"""
Node implementations for the RAG Agent pipeline.
Each node is a function that takes AgentState and returns modified AgentState.
"""

import re
import hashlib
from typing import Callable
import numpy as np

from states import (
    AgentState, DocumentChunk, RetrievedChunk, 
    ParsedQuery, QueryIntent, LLMResponse
)
from prompts import build_system_prompt, format_context
from utils import generate_chunk_id, clean_text

# Type alias for node functions
NodeFunc = Callable[[AgentState], AgentState]


# =============================================================================
# NODE 1: PDF LOADER
# =============================================================================

def pdf_loader_node(state: AgentState) -> AgentState:
    """
    Loads PDF and extracts raw text.
    For now, this is a dummy that simulates loading the Hands-On ML book.
    Replace with actual PDF parsing (PyPDF2, pdfplumber, etc.) later.
    """
    state.current_node = "pdf_loader"
    
    if not state.pdf_path:
        state.errors.append("No PDF path provided")
        return state
    
    try:
        # DUMMY: Simulating book content structure
        # In production, replace with actual PDF extraction
        state.raw_text = _simulate_book_extraction()
        state.total_pages = 856  # Approx pages in Hands-On ML
        
    except Exception as e:
        state.errors.append(f"PDF loading failed: {str(e)}")
    
    return state


def _simulate_book_extraction() -> str:
    """Simulates extracted text from Hands-On ML book."""
    # This dummy content mirrors the book's structure
    return """
    CHAPTER 1: The Machine Learning Landscape
    
    Machine Learning is the field of study that gives computers the ability to learn 
    without being explicitly programmed. This chapter introduces the fundamental concepts.
    
    Types of Machine Learning Systems:
    - Supervised Learning
    - Unsupervised Learning  
    - Reinforcement Learning
    
    ```python
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    ```
    
    CHAPTER 4: Training Models
    
    Linear Regression is one of the simplest models. The Normal Equation provides 
    a closed-form solution: theta = (X^T X)^(-1) X^T y
    
    Gradient Descent is an optimization algorithm that iteratively adjusts parameters
    to minimize a cost function.
    
    ```python
    eta = 0.1  # learning rate
    n_iterations = 1000
    theta = np.random.randn(2, 1)
    
    for iteration in range(n_iterations):
        gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
        theta = theta - eta * gradients
    ```
    
    CHAPTER 10: Introduction to Artificial Neural Networks with Keras
    
    Neural networks are inspired by biological neurons. The perceptron is the simplest
    form of a neural network.
    
    ```python
    import tensorflow as tf
    from tensorflow import keras
    
    model = keras.Sequential([
        keras.layers.Dense(300, activation='relu', input_shape=[8]),
        keras.layers.Dense(100, activation='relu'),
        keras.layers.Dense(1)
    ])
    
    model.compile(loss='mse', optimizer='sgd')
    model.fit(X_train, y_train, epochs=20)
    ```
    
    CHAPTER 14: Deep Computer Vision Using Convolutional Neural Networks
    
    CNNs use convolutional layers to detect features in images. Key concepts include
    filters, feature maps, pooling, and stride.
    
    ```python
    model = keras.Sequential([
        keras.layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=[28, 28, 1]),
        keras.layers.MaxPooling2D(pool_size=2),
        keras.layers.Conv2D(64, kernel_size=3, activation='relu'),
        keras.layers.MaxPooling2D(pool_size=2),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    ```
    """


# =============================================================================
# NODE 2: CHUNKING & EMBEDDING
# =============================================================================

def chunking_embedding_node(state: AgentState) -> AgentState:
    """
    Chunks the raw text intelligently and generates embeddings.
    Strategy: 
    - Preserve code blocks as single chunks
    - Split text by sections/paragraphs
    - Add overlap for context continuity
    """
    state.current_node = "chunking_embedding"
    
    if not state.raw_text:
        state.errors.append("No raw text to chunk")
        return state
    
    try:
        chunks = _smart_chunk(state.raw_text)
        
        # Generate embeddings for each chunk
        for chunk in chunks:
            chunk.embedding = _generate_embedding(chunk.content)
        
        # Link chunks for context
        for i, chunk in enumerate(chunks):
            if i > 0:
                chunk.prev_chunk_id = chunks[i-1].chunk_id
            if i < len(chunks) - 1:
                chunk.next_chunk_id = chunks[i+1].chunk_id
        
        state.chunks = chunks
        
    except Exception as e:
        state.errors.append(f"Chunking failed: {str(e)}")
    
    return state


def _smart_chunk(text: str, max_chunk_size: int = 500) -> list[DocumentChunk]:
    """
    Intelligently chunks text while preserving code blocks and context.
    """
    chunks = []
    current_chapter = "Unknown"
    current_section = ""
    
    # Split by chapters first
    chapter_pattern = r'(CHAPTER \d+:[^\n]+)'
    parts = re.split(chapter_pattern, text)
    
    chunk_counter = 0
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
            
        # Check if this is a chapter header
        if re.match(r'CHAPTER \d+:', part):
            current_chapter = part
            continue
        
        # Extract code blocks separately
        code_pattern = r'```python(.*?)```'
        segments = re.split(code_pattern, part, flags=re.DOTALL)
        
        is_code = False
        for segment in segments:
            segment = segment.strip()
            if not segment:
                is_code = not is_code
                continue
            
            if is_code:
                # Code block - keep as single chunk
                chunk_id = _generate_chunk_id(segment, chunk_counter)
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    content=segment,
                    chapter=current_chapter,
                    section=current_section,
                    chunk_type="code"
                ))
                chunk_counter += 1
            else:
                # Text content - split into smaller chunks
                text_chunks = _split_text(segment, max_chunk_size)
                for tc in text_chunks:
                    chunk_id = _generate_chunk_id(tc, chunk_counter)
                    chunks.append(DocumentChunk(
                        chunk_id=chunk_id,
                        content=tc,
                        chapter=current_chapter,
                        section=current_section,
                        chunk_type="text"
                    ))
                    chunk_counter += 1
            
            is_code = not is_code
    
    return chunks


def _split_text(text: str, max_size: int) -> list[str]:
    """Splits text into chunks with overlap."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current = [], ""
    
    for sent in sentences:
        if len(current) + len(sent) > max_size and current:
            chunks.append(current.strip())
            # Keep last sentence for overlap
            overlap = sentences[sentences.index(sent)-1] if sentences.index(sent) > 0 else ""
            current = overlap + " " + sent
        else:
            current += " " + sent
    
    if current.strip():
        chunks.append(current.strip())
    
    return chunks


def _generate_chunk_id(content: str, counter: int) -> str:
    """Generates unique chunk ID."""
    hash_val = hashlib.md5(content.encode()).hexdigest()[:8]
    return f"chunk_{counter}_{hash_val}"


def _generate_embedding(text: str, dim: int = 384) -> list[float]:
    """
    Generates embedding vector for text.
    DUMMY: Creates deterministic pseudo-embedding based on text features.
    Replace with actual embedding model (sentence-transformers, OpenAI, etc.)
    """
    np.random.seed(hash(text) % 2**32)
    
    # Create feature-based embedding (dummy but somewhat meaningful)
    base = np.random.randn(dim).astype(float)
    
    # Adjust based on content characteristics
    if 'def ' in text or 'import ' in text:
        base[:50] += 0.5  # Code indicator
    if 'neural' in text.lower() or 'network' in text.lower():
        base[50:100] += 0.3
    if 'gradient' in text.lower() or 'descent' in text.lower():
        base[100:150] += 0.3
    
    # Normalize
    norm = np.linalg.norm(base)
    return (base / norm).tolist()


# =============================================================================
# NODE 3: USER QUERY PARSING
# =============================================================================

def user_query_node(state: AgentState) -> AgentState:
    """
    Parses user query to extract intent, topics, keywords, etc.
    """
    state.current_node = "user_query"
    
    if not state.user_query:
        state.errors.append("No user query provided")
        return state
    
    try:
        query = state.user_query.lower()
        
        # Detect intent
        intent = _detect_intent(query)
        
        # Extract topics and keywords
        topics = _extract_topics(query)
        keywords = _extract_keywords(query)
        
        # Detect if code is requested
        code_lang = "python" if any(w in query for w in 
            ['code', 'implement', 'write', 'show me', 'example']) else None
        
        # Detect complexity
        complexity = _detect_complexity(query)
        
        state.parsed_query = ParsedQuery(
            raw_query=state.user_query,
            intent=intent,
            topics=topics,
            keywords=keywords,
            code_language=code_lang,
            complexity_hint=complexity
        )
        
    except Exception as e:
        state.errors.append(f"Query parsing failed: {str(e)}")
    
    return state


def _detect_intent(query: str) -> QueryIntent:
    """Detects the intent of the query."""
    if any(w in query for w in ['implement', 'code', 'write', 'build', 'create']):
        return QueryIntent.CODE_REQUEST
    if any(w in query for w in ['difference', 'compare', 'vs', 'versus']):
        return QueryIntent.COMPARISON
    if any(w in query for w in ['error', 'bug', 'fix', 'wrong', 'not working']):
        return QueryIntent.DEBUGGING
    if any(w in query for w in ['tutorial', 'walk through', 'step by step', 'guide']):
        return QueryIntent.TUTORIAL
    return QueryIntent.CONCEPTUAL


def _extract_topics(query: str) -> list[str]:
    """Extracts ML/AI topics from query."""
    known_topics = [
        'neural network', 'cnn', 'rnn', 'lstm', 'transformer',
        'gradient descent', 'backpropagation', 'linear regression',
        'logistic regression', 'decision tree', 'random forest',
        'svm', 'clustering', 'pca', 'autoencoder', 'gan',
        'reinforcement learning', 'keras', 'tensorflow', 'sklearn'
    ]
    return [t for t in known_topics if t in query.lower()]


def _extract_keywords(query: str) -> list[str]:
    """Extracts important keywords."""
    stopwords = {'the', 'a', 'an', 'is', 'are', 'what', 'how', 'why', 'can', 'do', 'me', 'i', 'to'}
    words = re.findall(r'\b\w+\b', query.lower())
    return [w for w in words if w not in stopwords and len(w) > 2]


def _detect_complexity(query: str) -> str:
    """Detects desired complexity level."""
    if any(w in query for w in ['basic', 'simple', 'beginner', 'intro']):
        return "beginner"
    if any(w in query for w in ['advanced', 'complex', 'deep dive', 'detailed']):
        return "advanced"
    return "intermediate"


# =============================================================================
# NODE 4: VECTOR SEARCH
# =============================================================================

def vector_search_node(state: AgentState, top_k: int = 10) -> AgentState:
    """
    Performs vector similarity search to retrieve relevant chunks.
    """
    state.current_node = "vector_search"
    
    if not state.parsed_query or not state.chunks:
        state.errors.append("Missing query or chunks for search")
        return state
    
    try:
        # Generate query embedding
        query_embedding = np.array(_generate_embedding(state.parsed_query.raw_query))
        
        # Calculate similarities
        scored_chunks = []
        for chunk in state.chunks:
            chunk_emb = np.array(chunk.embedding)
            similarity = float(np.dot(query_embedding, chunk_emb))
            scored_chunks.append(RetrievedChunk(
                chunk=chunk,
                similarity_score=similarity
            ))
        
        # Sort by similarity and take top_k
        scored_chunks.sort(key=lambda x: x.similarity_score, reverse=True)
        state.retrieved_chunks = scored_chunks[:top_k]
        
    except Exception as e:
        state.errors.append(f"Vector search failed: {str(e)}")
    
    return state


# =============================================================================
# NODE 5: RERANKING
# =============================================================================

def reranking_node(state: AgentState, top_k: int = 5) -> AgentState:
    """
    Reranks retrieved chunks based on deeper relevance analysis.
    Outputs relevance percentages.
    """
    state.current_node = "reranking"
    
    if not state.retrieved_chunks or not state.parsed_query:
        state.errors.append("Missing chunks or query for reranking")
        return state
    
    try:
        query = state.parsed_query
        
        for rc in state.retrieved_chunks:
            score = rc.similarity_score * 0.4  # Base similarity weight
            
            # Boost for topic matches
            for topic in query.topics:
                if topic in rc.chunk.content.lower():
                    score += 0.2
            
            # Boost for keyword matches
            keyword_matches = sum(1 for kw in query.keywords if kw in rc.chunk.content.lower())
            score += keyword_matches * 0.05
            
            # Boost code chunks for code requests
            if query.intent == QueryIntent.CODE_REQUEST and rc.chunk.chunk_type == "code":
                score += 0.25
            
            # Boost text chunks for conceptual queries
            if query.intent == QueryIntent.CONCEPTUAL and rc.chunk.chunk_type == "text":
                score += 0.1
            
            rc.rerank_score = min(score, 1.0)  # Cap at 1.0
        
        # Sort by rerank score
        state.retrieved_chunks.sort(key=lambda x: x.rerank_score, reverse=True)
        
        # Calculate relevance percentages (normalized)
        max_score = state.retrieved_chunks[0].rerank_score if state.retrieved_chunks else 1
        for rc in state.retrieved_chunks:
            rc.relevance_percentage = round((rc.rerank_score / max_score) * 100, 1)
        
        # Keep top_k
        state.reranked_chunks = state.retrieved_chunks[:top_k]
        
    except Exception as e:
        state.errors.append(f"Reranking failed: {str(e)}")
    
    return state


# =============================================================================
# NODE 6: CONTEXT ASSEMBLY
# =============================================================================

def context_assembly_node(state: AgentState) -> AgentState:
    """
    Assembles context for LLM input from reranked chunks.
    Creates appropriate system prompt based on query intent.
    """
    state.current_node = "context_assembly"
    
    if not state.reranked_chunks or not state.parsed_query:
        state.errors.append("Missing reranked chunks or query")
        return state
    
    try:
        query = state.parsed_query
        
        # Build context from chunks
        context_parts = []
        for i, rc in enumerate(state.reranked_chunks):
            chunk = rc.chunk
            context_parts.append(
                f"[Source {i+1} - {chunk.chapter} | Relevance: {rc.relevance_percentage}%]\n"
                f"{chunk.content}\n"
            )
        
        state.assembled_context = "\n---\n".join(context_parts)
        
        # Build system prompt based on intent
        state.system_prompt = _build_system_prompt(query)
        
    except Exception as e:
        state.errors.append(f"Context assembly failed: {str(e)}")
    
    return state


def _build_system_prompt(query: ParsedQuery) -> str:
    """Builds system prompt based on query intent."""
    base = """You are an expert ML/AI teaching assistant with deep knowledge of 
the 'Hands-On Machine Learning' book by Aurélien Géron. Use the provided context 
to answer questions accurately. Always cite which source you're using."""
    
    intent_specific = {
        QueryIntent.CODE_REQUEST: "\n\nProvide working Python code examples. Explain each part briefly.",
        QueryIntent.CONCEPTUAL: "\n\nExplain concepts clearly with intuition before formulas.",
        QueryIntent.DEBUGGING: "\n\nHelp identify the issue and provide corrected code if applicable.",
        QueryIntent.COMPARISON: "\n\nStructure your response to clearly compare and contrast.",
        QueryIntent.TUTORIAL: "\n\nProvide step-by-step guidance with code at each step."
    }
    
    complexity_hint = {
        "beginner": " Keep explanations simple and avoid jargon.",
        "intermediate": " Balance theory and practice.",
        "advanced": " Include mathematical details and edge cases."
    }
    
    return base + intent_specific.get(query.intent, "") + complexity_hint.get(query.complexity_hint, "")


# =============================================================================
# NODE 7: LLM REASONING
# =============================================================================

def llm_reasoning_node(state: AgentState) -> AgentState:
    """
    Calls LLM to generate final response.
    DUMMY: Simulates LLM response. Replace with actual LLM call.
    """
    state.current_node = "llm_reasoning"
    
    if not state.assembled_context or not state.parsed_query:
        state.errors.append("Missing context or query for LLM")
        return state
    
    try:
        # DUMMY: Simulate LLM response
        # In production, replace with actual LLM API call
        response = _simulate_llm_response(state)
        state.response = response
        
    except Exception as e:
        state.errors.append(f"LLM reasoning failed: {str(e)}")
    
    return state


def _simulate_llm_response(state: AgentState) -> LLMResponse:
    """
    Simulates LLM response based on query and context.
    Replace with actual LLM integration (OpenAI, Anthropic, local model, etc.)
    """
    query = state.parsed_query
    chunks = state.reranked_chunks
    
    # Build dummy response based on intent
    if query.intent == QueryIntent.CODE_REQUEST:
        answer = f"Based on the context from {chunks[0].chunk.chapter}, here's how to implement this:\n\n"
        code = chunks[0].chunk.content if chunks[0].chunk.chunk_type == "code" else "# Example code here"
        return LLMResponse(
            answer=answer,
            code_snippets=[code],
            sources=[c.chunk.chunk_id for c in chunks[:3]],
            confidence=0.85
        )
    
    elif query.intent == QueryIntent.CONCEPTUAL:
        answer = f"Let me explain based on {chunks[0].chunk.chapter}:\n\n{chunks[0].chunk.content[:200]}..."
        return LLMResponse(
            answer=answer,
            code_snippets=[],
            sources=[c.chunk.chunk_id for c in chunks[:3]],
            confidence=0.9
        )
    
    else:
        answer = f"Here's what I found relevant to your query:\n\n"
        for i, c in enumerate(chunks[:3]):
            answer += f"{i+1}. From {c.chunk.chapter}: {c.chunk.content[:100]}...\n"
        return LLMResponse(
            answer=answer,
            code_snippets=[],
            sources=[c.chunk.chunk_id for c in chunks[:3]],
            confidence=0.8
        )