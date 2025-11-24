"""
Node implementations for the RAG Agent pipeline.
Each node is a function that takes AgentState and returns modified AgentState.
"""

import re
import os
import sys
from typing import List
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# Fixed imports - removed the broken ones since we don't need them for this pipeline
from rag_based_book_bot.agents.states import (
    AgentState, DocumentChunk, RetrievedChunk, 
    ParsedQuery, QueryIntent, LLMResponse
)

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "coding-books")
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "books_rag")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_pc = None
_index = None
_model = None

def get_pinecone_index():
    """Get Pinecone index (lazy initialization)"""
    global _pc, _index
    if _index is None:
        _pc = Pinecone(api_key=PINECONE_API_KEY)
        _index = _pc.Index(INDEX_NAME)
    return _index

def get_embedding_model():
    """Get embedding model (lazy initialization)"""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def pdf_loader_node(state: AgentState) -> AgentState:
    """PDF loader node - not used in direct query pipeline"""
    state.current_node = "pdf_loader"
    state.errors.append("PDF loader not implemented - use book_ingestion.py instead")
    return state


def chunking_embedding_node(state: AgentState) -> AgentState:
    """Chunking and embedding node - not used in direct query pipeline"""
    state.current_node = "chunking_embedding"
    state.errors.append("Chunking not implemented - use book_ingestion.py instead")
    return state


def user_query_node(state: AgentState) -> AgentState:
    """Parse user query"""
    state.current_node = "user_query"
    
    if not state.user_query:
        state.errors.append("No user query provided")
        return state
    
    try:
        query = state.user_query.lower()
        
        state.parsed_query = ParsedQuery(
            raw_query=state.user_query,
            intent=_detect_intent(query),
            topics=_extract_topics(query),
            keywords=_extract_keywords(query),
            code_language="python" if any(w in query for w in 
                ['code', 'implement', 'write', 'show me', 'example']) else None,
            complexity_hint=_detect_complexity(query)
        )
        
    except Exception as e:
        state.errors.append(f"Query parsing failed: {str(e)}")
    
    return state


def _detect_intent(query: str) -> QueryIntent:
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
    known_topics = [
        'neural network', 'cnn', 'rnn', 'lstm', 'transformer',
        'gradient descent', 'backpropagation', 'linear regression',
        'logistic regression', 'decision tree', 'random forest',
        'svm', 'clustering', 'pca', 'autoencoder', 'gan',
        'reinforcement learning', 'keras', 'tensorflow', 'sklearn'
    ]
    return [t for t in known_topics if t in query.lower()]


def _extract_keywords(query: str) -> list[str]:
    stopwords = {'the', 'a', 'an', 'is', 'are', 'what', 'how', 'why', 'can', 'do', 'me', 'i', 'to'}
    words = re.findall(r'\b\w+\b', query.lower())
    return [w for w in words if w not in stopwords and len(w) > 2]


def _detect_complexity(query: str) -> str:
    if any(w in query for w in ['basic', 'simple', 'beginner', 'intro']):
        return "beginner"
    if any(w in query for w in ['advanced', 'complex', 'deep dive', 'detailed']):
        return "advanced"
    return "intermediate"


def vector_search_node(state: AgentState, top_k: int = 10) -> AgentState:
    """
    Direct Pinecone vector search - uses the data ingested by book_ingestion.py
    """
    state.current_node = "vector_search"
    
    if not state.parsed_query:
        state.errors.append("Missing query for search")
        return state
    
    try:
        # Get Pinecone index and embedding model
        index = get_pinecone_index()
        model = get_embedding_model()
        
        # Generate query embedding
        query_embedding = model.encode(state.parsed_query.raw_query).tolist()
        
        # Build filter if book/chapter specified
        filter_dict = {}
        if hasattr(state, 'book_filter') and state.book_filter:
            filter_dict["book_title"] = state.book_filter
        if hasattr(state, 'chapter_filter') and state.chapter_filter:
            filter_dict["chapter_numbers"] = {"$in": [state.chapter_filter]}
        
        # Query Pinecone directly
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=NAMESPACE,
            filter=filter_dict if filter_dict else None,
            include_metadata=True
        )
        
        # Convert Pinecone results to RetrievedChunk objects
        retrieved_chunks = []
        
        for match in results.get("matches", []):
            metadata = match.get("metadata", {})
            
            # Handle metadata lists safely
            chapter_titles = metadata.get("chapter_titles", [])
            chapter_numbers = metadata.get("chapter_numbers", [])
            section_titles = metadata.get("section_titles", [])
            
            chapter_title = chapter_titles[0] if chapter_titles else ""
            chapter_number = chapter_numbers[0] if chapter_numbers else ""
            
            # Create DocumentChunk with metadata from book_ingestion
            chunk = DocumentChunk(
                chunk_id=match["id"],
                content=metadata.get("text", ""),
                chapter=f"{chapter_number}: {chapter_title}" if chapter_number else chapter_title,
                section=", ".join(section_titles) if section_titles else "",
                page_number=metadata.get("page_start"),
                chunk_type="code" if metadata.get("contains_code") else "text"
            )
            
            retrieved_chunks.append(RetrievedChunk(
                chunk=chunk,
                similarity_score=match.get("score", 0.0)
            ))
        
        state.retrieved_chunks = retrieved_chunks
        
    except Exception as e:
        state.errors.append(f"Vector search failed: {str(e)}")
    
    return state


def reranking_node(state: AgentState, top_k: int = 5) -> AgentState:
    """Rerank based on query intent and content type"""
    state.current_node = "reranking"
    
    if not state.retrieved_chunks or not state.parsed_query:
        state.errors.append("Missing chunks or query for reranking")
        return state
    
    try:
        query = state.parsed_query
        
        for rc in state.retrieved_chunks:
            score = rc.similarity_score * 0.4
            
            # Boost for topic matches
            for topic in query.topics:
                if topic in rc.chunk.content.lower():
                    score += 0.2
            
            # Boost for keyword matches
            keyword_matches = sum(1 for kw in query.keywords 
                                 if kw in rc.chunk.content.lower())
            score += keyword_matches * 0.05
            
            # Boost code chunks for code requests
            if query.intent == QueryIntent.CODE_REQUEST and rc.chunk.chunk_type == "code":
                score += 0.25
            
            # Boost text chunks for conceptual queries
            if query.intent == QueryIntent.CONCEPTUAL and rc.chunk.chunk_type == "text":
                score += 0.1
            
            rc.rerank_score = min(score, 1.0)
        
        # Sort and calculate percentages
        state.retrieved_chunks.sort(key=lambda x: x.rerank_score, reverse=True)
        
        max_score = state.retrieved_chunks[0].rerank_score if state.retrieved_chunks else 1
        for rc in state.retrieved_chunks:
            rc.relevance_percentage = round((rc.rerank_score / max_score) * 100, 1)
        
        state.reranked_chunks = state.retrieved_chunks[:top_k]
        
    except Exception as e:
        state.errors.append(f"Reranking failed: {str(e)}")
    
    return state


def context_assembly_node(state: AgentState) -> AgentState:
    """Assemble context with rich metadata from ingestion"""
    state.current_node = "context_assembly"
    
    if not state.reranked_chunks or not state.parsed_query:
        state.errors.append("Missing reranked chunks or query")
        return state
    
    try:
        context_parts = []
        
        for i, rc in enumerate(state.reranked_chunks):
            chunk = rc.chunk
            
            # Format source
            context_parts.append(
                f"[SOURCE {i+1}]\n"
                f"Chapter: {chunk.chapter}\n"
                f"Section: {chunk.section}\n"
                f"Page: {chunk.page_number}\n"
                f"Type: {'CODE' if chunk.chunk_type == 'code' else 'TEXT'}\n"
                f"Relevance: {rc.relevance_percentage}%\n\n"
                f"{chunk.content}\n"
            )
        
        state.assembled_context = "\n{'='*70}\n".join(context_parts)
        state.system_prompt = _build_system_prompt(state.parsed_query)
        
    except Exception as e:
        state.errors.append(f"Context assembly failed: {str(e)}")
    
    return state


def _build_system_prompt(query: ParsedQuery) -> str:
    """Build system prompt based on intent"""
    base = """You are an expert ML/AI assistant with access to technical books.
Use the provided sources to give accurate, well-cited answers.
Always reference sources by number [SOURCE 1], [SOURCE 2], etc."""
    
    intent_prompts = {
        QueryIntent.CODE_REQUEST: "\n\nProvide working code with explanations.",
        QueryIntent.CONCEPTUAL: "\n\nExplain clearly with examples.",
        QueryIntent.COMPARISON: "\n\nCompare and contrast systematically.",
        QueryIntent.DEBUGGING: "\n\nIdentify issues and provide fixes.",
        QueryIntent.TUTORIAL: "\n\nProvide step-by-step guidance."
    }
    
    complexity = {
        "beginner": " Use simple language.",
        "intermediate": " Balance theory and practice.",
        "advanced": " Include technical details."
    }
    
    return base + intent_prompts.get(query.intent, "") + complexity.get(query.complexity_hint, "")


def llm_reasoning_node(state: AgentState) -> AgentState:
    """Call OpenAI LLM for answer generation"""
    state.current_node = "llm_reasoning"
    
    if not state.reranked_chunks or not state.parsed_query:
        state.errors.append("Missing context or query for LLM")
        return state
    
    # Build context if not already assembled
    if not state.assembled_context:
        state = context_assembly_node(state)
    
    try:
        from openai import OpenAI
        
        # Initialize OpenAI client
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": state.system_prompt},
                {"role": "user", "content": f"{state.assembled_context}\n\nQuestion: {state.parsed_query.raw_query}"}
            ],
            temperature=0.7
        )
        
        # Extract answer from response
        answer = response.choices[0].message.content
        
        # Extract code snippets from chunks (if any)
        chunks = state.reranked_chunks
        code_snippets = [c.chunk.content for c in chunks if c.chunk.chunk_type == "code"]
        
        # Build response object
        state.response = LLMResponse(
            answer=answer,
            code_snippets=code_snippets[:2],
            sources=[c.chunk.chunk_id for c in chunks[:3]],
            confidence=0.85
        )
        
    except Exception as e:
        state.errors.append(f"LLM reasoning failed: {str(e)}")
        # Fallback response if LLM fails
        if state.reranked_chunks:
            chunks = state.reranked_chunks
            fallback_answer = f"Error calling LLM. Here's the retrieved content:\n\n"
            fallback_answer += f"From {chunks[0].chunk.chapter}:\n"
            fallback_answer += chunks[0].chunk.content[:500] + "..."
            
            state.response = LLMResponse(
                answer=fallback_answer,
                code_snippets=[],
                sources=[c.chunk.chunk_id for c in chunks[:3]],
                confidence=0.5
            )
    
    return state