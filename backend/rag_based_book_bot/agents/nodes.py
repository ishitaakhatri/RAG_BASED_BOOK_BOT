"""
Updated Node implementations with LangChain and Gemini

CHANGES:
- Switched from direct OpenAI API calls to LangChain
- Using Google's Gemini (ChatGoogleGenerativeAI)
- Enhanced metadata handling to preserve book_title throughout pipeline
- Better chunk formatting with book information
"""

import re
import os
import json
from typing import List
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from rag_based_book_bot.agents.states import (
    AgentState, DocumentChunk, RetrievedChunk, 
    ParsedQuery, QueryIntent, LLMResponse
)

# Import the new retrieval components
from rag_based_book_bot.retrieval.cross_encoder_reranker import CrossEncoderReranker
from rag_based_book_bot.retrieval.multi_hop_expander import MultiHopExpander
from rag_based_book_bot.retrieval.cluster_manager import ClusterManager
from rag_based_book_bot.retrieval.context_compressor import EnhancedContextCompressor

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "coding-books")
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "books_rag")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ============================================================================
# LANGCHAIN LLM INITIALIZATION (Gemini)
# ============================================================================

# Initialize Gemini LLM at module level
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.7,
    convert_system_message_to_human=True  # Gemini doesn't support system messages natively
)

# Global instances (lazy loading)
_pc = None
_index = None
_model = None
_cross_encoder = None
_multi_hop = None
_cluster_manager = None
_compressor = None

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

def get_cross_encoder():
    """Get cross-encoder reranker"""
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoderReranker()
    return _cross_encoder

def get_multi_hop_expander():
    """Get multi-hop expander"""
    global _multi_hop
    if _multi_hop is None:
        _multi_hop = MultiHopExpander()
    return _multi_hop

def get_cluster_manager():
    """Get cluster manager"""
    global _cluster_manager
    if _cluster_manager is None:
        _cluster_manager = ClusterManager(n_clusters=100)
    return _cluster_manager

def get_compressor(target_tokens=2000, max_tokens=4000):
    """Get context compressor with configurable token limits"""
    return EnhancedContextCompressor(
        target_tokens=target_tokens,
        max_tokens=max_tokens
    )


# ============================================================================
# UPDATED: LLM-BASED QUERY PARSING NODE WITH LANGCHAIN
# ============================================================================

def user_query_node(state: AgentState) -> AgentState:
    """
    ✨ LLM-based query parsing for intelligent intent detection using LangChain
    """
    state.current_node = "user_query"
    
    if not state.user_query:
        state.errors.append("No user query provided")
        return state
    
    try:
        print(f"\n[Query Parsing] Analyzing: '{state.user_query[:60]}...'")
        
        # Use LLM for intelligent parsing
        parsed_data = _parse_query_with_llm(state.user_query)
        
        state.parsed_query = ParsedQuery(
            raw_query=state.user_query,
            intent=QueryIntent[parsed_data['intent']],
            topics=parsed_data['topics'],
            keywords=parsed_data['keywords'],
            code_language=parsed_data['code_language'],
            complexity_hint=parsed_data['complexity_hint']
        )
        
        print(f"  ✅ Intent: {parsed_data['intent']}")
        print(f"  📚 Topics: {', '.join(parsed_data['topics'][:3])}")
        if parsed_data['code_language']:
            print(f"  💻 Language: {parsed_data['code_language']}")
        print(f"  📊 Level: {parsed_data['complexity_hint']}")
        
    except Exception as e:
        print(f"  ⚠️ LLM parsing failed: {e}")
        print(f"  → Using fallback heuristics")
        state.parsed_query = _fallback_parse_query(state.user_query)
    
    return state



def query_rewriter_node(state: AgentState, num_variations: int = 3) -> AgentState:
    """
    Query Rewriting Node - Generates alternative query formulations
    
    Uses Gemini LLM to create semantically similar but differently phrased queries
    to improve retrieval recall by covering different ways of expressing the same intent.
    
    Args:
        state: Current agent state with parsed_query
        num_variations: Number of alternative queries to generate (default: 3)
    
    Returns:
        Updated state with rewritten_queries populated
    """
    state.current_node = "query_rewriter"
    
    if not state.parsed_query:
        state.errors.append("Missing parsed query for rewriting")
        return state
    
    try:
        print(f"\n[Query Rewriting] Generating {num_variations} alternative queries...")
        
        # Use LLM to generate query variations
        rewritten = _generate_query_variations(
            state.parsed_query.raw_query,
            state.parsed_query.intent,
            num_variations
        )
        
        state.rewritten_queries = rewritten
        
        print(f"  ✅ Generated {len(rewritten)} variations:")
        for i, query in enumerate(rewritten, 1):
            print(f"     {i}. {query}")
        
    except Exception as e:
        print(f"  ⚠️ Query rewriting failed: {e}")
        print(f"  → Continuing with original query only")
        state.rewritten_queries = []
    
    return state

def _generate_query_variations(query: str, intent: QueryIntent, num_variations: int = 3) -> list[str]:
    """Generate alternative query formulations using Gemini"""
    
    system_prompt = """You are an expert at reformulating search queries to improve information retrieval.

Your task: Generate alternative phrasings of the user's query that:
1. Preserve the original intent and meaning
2. Use different vocabulary and sentence structures  
3. Cover different angles or aspects of the same question
4. Are optimized for semantic search in technical documentation

Guidelines:
- Keep queries concise (1-2 sentences max)
- Use synonyms and related technical terms
- Rephrase from different perspectives (e.g., "how to X" → "implementing X", "X tutorial")
- For code requests, vary between implementation-focused and explanation-focused
- Don't add new requirements or constraints not in the original query

Return ONLY a JSON array of strings, nothing else:
["variation 1", "variation 2", "variation 3"]"""

    intent_hints = {
        QueryIntent.CONCEPTUAL: "Focus on understanding, explanation, and theoretical aspects.",
        QueryIntent.CODE_REQUEST: "Vary between implementation details, code examples, and practical usage.",
        QueryIntent.DEBUGGING: "Include variations about troubleshooting, error fixing, and problem solving.",
        QueryIntent.COMPARISON: "Rephrase as differences, pros/cons, or when to use each option.",
        QueryIntent.TUTORIAL: "Vary between step-by-step guides, walkthroughs, and practical examples."
    }
    
    user_prompt = f"""Original query: "{query}"

Intent: {intent.value}
Hint: {intent_hints.get(intent, "")}

Generate {num_variations} alternative phrasings."""

    try:
        # Use LangChain to invoke Gemini
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = llm.invoke(messages)
        response_text = response.content.strip()
        
        # --- FIXED BROKEN STRINGS BELOW ---
        if response_text.startswith("```"):
            response_text = response_text.replace("```json", "").replace("```", "")
        elif response_text.startswith("```"):
            response_text = response_text.replace("```", "")
        # --- END FIX ---

        # Parse JSON array
        variations = json.loads(response_text)
        
        # Validate
        if isinstance(variations, list) and len(variations) > 0:
            return variations[:num_variations]  # Limit to requested number
        else:
            print(f"  ⚠️ Invalid response format: {variations}")
            return _fallback_query_variations(query, num_variations)
            
    except Exception as e:
        print(f"  ⚠️ LLM query rewriting failed: {e}")
        return _fallback_query_variations(query, num_variations)



def _fallback_query_variations(query: str, num_variations: int = 3) -> list[str]:
    """Fallback query variations using simple heuristics"""
    variations = []
    query_lower = query.lower()
    
    # Variation 1: Rephrase question words
    var1 = query
    replacements = {
        "how do i": "implementing",
        "what is": "explanation of",
        "how to": "guide for",
        "why does": "reason for",
        "can i": "method to"
    }
    for old, new in replacements.items():
        if old in query_lower:
            var1 = query_lower.replace(old, new)
            break
    if var1 != query_lower:
        variations.append(var1)
    
    # Variation 2: Add context words
    context_words = {
        "conceptual": ["understand", "explain", "concept"],
        "code": ["implement", "code", "example"],
        "debugging": ["fix", "troubleshoot", "debug"],
        "comparison": ["difference", "compare", "versus"],
        "tutorial": ["tutorial", "guide", "walkthrough"]
    }
    
    # Pick context based on keywords
    for category, words in context_words.items():
        if any(w in query_lower for w in words):
            var2 = f"{words} {query}"
            variations.append(var2)
            break
    
    # Variation 3: Extract key terms
    import re
    words = re.findall(r'\b\w{4,}\b', query_lower)
    if len(words) >= 2:
        var3 = " ".join(words[:min(5, len(words))])
        variations.append(var3)
    
    # Ensure we have enough variations
    while len(variations) < num_variations:
        variations.append(query)  # Use original as fallback
    
    return variations[:num_variations]





def _parse_query_with_llm(query: str) -> dict:
    """Intelligent query parsing using Gemini via LangChain"""
    
    system_prompt = """You are an expert query analyzer for a coding book learning assistant.

Analyze user queries to help retrieve the most relevant content from programming books.

For each query, extract:

1. **intent** (choose exactly ONE):
   - CONCEPTUAL: Understanding theory, explanations, "what is X?"
   - CODE_REQUEST: Wants code examples, implementations, "show me code"
   - DEBUGGING: Fixing errors, troubleshooting, "why isn't this working?"
   - COMPARISON: Comparing options, "difference between X and Y"
   - TUTORIAL: Step-by-step guide, "how to build X"

2. **topics**: Key technical concepts (3-5 items, lowercase)
   Examples: ["neural networks", "gradient descent", "tensorflow"]

3. **keywords**: Search terms (5-10 words, lowercase, no stopwords)
   Examples: ["neural", "network", "training", "loss", "function"]

4. **code_language**: Programming language or null
   Examples: "python", "javascript", "java", null

5. **complexity_hint** (choose ONE):
   - beginner: New to programming or the topic
   - intermediate: Has some experience, wants practical knowledge  
   - advanced: Expert-level, wants deep technical details

Respond with ONLY valid JSON:
{
  "intent": "CONCEPTUAL",
  "topics": ["deep learning", "cnn"],
  "keywords": ["convolution", "neural", "network", "layer"],
  "code_language": "python",
  "complexity_hint": "intermediate"
}"""

    user_prompt = f'Analyze this query: "{query}"'

    # Use LangChain to invoke Gemini
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    
    # Parse the response
    response_text = response.content.strip()
    
    # Remove markdown code blocks if present
    if response_text.startswith("```json"):
        response_text = response_text.replace("```json", "").replace("```", "").strip()
    elif response_text.startswith("```"):
        response_text = response_text.replace("```", "").strip()
    
    parsed = json.loads(response_text)
    
    # Validate and set defaults
    return {
        'intent': parsed.get('intent', 'CONCEPTUAL'),
        'topics': parsed.get('topics', []),
        'keywords': parsed.get('keywords', []),
        'code_language': parsed.get('code_language'),
        'complexity_hint': parsed.get('complexity_hint', 'intermediate')
    }


def _fallback_parse_query(query: str) -> ParsedQuery:
    """Fallback parser using simple heuristics"""
    query_lower = query.lower()
    
    # Intent detection
    if any(w in query_lower for w in ['implement', 'code', 'write', 'build', 'show me']):
        intent = QueryIntent.CODE_REQUEST
    elif any(w in query_lower for w in ['difference', 'compare', 'vs', 'versus']):
        intent = QueryIntent.COMPARISON
    elif any(w in query_lower for w in ['error', 'bug', 'fix', 'wrong', 'debug']):
        intent = QueryIntent.DEBUGGING
    elif any(w in query_lower for w in ['tutorial', 'walk through', 'step by step']):
        intent = QueryIntent.TUTORIAL
    else:
        intent = QueryIntent.CONCEPTUAL
    
    # Extract keywords
    stopwords = {'the', 'a', 'an', 'is', 'are', 'what', 'how', 'why', 'can', 'do', 'me', 'i', 'to'}
    words = re.findall(r'\b\w+\b', query_lower)
    keywords = [w for w in words if w not in stopwords and len(w) > 2]
    
    # Detect language
    code_language = None
    for lang in ['python', 'javascript', 'java', 'cpp', 'c++', 'go', 'rust', 'typescript']:
        if lang in query_lower:
            code_language = lang
            break
    
    # Detect complexity
    if any(w in query_lower for w in ['basic', 'simple', 'beginner', 'intro']):
        complexity = "beginner"
    elif any(w in query_lower for w in ['advanced', 'complex', 'deep dive', 'detailed']):
        complexity = "advanced"
    else:
        complexity = "intermediate"
    
    return ParsedQuery(
        raw_query=query,
        intent=intent,
        topics=[],
        keywords=keywords[:10],
        code_language=code_language,
        complexity_hint=complexity
    )


# ============================================================================
# EXISTING NODES (unchanged)
# ============================================================================

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


# ============================================================================
# RETRIEVAL NODES (unchanged - no LLM calls here)
# ============================================================================

def vector_search_node(state: AgentState, top_k: int = 50) -> AgentState:
    """PASS 1: Coarse Semantic Search with Query Expansion"""
    state.current_node = "vector_search"
    
    if not state.parsed_query:
        state.errors.append("Missing query for search")
        return state
    
    try:
        # Collect all queries (original + rewritten)
        all_queries = [state.parsed_query.raw_query] + state.rewritten_queries
        
        print(f"\n[PASS 1] Vector Search (top_k={top_k})")
        print(f"  → Searching with {len(all_queries)} queries (1 original + {len(state.rewritten_queries)} rewritten)")
        
        index = get_pinecone_index()
        model = get_embedding_model()
        
        filter_dict = {}
        if hasattr(state, 'book_filter') and state.book_filter:
            filter_dict["book_title"] = state.book_filter
        if hasattr(state, 'chapter_filter') and state.chapter_filter:
            filter_dict["chapter_numbers"] = {"$in": [state.chapter_filter]}
        
        # Perform parallel searches for all queries
        all_results = {}  # Use dict to deduplicate by chunk_id
        
        for i, query_text in enumerate(all_queries):
            query_embedding = model.encode(query_text).tolist()
            
            results = index.query(
                vector=query_embedding,
                top_k=top_k,
                namespace=NAMESPACE,
                filter=filter_dict if filter_dict else None,
                include_metadata=True
            )
            
            # Merge results, keeping highest score for each chunk_id
            for match in results.get("matches", []):
                chunk_id = match["id"]
                current_score = match.get("score", 0.0)
                
                if chunk_id not in all_results or current_score > all_results[chunk_id].get("score", 0):
                    all_results[chunk_id] = match
        
        # Convert deduplicated results to RetrievedChunk objects
        retrieved_chunks = []
        
        for match in all_results.values():
            metadata = match.get("metadata", {})
            
            # Get book metadata
            book_title = metadata.get("book_title", "Unknown Book")
            author = metadata.get("author", "Unknown Author")
            
            # Get page info
            page_start = metadata.get("page_start")
            page_end = metadata.get("page_end")
            chunk_index = metadata.get("chunk_index", 0)
            
            # Try to get chapter info
            chapter_titles = metadata.get("chapter_titles", [])
            chapter_numbers = metadata.get("chapter_numbers", [])
            section_titles = metadata.get("section_titles", [])
            
            # Build chapter string
            if chapter_titles and chapter_numbers:
                chapter_str = f"Chapter {chapter_numbers}: {chapter_titles}"
            elif page_start and page_end and page_start != page_end:
                chapter_str = f"Pages {page_start}-{page_end}"
            elif page_start:
                chapter_str = f"Page {page_start}"
            else:
                chapter_str = f"Semantic Chunk #{chunk_index + 1}"
            
            chunk = DocumentChunk(
                chunk_id=match["id"],
                content=metadata.get("text", ""),
                chapter=chapter_str,
                section=", ".join(section_titles) if section_titles else "",
                page_number=page_start,
                chunk_type="code" if metadata.get("contains_code") else "text",
                book_title=book_title,
                author=author,
                chapter_title="",
                chapter_number=""
            )
            
            retrieved_chunks.append(RetrievedChunk(
                chunk=chunk,
                similarity_score=match.get("score", 0.0)
            ))
        
        # Sort by score descending
        retrieved_chunks.sort(key=lambda x: x.similarity_score, reverse=True)
        
        state.retrieved_chunks = retrieved_chunks
        print(f"  → Retrieved {len(retrieved_chunks)} unique candidates (deduplicated)")
        
    except Exception as e:
        state.errors.append(f"Vector search failed: {str(e)}")
    
    return state



def reranking_node(state: AgentState, top_k: int = 15) -> AgentState:
    """PASS 2: Cross-Encoder Reranking - FIXED for semantic chunking"""
    state.current_node = "reranking"
    
    if not state.retrieved_chunks or not state.parsed_query:
        state.errors.append("Missing chunks or query for reranking")
        return state
    
    try:
        print(f"\n[PASS 2] Cross-Encoder Reranking (top_k={top_k})")
        
        cross_encoder = get_cross_encoder()
        
        chunks_data = []
        for rc in state.retrieved_chunks:
            chunks_data.append({
                'text': rc.chunk.content,
                'metadata': {
                    'chunk_id': rc.chunk.chunk_id,
                    'chapter': rc.chunk.chapter,  # ← Already has meaningful info
                    'page': rc.chunk.page_number,
                    'type': rc.chunk.chunk_type,
                    'book_title': rc.chunk.book_title,
                    'author': rc.chunk.author
                },
                'similarity_score': rc.similarity_score
            })
        
        reranked = cross_encoder.rerank_with_metadata(
            state.parsed_query.raw_query,
            chunks_data,
            top_k=top_k
        )
        
        reranked_chunks = []
        for chunk_data in reranked:
            chunk = DocumentChunk(
                chunk_id=chunk_data['metadata']['chunk_id'],
                content=chunk_data['text'],
                chapter=chunk_data['metadata']['chapter'],  # ← Preserved
                section="",
                page_number=chunk_data['metadata']['page'],
                chunk_type=chunk_data['metadata']['type'],
                book_title=chunk_data['metadata'].get('book_title', 'Unknown Book'),
                author=chunk_data['metadata'].get('author', 'Unknown Author')
            )
            
            reranked_chunks.append(RetrievedChunk(
                chunk=chunk,
                similarity_score=chunk_data['similarity_score'],
                rerank_score=chunk_data['cross_encoder_score'],
                relevance_percentage=round(chunk_data['final_score'] * 100, 1)
            ))
        
        state.reranked_chunks = reranked_chunks
        print(f"  → Reranked to {len(reranked_chunks)} high-precision chunks")
        
    except Exception as e:
        state.errors.append(f"Reranking failed: {str(e)}")
        state.reranked_chunks = state.retrieved_chunks[:top_k]
    
    return state


def multi_hop_expansion_node(state: AgentState, max_hops: int = 2) -> AgentState:
    """PASS 3: Multi-Hop Retrieval - FIXED for semantic chunking"""
    state.current_node = "multi_hop_expansion"
    
    if not state.reranked_chunks or not state.parsed_query:
        state.errors.append("Missing reranked chunks for multi-hop")
        return state
    
    try:
        print(f"\n[PASS 3] Multi-Hop Expansion (max_hops={max_hops})")
        
        expander = get_multi_hop_expander()
        
        initial_results = []
        for rc in state.reranked_chunks[:10]:
            initial_results.append({
                'id': rc.chunk.chunk_id,
                'text': rc.chunk.content,
                'score': rc.rerank_score,
                'book_title': rc.chunk.book_title,
                'author': rc.chunk.author,
                'chapter': rc.chunk.chapter,      # ADD THIS LINE
                'page': rc.chunk.page_number      # ADD THIS LINE
            })
        
        def retrieval_fn(query_text: str, top_k: int = 5):
            index = get_pinecone_index()
            model = get_embedding_model()
            
            emb = model.encode(query_text).tolist()
            results = index.query(
                vector=emb,
                top_k=top_k,
                namespace=NAMESPACE,
                include_metadata=True
            )
            
            retrieved = []
            for m in results.get('matches', []):
                metadata = m.get('metadata', {})
                
                # FIXED: Build chapter string for expanded chunks
                page_start = metadata.get('page_start')
                page_end = metadata.get('page_end')
                chunk_index = metadata.get('chunk_index', 0)
                
                if page_start and page_end and page_start != page_end:
                    chapter_str = f"Pages {page_start}-{page_end}"
                elif page_start:
                    chapter_str = f"Page {page_start}"
                else:
                    chapter_str = f"Semantic Chunk #{chunk_index + 1}"
                
                retrieved.append({
                    'id': m['id'],
                    'text': metadata.get('text', ''),
                    'score': m.get('score', 0),
                    'book_title': metadata.get('book_title', 'Unknown Book'),
                    'author': metadata.get('author', 'Unknown Author'),
                    'chapter': chapter_str,
                    'page': page_start
                })
            
            return retrieved
        
        expanded_results = expander.multi_hop_retrieve(
            state.parsed_query.raw_query,
            initial_results,
            retrieval_fn,
            max_hops=max_hops,
            top_k_per_hop=3
        )
        
        # Add expanded chunks to state
        for exp_result in expanded_results[len(initial_results):]:
            chunk = DocumentChunk(
                chunk_id=exp_result['id'],
                content=exp_result['text'],
                chapter=exp_result.get('chapter', 'Multi-hop Result'),
                section="",
                page_number=exp_result.get('page'),
                chunk_type="text",
                book_title=exp_result.get('book_title', 'Unknown Book'),
                author=exp_result.get('author', 'Unknown Author')
            )
            
            state.reranked_chunks.append(RetrievedChunk(
                chunk=chunk,
                similarity_score=exp_result['score'],
                rerank_score=exp_result['score'] * 0.8,
                relevance_percentage=exp_result['score'] * 80
            ))
        
        print(f"  → Expanded to {len(state.reranked_chunks)} total chunks")
        
    except Exception as e:
        print(f"  ⚠️ Multi-hop expansion failed: {e}")
    
    return state


def cluster_expansion_node(state: AgentState) -> AgentState:
    """PASS 4: Cluster-Based Expansion"""
    state.current_node = "cluster_expansion"
    
    try:
        print(f"\n[PASS 4] Cluster Expansion")
        
        cluster_manager = get_cluster_manager()
        
        if not cluster_manager.chunk_to_cluster:
            print("  ⚠️ No clusters available")
            return state
        
        chunk_ids = [rc.chunk.chunk_id for rc in state.reranked_chunks[:10]]
        neighbor_ids = cluster_manager.get_cluster_neighbors(chunk_ids, max_neighbors=5)
        
        print(f"  → Found {len(neighbor_ids)} cluster neighbors")
        
    except Exception as e:
        print(f"  ⚠️ Cluster expansion failed: {e}")
    
    return state


def context_assembly_node(state: AgentState, max_tokens: int = 2500) -> AgentState:
    """PASS 5: Compression & Assembly"""
    state.current_node = "context_assembly"
    
    if not state.reranked_chunks or not state.parsed_query:
        state.errors.append("Missing reranked chunks or query")
        return state
    
    try:
        print(f"\n[PASS 5] Context Compression & Assembly")
        
        compressor = get_compressor(
            target_tokens=int(max_tokens * 0.8),
            max_tokens=max_tokens
        )
        
        chunks_for_compression = []
        for rc in state.reranked_chunks:
            chunks_for_compression.append({
                'text': rc.chunk.content,
                'metadata': {
                    'chapter_title': rc.chunk.chapter,
                    'page_start': rc.chunk.page_number,
                    'contains_code': rc.chunk.chunk_type == 'code',
                    'book_title': rc.chunk.book_title,
                    'author': rc.chunk.author
                },
                'score': rc.rerank_score,
                'chunk_type': rc.chunk.chunk_type
            })
        
        compressed_context = compressor.compress_context(
            chunks_for_compression,
            state.parsed_query.raw_query,
            preserve_code=True
        )
        
        state.assembled_context = compressed_context
        state.system_prompt = _build_system_prompt(state.parsed_query)
        
    except Exception as e:
        state.errors.append(f"Context assembly failed: {str(e)}")
    
    return state


def _build_system_prompt(query: ParsedQuery) -> str:
    """Build system prompt based on intent and complexity"""
    base = """You are an expert programming tutor with deep knowledge of coding books and technical documentation.

Your role:
- Guide learners through programming concepts
- Provide clear explanations with relevant examples from the books
- Generate new and accurate code based on book examples and best practices
- Make sure to explain important keywords present in the answer with sufficient length
- Give verbose answers explaining everything and give brief explanations of the code
- The users will be someone having no knowledge about the topics, so explain everything that can be confusing for a new user
- ALWAYS mention the book title when referencing examples or concepts from specific books
- If the user asks for code, give more priority to code. dont add too much comments in code, add short and precise comments

Always reference sources WITH BOOK TITLES and ensure code is correct and follows best practices."""
    
    intent_prompts = {
        QueryIntent.CODE_REQUEST: "\n\n**Focus**: Provide working, well-commented code with explanations and cite the source book.",
        QueryIntent.CONCEPTUAL: "\n\n**Focus**: Explain concepts clearly with examples and mention which books they come from.",
        QueryIntent.COMPARISON: "\n\n**Focus**: Compare systematically with pros/cons, citing specific books.",
        QueryIntent.DEBUGGING: "\n\n**Focus**: Identify issues and provide fixes with book references.",
        QueryIntent.TUTORIAL: "\n\n**Focus**: Provide step-by-step guidance with book citations."
    }
    
    complexity = {
        "beginner": " Use simple language and basic examples.",
        "intermediate": " Balance theory and practice.",
        "advanced": " Include technical details and edge cases."
    }
    
    return base + intent_prompts.get(query.intent, "") + complexity.get(query.complexity_hint, "")


# ============================================================================
# UPDATED: LLM REASONING NODE WITH LANGCHAIN & GEMINI
# ============================================================================

def llm_reasoning_node(state: AgentState) -> AgentState:
    """Call Gemini LLM via LangChain for answer generation"""
    state.current_node = "llm_reasoning"
    
    if not state.reranked_chunks or not state.parsed_query:
        state.errors.append("Missing context or query for LLM")
        return state
    
    if not state.assembled_context:
        state = context_assembly_node(state, max_tokens=2500)
    
    try:
        print(f"\n[FINAL] LLM Reasoning")
        
        # Prepare messages for LangChain
        messages = [
            SystemMessage(content=state.system_prompt),
            HumanMessage(content=f"{state.assembled_context}\n\nQuestion: {state.parsed_query.raw_query}")
        ]
        
        # Invoke Gemini via LangChain
        response = llm.invoke(messages)
        answer = response.content
        
        chunks = state.reranked_chunks
        code_snippets = [c.chunk.content for c in chunks if c.chunk.chunk_type == "code"]
        
        state.response = LLMResponse(
            answer=answer,
            code_snippets=code_snippets[:2],
            sources=[c.chunk.chunk_id for c in chunks[:3]],
            confidence=0.85
        )
        
        print(f"  ✅ Answer generated ({len(answer)} chars)")
        
    except Exception as e:
        state.errors.append(f"LLM reasoning failed: {str(e)}")
        if state.reranked_chunks:
            chunks = state.reranked_chunks
            fallback_answer = f"Error calling LLM. Here's the retrieved content:\n\n"
            fallback_answer += f"From {chunks[0].chunk.book_title} - {chunks[0].chunk.chapter}:\n"
            fallback_answer += chunks[0].chunk.content[:500] + "..."
            
            state.response = LLMResponse(
                answer=fallback_answer,
                code_snippets=[],
                sources=[c.chunk.chunk_id for c in chunks[:3]],
                confidence=0.5
            )
    
    return state