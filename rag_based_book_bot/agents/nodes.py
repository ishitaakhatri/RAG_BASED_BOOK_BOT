"""
Updated Node implementations with LLM-based query parsing

CHANGES:
- Replaced hardcoded user_query_node with LLM-based parsing
- More accurate intent detection
- Better topic and keyword extraction
- Automatic programming language detection
"""

import re
import os
import json
from typing import List
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from openai import OpenAI

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

# Global instances (lazy loading)
_pc = None
_index = None
_model = None
_cross_encoder = None
_multi_hop = None
_cluster_manager = None
_compressor = None
_llm_client = None

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

def get_llm_client():
    """Get cached OpenAI client for query parsing"""
    global _llm_client
    if _llm_client is None:
        _llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _llm_client


# ============================================================================
# UPDATED: LLM-BASED QUERY PARSING NODE
# ============================================================================

def user_query_node(state: AgentState) -> AgentState:
    """
    ✨ NEW: LLM-based query parsing for intelligent intent detection
    
    Uses GPT-4 to understand:
    - User's learning intent (conceptual, coding, debugging, etc.)
    - Key technical topics to search for
    - Programming language context
    - Appropriate complexity level
    
    Falls back to heuristics if LLM fails.
    """
    state.current_node = "user_query"
    
    if not state.user_query:
        state.errors.append("No user query provided")
        return state
    
    try:
        print(f"\n[Query Parsing] Analyzing: '{state.user_query[:60]}...'")
        
        # 🔥 NEW: Use LLM for intelligent parsing
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


def _parse_query_with_llm(query: str) -> dict:
    """
    🔥 NEW: Intelligent query parsing using GPT-4
    
    This replaces all hardcoded heuristics with LLM understanding.
    """
    client = get_llm_client()
    
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

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
        max_tokens=400,
        response_format={"type": "json_object"}
    )
    
    parsed = json.loads(response.choices[0].message.content)
    
    # Validate and set defaults
    return {
        'intent': parsed.get('intent', 'CONCEPTUAL'),
        'topics': parsed.get('topics', []),
        'keywords': parsed.get('keywords', []),
        'code_language': parsed.get('code_language'),
        'complexity_hint': parsed.get('complexity_hint', 'intermediate')
    }


def _fallback_parse_query(query: str) -> ParsedQuery:
    """
    Fallback parser using simple heuristics.
    Used when LLM parsing fails.
    """
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
# RETRIEVAL NODES (unchanged from original)
# ============================================================================

def vector_search_node(state: AgentState, top_k: int = 50) -> AgentState:
    """PASS 1: Coarse Semantic Search"""
    state.current_node = "vector_search"
    
    if not state.parsed_query:
        state.errors.append("Missing query for search")
        return state
    
    try:
        print(f"\n[PASS 1] Vector Search (top_k={top_k})")
        
        index = get_pinecone_index()
        model = get_embedding_model()
        
        query_embedding = model.encode(state.parsed_query.raw_query).tolist()
        
        filter_dict = {}
        if hasattr(state, 'book_filter') and state.book_filter:
            filter_dict["book_title"] = state.book_filter
        if hasattr(state, 'chapter_filter') and state.chapter_filter:
            filter_dict["chapter_numbers"] = {"$in": [state.chapter_filter]}
        
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=NAMESPACE,
            filter=filter_dict if filter_dict else None,
            include_metadata=True
        )
        
        retrieved_chunks = []
        
        for match in results.get("matches", []):
            metadata = match.get("metadata", {})
            
            chapter_titles = metadata.get("chapter_titles", [])
            chapter_numbers = metadata.get("chapter_numbers", [])
            section_titles = metadata.get("section_titles", [])
            
            chapter_title = chapter_titles[0] if chapter_titles else ""
            chapter_number = chapter_numbers[0] if chapter_numbers else ""
            
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
        print(f"  → Retrieved {len(retrieved_chunks)} candidates")
        
    except Exception as e:
        state.errors.append(f"Vector search failed: {str(e)}")
    
    return state


def reranking_node(state: AgentState, top_k: int = 15) -> AgentState:
    """PASS 2: Cross-Encoder Reranking"""
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
                    'chapter': rc.chunk.chapter,
                    'page': rc.chunk.page_number,
                    'type': rc.chunk.chunk_type
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
                chapter=chunk_data['metadata']['chapter'],
                section="",
                page_number=chunk_data['metadata']['page'],
                chunk_type=chunk_data['metadata']['type']
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
    """PASS 3: Multi-Hop Retrieval"""
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
                'score': rc.rerank_score
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
            
            return [
                {
                    'id': m['id'],
                    'text': m.get('metadata', {}).get('text', ''),
                    'score': m.get('score', 0)
                }
                for m in results.get('matches', [])
            ]
        
        expanded_results = expander.multi_hop_retrieve(
            state.parsed_query.raw_query,
            initial_results,
            retrieval_fn,
            max_hops=max_hops,
            top_k_per_hop=3
        )
        
        for exp_result in expanded_results[len(initial_results):]:
            chunk = DocumentChunk(
                chunk_id=exp_result['id'],
                content=exp_result['text'],
                chapter="Multi-hop Result",
                section="",
                chunk_type="text"
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


def context_assembly_node(state: AgentState, max_tokens) -> AgentState:
    """PASS 5: Compression & Assembly"""
    state.current_node = "context_assembly"
    
    if not state.reranked_chunks or not state.parsed_query:
        state.errors.append("Missing reranked chunks or query")
        return state
    
    try:
        print(f"\n[PASS 5] Context Compression & Assembly")
        
        from rag_based_book_bot.retrieval.context_compressor import EnhancedContextCompressor
        
        compressor = EnhancedContextCompressor(
            target_tokens=int(max_tokens * 0.8),
            max_tokens=max_tokens,
            semantic_threshold=0.92
        )
        
        chunks_for_compression = []
        for rc in state.reranked_chunks:
            chunks_for_compression.append({
                'text': rc.chunk.content,
                'metadata': {
                    'chapter_title': rc.chunk.chapter,
                    'page_start': rc.chunk.page_number,
                    'contains_code': rc.chunk.chunk_type == 'code'
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
- Generate accurate code based on book examples and best practices
- Cite sources using [SOURCE N] notation

Always reference sources and ensure code is correct and follows best practices."""
    
    intent_prompts = {
        QueryIntent.CODE_REQUEST: "\n\n**Focus**: Provide working, well-commented code with explanations.",
        QueryIntent.CONCEPTUAL: "\n\n**Focus**: Explain concepts clearly with examples.",
        QueryIntent.COMPARISON: "\n\n**Focus**: Compare systematically with pros/cons.",
        QueryIntent.DEBUGGING: "\n\n**Focus**: Identify issues and provide fixes.",
        QueryIntent.TUTORIAL: "\n\n**Focus**: Provide step-by-step guidance."
    }
    
    complexity = {
        "beginner": " Use simple language and basic examples.",
        "intermediate": " Balance theory and practice.",
        "advanced": " Include technical details and edge cases."
    }
    
    return base + intent_prompts.get(query.intent, "") + complexity.get(query.complexity_hint, "")


def llm_reasoning_node(state: AgentState) -> AgentState:
    """Call OpenAI LLM for answer generation"""
    state.current_node = "llm_reasoning"
    
    if not state.reranked_chunks or not state.parsed_query:
        state.errors.append("Missing context or query for LLM")
        return state
    
    if not state.assembled_context:
        state = context_assembly_node(state, max_tokens=2500)
    
    try:
        client = get_llm_client()
        
        print(f"\n[FINAL] LLM Reasoning")
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": state.system_prompt},
                {"role": "user", "content": f"{state.assembled_context}\n\nQuestion: {state.parsed_query.raw_query}"}
            ],
            temperature=0.7,
            max_tokens=2048
        )
        
        answer = response.choices[0].message.content
        
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
            fallback_answer += f"From {chunks[0].chunk.chapter}:\n"
            fallback_answer += chunks[0].chunk.content[:500] + "..."
            
            state.response = LLMResponse(
                answer=fallback_answer,
                code_snippets=[],
                sources=[c.chunk.chunk_id for c in chunks[:3]],
                confidence=0.5
            )
    
    return state