"""
Updated Node implementations with full 5-pass retrieval.

PASSES:
1. Vector Search (coarse)
2. Cross-Encoder Reranking (precision)
3. Multi-Hop Expansion (cross-chapter)
4. Cluster Expansion (avoid chapter bias)
5. Compression (token management)
"""

import re
import os
from typing import List
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

from rag_based_book_bot.agents.states import (
    AgentState, DocumentChunk, RetrievedChunk, 
    ParsedQuery, QueryIntent, LLMResponse
)

# Import the new retrieval components
from rag_based_book_bot.retrieval.cross_encoder_reranker import CrossEncoderReranker
from rag_based_book_bot.retrieval.multi_hop_expander import MultiHopExpander
from rag_based_book_bot.retrieval.cluster_manager import ClusterManager
from rag_based_book_bot.retrieval.context_compressor import ContextCompressor

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
        # Try to load pre-built clusters
        # In production, you'd load based on book_id
        # cluster_manager.load("default_book")
    return _cluster_manager

def get_compressor(target_tokens=2000, max_tokens=4000):
    """Get context compressor with configurable token limits"""
    return ContextCompressor(
        target_tokens=target_tokens,
        max_tokens=max_tokens
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


# ============================================================================
# UPDATED NODES WITH 5-PASS INTEGRATION
# ============================================================================

def vector_search_node(state: AgentState, top_k: int = 50) -> AgentState:
    """
    PASS 1: Coarse Semantic Search
    
    Retrieve broad set of candidates (50-100).
    NOT directly passed to LLM - will be refined in later passes.
    """
    state.current_node = "vector_search"
    
    if not state.parsed_query:
        state.errors.append("Missing query for search")
        return state
    
    try:
        print(f"\n[PASS 1] Vector Search (top_k={top_k})")
        
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
        
        # Query Pinecone (broad search)
        results = index.query(
            vector=query_embedding,
            top_k=top_k,  # Higher than before!
            namespace=NAMESPACE,
            filter=filter_dict if filter_dict else None,
            include_metadata=True
        )
        
        # Convert to RetrievedChunk objects
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
    """
    PASS 2: Cross-Encoder Reranking
    
    Use BERT cross-encoder for true relevance scoring.
    Reduces 50 candidates → 15 high-precision results.
    """
    state.current_node = "reranking"
    
    if not state.retrieved_chunks or not state.parsed_query:
        state.errors.append("Missing chunks or query for reranking")
        return state
    
    try:
        print(f"\n[PASS 2] Cross-Encoder Reranking (top_k={top_k})")
        
        cross_encoder = get_cross_encoder()
        
        # Prepare chunks for reranking
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
        
        # Rerank with cross-encoder
        reranked = cross_encoder.rerank_with_metadata(
            state.parsed_query.raw_query,
            chunks_data,
            top_k=top_k
        )
        
        # Convert back to RetrievedChunk objects
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
        # Fallback: use original chunks
        state.reranked_chunks = state.retrieved_chunks[:top_k]
    
    return state


def multi_hop_expansion_node(state: AgentState, max_hops: int = 2) -> AgentState:
    """
    PASS 3: Multi-Hop Retrieval
    
    Extract concepts from top results, query again.
    Finds cross-chapter information.
    """
    state.current_node = "multi_hop_expansion"
    
    if not state.reranked_chunks or not state.parsed_query:
        state.errors.append("Missing reranked chunks for multi-hop")
        return state
    
    try:
        print(f"\n[PASS 3] Multi-Hop Expansion (max_hops={max_hops})")
        
        expander = get_multi_hop_expander()
        
        # Convert to format expected by expander
        initial_results = []
        for rc in state.reranked_chunks[:10]:  # Top 10 for expansion
            initial_results.append({
                'id': rc.chunk.chunk_id,
                'text': rc.chunk.content,
                'score': rc.rerank_score
            })
        
        # Define retrieval function for secondary queries
        def retrieval_fn(query_text: str, top_k: int = 5):
            """Mini retrieval for secondary queries"""
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
        
        # Perform multi-hop expansion
        expanded_results = expander.multi_hop_retrieve(
            state.parsed_query.raw_query,
            initial_results,
            retrieval_fn,
            max_hops=max_hops,
            top_k_per_hop=3
        )
        
        # Add expanded results to state (mark as secondary)
        for exp_result in expanded_results[len(initial_results):]:  # Only new ones
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
                rerank_score=exp_result['score'] * 0.8,  # Slightly lower weight
                relevance_percentage=exp_result['score'] * 80
            ))
        
        print(f"  → Expanded to {len(state.reranked_chunks)} total chunks")
        
    except Exception as e:
        print(f"  ⚠️ Multi-hop expansion failed: {e}")
        # Continue without expansion
    
    return state


def cluster_expansion_node(state: AgentState) -> AgentState:
    """
    PASS 4: Cluster-Based Expansion
    
    Find semantically similar chunks from same clusters.
    Avoids chapter-name bias.
    """
    state.current_node = "cluster_expansion"
    
    # Note: This requires clusters to be built during ingestion
    # For now, we'll skip if clusters aren't available
    
    try:
        print(f"\n[PASS 4] Cluster Expansion")
        
        cluster_manager = get_cluster_manager()
        
        # Check if clusters are available
        if not cluster_manager.chunk_to_cluster:
            print("  ⚠️ No clusters available (run cluster building during ingestion)")
            return state
        
        # Get chunk IDs from current results
        chunk_ids = [rc.chunk.chunk_id for rc in state.reranked_chunks[:10]]
        
        # Find cluster neighbors
        neighbor_ids = cluster_manager.get_cluster_neighbors(
            chunk_ids,
            max_neighbors=5
        )
        
        # Retrieve neighbor chunks (would need lookup in production)
        print(f"  → Found {len(neighbor_ids)} cluster neighbors")
        
    except Exception as e:
        print(f"  ⚠️ Cluster expansion failed: {e}")
    
    return state


def context_assembly_node(state: AgentState,max_tokens) -> AgentState:
    """
    PASS 5: Compression & Assembly
    
    Deduplicate, summarize, compress context to fit token budget.
    """
    state.current_node = "context_assembly"
    
    if not state.reranked_chunks or not state.parsed_query:
        state.errors.append("Missing reranked chunks or query")
        return state
    
    try:
        print(f"\n[PASS 5] Context Compression & Assembly")
        
        compressor = ContextCompressor(
            target_tokens=int(max_tokens * 0.8),  # Target 80% of max
            max_tokens=max_tokens
        )
        
        # Convert chunks to format for compressor
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
        
        # Compress context
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
        
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        print(f"\n[FINAL] LLM Reasoning")
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": state.system_prompt},
                {"role": "user", "content": f"{state.assembled_context}\n\nQuestion: {state.parsed_query.raw_query}"}
            ],
            temperature=0.7
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
        # Fallback response
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