"""
Cross-Encoder Reranking Node
Uses cross-encoder model for precise relevance scoring
"""
import logging
from typing import List
from sentence_transformers import CrossEncoder

from rag_based_book_bot.agents.states import AgentState, RetrievedChunk

logger = logging.getLogger("cross_encoder_reranker")

# Global cross-encoder (lazy initialization)
_cross_encoder = None


def get_cross_encoder():
    """Lazy load cross-encoder model"""
    global _cross_encoder
    if _cross_encoder is None:
        logger.info("Loading cross-encoder model: cross-encoder/ms-marco-MiniLM-L-6-v2")
        _cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        logger.info("✅ Cross-encoder loaded")
    return _cross_encoder


def cross_encoder_rerank_node(state: AgentState, top_k: int = 15) -> AgentState:
    """
    PASS 2: Cross-Encoder Reranking
    
    Takes broad vector search results and re-scores them using a cross-encoder
    for true semantic relevance.
    
    Args:
        state: Current agent state with retrieved_chunks
        top_k: Number of chunks to keep after reranking (default: 15)
    
    Returns:
        Updated state with reranked chunks
    """
    state.current_node = "cross_encoder_rerank"
    
    if not state.retrieved_chunks or not state.parsed_query:
        state.errors.append("Missing chunks or query for cross-encoder reranking")
        return state
    
    try:
        logger.info(f"🔄 Cross-encoder reranking {len(state.retrieved_chunks)} chunks...")
        
        # Get cross-encoder model
        cross_encoder = get_cross_encoder()
        
        # Prepare query-chunk pairs
        query = state.parsed_query.raw_query
        pairs = [
            (query, chunk.chunk.content)
            for chunk in state.retrieved_chunks
        ]
        
        # Score all pairs in batch (MUCH faster than one-by-one)
        logger.info(f"   Scoring {len(pairs)} query-chunk pairs...")
        scores = cross_encoder.predict(pairs)
        
        # Update chunks with cross-encoder scores
        for chunk, score in zip(state.retrieved_chunks, scores):
            chunk.rerank_score = float(score)
        
        # Sort by cross-encoder score
        state.retrieved_chunks.sort(key=lambda x: x.rerank_score, reverse=True)
        
        # Keep top-k
        state.retrieved_chunks = state.retrieved_chunks[:top_k]
        
        # Calculate relevance percentages
        if state.retrieved_chunks:
            max_score = state.retrieved_chunks[0].rerank_score
            for chunk in state.retrieved_chunks:
                chunk.relevance_percentage = round((chunk.rerank_score / max_score) * 100, 1)
        
        logger.info(f"✅ Cross-encoder reranking complete. Top {len(state.retrieved_chunks)} chunks selected")
        
        # Debug output
        if state.retrieved_chunks:
            logger.info(f"   Top score: {state.retrieved_chunks[0].rerank_score:.4f}")
            logger.info(f"   Bottom score: {state.retrieved_chunks[-1].rerank_score:.4f}")
        
    except Exception as e:
        state.errors.append(f"Cross-encoder reranking failed: {str(e)}")
        logger.error(f"❌ Cross-encoder error: {e}")
    
    return state


def score_query_chunk_pairs(query: str, chunks: List[str]) -> List[float]:
    """
    Utility function to score query-chunk pairs.
    Can be used standalone for testing.
    
    Args:
        query: User query
        chunks: List of chunk texts
    
    Returns:
        List of scores (0-1, higher = more relevant)
    """
    cross_encoder = get_cross_encoder()
    pairs = [(query, chunk) for chunk in chunks]
    scores = cross_encoder.predict(pairs)
    return scores.tolist()