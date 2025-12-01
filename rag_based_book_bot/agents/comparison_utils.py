"""
Create this new file: rag_based_book_bot/agents/comparison_utils.py

Utility functions for comparing pipeline results
"""

from typing import Dict, List, Set, Tuple
from rag_based_book_bot.agents.states import AgentState, RetrievedChunk


def calculate_chunk_overlap(
    chunks_a: List[RetrievedChunk],
    chunks_b: List[RetrievedChunk]
) -> Dict:
    """
    Calculate overlap between two sets of chunks.
    
    Returns:
        Dict with overlap statistics
    """
    ids_a = {chunk.chunk.chunk_id for chunk in chunks_a}
    ids_b = {chunk.chunk.chunk_id for chunk in chunks_b}
    
    overlap = ids_a & ids_b
    unique_to_a = ids_a - ids_b
    unique_to_b = ids_b - ids_a
    
    return {
        'total_a': len(ids_a),
        'total_b': len(ids_b),
        'overlap_count': len(overlap),
        'overlap_ids': list(overlap),
        'unique_to_a_count': len(unique_to_a),
        'unique_to_a_ids': list(unique_to_a),
        'unique_to_b_count': len(unique_to_b),
        'unique_to_b_ids': list(unique_to_b),
        'overlap_percentage': round(len(overlap) / max(len(ids_a), 1) * 100, 1)
    }


def get_rank_changes(
    chunks_a: List[RetrievedChunk],
    chunks_b: List[RetrievedChunk]
) -> List[Dict]:
    """
    Find chunks that appear in both lists and track rank changes.
    
    Returns:
        List of dicts with chunk_id, rank_a, rank_b, change
    """
    # Build rank maps
    rank_map_a = {chunk.chunk.chunk_id: i for i, chunk in enumerate(chunks_a)}
    rank_map_b = {chunk.chunk.chunk_id: i for i, chunk in enumerate(chunks_b)}
    
    # Find common chunks
    common_ids = set(rank_map_a.keys()) & set(rank_map_b.keys())
    
    rank_changes = []
    for chunk_id in common_ids:
        rank_a = rank_map_a[chunk_id]
        rank_b = rank_map_b[chunk_id]
        change = rank_a - rank_b  # Positive = moved up in B
        
        rank_changes.append({
            'chunk_id': chunk_id,
            'rank_baseline': rank_a,
            'rank_reranked': rank_b,
            'change': change,
            'moved_up': change > 0
        })
    
    # Sort by absolute change (biggest movers first)
    rank_changes.sort(key=lambda x: abs(x['change']), reverse=True)
    
    return rank_changes


def get_score_statistics(chunks: List[RetrievedChunk]) -> Dict:
    """Get statistical summary of chunk scores."""
    if not chunks:
        return {
            'count': 0,
            'avg_similarity': 0,
            'avg_rerank': 0,
            'avg_relevance': 0,
            'min_score': 0,
            'max_score': 0
        }
    
    similarities = [c.similarity_score for c in chunks]
    rerank_scores = [c.rerank_score for c in chunks]
    relevances = [c.relevance_percentage for c in chunks]
    
    return {
        'count': len(chunks),
        'avg_similarity': round(sum(similarities) / len(similarities), 3),
        'avg_rerank': round(sum(rerank_scores) / len(rerank_scores), 3),
        'avg_relevance': round(sum(relevances) / len(relevances), 1),
        'min_score': round(min(similarities), 3),
        'max_score': round(max(similarities), 3)
    }


def compare_pipeline_results(
    state_baseline: AgentState,
    state_reranked: AgentState
) -> Dict:
    """
    Comprehensive comparison of two pipeline results.
    
    Args:
        state_baseline: State from baseline pipeline (no reranking)
        state_reranked: State from reranked pipeline
    
    Returns:
        Dict with all comparison metrics
    """
    # Get chunks for comparison
    chunks_baseline = state_baseline.reranked_chunks[:15]
    chunks_reranked = state_reranked.reranked_chunks[:15]
    
    # Calculate metrics
    overlap = calculate_chunk_overlap(chunks_baseline, chunks_reranked)
    rank_changes = get_rank_changes(chunks_baseline, chunks_reranked)
    stats_baseline = get_score_statistics(chunks_baseline)
    stats_reranked = get_score_statistics(chunks_reranked)
    
    # Answer comparison
    answer_baseline = state_baseline.response.answer if state_baseline.response else ""
    answer_reranked = state_reranked.response.answer if state_reranked.response else ""
    
    return {
        'overlap': overlap,
        'rank_changes': rank_changes,
        'stats_baseline': stats_baseline,
        'stats_reranked': stats_reranked,
        'answers': {
            'baseline_length': len(answer_baseline),
            'reranked_length': len(answer_reranked),
            'baseline_answer': answer_baseline,
            'reranked_answer': answer_reranked
        },
        'summary': {
            'chunks_different': overlap['unique_to_b_count'],
            'biggest_rank_change': rank_changes[0]['change'] if rank_changes else 0,
            'avg_score_improvement': round(
                stats_reranked['avg_relevance'] - stats_baseline['avg_relevance'], 1
            )
        }
    }