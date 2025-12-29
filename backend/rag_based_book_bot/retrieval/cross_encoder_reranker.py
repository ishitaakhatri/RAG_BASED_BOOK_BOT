"""
Cross-Encoder Reranker (PASS 2)

Uses BERT-based cross-encoder to compute true relevance scores
between query and candidate chunks.

This is the REAL Pass 2 - not just keyword matching.
[REF] Centralized Configuration
"""

from typing import List, Tuple
from sentence_transformers import CrossEncoder
import numpy as np

# NEW: Import Config
from app_config import get_config
settings = get_config()


class CrossEncoderReranker:
    """
    True cross-encoder reranking using BERT.
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize cross-encoder model.
        
        Args:
            model_name: HuggingFace cross-encoder model name
        """
        # Use config default if not provided
        self.model_name = model_name or settings.retrieval.cross_encoder_model
        
        print(f"Loading cross-encoder: {self.model_name}")
        self.model = CrossEncoder(self.model_name, max_length=512)
        print("✅ Cross-encoder loaded")
    
    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, dict]],
        top_k: int = 10
    ) -> List[Tuple[str, dict, float]]:
        """
        Rerank candidates using cross-encoder.
        
        Args:
            query: User query text
            candidates: List of (chunk_text, metadata) tuples
            top_k: Number of top results to return
        
        Returns:
            List of (chunk_text, metadata, cross_encoder_score) tuples
        """
        if not candidates:
            return []
        
        # Prepare pairs for cross-encoder
        pairs = [[query, chunk_text] for chunk_text, _ in candidates]
        
        # Get cross-encoder scores (this is the magic!)
        scores = self.model.predict(pairs)
        
        # Combine with original data
        reranked = []
        for i, (chunk_text, metadata) in enumerate(candidates):
            reranked.append((chunk_text, metadata, float(scores[i])))
        
        # Sort by score (descending)
        reranked.sort(key=lambda x: x[2], reverse=True)
        
        # Return top-k
        return reranked[:top_k]
    
    def rerank_with_metadata(
        self,
        query: str,
        chunks_with_scores: List[dict],
        top_k: int = 10
    ) -> List[dict]:
        """
        Rerank chunks that already have similarity scores.
        
        Args:
            query: User query
            chunks_with_scores: List of dicts with 'text', 'metadata', 'similarity_score'
            top_k: Number to return
        
        Returns:
            Reranked list with added 'cross_encoder_score' field
        """
        if not chunks_with_scores:
            return []
        
        # Extract texts
        texts = [chunk['text'] for chunk in chunks_with_scores]
        
        # Prepare pairs
        pairs = [[query, text] for text in texts]
        
        # Score
        scores = self.model.predict(pairs)
        
        # Add scores to chunks
        for i, chunk in enumerate(chunks_with_scores):
            chunk['cross_encoder_score'] = float(scores[i])
            # Combined score (blend semantic + cross-encoder)
            chunk['final_score'] = (
                0.3 * chunk.get('similarity_score', 0) +
                0.7 * float(scores[i])
            )
        
        # Sort by final score
        chunks_with_scores.sort(key=lambda x: x['final_score'], reverse=True)
        
        return chunks_with_scores[:top_k]


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Initialize
    reranker = CrossEncoderReranker()
    
    # Example candidates from vector search
    query = "How to implement gradient descent in Python?"
    
    candidates = [
        ("Gradient descent is an optimization algorithm...", {"page": 10}),
        ("Neural networks use backpropagation...", {"page": 25}),
        ("To implement GD in Python: optimizer = SGD()...", {"page": 15}),
        ("Linear regression can be solved analytically...", {"page": 8}),
    ]
    
    # Rerank
    reranked = reranker.rerank(query, candidates, top_k=3)
    
    print("\n=== RERANKED RESULTS ===")
    for i, (text, metadata, score) in enumerate(reranked, 1):
        print(f"\n{i}. Score: {score:.4f}")
        print(f"   Page: {metadata['page']}")
        print(f"   Text: {text[:80]}...")