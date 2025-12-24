"""
Embedding utilities for conversation turns

Handles creating embeddings from question-answer pairs for semantic search
"""

from typing import List, Dict
from sentence_transformers import SentenceTransformer
import os

# Use same embedding model as document chunks for consistency
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

# Lazy load embedding model
_embedding_model = None


def get_embedding_model() -> SentenceTransformer:
    """Get or initialize embedding model (lazy loading)"""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def format_turn_for_embedding(user_query: str, assistant_response: str, max_response_length: int = 500) -> str:
    """
    Format a conversation turn for embedding
    
    Combines query and response in a structured format optimized for semantic search.
    
    Args:
        user_query: User's question
        assistant_response: Assistant's answer
        max_response_length: Max chars from response to include (to avoid huge embeddings)
    
    Returns:
        Formatted text for embedding
    
    Example:
        Input: "What is CNN?", "Convolutional Neural Networks are..."
        Output: "Q: What is CNN?\nA: Convolutional Neural Networks are..."
    """
    # Truncate response if too long (keep most relevant part - beginning)
    truncated_response = assistant_response[:max_response_length]
    if len(assistant_response) > max_response_length:
        truncated_response += "..."
    
    # Format as Q&A pair
    formatted_text = f"Q: {user_query}\nA: {truncated_response}"
    
    return formatted_text


def embed_conversation_turn(user_query: str, assistant_response: str) -> List[float]:
    """
    Create embedding vector for a conversation turn
    
    Args:
        user_query: User's question
        assistant_response: Assistant's answer
    
    Returns:
        384-dimensional embedding vector
    
    Example:
        >>> embed_conversation_turn("What is CNN?", "Convolutional Neural Networks...")
        [0.234, 0.567, -0.123, ...]  # 384 dimensions
    """
    model = get_embedding_model()
    
    # Format turn for embedding
    text = format_turn_for_embedding(user_query, assistant_response)
    
    # Generate embedding
    embedding = model.encode(text).tolist()
    
    return embedding


def batch_embed_turns(turns: List[Dict[str, str]]) -> List[List[float]]:
    """
    Embed multiple conversation turns in batch (more efficient)
    
    Args:
        turns: List of dicts with 'user_query' and 'assistant_response' keys
    
    Returns:
        List of embedding vectors
    
    Example:
        >>> turns = [
        ...     {"user_query": "What is CNN?", "assistant_response": "CNNs are..."},
        ...     {"user_query": "Show code", "assistant_response": "Here's code..."}
        ... ]
        >>> batch_embed_turns(turns)
        [[0.234, ...], [0.456, ...]]
    """
    model = get_embedding_model()
    
    # Format all turns
    texts = [
        format_turn_for_embedding(turn['user_query'], turn['assistant_response'])
        for turn in turns
    ]
    
    # Batch encode (faster than one-by-one)
    embeddings = model.encode(texts).tolist()
    
    return embeddings


def embed_query_for_search(query: str) -> List[float]:
    """
    Embed a search query to find relevant conversation turns
    
    Args:
        query: Search query (e.g., "What did you tell me about CNNs?")
    
    Returns:
        384-dimensional embedding vector
    """
    model = get_embedding_model()
    embedding = model.encode(query).tolist()
    return embedding
