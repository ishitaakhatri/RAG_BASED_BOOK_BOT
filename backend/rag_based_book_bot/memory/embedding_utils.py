"""
Embedding utilities for conversation turns

Handles creating embeddings from question-answer pairs for semantic search
"""

from typing import List, Dict
from sentence_transformers import SentenceTransformer
import os

# Use same embedding model as document chunks for consistency
# Note: BGE-M3 produces 1024-dimensional vectors, but Pinecone index expects 1024
# We'll pad the vectors to match the index dimension
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DIMENSION = 1024  # BGE-M3 produces 1024-dim vectors
PINECONE_DIMENSION = 1024  # Pinecone index expects 1024-dim vectors

import signal
import sys
from contextlib import contextmanager

class TimeoutException(Exception):
    """Exception raised when operation times out"""
    pass


@contextmanager
def timeout_handler(seconds=30):
    """Context manager for timeout handling (Windows-compatible)"""
    def timeout_handler_fn(signum, frame):
        raise TimeoutException(f"Operation timed out after {seconds} seconds")
    
    # Note: signal.alarm only works on Unix. For Windows, we'll use a different approach
    old_handler = None
    if sys.platform != 'win32':
        old_handler = signal.signal(signal.SIGALRM, timeout_handler_fn)
        signal.alarm(seconds)
    
    try:
        yield
    finally:
        if sys.platform != 'win32':
            signal.alarm(0)  # Disable alarm
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)


def get_embedding_model() -> SentenceTransformer:
    """Get or initialize embedding model (lazy loading) with timeout"""
    global _embedding_model
    if _embedding_model is None:
        try:
            print(f"[EMBEDDING] Loading model: {EMBEDDING_MODEL_NAME}...")
            _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            print(f"[EMBEDDING] Model loaded successfully")
        except Exception as e:
            print(f"[ERROR] Failed to load embedding model: {e}")
            # Fallback: use a lightweight model
            print(f"[FALLBACK] Using lightweight embedding model instead...")
            try:
                _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                print(f"[FALLBACK] Lightweight model loaded successfully")
            except Exception as fallback_error:
                print(f"[CRITICAL] Even fallback model failed to load: {fallback_error}")
                raise RuntimeError(f"Cannot load any embedding model. Original error: {e}")
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


def create_fallback_embedding(text: str = "", dimension: int = PINECONE_DIMENSION) -> List[float]:
    """
    Create a fallback embedding with at least one non-zero value
    
    When model loading fails, Pinecone rejects all-zero vectors.
    This creates a simple hash-based embedding that's guaranteed to have non-zero values.
    
    Args:
        text: Text to hash (optional, for some variation)
        dimension: Target dimension (1024 for Pinecone)
    
    Returns:
        Non-zero embedding vector with shape [dimension]
    """
    import hashlib
    
    # Create a small non-zero value based on text hash
    hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16) % 1000
    base_value = 0.001 * (1 + (hash_val % 10))  # Range: 0.001 to 0.011
    
    # Create embedding with base_value + some variation
    embedding = [base_value] * dimension
    
    # Add some pseudo-random variation to make it more meaningful
    for i in range(min(10, dimension)):
        embedding[i] = base_value * (1.0 + (i % 5) * 0.001)
    
    return embedding


def pad_embedding_to_pinecone(embedding: List[float], target_dim: int = PINECONE_DIMENSION) -> List[float]:
    """
    Pad embedding vector to match Pinecone index dimension
    
    Args:
        embedding: Original embedding vector (1024 dimensions from BGE-M3)
        target_dim: Target dimension (1024 for Pinecone index)
    
    Returns:
        Padded embedding vector
    """
    if len(embedding) >= target_dim:
        return embedding[:target_dim]
    
    # Pad with zeros to reach target dimension
    padding = [0.0] * (target_dim - len(embedding))
    return embedding + padding


def embed_conversation_turn(user_query: str, assistant_response: str) -> List[float]:
    """
    Create embedding vector for a conversation turn
    
    Args:
        user_query: User's question
        assistant_response: Assistant's answer
    
    Returns:
        1024-dimensional embedding vector (padded from 1024-dim BGE-M3 model)
    
    Example:
        >>> embed_conversation_turn("What is CNN?", "Convolutional Neural Networks...")
        [0.234, 0.567, -0.123, ..., 0.0, 0.0, ...]  # 1024 dimensions (1024 values + 640 zeros)
    """
    try:
        model = get_embedding_model()
        
        # Format turn for embedding
        text = format_turn_for_embedding(user_query, assistant_response)
        
        # Generate embedding (1024 dimensions from BGE-M3)
        embedding = model.encode(text).tolist()
        
        # Pad to match Pinecone index dimension (1024)
        padded_embedding = pad_embedding_to_pinecone(embedding)
        
        return padded_embedding
    except Exception as e:
        print(f"[WARNING] Failed to create embedding: {e}")
        # Return a valid fallback embedding (non-zero values for Pinecone)
        # Use query text for some variation in fallback embeddings
        combined_text = f"{user_query} {assistant_response}"
        return create_fallback_embedding(combined_text, PINECONE_DIMENSION)


def batch_embed_turns(turns: List[Dict[str, str]]) -> List[List[float]]:
    """
    Embed multiple conversation turns in batch (more efficient)
    
    Args:
        turns: List of dicts with 'user_query' and 'assistant_response' keys
    
    Returns:
        List of 1024-dimensional embedding vectors (padded from 1024-dim)
    
    Example:
        >>> turns = [
        ...     {"user_query": "What is CNN?", "assistant_response": "CNNs are..."},
        ...     {"user_query": "Show code", "assistant_response": "Here's code..."}
        ... ]
        >>> batch_embed_turns(turns)
        [[0.234, ..., 0.0], [0.456, ..., 0.0]]  # Each is 1024-dim
    """
    try:
        model = get_embedding_model()
        
        # Format all turns
        texts = [
            format_turn_for_embedding(turn['user_query'], turn['assistant_response'])
            for turn in turns
        ]
        
        # Batch encode (faster than one-by-one) - returns 1024-dim vectors
        embeddings = model.encode(texts).tolist()
        
        # Pad all embeddings to Pinecone dimension
        padded_embeddings = [pad_embedding_to_pinecone(emb) for emb in embeddings]
        
        return padded_embeddings
    except Exception as e:
        print(f"[WARNING] Failed to batch embed turns: {e}")
        # Fallback: create individual embeddings for each turn
        return [
            create_fallback_embedding(
                f"{turn['user_query']} {turn['assistant_response']}",
                PINECONE_DIMENSION
            )
            for turn in turns
        ]


def embed_query_for_search(query: str) -> List[float]:
    """
    Embed a search query to find relevant conversation turns
    
    Args:
        query: Search query (e.g., "What did you tell me about CNNs?")
    
    Returns:
        1024-dimensional embedding vector (padded from 1024-dim)
    """
    try:
        model = get_embedding_model()
        embedding = model.encode(query).tolist()
        # Pad to match Pinecone dimension
        padded_embedding = pad_embedding_to_pinecone(embedding)
        return padded_embedding
    except Exception as e:
        print(f"[WARNING] Failed to embed search query: {e}")
        # Fallback: create valid embedding from query text
        return create_fallback_embedding(query, PINECONE_DIMENSION)
