"""
Conversation storage using Pinecone vector database

Stores conversation turns as vectors for semantic search and retrieval.
Each turn is embedded and stored with metadata for context resolution.
"""

from typing import List, Dict, Optional, Tuple
from pinecone import Pinecone
import os
import time
from dotenv import load_dotenv, find_dotenv

from .embedding_utils import (
    embed_conversation_turn,
    embed_query_for_search,
    format_turn_for_embedding
)

load_dotenv(find_dotenv(), override=True)

# Pinecone configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "coding-books")
NAMESPACE_CONVERSATIONS = "conversations"

# Global Pinecone instances (lazy loading)
_pc = None
_index = None


def get_pinecone_index():
    """Get Pinecone index (lazy initialization)"""
    global _pc, _index
    if _index is None:
        _pc = Pinecone(api_key=PINECONE_API_KEY)
        _index = _pc.Index(INDEX_NAME)
    return _index


def save_conversation_turn(
    session_id: str,
    turn_number: int,
    user_query: str,
    assistant_response: str,
    resolved_query: Optional[str] = None,
    needs_retrieval: bool = True,
    referenced_turn: Optional[int] = None,
    sources_used: Optional[List[str]] = None,
    user_id: Optional[str] = None
) -> Dict:
    """
    Save a conversation turn to Pinecone
    
    Args:
        session_id: Unique session identifier
        turn_number: Turn number in conversation (1, 2, 3, ...)
        user_query: Original user question
        assistant_response: Assistant's answer
        resolved_query: Standalone query after context resolution
        needs_retrieval: Whether retrieval was needed
        referenced_turn: Which previous turn was referenced (if any)
        sources_used: List of chunk IDs used as sources
        user_id: Optional user identifier
    
    Returns:
        Dict with save confirmation
    """
    try:
        index = get_pinecone_index()
        
        # Create embedding for Q+A pair
        embedding = embed_conversation_turn(user_query, assistant_response)
        
        # Create vector ID
        vector_id = f"{session_id}_turn{turn_number}"
        
        # Build metadata
        metadata = {
            # Session tracking
            "session_id": session_id,
            "turn_number": turn_number,
            "timestamp": time.time(),
            
            # Conversation content
            "user_query": user_query,
            "assistant_response": assistant_response[:1000],  # Limit to 1000 chars for metadata
            
            # Context resolution
            "resolved_query": resolved_query or user_query,
            "needs_retrieval": needs_retrieval,
            
            # For search and filtering
            "combined_text": format_turn_for_embedding(user_query, assistant_response, max_response_length=500)
        }
        
        # Optional fields
        if referenced_turn is not None:
            metadata["referenced_turn"] = referenced_turn
        
        if sources_used:
            metadata["sources_used"] = ",".join(sources_used[:5])  # Limit to 5 sources
        
        if user_id:
            metadata["user_id"] = user_id
        
        # Upsert to Pinecone
        index.upsert(
            vectors=[{
                "id": vector_id,
                "values": embedding,
                "metadata": metadata
            }],
            namespace=NAMESPACE_CONVERSATIONS
        )
        
        print(f"✅ Saved conversation turn: {vector_id}")
        
        return {
            "success": True,
            "vector_id": vector_id,
            "session_id": session_id,
            "turn_number": turn_number
        }
        
    except Exception as e:
        print(f"❌ Failed to save conversation turn: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def load_conversation(session_id: str, max_turns: int = 10) -> List[Dict]:
    """
    Load conversation history for a session
    
    Args:
        session_id: Session identifier
        max_turns: Maximum number of turns to load
    
    Returns:
        List of conversation turns, ordered by turn_number
    """
    try:
        index = get_pinecone_index()
        
        # Query Pinecone with a dummy vector to get all matches for session
        # (We use metadata filtering, so the query vector doesn't matter)
        dummy_vector = [0.0] * 384
        
        results = index.query(
            vector=dummy_vector,
            filter={"session_id": session_id},
            top_k=max_turns,
            namespace=NAMESPACE_CONVERSATIONS,
            include_metadata=True
        )
        
        # Extract turns from results
        turns = []
        for match in results.get("matches", []):
            metadata = match.get("metadata", {})
            turns.append({
                "turn_number": metadata.get("turn_number", 0),
                "user_query": metadata.get("user_query", ""),
                "assistant_response": metadata.get("assistant_response", ""),
                "resolved_query": metadata.get("resolved_query"),
                "needs_retrieval": metadata.get("needs_retrieval", True),
                "referenced_turn": metadata.get("referenced_turn"),
                "timestamp": metadata.get("timestamp", 0),
                "sources_used": metadata.get("sources_used", "").split(",") if metadata.get("sources_used") else []
            })
        
        # Sort by turn number
        turns.sort(key=lambda x: x["turn_number"])
        
        print(f"📥 Loaded {len(turns)} turns for session {session_id}")
        
        return turns
        
    except Exception as e:
        print(f"❌ Failed to load conversation: {e}")
        return []


def search_conversation_context(
    session_id: str,
    query: str,
    top_k: int = 3
) -> List[Dict]:
    """
    Semantic search over conversation history
    
    Finds most relevant past turns for a given query using vector similarity.
    
    Args:
        session_id: Session to search within
        query: Search query (e.g., "What did you say about advantages?")
        top_k: Number of relevant turns to return
    
    Returns:
        List of relevant conversation turns with similarity scores
    """
    try:
        index = get_pinecone_index()
        
        # Embed the search query
        query_embedding = embed_query_for_search(query)
        
        # Search in Pinecone with session filter
        results = index.query(
            vector=query_embedding,
            filter={"session_id": session_id},
            top_k=top_k,
            namespace=NAMESPACE_CONVERSATIONS,
            include_metadata=True
        )
        
        # Extract relevant turns
        relevant_turns = []
        for match in results.get("matches", []):
            metadata = match.get("metadata", {})
            relevant_turns.append({
                "turn_number": metadata.get("turn_number", 0),
                "user_query": metadata.get("user_query", ""),
                "assistant_response": metadata.get("assistant_response", ""),
                "resolved_query": metadata.get("resolved_query"),
                "timestamp": metadata.get("timestamp", 0),
                "relevance_score": match.get("score", 0.0),
                "sources_used": metadata.get("sources_used", "").split(",") if metadata.get("sources_used") else []
            })
        
        print(f"🔍 Found {len(relevant_turns)} relevant turns for query: '{query[:50]}...'")
        
        return relevant_turns
        
    except Exception as e:
        print(f"❌ Failed to search conversation context: {e}")
        return []


def get_session_turns(session_id: str) -> List[Dict]:
    """
    Get all turns for a session (alias for load_conversation with no limit)
    
    Args:
        session_id: Session identifier
    
    Returns:
        All conversation turns for the session
    """
    return load_conversation(session_id, max_turns=100)


def delete_session(session_id: str) -> Dict:
    """
    Delete all conversation turns for a session
    
    Args:
        session_id: Session to delete
    
    Returns:
        Deletion confirmation
    """
    try:
        index = get_pinecone_index()
        
        # Get all vector IDs for this session
        dummy_vector = [0.0] * 384
        results = index.query(
            vector=dummy_vector,
            filter={"session_id": session_id},
            top_k=100,
            namespace=NAMESPACE_CONVERSATIONS,
            include_metadata=True
        )
        
        # Extract IDs
        vector_ids = [match["id"] for match in results.get("matches", [])]
        
        if vector_ids:
            # Delete vectors
            index.delete(
                ids=vector_ids,
                namespace=NAMESPACE_CONVERSATIONS
            )
            print(f"🗑️ Deleted {len(vector_ids)} turns for session {session_id}")
        
        return {
            "success": True,
            "session_id": session_id,
            "deleted_turns": len(vector_ids)
        }
        
    except Exception as e:
        print(f"❌ Failed to delete session: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def get_conversation_stats(session_id: str) -> Dict:
    """
    Get statistics for a conversation session
    
    Args:
        session_id: Session identifier
    
    Returns:
        Stats including total turns, first/last interaction time, etc.
    """
    turns = get_session_turns(session_id)
    
    if not turns:
        return {
            "session_id": session_id,
            "total_turns": 0,
            "exists": False
        }
    
    timestamps = [t.get("timestamp", 0) for t in turns]
    
    return {
        "session_id": session_id,
        "total_turns": len(turns),
        "exists": True,
        "first_interaction": min(timestamps) if timestamps else 0,
        "last_interaction": max(timestamps) if timestamps else 0,
        "retrieval_turns": sum(1 for t in turns if t.get("needs_retrieval", True)),
        "history_turns": sum(1 for t in turns if not t.get("needs_retrieval", True))
    }
