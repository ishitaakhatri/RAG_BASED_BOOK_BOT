"""
Enhanced Conversation Storage with Full Session Management

Features:
- Persistent storage in Pinecone
- Session listing and metadata
- Cross-session search
- Robust error handling
- Automatic retry logic
"""

from typing import List, Dict, Optional
from pinecone import Pinecone
import os
import time
from dotenv import load_dotenv, find_dotenv
from functools import wraps
import logging

from .embedding_utils import (
    embed_conversation_turn,
    embed_query_for_search,
    format_turn_for_embedding
)

load_dotenv(find_dotenv(), override=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pinecone configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "coding-books")
NAMESPACE_CONVERSATIONS = "conversations"
NAMESPACE_SESSION_META = "session_metadata"

# Global instances (lazy loading)
_pc = None
_index = None


def retry_on_failure(max_retries=3, delay=1.0):
    """Decorator for retry logic on Pinecone operations"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"{func.__name__} failed after {max_retries} attempts: {e}")
                        raise
                    logger.warning(f"{func.__name__} attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(delay * (attempt + 1))
            return None
        return wrapper
    return decorator


def get_pinecone_index():
    """Get Pinecone index (lazy initialization with error handling)"""
    global _pc, _index
    if _index is None:
        try:
            if not PINECONE_API_KEY:
                raise ValueError("PINECONE_API_KEY not found in environment")
            _pc = Pinecone(api_key=PINECONE_API_KEY)
            _index = _pc.Index(INDEX_NAME)
            logger.info(f"✅ Connected to Pinecone index: {INDEX_NAME}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Pinecone: {e}")
            raise
    return _index


@retry_on_failure(max_retries=3)
def save_conversation_turn(
    session_id: str,
    turn_number: int,
    user_query: str,
    assistant_response: str,
    resolved_query: Optional[str] = None,
    needs_retrieval: bool = True,
    referenced_turn: Optional[int] = None,
    sources_used: Optional[List[Dict]] = None,
    user_id: Optional[str] = None
) -> Dict:
    """
    Save a conversation turn to Pinecone with retry logic
    
    Args:
        session_id: Unique session identifier
        turn_number: Turn number in conversation (1, 2, 3, ...)
        user_query: Original user question
        assistant_response: Assistant's answer
        resolved_query: Standalone query after context resolution
        needs_retrieval: Whether retrieval was needed
        referenced_turn: Which previous turn was referenced
        sources_used: List of chunk IDs used as sources
        user_id: Optional user identifier
    
    Returns:
        Dict with save confirmation
    """
    try:
        index = get_pinecone_index()
        
        # Create embedding
        embedding = embed_conversation_turn(user_query, assistant_response)
        
        # Vector ID
        vector_id = f"{session_id}_turn{turn_number}"
        
        # Build metadata (all strings, ints, or floats for Pinecone)
        metadata = {
            "session_id": session_id,
            "turn_number": int(turn_number),
            "timestamp": float(time.time()),
            "user_query": user_query,  # Limit length
            "assistant_response": assistant_response,
            "resolved_query": (resolved_query or user_query),
            "needs_retrieval": str(needs_retrieval),  # Convert bool to string
            "combined_text": format_turn_for_embedding(user_query, assistant_response, max_response_length=500)
        }
        
        # Optional fields
        if referenced_turn is not None:
            metadata["referenced_turn"] = int(referenced_turn)
        
        if sources_used:
            metadata["sources_used"] = ",".join(sources_used[:5])
        
        if user_id:
            metadata["user_id"] = str(user_id)
        
        # Upsert to Pinecone
        index.upsert(
            vectors=[{
                "id": vector_id,
                "values": embedding,
                "metadata": metadata
            }],
            namespace=NAMESPACE_CONVERSATIONS
        )
        
        # Update session metadata
        update_session_metadata(session_id, user_query, turn_number, user_id)
        
        logger.info(f"✅ Saved: {vector_id}")
        
        return {
            "success": True,
            "vector_id": vector_id,
            "session_id": session_id,
            "turn_number": turn_number
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to save turn: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@retry_on_failure(max_retries=3)
def load_conversation(session_id: str, max_turns: int = 10) -> List[Dict]:
    """
    Load conversation history for a session
    
    Args:
        session_id: Session identifier
        max_turns: Maximum turns to load
    
    Returns:
        List of conversation turns, ordered by turn_number
    """
    try:
        index = get_pinecone_index()
        
        # Query with session filter
        dummy_vector = [0.0] * 1024
        
        results = index.query(
            vector=dummy_vector,
            filter={"session_id": session_id},
            top_k=max_turns * 2,  # Fetch more to ensure we get all
            namespace=NAMESPACE_CONVERSATIONS,
            include_metadata=True
        )
        
        # Extract turns
        turns = []
        for match in results.get("matches", []):
            metadata = match.get("metadata", {})
            
            # Convert bool string back to bool
            needs_retrieval = metadata.get("needs_retrieval", "True") == "True"
            
            turns.append({
                "turn_number": int(metadata.get("turn_number", 0)),
                "user_query": metadata.get("user_query", ""),
                "assistant_response": metadata.get("assistant_response", ""),
                "resolved_query": metadata.get("resolved_query"),
                "needs_retrieval": needs_retrieval,
                "referenced_turn": metadata.get("referenced_turn"),
                "timestamp": float(metadata.get("timestamp", 0)),
                "sources_used": metadata.get("sources_used", "").split(",") if metadata.get("sources_used") else []
            })
        
        # Sort by turn number
        turns.sort(key=lambda x: x["turn_number"])
        
        # Limit to max_turns
        turns = turns[:max_turns]
        
        logger.info(f"📥 Loaded {len(turns)} turns for session {session_id}")
        
        return turns
        
    except Exception as e:
        logger.error(f"❌ Failed to load conversation: {e}")
        return []


@retry_on_failure(max_retries=3)
def search_conversation_context(
    session_id: str,
    query: str,
    top_k: int = 3
) -> List[Dict]:
    """
    Semantic search over conversation history
    
    Args:
        session_id: Session to search within
        query: Search query
        top_k: Number of relevant turns to return
    
    Returns:
        List of relevant conversation turns with similarity scores
    """
    try:
        index = get_pinecone_index()
        
        # Embed query
        query_embedding = embed_query_for_search(query)
        
        # Search with session filter
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
                "turn_number": int(metadata.get("turn_number", 0)),
                "user_query": metadata.get("user_query", ""),
                "assistant_response": metadata.get("assistant_response", ""),
                "resolved_query": metadata.get("resolved_query"),
                "timestamp": float(metadata.get("timestamp", 0)),
                "relevance_score": float(match.get("score", 0.0)),
                "sources_used": metadata.get("sources_used", "").split(",") if metadata.get("sources_used") else []
            })
        
        logger.info(f"🔍 Found {len(relevant_turns)} relevant turns")
        
        return relevant_turns
        
    except Exception as e:
        logger.error(f"❌ Search failed: {e}")
        return []


@retry_on_failure(max_retries=3)
def list_all_sessions(user_id: Optional[str] = None, limit: int = 50) -> List[Dict]:
    """
    List all conversation sessions
    
    Args:
        user_id: Optional filter by user
        limit: Maximum sessions to return
    
    Returns:
        List of session summaries
    """
    try:
        index = get_pinecone_index()
        
        # Query session metadata namespace
        dummy_vector = [1.0] * 1024
        
        filter_dict = {}
        if user_id:
            filter_dict["user_id"] = str(user_id)
        
        results = index.query(
            vector=dummy_vector,
            filter=filter_dict if filter_dict else None,
            top_k=limit,
            namespace=NAMESPACE_SESSION_META,
            include_metadata=True
        )
        
        sessions = []
        for match in results.get("matches", []):
            metadata = match.get("metadata", {})
            
            # Skip initialization vectors
            if metadata.get("session_id") == "__init__":
                continue
            
            sessions.append({
                "session_id": metadata.get("session_id", ""),
                "title": metadata.get("title", "Untitled"),
                "last_message": metadata.get("last_message", ""),
                "message_count": int(metadata.get("message_count", 0)),
                "created_at": float(metadata.get("created_at", 0)),
                "updated_at": float(metadata.get("updated_at", 0))
            })
        
        # Sort by updated_at (most recent first)
        sessions.sort(key=lambda x: x["updated_at"], reverse=True)
        
        logger.info(f"📋 Listed {len(sessions)} sessions")
        
        return sessions
        
    except Exception as e:
        logger.error(f"❌ Failed to list sessions: {e}")
        return []


@retry_on_failure(max_retries=3)
def search_across_sessions(
    query: str,
    user_id: Optional[str] = None,
    top_k: int = 10
) -> List[Dict]:
    """
    Search across all conversation sessions
    
    Args:
        query: Search query
        user_id: Optional filter by user
        top_k: Number of results to return
    
    Returns:
        List of matching conversation turns from any session
    """
    try:
        index = get_pinecone_index()
        
        # Embed query
        query_embedding = embed_query_for_search(query)
        
        filter_dict = {}
        if user_id:
            filter_dict["user_id"] = str(user_id)
        
        # Search across all conversations
        results = index.query(
            vector=query_embedding,
            filter=filter_dict if filter_dict else None,
            top_k=top_k,
            namespace=NAMESPACE_CONVERSATIONS,
            include_metadata=True
        )
        
        # Extract matches
        matches = []
        for match in results.get("matches", []):
            metadata = match.get("metadata", {})
            matches.append({
                "session_id": metadata.get("session_id", ""),
                "turn_number": int(metadata.get("turn_number", 0)),
                "user_query": metadata.get("user_query", ""),
                "assistant_response": metadata.get("assistant_response", ""),
                "timestamp": float(metadata.get("timestamp", 0)),
                "relevance_score": float(match.get("score", 0.0))
            })
        
        logger.info(f"🔍 Cross-session search: {len(matches)} matches")
        
        return matches
        
    except Exception as e:
        logger.error(f"❌ Cross-session search failed: {e}")
        return []


@retry_on_failure(max_retries=3)
def update_session_metadata(
    session_id: str,
    last_message: str,
    turn_number: int,
    user_id: Optional[str] = None
):
    """Update session metadata for listing"""
    try:
        index = get_pinecone_index()
        
        # Generate title from first message
        title = last_message[:50]
        if len(last_message) > 50:
            title += "..."
        
        current_time = time.time()
        
        metadata = {
            "session_id": session_id,
            "title": title,
            "last_message": last_message[:200],
            "message_count": int(turn_number),
            "updated_at": float(current_time)
        }
        
        # Add created_at only for first turn
        if turn_number == 1:
            metadata["created_at"] = float(current_time)
        
        if user_id:
            metadata["user_id"] = str(user_id)
        
        # Upsert metadata
        index.upsert(
            vectors=[{
                "id": f"session_{session_id}",
                "values": [1.0] * 1024,  # Dummy vector
                "metadata": metadata
            }],
            namespace=NAMESPACE_SESSION_META
        )
        
        logger.info(f"✅ Updated metadata for session {session_id}")
        
    except Exception as e:
        logger.error(f"⚠️ Failed to update metadata: {e}")


def get_session_turns(session_id: str) -> List[Dict]:
    """Get all turns for a session (alias)"""
    return load_conversation(session_id, max_turns=100)


@retry_on_failure(max_retries=3)
def delete_session(session_id: str) -> Dict:
    """
    Delete all conversation turns and metadata for a session
    
    Args:
        session_id: Session to delete
    
    Returns:
        Deletion confirmation
    """
    try:
        index = get_pinecone_index()
        
        # Delete conversation turns
        dummy_vector = [0.0] * 1024
        results = index.query(
            vector=dummy_vector,
            filter={"session_id": session_id},
            top_k=1000,
            namespace=NAMESPACE_CONVERSATIONS,
            include_metadata=True
        )
        
        vector_ids = [match["id"] for match in results.get("matches", [])]
        
        if vector_ids:
            index.delete(
                ids=vector_ids,
                namespace=NAMESPACE_CONVERSATIONS
            )
        
        # Delete session metadata
        try:
            index.delete(
                ids=[f"session_{session_id}"],
                namespace=NAMESPACE_SESSION_META
            )
        except:
            pass  # Metadata might not exist
        
        logger.info(f"🗑️ Deleted session {session_id}: {len(vector_ids)} turns")
        
        return {
            "success": True,
            "session_id": session_id,
            "deleted_turns": len(vector_ids)
        }
        
    except Exception as e:
        logger.error(f"❌ Delete failed: {e}")
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
        Stats including total turns, timestamps, etc.
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