# File: rag_based_book_bot/agents/memory_manager.py
# CREATE THIS NEW FILE

import json
import os
from datetime import datetime
from typing import List, Optional, Set, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

from rag_based_book_bot.agents.states import (
    ConversationMemory, QueryMemoryRecord, FeedbackRecord, AgentState
)

# Initialize embedding model (same as nodes.py)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MEMORY_DIR = os.getenv("MEMORY_DIR", "./conversation_memory")
SIMILARITY_THRESHOLD = 0.85  # High threshold to avoid noise

_model = None

def get_embedding_model():
    """Get embedding model (lazy initialization)"""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def initialize_conversation_memory(conversation_id: str) -> ConversationMemory:
    """
    Initialize or load existing conversation memory
    """
    memory = ConversationMemory(conversation_id=conversation_id)
    
    # Try to load from disk if it exists
    memory_file = os.path.join(MEMORY_DIR, f"{conversation_id}.json")
    if os.path.exists(memory_file):
        memory = load_conversation_memory(conversation_id)
    else:
        # Create new memory
        os.makedirs(MEMORY_DIR, exist_ok=True)
    
    return memory


def generate_query_embedding(query_text: str) -> List[float]:
    """
    Generate embedding for a query
    Returns: embedding vector as list
    """
    model = get_embedding_model()
    embedding = model.encode(query_text)
    return embedding.tolist()


def find_similar_previous_query(
    query_embedding: List[float],
    conversation_memory: ConversationMemory,
    threshold: float = SIMILARITY_THRESHOLD
) -> Optional[QueryMemoryRecord]:
    """
    Find if current query is semantically similar to any previous query
    Returns: Previous QueryMemoryRecord if similarity > threshold, else None
    """
    if not conversation_memory.query_history:
        return None
    
    current_embedding = np.array(query_embedding)
    
    for previous_record in conversation_memory.get_last_n_queries(n=10):
        previous_embedding = np.array(previous_record.query_embedding)
        
        # Calculate cosine similarity
        similarity = np.dot(current_embedding, previous_embedding) / (
            np.linalg.norm(current_embedding) * np.linalg.norm(previous_embedding)
        )
        
        if similarity >= threshold:
            return previous_record
    
    return None


def get_already_retrieved_chunks(
    conversation_memory: ConversationMemory
) -> Set[str]:
    """
    Get all chunk IDs already retrieved in this conversation
    Returns: Set of chunk_ids to exclude from new retrieval
    """
    return conversation_memory.seen_chunk_ids


def save_query_record(
    state: AgentState,
    retrieved_chunk_ids: List[str],
    book_title: str,
    chapter: str
) -> None:
    """
    Save current query record to memory
    Call this AFTER vector search retrieves chunks
    """
    if not state.conversation_memory or not state.current_query_embedding:
        return
    
    book_chapter_combo = f"{book_title} - {chapter}" if chapter else book_title
    
    record = QueryMemoryRecord(
        query_text=state.user_query,
        query_embedding=state.current_query_embedding,
        retrieved_chunk_ids=retrieved_chunk_ids,
        book_chapter_combo=book_chapter_combo,
        timestamp=datetime.now().isoformat(),
        user_feedback=None
    )
    
    state.conversation_memory.add_query_record(record)
    
    # Update explored book/chapters count
    state.conversation_memory.explored_book_chapters[book_chapter_combo] = (
        state.conversation_memory.explored_book_chapters.get(book_chapter_combo, 0) + 1
    )
    
    # Persist to disk
    save_conversation_memory(state.conversation_memory)


def record_user_feedback(
    state: AgentState,
    feedback_type: str,  # "helpful" or "wrong"
    chunk_id: str,
    book_title: str,
    chapter: str
) -> None:
    """
    Record user feedback on a specific chunk
    Call this when user marks an answer as helpful or wrong
    """
    if not state.conversation_memory or not state.current_query_embedding:
        return
    
    feedback = FeedbackRecord(
        chunk_id=chunk_id,
        query_embedding=state.current_query_embedding,
        feedback_type=feedback_type,
        timestamp=datetime.now().isoformat(),
        book_title=book_title,
        chapter=chapter
    )
    
    state.conversation_memory.add_feedback(feedback)
    save_conversation_memory(state.conversation_memory)


def get_feedback_for_similar_chunks(
    query_embedding: List[float],
    conversation_memory: ConversationMemory,
    feedback_type: str = "helpful"
) -> List[str]:
    """
    Get chunk IDs that had positive feedback from semantically similar queries
    Use this to BOOST certain chunks in reranking
    """
    if not conversation_memory.feedback_records:
        return []
    
    current_embedding = np.array(query_embedding)
    boosted_chunk_ids = []
    
    for feedback_record in conversation_memory.feedback_records:
        if feedback_record.feedback_type != feedback_type:
            continue
        
        previous_embedding = np.array(feedback_record.query_embedding)
        similarity = np.dot(current_embedding, previous_embedding) / (
            np.linalg.norm(current_embedding) * np.linalg.norm(previous_embedding)
        )
        
        # Lower threshold for feedback (0.75) because we want to be helpful
        if similarity >= 0.75:
            boosted_chunk_ids.append(feedback_record.chunk_id)
    
    return boosted_chunk_ids


def get_explored_book_chapters(
    conversation_memory: ConversationMemory
) -> List[Tuple[str, int]]:
    """
    Get list of book/chapter combinations already explored
    Returns: List of (book_chapter_combo, count) tuples
    """
    return sorted(
        conversation_memory.explored_book_chapters.items(),
        key=lambda x: x[1],
        reverse=True
    )


def save_conversation_memory(memory: ConversationMemory) -> None:
    """
    Persist conversation memory to disk as JSON
    """
    os.makedirs(MEMORY_DIR, exist_ok=True)
    
    memory_file = os.path.join(MEMORY_DIR, f"{memory.conversation_id}.json")
    
    # Convert to serializable format
    memory_dict = {
        "conversation_id": memory.conversation_id,
        "query_history": [
            {
                "query_text": record.query_text,
                "query_embedding": record.query_embedding,
                "retrieved_chunk_ids": record.retrieved_chunk_ids,
                "book_chapter_combo": record.book_chapter_combo,
                "timestamp": record.timestamp,
                "user_feedback": {
                    "chunk_id": record.user_feedback.chunk_id,
                    "feedback_type": record.user_feedback.feedback_type,
                    "timestamp": record.user_feedback.timestamp,
                } if record.user_feedback else None
            }
            for record in memory.query_history
        ],
        "seen_chunk_ids": list(memory.seen_chunk_ids),
        "explored_book_chapters": memory.explored_book_chapters,
        "feedback_records": [
            {
                "chunk_id": fb.chunk_id,
                "query_embedding": fb.query_embedding,
                "feedback_type": fb.feedback_type,
                "timestamp": fb.timestamp,
                "book_title": fb.book_title,
                "chapter": fb.chapter,
            }
            for fb in memory.feedback_records
        ]
    }
    
    with open(memory_file, 'w') as f:
        json.dump(memory_dict, f, indent=2)


def load_conversation_memory(conversation_id: str) -> ConversationMemory:
    """
    Load conversation memory from disk
    """
    memory_file = os.path.join(MEMORY_DIR, f"{conversation_id}.json")
    
    if not os.path.exists(memory_file):
        return ConversationMemory(conversation_id=conversation_id)
    
    with open(memory_file, 'r') as f:
        memory_dict = json.load(f)
    
    memory = ConversationMemory(
        conversation_id=conversation_id,
        seen_chunk_ids=set(memory_dict.get("seen_chunk_ids", [])),
        explored_book_chapters=memory_dict.get("explored_book_chapters", {}),
    )
    
    # Reconstruct query history
    for record_dict in memory_dict.get("query_history", []):
        feedback = None
        if record_dict.get("user_feedback"):
            feedback = FeedbackRecord(
                chunk_id=record_dict["user_feedback"]["chunk_id"],
                query_embedding=[],  # Not needed for loaded feedback
                feedback_type=record_dict["user_feedback"]["feedback_type"],
                timestamp=record_dict["user_feedback"]["timestamp"],
                book_title="",
                chapter=""
            )
        
        record = QueryMemoryRecord(
            query_text=record_dict["query_text"],
            query_embedding=record_dict["query_embedding"],
            retrieved_chunk_ids=record_dict["retrieved_chunk_ids"],
            book_chapter_combo=record_dict["book_chapter_combo"],
            timestamp=record_dict["timestamp"],
            user_feedback=feedback
        )
        memory.query_history.append(record)
    
    # Reconstruct feedback records
    for fb_dict in memory_dict.get("feedback_records", []):
        feedback = FeedbackRecord(
            chunk_id=fb_dict["chunk_id"],
            query_embedding=fb_dict["query_embedding"],
            feedback_type=fb_dict["feedback_type"],
            timestamp=fb_dict["timestamp"],
            book_title=fb_dict["book_title"],
            chapter=fb_dict["chapter"]
        )
        memory.feedback_records.append(feedback)
    
    return memory