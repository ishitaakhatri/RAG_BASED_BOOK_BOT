"""
Enhanced Memory Package with Full Session Management

Features:
- Persistent conversation storage in Pinecone
- Session listing and search
- Robust error handling
- Semantic search across sessions
"""

from rag_based_book_bot.memory.conversation_store import (
    save_conversation_turn,
    load_conversation,
    search_conversation_context,
    get_session_turns,
    delete_session,
    get_conversation_stats,
    list_all_sessions,
    search_across_sessions,
    update_session_metadata
)

from rag_based_book_bot.memory.embedding_utils import (
    embed_conversation_turn,
    format_turn_for_embedding,
    batch_embed_turns
)

__all__ = [
    # Core conversation operations
    'save_conversation_turn',
    'load_conversation',
    'search_conversation_context',
    'get_session_turns',
    'delete_session',
    'get_conversation_stats',
    
    # Session management
    'list_all_sessions',
    'search_across_sessions',
    'update_session_metadata',
    
    # Embedding utilities
    'embed_conversation_turn',
    'format_turn_for_embedding',
    'batch_embed_turns'
]