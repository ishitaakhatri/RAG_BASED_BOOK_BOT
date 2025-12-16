"""
Memory package for conversation storage using Pinecone

This package handles conversation memory using Pinecone vector database:
- Store conversation turns as vectors
- Semantic search over conversation history
- Session management
- Context retrieval for follow-up questions
"""

from rag_based_book_bot.memory.conversation_store import (
    save_conversation_turn,
    load_conversation,
    search_conversation_context,
    get_session_turns,
    delete_session,
    get_conversation_stats
)

from rag_based_book_bot.memory.embedding_utils import (
    embed_conversation_turn,
    format_turn_for_embedding,
    batch_embed_turns
)

__all__ = [
    # Conversation storage operations
    'save_conversation_turn',
    'load_conversation',
    'search_conversation_context',
    'get_session_turns',
    'delete_session',
    'get_conversation_stats',
    
    # Embedding utilities
    'embed_conversation_turn',
    'format_turn_for_embedding',
    'batch_embed_turns'
]
