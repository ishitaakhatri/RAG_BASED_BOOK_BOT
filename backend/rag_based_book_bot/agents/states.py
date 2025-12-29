"""
Agent State for LangGraph Pipeline
Updated to use TypedDict for LangGraph compatibility
"""

from typing import List, Dict, Any, Optional, TypedDict
from dataclasses import dataclass
from enum import Enum


class QueryIntent(Enum):
    CONCEPTUAL = "conceptual"
    CODE_REQUEST = "code_request"
    DEBUGGING = "debugging"
    COMPARISON = "comparison"
    TUTORIAL = "tutorial"


@dataclass
class ParsedQuery:
    raw_query: str
    intent: QueryIntent
    topics: List[str]
    keywords: List[str]
    code_language: Optional[str] = None
    complexity_hint: str = "intermediate"


@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation history"""
    user_query: str
    assistant_response: str
    timestamp: float = 0.0
    sources_used: List[str] = None
    resolved_query: Optional[str] = None
    needs_retrieval: bool = True
    referenced_turn: Optional[int] = None
    
    def __post_init__(self):
        if self.sources_used is None:
            self.sources_used = []


@dataclass
class DocumentChunk:
    """Represents a chunk of text/code from a document"""
    chunk_id: str
    content: str
    chapter: str
    section: str
    page_number: int
    chunk_type: str
    book_title: str
    author: str
    chapter_title: str = ""
    chapter_number: str = ""
    preview: str = ""


@dataclass
class RetrievedChunk:
    """A chunk retrieved from vector DB with similarity scores"""
    chunk: DocumentChunk
    similarity_score: float = 0.0
    rerank_score: float = 0.0
    relevance_percentage: float = 0.0


@dataclass
class LLMResponse:
    answer: str
    code_snippets: List[str]
    sources: List[str]
    confidence: float


# ============================================================================
# LANGGRAPH STATE - TypedDict for proper state management
# ============================================================================

class AgentState(TypedDict, total=False):
    """
    LangGraph-compatible state using TypedDict.
    
    All fields are optional (total=False) to allow partial updates.
    LangGraph will merge returned dicts into the state automatically.
    """
    
    # ===== Input Fields =====
    user_query: str
    session_id: str
    user_id: Optional[str]
    
    # ===== Conversation History =====
    conversation_history: List[ConversationTurn]
    chat_history: List[Dict[str, str]]  # Backward compatibility
    max_history_turns: int
    
    # ===== Configuration =====
    pass1_k: int
    pass2_k: int
    pass3_enabled: bool
    max_tokens: int
    book_filter: Optional[str]
    chapter_filter: Optional[str]
    
    # ===== Processing State =====
    parsed_query: Optional[ParsedQuery]
    rewritten_queries: List[str]
    resolved_query: Optional[str]
    needs_retrieval: bool
    referenced_turn: Optional[int]
    relevant_past_turns: List[ConversationTurn]
    
    # ===== Retrieval State =====
    retrieved_chunks: List[RetrievedChunk]
    reranked_chunks: List[RetrievedChunk]
    assembled_context: str
    system_prompt: str
    
    # ===== Output =====
    response: Optional[LLMResponse]
    errors: List[str]
    current_node: str
    
    # ===== Debugging / Visualization =====
    pipeline_snapshots: List[Dict[str, Any]]


# ============================================================================
# Helper function to create initial state
# ============================================================================

def create_initial_state(
    user_query: str,
    session_id: str = "default",
    user_id: Optional[str] = None,
    conversation_history: Optional[List[ConversationTurn]] = None,
    **kwargs
) -> AgentState:
    """
    Create initial state with default values.
    
    Args:
        user_query: User's question
        session_id: Session identifier
        user_id: Optional user identifier
        conversation_history: Previous conversation turns
        **kwargs: Additional configuration (pass1_k, pass2_k, etc.)
    
    Returns:
        AgentState dict ready for LangGraph
    """
    return AgentState(
        # Input
        user_query=user_query,
        session_id=session_id,
        user_id=user_id,
        
        # History
        conversation_history=conversation_history or [],
        chat_history=[],
        max_history_turns=kwargs.get("max_history_turns", 5),
        
        # Configuration
        pass1_k=kwargs.get("pass1_k", 50),
        pass2_k=kwargs.get("pass2_k", 10),
        pass3_enabled=kwargs.get("pass3_enabled", True),
        max_tokens=kwargs.get("max_tokens", 4000),
        book_filter=kwargs.get("book_filter"),
        chapter_filter=kwargs.get("chapter_filter"),
        
        # Processing State (initialized)
        parsed_query=None,
        rewritten_queries=[],
        resolved_query=None,
        needs_retrieval=True,
        referenced_turn=None,
        relevant_past_turns=[],
        
        # Retrieval State (initialized)
        retrieved_chunks=[],
        reranked_chunks=[],
        assembled_context="",
        system_prompt="",
        
        # Output (initialized)
        response=None,
        errors=[],
        current_node="start",
        
        # Debugging
        pipeline_snapshots=[]
    )