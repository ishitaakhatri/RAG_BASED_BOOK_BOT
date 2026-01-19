"""
Agent State for LangGraph Pipeline
Compatible with LangGraph 0.2+
"""

from typing import List, Dict, Any, Optional, TypedDict
from dataclasses import dataclass, field
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
    sources_used: List[str] = field(default_factory=list)
    resolved_query: Optional[str] = None
    needs_retrieval: bool = True
    referenced_turn: Optional[int] = None


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
    search_summary: Optional[str] = None


# ============================================================================
# LANGGRAPH STATE - TypedDict
# ============================================================================

class AgentState(TypedDict, total=False):
    """
    LangGraph State Definition.
    Using total=False allows us to return partial updates from nodes.
    """
    
    # ===== Input Fields =====
    user_query: str
    session_id: str
    user_id: Optional[str]
    
    # ===== Conversation History =====
    conversation_history: List[ConversationTurn]
    chat_history: List[Dict[str, str]]
    max_history_turns: int
    
    # ===== Configuration =====
    pass1_k: int
    pass2_k: int
    pass3_enabled: bool
    max_tokens: int
    book_filter: Optional[List[str]]  # List of book titles to filter by
    chapter_filter: Optional[str]
    
    # ===== Relevance Check =====
    intent: Optional[str]
    intent_confidence: float
    
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
    
    # ===== Debugging =====
    pipeline_snapshots: List[Dict[str, Any]]


def create_initial_state(
    user_query: str,
    session_id: str = "default",
    user_id: Optional[str] = None,
    conversation_history: Optional[List[ConversationTurn]] = None,
    **kwargs
) -> AgentState:
    """Factory to create a clean initial state"""
    return AgentState(
        user_query=user_query,
        session_id=session_id,
        user_id=user_id,
        conversation_history=conversation_history or [],
        chat_history=[],
        max_history_turns=kwargs.get("max_history_turns", 5),
        pass1_k=kwargs.get("pass1_k", 50),
        pass2_k=kwargs.get("pass2_k", 10),
        pass3_enabled=kwargs.get("pass3_enabled", True),
        max_tokens=kwargs.get("max_tokens", 4000),
        book_filter=kwargs.get("book_filter"),
        chapter_filter=kwargs.get("chapter_filter"),
        intent=None,
        intent_confidence=0.0,
        parsed_query=None,
        rewritten_queries=[],
        resolved_query=None,
        needs_retrieval=True,
        referenced_turn=None,
        relevant_past_turns=[],
        retrieved_chunks=[],
        reranked_chunks=[],
        assembled_context="",
        system_prompt="",
        response=None,
        errors=[],
        current_node="start",
        pipeline_snapshots=[]
    )