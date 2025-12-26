from typing import List, Dict, Any, Optional
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
    # NEW FIELDS for frontend preview and hierarchy
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

@dataclass
class AgentState:
    """
    Shared state object passed through the LangGraph pipeline.
    Contains input, configuration, processing state, and output.
    """
    # Input
    user_query: str
    session_id: str = "default"
    user_id: Optional[str] = None
    
    # History (Required by memory_nodes.py and main.py)
    conversation_history: List[ConversationTurn] = field(default_factory=list)
    chat_history: List[Dict[str, str]] = field(default_factory=list) # Kept for backward compatibility if needed
    max_history_turns: int = 5
    
    # Configuration
    pass1_k: int = 50
    pass2_k: int = 10
    pass3_enabled: bool = True
    max_tokens: int = 30000
    book_filter: Optional[str] = None
    chapter_filter: Optional[str] = None
    
    # Processing State
    parsed_query: Optional[ParsedQuery] = None
    rewritten_queries: List[str] = field(default_factory=list)
    resolved_query: Optional[str] = None
    needs_retrieval: bool = True
    referenced_turn: Optional[int] = None
    relevant_past_turns: List[ConversationTurn] = field(default_factory=list)
    
    # Retrieval State
    retrieved_chunks: List[RetrievedChunk] = field(default_factory=list)
    reranked_chunks: List[RetrievedChunk] = field(default_factory=list)
    assembled_context: str = ""
    system_prompt: str = ""
    
    # Output
    response: Optional[LLMResponse] = None
    errors: List[str] = field(default_factory=list)
    current_node: str = "start"
    
    # Debugging / Visualization
    pipeline_snapshots: List[Dict[str, Any]] = field(default_factory=list)