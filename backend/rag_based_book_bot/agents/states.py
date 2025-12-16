"""
State definitions for the RAG Agent pipeline.
These dataclasses define the structure of data flowing between nodes.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum
import time  # NEW IMPORT


class QueryIntent(Enum):
    """Types of user query intents."""
    CONCEPTUAL = "conceptual"      # "What is gradient descent?"
    CODE_REQUEST = "code_request"  # "Show me how to implement..."
    DEBUGGING = "debugging"        # "Why isn't my model converging?"
    COMPARISON = "comparison"      # "Difference between CNN and RNN?"
    TUTORIAL = "tutorial"          # "Walk me through building..."


@dataclass
class DocumentChunk:
    """A single chunk of document content."""
    chunk_id: str
    content: str
    chapter: str
    section: str
    page_number: Optional[int] = None
    chunk_type: str = "text"  # "text", "code", "mixed"
    embedding: list[float] = field(default_factory=list)
    
    # Additional metadata
    book_title: Optional[str] = None
    author: Optional[str] = None
    chapter_title: Optional[str] = None
    chapter_number: Optional[str] = None
    section_titles: List[str] = field(default_factory=list)
    section_numbers: List[str] = field(default_factory=list)
    subsection_titles: List[str] = field(default_factory=list)
    subsection_numbers: List[str] = field(default_factory=list)
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    chunk_index: int = 0
    contains_code: bool = False
    
    # Context linking
    prev_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None


@dataclass
class RetrievedChunk:
    """A chunk with retrieval and reranking scores."""
    chunk: DocumentChunk
    similarity_score: float = 0.0
    rerank_score: float = 0.0
    relevance_percentage: float = 0.0


@dataclass 
class ParsedQuery:
    """Structured representation of user query."""
    raw_query: str
    intent: QueryIntent = QueryIntent.CONCEPTUAL
    topics: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    code_language: Optional[str] = None  # If code is requested
    complexity_hint: str = "intermediate"  # beginner, intermediate, advanced
    

@dataclass
class LLMResponse:
    """Final response from the LLM."""
    answer: str
    code_snippets: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)  # chunk_ids used
    confidence: float = 0.0


# ============================================================================
# NEW: Conversation Memory Classes
# ============================================================================

@dataclass
class ConversationTurn:
    """
    Single Q&A turn in a conversation
    
    Represents one complete interaction with user query and assistant response.
    Used for conversation memory and context resolution.
    """
    user_query: str
    assistant_response: str
    timestamp: float = field(default_factory=time.time)
    sources_used: List[str] = field(default_factory=list)
    
    # Context resolution metadata
    resolved_query: Optional[str] = None  # Standalone query after context resolution
    needs_retrieval: bool = True  # Whether retrieval was needed
    referenced_turn: Optional[int] = None  # Which previous turn was referenced


@dataclass
class AgentState:
    """
    Main state object that flows through the entire pipeline.
    Each node reads from and writes to this state.
    """
    # Input
    pdf_path: Optional[str] = None
    user_query: Optional[str] = None
    
    # After PDF Loading
    raw_text: str = ""
    total_pages: int = 0
    
    # After Chunking & Embedding
    chunks: list[DocumentChunk] = field(default_factory=list)
    
    # After Query Parsing
    parsed_query: Optional[ParsedQuery] = None
    
    # After Query Rewriting
    rewritten_queries: list[str] = field(default_factory=list)
    
    # After Vector Search
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    
    # After Reranking
    reranked_chunks: list[RetrievedChunk] = field(default_factory=list)
    
    # After Context Assembly
    assembled_context: str = ""
    system_prompt: str = ""
    
    # Final Output
    response: Optional[LLMResponse] = None
    
    # Filters and Configuration
    book_filter: Optional[str] = None
    chapter_filter: Optional[str] = None
    pass1_k: int = 50                    # Top-k for Pass 1 (vector search)
    pass2_k: int = 15                    # Top-k for Pass 2 (reranking)
    pass3_enabled: bool = True           # Enable/disable multi-hop expansion
    max_tokens: int = 2500               # Max tokens for context assembly
    
    # ============================================================================
    # NEW: Conversation Memory Fields
    # ============================================================================
    
    # Conversation history (loaded from Pinecone)
    conversation_history: List[ConversationTurn] = field(default_factory=list)
    
    # Context resolution results
    resolved_query: Optional[str] = None  # Standalone query (no pronouns/references)
    needs_retrieval: bool = True  # Whether retrieval needed or can answer from history
    referenced_turn: Optional[int] = None  # Which previous turn is referenced (1, 2, 3...)
    
    # Relevant past context (from semantic search over conversation)
    relevant_past_turns: List[ConversationTurn] = field(default_factory=list)
    
    # Session tracking
    session_id: Optional[str] = None  # Current session identifier
    user_id: Optional[str] = None  # Optional user identifier
    
    # Configuration
    max_history_turns: int = 5  # How many past turns to consider for context
    
    # ============================================================================
    # End of Conversation Memory Fields
    # ============================================================================
    
    # Pipeline metadata
    current_node: str = ""
    errors: list[str] = field(default_factory=list)
    pipeline_snapshots: List[dict] = field(default_factory=list)  # Track chunk counts per stage
