"""
State definitions for the RAG Agent pipeline.
These dataclasses define the structure of data flowing between nodes.
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


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
    
    # Metadata for context
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
    
    # After Vector Search
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    
    # After Reranking
    reranked_chunks: list[RetrievedChunk] = field(default_factory=list)
    
    # After Context Assembly
    assembled_context: str = ""
    system_prompt: str = ""
    
    # Final Output
    response: Optional[LLMResponse] = None
    
    # Pipeline metadata
    current_node: str = ""
    errors: list[str] = field(default_factory=list)