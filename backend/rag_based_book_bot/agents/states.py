import operator
from typing import List, Dict, Any, Optional, Annotated
from pydantic import BaseModel, Field
from enum import Enum

class QueryIntent(str, Enum):
    CONCEPTUAL = "conceptual"
    CODE_REQUEST = "code_request"
    DEBUGGING = "debugging"
    COMPARISON = "comparison"
    TUTORIAL = "tutorial"

class ParsedQuery(BaseModel):
    raw_query: str
    intent: QueryIntent
    topics: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    code_language: Optional[str] = None
    complexity_hint: str = "intermediate"

class ConversationTurn(BaseModel):
    user_query: str
    assistant_response: str
    timestamp: float = 0.0
    sources_used: List[str] = Field(default_factory=list)
    resolved_query: Optional[str] = None
    needs_retrieval: bool = True
    referenced_turn: Optional[int] = None

class DocumentChunk(BaseModel):
    chunk_id: str
    content: str
    chapter: str
    section: str = ""
    page_number: int = 0
    chunk_type: str = "text"
    book_title: str = "Unknown Book"
    author: str = "Unknown Author"
    chapter_title: str = ""
    preview: str = ""

class RetrievedChunk(BaseModel):
    chunk: DocumentChunk
    similarity_score: float = 0.0
    rerank_score: float = 0.0
    relevance_percentage: float = 0.0

class LLMResponse(BaseModel):
    answer: str
    code_snippets: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)
    confidence: float = 0.0

class AgentState(BaseModel):
    # Input
    user_query: str
    session_id: str = "default"
    user_id: Optional[str] = None
    
    # Configuration
    pass1_k: int = 50
    pass2_k: int = 10
    pass3_enabled: bool = True
    max_tokens: int = 4000
    book_filter: Optional[str] = None
    chapter_filter: Optional[str] = None
    
    # History
    conversation_history: List[ConversationTurn] = Field(default_factory=list)
    
    # Processing
    parsed_query: Optional[ParsedQuery] = None
    rewritten_queries: List[str] = Field(default_factory=list)
    resolved_query: Optional[str] = None
    needs_retrieval: bool = True
    referenced_turn: Optional[int] = None
    
    # Retrieval
    retrieved_chunks: List[RetrievedChunk] = Field(default_factory=list)
    reranked_chunks: List[RetrievedChunk] = Field(default_factory=list)
    assembled_context: str = ""
    
    # Output
    response: Optional[LLMResponse] = None
    
    # Appenders
    errors: Annotated[List[str], operator.add] = Field(default_factory=list)
    pipeline_snapshots: Annotated[List[Dict[str, Any]], operator.add] = Field(default_factory=list)