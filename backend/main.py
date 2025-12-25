"""
FastAPI Backend - Production-Ready RAG with LangGraph
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from fastapi import WebSocket, WebSocketDisconnect
import asyncio
import logging
import os
import tempfile
import time
import hashlib
from uuid import uuid4
from dotenv import load_dotenv

# --- Internal Imports ---
from rag_based_book_bot.document_ingestion.progress_tracker import get_progress_tracker
from rag_based_book_bot.document_ingestion.enhanced_ingestion import (
    EnhancedBookIngestorPaddle,
    IngestorConfig
)
# Import State and Graph directly
from rag_based_book_bot.agents.states import AgentState, ConversationTurn, LLMResponse
from rag_based_book_bot.agents.graph import build_query_graph
from rag_based_book_bot.agents.nodes import get_pinecone_index

from rag_based_book_bot.memory.conversation_store import (
    save_conversation_turn,
    load_conversation,
    list_all_sessions,
    search_across_sessions,
    delete_session,
    get_conversation_stats
)

load_dotenv()
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)

# ============================================================================
# GLOBALS & LIFESPAN
# ============================================================================

# Initialize Graph
rag_graph = build_query_graph()

@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_running_loop()
    tracker = get_progress_tracker()
    tracker.set_loop(loop)
    logger.info("✅ Initialized progress tracker")
    logger.info("✅ LangGraph Workflow Compiled")
    yield

app = FastAPI(title="RAG Book Bot API", version="4.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# MODELS
# ============================================================================

class ChunkDetail(BaseModel):
    chunk_id: str
    chapter: str
    page: Optional[int]
    relevance: float
    type: str
    content_preview: str = ""
    source: str = ""
    book_title: str = "Unknown Book"
    author: str = "Unknown Author"

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    book_filter: Optional[str] = None
    chapter_filter: Optional[str] = None
    top_k: int = 5
    pass1_k: int = 50
    pass2_k: int = 15
    pass3_enabled: bool = True
    max_tokens: int = 2500
    force_retrieval: bool = False

class PipelineStage(BaseModel):
    stage_name: str
    chunk_count: int
    chunks: List[ChunkDetail]

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict] # Detailed source info
    confidence: float
    stats: dict
    pipeline_stages: List[PipelineStage]
    rewritten_queries: List[str] = []
    session_id: str
    conversation_turn: int
    resolved_query: Optional[str] = None
    answered_from_history: bool = False
    needs_context: bool = False

class IngestResponse(BaseModel):
    success: bool
    result: Optional[dict] = None
    error: Optional[str] = None
    
class BookInfo(BaseModel):
    title: str
    author: str
    total_chunks: int
    code_chunks: int = 0
    text_chunks: int = 0
    indexed_at: Optional[float] = None

class BooksResponse(BaseModel):
    books: List[BookInfo]

class SessionListResponse(BaseModel):
    sessions: List[dict]
    total: int

# ============================================================================
# HELPERS
# ============================================================================

def convert_pinecone_to_conversation_turns(pinecone_turns: List[dict]) -> List[ConversationTurn]:
    return [
        ConversationTurn(
            user_query=turn.get('user_query', ''),
            assistant_response=turn.get('assistant_response', ''),
            timestamp=turn.get('timestamp', time.time()),
            sources_used=turn.get('sources_used', []),
            resolved_query=turn.get('resolved_query'),
            needs_retrieval=turn.get('needs_retrieval', True),
            referenced_turn=turn.get('referenced_turn')
        )
        for turn in pinecone_turns
    ]

def extract_pipeline_stages(snapshots: List[dict]) -> List[PipelineStage]:
    """Converts internal graph snapshots to frontend pipeline stages."""
    stages = []
    for snap in snapshots:
        chunks = []
        raw_chunks = snap.get("chunks", [])
        
        # Handle both list of dicts or list of objects if necessary
        for c in raw_chunks:
            # Check if it's a RetrievedChunk object or dict
            chunk_data = c.get("chunk", {}) if isinstance(c, dict) else getattr(c, "chunk", {})
            if hasattr(chunk_data, "dict"): chunk_data = chunk_data.dict()
            if hasattr(c, "dict"): c = c.dict()
            
            chunks.append(ChunkDetail(
                chunk_id=chunk_data.get("chunk_id", ""),
                chapter=chunk_data.get("chapter", ""),
                page=chunk_data.get("page_number", 0),
                relevance=c.get("relevance_percentage", 0),
                type=chunk_data.get("chunk_type", "text"),
                book_title=chunk_data.get("book_title", "Unknown"),
                content_preview=chunk_data.get("preview", "") or chunk_data.get("content", "")[:100]
            ))
            
        stages.append(PipelineStage(
            stage_name=snap.get("stage", "Unknown"),
            chunk_count=snap.get("chunk_count", 0),
            chunks=chunks
        ))
    return stages

def get_available_books() -> List[BookInfo]:
    try:
        index = get_pinecone_index()
        metadata_namespace = "books_metadata"
        results = index.query(
            vector=[1.0] * 1024,
            top_k=10000,
            namespace=metadata_namespace,
            include_metadata=True
        )
        books_info = []
        for match in results.get("matches", []):
            metadata = match.get("metadata", {})
            if metadata.get("book_title") and metadata.get("book_title") != "__init__":
                books_info.append(BookInfo(
                    title=metadata.get("book_title", "Unknown"),
                    author=metadata.get("author", "Unknown"),
                    total_chunks=int(metadata.get("total_chunks", 0)),
                    code_chunks=int(metadata.get("code_chunks", 0)),
                    text_chunks=int(metadata.get("text_chunks", 0)),
                    indexed_at=metadata.get("indexed_at")
                ))
        return books_info
    except Exception as e:
        print(f"Error fetching books: {e}")
        return []

def store_book_metadata(book_title: str, author: str, total_chunks: int, code_chunks: int = 0):
    try:
        index = get_pinecone_index()
        book_id = hashlib.md5(book_title.encode()).hexdigest()
        index.upsert(
            vectors=[{
                "id": book_id,
                "values": [1.0] * 1024,
                "metadata": {
                    "book_title": book_title,
                    "author": author,
                    "total_chunks": total_chunks,
                    "code_chunks": code_chunks,
                    "text_chunks": total_chunks - code_chunks,
                    "indexed_at": time.time()
                }
            }],
            namespace="books_metadata"
        )
    except Exception as e:
        print(f"Failed to store metadata: {e}")

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    try:
        print(f"\n{'='*80}\n[NEW QUERY] {request.query}")
        
        session_id = request.session_id or str(uuid4())
        
        # 1. Load History
        pinecone_turns = load_conversation(session_id, max_turns=10)
        conversation_history = convert_pinecone_to_conversation_turns(pinecone_turns)
        
        # 2. Prepare Graph Input (AgentState)
        initial_state = AgentState(
            user_query=request.query,
            session_id=session_id,
            user_id=request.user_id,
            pass1_k=request.pass1_k,
            pass2_k=request.pass2_k,
            pass3_enabled=request.pass3_enabled,
            max_tokens=request.max_tokens,
            book_filter=request.book_filter,
            chapter_filter=request.chapter_filter,
            conversation_history=conversation_history
        )
        
        # 3. Invoke LangGraph
        # Use ainvoke for async execution
        final_state_dict = await rag_graph.ainvoke(initial_state)
        
        # 4. Extract Results from State
        # Note: LangGraph returns a dict representation of the state
        response_obj = final_state_dict.get("response")
        
        if not response_obj:
            # Fallback if no response generated
            answer = "I'm sorry, I encountered an error processing your request."
            confidence = 0.0
            sources_ids = []
            code_snippets = []
        else:
            # response_obj might be a dict or Pydantic model depending on how LangGraph serialization works
            if isinstance(response_obj, dict):
                answer = response_obj.get("answer", "")
                confidence = response_obj.get("confidence", 0.0)
                sources_ids = response_obj.get("sources", [])
                code_snippets = response_obj.get("code_snippets", [])
            else:
                answer = response_obj.answer
                confidence = response_obj.confidence
                sources_ids = response_obj.sources
                code_snippets = response_obj.code_snippets

        # 5. Reconstruct Source Details
        # We need to map source IDs back to the chunk details found in retrieval
        full_sources = []
        reranked = final_state_dict.get("reranked_chunks", [])
        
        for rc in reranked:
            # Handle if rc is dict or object
            rc_data = rc if isinstance(rc, dict) else rc.dict()
            chunk_data = rc_data.get("chunk", {})
            
            if chunk_data.get("chunk_id") in sources_ids:
                full_sources.append({
                    "chunk_id": chunk_data.get("chunk_id"),
                    "book_title": chunk_data.get("book_title"),
                    "page": chunk_data.get("page_number"),
                    "content": chunk_data.get("content"),
                    "relevance": rc_data.get("relevance_percentage", 0)
                })

        # 6. Save Turn
        turn_number = len(conversation_history) + 1
        save_conversation_turn(
            session_id=session_id,
            turn_number=turn_number,
            user_query=request.query,
            assistant_response=answer,
            resolved_query=final_state_dict.get("resolved_query", request.query),
            needs_retrieval=final_state_dict.get("needs_retrieval", True),
            referenced_turn=final_state_dict.get("referenced_turn"),
            sources_used=sources_ids,
            user_id=request.user_id
        )
        
        # 7. Format Output
        pipeline_stages = extract_pipeline_stages(final_state_dict.get("pipeline_snapshots", []))
        
        stats = {
            "snapshots": final_state_dict.get("pipeline_snapshots", []),
            "total_chunks_found": len(final_state_dict.get("retrieved_chunks", [])),
            "final_context_chunks": len(final_state_dict.get("reranked_chunks", []))
        }

        return QueryResponse(
            answer=answer,
            sources=full_sources[:5], # Return top 5 full source objects
            confidence=confidence,
            stats=stats,
            pipeline_stages=pipeline_stages,
            rewritten_queries=final_state_dict.get("rewritten_queries", []),
            session_id=session_id,
            conversation_turn=turn_number,
            resolved_query=final_state_dict.get("resolved_query"),
            answered_from_history=not final_state_dict.get("needs_retrieval", True),
            needs_context=False
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.get("/sessions", response_model=SessionListResponse)
async def get_sessions(limit: int = 50):
    """Fetch recent chat sessions."""
    try:
        sessions = list_all_sessions(limit=limit)
        return SessionListResponse(sessions=sessions, total=len(sessions))
    except Exception as e:
        print(f"Error fetching sessions: {e}")
        return SessionListResponse(sessions=[], total=0)

@app.get("/books", response_model=BooksResponse)
async def list_books():
    books = get_available_books()
    return BooksResponse(books=books)

@app.post("/ingest", response_model=IngestResponse)
def ingest_book(
    file: UploadFile = File(...),
    book_title: Optional[str] = None,
    author: Optional[str] = None
):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files supported")
    
    tmp_path = None
    try:
        tracker = get_progress_tracker()
        tracker.reset()
        
        final_title = book_title or file.filename
        final_author = author or "Unknown"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(file.file.read())
            tmp_path = tmp.name
            
        config = IngestorConfig()
        ingestor = EnhancedBookIngestorPaddle(config=config)
        result = ingestor.ingest_book(tmp_path, final_title, final_author)
        
        store_book_metadata(final_title, final_author, result.get('chunks', 0), result.get('code_chunks', 0))
        
        if os.path.exists(tmp_path): os.unlink(tmp_path)
        return IngestResponse(success=True, result=result)
        
    except Exception as e:
        if tmp_path and os.path.exists(tmp_path): os.unlink(tmp_path)
        return IngestResponse(success=False, error=str(e))

@app.get("/health")
async def health_check():
    return {"status": "online", "mode": "langgraph-prod"}

@app.websocket("/ws/ingest")
async def websocket_ingestion_progress(websocket: WebSocket):
    await websocket.accept()
    tracker = get_progress_tracker()
    
    is_connected = True
    
    async def send_update(state):
        nonlocal is_connected
        if not is_connected: return
        try:
            if websocket.client_state.name == "CONNECTED":
                await websocket.send_json(state.to_dict())
            else:
                is_connected = False
        except Exception:
            is_connected = False

    tracker.on_progress(send_update)
    
    try:
        while is_connected:
            await asyncio.sleep(1)
            try:
                await websocket.receive_text()
            except WebSocketDisconnect:
                is_connected = False
    finally:
        tracker.remove_callback(send_update)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)