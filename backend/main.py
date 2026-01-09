"""
FastAPI Backend - Production-Ready RAG with Conversation Memory & Clerk Authentication
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional, List, Dict
from fastapi import WebSocket, WebSocketDisconnect
import asyncio
import logging
import os
import tempfile
import time
import hashlib
from uuid import uuid4
from dotenv import load_dotenv

# Redis client
import redis

from app_config import get_config
from auth_utils import verify_clerk_token

# NEW: Import DB Init
from rag_based_book_bot.db import init_db 

load_dotenv()
settings = get_config()

# Redis connection (default localhost:6379 for local dev, 'redis' for Docker)
redis_host = os.getenv('REDIS_HOST') or ('localhost' if os.getenv('ENV', 'local') == 'local' else 'redis')
redis_port = int(os.getenv('REDIS_PORT', 6379))
redis_client = redis.Redis(host=redis_host, port=redis_port, db=0, decode_responses=True)

logger = logging.getLogger("main")
logger.setLevel(settings.log_level)
logger.propagate = True

from rag_based_book_bot.document_ingestion.enhanced_ingestion import (
    EnhancedBookIngestorPaddle,
    IngestorConfig
)
from rag_based_book_bot.document_ingestion.progress_tracker import (
    create_tracker, get_tracker, remove_tracker
)
from rag_based_book_bot.agents.states import AgentState, ConversationTurn, create_initial_state
from rag_based_book_bot.agents.graph import build_query_graph
from rag_based_book_bot.agents.nodes import get_pinecone_index

from rag_based_book_bot.memory import (
    save_conversation_turn,
    load_conversation,
    delete_session,
    list_all_sessions,
    search_across_sessions,
    get_session_metadata
)

# ============================================================================
# LIFESPAN & STARTUP
# ============================================================================

query_graph_app = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern lifespan handler"""
    global query_graph_app
    
    # 🔥 Initialize Database Tables (Async)
    print("🚀 Initializing Database...")
    await init_db()
    print("✅ Database ready.")
    
    query_graph_app = build_query_graph(enable_persistence=True)
    logger.info(f"🚀 RAG Book Bot API started successfully (Env: {settings.environment})")
    
    yield
    
    logger.info("🛑 Application shutting down...")

app = FastAPI(title="RAG Book Bot API", version="4.3.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# QUERY CANCELLATION TRACKING
# ============================================================================
active_queries: Dict[str, asyncio.Task] = {}
query_cancellation_events: Dict[str, asyncio.Event] = {}

# ============================================================================
# MODELS
# ============================================================================

class ChunkDetail(BaseModel):
    chunk_id: str
    chapter: str
    page: Optional[int]
    relevance: float
    type: str
    content_preview: str
    source: str
    book_title: str = "Unknown Book"
    author: str = "Unknown Author"

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    book_filter: Optional[str] = None
    chapter_filter: Optional[str] = None
    search_mode: Optional[str] = "all" 
    top_k: int = 5
    pass1_k: int = settings.retrieval.pass1_top_k
    pass2_k: int = settings.retrieval.pass2_top_k
    pass3_enabled: bool = settings.retrieval.pass3_enabled
    max_tokens: int = settings.retrieval.max_context_tokens
    force_retrieval: bool = False

class PipelineStage(BaseModel):
    stage_name: str
    chunk_count: int
    chunks: List[ChunkDetail]

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
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

class ConversationHistoryResponse(BaseModel):
    session_id: str
    total_turns: int
    turns: List[dict]

class SessionSummary(BaseModel):
    session_id: str
    title: str
    last_message: str
    message_count: int
    created_at: float
    updated_at: float

class SessionListResponse(BaseModel):
    sessions: List[SessionSummary]
    total: int

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_book_filename(filename: str) -> tuple[str, str]:
    name_without_ext = filename.rsplit('.', 1)[0] if '.' in filename else filename
    separator = ' - '
    if separator in name_without_ext:
        first_dash_index = name_without_ext.index(separator)
        title = name_without_ext[:first_dash_index].strip()
        author = name_without_ext[first_dash_index + len(separator):].strip()
    else:
        title = name_without_ext.strip()
        author = "Unknown"
    return title, author

def get_available_books() -> List[BookInfo]:
    try:
        # Try Redis cache first
        cached_books = redis_client.get('books_metadata')
        if cached_books:
            import json
            books_list = json.loads(cached_books)
            return [BookInfo(**b) for b in books_list]

        # If not cached, fetch from Pinecone
        index = get_pinecone_index()
        metadata_namespace = settings.vector_db.metadata_namespace
        results = index.query(
            vector=[1.0] * settings.vector_db.dimension,
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
                    total_chunks=metadata.get("total_chunks", 0),
                    code_chunks=metadata.get("code_chunks", 0),
                    text_chunks=metadata.get("text_chunks", 0),
                    indexed_at=metadata.get("indexed_at")
                ))
        if books_info:
            books_info.sort(key=lambda x: x.indexed_at or 0, reverse=True)
            # Cache in Redis for 10 minutes
            import json
            redis_client.setex('books_metadata', 600, json.dumps([b.dict() for b in books_info]))
            return books_info
        return []
    except Exception as e:
        print(f"❌ Error fetching books: {e}")
        return []

def store_book_metadata(book_title: str, author: str, total_chunks: int, code_chunks: int = 0):
    try:
        index = get_pinecone_index()
        metadata_namespace = settings.vector_db.metadata_namespace
        book_id = hashlib.md5(book_title.encode()).hexdigest()
        index.upsert(
            vectors=[{
                "id": book_id,
                "values": [1.0] * settings.vector_db.dimension,
                "metadata": {
                    "book_title": book_title,
                    "author": author,
                    "total_chunks": total_chunks,
                    "code_chunks": code_chunks,
                    "text_chunks": total_chunks - code_chunks,
                    "indexed_at": time.time()
                }
            }],
            namespace=metadata_namespace
        )
        # Invalidate Redis cache
        redis_client.delete('books_metadata')
    except Exception as e:
        print(f"⚠️ Failed to store metadata: {e}")

def format_chunk_detail(chunk, source: str) -> ChunkDetail:
    if hasattr(chunk, 'relevance_percentage') and chunk.relevance_percentage is not None and chunk.relevance_percentage > 0:
        relevance = chunk.relevance_percentage
    elif hasattr(chunk, 'similarity_score') and chunk.similarity_score is not None:
        relevance = chunk.similarity_score * 100
    elif hasattr(chunk, 'rerank_score') and chunk.rerank_score is not None:
        relevance = chunk.rerank_score * 100
    else:
        relevance = 0.0
    
    return ChunkDetail(
        chunk_id=chunk.chunk.chunk_id,
        chapter=chunk.chunk.chapter,
        page=chunk.chunk.page_number,
        relevance=relevance,
        type=chunk.chunk.chunk_type,
        content_preview=chunk.chunk.content[:200] + "..." if len(chunk.chunk.content) > 200 else chunk.chunk.content,
        source=source,
        book_title=chunk.chunk.book_title or "Unknown Book",
        author=chunk.chunk.author or "Unknown Author"
    )

def extract_pipeline_stages(state: AgentState, executed_nodes: List[str]) -> List[PipelineStage]:
    stage_mapping = {
        "vector_search": "Pass 1: Vector Search",
        "reranking": "Pass 2: Cross-Encoder Reranking",
        "multi_hop_expansion": "Pass 3: Multi-Hop Expansion",
        "cluster_expansion": "Pass 4: Cluster Expansion",
        "context_assembly": "Pass 5: Context Assembly",
        "answer_from_history": "Answered from Memory"
    }
    pipeline_stages = []
    pipeline_snapshots = state.get("pipeline_snapshots", [])
    for snapshot in pipeline_snapshots:
        stage_name = snapshot.get("stage", "")
        display_name = stage_mapping.get(stage_name, stage_name)
        chunks = []
        for chunk in snapshot.get("chunks", [])[:10]:
            chunks.append(format_chunk_detail(chunk, stage_name))
        pipeline_stages.append(PipelineStage(
            stage_name=display_name,
            chunk_count=snapshot.get("chunk_count", 0),
            chunks=chunks
        ))
    return pipeline_stages

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
        ) for turn in pinecone_turns
    ]

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {"status": "online", "message": "RAG Book Bot API v4.3 - Async Postgres + Pinecone", "auth": "Clerk (Enabled)"}

@app.get("/books", response_model=BooksResponse)
async def list_books():
    books = get_available_books()
    return BooksResponse(books=books)

@app.post("/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequest, 
    user_claims: dict = Depends(verify_clerk_token)
):
    global query_graph_app
    session_id = None
    try:
        user_id = user_claims.get("sub")
        # Ensure we don't modify the Pydantic model in place with extra fields unexpectedly
        # Just use local variables for logic
        
        session_id = request.session_id or str(uuid4())
        query_cancellation_events[session_id] = asyncio.Event()
        
        # 🔥 UPDATED: Async Call to Load History
        pinecone_turns = await load_conversation(session_id, max_turns=10)
        conversation_history = convert_pinecone_to_conversation_turns(pinecone_turns)
        
        initial_state = create_initial_state(
            user_query=request.query,
            session_id=session_id,
            user_id=user_id,
            conversation_history=conversation_history,
            max_history_turns=5,
            book_filter=request.book_filter,
            chapter_filter=request.chapter_filter,
            pass1_k=request.pass1_k,
            pass2_k=request.pass2_k,
            pass3_enabled=request.pass3_enabled,
            max_tokens=request.max_tokens
        )
        
        config = {"configurable": {"thread_id": session_id}}
        query_task = asyncio.create_task(query_graph_app.ainvoke(initial_state, config=config))
        active_queries[session_id] = query_task
        
        try:
            final_state = await asyncio.wait_for(query_task, timeout=None)
        except asyncio.CancelledError:
            if session_id in active_queries: del active_queries[session_id]
            if session_id in query_cancellation_events: del query_cancellation_events[session_id]
            raise HTTPException(status_code=499, detail="Query was cancelled by user")
        
        if final_state.get("errors"):
            raise HTTPException(status_code=500, detail=f"Pipeline errors: {'; '.join(final_state['errors'])}")
        
        turn_number = len(conversation_history) + 1
        
        try:
            # 🔥 UPDATED: Async Save with Search Payload
            await save_conversation_turn(
                session_id=session_id,
                turn_number=turn_number,
                user_query=request.query,
                assistant_response=final_state["response"].answer,
                search_summary=final_state["response"].search_summary, # <--- Pass Summary
                resolved_query=final_state.get("resolved_query"),
                needs_retrieval=final_state.get("needs_retrieval", True),
                referenced_turn=final_state.get("referenced_turn"),
                sources_used=[c.chunk.chunk_id for c in final_state.get("reranked_chunks", [])[:5]],
                user_id=user_id
            )
        except Exception as e:
            logger.error(f"[ERROR] History save failed: {e}")
        
        executed_nodes = [s.get("stage") for s in final_state.get("pipeline_snapshots", [])]
        pipeline_stages = extract_pipeline_stages(final_state, executed_nodes)
        
        sources = []
        for rc in final_state.get("reranked_chunks", [])[:5]:
            sources.append({
                "chunk_id": rc.chunk.chunk_id,
                "chapter": rc.chunk.chapter,
                "page": rc.chunk.page_number,
                "relevance": rc.relevance_percentage,
                "type": rc.chunk.chunk_type,
                "book_title": rc.chunk.book_title or "Unknown Book",
                "author": rc.chunk.author or "Unknown Author"
            })
        
        pass1_count = 0
        pass2_count = 0
        pass3_count = 0
        final_count = 0
        
        for snapshot in final_state.get("pipeline_snapshots", []):
            stage = snapshot.get("stage", "")
            chunk_count = snapshot.get("chunk_count", 0)
            if stage == "vector_search":
                pass1_count = chunk_count
            elif stage == "reranking":
                pass2_count = chunk_count
            elif stage == "multi_hop_expansion":
                pass3_count = chunk_count
            elif stage == "context_assembly":
                final_count = chunk_count
        
        if final_count == 0:
            final_count = len(final_state.get("reranked_chunks", []))
        
        stats = {
            "total_stages": len(executed_nodes),
            "executed_nodes": executed_nodes,
            "conversation_turn": turn_number,
            "tokens": len(final_state.get("assembled_context", "").split()),
            "pass1": pass1_count,
            "pass2": pass2_count,
            "pass3": pass3_count,
            "final": final_count
        }
        
        if session_id in active_queries: del active_queries[session_id]
        if session_id in query_cancellation_events: del query_cancellation_events[session_id]
        
        return QueryResponse(
            answer=final_state["response"].answer,
            sources=sources,
            confidence=final_state["response"].confidence,
            stats=stats,
            pipeline_stages=pipeline_stages,
            rewritten_queries=final_state.get("rewritten_queries", []),
            session_id=session_id,
            conversation_turn=turn_number,
            resolved_query=final_state.get("resolved_query"),
            answered_from_history=not final_state.get("needs_retrieval", True),
            needs_context=bool(final_state.get("referenced_turn"))
        )
        
    except Exception as e:
        if session_id in active_queries: del active_queries[session_id]
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.post("/cancel-query")
async def cancel_query(session_id: str = Query(...), user_claims: dict = Depends(verify_clerk_token)):
    try:
        if session_id in query_cancellation_events: query_cancellation_events[session_id].set()
        if session_id in active_queries:
            task = active_queries[session_id]
            if not task.done():
                task.cancel()
                try: await task
                except asyncio.CancelledError: pass
            del active_queries[session_id]
        if session_id in query_cancellation_events: del query_cancellation_events[session_id]
        return {"status": "cancelled", "message": f"Query cancelled for session: {session_id}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# --- BACKGROUND INGESTION TASK ---
def process_ingestion_task(task_id: str, temp_path: str, book_title: str, author: str):
    """Background task to handle ingestion without blocking the HTTP request"""
    try:
        logger.info(f"🧵 Starting background ingestion for {book_title} (Task: {task_id})")
        config = IngestorConfig(
            similarity_threshold=settings.ingestion.similarity_threshold,
            min_chunk_size=settings.ingestion.min_chunk_size,
            max_chunk_size=settings.ingestion.max_chunk_size,
            use_grobid=settings.ingestion.use_grobid,
            debug=False
        )
        ingestor = EnhancedBookIngestorPaddle(config=config)
        
        # Run ingestion with task_id
        result = ingestor.ingest_book(pdf_path=temp_path, book_title=book_title, author=author, task_id=task_id)
        
        store_book_metadata(book_title, author, result.get('chunks', 0), result.get('code_chunks', 0))
        
        if os.path.exists(temp_path):
            os.unlink(temp_path)
            
        logger.info(f"✅ Background ingestion finished for {book_title}")
        
    except Exception as e:
        logger.error(f"❌ Background ingestion failed: {e}")
        # Ensure tracker reports failure to frontend
        tracker = get_tracker(task_id)
        if tracker:
            tracker.add_error(str(e))
            tracker.finish(success=False)
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    finally:
        # Cleanup tracker after delay to allow frontend to receive final message
        time.sleep(5)
        remove_tracker(task_id)

@app.post("/ingest", response_model=IngestResponse)
async def ingest_book(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    book_title: Optional[str] = None,
    author: Optional[str] = None,
    user_claims: dict = Depends(verify_clerk_token)
):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files supported")
    
    tmp_path = None
    try:
        task_id = str(uuid4())
        
        # Initialize a tracker for this specific task
        tracker = create_tracker(task_id)
        tracker.set_loop(asyncio.get_running_loop())
        
        extracted_title, extracted_author = parse_book_filename(file.filename)
        final_book_title = book_title or extracted_title
        final_author = author or extracted_author
        
        # Create temp file
        fd, tmp_path = tempfile.mkstemp(suffix='.pdf')
        os.close(fd)
        
        # Write content asynchronously
        with open(tmp_path, 'wb') as f:
            content = await file.read()
            f.write(content)
            
        logger.info(f"📥 File received: {final_book_title}. Handing off to background task {task_id}.")

        # Add to background tasks
        background_tasks.add_task(
            process_ingestion_task, 
            task_id=task_id,
            temp_path=tmp_path, 
            book_title=final_book_title, 
            author=final_author
        )
        
        # Return success with task_id for WebSocket connection
        return IngestResponse(success=True, result={"task_id": task_id, "message": "Ingestion started"})
        
    except Exception as e:
        if tmp_path and os.path.exists(tmp_path): os.unlink(tmp_path)
        return IngestResponse(success=False, error=str(e))

@app.websocket("/ws/ingest/{task_id}")
async def websocket_ingestion_progress(websocket: WebSocket, task_id: str):
    await websocket.accept()
    
    # Get the specific tracker for this task
    tracker = get_tracker(task_id)
    if not tracker:
        await websocket.close(code=4004, reason="Task not found or expired")
        return

    is_connected = True
    
    async def send_update(state):
        nonlocal is_connected
        if not is_connected: return
        try:
            if websocket.client_state.name == "CONNECTED": 
                await websocket.send_json(state.to_dict())
            else: 
                is_connected = False
        except: pass

    tracker.on_progress(send_update)
    
    # Send initial state immediately
    await send_update(tracker.state)
    
    try:
        while is_connected:
            try: await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
            except (asyncio.TimeoutError, WebSocketDisconnect): continue
    finally:
        is_connected = False
        tracker.remove_callback(send_update)
        try: await websocket.close()
        except: pass

@app.get("/sessions", response_model=SessionListResponse)
async def list_sessions(limit: int = Query(50), user_claims: dict = Depends(verify_clerk_token)):
    user_id = user_claims.get("sub")
    # 🔥 UPDATED: Async Call
    sessions = await list_all_sessions(user_id=user_id, limit=limit)
    return SessionListResponse(sessions=sessions, total=len(sessions))

@app.get("/conversation/{session_id}", response_model=ConversationHistoryResponse)
async def get_conversation_history(session_id: str, user_claims: dict = Depends(verify_clerk_token)):
    user_id = user_claims.get("sub")
    meta = get_session_metadata(session_id)
    if meta:
        owner_id = meta.get("user_id")
        if owner_id and owner_id != user_id:
            raise HTTPException(status_code=403, detail="Unauthorized access to this session")

    # 🔥 UPDATED: Async Call
    turns = await load_conversation(session_id, max_turns=100)
    if not turns: raise HTTPException(status_code=404, detail="Conversation not found")
    return ConversationHistoryResponse(session_id=session_id, total_turns=len(turns), turns=turns)

@app.delete("/conversation/{session_id}")
async def delete_conversation(session_id: str, user_claims: dict = Depends(verify_clerk_token)):
    user_id = user_claims.get("sub")
    meta = get_session_metadata(session_id)
    if meta:
        owner_id = meta.get("user_id")
        if owner_id and owner_id != user_id:
            raise HTTPException(status_code=403, detail="Unauthorized to delete this session")

    # 🔥 UPDATED: Async Call
    result = await delete_session(session_id)
    if not result.get('success'): raise HTTPException(status_code=500, detail="Failed")
    return {"message": "Deleted", "session_id": session_id}

@app.get("/search/sessions")
async def search_sessions(query: str, limit: int = 10, user_claims: dict = Depends(verify_clerk_token)):
    user_id = user_claims.get("sub")
    # 🔥 UPDATED: Async Call
    results = await search_across_sessions(query=query, user_id=user_id, top_k=limit)
    return {"query": query, "results": results, "total": len(results)}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)