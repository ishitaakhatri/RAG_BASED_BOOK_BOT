"""
FastAPI Backend - Production-Ready RAG with Conversation Memory & Clerk Authentication
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional, List, Dict, Annotated
import asyncio
import logging
import os
import time
import hashlib
import json
import boto3
from uuid import uuid4
from dotenv import load_dotenv
from redis import Redis

from app_config import get_config, get_settings
from auth_utils import verify_clerk_token

# NEW: Import DB Init & Celery Task
from rag_based_book_bot.db import init_db 
from tasks import ingest_book_task

load_dotenv()
config = get_config()
settings = get_settings()

logger = logging.getLogger("main")
logger.setLevel(config.log_level)
logger.propagate = True

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
# INITIALIZATION
# ============================================================================

# S3 Client for API Uploads
s3_client = boto3.client(
    's3',
    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    region_name=settings.AWS_DEFAULT_REGION
)

# Redis Client for Status Polling
redis_client = Redis.from_url(settings.REDIS_URL, decode_responses=True)

query_graph_app = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern lifespan handler"""
    global query_graph_app
    
    # Initialize Database Tables (Async)
    print("🚀 Initializing Database...")
    await init_db()
    print("✅ Database ready.")
    
    query_graph_app = build_query_graph(enable_persistence=True)
    logger.info(f"🚀 RAG Book Bot API started successfully (Env: {config.environment})")
    
    yield
    
    logger.info("🛑 Application shutting down...")

app = FastAPI(title="RAG Book Bot API", version="5.0.0", lifespan=lifespan)

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
    pass1_k: int = config.retrieval.pass1_top_k
    pass2_k: int = config.retrieval.pass2_top_k
    pass3_enabled: bool = config.retrieval.pass3_enabled
    max_tokens: int = config.retrieval.max_context_tokens
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
        index = get_pinecone_index()
        metadata_namespace = config.vector_db.metadata_namespace
        results = index.query(
            vector=[1.0] * config.vector_db.dimension,
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
            return books_info
        return []
    except Exception as e:
        print(f"❌ Error fetching books: {e}")
        return []

def format_chunk_detail(chunk, source: str) -> ChunkDetail:
    if hasattr(chunk, 'relevance_percentage') and chunk.relevance_percentage is not None and chunk.relevance_percentage > 0:
        relevance = chunk.relevance_percentage
    elif hasattr(chunk, 'similarity_score') and chunk.similarity_score is not None:
        relevance = chunk.similarity_score * 100
    elif hasattr(chunk, 'rerank_score') and chunk.rerank_score is not None:
        relevance = chunk.rerank_score * 100
    else:
        relevance = 0.0
    
    # Ensure content preview is safe
    raw_content = chunk.chunk.content if hasattr(chunk.chunk, 'content') and chunk.chunk.content else ""
    preview = raw_content[:200] + "..." if len(raw_content) > 200 else raw_content
    
    return ChunkDetail(
        chunk_id=chunk.chunk.chunk_id,
        chapter=chunk.chunk.chapter or "Unknown Chapter",
        page=chunk.chunk.page_number,
        relevance=relevance,
        type=chunk.chunk.chunk_type or "text",
        content_preview=preview,
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
            try:
                chunks.append(format_chunk_detail(chunk, stage_name))
            except Exception as e:
                logger.error(f"Error formatting chunk for stage {stage_name}: {e}")
        
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
    return {"status": "online", "message": "RAG Book Bot API v5.0 - Distributed Architecture", "auth": "Clerk (Enabled)"}

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
    start_time = time.time()
    
    try:
        user_id = user_claims.get("sub")
        
        session_id = request.session_id or str(uuid4())
        query_cancellation_events[session_id] = asyncio.Event()
        
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
        
        # Save history
        try:
            await save_conversation_turn(
                session_id=session_id,
                turn_number=turn_number,
                user_query=request.query,
                assistant_response=final_state["response"].answer,
                search_summary=final_state["response"].search_summary,
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
        
        # FIX: Ensure sources contain all fields expected by frontend (title, content, etc.)
        sources = []
        for rc in final_state.get("reranked_chunks", [])[:5]:
            safe_content = rc.chunk.content if hasattr(rc.chunk, 'content') and rc.chunk.content else ""
            sources.append({
                "chunk_id": rc.chunk.chunk_id,
                "chapter": rc.chunk.chapter,
                "page": rc.chunk.page_number,
                "relevance": rc.relevance_percentage,
                "type": rc.chunk.chunk_type,
                "book_title": rc.chunk.book_title or "Unknown Book",
                "title": rc.chunk.book_title or "Unknown Book", # Added for frontend compatibility
                "author": rc.chunk.author or "Unknown Author",
                "content": safe_content # Added for frontend display
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
        
        total_time = time.time() - start_time
        
        stats = {
            "total_stages": len(executed_nodes),
            "executed_nodes": executed_nodes,
            "conversation_turn": turn_number,
            "tokens": len(final_state.get("assembled_context", "").split()),
            "pass1": pass1_count,
            "pass2": pass2_count,
            "pass3": pass3_count,
            "final": final_count,
            "total_time": total_time, # Added for frontend stats
            "retrieval_time": total_time * 0.7, # Estimate if not available
            "generation_time": total_time * 0.3, # Estimate if not available
            "tokens_used": len(final_state.get("assembled_context", "").split()) # Added for frontend stats
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
        logger.error(f"Query processing failed: {e}")
        if session_id and session_id in active_queries: del active_queries[session_id]
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

# --- INGESTION ENDPOINTS (DISTRIBUTED) ---

@app.post("/ingest", response_model=IngestResponse)
def ingest_book(
    file: Annotated[UploadFile, File(...)],
    user_claims: Annotated[dict, Depends(verify_clerk_token)],
    book_title: Optional[str] = None,
    author: Optional[str] = "Unknown"
):
    """
    Synchronous endpoint (using 'def') to handle blocking S3 uploads safely.
    FastAPI runs this in a threadpool, preventing the event loop from blocking.
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files supported")
    
    task_id = str(uuid4())
    s3_key = f"uploads/{task_id}/{file.filename}"
    
    try:
        # 1. Parse Metadata
        extracted_title, extracted_author = parse_book_filename(file.filename)
        final_book_title = book_title or extracted_title
        final_author = author or extracted_author
        
        # 2. Upload to S3 (Blocking I/O)
        logger.info(f"📤 Uploading {file.filename} to S3 bucket {settings.S3_BUCKET_NAME}...")
        s3_client.upload_fileobj(file.file, settings.S3_BUCKET_NAME, s3_key)
        
        # 3. Dispatch Celery Task (Non-blocking)
        logger.info(f"🚀 Dispatching Celery task for {task_id}")
        ingest_book_task.delay(
            task_id=task_id, 
            s3_key=s3_key, 
            book_title=final_book_title, 
            author=final_author
        )
        
        return IngestResponse(
            success=True, 
            result={"task_id": task_id, "message": "Ingestion started"}
        )
        
    except Exception as e:
        logger.error(f"❌ Ingestion initiation failed: {e}")
        return IngestResponse(success=False, error=str(e))

@app.websocket("/ws/ingest/{task_id}")
async def websocket_ingestion_progress(websocket: WebSocket, task_id: str):
    """
    Async WebSocket to poll Redis for ingestion progress.
    """
    await websocket.accept()
    redis_key = f"task:{task_id}"
    
    try:
        while True:
            # Poll Redis (Use to_thread for blocking redis calls)
            data = await asyncio.to_thread(redis_client.get, redis_key)
            
            if data:
                state = json.loads(data)
                await websocket.send_json(state)
                
                if state.get("status") in ["completed", "failed"]:
                    break
            else:
                # If key missing, assume queued or initializing
                await websocket.send_json({"status": "queued", "percentage": 0})
            
            # Poll every 0.5s to balance responsiveness and load
            await asyncio.sleep(0.5)
            
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.close()
        except:
            pass

@app.get("/sessions", response_model=SessionListResponse)
async def list_sessions(limit: int = Query(50), user_claims: dict = Depends(verify_clerk_token)):
    user_id = user_claims.get("sub")
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

    result = await delete_session(session_id)
    if not result.get('success'): raise HTTPException(status_code=500, detail="Failed")
    return {"message": "Deleted", "session_id": session_id}

@app.get("/search/sessions")
async def search_sessions(query: str, limit: int = 10, user_claims: dict = Depends(verify_clerk_token)):
    user_id = user_claims.get("sub")
    results = await search_across_sessions(query=query, user_id=user_id, top_k=limit)
    return {"query": query, "results": results, "total": len(results)}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)