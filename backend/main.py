"""
FastAPI Backend - Production-Ready RAG with Conversation Memory

Features:
- Persistent session storage in Pinecone
- Smart context resolution with LLM
- Full conversation history management
- Robust error handling
- Code/Text chunk separation
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from fastapi import WebSocket, WebSocketDisconnect
import asyncio
import json
from rag_based_book_bot.document_ingestion.progress_tracker import get_progress_tracker
import os
import tempfile
import time
import hashlib
from uuid import uuid4
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from rag_based_book_bot.document_ingestion.enhanced_ingestion import (
    EnhancedBookIngestorPaddle,
    IngestorConfig
)
from rag_based_book_bot.agents.states import AgentState, ConversationTurn
from rag_based_book_bot.agents.graph import build_query_graph
from rag_based_book_bot.agents.nodes import get_pinecone_index, get_embedding_model

from rag_based_book_bot.memory import (
    save_conversation_turn,
    load_conversation,
    get_conversation_stats,
    delete_session,
    list_all_sessions,
    search_across_sessions
)

app = FastAPI(title="RAG Book Bot API", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------
# STARTUP EVENTS
# -----------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    """Initialize global resources"""
    # Capture the main event loop for the progress tracker
    # This allows worker threads to send WebSocket updates safely
    loop = asyncio.get_running_loop()
    tracker = get_progress_tracker()
    tracker.set_loop(loop)
    print("✅ Initialized progress tracker with main event loop")


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
    top_k: int = 5
    pass1_k: int = 50
    pass2_k: int = 15
    pass3_enabled: bool = True
    max_tokens: int = 2500
    force_retrieval: bool = False  # Override memory detection


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
    needs_context: bool = False  # NEW: indicates if query needed context


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
    """Parse book filename in format: 'Title - Author.pdf'"""
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
    """Retrieve all books from metadata namespace"""
    try:
        index = get_pinecone_index()
        metadata_namespace = "books_metadata"
        
        try:
            # ✅ FIX: Use 1024 dimensions for BGE-M3
            results = index.query(
                vector=[1.0] * 1024,  # ← FIXED from 384
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
        except Exception as e:
            print(f"Metadata namespace error: {e}")
        
        return []
        
    except Exception as e:
        print(f"❌ Error fetching books: {e}")
        return []


def store_book_metadata(book_title: str, author: str, total_chunks: int, code_chunks: int = 0):
    """Store book metadata in separate namespace"""
    try:
        index = get_pinecone_index()
        metadata_namespace = "books_metadata"
        book_id = hashlib.md5(book_title.encode()).hexdigest()
        
        # ✅ FIX: Use 1024 dimensions for BGE-M3
        index.upsert(
            vectors=[{
                "id": book_id,
                "values": [1.0] * 1024,  # ← FIXED from 384
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
        print(f"✅ Stored metadata for: {book_title}")
    except Exception as e:
        print(f"⚠️ Failed to store metadata: {e}")



def format_chunk_detail(chunk, source: str) -> ChunkDetail:
    """Format chunk for API response"""
    return ChunkDetail(
        chunk_id=chunk.chunk.chunk_id,
        chapter=chunk.chunk.chapter,
        page=chunk.chunk.page_number,
        relevance=chunk.relevance_percentage,
        type=chunk.chunk.chunk_type,
        content_preview=chunk.chunk.content[:200] + "..." if len(chunk.chunk.content) > 200 else chunk.chunk.content,
        source=source,
        book_title=chunk.chunk.book_title or "Unknown Book",
        author=chunk.chunk.author or "Unknown Author"
    )


def extract_pipeline_stages(state: AgentState, executed_nodes: List[str]) -> List[PipelineStage]:
    """Extract pipeline stages from state snapshots"""
    stage_mapping = {
        "vector_search": "Pass 1: Vector Search",
        "reranking": "Pass 2: Cross-Encoder Reranking",
        "multi_hop_expansion": "Pass 3: Multi-Hop Expansion",
        "cluster_expansion": "Pass 4: Cluster Expansion",
        "context_assembly": "Pass 5: Context Assembly",
        "answer_from_history": "Answered from Memory"
    }
    
    pipeline_stages = []
    
    for snapshot in state.pipeline_snapshots:
        stage_name = snapshot.get("stage", "")
        
        if stage_name not in executed_nodes:
            continue
        
        display_name = stage_mapping.get(stage_name, stage_name)
        
        if snapshot.get("skipped"):
            display_name += " (SKIPPED)"
        elif snapshot.get("answered_from_memory"):
            display_name = "Answered from Conversation Memory"
        elif snapshot.get("new_chunks_added"):
            display_name += f" (+{snapshot['new_chunks_added']} new)"
        elif snapshot.get("removed_duplicates"):
            display_name += f" (-{snapshot['removed_duplicates']} duplicates)"
        
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
    """Convert Pinecone conversation turns to ConversationTurn objects"""
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


def generate_session_title(first_query: str) -> str:
    """Generate a title for the session from first query"""
    title = first_query[:50]
    if len(first_query) > 50:
        title += "..."
    return title


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "online",
        "message": "RAG Book Bot API v4.0 - Production Ready",
        "features": [
            "graph_execution",
            "persistent_conversation_memory",
            "smart_context_detection",
            "full_session_management",
            "code_text_separation",
            "robust_error_handling"
        ]
    }


@app.get("/books", response_model=BooksResponse)
async def list_books():
    """Get list of available books"""
    books = get_available_books()
    return BooksResponse(books=books)


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process query with smart conversation memory
    
    Features:
    - Loads conversation history from Pinecone
    - LLM determines if context is needed
    - Can answer from history without retrieval
    - Saves conversation turn after response
    - Robust error handling
    """
    try:
        print(f"\n{'='*80}")
        print(f"[NEW QUERY] {request.query}")
        
        # Get or create session
        session_id = request.session_id or str(uuid4())
        
        # Load conversation history
        print(f"[SESSION] {session_id}")
        print(f"[LOADING] Conversation history...")
        
        pinecone_turns = load_conversation(session_id, max_turns=10)
        conversation_history = convert_pinecone_to_conversation_turns(pinecone_turns)
        
        print(f"[HISTORY] Loaded {len(conversation_history)} previous turns")
        print(f"{'='*80}")
        
        # Initialize state
        state = AgentState(
            user_query=request.query,
            conversation_history=conversation_history,
            session_id=session_id,
            user_id=request.user_id,
            max_history_turns=5,
            book_filter=request.book_filter,
            chapter_filter=request.chapter_filter,
            pass1_k=request.pass1_k,
            pass2_k=request.pass2_k,
            pass3_enabled=request.pass3_enabled,
            max_tokens=request.max_tokens
        )
        
        # Build and execute graph
        print("\n[GRAPH] Building query pipeline...")
        query_graph = build_query_graph()
        
        print("[GRAPH] Executing pipeline...")
        result = query_graph.execute(state)
        
        # Check execution result
        if not result.success:
            error_msg = result.error_message or "Unknown pipeline error"
            failed_at = result.failed_node or "unknown stage"
            raise HTTPException(
                status_code=500,
                detail=f"Pipeline failed at {failed_at}: {error_msg}"
            )
        
        final_state = result.final_state
        
        # Check for errors
        if final_state.errors:
            raise HTTPException(
                status_code=500,
                detail=f"Pipeline errors: {'; '.join(final_state.errors)}"
            )
        
        # Check for response
        if not final_state.response:
            raise HTTPException(
                status_code=500,
                detail="Pipeline completed but no response generated"
            )
        
        # Calculate turn number
        turn_number = len(conversation_history) + 1
        
        # Save conversation turn
        print(f"\n[SAVING] Conversation turn to Pinecone...")
        save_result = save_conversation_turn(
            session_id=session_id,
            turn_number=turn_number,
            user_query=request.query,
            assistant_response=final_state.response.answer,
            resolved_query=final_state.resolved_query,
            needs_retrieval=final_state.needs_retrieval,
            referenced_turn=final_state.referenced_turn,
            sources_used=[c.chunk.chunk_id for c in final_state.reranked_chunks[:5]] if final_state.reranked_chunks else [],
            user_id=request.user_id
        )
        
        if save_result.get('success'):
            print(f"✅ Saved turn #{turn_number}: {save_result.get('vector_id')}")
        else:
            print(f"⚠️ Failed to save: {save_result.get('error')}")
        
        # Extract pipeline stages
        pipeline_stages = extract_pipeline_stages(final_state, result.executed_nodes)
        
        # Build sources
        sources = []
        if final_state.reranked_chunks:
            for rc in final_state.reranked_chunks[:5]:
                sources.append({
                    "chunk_id": rc.chunk.chunk_id,
                    "chapter": rc.chunk.chapter,
                    "page": rc.chunk.page_number,
                    "relevance": rc.relevance_percentage,
                    "type": rc.chunk.chunk_type,
                    "book_title": rc.chunk.book_title or "Unknown Book",
                    "author": rc.chunk.author or "Unknown Author"
                })
        
        # Calculate stats
        stats = {
            "total_stages": len(result.executed_nodes),
            "executed_nodes": result.executed_nodes,
            "conversation_turn": turn_number,
            "referenced_turn": final_state.referenced_turn,
            "answered_from_history": not final_state.needs_retrieval,
            "pass1": next((s.get("chunk_count", 0) for s in final_state.pipeline_snapshots if s.get("stage") == "vector_search"), 0) if final_state.needs_retrieval else 0,
            "pass2": next((s.get("chunk_count", 0) for s in final_state.pipeline_snapshots if s.get("stage") == "reranking"), 0) if final_state.needs_retrieval else 0,
            "final": len(final_state.reranked_chunks) if final_state.reranked_chunks else 0,
            "tokens": len(final_state.assembled_context.split()) if final_state.assembled_context else 0,
            "rewritten_queries_count": len(final_state.rewritten_queries)
        }
        
        print(f"\n{'='*80}")
        print(f"[SUCCESS] Pipeline completed")
        print(f"[ANSWER] {len(final_state.response.answer)} characters")
        print(f"[SESSION] Saved as turn #{turn_number}")
        if final_state.referenced_turn:
            print(f"[CONTEXT] Referenced turn #{final_state.referenced_turn}")
        if not final_state.needs_retrieval:
            print(f"[MEMORY] Answered from conversation history")
        print(f"{'='*80}\n")
        
        return QueryResponse(
            answer=final_state.response.answer,
            sources=sources,
            confidence=final_state.response.confidence,
            stats=stats,
            pipeline_stages=pipeline_stages,
            rewritten_queries=final_state.rewritten_queries,
            session_id=session_id,
            conversation_turn=turn_number,
            resolved_query=final_state.resolved_query,
            answered_from_history=not final_state.needs_retrieval,
            needs_context=bool(final_state.referenced_turn)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"\n{'='*80}")
        print(f"[ERROR] {traceback.format_exc()}")
        print(f"{'='*80}\n")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/ingest", response_model=IngestResponse)
def ingest_book(
    file: UploadFile = File(...),
    book_title: Optional[str] = None,
    author: Optional[str] = None
):
    """
    Ingest a new PDF book with real-time progress tracking
    
    CHANGED: Now a synchronous 'def' (not 'async def'). 
    FastAPI will run this in a separate thread pool, preventing the heavy
    ingestion process from blocking the main event loop. This allows 
    WebSockets to stay alive and send progress updates.
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files supported")
    
    tmp_path = None
    try:
        # Reset tracker for new ingestion
        tracker = get_progress_tracker()
        tracker.reset()
        
        # Auto-extract from filename
        if not book_title or not author:
            extracted_title, extracted_author = parse_book_filename(file.filename)
            final_book_title = book_title or extracted_title
            final_author = author or extracted_author
            
            print(f"📖 Auto-extracted: '{final_book_title}' by {final_author}")
        else:
            final_book_title = book_title
            final_author = author
        
        # Save temporarily
        # Since this is a synchronous function now, we use standard sync IO
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            # Synchronous read from the UploadFile wrapper
            content = file.file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        print(f"📥 Received file: {file.filename}")
        print(f"📊 Starting ingestion for '{final_book_title}'")
        
        # Ingest
        config = IngestorConfig(
            similarity_threshold=0.75,
            min_chunk_size=200,
            max_chunk_size=1500,
            debug=False
        )
        ingestor = EnhancedBookIngestorPaddle(config=config)
        
        # This is the heavy blocking call - now runs in thread pool
        result = ingestor.ingest_book(
            pdf_path=tmp_path,
            book_title=final_book_title,
            author=final_author
        )
        
        # Store metadata
        code_chunks = result.get('code_chunks', 0)
        total_chunks = result.get('chunks', 0)
        
        store_book_metadata(
            book_title=final_book_title,
            author=final_author,
            total_chunks=total_chunks,
            code_chunks=code_chunks
        )
        
        # Cleanup
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        
        print(f"✅ Ingestion complete: {total_chunks} chunks from {result.get('total_pages', 0)} pages")
        
        return IngestResponse(success=True, result=result)
        
    except Exception as e:
        print(f"❌ Ingest error: {str(e)}")
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass
        
        return IngestResponse(success=False, error=str(e))

@app.websocket("/ws/ingest")
async def websocket_ingestion_progress(websocket: WebSocket):
    """
    WebSocket endpoint for real-time ingestion progress updates

    Frontend connects with:
    ws://localhost:8000/ws/ingest
    """
    await websocket.accept()
    tracker = get_progress_tracker()

    print("🔌 WebSocket client connected for ingestion progress")

    async def send_update(state):
        """Send progress updates to the WebSocket client"""
        try:
            await websocket.send_json(state.to_dict())
        except Exception as e:
            print(f"⚠️ WebSocket send failed: {e}")

    # ✅ Register callback
    tracker.on_progress(send_update)

    # ✅ Send initial state immediately (prevents blank UI)
    try:
        # FIX: Do not send "completed" status immediately on connection.
        # This prevents the frontend from thinking the NEW job is already finished 
        # based on the OLD job's status.
        initial_state = tracker.get_state()
        if initial_state.get("status") != "completed":
            await websocket.send_json(initial_state)
    except Exception:
        pass

    try:
        # ✅ Push-only connection (no receive loop needed)
        while True:
            await asyncio.sleep(1)

    except WebSocketDisconnect:
        print("🔌 WebSocket client disconnected")

    finally:
        # ✅ IMPORTANT: Remove callback to prevent memory leaks
        tracker.remove_callback(send_update)

        try:
            await websocket.close()
        except Exception:
            pass



@app.get("/sessions", response_model=SessionListResponse)
async def list_sessions(
    user_id: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=100)
):
    """List all conversation sessions"""
    try:
        sessions = list_all_sessions(user_id=user_id, limit=limit)
        return SessionListResponse(
            sessions=sessions,
            total=len(sessions)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list sessions: {str(e)}")


@app.get("/conversation/{session_id}", response_model=ConversationHistoryResponse)
async def get_conversation_history(session_id: str):
    """Get conversation history for a session"""
    try:
        turns = load_conversation(session_id, max_turns=100)
        
        if not turns:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        return ConversationHistoryResponse(
            session_id=session_id,
            total_turns=len(turns),
            turns=turns
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load conversation: {str(e)}")


@app.get("/conversation/{session_id}/stats")
async def get_session_stats(session_id: str):
    """Get statistics for a conversation session"""
    try:
        stats = get_conversation_stats(session_id)
        
        if not stats.get('exists'):
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        return stats
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@app.delete("/conversation/{session_id}")
async def delete_conversation(session_id: str):
    """Delete a conversation"""
    try:
        result = delete_session(session_id)
        
        if not result.get('success'):
            raise HTTPException(status_code=500, detail=result.get('error', 'Unknown error'))
        
        return {
            "message": f"Conversation deleted successfully",
            "session_id": session_id,
            "deleted_turns": result.get('deleted_turns', 0)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete conversation: {str(e)}")


@app.get("/search/sessions")
async def search_sessions(
    query: str = Query(..., min_length=1),
    user_id: Optional[str] = Query(None),
    limit: int = Query(10, ge=1, le=50)
):
    """Search across all conversation sessions"""
    try:
        results = search_across_sessions(
            query=query,
            user_id=user_id,
            top_k=limit
        )
        return {
            "query": query,
            "results": results,
            "total": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        index = get_pinecone_index()
        stats = index.describe_index_stats()
        
        return {
            "status": "healthy",
            "version": "4.0.0",
            "execution_mode": "graph-based",
            "memory_backend": "pinecone",
            "pinecone": "connected",
            "total_vectors": stats.get('total_vector_count', 0),
            "namespaces": {
                "books": stats.get('namespaces', {}).get('books_rag', {}).get('vector_count', 0),
                "conversations": stats.get('namespaces', {}).get('conversations', {}).get('vector_count', 0),
                "metadata": stats.get('namespaces', {}).get('books_metadata', {}).get('vector_count', 0)
            },
            "books": len(get_available_books())
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)