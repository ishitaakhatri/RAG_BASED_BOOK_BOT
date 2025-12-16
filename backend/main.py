"""
FastAPI Backend - GRAPH-BASED EXECUTION
Key changes:
1. Removed manual node orchestration
2. Removed query_pinecone_with_expansion(), convert_matches_to_chunks(), format_chunk_detail()
3. Uses graph.execute(state) for pipeline execution
4. Extracts pipeline stages from ExecutionResult
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import os
import tempfile
import time
import hashlib
from dotenv import load_dotenv
load_dotenv()

from rag_based_book_bot.document_ingestion.enhanced_ingestion import (
    EnhancedBookIngestorPaddle,
    IngestorConfig
)
from rag_based_book_bot.agents.states import AgentState
from rag_based_book_bot.agents.graph import build_query_graph # NEW IMPORT
from rag_based_book_bot.agents.nodes import get_pinecone_index, get_embedding_model

app = FastAPI(title="RAG Book Bot API", version="2.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
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
    content_preview: str
    source: str
    book_title: str = "Unknown Book"
    author: str = "Unknown Author"


class QueryRequest(BaseModel):
    query: str
    book_filter: Optional[str] = None
    chapter_filter: Optional[str] = None
    top_k: int = 5
    pass1_k: int = 50
    pass2_k: int = 15
    pass3_enabled: bool = True
    max_tokens: int = 2500


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


class IngestResponse(BaseModel):
    success: bool
    result: Optional[dict] = None
    error: Optional[str] = None


class BookInfo(BaseModel):
    title: str
    author: str
    total_chunks: int
    indexed_at: Optional[float] = None


class BooksResponse(BaseModel):
    books: List[BookInfo]


# ============================================================================
# HELPER FUNCTIONS (kept for ingestion and book listing)
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


def store_book_metadata(book_title: str, author: str, total_chunks: int):
    """Store book metadata in separate namespace"""
    try:
        index = get_pinecone_index()
        metadata_namespace = "books_metadata"
        book_id = hashlib.md5(book_title.encode()).hexdigest()
        
        index.upsert(
            vectors=[{
                "id": book_id,
                "values": [1.0] * 384,
                "metadata": {
                    "book_title": book_title,
                    "author": author,
                    "total_chunks": total_chunks,
                    "indexed_at": time.time()
                }
            }],
            namespace=metadata_namespace
        )
        print(f"✅ Stored metadata for: {book_title}")
    except Exception as e:
        print(f"⚠️ Failed to store metadata: {e}")


def get_available_books() -> List[BookInfo]:
    """Retrieve all books from metadata namespace"""
    try:
        index = get_pinecone_index()
        metadata_namespace = "books_metadata"
        
        try:
            results = index.query(
                vector=[1.0] * 384,
                top_k=10000,
                namespace=metadata_namespace,
                include_metadata=True
            )
            
            books_info = []
            for match in results.get("matches", []):
                metadata = match.get("metadata", {})
                if metadata.get("book_title"):
                    books_info.append(BookInfo(
                        title=metadata.get("book_title", "Unknown"),
                        author=metadata.get("author", "Unknown"),
                        total_chunks=metadata.get("total_chunks", 0),
                        indexed_at=metadata.get("indexed_at")
                    ))
            
            if books_info:
                books_info.sort(key=lambda x: x.indexed_at or 0, reverse=True)
                return books_info
        except Exception as e:
            print(f"Metadata namespace error: {e}")
        
        # Fallback
        namespace = os.getenv("PINECONE_NAMESPACE", "books_rag")
        books_set = {}
        
        import random
        for _ in range(10):
            random_vector = [random.uniform(-1, 1) for _ in range(384)]
            results = index.query(
                vector=random_vector,
                top_k=1000,
                namespace=namespace,
                include_metadata=True
            )
            
            for match in results.get("matches", []):
                metadata = match.get("metadata", {})
                book_title = metadata.get("book_title")
                if book_title and book_title not in books_set:
                    books_set[book_title] = BookInfo(
                        title=book_title,
                        author=metadata.get("author", "Unknown"),
                        total_chunks=0,
                        indexed_at=None
                    )
        
        return list(books_set.values())
        
    except Exception as e:
        print(f"❌ Error fetching books: {e}")
        return []


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
    """
    Extract pipeline stages from state snapshots
    Maps internal node names to user-friendly stage names
    """
    stage_mapping = {
        "vector_search": "Pass 1: Vector Search",
        "reranking": "Pass 2: Cross-Encoder Reranking",
        "multi_hop_expansion": "Pass 3: Multi-Hop Expansion",
        "cluster_expansion": "Pass 4: Cluster Expansion",
        "context_assembly": "Pass 5: Context Assembly & Deduplication"
    }
    
    pipeline_stages = []
    
    for snapshot in state.pipeline_snapshots:
        stage_name = snapshot.get("stage", "")
        
        # Skip if not in executed nodes
        if stage_name not in executed_nodes:
            continue
        
        # Get user-friendly name
        display_name = stage_mapping.get(stage_name, stage_name)
        
        # Add context to display name
        if snapshot.get("skipped"):
            display_name += " (SKIPPED)"
        elif snapshot.get("new_chunks_added"):
            display_name += f" (+{snapshot['new_chunks_added']} new)"
        elif snapshot.get("removed_duplicates"):
            display_name += f" (-{snapshot['removed_duplicates']} duplicates)"
        
        # Format chunks
        chunks = []
        for chunk in snapshot.get("chunks", [])[:10]:
            chunks.append(format_chunk_detail(chunk, stage_name))
        
        pipeline_stages.append(PipelineStage(
            stage_name=display_name,
            chunk_count=snapshot.get("chunk_count", 0),
            chunks=chunks
        ))
    
    return pipeline_stages


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Health check"""
    return {"status": "online", "message": "RAG Book Bot API - Graph-Based Execution v2.0"}


@app.get("/books", response_model=BooksResponse)
async def list_books():
    """Get list of available books"""
    books = get_available_books()
    return BooksResponse(books=books)


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process query with GRAPH-BASED EXECUTION
    
    Uses LangGraph to orchestrate the 5-pass retrieval pipeline:
    1. Query Parser → 2. Query Rewriter → 3. Vector Search → 
    4. Cross-Encoder Reranking → 5. Multi-Hop Expansion → 
    6. Cluster Expansion → 7. Context Assembly → 8. LLM Reasoning
    """
    try:
        print(f"\n{'='*80}")
        print(f"[NEW QUERY] {request.query}")
        print(f"{'='*80}")
        
        # Initialize state with all parameters
        state = AgentState(
            user_query=request.query,
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
        
        # Extract final state
        final_state = result.final_state
        
        # Check for errors in state
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
        
        # Extract pipeline stages from snapshots
        pipeline_stages = extract_pipeline_stages(final_state, result.executed_nodes)
        
        # Build sources from final chunks
        sources = []
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
            "pass1": next((s.get("chunk_count", 0) for s in final_state.pipeline_snapshots if s.get("stage") == "vector_search"), 0),
            "pass2": next((s.get("chunk_count", 0) for s in final_state.pipeline_snapshots if s.get("stage") == "reranking"), 0),
            "pass3": next((s.get("chunk_count", 0) for s in final_state.pipeline_snapshots if s.get("stage") == "multi_hop_expansion"), 0),
            "final": len(final_state.reranked_chunks),
            "tokens": len(final_state.assembled_context.split()) if final_state.assembled_context else 0,
            "rewritten_queries_count": len(final_state.rewritten_queries)
        }
        
        print(f"\n{'='*80}")
        print(f"[SUCCESS] Pipeline completed: {len(result.executed_nodes)} stages executed")
        print(f"[ANSWER] {len(final_state.response.answer)} characters")
        print(f"{'='*80}\n")
        
        return QueryResponse(
            answer=final_state.response.answer,
            sources=sources,
            confidence=final_state.response.confidence,
            stats=stats,
            pipeline_stages=pipeline_stages,
            rewritten_queries=final_state.rewritten_queries
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
async def ingest_book(
    file: UploadFile = File(...),
    book_title: Optional[str] = None,
    author: Optional[str] = None
):
    """Ingest a new PDF book"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files supported")
    
    try:
        # Auto-extract from filename if not provided
        if not book_title or not author:
            extracted_title, extracted_author = parse_book_filename(file.filename)
            final_book_title = book_title or extracted_title
            final_author = author or extracted_author
            
            print(f"📖 Auto-extracted from filename:")
            print(f"   Title: {final_book_title}")
            print(f"   Author: {final_author}")
        else:
            final_book_title = book_title
            final_author = author
        
        # Save temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Ingest
        config = IngestorConfig(
            similarity_threshold=0.75,
            min_chunk_size=200,
            max_chunk_size=1500,
            debug=False
        )
        ingestor = EnhancedBookIngestorPaddle(config=config)
        
        result = ingestor.ingest_book(
            pdf_path=tmp_path,
            book_title=final_book_title,
            author=final_author
        )
        
        # Store metadata
        store_book_metadata(
            book_title=final_book_title,
            author=final_author,
            total_chunks=result.get('total_chunks', 0)
        )
        
        # Cleanup
        os.unlink(tmp_path)
        
        return IngestResponse(success=True, result=result)
        
    except Exception as e:
        if 'tmp_path' in locals():
            try:
                os.unlink(tmp_path)
            except:
                pass
        
        return IngestResponse(success=False, error=str(e))


@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        index = get_pinecone_index()
        stats = index.describe_index_stats()
        
        return {
            "status": "healthy",
            "version": "2.0.0",
            "execution_mode": "graph-based",
            "pinecone": "connected",
            "total_vectors": stats.get('total_vector_count', 0),
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
