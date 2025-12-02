"""
FastAPI Backend for RAG Book Bot with Pipeline Details
FIXED: Books list now loads properly using metadata namespace
FIXED: Auto-extract book title and author from filename
"""
import sys
import os
if sys.platform == "win32":
    # Force UTF-8 encoding on Windows
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
    if sys.stderr.encoding != 'utf-8':
        sys.stderr.reconfigure(encoding='utf-8')
    # Also set environment variable for subprocess
    os.environ['PYTHONIOENCODING'] = 'utf-8'

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import os
import tempfile
import time
import hashlib
from dotenv import load_dotenv

from rag_based_book_bot.document_ingestion.enhanced_ingestion import (
    EnhancedBookIngestorPaddle,
    IngestorConfig
)
from rag_based_book_bot.agents.nodes import (
    user_query_node,
    vector_search_node,
    reranking_node,
    multi_hop_expansion_node,
    cluster_expansion_node,
    context_assembly_node,
    llm_reasoning_node,
    get_pinecone_index,
    get_embedding_model
)
from rag_based_book_bot.agents.states import AgentState

load_dotenv()

app = FastAPI(title="RAG Book Bot API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_book_filename(filename: str) -> tuple[str, str]:
    """
    Parse book filename in format: "Title - Author.pdf"
    Uses the FIRST " - " (dash with spaces) as the separator.
    
    Examples:
        "AI Engineering - Chip Huyen.pdf" 
        → ("AI Engineering", "Chip Huyen")
        
        "Developing Apps with GPT-4 and ChatGPT - Olivier Caelen & Marie-Alice Blete.pdf"
        → ("Developing Apps with GPT-4 and ChatGPT", "Olivier Caelen & Marie-Alice Blete")
        
        "Machine Learning - A Probabilistic Perspective - Kevin Murphy.pdf"
        → ("Machine Learning", "A Probabilistic Perspective - Kevin Murphy")
        
        "Simple Book.pdf"
        → ("Simple Book", "Unknown")
    
    Note: Hyphens without spaces (like "Marie-Alice" or "GPT-4") are NOT treated as separators.
    """
    # Remove extension
    name_without_ext = filename.rsplit('.', 1)[0] if '.' in filename else filename
    
    # Find the FIRST occurrence of " - " (with spaces on both sides)
    separator = ' - '
    
    if separator in name_without_ext:
        # Split ONLY on the first " - "
        first_dash_index = name_without_ext.index(separator)
        title = name_without_ext[:first_dash_index].strip()
        author = name_without_ext[first_dash_index + len(separator):].strip()
    else:
        # No separator found - entire name is title
        title = name_without_ext.strip()
        author = "Unknown"
    
    return title, author


# Request/Response Models
class QueryRequest(BaseModel):
    query: str
    book_filter: Optional[str] = None
    chapter_filter: Optional[str] = None
    top_k: int = 5
    pass1_k: int = 50
    pass2_k: int = 15
    pass3_enabled: bool = True
    max_tokens: int = 2500

class ChunkDetail(BaseModel):
    chunk_id: str
    chapter: str
    page: Optional[int]
    relevance: float
    type: str
    content_preview: str
    source: str

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
# FIXED: Book Metadata Management
# ============================================================================

def store_book_metadata(book_title: str, author: str, total_chunks: int):
    """Store book metadata in separate namespace for instant retrieval"""
    try:
        index = get_pinecone_index()
        metadata_namespace = "books_metadata"
        
        # Create deterministic ID
        book_id = hashlib.md5(book_title.encode()).hexdigest()
        
        # Store with dummy vector
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
    """
    FIXED: Retrieve ALL books from metadata namespace
    Falls back to querying main namespace if needed
    """
    try:
        index = get_pinecone_index()
        metadata_namespace = "books_metadata"
        
        # Try to get from metadata namespace (fast)
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
                print(f"✅ Found {len(books_info)} books from metadata namespace")
                # Sort by indexed_at (most recent first)
                books_info.sort(key=lambda x: x.indexed_at or 0, reverse=True)
                return books_info
        except Exception as e:
            print(f"Metadata namespace not found: {e}")
        
        # Fallback: Query main namespace with multiple random vectors
        print("Using fallback: querying main namespace...")
        namespace = os.getenv("PINECONE_NAMESPACE", "books_rag")
        books_set = {}  # Use dict to store book info
        
        # Multiple queries for better coverage
        import random
        for _ in range(10):  # 10 queries with different vectors
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
                        total_chunks=0,  # Unknown from main namespace
                        indexed_at=None
                    )
        
        # Also try zero vector
        results = index.query(
            vector=[0.0] * 384,
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
        
        books_list = list(books_set.values())
        print(f"✅ Found {len(books_list)} books from main namespace")
        return books_list
        
    except Exception as e:
        print(f"❌ Error fetching books: {e}")
        return []


def query_pinecone(query_text, top_k=50, book_filter=None, chapter_filter=None):
    """Query Pinecone for relevant chunks"""
    try:
        index = get_pinecone_index()
        model = get_embedding_model()
        namespace = os.getenv("PINECONE_NAMESPACE", "books_rag")
        
        query_embedding = model.encode(query_text).tolist()
        
        filter_dict = {}
        if book_filter:
            filter_dict["book_title"] = book_filter
        if chapter_filter:
            filter_dict["chapter_numbers"] = {"$in": [chapter_filter]}
        
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=namespace,
            filter=filter_dict if filter_dict else None,
            include_metadata=True
        )
        
        return results.get("matches", [])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pinecone query failed: {str(e)}")


def convert_matches_to_chunks(matches):
    """Convert Pinecone matches to RetrievedChunk objects"""
    from rag_based_book_bot.agents.states import DocumentChunk, RetrievedChunk
    
    chunks = []
    for match in matches:
        metadata = match.get("metadata", {})
        
        chapter_numbers = metadata.get("chapter_numbers", [])
        chapter_titles = metadata.get("chapter_titles", [])
        section_titles = metadata.get("section_titles", [])
        
        chapter_num = chapter_numbers[0] if chapter_numbers else ""
        chapter_title = chapter_titles[0] if chapter_titles else ""
        
        chunk = DocumentChunk(
            chunk_id=match["id"],
            content=metadata.get("text", ""),
            chapter=f"{chapter_num}: {chapter_title}" if chapter_num else chapter_title,
            section=", ".join(section_titles) if section_titles else "",
            page_number=metadata.get("page_start"),
            chunk_type="code" if metadata.get("contains_code") else "text"
        )
        
        chunks.append(RetrievedChunk(
            chunk=chunk,
            similarity_score=match.get("score", 0.0),
            rerank_score=match.get("score", 0.0),
            relevance_percentage=round(match.get("score", 0.0) * 100, 1)
        ))
    
    return chunks


def format_chunk_detail(chunk, source: str) -> ChunkDetail:
    """Format a RetrievedChunk into ChunkDetail for frontend"""
    return ChunkDetail(
        chunk_id=chunk.chunk.chunk_id,
        chapter=chunk.chunk.chapter,
        page=chunk.chunk.page_number,
        relevance=chunk.relevance_percentage,
        type=chunk.chunk.chunk_type,
        content_preview=chunk.chunk.content[:200] + "..." if len(chunk.chunk.content) > 200 else chunk.chunk.content,
        source=source
    )


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Health check"""
    return {"status": "online", "message": "RAG Book Bot API"}


@app.get("/books", response_model=BooksResponse)
async def list_books():
    """Get list of available books with metadata - FIXED!"""
    books = get_available_books()
    return BooksResponse(books=books)


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process user query through 5-pass retrieval pipeline"""
    try:
        pipeline_stages = []
        
        # Step 1: Parse query
        state = AgentState(user_query=request.query)
        state = user_query_node(state)
        
        # Step 2: Pass 1 - Vector Search
        matches = query_pinecone(
            request.query,
            top_k=request.pass1_k,
            book_filter=request.book_filter,
            chapter_filter=request.chapter_filter
        )
        
        if not matches:
            raise HTTPException(status_code=404, detail="No relevant content found")
        
        state.retrieved_chunks = convert_matches_to_chunks(matches)
        
        # Track Pass 1
        pass1_chunks = [format_chunk_detail(chunk, "pass1") for chunk in state.retrieved_chunks]
        pipeline_stages.append(PipelineStage(
            stage_name="Pass 1: Vector Search",
            chunk_count=len(pass1_chunks),
            chunks=pass1_chunks
        ))
        
        # Step 3: Pass 2 - Reranking
        state = reranking_node(state, top_k=request.pass2_k)
        
        pass2_chunks = [format_chunk_detail(chunk, "pass2_reranked") for chunk in state.reranked_chunks]
        pipeline_stages.append(PipelineStage(
            stage_name="Pass 2: Cross-Encoder Reranking",
            chunk_count=len(pass2_chunks),
            chunks=pass2_chunks
        ))
        
        pass2_count = len(state.reranked_chunks)
        
        # Step 4: Pass 3 - Multi-Hop (optional)
        if request.pass3_enabled:
            state = multi_hop_expansion_node(state, max_hops=2)
            
            pass3_new_chunks = [
                format_chunk_detail(chunk, "pass3_multihop") 
                for chunk in state.reranked_chunks[pass2_count:]
            ]
            pipeline_stages.append(PipelineStage(
                stage_name="Pass 3: Multi-Hop Expansion",
                chunk_count=len(pass3_new_chunks),
                chunks=pass3_new_chunks
            ))
        
        # Step 5: Pass 4 - Cluster Expansion
        state = cluster_expansion_node(state)
        
        # Step 6: Pass 5 - Context Assembly
        state = context_assembly_node(state, max_tokens=request.max_tokens)
        
        # Step 7: LLM Reasoning
        state = llm_reasoning_node(state)
        
        # Track Final Chunks
        final_chunks = [format_chunk_detail(chunk, "final") for chunk in state.reranked_chunks[:10]]
        pipeline_stages.append(PipelineStage(
            stage_name="Final: After Compression & Deduplication",
            chunk_count=len(final_chunks),
            chunks=final_chunks
        ))
        
        if state.errors:
            raise HTTPException(status_code=500, detail="; ".join(state.errors))
        
        # Format response
        sources = [
            {
                "chunk_id": rc.chunk.chunk_id,
                "chapter": rc.chunk.chapter,
                "page": rc.chunk.page_number,
                "relevance": rc.relevance_percentage,
                "type": rc.chunk.chunk_type
            }
            for rc in state.reranked_chunks[:5]
        ]
        
        stats = {
            "pass1": len(matches),
            "pass2": request.pass2_k,
            "final": len(state.reranked_chunks),
            "tokens": len(state.assembled_context.split()) if state.assembled_context else 0
        }
        
        return QueryResponse(
            answer=state.response.answer,
            sources=sources,
            confidence=state.response.confidence,
            stats=stats,
            pipeline_stages=pipeline_stages
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@app.post("/ingest", response_model=IngestResponse)
async def ingest_book(
    file: UploadFile = File(...),
    book_title: Optional[str] = None,
    author: Optional[str] = None
):
    """
    Ingest a new PDF book - Auto-extracts title and author from filename
    
    Filename format: "Book Title - Author Name.pdf"
    You can override with explicit book_title and author parameters.
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
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
        
        # Save uploaded file temporarily
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
        
        # FIXED: Store book metadata for instant retrieval
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