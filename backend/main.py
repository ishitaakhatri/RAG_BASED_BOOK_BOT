"""
FastAPI Backend for RAG Book Bot with Pipeline Details
Run with: uvicorn main:app --reload
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import os
import tempfile
from dotenv import load_dotenv

# Import your existing RAG components
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

# CORS middleware to allow React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    source: str  # "pass1", "pass2_reranked", "pass3_multihop", "final"

class PipelineStage(BaseModel):
    stage_name: str
    chunk_count: int
    chunks: List[ChunkDetail]

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    confidence: float
    stats: dict
    pipeline_stages: List[PipelineStage]  # NEW: Detailed pipeline data

class IngestResponse(BaseModel):
    success: bool
    result: Optional[dict] = None
    error: Optional[str] = None

class BooksResponse(BaseModel):
    books: List[str]


# Helper Functions
def get_available_books() -> List[str]:
    """Get list of books from Pinecone"""
    try:
        index = get_pinecone_index()
        model = get_embedding_model()
        namespace = os.getenv("PINECONE_NAMESPACE", "books_rag")
        
        results = index.query(
            vector=[0.0] * 384,
            top_k=100,
            namespace=namespace,
            include_metadata=True
        )
        
        books = set()
        for match in results.get("matches", []):
            book_title = match.get("metadata", {}).get("book_title")
            if book_title:
                books.add(book_title)
        
        return sorted(list(books))
    except Exception as e:
        print(f"Error fetching books: {e}")
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


# API Endpoints
@app.get("/")
async def root():
    """Health check"""
    return {"status": "online", "message": "RAG Book Bot API"}


@app.get("/books", response_model=BooksResponse)
async def list_books():
    """Get list of available books"""
    books = get_available_books()
    return BooksResponse(books=books)


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process user query through 5-pass retrieval pipeline WITH DETAILED STAGES
    """
    try:
        # Initialize tracking for pipeline stages
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
        
        # TRACK PASS 1
        pass1_chunks = [format_chunk_detail(chunk, "pass1") for chunk in state.retrieved_chunks]
        pipeline_stages.append(PipelineStage(
            stage_name="Pass 1: Vector Search",
            chunk_count=len(pass1_chunks),
            chunks=pass1_chunks
        ))
        
        # Step 3: Pass 2 - Reranking
        state = reranking_node(state, top_k=request.pass2_k)
        
        # TRACK PASS 2
        pass2_chunks = [format_chunk_detail(chunk, "pass2_reranked") for chunk in state.reranked_chunks]
        pipeline_stages.append(PipelineStage(
            stage_name="Pass 2: Cross-Encoder Reranking",
            chunk_count=len(pass2_chunks),
            chunks=pass2_chunks
        ))
        
        # Remember Pass 2 count for comparison
        pass2_count = len(state.reranked_chunks)
        
        # Step 4: Pass 3 - Multi-Hop (optional)
        if request.pass3_enabled:
            state = multi_hop_expansion_node(state, max_hops=2)
            
            # TRACK PASS 3 (only new chunks added)
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
        
        # TRACK FINAL CHUNKS (after compression/deduplication)
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
            pipeline_stages=pipeline_stages  # NEW: Include pipeline details
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@app.post("/ingest", response_model=IngestResponse)
async def ingest_book(
    file: UploadFile = File(...),
    book_title: Optional[str] = None,
    author: str = "Unknown"
):
    """
    Ingest a new PDF book into the vector database
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Ingest using enhanced ingestor
        config = IngestorConfig(
            similarity_threshold=0.75,
            min_chunk_size=200,
            max_chunk_size=1500,
            debug=False
        )
        ingestor = EnhancedBookIngestorPaddle(config=config)
        
        result = ingestor.ingest_book(
            pdf_path=tmp_path,
            book_title=book_title or file.filename.replace('.pdf', ''),
            author=author
        )
        
        # Cleanup
        os.unlink(tmp_path)
        
        return IngestResponse(success=True, result=result)
        
    except Exception as e:
        # Cleanup on error
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
        # Test Pinecone connection
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