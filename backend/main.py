"""
FastAPI Backend - UPDATED for Marker + Hierarchical Chunking
Located at: backend/main.py
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

# NEW: Import Marker ingestor
from rag_based_book_bot.document_ingestion.marker_ingestion import MarkerBookIngestor
from rag_based_book_bot.document_ingestion.marker_config import get_default_config

# Import retrieval components
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

# NEW: Import hierarchical retriever
from rag_based_book_bot.retrieval.hierarchical_retrieval import HierarchicalRetriever

load_dotenv()

app = FastAPI(title="RAG Book Bot API with Marker", version="2.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Marker ingestor (GPU by default)
USE_GPU = os.getenv("USE_GPU", "true").lower() == "true"
marker_ingestor = MarkerBookIngestor(use_gpu=USE_GPU)

# Initialize hierarchical retriever
hierarchical_retriever = None


# ============================================================================
# MODELS
# ============================================================================

class ChunkDetail(BaseModel):
    chunk_id: str
    chapter: str
    section: Optional[str] = None
    subsection: Optional[str] = None
    page: Optional[int]
    relevance: float
    type: str
    content_preview: str
    source: str
    book_title: str = "Unknown Book"
    author: str = "Unknown Author"
    level: int = 1  # NEW: hierarchy level


class QueryRequest(BaseModel):
    query: str
    book_filter: Optional[str] = None
    chapter_filter: Optional[str] = None
    top_k: int = 5
    pass1_k: int = 50
    pass2_k: int = 15
    pass3_enabled: bool = True
    max_tokens: int = 2500
    expand_hierarchy: bool = True  # NEW: expand children


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
        
        return []
        
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
            filter_dict["chapter"] = chapter_filter
        
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=namespace,
            filter=filter_dict if filter_dict else None,
            include_metadata=True
        )
        
        return results.get("matches", [])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


def convert_matches_to_chunks(matches):
    """Convert Pinecone matches to RetrievedChunk objects"""
    from rag_based_book_bot.agents.states import DocumentChunk, RetrievedChunk
    
    chunks = []
    for match in matches:
        metadata = match.get("metadata", {})
        
        # Build chapter string from hierarchy
        chapter_parts = []
        if metadata.get("chapter"):
            chapter_parts.append(metadata["chapter"])
        if metadata.get("section"):
            chapter_parts.append(f" > {metadata['section']}")
        if metadata.get("subsection"):
            chapter_parts.append(f" > {metadata['subsection']}")
        
        chapter_str = "".join(chapter_parts) if chapter_parts else "Unknown"
        
        chunk = DocumentChunk(
            chunk_id=metadata.get("chunk_id", match["id"]),
            content=metadata.get("text", ""),
            chapter=chapter_str,
            section=metadata.get("section", ""),
            page_number=metadata.get("page_start"),
            chunk_type="code" if metadata.get("contains_code") else "text",
            book_title=metadata.get("book_title", "Unknown Book"),
            author=metadata.get("author", "Unknown Author"),
            chapter_title=metadata.get("chapter", ""),
            chapter_number=""
        )
        
        chunks.append(RetrievedChunk(
            chunk=chunk,
            similarity_score=match.get("score", 0.0),
            rerank_score=match.get("score", 0.0),
            relevance_percentage=round(match.get("score", 0.0) * 100, 1)
        ))
    
    return chunks


def format_chunk_detail(chunk, source: str) -> ChunkDetail:
    """Format chunk detail with hierarchy info"""
    metadata = chunk.chunk if hasattr(chunk, 'chunk') else chunk
    
    return ChunkDetail(
        chunk_id=metadata.chunk_id if hasattr(metadata, 'chunk_id') else metadata.get('chunk_id', ''),
        chapter=metadata.chapter if hasattr(metadata, 'chapter') else metadata.get('chapter', ''),
        section=metadata.section if hasattr(metadata, 'section') else metadata.get('section'),
        subsection=metadata.subsection if hasattr(metadata, 'subsection') else metadata.get('subsection'),
        page=metadata.page_number if hasattr(metadata, 'page_number') else metadata.get('page_number'),
        relevance=chunk.relevance_percentage if hasattr(chunk, 'relevance_percentage') else 0.0,
        type=metadata.chunk_type if hasattr(metadata, 'chunk_type') else metadata.get('chunk_type', 'text'),
        content_preview=metadata.content[:200] if hasattr(metadata, 'content') else metadata.get('content', '')[:200],
        source=source,
        book_title=metadata.book_title if hasattr(metadata, 'book_title') else metadata.get('book_title', 'Unknown'),
        author=metadata.author if hasattr(metadata, 'author') else metadata.get('author', 'Unknown'),
        level=metadata.level if hasattr(metadata, 'level') else metadata.get('level', 1)
    )


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "online",
        "message": "RAG Book Bot API with Marker",
        "version": "2.0.0",
        "gpu_enabled": USE_GPU
    }


@app.get("/books", response_model=BooksResponse)
async def list_books():
    """Get list of available books"""
    books = get_available_books()
    return BooksResponse(books=books)


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process query with hierarchical retrieval"""
    global hierarchical_retriever
    
    try:
        pipeline_stages = []
        
        # Initialize hierarchical retriever if needed
        if hierarchical_retriever is None:
            index = get_pinecone_index()
            namespace = os.getenv("PINECONE_NAMESPACE", "books_rag")
            hierarchical_retriever = HierarchicalRetriever(index, namespace)
        
        # Parse query
        state = AgentState(user_query=request.query)
        state = user_query_node(state)
        
        # Pass 1: Vector Search
        matches = query_pinecone(
            request.query,
            top_k=request.pass1_k,
            book_filter=request.book_filter,
            chapter_filter=request.chapter_filter
        )
        
        if not matches:
            raise HTTPException(status_code=404, detail="No content found")
        
        # NEW: Hierarchical expansion
        if request.expand_hierarchy:
            matches = hierarchical_retriever.smart_expansion(matches, request.query)
        
        state.retrieved_chunks = convert_matches_to_chunks(matches)
        
        # Record Pass 1
        pass1_chunks = [format_chunk_detail(chunk, "pass1") for chunk in state.retrieved_chunks]
        pipeline_stages.append(PipelineStage(
            stage_name="Pass 1: Vector Search + Hierarchy",
            chunk_count=len(state.retrieved_chunks),
            chunks=pass1_chunks[:10]
        ))
        
        pass1_count = len(state.retrieved_chunks)
        
        # Pass 2: Reranking
        state = reranking_node(state, top_k=request.pass2_k)
        
        pass2_chunks = [format_chunk_detail(chunk, "pass2_reranked") for chunk in state.reranked_chunks]
        pipeline_stages.append(PipelineStage(
            stage_name="Pass 2: Cross-Encoder Reranking",
            chunk_count=len(state.reranked_chunks),
            chunks=pass2_chunks[:10]
        ))
        
        pass2_count = len(state.reranked_chunks)
        
        # Pass 3: Multi-Hop (optional)
        if request.pass3_enabled:
            before_expansion = len(state.reranked_chunks)
            state = multi_hop_expansion_node(state, max_hops=2)
            after_expansion = len(state.reranked_chunks)
            
            new_chunks_added = after_expansion - before_expansion
            if new_chunks_added > 0:
                pass3_new_chunks = [
                    format_chunk_detail(chunk, "pass3_multihop") 
                    for chunk in state.reranked_chunks[before_expansion:after_expansion]
                ]
                pipeline_stages.append(PipelineStage(
                    stage_name=f"Pass 3: Multi-Hop Expansion (+{new_chunks_added} new)",
                    chunk_count=after_expansion,
                    chunks=pass3_new_chunks[:10]
                ))
        
        pass3_count = len(state.reranked_chunks)
        
        # Pass 4: Cluster Expansion
        state = cluster_expansion_node(state)
        
        # Pass 5: Context Assembly
        before_compression = len(state.reranked_chunks)
        state = context_assembly_node(state, max_tokens=request.max_tokens)
        
        # LLM Reasoning
        state = llm_reasoning_node(state)
        
        # Final stage
        after_compression = len(state.reranked_chunks)
        final_chunks = [format_chunk_detail(chunk, "final") for chunk in state.reranked_chunks]
        
        removed_count = before_compression - after_compression
        pipeline_stages.append(PipelineStage(
            stage_name=f"Final: After Deduplication (-{removed_count} duplicates)" if removed_count > 0 else "Final: Ready for LLM",
            chunk_count=after_compression,
            chunks=final_chunks[:10]
        ))
        
        if state.errors:
            raise HTTPException(status_code=500, detail="; ".join(state.errors))
        
        # Build sources
        sources = [
            {
                "chunk_id": rc.chunk.chunk_id,
                "chapter": rc.chunk.chapter,
                "section": rc.chunk.section,
                "page": rc.chunk.page_number,
                "relevance": rc.relevance_percentage,
                "type": rc.chunk.chunk_type,
                "book_title": rc.chunk.book_title,
                "author": rc.chunk.author
            }
            for rc in state.reranked_chunks[:5]
        ]
        
        # Stats
        stats = {
            "pass1": pass1_count,
            "pass2": pass2_count,
            "pass3": pass3_count,
            "final": after_compression,
            "tokens": len(state.assembled_context.split()) if state.assembled_context else 0,
            "removed_duplicates": removed_count if removed_count > 0 else 0
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
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/ingest", response_model=IngestResponse)
async def ingest_book(
    file: UploadFile = File(...),
    book_title: Optional[str] = None,
    author: Optional[str] = None
):
    """Ingest a new PDF book using Marker"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files supported")
    
    try:
        # Auto-extract from filename if not provided
        if not book_title or not author:
            extracted_title, extracted_author = parse_book_filename(file.filename)
            final_book_title = book_title or extracted_title
            final_author = author or extracted_author
        else:
            final_book_title = book_title
            final_author = author
        
        # Save temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Ingest using Marker
        result = marker_ingestor.ingest_book(
            pdf_path=tmp_path,
            book_title=final_book_title,
            author=final_author,
            save_markdown=True
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
            "pinecone": "connected",
            "total_vectors": stats.get('total_vector_count', 0),
            "books": len(get_available_books()),
            "gpu_enabled": USE_GPU,
            "marker_ready": marker_ingestor is not None
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)