"""
FastAPI Backend - WITH QUERY REWRITING SUPPORT
Key additions:
1. Query rewriting node integration
2. Parallel search with original + rewritten queries
3. Rewritten queries included in API response
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

from rag_based_book_bot.document_ingestion.enhanced_ingestion import (
    EnhancedBookIngestorPaddle,
    IngestorConfig
)
from rag_based_book_bot.agents.nodes import (
    user_query_node,
    query_rewriter_node,  # NEW IMPORT
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
    rewritten_queries: List[str] = []  # NEW FIELD


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


def query_pinecone_with_expansion(query_text, rewritten_queries, top_k=50, book_filter=None, chapter_filter=None):
    """
    NEW: Query Pinecone with original + rewritten queries
    Performs parallel searches and deduplicates results
    """
    try:
        index = get_pinecone_index()
        model = get_embedding_model()
        namespace = os.getenv("PINECONE_NAMESPACE", "books_rag")
        
        # Collect all queries
        all_queries = [query_text] + rewritten_queries
        
        # Build filter
        filter_dict = {}
        if book_filter:
            filter_dict["book_title"] = book_filter
        if chapter_filter:
            filter_dict["chapter_numbers"] = {"$in": [chapter_filter]}
        
        # Store results in dict to deduplicate by chunk_id
        all_results = {}
        
        for query in all_queries:
            query_embedding = model.encode(query).tolist()
            
            results = index.query(
                vector=query_embedding,
                top_k=top_k,
                namespace=namespace,
                filter=filter_dict if filter_dict else None,
                include_metadata=True
            )
            
            # Merge results, keeping highest score for each chunk
            for match in results.get("matches", []):
                chunk_id = match["id"]
                current_score = match.get("score", 0.0)
                
                if chunk_id not in all_results or current_score > all_results[chunk_id].get("score", 0):
                    all_results[chunk_id] = match
        
        # Convert to list and sort by score
        matches = list(all_results.values())
        matches.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        return matches
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


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
            chunk_type="code" if metadata.get("contains_code") else "text",
            book_title=metadata.get("book_title", "Unknown Book"),
            author=metadata.get("author", "Unknown Author"),
            chapter_title=chapter_titles,
            chapter_number=chapter_numbers
        )
        
        chunks.append(RetrievedChunk(
            chunk=chunk,
            similarity_score=match.get("score", 0.0),
            rerank_score=match.get("score", 0.0),
            relevance_percentage=round(match.get("score", 0.0) * 100, 1)
        ))
    
    return chunks


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


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Health check"""
    return {"status": "online", "message": "RAG Book Bot API with Query Rewriting"}


@app.get("/books", response_model=BooksResponse)
async def list_books():
    """Get list of available books"""
    books = get_available_books()
    return BooksResponse(books=books)


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process query with QUERY REWRITING and 5-pass retrieval pipeline
    
    NEW: Generates alternative query formulations before retrieval
    """
    try:
        pipeline_stages = []
        
        # Parse query
        state = AgentState(user_query=request.query)
        state = user_query_node(state)
        
        # NEW: Query Rewriting - Generate alternative formulations
        print("\n[Query Rewriting] Generating alternative queries...")
        state = query_rewriter_node(state, num_variations=3)
        print(f"  → Generated {len(state.rewritten_queries)} variations")
        for i, q in enumerate(state.rewritten_queries, 1):
            print(f"     {i}. {q}")
        
        # Pass 1: Vector Search with query expansion
        print(f"\n[Pass 1] Searching with {len(state.rewritten_queries) + 1} queries...")
        matches = query_pinecone_with_expansion(
            request.query,
            state.rewritten_queries,  # NEW: Pass rewritten queries
            top_k=request.pass1_k,
            book_filter=request.book_filter,
            chapter_filter=request.chapter_filter
        )
        
        if not matches:
            raise HTTPException(status_code=404, detail="No content found")
        
        state.retrieved_chunks = convert_matches_to_chunks(matches)
        
        # PASS 1 STAGE
        pass1_chunks = [format_chunk_detail(chunk, "pass1") for chunk in state.retrieved_chunks]
        pipeline_stages.append(PipelineStage(
            stage_name=f"Pass 1: Vector Search (with {len(state.rewritten_queries)} rewritten queries)",
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
        
        # FINAL STAGE
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
                "page": rc.chunk.page_number,
                "relevance": rc.relevance_percentage,
                "type": rc.chunk.chunk_type,
                "book_title": rc.chunk.book_title or "Unknown Book",
                "author": rc.chunk.author or "Unknown Author"
            }
            for rc in state.reranked_chunks[:5]
        ]
        
        # Calculate stats
        stats = {
            "pass1": pass1_count,
            "pass2": pass2_count,
            "pass3": pass3_count,
            "final": after_compression,
            "tokens": len(state.assembled_context.split()) if state.assembled_context else 0,
            "removed_duplicates": removed_count if removed_count > 0 else 0,
            "rewritten_queries_count": len(state.rewritten_queries)  # NEW
        }
        
        return QueryResponse(
            answer=state.response.answer,
            sources=sources,
            confidence=state.response.confidence,
            stats=stats,
            pipeline_stages=pipeline_stages,
            rewritten_queries=state.rewritten_queries  # NEW: Include in response
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"ERROR: {traceback.format_exc()}")
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
