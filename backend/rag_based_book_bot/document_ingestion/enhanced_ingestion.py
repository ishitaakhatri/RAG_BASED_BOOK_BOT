# enhanced_ingestion.py
"""
ENHANCED Ingestor with GROBID + Hierarchical Chunking + Real-time Logging
Optimized for Memory Efficiency and UI Responsiveness
"""
import os
import uuid
import logging
import requests
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from rag_based_book_bot.document_ingestion.progress_tracker import (
    get_progress_tracker, ProgressTracker
)
# PDF processing fallback
import pdfplumber

# Embeddings
from sentence_transformers import SentenceTransformer

# Pinecone
try:
    from pinecone import Pinecone
    _HAS_PINECONE = True
except Exception:
    _HAS_PINECONE = False

# Import Chunkers
from rag_based_book_bot.document_ingestion.ingestion.sementic_chunker import (
    SemanticChunker, create_semantic_chunker
)
from rag_based_book_bot.document_ingestion.ingestion.hierarchical_chunker import (
    HierarchicalChunker
)
from rag_based_book_bot.document_ingestion.ingestion.grobid_parser import (
    GrobidTEIParser
)

# Import shared model getter
from rag_based_book_bot.memory.embedding_utils import get_embedding_model

# Config
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX_NAME", "coding-books")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "books_rag")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

# ✅ Configure logging at module level
logger = logging.getLogger("enhanced_ingestion")
logger.setLevel(logging.INFO)
logger.propagate = True

@dataclass
class IngestorConfig:
    similarity_threshold: float = 0.75
    min_chunk_size: int = 200
    max_chunk_size: int = 1000
    use_grobid: bool = True
    debug: bool = False

class SemanticBookIngestor:
    def __init__(self, config: Optional[IngestorConfig] = None):
        self.config = config or IngestorConfig()
        
        # Initialize or get the shared embedding model
        self.embedding_model = get_embedding_model()
        
        self.hierarchical_chunker = HierarchicalChunker(
            max_chunk_tokens=self.config.max_chunk_size,
            overlap=100
        )
        self.semantic_chunker = create_semantic_chunker(
            similarity_threshold=self.config.similarity_threshold,
            min_chunk_size=self.config.min_chunk_size,
            max_chunk_size=self.config.max_chunk_size,
            embedding_model=self.embedding_model
        )
        
        self.grobid_parser = GrobidTEIParser()
        self.pinecone_index = self._init_pinecone()
        # Check Grobid only if enabled in env and config
        self.grobid_available = self._check_grobid_health() if (os.getenv("GROBID_ENABLED", "true").lower() == "true" and self.config.use_grobid) else False
        
        # Get tracker
        self.tracker = get_progress_tracker()
        logger.info("✅ SemanticBookIngestor initialized")

    def _init_pinecone(self):
        if not _HAS_PINECONE or not PINECONE_API_KEY:
            logger.warning("Pinecone credentials missing.")
            return None
        try:
            pc = Pinecone(api_key=PINECONE_API_KEY)
            return pc.Index(PINECONE_INDEX)
        except Exception as e:
            logger.error(f"Pinecone init failed: {e}")
            return None

    def _check_grobid_health(self) -> bool:
        grobid_url = os.getenv("GROBID_URL", "http://localhost:8070/api")
        try:
            resp = requests.get(f"{grobid_url}/isalive", timeout=2)
            return resp.status_code == 200
        except Exception:
            return False

    def _process_pdf_with_grobid(self, pdf_path: str) -> Optional[Dict]:
        grobid_url = os.getenv("GROBID_URL", "http://localhost:8070/api")
        grobid_timeout = int(os.getenv("GROBID_TIMEOUT", "300"))
        try:
            logger.info("🔬 Sending PDF to GROBID...")
            url = f"{grobid_url}/processFulltextDocument"
            with open(pdf_path, 'rb') as f:
                files = {'input': (os.path.basename(pdf_path), f, 'application/pdf')}
                resp = requests.post(url, files=files, timeout=grobid_timeout)
            
            if resp.status_code == 200:
                return self.grobid_parser.parse_tei_xml(resp.text)
            return None
        except Exception as e:
            logger.warning(f"GROBID error: {e}")
            return None

    def ingest_book(self, pdf_path: str, book_title: Optional[str] = None, author: str = "Unknown") -> Dict:
        """
        Ingest a PDF book with progress tracking and real-time logging
        """
        try:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF not found: {pdf_path}")
            
            book_id = str(uuid.uuid4())
            if not book_title:
                book_title = os.path.basename(pdf_path).replace('.pdf', '')
            
            logger.info(f"🚀 Starting ingestion for: '{book_title}'")
            
            # Reset tracker for new ingestion
            tracker = get_progress_tracker()
            tracker.reset()
            logger.info(f"📋 Tracker reset - starting fresh ingestion")
            
            # Get page count first
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
            
            logger.info(f"📄 PDF loaded: {total_pages} pages")
            tracker.start_ingestion(pdf_path, total_pages=total_pages, book_title=book_title, author=author)
            
            chunks = []
            method = "semantic_fallback"

            # Try GROBID first if available
            if self.grobid_available:
                logger.info("🔬 GROBID is available, attempting hierarchical chunking...")
                try:
                    grobid_data = self._process_pdf_with_grobid(pdf_path)
                    if grobid_data and grobid_data.get("success"):
                        logger.info("🌳 GROBID success! Using Hierarchical Tree Chunking")
                        chunks = self.hierarchical_chunker.process_document_tree(
                            grobid_data['sections'], book_title, author
                        )
                        method = "hierarchical"
                        tracker.update_chunks(len(chunks))
                except Exception as e:
                    tracker.add_log(f"⚠️ GROBID failed: {str(e)}, using semantic chunking", "WARNING")
                    logger.warning(f"GROBID chunking failed: {e}, falling back to semantic chunking")
                    tracker.add_error(f"GROBID error: {str(e)}")

            # Fallback to semantic chunking
            if not chunks:
                logger.info("📊 Falling back to Semantic Chunking")
                logger.info("⚙️ Initializing semantic chunker...")
                
                try:
                    logger.info("📖 Reading PDF pages...")
                    with pdfplumber.open(pdf_path) as pdf:
                        pages_text = [{"page": i+1, "text": p.extract_text() or ""} for i, p in enumerate(pdf.pages)]
                        total_pages = len(pdf.pages)
                        total_batches = (total_pages + 19) // 20
                        
                        logger.info(f"📊 Processing {total_pages} pages in {total_batches} batches")
                        tracker.update_total_pages(total_pages)
                        
                        def chunking_progress(batch_num: int, current_page: int):
                            logger.info(f"⚙️ Processing batch {batch_num}/{total_batches} (page {current_page})")
                            tracker.update_batch(batch_num, total_batches, current_page)
                            # Yield CPU briefly during heavy chunking to keep WebSocket alive
                            time.sleep(0.01)

                        tracker.start_chunking()
                        logger.info("🔄 Starting semantic chunking process...")    
                        
                        chunks = self.semantic_chunker.chunk_pages_batched(
                            pages_text, 
                            book_title, 
                            author,
                            progress_callback=chunking_progress
                        )
                    
                    tracker.update_chunks(len(chunks))
                    
                except Exception as e:
                    error_msg = f"Semantic chunking failed: {str(e)}"
                    logger.error(error_msg)
                    tracker.add_error(error_msg)
                    raise

            if not chunks:
                error_msg = "No chunks generated from PDF"
                logger.error(error_msg)
                tracker.add_error(error_msg)
                raise ValueError(error_msg)

            tracker.add_log(f"✅ Generated {len(chunks)} chunks using {method}")
            logger.info(f"✅ Generated {len(chunks)} chunks using {method}")
            
            # Embed and upsert (Now using optimized batching)
            try:
                self._embed_and_upsert_batched(chunks, book_id, book_title, author, tracker=tracker)
            except Exception as e:
                error_msg = f"Embedding/upsert failed: {str(e)}"
                logger.error(error_msg)
                tracker.add_error(error_msg)
                raise
            
            tracker.finish(success=True)
            tracker.add_log("✅ Ingestion completed successfully")
            logger.info(f"✅ Book ingestion complete: {len(chunks)} chunks, {book_id}")
            
            return {
                "book_id": book_id, 
                "chunks": len(chunks), 
                "method": method,
                "total_pages": total_pages,
            }
        
        except Exception as e:
            logger.error(f"❌ Ingestion failed for '{book_title}': {str(e)}")
            if 'tracker' in locals():
                tracker.add_error(str(e))
                tracker.finish(success=False)
            raise
        
    def _embed_and_upsert_batched(
        self, 
        chunks: List[Tuple[str, Dict]], 
        book_id: str, 
        book_title: str, 
        author: str,
        tracker: Optional[ProgressTracker] = None
    ):
        """
        Memory-optimized generation and upsert
        Processes small batches: Encode -> Upsert -> Release Memory
        """
        if not self.pinecone_index:
            logger.warning("Pinecone index not initialized, skipping upsert")
            return

        total_chunks = len(chunks)
        # Process in small batches to save RAM and CPU time
        # Reduced batch size ensures frequent CPU yielding and updates
        BATCH_SIZE = 32 
        
        logger.info(f"🧠 Starting stream processing for {total_chunks} chunks...")
        
        if tracker:
            tracker.add_log(f"🧠 Processing {total_chunks} chunks (Embedding + Upserting)...")
            # We stay in "embedding" phase visually to prevent jitter, or toggle between them
            tracker.start_embedding()
        
        try:
            for batch_start in range(0, total_chunks, BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, total_chunks)
                chunk_batch = chunks[batch_start:batch_end]
                
                # 1. Prepare texts (Only for this batch)
                texts = [c[0] for c in chunk_batch]
                
                # 2. Generate embeddings (Only for this batch)
                # show_progress_bar=False prevents console clutter
                embeddings = self.embedding_model.encode(
                    texts, 
                    batch_size=BATCH_SIZE, 
                    show_progress_bar=False
                )
                
                # 3. Prepare vectors
                vectors = []
                for i, (text, meta) in enumerate(chunk_batch):
                    global_idx = batch_start + i
                    
                    if "preview" not in meta:
                        preview_text = text.strip()[:100].replace('\n', ' ') + "..."
                        meta["preview"] = preview_text

                    if "chapter_title" not in meta:
                        meta["chapter_title"] = meta.get("section_title", "General Content")
                        if meta["chapter_title"] == "General Content":
                             meta["section_title"] = f"Part {meta.get('chunk_index', global_idx)}"
                    
                    clean_meta = {
                        "text": text,
                        "book_id": book_id,
                        "book_title": book_title,
                        "author": author,
                        "chapter_title": meta.get("chapter_title", "Unknown Chapter"),
                        "section_title": meta.get("section_title", "Unknown Section"),
                        "preview": meta.get("preview", ""),
                        "hierarchy_path": meta.get("hierarchy_path", "root"),
                        "hierarchy_level": int(meta.get("hierarchy_level", 0)),
                        "chunk_index": int(meta.get("chunk_index", global_idx)),
                        "chunk_type": meta.get("chunk_type", "text_block"),
                        "page_number": int(meta.get("page_start", 0))
                    }
                    
                    vectors.append({
                        "id": f"{book_id}_{global_idx}",
                        "values": embeddings[i].tolist(),
                        "metadata": clean_meta
                    })
                
                # 4. Upsert this batch immediately
                self.pinecone_index.upsert(
                    vectors=vectors, 
                    namespace=PINECONE_NAMESPACE
                )
                
                # 5. Update Progress
                if tracker:
                    # Update counts
                    tracker.update_embeddings(batch_end)
                    # We can manually set the upsert count to track progress
                    tracker.state.vectors_upserted = batch_end
                    
                    # Log EVERY batch so the user knows it's working
                    batch_num = (batch_start // BATCH_SIZE) + 1
                    total_batches = (total_chunks + BATCH_SIZE - 1) // BATCH_SIZE
                    
                    msg = f"✅ Processed batch {batch_num}/{total_batches} ({batch_end}/{total_chunks} chunks)"
                    logger.info(msg)
                    tracker.add_log(msg)

                # 6. CRITICAL: Sleep briefly to yield CPU to the main thread
                # Increased to 0.1s to ensure WebSocket has enough time to breathe
                time.sleep(0.1) 
            
            # Finalize progress state
            if tracker:
                tracker.start_upsert() # Jump to 85%
                tracker.update_upsert(total_chunks) # Jump to 95%
                tracker.add_log(f"✅ Successfully processed {total_chunks} chunks")
                
            logger.info("✅ Batch processing complete.")
            
        except Exception as e:
            error_msg = f"Processing error at batch {batch_start}: {str(e)}"
            logger.error(error_msg)
            if tracker:
                tracker.add_error(error_msg)
            raise


def EnhancedBookIngestorPaddle(config: Optional[IngestorConfig] = None):
    return SemanticBookIngestor(config)