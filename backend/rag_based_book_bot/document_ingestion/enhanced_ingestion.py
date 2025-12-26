# enhanced_ingestion.py
"""
ENHANCED Ingestor with GROBID + Hierarchical Chunking + Real-time Logging
"""
import os
import uuid
import logging
import requests
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
BATCH_SIZE = 100

# GROBID Config
GROBID_URL = os.getenv("GROBID_URL", "http://localhost:8070/api")
GROBID_ENABLED = os.getenv("GROBID_ENABLED", "true").lower() == "true"
GROBID_TIMEOUT = int(os.getenv("GROBID_TIMEOUT", "300"))

# ✅ Configure logging at module level
logger = logging.getLogger("enhanced_ingestion")
logger.setLevel(logging.INFO)  # Use logging constant, not string
logger.propagate = True  # Allow logs to propagate to handlers

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
        self.grobid_available = self._check_grobid_health() if (GROBID_ENABLED and self.config.use_grobid) else False
        
        # Get tracker - handlers are set up automatically on first access
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
        try:
            resp = requests.get(f"{GROBID_URL}/isalive", timeout=2)
            return resp.status_code == 200
        except Exception:
            return False

    def _process_pdf_with_grobid(self, pdf_path: str) -> Optional[Dict]:
        try:
            logger.info("🔬 Sending PDF to GROBID...")
            url = f"{GROBID_URL}/processFulltextDocument"
            with open(pdf_path, 'rb') as f:
                files = {'input': (os.path.basename(pdf_path), f, 'application/pdf')}
                resp = requests.post(url, files=files, timeout=GROBID_TIMEOUT)
            
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
            
            # Embed and upsert
            try:
                self._embed_and_upsert(chunks, book_id, book_title, author, tracker=tracker)
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
                "total_pages": total_pages if 'total_pages' in locals() else 0,
                "total_batches": total_batches if 'total_batches' in locals() else 0
            }
        
        except Exception as e:
            logger.error(f"❌ Ingestion failed for '{book_title}': {str(e)}")
            if 'tracker' in locals():
                tracker.add_error(str(e))
                tracker.finish(success=False)
            raise
        
    def _embed_and_upsert(
        self, 
        chunks: List[Tuple[str, Dict]], 
        book_id: str, 
        book_title: str, 
        author: str,
        tracker: Optional[ProgressTracker] = None
    ):
        """Generate embeddings and upsert to Pinecone with progress tracking"""
        if not self.pinecone_index:
            logger.warning("Pinecone index not initialized, skipping upsert")
            return

        try:
            texts = [c[0] for c in chunks]
            
            logger.info(f"🧠 Starting embedding generation for {len(texts)} chunks...")
            
            if tracker:
                tracker.add_log(f"🧠 Generating embeddings for {len(texts)} chunks...")
                tracker.start_embedding()
            
            logger.info("⚡ Running embedding model...")
            embeddings = self.embedding_model.encode(
                texts, 
                batch_size=64, 
                show_progress_bar=False  # Disable progress bar to avoid log clutter
            )
            
            logger.info(f"✅ Embeddings generated: {len(embeddings)}/{len(texts)}")
            
            if tracker:
                tracker.update_embeddings(len(embeddings))
                tracker.add_log(f"✅ Generated {len(embeddings)} embeddings")
            
            # Prepare vectors
            vectors = []
            for i, (text, meta) in enumerate(chunks):
                
                if "preview" not in meta:
                    preview_text = text.strip()[:100].replace('\n', ' ') + "..."
                    meta["preview"] = preview_text

                if "chapter_title" not in meta:
                    meta["chapter_title"] = meta.get("section_title", "General Content")
                    if meta["chapter_title"] == "General Content":
                         meta["section_title"] = f"Part {meta.get('chunk_index', i)}"
                
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
                    "chunk_index": int(meta.get("chunk_index", i)),
                    "chunk_type": meta.get("chunk_type", "text_block"),
                    "page_number": int(meta.get("page_start", 0))
                }
                
                vectors.append({
                    "id": f"{book_id}_{i}",
                    "values": embeddings[i].tolist(),
                    "metadata": clean_meta
                })
            
            logger.info(f"📤 Starting Pinecone upsert for {len(vectors)} vectors...")
            
            if tracker:
                tracker.add_log(f"📤 Upserting {len(vectors)} vectors to Pinecone...")
                tracker.start_upsert()
            
            logger.info(f"📦 Upserting in batches of {BATCH_SIZE}...")
            
            for batch_idx in range(0, len(vectors), BATCH_SIZE):
                batch = vectors[batch_idx:batch_idx + BATCH_SIZE]
                
                # ✅ FIX: Log this less frequently to avoid flooding the websocket channel
                # logger.info(f"📦 Upserting batch {(batch_idx // BATCH_SIZE) + 1}/{(len(vectors) + BATCH_SIZE - 1) // BATCH_SIZE}")
                
                self.pinecone_index.upsert(
                    vectors=batch, 
                    namespace=PINECONE_NAMESPACE
                )
                
                if tracker:
                    vectors_upserted = min(batch_idx + BATCH_SIZE, len(vectors))
                    tracker.update_upsert(vectors_upserted)
                    
                    if (batch_idx + BATCH_SIZE) % (BATCH_SIZE * 5) == 0 or (batch_idx + BATCH_SIZE) >= len(vectors):
                        tracker.add_log(f"📤 Upserted {vectors_upserted}/{len(vectors)} vectors")
                        logger.info(f"📤 Upserted {vectors_upserted}/{len(vectors)} vectors")
            
            if tracker:
                tracker.add_log("✅ Upsert complete")
            
            logger.info("✅ Upsert complete.")
            
        except Exception as e:
            error_msg = f"Embedding/upsert error: {str(e)}"
            logger.error(error_msg)
            if tracker:
                tracker.add_error(error_msg)
            raise


def EnhancedBookIngestorPaddle(config: Optional[IngestorConfig] = None):
    return SemanticBookIngestor(config)