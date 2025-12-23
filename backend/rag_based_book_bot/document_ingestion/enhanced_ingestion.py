"""
ENHANCED Ingestor with GROBID + Hierarchical Chunking
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

# Config
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX_NAME", "coding-books")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "books_rag")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
BATCH_SIZE = 100

# GROBID Config
GROBID_URL = os.getenv("GROBID_URL", "http://localhost:8070/api")
GROBID_ENABLED = os.getenv("GROBID_ENABLED", "true").lower() == "true"
GROBID_TIMEOUT = int(os.getenv("GROBID_TIMEOUT", "300"))

logging.basicConfig(level=os.getenv("INGESTOR_LOG_LEVEL", "INFO"))
logger = logging.getLogger("enhanced_ingestion")

@dataclass
class IngestorConfig:
    similarity_threshold: float = 0.75
    min_chunk_size: int = 128
    max_chunk_size: int = 256
    use_grobid: bool = True
    debug: bool = False

class SemanticBookIngestor:
    def __init__(self, config: Optional[IngestorConfig] = None):
        self.config = config or IngestorConfig()
        
        self.hierarchical_chunker = HierarchicalChunker(
            max_chunk_tokens=self.config.max_chunk_size,
            overlap=100
        )
        self.semantic_chunker = create_semantic_chunker(
            similarity_threshold=self.config.similarity_threshold,
            min_chunk_size=self.config.min_chunk_size,
            max_chunk_size=self.config.max_chunk_size
        )
        
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.grobid_parser = GrobidTEIParser()
        self.pinecone_index = self._init_pinecone()
        self.grobid_available = self._check_grobid_health() if (GROBID_ENABLED and self.config.use_grobid) else False

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
        Ingest a PDF book with progress tracking
        
        Args:
            pdf_path: Path to PDF file
            book_title: Title of the book (auto-extracted from filename if not provided)
            author: Author name
            
        Returns:
            Dict with book_id, chunks count, and method used
        """
        try:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF not found: {pdf_path}")
            
            book_id = str(uuid.uuid4())
            if not book_title:
                book_title = os.path.basename(pdf_path).replace('.pdf', '')
            
            logger.info(f"🚀 Ingesting: '{book_title}'")
            
            # Initialize tracker
            tracker = get_progress_tracker()
            tracker.start_ingestion(pdf_path, total_pages=0, book_title=book_title, author=author)
            
            chunks = []
            method = "semantic_fallback"

            # Try GROBID first if available
            if self.grobid_available:
                try:
                    grobid_data = self._process_pdf_with_grobid(pdf_path)
                    if grobid_data and grobid_data.get("success"):
                        logger.info("🌳 Using Hierarchical Tree Chunking")
                        chunks = self.hierarchical_chunker.process_document_tree(
                            grobid_data['sections'], book_title, author
                        )
                        method = "hierarchical"
                        tracker.update_chunks(len(chunks))
                except Exception as e:
                    logger.warning(f"GROBID chunking failed: {e}, falling back to semantic chunking")
                    tracker.add_error(f"GROBID error: {str(e)}")

            # Fallback to semantic chunking if GROBID failed or not available
            if not chunks:
                logger.info("📊 Using Flat Semantic Chunking (Fallback)")
                
                try:
                    with pdfplumber.open(pdf_path) as pdf:
                        pages_text = [{"page": i+1, "text": p.extract_text() or ""} for i, p in enumerate(pdf.pages)]
                        total_pages = len(pdf.pages)
                        total_batches = (total_pages + 19) // 20  # Batch size = 20 pages
                        
                        # Update tracker with total pages
                        tracker.state.total_pages = total_pages
                        
                        # Define progress callback for batch processing
                        def chunking_progress(batch_num: int, current_page: int):
                            tracker.update_batch(batch_num, total_batches, current_page)
                        
                        # Process pages with progress tracking
                        chunks = self.semantic_chunker.chunk_pages_batched(
                            pages_text, 
                            book_title, 
                            author,
                            progress_callback=chunking_progress
                        )
                    
                    # Update tracker after chunking complete
                    tracker.start_chunking()
                    tracker.update_chunks(len(chunks))
                    
                except Exception as e:
                    error_msg = f"Semantic chunking failed: {str(e)}"
                    logger.error(error_msg)
                    tracker.add_error(error_msg)
                    tracker.finish(success=False)
                    raise

            if not chunks:
                error_msg = "No chunks generated from PDF"
                logger.error(error_msg)
                tracker.add_error(error_msg)
                tracker.finish(success=False)
                raise ValueError(error_msg)

            logger.info(f"✅ Generated {len(chunks)} chunks using {method}")
            
            # Start embedding phase
            tracker.start_embedding()
            
            # Embed and upsert with progress tracking
            try:
                self._embed_and_upsert(chunks, book_id, book_title, author, tracker=tracker)
            except Exception as e:
                error_msg = f"Embedding/upsert failed: {str(e)}"
                logger.error(error_msg)
                tracker.add_error(error_msg)
                tracker.finish(success=False)
                raise
            
            # Success
            tracker.finish(success=True)
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
        """
        Generate embeddings and upsert to Pinecone with progress tracking
        
        Args:
            chunks: List of (text, metadata) tuples
            book_id: Unique book identifier
            book_title: Title of the book
            author: Author name
            tracker: Optional progress tracker for real-time updates
        """
        if not self.pinecone_index:
            logger.warning("Pinecone index not initialized, skipping upsert")
            return

        try:
            # Extract texts and generate embeddings
            texts = [c[0] for c in chunks]
            logger.info(f"Generating embeddings for {len(texts)} chunks...")
            
            embeddings = self.embedding_model.encode(
                texts, 
                batch_size=64, 
                show_progress_bar=True
            )
            
            # Update tracker with embedding progress
            if tracker:
                tracker.update_embeddings(len(embeddings))
                logger.info(f"📊 Embeddings generated: {len(embeddings)}/{len(texts)}")
            
            # Prepare vectors for upsert
            vectors = []
            for i, (text, meta) in enumerate(chunks):
                clean_meta = {
                    "text": text,
                    "book_id": book_id,
                    "book_title": book_title,
                    "author": author,
                    "hierarchy_path": meta.get("hierarchy_path", "root"),
                    "hierarchy_level": int(meta.get("hierarchy_level", 0)),
                    "section_title": meta.get("section_title", "Unknown"),
                    "chunk_index": int(meta.get("chunk_index", i)),
                    "chunk_type": meta.get("chunk_type", "text_block"),
                    "page_number": int(meta.get("page_start", 0))
                }
                vectors.append({
                    "id": f"{book_id}_{i}",
                    "values": embeddings[i].tolist(),
                    "metadata": clean_meta
                })
            
            # Upsert vectors in batches with progress tracking
            logger.info(f"Upserting {len(vectors)} vectors to Pinecone...")
            
            if tracker:
                tracker.start_upsert()
            
            for batch_idx in range(0, len(vectors), BATCH_SIZE):
                batch = vectors[batch_idx:batch_idx + BATCH_SIZE]
                self.pinecone_index.upsert(
                    vectors=batch, 
                    namespace=PINECONE_NAMESPACE
                )
                
                # Update tracker after each batch
                if tracker:
                    vectors_upserted = min(batch_idx + BATCH_SIZE, len(vectors))
                    tracker.update_upsert(vectors_upserted)
                    
                    # Log progress
                    if (batch_idx + BATCH_SIZE) % (BATCH_SIZE * 5) == 0 or (batch_idx + BATCH_SIZE) >= len(vectors):
                        logger.info(f"📤 Upserted {vectors_upserted}/{len(vectors)} vectors")
            
            logger.info("✅ Upsert complete.")
            
        except Exception as e:
            error_msg = f"Embedding/upsert error: {str(e)}"
            logger.error(error_msg)
            if tracker:
                tracker.add_error(error_msg)
            raise
# Factory
def EnhancedBookIngestorPaddle(config: Optional[IngestorConfig] = None):
    return SemanticBookIngestor(config)