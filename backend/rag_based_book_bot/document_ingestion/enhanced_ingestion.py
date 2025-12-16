"""
ENHANCED Ingestor with GROBID + Hierarchical Chunking
"""
import os
import uuid
import logging
import requests
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

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
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
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
    min_chunk_size: int = 200
    max_chunk_size: int = 1000
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
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        book_id = str(uuid.uuid4())
        if not book_title:
            book_title = os.path.basename(pdf_path).replace('.pdf', '')
        
        logger.info(f"🚀 Ingesting: '{book_title}'")
        chunks = []
        method = "semantic_fallback"

        if self.grobid_available:
            grobid_data = self._process_pdf_with_grobid(pdf_path)
            if grobid_data and grobid_data.get("success"):
                logger.info("🌳 Using Hierarchical Tree Chunking")
                chunks = self.hierarchical_chunker.process_document_tree(
                    grobid_data['sections'], book_title, author
                )
                method = "hierarchical"

        if not chunks:
            logger.info("📊 Using Flat Semantic Chunking (Fallback)")
            with pdfplumber.open(pdf_path) as pdf:
                pages_text = [{"page": i+1, "text": p.extract_text() or ""} for i, p in enumerate(pdf.pages)]
            chunks = self.semantic_chunker.chunk_pages_batched(pages_text, book_title, author)

        logger.info(f"✅ Generated {len(chunks)} chunks using {method}")
        self._embed_and_upsert(chunks, book_id, book_title, author)
        
        return {"book_id": book_id, "chunks": len(chunks), "method": method}

    def _embed_and_upsert(self, chunks: List[Tuple[str, Dict]], book_id, book_title, author):
        if not self.pinecone_index: return

        texts = [c[0] for c in chunks]
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embedding_model.encode(texts, batch_size=64, show_progress_bar=True)
        
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

        for i in range(0, len(vectors), BATCH_SIZE):
            self.pinecone_index.upsert(vectors=vectors[i:i+BATCH_SIZE], namespace=PINECONE_NAMESPACE)
        logger.info("✅ Upsert complete.")

# Factory
def EnhancedBookIngestorPaddle(config: Optional[IngestorConfig] = None):
    return SemanticBookIngestor(config)