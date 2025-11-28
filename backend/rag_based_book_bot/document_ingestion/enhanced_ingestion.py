"""
SIMPLIFIED Enhanced Ingestor with Semantic Chunking
No hierarchy detection - pure semantic approach
"""
import os
import uuid
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# PDF processing
import pdfplumber

# Embeddings
from sentence_transformers import SentenceTransformer

# Pinecone
try:
    from pinecone import Pinecone, ServerlessSpec
    _HAS_PINECONE = True
except Exception:
    _HAS_PINECONE = False

# Import our semantic chunker
from rag_based_book_bot.document_ingestion.ingestion.sementic_chunker import (
    SemanticChunker, create_semantic_chunker
)

# Config
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX_NAME", "coding-books")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "books_rag")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
BATCH_SIZE = 100  # For Pinecone upsert

logging.basicConfig(level=os.getenv("INGESTOR_LOG_LEVEL", "INFO"))
logger = logging.getLogger("semantic_ingestor")


@dataclass
class IngestorConfig:
    """Configuration for semantic ingestion"""
    similarity_threshold: float = 0.75  # Lower = more topic splits
    min_chunk_size: int = 200
    max_chunk_size: int = 1500
    debug: bool = False


class SemanticBookIngestor:
    """
    Simplified book ingestor using pure semantic chunking.
    No hierarchy detection - just clean semantic boundaries.
    """
    
    def __init__(self, config: Optional[IngestorConfig] = None):
        self.config = config or IngestorConfig()
        
        # Initialize semantic chunker
        self.chunker = create_semantic_chunker(
            similarity_threshold=self.config.similarity_threshold,
            min_chunk_size=self.config.min_chunk_size,
            max_chunk_size=self.config.max_chunk_size
        )
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info(f"Loaded embedding model: {EMBEDDING_MODEL}")
        
        # Initialize Pinecone
        self.pinecone_index = None
        if _HAS_PINECONE and PINECONE_API_KEY:
            try:
                self.pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
                
                # Create index if it doesn't exist
                existing_indexes = [idx.name for idx in self.pinecone_client.list_indexes()]
                if PINECONE_INDEX not in existing_indexes:
                    logger.info(f"Creating Pinecone index: {PINECONE_INDEX}")
                    dim = len(self.embedding_model.encode(["test"])[0])
                    self.pinecone_client.create_index(
                        name=PINECONE_INDEX,
                        dimension=dim,
                        metric="cosine",
                        spec=ServerlessSpec(cloud="aws", region="us-east-1")
                    )
                
                self.pinecone_index = self.pinecone_client.Index(PINECONE_INDEX)
                logger.info(f"Connected to Pinecone index: {PINECONE_INDEX}")
            except Exception as e:
                logger.warning(f"Pinecone init error: {e}")
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict]:
        """
        Simple PDF text extraction - no fancy hierarchy detection.
        Just extracts text page by page.
        """
        pages_text = []
        
        logger.info(f"Extracting text from PDF: {pdf_path}")
        
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            logger.info(f"Total pages: {total_pages}")
            
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                
                if text.strip():
                    pages_text.append({
                        "page": page_num,
                        "text": text
                    })
                
                # Progress logging
                if page_num % 50 == 0:
                    logger.info(f"Processed {page_num}/{total_pages} pages")
        
        logger.info(f"Extracted text from {len(pages_text)} pages")
        return pages_text
    
    def _embed_and_upsert(
        self,
        chunks: List[Tuple[str, Dict]],
        book_id: str
    ):
        """Embed chunks and upsert to Pinecone"""
        if not self.pinecone_index:
            logger.warning("Pinecone not configured - skipping upsert")
            return
        
        if not chunks:
            logger.warning("No chunks to embed")
            return
        
        logger.info(f"Embedding and upserting {len(chunks)} chunks...")
        
        # Extract texts for embedding
        texts = [chunk[0] for chunk in chunks]
        
        # Generate embeddings in batches
        logger.info("Generating embeddings...")
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32
        )
        
        # Prepare vectors for upsert
        vectors = []
        for i, ((chunk_text, metadata), embedding) in enumerate(zip(chunks, embeddings)):
            # Create metadata for Pinecone (flat structure)
            pinecone_metadata = {
                "text": chunk_text[:1000],  # Store preview only
                "book_id": book_id,
                "book_title": metadata.get("book_title", "Unknown"),
                "author": metadata.get("author", "Unknown"),
                "page_start": int(metadata.get("page_start", 1)),
                "page_end": int(metadata.get("page_end", 1)),
                "chunk_index": int(metadata.get("chunk_index", i)),
                "contains_code": bool(metadata.get("contains_code", False)),
                "token_count": int(metadata.get("token_count", 0))
            }
            
            vectors.append({
                "id": str(uuid.uuid4()),
                "values": embedding.tolist(),
                "metadata": pinecone_metadata
            })
        
        # Upsert in batches
        logger.info("Upserting to Pinecone...")
        for i in range(0, len(vectors), BATCH_SIZE):
            batch = vectors[i:i + BATCH_SIZE]
            try:
                self.pinecone_index.upsert(
                    vectors=batch,
                    namespace=PINECONE_NAMESPACE
                )
                batch_num = i // BATCH_SIZE + 1
                total_batches = (len(vectors) + BATCH_SIZE - 1) // BATCH_SIZE
                logger.info(f"Upserted batch {batch_num}/{total_batches}")
            except Exception as e:
                logger.error(f"Failed to upsert batch: {e}")
        
        logger.info(f"✅ Successfully upserted {len(vectors)} vectors to Pinecone")
    
    def ingest_book(
        self,
        pdf_path: str,
        book_title: Optional[str] = None,
        author: str = "Unknown"
    ) -> Dict:
        """
        Main ingestion function - semantic chunking approach.
        
        Args:
            pdf_path: Path to PDF file
            book_title: Title of the book
            author: Author name
            
        Returns:
            Dictionary with ingestion statistics
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        book_id = str(uuid.uuid4())
        book_title = book_title or os.path.basename(pdf_path)
        
        logger.info(f"🚀 Starting semantic ingestion: {book_title}")
        logger.info(f"Settings: threshold={self.config.similarity_threshold}, "
                   f"chunk_size={self.config.min_chunk_size}-{self.config.max_chunk_size}")
        
        # Step 1: Extract text from PDF (simple)
        pages_text = self.extract_text_from_pdf(pdf_path)
        total_pages = len(pages_text)
        
        if not pages_text:
            raise ValueError("No text extracted from PDF")
        
        # Step 2: Semantic chunking
        logger.info("📊 Starting semantic chunking...")
        chunks = self.chunker.chunk_pages_batched(
            pages_text, 
            book_title, 
            author,
            batch_size=20  # Process 20 pages at a time
        )
        
        logger.info(f"✅ Created {len(chunks)} semantic chunks")
        
        # Calculate stats
        total_tokens = sum(meta.get("token_count", 0) for _, meta in chunks)
        avg_tokens = total_tokens / len(chunks) if chunks else 0
        code_chunks = sum(1 for _, meta in chunks if meta.get("contains_code"))
        
        # Step 3: Embed and upsert to Pinecone
        self._embed_and_upsert(chunks, book_id)
        
        # Return statistics
        result = {
            "title": book_title,
            "author": author,
            "total_pages": total_pages,
            "total_chunks": len(chunks),
            "code_chunks": code_chunks,
            "text_chunks": len(chunks) - code_chunks,
            "avg_tokens_per_chunk": round(avg_tokens, 1),
            "min_tokens": min((meta.get("token_count", 0) for _, meta in chunks), default=0),
            "max_tokens": max((meta.get("token_count", 0) for _, meta in chunks), default=0)
        }
        
        logger.info(f"✅ Ingestion complete!")
        logger.info(f"📈 Stats: {result}")
        
        return result


# Factory function for backward compatibility
def EnhancedBookIngestorPaddle(config: Optional[IngestorConfig] = None):
    """
    Factory function - returns semantic ingestor.
    Kept for backward compatibility with app.py
    """
    return SemanticBookIngestor(config)