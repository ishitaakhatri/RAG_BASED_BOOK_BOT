"""
ENHANCED Ingestor with GROBID + Semantic Chunking
Now supports structure-aware document parsing via GROBID
"""
import os
import uuid
import logging
import requests
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

# Import GROBID parser
from rag_based_book_bot.document_ingestion.ingestion.grobid_parser import (
    GrobidTEIParser
)

# Config
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX_NAME", "coding-books")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "books_rag")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
BATCH_SIZE = 100  # For Pinecone upsert

# GROBID Config
GROBID_URL = os.getenv("GROBID_URL", "http://localhost:8070/api")
GROBID_ENABLED = os.getenv("GROBID_ENABLED", "true").lower() == "true"
GROBID_TIMEOUT = int(os.getenv("GROBID_TIMEOUT", "300"))

logging.basicConfig(level=os.getenv("INGESTOR_LOG_LEVEL", "INFO"))
logger = logging.getLogger("semantic_ingestor")


@dataclass
class IngestorConfig:
    """Configuration for semantic ingestion"""
    similarity_threshold: float = 0.75  # Lower = more topic splits
    min_chunk_size: int = 200
    max_chunk_size: int = 1500
    debug: bool = False
    use_grobid: bool = True  # NEW: Enable/disable GROBID


class SemanticBookIngestor:
    """
    Enhanced book ingestor with GROBID structure extraction + semantic chunking
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
        
        # Initialize GROBID parser
        self.grobid_parser = GrobidTEIParser()
        
        # Check GROBID availability
        self.grobid_available = False
        if GROBID_ENABLED and self.config.use_grobid:
            self.grobid_available = self._check_grobid_health()
        
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
    
    def _check_grobid_health(self) -> bool:
        """Check if GROBID service is available"""
        try:
            response = requests.get(
                f"{GROBID_URL}/isalive",
                timeout=5
            )
            if response.status_code == 200:
                logger.info(f"✅ GROBID service available at {GROBID_URL}")
                return True
        except Exception as e:
            logger.warning(f"⚠️ GROBID service not available: {e}")
        
        return False
    
    def _process_pdf_with_grobid(self, pdf_path: str) -> Optional[Dict]:
        """
        Process PDF using GROBID service
        
        Returns:
            Parsed data dict or None if failed
        """
        try:
            logger.info("🔬 Processing PDF with GROBID...")
            
            # Call GROBID processFulltextDocument endpoint
            url = f"{GROBID_URL}/processFulltextDocument"
            
            with open(pdf_path, 'rb') as pdf_file:
                files = {
                    'input': (
                        os.path.basename(pdf_path),
                        pdf_file,
                        'application/pdf'
                    )
                }
                
                response = requests.post(
                    url,
                    files=files,
                    timeout=GROBID_TIMEOUT
                )
            
            if response.status_code == 200:
                tei_xml = response.text
                
                # Parse TEI-XML
                parsed_data = self.grobid_parser.parse_tei_xml(tei_xml)
                
                if parsed_data.get("success"):
                    logger.info(f"✅ GROBID extracted {len(parsed_data['sections'])} sections")
                    return parsed_data
                else:
                    logger.warning(f"⚠️ GROBID parsing failed: {parsed_data.get('error')}")
                    return None
            else:
                logger.warning(f"⚠️ GROBID request failed: {response.status_code}")
                return None
                
        except Exception as e:
            logger.warning(f"⚠️ GROBID processing failed: {e}")
            return None
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict]:
        """
        Hybrid PDF extraction: Try GROBID first, fallback to pdfplumber
        
        Returns:
            List of page dicts with text and optional structure info
        """
        # Try GROBID if enabled and available
        if self.grobid_available:
            grobid_data = self._process_pdf_with_grobid(pdf_path)
            
            if grobid_data and grobid_data.get("sections"):
                # Convert GROBID sections to page format
                pages = self.grobid_parser.convert_to_pages_format(grobid_data)
                
                if pages:
                    logger.info(f"✅ Using GROBID extraction: {len(pages)} structured sections")
                    return pages
        
        # Fallback to pdfplumber
        logger.info("📄 Using pdfplumber fallback extraction...")
        return self._extract_with_pdfplumber(pdf_path)
    
    def _extract_with_pdfplumber(self, pdf_path: str) -> List[Dict]:
        """
        Fallback: Simple PDF text extraction with pdfplumber
        Original method without structure
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
                        "text": text,
                        "structure": None  # No structure info from pdfplumber
                    })
                
                # Progress logging
                if page_num % 50 == 0:
                    logger.info(f"Processed {page_num}/{total_pages} pages")
        
        logger.info(f"Extracted text from {len(pages_text)} pages")
        return pages_text
    
    def _embed_and_upsert(
        self,
        chunks: List[Tuple[str, Dict]],
        book_id: str,
        book_title: str,
        author: str
    ):
        """
        Embed chunks and upsert to Pinecone with COMPLETE metadata
        Enhanced to include GROBID structure info
        """
        if not self.pinecone_index:
            logger.warning("Pinecone not configured - skipping upsert")
            return
        
        if not chunks:
            logger.warning("No chunks to embed")
            return
        
        logger.info(f"Embedding and upserting {len(chunks)} chunks for '{book_title}' by {author}...")
        
        # Extract texts for embedding
        texts = [chunk[0] for chunk in chunks]
        
        # Generate embeddings in batches
        logger.info("Generating embeddings...")
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32
        )
        
        # Prepare vectors for upsert with COMPLETE metadata
        vectors = []
        for i, ((chunk_text, metadata), embedding) in enumerate(zip(chunks, embeddings)):
            # Create ENHANCED metadata for Pinecone
            pinecone_metadata = {
                "text": chunk_text[:1000],  # Store preview only
                "book_id": book_id,
                "book_title": book_title,
                "author": author,
                "page_start": int(metadata.get("page_start", 1)),
                "page_end": int(metadata.get("page_end", 1)),
                "chunk_index": int(metadata.get("chunk_index", i)),
                "contains_code": bool(metadata.get("contains_code", False)),
                "token_count": int(metadata.get("token_count", 0)),
                # Legacy fields for compatibility
                "chapter_titles": [],
                "chapter_numbers": [],
                "section_titles": [],
            }
            
            # Add GROBID structure info if available
            if metadata.get("section_title"):
                pinecone_metadata["section_title"] = metadata["section_title"]
            if metadata.get("section_type"):
                pinecone_metadata["section_type"] = metadata["section_type"]
            if metadata.get("section_number"):
                pinecone_metadata["section_number"] = metadata["section_number"]
            if metadata.get("formulas"):
                pinecone_metadata["has_formulas"] = True
                pinecone_metadata["formula_count"] = len(metadata["formulas"])
            
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
        
        logger.info(f"✅ Successfully upserted {len(vectors)} vectors for '{book_title}' to Pinecone")
    
    def ingest_book(
        self,
        pdf_path: str,
        book_title: Optional[str] = None,
        author: str = "Unknown"
    ) -> Dict:
        """
        Main ingestion function - GROBID + semantic chunking
        
        Args:
            pdf_path: Path to PDF file
            book_title: Title of the book (REQUIRED)
            author: Author name (defaults to "Unknown")
            
        Returns:
            Dictionary with ingestion statistics
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        # Generate unique book ID
        book_id = str(uuid.uuid4())
        
        # Ensure we have a book title
        if not book_title:
            book_title = os.path.basename(pdf_path).replace('.pdf', '')
        
        logger.info(f"🚀 Starting enhanced ingestion")
        logger.info(f"   Book: '{book_title}' by {author}")
        logger.info(f"   GROBID: {'Enabled' if self.grobid_available else 'Disabled (using pdfplumber)'}")
        logger.info(f"   Settings: threshold={self.config.similarity_threshold}, "
                   f"chunk_size={self.config.min_chunk_size}-{self.config.max_chunk_size}")
        
        # Step 1: Extract text from PDF (GROBID or pdfplumber)
        pages_text = self.extract_text_from_pdf(pdf_path)
        total_pages = len(pages_text)
        
        if not pages_text:
            raise ValueError("No text extracted from PDF")
        
        # Check if we have structure info from GROBID
        has_structure = any(p.get("structure") for p in pages_text)
        if has_structure:
            logger.info("✅ Using structure-aware chunking (GROBID data available)")
        
        # Step 2: Semantic chunking with structure hints
        logger.info("📊 Starting semantic chunking...")
        chunks = self.chunker.chunk_pages_batched(
            pages_text, 
            book_title,
            author,
            batch_size=20
        )
        
        logger.info(f"✅ Created {len(chunks)} semantic chunks")
        
        # Calculate stats
        total_tokens = sum(meta.get("token_count", 0) for _, meta in chunks)
        avg_tokens = total_tokens / len(chunks) if chunks else 0
        code_chunks = sum(1 for _, meta in chunks if meta.get("contains_code"))
        structured_chunks = sum(1 for _, meta in chunks if meta.get("section_title"))
        
        # Step 3: Embed and upsert to Pinecone
        self._embed_and_upsert(chunks, book_id, book_title, author)
        
        # Return statistics
        result = {
            "title": book_title,
            "author": author,
            "book_id": book_id,
            "total_pages": total_pages,
            "total_chunks": len(chunks),
            "code_chunks": code_chunks,
            "text_chunks": len(chunks) - code_chunks,
            "structured_chunks": structured_chunks,
            "avg_tokens_per_chunk": round(avg_tokens, 1),
            "min_tokens": min((meta.get("token_count", 0) for _, meta in chunks), default=0),
            "max_tokens": max((meta.get("token_count", 0) for _, meta in chunks), default=0),
            "grobid_used": has_structure
        }
        
        logger.info(f"✅ Ingestion complete for '{book_title}'!")
        logger.info(f"📈 Stats: {result}")
        
        return result


# Factory function for backward compatibility
def EnhancedBookIngestorPaddle(config: Optional[IngestorConfig] = None):
    """
    Factory function - returns semantic ingestor with GROBID support.
    Kept for backward compatibility with main.py
    """
    return SemanticBookIngestor(config)
