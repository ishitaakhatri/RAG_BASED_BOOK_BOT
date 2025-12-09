"""
Marker-based PDF to Markdown Conversion with Hierarchical Chunking
Located at: backend/rag_based_book_bot/document_ingestion/marker_ingestion.py
"""
import os
import uuid
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# UPDATED: New Marker API imports
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

# Pinecone for vector storage
from pinecone import Pinecone, ServerlessSpec

# Embeddings
from sentence_transformers import SentenceTransformer

# Our modules
from .marker_config import MarkerConfig, get_default_config
from .hierarchical_chunker import HierarchicalMarkdownChunker

# Config
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX_NAME", "coding-books")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "books_rag")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
BATCH_SIZE = 100

logging.basicConfig(level=os.getenv("INGESTOR_LOG_LEVEL", "INFO"))
logger = logging.getLogger("marker_ingestor")


class MarkerBookIngestor:
    """
    Book ingestion using Marker for PDF to Markdown conversion
    with hierarchical chunking
    """
    
    def __init__(
        self,
        marker_config: Optional[MarkerConfig] = None,
        max_chunk_tokens: int = 1500,
        use_gpu: bool = True
    ):
        """
        Initialize ingestor
        
        Args:
            marker_config: Configuration for Marker
            max_chunk_tokens: Maximum tokens per chunk
            use_gpu: Whether to use GPU (auto-detects if available)
        """
        self.marker_config = marker_config or get_default_config(use_gpu=use_gpu)
        
        # UPDATED: Load Marker models using new API
        logger.info("Loading Marker models...")
        self.artifact_dict = create_model_dict()
        logger.info("✅ Marker models loaded")
        
        # Initialize chunker
        self.chunker = HierarchicalMarkdownChunker(
            max_chunk_tokens=max_chunk_tokens,
            min_chunk_tokens=200,
            overlap_tokens=100
        )
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        
        # Move embedding model to GPU if available
        if self.marker_config.device == "cuda":
            import torch
            self.embedding_model = self.embedding_model.to('cuda')
            logger.info("✅ Embedding model moved to GPU")
        
        # Initialize Pinecone
        self.pinecone_index = None
        if PINECONE_API_KEY:
            try:
                self.pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
                
                # Create index if needed
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
                logger.info(f"✅ Connected to Pinecone index: {PINECONE_INDEX}")
            except Exception as e:
                logger.warning(f"Pinecone init error: {e}")
    
    def convert_pdf_to_markdown(self, pdf_path: str, output_dir: Optional[str] = None) -> str:
            """
            Convert PDF to Markdown using Marker
            
            Args:
                pdf_path: Path to PDF file
                output_dir: Directory to save markdown (optional)
            
            Returns:
                Markdown text
            """
            logger.info(f"📄 Converting PDF to Markdown: {pdf_path}")
            logger.info(f"   Using device: {self.marker_config.device}")
            
            # 1. Prepare configuration dictionary
            #    Note: 'languages' is the key in config, but MarkerConfig uses 'langs'
            marker_kwargs = self.marker_config.to_marker_kwargs()
            converter_config = {
                "max_pages": marker_kwargs.get("max_pages"),
                "batch_multiplier": marker_kwargs.get("batch_multiplier"),
                "languages": marker_kwargs.get("langs"), # Remap 'langs' -> 'languages'
                "disable_image_extraction": not marker_kwargs.get("extract_images", False) # Inverted logic
            }
            
            # Remove None values so defaults take over
            converter_config = {k: v for k, v in converter_config.items() if v is not None}

            # 2. Initialize converter with config
            converter = PdfConverter(
                artifact_dict=self.artifact_dict,
                config=converter_config  # Pass config here!
            )
            
            # 3. Call converter with ONLY filepath
            rendered = converter(pdf_path)
            
            # Extract text and metadata
            markdown_text, _, images = text_from_rendered(rendered)
            metadata = rendered.metadata if hasattr(rendered, 'metadata') else {}
            
            logger.info(f"✅ Conversion complete!")
            logger.info(f"   Pages processed: {metadata.get('pages', 'unknown')}")
            logger.info(f"   Markdown length: {len(markdown_text)} chars")
            
            # Optionally save markdown
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(exist_ok=True, parents=True)
                
                pdf_name = Path(pdf_path).stem
                md_path = output_dir / f"{pdf_name}.md"
                
                with open(md_path, 'w', encoding='utf-8') as f:
                    f.write(markdown_text)
                
                logger.info(f"   Saved markdown to: {md_path}")
            
            return markdown_text
    
    def ingest_book(
        self,
        pdf_path: str,
        book_title: Optional[str] = None,
        author: str = "Unknown",
        save_markdown: bool = True
    ) -> Dict:
        """
        Main ingestion pipeline
        
        Args:
            pdf_path: Path to PDF
            book_title: Book title (auto-extracted from filename if None)
            author: Author name (auto-extracted from filename if None)
            save_markdown: Whether to save intermediate markdown
        
        Returns:
            Ingestion statistics
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        # Auto-extract title/author from filename if not provided
        if not book_title or not author or author == "Unknown":
            extracted_title, extracted_author = self._parse_filename(pdf_path)
            book_title = book_title or extracted_title
            author = author if author != "Unknown" else extracted_author
        
        book_id = str(uuid.uuid4())
        
        logger.info(f"\n{'='*70}")
        logger.info(f"🚀 Starting Marker-based ingestion")
        logger.info(f"{'='*70}")
        logger.info(f"📚 Book: '{book_title}' by {author}")
        logger.info(f"🔧 Device: {self.marker_config.device}")
        logger.info(f"📦 Book ID: {book_id}")
        
        # Step 1: Convert PDF to Markdown
        markdown_dir = "markdown_output" if save_markdown else None
        markdown_text = self.convert_pdf_to_markdown(pdf_path, markdown_dir)
        
        if not markdown_text.strip():
            raise ValueError("No text extracted from PDF")
        
        # Step 2: Hierarchical chunking
        logger.info(f"\n📊 Starting hierarchical chunking...")
        chunks = self.chunker.chunk_markdown(markdown_text, book_title, author)
        
        logger.info(f"✅ Created {len(chunks)} hierarchical chunks")
        
        # Calculate stats
        total_tokens = sum(meta['token_count'] for _, meta in chunks)
        avg_tokens = total_tokens / len(chunks) if chunks else 0
        code_chunks = sum(1 for _, meta in chunks if meta['contains_code'])
        
        # Count hierarchy levels
        level_counts = {}
        for _, meta in chunks:
            level = meta['level']
            level_counts[level] = level_counts.get(level, 0) + 1
        
        logger.info(f"\n📈 Chunk Statistics:")
        logger.info(f"   Total chunks: {len(chunks)}")
        logger.info(f"   Code chunks: {code_chunks}")
        logger.info(f"   Text chunks: {len(chunks) - code_chunks}")
        logger.info(f"   Avg tokens: {avg_tokens:.1f}")
        logger.info(f"   Hierarchy distribution:")
        for level, count in sorted(level_counts.items()):
            level_name = ['Chapter', 'Section', 'Subsection', 'Subsubsection'][level-1] if level <= 4 else f'Level {level}'
            logger.info(f"     {level_name}: {count}")
        
        # Step 3: Embed and upsert to Pinecone
        self._embed_and_upsert(chunks, book_id, book_title, author)
        
        # Return statistics
        result = {
            "title": book_title,
            "author": author,
            "book_id": book_id,
            "total_chunks": len(chunks),
            "code_chunks": code_chunks,
            "text_chunks": len(chunks) - code_chunks,
            "avg_tokens_per_chunk": round(avg_tokens, 1),
            "min_tokens": min((meta['token_count'] for _, meta in chunks), default=0),
            "max_tokens": max((meta['token_count'] for _, meta in chunks), default=0),
            "hierarchy_distribution": level_counts
        }
        
        logger.info(f"\n{'='*70}")
        logger.info(f"✅ Ingestion complete for '{book_title}'!")
        logger.info(f"{'='*70}\n")
        
        return result
    
    def _embed_and_upsert(
        self,
        chunks: List[Tuple[str, Dict]],
        book_id: str,
        book_title: str,
        author: str
    ):
        """Embed chunks and upsert to Pinecone"""
        if not self.pinecone_index:
            logger.warning("Pinecone not configured - skipping upsert")
            return
        
        if not chunks:
            logger.warning("No chunks to embed")
            return
        
        logger.info(f"\n🔢 Embedding and upserting {len(chunks)} chunks...")
        
        # Extract texts
        texts = [chunk_text for chunk_text, _ in chunks]
        
        # Generate embeddings (GPU-accelerated if available)
        logger.info("   Generating embeddings...")
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32 if self.marker_config.device == "cuda" else 16
        )
        
        # Prepare vectors
        vectors = []
        for i, ((chunk_text, metadata), embedding) in enumerate(zip(chunks, embeddings)):
            # Build Pinecone metadata
            pinecone_metadata = {
                "text": chunk_text[:1000],  # Preview only
                "book_id": book_id,
                "book_title": book_title,
                "author": author,
                "chunk_id": metadata['chunk_id'],
                "level": int(metadata['level']),
                "title": metadata['title'],
                # FIX: Use 'or ""' to explicitly convert None to empty string
                "chapter": metadata.get('chapter') or "",
                "section": metadata.get('section') or "",
                "subsection": metadata.get('subsection') or "",
                "subsubsection": metadata.get('subsubsection') or "",
                "chapter_id": metadata.get('chapter_id') or "",
                "section_id": metadata.get('section_id') or "",
                "subsection_id": metadata.get('subsection_id') or "",
                "parent_id": metadata.get('parent_id') or "",
                "children_ids": metadata.get('children_ids') or [],
                "page_start": int(metadata.get('page_start') or 0),
                "page_end": int(metadata.get('page_end') or 0),
                "token_count": int(metadata['token_count']),
                "contains_code": bool(metadata['contains_code']),
            }
            
            vectors.append({
                "id": str(uuid.uuid4()),
                "values": embedding.tolist(),
                "metadata": pinecone_metadata
            })
        
        # Upsert in batches
        logger.info("   Upserting to Pinecone...")
        for i in range(0, len(vectors), BATCH_SIZE):
            batch = vectors[i:i + BATCH_SIZE]
            try:
                self.pinecone_index.upsert(
                    vectors=batch,
                    namespace=PINECONE_NAMESPACE
                )
                batch_num = i // BATCH_SIZE + 1
                total_batches = (len(vectors) + BATCH_SIZE - 1) // BATCH_SIZE
                logger.info(f"   Batch {batch_num}/{total_batches} uploaded")
            except Exception as e:
                logger.error(f"Failed to upsert batch: {e}")
        
        logger.info(f"✅ Successfully upserted {len(vectors)} vectors to Pinecone")
    
    def _parse_filename(self, filename: str) -> Tuple[str, str]:
        """Parse 'Title - Author.pdf' format"""
        name = Path(filename).stem
        separator = ' - '
        
        if separator in name:
            parts = name.split(separator, 1)
            return parts[0].strip(), parts[1].strip()
        else:
            return name.strip(), "Unknown"