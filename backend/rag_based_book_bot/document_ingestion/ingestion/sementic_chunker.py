"""
Semantic Chunker - Embedding-based text chunking
Enhanced with GROBID structure-aware chunking
"""
import re
import tiktoken
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass
import logging

logger = logging.getLogger("semantic_chunker")


@dataclass
class SemanticChunk:
    """Represents a semantically coherent chunk of text"""
    text: str
    page_start: int
    page_end: int
    token_count: int
    chunk_index: int
    contains_code: bool
    book_title: str = "Unknown Book"
    author: str = "Unknown Author"
    # NEW: Structure fields from GROBID
    section_title: Optional[str] = None
    section_type: Optional[str] = None
    section_number: Optional[str] = None
    formulas: Optional[List[str]] = None


class SemanticChunker:
    """
    Chunks text based on semantic similarity between sentences.
    Enhanced to respect structural boundaries from GROBID.
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        similarity_threshold: float = 0.75,
        min_chunk_size: int = 150,
        max_chunk_size: int = 300,
        encoding_name: str = "cl100k_base",
        embedding_model: Optional[SentenceTransformer] = None
    ):
        """
        Args:
            model_name: Sentence transformer model for embeddings
            similarity_threshold: Cosine similarity threshold (0-1). 
                                Lower = more topic splits
            min_chunk_size: Minimum tokens per chunk
            max_chunk_size: Maximum tokens per chunk
            embedding_model: Optional pre-loaded SentenceTransformer instance
        """
        if embedding_model:
            self.model = embedding_model
        else:
            logger.info(f"Loading embedding model: {model_name}")
            self.model = SentenceTransformer(model_name)
            
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.tokenizer = tiktoken.get_encoding(encoding_name)
        
        logger.info(f"SemanticChunker initialized: threshold={similarity_threshold}, "
                   f"chunk_size={min_chunk_size}-{max_chunk_size}")
    
    def chunk_pages_batched(
        self,
        pages_text: List[Dict],
        book_title: str,
        author: str,
        batch_size: int = 20,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Tuple[str, Dict]]:
        """
        Chunk pages in batches with structure awareness
        
        Args:
            pages_text: List of dicts with 'page', 'text', and optional 'structure' keys
            book_title: Title of the book
            author: Author name
            batch_size: Number of pages to process at once
            progress_callback: Optional callback function(batch_num: int, current_page: int)
            
        Returns:
            List of (chunk_text, metadata) tuples
        """
        all_chunks = []
        global_chunk_idx = 0
        total_pages = len(pages_text)
        
        # Check if we have structure info
        has_structure = any(p.get("structure") for p in pages_text)
        
        logger.info(f"Starting batched processing of {total_pages} pages for '{book_title}' by {author}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Structure info: {'Available (GROBID)' if has_structure else 'Not available'}")
        
        if has_structure:
            # Structure-aware chunking (GROBID data)
            all_chunks = self._chunk_with_structure(
                pages_text,
                book_title,
                author
            )
        else:
            # Original semantic chunking (no structure)
            all_chunks = self._chunk_without_structure(
                pages_text,
                book_title,
                author,
                batch_size,
                progress_callback
            )

        logger.info(f"✅ Created {len(all_chunks)} semantic chunks from {total_pages} pages")
        logger.info(f"   Book: '{book_title}' by {author}")
        return all_chunks
    
    def _chunk_with_structure(
        self,
        pages_text: List[Dict],
        book_title: str,
        author: str
    ) -> List[Tuple[str, Dict]]:
        """
        Structure-aware chunking using GROBID section boundaries
        Respects section boundaries as hard splits
        """
        all_chunks = []
        global_chunk_idx = 0
        
        logger.info("🏗️ Using structure-aware chunking (GROBID mode)")
        
        for page_data in pages_text:
            text = page_data["text"]
            page_num = page_data["page"]
            structure = page_data.get("structure", {})
            
            if not text.strip():
                continue
            
            # Extract structure info
            section_title = structure.get("title", "") if structure else ""
            section_type = structure.get("type", "") if structure else ""
            section_number = structure.get("section_number", "") if structure else ""
            formulas = structure.get("formulas", []) if structure else []
            contains_code = structure.get("contains_code", False) if structure else False
            
            # Check if this section is small enough to be one chunk
            token_count = self._count_tokens(text)
            
            if token_count <= self.max_chunk_size and token_count >= self.min_chunk_size:
                # Single chunk for this section
                metadata = {
                    "book_title": book_title,
                    "author": author,
                    "page_start": page_num,
                    "page_end": page_num,
                    "chunk_index": global_chunk_idx,
                    "contains_code": contains_code or self._detect_code(text),
                    "token_count": token_count,
                    "section_title": section_title,
                    "section_type": section_type,
                    "section_number": section_number,
                    "formulas": formulas
                }
                
                all_chunks.append((text, metadata))
                global_chunk_idx += 1
                
            elif token_count > self.max_chunk_size:
                # Section too large, apply semantic chunking within section
                logger.info(f"  Section '{section_title}' too large ({token_count} tokens), applying semantic split")
                
                section_chunks = self._chunk_text_semantic_fast(
                    text,
                    page_num,
                    page_num,
                    book_title,
                    author,
                    section_info={
                        "title": section_title,
                        "type": section_type,
                        "number": section_number,
                        "formulas": formulas
                    }
                )
                
                # Convert to tuples with metadata
                for chunk in section_chunks:
                    metadata = {
                        "book_title": book_title,
                        "author": author,
                        "page_start": chunk.page_start,
                        "page_end": chunk.page_end,
                        "chunk_index": global_chunk_idx,
                        "contains_code": chunk.contains_code,
                        "token_count": chunk.token_count,
                        "section_title": chunk.section_title,
                        "section_type": chunk.section_type,
                        "section_number": chunk.section_number,
                        "formulas": chunk.formulas or []
                    }
                    
                    all_chunks.append((chunk.text, metadata))
                    global_chunk_idx += 1
            
            else:
                # Section too small, might merge with next in future enhancement
                # For now, skip or merge with previous
                if token_count > 0:
                    logger.info(f"  Section '{section_title}' below minimum ({token_count} tokens), keeping as-is")
                    metadata = {
                        "book_title": book_title,
                        "author": author,
                        "page_start": page_num,
                        "page_end": page_num,
                        "chunk_index": global_chunk_idx,
                        "contains_code": contains_code or self._detect_code(text),
                        "token_count": token_count,
                        "section_title": section_title,
                        "section_type": section_type,
                        "section_number": section_number,
                        "formulas": formulas
                    }
                    
                    all_chunks.append((text, metadata))
                    global_chunk_idx += 1
        
        return all_chunks
    
    def _chunk_without_structure(
        self,
        pages_text: List[Dict],
        book_title: str,
        author: str,
        batch_size: int,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Tuple[str, Dict]]:
        """
        Original semantic chunking without structure hints
        Fallback when GROBID data not available
        """
        all_chunks = []
        global_chunk_idx = 0
        total_pages = len(pages_text)
        
        logger.info("📄 Using pure semantic chunking (no structure info)")
        
        # Process in batches
        for batch_start in range(0, total_pages, batch_size):
            batch_end = min(batch_start + batch_size, total_pages)
            batch_pages = pages_text[batch_start:batch_end]
            
            # Progress
            logger.info(f"📊 Processing pages {batch_start+1}-{batch_end} ({batch_end/total_pages*100:.1f}% complete)")
            
           

            
            # Combine batch pages into one text block
            combined_text = ""
            page_boundaries = []
            
            for page_data in batch_pages:
                page_boundaries.append(len(combined_text))
                combined_text += page_data["text"] + "\n\n"
            
            if not combined_text.strip():
                continue
            
            # Chunk the combined text
            batch_chunks = self._chunk_text_semantic_fast(
                combined_text,
                batch_start + 1,
                batch_end,
                book_title,
                author
            )

            if progress_callback:
                batch_num = (batch_start // batch_size) + 1
                progress_callback(batch_num, batch_end)
            
            # Add to results with metadata
            for chunk in batch_chunks:
                metadata = {
                    "book_title": book_title,
                    "author": author,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "chunk_index": global_chunk_idx,
                    "contains_code": chunk.contains_code,
                    "token_count": chunk.token_count
                }
                
                all_chunks.append((chunk.text, metadata))
                global_chunk_idx += 1
        
        return all_chunks

    def _chunk_text_semantic_fast(
        self, 
        text: str, 
        page_start: int, 
        page_end: int,
        book_title: str,
        author: str,
        section_info: Optional[Dict] = None
    ) -> List[SemanticChunk]:
        """
        Faster semantic chunking with optional section metadata
        
        Args:
            section_info: Optional dict with 'title', 'type', 'number', 'formulas'
        """
        sentences = self._split_sentences(text)
        
        if not sentences:
            return []
        
        # Embed in one batch
        logger.debug(f"   Embedding {len(sentences)} sentences...")
        embeddings = self.model.encode(
            sentences,
            show_progress_bar=False,
            convert_to_numpy=True,
            batch_size=16
        )
        
        # Group sentences by similarity
        chunks = []
        current_sentences = [sentences[0]]
        current_embedding = embeddings[0]
        
        for i in range(1, len(sentences)):
            similarity = cosine_similarity(
                [current_embedding],
                [embeddings[i]]
            )[0][0]
            
            current_text = ' '.join(current_sentences)
            current_tokens = self._count_tokens(current_text)
            
            should_split = (
                similarity < self.similarity_threshold or
                current_tokens >= self.max_chunk_size
            )
            
            if should_split and current_tokens >= self.min_chunk_size:
                chunk = self._create_chunk(
                    current_sentences,
                    page_start,
                    page_end,
                    len(chunks),
                    book_title,
                    author,
                    section_info
                )
                chunks.append(chunk)
                
                current_sentences = [sentences[i]]
                current_embedding = embeddings[i]
            else:
                current_sentences.append(sentences[i])
                current_embedding = (current_embedding + embeddings[i]) / 2
        
        # Add final chunk
        if current_sentences:
            final_text = ' '.join(current_sentences)
            if self._count_tokens(final_text) >= self.min_chunk_size:
                chunk = self._create_chunk(
                    current_sentences,
                    page_start,
                    page_end,
                    len(chunks),
                    book_title,
                    author,
                    section_info
                )
                chunks.append(chunk)
            elif chunks:
                # Merge with last chunk if too small
                chunks[-1].text += ' ' + final_text
                chunks[-1].token_count = self._count_tokens(chunks[-1].text)
        
        return chunks
    
    def _create_chunk(
        self,
        sentences: List[str],
        page_start: int,
        page_end: int,
        chunk_idx: int,
        book_title: str,
        author: str,
        section_info: Optional[Dict] = None
    ) -> SemanticChunk:
        """Create a SemanticChunk with optional structure metadata"""
        text = ' '.join(sentences)
        
        # Extract section info if provided
        section_title = section_info.get("title") if section_info else None
        section_type = section_info.get("type") if section_info else None
        section_number = section_info.get("number") if section_info else None
        formulas = section_info.get("formulas") if section_info else None
        
        return SemanticChunk(
            text=text,
            page_start=page_start,
            page_end=page_end,
            token_count=self._count_tokens(text),
            chunk_index=chunk_idx,
            contains_code=self._detect_code(text),
            book_title=book_title,
            author=author,
            section_title=section_title,
            section_type=section_type,
            section_number=section_number,
            formulas=formulas
        )
    
    def _split_sentences(self, text: str) -> List[str]:
        """Smart sentence splitting that preserves code blocks"""
        # First, protect code blocks
        code_pattern = r'``````|`[^`]+`'
        code_blocks = []
        
        def replace_code(match):
            code_blocks.append(match.group(0))
            return f" __CODE_BLOCK_{len(code_blocks)-1}__ "
        
        text_protected = re.sub(code_pattern, replace_code, text)
        
        # Split into sentences
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, text_protected)
        
        # Also split on double newlines (paragraph breaks)
        final_sentences = []
        for sent in sentences:
            if '\n\n' in sent:
                final_sentences.extend(sent.split('\n\n'))
            else:
                final_sentences.append(sent)
        
        # Restore code blocks
        restored = []
        for sent in final_sentences:
            for i, code in enumerate(code_blocks):
                sent = sent.replace(f"__CODE_BLOCK_{i}__", code)
            
            sent = sent.strip()
            if sent:
                restored.append(sent)
        
        return restored
    
    def _detect_code(self, text: str) -> bool:
        """Detect if text contains code snippets"""
        code_indicators = [
            r'```',
            r'^\s{4,}\w+',
            r'\bdef\s+\w+\s*\(',
            r'\bclass\s+\w+',
            r'\bimport\s+\w+',
            r'\bfrom\s+\w+\s+import\b',
            r'public\s+(class|static|void)',
            r'function\s+\w+\s*\(',
            r'=>',
        ]
        
        for pattern in code_indicators:
            if re.search(pattern, text, flags=re.MULTILINE):
                return True
        
        # Check symbol density
        symbols = sum(text.count(s) for s in '{}[]();=<>+-*/')
        if len(text) > 50 and (symbols / len(text)) > 0.03:
            return True
        
        return False
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))


# Convenience function for quick usage
def create_semantic_chunker(
    similarity_threshold: float = 0.75,
    min_chunk_size: int = 200,
    max_chunk_size: int = 1500,
    embedding_model: Optional[SentenceTransformer] = None
) -> SemanticChunker:
    """
    Factory function to create a semantic chunker with custom settings.
    """
    return SemanticChunker(
        similarity_threshold=similarity_threshold,
        min_chunk_size=min_chunk_size,
        max_chunk_size=max_chunk_size,
        embedding_model=embedding_model
    )