"""
Semantic Chunker - Embedding-based text chunking
Groups semantically similar sentences together
"""
import re
import tiktoken
import numpy as np
from typing import List, Dict, Tuple
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


class SemanticChunker:
    """
    Chunks text based on semantic similarity between sentences.
    Groups related content together, splits at topic boundaries.
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold: float = 0.75,
        min_chunk_size: int = 200,
        max_chunk_size: int = 1500,
        encoding_name: str = "cl100k_base"
    ):
        """
        Args:
            model_name: Sentence transformer model for embeddings
            similarity_threshold: Cosine similarity threshold (0-1). 
                                Lower = more topic splits
            min_chunk_size: Minimum tokens per chunk
            max_chunk_size: Maximum tokens per chunk
        """
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
        batch_size: int = 20  # Process 20 pages at once
    ) -> List[Tuple[str, Dict]]:
        """
        Chunk pages in batches for much better performance.
        Combines multiple pages before embedding.
        """
        all_chunks = []
        global_chunk_idx = 0
        total_pages = len(pages_text)
        
        logger.info(f"Starting batched processing of {total_pages} pages (batch_size={batch_size})...")
        
        # Process in batches
        for batch_start in range(0, total_pages, batch_size):
            batch_end = min(batch_start + batch_size, total_pages)
            batch_pages = pages_text[batch_start:batch_end]
            
            # Progress
            logger.info(f"📊 Processing pages {batch_start+1}-{batch_end} ({batch_end/total_pages*100:.1f}% complete)")
            
            # Combine batch pages into one text block
            combined_text = ""
            page_boundaries = []  # Track where each page starts
            
            for page_data in batch_pages:
                page_boundaries.append(len(combined_text))
                combined_text += page_data["text"] + "\n\n"
            
            if not combined_text.strip():
                continue
            
            # Chunk the combined text
            batch_chunks = self._chunk_text_semantic_fast(
                combined_text,
                batch_start + 1,  # First page in batch
                batch_end         # Last page in batch
            )
            
            # Add to results
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
        
        logger.info(f"✅ Created {len(all_chunks)} semantic chunks from {total_pages} pages")
        return all_chunks

    def _chunk_text_semantic_fast(self, text: str, page_start: int, page_end: int) -> List[SemanticChunk]:
        """
        Faster semantic chunking - processes larger text blocks at once
        """
        sentences = self._split_sentences(text)
        
        if not sentences:
            return []
        
        # Embed in one batch (MUCH faster than one-by-one)
        logger.info(f"   Embedding {len(sentences)} sentences...")
        embeddings = self.model.encode(
            sentences,
            show_progress_bar=False,
            convert_to_numpy=True,
            batch_size=64  # Larger batch size
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
                chunk = SemanticChunk(
                    text=current_text,
                    page_start=page_start,
                    page_end=page_end,
                    token_count=current_tokens,
                    chunk_index=len(chunks),
                    contains_code=self._detect_code(current_text)
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
                chunk = SemanticChunk(
                    text=final_text,
                    page_start=page_start,
                    page_end=page_end,
                    token_count=self._count_tokens(final_text),
                    chunk_index=len(chunks),
                    contains_code=self._detect_code(final_text)
                )
                chunks.append(chunk)
            elif chunks:
                chunks[-1].text += ' ' + final_text
                chunks[-1].token_count = self._count_tokens(chunks[-1].text)
        
        return chunks
    
    def _chunk_text_semantic(self, text: str, page_num: int) -> List[SemanticChunk]:
        """
        Core semantic chunking logic for a single page/section.
        """
        # Split into sentences
        sentences = self._split_sentences(text)
        
        if not sentences:
            return []
        
        # Embed all sentences
        embeddings = self.model.encode(
            sentences, 
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        # Group sentences by similarity
        chunks = []
        current_sentences = [sentences[0]]
        current_embedding = embeddings[0]
        
        for i in range(1, len(sentences)):
            # Calculate similarity with current chunk
            similarity = cosine_similarity(
                [current_embedding], 
                [embeddings[i]]
            )[0][0]
            
            # Get token count of current chunk
            current_text = ' '.join(current_sentences)
            current_tokens = self._count_tokens(current_text)
            
            # Decision: add to current chunk or start new?
            should_split = (
                similarity < self.similarity_threshold or  # Topic changed
                current_tokens >= self.max_chunk_size      # Chunk too large
            )
            
            if should_split and current_tokens >= self.min_chunk_size:
                # Finalize current chunk
                chunk = self._create_chunk(
                    current_sentences,
                    page_num,
                    len(chunks)
                )
                chunks.append(chunk)
                
                # Start new chunk
                current_sentences = [sentences[i]]
                current_embedding = embeddings[i]
            else:
                # Add to current chunk
                current_sentences.append(sentences[i])
                # Update embedding (running average for efficiency)
                current_embedding = (current_embedding + embeddings[i]) / 2
        
        # Add final chunk
        if current_sentences:
            final_text = ' '.join(current_sentences)
            if self._count_tokens(final_text) >= self.min_chunk_size:
                chunk = self._create_chunk(
                    current_sentences,
                    page_num,
                    len(chunks)
                )
                chunks.append(chunk)
            elif chunks:
                # Too small, merge with last chunk
                chunks[-1].text += ' ' + final_text
                chunks[-1].token_count = self._count_tokens(chunks[-1].text)
        
        return chunks
    
    def _create_chunk(
        self,
        sentences: List[str],
        page_num: int,
        chunk_idx: int
    ) -> SemanticChunk:
        """Create a SemanticChunk from sentences"""
        text = ' '.join(sentences)
        
        return SemanticChunk(
            text=text,
            page_start=page_num,
            page_end=page_num,
            token_count=self._count_tokens(text),
            chunk_index=chunk_idx,
            contains_code=self._detect_code(text)
        )
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        Smart sentence splitting that preserves code blocks.
        """
        # First, protect code blocks
        code_pattern = r'```[\s\S]*?```|`[^`]+`'
        code_blocks = []
        
        def replace_code(match):
            code_blocks.append(match.group(0))
            return f" __CODE_BLOCK_{len(code_blocks)-1}__ "
        
        text_protected = re.sub(code_pattern, replace_code, text)
        
        # Split into sentences (improved pattern)
        # Split on: period/question/exclamation followed by space and capital letter
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
            r'```',                          # Markdown code blocks
            r'^\s{4,}\w+',                   # Indented code
            r'\bdef\s+\w+\s*\(',            # Python functions
            r'\bclass\s+\w+',                # Class definitions
            r'\bimport\s+\w+',               # Import statements
            r'\bfrom\s+\w+\s+import\b',     # From imports
            r'public\s+(class|static|void)', # Java/C#
            r'function\s+\w+\s*\(',          # JavaScript functions
            r'=>',                           # Arrow functions
        ]
        
        for pattern in code_indicators:
            if re.search(pattern, text, flags=re.MULTILINE):
                return True
        
        # Check symbol density (high = likely code)
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
    max_chunk_size: int = 1500
) -> SemanticChunker:
    """
    Factory function to create a semantic chunker with custom settings.
    
    Args:
        similarity_threshold: Lower = more splits (more specific chunks)
                            Higher = fewer splits (larger chunks)
                            Recommended: 0.70-0.80
        min_chunk_size: Minimum tokens per chunk
        max_chunk_size: Maximum tokens per chunk
    """
    return SemanticChunker(
        similarity_threshold=similarity_threshold,
        min_chunk_size=min_chunk_size,
        max_chunk_size=max_chunk_size
    )