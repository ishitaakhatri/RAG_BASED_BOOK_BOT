"""
Enhanced Context Compressor with SEMANTIC deduplication
Updated to display book titles with sources
"""

import tiktoken
from typing import List, Dict, Set, Tuple
import re
from difflib import SequenceMatcher
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class EnhancedContextCompressor:
    """
    Compresses context with DUAL-LEVEL deduplication:
    1. Character-level (fast, exact duplicates)
    2. Semantic-level (slow, paraphrased duplicates)
    
    Enhanced to preserve and display book titles
    """
    
    def __init__(
        self,
        target_tokens: int = 20000,
        max_tokens: int = 30000,
        encoding_name: str = "cl100k_base",
        semantic_threshold: float = 0.92
    ):
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.tokenizer = tiktoken.get_encoding(encoding_name)

        # Semantic deduplication
        self.semantic_threshold = semantic_threshold
        self.embedding_model = None  # Lazy load

    
    def _get_embedding_model(self):
        """Lazy load embedding model (only when needed)"""
        if self.embedding_model is None:
            print("  Loading embedding model for semantic deduplication...")
            self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return self.embedding_model
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))
    
    def deduplicate_chunks(
        self,
        chunks: List[Dict],
        use_semantic: bool = True,
        character_threshold: float = 0.85,
        semantic_threshold: float = None
    ) -> Tuple[List[Dict], Dict]:
        """DUAL-LEVEL deduplication"""
        if not chunks:
            return [], {}
        
        semantic_threshold = semantic_threshold or self.semantic_threshold
        
        # PHASE 1: Character-level deduplication (FAST)
        print(f"  Phase 1: Character-level deduplication...")
        unique_chunks_char, char_removed = self._deduplicate_character_level(
            chunks, character_threshold
        )
        
        # PHASE 2: Semantic deduplication (SLOWER)
        if use_semantic and len(unique_chunks_char) > 1:
            print(f"  Phase 2: Semantic deduplication (threshold={semantic_threshold})...")
            unique_chunks_final, sem_removed = self._deduplicate_semantic_level(
                unique_chunks_char, semantic_threshold
            )
        else:
            unique_chunks_final = unique_chunks_char
            sem_removed = 0
        
        stats = {
            'original_count': len(chunks),
            'character_removed': char_removed,
            'semantic_removed': sem_removed,
            'final_count': len(unique_chunks_final),
            'total_removed': len(chunks) - len(unique_chunks_final)
        }
        
        print(f"  Deduplication complete: {len(chunks)} → {len(unique_chunks_final)} chunks "
              f"(char: -{char_removed}, semantic: -{sem_removed})")
        
        return unique_chunks_final, stats
    
    def _deduplicate_character_level(
        self,
        chunks: List[Dict],
        threshold: float
    ) -> Tuple[List[Dict], int]:
        """Character-level deduplication using SequenceMatcher"""
        unique_chunks = []
        seen_texts = []
        removed = 0
        
        for chunk in chunks:
            text = chunk.get('text', '')
            
            if not text:
                continue
            
            # Check character similarity with existing chunks
            is_duplicate = False
            for seen_text in seen_texts:
                similarity = SequenceMatcher(None, text, seen_text).ratio()
                if similarity > threshold:
                    is_duplicate = True
                    removed += 1
                    break
            
            if not is_duplicate:
                unique_chunks.append(chunk)
                seen_texts.append(text)
        
        return unique_chunks, removed
    
    def _deduplicate_semantic_level(
        self,
        chunks: List[Dict],
        threshold: float
    ) -> Tuple[List[Dict], int]:
        """Semantic deduplication using embeddings"""
        if len(chunks) <= 1:
            return chunks, 0
        
        # Get embedding model
        model = self._get_embedding_model()
        
        # Extract texts
        texts = [chunk.get('text', '') for chunk in chunks]
        
        # Generate embeddings (batch for efficiency)
        embeddings = model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            batch_size=32
        )
        
        # Find duplicates using cosine similarity
        unique_chunks = []
        unique_embeddings = []
        removed = 0
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            is_duplicate = False
            
            # Compare with already-selected chunks
            if unique_embeddings:
                similarities = cosine_similarity(
                    [embedding],
                    unique_embeddings
                )[0]
                
                # If ANY similarity > threshold, it's a duplicate
                if np.max(similarities) > threshold:
                    is_duplicate = True
                    removed += 1
            
            if not is_duplicate:
                unique_chunks.append(chunk)
                unique_embeddings.append(embedding)
        
        return unique_chunks, removed
    
    def extract_key_sentences(
        self,
        text: str,
        query: str,
        max_sentences: int = 5
    ) -> str:
        """Extract most relevant sentences from text"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if len(sentences) <= max_sentences:
            return text
        
        # Score sentences by keyword overlap
        query_words = set(query.lower().split())
        scored_sentences = []
        
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            overlap = len(query_words & sentence_words)
            scored_sentences.append((sentence, overlap))
        
        # Sort by score
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        # Take top sentences
        top_sentences = [s for s, _ in scored_sentences[:max_sentences]]
        
        return '. '.join(top_sentences) + '.'
    
    def compress_context(
        self,
        chunks: List[Dict],
        query: str,
        preserve_code: bool = True,
        use_semantic_dedup: bool = True
    ) -> str:
        """
        Main compression pipeline with semantic deduplication.
        Enhanced to display book titles with sources.
        
        Args:
            chunks: List of chunk dicts with 'text' and 'metadata' fields
            query: User query
            preserve_code: Keep code blocks intact
            use_semantic_dedup: Whether to use semantic deduplication
        
        Returns:
            Compressed context string with book titles
        """
        if not chunks:
            return ""
        
        print(f"\n[Context Compression Pipeline]")
        print(f"  Input: {len(chunks)} chunks")
        
        # Step 1: Deduplicate (character + semantic)
        unique_chunks, dedup_stats = self.deduplicate_chunks(
            chunks,
            use_semantic=use_semantic_dedup
        )
        
        # Step 2: Separate code and text chunks
        code_chunks = []
        text_chunks = []
        
        for chunk in unique_chunks:
            metadata = chunk.get('metadata', {})
            chunk_type = chunk.get('chunk_type', 'text')
            
            if metadata.get('contains_code') or chunk_type == 'code':
                code_chunks.append(chunk)
            else:
                text_chunks.append(chunk)
        
        print(f"  Split: {len(code_chunks)} code, {len(text_chunks)} text chunks")
        
        # Step 3: Build compressed context
        context_parts = []
        current_tokens = 0
        
        # Always include code chunks (high value)
        if preserve_code:
            for chunk in code_chunks[:3]:  # Top 3 code chunks
                text = chunk.get('text', '')
                tokens = self.count_tokens(text)
                
                if current_tokens + tokens <= self.max_tokens:
                    context_parts.append({
                        'text': text,
                        'metadata': chunk.get('metadata', {}),
                        'type': 'code'
                    })
                    current_tokens += tokens
        
        # Add text chunks (summarize if needed)
        for chunk in text_chunks:
            text = chunk.get('text', '')
            tokens = self.count_tokens(text)
            
            # If chunk is too large, summarize it
            if tokens > 3000:
                text = self.extract_key_sentences(text, query, max_sentences=4)
                tokens = self.count_tokens(text)
            
            if current_tokens + tokens <= self.target_tokens:
                context_parts.append({
                    'text': text,
                    'metadata': chunk.get('metadata', {}),
                    'type': 'text'
                })
                current_tokens += tokens
            elif current_tokens >= self.target_tokens:
                break
        
        # Step 4: Format final context WITH BOOK TITLES
        formatted_context = self._format_context_with_books(context_parts)
        
        final_tokens = self.count_tokens(formatted_context)
        print(f"  Output: {final_tokens} tokens (target: {self.target_tokens})")
        print(f"  Deduplication saved: {dedup_stats['total_removed']} chunks\n")
        
        return formatted_context
    
    def _format_context_with_books(self, context_parts: List[Dict]) -> str:
        """
        Format compressed context for LLM with BOOK TITLES prominently displayed
        """
        formatted = []
        
        for i, part in enumerate(context_parts, 1):
            metadata = part.get('metadata', {})
            text = part['text']
            part_type = part['type']
            
            # Extract book information
            book_title = metadata.get('book_title', 'Unknown Book')
            author = metadata.get('author', 'Unknown Author')
            chapter = metadata.get('chapter_title', 'Unknown Chapter')
            page = metadata.get('page_start', '?')
            
            # Build header with BOOK TITLE prominently
            header = f"[SOURCE {i}] ({part_type.upper()})\n"
            header += f"📚 Book: '{book_title}' by {author}\n"
            header += f"📖 Chapter: {chapter} | Page: {page}"
            
            formatted.append(f"{header}\n{'-' * 70}\n{text}\n")
        
        return "\n" + "="*70 + "\nCONTEXT FROM BOOKS:\n" + "="*70 + "\n\n" + "\n".join(formatted)


# ============================================================================
# COMPARISON: OLD vs NEW
# ============================================================================

if __name__ == "__main__":
    print("=== TESTING SEMANTIC DEDUPLICATION WITH BOOK TITLES ===\n")
    
    # Test chunks with book metadata
    test_chunks = [
        {
            'text': "Gradient descent is an optimization algorithm used to minimize loss functions in machine learning.",
            'metadata': {
                'book_title': 'Hands-On Machine Learning',
                'author': 'Aurélien Géron',
                'chapter_title': 'Training Models',
                'page_start': 45
            },
            'score': 0.9
        },
        {
            'text': "Neural networks use backpropagation to compute gradients for training.",
            'metadata': {
                'book_title': 'Deep Learning with Python',
                'author': 'François Chollet',
                'chapter_title': 'Neural Networks',
                'page_start': 75
            },
            'score': 0.8
        }
    ]
    
    compressor = EnhancedContextCompressor(
        target_tokens=1000,
        semantic_threshold=0.90
    )
    
    # Test compression with book titles
    print("COMPRESSING WITH BOOK TITLES:")
    compressed = compressor.compress_context(
        test_chunks,
        "How does gradient descent work?",
        use_semantic_dedup=True
    )
    
    print("\nFORMATTED OUTPUT:")
    print(compressed)
    
    print("\n✅ Book titles are now displayed with sources!")