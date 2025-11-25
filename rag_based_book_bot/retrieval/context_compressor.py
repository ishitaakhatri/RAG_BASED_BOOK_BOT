"""
Context Compressor (PASS 5)

Summarizes, deduplicates, and compresses context before LLM.
Keeps token usage under control while preserving critical information.
"""

import tiktoken
from typing import List, Dict, Set
import re
from difflib import SequenceMatcher
from openai import OpenAI
import os


class ContextCompressor:
    """
    Compresses context to fit token budget while preserving quality.
    """
    
    def __init__(
        self,
        target_tokens: int = 2000,
        max_tokens: int = 4000,
        encoding_name: str = "cl100k_base"
    ):
        """
        Initialize compressor.
        
        Args:
            target_tokens: Target context size (ideal)
            max_tokens: Maximum context size (hard limit)
            encoding_name: Tokenizer to use
        """
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.tokenizer = tiktoken.get_encoding(encoding_name)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))
    
    def deduplicate_chunks(
        self,
        chunks: List[Dict],
        similarity_threshold: float = 0.85
    ) -> List[Dict]:
        """
        Remove highly similar chunks.
        
        Args:
            chunks: List of chunk dicts with 'text' field
            similarity_threshold: Similarity ratio above which chunks are considered duplicates
        
        Returns:
            Deduplicated list
        """
        if not chunks:
            return []
        
        unique_chunks = []
        seen_texts = []
        
        for chunk in chunks:
            text = chunk.get('text', '')
            
            if not text:
                continue
            
            # Check similarity with existing chunks
            is_duplicate = False
            for seen_text in seen_texts:
                similarity = self._text_similarity(text, seen_text)
                if similarity > similarity_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_chunks.append(chunk)
                seen_texts.append(text)
        
        return unique_chunks
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Compute text similarity ratio"""
        return SequenceMatcher(None, text1, text2).ratio()
    
    def extract_key_sentences(
        self,
        text: str,
        query: str,
        max_sentences: int = 5
    ) -> str:
        """
        Extract most relevant sentences from text.
        
        Args:
            text: Full text
            query: User query for relevance scoring
            max_sentences: Maximum sentences to keep
        
        Returns:
            Compressed text with key sentences
        """
        # Split into sentences
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
    
    def summarize_chunk(
        self,
        text: str,
        max_length: int = 200
    ) -> str:
        """
        Summarize a single chunk using LLM.
        
        Args:
            text: Chunk text
            max_length: Maximum summary length in tokens
        
        Returns:
            Summarized text
        """
        if self.count_tokens(text) <= max_length:
            return text
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Summarize the following text, preserving key technical details, code examples, and important facts."},
                    {"role": "user", "content": text[:3000]}  # Limit input
                ],
                temperature=0.3,
                max_tokens=max_length
            )
            
            summary = response.choices[0].message.content.strip()
            return summary
            
        except Exception as e:
            print(f"Summarization failed: {e}")
            # Fallback: truncate
            return self.extract_key_sentences(text, "", max_sentences=3)
    
    def compress_context(
        self,
        chunks: List[Dict],
        query: str,
        preserve_code: bool = True
    ) -> str:
        """
        Main compression pipeline.
        
        Args:
            chunks: List of chunk dicts with 'text', 'metadata', 'score'
            query: User query
            preserve_code: Whether to keep code blocks intact
        
        Returns:
            Compressed context string ready for LLM
        """
        if not chunks:
            return ""
        
        # Step 1: Deduplicate
        unique_chunks = self.deduplicate_chunks(chunks)
        print(f"  Deduplication: {len(chunks)} → {len(unique_chunks)} chunks")
        
        # Step 2: Separate code and text chunks
        code_chunks = []
        text_chunks = []
        
        for chunk in unique_chunks:
            if chunk.get('metadata', {}).get('contains_code') or chunk.get('chunk_type') == 'code':
                code_chunks.append(chunk)
            else:
                text_chunks.append(chunk)
        
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
            if tokens > 300:
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
        
        # Step 4: Format final context
        formatted_context = self._format_context(context_parts)
        
        final_tokens = self.count_tokens(formatted_context)
        print(f"  Compression: {final_tokens} tokens (target: {self.target_tokens})")
        
        return formatted_context
    
    def _format_context(self, context_parts: List[Dict]) -> str:
        """Format compressed context for LLM"""
        formatted = []
        
        for i, part in enumerate(context_parts, 1):
            metadata = part.get('metadata', {})
            text = part['text']
            part_type = part['type']
            
            # Build header
            chapter = metadata.get('chapter_title', 'Unknown')
            page = metadata.get('page_start', '?')
            
            header = f"[SOURCE {i}] ({part_type.upper()}) - Chapter: {chapter}, Page: {page}"
            
            formatted.append(f"{header}\n{text}\n")
        
        return "\n" + "="*70 + "\n".join(formatted)
    
    def get_compression_stats(self, chunks: List[Dict]) -> Dict:
        """Get statistics about compression"""
        total_tokens = sum(self.count_tokens(c.get('text', '')) for c in chunks)
        
        return {
            'n_chunks': len(chunks),
            'total_tokens': total_tokens,
            'avg_tokens_per_chunk': total_tokens / len(chunks) if chunks else 0,
            'target_tokens': self.target_tokens,
            'compression_ratio': self.target_tokens / total_tokens if total_tokens > 0 else 0
        }


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    compressor = ContextCompressor(target_tokens=1000)
    
    # Simulate chunks
    chunks = [
        {
            'text': "Gradient descent is an optimization algorithm used to minimize loss functions. " * 20,
            'metadata': {'chapter_title': 'Chapter 5', 'page_start': 45, 'contains_code': False},
            'score': 0.9
        },
        {
            'text': "def gradient_descent(X, y, lr=0.01):\n    theta = np.zeros(X.shape[1])\n    for i in range(1000):\n        gradient = X.T.dot(X.dot(theta) - y)\n        theta -= lr * gradient\n    return theta",
            'metadata': {'chapter_title': 'Chapter 5', 'page_start': 47, 'contains_code': True},
            'score': 0.95
        },
        {
            'text': "Gradient descent is an optimization algorithm used to minimize loss functions. " * 20,  # Duplicate
            'metadata': {'chapter_title': 'Chapter 6', 'page_start': 60, 'contains_code': False},
            'score': 0.85
        }
    ]
    
    query = "How to implement gradient descent?"
    
    # Compress
    compressed = compressor.compress_context(chunks, query)
    
    print("\n=== COMPRESSED CONTEXT ===")
    print(compressed)
    
    # Stats
    stats = compressor.get_compression_stats(chunks)
    print("\n=== STATS ===")
    for k, v in stats.items():
        print(f"{k}: {v}")