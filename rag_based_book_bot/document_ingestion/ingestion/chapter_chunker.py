"""
Chapter-Aware Chunker

Implements sliding window chunking that respects chapter boundaries.
Chunks never span across different chapters.
"""
import tiktoken
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ChunkMetadata:
    """Metadata for a single chunk"""
    book_title: str
    author: str
    chapter_title: Optional[str]
    chapter_number: Optional[str]
    section_titles: List[str]
    section_numbers: List[str]
    subsection_titles: List[str]
    subsection_numbers: List[str]
    page_start: Optional[int]
    page_end: Optional[int]
    chunk_index: int
    contains_code: bool


class ChapterChunker:
    """Chunks text with chapter boundary awareness"""
    
    def __init__(
        self,
        chunk_size: int = 1500,
        overlap: int = 200,
        encoding_name: str = "cl100k_base"
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokenizer = tiktoken.get_encoding(encoding_name)
    
    def chunk_chapters(
        self,
        chapters: List[Dict],
        elements: List[Dict],
        book_title: str,
        author: str = "Unknown"
    ) -> List[Tuple[str, ChunkMetadata]]:
        """
        Chunk all chapters while respecting chapter boundaries.
        
        Args:
            chapters: List of BookNode objects at chapter level
            elements: Original parsed elements
            book_title: Title of the book
            author: Author name
        
        Returns:
            List of (chunk_text, metadata) tuples
        """
        all_chunks = []
        global_chunk_idx = 0
        
        for chapter in chapters:
            # Extract text for this chapter
            chapter_text, chapter_elements = self._extract_chapter_text(
                chapter,
                elements
            )
            
            if not chapter_text.strip():
                continue
            
            # Tokenize chapter
            tokens = self.tokenizer.encode(chapter_text)
            
            # Chunk within this chapter
            chapter_chunks = self._chunk_chapter_tokens(
                tokens,
                chapter_text,
                chapter,
                chapter_elements,
                book_title,
                author,
                global_chunk_idx
            )
            
            all_chunks.extend(chapter_chunks)
            global_chunk_idx += len(chapter_chunks)
        
        return all_chunks
    
    def _extract_chapter_text(
        self,
        chapter: Dict,
        elements: List[Dict]
    ) -> Tuple[str, List[Dict]]:
        """Extract text and elements for a single chapter"""
        start_idx = chapter.get("start_element_idx") or 0
        end_idx = chapter.get("end_element_idx")
        
        # Handle None end_idx
        if end_idx is None:
            end_idx = len(elements) - 1
        
        chapter_elements = []
        text_parts = []
        
        # Ensure valid range
        start_idx = max(0, start_idx)
        end_idx = min(end_idx, len(elements) - 1)
        
        for i in range(start_idx, end_idx + 1):
            if i >= len(elements):
                break
            
            el = elements[i]
            text = el.get("text", "") or ""
            
            if text.strip():
                text_parts.append(text)
                chapter_elements.append(el)
        
        chapter_text = "\n\n".join(text_parts)
        return chapter_text, chapter_elements
    
    def _chunk_chapter_tokens(
        self,
        tokens: List[int],
        chapter_text: str,
        chapter: Dict,
        chapter_elements: List[Dict],
        book_title: str,
        author: str,
        start_chunk_idx: int
    ) -> List[Tuple[str, ChunkMetadata]]:
        """Chunk a single chapter's tokens with sliding window"""
        chunks = []
        idx = 0
        step = max(1, self.chunk_size - self.overlap)
        local_chunk_idx = 0
        
        while idx < len(tokens):
            # Extract chunk window (capped at chapter end)
            window_end = min(idx + self.chunk_size, len(tokens))
            chunk_tokens = tokens[idx:window_end]
            
            if not chunk_tokens:
                break
            
            # Decode tokens to text
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            # Get sections/subsections covered by this chunk
            sections_info = self._get_sections_in_chunk(
                chunk_text,
                chapter,
                chapter_elements,
                idx,
                window_end,
                len(tokens)
            )
            
            # Detect if chunk contains code
            contains_code = self._detect_code(chunk_text)
            
            # Build metadata
            metadata = ChunkMetadata(
                book_title=book_title,
                author=author,
                chapter_title=chapter.get("title"),
                chapter_number=chapter.get("number"),
                section_titles=sections_info["section_titles"],
                section_numbers=sections_info["section_numbers"],
                subsection_titles=sections_info["subsection_titles"],
                subsection_numbers=sections_info["subsection_numbers"],
                page_start=chapter.get("start_page"),
                page_end=chapter.get("end_page"),
                chunk_index=start_chunk_idx + local_chunk_idx,
                contains_code=contains_code
            )
            
            chunks.append((chunk_text, metadata))
            local_chunk_idx += 1
            
            # Move to next window
            idx += step
            
            # If we're very close to the end, just include it in this chunk
            if idx < len(tokens) and idx + self.chunk_size > len(tokens):
                if len(tokens) - idx < self.overlap:
                    break  # Too small to make another chunk
        
        return chunks
    
    def _get_sections_in_chunk(
        self,
        chunk_text: str,
        chapter: Dict,
        chapter_elements: List[Dict],
        token_start: int,
        token_end: int,
        total_tokens: int
    ) -> Dict[str, List]:
        """Identify sections and subsections within this chunk"""
        # For now, simple approach: check if chapter has children
        section_titles = []
        section_numbers = []
        subsection_titles = []
        subsection_numbers = []
        
        children = chapter.get("children", [])
        
        for child in children:
            # Check if this section overlaps with chunk
            # (simplified - in production you'd want token-level tracking)
            if child.get("level") == 2:  # Section
                section_titles.append(child.get("title", ""))
                section_numbers.append(child.get("number") or "")
                
                # Check subsections
                for subchild in child.get("children", []):
                    if subchild.get("level") == 3:
                        subsection_titles.append(subchild.get("title", ""))
                        subsection_numbers.append(subchild.get("number") or "")
        
        return {
            "section_titles": section_titles,
            "section_numbers": section_numbers,
            "subsection_titles": subsection_titles,
            "subsection_numbers": subsection_numbers
        }
    
    def _detect_code(self, text: str) -> bool:
        """Detect if text contains code snippets"""
        import re
        
        patterns = [
            r"^\s{4,}\w+",  # Indented code
            r"^\s*def\s+\w+\(",  # Python function
            r"^\s*class\s+\w+",  # Class definition
            r"```",  # Markdown code block
            r"\bimport\s+\w+",  # Import statement
            r"\bfrom\s+\w+\s+import\b",  # From import
            r"public\s+(class|static|void)",  # Java/C#
            r"function\s+\w+\s*\(",  # JavaScript function
        ]
        
        for pattern in patterns:
            if re.search(pattern, text, flags=re.MULTILINE):
                return True
        
        # Check symbol density
        symbols = sum(text.count(s) for s in "{}[]();=<>:+-*/\\")
        if len(text) > 50 and (symbols / len(text)) > 0.03:
            return True
        
        return False
    
    def estimate_chunks(self, total_tokens: int) -> int:
        """Estimate number of chunks for given token count"""
        step = max(1, self.chunk_size - self.overlap)
        return max(1, (total_tokens + step - 1) // step)
