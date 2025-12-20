"""
Proposition-Based Chunker for GROBID Sections
Flattens hierarchical GROBID structure into proposition-content pairs
"""
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger("proposition_chunker")


class PropositionChunker:
    """
    Converts hierarchical GROBID sections into flat proposition-based chunks
    where headings are the retrieval units and full text is stored in metadata.
    """
    
    def flatten_grobid_sections(
        self, 
        sections: List[Dict], 
        book_title: str, 
        author: str
    ) -> List[Tuple[str, Dict]]:
        """
        Flatten nested GROBID sections into (heading, metadata) tuples.
        
        Args:
            sections: List of chapter dicts from GROBID parser
            book_title: Title of the book
            author: Author name
            
        Returns:
            List of (heading, metadata) tuples where:
            - heading: Section title with full hierarchy (for embedding)
            - metadata: Contains full_text, hierarchy info, etc.
        """
        flat_chunks = []
        chunk_index = 0
        
        for chapter in sections:
            # Add the chapter itself
            chapter_heading = " > ".join(chapter["path"])
            chapter_meta = {
                "full_text": chapter["text"],
                "heading": chapter_heading,
                "hierarchy_path": chapter_heading,
                "hierarchy_level": 1,
                "section_title": chapter["title"],
                "chunk_index": chunk_index,
                "chunk_type": "chapter",
                "book_title": book_title,
                "author": author
            }
            flat_chunks.append((chapter_heading, chapter_meta))
            chunk_index += 1
            
            # Add all subsections
            for subsection in chapter.get("subsections", []):
                subsection_heading = " > ".join(subsection["path"])
                subsection_meta = {
                    "full_text": subsection["text"],
                    "heading": subsection_heading,
                    "hierarchy_path": subsection_heading,
                    "hierarchy_level": 2,
                    "section_title": subsection["title"],
                    "chunk_index": chunk_index,
                    "chunk_type": "section",
                    "book_title": book_title,
                    "author": author
                }
                flat_chunks.append((subsection_heading, subsection_meta))
                chunk_index += 1
        
        logger.info(f"✅ Flattened {len(flat_chunks)} proposition-based chunks from GROBID")
        return flat_chunks


def create_proposition_chunker() -> PropositionChunker:
    """Factory function to create PropositionChunker"""
    return PropositionChunker()
