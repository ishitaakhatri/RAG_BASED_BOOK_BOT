"""
Hierarchical Chunker
Flattens a document tree into vector-ready chunks while preserving context/breadcrumbs.
"""
import logging
from typing import List, Dict, Tuple
import tiktoken

logger = logging.getLogger("hierarchical_chunker")

class HierarchicalChunker:
    """
    Chunks text based on GROBID structure tree.
    Enforces that every chunk carries its hierarchical context.
    """
    
    def __init__(self, max_chunk_tokens: int = 1000, overlap: int = 100, encoding_name: str = "cl100k_base"):
        self.tokenizer = tiktoken.get_encoding(encoding_name)
        self.max_tokens = max_chunk_tokens
        self.overlap = overlap

    def process_document_tree(self, sections_tree: List[Dict], book_title: str, author: str) -> List[Tuple[str, Dict]]:
        """
        Walks the GROBID tree and produces flat, context-aware chunks.
        """
        all_chunks = []
        
        for section_node in sections_tree:
            all_chunks.extend(self._process_node(section_node, book_title, author))
            
        return all_chunks

    def _process_node(self, node: Dict, book_title: str, author: str) -> List[Tuple[str, Dict]]:
        """
        Recursively process a node and its children.
        """
        chunks = []
        
        # 1. Construct the Context Header
        hierarchy_path_str = " > ".join(node["path"])
        context_header = f"Context: {book_title} > {hierarchy_path_str}\n"
        
        context_tokens = len(self.tokenizer.encode(context_header))
        
        # 2. Chunk the text of THIS node
        if node["text"] and node["text"].strip():
            available_tokens = self.max_tokens - context_tokens
            if available_tokens < 200: available_tokens = 200
            
            text_segments = self._sliding_window_split(node["text"], available_tokens)
            
            for i, segment in enumerate(text_segments):
                final_text = f"{context_header}\n{segment}"
                
                metadata = {
                    "book_title": book_title,
                    "author": author,
                    "section_title": node["title"],
                    "hierarchy_path": hierarchy_path_str,
                    "hierarchy_level": len(node["path"]),
                    "chunk_type": "text_block",
                    "chunk_index": i
                }
                chunks.append((final_text, metadata))

        # 3. Recursively process Children
        for subsection in node["subsections"]:
            chunks.extend(self._process_node(subsection, book_title, author))
            
        return chunks

    def _sliding_window_split(self, text: str, max_tokens: int) -> List[str]:
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return [text]
            
        chunks = []
        step = max(1, max_tokens - self.overlap)
        
        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i : i + max_tokens]
            chunks.append(self.tokenizer.decode(chunk_tokens))
            
        return chunks