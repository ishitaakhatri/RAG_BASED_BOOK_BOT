"""
Hierarchical Chunker for Marker Markdown Output
Located at: backend/rag_based_book_bot/document_ingestion/hierarchical_chunker.py

Chunks markdown by respecting document hierarchy (chapters, sections, subsections)
"""
import re
import tiktoken
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class HierarchyNode:
    """Represents a node in document hierarchy"""
    level: int  # 1=chapter, 2=section, 3=subsection, 4=subsubsection
    title: str
    content: str = ""
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    parent_id: Optional[str] = None
    node_id: str = ""
    children: List[str] = field(default_factory=list)  # List of child node IDs
    
    def __post_init__(self):
        if not self.node_id:
            # Generate unique ID from title and level
            import hashlib
            raw = f"{self.level}_{self.title}_{self.page_start}"
            self.node_id = hashlib.md5(raw.encode()).hexdigest()[:12]


@dataclass
class HierarchicalChunk:
    """A chunk with full hierarchical metadata"""
    chunk_id: str
    content: str
    level: int
    title: str
    
    # Full hierarchy path
    chapter: Optional[str] = None
    section: Optional[str] = None
    subsection: Optional[str] = None
    subsubsection: Optional[str] = None
    
    # Hierarchy IDs for retrieval
    chapter_id: Optional[str] = None
    section_id: Optional[str] = None
    subsection_id: Optional[str] = None
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    
    # Other metadata
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    token_count: int = 0
    contains_code: bool = False
    
    # Book metadata
    book_title: str = "Unknown"
    author: str = "Unknown"


class HierarchicalMarkdownChunker:
    """
    Chunks Marker's markdown output hierarchically
    
    Respects document structure:
    - # Chapter
    - ## Section  
    - ### Subsection
    - #### Subsubsection
    """
    
    def __init__(
        self,
        max_chunk_tokens: int = 1500,
        min_chunk_tokens: int = 200,
        overlap_tokens: int = 100,
        encoding_name: str = "cl100k_base"
    ):
        self.max_chunk_tokens = max_chunk_tokens
        self.min_chunk_tokens = min_chunk_tokens
        self.overlap_tokens = overlap_tokens
        self.tokenizer = tiktoken.get_encoding(encoding_name)
        
        # Heading patterns (Marker uses # syntax)
        self.heading_pattern = re.compile(r'^(#{1,4})\s+(.+)$', re.MULTILINE)
        self.page_pattern = re.compile(r'\{==\s*Page\s+(\d+)\s*==\}', re.IGNORECASE)
    
    def chunk_markdown(
        self,
        markdown_text: str,
        book_title: str,
        author: str
    ) -> List[Tuple[str, Dict]]:
        """
        Main entry point: chunk markdown into hierarchical chunks
        
        Returns:
            List of (chunk_text, metadata_dict) tuples
        """
        print(f"📚 Chunking '{book_title}' by {author}")
        
        # Step 1: Build hierarchy tree
        hierarchy = self._build_hierarchy_tree(markdown_text)
        print(f"   Built hierarchy: {len(hierarchy)} nodes")
        
        # Step 2: Chunk each node
        chunks = []
        for node_id, node in hierarchy.items():
            node_chunks = self._chunk_node(node, hierarchy, book_title, author)
            chunks.extend(node_chunks)
        
        print(f"   Created {len(chunks)} hierarchical chunks")
        
        # Convert to (text, metadata) format
        result = []
        for chunk in chunks:
            metadata = {
                "chunk_id": chunk.chunk_id,
                "book_title": chunk.book_title,
                "author": chunk.author,
                "level": chunk.level,
                "title": chunk.title,
                "chapter": chunk.chapter,
                "section": chunk.section,
                "subsection": chunk.subsection,
                "subsubsection": chunk.subsubsection,
                "chapter_id": chunk.chapter_id,
                "section_id": chunk.section_id,
                "subsection_id": chunk.subsection_id,
                "parent_id": chunk.parent_id,
                "children_ids": chunk.children_ids,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "token_count": chunk.token_count,
                "contains_code": chunk.contains_code,
            }
            result.append((chunk.content, metadata))
        
        return result
    
    def _build_hierarchy_tree(self, markdown: str) -> Dict[str, HierarchyNode]:
        """Build hierarchy tree from markdown headings"""
        nodes = {}
        current_page = 1
        
        # Find all headings with their positions
        lines = markdown.split('\n')
        current_node_stack = []  # Stack to track current hierarchy
        current_content = []
        
        for line in lines:
            # Check for page markers
            page_match = self.page_pattern.search(line)
            if page_match:
                current_page = int(page_match.group(1))
                continue
            
            # Check for headings
            heading_match = self.heading_pattern.match(line)
            
            if heading_match:
                # Save previous node's content
                if current_node_stack:
                    current_node = current_node_stack[-1]
                    current_node.content = '\n'.join(current_content).strip()
                    current_content = []
                
                # Create new node
                level = len(heading_match.group(1))  # Count #'s
                title = heading_match.group(2).strip()
                
                node = HierarchyNode(
                    level=level,
                    title=title,
                    page_start=current_page,
                    page_end=current_page
                )
                
                # Pop stack to find parent
                while current_node_stack and current_node_stack[-1].level >= level:
                    current_node_stack.pop()
                
                # Set parent relationship
                if current_node_stack:
                    parent = current_node_stack[-1]
                    node.parent_id = parent.node_id
                    parent.children.append(node.node_id)
                
                nodes[node.node_id] = node
                current_node_stack.append(node)
            else:
                # Accumulate content
                current_content.append(line)
        
        # Save last node's content
        if current_node_stack:
            current_node = current_node_stack[-1]
            current_node.content = '\n'.join(current_content).strip()
        
        return nodes
    
    def _chunk_node(
        self,
        node: HierarchyNode,
        hierarchy: Dict[str, HierarchyNode],
        book_title: str,
        author: str
    ) -> List[HierarchicalChunk]:
        """Chunk a single hierarchy node"""
        if not node.content.strip():
            return []
        
        # Get hierarchy path
        chapter, section, subsection, subsubsection = self._get_hierarchy_path(node, hierarchy)
        chapter_id, section_id, subsection_id = self._get_hierarchy_ids(node, hierarchy)
        
        # Token count
        tokens = self.tokenizer.encode(node.content)
        
        chunks = []
        
        # If content fits in one chunk
        if len(tokens) <= self.max_chunk_tokens:
            chunk = HierarchicalChunk(
                chunk_id=f"{node.node_id}_0",
                content=node.content,
                level=node.level,
                title=node.title,
                chapter=chapter,
                section=section,
                subsection=subsection,
                subsubsection=subsubsection,
                chapter_id=chapter_id,
                section_id=section_id,
                subsection_id=subsection_id,
                parent_id=node.parent_id,
                children_ids=node.children,
                page_start=node.page_start,
                page_end=node.page_end,
                token_count=len(tokens),
                contains_code=self._detect_code(node.content),
                book_title=book_title,
                author=author
            )
            chunks.append(chunk)
        else:
            # Split into multiple chunks with overlap
            chunk_idx = 0
            for i in range(0, len(tokens), self.max_chunk_tokens - self.overlap_tokens):
                chunk_tokens = tokens[i:i + self.max_chunk_tokens]
                chunk_text = self.tokenizer.decode(chunk_tokens)
                
                chunk = HierarchicalChunk(
                    chunk_id=f"{node.node_id}_{chunk_idx}",
                    content=chunk_text,
                    level=node.level,
                    title=f"{node.title} (part {chunk_idx + 1})",
                    chapter=chapter,
                    section=section,
                    subsection=subsection,
                    subsubsection=subsubsection,
                    chapter_id=chapter_id,
                    section_id=section_id,
                    subsection_id=subsection_id,
                    parent_id=node.parent_id,
                    children_ids=node.children,
                    page_start=node.page_start,
                    page_end=node.page_end,
                    token_count=len(chunk_tokens),
                    contains_code=self._detect_code(chunk_text),
                    book_title=book_title,
                    author=author
                )
                chunks.append(chunk)
                chunk_idx += 1
        
        return chunks
    
    def _get_hierarchy_path(
        self,
        node: HierarchyNode,
        hierarchy: Dict[str, HierarchyNode]
    ) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        """Get full hierarchy path (chapter, section, subsection, subsubsection)"""
        path = [None, None, None, None]  # Max 4 levels
        
        current = node
        while current:
            if current.level <= 4:
                path[current.level - 1] = current.title
            
            if current.parent_id:
                current = hierarchy.get(current.parent_id)
            else:
                break
        
        return tuple(path)
    
    def _get_hierarchy_ids(
        self,
        node: HierarchyNode,
        hierarchy: Dict[str, HierarchyNode]
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Get chapter_id, section_id, subsection_id"""
        chapter_id = None
        section_id = None
        subsection_id = None
        
        current = node
        while current:
            if current.level == 1:
                chapter_id = current.node_id
            elif current.level == 2:
                section_id = current.node_id
            elif current.level == 3:
                subsection_id = current.node_id
            
            if current.parent_id:
                current = hierarchy.get(current.parent_id)
            else:
                break
        
        return chapter_id, section_id, subsection_id
    
    def _detect_code(self, text: str) -> bool:
        """Detect if text contains code"""
        code_patterns = [
            r'```',  # Code blocks
            r'^\s{4,}\w+',  # Indented code
            r'\bdef\s+\w+\(',
            r'\bclass\s+\w+',
            r'\bimport\s+\w+',
            r'\bfunction\s+\w+\(',
        ]
        
        for pattern in code_patterns:
            if re.search(pattern, text, re.MULTILINE):
                return True
        
        return False