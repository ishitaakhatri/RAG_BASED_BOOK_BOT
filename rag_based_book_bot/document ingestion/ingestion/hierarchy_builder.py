"""
Hierarchy Builder

Builds and validates hierarchical tree structure from TOC and detected headings.
Merges multiple detection strategies and resolves conflicts.
"""
from typing import List, Optional, Dict
from dataclasses import dataclass, field


@dataclass
class BookNode:
    """Represents a node in the book's hierarchical structure"""
    title: str
    level: int  # 0=book, 1=chapter, 2=section, 3=subsection
    number: Optional[str] = None
    start_page: Optional[int] = None
    end_page: Optional[int] = None
    start_element_idx: Optional[int] = None
    end_element_idx: Optional[int] = None
    parent: Optional['BookNode'] = None
    children: List['BookNode'] = field(default_factory=list)
    confidence: float = 1.0  # Detection confidence (0-1)
    source: str = "unknown"  # toc, font, pattern, style
    
    def add_child(self, child: 'BookNode'):
        """Add a child node"""
        child.parent = self
        self.children.append(child)
    
    def get_hierarchy_path(self) -> List[str]:
        """Get full hierarchical path from root"""
        path = []
        node = self
        while node.parent:
            path.insert(0, node.title)
            node = node.parent
        return path
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation"""
        return {
            "title": self.title,
            "number": self.number,
            "level": self.level,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "start_element_idx": self.start_element_idx,
            "end_element_idx": self.end_element_idx,
            "confidence": self.confidence,
            "source": self.source,
            "children": [child.to_dict() for child in self.children]
        }


class HierarchyBuilder:
    """Builds book hierarchy from multiple sources"""
    
    def __init__(self):
        self.root = None
    
    def build_from_toc(self, toc_entries: List[Dict]) -> BookNode:
        """
        Build hierarchy tree from TOC entries.
        
        Args:
            toc_entries: List of TOC entries from TOCParser
        
        Returns:
            Root BookNode
        """
        root = BookNode(title="Root", level=0, source="root")
        
        parent_stack = [root]  # Stack to track parent at each level
        
        for entry in toc_entries:
            node = BookNode(
                title=entry["title"],
                level=entry["level"],
                number=entry.get("number"),
                start_page=entry.get("page"),
                confidence=0.9,  # High confidence for TOC
                source="toc"
            )
            
            # Find appropriate parent
            while len(parent_stack) > 1 and parent_stack[-1].level >= node.level:
                parent_stack.pop()
            
            parent = parent_stack[-1]
            parent.add_child(node)
            parent_stack.append(node)
        
        self.root = root
        return root
    
    def merge_detected_headings(
        self,
        root: BookNode,
        detected_headings: List[Dict],
        elements: List[Dict]
    ) -> BookNode:
        """
        Merge detected headings with TOC-based structure.
        Fills gaps and adds missing sections.
        
        Args:
            root: Existing root node from TOC
            detected_headings: List of headings detected from font/pattern analysis
            elements: Original parsed elements
        
        Returns:
            Updated root node
        """
        # Build a map of existing nodes by page
        existing_by_page = {}
        self._map_nodes_by_page(root, existing_by_page)
        
        # Process detected headings
        for heading in detected_headings:
            page = heading.get("page")
            title = heading.get("text", "").strip()
            level = heading.get("level", 2)
            number = heading.get("num")
            element_idx = heading.get("index")
            confidence = heading.get("confidence", 0.6)
            
            # Check if already exists in TOC
            if page and page in existing_by_page:
                # Update element index if not set
                existing_node = existing_by_page[page]
                if existing_node.start_element_idx is None:
                    existing_node.start_element_idx = element_idx
                continue
            
            # New heading not in TOC - add it
            node = BookNode(
                title=title,
                level=level,
                number=number,
                start_page=page,
                start_element_idx=element_idx,
                confidence=confidence,
                source="detected"
            )
            
            # Find appropriate parent
            self._insert_node(root, node, elements)
        
        return root
    
    def _map_nodes_by_page(self, node: BookNode, page_map: Dict):
        """Recursively map all nodes by their start page"""
        if node.start_page:
            page_map[node.start_page] = node
        
        for child in node.children:
            self._map_nodes_by_page(child, page_map)
    
    def _insert_node(self, root: BookNode, node: BookNode, elements: List[Dict]):
        """Insert a node into the hierarchy at the appropriate position"""
        # Find best parent based on level and position
        candidates = []
        self._find_parent_candidates(root, node, candidates)
        
        if candidates:
            # Choose best parent (closest in page/index)
            best_parent = min(
                candidates,
                key=lambda p: abs((p.start_page or 0) - (node.start_page or 0))
            )
            best_parent.add_child(node)
        else:
            # Default to root
            root.add_child(node)
    
    def _find_parent_candidates(
        self,
        current: BookNode,
        node: BookNode,
        candidates: List[BookNode]
    ):
        """Recursively find suitable parent candidates"""
        # Parent must be one level higher
        if current.level == node.level - 1:
            # Must come before the node
            if current.start_page and node.start_page:
                if current.start_page <= node.start_page:
                    candidates.append(current)
        
        for child in current.children:
            self._find_parent_candidates(child, node, candidates)
    
    def fill_page_ranges(self, root: BookNode, total_pages: int):
        """
        Fill end_page and end_element_idx for all nodes.
        Each node's range extends to the start of the next sibling or parent's end.
        """
        self._fill_ranges_recursive(root, total_pages)
    
    def _fill_ranges_recursive(self, node: BookNode, total_pages: int):
        """Recursively fill ranges"""
        if not node.children:
            # Leaf node - will be filled by sibling or parent
            return
        
        # Sort children by start page/index
        node.children.sort(
            key=lambda n: (n.start_page or 0, n.start_element_idx or 0)
        )
        
        # Fill end ranges for each child
        for i, child in enumerate(node.children):
            if i < len(node.children) - 1:
                # End is next sibling's start - 1
                next_child = node.children[i + 1]
                child.end_page = (next_child.start_page - 1) if next_child.start_page else child.start_page
                child.end_element_idx = (next_child.start_element_idx - 1) if next_child.start_element_idx else child.start_element_idx
            else:
                # Last child - extends to parent's end
                child.end_page = node.end_page or total_pages
                child.end_element_idx = node.end_element_idx
            
            # Recurse
            self._fill_ranges_recursive(child, total_pages)
    
    def add_front_matter(self, root: BookNode) -> BookNode:
        """Add front matter section if content exists before first chapter"""
        if not root.children:
            return root
        
        first_child = min(root.children, key=lambda n: n.start_page or 0)
        first_page = first_child.start_page or 1
        
        if first_page > 1:
            front_matter = BookNode(
                title="Front Matter",
                level=1,
                start_page=1,
                end_page=first_page - 1,
                start_element_idx=0,
                end_element_idx=(first_child.start_element_idx or 1) - 1,
                confidence=1.0,
                source="inferred"
            )
            root.children.insert(0, front_matter)
            front_matter.parent = root
        
        return root
    
    def get_flat_chapters(self, root: BookNode) -> List[BookNode]:
        """Get flat list of all chapter-level nodes (level 1)"""
        chapters = []
        for child in root.children:
            if child.level == 1:
                chapters.append(child)
        return chapters
    
    def get_all_nodes_at_level(self, root: BookNode, level: int) -> List[BookNode]:
        """Get all nodes at a specific hierarchy level"""
        nodes = []
        self._collect_nodes_at_level(root, level, nodes)
        return nodes
    
    def _collect_nodes_at_level(self, node: BookNode, target_level: int, result: List[BookNode]):
        """Recursively collect nodes at target level"""
        if node.level == target_level:
            result.append(node)
        
        for child in node.children:
            self._collect_nodes_at_level(child, target_level, result)
    
    def print_tree(self, node: Optional[BookNode] = None, indent: int = 0):
        """Print hierarchy tree for debugging"""
        if node is None:
            node = self.root
        
        if node:
            prefix = "  " * indent
            page_info = f"p.{node.start_page}-{node.end_page}" if node.start_page else "no page"
            num_info = f"[{node.number}]" if node.number else ""
            print(f"{prefix}{num_info} {node.title} ({page_info}, confidence={node.confidence:.2f}, source={node.source})")
            
            for child in node.children:
                self.print_tree(child, indent + 1)
