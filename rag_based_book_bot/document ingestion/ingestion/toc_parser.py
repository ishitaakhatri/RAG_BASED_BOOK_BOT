"""
Table of Contents (TOC) Parser

Extracts hierarchical structure from PDF Table of Contents.
Handles various TOC formats and maps entries to actual content pages.
"""
import re
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass


# Regex patterns for TOC entry detection
TOC_PATTERNS = [
    # "Chapter 1: Title ........... 15"
    re.compile(r'^(?P<prefix>chapter|ch\.?)\s+(?P<num>\d+)[:\s-]*(?P<title>.*?)\s*\.{2,}\s*(?P<page>\d+)\s*$', re.I),
    # "1.2.3 Title ........... 20"
    re.compile(r'^(?P<num>\d+(\.\d+)*)\s+(?P<title>.*?)\s*\.{2,}\s*(?P<page>\d+)\s*$'),
    # "Title ........... 15" (no numbering)
    re.compile(r'^(?P<title>[A-Z].*?)\s*\.{2,}\s*(?P<page>\d+)\s*$'),
    # "1.2.3 Title    20" (no dots, just spaces)
    re.compile(r'^(?P<num>\d+(\.\d+)*)\s+(?P<title>.+?)\s{3,}(?P<page>\d+)\s*$'),
    # "Title    20" (no numbering, no dots)
    re.compile(r'^(?P<title>[A-Z][^.]{3,}?)\s{3,}(?P<page>\d+)\s*$'),
]


@dataclass
class TOCEntry:
    """Represents a single TOC entry"""
    title: str
    number: Optional[str]
    page: int
    level: int  # 1=chapter, 2=section, 3=subsection
    raw_text: str


class TOCParser:
    """Extracts and parses Table of Contents from PDF elements"""
    
    def __init__(self):
        self.toc_keywords = [
            "table of contents",
            "contents",
            "table des matières",  # French
            "índice",  # Spanish/Portuguese
        ]
    
    def detect_toc_region(self, elements: List[Dict]) -> Optional[Tuple[int, int]]:
        """
        Detect the start and end indices of TOC in parsed elements.
        
        Args:
            elements: List of parsed PDF elements (from unstructured or pdfplumber)
        
        Returns:
            Tuple of (start_idx, end_idx) or None if TOC not found
        """
        start_idx = None
        
        # Scan first 100 elements for TOC keywords
        for i, el in enumerate(elements[:100]):
            text = (el.get("text", "") or "").strip().lower()
            
            # Check if this element contains TOC keywords
            for keyword in self.toc_keywords:
                if keyword in text and len(text) < 50:  # Short heading
                    start_idx = i
                    break
            
            if start_idx is not None:
                break
        
        if start_idx is None:
            return None
        
        # Find end of TOC: usually when we hit long narrative paragraphs
        # or recognize main content (like "Chapter 1", "Introduction" with large gap)
        end_idx = None
        toc_entry_count = 0
        
        for i in range(start_idx + 1, min(start_idx + 500, len(elements))):
            text = (elements[i].get("text", "") or "").strip()
            
            if not text:
                continue
            
            # Check if this looks like a TOC entry
            is_toc_entry = any(pattern.match(text) for pattern in TOC_PATTERNS)
            
            if is_toc_entry:
                toc_entry_count += 1
            elif toc_entry_count > 5:  # We've seen enough TOC entries
                # Check if this is start of main content
                if len(text) > 200 or \
                   re.match(r'^(chapter|introduction|preface)\s+\d+', text, re.I):
                    end_idx = i - 1
                    break
        
        if end_idx is None:
            end_idx = min(start_idx + 400, len(elements) - 1)
        
        return (start_idx, end_idx) if toc_entry_count > 3 else None
    
    def parse_toc_text(self, text: str) -> Optional[TOCEntry]:
        """
        Parse a single TOC line into a TOCEntry.
        
        Args:
            text: Single line of text from TOC
        
        Returns:
            TOCEntry or None if not a valid TOC entry
        """
        text = text.strip()
        
        if not text or len(text) < 3:
            return None
        
        # Try each pattern
        for pattern in TOC_PATTERNS:
            match = pattern.match(text)
            if match:
                groups = match.groupdict()
                
                # Extract components
                title = groups.get("title", "").strip()
                number = groups.get("num", "").strip() if "num" in groups else None
                page_str = groups.get("page", "")
                
                # Validate page number
                try:
                    page = int(page_str)
                except (ValueError, TypeError):
                    continue
                
                # Skip if title is empty or too short
                if not title or len(title) < 2:
                    continue
                
                # Determine level from numbering
                level = self._determine_level(number, title)
                
                return TOCEntry(
                    title=title,
                    number=number,
                    page=page,
                    level=level,
                    raw_text=text
                )
        
        return None
    
    def _determine_level(self, number: Optional[str], title: str) -> int:
        """Determine hierarchy level from number format"""
        if not number:
            # No numbering - guess from title
            title_lower = title.lower()
            if any(word in title_lower for word in ["chapter", "part", "appendix"]):
                return 1
            return 2
        
        # Count dots to determine level
        # "1" -> level 1 (chapter)
        # "1.1" -> level 2 (section)
        # "1.1.1" -> level 3 (subsection)
        dot_count = number.count(".")
        return min(dot_count + 1, 3)
    
    def extract_toc(self, elements: List[Dict]) -> Optional[List[TOCEntry]]:
        """
        Extract complete TOC from parsed PDF elements.
        
        Args:
            elements: List of parsed PDF elements
        
        Returns:
            List of TOCEntry objects or None if TOC not found
        """
        # Detect TOC region
        toc_region = self.detect_toc_region(elements)
        
        if not toc_region:
            return None
        
        start_idx, end_idx = toc_region
        
        # Extract and parse TOC entries
        toc_entries = []
        
        for i in range(start_idx + 1, end_idx + 1):
            text = elements[i].get("text", "") or ""
            
            # Handle multi-line elements
            for line in text.splitlines():
                line = line.strip()
                
                if not line:
                    continue
                
                # Skip the "Contents" heading itself
                if line.lower() in self.toc_keywords:
                    continue
                
                # Try to parse as TOC entry
                entry = self.parse_toc_text(line)
                
                if entry:
                    toc_entries.append(entry)
        
        # Validate: should have at least a few entries
        if len(toc_entries) < 3:
            return None
        
        # Post-process: ensure page numbers are increasing (mostly)
        toc_entries = self._validate_and_clean(toc_entries)
        
        return toc_entries if toc_entries else None
    
    def _validate_and_clean(self, entries: List[TOCEntry]) -> List[TOCEntry]:
        """Remove outliers and validate TOC entries"""
        if not entries:
            return []
        
        # Remove entries with suspiciously high page numbers (likely parsing errors)
        max_reasonable_page = max(e.page for e in entries) if entries else 1000
        
        # Filter outliers: page numbers that are way beyond reasonable
        cleaned = []
        for entry in entries:
            # Skip entries with page > 9999 (likely parsing error)
            if entry.page > 9999:
                continue
            
            # Skip duplicates (same page and title)
            is_duplicate = any(
                e.page == entry.page and e.title == entry.title
                for e in cleaned
            )
            
            if not is_duplicate:
                cleaned.append(entry)
        
        return cleaned
    
    def build_hierarchy_from_toc(self, toc_entries: List[TOCEntry]) -> List[Dict]:
        """
        Build hierarchical structure from TOC entries.
        
        Returns:
            List of hierarchy nodes with parent-child relationships
        """
        hierarchy = []
        parent_stack = []  # Stack to track parent nodes at each level
        
        for entry in toc_entries:
            node = {
                "title": entry.title,
                "number": entry.number,
                "page": entry.page,
                "level": entry.level,
                "children": [],
                "parent": None,
            }
            
            # Pop stack until we find appropriate parent
            while parent_stack and parent_stack[-1]["level"] >= entry.level:
                parent_stack.pop()
            
            if parent_stack:
                # Attach to parent
                parent = parent_stack[-1]
                node["parent"] = parent
                parent["children"].append(node)
            else:
                # Top-level node
                hierarchy.append(node)
            
            # Add to stack
            parent_stack.append(node)
        
        return hierarchy
