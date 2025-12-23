"""
GROBID TEI-XML Parser (Safe 2-Level Hierarchy)
Robust strategy for books with NO numbering and NO consistent capitalization.
Structure: Book -> Chapter -> [All Sections as Siblings]
"""
import re
import logging
from typing import List, Dict, Optional
from lxml import etree

logger = logging.getLogger("grobid_parser")

class GrobidTEIParser:
    """
    Parser that enforces a strict 2-Level Hierarchy:
    1. Level 1 (Anchor): Detected via 'Chapter X', 'Part I'.
    2. Level 2 (Content): All other sections are treated as direct children of the current Anchor.
    3. Skips Front/Back matter (Preface, Index, Copyright).
    """
    
    TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}
    
    def __init__(self):
        self.sections = []
        self.metadata = {}
        
        # 1. ANCHOR PATTERN: The only triggers for a new top-level container.
        # Matches: "Chapter 1", "Part I", "Module 5", "Unit 2", "1. Introduction"
        self.anchor_pattern = re.compile(
            r"^(chapter\s+\d+|part\s+[IVX\d]+|module\s+\d+|unit\s+\d+|^\d+\.\s+([A-Z]|$))", 
            re.IGNORECASE
        )

        # 2. IGNORE PATTERN: Explicitly skip these sections.
        self.ignore_pattern = re.compile(
            r"^(preface|foreword|acknowledg+ments?|copyright|table\s+of\s+contents|contents|list\s+of\s+|dedication|abstract|about\s+the\s+author|colophon|index|bibliography|references)", 
            re.IGNORECASE
        )

    def parse_tei_xml(self, tei_content: str) -> Dict:
        """
        Parse TEI-XML content from GROBID into a tree structure.
        """
        try:
            # Handle both bytes and string input
            if isinstance(tei_content, bytes):
                root = etree.fromstring(tei_content)
            else:
                root = etree.fromstring(tei_content.encode('utf-8'))
            
            # Extract metadata
            self.metadata = self._extract_metadata(root)
            
            # Extract structured tree
            self.sections = self._extract_book_structure(root)
            
            return {
                "metadata": self.metadata,
                "sections": self.sections,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Failed to parse TEI-XML: {e}")
            return {
                "metadata": {},
                "sections": [],
                "success": False,
                "error": str(e)
            }
    
    def _extract_book_structure(self, root) -> List[Dict]:
        """
        Iterates through divs to reconstruct the book tree.
        """
        body = root.find(".//tei:text/tei:body", self.TEI_NS)
        if body is None:
            return []

        all_divs = body.findall("tei:div", self.TEI_NS)
        
        structured_chapters = []
        current_chapter_node = None
        
        # 'Content Started' flag prevents skipping legitimate "Introduction" chapters
        # just because they appear before "Chapter 1"
        has_content_started = False

        for div in all_divs:
            node = self._process_div_content(div)
            title = node["title"]
            clean_title = title.strip().lower()

            # --- 1. SKIP LOGIC ---
            # If we haven't hit Chapter 1 yet, be aggressive about skipping junk
            if not has_content_started:
                if self.ignore_pattern.match(title):
                    logger.info(f"Skipping Front Matter: {title}")
                    continue
                # Skip tiny generic snippets at the start (artifacts/page headers)
                if len(node["text"]) < 50 and "intro" not in clean_title:
                    continue

            # --- 2. ANCHOR LOGIC (New Chapter) ---
            if self.anchor_pattern.match(title):
                has_content_started = True
                
                # Create NEW Chapter Node
                current_chapter_node = {
                    "title": title,
                    "text": node["text"],
                    "path": [title],      # Path Context: [Chapter 1]
                    "subsections": [] 
                }
                structured_chapters.append(current_chapter_node)
                logger.info(f"New Chapter Detected: {title}")
                
            # --- 3. CHILD LOGIC (Section inside Chapter) ---
            elif current_chapter_node:
                # We are inside a chapter. 
                # Since we can't distinguish Section vs Subsection, we treat ALL as direct children.
                # This creates a flat list of sections under the chapter.
                
                # Update Path Context: [Chapter 1, Section Title]
                node["path"] = current_chapter_node["path"] + [title]
                current_chapter_node["subsections"].append(node)
                
            # --- 4. ORPHAN LOGIC (Before Chapter 1) ---
            else:
                # This handles "Introduction" or "Prologue" that appears before Chapter 1
                # We treat them as root-level chapters themselves.
                has_content_started = True 
                node["path"] = [title]
                structured_chapters.append(node)
                logger.info(f"Root Level Section (Pre-Chapter): {title}")

        return structured_chapters

    def _process_div_content(self, div_element) -> Dict:
        """
        Process a single div to extract text and title.
        """
        # Get Title
        head = div_element.find("tei:head", self.TEI_NS)
        title = self._extract_text_recursive(head).strip() if head is not None else "Untitled Section"
        
        # Get Content (Paragraphs, Formulas, Lists)
        content_text_parts = []
        
        for child in div_element:
            # Skip the head we already processed
            if child == head:
                continue
                
            tag = child.tag.replace(f"{{{self.TEI_NS['tei']}}}", "")
            
            if tag == "p":
                text = self._extract_text_recursive(child).strip()
                if text:
                    content_text_parts.append(text)
            elif tag == "formula":
                 text = child.text if child.text else ""
                 if text:
                     content_text_parts.append(f"[FORMULA: {text}]")
            elif tag == "list":
                 for item in child.findall(".//tei:item", self.TEI_NS):
                    text = self._extract_text_recursive(item).strip()
                    if text:
                        content_text_parts.append(f"• {text}")

        return {
            "title": title,
            "text": "\n\n".join(content_text_parts),
            "path": [title], # Placeholder, will be updated by parent logic
            "subsections": [] 
        }

    def _extract_metadata(self, root) -> Dict:
        """Extract document metadata from TEI header"""
        metadata = {
            "title": "Unknown Title",
            "authors": [],
            "abstract": ""
        }
        
        try:
            # Title
            title_elem = root.find(".//tei:titleStmt/tei:title", self.TEI_NS)
            if title_elem is not None and title_elem.text:
                metadata["title"] = title_elem.text.strip()
            
            # Authors
            authors = root.findall(".//tei:sourceDesc//tei:author", self.TEI_NS)
            for author in authors:
                surname = author.find(".//tei:surname", self.TEI_NS)
                if surname is not None and surname.text:
                    metadata["authors"].append(surname.text.strip())
            
            # Abstract
            abstract = root.find(".//tei:profileDesc//tei:abstract", self.TEI_NS)
            if abstract is not None:
                metadata["abstract"] = self._extract_text_recursive(abstract).strip()
            
        except Exception as e:
            logger.warning(f"Error extracting metadata: {e}")
        
        return metadata

    def _extract_text_recursive(self, element) -> str:
        """Recursively extract text from element and children"""
        if element is None:
            return ""
        return "".join(element.itertext())