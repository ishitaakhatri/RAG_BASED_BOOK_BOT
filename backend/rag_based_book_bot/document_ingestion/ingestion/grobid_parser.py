"""
GROBID TEI-XML Parser (Safe 2-Level Hierarchy)
Robust strategy for Books (Strict) and Research Papers (Flat/Abstract-First).
Structure: 
  - Books: Book -> Chapter -> [Sections]
  - Papers: Paper -> Abstract/Intro -> [Sections]
"""
import re
import logging
from typing import List, Dict, Optional
from lxml import etree

logger = logging.getLogger("grobid_parser")

class GrobidTEIParser:
    """
    Parser that enforces a strict 2-Level Hierarchy but adapts for Papers:
    1. Level 1 (Anchor): Detected via 'Chapter X', 'Part I', or standard Paper headers ('Abstract', 'Introduction').
    2. Level 2 (Content): All other sections are treated as direct children of the current Anchor.
    3. Book Mode: Skips Front/Back matter (Preface, Index, Copyright).
    4. Paper Mode: Preserves Abstract, Introduction, Conclusion.
    """
    
    TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}
    
    def __init__(self):
        self.sections = []
        self.metadata = {}
        
        # 1. ANCHOR PATTERN: Triggers for a new top-level container.
        # Matches: "Chapter 1", "Part I", "1. Introduction"
        self.anchor_pattern = re.compile(
            r"^(chapter\s+\d+|part\s+[IVX\d]+|module\s+\d+|unit\s+\d+|^\d+\.\s+([A-Z]|$))", 
            re.IGNORECASE
        )

        # 2. IGNORE PATTERNS
        # Strict ignoring for books (skip front matter)
        self.book_ignore_pattern = re.compile(
            r"^(preface|foreword|acknowledg+ments?|copyright|table\s+of\s+contents|contents|list\s+of\s+|dedication|colophon|index|bibliography|references)", 
            re.IGNORECASE
        )
        
        # Looser ignoring for papers (Keep Abstract, References might be excluded if desired, but we keep text flow)
        self.paper_ignore_pattern = re.compile(
            r"^(copyright|table\s+of\s+contents|contents|list\s+of\s+tables|list\s+of\s+figures)", 
            re.IGNORECASE
        )

    def parse_tei_xml(self, tei_content: str, is_paper: bool = False) -> Dict:
        """
        Parse TEI-XML content from GROBID into a tree structure.
        Args:
            tei_content: The XML string from GROBID.
            is_paper: If True, uses paper-specific extraction (preserves Abstract, etc).
        """
        try:
            # Handle both bytes and string input
            if isinstance(tei_content, bytes):
                root = etree.fromstring(tei_content)
            else:
                root = etree.fromstring(tei_content.encode('utf-8'))
            
            # Extract metadata
            self.metadata = self._extract_metadata(root)
            
            # Extract structured tree based on document type
            self.sections = self._extract_structure(root, is_paper)
            
            # Fallback for Papers: If no sections found, try to grab the whole body as one chunk
            if is_paper and not self.sections:
                self.sections = self._extract_flat_body(root)
            
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
    
    def _extract_structure(self, root, is_paper: bool) -> List[Dict]:
        """
        Iterates through divs to reconstruct the document tree.
        """
        body = root.find(".//tei:text/tei:body", self.TEI_NS)
        if body is None:
            return []

        all_divs = body.findall("tei:div", self.TEI_NS)
        
        structured_chapters = []
        current_anchor_node = None
        
        # 'Content Started' flag prevents skipping legitimate "Introduction" chapters
        has_content_started = False
        
        # Select the correct ignore pattern
        ignore_pattern = self.paper_ignore_pattern if is_paper else self.book_ignore_pattern

        # FOR PAPERS: Inject Abstract as the first section if it exists in metadata
        if is_paper and self.metadata.get("abstract"):
            abstract_node = {
                "title": "Abstract",
                "text": self.metadata["abstract"],
                "path": ["Abstract"],
                "subsections": []
            }
            structured_chapters.append(abstract_node)
            has_content_started = True

        for div in all_divs:
            node = self._process_div_content(div)
            title = node["title"]
            clean_title = title.strip().lower()

            # --- 1. SKIP LOGIC ---
            if not has_content_started:
                if ignore_pattern.match(title):
                    logger.info(f"Skipping Matter: {title}")
                    continue
                # For books, skip tiny generic snippets at start. For papers, be more lenient.
                if not is_paper and len(node["text"]) < 50 and "intro" not in clean_title:
                    continue

            # --- 2. ANCHOR LOGIC (New Chapter/Section) ---
            # For papers, "Introduction", "Methodology" are anchors even if not numbered
            is_anchor = False
            if self.anchor_pattern.match(title):
                is_anchor = True
            elif is_paper and ("introduction" in clean_title or "conclusion" in clean_title or "results" in clean_title):
                is_anchor = True

            if is_anchor:
                has_content_started = True
                
                # Create NEW Anchor Node
                current_anchor_node = {
                    "title": title,
                    "text": node["text"],
                    "path": [title],      # Path Context: [Chapter 1]
                    "subsections": [] 
                }
                structured_chapters.append(current_anchor_node)
                logger.info(f"New Anchor Detected: {title}")
                
            # --- 3. CHILD LOGIC (Subsection) ---
            elif current_anchor_node:
                # We are inside an anchor. Treat as child.
                node["path"] = current_anchor_node["path"] + [title]
                current_anchor_node["subsections"].append(node)
                
            # --- 4. ORPHAN LOGIC (Before first anchor) ---
            else:
                # This handles "Introduction" in books before Ch1, or initial text in papers
                has_content_started = True 
                node["path"] = [title]
                structured_chapters.append(node)
                logger.info(f"Root Level Section: {title}")

        return structured_chapters

    def _extract_flat_body(self, root) -> List[Dict]:
        """Fallback for papers with no internal structure (just text)"""
        body = root.find(".//tei:text/tei:body", self.TEI_NS)
        if body is None:
            return []
            
        text = self._extract_text_recursive(body).strip()
        if not text:
            return []
            
        return [{
            "title": "Full Content", 
            "text": text, 
            "path": ["Paper"], 
            "subsections": []
        }]

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