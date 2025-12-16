"""
GROBID TEI-XML Parser (Book-Optimized)
Patches Grobid's flat structure by inferring hierarchy from Chapter titles.
"""
import re
import logging
from typing import List, Dict, Optional
from lxml import etree

logger = logging.getLogger("grobid_parser")

class GrobidTEIParser:
    """
    Parser for GROBID's TEI-XML output that manually reconstructs 
    Book -> Chapter -> Section hierarchy by analyzing titles.
    """
    
    # TEI namespace
    TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}
    
    def __init__(self):
        self.sections = []
        self.metadata = {}
        # Regex to identify Chapter starts. 
        # Matches: "Chapter 1", "1. Introduction", "PART I", "Module 1"
        self.chapter_pattern = re.compile(
            r"^(chapter\s+\d+|part\s+[IVX\d]+|module\s+\d+|^\d+\.\s+([A-Z]|$))", 
            re.IGNORECASE
        )
    
    def parse_tei_xml(self, tei_content: str) -> Dict:
        """
        Parse TEI-XML content from GROBID into a tree structure.
        """
        try:
            # Parse XML bytes
            root = etree.fromstring(tei_content.encode('utf-8'))
            
            # Extract metadata (Title, Authors, Abstract)
            self.metadata = self._extract_metadata(root)
            
            # Extract structured tree of sections (Custom Logic)
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
        Iterates through flat Grobid divs and reconstructs Book -> Chapter -> Section tree.
        Filters out front matter (pages before Chapter 1).
        """
        body = root.find(".//tei:text/tei:body", self.TEI_NS)
        if body is None:
            return []

        all_divs = body.findall("tei:div", self.TEI_NS)
        
        structured_chapters = []
        current_chapter = None
        
        # Flag to skip front matter (Preface, Foreword, etc.)
        is_content_started = False

        for div in all_divs:
            # 1. Process the raw div to get title and text
            node = self._process_div_flat(div)
            title = node["title"]
            
            # 2. Check if this div marks the start of a new Chapter
            if self.chapter_pattern.match(title):
                is_content_started = True # Stop skipping front matter
                
                # Save the previous chapter if it exists
                if current_chapter:
                    structured_chapters.append(current_chapter)
                
                # Init new Chapter node
                current_chapter = {
                    "title": title,
                    "text": node["text"], # Add the chapter intro text
                    "path": [title],      # Path is just [Chapter Title]
                    "subsections": []
                }
                logger.info(f"New Chapter Detected: {title}")

            elif is_content_started:
                # We are inside a chapter, treat this div as a subsection
                if current_chapter:
                    # Update path to include Chapter parent
                    node["path"] = current_chapter["path"] + [title]
                    current_chapter["subsections"].append(node)
                else:
                    # Edge case: Content started but no "Chapter" title found yet.
                    # Treat as top-level section.
                    structured_chapters.append(node)
            else:
                # We are still in front matter (Preface, TOC, etc.) -> SKIP
                continue

        # Append the final chapter
        if current_chapter:
            structured_chapters.append(current_chapter)

        # Fallback: If regex failed to find ANY chapters (e.g. flat book), return all flat nodes
        if not structured_chapters and all_divs:
            logger.warning("No chapters detected via Regex. Returning flat structure.")
            return [self._process_div_flat(div) for div in all_divs]

        return structured_chapters

    def _process_div_flat(self, div_element) -> Dict:
        """
        Process a single div to extract text and title, without recursion.
        """
        # Get Title
        head = div_element.find("tei:head", self.TEI_NS)
        title = self._extract_text_recursive(head).strip() if head is not None else "Untitled Section"
        
        # Get Content (Paragraphs & Formulas)
        content_text_parts = []
        for child in div_element:
            # Skip the head we already processed
            if child == head:
                continue
                
            if child.tag == f"{{{self.TEI_NS['tei']}}}p":
                text = self._extract_text_recursive(child).strip()
                if text:
                    content_text_parts.append(text)
            elif child.tag == f"{{{self.TEI_NS['tei']}}}formula":
                 text = child.text if child.text else ""
                 if text:
                     content_text_parts.append(f"[FORMULA: {text}]")

        return {
            "title": title,
            "text": "\n\n".join(content_text_parts),
            "path": [title], # Default path (will be overwritten if it becomes a child)
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
        texts = []
        if element.text:
            texts.append(element.text)
        for child in element:
            texts.append(self._extract_text_recursive(child))
            if child.tail:
                texts.append(child.tail)
        return "".join(texts)