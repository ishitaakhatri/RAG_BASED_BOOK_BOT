"""
GROBID TEI-XML Parser (Hierarchical)
Extracts a nested tree structure (Book -> Chapter -> Section) from GROBID's TEI output.
"""
import re
import logging
from typing import List, Dict, Optional
from lxml import etree

logger = logging.getLogger("grobid_parser")

class GrobidTEIParser:
    """Parser for GROBID's TEI-XML output format producing a hierarchical tree."""
    
    # TEI namespace
    TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}
    
    def __init__(self):
        self.sections = []
        self.metadata = {}
    
    def parse_tei_xml(self, tei_content: str) -> Dict:
        """
        Parse TEI-XML content from GROBID into a tree structure.
        """
        try:
            # Parse XML bytes
            root = etree.fromstring(tei_content.encode('utf-8'))
            
            # Extract metadata (Title, Authors, Abstract)
            self.metadata = self._extract_metadata(root)
            
            # Extract structured tree of sections
            self.sections = self._extract_sections_tree(root)
            
            return {
                "metadata": self.metadata,
                "sections": self.sections,  # This is now a nested tree
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

    def _extract_sections_tree(self, root) -> List[Dict]:
        """
        Extract sections recursively to build a hierarchy tree.
        Returns a list of top-level section nodes (e.g. Chapters).
        """
        body = root.find(".//tei:text/tei:body", self.TEI_NS)
        if body is None:
            return []

        # Start processing from the top-level divs in the body
        return [self._process_div_recursive(div) for div in body.findall("tei:div", self.TEI_NS)]

    def _process_div_recursive(self, div_element, parent_path=None) -> Dict:
        """
        Recursively process a div and its children to build a tree node.
        """
        if parent_path is None:
            parent_path = []

        # 1. Get Header (Title)
        head = div_element.find("tei:head", self.TEI_NS)
        title = self._extract_text_recursive(head).strip() if head is not None else "Untitled Section"
        
        # Current hierarchical path
        current_path = parent_path + [title]

        # 2. Get Direct Content (Paragraphs in this specific div)
        content_text_parts = []
        for child in div_element:
            if child.tag == f"{{{self.TEI_NS['tei']}}}p":
                text = self._extract_text_recursive(child).strip()
                if text:
                    content_text_parts.append(text)
            elif child.tag == f"{{{self.TEI_NS['tei']}}}formula":
                 text = child.text if child.text else ""
                 if text:
                     content_text_parts.append(f"[FORMULA: {text}]")

        full_text = "\n\n".join(content_text_parts)

        # 3. Build current node structure
        current_node = {
            "title": title,
            "text": full_text,
            "path": current_path,
            "subsections": []
        }

        # 4. Recursively process nested divs (Subsections)
        for child_div in div_element.findall("tei:div", self.TEI_NS):
            child_node = self._process_div_recursive(child_div, current_path)
            if child_node["text"] or child_node["subsections"]:
                current_node["subsections"].append(child_node)

        return current_node

    def _extract_text_recursive(self, element) -> str:
        """Recursively extract text from element and children"""
        texts = []
        if element.text:
            texts.append(element.text)
        for child in element:
            texts.append(self._extract_text_recursive(child))
            if child.tail:
                texts.append(child.tail)
        return "".join(texts)