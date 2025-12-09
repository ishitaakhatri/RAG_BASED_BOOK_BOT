"""
GROBID TEI-XML Parser
Extracts structured content from GROBID's TEI output
"""
import re
import logging
from typing import List, Dict, Optional, Tuple
from lxml import etree

logger = logging.getLogger("grobid_parser")


class GrobidTEIParser:
    """Parser for GROBID's TEI-XML output format"""
    
    # TEI namespace
    TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}
    
    def __init__(self):
        self.sections = []
        self.metadata = {}
    
    def parse_tei_xml(self, tei_content: str) -> Dict:
        """
        Parse TEI-XML content from GROBID
        
        Args:
            tei_content: TEI-XML string
            
        Returns:
            Dictionary with structured content
        """
        try:
            root = etree.fromstring(tei_content.encode('utf-8'))
            
            # Extract metadata
            self.metadata = self._extract_metadata(root)
            
            # Extract structured sections
            self.sections = self._extract_sections(root)
            
            # Extract references/citations
            references = self._extract_references(root)
            
            return {
                "metadata": self.metadata,
                "sections": self.sections,
                "references": references,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Failed to parse TEI-XML: {e}")
            return {
                "metadata": {},
                "sections": [],
                "references": [],
                "success": False,
                "error": str(e)
            }
    
    def _extract_metadata(self, root) -> Dict:
        """Extract document metadata from TEI header"""
        metadata = {
            "title": "",
            "authors": [],
            "abstract": "",
            "keywords": []
        }
        
        try:
            # Title
            title_elem = root.find(".//tei:titleStmt/tei:title", self.TEI_NS)
            if title_elem is not None and title_elem.text:
                metadata["title"] = title_elem.text.strip()
            
            # Authors
            authors = root.findall(".//tei:sourceDesc//tei:author", self.TEI_NS)
            for author in authors:
                forename = author.find(".//tei:forename", self.TEI_NS)
                surname = author.find(".//tei:surname", self.TEI_NS)
                
                name_parts = []
                if forename is not None and forename.text:
                    name_parts.append(forename.text.strip())
                if surname is not None and surname.text:
                    name_parts.append(surname.text.strip())
                
                if name_parts:
                    metadata["authors"].append(" ".join(name_parts))
            
            # Abstract
            abstract = root.find(".//tei:profileDesc//tei:abstract", self.TEI_NS)
            if abstract is not None:
                abstract_text = self._extract_text_recursive(abstract)
                metadata["abstract"] = abstract_text.strip()
            
            # Keywords
            keywords = root.findall(".//tei:keywords//tei:term", self.TEI_NS)
            metadata["keywords"] = [
                kw.text.strip() for kw in keywords 
                if kw.text and kw.text.strip()
            ]
            
        except Exception as e:
            logger.warning(f"Error extracting metadata: {e}")
        
        return metadata
    
    def _extract_sections(self, root) -> List[Dict]:
        """
        Extract hierarchical sections from document body
        Returns list of sections with structure information
        """
        sections = []
        
        try:
            body = root.find(".//tei:text/tei:body", self.TEI_NS)
            if body is None:
                return sections
            
            # Process all div elements (sections)
            divs = body.findall(".//tei:div", self.TEI_NS)
            
            for div_idx, div in enumerate(divs):
                section = self._process_div_element(div, div_idx)
                if section and section.get("text"):
                    sections.append(section)
            
        except Exception as e:
            logger.error(f"Error extracting sections: {e}")
        
        return sections
    
    def _process_div_element(self, div, section_idx: int) -> Optional[Dict]:
        """Process a single div (section) element"""
        try:
            # Get section type and number
            section_type = div.get("type", "section")
            section_n = div.get("n", "")
            
            # Extract section heading
            head = div.find("tei:head", self.TEI_NS)
            section_title = ""
            if head is not None:
                section_title = self._extract_text_recursive(head)
            
            # Extract all paragraphs in this section
            paragraphs = []
            for p in div.findall(".//tei:p", self.TEI_NS):
                p_text = self._extract_text_recursive(p)
                if p_text.strip():
                    paragraphs.append(p_text.strip())
            
            # Combine into section text
            section_text = "\n\n".join(paragraphs)
            
            if not section_text.strip():
                return None
            
            # Check for code blocks
            contains_code = self._detect_code_in_element(div)
            
            # Extract formulas/equations
            formulas = self._extract_formulas(div)
            
            return {
                "section_idx": section_idx,
                "section_type": section_type,
                "section_number": section_n,
                "section_title": section_title.strip(),
                "text": section_text,
                "contains_code": contains_code,
                "formulas": formulas,
                "paragraph_count": len(paragraphs)
            }
            
        except Exception as e:
            logger.warning(f"Error processing div element: {e}")
            return None
    
    def _extract_text_recursive(self, element) -> str:
        """Recursively extract text from element and children"""
        texts = []
        
        if element.text:
            texts.append(element.text)
        
        for child in element:
            # Handle special elements
            if child.tag.endswith("formula"):
                formula_text = child.text or ""
                texts.append(f"[FORMULA: {formula_text}]")
            elif child.tag.endswith("ref"):
                ref_text = child.text or ""
                ref_type = child.get("type", "")
                if ref_type == "bibr":
                    texts.append(f"[{ref_text}]")
                else:
                    texts.append(ref_text)
            else:
                texts.append(self._extract_text_recursive(child))
            
            if child.tail:
                texts.append(child.tail)
        
        return "".join(texts)
    
    def _detect_code_in_element(self, element) -> bool:
        """Check if element contains code blocks"""
        # Look for code-related tags or patterns
        code_elements = element.findall(".//tei:code", self.TEI_NS)
        if code_elements:
            return True
        
        # Check text content for code patterns
        text = self._extract_text_recursive(element)
        code_patterns = [
            r'\bdef\s+\w+\s*\(',
            r'\bclass\s+\w+',
            r'\bimport\s+\w+',
            r'function\s+\w+\s*\(',
            r'public\s+(class|static|void)'
        ]
        
        for pattern in code_patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    def _extract_formulas(self, element) -> List[str]:
        """Extract mathematical formulas from element"""
        formulas = []
        
        formula_elements = element.findall(".//tei:formula", self.TEI_NS)
        for formula in formula_elements:
            formula_text = self._extract_text_recursive(formula)
            if formula_text.strip():
                formulas.append(formula_text.strip())
        
        return formulas
    
    def _extract_references(self, root) -> List[Dict]:
        """Extract bibliography/references"""
        references = []
        
        try:
            # Find bibliography section
            bibl_structs = root.findall(".//tei:back//tei:listBibl//tei:biblStruct", self.TEI_NS)
            
            for idx, bibl in enumerate(bibl_structs):
                ref = self._parse_bibl_struct(bibl, idx)
                if ref:
                    references.append(ref)
        
        except Exception as e:
            logger.warning(f"Error extracting references: {e}")
        
        return references
    
    def _parse_bibl_struct(self, bibl, ref_idx: int) -> Optional[Dict]:
        """Parse a single biblStruct element"""
        try:
            # Extract title
            title = bibl.find(".//tei:title", self.TEI_NS)
            title_text = title.text.strip() if title is not None and title.text else ""
            
            # Extract authors
            authors = []
            author_elements = bibl.findall(".//tei:author", self.TEI_NS)
            for author in author_elements:
                forename = author.find(".//tei:forename", self.TEI_NS)
                surname = author.find(".//tei:surname", self.TEI_NS)
                
                name = ""
                if forename is not None and forename.text:
                    name += forename.text.strip() + " "
                if surname is not None and surname.text:
                    name += surname.text.strip()
                
                if name.strip():
                    authors.append(name.strip())
            
            # Extract year
            date = bibl.find(".//tei:date", self.TEI_NS)
            year = date.get("when", "") if date is not None else ""
            
            if not title_text:
                return None
            
            return {
                "ref_id": f"ref_{ref_idx}",
                "title": title_text,
                "authors": authors,
                "year": year
            }
            
        except Exception as e:
            logger.warning(f"Error parsing biblStruct: {e}")
            return None
    
    def convert_to_pages_format(self, parsed_data: Dict) -> List[Dict]:
        """
        Convert parsed GROBID data to pages format expected by semantic chunker
        
        Returns:
            List of dicts with 'page', 'text', 'structure' keys
        """
        pages = []
        
        if not parsed_data.get("success"):
            return pages
        
        sections = parsed_data.get("sections", [])
        
        # Group sections into logical pages (by section for now)
        for section in sections:
            page_data = {
                "page": section["section_idx"] + 1,  # 1-indexed
                "text": section["text"],
                "structure": {
                    "type": section["section_type"],
                    "title": section["section_title"],
                    "section_number": section["section_number"],
                    "contains_code": section["contains_code"],
                    "formulas": section.get("formulas", [])
                }
            }
            pages.append(page_data)
        
        return pages


# Convenience function
def parse_grobid_tei(tei_content: str) -> Dict:
    """
    Quick function to parse GROBID TEI-XML
    
    Args:
        tei_content: TEI-XML string from GROBID
        
    Returns:
        Parsed data dictionary
    """
    parser = GrobidTEIParser()
    return parser.parse_tei_xml(tei_content)
