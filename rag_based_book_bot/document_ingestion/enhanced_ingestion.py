"""
Enhanced Book Ingestion Pipeline

Multi-strategy approach for accurate chapter/section detection:
1. TOC parsing (primary)
2. Font-based heading detection
3. Pattern-based detection
4. Element type detection (from unstructured)
5. Chapter-aware chunking (no cross-chapter chunks)
6. Pinecone upload with rich hierarchical metadata
"""
import os
import re
import uuid
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
os.environ['UNSTRUCTURED_DISABLE_INFERENCE'] = '1'


import pdfplumber
# try:
#     from unstructured.partition.pdf import partition_pdf
#     HAS_UNSTRUCTURED = True
# except Exception as e:
#     print(f"Warning: Could not import unstructured: {e}")
#     HAS_UNSTRUCTURED = False

from sentence_transformers import SentenceTransformer
import tiktoken
from dotenv import load_dotenv
from pinecone import Pinecone

from rag_based_book_bot.document_ingestion.ingestion.toc_parser import TOCParser, TOCEntry
from rag_based_book_bot.document_ingestion.ingestion.hierarchy_builder import HierarchyBuilder, BookNode
from rag_based_book_bot.document_ingestion.ingestion.chapter_chunker import ChapterChunker, ChunkMetadata


load_dotenv()

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "coding-books"
NAMESPACE = "books_rag"

CHUNK_SIZE_TOKENS = 1500
CHUNK_OVERLAP_TOKENS = 200
BATCH_SIZE = 100

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ENCODING_NAME = "cl100k_base"

# Patterns for chapter detection
CHAPTER_PATTERNS = [
    re.compile(r"^(?:chapter|ch\.?)\s+(\d+)\b[:\s-]*(.*)$", re.I),
    re.compile(r"^(\d+)[:\.\s-]+(.+)$"),  # "1. Title" or "1: Title"
    re.compile(r"^(appendix|epilogue|introduction|preface|foreword)\b[:\s-]*(.*)$", re.I),
]


@dataclass
class BookMetadata:
    """Metadata about the book"""
    title: str
    author: str = "Unknown"
    source_file: str = ""
    total_pages: int = 0
    total_chapters: int = 0


class EnhancedBookIngestor:
    """Enhanced book ingestion with multi-strategy structure detection"""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.tokenizer = tiktoken.get_encoding(ENCODING_NAME)
        
        if not PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY not set")
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = self.pc.Index(INDEX_NAME)
        
        # Initialize sub-components
        self.toc_parser = TOCParser()
        self.hierarchy_builder = HierarchyBuilder()
        self.chapter_chunker = ChapterChunker(
            chunk_size=CHUNK_SIZE_TOKENS,
            overlap=CHUNK_OVERLAP_TOKENS
        )
    
    def ingest_book(self, pdf_path: str, book_title: Optional[str] = None, author: str = "Unknown") -> BookMetadata:
        """
        Main ingestion pipeline.
        
        Args:
            pdf_path: Path to PDF file
            book_title: Optional book title (defaults to filename)
            author: Author name
        
        Returns:
            BookMetadata with ingestion statistics
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        print(f"\n{'='*60}")
        print(f"INGESTING: {os.path.basename(pdf_path)}")
        print(f"{'='*60}\n")
        
        # === PASS 1: Parse PDF with multiple strategies ===
        print("[1/6] Parsing PDF with multiple strategies...")
        elements = self._parse_pdf_multiway(pdf_path)
        total_pages = self._get_total_pages(pdf_path)
        
        print(f"  → Extracted {len(elements)} elements from {total_pages} pages")
        
        # === PASS 2: Try to detect and parse TOC ===
        print("[2/6] Detecting Table of Contents...")
        toc_entries = self.toc_parser.extract_toc(elements)
        
        if toc_entries:
            print(f"  → Found TOC with {len(toc_entries)} entries")
            if self.debug:
                for entry in toc_entries[:10]:
                    print(f"     - [{entry.level}] {entry.number} {entry.title} (p.{entry.page})")
        else:
            print("  → No TOC detected, will use heading detection")
        
        # === PASS 3: Detect headings from multiple sources ===
        print("[3/6] Detecting headings from font/pattern/structure...")
        detected_headings = self._detect_headings_multiway(pdf_path, elements)
        
        print(f"  → Detected {len(detected_headings)} potential headings")
        
        # === PASS 4: Build hierarchical structure ===
        print("[4/6] Building hierarchical structure...")
        if toc_entries:
            # Convert TOC entries to hierarchy format
            toc_hierarchy = []
            for entry in toc_entries:
                toc_hierarchy.append({
                    "title": entry.title,
                    "number": entry.number,
                    "page": entry.page,
                    "level": entry.level
                })
            root = self.hierarchy_builder.build_from_toc(toc_hierarchy)
        else:
            # Build from detected headings only
            root = BookNode(title="Root", level=0, source="root")
        
        # Merge detected headings
        if detected_headings:
            root = self.hierarchy_builder.merge_detected_headings(
                root, detected_headings, elements
            )
        
        # Add front matter if needed
        root = self.hierarchy_builder.add_front_matter(root)
        
        # Fill page ranges
        root.end_page = total_pages
        root.end_element_idx = len(elements) - 1
        self.hierarchy_builder.fill_page_ranges(root, total_pages)
        
        # Get chapters
        chapters = self.hierarchy_builder.get_flat_chapters(root)
        
        print(f"  → Built hierarchy with {len(chapters)} chapters")
        
        if self.debug:
            print("\n  Hierarchy Tree:")
            self.hierarchy_builder.print_tree(root)
            print()
        
        # === PASS 5: Chapter-aware chunking ===
        print("[5/6] Chunking with chapter boundaries...")
        
        # Convert chapters to dict format for chunker
        chapter_dicts = [self._node_to_dict(ch, elements) for ch in chapters]
        
        chunks = self.chapter_chunker.chunk_chapters(
            chapter_dicts,
            elements,
            book_title or os.path.basename(pdf_path),
            author
        )
        
        print(f"  → Created {len(chunks)} chunks (respecting chapter boundaries)")
        
        # === PASS 6: Embed and upsert to Pinecone ===
        print("[6/6] Generating embeddings and uploading to Pinecone...")
        self._embed_and_upsert(chunks)
        
        print(f"  → Successfully uploaded {len(chunks)} chunks\n")
        
        # Create book metadata
        metadata = BookMetadata(
            title=book_title or os.path.basename(pdf_path),
            author=author,
            source_file=pdf_path,
            total_pages=total_pages,
            total_chapters=len(chapters)
        )
        
        print(f"{'='*60}")
        print(f"INGESTION COMPLETE")
        print(f"  Book: {metadata.title}")
        print(f"  Chapters: {metadata.total_chapters}")
        print(f"  Pages: {metadata.total_pages}")
        print(f"  Chunks: {len(chunks)}")
        print(f"{'='*60}\n")
        
        return metadata
    
    def _parse_pdf_multiway(self, pdf_path: str) -> List[Dict]:
        """Parse PDF using pdfplumber (skip unstructured to avoid DLL issues)"""
        elements = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                
                # Split into paragraphs
                paragraphs = text.split('\n\n')
                
                for para in paragraphs:
                    if para.strip():
                        elements.append({
                            "index": len(elements),
                            "type": "NarrativeText",
                            "text": para.strip(),
                            "metadata": {"page_number": page_num},
                            "page": page_num
                        })
        
        return elements

    
    def _get_total_pages(self, pdf_path: str) -> int:
        """Get total number of pages"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                return len(pdf.pages)
        except:
            return 0
    
    def _detect_headings_multiway(self, pdf_path: str, elements: List[Dict]) -> List[Dict]:
        """Detect headings using multiple strategies"""
        headings = []
        
        # Strategy 1: Font-based detection with pdfplumber
        try:
            font_headings = self._detect_font_based_headings(pdf_path)
            headings.extend(font_headings)
        except Exception as e:
            print(f"  Warning: Font-based detection failed ({e})")
        
        # Strategy 2: Element type-based (from unstructured)
        type_headings = self._detect_type_based_headings(elements)
        headings.extend(type_headings)
        
        # Strategy 3: Pattern-based detection
        pattern_headings = self._detect_pattern_based_headings(elements)
        headings.extend(pattern_headings)
        
        # Deduplicate by page and title
        unique_headings = []
        seen = set()
        
        for h in headings:
            key = (h.get("page"), h.get("text", "").strip()[:50])
            if key not in seen:
                seen.add(key)
                unique_headings.append(h)
        
        # Sort by page then level
        unique_headings.sort(key=lambda h: (h.get("page") or 0, -h.get("level", 2)))
        
        return unique_headings
    
    def _detect_font_based_headings(self, pdf_path: str) -> List[Dict]:
        """Detect headings based on font size using pdfplumber"""
        headings = []
        
        with pdfplumber.open(pdf_path) as pdf:
            all_sizes = []
            page_data = []
            
            # Collect font sizes
            for page_num, page in enumerate(pdf.pages, start=1):
                chars = page.chars
                
                if not chars:
                    continue
                
                # Group chars into lines
                lines = self._group_chars_to_lines(chars)
                page_data.append((page_num, lines, page))
                
                for line in lines:
                    if line["size"] > 0:
                        all_sizes.append(line["size"])
            
            if not all_sizes:
                return []
            
            # Calculate median size
            median_size = sorted(all_sizes)[len(all_sizes) // 2]
            threshold = median_size * 1.2
            
            # Find large-font lines
            for page_num, lines, page in page_data:
                for line in lines:
                    text = line["text"].strip()
                    size = line["size"]
                    
                    if not text or len(text) < 3:
                        continue
                    
                    # Check if large font or matches chapter pattern
                    is_large = size >= threshold
                    matches_pattern = any(p.match(text) for p in CHAPTER_PATTERNS)
                    is_centered = abs(line["x0"] + line["x1"] - page.width) / page.width < 0.2
                    is_near_top = line["top"] < 0.3 * page.height
                    
                    if is_large or matches_pattern or (is_centered and is_near_top):
                        # Determine level
                        level = self._guess_level_from_text(text, size, median_size)
                        
                        # Extract number
                        num = self._extract_number(text)
                        
                        headings.append({
                            "page": page_num,
                            "text": text,
                            "level": level,
                            "num": num,
                            "confidence": 0.7,
                            "source": "font"
                        })
        
        return headings
    
    def _group_chars_to_lines(self, chars: List[Dict]) -> List[Dict]:
        """Group characters into lines"""
        lines_map = {}
        
        for ch in chars:
            top = round(ch.get("top", 0))
            if top not in lines_map:
                lines_map[top] = []
            lines_map[top].append(ch)
        
        lines = []
        for top in sorted(lines_map.keys()):
            chars_in_line = lines_map[top]
            text = "".join(c.get("text", "") for c in chars_in_line)
            sizes = [c.get("size", 0) for c in chars_in_line if c.get("size")]
            avg_size = sum(sizes) / len(sizes) if sizes else 0
            x0 = min(c.get("x0", 0) for c in chars_in_line)
            x1 = max(c.get("x1", 0) for c in chars_in_line)
            
            lines.append({
                "text": text.strip(),
                "size": avg_size,
                "x0": x0,
                "x1": x1,
                "top": top
            })
        
        return lines
    
    def _detect_type_based_headings(self, elements: List[Dict]) -> List[Dict]:
        """Detect headings from element types (unstructured)"""
        headings = []
        
        for el in elements:
            el_type = (el.get("type") or "").lower()
            text = (el.get("text") or "").strip()
            
            if not text:
                continue
            
            # Check if title/heading type
            if "title" in el_type or "heading" in el_type:
                level = self._guess_level_from_text(text, 14, 12)
                num = self._extract_number(text)
                
                headings.append({
                    "page": el.get("page"),
                    "text": text,
                    "level": level,
                    "num": num,
                    "index": el.get("index"),
                    "confidence": 0.8,
                    "source": "type"
                })
        
        return headings
    
    def _detect_pattern_based_headings(self, elements: List[Dict]) -> List[Dict]:
        """Detect headings using regex patterns"""
        headings = []
        
        for el in elements:
            text = (el.get("text") or "").strip()
            
            if not text:
                continue
            
            # Try each chapter pattern
            for pattern in CHAPTER_PATTERNS:
                if pattern.match(text):
                    level = self._guess_level_from_text(text, 14, 12)
                    num = self._extract_number(text)
                    
                    headings.append({
                        "page": el.get("page"),
                        "text": text,
                        "level": level,
                        "num": num,
                        "index": el.get("index"),
                        "confidence": 0.75,
                        "source": "pattern"
                    })
                    break
        
        return headings
    
    def _guess_level_from_text(self, text: str, size: float, median_size: float) -> int:
        """Guess hierarchy level from text and font size"""
        text_lower = text.lower()
        
        # Level 1: Chapters
        if "chapter" in text_lower or "appendix" in text_lower or "part" in text_lower:
            return 1
        
        # Check numbering
        num = self._extract_number(text)
        if num:
            dots = num.count(".")
            if dots == 0:
                return 1  # "1" or "Chapter 1"
            elif dots == 1:
                return 2  # "1.1"
            else:
                return 3  # "1.1.1"
        
        # Use font size if available
        if size > median_size * 1.5:
            return 1
        elif size > median_size * 1.2:
            return 2
        else:
            return 3
    
    def _extract_number(self, text: str) -> Optional[str]:
        """Extract chapter/section number from text"""
        # Try patterns
        for pattern in CHAPTER_PATTERNS:
            match = pattern.match(text)
            if match:
                groups = match.groups()
                if groups:
                    return groups[0]
        
        # Try simple number pattern
        match = re.match(r"^(\d+(\.\d+)*)", text)
        if match:
            return match.group(1)
        
        return None
    
    def _node_to_dict(self, node: BookNode, elements: List[Dict]) -> Dict:
        """Convert BookNode to dictionary for chunker"""
        result = {
            "title": node.title,
            "number": node.number,
            "level": node.level,
            "start_page": node.start_page,
            "end_page": node.end_page,
            "start_element_idx": node.start_element_idx,
            "end_element_idx": node.end_element_idx,
            "children": []
        }
        
        # Recursively convert children
        for child in node.children:
            result["children"].append(self._node_to_dict(child, elements))
        
        return result
    
    def _embed_and_upsert(self, chunks: List[Tuple[str, ChunkMetadata]]):
        """Generate embeddings and upsert to Pinecone"""
        if not chunks:
            return
        
        # Extract texts
        texts = [chunk_text for chunk_text, _ in chunks]
        
        # Generate embeddings
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Prepare batch for Pinecone
        batch = []
        
        for (chunk_text, metadata), emb in zip(chunks, embeddings):
            pinecone_metadata = {
                "book_title": metadata.book_title or "",
                "author": metadata.author or "",
                "chapter_title": metadata.chapter_title or "",
                "chapter_number": metadata.chapter_number or "",
                "section_titles": metadata.section_titles or [],
                "section_numbers": metadata.section_numbers or [],
                "subsection_titles": metadata.subsection_titles or [],
                "subsection_numbers": metadata.subsection_numbers or [],
                "page_start": metadata.page_start or -1,
                "page_end": metadata.page_end or -1,
                "chunk_index": metadata.chunk_index,
                "contains_code": metadata.contains_code
            }
            
            batch.append({
                "id": str(uuid.uuid4()),
                "values": emb.tolist(),
                "metadata": pinecone_metadata
            })
            
            # Upsert when batch is full
            if len(batch) >= BATCH_SIZE:
                self.index.upsert(vectors=batch, namespace=NAMESPACE)
                batch = []
        
        # Upsert remaining
        if batch:
            self.index.upsert(vectors=batch, namespace=NAMESPACE)


if __name__ == "__main__":
    # Example usage
    ingestor = EnhancedBookIngestor(debug=True)
    
    # Replace with your PDF path
    pdf_path = r"C:/Users/Yuvraj/Downloads/hands-onmachinelearningwithscikit-learnkerasandtensorflow.pdf"
    
    if os.path.exists(pdf_path):
        metadata = ingestor.ingest_book(
            pdf_path,
            book_title="Hands-On Machine Learning",
            author="Aurélien Géron"
        )
        
        print("\nIngestion successful!")
        print(f"  Title: {metadata.title}")
        print(f"  Author: {metadata.author}")
        print(f"  Chapters: {metadata.total_chapters}")
        print(f"  Pages: {metadata.total_pages}")
    else:
        print(f"Error: PDF not found at {pdf_path}")
        print("Please update the path in the script.")
