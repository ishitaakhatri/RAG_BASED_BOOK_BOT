# enhanced_ingestor_paddle.py
"""
PaddleOCR-based Enhanced Ingestor (Windows-friendly, GPU with CUDA 12.9 support)
WITH PROPER EMBEDDING AND PINECONE UPSERT
"""
import os
import re
import io
import json
import uuid
import logging
import warnings
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any

# pdf + imaging
import pdfplumber
from PIL import Image
import numpy as np

# PaddleOCR (optional)
try:
    from paddleocr import PaddleOCR
    _HAS_PADDLEOCR = True
except Exception:
    _HAS_PADDLEOCR = False

# OCR fallback (Tesseract)
try:
    import pytesseract
    _HAS_TESSERACT = True
except Exception:
    _HAS_TESSERACT = False

# LLM / embeddings / utils
import tiktoken
from sentence_transformers import SentenceTransformer

try:
    import openai
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False

# Pinecone
try:
    from pinecone import Pinecone, ServerlessSpec
    _HAS_PINECONE = True
except Exception:
    _HAS_PINECONE = False

# local imports (your modules)
from rag_based_book_bot.document_ingestion.ingestion.toc_parser import TOCParser, TOCEntry  # type: ignore
from rag_based_book_bot.document_ingestion.ingestion.hierarchy_builder import HierarchyBuilder, BookNode  # type: ignore
from rag_based_book_bot.document_ingestion.ingestion.chapter_chunker import ChapterChunker, ChunkMetadata  # type: ignore

# config & logging
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX_NAME", "coding-books")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "books_rag")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
ENCODING_NAME = os.getenv("ENCODING_NAME", "cl100k_base")
VERIFIER_BATCH = int(os.getenv("VERIFIER_BATCH", "16"))
VERIFIER_MAX_TOKENS = int(os.getenv("VERIFIER_MAX_TOKENS", "512"))
USE_CUDA = os.getenv("USE_CUDA", "1") == "1"
BATCH_SIZE = 100  # For Pinecone upsert

logging.basicConfig(level=os.getenv("INGESTOR_LOG_LEVEL", "INFO"))
logger = logging.getLogger("ingestor_paddle")

CHAPTER_PATTERNS = [
    re.compile(r"^(?:chapter|ch\.?)\s+(\d+)\b[:\s-]*(.*)$", re.I),
    re.compile(r"^(\d+)[\.\:\s-]+(.+)$"),
    re.compile(r"^(appendix|epilogue|introduction|preface|foreword)\b[:\s-]*(.*)$", re.I),
]

@dataclass
class IngestorConfig:
    chunk_size: int = 1500
    overlap: int = 200
    encoding_name: str = ENCODING_NAME
    debug: bool = False

# --------------------------
# PaddleOCR wrapper - FIXED
# --------------------------
class PaddleLayoutWrapper:
    """Uses PaddleOCR for text extraction with fallback to pdfplumber"""

    def __init__(self, lang: str = "en"):
        self.lang = lang
        self.ocr = None
        if _HAS_PADDLEOCR:
            try:
                # Fixed: Remove use_gpu and show_log parameters
                # PaddleOCR automatically detects GPU availability
                self.ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang=self.lang
                )
                logger.info("PaddleOCR initialized successfully.")
            except Exception as e:
                logger.warning(f"PaddleOCR init failed: {e}")
                self.ocr = None
        else:
            logger.info("PaddleOCR not installed; will fallback to pdfplumber.")

    def page_image_to_blocks(self, pil_image: Image.Image) -> List[Dict]:
        """Return list of blocks with text and bounding boxes"""
        if not self.ocr:
            return []
        try:
            result = self.ocr.ocr(np.array(pil_image), det=True, rec=True, cls=True)
            blocks = []
            for line in result[0] if result and result[0] else []:
                try:
                    box = line[0]
                    text = line[1][0] if len(line) > 1 else ""
                    conf = float(line[1][1]) if len(line) > 1 else 0.0
                    xs = [pt[0] for pt in box]
                    ys = [pt[1] for pt in box]
                    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
                    blocks.append({
                        "bbox": [int(x0), int(y0), int(x1), int(y1)],
                        "text": text.strip(),
                        "score": conf,
                        "type": "Text"
                    })
                except Exception:
                    continue
            return blocks
        except Exception as e:
            logger.warning(f"PaddleOCR.ocr failed: {e}")
            return []

# --------------------------
# LLM verifier (simplified)
# --------------------------
class LLMVerifier:
    def __init__(self, batch_size: int = VERIFIER_BATCH, max_tokens: int = VERIFIER_MAX_TOKENS):
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.mode = "heuristic"
        logger.info("Verifier: heuristics only (fast mode).")

    def _heuristic_label(self, it: Dict):
        fs = float(it.get("font_size") or 0)
        top = float(it.get("top_ratio") or 0)
        centered = bool(it.get("is_centered"))
        if fs >= 28 and top < 0.2 and centered:
            return "CHAPTER", 0.9
        if fs >= 18 and it.get("space_before", 0) >= 30:
            return "SECTION", 0.8
        if top < 0.12 and fs <= 12 and centered:
            return "HEADER", 0.6
        return "NOT_HEADING", 0.4

    def verify(self, candidates: List[Dict]) -> List[Dict]:
        results = []
        for c in candidates:
            lbl, conf = self._heuristic_label(c)
            c["label"] = lbl
            c["confidence"] = conf
            results.append(c)
        return results

# --------------------------
# The main enhanced ingestor
# --------------------------
class EnhancedBookIngestorPaddle:
    def __init__(self, config: Optional[IngestorConfig] = None):
        self.config = config or IngestorConfig()
        self.paddle_wrapper = PaddleLayoutWrapper()  # Fixed: removed use_gpu parameter
        self.hierarchy_builder = HierarchyBuilder()
        self.chunker = ChapterChunker(
            chunk_size=self.config.chunk_size,
            overlap=self.config.overlap,
            encoding_name=self.config.encoding_name
        )
        self.verifier = LLMVerifier()
        self.tokenizer = tiktoken.get_encoding(self.config.encoding_name)
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info(f"Loaded embedding model: {EMBEDDING_MODEL}")
        
        # Initialize Pinecone
        self.pinecone_index = None
        if _HAS_PINECONE and PINECONE_API_KEY:
            try:
                self.pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
                
                # Create index if it doesn't exist
                existing_indexes = [idx.name for idx in self.pinecone_client.list_indexes()]
                if PINECONE_INDEX not in existing_indexes:
                    logger.info(f"Creating Pinecone index: {PINECONE_INDEX}")
                    dim = len(self.embedding_model.encode(["test"])[0])
                    self.pinecone_client.create_index(
                        name=PINECONE_INDEX,
                        dimension=dim,
                        metric="cosine",
                        spec=ServerlessSpec(cloud="aws", region="us-east-1")
                    )
                
                self.pinecone_index = self.pinecone_client.Index(PINECONE_INDEX)
                logger.info(f"Connected to Pinecone index: {PINECONE_INDEX}")
            except Exception as e:
                logger.warning(f"Pinecone init error: {e}")
        
        self.toc_parser = TOCParser()

    def _get_total_pages(self, pdf_path: str) -> int:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                return len(pdf.pages)
        except:
            return 0

    def _parse_pdf_to_elements(self, pdf_path: str) -> List[Dict]:
        elements = []
        with pdfplumber.open(pdf_path) as pdf:
            for pnum, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                paras = [para.strip() for para in text.split("\n\n") if para.strip()]
                for para in paras:
                    elements.append({
                        "index": len(elements),
                        "type": "NarrativeText",
                        "text": para,
                        "page": pnum,
                        "metadata": {"page_number": pnum}
                    })
        return elements

    def _render_page_image(self, pdf_path: str, page_num: int):
        try:
            with pdfplumber.open(pdf_path) as pdf:
                page = pdf.pages[page_num - 1]
                try:
                    return page.to_image(resolution=150).original
                except Exception:
                    return None
        except Exception as e:
            logger.warning(f"render page image failed: {e}")
            return None

    def _pdfplumber_text_blocks(self, page) -> List[Dict]:
        blocks = []
        try:
            txt = page.extract_text() or ""
            if txt.strip():
                blocks.append({
                    "text": txt.strip(),
                    "bbox": [0, 0, page.width, page.height],
                    "top": 0,
                    "font_size": 12,
                    "type": "Text"
                })
        except Exception as e:
            logger.warning(f"_pdfplumber_text_blocks error: {e}")
        return blocks

    def _extract_page_blocks(self, pdf_path: str, page_num: int) -> List[Dict]:
        blocks = []
        page_img = self._render_page_image(pdf_path, page_num)
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_num - 1]
            pw, ph = page.width, page.height
            if page_img is not None:
                vis_blocks = self.paddle_wrapper.page_image_to_blocks(page_img)
                if vis_blocks:
                    for vb in vis_blocks:
                        x0, y0, x1, y1 = vb.get("bbox", [0, 0, pw, ph])
                        text = vb.get("text", "")
                        blocks.append({
                            "id": str(uuid.uuid4()),
                            "text": text,
                            "page": page_num,
                            "bbox": [x0, y0, x1, y1],
                            "font_size": None,
                            "top": y0,
                            "type": "Text",
                            "score": vb.get("score", 0.0)
                        })
                    return blocks
            txt_blocks = self._pdfplumber_text_blocks(page)
            for tb in txt_blocks:
                blocks.append({
                    "id": str(uuid.uuid4()),
                    "text": tb.get("text", ""),
                    "page": page_num,
                    "bbox": tb.get("bbox", [0, 0, pw, ph]),
                    "font_size": tb.get("font_size"),
                    "top": tb.get("top", 0),
                    "type": tb.get("type", "Text")
                })
        return blocks

    def _is_centered(self, bbox, page_w):
        x0, _, x1, _ = bbox
        center = (x0 + x1) / 2.0
        return abs(center - page_w / 2.0) < (page_w * 0.18)

    def _estimate_font_from_text(self, text: str):
        L = len(text)
        if L < 30: return 22.0
        if L < 80: return 16.0
        return 12.0

    def detect_heading_candidates(self, pdf_path: str) -> List[Dict]:
        candidates = []
        total = self._get_total_pages(pdf_path)
        for p in range(1, min(total + 1, 50)):  # Limit to first 50 pages for speed
            try:
                blocks = self._extract_page_blocks(pdf_path, p)
                if not blocks: continue
                sorted_blocks = sorted(blocks, key=lambda b: b.get("top", 0))
                with pdfplumber.open(pdf_path) as pdf:
                    page = pdf.pages[p - 1]
                    pw, ph = page.width, page.height
                for idx, b in enumerate(sorted_blocks):
                    text = (b.get("text") or "").strip()
                    if not text or len(text) < 2: continue
                    bbox = b.get("bbox", [0, 0, pw, ph])
                    top = b.get("top", 0)
                    top_ratio = top / ph if ph else 0.0
                    is_centered = self._is_centered(bbox, pw)
                    prev_top = sorted_blocks[idx - 1].get("top") if idx > 0 else None
                    next_top = sorted_blocks[idx + 1].get("top") if idx + 1 < len(sorted_blocks) else None
                    space_before = (top - prev_top) if prev_top is not None else 0
                    space_after = ((next_top - (bbox[3] if len(bbox) > 3 else top)) if next_top is not None else 0)
                    font_size = b.get("font_size") or self._estimate_font_from_text(text)
                    snippet = text if len(text) < 400 else text[:400]
                    candidates.append({
                        "id": b.get("id") or str(uuid.uuid4()),
                        "text": text,
                        "page": p,
                        "bbox": bbox,
                        "font_size": font_size,
                        "top_ratio": top_ratio,
                        "is_centered": is_centered,
                        "space_before": space_before,
                        "space_after": space_after,
                        "context_snippet": snippet,
                        "source": b.get("type", "Text")
                    })
            except Exception as e:
                logger.warning(f"candidate detect failed page {p}: {e}")
        logger.info(f"Detected {len(candidates)} raw candidates.")
        return candidates

    def _candidates_to_detected_headings(self, candidates: List[Dict]) -> List[Dict]:
        out = []
        for c in candidates:
            num = None
            for pat in CHAPTER_PATTERNS:
                m = pat.match(c["text"])
                if m:
                    num = m.group(1)
                    break
            guessed_level = 1 if (c.get("font_size") or 0) >= 22 else 2
            out.append({
                "page": c.get("page"),
                "text": c.get("text"),
                "level": guessed_level,
                "num": num,
                "index": None,
                "confidence": 0.4,
                "source": c.get("source", "vision")
            })
        return out

    def _smooth_boundaries(self, root: BookNode, total_pages: int) -> BookNode:
        if not root or not root.children: return root
        root.children.sort(key=lambda n: (n.start_page or 0, n.start_element_idx or 0))
        for i, child in enumerate(root.children):
            if child.start_page is None:
                child.start_page = 1
            if i < len(root.children) - 1:
                nextc = root.children[i + 1]
                child.end_page = (nextc.start_page - 1) if nextc.start_page else child.end_page or total_pages
            else:
                child.end_page = child.end_page or total_pages
            self.hierarchy_builder._fill_ranges_recursive(child, total_pages)
        return root

    def _embed_and_upsert(self, chunks: List[Tuple[str, ChunkMetadata]], book_id: str):
        """Embed chunks and upsert to Pinecone"""
        if not self.pinecone_index:
            logger.warning("Pinecone not configured - skipping upsert")
            return
        
        if not chunks:
            logger.warning("No chunks to embed")
            return
        
        logger.info(f"Embedding and upserting {len(chunks)} chunks...")
        
        # Extract texts for embedding
        texts = [chunk[0] for chunk in chunks]
        
        # Generate embeddings in batches
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True, batch_size=32)
        
        # Prepare vectors for upsert
        vectors = []
        for i, ((chunk_text, metadata), embedding) in enumerate(zip(chunks, embeddings)):
            # Convert metadata to dict for Pinecone
            # Handle singular fields from ChunkMetadata - convert to lists for compatibility
            metadata_dict = {
                "text": chunk_text[:1000],  # Store preview only
                "book_id": book_id,
                "book_title": metadata.book_title,
                "author": metadata.author,
                "chapter_titles": [metadata.chapter_title] if metadata.chapter_title else [],
                "chapter_numbers": [str(metadata.chapter_number) if metadata.chapter_number else ""],
                "section_titles": metadata.section_titles if metadata.section_titles else [],
                "section_numbers": [str(n) if n else "" for n in (metadata.section_numbers if metadata.section_numbers else [])],
                "subsection_titles": metadata.subsection_titles if metadata.subsection_titles else [],
                "subsection_numbers": [str(n) if n else "" for n in (metadata.subsection_numbers if metadata.subsection_numbers else [])],
                "page_start": int(metadata.page_start) if metadata.page_start else 1,
                "page_end": int(metadata.page_end) if metadata.page_end else 1,
                "chunk_index": int(metadata.chunk_index),
                "contains_code": bool(metadata.contains_code)
            }
            
            vectors.append({
                "id": str(uuid.uuid4()),
                "values": embedding.tolist(),
                "metadata": metadata_dict
            })
        
        # Upsert in batches
        for i in range(0, len(vectors), BATCH_SIZE):
            batch = vectors[i:i + BATCH_SIZE]
            try:
                self.pinecone_index.upsert(vectors=batch, namespace=PINECONE_NAMESPACE)
                logger.info(f"Upserted batch {i // BATCH_SIZE + 1}/{(len(vectors) + BATCH_SIZE - 1) // BATCH_SIZE}")
            except Exception as e:
                logger.error(f"Failed to upsert batch: {e}")
        
        logger.info(f"Successfully upserted {len(vectors)} vectors to Pinecone")

    def ingest_book(self, pdf_path: str, book_title: Optional[str] = None, author: str = "Unknown") -> Dict:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(pdf_path)
        
        book_id = str(uuid.uuid4())
        total = self._get_total_pages(pdf_path)
        book_title = book_title or os.path.basename(pdf_path)
        logger.info(f"Ingesting {book_title} ({total} pages)")

        # Parse elements
        elements = self._parse_pdf_to_elements(pdf_path)
        
        # Try TOC parsing
        toc_entries = None
        try:
            toc_entries = self.toc_parser.extract_toc(elements)
            if toc_entries:
                logger.info(f"TOC parsed with {len(toc_entries)} entries")
        except Exception as e:
            logger.warning(f"TOCParser failed: {e}")

        # Detect heading candidates
        raw_candidates = self.detect_heading_candidates(pdf_path)

        # Verify with LLM
        verified = self.verifier.verify(raw_candidates)
        filtered = [v for v in verified if (v.get("label") in ("CHAPTER", "SECTION", "SUBSECTION") or v.get("confidence", 0) >= 0.6)]
        logger.info(f"{len(filtered)} candidates passed verification.")

        detected_headings = self._candidates_to_detected_headings(filtered)

        # Build hierarchy
        if toc_entries:
            toc_hierarchy = []
            for te in toc_entries:
                toc_hierarchy.append({"title": te.title, "number": te.number, "page": te.page, "level": te.level})
            root = self.hierarchy_builder.build_from_toc(toc_hierarchy)
        else:
            root = BookNode(title="Root", level=0, source="root")

        if detected_headings:
            root = self.hierarchy_builder.merge_detected_headings(root, detected_headings, elements)

        root = self.hierarchy_builder.add_front_matter(root)
        root.end_page = total
        root.end_element_idx = len(elements) - 1
        self.hierarchy_builder.fill_page_ranges(root, total)
        root = self._smooth_boundaries(root, total)

        # Get chapters and chunk
        chapters = self.hierarchy_builder.get_flat_chapters(root)
        logger.info(f"Built hierarchy with {len(chapters)} chapters.")
        
        chapter_dicts = []
        def node_to_dict(node: BookNode):
            return {
                "title": node.title,
                "number": node.number,
                "level": node.level,
                "start_page": node.start_page,
                "end_page": node.end_page,
                "start_element_idx": node.start_element_idx,
                "end_element_idx": node.end_element_idx,
                "children": [node_to_dict(c) for c in node.children]
            }
        
        for ch in chapters:
            chapter_dicts.append(node_to_dict(ch))
        
        chunks = self.chunker.chunk_chapters(chapter_dicts, elements, book_title, author)
        logger.info(f"Produced {len(chunks)} chunks.")

        # Embed and upsert to Pinecone
        self._embed_and_upsert(chunks, book_id)

        return {
            "title": book_title,
            "author": author,
            "total_pages": total,
            "total_chapters": len(chapters),
            "total_chunks": len(chunks),
            "root": root
        }