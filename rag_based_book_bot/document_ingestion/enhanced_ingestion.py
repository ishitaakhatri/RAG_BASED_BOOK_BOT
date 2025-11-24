# enhanced_ingestor_paddle.py
"""
PaddleOCR-based Enhanced Ingestor (Windows-friendly, GPU with CUDA 12.9 support)

Place this file next to:
 - rag_based_book_bot/document_ingestion/ingestion/toc_parser.py
 - rag_based_book_bot/document_ingestion/ingestion/hierarchy_builder.py
 - rag_based_book_bot/document_ingestion/ingestion/chapter_chunker.py

Environment variables:
 - OPENAI_API_KEY (optional) for LLM verifier
 - PINECONE_API_KEY (optional) for upserts
 - LOCAL_LLM_MODEL (optional) for local verifier fallback
 - EMBEDDING_MODEL, ENCODING_NAME as before
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
    from paddleocr import PaddleOCR, draw_ocr
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

# Pinecone optional
try:
    from pinecone import Pinecone
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
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "coding-books")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "books_rag")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
ENCODING_NAME = os.getenv("ENCODING_NAME", "cl100k_base")
VERIFIER_BATCH = int(os.getenv("VERIFIER_BATCH", "16"))
VERIFIER_MAX_TOKENS = int(os.getenv("VERIFIER_MAX_TOKENS", "512"))
USE_CUDA = os.getenv("USE_CUDA", "1") == "1"

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
# PaddleOCR wrapper
# --------------------------
class PaddleLayoutWrapper:
    """
    Uses PaddleOCR doc-parser (PP-Structure) to detect layout + extract text boxes.
    Falls back to pdfplumber or pytesseract if PaddleOCR is missing.
    """

    def __init__(self, use_gpu: bool = True, lang: str = "en"):
        self.use_gpu = use_gpu and USE_CUDA
        self.lang = lang
        self.ocr = None
        if _HAS_PADDLEOCR:
            # PaddleOCR autodetects GPU if installed with GPU wheel
            try:
                # doc-parser includes pp-structure; use det + rec + table/structure
                # 'use_angle_cls' helps with rotated text
                self.ocr = PaddleOCR(use_angle_cls=True, lang=self.lang, use_gpu=self.use_gpu, show_log=False)
                logger.info(f"PaddleOCR initialized (use_gpu={self.use_gpu}).")
            except Exception as e:
                logger.warning(f"PaddleOCR init failed: {e}")
                self.ocr = None
        else:
            logger.info("PaddleOCR not installed; will fallback to pytesseract/pdfplumber.")

    def page_image_to_blocks(self, pil_image: Image.Image) -> List[Dict]:
        """
        Return list of blocks: { 'bbox': [x0,y0,x1,y1], 'text': str, 'score': float, 'type': 'Text' }
        Uses PaddleOCR. If not available, returns [] and caller falls back to text-block extraction.
        """
        if not self.ocr:
            return []
        try:
            # PaddleOCR.ocr returns list of [bbox, (text, conf)] entries per line/block
            result = self.ocr.ocr(np.array(pil_image), det=True, rec=True, cls=True)
            blocks = []
            # result can be nested (list of lines)
            for line in result:
                try:
                    # line: [ [[x1,y1],[x2,y2],...], (text, conf) ]
                    box = line[0]
                    text = line[1][0] if len(line) > 1 else ""
                    conf = float(line[1][1]) if len(line) > 1 else 0.0
                    # convert box polygon to bbox
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
# LLM verifier (same pattern as previous)
# --------------------------
class LLMVerifier:
    def __init__(self, batch_size: int = VERIFIER_BATCH, max_tokens: int = VERIFIER_MAX_TOKENS):
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.mode = "heuristic"
        self.openai_available = _HAS_OPENAI and bool(OPENAI_API_KEY)
        self.local_available = _HAS_TRANSFORMERS and bool(LOCAL_LLM_MODEL)
        if self.openai_available:
            openai.api_key = OPENAI_API_KEY
            self.mode = "openai"
            logger.info("Verifier: OpenAI mode.")
        elif self.local_available:
            try:
                self.local_tokenizer = AutoTokenizer.from_pretrained(LOCAL_LLM_MODEL, use_fast=True)
                self.local_model = AutoModelForCausalLM.from_pretrained(
                    LOCAL_LLM_MODEL,
                    torch_dtype=torch.float16 if torch.cuda.is_available() and USE_CUDA else torch.float32,
                    device_map="auto" if torch.cuda.is_available() and USE_CUDA else None,
                    low_cpu_mem_usage=True
                )
                device = 0 if torch.cuda.is_available() and USE_CUDA else -1
                self.generator = pipeline("text-generation", model=self.local_model, tokenizer=self.local_tokenizer, device=device)
                self.mode = "local"
                logger.info(f"Verifier: local HF model {LOCAL_LLM_MODEL}.")
            except Exception as e:
                logger.warning(f"Local verifier load failed: {e}; falling back to heuristics.")
                self.mode = "heuristic"
        else:
            logger.info("Verifier: heuristics only (no OpenAI/local).")

    def _build_prompt(self, items: List[Dict]) -> str:
        instruct = (
            "Classify each item as one of: CHAPTER, SECTION, SUBSECTION, HEADER, FOOTER, NOT_HEADING. "
            "Return a JSON array of objects with id, label and confidence (0.0-1.0)."
        )
        parts = [instruct]
        for it in items:
            snippet = (it.get("context_snippet") or "").replace("\n", " ")[:300]
            parts.append(json.dumps({
                "id": it["id"],
                "text": it.get("text","")[:200],
                "page": it.get("page"),
                "font_size": it.get("font_size"),
                "is_centered": it.get("is_centered"),
                "space_before": it.get("space_before"),
                "space_after": it.get("space_after"),
                "top_ratio": it.get("top_ratio"),
                "context_snippet": snippet
            }))
        return "\n\n".join(parts)

    def _parse_json_safe(self, s: str):
        try:
            return json.loads(s)
        except Exception:
            m = re.search(r'(\[.*\])', s, flags=re.S)
            if m:
                return json.loads(m.group(1))
        return None

    def _call_openai(self, prompt: str) -> Optional[List[Dict]]:
        try:
            resp = openai.ChatCompletion.create(
                model=os.getenv("OPENAI_VERIFIER_MODEL", "gpt-4o-mini"),
                messages=[{"role":"user","content":prompt}],
                max_tokens=self.max_tokens,
                temperature=0.0
            )
            text = resp["choices"][0]["message"]["content"]
            return self._parse_json_safe(text)
        except Exception as e:
            logger.warning(f"OpenAI verifier failed: {e}")
            return None

    def _call_local(self, prompt: str) -> Optional[List[Dict]]:
        try:
            out = self.generator(prompt, max_length=self.max_tokens, do_sample=False)
            text = out[0].get("generated_text","")
            return self._parse_json_safe(text)
        except Exception as e:
            logger.warning(f"Local verifier failed: {e}")
            return None

    def _heuristic_label(self, it: Dict):
        fs = float(it.get("font_size") or 0)
        top = float(it.get("top_ratio") or 0)
        centered = bool(it.get("is_centered"))
        if fs >= 28 and top < 0.2 and centered:
            return "CHAPTER", 0.9
        if fs >= 18 and it.get("space_before",0) >= 30:
            return "SECTION", 0.8
        if top < 0.12 and fs <= 12 and centered:
            return "HEADER", 0.6
        return "NOT_HEADING", 0.4

    def verify(self, candidates: List[Dict]) -> List[Dict]:
        if not candidates:
            return []
        results = []
        if self.mode == "heuristic":
            for c in candidates:
                lbl, conf = self._heuristic_label(c)
                c["label"] = lbl; c["confidence"] = conf
                results.append(c)
            return results

        for i in range(0, len(candidates), self.batch_size):
            batch = candidates[i:i+self.batch_size]
            prompt = self._build_prompt(batch)
            parsed = None
            if self.mode == "openai":
                parsed = self._call_openai(prompt)
            elif self.mode == "local":
                parsed = self._call_local(prompt)

            if parsed and isinstance(parsed, list):
                mapping = {str(p.get("id")): p for p in parsed if isinstance(p, dict) and p.get("id")}
                for it in batch:
                    pid = str(it["id"])
                    if pid in mapping:
                        r = mapping[pid]
                        it["label"] = r.get("label","NOT_HEADING")
                        it["confidence"] = float(r.get("confidence",0.0))
                        results.append(it)
                    else:
                        lbl, conf = self._heuristic_label(it)
                        it["label"]=lbl; it["confidence"]=conf
                        results.append(it)
            else:
                for it in batch:
                    lbl, conf = self._heuristic_label(it)
                    it["label"]=lbl; it["confidence"]=conf
                    results.append(it)
        return results

# --------------------------
# The main enhanced ingestor
# --------------------------
class EnhancedBookIngestorPaddle:
    def __init__(self, config: Optional[IngestorConfig]=None):
        self.config = config or IngestorConfig()
        self.paddle_wrapper = PaddleLayoutWrapper(use_gpu=USE_CUDA)
        self.hierarchy_builder = HierarchyBuilder()
        self.chunker = ChapterChunker(chunk_size=self.config.chunk_size, overlap=self.config.overlap, encoding_name=self.config.encoding_name)
        self.verifier = LLMVerifier()
        self.tokenizer = tiktoken.get_encoding(self.config.encoding_name)
        # optional integrations
        self.toc_parser = TOCParser() if 'TOCParser' in globals() else None
        self.pinecone_index = None
        if _HAS_PINECONE and PINECONE_API_KEY:
            try:
                self.pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
                self.pinecone_index = self.pinecone_client.Index(PINECONE_INDEX)
            except Exception as e:
                logger.warning(f"Pinecone init error: {e}")

    def _get_total_pages(self, pdf_path: str) -> int:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                return len(pdf.pages)
        except:
            return 0

    def _parse_pdf_to_elements(self, pdf_path: str) -> List[Dict]:
        elements=[]
        with pdfplumber.open(pdf_path) as pdf:
            for pnum,page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                paras = [para.strip() for para in text.split("\n\n") if para.strip()]
                for para in paras:
                    elements.append({"index":len(elements), "type":"NarrativeText","text":para, "page":pnum, "metadata":{"page_number":pnum}})
        return elements

    def _render_page_image(self, pdf_path: str, page_num: int):
        try:
            with pdfplumber.open(pdf_path) as pdf:
                page = pdf.pages[page_num-1]
                try:
                    return page.to_image(resolution=150).original
                except Exception:
                    return None
        except Exception as e:
            logger.warning(f"render page image failed: {e}")
            return None

    def _pdfplumber_text_blocks(self, page) -> List[Dict]:
        blocks=[]
        try:
            words = page.extract_words(use_text_flow=True)
            if not words:
                txt = page.extract_text() or ""
                if txt.strip():
                    blocks.append({"text":txt.strip(),"bbox":[0,0,page.width,page.height],"top":0,"font_size":12,"type":"Text"})
                return blocks
            lines={}
            for w in words:
                top = round(w.get("top",0))
                lines.setdefault(top,[]).append(w)
            for top in sorted(lines.keys()):
                words_line = lines[top]
                text = " ".join(w.get("text","") for w in words_line).strip()
                if not text: continue
                x0=min(w.get("x0",0) for w in words_line)
                x1=max(w.get("x1",0) for w in words_line)
                bbox=[x0,top,x1,top+10]
                blocks.append({"text":text,"bbox":bbox,"top":top,"font_size":None,"type":"Text"})
        except Exception as e:
            logger.warning(f"_pdfplumber_text_blocks error: {e}")
        return blocks

    def _extract_page_blocks(self, pdf_path: str, page_num: int) -> List[Dict]:
        blocks=[]
        page_img = self._render_page_image(pdf_path, page_num)
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_num-1]
            pw, ph = page.width, page.height
            if page_img is not None:
                # try paddle
                vis_blocks = self.paddle_wrapper.page_image_to_blocks(page_img)
                if vis_blocks:
                    for vb in vis_blocks:
                        x0,y0,x1,y1 = vb.get("bbox",[0,0,pw,ph])
                        text = vb.get("text","")
                        blocks.append({"id":str(uuid.uuid4()), "text":text, "page":page_num, "bbox":[x0,y0,x1,y1], "font_size":None, "top":y0, "type":"Text", "score":vb.get("score",0.0)})
                    return blocks
            # fallback to pdfplumber grouping
            txt_blocks = self._pdfplumber_text_blocks(page)
            for tb in txt_blocks:
                blocks.append({"id":str(uuid.uuid4()), "text":tb.get("text",""), "page":page_num, "bbox":tb.get("bbox",[0,0,pw,ph]), "font_size":tb.get("font_size"), "top":tb.get("top",0), "type":tb.get("type","Text")})
        return blocks

    def _is_centered(self, bbox, page_w):
        x0,_,x1,_ = bbox
        center=(x0+x1)/2.0
        return abs(center - page_w/2.0) < (page_w * 0.18)

    def _estimate_font_from_text(self, text:str):
        L=len(text)
        if L<30: return 22.0
        if L<80: return 16.0
        return 12.0

    def detect_heading_candidates(self, pdf_path:str)->List[Dict]:
        candidates=[]
        total=self._get_total_pages(pdf_path)
        for p in range(1,total+1):
            try:
                blocks = self._extract_page_blocks(pdf_path,p)
                if not blocks: continue
                sorted_blocks = sorted(blocks, key=lambda b: b.get("top",0))
                with pdfplumber.open(pdf_path) as pdf:
                    page = pdf.pages[p-1]
                    pw,ph = page.width, page.height
                for idx,b in enumerate(sorted_blocks):
                    text=(b.get("text") or "").strip()
                    if not text or len(text)<2: continue
                    bbox=b.get("bbox",[0,0,pw,ph])
                    top=b.get("top",0)
                    top_ratio = top/ph if ph else 0.0
                    is_centered=self._is_centered(bbox,pw)
                    prev_top = sorted_blocks[idx-1].get("top") if idx>0 else None
                    next_top = sorted_blocks[idx+1].get("top") if idx+1<len(sorted_blocks) else None
                    space_before = (top - prev_top) if prev_top is not None else 0
                    space_after = ( (next_top - (bbox[3] if len(bbox)>3 else top)) if next_top is not None else 0 )
                    font_size = b.get("font_size") or self._estimate_font_from_text(text)
                    snippet = text if len(text)<400 else text[:400]
                    candidates.append({"id":b.get("id") or str(uuid.uuid4()), "text":text, "page":p, "bbox":bbox, "font_size":font_size, "top_ratio":top_ratio, "is_centered":is_centered, "space_before":space_before, "space_after":space_after, "context_snippet":snippet, "source":b.get("type","Text")})
            except Exception as e:
                logger.warning(f"candidate detect failed page {p}: {e}")
        logger.info(f"Detected {len(candidates)} raw candidates.")
        return candidates

    def _candidates_to_detected_headings(self, candidates:List[Dict])->List[Dict]:
        out=[]
        for c in candidates:
            num=None
            for pat in CHAPTER_PATTERNS:
                m=pat.match(c["text"])
                if m:
                    num=m.group(1)
                    break
            guessed_level=1 if (c.get("font_size") or 0)>=22 else 2
            out.append({"page":c.get("page"), "text":c.get("text"), "level":guessed_level, "num":num, "index":None, "confidence":0.4, "source":c.get("source","vision")})
        return out

    def _smooth_boundaries(self, root:BookNode, total_pages:int)->BookNode:
        if not root or not root.children: return root
        root.children.sort(key=lambda n: (n.start_page or 0, n.start_element_idx or 0))
        for i,child in enumerate(root.children):
            if child.start_page is None:
                child.start_page = 1
            if i < len(root.children)-1:
                nextc = root.children[i+1]
                child.end_page = (nextc.start_page - 1) if nextc.start_page else child.end_page or total_pages
            else:
                child.end_page = child.end_page or total_pages
            self.hierarchy_builder._fill_ranges_recursive(child, total_pages)
        return root

    def ingest_book(self, pdf_path:str, book_title:Optional[str]=None, author:str="Unknown")->Dict:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(pdf_path)
        total=self._get_total_pages(pdf_path)
        book_title = book_title or os.path.basename(pdf_path)
        logger.info(f"Ingesting {book_title} ({total} pages)")

        # pre-parse elements
        elements = self._parse_pdf_to_elements(pdf_path)
        toc_entries=None
        if self.toc_parser:
            try:
                toc_entries = self.toc_parser.extract_toc(elements)
                if toc_entries:
                    logger.info(f"TOC parsed with {len(toc_entries)} entries")
            except Exception as e:
                logger.warning(f"TOCParser failed: {e}")

        # detect candidates
        raw_candidates = self.detect_heading_candidates(pdf_path)

        # verify with LLM
        verified = self.verifier.verify(raw_candidates)
        filtered = [v for v in verified if (v.get("label") in ("CHAPTER","SECTION","SUBSECTION") or v.get("confidence",0)>=0.6)]
        logger.info(f"{len(filtered)} candidates passed verification.")

        detected_headings = self._candidates_to_detected_headings(filtered)

        # build hierarchy
        if toc_entries:
            toc_hierarchy=[]
            for te in toc_entries:
                toc_hierarchy.append({"title":te.title,"number":te.number,"page":te.page,"level":te.level})
            root = self.hierarchy_builder.build_from_toc(toc_hierarchy)
        else:
            root = BookNode(title="Root", level=0, source="root")

        if detected_headings:
            root = self.hierarchy_builder.merge_detected_headings(root, detected_headings, elements)

        root = self.hierarchy_builder.add_front_matter(root)
        root.end_page = total
        root.end_element_idx = len(elements)-1
        self.hierarchy_builder.fill_page_ranges(root, total)
        root = self._smooth_boundaries(root, total)

        # get chapters and chunk
        chapters = self.hierarchy_builder.get_flat_chapters(root)
        logger.info(f"Built hierarchy with {len(chapters)} chapters.")
        chapter_dicts = []
        def node_to_dict(node:BookNode):
            return {"title":node.title,"number":node.number,"level":node.level,"start_page":node.start_page,"end_page":node.end_page,"start_element_idx":node.start_element_idx,"end_element_idx":node.end_element_idx,"children":[node_to_dict(c) for c in node.children]}
        for ch in chapters:
            chapter_dicts.append(node_to_dict(ch))
        chunks = self.chunker.chunk_chapters(chapter_dicts, elements, book_title, author)
        logger.info(f"Produced {len(chunks)} chunks.")

        # Placeholder: embed & upsert
        if self.pinecone_index:
            logger.info("Pinecone upsert placeholder - implement embeddings.")
        else:
            logger.info("Pinecone not configured - skipping upsert.")

        return {"title":book_title, "author":author, "total_pages":total, "total_chapters":len(chapters), "total_chunks":len(chunks), "root":root}
