import os
import re
import uuid
from typing import List, Optional
from dataclasses import dataclass

import pdfplumber
from sentence_transformers import SentenceTransformer
import tiktoken
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "coding-books"
NAMESPACE = "books_rag"

CHUNK_SIZE_TOKENS = 1500
CHUNK_OVERLAP_TOKENS = 200
BATCH_SIZE = 100
CHUNK_PREVIEW_LEN = 0  # no preview stored per user request

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ENCODING_NAME = "cl100k_base"

LEADING_NUMBER_RE = re.compile(r"^(?P<number>\d+(?:\.\d+)*)\s*[:\-–—]?\s*(?P<title>.+)$", re.I)
CHAPTER_WORD_RE = re.compile(r"^(?:chapter|ch)\s+(\d+)\b[:\s-]*(.*)$", re.I)


@dataclass
class BookMetadata:
    title: str
    author: str = "Unknown"
    source_file: str = ""


@dataclass
class SectionMeta:
    title: str
    level: int
    start_page: int
    end_page: int
    chapter_title: Optional[str]
    chapter_number: Optional[str]
    section_title: Optional[str]
    section_number: Optional[str]
    subsection_title: Optional[str]
    subsection_number: Optional[str]


@dataclass
class Chunk:
    text: str  # used for embedding only, not stored
    book_id: str
    book_title: str
    author: str
    chapter_titles: List[str]
    chapter_numbers: List[Optional[str]]
    section_titles: List[str]
    section_numbers: List[Optional[str]]
    subsection_titles: List[str]
    subsection_numbers: List[Optional[str]]
    page_start: int
    page_end: int
    chunk_index: int
    contains_code: bool


def extract_number_and_title(text: str):
    s = (text or "").strip()
    if not s:
        return None, s
    m = CHAPTER_WORD_RE.match(s)
    if m:
        return m.group(1), (m.group(2) or s).strip()
    m = LEADING_NUMBER_RE.match(s)
    if m:
        return m.group("number"), m.group("title").strip()
    return None, s


def simple_contains_code(text: str) -> bool:
    indicators = [
        r"^\s{4,}\w+",
        r"^\s*def\s+\w+\(",
        r"^\s*class\s+\w+",
        r"```",
        r";\s*$",
        r"\bimport\s+\w+",
        r"\bfrom\s+\w+\s+import\b",
    ]
    for pat in indicators:
        if re.search(pat, text, flags=re.MULTILINE):
            return True
    symbols = sum(text.count(s) for s in "{}[]();=<>:+-*/\\")
    return len(text) > 0 and (symbols / len(text)) > 0.01


def clean_list(lst):
    out = []
    for x in lst or []:
        out.append("" if x is None else str(x))
    return out


class BookIngestionService:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.encoding = tiktoken.get_encoding(ENCODING_NAME)

        if not PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY not set")

        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        existing = []
        try:
            existing = self.pc.list_indexes()
        except Exception:
            existing = []
        if INDEX_NAME not in existing:
            dim = len(self.model.encode(["hello"])[0])
            self.pc.create_index(name=INDEX_NAME, dimension=dim, metric="cosine")
        self.index = self.pc.Index(INDEX_NAME)

    def _line_from_chars(self, chars):
        # group chars by rounded top coordinate to form lines
        lines = {}
        for ch in chars:
            top = round(ch.get("top", 0))
            if top not in lines:
                lines[top] = []
            lines[top].append(ch)
        out = []
        for top in sorted(lines.keys()):
            spans = lines[top]
            text = "".join([s.get("text", "") for s in spans])
            sizes = [s.get("size", 0) for s in spans if s.get("size")]
            fonts = [s.get("fontname", "") for s in spans if s.get("fontname")]
            x0 = min(s.get("x0", 0) for s in spans)
            x1 = max(s.get("x1", 0) for s in spans)
            avg_size = sum(sizes) / len(sizes) if sizes else 0
            out.append({"text": text.strip(), "size": avg_size, "fontnames": fonts, "x0": x0, "x1": x1, "top": top})
        return out

    def _is_monospace_fontname(self, name: str) -> bool:
        if not name:
            return False
        name = name.lower()
        monos = ["courier", "mono", "consolas", "menlo", "monaco", "monospace"]
        return any(m in name for m in monos)

    def extract_headings(self, pdf_path: str):
        headings = []  # list of dict(page, text, size, fontname, x0, x1, top)
        page_texts = []
        with pdfplumber.open(pdf_path) as pdf:
            # collect all line sizes to compute thresholds
            all_sizes = []
            for i, page in enumerate(pdf.pages, start=1):
                chars = page.chars
                lines = self._line_from_chars(chars)
                page_text = page.extract_text() or ""
                page_texts.append(page_text)
                for ln in lines:
                    avg_fontname = ln["fontnames"][0] if ln["fontnames"] else ""
                    if self._is_monospace_fontname(avg_fontname):
                        continue
                    text = ln["text"]
                    if not text or len(text) < 3:
                        continue
                    all_sizes.append(ln["size"])
                    headings.append({
                        "page": i,
                        "text": text,
                        "size": ln["size"],
                        "fontname": avg_fontname,
                        "x0": ln["x0"],
                        "x1": ln["x1"],
                        "top": ln["top"]
                    })
            if not headings:
                return [], page_texts

            median_size = sorted(all_sizes)[max(0, len(all_sizes) // 2)]
            # filter probable headings: font size noticeably larger than median or numeric pattern
            candidates = []
            for h in headings:
                text = h["text"]
                size = h["size"] or 0
                numeric = bool(re.match(r"^(chapter\s+\d+|\d+(\.\d+)*\b)", text.lower()))
                near_top = h["top"] < 0.2 * pdf.pages[h["page"] - 1].height
                centered = abs((h["x0"] + h["x1"]) / 2 - (pdf.pages[h["page"] - 1].width / 2)) < 0.25 * pdf.pages[h["page"] - 1].width
                if numeric or size >= median_size * 1.2 or near_top or centered:
                    candidates.append(h)
            if not candidates:
                return [], page_texts

            # cluster sizes into up to 3 levels
            sizes = sorted({round(c["size"], 2) for c in candidates}, reverse=True)
            sizes = sizes[:3]
            size_to_level = {s: min(i + 1, 3) for i, s in enumerate(sizes)}
            normalized = []
            for c in candidates:
                nearest = min(sizes, key=lambda s: abs(s - round(c["size"], 2)))
                lvl = size_to_level[nearest]
                normalized.append({
                    "page": c["page"],
                    "text": c["text"],
                    "level": lvl,
                    "size": c["size"]
                })
            normalized.sort(key=lambda x: (x["page"], -x["level"]))
            return normalized, page_texts

    def build_sections_from_headings(self, headings, total_pages: int):
        if not headings:
            return [SectionMeta(
                title="Full Book", level=1, start_page=1, end_page=total_pages,
                chapter_title=None, chapter_number=None,
                section_title=None, section_number=None,
                subsection_title=None, subsection_number=None
            )]
        sections: List[SectionMeta] = []
        for i, h in enumerate(headings):
            level = h["level"]
            title = h["text"]
            start_page = h["page"]
            end_page = total_pages
            for j in range(i + 1, len(headings)):
                if headings[j]["level"] <= level:
                    end_page = headings[j]["page"] - 1
                    break
            num, name = extract_number_and_title(title)
            chapter_title = chapter_number = None
            section_title = section_number = None
            subsection_title = subsection_number = None
            if level == 1:
                chapter_title = name
                chapter_number = num
            elif level == 2:
                section_title = name
                section_number = num
                for prev in reversed(headings[:i]):
                    if prev["level"] == 1:
                        pn, pname = extract_number_and_title(prev["text"])
                        chapter_title = pname
                        chapter_number = pn
                        break
            else:
                subsection_title = name
                subsection_number = num
                for prev in reversed(headings[:i]):
                    if prev["level"] == 2 and section_title is None:
                        sn, sname = extract_number_and_title(prev["text"])
                        section_title = sname
                        section_number = sn
                    if prev["level"] == 1 and chapter_title is None:
                        pn, pname = extract_number_and_title(prev["text"])
                        chapter_title = pname
                        chapter_number = pn
            sections.append(SectionMeta(
                title=title, level=level,
                start_page=start_page, end_page=end_page,
                chapter_title=chapter_title, chapter_number=chapter_number,
                section_title=section_title, section_number=section_number,
                subsection_title=subsection_title, subsection_number=subsection_number
            ))
        sections.sort(key=lambda s: s.start_page)
        if sections[0].start_page > 1:
            sections.insert(0, SectionMeta(
                title="Front Matter", level=1,
                start_page=1, end_page=sections[0].start_page - 1,
                chapter_title="Front Matter", chapter_number=None,
                section_title=None, section_number=None,
                subsection_title=None, subsection_number=None
            ))
        for i in range(len(sections) - 1):
            if sections[i].end_page < sections[i + 1].start_page - 1:
                sections[i].end_page = sections[i + 1].start_page - 1
        sections[-1].end_page = max(sections[-1].end_page, total_pages)
        return sections

    def process_pdf(self, pdf_path: str):
        headings, page_texts = self.extract_headings(pdf_path)
        # open to get page count and to build page-level text reliably
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            if not page_texts or len(page_texts) != total_pages:
                page_texts = [p.extract_text() or "" for p in pdf.pages]

        sections = self.build_sections_from_headings(headings, len(page_texts))

        # build per-page token spans
        page_spans = []
        all_tokens = []
        for pnum, page_text in enumerate(page_texts, start=1):
            toks = self.encoding.encode(page_text)
            start = len(all_tokens)
            all_tokens.extend(toks)
            end = len(all_tokens)
            page_spans.append({"page": pnum, "token_start": start, "token_end": end})

        total_tokens = len(all_tokens)
        section_spans = []
        for s in sections:
            start_tok = None
            end_tok = None
            for span in page_spans:
                if span["page"] == s.start_page:
                    start_tok = span["token_start"]
                if span["page"] == s.end_page:
                    end_tok = span["token_end"]
            section_spans.append({
                "section": s,
                "token_start": start_tok or 0,
                "token_end": end_tok or total_tokens
            })

        chunks: List[Chunk] = []
        idx = 0
        step = max(1, CHUNK_SIZE_TOKENS - CHUNK_OVERLAP_TOKENS)
        chunk_idx = 0
        book_id = str(uuid.uuid4())

        while idx < total_tokens:
            window = all_tokens[idx: idx + CHUNK_SIZE_TOKENS]
            if not window:
                break
            chunk_text = self.encoding.decode(window)
            window_start = idx
            window_end = idx + len(window)
            pages = [span["page"] for span in page_spans
                     if not (window_start >= span["token_end"] or window_end <= span["token_start"])]
            page_start = min(pages) if pages else 1
            page_end = max(pages) if pages else 1

            chapters, chapter_nums = [], []
            section_titles, section_nums = [], []
            subsection_titles, subsection_nums = [], []

            for sp in section_spans:
                s = sp["section"]
                if not (window_start >= sp["token_end"] or window_end <= sp["token_start"]):
                    if s.chapter_title:
                        chapters.append(s.chapter_title)
                        chapter_nums.append(s.chapter_number)
                    elif s.level == 1:
                        chapters.append(s.title)
                        chapter_nums.append(s.chapter_number)

                    if s.section_title:
                        section_titles.append(s.section_title)
                        section_nums.append(s.section_number)
                    elif s.level == 2:
                        section_titles.append(s.title)
                        section_nums.append(s.section_number)

                    if s.subsection_title:
                        subsection_titles.append(s.subsection_title)
                        subsection_nums.append(s.subsection_number)

            def dedup(x):
                out = []
                seen = set()
                for i in x:
                    key = i or "__NONE__"
                    if key not in seen:
                        seen.add(key)
                        out.append(i)
                return out

            chunks.append(Chunk(
                text=chunk_text,
                book_id=book_id,
                book_title=os.path.basename(pdf_path),
                author="Unknown",
                chapter_titles=dedup(chapters),
                chapter_numbers=dedup(chapter_nums),
                section_titles=dedup(section_titles),
                section_numbers=dedup(section_nums),
                subsection_titles=dedup(subsection_titles),
                subsection_numbers=dedup(subsection_nums),
                page_start=page_start,
                page_end=page_end,
                chunk_index=chunk_idx,
                contains_code=simple_contains_code(chunk_text)
            ))

            chunk_idx += 1
            idx += step

        return BookMetadata(title=os.path.basename(pdf_path), author="Unknown", source_file=pdf_path), chunks

    def embed_and_upsert(self, chunks: List[Chunk]):
        if not chunks:
            return
        texts = [c.text for c in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True)

        batch = []
        for c, emb in zip(chunks, embeddings):
            md = {
                "book_id": c.book_id,
                "book_title": c.book_title or "",
                "author": c.author or "",
                "chapter_titles": clean_list(c.chapter_titles),
                "chapter_numbers": clean_list(c.chapter_numbers),
                "section_titles": clean_list(c.section_titles),
                "section_numbers": clean_list(c.section_numbers),
                "subsection_titles": clean_list(c.subsection_titles),
                "subsection_numbers": clean_list(c.subsection_numbers),
                "page_start": int(c.page_start),
                "page_end": int(c.page_end),
                "chunk_index": int(c.chunk_index),
                "contains_code": bool(c.contains_code)
            }

            batch.append({
                "id": str(uuid.uuid4()),
                "values": emb.tolist(),
                "metadata": md
            })

            if len(batch) >= BATCH_SIZE:
                self.index.upsert(vectors=batch, namespace=NAMESPACE)
                batch = []

        if batch:
            self.index.upsert(vectors=batch, namespace=NAMESPACE)

    def ingest_book(self, pdf_path: str) -> BookMetadata:
        book_meta, chunks = self.process_pdf(pdf_path)
        self.embed_and_upsert(chunks)
        return book_meta


if __name__ == "__main__":
    svc = BookIngestionService()
    svc.ingest_book("C:/Users/Yuvraj/Downloads/hands-onmachinelearningwithscikit-learnkerasandtensorflow.pdf")

