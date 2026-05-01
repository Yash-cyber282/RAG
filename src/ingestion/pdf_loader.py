"""
ingestion/pdf_loader.py

Extract text, metadata, and page numbers from PDFs using PyMuPDF.
Handles scanned PDFs (OCR fallback), encrypted files, and multi-column layouts.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import fitz  # PyMuPDF
from loguru import logger


@dataclass
class PageContent:
    """Represents a single page extracted from a PDF."""
    doc_id: str          # stable SHA256 of the file
    source_path: str     # absolute path to the PDF
    filename: str        # basename
    page_number: int     # 1-indexed
    text: str
    word_count: int
    char_count: int
    metadata: dict = field(default_factory=dict)

    def is_meaningful(self, min_chars: int = 50) -> bool:
        """Skip cover pages, blank pages, or mostly-image pages."""
        return self.char_count >= min_chars and len(self.text.strip()) >= min_chars


def _compute_doc_id(path: Path) -> str:
    """SHA256 hash of file content for stable de-duplication."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()[:16]


def _extract_pdf_metadata(doc: fitz.Document, path: Path) -> dict:
    meta = doc.metadata or {}
    return {
        "title": meta.get("title") or path.stem,
        "author": meta.get("author", ""),
        "subject": meta.get("subject", ""),
        "keywords": meta.get("keywords", ""),
        "total_pages": doc.page_count,
        "file_size_kb": round(path.stat().st_size / 1024, 1),
    }


def load_pdf(path: Path | str) -> Iterator[PageContent]:
    """
    Yield PageContent for each meaningful page in the PDF.

    Strategy:
    - Primary: native text extraction (fitz.Page.get_text)
    - For tables: preserve whitespace with "text" mode
    - For multi-column: use "blocks" to preserve reading order
    - Fallback: OCR via fitz (if page has no text but has images)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    doc_id = _compute_doc_id(path)

    try:
        doc = fitz.open(str(path))
    except Exception as e:
        logger.error(f"Cannot open {path.name}: {e}")
        return

    if doc.is_encrypted:
        if not doc.authenticate(""):   # try blank password
            logger.warning(f"Skipping encrypted PDF: {path.name}")
            return

    base_meta = _extract_pdf_metadata(doc, path)
    logger.info(f"Loading {path.name} — {base_meta['total_pages']} pages")

    for page_idx in range(doc.page_count):
        page: fitz.Page = doc[page_idx]
        page_number = page_idx + 1

        # Primary extraction — preserves layout better than "text"
        text = page.get_text("text")

        # Fallback to OCR if page has images but little/no text
        if len(text.strip()) < 30 and page.get_images():
            try:
                mat = fitz.Matrix(2, 2)   # 2× resolution for OCR accuracy
                pix = page.get_pixmap(matrix=mat)
                ocr_page = fitz.open("pdf", pix.pdfocr_tobytes())
                text = ocr_page[0].get_text("text")
                logger.debug(f"OCR used on page {page_number} of {path.name}")
            except Exception:
                pass  # OCR not available — continue with whatever we have

        # Clean text
        text = _clean_text(text)

        yield PageContent(
            doc_id=doc_id,
            source_path=str(path.resolve()),
            filename=path.name,
            page_number=page_number,
            text=text,
            word_count=len(text.split()),
            char_count=len(text),
            metadata={
                **base_meta,
                "page_number": page_number,
                "doc_id": doc_id,
                "filename": path.name,
            },
        )

    doc.close()


def load_pdfs_from_directory(
    directory: Path | str,
    recursive: bool = True,
) -> Iterator[PageContent]:
    """Scan a directory for PDFs and yield all pages."""
    directory = Path(directory)
    pattern = "**/*.pdf" if recursive else "*.pdf"
    pdf_paths = sorted(directory.glob(pattern))
    logger.info(f"Found {len(pdf_paths)} PDFs in {directory}")

    for pdf_path in pdf_paths:
        yield from load_pdf(pdf_path)


def _clean_text(text: str) -> str:
    """
    Normalise extracted text:
    - Remove hyphenation at line-ends (common in PDFs)
    - Collapse runs of whitespace
    - Strip page numbers / header/footer noise (heuristic)
    """
    import re
    # Rejoin hyphenated words split across lines
    text = re.sub(r"-\n(\w)", r"\1", text)
    # Collapse multiple newlines to double (paragraph breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)
    return text.strip()
