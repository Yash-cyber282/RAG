"""
ingestion/chunker.py

Token-aware sentence-boundary chunking — pure Python, no llama_index needed.
Each chunk preserves full source metadata for citation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator

from loguru import logger

from src.ingestion.pdf_loader import PageContent


@dataclass
class DocumentChunk:
    """One retrievable unit of text with full provenance."""
    chunk_id: str
    doc_id: str
    source_path: str
    filename: str
    page_number: int
    chunk_index: int
    text: str
    token_count: int
    metadata: dict = field(default_factory=dict)


def _make_chunk_id(doc_id: str, page: int, idx: int) -> str:
    return f"{doc_id}_p{page:04d}_c{idx:04d}"


def _split_sentences(text: str) -> list[str]:
    """
    Simple but robust sentence splitter — no external dependencies.
    Splits on '.', '!', '?' followed by whitespace + capital letter,
    and on double newlines (paragraph breaks).
    """
    # Normalise line breaks
    text = re.sub(r"\r\n|\r", "\n", text)

    # Split on paragraph breaks first
    paragraphs = re.split(r"\n{2,}", text)

    sentences: list[str] = []
    # Sentence-ending pattern: ., !, ? followed by space + uppercase or end
    sent_re = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        parts = sent_re.split(para)
        sentences.extend(p.strip() for p in parts if p.strip())

    return sentences


def _word_count(text: str) -> int:
    return len(text.split())


def _split_into_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Greedily pack sentences into chunks of ~chunk_size words,
    with a word-level overlap between consecutive chunks.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for sent in sentences:
        sent_words = _word_count(sent)

        # If a single sentence exceeds chunk_size, hard-split it by words
        if sent_words > chunk_size:
            words = sent.split()
            for start in range(0, len(words), chunk_size - chunk_overlap):
                piece = " ".join(words[start: start + chunk_size])
                if piece:
                    chunks.append(piece)
            continue

        if current_words + sent_words > chunk_size and current:
            chunks.append(" ".join(current))
            # Carry-over overlap: keep sentences from the end until we have
            # ~chunk_overlap words worth of context
            overlap_words = 0
            overlap_sents: list[str] = []
            for s in reversed(current):
                w = _word_count(s)
                if overlap_words + w > chunk_overlap:
                    break
                overlap_sents.insert(0, s)
                overlap_words += w
            current = overlap_sents
            current_words = overlap_words

        current.append(sent)
        current_words += sent_words

    if current:
        chunks.append(" ".join(current))

    return [c for c in chunks if c.strip()]


class SemanticChunker:
    """
    Splits PageContent objects into overlapping word-count chunks
    that respect sentence boundaries.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_page(
        self, page: PageContent, global_chunk_counter: list[int] | None = None
    ) -> list[DocumentChunk]:
        """Split a single page into chunks. Returns empty list for blank pages."""
        if not page.is_meaningful():
            return []

        raw_chunks = _split_into_chunks(page.text, self.chunk_size, self.chunk_overlap)

        chunks = []
        for i, text in enumerate(raw_chunks):
            text = text.strip()
            if not text:
                continue

            token_count = _word_count(text)
            chunk_id = _make_chunk_id(page.doc_id, page.page_number, i)

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=page.doc_id,
                    source_path=page.source_path,
                    filename=page.filename,
                    page_number=page.page_number,
                    chunk_index=i,
                    text=text,
                    token_count=token_count,
                    metadata={
                        **page.metadata,
                        "chunk_id": chunk_id,
                        "chunk_index": i,
                        "token_count": token_count,
                    },
                )
            )

        return chunks

    def chunk_pages(self, pages: Iterator[PageContent]) -> Iterator[DocumentChunk]:
        """Yield chunks from a stream of pages."""
        doc_chunk_counts: dict[str, int] = {}
        total = 0
        for page in pages:
            for chunk in self.chunk_page(page):
                doc_chunk_counts[chunk.doc_id] = (
                    doc_chunk_counts.get(chunk.doc_id, 0) + 1
                )
                total += 1
                yield chunk

        logger.info(
            f"Chunking complete — {total} chunks from {len(doc_chunk_counts)} documents"
        )
