"""
ingestion/chunker.py

Token-aware semantic chunking using LlamaIndex's SentenceSplitter.
Each chunk preserves full source metadata for citation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from loguru import logger

from src.ingestion.pdf_loader import PageContent


@dataclass
class DocumentChunk:
    """One retrievable unit of text with full provenance."""
    chunk_id: str          # globally unique
    doc_id: str            # parent document
    source_path: str
    filename: str
    page_number: int
    chunk_index: int       # position within the document
    text: str
    token_count: int
    metadata: dict = field(default_factory=dict)


def _make_chunk_id(doc_id: str, page: int, idx: int) -> str:
    return f"{doc_id}_p{page:04d}_c{idx:04d}"


class SemanticChunker:
    """
    Splits PageContent objects into overlapping token chunks.

    Strategy:
    - SentenceSplitter respects sentence boundaries → fewer mid-sentence cuts
    - chunk_size / chunk_overlap configurable via settings
    - Each chunk tagged with page, document, and absolute chunk index
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            paragraph_separator="\n\n",
        )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_page(
        self, page: PageContent, global_chunk_counter: list[int] | None = None
    ) -> list[DocumentChunk]:
        """Split a single page into chunks. Returns empty list for blank pages."""
        if not page.is_meaningful():
            return []

        node = TextNode(text=page.text, metadata=page.metadata)
        sub_nodes: list[TextNode] = self.splitter.get_nodes_from_documents([node])  # type: ignore

        chunks = []
        for i, sub in enumerate(sub_nodes):
            text = sub.get_content().strip()
            if not text:
                continue

            token_count = len(text.split())  # approximate; use tiktoken for exact
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
