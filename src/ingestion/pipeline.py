"""
ingestion/pipeline.py

Top-level ingestion orchestrator. Glues loader → chunker → vector_store.
Designed for both CLI use and import from the Streamlit app.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from loguru import logger
from tqdm import tqdm

from src.config import settings
from src.ingestion.chunker import SemanticChunker
from src.ingestion.pdf_loader import load_pdf, load_pdfs_from_directory
from src.ingestion.vector_store import VectorStore


class IngestionPipeline:
    def __init__(self, progress_callback: Callable[[str, int, int], None] | None = None):
        """
        progress_callback(message, current, total) — optional hook for Streamlit progress bars.
        """
        self.chunker = SemanticChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        self.vector_store = VectorStore()
        self._progress = progress_callback or (lambda msg, cur, tot: None)

    def ingest_pdf(self, pdf_path: Path | str) -> dict:
        """Ingest a single PDF. Returns summary dict."""
        pdf_path = Path(pdf_path)
        logger.info(f"Ingesting: {pdf_path.name}")

        pages = list(load_pdf(pdf_path))
        meaningful = [p for p in pages if p.is_meaningful()]
        self._progress(f"Loaded {len(meaningful)} pages from {pdf_path.name}", 0, 3)

        chunks_iter = self.chunker.chunk_pages(iter(meaningful))
        chunks = list(chunks_iter)
        self._progress(f"Created {len(chunks)} chunks", 1, 3)

        added = self.vector_store.upsert_chunks(iter(chunks))
        self._progress(f"Stored {added} chunks in ChromaDB", 2, 3)

        return {
            "filename": pdf_path.name,
            "pages_loaded": len(meaningful),
            "chunks_created": len(chunks),
            "chunks_stored": added,
            "doc_id": chunks[0].doc_id if chunks else None,
        }

    def ingest_directory(
        self, directory: Path | str, recursive: bool = True
    ) -> list[dict]:
        """Ingest all PDFs in a directory. Returns per-file summaries."""
        directory = Path(directory)
        pdfs = sorted(directory.glob("**/*.pdf" if recursive else "*.pdf"))
        logger.info(f"Starting batch ingestion of {len(pdfs)} PDFs")

        results = []
        for i, pdf in enumerate(tqdm(pdfs, desc="Ingesting PDFs")):
            self._progress(f"Processing {pdf.name}", i, len(pdfs))
            try:
                result = self.ingest_pdf(pdf)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed on {pdf.name}: {e}")
                results.append({"filename": pdf.name, "error": str(e)})

        total_chunks = sum(r.get("chunks_stored", 0) for r in results)
        logger.success(
            f"Batch ingestion done — {len(pdfs)} files, {total_chunks} chunks stored"
        )
        return results
