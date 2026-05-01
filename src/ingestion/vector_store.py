"""
ingestion/vector_store.py

ChromaDB integration with:
- FREE local embeddings via sentence-transformers (no API key needed)
- Batched upsert (avoids OOM on large ingestion)
- De-duplication by chunk_id
"""
from __future__ import annotations

import time
from typing import Iterator

import chromadb
from chromadb.config import Settings as ChromaSettings
from loguru import logger
from sentence_transformers import SentenceTransformer

from src.config import settings
from src.ingestion.chunker import DocumentChunk

# Free local embedding model — downloads once (~90MB), then cached on disk.
# all-MiniLM-L6-v2 is fast, small, and works great for semantic search.
_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
_embed_model: SentenceTransformer | None = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        import torch
        # Force CPU — avoids CUDA library conflicts on systems with GPU/driver mismatches
        device = "cpu"
        logger.info(f"Loading local embedding model '{_EMBED_MODEL_NAME}' on {device}...")
        _embed_model = SentenceTransformer(_EMBED_MODEL_NAME, device=device)
        logger.info("Embedding model loaded.")
    return _embed_model


class VectorStore:
    """
    Thin wrapper around ChromaDB using FREE local sentence-transformer embeddings.
    No OpenAI API key required for ingestion.
    """

    def __init__(self) -> None:
        self._client = chromadb.PersistentClient(
            path=str(settings.chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"ChromaDB ready — collection '{settings.chroma_collection_name}' "
            f"({self._collection.count()} existing chunks)"
        )

    @property
    def collection(self):
        return self._collection

    def count(self) -> int:
        return self._collection.count()

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed texts locally — free, no API call."""
        model = _get_embed_model()
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings.tolist()

    def upsert_chunks(self, chunks: Iterator[DocumentChunk]) -> int:
        """
        Batch-upsert chunks into ChromaDB using local embeddings.
        Returns count of newly added chunks.
        """
        batch_texts: list[str] = []
        batch_ids: list[str] = []
        batch_meta: list[dict] = []
        added = 0

        def _flush():
            nonlocal added
            if not batch_ids:
                return
            embeddings = self._embed_batch(batch_texts)
            self._collection.upsert(
                ids=batch_ids,
                embeddings=embeddings,
                documents=batch_texts,
                metadatas=batch_meta,
            )
            added += len(batch_ids)
            logger.debug(f"Upserted {len(batch_ids)} chunks (running total: {added})")
            batch_texts.clear()
            batch_ids.clear()
            batch_meta.clear()

        for chunk in chunks:
            batch_texts.append(chunk.text)
            batch_ids.append(chunk.chunk_id)
            batch_meta.append(chunk.metadata)

            if len(batch_ids) >= settings.batch_size:
                _flush()

        _flush()  # trailing batch
        logger.success(f"Upsert complete — {added} new/updated chunks")
        return added

    def delete_document(self, doc_id: str) -> int:
        """Remove all chunks belonging to a document."""
        results = self._collection.get(where={"doc_id": doc_id})
        ids = results["ids"]
        if ids:
            self._collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} chunks for doc_id={doc_id}")
        return len(ids)

    def list_documents(self) -> list[dict]:
        """Return one summary record per unique document."""
        results = self._collection.get(include=["metadatas"])
        seen: dict[str, dict] = {}
        for meta in results["metadatas"]:
            doc_id = meta.get("doc_id", "unknown")
            if doc_id not in seen:
                seen[doc_id] = {
                    "doc_id": doc_id,
                    "filename": meta.get("filename", ""),
                    "total_pages": meta.get("total_pages", 0),
                    "title": meta.get("title", ""),
                }
        return list(seen.values())

    def query_dense(self, query_embedding: list[float], top_k: int) -> list[dict]:
        """Pure dense (cosine) retrieval."""
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count() or 1),
            include=["documents", "metadatas", "distances"],
        )
        return [
            {
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1 - results["distances"][0][i],
            }
            for i in range(len(results["ids"][0]))
        ]

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string for retrieval."""
        return self._embed_batch([query])[0]
