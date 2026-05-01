"""
retrieval/reranker.py

Cross-encoder reranker using a local sentence-transformers model.
Reranking runs entirely locally — no extra API cost.

Why rerank?
- First-stage retrieval optimises for recall (get all relevant chunks in top-K).
- Reranker optimises for precision (push the best chunk to the top).
- Cross-encoders see both query and passage together, allowing
  deep interaction that bi-encoder embeddings miss.
"""
from __future__ import annotations

import functools

from loguru import logger
from sentence_transformers import CrossEncoder

from src.config import settings
from src.retrieval.hybrid_retriever import RetrievedChunk


@functools.lru_cache(maxsize=1)
def _load_reranker(model_name: str) -> CrossEncoder:
    """Load and cache the cross-encoder model (slow first call, free thereafter)."""
    logger.info(f"Loading reranker model: {model_name}")
    return CrossEncoder(model_name)


class Reranker:
    """
    Reranks retrieved chunks using a cross-encoder model.
    Falls back gracefully if the model is unavailable.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.reranker_model
        self._model: CrossEncoder | None = None

    def _get_model(self) -> CrossEncoder | None:
        try:
            return _load_reranker(self.model_name)
        except Exception as e:
            logger.warning(f"Reranker unavailable ({e}); using fusion scores only")
            return None

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """
        Score (query, chunk_text) pairs with the cross-encoder and re-sort.
        Returns top_k chunks sorted by reranker score descending.
        """
        top_k = top_k or settings.top_k_rerank

        if not chunks:
            return []

        model = self._get_model()
        if model is None:
            # Graceful degradation: keep fusion ranking
            return chunks[:top_k]

        pairs = [(query, chunk.text) for chunk in chunks]
        scores = model.predict(pairs, show_progress_bar=False)

        for chunk, score in zip(chunks, scores):
            chunk.fusion_score = float(score)  # overwrite with reranker score

        reranked = sorted(chunks, key=lambda c: c.fusion_score, reverse=True)
        logger.debug(
            f"Reranked {len(chunks)} → {top_k} chunks; "
            f"top score={reranked[0].fusion_score:.3f}"
        )
        return reranked[:top_k]
