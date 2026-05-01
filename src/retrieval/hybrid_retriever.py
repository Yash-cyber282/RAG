"""
retrieval/hybrid_retriever.py

Hybrid search = dense (cosine) + sparse (BM25) + reciprocal rank fusion.
Uses FREE local sentence-transformer embeddings for query encoding.
No OpenAI API key required.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger
from rank_bm25 import BM25Okapi

from src.config import settings
from src.ingestion.vector_store import VectorStore


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    filename: str
    page_number: int
    doc_id: str
    dense_score: float = 0.0
    bm25_score: float = 0.0
    fusion_score: float = 0.0
    metadata: dict = None  # type: ignore

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @property
    def source_label(self) -> str:
        return f"{self.filename} — p.{self.page_number}"


class HybridRetriever:
    """
    Retriever combining:
    1. Dense retrieval from ChromaDB (local embeddings — FREE)
    2. Sparse BM25 over the same corpus (in-memory)
    3. Reciprocal Rank Fusion (RRF) to merge results
    """

    def __init__(self, vector_store: VectorStore) -> None:
        self.vector_store = vector_store
        self._bm25: BM25Okapi | None = None
        self._bm25_corpus: list[dict] = []

    def _build_bm25_index(self) -> None:
        logger.info("Building BM25 index from ChromaDB corpus…")
        result = self.vector_store.collection.get(include=["documents", "metadatas"])
        self._bm25_corpus = [
            {
                "chunk_id": result["ids"][i],
                "text": result["documents"][i],
                "metadata": result["metadatas"][i],
            }
            for i in range(len(result["ids"]))
        ]
        tokenised = [doc["text"].lower().split() for doc in self._bm25_corpus]
        self._bm25 = BM25Okapi(tokenised)
        logger.success(f"BM25 index built over {len(self._bm25_corpus)} chunks")

    def _embed_query(self, query: str) -> list[float]:
        """Embed query using the same free local model used during ingestion."""
        return self.vector_store.embed_query(query)

    def _reciprocal_rank_fusion(
        self,
        dense_results: list[dict],
        bm25_results: list[dict],
        k: int = 60,
    ) -> list[dict]:
        scores: dict[str, float] = {}
        chunk_map: dict[str, dict] = {}

        for rank, item in enumerate(dense_results):
            cid = item["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1 / (k + rank + 1)
            item["dense_score"] = item.get("score", 0.0)
            chunk_map[cid] = item

        for rank, item in enumerate(bm25_results):
            cid = item["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1 / (k + rank + 1)
            if cid in chunk_map:
                chunk_map[cid]["bm25_score"] = item.get("score", 0.0)
            else:
                item["bm25_score"] = item.get("score", 0.0)
                chunk_map[cid] = item

        for cid, fused in scores.items():
            chunk_map[cid]["fusion_score"] = fused

        return sorted(chunk_map.values(), key=lambda x: x["fusion_score"], reverse=True)

    def retrieve(
        self,
        query: str,
        top_k_dense: int | None = None,
        top_k_bm25: int | None = None,
        top_k_final: int | None = None,
    ) -> list[RetrievedChunk]:
        top_k_dense = top_k_dense or settings.top_k_dense
        top_k_bm25 = top_k_bm25 or settings.top_k_bm25
        top_k_final = top_k_final or (top_k_dense + top_k_bm25)

        # Dense retrieval using local embeddings
        query_embedding = self._embed_query(query)
        dense_results = self.vector_store.query_dense(query_embedding, top_k_dense)

        # BM25 retrieval
        if self._bm25 is None:
            self._build_bm25_index()

        tokenised_query = query.lower().split()
        bm25_raw_scores: Any = self._bm25.get_scores(tokenised_query)
        top_bm25_idx = bm25_raw_scores.argsort()[::-1][:top_k_bm25]
        bm25_results = [
            {
                "chunk_id": self._bm25_corpus[i]["chunk_id"],
                "text": self._bm25_corpus[i]["text"],
                "metadata": self._bm25_corpus[i]["metadata"],
                "score": float(bm25_raw_scores[i]),
            }
            for i in top_bm25_idx
        ]

        fused = self._reciprocal_rank_fusion(dense_results, bm25_results)[:top_k_final]

        return [
            RetrievedChunk(
                chunk_id=item["chunk_id"],
                text=item["text"],
                filename=item["metadata"].get("filename", "unknown"),
                page_number=item["metadata"].get("page_number", 0),
                doc_id=item["metadata"].get("doc_id", ""),
                dense_score=item.get("dense_score", 0.0),
                bm25_score=item.get("bm25_score", 0.0),
                fusion_score=item.get("fusion_score", 0.0),
                metadata=item["metadata"],
            )
            for item in fused
        ]
