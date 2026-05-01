"""
generation/query_pipeline.py

Orchestrates the full retrieval → rerank → generate pipeline.
Single entry point for both Streamlit and CLI.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Generator

from loguru import logger

from src.generation.generator import (
    Citation,
    ConversationMemory,
    GeneratedAnswer,
    RAGGenerator,
)
from src.ingestion.vector_store import VectorStore
from src.retrieval.hybrid_retriever import HybridRetriever, RetrievedChunk
from src.retrieval.reranker import Reranker


@dataclass
class QueryResult:
    answer: str
    citations: list[Citation]
    retrieved_chunks: list[RetrievedChunk]
    query: str


class QueryPipeline:
    """
    Full RAG query pipeline:
    1. Hybrid retrieval (dense + BM25 + RRF)
    2. Cross-encoder reranking
    3. LLM response generation (streaming or batch)
    """

    def __init__(self) -> None:
        self._vector_store = VectorStore()
        self._retriever = HybridRetriever(self._vector_store)
        self._reranker = Reranker()
        self._generator = RAGGenerator()
        self.memory = ConversationMemory(max_turns=6)

    def query(
        self,
        question: str,
        use_cache: bool = True,
    ) -> QueryResult:
        """Non-streaming query. Returns complete answer."""
        logger.info(f"Query: {question!r}")

        # Step 1: Hybrid retrieval
        raw_chunks = self._retriever.retrieve(question)
        logger.debug(f"Retrieved {len(raw_chunks)} candidates")

        # Step 2: Rerank
        ranked_chunks = self._reranker.rerank(question, raw_chunks)
        logger.debug(f"Reranked to top {len(ranked_chunks)}")

        # Step 3: Generate
        result: GeneratedAnswer = self._generator.generate(
            question, ranked_chunks, self.memory, use_cache=use_cache
        )

        return QueryResult(
            answer=result.answer,
            citations=result.citations,
            retrieved_chunks=ranked_chunks,
            query=question,
        )

    def query_stream(
        self,
        question: str,
    ) -> tuple[Generator[str, None, None], list[Citation], list[RetrievedChunk]]:
        """
        Streaming query.
        Returns (token_generator, citations, chunks) — citations are ready
        immediately so the UI can render them before streaming completes.
        """
        logger.info(f"Stream query: {question!r}")

        raw_chunks = self._retriever.retrieve(question)
        ranked_chunks = self._reranker.rerank(question, raw_chunks)
        token_stream, citations = self._generator.generate_stream(
            question, ranked_chunks, self.memory
        )
        return token_stream, citations, ranked_chunks

    def reset_memory(self) -> None:
        self.memory.clear()
        logger.info("Conversation memory cleared")
