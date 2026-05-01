"""
generation/generator.py

Fully LOCAL LLM using HuggingFace Transformers pipeline.
No Ollama. No API key. No internet after first model download.

Default model: google/flan-t5-base (~250MB, CPU-friendly)
Swap via HFLOCAL_MODEL env var for larger models if you have GPU.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Generator, Iterator

import diskcache
from loguru import logger

from src.config import settings
from src.retrieval.hybrid_retriever import RetrievedChunk


# ─────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────

@dataclass
class Citation:
    filename: str
    page_number: int
    excerpt: str
    chunk_id: str
    score: float

    @property
    def label(self) -> str:
        return f"{self.filename} (p. {self.page_number})"


@dataclass
class GeneratedAnswer:
    answer: str
    citations: list[Citation]
    query: str
    model: str
    usage: dict = field(default_factory=dict)


# ─────────────────────────────────────────────
# System prompt / prompt builder
# ─────────────────────────────────────────────

def _build_prompt(query: str, chunks: list[RetrievedChunk], history: list[dict]) -> str:
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[{i}] Source: {chunk.filename}, Page {chunk.page_number}\n{chunk.text}"
        )
    context = "\n---\n".join(context_parts)

    history_str = ""
    if history:
        turns = []
        for msg in history:
            role = "User" if msg["role"] == "user" else "Assistant"
            turns.append(f"{role}: {msg['content']}")
        history_str = "\n".join(turns) + "\n\n"

    return (
        f"{history_str}"
        f"You are a document analyst. Answer ONLY using the context below. "
        f"Cite sources as [Source: filename, p.N].\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )


def _extract_citations(chunks: list[RetrievedChunk]) -> list[Citation]:
    return [
        Citation(
            filename=c.filename,
            page_number=c.page_number,
            excerpt=c.text[:200].strip() + ("…" if len(c.text) > 200 else ""),
            chunk_id=c.chunk_id,
            score=round(c.fusion_score, 4),
        )
        for c in chunks
    ]


# ─────────────────────────────────────────────
# Conversation memory
# ─────────────────────────────────────────────

class ConversationMemory:
    def __init__(self, max_turns: int = 4) -> None:
        self.max_turns = max_turns
        self.history: list[dict] = []

    def add(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-(self.max_turns * 2):]

    def as_messages(self) -> list[dict]:
        return list(self.history)

    def clear(self) -> None:
        self.history.clear()


# ─────────────────────────────────────────────
# Model loader (cached in memory)
# ─────────────────────────────────────────────

_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    from transformers import pipeline as hf_pipeline
    import torch

    model_name = settings.hf_model
    logger.info(f"Loading HuggingFace model: {model_name} (first run downloads it)")

    device = 0 if torch.cuda.is_available() else -1  # GPU if available, else CPU
    device_label = "GPU" if device == 0 else "CPU"
    logger.info(f"Running on {device_label}")

    # Detect model type for correct task
    name_lower = model_name.lower()
    if any(x in name_lower for x in ["t5", "bart", "pegasus"]):
        task = "text2text-generation"
    else:
        task = "text-generation"

    _pipeline = hf_pipeline(
        task,
        model=model_name,
        device=device,
        max_new_tokens=512,
        do_sample=False,          # deterministic for RAG
        temperature=None,
        top_p=None,
    )
    logger.success(f"Model ready: {model_name}")
    return _pipeline


def check_model_ready() -> tuple[bool, str]:
    """Check if transformers + torch are importable."""
    try:
        import transformers
        import torch
        return True, f"transformers {transformers.__version__}, torch {torch.__version__}"
    except ImportError as e:
        return False, str(e)


# ─────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────

class RAGGenerator:
    def __init__(self) -> None:
        self._model_name = settings.hf_model
        self._cache = diskcache.Cache(str(settings.cache_path))
        logger.info(f"RAGGenerator initialised — model: {self._model_name}")

    def _cache_key(self, query: str, chunk_ids: list[str]) -> str:
        payload = query + "|" + ",".join(sorted(chunk_ids))
        return "gen:" + hashlib.md5(payload.encode()).hexdigest()

    def _run_inference(self, prompt: str) -> str:
        pipe = _get_pipeline()
        result = pipe(prompt)
        if isinstance(result, list) and result:
            out = result[0]
            # text2text-generation returns {"generated_text": ...}
            # text-generation returns {"generated_text": full_prompt + answer}
            text = out.get("generated_text", "")
            # Strip the prompt prefix for causal LMs
            if text.startswith(prompt):
                text = text[len(prompt):]
            return text.strip()
        return ""

    def generate(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        memory: ConversationMemory | None = None,
        use_cache: bool = True,
    ) -> GeneratedAnswer:
        cache_key = self._cache_key(query, [c.chunk_id for c in chunks])
        if use_cache and cache_key in self._cache:
            logger.debug("Cache hit")
            return self._cache[cache_key]

        history = memory.as_messages() if memory else []
        prompt = _build_prompt(query, chunks, history)
        answer_text = self._run_inference(prompt)

        result = GeneratedAnswer(
            answer=answer_text,
            citations=_extract_citations(chunks),
            query=query,
            model=self._model_name,
        )
        self._cache.set(cache_key, result, expire=settings.cache_ttl_seconds)
        if memory:
            memory.add("user", query)
            memory.add("assistant", answer_text)
        return result

    def generate_stream(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        memory: ConversationMemory | None = None,
    ) -> tuple[Generator[str, None, None], list[Citation]]:
        """
        HuggingFace pipelines don't natively stream token-by-token easily,
        so we run inference fully and yield word-by-word for a streaming feel.
        """
        citations = _extract_citations(chunks)
        history = memory.as_messages() if memory else []
        prompt = _build_prompt(query, chunks, history)

        def _word_stream() -> Iterator[str]:
            answer = self._run_inference(prompt)
            words = answer.split(" ")
            for i, word in enumerate(words):
                yield word + ("" if i == len(words) - 1 else " ")
            if memory:
                memory.add("user", query)
                memory.add("assistant", answer)

        return _word_stream(), citations
