"""
generation/generator.py

Fully LOCAL LLM response generation using Ollama.
No API key. No internet. No cost.

Install Ollama: https://ollama.com/download
Then run:  ollama pull llama3.2
Then start: ollama serve
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generator, Iterator
import hashlib
import time
import diskcache
from loguru import logger

import requests

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
# System prompt
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert document analyst. Answer the user's question using ONLY the provided context passages.

Rules:
1. Base your answer solely on the context. Do not use prior knowledge.
2. If the context does not contain enough information, say so clearly.
3. Cite the specific document(s) your answer comes from using [Source: filename, p.N] inline.
4. Be concise but complete. Use bullet points for lists.
5. If asked about multiple documents, synthesise across all relevant sources.
"""


def _build_context_block(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[{i}] Source: {chunk.filename}, Page {chunk.page_number}\n"
            f"{chunk.text}\n"
        )
    return "\n---\n".join(parts)


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
    def __init__(self, max_turns: int = 6) -> None:
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
# Ollama client helpers
# ─────────────────────────────────────────────

def _ollama_base() -> str:
    return settings.ollama_base_url.rstrip("/")


def check_ollama_running() -> tuple[bool, str]:
    """Returns (is_running, error_message)."""
    try:
        r = requests.get(f"{_ollama_base()}/api/tags", timeout=3)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            return True, ", ".join(models) if models else "(no models pulled yet)"
        return False, f"Ollama returned status {r.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "Ollama is not running. Start it with: ollama serve"
    except Exception as e:
        return False, str(e)


def check_model_available(model: str) -> tuple[bool, str]:
    """Check if a specific model is pulled locally."""
    try:
        r = requests.get(f"{_ollama_base()}/api/tags", timeout=3)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            # Match with or without tag
            base = model.split(":")[0]
            for m in models:
                if m == model or m.split(":")[0] == base:
                    return True, m
            return False, f"Model '{model}' not found. Run: ollama pull {model}"
        return False, "Could not query Ollama models."
    except Exception as e:
        return False, str(e)


# ─────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────

class RAGGenerator:
    def __init__(self) -> None:
        self._model = settings.ollama_model
        self._base = _ollama_base()
        self._cache = diskcache.Cache(str(settings.cache_path))
        logger.info(f"RAGGenerator ready — Ollama model: {self._model} @ {self._base}")

    def _cache_key(self, query: str, chunk_ids: list[str]) -> str:
        payload = query + "|" + ",".join(sorted(chunk_ids))
        return "gen:" + hashlib.md5(payload.encode()).hexdigest()

    def _build_messages(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        memory: ConversationMemory | None,
    ) -> list[dict]:
        context = _build_context_block(chunks)
        user_content = f"Context:\n{context}\n\nQuestion: {query}"
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if memory and memory.history:
            messages.extend(memory.as_messages())
        messages.append({"role": "user", "content": user_content})
        return messages

    def generate(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        memory: ConversationMemory | None = None,
        use_cache: bool = True,
    ) -> GeneratedAnswer:
        cache_key = self._cache_key(query, [c.chunk_id for c in chunks])
        if use_cache and cache_key in self._cache:
            logger.debug("Cache hit — returning cached answer")
            return self._cache[cache_key]

        messages = self._build_messages(query, chunks, memory)

        response = requests.post(
            f"{self._base}/api/chat",
            json={
                "model": self._model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0.1},
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        answer_text = data.get("message", {}).get("content", "")

        result = GeneratedAnswer(
            answer=answer_text,
            citations=_extract_citations(chunks),
            query=query,
            model=self._model,
            usage={
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
            },
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
        citations = _extract_citations(chunks)
        messages = self._build_messages(query, chunks, memory)

        def _token_stream() -> Iterator[str]:
            import json
            full_answer = []
            with requests.post(
                f"{self._base}/api/chat",
                json={
                    "model": self._model,
                    "messages": messages,
                    "stream": True,
                    "options": {"temperature": 0.1},
                },
                stream=True,
                timeout=120,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = chunk.get("message", {}).get("content", "")
                    if text:
                        full_answer.append(text)
                        yield text
                    if chunk.get("done"):
                        break

            if memory:
                memory.add("user", query)
                memory.add("assistant", "".join(full_answer))

        return _token_stream(), citations
