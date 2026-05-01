"""
generation/generator.py

LLM response generation using OpenAI API.
Get your key at: https://platform.openai.com/api-keys
- Streaming support
- Conversation memory
- Source citation system
- Disk-cached responses
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generator, Iterator

import time
import diskcache
from loguru import logger
from openai import OpenAI, RateLimitError, AuthenticationError

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
# Generator
# ─────────────────────────────────────────────

class RAGGenerator:
    def __init__(self) -> None:
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_model
        self._cache = diskcache.Cache(str(settings.cache_path))
        logger.info(f"RAGGenerator ready — model: {self._model}")

    def _cache_key(self, query: str, chunk_ids: list[str]) -> str:
        import hashlib
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

        for attempt in range(3):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    max_tokens=1500,
                    temperature=0.1,
                    messages=messages,
                )
                break
            except RateLimitError:
                if attempt < 2:
                    time.sleep(15)
                    continue
                raise

        answer_text = response.choices[0].message.content or ""
        usage = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }

        result = GeneratedAnswer(
            answer=answer_text,
            citations=_extract_citations(chunks),
            query=query,
            model=self._model,
            usage=usage,
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
            full_answer = []
            stream = self._client.chat.completions.create(
                model=self._model,
                max_tokens=1500,
                temperature=0.1,
                messages=messages,
                stream=True,
            )
            for chunk in stream:
                text = chunk.choices[0].delta.content or ""
                if text:
                    full_answer.append(text)
                    yield text

            if memory:
                memory.add("user", query)
                memory.add("assistant", "".join(full_answer))

        return _token_stream(), citations
