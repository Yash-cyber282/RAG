"""
generation/generator.py

Fully LOCAL LLM using HuggingFace Transformers.
Uses AutoModel directly — compatible with transformers v4 and v5.
No Ollama. No API key. No internet after first model download.

Default model: google/flan-t5-base (~250MB, CPU-friendly)
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
# Prompt builder
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
        turns = [
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in history
        ]
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

_model_cache: dict = {}


def _is_seq2seq(model_name: str) -> bool:
    """Detect encoder-decoder models (T5, BART, etc.) vs causal LMs."""
    name = model_name.lower()
    return any(x in name for x in ["t5", "bart", "pegasus", "mt5", "mbart"])


def _get_model(model_name: str):
    """Load and cache model + tokenizer. Auto-detects seq2seq vs causal."""
    if model_name in _model_cache:
        return _model_cache[model_name]

    import torch
    from transformers import AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading model: {model_name} on {device.upper()} (first run downloads it)")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if _is_seq2seq(model_name):
        from transformers import AutoModelForSeq2SeqLM
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )

    model = model.to(device)
    model.eval()

    _model_cache[model_name] = (tokenizer, model, device)
    logger.success(f"Model ready: {model_name}")
    return _model_cache[model_name]


def _run_inference(prompt: str, model_name: str) -> str:
    import torch

    tokenizer, model, device = _get_model(model_name)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(device)

    with torch.no_grad():
        if _is_seq2seq(model_name):
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                num_beams=2,
                early_stopping=True,
            )
            # seq2seq: output is just the answer tokens
            answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        else:
            input_len = inputs["input_ids"].shape[1]
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            # causal LM: output includes the prompt, strip it
            answer = tokenizer.decode(
                output_ids[0][input_len:], skip_special_tokens=True
            )

    return answer.strip()


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
        answer_text = _run_inference(prompt, self._model_name)

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
        """Run inference then yield word-by-word for a streaming feel."""
        citations = _extract_citations(chunks)
        history = memory.as_messages() if memory else []
        prompt = _build_prompt(query, chunks, history)

        def _word_stream() -> Iterator[str]:
            answer = _run_inference(prompt, self._model_name)
            words = answer.split(" ")
            for i, word in enumerate(words):
                yield word + ("" if i == len(words) - 1 else " ")
            if memory:
                memory.add("user", query)
                memory.add("assistant", answer)

        return _word_stream(), citations
