"""
config.py — All settings from environment variables.
Fully LOCAL stack — no API keys needed:
  - Embeddings: sentence-transformers (local)
  - LLM: Ollama (local) — https://ollama.com
"""
from __future__ import annotations
import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    pass


class Settings:
    def __init__(self):
        pass

    # ── Ollama (local LLM) ────────────────────────────────────────────────
    @property
    def ollama_base_url(self) -> str:
        return os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

    @property
    def ollama_model(self) -> str:
        return os.environ.get("OLLAMA_MODEL", "llama3.2")

    # ── ChromaDB ─────────────────────────────────────────────────────────
    @property
    def chroma_persist_dir(self) -> str:
        env_val = os.environ.get("CHROMA_PERSIST_DIR", "")
        return env_val if env_val else str(_PROJECT_ROOT / "data" / "chroma_db")

    @property
    def chroma_collection_name(self) -> str:
        return os.environ.get("CHROMA_COLLECTION_NAME", "rag_documents")

    # ── Chunking ─────────────────────────────────────────────────────────
    @property
    def chunk_size(self) -> int:
        return int(os.environ.get("CHUNK_SIZE", "512"))

    @property
    def chunk_overlap(self) -> int:
        return int(os.environ.get("CHUNK_OVERLAP", "50"))

    @property
    def batch_size(self) -> int:
        return int(os.environ.get("BATCH_SIZE", "100"))

    # ── Retrieval ─────────────────────────────────────────────────────────
    @property
    def top_k_dense(self) -> int:
        return int(os.environ.get("TOP_K_DENSE", "10"))

    @property
    def top_k_bm25(self) -> int:
        return int(os.environ.get("TOP_K_BM25", "10"))

    @property
    def top_k_rerank(self) -> int:
        return int(os.environ.get("TOP_K_RERANK", "5"))

    # ── Cache ─────────────────────────────────────────────────────────────
    @property
    def cache_dir(self) -> str:
        env_val = os.environ.get("CACHE_DIR", "")
        return env_val if env_val else str(_PROJECT_ROOT / "data" / "cache")

    @property
    def cache_ttl_seconds(self) -> int:
        return int(os.environ.get("CACHE_TTL_SECONDS", "86400"))

    # ── Reranker ──────────────────────────────────────────────────────────
    @property
    def reranker_model(self) -> str:
        return os.environ.get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    # ── Resolved paths ────────────────────────────────────────────────────
    @property
    def chroma_dir(self) -> Path:
        p = Path(self.chroma_persist_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def cache_path(self) -> Path:
        p = Path(self.cache_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


settings = Settings()
