# RAG Document Intelligence

Ask questions across your PDF library. Answers are grounded in your documents using hybrid retrieval (dense + BM25 + reranking) and **Anthropic Claude** as the LLM.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
streamlit run app.py
```

## Get an API Key

1. Go to [console.anthropic.com](https://console.anthropic.com)
2. Create an account and generate an API key
3. Paste it in the sidebar when the app starts (or set `ANTHROPIC_API_KEY` in `.env`)

## Stack

| Component | Technology |
|---|---|
| LLM | Anthropic Claude (Haiku by default, configurable) |
| Embeddings | `all-MiniLM-L6-v2` via sentence-transformers (local, free) |
| Vector store | ChromaDB (local) |
| Retrieval | Hybrid dense + BM25 with RRF fusion |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local) |
| UI | Streamlit |

## Configuration (`.env`)

```
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-haiku-4-5-20251001   # or claude-sonnet-4-6 for higher quality
CHUNK_SIZE=512
CHUNK_OVERLAP=50
TOP_K_DENSE=10
TOP_K_BM25=10
TOP_K_RERANK=5
```
