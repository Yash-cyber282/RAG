# RAG Document Intelligence

Ask questions across your PDF library. Answers grounded in your documents using hybrid retrieval (dense + BM25 + reranking) and **OpenAI** as the LLM.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
streamlit run app.py
```

## Get an API Key

1. Go to [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Create a key and paste it in the sidebar when the app starts (or set `OPENAI_API_KEY` in `.env`)

## Stack

| Component | Technology |
|---|---|
| LLM | OpenAI (gpt-4o-mini by default, configurable) |
| Embeddings | `all-MiniLM-L6-v2` via sentence-transformers (local, free) |
| Vector store | ChromaDB (local) |
| Retrieval | Hybrid dense + BM25 with RRF fusion |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local) |
| UI | Streamlit |

## Configuration (`.env`)

```
OPENAI_API_KEY=sk-proj-...
OPENAI_MODEL=gpt-4o-mini   # or gpt-4o for higher quality
CHUNK_SIZE=512
CHUNK_OVERLAP=50
TOP_K_DENSE=10
TOP_K_BM25=10
TOP_K_RERANK=5
```
