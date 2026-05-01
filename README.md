# RAG Document Intelligence — 100% Local, Zero Dependencies on External Services

No Ollama. No API key. No subscriptions. Just `pip install` and run.

## Stack

| Component | Technology | Size | Cost |
|---|---|---|---|
| LLM | HuggingFace `flan-t5-base` | ~250MB | Free |
| Embeddings | `all-MiniLM-L6-v2` | ~90MB | Free |
| Vector store | ChromaDB | local | Free |
| Retrieval | Hybrid dense + BM25 + RRF | — | Free |
| Reranking | `ms-marco-MiniLM-L-6-v2` | ~80MB | Free |
| UI | Streamlit | — | Free |

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

On first run, models download automatically (~420MB total) and are cached locally. No internet needed after that.

## Switching models

Set `HF_MODEL` in your `.env` or via the sidebar dropdown:

```
# Fast & CPU-friendly (default)
HF_MODEL=google/flan-t5-base

# Better quality, still CPU-ok
HF_MODEL=google/flan-t5-large

# Best quality, needs GPU
HF_MODEL=mistralai/Mistral-7B-Instruct-v0.3
```
