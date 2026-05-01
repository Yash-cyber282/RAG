# RAG Document Intelligence — 100% Local

Ask questions across your PDF library with **zero API costs**.
Everything runs on your machine: no keys, no internet, no subscriptions.

## Stack

| Component | Technology | Cost |
|---|---|---|
| LLM | Ollama (llama3.2 / mistral / phi3 etc.) | Free |
| Embeddings | `all-MiniLM-L6-v2` sentence-transformers | Free |
| Vector store | ChromaDB (local) | Free |
| Retrieval | Hybrid dense + BM25 + RRF | Free |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Free |
| UI | Streamlit | Free |

## Setup

### 1. Install Ollama
Download from https://ollama.com/download and install for your OS.

### 2. Pull a model
```bash
ollama pull llama3.2       # recommended — fast, good quality
# or
ollama pull mistral
ollama pull phi3           # great for low-RAM machines
ollama pull gemma2
```

### 3. Start Ollama server
```bash
ollama serve
```

### 4. Install Python dependencies & run
```bash
pip install -r requirements.txt
streamlit run app.py
```

The app auto-detects all locally available Ollama models — switch between them via the dropdown in the sidebar.

## Configuration (optional `.env`)

```
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
CHUNK_SIZE=512
TOP_K_DENSE=10
TOP_K_BM25=10
TOP_K_RERANK=5
```
