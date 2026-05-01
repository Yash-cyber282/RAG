"""
app.py — RAG Document Intelligence (Streamlit app)
100% local. No API keys. No internet required.
Powered by Ollama (LLM) + sentence-transformers (embeddings) + ChromaDB (vector store).
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

st.set_page_config(
    page_title="RAG Document Intelligence",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.citation-box {
    background: #1e2a4a;
    border-left: 3px solid #4f6af5;
    padding: 0.6rem 1rem;
    border-radius: 4px;
    margin: 0.4rem 0;
    font-size: 0.87rem;
    color: #c9d1e8;
}
.citation-label { font-weight: 600; color: #7b93f5; font-size: 0.8rem;
    text-transform: uppercase; letter-spacing: 0.04em; }
.score-badge { background: #2a3560; color: #9eb0ff; padding: 1px 8px;
    border-radius: 12px; font-size: 0.77rem; margin-left: 6px; }
.chunk-preview { color: #8a96b8; font-size: 0.82rem; margin-top: 0.25rem; }
.error-card { background: #2d1a1a; border: 1px solid #7a2020; border-radius: 6px;
    padding: 0.8rem 1rem; margin: 0.5rem 0; color: #f08080; font-size: 0.9rem; }
.info-card { background: #1a2a2d; border: 1px solid #1e6a72; border-radius: 6px;
    padding: 0.8rem 1rem; margin: 0.5rem 0; color: #7ecfe0; font-size: 0.9rem; }
.setup-step { background: #1e1e2e; border: 1px solid #3a3a5c; border-radius: 6px;
    padding: 0.7rem 1rem; margin: 0.4rem 0; font-size: 0.85rem; color: #cdd6f4; }
.setup-step code { background: #313244; padding: 2px 6px; border-radius: 4px;
    font-family: monospace; color: #cba6f7; }
</style>
""", unsafe_allow_html=True)

for key, default in [
    ("messages", []),
    ("ollama_ok", False),
    ("pipeline", None),
    ("ingest_pipeline", None),
    ("ingest_log", []),
    ("selected_model", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Helpers ────────────────────────────────────────────────────────────────

def get_pipeline():
    if st.session_state.pipeline is None:
        from src.generation.query_pipeline import QueryPipeline
        st.session_state.pipeline = QueryPipeline()
    return st.session_state.pipeline


def get_ingest_pipeline():
    if st.session_state.ingest_pipeline is None:
        from src.ingestion.pipeline import IngestionPipeline
        st.session_state.ingest_pipeline = IngestionPipeline()
    return st.session_state.ingest_pipeline


def reset_pipeline():
    st.session_state.pipeline = None
    st.session_state.ingest_pipeline = None


# ── SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📚 Document Library")

    # ── Ollama status ─────────────────────────────────────────────────────
    from src.generation.generator import check_ollama_running, check_model_available
    from src.config import settings

    with st.expander("🖥️ Ollama (Local LLM)", expanded=not st.session_state.ollama_ok):
        ollama_running, ollama_info = check_ollama_running()

        if not ollama_running:
            st.error(f"❌ {ollama_info}")
            st.markdown("""
<div class="setup-step">
<b>1.</b> Install Ollama: <a href="https://ollama.com/download" target="_blank">ollama.com/download</a>
</div>
<div class="setup-step">
<b>2.</b> Pull a model: <code>ollama pull llama3.2</code>
</div>
<div class="setup-step">
<b>3.</b> Start the server: <code>ollama serve</code>
</div>
""", unsafe_allow_html=True)
            if st.button("🔄 Retry connection"):
                st.rerun()
            st.stop()

        # Ollama is running — pick model
        st.success(f"✅ Ollama running")

        # List available models
        import requests as _req
        try:
            _r = _req.get(f"{settings.ollama_base_url}/api/tags", timeout=3)
            available_models = [m["name"] for m in _r.json().get("models", [])]
        except Exception:
            available_models = []

        if not available_models:
            st.warning("No models pulled yet.")
            st.markdown("""
<div class="setup-step">Run: <code>ollama pull llama3.2</code></div>
""", unsafe_allow_html=True)
            if st.button("🔄 Refresh"):
                st.rerun()
            st.stop()

        default_model = settings.ollama_model
        default_idx = 0
        for i, m in enumerate(available_models):
            if m.split(":")[0] == default_model.split(":")[0]:
                default_idx = i
                break

        chosen = st.selectbox("Model", available_models, index=default_idx)
        if chosen != st.session_state.selected_model:
            st.session_state.selected_model = chosen
            os.environ["OLLAMA_MODEL"] = chosen
            reset_pipeline()

        st.session_state.ollama_ok = True
        st.caption(f"Using: `{chosen}` — swap anytime via the dropdown")

    if not st.session_state.ollama_ok:
        st.stop()

    # ── Upload PDFs ───────────────────────────────────────────────────────
    st.subheader("Upload PDFs")
    uploaded_files = st.file_uploader(
        "Drop PDFs here",
        type="pdf",
        accept_multiple_files=True,
        help="Max 200 MB per file.",
    )

    if uploaded_files and st.button("⚡ Ingest PDFs", type="primary"):
        st.session_state.ingest_log = []
        try:
            ingest = get_ingest_pipeline()
        except Exception as e:
            st.session_state.ingest_log.append(("error", f"Failed to initialise ingestion engine: {e}"))
            ingest = None

        if ingest:
            results, errors = [], []
            progress = st.progress(0, text="Starting…")
            for i, uf in enumerate(uploaded_files):
                progress.progress(i / len(uploaded_files), text=f"Processing {uf.name}…")
                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uf.read())
                        tmp_path = Path(tmp.name)

                    if tmp_path.stat().st_size < 100:
                        raise ValueError("File is too small to be a valid PDF.")

                    result = ingest.ingest_pdf(tmp_path)
                    chunks = result.get("chunks_stored", 0)
                    pages = result.get("pages_loaded", 0)
                    if chunks == 0:
                        raise ValueError(
                            f"0 chunks stored (pages_loaded={pages}). "
                            "PDF may be image-only, password-protected, or corrupt."
                        )
                    results.append(result)
                    st.session_state.ingest_log.append(("ok", f"✅ {uf.name}: {pages} pages → {chunks} chunks"))
                except Exception as e:
                    errors.append(uf.name)
                    st.session_state.ingest_log.append(("error", f"❌ {uf.name}: {e}"))
                finally:
                    if tmp_path and tmp_path.exists():
                        tmp_path.unlink(missing_ok=True)

            progress.progress(1.0, text="Done!")
            if results:
                total_chunks = sum(r.get("chunks_stored", 0) for r in results)
                st.session_state.ingest_log.insert(0, ("success",
                    f"✅ {len(results)} file(s) indexed — {total_chunks:,} chunks stored"))
                reset_pipeline()
            if errors:
                st.session_state.ingest_log.insert(0 if not results else 1, ("warn",
                    f"⚠️ {len(errors)} file(s) failed — see details below"))
            st.rerun()

    if st.session_state.ingest_log:
        with st.expander("📋 Last ingestion log", expanded=True):
            for level, msg in st.session_state.ingest_log:
                if level in ("success", "ok"):
                    st.success(msg)
                elif level == "error":
                    st.error(msg)
                elif level == "warn":
                    st.warning(msg)
                else:
                    st.info(msg)

    # ── Indexed documents ─────────────────────────────────────────────────
    st.subheader("Indexed Documents")
    try:
        qp = get_pipeline()
        from src.config import settings as _s
        st.caption(f"🗄 DB path: `{_s.chroma_dir}` · {qp._vector_store.count()} chunks")
        docs = qp._vector_store.list_documents()
        if docs:
            for doc in docs:
                with st.expander(f"📄 {doc['filename']}", expanded=False):
                    st.caption(f"Pages: {doc['total_pages']}  ·  ID: `{doc['doc_id']}`")
                    if st.button("🗑 Remove", key=f"del_{doc['doc_id']}"):
                        try:
                            qp._vector_store.delete_document(doc["doc_id"])
                            reset_pipeline()
                            st.success("Removed.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Delete failed: {e}")
        else:
            st.markdown('<div class="info-card">No documents indexed yet.<br>Upload PDFs above to get started.</div>', unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Could not load document list: {e}")

    st.divider()

    # ── Settings ──────────────────────────────────────────────────────────
    st.subheader("Settings")
    streaming   = st.toggle("Streaming response",    value=True)
    show_chunks = st.toggle("Show retrieved chunks", value=False)
    use_cache   = st.toggle("Use response cache",    value=True)

    if st.button("🗑 Clear conversation"):
        st.session_state.messages = []
        try:
            get_pipeline().reset_memory()
        except Exception:
            pass
        st.rerun()

    st.divider()
    try:
        n = get_pipeline()._vector_store.count()
        st.caption(f"🗄 **{n:,}** chunks indexed")
    except Exception:
        st.caption("🗄 — chunks indexed")
    st.caption(f"Ollama ({st.session_state.selected_model or settings.ollama_model}) · all-MiniLM-L6-v2 · Hybrid RAG")


# ── MAIN AREA ──────────────────────────────────────────────────────────────
st.title("🔍 RAG Document Intelligence")
st.caption("Ask questions across your PDF library. 100% local — no API key needed.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander(f"📎 Sources ({len(msg['citations'])})", expanded=False):
                for cit in msg["citations"]:
                    st.markdown(
                        f'<div class="citation-box">'
                        f'<span class="citation-label">{cit["label"]}</span>'
                        f'<span class="score-badge">score {cit["score"]}</span>'
                        f'<div class="chunk-preview">{cit["excerpt"]}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
        if show_chunks and msg.get("chunks"):
            with st.expander(f"🔬 Chunks ({len(msg['chunks'])})", expanded=False):
                for i, c in enumerate(msg["chunks"], 1):
                    st.markdown(f"**[{i}]** `{c['filename']}` p.{c['page_number']} — score {c['fusion_score']:.3f}")
                    st.code(c["text"][:400], language=None)

prompt = st.chat_input("Ask a question about your documents…")

if prompt:
    prompt = prompt.strip()
    if not prompt:
        st.warning("Please type a question.")
        st.stop()

    try:
        qp = get_pipeline()
        doc_count = qp._vector_store.count()
    except Exception as e:
        st.error(f"❌ Could not connect to the document store: {e}")
        st.stop()

    if doc_count == 0:
        st.warning("⚠️ No documents indexed yet. Upload PDFs in the sidebar first.")
        st.stop()

    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    full_answer = ""
    citations = []
    ranked_chunks = []

    with st.chat_message("assistant"):
        answer_box = st.empty()

        try:
            if streaming:
                token_stream, citations, ranked_chunks = qp.query_stream(prompt)
                for token in token_stream:
                    full_answer += token
                    answer_box.markdown(full_answer + "▌")
                answer_box.markdown(full_answer)
            else:
                with st.spinner("Thinking…"):
                    result = qp.query(prompt, use_cache=use_cache)
                full_answer = result.answer
                citations = result.citations
                ranked_chunks = result.retrieved_chunks
                answer_box.markdown(full_answer)

        except Exception as e:
            err_str = str(e)
            err_low = err_str.lower()
            if "connection" in err_low or "refused" in err_low:
                msg = "❌ **Ollama not reachable.** Make sure `ollama serve` is running."
            elif "model" in err_low and ("not found" in err_low or "pull" in err_low):
                msg = f"❌ **Model not found.** Run: `ollama pull {settings.ollama_model}`"
            elif "timeout" in err_low:
                msg = "❌ **Request timed out.** The model may be loading — try again in a moment."
            elif "no documents" in err_low or "collection" in err_low:
                msg = "❌ **No documents found.** Upload and ingest PDFs first."
            else:
                msg = f"❌ **Error:** {err_str}"

            answer_box.markdown(f'<div class="error-card">{msg}</div>', unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": msg, "citations": [], "chunks": []})
            st.stop()

        if citations:
            with st.expander(f"📎 Sources ({len(citations)})", expanded=True):
                for cit in citations:
                    st.markdown(
                        f'<div class="citation-box">'
                        f'<span class="citation-label">{cit.label}</span>'
                        f'<span class="score-badge">score {cit.score}</span>'
                        f'<div class="chunk-preview">{cit.excerpt}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        if show_chunks and ranked_chunks:
            with st.expander(f"🔬 Retrieved chunks ({len(ranked_chunks)})", expanded=False):
                for i, c in enumerate(ranked_chunks, 1):
                    st.markdown(f"**[{i}]** `{c.filename}` p.{c.page_number} — score {c.fusion_score:.3f}")
                    st.code(c.text[:400], language=None)

    if full_answer:
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_answer,
            "citations": [{"label": c.label, "excerpt": c.excerpt, "score": c.score} for c in citations],
            "chunks": [
                {"filename": c.filename, "page_number": c.page_number,
                 "text": c.text, "fusion_score": c.fusion_score}
                for c in ranked_chunks
            ] if show_chunks else [],
        })
