"""
app.py — RAG Document Intelligence
100% local. No API keys. No Ollama. No internet after first run.
LLM: HuggingFace Transformers | Embeddings: sentence-transformers | DB: ChromaDB
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
    background: #1e2a4a; border-left: 3px solid #4f6af5;
    padding: 0.6rem 1rem; border-radius: 4px;
    margin: 0.4rem 0; font-size: 0.87rem; color: #c9d1e8;
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
.model-card { background: #1e1e2e; border: 1px solid #3a3a5c; border-radius: 6px;
    padding: 0.7rem 1rem; margin: 0.3rem 0; font-size: 0.83rem; color: #cdd6f4; }
.model-card code { background: #313244; padding: 2px 6px; border-radius: 4px;
    font-family: monospace; color: #cba6f7; }
</style>
""", unsafe_allow_html=True)

for key, default in [
    ("messages", []),
    ("pipeline", None),
    ("ingest_pipeline", None),
    ("ingest_log", []),
    ("model_loaded", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default


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

    # ── Model info ────────────────────────────────────────────────────────
    from src.generation.generator import check_model_ready
    from src.config import settings

    with st.expander("🤖 Local LLM (HuggingFace)", expanded=True):
        ok, info = check_model_ready()
        if not ok:
            st.error(f"❌ Missing dependency: {info}")
            st.code("pip install transformers torch", language="bash")
            st.stop()

        current_model = os.environ.get("HF_MODEL", settings.hf_model)

        st.success(f"✅ Ready — {info}")
        st.caption(f"Model: `{current_model}`")

        with st.expander("🔄 Switch model", expanded=False):
            st.markdown("""
<div class="model-card">🟢 <b>flan-t5-base</b> — <code>google/flan-t5-base</code><br>~250MB · CPU · Fast · Recommended</div>
<div class="model-card">🟡 <b>flan-t5-large</b> — <code>google/flan-t5-large</code><br>~800MB · CPU · Better quality</div>
<div class="model-card">🔵 <b>flan-t5-xl</b> — <code>google/flan-t5-xl</code><br>~3GB · GPU recommended</div>
<div class="model-card">🟣 <b>Mistral 7B</b> — <code>mistralai/Mistral-7B-Instruct-v0.3</code><br>~14GB · GPU required</div>
""", unsafe_allow_html=True)
            new_model = st.text_input("Custom model ID", placeholder="google/flan-t5-base")
            if st.button("Apply model") and new_model.strip():
                os.environ["HF_MODEL"] = new_model.strip()
                # Force reload
                from src.generation import generator as _gen
                _gen._pipeline = None
                reset_pipeline()
                st.success(f"Switched to `{new_model.strip()}` — will load on next query.")
                st.rerun()

    # ── Upload PDFs ───────────────────────────────────────────────────────
    st.subheader("Upload PDFs")
    uploaded_files = st.file_uploader(
        "Drop PDFs here", type="pdf", accept_multiple_files=True
    )

    if uploaded_files and st.button("⚡ Ingest PDFs", type="primary"):
        st.session_state.ingest_log = []
        try:
            ingest = get_ingest_pipeline()
        except Exception as e:
            st.session_state.ingest_log.append(("error", f"Failed to init engine: {e}"))
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
                        raise ValueError("File too small to be a valid PDF.")
                    result = ingest.ingest_pdf(tmp_path)
                    chunks = result.get("chunks_stored", 0)
                    pages = result.get("pages_loaded", 0)
                    if chunks == 0:
                        raise ValueError(f"0 chunks stored (pages={pages}). PDF may be image-only or corrupt.")
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
                total = sum(r.get("chunks_stored", 0) for r in results)
                st.session_state.ingest_log.insert(0, ("success",
                    f"✅ {len(results)} file(s) indexed — {total:,} chunks stored"))
                reset_pipeline()
            if errors:
                st.session_state.ingest_log.insert(0 if not results else 1, ("warn",
                    f"⚠️ {len(errors)} file(s) failed"))
            st.rerun()

    if st.session_state.ingest_log:
        with st.expander("📋 Last ingestion log", expanded=True):
            for level, msg in st.session_state.ingest_log:
                if level in ("success", "ok"): st.success(msg)
                elif level == "error": st.error(msg)
                elif level == "warn": st.warning(msg)
                else: st.info(msg)

    # ── Indexed documents ─────────────────────────────────────────────────
    st.subheader("Indexed Documents")
    try:
        qp = get_pipeline()
        st.caption(f"🗄 DB: `{settings.chroma_dir}` · {qp._vector_store.count()} chunks")
        docs = qp._vector_store.list_documents()
        if docs:
            for doc in docs:
                with st.expander(f"📄 {doc['filename']}", expanded=False):
                    st.caption(f"Pages: {doc['total_pages']} · ID: `{doc['doc_id']}`")
                    if st.button("🗑 Remove", key=f"del_{doc['doc_id']}"):
                        try:
                            qp._vector_store.delete_document(doc["doc_id"])
                            reset_pipeline()
                            st.success("Removed.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Delete failed: {e}")
        else:
            st.markdown('<div class="info-card">No documents yet. Upload PDFs above.</div>',
                        unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Could not load document list: {e}")

    st.divider()

    st.subheader("Settings")
    streaming    = st.toggle("Streaming response",    value=True)
    show_chunks  = st.toggle("Show retrieved chunks", value=False)
    use_cache    = st.toggle("Use response cache",    value=True)

    if st.button("🗑 Clear conversation"):
        st.session_state.messages = []
        try: get_pipeline().reset_memory()
        except Exception: pass
        st.rerun()

    st.divider()
    try:
        n = get_pipeline()._vector_store.count()
        st.caption(f"🗄 **{n:,}** chunks indexed")
    except Exception:
        st.caption("🗄 — chunks indexed")
    st.caption(f"`{settings.hf_model}` · all-MiniLM-L6-v2 · Hybrid RAG · 100% local")


# ── MAIN AREA ──────────────────────────────────────────────────────────────
st.title("🔍 RAG Document Intelligence")
st.caption("100% local — no API key, no Ollama, no internet required after first run.")

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
                        f'</div>', unsafe_allow_html=True)
        if show_chunks and msg.get("chunks"):
            with st.expander(f"🔬 Chunks ({len(msg['chunks'])})", expanded=False):
                for i, c in enumerate(msg["chunks"], 1):
                    st.markdown(f"**[{i}]** `{c['filename']}` p.{c['page_number']} — {c['fusion_score']:.3f}")
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
        st.error(f"❌ Document store error: {e}")
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
                with st.spinner("Loading model & generating…" if not st.session_state.model_loaded else "Generating…"):
                    token_stream, citations, ranked_chunks = qp.query_stream(prompt)
                    for token in token_stream:
                        full_answer += token
                        answer_box.markdown(full_answer + "▌")
                    st.session_state.model_loaded = True
                answer_box.markdown(full_answer)
            else:
                with st.spinner("Loading model & generating…" if not st.session_state.model_loaded else "Generating…"):
                    result = qp.query(prompt, use_cache=use_cache)
                    st.session_state.model_loaded = True
                full_answer = result.answer
                citations = result.citations
                ranked_chunks = result.retrieved_chunks
                answer_box.markdown(full_answer)

        except Exception as e:
            err = str(e)
            err_low = err.lower()
            if "out of memory" in err_low or "oom" in err_low:
                msg = "❌ **Out of memory.** Try a smaller model like `google/flan-t5-base`."
            elif "no space" in err_low or "disk" in err_low:
                msg = "❌ **Disk full.** Free up space for model weights."
            elif "connection" in err_low or "network" in err_low:
                msg = "❌ **Network error** downloading model. Check your internet connection for first-time download."
            elif "no documents" in err_low or "collection" in err_low:
                msg = "❌ **No documents found.** Upload and ingest PDFs first."
            else:
                msg = f"❌ **Error:** {err}"
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
                        f'</div>', unsafe_allow_html=True)

        if show_chunks and ranked_chunks:
            with st.expander(f"🔬 Retrieved chunks ({len(ranked_chunks)})", expanded=False):
                for i, c in enumerate(ranked_chunks, 1):
                    st.markdown(f"**[{i}]** `{c.filename}` p.{c.page_number} — {c.fusion_score:.3f}")
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
