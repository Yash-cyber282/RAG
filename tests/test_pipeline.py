"""
tests/test_pipeline.py — Unit and integration tests

Run with: pytest tests/ -v
"""
import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ─── helpers ───────────────────────────────────────────

def make_fake_pdf_bytes() -> bytes:
    """Generate a minimal valid PDF binary for testing."""
    import fitz
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "This is a test document about machine learning and neural networks.")
    page.insert_text((72, 100), "Deep learning models are trained using gradient descent.")
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


# ─── ingestion tests ────────────────────────────────────

class TestPDFLoader:
    def test_load_real_pdf(self, tmp_path):
        from src.ingestion.pdf_loader import load_pdf
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(make_fake_pdf_bytes())

        pages = list(load_pdf(pdf_path))
        assert len(pages) == 1
        assert "machine learning" in pages[0].text.lower()
        assert pages[0].page_number == 1
        assert pages[0].filename == "test.pdf"

    def test_missing_file_raises(self):
        from src.ingestion.pdf_loader import load_pdf
        with pytest.raises(FileNotFoundError):
            list(load_pdf("/nonexistent/file.pdf"))

    def test_doc_id_stable(self, tmp_path):
        from src.ingestion.pdf_loader import load_pdf, _compute_doc_id
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(make_fake_pdf_bytes())
        id1 = _compute_doc_id(pdf_path)
        id2 = _compute_doc_id(pdf_path)
        assert id1 == id2


class TestChunker:
    def test_chunking_produces_chunks(self, tmp_path):
        from src.ingestion.chunker import SemanticChunker
        from src.ingestion.pdf_loader import PageContent

        chunker = SemanticChunker(chunk_size=50, chunk_overlap=10)
        page = PageContent(
            doc_id="abc123",
            source_path="/test.pdf",
            filename="test.pdf",
            page_number=1,
            text="Machine learning is a subset of AI. " * 30,
            word_count=180,
            char_count=1000,
        )
        chunks = chunker.chunk_page(page)
        assert len(chunks) >= 1
        for c in chunks:
            assert c.doc_id == "abc123"
            assert c.page_number == 1
            assert len(c.text) > 0

    def test_blank_page_skipped(self):
        from src.ingestion.chunker import SemanticChunker
        from src.ingestion.pdf_loader import PageContent

        chunker = SemanticChunker()
        blank = PageContent(
            doc_id="x", source_path="", filename="", page_number=1,
            text="   ", word_count=0, char_count=3
        )
        assert chunker.chunk_page(blank) == []


# ─── retrieval tests ────────────────────────────────────

class TestHybridRetriever:
    @patch("src.retrieval.hybrid_retriever.OpenAI")
    def test_rrf_merging(self, mock_ollama):
        from src.retrieval.hybrid_retriever import HybridRetriever

        mock_vs = MagicMock()
        mock_vs.query_dense.return_value = [
            {"chunk_id": "a", "text": "alpha", "metadata": {"filename": "f.pdf", "page_number": 1, "doc_id": "d1"}, "score": 0.9},
            {"chunk_id": "b", "text": "beta",  "metadata": {"filename": "f.pdf", "page_number": 2, "doc_id": "d1"}, "score": 0.7},
        ]
        mock_vs.collection.get.return_value = {
            "ids": ["a", "b", "c"],
            "documents": ["alpha", "beta", "gamma"],
            "metadatas": [
                {"filename": "f.pdf", "page_number": 1, "doc_id": "d1"},
                {"filename": "f.pdf", "page_number": 2, "doc_id": "d1"},
                {"filename": "f.pdf", "page_number": 3, "doc_id": "d1"},
            ],
        }

        retriever = HybridRetriever(mock_vs)
        retriever._ollama.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=[0.1] * 1536)]
        )

        results = retriever.retrieve("test query")
        assert len(results) > 0
        assert all(hasattr(r, "fusion_score") for r in results)
        # All fusion scores non-negative
        assert all(r.fusion_score >= 0 for r in results)


# ─── generation tests ────────────────────────────────────

class TestConversationMemory:
    def test_add_and_retrieve(self):
        from src.generation.generator import ConversationMemory
        mem = ConversationMemory(max_turns=2)
        mem.add("user", "Hello")
        mem.add("assistant", "Hi")
        mem.add("user", "How are you?")
        mem.add("assistant", "Fine")
        mem.add("user", "Tell me about RAG")
        mem.add("assistant", "RAG stands for…")

        messages = mem.as_messages()
        # max_turns=2 → only last 4 messages kept
        assert len(messages) == 4

    def test_clear(self):
        from src.generation.generator import ConversationMemory
        mem = ConversationMemory()
        mem.add("user", "test")
        mem.clear()
        assert mem.as_messages() == []
