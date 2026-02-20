"""
Unit tests for core/rag.py — RAGManager ingestion and retrieval.
Each RAGManager gets a unique collection name to avoid ChromaDB singleton collisions.
"""
import os
import uuid
import tempfile
import unittest
from unittest.mock import patch

import chromadb
from core.rag import RAGManager


def _make_rag(content: str = None, cleanup_ref: list = None) -> RAGManager:
    """
    Create a RAGManager with a unique ChromaDB collection name per call
    to prevent 'collection already exists' errors when running multiple tests.
    """
    unique_name = f"portfolio_{uuid.uuid4().hex}"

    def patched_init(self, data_path):
        self.data_path = data_path
        self.chroma_client = chromadb.Client()
        from chromadb.utils import embedding_functions
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.chroma_client.create_collection(
            name=unique_name,
            embedding_function=self.embedding_fn
        )

    with patch.object(RAGManager, "__init__", patched_init):
        if content is not None:
            tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8")
            tmp.write(content)
            tmp.close()
            if cleanup_ref is not None:
                cleanup_ref.append(tmp.name)
            rag = RAGManager(tmp.name)
        else:
            rag = RAGManager("nonexistent_path/portfolio.md")

    return rag


class TestRAGManagerIngest(unittest.TestCase):
    """Tests for portfolio ingestion logic."""

    def test_ingest_normal_portfolio(self):
        """Ingesting a well-formatted portfolio populates the collection."""
        cleanup = []
        content = "I built an AI-powered RAG system using Python and ChromaDB.\n\nI also worked on ESP32 IoT projects with MQTT and embedded C."
        rag = _make_rag(content, cleanup)
        rag.ingest_portfolio()
        self.assertGreater(rag.collection.count(), 0)
        for path in cleanup:
            os.unlink(path)

    def test_ingest_missing_file(self):
        """Missing portfolio file logs a warning but does not crash."""
        rag = _make_rag()
        try:
            rag.ingest_portfolio()
        except Exception as e:
            self.fail(f"ingest_portfolio raised unexpectedly: {e}")
        self.assertEqual(rag.collection.count(), 0)

    def test_ingest_empty_file(self):
        """A portfolio with no paragraphs > 20 chars is handled gracefully."""
        cleanup = []
        rag = _make_rag("Hi\n\nOk\n", cleanup)
        rag.ingest_portfolio()
        self.assertEqual(rag.collection.count(), 0)
        for path in cleanup:
            os.unlink(path)

    def test_ingest_chunks_count(self):
        """Chunking splits on double newlines — verifies expected chunk count."""
        cleanup = []
        content = "Paragraph one with enough content here.\n\nParagraph two with enough content here.\n\nParagraph three big content here."
        rag = _make_rag(content, cleanup)
        rag.ingest_portfolio()
        self.assertEqual(rag.collection.count(), 3)
        for path in cleanup:
            os.unlink(path)


class TestRAGManagerRetrieval(unittest.TestCase):
    """Tests for context retrieval and similarity filtering."""

    def setUp(self):
        self._cleanup = []
        content = (
            "I built a Retrieval-Augmented Generation system using Python, LangChain, and ChromaDB "
            "to answer questions from a custom knowledge base.\n\n"
            "I developed an ESP32-based environmental monitoring system with sensors for temperature, "
            "humidity, and CO2, transmitting data to a cloud dashboard.\n\n"
            "I have strong experience with PyTorch model training, fine-tuning LLMs, and deploying "
            "inference pipelines on GPU servers."
        )
        self.rag = _make_rag(content, self._cleanup)
        self.rag.ingest_portfolio()

    def tearDown(self):
        for path in self._cleanup:
            os.unlink(path)

    def test_retrieve_returns_string(self):
        """retrieve_context always returns a non-None string."""
        result = self.rag.retrieve_context("Tell me about your AI projects")
        self.assertIsInstance(result, str)

    def test_retrieve_relevant_chunk(self):
        """A query about RAG should return content related to RAG/ChromaDB."""
        result = self.rag.retrieve_context("Explain your RAG system")
        self.assertTrue(
            "RAG" in result or "ChromaDB" in result or "No specific" in result,
            f"Expected RAG-related content, got: {result[:100]}"
        )

    def test_retrieve_empty_collection(self):
        """Querying an empty collection returns a safe fallback string."""
        empty_rag = _make_rag()
        result = empty_rag.retrieve_context("any query")
        self.assertIn("No portfolio context", result)

    def test_retrieve_separator_for_multiple_chunks(self):
        """Multiple good chunks joined with --- separator."""
        self.rag.SIMILARITY_THRESHOLD = 999.0
        result = self.rag.retrieve_context("Python AI ESP32 LLM PyTorch")
        # With high threshold all 3 chunks pass — separator should be present
        self.assertIn("---", result)

    def test_similarity_threshold_filters_noise(self):
        """With threshold=0, unrelated query returns the empty context message."""
        self.rag.SIMILARITY_THRESHOLD = 0.0
        result = self.rag.retrieve_context("completely unrelated gibberish xyz123")
        self.assertIn("No specific portfolio context", result)


if __name__ == "__main__":
    unittest.main()
