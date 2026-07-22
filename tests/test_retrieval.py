"""Unit tests for ChromaStore vector store and VectorRetriever."""

from pathlib import Path

import pytest

from app.core.types import Chunk
from app.embedding.service import MiniLMEmbeddingService
from app.retrieval.retriever import VectorRetriever
from app.retrieval.store import ChromaStore


class TestChromaStore:
    """Test ChromaStore operations."""

    @pytest.fixture
    def store(self, tmp_path: Path) -> ChromaStore:
        persist_dir = str(tmp_path / "chroma_test")
        return ChromaStore(persist_dir=persist_dir, collection_name="test_collection")

    @pytest.fixture
    def embedding_service(self) -> MiniLMEmbeddingService:
        return MiniLMEmbeddingService(model_name="all-MiniLM-L6-v2")

    def test_upsert_and_get_by_id(
        self, store: ChromaStore, embedding_service: MiniLMEmbeddingService
    ) -> None:
        chunk = Chunk(
            chunk_id="testchunk1234567",
            source_filename="test.pdf",
            page_number=1,
            char_start=0,
            char_end=50,
            chunk_text="Aspirin is an anti-inflammatory drug.",
        )

        embeddings = embedding_service.embed_documents([chunk.chunk_text])
        added = store.upsert([chunk], embeddings)
        assert added == 1

        fetched = store.get_by_id("testchunk1234567")
        assert fetched is not None
        assert fetched.chunk_id == "testchunk1234567"
        assert fetched.chunk_text == chunk.chunk_text
        assert fetched.source_filename == "test.pdf"

    def test_idempotency_upsert(
        self, store: ChromaStore, embedding_service: MiniLMEmbeddingService
    ) -> None:
        """Upserting identical chunk twice should maintain count = 1."""
        chunk = Chunk(
            chunk_id="idem123456789012",
            source_filename="doc.pdf",
            page_number=2,
            char_start=10,
            char_end=60,
            chunk_text="Idempotency test text for vector store.",
        )
        embeddings = embedding_service.embed_documents([chunk.chunk_text])

        store.upsert([chunk], embeddings)
        count_first = store.count()

        store.upsert([chunk], embeddings)
        count_second = store.count()

        assert count_first == count_second == 1

    def test_vector_retriever(
        self, store: ChromaStore, embedding_service: MiniLMEmbeddingService
    ) -> None:
        chunk = Chunk(
            chunk_id="retriever1234567",
            source_filename="doc.pdf",
            page_number=3,
            char_start=100,
            char_end=200,
            chunk_text="Hypertension is defined as high blood pressure above 140/90 mmHg.",
        )
        embeddings = embedding_service.embed_documents([chunk.chunk_text])
        store.upsert([chunk], embeddings)

        retriever = VectorRetriever(embedding_service, store)
        results = retriever.retrieve("What is hypertension definition?", top_k=5)

        assert len(results) == 1
        assert results[0].chunk_id == "retriever1234567"
        assert results[0].score > 0.0
