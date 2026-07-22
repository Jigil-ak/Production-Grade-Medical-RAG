"""Vector store Protocol and ChromaDB implementation.

Protocol and implementation co-located — no centralized protocols.py.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.core.types import Chunk, RetrievedChunk


@runtime_checkable
class VectorStore(Protocol):
    """Interface for vector storage and retrieval."""

    def upsert(
        self, chunks: list[Chunk], embeddings: list[list[float]]
    ) -> None:
        """Insert or update chunks with their embeddings."""
        ...

    def query(
        self, embedding: list[float], top_k: int
    ) -> list[RetrievedChunk]:
        """Query for the top-k most similar chunks."""
        ...

    def get_by_id(self, chunk_id: str) -> Chunk | None:
        """O(1) lookup of a chunk by its ID. Returns None if not found."""
        ...


class ChromaStore:
    """ChromaDB vector store in embedded/persistent-directory mode.

    Uses CHROMA_PERSIST_DIR from settings. NOT server mode.
    chunk_id (SHA-256 hash, first 16 hex chars) is used directly as the
    Chroma document ID for O(1) lookups.
    """

    def __init__(self, persist_dir: str, collection_name: str = "medical_chunks") -> None:
        self._persist_dir = persist_dir
        self._collection_name = collection_name
        # Phase 1: initialize chromadb.PersistentClient here

    def upsert(
        self, chunks: list[Chunk], embeddings: list[list[float]]
    ) -> None:
        """Insert or update chunks. Idempotent — re-ingesting the same PDF does not duplicate."""
        raise NotImplementedError("Phase 1")

    def query(
        self, embedding: list[float], top_k: int
    ) -> list[RetrievedChunk]:
        """Query for top-k similar chunks by embedding."""
        raise NotImplementedError("Phase 1")

    def get_by_id(self, chunk_id: str) -> Chunk | None:
        """O(1) lookup via collection.get(ids=[chunk_id])."""
        raise NotImplementedError("Phase 1")
