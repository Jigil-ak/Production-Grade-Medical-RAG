"""Embedding service Protocol and MiniLM implementation.

The Protocol and its first concrete implementation live together in this
file — no centralized protocols.py dumping ground. Import the Protocol
for type hints; import the implementation only at construction time.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingService(Protocol):
    """Interface for text embedding services."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document strings."""
        ...

    def max_sequence_length(self) -> int:
        """Return the model's maximum input token count."""
        ...


class MiniLMEmbeddingService:
    """all-MiniLM-L6-v2 embedding service via sentence-transformers.

    The SentenceTransformer model is loaded lazily inside __init__,
    NOT at module import time. If a network hiccup occurs during model
    download, this fails when the service is actually instantiated,
    not during .env validation and app startup.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        # Phase 1 fills in the actual SentenceTransformer load.
        # Lazy loading: model instantiated HERE, not at import time.
        self._model_name = model_name
        self._model = None  # SentenceTransformer loaded in Phase 1

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        raise NotImplementedError("Phase 1")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document strings."""
        raise NotImplementedError("Phase 1")

    def max_sequence_length(self) -> int:
        """Return the model's maximum input token count (256 for MiniLM)."""
        raise NotImplementedError("Phase 1")
