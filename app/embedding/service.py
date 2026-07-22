"""Embedding service Protocol and MiniLM implementation.

The Protocol and its first concrete implementation live together in this
file — no centralized protocols.py dumping ground. Import the Protocol
for type hints; import the implementation only at construction time.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from sentence_transformers import SentenceTransformer

from app.core.logging import get_logger

logger = get_logger(__name__)


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
        """Initialize SentenceTransformer model inside constructor.

        Args:
            model_name: HuggingFace sentence-transformer model name.
        """
        logger.info("Initializing MiniLMEmbeddingService", model_name=model_name)
        self._model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string.

        Args:
            text: Query string to embed.

        Returns:
            Embedding vector as a list of floats (384 floats for MiniLM).
        """
        embedding = self._model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return embedding.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document strings.

        Args:
            texts: List of document strings.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []
        embeddings = self._model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
        return embeddings.tolist()

    def max_sequence_length(self) -> int:
        """Return the model's maximum input token count (256 for MiniLM)."""
        return int(self._model.max_seq_length)
