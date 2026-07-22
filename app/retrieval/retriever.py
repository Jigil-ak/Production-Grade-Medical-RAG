"""Vector retrieval pipeline.

Embeds query, searches vector store for top-k chunks using
settings.retrieval.vector_top_k, and logs query details with structlog.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.core.types import RetrievedChunk
from app.embedding.service import EmbeddingService
from app.retrieval.store import VectorStore

logger = get_logger(__name__)


class VectorRetriever:
    """Retriever implementing vector similarity search."""

    def __init__(self, embedding_service: EmbeddingService, vector_store: VectorStore) -> None:
        """Initialize with embedding service and vector store implementations."""
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Retrieve top-k similar chunks for a given query.

        Args:
            query: User's search query string.
            top_k: Number of chunks to retrieve (from settings.retrieval.vector_top_k).

        Returns:
            List of RetrievedChunk models.
        """
        if not query.strip():
            logger.warn("Empty query provided to VectorRetriever")
            return []

        # 1. Embed query
        query_embedding = self.embedding_service.embed_query(query)

        # 2. Query vector store
        chunks = self.vector_store.query(query_embedding, top_k=top_k)

        # 3. Log retrieval context
        chunk_ids = [c.chunk_id for c in chunks]
        logger.info(
            "Vector retrieval complete",
            query=query,
            top_k=top_k,
            returned_count=len(chunks),
            chunk_ids=chunk_ids,
        )

        return chunks
