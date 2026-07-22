"""Vector store Protocol and ChromaDB implementation.

Protocol and implementation co-located — no centralized protocols.py.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import chromadb

from app.core.constants import CHROMA_COLLECTION_NAME
from app.core.exceptions import RetrievalError
from app.core.logging import get_logger
from app.core.types import Chunk, RetrievedChunk

logger = get_logger(__name__)


@runtime_checkable
class VectorStore(Protocol):
    """Interface for vector storage and retrieval."""

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        """Insert or update chunks with their embeddings. Returns count of chunks upserted."""
        ...

    def query(self, embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        """Query for the top-k most similar chunks."""
        ...

    def get_by_id(self, chunk_id: str) -> Chunk | None:
        """O(1) lookup of a chunk by its ID. Returns None if not found."""
        ...

    def count(self) -> int:
        """Return total count of chunks stored in vector store."""
        ...


class ChromaStore:
    """ChromaDB vector store in embedded/persistent-directory mode.

    Uses CHROMA_PERSIST_DIR from settings. NOT server mode.
    chunk_id (SHA-256 hash, first 16 hex chars) is used directly as the
    Chroma document ID for O(1) lookups.
    """

    def __init__(
        self, persist_dir: str, collection_name: str = CHROMA_COLLECTION_NAME
    ) -> None:
        """Initialize ChromaDB PersistentClient and get or create collection.

        Args:
            persist_dir: Local filesystem path to persist Chroma DB.
            collection_name: Name of the Chroma collection.
        """
        logger.info("Initializing ChromaStore", persist_dir=persist_dir, collection_name=collection_name)
        try:
            self._client = chromadb.PersistentClient(path=persist_dir)
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as e:
            logger.error("Failed to initialize ChromaStore", error=str(e))
            raise RetrievalError(f"Failed to initialize ChromaDB at {persist_dir}: {e}") from e

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        """Insert or update chunks into Chroma. Idempotent — identical IDs overwrite cleanly.

        Args:
            chunks: List of Chunk models.
            embeddings: Parallel list of embedding vectors.

        Returns:
            Number of chunks upserted.
        """
        if not chunks:
            return 0

        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings")

        ids = [c.chunk_id for c in chunks]
        documents = [c.chunk_text for c in chunks]
        metadatas: list[dict[str, Any]] = [
            {
                "source_filename": c.source_filename,
                "page_number": c.page_number,
                "char_start": c.char_start,
                "char_end": c.char_end,
            }
            for c in chunks
        ]

        # Batch upsert to Chroma (respecting Chroma's max_batch_size limit)
        max_batch = getattr(self._client, "max_batch_size", 100)
        batch_size = min(100, max_batch)
        for i in range(0, len(chunks), batch_size):
            self._collection.upsert(
                ids=ids[i : i + batch_size],
                documents=documents[i : i + batch_size],
                embeddings=embeddings[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
            )

        logger.info("Upserted chunks to ChromaStore", count=len(chunks))
        return len(chunks)

    def query(self, embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        """Query for top-k similar chunks by embedding.

        Args:
            embedding: Query embedding vector.
            top_k: Number of nearest neighbors to retrieve.

        Returns:
            List of RetrievedChunk models ordered by relevance.
        """
        if self._collection.count() == 0:
            return []

        try:
            results = self._collection.query(
                query_embeddings=[embedding],
                n_results=min(top_k, self._collection.count()),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error("Chroma query failed", error=str(e))
            raise RetrievalError(f"Vector query failed: {e}") from e

        retrieved: list[RetrievedChunk] = []

        if results["ids"] and results["ids"][0]:
            chunk_ids = results["ids"][0]
            docs = results["documents"][0] if results["documents"] else []
            metas = results["metadatas"][0] if results["metadatas"] else []
            distances = results["distances"][0] if results["distances"] else []

            for cid, doc, meta, dist in zip(chunk_ids, docs, metas, distances):
                # Convert cosine distance to similarity score
                similarity = max(0.0, 1.0 - float(dist))

                retrieved.append(
                    RetrievedChunk(
                        chunk_id=cid,
                        source_filename=str(meta.get("source_filename", "")),
                        page_number=int(meta.get("page_number", 1)),
                        char_start=int(meta.get("char_start", 0)),
                        char_end=int(meta.get("char_end", 0)),
                        chunk_text=str(doc),
                        score=round(similarity, 4),
                        retrieval_method="vector",
                    )
                )

        return retrieved

    def get_by_id(self, chunk_id: str) -> Chunk | None:
        """O(1) lookup of a chunk by its document ID.

        Args:
            chunk_id: 16-hex SHA-256 chunk ID.

        Returns:
            Chunk model if found, None otherwise.
        """
        try:
            result = self._collection.get(ids=[chunk_id], include=["documents", "metadatas"])
        except Exception as e:
            logger.error("Chroma get_by_id failed", chunk_id=chunk_id, error=str(e))
            return None

        if not result["ids"] or len(result["ids"]) == 0:
            return None

        meta = result["metadatas"][0]
        doc = result["documents"][0]

        return Chunk(
            chunk_id=chunk_id,
            source_filename=str(meta.get("source_filename", "")),
            page_number=int(meta.get("page_number", 1)),
            char_start=int(meta.get("char_start", 0)),
            char_end=int(meta.get("char_end", 0)),
            chunk_text=str(doc),
        )

    def count(self) -> int:
        """Return total document count in collection."""
        return self._collection.count()
