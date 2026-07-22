"""BM25 keyword search index using bm25s.

Uses bm25s (not rank_bm25) — faster, lower memory, same algorithm,
right fit for the 4GB RAM budget.

Indexes the same chunk_ids as the vector store so results can be merged
directly by RRF in hybrid retrieval.
"""

from __future__ import annotations

from pathlib import Path

import bm25s
from bm25s.tokenization import Tokenizer

from app.core.exceptions import RetrievalError
from app.core.logging import get_logger
from app.core.types import Chunk, RetrievedChunk

logger = get_logger(__name__)


class BM25Index:
    """BM25 keyword index using bm25s library."""

    def __init__(self, index_dir: str = "./data/processed/bm25") -> None:
        """Initialize BM25 index and persistence directory.

        Args:
            index_dir: Directory path where BM25 index is stored/persisted.
        """
        self._index_dir = Path(index_dir)
        self._retriever: bm25s.BM25 | None = None
        self._chunk_map: dict[str, Chunk] = {}
        self._chunk_ids: list[str] = []

    def build_index(self, chunks: list[Chunk]) -> None:
        """Build BM25 index from a list of Chunk objects.

        TODO: For corpus > 10,000 chunks, revisit with a content-hash check
        (e.g., hash of all chunk_ids) to skip unnecessary rebuilds on startup
        once rebuild time becomes noticeable.

        Args:
            chunks: List of Chunk models extracted from ingestion.
        """
        if not chunks:
            logger.warn("Empty chunk list provided to BM25Index.build_index")
            return

        logger.info("Building BM25 index", chunk_count=len(chunks))

        self._chunk_map = {c.chunk_id: c for c in chunks}
        self._chunk_ids = [c.chunk_id for c in chunks]
        corpus_texts = [c.chunk_text for c in chunks]

        # Tokenize corpus texts using bm25s Tokenizer
        tokenizer = Tokenizer(stemmer=None)
        corpus_tokens = tokenizer.tokenize(corpus_texts, show_progress=False)

        # Create and index BM25 model
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)

        self._retriever = retriever

        # Persist index to disk
        self._save_index()
        logger.info("BM25 index built and persisted successfully", index_dir=str(self._index_dir))

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Search BM25 index for top-k matching chunks.

        Args:
            query: User query string.
            top_k: Number of results to retrieve (settings.retrieval.bm25_top_k).

        Returns:
            List of RetrievedChunk objects with BM25 score and method="bm25".
        """
        if self._retriever is None:
            # Try loading index from disk
            if not self._load_index():
                logger.warn("BM25 index is empty or not built yet")
                return []

        if not query.strip() or not self._chunk_ids:
            return []

        try:
            tokenizer = Tokenizer(stemmer=None)
            query_tokens = tokenizer.tokenize([query], show_progress=False)

            actual_k = min(top_k, len(self._chunk_ids))
            results, scores = self._retriever.retrieve(query_tokens, k=actual_k)

            retrieved: list[RetrievedChunk] = []

            if len(results) > 0 and len(results[0]) > 0:
                doc_indices = results[0]
                doc_scores = scores[0]

                for idx, score in zip(doc_indices, doc_scores):
                    if idx < len(self._chunk_ids):
                        cid = self._chunk_ids[idx]
                        if cid in self._chunk_map:
                            chunk = self._chunk_map[cid]
                            retrieved.append(
                                RetrievedChunk(
                                    chunk_id=chunk.chunk_id,
                                    source_filename=chunk.source_filename,
                                    page_number=chunk.page_number,
                                    char_start=chunk.char_start,
                                    char_end=chunk.char_end,
                                    chunk_text=chunk.chunk_text,
                                    score=round(float(score), 4),
                                    retrieval_method="bm25",
                                )
                            )

            return retrieved

        except Exception as e:
            logger.error("BM25 search failed", query=query, error=str(e))
            raise RetrievalError(f"BM25 search error: {e}") from e

    def _save_index(self) -> None:
        """Save BM25 index and chunk map to disk."""
        if self._retriever is not None:
            self._index_dir.mkdir(parents=True, exist_ok=True)
            self._retriever.save(str(self._index_dir), corpus=None)

    def _load_index(self) -> bool:
        """Load BM25 index from disk if present."""
        if self._index_dir.exists() and (self._index_dir / "data.json").exists():
            try:
                self._retriever = bm25s.BM25.load(str(self._index_dir), load_corpus=False)
                return True
            except Exception as e:
                logger.warn("Failed to load BM25 index from disk", error=str(e))
                return False
        return False
