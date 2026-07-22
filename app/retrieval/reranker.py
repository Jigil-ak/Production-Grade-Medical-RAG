"""Cross-encoder reranking using TinyBERT.

Uses cross-encoder/ms-marco-TinyBERT-L-2-v2 (14MB).
Explicitly NOT ms-marco-MiniLM-L-6-v2 (~250MB) — on a 4GB machine, that
plus Chroma plus the Python process risks swapping.
"""

from __future__ import annotations

from sentence_transformers import CrossEncoder

from app.core.logging import get_logger
from app.core.types import RetrievedChunk

logger = get_logger(__name__)


class TinyBERTReranker:
    """Cross-encoder reranker using 14MB TinyBERT model."""

    def __init__(
        self, model_name: str = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
    ) -> None:
        """Initialize CrossEncoder model.

        Args:
            model_name: CrossEncoder HuggingFace identifier.
        """
        logger.info("Initializing TinyBERTReranker", model_name=model_name)
        self._model_name = model_name
        self._model = CrossEncoder(model_name)

    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        """Score (query, chunk_text) pairs, re-sort, and keep top_k candidates.

        Args:
            query: Search query text.
            chunks: Candidate chunks retrieved from hybrid search.
            top_k: Number of reranked candidates to return (settings.retrieval.rerank_top_k).

        Returns:
            List of top_k RetrievedChunk objects sorted by cross-encoder score.
        """
        if not chunks or not query.strip():
            return []

        # Create (query, document) pairs for cross-encoder
        pairs = [(query, c.chunk_text) for c in chunks]

        # Predict cross-encoder relevance scores
        scores = self._model.predict(pairs, show_progress_bar=False)

        # Pair chunks with scores
        scored_chunks: list[tuple[float, RetrievedChunk]] = []
        for chunk, score in zip(chunks, scores):
            scored_chunks.append((float(score), chunk))

        # Sort descending by score
        scored_chunks.sort(key=lambda item: item[0], reverse=True)

        reranked: list[RetrievedChunk] = []
        for score, chunk in scored_chunks[:top_k]:
            reranked.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    source_filename=chunk.source_filename,
                    page_number=chunk.page_number,
                    char_start=chunk.char_start,
                    char_end=chunk.char_end,
                    chunk_text=chunk.chunk_text,
                    score=round(score, 4),
                    retrieval_method="reranked",
                )
            )

        logger.info(
            "Reranking complete",
            query=query,
            candidates_in=len(chunks),
            top_k_out=len(reranked),
        )

        return reranked
