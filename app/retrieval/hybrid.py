"""Hybrid retrieval via Reciprocal Rank Fusion (RRF).

Merges dense vector search results and sparse BM25 keyword search results:
  score(chunk) = sum over retrievers of 1 / (rank + k)
where k=60 default, configurable via settings.retrieval.rrf_k.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.core.types import RetrievedChunk
from app.retrieval.bm25_index import BM25Index
from app.retrieval.retriever import VectorRetriever

logger = get_logger(__name__)


class RRFHybridRetriever:
    """Hybrid retriever combining Vector and BM25 search with RRF rank fusion."""

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        bm25_index: BM25Index,
        rrf_k: int = 60,
    ) -> None:
        """Initialize hybrid retriever.

        Args:
            vector_retriever: VectorRetriever instance.
            bm25_index: BM25Index instance.
            rrf_k: Reciprocal Rank Fusion constant (default 60).
        """
        self.vector_retriever = vector_retriever
        self.bm25_index = bm25_index
        self.rrf_k = rrf_k

    def retrieve(
        self, query: str, vector_top_k: int, bm25_top_k: int
    ) -> list[RetrievedChunk]:
        """Retrieve and fuse candidate chunks using Reciprocal Rank Fusion.

        Args:
            query: Query string.
            vector_top_k: Top-k vector results to pull.
            bm25_top_k: Top-k BM25 results to pull.

        Returns:
            List of RetrievedChunk models ordered by RRF fusion score.
        """
        if not query.strip():
            return []

        # 1. Run vector search and BM25 search independently
        vector_results = self.vector_retriever.retrieve(query, top_k=vector_top_k)
        bm25_results = self.bm25_index.search(query, top_k=bm25_top_k)

        # 2. Map chunk_id to Chunk object and accumulate RRF scores
        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, RetrievedChunk] = {}

        # Process vector ranks (1-indexed)
        for rank, chunk in enumerate(vector_results, start=1):
            cid = chunk.chunk_id
            chunk_map[cid] = chunk
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rank + self.rrf_k))

        # Process BM25 ranks (1-indexed)
        for rank, chunk in enumerate(bm25_results, start=1):
            cid = chunk.chunk_id
            if cid not in chunk_map:
                chunk_map[cid] = chunk
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rank + self.rrf_k))

        # 3. Sort chunks by descending RRF score
        sorted_cids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

        fused_chunks: list[RetrievedChunk] = []
        for cid in sorted_cids:
            base_chunk = chunk_map[cid]
            fused_score = round(rrf_scores[cid], 6)

            fused_chunks.append(
                RetrievedChunk(
                    chunk_id=base_chunk.chunk_id,
                    source_filename=base_chunk.source_filename,
                    page_number=base_chunk.page_number,
                    char_start=base_chunk.char_start,
                    char_end=base_chunk.char_end,
                    chunk_text=base_chunk.chunk_text,
                    score=fused_score,
                    retrieval_method="hybrid",
                )
            )

        logger.info(
            "Hybrid RRF retrieval complete",
            query=query,
            vector_count=len(vector_results),
            bm25_count=len(bm25_results),
            fused_count=len(fused_chunks),
        )

        return fused_chunks
