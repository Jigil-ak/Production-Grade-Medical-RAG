"""Hybrid retrieval via Reciprocal Rank Fusion (RRF).

Phase 2 implementation. Merges vector search and BM25 results:
  score(chunk) = sum over retrievers of 1 / (rank + k)
k=60 default, configurable via settings.retrieval.rrf_k.
"""

# Phase 2 implementation
