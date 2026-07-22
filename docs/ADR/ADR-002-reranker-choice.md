# ADR-002: Reranker Model Choice (TinyBERT vs MiniLM-L-6)

## Status
Accepted

## Context

In Phase 2, we introduce cross-encoder reranking to re-order candidate chunks after Reciprocal Rank Fusion (RRF) hybrid retrieval. Two main options were evaluated under our resource constraints:

1. **`cross-encoder/ms-marco-TinyBERT-L-2-v2`**: 14MB model size, ~2-layer Transformer encoder.
2. **`cross-encoder/ms-marco-MiniLM-L-6-v2`**: ~250MB model size, 6-layer Transformer encoder.

The system runs under a hard **~4GB RAM ceiling** on CPU. The Python process already hosts:
- `all-MiniLM-L6-v2` embedding model (~80MB)
- `ChromaDB` in-process vector store & HNSW index
- `bm25s` in-memory index
- FastAPI application & PyMuPDF PDF loader

Loading a ~250MB cross-encoder model alongside Chroma and PyMuPDF risks triggering OS memory paging or swapping, causing unpredictable latency spikes.

## Decision

Use **`cross-encoder/ms-marco-TinyBERT-L-2-v2` (14MB)** as the default reranker model.

- Extremely small footprint (14MB vs 250MB).
- Fast CPU inference (<15ms per rerank pass of top candidates).
- Captures the majority of cross-encoder precision gains over pure RRF retrieval.

## Consequences

### Benefits
- Operates comfortably within the 4GB RAM budget with zero swapping risk.
- Minimal inference latency impact on CPU.

### Tradeoffs
- Slightly lower precision on fine-grained sentence pairs compared to the larger 6-layer MiniLM reranker.
- Documented as a future scaling option if system memory is expanded.
