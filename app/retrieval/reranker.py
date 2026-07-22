"""Cross-encoder reranking using TinyBERT.

Phase 2 implementation. Uses cross-encoder/ms-marco-TinyBERT-L-2-v2 (14MB).
Explicitly NOT ms-marco-MiniLM-L-6-v2 (~250MB) — on a 4GB machine, that
plus Chroma plus the Python process risks swapping.
"""

# Phase 2 implementation
