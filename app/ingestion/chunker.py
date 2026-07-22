"""Embedding-tokenizer-aware text chunker.

Uses the actual sentence-transformers tokenizer for the configured
EMBEDDING_MODEL_NAME to count tokens — NEVER tiktoken. tiktoken tokenizes
for OpenAI models, not MiniLM's WordPiece tokenizer, and MiniLM has a
hard 256-token input limit.

Target: 220-240 MiniLM tokens per chunk, 40-60 token overlap.
Prefers paragraph/sentence boundaries; hard token cutoffs only when a
paragraph itself exceeds the target.
"""

# Phase 1 implementation
