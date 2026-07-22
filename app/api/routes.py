"""FastAPI API routes for the Medical RAG system.

Endpoints:
  POST /ingest  — Ingest PDFs from /data/raw
  POST /query   — Question answering with citations
  GET  /chunk/{chunk_id} — O(1) chunk lookup by ID
"""

# Phase 1 implementation
