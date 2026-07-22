# Medical RAG — Production-Grade Medical Document Q&A

A production-grade Retrieval-Augmented Generation system for medical document question answering, built under hard resource constraints (~4GB RAM, CPU-only, free-tier APIs only).

## Architecture

See [docs/architecture.md](docs/architecture.md) for the full pipeline diagram and constraint rationale.

| Component | Technology | Rationale |
|-----------|------------|----------|
| PDF Extraction | PyMuPDF (fitz) | Lightweight, no Poppler/Tesseract dependency |
| Chunking | MiniLM tokenizer-aware | 220-240 tokens, never exceeds 256 limit |
| Embedding | all-MiniLM-L6-v2 | 384-dim, ~80MB, CPU-friendly |
| Vector Store | ChromaDB (persistent) | Embedded, no server, O(1) lookups ([ADR-001](docs/ADR/ADR-001-why-chroma-not-pinecone.md)) |
| Keyword Search | bm25s | Phase 2 — faster/lighter than rank_bm25 |
| Reranker | TinyBERT (14MB) | Phase 2 — not MiniLM-L-6 (~250MB) |
| LLM | Groq API (llama-3.3-70b) | No local weights, free tier |
| Observability | Langfuse Cloud | No self-hosted (OOM risk) |
| Eval | RAGAS | Phase 3 |

## Quick Start

### Prerequisites
- Python 3.10–3.12 (3.13 not supported — chromadb==0.5.0 lacks a wheel)
- A Groq API key ([console.groq.com](https://console.groq.com))

### Setup
```bash
# Clone and enter the repo
git clone <repo-url>
cd Production-Grade-Rag

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -e .

# Copy environment template and fill in your keys
copy .env.example .env
# Edit .env: add GROQ_API_KEY (required)
```

### Local Development

> ⚠️ **Do NOT use `docker compose up` locally.** The Docker Desktop daemon alone commits ~1.2GB of RAM before your Python process starts — fatal on a 4GB budget. Local dev runs `uvicorn` directly.

```bash
# Run the API server
uvicorn app.main:app --reload

# Run tests
pytest tests/ -v
```

### Data Setup

1. **Source PDF**: Place your medical PDF at `data/raw/` (you provide this — the system does not fetch or generate it).
2. **Ingest**: `POST /ingest` to process the PDF into chunks.
3. **Query**: `POST /query` with `{"question": "..."}` to get an answer with citations.
4. **Verify**: `GET /chunk/{chunk_id}` to look up any cited chunk by ID.

## Project Structure

```
app/
├── core/           # Types, config, exceptions, logging, constants
├── embedding/      # EmbeddingService Protocol + MiniLM implementation
├── ingestion/      # PDF loader, tokenizer-aware chunker
├── retrieval/      # VectorStore, BM25, hybrid fusion, reranker
├── generation/     # LLMClient, prompt provider, citation enforcer
├── api/            # FastAPI routes
├── config/prompts/ # YAML prompt templates (Phase 2)
└── eval/           # Golden dataset validation, RAGAS eval (Phase 3)
tests/
docs/
├── architecture.md
├── sequence_diagram.md
└── ADR/
data/
├── raw/            # Source PDF (you place this)
├── processed/      # ChromaDB, BM25 index (gitignored)
└── golden/         # Golden dataset (Phase 3, gitignored)
```

## Phase Status

| Phase | Status | Description |
|-------|--------|-------------|
| 0 | ✅ | Repo skeleton, config, types, logging, docs |
| 1 | ⬜ | Ingest → Chunk → Embed → Retrieve → Cite |
| 2 | ⬜ | Hybrid search, reranking, citation enforcement |
| 3 | ⬜ | Golden dataset validation, RAGAS eval, CI gate |
| 4 | ⬜ | Ops & maintenance guardrails |

## Design Decisions

See the [ADR directory](docs/ADR/) for documented architectural decisions:
- [ADR-001](docs/ADR/ADR-001-why-chroma-not-pinecone.md): Why ChromaDB over Pinecone
- ADR-002: Reranker model choice (Phase 2)
- ADR-003: Citation threshold recalibration (Phase 3)

## Storage Checkpoint — Phase 0

- `.venv`: ~1.35 GB (Torch CPU, sentence-transformers, ChromaDB, RAGAS, transformers, etc.)
- `data/`: ~0 MB (empty skeleton)
