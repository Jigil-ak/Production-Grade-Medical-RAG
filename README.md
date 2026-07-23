# Medical RAG — Production-Grade Medical Document Q&A

A production-grade Retrieval-Augmented Generation system for medical document question answering, built under hard resource constraints (~4GB RAM, CPU-only, free-tier APIs only).

## Architecture

See [docs/architecture.md](docs/architecture.md) for the full pipeline diagram and constraint rationale.

| Component | Technology | Rationale |
|-----------|------------|----------|
| PDF Extraction | PyMuPDF (fitz) | Lightweight, no Poppler/Tesseract dependency |
| Text Cleaning | De-hyphenation & Ligatures | `cleaner.py` cleans text BEFORE char offset calculation |
| Chunking | MiniLM tokenizer-aware | 220-240 tokens, never exceeds 256 limit |
| Embedding | all-MiniLM-L6-v2 | 384-dim, ~80MB, CPU-friendly |
| Vector Store | ChromaDB (persistent) | Embedded, no server, O(1) lookups ([ADR-001](docs/ADR/ADR-001-why-chroma-not-pinecone.md)) |
| Keyword Search | bm25s | Phase 2 — faster/lighter than rank_bm25 |
| Reranker | TinyBERT (14MB) | Phase 2 — not MiniLM-L-6 (~250MB) |
| LLM | Groq API (llama-3.3-70b) | No local weights, free tier |
| Observability | Langfuse Cloud | No self-hosted (OOM risk) |
| Eval | RAGAS | Phase 3 |

> [!NOTE]
> **Image & Diagram Tradeoff**: `get_text("blocks")` extracts layout text blocks without image rendering or OCR. Text baked directly into images or diagrams is invisible to this pipeline, as OCR (e.g., Tesseract/Poppler) is deliberately excluded to stay within the ~4GB RAM budget.

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

### Local Development (No Docker)

> ⚠️ **DO NOT run `docker compose up` locally.** The Docker Desktop daemon alone commits ~1.2GB of RAM before your Python process even starts, which is fatal against our hard ~4GB RAM budget. Docker is used for CI workflows only.
>
> Local development runs `uvicorn` directly:
> ```bash
> uvicorn app.main:app --reload
> ```

### Contributor Checklist

- Before committing changes to `data/golden/*.json`, run `validate_golden.py` locally (no `--schema-only` flag) against your locally-ingested Chroma store to confirm every `chunk_id` still resolves.

## Phase Status

| Phase | Status | Description |
|-------|--------|-------------|
| 0 | ✅ | Repo skeleton, config, types, logging, docs |
| 1 | ✅ | Ingest → Chunk → Embed → Retrieve → Cite |
| 2 | ✅ | Hybrid search, reranking, citation enforcement |
| 3 | ✅ | Golden dataset validation, RAGAS eval, CI gate |
| 4 | ✅ | Maintenance, system health monitoring & production readiness |

## Design Decisions

See the [ADR directory](docs/ADR/) for documented architectural decisions:
- [ADR-001](docs/ADR/ADR-001-why-chroma-not-pinecone.md): Why ChromaDB over Pinecone
- [ADR-002](docs/ADR/ADR-002-reranker-choice.md): Reranker model choice (TinyBERT 14MB vs MiniLM-L-6 250MB)
- [ADR-003](docs/ADR/ADR-003-citation-threshold-recalibration.md): Citation support threshold recalibration (0.65)

## Hybrid Search vs. Vector-Only Ranking

In Phase 2, hybrid retrieval combines dense vector search with sparse BM25 keyword search via Reciprocal Rank Fusion ($k=60$).

**Example Query**: `"achalasia diagnosis manometry"`

| Retrieval Strategy | Target Chunk Rank | Rationale |
|-------------------|-------------------|-----------|
| Vector-Only | **Rank 2** | Embedding vectors group generic gastrointestinal topics near the top |
| **Hybrid (RRF)** | **Rank 1 (Winner)** | BM25 keyword match for exact medical term `"achalasia"` boosts the rank to #1 |

## Storage Checkpoints

- **Phase 0**: `.venv` ~1.35 GB | `data/` ~0 MB
- **Phase 1**: `.venv` ~1.35 GB | `data/` ~15.5 MB (source PDF `Medical_book.pdf`)
- **Phase 2**: `.venv` ~1.35 GB | `data/` ~64.3 MB (Chroma persistent DB + BM25 index + NLTK data)
- **Phase 3 & 4**: `.venv` ~1.35 GB | `data/` ~64.3 MB (Full system ready for production)
