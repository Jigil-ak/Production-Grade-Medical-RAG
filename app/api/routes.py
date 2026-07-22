"""FastAPI API routes for the Medical RAG system.

Endpoints:
  POST /ingest  — Ingest PDFs from /data/raw
  POST /query   — Question answering with citations
  GET  /chunk/{chunk_id} — O(1) chunk lookup by ID
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.core.exceptions import IngestionError
from app.core.logging import get_logger
from app.core.types import Chunk, QueryResult
from app.embedding.service import MiniLMEmbeddingService
from app.generation.llm_client import GroqClient
from app.ingestion.chunker import TokenizerAwareChunker
from app.ingestion.loader import load_pdf
from app.retrieval.retriever import VectorRetriever
from app.retrieval.store import ChromaStore

logger = get_logger(__name__)
router = APIRouter()

# Singletons initialized lazily on first API call or startup
_embedding_service: MiniLMEmbeddingService | None = None
_vector_store: ChromaStore | None = None
_llm_client: GroqClient | None = None


def get_services() -> tuple[MiniLMEmbeddingService, ChromaStore, GroqClient]:
    """Get or initialize core service instances."""
    global _embedding_service, _vector_store, _llm_client

    settings = get_settings()

    if _embedding_service is None:
        _embedding_service = MiniLMEmbeddingService(model_name=settings.embedding_model_name)

    if _vector_store is None:
        _vector_store = ChromaStore(persist_dir=settings.chroma_persist_dir)

    if _llm_client is None:
        api_key = settings.groq_api_key.get_secret_value()
        _llm_client = GroqClient(api_key=api_key)

    return _embedding_service, _vector_store, _llm_client


class IngestResponse(BaseModel):
    """Structured response for /ingest endpoint."""

    status: str = Field(..., description="'success', 'partial', or 'failed'")
    chunks_added: int = Field(..., description="Number of new chunks added")
    chunks_skipped_duplicate: int = Field(0, description="Number of duplicate chunks skipped")
    pages_processed: int = Field(..., description="Number of PDF pages processed")
    errors: list[str] = Field(default_factory=list, description="Error messages if any")


class QueryRequest(BaseModel):
    """Payload for /query endpoint."""

    question: str = Field(..., min_length=2, description="Medical question to ask")


@router.post("/ingest", response_model=IngestResponse)
async def ingest_documents() -> IngestResponse:
    """Ingest PDF documents placed in /data/raw into vector store.

    Returns structured status with chunks_added and pages_processed.
    Raises 400 Bad Request if PDF files are corrupt, unreadable, or missing.
    """
    start_time = time.time()
    raw_dir = Path("./data/raw")

    if not raw_dir.exists() or not raw_dir.is_dir():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Directory /data/raw does not exist.",
        )

    pdf_files = list(raw_dir.glob("*.pdf"))
    if not pdf_files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No PDF files found in /data/raw/ to ingest.",
        )

    embedding_service, vector_store, _ = get_services()
    chunker = TokenizerAwareChunker(model_name=get_settings().embedding_model_name)

    total_chunks_added = 0
    total_pages_processed = 0
    errors: list[str] = []

    for pdf_path in pdf_files:
        try:
            filename, blocks = load_pdf(pdf_path)
            if not blocks:
                errors.append(f"No readable text extracted from {filename}")
                continue

            max_page = max(b.page_number for b in blocks)
            total_pages_processed += max_page

            chunks = chunker.chunk_page_blocks(filename, blocks)
            if not chunks:
                errors.append(f"No chunks generated for {filename}")
                continue

            chunk_texts = [c.chunk_text for c in chunks]
            embeddings = embedding_service.embed_documents(chunk_texts)

            added = vector_store.upsert(chunks, embeddings)
            total_chunks_added += added

        except IngestionError as e:
            logger.error("Ingestion failed for file", path=str(pdf_path), error=str(e))
            errors.append(f"{pdf_path.name}: {e}")
        except Exception as e:
            logger.error("Unexpected ingestion error", path=str(pdf_path), error=str(e))
            errors.append(f"{pdf_path.name}: Unexpected error: {e}")

    latency_ms = round((time.time() - start_time) * 1000, 2)

    if errors and total_chunks_added == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"PDF Ingestion failed: {'; '.join(errors)}",
        )

    final_status = "partial" if errors else "success"

    logger.info(
        "Ingestion completed",
        status=final_status,
        chunks_added=total_chunks_added,
        pages_processed=total_pages_processed,
        errors=errors,
        latency_ms=latency_ms,
    )

    return IngestResponse(
        status=final_status,
        chunks_added=total_chunks_added,
        chunks_skipped_duplicate=0,
        pages_processed=total_pages_processed,
        errors=errors,
    )


@router.post("/query", response_model=QueryResult)
async def query_medical_rag(request: QueryRequest) -> QueryResult:
    """Ask a question and return answer + verified citations.

    Required fields logged on every request:
    request_id, chunk_ids, prompt_version, latency_ms.
    """
    start_time = time.time()
    settings = get_settings()

    embedding_service, vector_store, llm_client = get_services()
    retriever = VectorRetriever(embedding_service, vector_store)

    # 1. Retrieve top-k context chunks
    retrieved_chunks = retriever.retrieve(
        query=request.question, top_k=settings.retrieval.vector_top_k
    )

    chunk_ids = [c.chunk_id for c in retrieved_chunks]

    # 2. Generate answer with citations
    result = llm_client.generate_answer_with_citations(
        question=request.question,
        retrieved_chunks=retrieved_chunks,
        prompt_version=settings.prompt_version,
    )

    latency_ms = round((time.time() - start_time) * 1000, 2)

    logger.info(
        "Query request processed",
        question=request.question,
        chunk_ids=chunk_ids,
        prompt_version=settings.prompt_version,
        latency_ms=latency_ms,
        status=result.status,
        confidence=result.confidence,
    )

    return result


@router.get("/chunk/{chunk_id}", response_model=Chunk)
async def get_chunk_by_id(chunk_id: str) -> Chunk:
    """O(1) lookup of a chunk by its 16-hex SHA-256 ID.

    Returns 404 Not Found if the chunk_id does not exist.
    """
    _, vector_store, _ = get_services()
    chunk = vector_store.get_by_id(chunk_id)

    if chunk is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chunk ID '{chunk_id}' not found in vector store.",
        )

    return chunk
