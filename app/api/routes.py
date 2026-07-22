"""FastAPI API routes for the Medical RAG system (Phase 2).

Endpoints:
  POST /ingest  — Ingest PDFs from /data/raw (Vector + BM25 index build)
  POST /query   — Hybrid retrieval -> Rerank -> LLM generation -> Citation enforcement
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
from app.generation.citation_enforcer import CitationEnforcer
from app.generation.llm_client import GroqClient
from app.generation.prompt_provider import YAMLPromptProvider
from app.ingestion.chunker import TokenizerAwareChunker
from app.ingestion.loader import load_pdf
from app.retrieval.bm25_index import BM25Index
from app.retrieval.hybrid import RRFHybridRetriever
from app.retrieval.reranker import TinyBERTReranker
from app.retrieval.retriever import VectorRetriever
from app.retrieval.store import ChromaStore

logger = get_logger(__name__)
router = APIRouter()

# Singletons initialized lazily on startup
_embedding_service: MiniLMEmbeddingService | None = None
_vector_store: ChromaStore | None = None
_bm25_index: BM25Index | None = None
_reranker: TinyBERTReranker | None = None
_prompt_provider: YAMLPromptProvider | None = None
_citation_enforcer: CitationEnforcer | None = None
_llm_client: GroqClient | None = None


def get_services() -> tuple[
    MiniLMEmbeddingService,
    ChromaStore,
    BM25Index,
    TinyBERTReranker,
    YAMLPromptProvider,
    CitationEnforcer,
    GroqClient,
]:
    """Get or initialize core Phase 2 service instances."""
    global _embedding_service, _vector_store, _bm25_index, _reranker
    global _prompt_provider, _citation_enforcer, _llm_client

    settings = get_settings()

    if _embedding_service is None:
        _embedding_service = MiniLMEmbeddingService(model_name=settings.embedding_model_name)

    if _vector_store is None:
        _vector_store = ChromaStore(persist_dir=settings.chroma_persist_dir)

    if _bm25_index is None:
        _bm25_index = BM25Index()

    if _reranker is None:
        _reranker = TinyBERTReranker()

    if _prompt_provider is None:
        _prompt_provider = YAMLPromptProvider()

    if _citation_enforcer is None:
        _citation_enforcer = CitationEnforcer(
            embedding_service=_embedding_service,
            support_threshold=settings.citation.support_threshold,
        )

    if _llm_client is None:
        api_key = settings.groq_api_key.get_secret_value()
        _llm_client = GroqClient(api_key=api_key)

    return (
        _embedding_service,
        _vector_store,
        _bm25_index,
        _reranker,
        _prompt_provider,
        _citation_enforcer,
        _llm_client,
    )


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
    """Ingest PDF documents from /data/raw into vector store and BM25 index."""
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

    embedding_service, vector_store, bm25_index, _, _, _, _ = get_services()
    chunker = TokenizerAwareChunker(model_name=get_settings().embedding_model_name)

    total_chunks_added = 0
    total_pages_processed = 0
    errors: list[str] = []
    all_ingested_chunks: list[Chunk] = []

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
            all_ingested_chunks.extend(chunks)

        except IngestionError as e:
            logger.error("Ingestion failed for file", path=str(pdf_path), error=str(e))
            errors.append(f"{pdf_path.name}: {e}")
        except Exception as e:
            logger.error("Unexpected ingestion error", path=str(pdf_path), error=str(e))
            errors.append(f"{pdf_path.name}: Unexpected error: {e}")

    # Build BM25 index from ingested chunks
    if all_ingested_chunks:
        bm25_index.build_index(all_ingested_chunks)

    latency_ms = round((time.time() - start_time) * 1000, 2)

    if errors and total_chunks_added == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"PDF Ingestion failed: {'; '.join(errors)}",
        )

    final_status = "partial" if errors else "success"

    logger.info(
        "Phase 2 Ingestion completed",
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
    """Execute Phase 2 pipeline: Hybrid Retrieve -> Rerank -> Generate -> Citation Enforce.

    Required fields logged on every request:
    request_id, chunk_ids, prompt_version, latency_ms.
    """
    start_time = time.time()
    settings = get_settings()

    (
        embedding_service,
        vector_store,
        bm25_index,
        reranker,
        prompt_provider,
        citation_enforcer,
        llm_client,
    ) = get_services()

    # 1. Hybrid Retrieval (Vector + BM25 via RRF)
    vector_retriever = VectorRetriever(embedding_service, vector_store)
    hybrid_retriever = RRFHybridRetriever(
        vector_retriever=vector_retriever,
        bm25_index=bm25_index,
        rrf_k=settings.retrieval.rrf_k,
    )

    fused_candidates = hybrid_retriever.retrieve(
        query=request.question,
        vector_top_k=settings.retrieval.vector_top_k,
        bm25_top_k=settings.retrieval.bm25_top_k,
    )

    # 2. Cross-Encoder Reranking (rerank down to rerank_top_k pool)
    reranked_chunks = reranker.rerank(
        query=request.question,
        chunks=fused_candidates,
        top_k=settings.retrieval.rerank_top_k,
    )

    # 3. Slice to final_answer_k candidates for LLM prompt context
    final_chunks = reranked_chunks[: settings.retrieval.final_answer_k]

    chunk_ids = [c.chunk_id for c in final_chunks]

    # 4. Prompt Versioning
    prompt_template = prompt_provider.get(
        name="answer_generation", version=settings.prompt_version
    )

    # 5. LLM Generation
    raw_result = llm_client.generate_answer_with_citations(
        question=request.question,
        retrieved_chunks=final_chunks,
        prompt_version=prompt_template.version,
    )

    # 6. Citation Enforcement (MAX-over-sentences support scoring)
    final_result = citation_enforcer.enforce_citations(
        raw_answer=raw_result.answer,
        raw_citations=raw_result.citations,
        retrieved_chunks=final_chunks,
        prompt_version=prompt_template.version,
    )

    latency_ms = round((time.time() - start_time) * 1000, 2)

    logger.info(
        "Phase 2 Query request processed",
        question=request.question,
        chunk_ids=chunk_ids,
        prompt_version=prompt_template.version,
        latency_ms=latency_ms,
        status=final_result.status,
        confidence=final_result.confidence,
    )

    return final_result


@router.get("/chunk/{chunk_id}", response_model=Chunk)
async def get_chunk_by_id(chunk_id: str) -> Chunk:
    """O(1) lookup of a chunk by its 16-hex SHA-256 ID."""
    _, vector_store, _, _, _, _, _ = get_services()
    chunk = vector_store.get_by_id(chunk_id)

    if chunk is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chunk ID '{chunk_id}' not found in vector store.",
        )

    return chunk
