"""Core data types for the Medical RAG pipeline.

These models are the source of truth for field names across the entire
codebase. Define new types here BEFORE writing any Protocol or service
that references them. Locked field names prevent downstream inconsistency
(e.g. `source` vs `source_filename`).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A text chunk extracted from a source document."""

    chunk_id: str = Field(..., description="SHA-256 hash (first 16 hex chars) of filename+char_start+char_end")
    source_filename: str = Field(..., description="Original PDF filename")
    page_number: int = Field(..., description="1-indexed page number in the source PDF")
    char_start: int = Field(..., ge=0, description="Start character offset in the page's extracted text")
    char_end: int = Field(..., gt=0, description="End character offset in the page's extracted text")
    chunk_text: str = Field(..., min_length=1, description="The actual text content of the chunk")


class RetrievedChunk(Chunk):
    """A chunk returned from a retrieval query, with score and method."""

    score: float = Field(..., description="Similarity or relevance score from retrieval")
    retrieval_method: str = Field(
        ...,
        description="Retrieval method used: 'vector', 'bm25', or 'hybrid'",
    )


class Citation(BaseModel):
    """A citation linking part of the LLM answer to a source chunk."""

    chunk_id: str = Field(..., description="ID of the cited chunk")
    source_filename: str = Field(..., description="Source PDF filename")
    page_number: int = Field(..., description="Page number in the source PDF")
    quoted_text: str = Field(..., description="Exact text from the chunk supporting this citation")


class QueryResult(BaseModel):
    """Full result of a /query request, including answer, citations, and enforcement status."""

    answer: str | None = Field(None, description="Generated answer, None if declined")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence score")
    citations: list[Citation] = Field(default_factory=list, description="Source citations")
    supporting_chunks: int = Field(0, description="Number of chunks that passed support threshold")
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Claims from the answer not supported by retrieved chunks",
    )
    status: str = Field(
        ...,
        description="Enforcement outcome: 'answered', 'partial', or 'declined'",
    )
    prompt_version: str = Field(..., description="Prompt version used for generation")


class PromptTemplate(BaseModel):
    """A versioned prompt template loaded from YAML."""

    name: str = Field(..., description="Prompt template name")
    version: str = Field(..., description="Semantic version string")
    system_prompt: str = Field(..., description="System prompt content")
    user_template: str = Field(..., description="User prompt template with {placeholders}")
    description: str = Field("", description="Human-readable description of this prompt")
    created_date: str = Field(..., description="ISO-8601 date string")
