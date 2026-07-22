"""Tests for core data types — validates schema is locked correctly."""

import pytest
from pydantic import ValidationError

from app.core.types import (
    Chunk,
    Citation,
    PromptTemplate,
    QueryResult,
    RetrievedChunk,
)


class TestChunk:
    """Validate Chunk model creation and field enforcement."""

    def test_valid_chunk(self) -> None:
        chunk = Chunk(
            chunk_id="abcdef0123456789",
            source_filename="medical.pdf",
            page_number=1,
            char_start=0,
            char_end=100,
            chunk_text="This is a test chunk of medical text.",
        )
        assert chunk.chunk_id == "abcdef0123456789"
        assert chunk.source_filename == "medical.pdf"
        assert chunk.page_number == 1
        assert chunk.char_start == 0
        assert chunk.char_end == 100

    def test_chunk_requires_all_fields(self) -> None:
        with pytest.raises(ValidationError):
            Chunk()  # type: ignore[call-arg]

    def test_chunk_text_cannot_be_empty(self) -> None:
        with pytest.raises(ValidationError):
            Chunk(
                chunk_id="abc",
                source_filename="f.pdf",
                page_number=1,
                char_start=0,
                char_end=1,
                chunk_text="",
            )

    def test_char_start_cannot_be_negative(self) -> None:
        with pytest.raises(ValidationError):
            Chunk(
                chunk_id="abc",
                source_filename="f.pdf",
                page_number=1,
                char_start=-1,
                char_end=10,
                chunk_text="text",
            )


class TestRetrievedChunk:
    """Validate RetrievedChunk extends Chunk with score and method."""

    def test_retrieved_chunk_has_score_and_method(self) -> None:
        rc = RetrievedChunk(
            chunk_id="abcdef0123456789",
            source_filename="medical.pdf",
            page_number=5,
            char_start=200,
            char_end=400,
            chunk_text="Some retrieved text.",
            score=0.85,
            retrieval_method="vector",
        )
        assert rc.score == 0.85
        assert rc.retrieval_method == "vector"


class TestCitation:
    """Validate Citation model."""

    def test_valid_citation(self) -> None:
        c = Citation(
            chunk_id="abcdef0123456789",
            source_filename="medical.pdf",
            page_number=3,
            quoted_text="Aspirin is used to reduce fever.",
        )
        assert c.chunk_id == "abcdef0123456789"
        assert c.page_number == 3


class TestQueryResult:
    """Validate QueryResult model."""

    def test_answered_result(self) -> None:
        qr = QueryResult(
            answer="Aspirin reduces fever.",
            confidence=0.92,
            citations=[
                Citation(
                    chunk_id="abc123",
                    source_filename="medical.pdf",
                    page_number=3,
                    quoted_text="Aspirin is used to reduce fever.",
                )
            ],
            supporting_chunks=1,
            unsupported_claims=[],
            status="answered",
            prompt_version="answer_generation_v1",
        )
        assert qr.status == "answered"
        assert qr.confidence == 0.92
        assert len(qr.citations) == 1

    def test_declined_result(self) -> None:
        qr = QueryResult(
            answer=None,
            confidence=0.1,
            citations=[],
            supporting_chunks=0,
            unsupported_claims=["No relevant information found"],
            status="declined",
            prompt_version="answer_generation_v1",
        )
        assert qr.status == "declined"
        assert qr.answer is None


class TestPromptTemplate:
    """Validate PromptTemplate model."""

    def test_valid_template(self) -> None:
        pt = PromptTemplate(
            name="answer_generation",
            version="v1",
            system_prompt="You are a medical assistant.",
            user_template="Context: {context}\nQuestion: {question}",
            description="Main answer generation prompt",
            created_date="2024-01-01",
        )
        assert pt.name == "answer_generation"
        assert pt.version == "v1"
