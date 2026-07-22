"""Tests for application configuration."""

import os

import pytest
from pydantic import ValidationError

from app.core.config import CitationConfig, RetrievalConfig, Settings


class TestRetrievalConfig:
    """Validate RetrievalConfig defaults match the spec."""

    def test_defaults(self) -> None:
        rc = RetrievalConfig()
        assert rc.vector_top_k == 20
        assert rc.bm25_top_k == 20
        assert rc.rerank_top_k == 5
        assert rc.final_answer_k == 4
        assert rc.rrf_k == 60


class TestCitationConfig:
    """Validate CitationConfig defaults match the spec."""

    def test_defaults(self) -> None:
        cc = CitationConfig()
        assert cc.support_threshold == 0.65


class TestSettings:
    """Validate Settings loads from environment."""

    def test_loads_from_env(self) -> None:
        """Settings should load GROQ_API_KEY from environment."""
        settings = Settings()  # type: ignore[call-arg]
        assert settings.groq_api_key.get_secret_value() == "test-key-not-real"

    def test_groq_api_key_required(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Settings must fail if GROQ_API_KEY is missing."""
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with pytest.raises(ValidationError):
            Settings(_env_file=None)  # type: ignore[call-arg]

    def test_default_values(self) -> None:
        settings = Settings()  # type: ignore[call-arg]
        assert settings.chroma_persist_dir == "./data/processed/chroma"
        assert settings.embedding_model_name == "all-MiniLM-L6-v2"
        assert settings.langfuse_host == "https://cloud.langfuse.com"
        assert settings.prompt_version == "answer_generation_v1"
        assert settings.environment == "dev"

    def test_retrieval_config_nested(self) -> None:
        settings = Settings()  # type: ignore[call-arg]
        assert settings.retrieval.vector_top_k == 20
        assert settings.retrieval.final_answer_k == 4

    def test_citation_config_nested(self) -> None:
        settings = Settings()  # type: ignore[call-arg]
        assert settings.citation.support_threshold == 0.65
