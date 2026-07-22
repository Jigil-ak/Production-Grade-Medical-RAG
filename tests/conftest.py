"""Shared test fixtures for the Medical RAG test suite."""

import os

import pytest


@pytest.fixture(autouse=True)
def _set_test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set required environment variables for all tests."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    monkeypatch.setenv("ENVIRONMENT", "dev")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", "./data/processed/chroma")
