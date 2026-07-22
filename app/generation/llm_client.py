"""LLM client Protocol and Groq implementation.

Protocol and implementation co-located — no centralized protocols.py.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """Interface for LLM generation."""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a response given system and user prompts."""
        ...


class GroqClient:
    """Groq API client for llama-3.3-70b-versatile.

    Reads GROQ_API_KEY from settings — never hardcoded or logged.
    """

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile") -> None:
        self._api_key = api_key
        self._model = model
        # Phase 1: initialize groq.Groq client here

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a response via Groq API."""
        raise NotImplementedError("Phase 1")
