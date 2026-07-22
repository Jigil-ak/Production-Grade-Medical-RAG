"""LLM client Protocol and Groq implementation.

Protocol and implementation co-located — no centralized protocols.py.
"""

from __future__ import annotations

import json
import re
from typing import Protocol, runtime_checkable

from groq import Groq

from app.core.constants import DEFAULT_LLM_MODEL
from app.core.logging import get_logger
from app.core.types import Citation, QueryResult, RetrievedChunk

logger = get_logger(__name__)


@runtime_checkable
class LLMClient(Protocol):
    """Interface for LLM generation."""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a raw string response given system and user prompts."""
        ...


class GroqClient:
    """Groq API client for llama-3.3-70b-versatile.

    Reads GROQ_API_KEY from settings — never hardcoded or logged.
    """

    def __init__(self, api_key: str, model: str = DEFAULT_LLM_MODEL) -> None:
        """Initialize Groq API client.

        Args:
            api_key: Groq API key secret string.
            model: Groq model name (default: llama-3.3-70b-versatile).
        """
        if not api_key:
            raise ValueError("Groq API key must be provided")

        self._model = model
        self._client = Groq(api_key=api_key)
        logger.info("Initialized GroqClient", model=self._model)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a completion from Groq API.

        Args:
            system_prompt: System prompt instructing the model behavior.
            user_prompt: User prompt containing retrieved context and question.

        Returns:
            Completion string output from the LLM.
        """
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=1024,
            )
            content = response.choices[0].message.content or ""
            return content
        except Exception as e:
            logger.error("Groq API call failed", model=self._model, error=str(e))
            raise RuntimeError(f"Groq API error: {e}") from e

    def generate_answer_with_citations(
        self, question: str, retrieved_chunks: list[RetrievedChunk], prompt_version: str
    ) -> QueryResult:
        """Generate structured answer with citations from retrieved context.

        Args:
            question: User question.
            retrieved_chunks: List of retrieved context chunks.
            prompt_version: Active prompt version identifier.

        Returns:
            QueryResult containing answer, citations, confidence, and status.
        """
        if not retrieved_chunks:
            return QueryResult(
                answer=None,
                confidence=0.0,
                citations=[],
                supporting_chunks=0,
                unsupported_claims=["No relevant source context was found."],
                status="declined",
                prompt_version=prompt_version,
            )

        # Build context block formatted with chunk_ids
        context_lines: list[str] = []
        chunk_map = {c.chunk_id: c for c in retrieved_chunks}

        for idx, chunk in enumerate(retrieved_chunks, 1):
            context_lines.append(
                f"--- [CHUNK {idx}] ---\n"
                f"chunk_id: {chunk.chunk_id}\n"
                f"source_filename: {chunk.source_filename}\n"
                f"page_number: {chunk.page_number}\n"
                f"content: {chunk.chunk_text}\n"
            )

        context_str = "\n".join(context_lines)

        system_prompt = (
            "You are a medical assistant answering questions strictly based on the provided source chunks.\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. Answer ONLY using information explicitly contained in the provided chunks.\n"
            "2. If the context does not contain sufficient information to answer the question, state clearly that you cannot answer based on the provided documents.\n"
            "3. Format your response as a valid JSON object with the following schema:\n"
            "{\n"
            '  "answer": "Your concise medical answer here.",\n'
            '  "citations": [\n'
            "    {\n"
            '      "chunk_id": "exact_chunk_id_cited",\n'
            '      "quoted_text": "short literal snippet supporting this point"\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "4. Do NOT fabricate chunk_ids. Only use chunk_ids listed in the provided chunks."
        )

        user_prompt = f"SOURCE CONTEXT:\n{context_str}\n\nUSER QUESTION: {question}"

        raw_response = self.generate(system_prompt, user_prompt)

        # Parse JSON response from LLM
        return self._parse_llm_json_response(
            raw_response, chunk_map, prompt_version
        )

    def _parse_llm_json_response(
        self,
        raw_response: str,
        chunk_map: dict[str, RetrievedChunk],
        prompt_version: str,
    ) -> QueryResult:
        """Parse JSON response from LLM output."""
        try:
            # Clean markdown code blocks if present
            cleaned = re.sub(r"^```(?:json)?\s*", "", raw_response.strip(), flags=re.MULTILINE)
            cleaned = re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE)

            data = json.loads(cleaned)
            answer = data.get("answer")
            raw_citations = data.get("citations", [])

            if not answer or "cannot answer" in answer.lower():
                return QueryResult(
                    answer=None,
                    confidence=0.0,
                    citations=[],
                    supporting_chunks=0,
                    unsupported_claims=["Context insufficient to answer question"],
                    status="declined",
                    prompt_version=prompt_version,
                )

            valid_citations: list[Citation] = []
            for cite in raw_citations:
                cid = cite.get("chunk_id", "").strip()
                quoted = cite.get("quoted_text", "").strip()

                if cid in chunk_map:
                    ref_chunk = chunk_map[cid]
                    valid_citations.append(
                        Citation(
                            chunk_id=cid,
                            source_filename=ref_chunk.source_filename,
                            page_number=ref_chunk.page_number,
                            quoted_text=quoted or ref_chunk.chunk_text[:100],
                        )
                    )

            status = "answered" if valid_citations else "partial"
            confidence = 0.9 if valid_citations else 0.5

            return QueryResult(
                answer=answer,
                confidence=confidence,
                citations=valid_citations,
                supporting_chunks=len(valid_citations),
                unsupported_claims=[],
                status=status,
                prompt_version=prompt_version,
            )

        except Exception as e:
            logger.warn("Failed to parse LLM JSON output, falling back to raw text", raw=raw_response, error=str(e))
            # Fallback: treat raw response as answer text
            return QueryResult(
                answer=raw_response,
                confidence=0.6,
                citations=[],
                supporting_chunks=0,
                unsupported_claims=[],
                status="partial",
                prompt_version=prompt_version,
            )
