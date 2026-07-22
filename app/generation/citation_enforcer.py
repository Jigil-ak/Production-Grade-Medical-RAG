"""Citation enforcement with corrected support-scoring algorithm.

CRITICAL — Support scoring uses max-over-sentences, NOT whole-chunk cosine:
1. Split cited chunk into sentences (nltk punkt).
2. Embed each sentence individually with MiniLM.
3. Embed the LLM's claim sentence from the answer.
4. Take MAX cosine similarity between claim and any sentence in that cited chunk.

This prevents the ~200 tokens of surrounding text from diluting the specific
cited sentence. Threshold: settings.citation.support_threshold (0.65 default).
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path

import nltk

from app.core.logging import get_logger
from app.core.types import Citation, QueryResult, RetrievedChunk
from app.embedding.service import EmbeddingService

logger = get_logger(__name__)


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two float vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot_prod = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_prod / (norm_a * norm_b)


class CitationEnforcer:
    """Enforces citation validation and MAX-over-sentences support scoring."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        support_threshold: float = 0.65,
        nltk_data_dir: str = "./data/processed/nltk_data",
    ) -> None:
        """Initialize citation enforcer.

        Args:
            embedding_service: EmbeddingService implementation.
            support_threshold: Minimum cosine similarity required for support (default 0.65).
            nltk_data_dir: Local path to store NLTK data (punkt).
        """
        self.embedding_service = embedding_service
        self.support_threshold = support_threshold
        self._nltk_data_dir = Path(nltk_data_dir)
        self._init_nltk()

    def _init_nltk(self) -> None:
        """Ensure NLTK punkt tokenizer is downloaded to local data/processed directory."""
        self._nltk_data_dir.mkdir(parents=True, exist_ok=True)
        nltk.data.path.append(str(self._nltk_data_dir))
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            try:
                nltk.download("punkt", download_dir=str(self._nltk_data_dir), quiet=True)
            except Exception as e:
                logger.warn("NLTK punkt download failed, regex sentence splitter fallback will be used", error=str(e))

    def split_sentences(self, text: str) -> list[str]:
        """Split text into sentences using NLTK punkt with regex fallback.

        Note on regex fallback: Known to mis-split on abbreviations common in medical
        text (e.g., 'Dr.', 'e.g.', 'vs.'), so NLTK punkt is preferred whenever available.
        """
        if not text.strip():
            return []
        try:
            return [s.strip() for s in nltk.tokenize.sent_tokenize(text) if s.strip()]
        except Exception:
            # Fallback regex splitter
            sentences = re.split(r"(?<=[.!?])\s+", text)
            return [s.strip() for s in sentences if s.strip()]

    def enforce_citations(
        self,
        raw_answer: str | None,
        raw_citations: list[Citation],
        retrieved_chunks: list[RetrievedChunk],
        prompt_version: str,
    ) -> QueryResult:
        """Enforce citation validity and compute sentence-level support scoring.

        Args:
            raw_answer: Generated answer text from LLM.
            raw_citations: List of Citation objects claimed by LLM.
            retrieved_chunks: List of RetrievedChunk objects passed in LLM context.
            prompt_version: Prompt version identifier.

        Returns:
            QueryResult containing verified citations, support score, and status.
        """
        if not raw_answer or not raw_citations or not retrieved_chunks:
            return QueryResult(
                answer=None,
                confidence=0.0,
                citations=[],
                supporting_chunks=0,
                unsupported_claims=["No valid supporting context or citations found."],
                status="declined",
                prompt_version=prompt_version,
            )

        # 1. Map retrieved chunks by chunk_id for O(1) lookup
        retrieved_map = {c.chunk_id: c for c in retrieved_chunks}

        # Step A: Filter out fabricated chunk_ids not in retrieved context
        valid_chunk_citations: list[Citation] = []
        for cite in raw_citations:
            if cite.chunk_id in retrieved_map:
                valid_chunk_citations.append(cite)
            else:
                logger.warn("Fabricated chunk_id detected and discarded", chunk_id=cite.chunk_id)

        if not valid_chunk_citations:
            return QueryResult(
                answer=None,
                confidence=0.0,
                citations=[],
                supporting_chunks=0,
                unsupported_claims=["All cited chunk_ids were fabricated or invalid."],
                status="declined",
                prompt_version=prompt_version,
            )

        # Step B: MAX-over-sentences support scoring
        answer_sentences = self.split_sentences(raw_answer)
        answer_sentence_embeddings = self.embedding_service.embed_documents(answer_sentences)

        supported_citations: list[Citation] = []
        unsupported_claims: list[str] = []
        max_scores: list[float] = []

        for cite in valid_chunk_citations:
            cited_chunk = retrieved_map[cite.chunk_id]

            # Split cited chunk into sentences
            chunk_sentences = self.split_sentences(cited_chunk.chunk_text)
            if not chunk_sentences:
                chunk_sentences = [cited_chunk.chunk_text]

            chunk_sentence_embeddings = self.embedding_service.embed_documents(chunk_sentences)

            # Compute MAX cosine similarity between any sentence in claim/answer and any sentence in chunk
            best_similarity = 0.0

            for claim_vec in answer_sentence_embeddings:
                for chunk_vec in chunk_sentence_embeddings:
                    sim = _cosine_similarity(claim_vec, chunk_vec)
                    if sim > best_similarity:
                        best_similarity = sim

            max_scores.append(best_similarity)

            if best_similarity >= self.support_threshold:
                supported_citations.append(cite)
            else:
                unsupported_claims.append(
                    f"Citation '{cite.chunk_id}' score {best_similarity:.3f} below threshold {self.support_threshold}"
                )

        supporting_count = len(supported_citations)

        # Determine overall outcome status
        if supporting_count == len(valid_chunk_citations):
            status = "answered"
            confidence = round(sum(max_scores) / len(max_scores), 4) if max_scores else 0.9
        elif supporting_count > 0:
            status = "partial"
            confidence = round(sum(max_scores) / len(max_scores), 4) if max_scores else 0.5
        else:
            status = "declined"
            confidence = 0.0

        if status != "answered":
            logger.warn(
                "Non-answered citation outcome",
                status=status,
                confidence=confidence,
                supporting_count=supporting_count,
                unsupported_claims=unsupported_claims,
            )

        return QueryResult(
            answer=raw_answer if status != "declined" else None,
            confidence=confidence,
            citations=supported_citations,
            supporting_chunks=supporting_count,
            unsupported_claims=unsupported_claims,
            status=status,
            prompt_version=prompt_version,
        )
