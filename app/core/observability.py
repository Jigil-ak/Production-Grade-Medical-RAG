"""Langfuse Cloud observability and tracing wrapper.

LANGFUSE_HOST must point at cloud.langfuse.com ONLY.
Self-hosted Langfuse requires Postgres + Docker daemon (~1.2GB RAM),
which is a guaranteed OOM on the 4GB budget.

Gracefully handles unconfigured API keys during offline development.
"""

from __future__ import annotations

from typing import Any

from langfuse import Langfuse

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_langfuse_client: Langfuse | None = None


def get_langfuse() -> Langfuse | None:
    """Get or initialize Langfuse client instance.

    Returns None if public/secret keys are unconfigured.
    """
    global _langfuse_client

    if _langfuse_client is not None:
        return _langfuse_client

    settings = get_settings()
    pub_key = settings.langfuse_public_key
    sec_key = settings.langfuse_secret_key.get_secret_value()

    if not pub_key or not sec_key:
        logger.debug("Langfuse keys unconfigured, tracing disabled")
        return None

    try:
        _langfuse_client = Langfuse(
            public_key=pub_key,
            secret_key=sec_key,
            host=settings.langfuse_host,
        )
        logger.info("Langfuse Cloud tracing initialized", host=settings.langfuse_host)
        return _langfuse_client
    except Exception as e:
        logger.warn("Failed to initialize Langfuse Cloud tracing", error=str(e))
        return None


def trace_query_event(
    question: str,
    prompt_version: str,
    retrieved_chunk_ids: list[str],
    latency_ms: float,
    status: str,
) -> None:
    """Record query execution trace to Langfuse Cloud.

    Args:
        question: User medical question.
        prompt_version: Active prompt template version string.
        retrieved_chunk_ids: List of retrieved chunk IDs.
        latency_ms: Query execution latency in milliseconds.
        status: Enforcement outcome status ('answered', 'partial', 'declined').
    """
    client = get_langfuse()
    if client is None:
        return

    try:
        trace = client.trace(
            name="medical_rag_query",
            metadata={
                "prompt_version": prompt_version,
                "retrieved_count": len(retrieved_chunk_ids),
                "chunk_ids": retrieved_chunk_ids,
                "latency_ms": latency_ms,
                "status": status,
            },
            input={"question": question},
        )
        trace.event(
            name="query_completion",
            metadata={"status": status, "latency_ms": latency_ms},
        )
    except Exception as e:
        logger.warn("Failed to log trace to Langfuse", error=str(e))
