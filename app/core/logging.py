"""Structured logging setup via structlog.

Renderer is selected based on the ENVIRONMENT setting:
  - dev (default)  → ConsoleRenderer (human-readable)
  - production     → JSONRenderer (machine-parseable)

Required fields on every relevant log line from Phase 1 onward:
  request_id, chunk_ids, prompt_version, latency_ms

These are bound via structlog's context binding, not passed ad-hoc.
"""

from __future__ import annotations

import logging
import os

import structlog


def setup_logging(environment: str | None = None) -> None:
    """Configure structlog processors and renderer.

    Args:
        environment: 'dev' or 'production'. Falls back to ENVIRONMENT
                     env var, then defaults to 'dev'.
    """
    env = environment or os.getenv("ENVIRONMENT", "dev")

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if env == "production":
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Also configure stdlib logging so third-party libraries
    # (uvicorn, chromadb) don't dump unformatted output.
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Factory for structured loggers — use as get_logger(__name__)."""
    return structlog.get_logger(name)
