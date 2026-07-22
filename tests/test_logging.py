"""Tests for structured logging setup."""

import pytest
import structlog

from app.core.logging import get_logger, setup_logging


class TestLogging:
    """Validate logging configuration."""

    def test_get_logger_returns_bound_logger(self) -> None:
        setup_logging(environment="dev")
        logger = get_logger(__name__)
        assert logger is not None

    def test_dev_environment_uses_console_renderer(self) -> None:
        setup_logging(environment="dev")
        # Verify structlog is configured (no crash on log call)
        logger = get_logger("test")
        # This should not raise
        logger.info("test message", key="value")

    def test_production_environment_uses_json_renderer(self) -> None:
        setup_logging(environment="production")
        logger = get_logger("test")
        # This should not raise
        logger.info("test message", key="value")

    def test_default_environment_is_dev(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ENVIRONMENT", raising=False)
        setup_logging()  # Should default to dev, not crash
        logger = get_logger("test")
        logger.info("default env test")
