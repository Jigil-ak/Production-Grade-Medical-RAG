"""Unit tests for Phase 4 maintenance, system health, and logging audit components."""

from fastapi.testclient import TestClient
import pytest

from app.api.routes import router
from app.core.config import Settings
from app.core.system_health import check_system_health, SystemHealthReport
from app.main import app
from scripts.audit_logging import audit_logging_completeness


client = TestClient(app)


class TestSystemHealth:
    """Test system health and memory ceiling monitoring."""

    def test_check_system_health_returns_valid_report(self) -> None:
        report = check_system_health()

        assert isinstance(report, SystemHealthReport)
        assert report.status in ("healthy", "degraded", "warning")
        assert report.process_rss_mb >= 0.0
        assert report.max_ram_mb == 4096
        assert report.ram_usage_percent >= 0.0

    def test_health_system_endpoint_status_200(self) -> None:
        response = client.get("/health/system")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("healthy", "degraded", "warning")
        assert "process_rss_mb" in data
        assert "max_ram_mb" in data
        assert data["max_ram_mb"] == 4096


class TestLoggingAudit:
    """Test logging audit completeness script."""

    def test_logging_audit_completeness(self) -> None:
        is_complete, findings = audit_logging_completeness("./app")
        assert is_complete is True
        assert len(findings) == 0
