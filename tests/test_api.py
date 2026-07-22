"""Integration tests for FastAPI endpoints: /health, /ingest, /query, and /chunk/{chunk_id}."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


class TestAPIRoutes:
    """Test API endpoints."""

    def test_health_check(self) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_chunk_not_found_returns_404(self) -> None:
        response = client.get("/chunk/non_existent_chunk_id")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
