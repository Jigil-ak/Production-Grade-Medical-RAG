"""FastAPI application entry point.

Local development: uvicorn app.main:app --reload
Do NOT use docker compose up locally — Docker Desktop daemon alone
commits ~1.2GB RAM, fatal on the 4GB budget.
"""

from fastapi import FastAPI

from app.core.constants import APP_NAME, APP_VERSION
from app.core.logging import setup_logging

# Configure structured logging on import — before any route is loaded.
setup_logging()

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="Production-grade medical document Q&A with citation enforcement.",
)


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Basic health check endpoint."""
    return {"status": "healthy", "version": APP_VERSION}
