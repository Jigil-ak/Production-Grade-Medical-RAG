"""Application configuration via pydantic-settings.

All tuneable parameters live here. Never hardcode a k value, threshold,
or model name anywhere else in the codebase — always read from
`settings.retrieval`, `settings.citation`, or a top-level Settings field.
"""

from __future__ import annotations

from pydantic import BaseModel, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class RetrievalConfig(BaseModel):
    """Retrieval pipeline parameters — all k-values defined here."""

    vector_top_k: int = 20
    bm25_top_k: int = 20  # unused until Phase 2, defined now
    rerank_top_k: int = 5  # unused until Phase 2, defined now
    final_answer_k: int = 4
    rrf_k: int = 60  # Reciprocal Rank Fusion constant, Phase 2


class CitationConfig(BaseModel):
    """Citation enforcement parameters."""

    support_threshold: float = 0.65  # unused until Phase 2, defined now
    # 0.65, not 0.7 — max-over-sentences is a tighter comparison than
    # whole-chunk averaging, so 0.7 would cause too many false declines.
    # Recalibrate in Phase 3 once real eval data exists.


class Settings(BaseSettings):
    """Application settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # --- API keys (NEVER logged or hardcoded) ---
    groq_api_key: SecretStr

    # --- Storage ---
    chroma_persist_dir: str = "./data/processed/chroma"

    # --- Embedding ---
    embedding_model_name: str = "all-MiniLM-L6-v2"

    # --- Observability ---
    # LANGFUSE_HOST must point at cloud.langfuse.com ONLY.
    # Self-hosted Langfuse requires Postgres + Docker daemon (~1.2GB RAM),
    # which is a guaranteed OOM on the 4GB budget. Non-negotiable.
    langfuse_host: str = "https://cloud.langfuse.com"
    langfuse_public_key: str = ""
    langfuse_secret_key: SecretStr = SecretStr("")

    # --- Prompt versioning ---
    prompt_version: str = "answer_generation_v1"

    # --- Environment ---
    environment: str = "dev"  # dev | production

    # --- Sub-configs ---
    retrieval: RetrievalConfig = RetrievalConfig()
    citation: CitationConfig = CitationConfig()


def get_settings() -> Settings:
    """Factory for Settings — call once at startup, pass the instance around."""
    return Settings()  # type: ignore[call-arg]
