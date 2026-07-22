"""System health and resource limit monitor.

Monitors process RSS memory against settings.max_ram_mb (config-driven,
default 4096MB reflecting 4GB RAM ceiling assumption) and tracks local disk storage.
"""

from __future__ import annotations

import os

from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def _get_process_rss_mb() -> float:
    """Get current process Resident Set Size (RSS) memory in Megabytes."""
    try:
        import psutil

        process = psutil.Process(os.getpid())
        return round(process.memory_info().rss / (1024 * 1024), 2)
    except ImportError:
        # Fallback using resource module on Unix or estimates
        try:
            import resource

            usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # KB on Linux, Bytes on macOS
            return round(usage / 1024.0, 2)
        except Exception:
            return 0.0


def _get_dir_size_mb(dir_path: str) -> float:
    """Calculate total size of files inside a directory in Megabytes."""
    total_bytes = 0
    try:
        path = os.path.abspath(dir_path)
        if os.path.exists(path):
            for root, _, files in os.walk(path):
                for f in files:
                    fp = os.path.join(root, f)
                    if os.path.isfile(fp):
                        total_bytes += os.path.getsize(fp)
    except Exception as e:
        logger.warn("Failed to measure directory size", path=dir_path, error=str(e))

    return round(total_bytes / (1024 * 1024), 2)


class SystemHealthReport(BaseModel):
    """Structured report of process memory, storage, and system status."""

    status: str = Field(..., description="'healthy', 'degraded', or 'warning'")
    process_rss_mb: float = Field(..., description="Process memory RSS in MB")
    max_ram_mb: int = Field(..., description="Configured RAM ceiling in MB")
    ram_usage_percent: float = Field(..., description="Process RSS as % of max_ram_mb")
    storage_processed_mb: float = Field(..., description="Chroma & BM25 processed data size in MB")
    storage_raw_mb: float = Field(..., description="Raw PDF data size in MB")
    environment: str = Field(..., description="Active environment (dev | production)")


def check_system_health() -> SystemHealthReport:
    """Check process memory against settings.max_ram_mb and calculate disk usage.

    Returns:
        SystemHealthReport model.
    """
    settings = get_settings()
    rss_mb = _get_process_rss_mb()
    max_mb = settings.max_ram_mb

    usage_pct = round((rss_mb / max_mb) * 100.0, 2) if max_mb > 0 else 0.0

    # Determine health status
    if usage_pct > 90.0:
        health_status = "warning"
        logger.warn("High RAM usage detected", rss_mb=rss_mb, max_ram_mb=max_mb, pct=usage_pct)
    elif usage_pct > 75.0:
        health_status = "degraded"
    else:
        health_status = "healthy"

    processed_mb = _get_dir_size_mb(settings.chroma_persist_dir)
    raw_mb = _get_dir_size_mb("./data/raw")

    return SystemHealthReport(
        status=health_status,
        process_rss_mb=rss_mb,
        max_ram_mb=max_mb,
        ram_usage_percent=usage_pct,
        storage_processed_mb=processed_mb,
        storage_raw_mb=raw_mb,
        environment=settings.environment,
    )
