"""Custom exception hierarchy for the Medical RAG pipeline.

Keep exceptions specific enough to be caught meaningfully in API routes
(e.g. IngestionError → 4xx, not a bare 500) but not so granular that
every module invents its own hierarchy.
"""


class MedicalRAGError(Exception):
    """Base exception for all Medical RAG errors."""


class IngestionError(MedicalRAGError):
    """Raised on PDF parse failures, corrupt files, or chunk creation errors."""


class RetrievalError(MedicalRAGError):
    """Raised on vector store or BM25 query failures."""


class CitationValidationError(MedicalRAGError):
    """Raised when a cited chunk_id cannot be resolved or is fabricated."""
