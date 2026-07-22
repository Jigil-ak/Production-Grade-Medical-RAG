"""PDF document loader using PyMuPDF (fitz).

Extracts text via get_text('blocks') for better reading order in
multi-column medical textbook layouts. Preserves page number per block.
No unstructured/poppler/tesseract — too heavy for the RAM budget.
"""

# Phase 1 implementation
