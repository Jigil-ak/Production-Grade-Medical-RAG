"""PDF document loader using PyMuPDF (fitz) with integrated text cleaning.

Extracts text via get_text('blocks') for better reading order in
multi-column medical textbook layouts.

Ingestion Order:
1. Extract raw blocks per page.
2. Filter boilerplate headers/footers (>30% page frequency).
3. De-hyphenate, expand ligatures, normalize whitespace.
4. THEN compute char_start/char_end against the CLEANED text.
5. Return PageBlock objects.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

from app.core.exceptions import IngestionError
from app.core.logging import get_logger
from app.ingestion.cleaner import (
    MIN_BLOCK_LENGTH,
    RawBlock,
    clean_block_text,
    identify_boilerplate_blocks,
)

logger = get_logger(__name__)


@dataclass(frozen=True)
class PageBlock:
    """A block of cleaned text extracted from a PDF page with accurate character offsets."""

    source_filename: str
    page_number: int  # 1-indexed
    char_start: int
    char_end: int
    text: str


def load_pdf(pdf_path: str | Path) -> tuple[str, list[PageBlock]]:
    """Load and extract cleaned text blocks from a PDF document using PyMuPDF.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        A tuple of (source_filename, list of PageBlock objects).

    Raises:
        IngestionError: If the file does not exist, is not a PDF, or is corrupt/encrypted.
    """
    path = Path(pdf_path)
    if not path.is_file():
        raise IngestionError(f"PDF file not found: {path}")

    filename = path.name

    try:
        doc = fitz.open(path)
    except Exception as e:
        logger.error("Failed to open PDF file", filename=filename, error=str(e))
        raise IngestionError(f"Failed to open or parse PDF file {filename}: {e}") from e

    if doc.is_encrypted:
        doc.close()
        raise IngestionError(f"PDF file is encrypted: {filename}")

    page_count = doc.page_count
    if page_count == 0:
        doc.close()
        raise IngestionError(f"PDF file is empty (0 pages): {filename}")

    # Step 1: Collect raw blocks across all pages
    raw_blocks: list[RawBlock] = []
    page_raw_map: dict[int, list[str]] = {}

    try:
        for page_idx in range(page_count):
            page_num = page_idx + 1  # 1-indexed page number
            page = doc.load_page(page_idx)
            raw_blocks_data = page.get_text("blocks")

            # Sort blocks by vertical position (y0), then horizontal (x0)
            text_blocks = [b for b in raw_blocks_data if len(b) >= 7 and b[6] == 0]
            text_blocks.sort(key=lambda b: (round(b[1], 1), round(b[0], 1)))

            page_texts: list[str] = []
            for b in text_blocks:
                t = b[4].strip()
                if t:
                    raw_blocks.append(RawBlock(page_number=page_num, text=t))
                    page_texts.append(t)

            page_raw_map[page_num] = page_texts

        doc.close()
    except Exception as e:
        if not doc.is_closed:
            doc.close()
        logger.error("Error reading blocks from PDF", filename=filename, error=str(e))
        raise IngestionError(f"Error processing PDF pages in {filename}: {e}") from e

    # Step 2: Identify boilerplate headers/footers appearing on >30% of pages
    boilerplate_keys = identify_boilerplate_blocks(raw_blocks, total_pages=page_count)

    # Step 3: Clean text and compute char_start / char_end against CLEANED text
    blocks: list[PageBlock] = []

    for page_num in range(1, page_count + 1):
        raw_texts = page_raw_map.get(page_num, [])
        current_offset = 0

        for raw_t in raw_texts:
            norm_key = re.sub(r"\s+", " ", raw_t.strip().lower())
            if norm_key in boilerplate_keys:
                continue  # Skip boilerplate running header/footer

            cleaned_text = clean_block_text(raw_t)
            if len(cleaned_text) < MIN_BLOCK_LENGTH:
                continue  # Skip near-empty leftover blocks

            char_start = current_offset
            char_end = current_offset + len(cleaned_text)
            current_offset = char_end + 1  # +1 boundary spacing

            blocks.append(
                PageBlock(
                    source_filename=filename,
                    page_number=page_num,
                    char_start=char_start,
                    char_end=char_end,
                    text=cleaned_text,
                )
            )

    logger.info(
        "PDF loaded and cleaned successfully",
        filename=filename,
        page_count=page_count,
        total_cleaned_blocks=len(blocks),
        boilerplate_removed=len(boilerplate_keys),
    )
    return filename, blocks
