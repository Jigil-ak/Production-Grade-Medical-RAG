"""Text cleaning and normalization module for PyMuPDF raw block text.

Runs BEFORE offset computation and chunking. Order matters: clean raw text
first, THEN compute char_start/char_end against cleaned text — cleaning after
offsets are assigned shifts downstream citations.

Features:
- De-hyphenation of line-break-split words (e.g. "hyper-\nglycemia" -> "hyperglycemia").
- Unicode ligature normalization (ﬁ -> fi, ﬂ -> fl).
- Whitespace normalization while preserving paragraph breaks (\\n\\n).
- Detection and removal of repeated header/footer boilerplate (>30% page frequency).
- Filtering of empty / near-empty blocks (< 10 chars).
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

from app.core.logging import get_logger

logger = get_logger(__name__)

# Boilerplate threshold: text appearing on > 30% of pages is treated as header/footer
BOILERPLATE_PAGE_THRESHOLD = 0.30

# Minimum characters required for a block to be kept after cleaning
MIN_BLOCK_LENGTH = 10


@dataclass(frozen=True)
class RawBlock:
    """Raw uncleaned block extracted from a PyMuPDF page."""

    page_number: int
    text: str


def dehyphenate_text(text: str) -> str:
    """De-hyphenate words split across line breaks.

    Only merges when a hyphen is immediately followed by a newline and a lowercase letter.
    Does NOT merge legitimate hyphenated terms like "insulin-dependent".

    Example:
        "hyper-\\nglycemia" -> "hyperglycemia"
        "insulin-dependent" -> "insulin-dependent"
    """
    # Regex: word character + hyphen + newline + lowercase letter
    pattern = re.compile(r"(\b\w+)-\n([a-z]\w*)", re.UNICODE)
    return pattern.sub(r"\1\2", text)


def normalize_ligatures_and_whitespace(text: str) -> str:
    """Normalize Unicode ligatures and whitespace while preserving paragraph breaks (\\n\\n).

    Args:
        text: Input string.

    Returns:
        Cleaned text with ligatures expanded and uniform spacing.
    """
    if not text:
        return ""

    # 1. Expand Unicode ligatures (NFKD decomposition converts ﬁ -> fi, ﬂ -> fl, etc.)
    text = unicodedata.normalize("NFKD", text)

    # Explicit replacement fallback for common ligatures
    ligature_map = {
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬀ": "ff",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
        "ﬅ": "st",
        "ﬆ": "st",
    }
    for lig, replacement in ligature_map.items():
        text = text.replace(lig, replacement)

    # 2. Preserve paragraph breaks (\\n\\n) while normalizing spaces inside paragraphs
    # Split text into paragraphs by double newlines
    paragraphs = re.split(r"\n\s*\n", text)
    cleaned_paras: list[str] = []

    for para in paragraphs:
        # Replace newlines and multiple spaces within a paragraph with a single space
        cleaned_para = re.sub(r"[\r\n\t\f\v]+", " ", para)
        cleaned_para = re.sub(r" +", " ", cleaned_para).strip()
        if cleaned_para:
            cleaned_paras.append(cleaned_para)

    return "\n\n".join(cleaned_paras)


def identify_boilerplate_blocks(
    raw_blocks: list[RawBlock], total_pages: int, threshold: float = BOILERPLATE_PAGE_THRESHOLD
) -> set[str]:
    """Identify boilerplate text strings (headers/footers) appearing on > threshold % of pages.

    Args:
        raw_blocks: List of RawBlock objects.
        total_pages: Total number of pages in the PDF document.
        threshold: Percentage of pages threshold (default 0.30 = 30%).

    Returns:
        Set of normalized boilerplate strings to filter out.
    """
    if total_pages <= 1:
        return set()

    # Track unique pages on which each normalized block string appears
    block_page_occurrences: dict[str, set[int]] = {}

    for b in raw_blocks:
        # Normalize for comparison (lowercase, collapsed whitespace)
        norm_key = re.sub(r"\s+", " ", b.text.strip().lower())
        if not norm_key or len(norm_key) < 3:
            continue

        if norm_key not in block_page_occurrences:
            block_page_occurrences[norm_key] = set()
        block_page_occurrences[norm_key].add(b.page_number)

    boilerplate: set[str] = set()
    min_pages_required = total_pages * threshold

    for norm_key, pages in block_page_occurrences.items():
        if len(pages) > min_pages_required:
            boilerplate.add(norm_key)

    if boilerplate:
        logger.info(
            "Identified boilerplate headers/footers for removal",
            total_pages=total_pages,
            threshold=threshold,
            boilerplate_count=len(boilerplate),
        )

    return boilerplate


def clean_block_text(text: str) -> str:
    """Clean a single block of text: de-hyphenate, expand ligatures, normalize whitespace."""
    if not text:
        return ""
    text = dehyphenate_text(text)
    text = normalize_ligatures_and_whitespace(text)
    return text
