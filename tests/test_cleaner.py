"""Unit tests for text cleaning, de-hyphenation, and boilerplate header/footer removal."""

from app.ingestion.cleaner import (
    RawBlock,
    clean_block_text,
    dehyphenate_text,
    identify_boilerplate_blocks,
    normalize_ligatures_and_whitespace,
)


class TestDehyphenation:
    """Validate de-hyphenation rules."""

    def test_dehyphenate_split_word(self) -> None:
        text = "Patients with hyper-\nglycemia require blood glucose monitoring."
        cleaned = dehyphenate_text(text)
        assert "hyperglycemia" in cleaned
        assert "hyper-\nglycemia" not in cleaned

    def test_preserve_legitimate_hyphenated_terms(self) -> None:
        """Legitimate hyphenated terms like 'insulin-dependent' must NOT be merged if not split across newline."""
        text = "Treatment for insulin-dependent diabetes and hyper-\nglycemia is essential."
        cleaned = dehyphenate_text(text)
        assert "insulin-dependent" in cleaned
        assert "hyperglycemia" in cleaned


class TestLigaturesAndWhitespace:
    """Validate ligature expansion and whitespace normalization."""

    def test_expand_ligatures(self) -> None:
        text = "ﬁrst page and ﬂuid retention"
        cleaned = normalize_ligatures_and_whitespace(text)
        assert "first page" in cleaned
        assert "fluid retention" in cleaned

    def test_preserve_paragraph_breaks(self) -> None:
        """Paragraph breaks (\\n\\n) must be preserved for chunker recursive splitting."""
        text = "First paragraph text with extra   spaces.\n\nSecond paragraph text."
        cleaned = normalize_ligatures_and_whitespace(text)
        assert "\n\n" in cleaned
        assert "First paragraph text with extra spaces." in cleaned
        assert "Second paragraph text." in cleaned


class TestBoilerplateRemoval:
    """Validate detection and removal of repeated headers/footers (>30% page frequency)."""

    def test_identify_boilerplate_header(self) -> None:
        # Create 10 synthetic pages where "GALE ENCYCLOPEDIA OF MEDICINE" appears on 5 pages (50% > 30%)
        raw_blocks: list[RawBlock] = []

        header_text = "GALE ENCYCLOPEDIA OF MEDICINE, SECOND EDITION"

        for page in range(1, 11):
            if page % 2 == 1:  # Appears on pages 1, 3, 5, 7, 9 (5 out of 10 pages = 50%)
                raw_blocks.append(RawBlock(page_number=page, text=header_text))

            raw_blocks.append(
                RawBlock(
                    page_number=page,
                    text=f"Unique medical topic content for page {page}.",
                )
            )

        boilerplate = identify_boilerplate_blocks(raw_blocks, total_pages=10, threshold=0.30)

        assert len(boilerplate) == 1
        assert header_text.lower() in boilerplate

    def test_unique_headers_not_flagged_as_boilerplate(self) -> None:
        raw_blocks = [
            RawBlock(page_number=1, text="Unique Chapter 1 Header"),
            RawBlock(page_number=2, text="Unique Chapter 2 Header"),
            RawBlock(page_number=3, text="Unique Chapter 3 Header"),
        ]

        boilerplate = identify_boilerplate_blocks(raw_blocks, total_pages=10, threshold=0.30)
        assert len(boilerplate) == 0
