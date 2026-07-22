"""Unit tests for PDF loader and tokenizer-aware chunker."""

from pathlib import Path

import pytest

from app.core.exceptions import IngestionError
from app.ingestion.chunker import TokenizerAwareChunker
from app.ingestion.loader import PageBlock, load_pdf


class TestPDFLoader:
    """Test PyMuPDF loader."""

    def test_load_non_existent_file_raises_ingestion_error(self) -> None:
        with pytest.raises(IngestionError, match="PDF file not found"):
            load_pdf("non_existent_file.pdf")

    def test_load_corrupt_file_raises_ingestion_error(self, tmp_path: Path) -> None:
        corrupt_file = tmp_path / "corrupt.pdf"
        corrupt_file.write_text("This is not a PDF file content.")

        with pytest.raises(IngestionError):
            load_pdf(corrupt_file)

    def test_load_valid_pdf_if_present(self) -> None:
        pdf_path = Path("./data/raw/Medical_book.pdf")
        if pdf_path.exists():
            filename, blocks = load_pdf(pdf_path)
            assert filename == "Medical_book.pdf"
            assert len(blocks) > 0
            assert all(isinstance(b, PageBlock) for b in blocks)


class TestTokenizerAwareChunker:
    """Test chunker token counting and bounds."""

    @pytest.fixture
    def chunker(self) -> TokenizerAwareChunker:
        return TokenizerAwareChunker(model_name="all-MiniLM-L6-v2", target_tokens=230, max_tokens=250)

    def test_chunk_token_count_under_256_limit(self, chunker: TokenizerAwareChunker) -> None:
        """Regression test: Every chunk generated must have <= 256 MiniLM tokens."""
        sample_blocks = [
            PageBlock(
                source_filename="sample.pdf",
                page_number=1,
                char_start=0,
                char_end=1500,
                text="Aspirin (acetylsalicylic acid) is a nonsteroidal anti-inflammatory drug (NSAID) used to reduce pain, fever, or inflammation. " * 30,
            )
        ]

        chunks = chunker.chunk_page_blocks("sample.pdf", sample_blocks)
        assert len(chunks) > 0

        for chunk in chunks:
            token_count = chunker.count_tokens(chunk.chunk_text)
            assert token_count <= 256, f"Chunk {chunk.chunk_id} exceeded MiniLM 256 limit with {token_count} tokens!"

    def test_chunk_id_determinism_and_idempotency(self, chunker: TokenizerAwareChunker) -> None:
        """Ingesting identical text twice must generate identical chunk_ids."""
        blocks = [
            PageBlock(
                source_filename="doc.pdf",
                page_number=1,
                char_start=0,
                char_end=100,
                text="Medical chunk content for testing idempotency.",
            )
        ]

        chunks_1 = chunker.chunk_page_blocks("doc.pdf", blocks)
        chunks_2 = chunker.chunk_page_blocks("doc.pdf", blocks)

        assert len(chunks_1) == len(chunks_2) == 1
        assert chunks_1[0].chunk_id == chunks_2[0].chunk_id
        assert len(chunks_1[0].chunk_id) == 16
