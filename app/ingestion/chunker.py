"""Embedding-tokenizer-aware text chunker.

Uses the actual sentence-transformers tokenizer for the configured
EMBEDDING_MODEL_NAME to count tokens — NEVER tiktoken. MiniLM has a
hard 256-token input limit.

Target: 220-240 MiniLM tokens per chunk, 40-60 token overlap.
Prefers paragraph/sentence boundaries; hard token cutoffs only when a
paragraph itself exceeds the target.
"""

from __future__ import annotations

import hashlib
import re
from typing import Sequence

from transformers import AutoTokenizer

from app.core.logging import get_logger
from app.core.types import Chunk
from app.ingestion.loader import PageBlock

logger = get_logger(__name__)


class TokenizerAwareChunker:
    """Chunker that uses HuggingFace Transformers tokenizer for exact token counting."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        target_tokens: int = 230,
        max_tokens: int = 250,
        overlap_tokens: int = 50,
    ) -> None:
        """Initialize chunker with target token settings.

        Args:
            model_name: HuggingFace model or tokenizer identifier.
            target_tokens: Target token count per chunk (220-240).
            max_tokens: Hard ceiling below MiniLM's 256 sequence length limit.
            overlap_tokens: Overlap in tokens (40-60).
        """
        # Map shorthand model name if needed
        hf_model_id = (
            "sentence-transformers/all-MiniLM-L6-v2"
            if model_name == "all-MiniLM-L6-v2"
            else model_name
        )
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def count_tokens(self, text: str) -> int:
        """Count exact tokens for a text string using the MiniLM WordPiece tokenizer."""
        if not text:
            return 0
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _split_text_into_sentences(self, text: str) -> list[str]:
        """Simple regex sentence splitter fallback for chunk boundary estimation."""
        # Split on sentence ending punctuation followed by space
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk_page_blocks(self, source_filename: str, blocks: Sequence[PageBlock]) -> list[Chunk]:
        """Chunk page blocks from a single PDF document.

        Args:
            source_filename: Name of the PDF file.
            blocks: Extracted PageBlock objects.

        Returns:
            List of Chunk objects with metadata and SHA-256 chunk_ids.
        """
        chunks: list[Chunk] = []

        for block in blocks:
            text = block.text.strip()
            if not text:
                continue

            block_tokens = self.count_tokens(text)

            # Case 1: Block fits within target_tokens
            if block_tokens <= self.target_tokens:
                chunk_id = self._generate_chunk_id(
                    source_filename, block.page_number, block.char_start, block.char_end
                )
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        source_filename=source_filename,
                        page_number=block.page_number,
                        char_start=block.char_start,
                        char_end=block.char_end,
                        chunk_text=text,
                    )
                )
                continue

            # Case 2: Block exceeds target_tokens, split recursively by paragraphs / sentences
            sub_chunks = self._split_large_block(source_filename, block)
            chunks.extend(sub_chunks)

        logger.info(
            "Chunking complete",
            source_filename=source_filename,
            total_blocks=len(blocks),
            total_chunks=len(chunks),
        )
        return chunks

    def _split_large_block(self, source_filename: str, block: PageBlock) -> list[Chunk]:
        """Split a large block into chunks respecting sentence boundaries and token limits."""
        paragraphs = [p.strip() for p in block.text.split("\n\n") if p.strip()]
        units: list[str] = []

        for para in paragraphs:
            para_tokens = self.count_tokens(para)
            if para_tokens <= self.target_tokens:
                units.append(para)
            else:
                sentences = self._split_text_into_sentences(para)
                for sent in sentences:
                    if self.count_tokens(sent) <= self.target_tokens:
                        units.append(sent)
                    else:
                        # Hard token window fallback for extremely long sentences
                        units.extend(self._hard_token_split(sent))

        chunks: list[Chunk] = []
        current_unit_group: list[str] = []
        current_token_count = 0
        char_cursor = block.char_start

        for unit in units:
            unit_tokens = self.count_tokens(unit)

            if current_token_count + unit_tokens > self.target_tokens and current_unit_group:
                chunk_text = " ".join(current_unit_group)
                char_end = char_cursor + len(chunk_text)
                chunk_id = self._generate_chunk_id(
                    source_filename, block.page_number, char_cursor, char_end
                )
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        source_filename=source_filename,
                        page_number=block.page_number,
                        char_start=char_cursor,
                        char_end=char_end,
                        chunk_text=chunk_text,
                    )
                )

                # Maintain overlap by retaining units that fit within overlap_tokens
                overlap_group: list[str] = []
                overlap_count = 0
                for prev_unit in reversed(current_unit_group):
                    prev_tokens = self.count_tokens(prev_unit)
                    if overlap_count + prev_tokens <= self.overlap_tokens:
                        overlap_group.insert(0, prev_unit)
                        overlap_count += prev_tokens
                    else:
                        break

                current_unit_group = overlap_group
                current_token_count = overlap_count
                char_cursor = char_end

            current_unit_group.append(unit)
            current_token_count += unit_tokens

        if current_unit_group:
            chunk_text = " ".join(current_unit_group)
            char_end = char_cursor + len(chunk_text)
            chunk_id = self._generate_chunk_id(
                source_filename, block.page_number, char_cursor, char_end
            )
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    source_filename=source_filename,
                    page_number=block.page_number,
                    char_start=char_cursor,
                    char_end=char_end,
                    chunk_text=chunk_text,
                )
            )

        return chunks

    def _hard_token_split(self, long_sentence: str) -> list[str]:
        """Fallback hard token splitter for sentences exceeding target_tokens."""
        encoded = self.tokenizer.encode(long_sentence, add_special_tokens=False)
        pieces: list[str] = []

        step = self.target_tokens - self.overlap_tokens
        for i in range(0, len(encoded), step):
            token_sub = encoded[i : i + self.target_tokens]
            decoded = self.tokenizer.decode(token_sub, skip_special_tokens=True).strip()
            if decoded:
                pieces.append(decoded)

        return pieces

    @staticmethod
    def _generate_chunk_id(filename: str, page_number: int, char_start: int, char_end: int) -> str:
        """Generate SHA-256 hash (truncated to 16 hex chars) for chunk identification."""
        raw_key = f"{filename}:{page_number}:{char_start}:{char_end}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:16]
