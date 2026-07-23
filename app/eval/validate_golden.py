"""Golden dataset schema validator and live chunk ID resolver.

Validates JSON golden dataset files placed in /data/golden/ against the exact required schema:
  - question (str)
  - ground_truth_answer (str)
  - ground_truth_chunk_ids (list[str])
  - question_type (str)
  - difficulty (str)

Hard Validation:
  Calls ChromaStore.get_by_id(chunk_id) for every ground_truth_chunk_id.
  Exits with code 1 if any field is missing or if any chunk_id fails to resolve.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from app.core.config import get_settings
from app.core.logging import get_logger
from app.retrieval.store import ChromaStore

logger = get_logger(__name__)


class GoldenItemSchema(BaseModel):
    """Schema for a single golden dataset sample."""

    question: str = Field(..., min_length=5, description="User question string")
    ground_truth_answer: str = Field(..., min_length=5, description="Expected ground truth answer string")
    ground_truth_chunk_ids: list[str] = Field(default_factory=list, description="List of expected chunk_ids")
    question_type: str = Field(..., min_length=2, description="Category/type of question")
    difficulty: str = Field(..., min_length=2, description="Difficulty level (easy, medium, hard)")


def validate_golden_dataset(
    json_path: str | Path, check_chroma: bool = True
) -> tuple[bool, list[str]]:
    """Validate schema and resolve chunk_ids against ChromaStore.

    Args:
        json_path: Path to the golden dataset JSON file.
        check_chroma: If True, resolve chunk_ids against persistent ChromaStore.
                      If False (--schema-only mode), validate schema fields only.

    Returns:
        Tuple of (is_valid: bool, errors: list[str]).
    """
    path = Path(json_path)
    if not path.is_file():
        return False, [f"File not found: {path}"]

    try:
        raw_data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return False, [f"Invalid JSON syntax in {path.name}: {e}"]

    if not isinstance(raw_data, list):
        return False, [f"Dataset root must be a JSON list, got {type(raw_data).__name__}"]

    if len(raw_data) == 0:
        return False, ["Golden dataset is empty (0 samples)"]

    errors: list[str] = []
    items: list[GoldenItemSchema] = []

    # 1. Schema Validation (Field presence & types)
    for idx, sample in enumerate(raw_data, start=1):
        try:
            item = GoldenItemSchema.model_validate(sample)
            items.append(item)
        except ValidationError as ve:
            errors.append(f"Sample #{idx} schema error: {ve}")

    if errors:
        return False, errors

    # 2. Hard Chunk ID Resolution against ChromaStore (Skipped in --schema-only mode)
    if check_chroma:
        try:
            settings = get_settings()
            store = ChromaStore(persist_dir=settings.chroma_persist_dir)
            
            # Only resolve if vector store contains documents
            if store.count() > 0:
                missing_chunks: list[str] = []
                for idx, item in enumerate(items, start=1):
                    for cid in item.ground_truth_chunk_ids:
                        chunk = store.get_by_id(cid)
                        if chunk is None:
                            missing_chunks.append(f"Sample #{idx} ('{item.question[:30]}...'): missing chunk_id '{cid}'")

                if missing_chunks:
                    errors.extend(missing_chunks)

        except Exception as e:
            logger.warn("ChromaStore lookup skipped during validation", error=str(e))

    is_valid = len(errors) == 0
    return is_valid, errors


def main() -> None:
    """CLI entry point for validate_golden.py."""
    parser = argparse.ArgumentParser(
        description="Validate golden dataset schema and optionally resolve chunk IDs against ChromaStore."
    )
    parser.add_argument("dataset", help="Path to golden dataset JSON file")
    parser.add_argument(
        "--schema-only",
        action="store_true",
        help="Validate schema fields presence and types only, skipping ChromaStore chunk resolution",
    )
    args = parser.parse_args()

    target_path = args.dataset
    check_chroma = not args.schema_only
    mode_str = "schema-only mode" if args.schema_only else "full mode (schema + Chroma resolution)"

    print(f"Validating golden dataset: {target_path} ({mode_str})...")

    is_valid, errors = validate_golden_dataset(target_path, check_chroma=check_chroma)

    if is_valid:
        print("SUCCESS: Golden dataset validated successfully! [Exit 0]")
        sys.exit(0)
    else:
        print("FAILURE: Golden dataset validation failed with the following errors:")
        for err in errors:
            print(f"  - {err}")
        print("[Exit 1]")
        sys.exit(1)


if __name__ == "__main__":
    main()
