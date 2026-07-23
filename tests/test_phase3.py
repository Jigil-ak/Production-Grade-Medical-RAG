"""Unit tests for Phase 3 evaluation & observability components."""

import json
from pathlib import Path

import pytest

from app.core.observability import trace_query_event
from app.eval.validate_golden import GoldenItemSchema, validate_golden_dataset


class TestGoldenValidator:
    """Test Golden Dataset schema validation."""

    def test_valid_golden_item_schema(self) -> None:
        item = GoldenItemSchema(
            question="What is hypertension?",
            ground_truth_answer="Hypertension is high blood pressure.",
            ground_truth_chunk_ids=["chunk123"],
            question_type="direct_fact",
            difficulty="easy",
        )
        assert item.question == "What is hypertension?"
        assert item.ground_truth_chunk_ids == ["chunk123"]

    def test_validate_golden_dataset_valid_file(self, tmp_path: Path) -> None:
        dataset_file = tmp_path / "valid_ds.json"
        data = [
            {
                "question": "What are symptoms of asthma?",
                "ground_truth_answer": "Wheezing, shortness of breath, and coughing.",
                "ground_truth_chunk_ids": [],
                "question_type": "symptom_diagnosis",
                "difficulty": "easy",
            }
        ]
        dataset_file.write_text(json.dumps(data), encoding="utf-8")

        is_valid, errors = validate_golden_dataset(dataset_file, check_chroma=False)
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_golden_dataset_invalid_schema(self, tmp_path: Path) -> None:
        dataset_file = tmp_path / "invalid_ds.json"
        data = [
            {
                "question": "Short",  # Min length violation
                "ground_truth_answer": "",  # Min length violation
            }
        ]
        dataset_file.write_text(json.dumps(data), encoding="utf-8")

        is_valid, errors = validate_golden_dataset(dataset_file, check_chroma=False)
        assert is_valid is False
        assert len(errors) > 0

    def test_validate_golden_dataset_schema_only_skips_chroma_lookup(self, tmp_path: Path) -> None:
        """Verify check_chroma=False (--schema-only) skips Chroma resolution even if chunk_ids exist."""
        dataset_file = tmp_path / "ds_with_ids.json"
        data = [
            {
                "question": "What are symptoms of asthma?",
                "ground_truth_answer": "Wheezing, shortness of breath, and coughing.",
                "ground_truth_chunk_ids": ["non_existent_chunk_123"],
                "question_type": "symptom_diagnosis",
                "difficulty": "easy",
            }
        ]
        dataset_file.write_text(json.dumps(data), encoding="utf-8")

        # When check_chroma=False (schema-only), non-existent chunk_ids do NOT cause validation failure
        is_valid, errors = validate_golden_dataset(dataset_file, check_chroma=False)
        assert is_valid is True
        assert len(errors) == 0


class TestObservability:
    """Test Langfuse observability wrapper."""

    def test_trace_query_event_unconfigured_does_not_raise(self) -> None:
        # Should not raise exception when Langfuse keys are empty
        trace_query_event(
            question="Test question",
            prompt_version="v1",
            retrieved_chunk_ids=["c1"],
            latency_ms=12.5,
            status="answered",
        )
