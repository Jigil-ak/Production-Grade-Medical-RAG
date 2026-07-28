"""Script to export real chunk evaluation fixture from local ChromaStore.

Exports actual extracted chunks matching ground_truth_chunk_ids from smoke_dataset.json
plus competing candidate chunks to data/fixtures/eval_chunks.json.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.types import Chunk
from app.embedding.service import MiniLMEmbeddingService
from app.retrieval.store import ChromaStore

logger = get_logger(__name__)


def export_eval_fixture() -> None:
    settings = get_settings()
    smoke_dataset_path = Path("data/golden/smoke_dataset.json")

    if not smoke_dataset_path.exists():
        raise FileNotFoundError(f"Smoke dataset not found at {smoke_dataset_path}")

    raw_data = json.loads(smoke_dataset_path.read_text(encoding="utf-8"))

    target_chunk_ids: set[str] = set()
    questions: list[str] = []

    for item in raw_data:
        questions.append(item["question"])
        for cid in item.get("ground_truth_chunk_ids", []):
            target_chunk_ids.add(cid)

    print(f"Target ground truth chunk IDs from dataset: {target_chunk_ids}")

    store = ChromaStore(persist_dir=settings.chroma_persist_dir)
    embedding_service = MiniLMEmbeddingService(model_name=settings.embedding_model_name)

    exported_chunks_map: dict[str, dict] = {}

    # 1. Fetch exact ground truth chunks
    for cid in target_chunk_ids:
        chunk = store.get_by_id(cid)
        if chunk:
            exported_chunks_map[cid] = chunk.model_dump()
        else:
            print(f"WARNING: Ground truth chunk_id '{cid}' not found in local ChromaStore!")

    # 2. Query competing candidate chunks for each question to build a realistic ranking pool
    for q in questions:
        query_emb = embedding_service.embed_query(q)
        retrieved = store.query(query_emb, top_k=10)
        for r_chunk in retrieved:
            if r_chunk.chunk_id not in exported_chunks_map:
                c_obj = Chunk.model_validate(r_chunk.model_dump())
                exported_chunks_map[r_chunk.chunk_id] = c_obj.model_dump()

    fixture_list = list(exported_chunks_map.values())
    fixtures_dir = Path("data/fixtures")
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    out_file = fixtures_dir / "eval_chunks.json"
    out_file.write_text(json.dumps(fixture_list, indent=2), encoding="utf-8")

    print(f"SUCCESS: Exported {len(fixture_list)} real chunks to {out_file.as_posix()}")


if __name__ == "__main__":
    export_eval_fixture()
