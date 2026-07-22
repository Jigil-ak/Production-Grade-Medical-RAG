"""RAGAS offline evaluation runner and quality gate script.

Evaluates the Medical RAG pipeline against golden dataset metrics:
  - faithfulness (target >= 0.80)
  - answer_relevancy
  - context_precision (target >= 0.75)
  - context_recall

Configuration & Constraint Highlights:
1. max_workers=1 set explicitly: RAGAS's default multi-processing spawns parallel
   subprocesses reloading Chroma + MiniLM (~300MB each), spiking RAM past the 4GB
   budget; max_workers=1 also respects Groq's free-tier rate limits.
2. Direct ChatGroq import: SANCTIONED EXCEPTION to Protocol-first rule. RAGAS
   requires a LangChain-native BaseChatModel; passing custom GroqClient raises
   AttributeError.
3. Tenacity 429 Retry: Handles Groq rate-limiting gracefully on 50+ question runs.
4. Diagnostic Grouping: Calculates and reports metrics aggregated OVERALL AND
   grouped by question_type.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from datasets import Dataset
# SANCTIONED EXCEPTION to Protocol-first rule:
# RAGAS 0.1.19's evaluate() engine requires a LangChain-native BaseChatModel instance
# (e.g. ChatGroq). Passing our custom GroqClient Protocol raises AttributeError inside
# RAGAS internals. Do not replace ChatGroq with GroqClient here!
from langchain_groq import ChatGroq
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import get_settings
from app.core.logging import get_logger
from app.embedding.service import MiniLMEmbeddingService
from app.eval.validate_golden import validate_golden_dataset
from app.generation.citation_enforcer import CitationEnforcer
from app.generation.llm_client import GroqClient
from app.generation.prompt_provider import YAMLPromptProvider
from app.ingestion.chunker import TokenizerAwareChunker
from app.ingestion.loader import load_pdf
from app.retrieval.bm25_index import BM25Index
from app.retrieval.hybrid import RRFHybridRetriever
from app.retrieval.reranker import TinyBERTReranker
from app.retrieval.retriever import VectorRetriever
from app.retrieval.store import ChromaStore

logger = get_logger(__name__)

# Minimum thresholds for CI gate
FAITHFULNESS_THRESHOLD = 0.80
CONTEXT_PRECISION_THRESHOLD = 0.75


@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    retry=retry_if_exception_type((Exception,)),
)
def run_ragas_evaluate_with_retry(
    dataset: Dataset, metrics: list[Any], llm: Any, embeddings: Any
) -> Any:
    """Execute RAGAS evaluate() with tenacity retry-with-backoff for Groq 429 rate limits.

    Explicitly specifies max_workers=1 to prevent RAM spikes > 4GB.
    """
    return evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        # max_workers=1 is CRITICAL:
        # 1. Spawning multiple workers duplicates Chroma + MiniLM (~300MB each) causing OOM on 4GB RAM.
        # 2. It respects Groq free-tier API rate limits (TPM/RPM).
        max_workers=1,
        is_async=False,
    )


def run_pipeline_inference(
    dataset_path: Path, mode: str = "hybrid"
) -> tuple[list[dict[str, Any]], list[str]]:
    """Run pipeline inference over golden dataset questions.

    Args:
        dataset_path: Path to golden dataset JSON file.
        mode: 'hybrid' (Phase 2) or 'vector_only' (Phase 1 baseline).

    Returns:
        Tuple of (ragas_samples, question_types).
    """
    settings = get_settings()
    raw_data = json.loads(dataset_path.read_text(encoding="utf-8"))

    embedding_service = MiniLMEmbeddingService(model_name=settings.embedding_model_name)
    vector_store = ChromaStore(persist_dir=settings.chroma_persist_dir)
    bm25_index = BM25Index()
    reranker = TinyBERTReranker()
    prompt_provider = YAMLPromptProvider()
    citation_enforcer = CitationEnforcer(embedding_service=embedding_service)
    llm_client = GroqClient(api_key=settings.groq_api_key.get_secret_value())

    vector_retriever = VectorRetriever(embedding_service, vector_store)
    hybrid_retriever = RRFHybridRetriever(vector_retriever, bm25_index, rrf_k=settings.retrieval.rrf_k)

    ragas_samples: list[dict[str, Any]] = []
    question_types: list[str] = []

    for item in raw_data:
        question = item["question"]
        ground_truth = item["ground_truth_answer"]
        q_type = item.get("question_type", "general")
        question_types.append(q_type)

        if mode == "vector_only":
            chunks = vector_retriever.retrieve(query=question, top_k=settings.retrieval.final_answer_k)
        else:
            candidates = hybrid_retriever.retrieve(
                query=question,
                vector_top_k=settings.retrieval.vector_top_k,
                bm25_top_k=settings.retrieval.bm25_top_k,
            )
            reranked = reranker.rerank(query=question, chunks=candidates, top_k=settings.retrieval.rerank_top_k)
            chunks = reranked[: settings.retrieval.final_answer_k]

        contexts = [c.chunk_text for c in chunks]

        # Generate answer
        prompt_tmpl = prompt_provider.get("answer_generation", settings.prompt_version)
        raw_res = llm_client.generate_answer_with_citations(question, chunks, prompt_tmpl.version)
        final_res = citation_enforcer.enforce_citations(raw_res.answer, raw_res.citations, chunks, prompt_tmpl.version)

        answer_text = final_res.answer or "The provided context does not contain sufficient information."

        ragas_samples.append(
            {
                "question": question,
                "answer": answer_text,
                "contexts": contexts if contexts else ["No context found."],
                "ground_truth": ground_truth,
            }
        )

    return ragas_samples, question_types


def evaluate_dataset(dataset_path: Path) -> dict[str, Any]:
    """Run RAGAS evaluation, calculate per-question_type metrics, and export JSON report."""
    settings = get_settings()
    api_key = settings.groq_api_key.get_secret_value()

    # ChatGroq judge model for RAGAS
    judge_llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key)

    # 1. Run Phase 2 Hybrid + Rerank pipeline
    logger.info("Running Phase 2 Hybrid Pipeline inference...")
    hybrid_samples, q_types = run_pipeline_inference(dataset_path, mode="hybrid")
    hybrid_ds = Dataset.from_list(hybrid_samples)

    # 2. Run Phase 1 Vector-Only baseline
    logger.info("Running Phase 1 Vector-Only Baseline inference...")
    vector_samples, _ = run_pipeline_inference(dataset_path, mode="vector_only")
    vector_ds = Dataset.from_list(vector_samples)

    eval_metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    # Evaluate hybrid pipeline
    logger.info("Evaluating Phase 2 Hybrid Pipeline with RAGAS (max_workers=1)...")
    hybrid_results = run_ragas_evaluate_with_retry(hybrid_ds, eval_metrics, judge_llm, None)

    # Evaluate vector-only baseline
    logger.info("Evaluating Phase 1 Vector-Only Baseline with RAGAS (max_workers=1)...")
    vector_results = run_ragas_evaluate_with_retry(vector_ds, eval_metrics, judge_llm, None)

    # 3. Diagnostic Breakdown by question_type
    per_type_scores: dict[str, dict[str, list[float]]] = {}
    df_results = hybrid_results.to_pandas()

    for idx, q_type in enumerate(q_types):
        if q_type not in per_type_scores:
            per_type_scores[q_type] = {
                "faithfulness": [],
                "answer_relevancy": [],
                "context_precision": [],
                "context_recall": [],
            }

        row = df_results.iloc[idx]
        for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            val = float(row.get(m, 0.0))
            per_type_scores[q_type][m].append(val)

    type_aggregates: dict[str, dict[str, float]] = {}
    for q_type, m_dict in per_type_scores.items():
        type_aggregates[q_type] = {
            m: round(sum(vals) / len(vals), 4) if vals else 0.0 for m, vals in m_dict.items()
        }

    # 4. Prepare Report
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "dataset": dataset_path.name,
        "sample_count": len(hybrid_samples),
        "overall_scores": {
            "hybrid_pipeline": {
                "faithfulness": round(float(hybrid_results.get("faithfulness", 0.0)), 4),
                "answer_relevancy": round(float(hybrid_results.get("answer_relevancy", 0.0)), 4),
                "context_precision": round(float(hybrid_results.get("context_precision", 0.0)), 4),
                "context_recall": round(float(hybrid_results.get("context_recall", 0.0)), 4),
            },
            "vector_only_baseline": {
                "faithfulness": round(float(vector_results.get("faithfulness", 0.0)), 4),
                "answer_relevancy": round(float(vector_results.get("answer_relevancy", 0.0)), 4),
                "context_precision": round(float(vector_results.get("context_precision", 0.0)), 4),
                "context_recall": round(float(vector_results.get("context_recall", 0.0)), 4),
            },
        },
        "by_question_type": type_aggregates,
        "threshold_gate": {
            "faithfulness_target": FAITHFULNESS_THRESHOLD,
            "context_precision_target": CONTEXT_PRECISION_THRESHOLD,
            "faithfulness_passed": float(hybrid_results.get("faithfulness", 0.0)) >= FAITHFULNESS_THRESHOLD,
            "context_precision_passed": float(hybrid_results.get("context_precision", 0.0)) >= CONTEXT_PRECISION_THRESHOLD,
        },
    }

    # Export report to /reports/
    reports_dir = Path("./reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_filename = f"eval_results_{int(time.time())}.json"
    report_path = reports_dir / report_filename
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    logger.info("Evaluation report saved", report_path=str(report_path))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on golden dataset.")
    parser.add_argument("--dataset", type=str, default="data/golden/smoke_dataset.json", help="Path to golden dataset")
    parser.add_argument("--smoke", action="store_true", help="Run fast smoke evaluation")
    args = parser.parse_args()

    ds_path = Path(args.dataset)
    print(f"=== Starting Phase 3 RAGAS Evaluation on {ds_path.name} ===")

    # 1. Validate dataset schema first
    is_valid, errors = validate_golden_dataset(ds_path)
    if not is_valid:
        print("ERROR: Dataset schema validation failed!")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)

    # 2. Run evaluation
    try:
        report = evaluate_dataset(ds_path)
    except Exception as e:
        print(f"ERROR: RAGAS evaluation failed: {e}")
        sys.exit(1)

    # 3. Print Results & Check Quality Gate
    overall = report["overall_scores"]["hybrid_pipeline"]
    gate = report["threshold_gate"]

    print("\n=== OVERALL RAGAS SCORES (Hybrid Pipeline) ===")
    print(f"  Faithfulness:       {overall['faithfulness']:.4f} (Target: >={FAITHFULNESS_THRESHOLD})")
    print(f"  Context Precision:  {overall['context_precision']:.4f} (Target: >={CONTEXT_PRECISION_THRESHOLD})")
    print(f"  Answer Relevancy:   {overall['answer_relevancy']:.4f}")
    print(f"  Context Recall:     {overall['context_recall']:.4f}")

    print("\n=== SCORES BY QUESTION TYPE ===")
    for q_type, scores in report["by_question_type"].items():
        print(f"  [{q_type}] Faithfulness: {scores['faithfulness']:.4f} | Precision: {scores['context_precision']:.4f}")

    passed = gate["faithfulness_passed"] and gate["context_precision_passed"]
    if passed:
        print("\nSUCCESS: All RAGAS quality gate thresholds passed! [Exit 0]")
        sys.exit(0)
    else:
        print("\nFAILURE: RAGAS quality gate threshold NOT met! [Exit 1]")
        sys.exit(1)


if __name__ == "__main__":
    main()
