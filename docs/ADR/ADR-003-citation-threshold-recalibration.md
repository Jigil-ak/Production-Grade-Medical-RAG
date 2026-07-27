# ADR-003: Citation Support Threshold Recalibration

## Status
Accepted

## Context

In Phase 2 & 3, we introduced sentence-level MAX-over-sentences support scoring (`CitationEnforcer`). In traditional RAG systems, cosine similarity is often calculated by averaging over the entire ~200-token chunk vector. Whole-chunk averaging dilutes specific cited facts with surrounding context text, artificially lowering support scores.

When transitioning to sentence-level sentence-to-sentence cosine similarity:
- Sentence-to-sentence vector comparison provides a much tighter, more focused similarity signal.
- The similarity distribution shifts slightly lower for individual short sentences compared to long context blocks.
- Using a `0.70` threshold with sentence-level scoring resulted in false declines for valid medical citations.

## Decision

Set the default citation support threshold to **`0.65`** (`settings.citation.support_threshold = 0.65`).

- Recalibrated based on offline RAGAS dataset evaluations (`app/eval/run_eval.py` on `smoke_dataset.json`).
- Empirical RAGAS Offline Evaluation Results:
  - **Context Precision**: **0.9444** (Hybrid Pipeline) vs **0.9028** (Vector-Only Baseline)
  - **Context Recall**: **1.0000** (Hybrid Pipeline) vs **1.0000** (Vector-Only Baseline)
  - **Faithfulness**: **0.6667** (Hybrid Pipeline) vs **0.5000** (Vector-Only Baseline)
- Sentence-level MAX-over-sentences matching with `support_threshold = 0.65` eliminates false-declined citations on valid medical answers while outperforming the baseline by +16.67% on Faithfulness and +4.16% on Context Precision.

## Consequences

### Benefits
- Reduces false declined answers on valid medical facts.
- Outperforms Phase 1 baseline across Context Precision (0.9444 vs 0.9028) and Faithfulness (0.6667 vs 0.5000).

### Tradeoffs
- Threshold must be re-evaluated whenever the embedding model is upgraded or changed (e.g. from MiniLM to BGE or E5).
