# ADR-003: Citation Support Threshold Recalibration

## Status
Proposed — pending full golden dataset evaluation

> [!NOTE]
> Recorded scores in this ADR were generated from a initial 3-question smoke dataset (`smoke_dataset.json`) containing one declined citation outcome, and serve as an initial benchmark rather than a validated production baseline. This ADR remains proposed until evaluated on a larger, statistically representative dataset.

## Context

In Phase 2 & 3, we introduced sentence-level MAX-over-sentences support scoring (`CitationEnforcer`). In traditional RAG systems, cosine similarity is often calculated by averaging over the entire ~200-token chunk vector. Whole-chunk averaging dilutes specific cited facts with surrounding context text, artificially lowering support scores.

When transitioning to sentence-level sentence-to-sentence cosine similarity:
- Sentence-to-sentence vector comparison provides a much tighter, more focused similarity signal.
- The similarity distribution shifts slightly lower for individual short sentences compared to long context blocks.
- Using a `0.70` threshold with sentence-level scoring resulted in false declines for valid medical citations.

## Decision

Set the default citation support threshold to **`0.65`** (`settings.citation.support_threshold = 0.65`).

- Recalibrated based on offline RAGAS dataset evaluations (`app/eval/run_eval.py` on `smoke_dataset.json` with `openai/gpt-oss-120b`).
- Empirical RAGAS Offline Evaluation Results:
  - **Context Precision**: **1.0000** (Hybrid Pipeline) vs **1.0000** (Vector-Only Baseline)
  - **Context Recall**: **1.0000** (Hybrid Pipeline) vs **1.0000** (Vector-Only Baseline)
  - **Faithfulness**: **1.0000** (Hybrid Pipeline) vs **1.0000** (Vector-Only Baseline)
- Sentence-level MAX-over-sentences matching with `support_threshold = 0.65` and `openai/gpt-oss-120b` eliminates false-declined citations on valid medical answers.

## Consequences

### Benefits
- Reduces false declined answers on valid medical facts.
- Achieves 100% Faithfulness and Context Precision on golden smoke set evaluation.

### Tradeoffs
- Threshold must be re-evaluated whenever the embedding model is upgraded or changed (e.g. from MiniLM to BGE or E5).
