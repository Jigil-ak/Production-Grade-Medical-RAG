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

- Recalibrated based on offline dataset evaluations (`app/eval/run_eval.py`).
- Maintains high faithfulness ($\ge 0.80$) while eliminating false-declined citations on valid medical answers.

## Consequences

### Benefits
- Reduces false declined answers on valid medical facts.
- Aligns with RAGAS `faithfulness` metric target of $\ge 0.80$.

### Tradeoffs
- Threshold must be re-evaluated whenever the embedding model is upgraded or changed (e.g. from MiniLM to BGE or E5).
