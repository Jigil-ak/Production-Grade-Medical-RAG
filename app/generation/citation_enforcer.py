"""Citation enforcement with corrected support-scoring algorithm.

Phase 2 implementation.

CRITICAL — support scoring uses max-over-sentences, NOT whole-chunk cosine:
  1. Split cited chunk into sentences (nltk punkt)
  2. Embed each sentence individually with MiniLM
  3. Embed the LLM's claim sentence
  4. Take MAX cosine similarity between claim and any sentence in chunk

This prevents the ~200 tokens of surrounding text from diluting the
specific cited sentence. Threshold: settings.citation.support_threshold
(0.65 default, recalibrated in Phase 3).
"""

# Phase 2 implementation
