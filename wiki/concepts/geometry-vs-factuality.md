---
title: Geometry vs Factuality (the falsification result)
type: concept
tags: [mechanism, falsification, expansion, embedding]
created: 2026-07-02
updated: 2026-07-02
status: maintained
---

# Geometry vs factuality — why generated queries fail

**The claim.** When a generated pseudo-document fails as a retrieval query,
the cause is *geometric* (it moved the embedded query away from the gold
region / shrank the gold-vs-distractor affinity margin), **not** factual (the
LLM hallucinated false law/medicine). The standard objection — reviewer C4,
and CSQE's own failure attribution ("LLM knowledge gaps/hallucination") — is
empirically falsified in our setting.

**The mechanism (positive half).** Per-query SCOPE retrieval benefit tracks
CE/embedding affinity movement toward gold: Spearman ≈ 0.44 pooled (legal),
≈ 0.5 on BEIR; monotone; crossover consistent with margin ≈ 0; replicated on
7 datasets under gte+CE (0.342), BM25-tantivy (0.354), E5-large-v2 (0.387) —
so not a single-retriever artifact. Gold-affinity movement dominates
confounds (partial-R² 0.13 vs ≤0.004 for length/format/domain).

**The falsification (negative half).**
- Geometry predicts failure: AUC 0.791 (legal/medical q200 5-dataset), 0.94
  (BEIR) — vs LLM-judged *real* factuality of the same pseudo-docs: AUC 0.581
  (gemma judge), 0.548 pooled (full-N gpt-4o re-judge; IRR ρ=0.681 between
  judges); marginal lift of factuality after geometry +0.001–0.003.
- Per-query perplexity/OOV ruled out earlier (corr ≈ 0).
- Strawman-proofing done: the "less-circular" target (real hallucination
  measure, independent judge, full N) survived — see credibility pass
  (A++ single-judge gpt-4o, 2026-05-31).

**Why it matters.** (1) It is the direct rebuttal to C4 ("pseudo-docs inherit
LLM fabrication") — they don't need to be true to work. (2) It redefines what
a fix looks like: steering *geometry* (exemplars, corpus anchoring, margin-aware
generation), not fact-checking the pseudo-doc. (3) It is our most defensible,
novel, cross-domain result — the recommended paper anchor since the 05-26
deep-read, now with the review pressure pointing the same way.

**Caveats to keep honest.** Factuality judges are imperfect (judge-judge IRR
0.681); "factuality barely predicts" ≠ "hallucination never hurts" — the claim
is about *marginal* predictive value after geometry; Housing was deferred in
the gpt-4o pass; Claude-judge triangulation still budget-gated.

## Links
[[weak-vs-strong-query-regime]] · [[query-drift]] · [[csqe]] ·
[[beir-phase1]] · [[factuality-falsification]] · [[three-retriever-generality]] ·
[[icml-ai4law-2026-rejection]] (C4) · [[direction-2026-07]]
