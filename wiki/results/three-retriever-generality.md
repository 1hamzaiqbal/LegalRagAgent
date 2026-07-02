---
title: Three-Retriever Generality — mechanism survives gte+CE, BM25, E5
type: result
tags: [mechanism, retriever, generality, credibility]
created: 2026-07-02
updated: 2026-07-02
date: 2026-05-29/31
verdict: win (closed the single-retriever-artifact critique)
evidence: docs/generated/credibility_C_three_retrievers_full_2026-05-29.md, docs/generated/credibility_C_e5_addendum_2026-05-29.md
---

# Three-retriever generality (credibility phase C/C++)

**Question.** Is the gold-affinity-movement mechanism an artifact of our
gte-large + MiniLM-CE stack?

**Answer: no.** SCOPE gold-affinity-delta → retrieval-gain Spearman, mean over
all 7 datasets (BarExamQA, HousingQA state-filtered, SciFact, NFCorpus, FiQA,
TREC-COVID, SciDocs), full corpora:
- gte + CE (original): **0.342**
- Tantivy BM25 (full-corpus sparse): **0.354**
- E5-large-v2 (independent dense): **0.387** — all three *means* clear the
  pre-stated 0.30 gate. Per-dataset caveat (keep honest): the closure criterion
  is the mean, not every cell — TREC-COVID SCOPE sits at 0.108/0.195/0.234
  across the three retrievers (dense qrels), and gte+CE NFCorpus (0.296) /
  SciDocs (0.299) are marginally under 0.30.

E5 spot-reads: BarExamQA raw 0.5% → HyDE 11.2% / SCOPE 11.7% Hit@5 (SCOPE
Spearman 0.344); HousingQA state-filtered raw 39.5% → SCOPE 47.0% (Spearman
0.454) but **HyDE 50.5% is stronger there** — do not claim SCOPE is the best
E5 expansion on Housing.

**Why it matters.** Closes the "single-retriever artifact" reviewer critique
in advance; makes the mechanism claim ([[geometry-vs-factuality]])
retriever-family-general (dense×2 + sparse). Also strengthens the case that
the [[weak-vs-strong-query-regime]] law is about query–corpus geometry, not
one encoder's quirks.

## Links
[[geometry-vs-factuality]] · [[beir-phase1]] · [[weak-vs-strong-query-regime]]
· [full report](../../docs/generated/credibility_C_three_retrievers_full_2026-05-29.md) ·
[E5 addendum](../../docs/generated/credibility_C_e5_addendum_2026-05-29.md)
