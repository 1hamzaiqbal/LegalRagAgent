---
title: Answer-Conversion Gap (retrieval–generation gap)
type: concept
tags: [rag, bottleneck, evaluation]
created: 2026-07-02
updated: 2026-07-02
status: maintained
---

# The answer-conversion gap

**Definition.** Large retrieval-exposure gains that fail to convert into
answer-accuracy gains. Established anchors: the retrieval–generation gap /
context-utilization literature — distractor harm ("Power of Noise",
[[power-noise-lostmiddle]]), position effects (Lost in the Middle), and the
QPP↔answer disconnect documented by Emami'26 and Tian'25.

**Our canonical instances.**
- BarExamQA: Hit@5 1.4 → 12.1 (~8×) moves average answers 72.3 → 72.9 (+0.6).
- Gold-evidence ceiling itself is low: 72.6 avg on BarExamQA — perfect
  retrieval ≈ parametric answering on the two big models; the corpus mainly
  helps the 8B (+5.5pp) and Housing (+2–5pp).
- Neighbor dilution: gold+4 retrieved neighbors *under-performs* gold-only by
  10–18pp on CaseHOLD/SCALR (golden_plus_neighbors rows) — extra context can
  bury the signal.
- Field echo: Emami'26 finds NQC correlates with nDCG@5 (r=0.33) but ~0 with
  answer quality.

**Why it matters.** It is the reviewers' strongest substantive point (C5/C9:
"answers driven by parametric knowledge") *and* the most under-owned research
space in the grounding doc's deep-read ("only observed, never modeled").
Nobody has a model of *when* exposure converts. Our margin machinery gives a
place to start: exposure converts when the gold's rank/margin in the final
context clears the reader's distractor tolerance — measurable per-query.

**Candidate attacks** (for [[direction-2026-07]]):
1. Model conversion per-query: P(answer flips correct | gold enters context at
   rank r, margin m, reader size). We have every log needed (signed rows with
   per-row retrieval + answers).
2. Reader-side interventions where conversion fails: answer-option grounding /
   evidence-conditioned reranking by answer-consistency ("does this passage
   change the answer?") rather than query-similarity — connects to
   [[expert-judgment-replication]] (better reward signal than exact-match).
3. Honest negative framing: in high-parametric-competence regimes, retrieval
   is for *grounding/citation*, not accuracy — evaluate faithfulness/citation
   quality instead (LegalBench-RAG-style, [[legal-rag-benchmarks]]).

## Links
[[power-noise-lostmiddle]] · [[emami-qpp-variant]] · [[tian-right-track]] ·
[[geometry-vs-factuality]] · [[weak-vs-strong-query-regime]] ·
[[icml-ai4law-2026-rejection]] · [[direction-2026-07]]
