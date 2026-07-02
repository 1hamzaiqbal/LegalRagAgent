---
title: Weak- vs Strong-Query Regime
type: concept
tags: [regime, retrieval, vocabulary-gap, expansion]
created: 2026-07-02
updated: 2026-07-02
status: maintained
---

# Weak- vs strong-query regime

**Definition.** A *weak-query* regime is one where the raw user question is a
poor retrieval handle for the target corpus — low lexical/semantic overlap
between question phrasing and answer-bearing text (BarExamQA: colloquial fact
patterns vs doctrinal rule language; raw Hit@5 1.4%). A *strong-query* regime
is one where the raw question is already corpus-shaped (HousingQA statutory
entailment questions: raw state-filtered Hit@5 36.9%; Legal-Link-EU where
questions cite document anchors: raw Hit@5 90.0%).

**The regime law (our central empirical claim).** Generative query expansion
([[generated-query-family]]) helps roughly in proportion to the vocabulary gap
and *hurts* when the gap is absent:
- Weak end: SCOPE/HyDE lift BarExamQA Hit@5 1.4 → ~10–12%; MuSiQue bridge-recall
  +15–16pp; corpus-steered expansion ([[csqe]]) collapses here (2.0%) because
  there is no good initial retrieval to steer with.
- Strong end: expansion is net-negative (HousingQA answers −3pp; on
  strong-query BEIR HyDE −31%, SCOPE −12% Hit@5) — the failure mode is
  [[query-drift]].
- Macro precedent: Weller'24 ([[weller-drift]]) — expansion helps weak
  retrievers/hurts strong ones. Our wedge is *per-query, geometric, label-free*:
  benefit tracks the CE gold-vs-distractor affinity margin
  ([[geometry-vs-factuality]]), with the help→hurt crossover holding
  *within* datasets, not just across them.

**Why it matters to us.** It is the theory that unifies every result we have:
the BarExam win, the Housing regression (review criticism C10), the BEIR
negatives, the CSQE crossover, and it motivates [[regime-routing]] as the
operational recipe. It is also the defensible framing left after the method
novelty claim died (C3/C6/C7 in [[icml-ai4law-2026-rejection]]).

**Status in our work.** Validated across 7 datasets × 3 retrievers
(gte+CE, BM25-tantivy, E5-large-v2; pooled Spearman 0.34–0.39) and two extra
domains (MedQA-USMLE, MuSiQue). Dataset-level perplexity is a weak regime
separator; per-query perplexity/OOV is useless (ruled out as mechanism).

**Open questions.**
- Can regime be predicted *pre-retrieval* cheaply enough to route in prod?
  (per-query QPP routing failed: τ≈0.11 — [[qpp]]; regime-level routing works)
- Is the weak-end advantage of generative expansion legal/medical-specific or
  general? Global-corpus weak-end evidence is currently legal-only.
- Where exactly is the crossover? Margin ≈ 0 is the pre-registered prediction.

## Links
[[vocabulary-gap]] · [[query-drift]] · [[regime-routing]] ·
[[geometry-vs-factuality]] · [[weller-drift]] · [[csqe]] · [[scope]] ·
[[beir-phase1]]
