---
title: Thesis v2 — a geometric account of the retrieval budget
type: concept
tags: [thesis, mechanism, judge, generalization]
created: 2026-07-02
updated: 2026-07-02
status: draft
---

# Thesis v2 (working) — where the retrieval budget should go, and why

One sentence: **query-side generation and candidate selection fail for
different, measurable, geometric reasons — generation pays off exactly when
the raw query's gold-affinity margin is low, and once candidates are pooled,
the binding constraint is the selector, which a small judge trained on free
outcome labels fixes.** Legal is the extreme weak-query end of this story,
not its scope.

## The three pillars and their evidence status

**P1 — Generation is a low-margin instrument** (when to *expand*).
Per-query generative-expansion benefit tracks CE gold-affinity movement
(ρ≈0.34–0.50), crossover near margin≈0, within-dataset; failures are
geometric, not hallucinated (AUC 0.79–0.94 vs 0.55–0.58) and not
perplexity/OOV (≈0.51).
- Tested: 7 datasets (legal/medical/scientific/web) × 3 retrievers × 4
  generators — [[affinity-margin-mechanism]], [[beir-phase1]],
  [[three-retriever-generality]], [[factuality-falsification]].
- Pending: **leakage confound** ([[yoon2025leakage]] audit running
  2026-07-02 — does the weak-query lift survive on generations not entailed
  by gold?); answer-side conversion modeling.

**P2 — Answer-conditioning is a drift dampener, not a retrieval booster**
(how to expand). Snap-conditioned generation ≈ HyDE on weak-query retrieval
(BarExamQA ns 3/4 models) and on answers (13/16 pairs ns), but drifts far
less on strong queries: +16–45pp Hit@5 over HyDE, 19/20 cells significant,
4 generators × 5 BEIR datasets ([[snap-vs-hyde-ledger]],
[retrieval significance](../../docs/generated/retrieval_significance_2026-07-02.md)).
- Pending: *why* (mechanism hypothesis: committing to an answer frame first
  constrains topic support — needs the C12 keep-vs-discard/conclusion-banned
  ablations); the Housing generator-flip anomaly (Gemma pro-SCOPE +7.4pp,
  Llama-70B pro-HyDE −11.8pp — what property of the generator decides?).

**P3 — After pooling, selection is the binding constraint** (when to stop
expanding and start judging). Pooling raw∪generated candidates creates recall
headroom everywhere, but a general-domain CE buries it (gold median rank 4–5;
BarExam pool 3.9% vs SCOPE-alone 12.0%). A 9B judge LoRA-trained on *free
outcome labels* (gold ids + retrieved hard negatives, no human annotation)
un-buries it: Hit@5 20.6% vs CE 3.8% (p=1.4e-17), 90% of pool ceiling, and
trained>prompted (p=1e-04) — [[judge-pilot-v0-results]].
- Pending: **strong/intermediate-regime replication** (Housing judge — in
  prep); **cross-domain replication** (same recipe on a BEIR corpus);
  **answer conversion** (judge-top5 answer run); if P3 holds across regimes,
  [[regime-routing]] collapses into "always pool + trained judge," which is a
  simpler and stronger operational law than routing.

## Falsifiable predictions (pre-stated)
1. Leakage audit: unmatched-generation lift on BarExamQA stays > 0 (geometry
   account) rather than ≤ 0 (leakage account).
2. Housing judge: trained-judge pool reranking beats both raw-top5 and the CE
   pool on state-filtered Housing (i.e., P3 is regime-independent even where
   expansion alone hurts).
3. A same-recipe judge on a non-legal corpus (e.g. SciDocs/FiQA pools)
   improves over the CE by a margin proportional to how much the CE buries
   gold there (bury-rate measurable in advance).
4. C12 ablations: passing a0 to the answer call hurts ≥0pp (guardrail), and
   conclusion-banned generation (ParSeR-style) matches snap-conditioned
   retrieval (i.e., the *draft* is dispensable; the *frame commitment* isn't).

## External pressure (2026-07-02 ingest — engage, don't ignore)
- **[[lexpath2026]]**: IRAC-guided expansion beats HyDE/Query2doc on all three
  Chinese legal sets — the strongest published legal expansion. Their ablation
  independently shows raw-query fusion adds noise on weak queries (our
  pool-destroys-weak finding) and their Appendix D reproduces the
  answer-conversion gap on frontier models (Claude 33.3→29.1 with retrieval).
  Cheap counter-experiment queued: IRAC-structured SCOPE prompt variant.
- **[[reuter2025sac]]**: Document-Level Retrieval Mismatch (~46% at k=1) is
  corpus-side evidence for geometric confusability (supports P1/P4) — but a
  ~150-char generic summary prefix halves it offline. Open interaction: does
  query-side generation still pay on a SAC-fixed index? Must be flagged as
  untested.
- **[[qe-survey-2025]]**: the survey states selective/gated use of
  zero-grounding generative QE as best practice (our regime law as folklore —
  cite, then show we *measure* the gate) and names query-quality prediction
  as the open gap; MUGI's multi-pseudo-doc gains vs our 3SCOPE null must be
  positioned by regime.

## What this thesis is NOT
- Not "SCOPE beats HyDE" (dead — C7).
- Not "expansion is good" (it is net-negative ungated on strong queries).
- Not legal-specific: legal supplies the extreme weak end (TF-IDF
  query–gold cosine 0.07 vs 0.25 for open-domain, [[zheng-cslaw]]); medical
  (MedQA +7pp answers p=0.007) and scientific (BEIR) supply the middle and
  strong ends.

## Links
[[weak-vs-strong-query-regime]] · [[geometry-vs-factuality]] ·
[[snap-vs-hyde-ledger]] · [[judge-pilot-v0-results]] · [[regime-routing]] ·
[[answer-conversion-gap]] · [[direction-2026-07]] ·
[[icml-ai4law-2026-rejection]]
