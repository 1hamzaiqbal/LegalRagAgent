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
- **Prediction 1 SUPPORTED (2026-07-02)**: the leakage confound is rejected —
  the weak-query lift survives at +6pp (>4× raw, p=1.1e-20 question-level) on
  the ~85% of generations with no gold-entailed sentence
  ([[leakage-audit-barexam]]). Leakage amplifies (+28pp on matched rows) but
  does not explain the effect.
- **BEIR leakage replication also done (2026-07-02)**: matched rates 0–7% on
  scientific corpora and help_m=0 in all cells — expansion help is never
  leakage-gated in either regime ([[leakage-audit-barexam]] §BEIR).
- Still pending: answer-side conversion modeling; MedQA leakage variant (no
  gold qrels — needs a proxy design).

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
- **Prediction 2 SUPPORTED (2026-07-02)**: Housing (strong regime) trained
  judge 55.0% vs CE-pool 38.2% (p=2.5e-23), 96.5% gold-in-pool conversion —
  P3 holds on both regimes; [[regime-routing]] superseded by "always pool +
  trained judge" ([[judge-pilot-housing]]).
- **Prediction 3 REVISED (2026-07-02)**: the selector-bottleneck half
  transfers cross-domain (zero-shot judge > CE in all 3 domains: +11.5/+14.6/
  +8.5pp), but **label-training helps only where labels encode judged
  relevance** — on SciDocs' citation-proxy gold, training *hurt* (−14pp vs
  zero-shot, p=6.5e-06). P3 restated: *after pooling, selection binds, and a
  judge trained on quality labels fixes it; proxy labels can make it worse
  than prompting* ([[judge-pilot-scidocs]]). This sharpens the
  [[expert-judgment-replication]] motivation: label quality is the scarce
  resource, exactly TM's thesis.
- **Answer-conversion RESOLVED as a two-regime law (2026-07-02,
  [[judge-answer-conversion]])**: BarExamQA/70B — 5.4× exposure does not
  convert (gold-present +2.4pp, gold-absent −3.8pp, break-even Hit@5 ≈61% vs
  22.8% ceiling). Housing/70B — conversion pays monotonically: llm_only 54.2
  → CE-ev 61.8 → SCOPE-ev 63.2 → **judge-ev 65.6% (+11.4pp, p=5.5e-08;
  beats CE arm p=0.048)**, and even gold-absent evidence helps (+12.0pp).
  **The cost term is regime-dependent**: distractor tax on parametric-strong
  MC, neighboring-provision value on statutory entailment. Full pipeline law:
  expansion ← query margin; selection ← pool confusability; conversion ←
  task evidence-value — all measurable in advance.
- **FiQA resolves the label-semantics question (2026-07-02,
  [[judge-pilot-fiqa]])**: zero-shot judge > CE in all four domains
  (+8.5..+14.6pp, all p≤3e-05 — the general-domain CE is universally the
  weakest link); training = label quality × headroom (legal: helps; FiQA
  human labels at ceiling: neutral/safe; SciDocs proxy labels: harmful). P3
  final form.
- Still pending: pre-registered held-out validation of the three-dial law on
  an unseen corpus (e.g. Legal-Link-EU end-to-end); lawyer-label rung of
  Path C.

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

## Benchmark weighting (user steer, 2026-07-02)
HousingQA is a **supporting** dataset, not a headline: acknowledged-noisy
gold, answer-bound Y/N structure, mechanically-split questions
([[zheng-cslaw]] caveats; advisor preference on record). Lead evidence for
the conversion-pays regime should be **MedQA-USMLE** (full-N 4-arm matrix
running 2026-07-02; q200 probe already showed the only significant SCOPE
answer win over llm_only: +5.5pp p=0.019) plus BarExamQA/Legal-Link-EU/BEIR
for the other dials. Housing results stay as replications.

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
