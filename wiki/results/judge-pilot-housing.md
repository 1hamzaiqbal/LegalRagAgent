---
title: Judge Pilot — Housing (strong regime): pool + trained judge wins again
type: result
tags: [judge, tinker, housing, strong-regime, win]
created: 2026-07-02
updated: 2026-07-02
date: 2026-07-02
verdict: win (thesis-v2 prediction 2 supported; regime routing superseded)
evidence: scripts/judge_pilot/data/housing_eval_results.json, scripts/judge_pilot/data/housing_train_info.json
---

# Housing judge — the strong-regime replication

**Question** ([[thesis-v2]] prediction 2): does pool + trained-judge reranking
also beat raw-top5 and the CE pool on the strong/intermediate regime — where
expansion *alone* hurts and the raw query is already corpus-shaped? If yes,
[[regime-routing]] collapses into "always pool + trained judge."

**Setup.** Same recipe as [[judge-pilot-v0-results]], new domain slice:
HousingQA state-filtered, **group-level** splits (Y/N questions share source
groups — leakage-safe), 5,000 train pairs (multi-gold aware, ≤4 hard
negatives/question), Qwen3.5-9B LoRA rank 32 on Tinker (120 steps, loss →
0.125, 13 min). Eval: 500 held-out raw∪SCOPE pools (Gemma, statefilter),
gold-in-pool ceiling **57.0%**. Statute texts hydrated by direct CSV scan of
the 3.2GB corpus on EIT (9,913/9,913 ids).

| Arm (same pools) | Hit@5 | MRR@5 | gold-in-pool conversion |
|---|---:|---:|---:|
| raw-question top5 (cached) | 33.4% | 0.223 | — |
| CE ms-marco pool rerank (cached) | 38.2% | 0.248 | 67.0% (191/285) |
| SCOPE-alone top5 (cached) | 41.2% | 0.264 | — |
| judge-zeroshot | 52.8% | 0.385 | 92.6% |
| **judge-trained** | **55.0%** | **0.477** | **96.5% (275/285)** |

McNemar (exact): trained vs CE-pool **86/2, p=2.5e-23** · vs SCOPE-alone
88/19, p=8.5e-12 · vs raw 118/10, p=1.5e-24 · vs zeroshot 18/7, p=0.043
(training's larger win here is *ranking*: MRR +0.092 over zeroshot).

**Reading.**
1. **P3 is regime-independent**: the selector, not the candidate generator,
   binds on both ends. Weak (BarExam): trained judge 20.6% vs CE 3.8%,
   90% conversion. Strong (Housing): 55.0% vs 38.2%, 96.5% conversion.
2. **[[regime-routing]] is superseded as the primary recipe**: always-pool +
   trained-judge beats every routed alternative measured to date on both
   regimes. Routing survives only as the judge-less fallback.
3. **The CE was leaving ~1/3 of available gold on the table** (67% vs 96.5%
   conversion) even on the regime where it works best — consistent with the
   CE-buries-gold diagnosis and [[reuter2025sac]]'s confusability framing.
4. Trained-vs-prompted narrows on the strong regime (+2.2pp Hit@5, p=0.043)
   but ranking quality still improves markedly — label-training's value
   scales with how *confusable* the pool is.

**Caveats.** Retrieval-side only (the Housing [[answer-conversion-gap]] is
real: May's prompted-judge probe raised exposure without answer gains — the
judge-evidence answer run is the required next cell). Ceiling 57% binds;
in-distribution benchmark-gold supervision; single generator's pools (Gemma);
one seed. Housing gold labels are acknowledged-noisy ([[zheng-cslaw]]).

**Cost note.** Tinker spend to date (est.): BarExam train+3 evals + Housing
train+2 evals ≈ $50–70 of the $150 credits (check dashboard); within the
$125 loop budget.

## Cross-task transfer (2026-07-02, Tinker spend-down battery)

The **BarExam-trained** 9B scored on these same Housing pools: Hit@5 46.4% /
MRR 0.300 / 81.4% conversion — above the CE (38.2%) but **below Housing
zero-shot** (52.8%) and far below the Housing-trained judge (55.0%).
Reading: label-training *specializes* toward the source task's relevance
notion (controlling MBE rule ≠ statutory basis for a state); judgment
training is not automatically legal-general. A deployable "legal judge"
needs mixed-task labels — queued as the first free EIT-lane experiment
(train barexam+housing combined, eval both).

**RESOLVED same day ([[judge-mixed-legal]])**: the mixed barexam+housing
judge holds both domains with zero specialization tax — BarExam 22.1%
(above the 20.6% specialist, b/c=7/1 p=0.070) and Housing 55.4% (tied with
the 55.0% specialist, p=0.625), trained for $0 on the EIT free lane.
Specialization was an artifact of single-domain training, not a limit.

## Links
[[thesis-v2]] (prediction 2 ✓) · [[judge-pilot-v0-results]] ·
[[regime-routing]] (superseded) · [[pooling-regime]] ·
[[answer-conversion-gap]] · [[expert-judgment-replication]] ·
[[direction-2026-07]]
