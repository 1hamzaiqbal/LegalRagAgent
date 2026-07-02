---
title: Judge-Evidence Answer Run — the conversion wall, decomposed
type: result
tags: [answer-conversion, judge, barexam, negative, break-even]
created: 2026-07-02
updated: 2026-07-02
date: 2026-07-02
verdict: informative negative — exposure converts per-row, but distractor cost dominates below a measurable break-even
evidence: logs/eval_rag_simple_groq-llama70b_20260702_042708_barexam_detail.jsonl (judge), _043852_ (llm_only), _045017_ (ce), _050224_ (scope)
---

# Does fixing the selector convert to answers? (BarExamQA, 70B)

**Question** ([[thesis-v2]], the wall behind P3): the trained judge lifted
Hit@5 3.8→20.6% on identical pools — does that convert to answer accuracy?

**Setup.** Four fully-paired arms on the 399 held-out judge-test questions,
groq-llama70b, same day, harness **strict retrieval-cache replay** (synthetic
caches: judge-trained top5 / CE-pool top5 / SCOPE top5; full-text doc cache;
`EVAL_QA_CSV` subset override; `NO_SILENT_FALLBACK=1`,
`EVAL_FINAL_FORMAT_RETRY=1`, 2048-token cap). ~1,600 Groq calls.

| Arm (evidence @5) | Hit@5 of evidence | Answer acc |
|---|---:|---:|
| llm_only (no evidence) | — | **77.7%** |
| CE-pool top5 | 3.8% | 76.7% |
| SCOPE top5 | 12.0% | 76.2% |
| judge-trained top5 | **20.6%** | 75.2% |

All pairwise deltas ns (judge vs llm_only −2.51pp, b/c=26/36, p=0.25).
**A 5.4× gold-exposure improvement over the CE produced zero answer gain.**

**The decomposition (the finding).** Split each evidence arm by whether gold
is actually in the top-5:
- Judge arm, gold-present (n=82): **82.9%** vs llm_only 80.5% on the same
  rows → **+2.4pp**. Exposure DOES convert when it happens.
- Judge arm, gold-absent (n=317): 73.2% vs llm_only 77.0% → **−3.8pp**.
  All-distractor evidence actively hurts a 70B parametric-strong reader.
- Same signs in the CE and SCOPE arms (gold-present +6.7/−2.1pp at tiny n).

**Break-even model**: evidence pays iff
$Hit@5 \times gain > (1-Hit@5) \times cost$ → with +2.4/−3.8pp,
break-even ≈ **61% Hit@5** — ~3× the 22.8% pool ceiling. No selector,
however perfect, can cross the wall on these pools; the binding constraint
has moved *up the stack* to **candidate-pool recall** (expansion depth/
quality — P1's territory) and **evidence-conditional answering**.

**Score-gated evidence (post-hoc diagnostic).** Injecting evidence only when
the judge's top score ≥ τ: best case 78.4% (τ=2.0, 120 evidence rows) vs
77.7% llm_only — +0.75pp, post-hoc-optimistic, ns. Confidence gating alone
cannot rescue conversion at this ceiling.

**What it changes.**
1. [[answer-conversion-gap]] now has a *quantitative per-row model* on
   BarExamQA (+gain/−cost/break-even), not just an observation — this is the
   "model the conversion" attack the field punts on (Emami'26 et al.).
2. The judge line stays a retrieval-layer contribution; its answer-layer
   payoff requires either bigger pools (deeper k, better expansion — measure
   ceiling vs k) or readers/prompts robust to distractor-only contexts
   (evidence-use instruction, abstention) — both queued.
3. Caveat honesty: 70B on MC bar exam = maximal parametric competence; the
   Zheng gold-ceiling (~+0.5pp at 70B on gold evidence, their Table) already
   predicted a low ceiling here. Housing (retrieval-positive regime,
   gold ceiling +5pp, judge conversion 96.5%) is the arm where conversion
   should actually pay — **that is the decisive next answer run.**

## Housing arms (same day) — conversion PAYS on the evidence-valuable regime

Same design, 500 held-out Housing questions (state-filtered strict replay,
logs `eval_*_20260702_{061002,061742,062828,063923}_housing_detail.jsonl`):

| Arm | Hit@5 of evidence | Answer acc | vs llm_only |
|---|---:|---:|---|
| llm_only | — | 54.2% | — |
| CE-pool top5 | 38.2% | 61.8% | +7.6pp, p=1.6e-04 |
| SCOPE top5 | 41.2% | 63.2% | +9.0pp, p=8.1e-05 |
| **judge-trained top5** | **55.0%** | **65.6%** | **+11.4pp, p=5.5e-08** |

Judge beats the CE arm on answers too (+3.8pp, b/c=51/32, **p=0.048**);
judge-vs-SCOPE +2.4pp (ns at N=500). Ordering matches evidence quality
exactly: better selection → better answers, monotone.

**The two-regime conversion contrast (the paper's closing figure):**

| | BarExamQA/70B | HousingQA/70B |
|---|---:|---:|
| best evidence arm vs llm_only | −2.5pp (ns) | **+11.4pp (p=5e-08)** |
| gold-present evidence effect | +2.4pp | +10.9pp |
| gold-absent evidence effect | **−3.8pp** | **+12.0pp** |

The cost term of the break-even model is *itself regime-dependent*: on
parametric-strong MC, non-gold evidence is a distractor tax; on statutory
entailment, even non-gold same-state statutes carry answer-value (neighboring
provisions). So the full pipeline law reads: **expansion is governed by the
query-side margin; selection is governed by pool confusability; and whether
any of it reaches answers is governed by the evidence answer-value of the
task — all three measurable in advance.** Selector improvements convert 1:1
where evidence has value (Housing: CE 61.8 → judge 65.6 tracks Hit@5 38→55)
and cannot convert where it doesn't (BarExam).

Caveat: this 500-question subset runs hotter than the signed full-N llm_only
(54.2% vs 44.8%) — group-level sampling; all comparisons are within-subset
paired, so unaffected.

## The reader-size 2×2 (same day, groq-llama8b arms) — dial 3 sharpens

Identical evidence artifacts, answer model swapped to Llama-3.1-8B:

| best-evidence vs llm_only | BarExamQA | HousingQA |
|---|---:|---:|
| **70B reader** | −2.5pp ns (llm_only 77.7%) | **+11.4pp p=5e-08** (llm_only 54.2%) |
| **8B reader** | **+11.8pp p=5.6e-05** (llm_only 54.9%) | −2.8pp ns (llm_only 62.8%) |

At 8B the regimes *invert*: BarExam evidence pays (judge +8.8pp p=0.0026,
SCOPE-ev +11.8pp; gold-present +14.6pp AND gold-absent +7.3pp — even
imperfect evidence helps a weak reader), Housing evidence stops paying
(gold-present −6.5pp: the 8B can't integrate statutes it already answers
above its integration ability).

**Unified conversion law**: evidence pays iff the reader's parametric
competence on the task (measured by plain llm_only accuracy) is low —
in these four cells the crossover sits around llm_only ≈ 60%. "Task
evidence-value" was the fixed-reader special case; the general dial is the
**reader–task parametric deficit**, and its pre-test costs one llm_only run.
(Also note: at 8B/BarExam, SCOPE-evidence 66.7% ≥ judge-evidence 63.7%
despite less gold — weak readers may benefit from topically-broad context
beyond exact gold; judge-vs-scope −3.0pp ns, flag for replication.)

## Links
[[thesis-v2]] · [[answer-conversion-gap]] · [[judge-pilot-v0-results]] ·
[[judge-pilot-housing]] · [[zheng-cslaw]] · [[icml-ai4law-2026-rejection]] (C5/C9)
