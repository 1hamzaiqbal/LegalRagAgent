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

## Links
[[thesis-v2]] · [[answer-conversion-gap]] · [[judge-pilot-v0-results]] ·
[[judge-pilot-housing]] · [[zheng-cslaw]] · [[icml-ai4law-2026-rejection]] (C5/C9)
