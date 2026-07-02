---
title: Judge Capacity Dial — label-bound, not capacity-bound
type: result
tags: [judge, capacity, tinker, scaling]
created: 2026-07-02
updated: 2026-07-02
date: 2026-07-02
verdict: informative negative — 27B ≤ 9B at the judge task; training helps at both scales
evidence: scripts/judge_pilot/data/eval_results_9b.json, scripts/judge_pilot/data/train_info_27b.json, scripts/judge_pilot/data/eval_results.json
---

# Does judge capacity move the ceiling? (BarExamQA, identical 399 pools)

**Question** (Tinker spend-down, capacity dial): does a larger judge close
more of the pool ceiling — and can prompted frontier scale replace training?

| Judge | zeroshot Hit@5 (conv) | trained Hit@5 (conv) | training Δ |
|---|---:|---:|---:|
| Qwen3.5-9B | 15.3% (67.0%) | **20.6% (90.1%)** | +5.3pp |
| Qwen3.6-27B | 14.0% (61.5%) | 18.5% (81.3%) | +4.5pp |
| Qwen3-235B-A22B-Instruct (prompted) | 15.3% (67.0%) | — | — |

The 235B tie with 9B is exact at the top-5 cut but from genuinely different
score distributions (0/7,795 identical scores; mean |Δscore| 5.1) — the
zero-shot curve is **flat from 9B to 235B**.

(CE 3.8% / SCOPE-alone 12.0% / raw 1.3% on the same pools. 27B train:
120 steps, 23 min, same 3,500 pairs/hyperparameters.)

**Reading.**
1. **3× parameters bought nothing** — slightly negative at both zeroshot and
   trained. Relevance discrimination on 1.5K-char legal passages saturates
   by 9B; the binding constraints are the labels and the pool ceiling
   (22.8%), not model capacity.
2. **Training's lift replicates at a second scale** (+4.5pp/+5.3pp) — the
   labels-help finding is not a 9B quirk.
3. **Prompted scale does not buy judgment**: 26× active-parameter scale-up
   (9B → 235B-A22B MoE) moves zero-shot Hit@5 by 0.0pp; the trained 9B beats
   the prompted 235B by +5.3pp Hit@5 / +23pp conversion. This is the
   Thinking-Machines contrast in its purest form on our data: judgment comes
   from labels, not scale. Spend on labels, not parameters.
4. Caveat: Qwen3.6 is a different generation than Qwen3.5 (not a pure size
   ablation); one seed; same-prompt transfer may favor the model family the
   prompt was tuned on.

## Links
[[judge-pilot-v0-results]] · [[judge-pilot-housing]] · [[judge-pilot-fiqa]] ·
[[thinking-machines-expert-judgment]] · [[thesis-v2]] (P3)
