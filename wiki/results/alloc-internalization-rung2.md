---
title: Rung 2 — a 9B internalizes regime-level allocation from outcome labels; no per-question edge; frontier-positive under cost pressure
type: result
tags: [bandit, allocation, internalization, bridge-rung-2, e1, mixed]
created: 2026-07-02
updated: 2026-07-02
date: 2026-07-02
verdict: mixed — regime-level allocation learned (trained ≫ zero-shot), λ=0 per-question edge absent (consistent with E0), cost-aware mixtures trace points above the fixed-arm frontier (ns)
evidence: docs/generated/alloc_rung2_2026-07-02.md, scripts/judge_pilot/data/alloc_scores_{trained,zeroshot}.jsonl (EIT job 93770)
---

# E1 / rung 2 — the internalized allocation predictor

**Question** ([[opd-skill0-design]] E1): can a 9B *internalize* the
allocation decision — predicting from question text alone whether a given
reader succeeds under a given retrieval strategy — from sparse outcome
labels, where rung 1 showed cheap external features cannot?

**Setup.** Qwen3.5-9B LoRA (judge recipe) on 6,136
(question, reader, strategy)→Yes/No pairs across all five cells,
**rung-1-identical splits**; policy = argmax(sigmoid(score) − λ·cost);
zero-shot same-model scoring as the prompted control. Free EIT A100 (job
93770, ~2h). Full tables:
[alloc_rung2_2026-07-02](../../docs/generated/alloc_rung2_2026-07-02.md).

| Cell | best fixed | 9B-trained @λ=0 | 9B-zeroshot @λ=0 | trained action mix (sensible?) |
|---|---:|---:|---:|---|
| BarExam/70B | llm_only 77.5% | 75.5% (p=0.57) | 75.5% | ✗ under-picks llm_only (5/200) — fails to learn "evidence hurts strong readers" |
| BarExam/8B | scope 65.0% | 63.5% (p=0.63) | 59.0% | ✓ scope-heavy (140/200) |
| Housing/70B | judge 67.2% | 64.8% (p=0.11) | 58.8% (p=0.001) | ✓ judge-heavy (174/250) |
| Housing/8B | ce 66.4% | 66.0% (p=1.0) | 63.6% | ~ judge-heavy |
| MedQA/70B | llm_only 84.6% | 84.8% (p=1.0) | 84.6% | ~ llm_only+scope mix (the two best arms) |

**Readings.**
1. **Training internalizes regime-level allocation.** The trained model's
   action distributions track each cell's best arm (judge on Housing/70B,
   scope on BarExam/8B, llm_only+scope on MedQA), and it beats the
   zero-shot prompted 9B decisively where allocation matters (Housing/70B
   +6.0pp, BarExam/8B +4.5pp). Sparse outcome labels do teach the *regime*
   map — the same map the three dials describe.
2. **No per-question edge at λ=0** — ties/trails the best fixed arm in all
   five cells. Consistent with [[offline-bandit-v0]]: the per-question
   headroom (oracle 80–92%) stays unreachable, now also from the 9B's own
   reading of the question. The allocation signal, if extractable at all,
   is not in question text + outcome labels at this N.
3. **The one systematic failure is diagnostic**: on BarExam/70B the trained
   model almost never chooses llm_only (5/200) — it never learned the
   hardest, most counter-intuitive rule ("for a strong reader, retrieving
   *hurts*"). That rule is written explicitly in the E2/E3 skill file
   (`scripts/opd/skills/allocation.md`, Rule 1) — a concrete, falsifiable
   target for whether skill-augmented distillation fixes what labels miss.
4. **Cost-aware mixtures trace points above the fixed-arm frontier** (new
   vs rung 1): Housing/8B @λ=0.02 → **69.2% @ 1.59 ktok** vs best fixed
   66.4% @ 2.65 (b/c=29/22, p=0.40) and vs llm_only 64.8% @ 0.25 (p=0.14);
   MedQA @λ=0.02 → 85.1% @ 0.90 vs llm_only 84.6% @ 0.65 (p=0.65).
   Directionally frontier-dominating, **not significant**, and λ chosen
   post-hoc — treat as an observation to pre-register at E3, not a win.

**Caveats.** One seed; uncalibrated sigmoid of the Yes/No logit gap for the
cost-aware policy; post-hoc λ; per-cell N small for policy-vs-fixed
McNemars; barexam questions shared across reader cells (splits aligned, no
leakage, but cells are not independent).

**Ladder consequence** ([[opd-skill0-design]]): E1 sets the sparse-label
bar: regime-level yes, per-question no, frontier hints. E2 (does a big
model + skill file allocate better than without?) and E3 (dense OPD signal
vs these sparse labels) now have a sharp target: fix the BarExam/70B
llm_only failure and make the frontier points significant.

**Reproduce.** `scripts/bandit/build_alloc_dataset.py` → EIT
`judge_lane6_alloc.sbatch` → `scripts/bandit/analyze_rung2.py`.

## Links
[[opd-skill0-design]] · [[skill-distillation-bridge]] ·
[[offline-bandit-v0]] · [[judge-answer-conversion]] · [[thesis-v2]]
