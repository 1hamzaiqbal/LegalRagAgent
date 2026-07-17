---
title: CRAFT — Counterfactual Credit Assignment for Privileged Self-Teachers
type: source
tags: [opd, self-distillation, counterfactual-credit, agent-rl, teacher-trust]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2606.29476
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2606.29476.pdf
code: unavailable during double-blind review
authors: Anonymous authors
year: 2026
---

# CRAFT

## TL;DR

CRAFT is the strongest wording collision with “a student should sometimes
oppose its teacher.” It assigns signed token credit: a token preferred by a
privileged self-teacher may receive negative credit when sibling rollouts
suggest it would reduce return. Its counterfactual is nevertheless an
importance-weighted, group-level surrogate over GRPO siblings—not a repeated
fixed-state intervention—and the teacher is the same model with privileged
context, not a stronger model exporting an external-action policy to a weaker
student.

## Exact object

The privileged teacher is the current model conditioned on extra context
`s+`, such as a skill annotation, verifier hint, or successful trace. For a
sampled token, the teacher/student likelihood gap is

`Delta_t = log pi_T(y_t | s_t+) - log pi_theta(y_t | s_t)`.

The ideal counterfactual token influence asks how expected advantage would
change if the action at token `t` were sampled from the teacher and the
continuation rerolled. CRAFT does **not** run that intervention. It estimates
instead

`CTI_hat_t^(i) = sum_(j != i) w_t^(j) A^(j) - A^(i)`,

where sibling weights are proportional to `exp(Delta_t^(j) / tau)`. This is a
self-normalized importance-sampling tilt over the existing GRPO group. Signed
credit derived from this estimate drives a REINFORCE-style token loss, while
an EMA controller couples credit strength and reference-KL strength.

The paper is unusually explicit about identification: the estimator is exact
for a **group-level state-marginal estimand**. Interpreting it as a
fixed-state, per-trajectory counterfactual requires within-group
exchangeability, exact at `t=0` and progressively less plausible downstream.

## Evidence

- Environments: ALFWorld, Search-QA, and WebShop.
- Models: Qwen3-1.7B, Qwen2.5-3B, Qwen2.5-7B, and Qwen3-8B.
- Each cell uses group size 8 and 150 training steps.
- CRAFT is compared with GRPO, GRPO+OPSD, Skill-SD, RLSD, SDAR, and
  Adaptive-CRINGE. Most older-method cells are imported from the SDAR suite;
  Adaptive-CRINGE and CRAFT are newly measured.
- CRAFT exceeds Adaptive-CRINGE in all 12 reported model×environment cells.
  Gains are largest for Qwen3-1.7B: +2.6 ALFWorld, +1.9 Search-QA, and +2.3
  WebShop.
- The Qwen2.5-3B ablation gives most of the improvement to the counterfactual
  credit pillar; the two control/KL pillars add only a few tenths of a point.

## Limitations and open validity tests

- The “counterfactual” does not hold the prefix fixed. Later-position sibling
  trajectories differ in their states as well as their sampled actions.
- `G=8` is far from the asymptotic consistency regime, yet the paper reports
  no effective sample size, importance-weight concentration, or direct bias
  audit.
- The four model scales are four self-distillation runs. They do not form a
  teacher×student transport matrix across scale.
- Main tables have no independent-seed uncertainty or significance tests.
- Most baseline cells were not rerun in the same campaign.
- The method introduces nine hyperparameters; the negative-KL branch is a
  heuristic policy update rather than an unbiased forward-KL estimator.
- OOD evaluation is mainly within-domain template holdout. Random irrelevant
  skill retrieval is tested only in one ALFWorld/model setting.

## Bearing on our work

CRAFT closes the broad claim that teacher-preferred actions can receive
negative counterfactual credit. The remaining distinction for
[[compute_elasticity_handoff_2026-07-17/10-student-specific-action-value]] is exact and narrower:

- **CRAFT:** same-model privileged-context token tilt; approximate sibling
  counterfactual; no explicit action price.
- **Our candidate:** different teacher and student models; the same fixed
  external action; repeated randomized forced outcomes; task-level target
  utility; explicit cost; teacher-policy transport regret.

The most valuable baseline is empirical rather than architectural: collect
CRAFT-style sibling estimates alongside actual forced rerolls, then measure
sign agreement, bias, effective sample size, and degradation with token
position. We should also isolate the cross-scale cell “teacher prefers the
action while target-student utility is negative,” which CRAFT motivates but
does not measure.

## Code custody

No public repository was available on 2026-07-17. Appendix A says the
implementation repository, base commit, and phase tags are withheld during
double-blind review and will be restored after anonymity. This is an explicit
unavailable-code status, not a failed search or license to reconstruct the
authors' artifact.

## Raw source

EIT PDF `papers/arxiv_2606.29476.pdf`.

## Links

[[compute_elasticity_handoff_2026-07-17/10-student-specific-action-value]] · [[action-value-transport-reading-packet-2026-07-17]] ·
[[reward-gated-opd]] · [[sdar]] · [[token-teachability]]
