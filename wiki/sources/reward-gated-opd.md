---
title: Reward-Gated On-Policy Distillation
type: source
tags: [opd, verifier, reward-gating, reasoning]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2607.04037
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2607.04037.pdf
code: https://github.com/UoC-tail/RG-OPD
authors: Mohammad Sadegh Akhondzadeh et al.
year: 2026
---

# Reward-Gated On-Policy Distillation

## TL;DR

RG-OPD applies reverse-KL teacher supervision only when verifier reward and the
teacher–student likelihood gap agree directionally. It prevents a teacher from
pulling the student away from a verified success or reinforcing a failure the
teacher also likes.

## Method and evidence

- For a positive trajectory, distill only if the teacher assigns it higher
  likelihood than the student; for a negative trajectory, distill only if the
  teacher assigns it lower likelihood.
- Qwen2.5-14B teaches Qwen2.5-1.5B on UltraInteract.
- At 1K generation length, the reported average is 46.94 versus 44.02 for
  reverse-KL; at 8K the paper reports a larger average advantage.
- Instruction following is an exception: RG-OPD remains below the untuned
  student on IFEval, so the gate does not eliminate all knowledge/behavior
  loss.
- The paper's stated GitHub URL returned “repository not found” during the
  2026-07-17 vault pass; the PDF is archived, but no code checkout is pinned.

### Formal gate

For student trajectory `i`, the method compares summed teacher and student
log-likelihoods, `L_T^(i)` and `L_S^(i)`, and the trajectory's GRPO advantage
`A_i`. It retains the trajectory for reverse-KL supervision only when

`(A_i > 0 and L_T > L_S + delta)` or
`(A_i <= 0 and L_T < L_S - delta)`.

This is an **abstention gate**. It can suppress teacher supervision, but it
does not assign negative credit to the teacher or estimate that the opposite
action would have been better. The top-50 reverse-KL approximation includes a
residual-tail correction (Appendix A.2).

### Experimental design and reported results

- Qwen2.5-14B-Instruct teaches Qwen2.5-1.5B-Instruct on an UltraInteract
  subset for three epochs.
- Evaluation spans GSM8K, GSM-Plus, MATH, MMLU-Pro-Math, MBPP, IFEval, and
  four additional reasoning/knowledge tasks.
- At 1K generation length, the reported six-task average is `46.94` for
  RG-OPD, `44.02` for reverse-KL, and approximately `42.03` for TSD-KD.
- At 8K, the corresponding RG-OPD and reverse-KL averages are `53.74` and
  `50.90`; the untuned student is `45.59`.
- IFEval is a persistent counterexample: RG-OPD remains about 3.5 points below
  the untuned student at both lengths.
- The paper's prose calls `53.74 - 50.90` a 6.8-point gain in one location;
  the arithmetic is 2.84 points, consistent with the abstract's 2.8-point
  claim.

### Limits that matter here

- `A_i` is a group-relative GRPO advantage, but parts of the prose describe
  its sign as if it were simply correct versus incorrect. Those notions can
  diverge when sibling rewards tie or have different magnitudes.
- Low teacher likelihood on a failed trace does not establish that the
  teacher knows a better trace.
- The decision is trajectory-level: every token in an admitted trace receives
  supervision, whether or not it caused success.
- Only one teacher/student pair is evaluated. Reported variation is across
  generation seeds from one trained checkpoint, not independent training
  runs.
- The likelihood gap is not a counterfactual action-value estimate and has no
  external-action price.

## Bearing on our work

Reward-gated OPD already occupies a generic “trust the teacher only when useful”
claim. It is a mandatory method baseline, not our novelty. Compute-elasticity
evaluation adds a different failure test: whether the gate preserves multiple
effort modes and long-budget behavior rather than improving only average task
score.

For [[compute_elasticity_handoff_2026-07-17/10-student-specific-action-value]], RG-OPD is the nearest baseline for
**verifier-gated teacher trust**, but it answers a different question. It asks
whether realized reward and teacher likelihood point in compatible directions;
we ask whether a fixed external action has positive net value for the target
student and whether importing the teacher's action boundary creates student
regret.

Useful follow-ups are:

1. compare the RG gate with the sign of an independently estimated student
   action value;
2. report utility-weighted false acceptance and rejection of teacher advice;
3. separate correctness, raw verifier reward, and group-relative advantage;
4. if training becomes justified, compare vanilla OPD, RG-OPD,
   student-value-gated OPD, and no distillation at matched rollout compute.

## Code custody

The paper advertises `https://github.com/UoC-tail/RG-OPD`. The URL returned
HTTP 404 and `git ls-remote` failed on 2026-07-17. Record this as **claimed
official repository, currently inaccessible**; do not substitute an
unofficial implementation.

## Raw source

EIT PDF `papers/arxiv_2607.04037.pdf`.

## Links

[[compute_elasticity_handoff_2026-07-17/10-student-specific-action-value]] · [[action-value-transport-reading-packet-2026-07-17]] ·
[[compute-elasticity-distillation]] · [[rethinking-privileged-opd]] · [[sdar]]
