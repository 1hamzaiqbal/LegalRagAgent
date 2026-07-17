---
title: OPCD — On-Policy Context Distillation
type: source
tags: [opd, context-distillation, cross-scale, privileged-context, skills]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2602.12275
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2602.12275.pdf
repo: /engrfs/project/jacobsn/hiqbal/literature/legalrag/repos/LMOps-opcd
authors: Zhang et al.
year: 2026
---

# OPCD

## TL;DR

On-Policy Context Distillation has the deployment-conditioned student generate
without privileged context, while a frozen context-conditioned teacher scores
the same student prefixes. Reverse KL then transfers the context's behavior
into the student. It includes larger-teacher-to-smaller-student transfer and
distillation of optimized system prompts.

This is an exact collision with the broad “optimized skill in teacher context
plus cross-scale OPD” mechanism. It is a mandatory baseline and likely the
right engine if [[skill-lifecycle-research-snapshot-2026-07-17]] reaches its
cross-scale stage.

## Method

- The student samples on-policy from `pi_student(y | x)` without context `c`.
- A teacher scores the same prefixes as `pi_teacher(. | c, x, y_<t)`.
- Training minimizes reverse KL, approximated over the student's top 256
  tokens.
- The paper studies experiential knowledge extracted from solution traces and
  system prompts optimized with MetaSPO.
- Its default is a frozen teacher; same-weight self-distillation and a larger
  cross-size teacher are also tested.

## Evidence

- With random contexts, math accuracy is `79.7 +/- 0.5` for OPCD versus
  `78.5 +/- 0.5` for off-policy context distillation; Frozen Lake is
  `26.5 +/- 6.4` versus `22.9 +/- 4.0`.
- With validation-filtered contexts, the corresponding comparisons are
  `80.9 vs 79.5` on math, `38.3 vs 35.2` on Frozen Lake, and
  `53.9 vs 51.6` on Sokoban.
- Medical optimized-prompt gains over off-policy distillation are 1.5, 5.3,
  and 3.8 points across three tested models. One safety configuration is 0.2
  points worse, so OPCD is not uniformly superior.
- Raw trace context hurts: `70.5` versus a `75.1` base. Extracted knowledge
  reaches `77.4`, and OPCD reaches `79.7`. Context quality is therefore a real
  safety variable, not merely a convenience.
- A frozen teacher strongly beats continuously updated self-distillation in
  the reported Sokoban (`53.9 vs 18.8`) and medical (`56.8 vs 50.0`) tests.

## Novelty boundary

OPCD already owns the following combination:

1. context or skill available only to a teacher view;
2. context-free student rollouts under deployment conditioning;
3. reverse-KL on those student trajectories;
4. optimized prompt behavior moved into weights;
5. larger teacher to smaller student transfer.

It does not compare the same skill's contextual utility for source and target
models, test whether source-context utility ranks teaching artifacts, measure
forced-action value for each actor, or optimize under target-student regret.
Those are the remaining measurement questions—not the act of context
distillation itself.

## Limits and audit cautions

There is no task-reward safeguard or signed skill-helpfulness gate. Narrow
domains, raw-context failures, and the weak self-distillation result limit
generalization. Most tables lack conventional multi-seed uncertainty.
Appendix procedures select a best checkpoint, and in one description the
three best checkpoints, using test accuracy; paper-facing comparisons should
therefore note that the test surface also participates in model selection.

## Code custody

- Official code:
  https://github.com/microsoft/LMOps/tree/main/opcd.
- Persistent EIT sparse checkout pinned at
  `23610e8491bcb45ef4ae4e9d47b109eb70597501` on 2026-07-17.
- PDF SHA-256:
  `9b37000d483584ff333d61cebc9f485afee367923d45128aea5b50f11db613af`.
- The pinned checkout includes post-release fixes to on-policy minibatching
  and top-k padding/attention masking; preserve the commit in reproductions.

## Links

[[skill-lifecycle-research-snapshot-2026-07-17]] · [[skillopt]] · [[skill-sd]] ·
[[seed-self-evolving-opd]] · [[opd-distillation]]
