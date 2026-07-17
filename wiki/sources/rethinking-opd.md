---
title: Rethinking On-Policy Distillation of Large Language Models
type: source
tags: [opd, distillation, reasoning, teacher-student]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2604.13016
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2604.13016.pdf
authors: Yaxuan Li et al.
year: 2026
---

# Rethinking On-Policy Distillation

## TL;DR

This paper argues that OPD succeeds only when the teacher and student have
compatible thinking patterns and the teacher supplies genuinely new knowledge,
not merely a higher benchmark score.

## Key findings

- Successful runs show rising overlap among high-probability teacher/student
  tokens; the shared top-k set carries 97–99% of combined probability mass.
- Same-recipe larger teachers may offer little transferable signal even when
  their benchmark score is higher.
- Off-policy teacher-rollout warmup and teacher-aligned prompt selection can
  rescue failing OPD by improving initial overlap.
- Dense teacher reward becomes less reliable with trajectory depth, creating a
  ceiling for long reasoning and agents.
- The experiments are mathematical; cross-domain, self-distillation, and
  long-horizon extensions are left open.

## Bearing on our work

Teacher size is not a sufficient E2 gate. We must measure a condition-specific
skill/frontier gap and thinking-pattern compatibility. This paper also argues
for a short off-policy cold start before an OPD arm and for explicit long-budget
audits.

## Raw source

EIT PDF `papers/arxiv_2604.13016.pdf`.

## Links

[[compute-elasticity-distillation]] · [[reward-gated-opd]] ·
[[rethinking-privileged-opd]] · [[turnopd]]
