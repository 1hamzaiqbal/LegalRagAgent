---
title: What Should a Skill Remember? Cost-Aware Skill Rewriting
type: source
tags: [skills, cost, rewriting, agents]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2606.09421
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2606.09421.pdf
code: https://github.com/1Reminding/Skill_EE
authors: Qinghua Xing et al.
year: 2026
---

# Cost-aware skill rewriting

## TL;DR

This work treats skill rewriting as quality-preserving economic optimization,
because shortening a skill can increase downstream exploration/debugging cost.
It profiles task/skill structure, generates anchor-preserving rewrites, and
learns a task-conditioned rewrite-strategy selector.

## Evidence

- SkillsBench evaluation separates direct skill tokens, downstream agent
  tokens, total execution cost, quality retention, and overruns.
- The learned policy reports 7.0% total-cost and 6.0% downstream-token savings
  on the main held-out evaluation.
- Frozen cross-model transfer reports larger average savings while preserving
  verifier quality.
- This edits external skill documents; it does not internalize skills into
  model parameters.

## Bearing on our work

Cost-aware skill design is already occupied. The remaining SKILL0-style
question is parameter internalization of *cost-conditioned action selection*,
not compression or selection of an external skill document.

## Raw source

EIT PDF `papers/arxiv_2606.09421.pdf`; pinned repository `repos/SkillEE`.

## Links

[[compute-elasticity-distillation]] · [[skill0]] · [[skill1]]
