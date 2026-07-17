---
title: Strategy-Guided Policy Optimization
type: source
tags: [strategy-distillation, opd, reasoning, privileged-context]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2606.24064
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2606.24064.pdf
authors: Tianyuan Shi et al.
year: 2026
---

# Strategy-Guided Policy Optimization

## TL;DR

SGPO extracts reusable natural-language strategies from strong-model
responses, compares autonomous and strategy-guided student rollouts, and uses
forward-KL to transfer the distributional shift at strategy-critical tokens.
The distillation weight increases when guidance helps and vanishes as the
student becomes competent.

## Evidence and limits

- Evaluated on four math benchmarks across Qwen2.5 and Llama-3.2 families.
- The paper reports a 2.2-point average gain over its strongest Qwen2.5-7B
  baseline.
- It deliberately preserves alternative student strategies rather than
  collapsing to one teacher trajectory.
- It does not condition strategy selection on an inference-time resource
  price or include external tool actions.

## Bearing on our work

Reusable strategy distillation, marginal-helpfulness gating, and teacher-side
strategy descriptions are occupied. The possible contribution is the
orthogonal control problem: does the student internalize *which strategy or
tool is worth its cost under a supplied price*, including choosing none?

## Raw source

EIT PDF `papers/arxiv_2606.24064.pdf`. No primary code repository was located
during this pass.

## Links

[[compute-elasticity-distillation]] · [[agent-distillation-tools]] · [[skill0]]
