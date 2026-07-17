---
title: Rethinking On-Policy Self-Distillation for Thinking Models
type: source
tags: [opd, privileged-context, negative-result, reasoning, test-time-compute]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2607.05184
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2607.05184.pdf
authors: Simran Kaur et al.
year: 2026
---

# Rethinking On-Policy Self-Distillation for Thinking Models

## TL;DR

Privileged-context distillation can improve short-budget results while harming
the long-budget reasoning of thinking models. Across five Qwen3 and OLMo
models, the paper reports relative avg@16 degradation up to 17%.

## Mechanism evidence

- Privileged context lowers the rate of high-entropy reasoning forks.
- The trained models use fewer verification, backtracking, reconsideration,
  and hedging markers even after length normalization.
- Full gold demonstrations cause more long-budget compression than sparse
  final-answer-only PI.
- Unprivileged OPD can improve the same short-budget setup, while adding
  privileged teacher context reverses the gain.
- The authors do not propose a fix and restrict experiments to verifiable math.

## Bearing on compute elasticity

This is the strongest direct motivation for studying *frontier preservation*.
A student can look better at the training budget while losing the behaviors
that make extra inference compute useful. Any OPD/PI method must evaluate a
budget curve and behavior markers, not one pass@1 point or average response
length.

## Raw source

EIT PDF `papers/arxiv_2607.05184.pdf`.

## Links

[[compute-elasticity-distillation]] · [[privileged-information-distillation]] ·
[[rethinking-opd]]
