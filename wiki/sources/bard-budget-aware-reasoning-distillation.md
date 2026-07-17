---
title: BARD — Budget-Aware Reasoning Distillation
type: source
tags: [reasoning-distillation, budget-control, small-models, rl]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2511.01470
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2511.01470.pdf
authors: Lujie Niu et al.
year: 2025
---

# BARD

## TL;DR

BARD already performs budget-conditioned reasoning distillation. It first SFTs
an 8B student on teacher CoTs compressed to multiple user-specified token
budgets, then applies RL with a multiplicative accuracy × budget-fidelity
reward.

## Evidence

- The user supplies an upper bound on thinking tokens.
- A powerful teacher generates a long CoT; an expert compressor creates
  multiple budget-specific versions of the same reasoning.
- The RL phase is necessary: direct RL without budget-aware SFT collapses, and
  an additive reward learns the shortcut of short wrong answers.
- The paper evaluates budgets from 500 to 8,000 tokens on AIME24, AIME25, and
  GPQA.
- Its qualitative analysis explicitly claims strategy shifts: tight budgets
  prune exploration/checking, while high budgets add alternative paths,
  verification, and self-correction.

## Bearing on our work

This closes the broad novelty claim in the first draft of
[[compute-elasticity-distillation]]. Neither “distill a controllable
accuracy–token frontier” nor “teach different reasoning strategies at
different budgets” is new. The surviving candidate must extend the condition
to costly *agent actions* and transfer skill-conditioned action selection, or
be an evaluation study of which distillation methods preserve elasticity.

## Raw source

EIT PDF `papers/arxiv_2511.01470.pdf`. No primary code repository was located
during this pass.

## Links

[[compute-elasticity-distillation]] · [[l1-length-control]] · [[crisp]]
