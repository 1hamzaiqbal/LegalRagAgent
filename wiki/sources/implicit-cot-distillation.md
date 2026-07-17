---
title: Implicit Chain-of-Thought Reasoning via Knowledge Distillation
type: source
tags: [implicit-cot, hidden-states, distillation, reasoning-efficiency]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2311.01460
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2311.01460.pdf
code: https://github.com/da03/implicit_chain_of_thought
authors: Yuntian Deng et al.
year: 2023
---

# Implicit Chain-of-Thought via Knowledge Distillation

## TL;DR

This work distills an explicit-CoT teacher's hidden-state trajectory into an
emulator and student that reason “vertically” through hidden layers and emit
the answer without verbalizing the chain.

## Evidence and limitations

- On synthetic multiplication, implicit CoT solves cases unavailable to the
  no-CoT baseline and approaches no-CoT latency.
- On the paper's GSM8K augmentation, it reaches 22% direct-answer accuracy.
- It remains substantially below explicit CoT accuracy.
- The method depends on teacher hidden states and loses the interpretability of
  a textual rationale.

## Bearing on our work

Reasoning compression into hidden trajectories predates J-space. This is a
latent-efficiency baseline, not evidence that a model preserves a controllable
accuracy–cost frontier. It also requires open weights for the teacher.

## Raw source

EIT PDF `papers/arxiv_2311.01460.pdf`.

## Links

[[compute-elasticity-distillation]] · [[coconut]] · [[lori]]
