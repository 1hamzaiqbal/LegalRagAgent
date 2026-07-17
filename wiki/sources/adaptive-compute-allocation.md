---
title: Adaptive Test-Time Compute Allocation via Constrained Policy Optimization
type: source
tags: [test-time-compute, allocation, budget, reasoning]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2604.14853
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2604.14853.pdf
code: https://github.com/zhiyuanZhai20/AdaCompute-LLM
authors: Zhiyuan Zhai, Bingcong Li, Bingnan Xiao, Ming Li, Xin Wang
year: 2026
---

# Adaptive Test-Time Compute Allocation

## TL;DR

AdaCompute formalizes per-question test-time sampling allocation under an
average budget. It first computes a Lagrangian oracle
`argmax_b Acc(x,b) - lambda*C(b)`, then trains a lightweight classifier to
imitate the oracle from cheap prompt and one-pass features.

## Evidence from the paper

- The budget set is `{1,2,4,8,16}` independent samples, aggregated by
  self-consistency.
- The learned controller is external to the LLM; the base model is frozen.
- MATH/GSM8K experiments use DeepSeek-V3, GPT-4o-mini, and Qwen2.5-7B.
- The paper reports up to 12.8% relative accuracy improvement under matched
  budgets and over 91% oracle-action imitation.
- The repository describes closing 82–95% of the oracle gap and provides the
  full solve/feature/classifier/evaluation pipeline.

## Bearing on compute elasticity

This occupies generic input-adaptive test-time allocation and the same
Lagrangian price construction. [[compute-elasticity-distillation]] must not
claim novelty for `Acc - lambda*cost`, difficulty-aware allocation, or learned
budget assignment. Its distinct question is whether a smaller *generative
policy* can inherit the teacher's conditioned frontier and behaviors rather
than rely on an external sample-count classifier.

## Raw source

EIT PDF `papers/arxiv_2604.14853.pdf`; pinned repository
`repos/AdaCompute-LLM`.

## Links

[[compute-elasticity-distillation]] · [[effort-conditioned-resource-allocation]]
