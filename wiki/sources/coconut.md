---
title: Training Large Language Models to Reason in a Continuous Latent Space
type: source
tags: [coconut, latent-reasoning, test-time-compute, planning]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2412.06769
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2412.06769.pdf
code: https://github.com/facebookresearch/coconut
authors: Shibo Hao et al.
year: 2024
---

# Coconut: Continuous Latent Reasoning

## TL;DR

Coconut feeds the last hidden state back as the next input embedding during a
latent “continuous thought” phase. A curriculum gradually replaces explicit
reasoning steps with latent steps.

## Evidence and boundaries

- The paper reports advantages on planning-heavy logical reasoning and a
  better accuracy/token tradeoff on several tasks.
- Probing suggests one continuous thought can encode multiple alternative next
  steps, interpreted as a latent breadth-first search.
- Training directly from question/answer pairs without the curriculum does not
  beat no-CoT; language reasoning supervision remains important.
- The number of latent steps is fixed in the experiments. Autonomous stopping
  is proposed as future work.

## Bearing on our work

Coconut offers an open-weight latent-compute baseline and a possible later
student architecture. It does not transfer a cost-conditioned teacher frontier
or expose API-compatible training.

## Raw source

EIT PDF `papers/arxiv_2412.06769.pdf`; pinned repository `repos/coconut`.

## Links

[[compute-elasticity-distillation]] · [[implicit-cot-distillation]] · [[lori]]
