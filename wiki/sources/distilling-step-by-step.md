---
title: Distilling Step-by-Step
type: source
tags: [distillation, rationales, specialist-models, small-models]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2305.02301
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2305.02301.pdf
code: https://github.com/google-research/distilling-step-by-step
authors: Cheng-Yu Hsieh et al.
year: 2023
---

# Distilling Step-by-Step

## TL;DR

The method uses a large model's rationales as an auxiliary training target for
a task-specific student. The paper reports that a 770M T5 student can beat
few-shot CoT from 540B PaLM on several classification/reasoning benchmarks.

## What the result does and does not show

- The student predicts labels and rationales as separate tasks; rationale
  generation can be omitted at inference.
- The paper reports beating the teacher baseline on three of four datasets,
  with far less model capacity and labeled data.
- It requires roughly ten human-written demonstrations to elicit teacher
  rationales.
- Teacher rationale quality limits the method; complex planning remains a
  stated concern.

## Bearing on our work

This establishes that a narrow specialist can outperform a much larger
prompted generalist. The novel target cannot simply be “small beats large”; it
must concern transfer and preservation of a controllable effort frontier under
matched evaluation and cost.

## Raw source

EIT PDF `papers/arxiv_2305.02301.pdf`; pinned repository
`repos/distilling-step-by-step`.

## Links

[[compute-elasticity-distillation]] · [[deepseek-r1-distillation]] ·
[[thinking-machines-expert-judgment]]
