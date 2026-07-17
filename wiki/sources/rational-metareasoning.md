---
title: Rational Metareasoning for Large Language Models
type: source
tags: [metareasoning, value-of-computation, cost, reasoning]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2410.05563
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2410.05563.pdf
authors: C. Nicolo De Sabbata, Theodore R. Sumers, Thomas L. Griffiths
year: 2024
---

# Rational Metareasoning for LLMs

## TL;DR

This work trains models to reason only when the expected value of computation
justifies its token cost. Expert Iteration selects high-value rollouts under a
value-of-computation reward, then distills them with cross-entropy.

## Evidence and limits

- Reported token reductions are 23–42% across the tested model/dataset cells
  while maintaining or improving performance.
- The method learns one cost-aware policy rather than user control over a
  budget continuum.
- The authors explicitly identify agentic tool/API costs as future work.

## Bearing on our work

Value-of-computation and selective reasoning are prior art. The clearest
remaining extension is cross-scale, condition-controlled choice between
internal reasoning and costly external actions—not a new token penalty.

## Raw source

EIT PDF `papers/arxiv_2410.05563.pdf`.

## Links

[[compute-elasticity-distillation]] · [[adaptive-compute-allocation]]
