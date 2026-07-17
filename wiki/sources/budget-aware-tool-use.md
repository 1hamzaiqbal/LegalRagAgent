---
title: Budget-Aware Tool-Use Enables Effective Agent Scaling
type: source
tags: [tool-use, test-time-scaling, budgets, cost]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2511.17006
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2511.17006.pdf
authors: Liu et al.
year: 2025
---

# Budget-aware tool use and BATS

## TL;DR

This paper shows that merely granting more tool calls does not make an agent
use them well. A prompt-level Budget Tracker exposes used and remaining calls,
while the training-free BATS orchestration adapts planning and verification to
the remaining budget. Evaluation uses a unified post-hoc economic cost for
tokens and tool calls.

## What it occupies

- Explicit inference-time budget awareness as a strong prompt-only baseline.
- Dynamic decisions to deepen a promising lead, stop, or pivot under remaining
  resources.
- Cost/performance curves that jointly account for internal token consumption
  and external actions.

## Important limits and opening

The hard constraint is tool-call budget; token use enters the realized cost
metric rather than a learned multidimensional allocation policy. The authors'
limitations explicitly leave joint resource constraints and principled
resource allocation open. That supports—but does not prove—the narrower
[[effort-conditioned-resource-allocation]] question.

Any learned policy must beat Budget Tracker/BATS at matched task, model,
retriever, and realized cost. Otherwise training added complexity without a
scientific or practical gain.

## Raw source

EIT PDF: `papers/arxiv_2511.17006.pdf`.

## Links

[[acting-less-otc]] · [[autosearch]] ·
[[budget-constrained-agentic-search]] · [[three-dial]]
