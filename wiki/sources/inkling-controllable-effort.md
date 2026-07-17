---
title: Inkling — Controllable Thinking Effort
type: source
tags: [reasoning-efficiency, effort-control, agentic-rl]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://thinkingmachines.ai/news/introducing-inkling/
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/web/thinking-machines-inkling.html
authors: Thinking Machines Lab
year: 2026
---

# Inkling: controllable thinking effort

## TL;DR

Thinking Machines reports that Inkling's large-scale RL varied the system-level
effort instruction and per-token cost across samples. The resulting model uses
different reasoning lengths under different effort settings and exposes a
cost/performance sweep at inference.

## What the primary source establishes

- The post reports more than 30 million asynchronous RL rollouts.
- Effort was varied per sample through both a system message and per-token
  cost, rather than training only one maximum-effort behavior.
- The released model exposes a continuous effort setting and shows different
  performance/token operating points.
- More concise reasoning style emerged during RL even though stylistic
  compression was not directly rewarded.

## What it does not establish publicly

The post is a model-release report, not a complete methods paper. It does not
give the exact reward equation, distribution of per-token costs, coupling
between effort text and cost, or ablations separating those two signals.
Accordingly, `task reward - lambda * token count` is our shorthand
interpretation, not a quoted Inkling objective.

## Bearing on our work

Inkling supports the mechanism intuition in
[[effort-conditioned-resource-allocation]], but not a standalone novelty claim.
[[l1-length-control]] already trains one model to obey varied prompt-level
reasoning budgets, while [[autosearch]] and [[acting-less-otc]] extend
efficiency learning to external actions. Our possible opening is the joint,
reader-conditioned choice among thinking, evidence acquisition, and
verification.

## Raw source

EIT snapshot: `web/thinking-machines-inkling.html`.

## Links

[[training-language-models-to-reason-efficiently]] · [[l1-length-control]] ·
[[compute-elasticity-distillation]] · [[three-dial]]
