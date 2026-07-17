---
title: Acting Less Is Reasoning More — OTC-PO
type: source
tags: [tool-use, agentic-rl, cost, search-effort]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2504.14870
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2504.14870.pdf
authors: Wang et al.
year: 2025
---

# Acting Less Is Reasoning More: Optimal Tool Calls

## TL;DR

OTC-PO adds tool-use efficiency to PPO and GRPO. For each question/model pair,
it approximates the minimum tool calls among correct trajectories and rewards
correct rollouts that approach that count.

## What it occupies

- Correctness-minus-unnecessary-tool-use as an RL objective.
- Per-question and per-model adaptation rather than a single global tool
  quota.
- Search and code tools, with tool productivity reported as correct answers per
  call.

## Limits relative to our candidate

The primary cost is scalar tool-call count; the model is not asked to respond
to a user-selected cost vector, trade reasoning against evidence/context cost,
or predict whether a particular evidence set helps a separate downstream
reader. Those distinctions define the bar in
[[effort-conditioned-resource-allocation]], not the generic act-less reward.

## Raw source

EIT PDF: `papers/arxiv_2504.14870.pdf`.

## Links

[[autosearch]] · [[budget-aware-tool-use]] · [[three-dial]]
