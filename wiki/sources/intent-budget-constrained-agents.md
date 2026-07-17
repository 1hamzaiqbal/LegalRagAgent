---
title: INTENT — Budget-Constrained Agentic LLMs
type: source
tags: [agents, tool-use, budget, planning]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2602.11541
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2602.11541.pdf
year: 2026
---

# INTENT

## TL;DR

INTENT formalizes tool-augmented agents with priced, stochastic tool calls and
a hard monetary budget. It uses intention-based inference-time planning rather
than training a price-conditioned student.

## Evidence and bearing

The paper constructs a cost-augmented StableToolBench with changing prices and
budgets and reports that standalone or prompt-only agents frequently violate
budget constraints. This is a direct baseline for any dynamic-price tool-use
claim. The remaining question in [[01-research-question-and-novelty]] is
cross-scale preservation of a same-task price-response policy.

## Raw source

EIT PDF `papers/arxiv_2602.11541.pdf`. No primary code repository was located
during this pass.
