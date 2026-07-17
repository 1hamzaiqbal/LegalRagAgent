---
title: Budget-Constrained Agentic LLM Search (BCAS)
type: source
tags: [agentic-search, cost, budgets, rag]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2603.08877
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2603.08877.pdf
authors: McCleary and Ghawaly
year: 2026
---

# Quantifying Accuracy and Cost in Budget-Constrained Agentic Search

## TL;DR

BCAS measures search-depth, retrieval-strategy, and completion-token budgets
across six models and three QA datasets. Returns usually diminish after about
three searches; hybrid retrieval with lightweight reranking produces the
largest average gains, and extra retrieval budget often matters more than extra
generation budget.

## Bearing on our work

A static budget sweep is not a novel three-dial contribution. The opening is a
reader-conditioned policy that estimates the marginal value of the **next**
retrieval action, stops when expected utility turns non-positive, and reports a
cost/accuracy frontier against BCAS-style fixed-budget baselines.

## Caveats

The study lacks a pure non-agentic single-pass baseline in parts of its grid,
concentrates key ablations on HotpotQA, and inherits provider/cost variability.

## Links

[[three-dial]] · [[offline-bandit-v0]] · [[opd-skill0-design]]
