---
title: AutoSearch — Adaptive Search Depth via RL
type: source
tags: [agentic-rag, search-effort, reinforcement-learning, cost]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2604.17337
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2604.17337.pdf
authors: Sun et al.
year: 2026
code: https://github.com/bofusun/AutoSearch
---

# AutoSearch: adaptive search depth for efficient agentic RAG

## TL;DR

AutoSearch trains an agent to find a capability-aware minimal sufficient
search depth. At every retrieval step, the current policy produces an
intermediate answer; the earliest exact match defines the trajectory's target
depth, and rewards favor useful steps while penalizing later over-search.

## Why this is a close neighbor

- The paper explicitly argues that appropriate depth depends jointly on
  question complexity and model capability.
- Its search-quality reward is the marginal gain in intermediate-answer F1
  from the new step.
- It compares against Search-R1, StepSearch, and HiPRAG and reports answer
  quality together with search depth, efficiency, over-search, and latency.
- Its own limitation notes that experiments use relatively low maximum search
  depths.

## Differentiation bar

“Capability-aware search effort” and “marginal answer improvement per search”
are occupied. The remaining distinction in [[three-dial]] is cross-reader,
counterfactual evidence-set utility: the searching policy and downstream
reader need not be the same model; evidence can harm; and set sufficiency,
conflict, authority, context cost, and user-specified prices affect the next
action. AutoSearch must nevertheless be implemented or faithfully reproduced
as a primary baseline.

## Raw source

EIT PDF: `papers/arxiv_2604.17337.pdf`; pinned code: `repos/AutoSearch`.

## Links

[[effort-conditioned-resource-allocation]] · [[acting-less-otc]] ·
[[budget-constrained-agentic-search]] · [[three-dial]]
