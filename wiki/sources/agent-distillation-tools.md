---
title: Distilling LLM Agent into Small Models with Retrieval and Code Tools
type: source
tags: [agent-distillation, tool-use, retrieval, code, small-models]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2505.17612
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2505.17612.pdf
code: https://github.com/Nardien/agent-distillation
authors: Minki Kang, Jongwon Jeong, Seanie Lee, Jaewoong Cho, Sung Ju Hwang
year: 2025
---

# Agent Distillation with Retrieval and Code Tools

## TL;DR

This work distills complete thought/action/observation trajectories from a
Qwen2.5-32B retrieval-and-code agent into 0.5B–7B students. A first-thought
prefix improves teacher trajectories; self-consistent action generation
improves student test-time robustness.

## Evidence and limits

- Agent-distilled students beat CoT-distilled and RAG-enhanced CoT baselines
  across several math/factual tasks.
- The paper reports 3B/7B students matching or exceeding next-tier larger
  models in some cells.
- Student tool-use quality depends strongly on teacher trajectory composition.
- It uses one teacher and a fixed tool regime; it does not condition the
  deployment policy on token/tool prices or evaluate price response.

## Bearing on our work

Full agent behavior and tool use can already be distilled. A surviving gap
must concern *cost-conditioned tool choice*—whether one student substitutes
reasoning, retrieval, code, verification, and stopping when relative prices
change—rather than generic agent distillation.

## Raw source

EIT PDF `papers/arxiv_2505.17612.pdf`; pinned repository
`repos/agent-distillation`.

## Links

[[compute-elasticity-distillation]] · [[strategy-guided-policy-optimization]]
