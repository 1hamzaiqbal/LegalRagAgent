---
title: Skill1 — Unified Evolution of Skill-Augmented Agents via RL
type: source
tags: [agentic-rl, skills, selection, distillation]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2605.06130
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2605.06130.pdf
authors: Shi et al.
year: 2026
code: https://github.com/AlphaLab-USTC/Skill1
---

# Skill1: Unified Evolution of Skill-Augmented Agents via RL

## TL;DR

Skill1 trains one policy to search for, rank, use, and internalize reusable
skills from a persistent library using a shared task-outcome signal. It reports
97.5% on ALFWorld and improvements on WebShop. This occupies the broad claim
that skill selection and distillation can co-evolve in one agent.

## What matters to us

- The policy generates a skill-search query, reranks candidates, acts with the
  selected skill, and distills successful behavior back into itself.
- A low-frequency outcome-credit signal trains selection; a higher-frequency
  residual signal trains utilization/distillation.
- The method links library maintenance and parametric internalization instead
  of treating them as separate stages.

## Consequence and differentiation

We cannot claim novelty for generic skill evolution or “skills disappear into
the weights.” The remaining research question is narrower: can a stronger
reader or skill-augmented retrieval policy teach a smaller model the
reader-conditioned marginal value of search/evidence in a professional legal
domain, and improve a measured cost/accuracy frontier?

## Links

[[skill0]] · [[sdar]] · [[skill-distillation-bridge]] ·
[[opd-skill0-design]]
