---
title: ArbGraph — Conflict-Aware Evidence Arbitration
type: source
tags: [rag, conflict, graph, arbitration, long-form]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2604.18362
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2604.18362.pdf
authors: Niu et al.
year: 2026
---

# ArbGraph

## TL;DR

ArbGraph decomposes evidence into atomic claims, builds support/contradiction
relations, and propagates credibility signals before long-form generation.
The central thesis is that evidence consistency can matter more than evidence
breadth and that arbitration should be separated from generation.

## Bearing on our work

Generic graph-based conflict arbitration is already occupied. The legal
opening is not another unlabeled support graph; it is an authority-aware graph
whose edge/priority semantics reflect jurisdiction, court hierarchy, time, and
precedential treatment, evaluated on downstream legal decisions and cost.

## Links

[[conflictrag]] · [[sure-rag]] · [[three-dial]]
