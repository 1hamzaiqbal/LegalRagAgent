---
title: CUE-R — Intervention-Based Evidence Utility in RAG
type: source
tags: [rag, evidence-utility, interventions, non-additivity]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2604.05467
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2604.05467.pdf
authors: Jain and Vedam
year: 2026
---

# CUE-R: Beyond the Final Answer in RAG

## TL;DR

CUE-R measures individual evidence utility using REMOVE, REPLACE, and DUPLICATE
interventions, then records changes in correctness, grounding, confidence
error, and observable traces. It also demonstrates that evidence effects are
not simply additive when multiple supporting items are present.

## Bearing on our work

CUE-R is the closest methodological neighbor to a causal helpfulness metric.
Our differentiation must be empirical and structural: reader-size/task
crossovers, gold-present and gold-absent effects, evidence **sets**, repeated
outcomes, legal data, and search cost. Its intervention operators should be
adopted as baselines rather than reinvented.

## Limitations we can address

The paper uses small 100–200-example slices, single-shot RAG, shallow proxy
signals, a limited intervention family, and leaves combinatorial evidence
interactions open.

## Links

[[helpfulness-benchmark]] · [[answer-conversion-gap]] · [[three-dial]]
