---
title: Predicting Retrieval Utility and Answer Quality in RAG
type: source
tags: [rag, utility, qpp, answer-quality, prior-art]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2601.14546
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2601.14546.pdf
authors: Tian, Ganguly, and Macdonald
year: 2026
---

# Predicting Retrieval Utility and Answer Quality in RAG

## TL;DR

This paper explicitly defines retrieval utility as the performance difference
between RAG and no-context generation. It studies retrieval performance
prediction (RPP) and generation performance prediction (GPP), finding that
standard QPP signals help inconsistently and that utility is difficult because
retrieved context interacts with the model’s parametric knowledge.

## Bearing on our work

The generic pitch “predict whether retrieval helps the LLM” is already direct
prior art. The useful opening is the part this paper leaves underdeveloped:
paired causal outcomes for particular readers, harm and break-even analysis,
evidence-set interactions, professional legal tasks, and explicit action cost.
Its predictors are necessary baselines for a three-dial controller.

## Links

[[helpfulness-benchmark]] · [[qpp]] · [[answer-conversion-gap]] ·
[[three-dial]]
