---
title: LRAGE: Legal Retrieval Augmented Generation Evaluation Tool
type: source
tags: [legal-rag, evaluation, component-ablation, benchmark]
created: 2026-07-17
updated: 2026-07-17
status: triaged
url: https://arxiv.org/abs/2504.01840
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2504.01840v1.pdf
authors: Oh et al.
year: 2025
venue: arXiv
code: https://github.com/hoorangyee/LRAGE
---

# LRAGE

## Why it matters

LRAGE treats legal RAG performance as the interaction of corpus, retriever,
reranker, reader LLM, and evaluation rubric. It supports Korean, English, and
Chinese legal benchmarks and exposes both CLI and GUI experiment surfaces.

This does not pre-empt the three-dial estimand, but it does pre-empt any broad
claim that jointly varying RAG components is new. Our contribution must be the
paired reader-conditioned utility of an evidence set and the marginal decision
to retrieve or stop, not another component grid.

## Design consequences

- Report absolute performance and component deltas; never collapse retriever,
  reranker, reader, and rubric changes into one method comparison.
- Preserve the reader axis explicitly: the same retrieved set can reverse sign
  across readers even when retrieval metrics are fixed.
- Compare against a component-ablation framework such as LRAGE when arguing
  that a learned effort controller adds value beyond fixed configurations.

## Reading state

The abstract, system framing, related-work positioning, and component contract
were reviewed during archive migration. A full table-by-table read and code
checkout remain TODO before submission-facing related-work language is frozen.

## Links

[[three-dial]] · [[zheng-cslaw]] · [[legal-rag-benchmarks-src]] ·
[[answer-conversion-gap]] · [[predicting-retrieval-utility]]

## Raw source

- EIT PDF: `papers/arxiv_2504.01840v1.pdf`; pinned code: `repos/LRAGE`
- SHA-256: `c7ce2b8871ce4486ea70170d630e74a4e178ede8e16ff2f92c294fe46882a370`
