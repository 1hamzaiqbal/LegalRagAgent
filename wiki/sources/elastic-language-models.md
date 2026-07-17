---
title: On Elastic Language Models
type: source
tags: [compute-elasticity, distillation, dynamic-depth, retrieval]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2311.07204
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2311.07204.pdf
authors: Chen Zhang, Benyou Wang, Dawei Song
year: 2023
---

# On Elastic Language Models

## TL;DR

ElasticLM already uses the term “compute elasticity” for one distilled model
that serves multiple latency/performance operating points. It trains nested
parameter-sharing submodels and schedules depth according to request load.

## Scope

- BERT-like masked language models rather than generative reasoning models.
- Task-agnostic relation-alignment distillation from BERT/Condenser/PROP.
- Evaluated on GLUE, Natural Questions, TriviaQA, and MS MARCO retrieval or
  reranking.
- Elasticity is architectural depth/parameter use and serving concurrency,
  not a learned deliberative action policy.

## Bearing on our work

“Compute elasticity” cannot be presented as a new term or general model
property. If retained, it must be qualified as *behavioral* or *agentic*
elasticity and cite this prior. The more precise candidate name is
cost-conditioned skill internalization or resource-conditioned agent
distillation.

## Raw source

EIT PDF `papers/arxiv_2311.07204.pdf`. The authors did not provide an official
code repository in the paper or search results reviewed here.

## Links

[[compute-elasticity-distillation]] · [[bard-budget-aware-reasoning-distillation]]
