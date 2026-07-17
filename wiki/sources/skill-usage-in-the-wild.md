---
title: How Well Do Agentic Skills Work in the Wild
type: source
tags: [skills, evaluation, retrieval, refinement, cross-model]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2604.04323
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2604.04323.pdf
repo: /engrfs/project/jacobsn/hiqbal/literature/legalrag/repos/Skill-Usage
authors: Liu et al.
year: 2026
---

# How Well Do Agentic Skills Work in the Wild

## TL;DR

This study tests skill retrieval, loading, and refinement against 34,198 public
skill packages using three model–harness pairs. Skill benefit becomes fragile
under realistic retrieval and can reverse by reader. It closes generic claims
about robust, model-independent contextual skill utility.

The treatment pipeline changes what each reader retrieves, loads, and refines.
It does not rank several identical same-task artifacts across readers or move
them into weights.

## Evidence

- Evaluation covers 84 SkillsBench tasks after exclusions, 89 Terminal-Bench
  2.0 tasks, and three runs per condition.
- Claude Opus 4.6/Claude Code scores 35.4% with no skill, 55.4% with
  force-loaded curated skills, and 38.4% after retrieval without curated
  skills.
- Kimi K2.5/Terminus-2 scores 21.8% no-skill and 19.8% with retrieved
  non-curated skills; Qwen3.5-397B/Qwen-Code scores 20.5% and 19.7%.
- Curated-skill loading is about 62% for Claude and 86% for Kimi, yet Kimi
  gains less: loading is not utility.
- Semantic retrieval Recall@3/5/10 is 38.1/47.0/52.3; agentic hybrid retrieval
  with content reaches 57.3/65.5/68.3.
- With curated skills available, query-specific refinement changes Claude
  40.1 to 48.2, Kimi 33.5 to 26.7, and Qwen 26.7 to 30.8.

## Novelty boundary

The work occupies contextual help/null/harm under realistic selection,
reader-dependent refinement, and the distinction between availability,
loading, and benefit. Each reader can receive a different retrieved/refined
set, so it does not cross exact artifact bytes. Its force-loaded condition has
one curated task bundle, not several same-task candidates; it reports no
per-artifact rank transport, top-k agreement, or source-selection regret.
Everything remains external context.

The design implication is strict: freeze candidate set, bytes, renderer,
availability, and loading policy for the causal matrix. Record both available
and actually loaded/used. Model and harness must be separated where possible.

## Code and data custody

- Official repository: https://github.com/UCSB-NLP-Chang/Skill-Usage.
- EIT checkout pinned at
  `03446d16f7b659ccc93ac5bd512f62e9b7fabb45` on 2026-07-17; no tags,
  releases, or detected repository-level license.
- Dataset: https://huggingface.co/datasets/Shiyu-Lab/Skill-Usage, observed
  commit `bc2a24d21a013cf30596dc758dbf44750e7211f6`. The dataset payload is
  linked but not mirrored in the Git-repository manifest.
- PDF SHA-256:
  `dd3c1ec25f258e655413751192b2dec9dfefc3063c28882748efa86e446e6ad4`.

## Links

[[skill-lifecycle-research-snapshot-2026-07-17]] · [[skillsbench]] ·
[[skillrae]] · [[skillsinjector]]
