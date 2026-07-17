---
title: Skill0.5 — Joint Skill Internalization and Utilization
type: source
tags: [skills, internalization, externalization, agentic-rl, ood]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2605.28424
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2605.28424.pdf
repo: /engrfs/project/jacobsn/hiqbal/literature/legalrag/repos/Skill0_5
authors: Zhu et al.
year: 2026
---

# Skill0.5

## TL;DR

Skill0.5 rejects an all-context versus all-weights choice. It internalizes
general skills through privileged distillation on hard tasks, applies ordinary
GRPO on medium tasks, and deliberately keeps task-specific skills external for
OOD use. Paired with/without-specific-skill probes on easy tasks discourage
the policy from bypassing new external guidance.

This closes the broad idea that an agent should place some skills in weights
and others in context. It does not learn placement from model-specific
teachability, retention, update frequency, or cost.

## Method

- A pass-rate router assigns each task to hard, medium, or easy.
- Hard tasks use successful general-skill-augmented teacher rollouts and
  token-level Jensen-Shannon distillation into the standard-prompt student.
- Medium tasks use ordinary GRPO.
- Easy tasks compare success with and without the retrieved task-specific
  skill and use the utilization difference as an advantage correction.
- At deployment, general skills are absent from context, while unseen
  task-specific skills remain available through retrieval.

## Evidence and limits

On ALFWorld and WebShop, the paper reports gains over SkillRL, SKILL0, SLIM,
and memory/RL baselines, especially on held-out task domains. The
internalize-only and utilize-only ablations both lose to the joint method.

The evidence is limited to two text environments, a manually semantic
general/specific split, one skill-bank construction, and relatively narrow OOD
definitions. It does not compare source-optimized versus student-optimized
skills, continual skill revisions, or context rescue after forgetting.

## Code custody

- Official repository: https://github.com/JasonZhujp/Skill0_5.
- EIT checkout pinned at
  `703e635619901c9c84f76caff0907a37d1a262a8` on 2026-07-17.
- PDF SHA-256:
  `02565c09fd529fabb8bc2ef805e9f6c4c6943a04a0e58b5ca13ea85f173e8cba`.

## Links

[[skill-lifecycle-research-snapshot-2026-07-17]] · [[skill0]] · [[skillc]] ·
[[skill-sd]] · [[latent-skill]]
