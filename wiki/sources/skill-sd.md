---
title: Skill-SD — Skill-Conditioned Self-Distillation for Multi-turn LLM Agents
type: source
tags: [skills, self-distillation, opd, agentic-rl, privileged-context]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2604.10674
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2604.10674.pdf
authors: Wang et al.
year: 2026
---

# Skill-SD

## TL;DR

Skill-SD is the closest direct collision with “skills + OPD + withdrawal.” The
student generates on-policy multi-turn trajectories under the plain task
prompt. An auxiliary model summarizes completed trajectories into compact
success, mistake, and workflow skills. Those skills condition only a teacher
view of the current policy, which re-scores the student's tokens. An
importance-weighted reverse-KL term is trained jointly with GRPO reward; the
deployment student receives no skill.

## Method and evidence

- The same student parameters define a plain-prompt student and a
  skill-conditioned teacher view; the teacher checkpoint is dynamically
  synchronized during the main method.
- Student-owned rollouts preserve the deployment conditioning. Off-policy
  teacher-owned rollout variants collapse during training.
- Skills are task-local and selected with a UCB rule; they are generated from
  past trajectories rather than optimized as a persistent held-out-gated
  artifact.
- On Qwen3-4B-Instruct-2507, the paper reports 64.9% AppWorld accuracy and
  62.5% Sokoban accuracy, versus 50.9% and 51.6% for vanilla GRPO and 22.8%
  and 21.9% for vanilla OPD.
- Direct skill-augmented GRPO underperforms vanilla GRPO on both reported
  benchmarks, illustrating train/deployment prompt mismatch.
- Dynamic synchronization beats the frozen-teacher ablation by 15.8 points on
  AppWorld and 12.5 on Sokoban at the reported endpoints.

## Novelty boundary

Skill-SD closes generic claims that natural-language skills can be generated
from trajectories, supplied only to a teacher, and internalized into a
plain-prompt policy through reverse-KL plus reward. [[skillopt]] remains
different because it maintains one versioned artifact and accepts edits on a
held-out selection set. The open question is whether such an artifact is good
**teaching material for another model**, not whether the two systems can be
chained.

## Limits

Same-size self-distillation, two agent benchmarks, task-local skills, and no
cross-scale or optimized-skill study. Sampled-token rather than
full-vocabulary distillation reduces cost but changes the target. The paper's
teacher synchronization result conflicts in direction with sequential fact
writing in [[continual-facts-in-weights]]; the regimes are different enough
that neither can be generalized a priori.

## Code custody

The official project page https://k1xe.github.io/skill-sd/ says “Code Coming
Soon” as of 2026-07-17. No repository was substituted.

- PDF SHA-256:
  `cc3d09abbe7ef6c4e0fefbdff3630721220aceb4bbcee053713b255f61c78ff7`.

## Links

[[skill-lifecycle-research-snapshot-2026-07-17]] · [[skillopt]] · [[skill0]] ·
[[skillc]] · [[seed-self-evolving-opd]] · [[opd-distillation]]
