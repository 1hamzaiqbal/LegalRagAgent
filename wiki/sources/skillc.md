---
title: SkillC — Autonomous Skill Internalization via Contrastive Credit
type: source
tags: [skills, internalization, contrastive-credit, agentic-rl, curriculum]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2605.27899
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2605.27899.pdf
authors: Lin et al.
year: 2026
---

# SkillC

## TL;DR

SkillC samples paired skill-injected and skill-free rollouts inside the same
policy update. It turns the observed skill-dependence gap into direct credit
for autonomous success and uses a smoothed validation gap to control
attribution strength, rollout allocation, and monotonic skill retirement.

This is a direct precedent for measuring “does this context still help this
reader?” and using the result to drive internalization. It uses fixed skills
and one policy; it does not optimize the teaching artifact or transfer across
model scale.

## Method and evidence

- The task-level contrast is
  `Delta(x) = E[R | skill] - E[R | no skill]`.
- A dual-stream advantage preserves global reward ranking while downweighting
  skill-dependent trajectories and upweighting skill-free success when the
  contrast is positive.
- A validation-level moving average controls skill-rollout fraction and
  retirement; once retired, a skill does not re-enter.
- The paper reports improvements over SKILL0 on ALFWorld and WebShop without
  runtime skills. Paired sampling costs about 26% early overhead and about 30%
  total extra compute relative to GRPO in the reported setup.

The abstract and conclusion quote slightly different improvement summaries
(5.5/4.4 versus 4.7/3.1 points), apparently under different comparators or
summaries. Paper-facing use should cite the exact table cell, not the prose
headline.

## Limits

Fixed task-aligned skills, same-policy comparison, two environments, and a
validation-coverage bottleneck. The one-sided correction is set to zero when
the skill is neutral or harmful; it does not learn a signed replacement skill.
Rare/noisy task types can retire prematurely. There is no cross-scale teacher,
SkillOpt artifact, or sequential retention study.

## Code custody

The current paper advertises no official repository. The PDF is archived with
SHA-256
`88f24aa2bd32712d75750e22c6fc0367427a3acc3a308d492ff73469ea42de1d`.

## Links

[[skill-lifecycle-research-snapshot-2026-07-17]] · [[skill0]] · [[skillopt]] ·
[[skill-sd]] · [[skill-zero-five]]
