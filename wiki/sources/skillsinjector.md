---
title: SkillsInjector — Dynamic Skill Context Construction
type: source
tags: [skills, utility, selection, adaptive-budget, context-construction]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2605.29794
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2605.29794.pdf
authors: Li et al.
year: 2026
---

# SkillsInjector

## TL;DR

SkillsInjector already defines execution-grounded, per-task marginal skill
utility for a frozen reader, learns an adaptive skill budget, and models how a
set of injected descriptions should be presented. It closes claims that skill
helpfulness, harm, or adaptive context effort are unmeasured.

It does not compare the same fixed artifacts across readers or transfer them
into weights. Its utility definition is a mandatory existing baseline, not a
new metric for this project.

## Method

For task `t`, frozen actor, and injected skill set `C`, the paper defines

`U(t, C) = E[r(trajectory)]`

and single-skill marginal benefit

`Delta(t, s) = U(t, {s}) - U(t, empty)`.

Repeated executions supervise a planner that ranks skills and admits every
skill above a held-out threshold, including none. An 8B set-aware renderer
then rewrites selected descriptions conditional on the task and neighboring
skills while leaving skill bodies fixed.

## Evidence

- In the running example, no-skill success is 0.60 over five trials and the
  82 skills range from -0.20 to +0.40 marginal benefit.
- SkillsInjector scores 60.0/61.4/67.0 on Airline/Retail/Telecom versus
  37.6/51.2/40.0 with no skill and 24.4/40.5/24.6 with the full library.
- It scores 22.6 on SkillsBench and 82.7 on ALFWorld versus 5.2 and 67.1
  without skills; the abstract reports gains over the strongest baselines of
  6.1 and 7.3 points.
- Helpfulness and cost diverge. In Airline/Retail/Telecom, 83%/95%/60% of
  skills help, while 70%/68%/83% increase interaction cost.
- Removing the planner lowers the three tau2 scores to 47.2/51.4/52.5; a fixed
  budget gives 51.6/56.5/56.1. Removing the renderer gives
  55.2/59.6/65.8 and adds 6.1–8.2 agent messages.
- Primary evaluations use five seeds.

## Novelty boundary

SkillsInjector already owns repeated-execution single-skill lift, explicit
harm, target-specific utility ranking, adaptive selection count, context-set
interaction, description presentation, and execution cost. Its `Delta` is the
ordinary context-utility quantity required by
[[skill-lifecycle-research-snapshot-2026-07-17]].

It uses one frozen actor per benchmark: Qwen3-235B for tau2/SkillsBench and
Qwen3-8B for ALFWorld, with different task and skill libraries. It therefore
does not compare artifact rankings across readers. It also does not change the
actor's weights, withdraw a skill after training, or measure teaching utility.
Because the renderer rewrites descriptions, the final injected artifact is not
held fixed unless that component is frozen as an experimental factor.

## Required experimental consequence

Any utility-transport study should reuse SkillsInjector's no-skill arm,
repeated executions, fixed-budget controls, cost measurements, and marginal
benefit definition. The contribution must be ordering transport across reader
and placement, not renaming contextual utility.

## Code custody

- ArXiv v1 says code will be released upon publication.
- No official repository or project page was found as of 2026-07-17.
- PDF SHA-256:
  `151abc67a8717616d31f9cb832d78c9029f1cacab4e3b4d64e37a152191b2618`.

## Links

[[skill-lifecycle-research-snapshot-2026-07-17]] · [[ctx2skill]] ·
[[skillrae]] · [[skillopt]]
