---
title: OPID — On-Policy Skill Distillation for Agentic RL
type: source
tags: [agents, opd, skills, reinforcement-learning]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2606.26790
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2606.26790.pdf
repo: /engrfs/project/jacobsn/hiqbal/literature/legalrag/repos/OPID
year: 2026
---

# OPID

## TL;DR

OPID extracts episode-level workflows and critical step-level skills from
completed on-policy trajectories, then distills the skill-conditioned behavior
back into the policy alongside outcome-based RL.

## Evidence and bearing

It evaluates ALFWorld, WebShop, and search QA, and compares the sampled action
under ordinary and skill-conditioned context to construct a skill advantage.
This closes broad “Skill0 + OPD for agents” novelty. It is an important later
baseline, but it does not make changing resource price or unseen-price
cross-scale preservation the audited object.

## Raw source

EIT PDF `papers/arxiv_2606.26790.pdf`; pinned repo `repos/OPID`.
