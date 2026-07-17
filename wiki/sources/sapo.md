---
title: SAPO — Co-Evolving Skill Generation and Policy Optimization
type: source
tags: [skills, utility, reinforcement-learning, skill-bank, agents]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2606.08755
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2606.08755.pdf
repo: /engrfs/project/jacobsn/hiqbal/literature/legalrag/repos/skill_augmented_agent
authors: Zhang et al.
year: 2026
---

# SAPO

## TL;DR

SAPO already measures a candidate skill's context-conditional marginal reward
under the current policy and co-retrieved skill set, then uses that utility to
gate storage, rerank retrieval, prune outdated skills, and train the policy as
a skill generator. Generic claims about first execution-grounded, harmful, or
policy-dependent skill utility are occupied.

SAPO co-evolves its candidates, policy, context, and evaluation queries. It
does not cross the same versioned candidate set over fixed readers or compare
contextual ordering with post-withdrawal acquisition ordering.

## Utility and method

For policy `pi`, task `x`, retrieved set `S`, and candidate `s`, SAPO estimates

`u(x,s) = mean reward[pi(.|x,S+s)] - mean reward[pi(.|x,S)]`.

It averages this over `K` related induction queries, then promotes a skill only
if utility is positive, in the top fraction, and nonredundant with the long-
term bank. Utility-weighted likelihood trains the policy to generate useful
skills; generation likelihood becomes a cheaper reranking/pruning score as the
policy evolves.

## Evidence

- ALFWorld aggregate success is 92.2 for SAPO versus 89.9 for SkillRL.
- WebShop score/success is 90.5/78.1 versus 85.2/72.7 for SkillRL.
- Search-QA average is 47.8 versus 45.5 for SkillRL.
- Removing utility validation gives 90.6 ALFWorld and 83.0/75.0 WebShop
  score/success, versus 92.2 and 90.5/78.1 for the full method.
- Frontier-model-generated candidates have mixed mean utility; promoted and
  discarded subsets separate. Skill value can become outdated as the policy
  changes.

## Novelty boundary

SAPO occupies contextual marginal utility beyond existing skills, explicit
help/harm, policy dependence, utility-gated admission, and skill-bank
maintenance. It does not establish a common held-out ordering of multiple
independent artifacts: candidates are generated from base rollouts and usually
evaluated on their own induction queries while the reader and bank evolve.
Different backbones generate different artifacts, and there is no clean
artifact-specific withdrawal/weight column.

The candidate itself is induced from the base rollouts whose outcomes also
form the subtraction arm. Without paired random draws/common seeds, this can
inherit retry or regression-to-the-mean bias. Main utility curves have no
reported confidence intervals or independent seed counts.

## Required experimental consequence

For [[skill-lifecycle-research-snapshot-2026-07-17]], SAPO is the mandatory
contextual-utility baseline. Freeze reader checkpoint, co-skill context, exact
artifact bytes, and common held-out tasks; reset the same target checkpoint
for each artifact; and compare internalization against matched direct training.

## Code custody

- Claimed repository: https://github.com/zzwjames/skill_augmented_agent.
- EIT checkout pinned at
  `5410692f9a2e2fb4a1a564e2ea370dae13b7ed49` on 2026-07-17.
- The repository is a placeholder: one commit and a README containing only
  `# SAPO`; no code, tags, or releases.
- PDF SHA-256:
  `1f6e064f8f89b2c7ccbbe18ef1b3eed4cf99e7438f032968651dd734752782fb`.

## Links

[[skill-lifecycle-research-snapshot-2026-07-17]] · [[skillsinjector]] ·
[[skillopt]] · [[lifeskill]]
