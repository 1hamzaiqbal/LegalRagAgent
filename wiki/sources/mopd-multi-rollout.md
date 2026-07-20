---
title: MOPD - Multi-Rollout On-Policy Distillation
type: source
tags: [opd, multi-rollout, peer-conditioning, verifier, naming-collision]
created: 2026-07-20
updated: 2026-07-20
status: archived; abstract-level intake, full audit pending
url: https://arxiv.org/abs/2605.12652
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2605.12652.pdf
authors: Yu et al.
year: 2026
---

# MOPD - Multi-Rollout On-Policy Distillation

## Intake boundary

This is a different “MOPD” from [[mopd-multi-teacher]]. It uses the student's
other rollouts for the same prompt as privileged teacher context: successful
peers supply positive evidence, while failed peers can supply contrastive
negative evidence. The paper studies positive-peer imitation and a mixed
success/failure construction across code, math, science QA, and tool use.

The immediate relevance is methodological and terminological. Our group-of-four
math rollout already exposes successes and failures, so a future teacher could
condition on that group rather than score each sample independently. That
would change the teacher information set and would require a separate
preregistration, comparison against ordinary per-rollout OPD, and careful
custody of which peer outcomes were visible. It is not part of the current
O-teacher campaign.

This page is intentionally an abstract-level intake. Do not cite detailed
mechanism, numbers, or implementation claims until the full PDF and any
released code have been audited.

## Version and custody

- Audited metadata: arXiv v2, revised 1 June 2026.
- PDF SHA-256:
  `e56bac4079331cd2c68a7925623786a0742a4a774a4fd5bd4c8b968ab94f1bf6`.
- Persistent PDF:
  `/engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2605.12652.pdf`.

## Links

[[mopd-multi-teacher]] · [[verl-opd-trainer]] · [[opd-math-source-transfer]] ·
[[opd-distillation]]
