---
title: LifeSkill — Skill-Enhanced Test-Time Co-Evolution
type: source
tags: [skills, internalization, lifelong-learning, reinforcement-learning]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2606.04815
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2606.04815.pdf
authors: Mao et al.
year: 2026
---

# LifeSkill

## TL;DR

LifeSkill is a direct context-to-weights collision. It generates skills after
failed attempts, samples skill-conditioned successful trajectories, removes
the skill text, and reward-weights the same trajectory under the original task
input. The deployed policy uses zero retrieved experiences.

It does not hold artifacts fixed across readers or attribute post-withdrawal
gain to particular artifacts, so artifact-ordering transport remains open.

## Method

Verifier-Guided Skill Learning scores a candidate by conditional success:

`R_i = mean_j V(q, trajectory_i,j | skill_i)`.

Online Skill Internalization then takes successful `(q, skill, trajectory)`
samples, removes the explicit skill from the conditioning context, and updates
the policy with `-reward * log pi(trajectory | q)`. Skill extractor and policy
use separate LoRA adapters and co-evolve during the task stream.

`R_i` is not a no-skill delta. Because skills are proposed only after an
initial failure, it conflates skill benefit with a stochastic second attempt.

## Evidence

- With Llama-3.1-8B, LifeSkill scores 0.82/0.64/0.32 on DB/OS/KG, average
  0.59, versus the strongest training-based average of 0.52 for RFT.
- Removing Online Skill Internalization yields 0.79/0.60 on DB/OS versus
  0.82/0.64 for the full system.
- With Qwen2.5-7B, independently evolved LifeSkill scores
  0.83/0.65/0.38, average 0.62.
- Post-stream retention is 0.708/0.552/0.283 on DB/OS/KG.
- Three task-order runs report DB `0.820 +/- 0.044` and OS
  `0.640 +/- 0.056`.

## Novelty boundary

LifeSkill occupies execution-grounded skill-conditioned scoring, online
skill-guided exploration, scaffold removal, and parametric adaptation. Its
skills are freshly sampled per failure; reader and extractor co-evolve; and
the Llama/Qwen runs regenerate different artifacts. It never compares the
same artifact's contextual lift and post-withdrawal lift, crosses that artifact
across readers, or ranks fixed candidates in either placement.

The main table's `Experience=0` describes deployment prompts, not a clean
all-task withdrawal counterfactual: its online loop still uses freshly
generated skill-conditioned retries. Add a no-skill second-retry arm in any
adaptation and compare against matched direct training.

## Code custody

- ArXiv v2 says code and models will be released.
- No official repository was found as of 2026-07-17.
- PDF SHA-256:
  `ade2f426ed7af13c603ef41c1465fbf8261ecb71b6dc577fbbbd4a402e24222c`.

## Links

[[skill-lifecycle-research-snapshot-2026-07-17]] · [[skill0]] · [[skillc]] ·
[[sapo]] · [[constant-context-skill-learning]]
