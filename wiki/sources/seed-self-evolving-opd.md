---
title: SEED — Self-Evolving On-Policy Distillation for Agentic RL
type: source
tags: [skills, opd, self-evolution, agentic-rl, internalization]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2607.14777
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2607.14777.pdf
repo: /engrfs/project/jacobsn/hiqbal/literature/legalrag/repos/SEED
authors: Wu et al.
year: 2026
---

# SEED

## TL;DR

SEED teaches a policy to turn completed trajectories into hindsight skills,
then repeatedly re-scores new on-policy actions with and without those skills.
The skill-induced log-probability shift gates an OPD term trained jointly with
GRPO. The analyzer and skills disappear at deployment.

This closes the generic claim that self-evolving natural-language skills can
be internalized through task reward plus gap-gated on-policy distillation. It
is especially close to the corrected E3 design already recorded locally.

## Method

1. An external GLM-5.2 annotates ordinary trajectories with episode-level
   hindsight skills; SFT teaches the backbone to generate such analyses.
2. The current frozen policy acts and separately analyzes its trajectories.
3. Sampled actions are re-scored under plain and skill-augmented contexts.
4. A sigmoid of `logp_skill - logp_plain` gates the dense OPD signal.
5. Training uses `L_GRPO + lambda_opd * L_opd`; the skill branch and gate are
   detached. The reported settings use beta 5 and `lambda_opd = 0.01`.

The result is same-model self-distillation rather than transfer from an
independent, larger teacher.

## Evidence

- On Qwen2.5-3B, SEED versus GRPO scores `91.8 vs 75.0` on ALFWorld,
  `45.7 vs 36.4` on Search-QA, and `78.9 vs 63.3` WebShop success.
- The ALFWorld ablation scores `91.8` for full SEED, `86.0` without
  hindsight-skill SFT, `87.0` without self-evolving OPD, and `84.4` with
  static instead of on-policy skills.
- It is not uniformly best. For Qwen2.5-7B, Search-QA `48.6` trails the
  `49.0` reported for RLSD/SDAR, while WebShop success `78.1` trails SDAR's
  `82.8`.

## Novelty boundary

SEED is not [[skillopt]]: it does not maintain a bounded-edit, versioned skill
document accepted on a held-out gate. It is, however, a near-exact collision
with `SKILL0-style training-only skills + task RL + negative-gap-gated OPD`.

Remaining questions include whether an independently optimized artifact is
good teaching material for another model, whether cross-scale transfer
preserves the target student's action value, and where a revisable skill
should live after repeated updates. SEED must be compared directly if a new
method advances beyond measurement.

## Limits

The authors note that shared actor/analyzer blind spots can reinforce bad
rules, confidence gating does not establish semantic correctness, and paired
analysis/scoring cost grows with horizon and multimodal context. Stage 1 still
depends on an external analyzer. The paper reports no independent training-
seed confidence intervals or significance tests, and its main ablation is
localized to ALFWorld. Calls, tokens, latency, and price-conditioned utility
are not evaluated.

## Code custody

- Official repository: https://github.com/jinyangwu/SEED.
- Project page: https://jinyangwu.github.io/seed/.
- Persistent EIT checkout pinned at
  `2cf2fadca3c5aba28da68e8e1405182ba8d90e6c` on 2026-07-17.
- PDF SHA-256:
  `d292c0320f3beaf9d5561c24c9a044fe9286dcc2217f899a5300fb5550dca565`.

## Links

[[skill-lifecycle-research-snapshot-2026-07-17]] · [[skill-sd]] · [[skill0]] ·
[[skillc]] · [[opcd]] · [[opd-distillation]]
