---
title: eXTC — Structured Prompt Optimization, Distillation, and RL
type: source
tags: [prompt-optimization, distillation, reinforcement-learning, classification, legal]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2605.29076
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2605.29076.pdf
authors: Yang et al.
year: 2026
---

# eXTC

## TL;DR

eXTC is a three-stage classification pipeline: optimize a natural-language
SOP/rulebook, use a larger model to generate SOP-grounded correct reasoning
traces for a smaller Qwen3-4B model, then apply task-reward RL on teacher-
failed cases. The student deploys without the SOP.

It is not OPD and not a general agent-skill system, but it closes novelty based
on the literal `optimize text → distill to a smaller model → RL` stage shape.
ContractNLI also makes it a direct legal-domain neighbor.

## Method and evidence

1. Structured prompt optimization edits a natural-language SOP/rulebook.
2. GPT-4.1-mini produces correct SOP-grounded traces; rejection-sampled traces
   SFT the student without requiring the SOP at inference.
3. BD-GRPO trains on cases the teacher failed, using task reward.

On ContractNLI, eXTC reports macro-F1/balanced accuracy `0.849/0.882`, versus
`0.802/0.811` for reasoning SFT and `0.847/0.865` for label-only SFT. The
result is not uniform: on ICLR Review, `0.825/0.829` does not beat SFT's
`0.825/0.832`; on MIMIC, `0.652/0.642` improves macro-F1 but has much lower
balanced accuracy than SFT's `0.726`.

## Novelty boundary

eXTC does not optimize a portable Codex-style `SKILL.md`, perform
context-conditioned reverse-KL, compare source and target contextual utility,
or study continual skill revision. It does establish that composing textual
rule optimization, large-to-small reasoning distillation, and outcome RL is
not itself new. A new project must make a sharper empirical claim and compare
the complete chain with direct target training and every shorter path.

## Limits and code custody

The study uses one seed (`42`), best-validation-checkpoint selection,
proprietary LLMs, judge-based rationale evaluation, and no multimodal setting.
The authors acknowledge that generated explanations may be unfaithful. The
paper/source says code is released but provides no repository URL, and no
official public repository was identifiable on 2026-07-17.

- PDF SHA-256:
  `6d3110e2e7e1722291d217cfd955c11130f4925963bb9ad6c5318b2ee23999fc`.

## Links

[[skill-lifecycle-research-snapshot-2026-07-17]] · [[skillopt]] · [[opcd]] ·
[[seed-self-evolving-opd]] · [[skill-sd]]
