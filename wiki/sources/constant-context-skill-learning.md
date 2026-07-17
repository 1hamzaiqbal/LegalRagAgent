---
title: From History to State — Constant-Context Skill Learning
type: source
tags: [skills, context-to-weights, lora, agents, constant-context]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2605.05413
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2605.05413.pdf
authors: Xie et al.
year: 2026
---

# From History to State

## TL;DR

This paper moves recurring agent procedures and growing histories into
task-family LoRA modules, leaving only a deterministic compact state block at
runtime. It is a direct and serious context-to-modular-weights precedent.

It does not train from or rank the same fixed textual skill artifacts across
readers and placements, and its context-versus-weight comparison is not
cost- or data-matched. It is therefore a mandatory modular-weight baseline,
not a complete collision with artifact-level utility transport.

## Method

Successful trajectories are grouped by task family. A hand-specified tracker
turns interaction history into a compact state block. Each family receives a
separate LoRA adapter trained by step-level SFT on successful trajectories and
then online GRPO-style RL with four rollouts per instance. The backbone stays
frozen. GPT-5.5 drafts family-specific reward rules offline; humans validate
them before deterministic runtime use.

For Qwen3-8B, the rank-64 LoRA has about 175M trainable parameters, or 2.09%
of the base model, and each task-family adapter occupies 0.5–0.7 GB.

## Evidence

- Qwen3-8B SFT+RL reaches 83.6%/89.6% ALFWorld seen/unseen success, 84.0
  WebShop score with 76.8% success, and 72.8/62.9% seen plus 79.7/66.4%
  unseen SciWorld score/success.
- On WebShop, a compact state in an untrained prompt gives 23.6% success, SFT
  gives 62.2%, and SFT+RL gives 76.8%.
- Prompt tokens per turn are reduced relative to a controlled ReAct-1step
  baseline: 183.8 versus 380.0 on ALFWorld seen, 488.0 versus 1,059.0 on
  WebShop, and roughly 492–496 versus 1,443–1,481 on SciWorld.
- Reported single-A100-80GB wall times are approximately 173/143/90 minutes
  for SFT and 650/86/130 minutes for RL across the three environments.

## Novelty boundary

The work closes broad claims that recurring procedural context has not been
moved into lightweight modular weights or that bounded runtime state cannot
support trained agents. It does not transport fixed textual candidates,
compare their reader-specific contextual rankings, or measure their
post-withdrawal acquisition rankings.

Its ReAct comparison is not a matched placement experiment: ReAct never sees
the same source artifact; the adapter gets expert SFT and online RL; output
formats differ; and reported token cost excludes training/amortization.
Qwen/Llama rows use independently trained backbone-specific adapters rather
than one artifact transported across readers.

## Limits

The tracker schema and reward rules are task-family-specific, only text-mode
environments are tested, and a new procedure requires a new adapter. Published
method comparisons are uncontrolled. The paper reports one training seed
(`42`); its three-seed uncertainty is over inference, not independent
training runs.

## Code custody

- No official repository or project page was found from the paper or targeted
  searches as of 2026-07-17.
- PDF SHA-256:
  `13a6d56999ad5ead28b6a54341488520cecd7b4c444655e41ccd91d2d98812ae`.

## Links

[[skill-lifecycle-research-snapshot-2026-07-17]] · [[latent-skill]] ·
[[skill0]] · [[skillc]] · [[skillrae]]
