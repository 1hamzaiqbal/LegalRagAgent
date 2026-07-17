---
title: LatentSkill — Textual Skills to Generated LoRA Adapters
type: source
tags: [skills, lora, hypernetwork, context-compression, modular-weights]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2606.06087
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2606.06087.pdf
repo: /engrfs/project/jacobsn/hiqbal/literature/legalrag/repos/LatentSkill
authors: Fan et al.
year: 2026
---

# LatentSkill

## TL;DR

LatentSkill trains a hypernetwork to compile a textual skill into a cached,
plug-and-play LoRA adapter for a frozen Qwen3-8B backbone. It is a modular
alternative to permanently baking every skill into one student's weights.

It closes generic novelty around “textual skills become efficient weight
skills,” while leaving source-versus-target teachability, cross-scale action
value, continual revision, and robust skill placement open.

## Method and evidence

- Training uses 171,000 skill documents, roughly 300M skill tokens, plus
  teacher trajectories; the base model remains frozen.
- ALFWorld success is `74.3 vs 52.9` for LatentSkill versus in-context skills
  on seen tasks and `69.4 vs 56.0` on unseen tasks. Prefill falls from roughly
  1.21–1.23K to 0.44K tokens.
- Search-QA averages `35.6 vs 32.6`, with approximately `0.31K vs 1.10K`
  tokens per step.
- The aggregate hides reversals: LatentSkill loses on 2Wiki
  (`32.0 vs 39.8`) and Bamboogle (`25.6 vs 38.4`).
- Adapter strength is fragile. Unseen ALFWorld peaks near 70.9% at alpha 0.5
  and falls to 8.21% at alpha 1.2; naive adapter merging also interferes.

## Bearing on the proposed chain

The existence of a text-to-LoRA compiler makes “context or weights” at least a
three-way comparison:

1. keep the textual skill in context;
2. internalize it irreversibly through RL or distillation;
3. compile it into a selectable, removable adapter.

Any lifecycle study should include a modular-weight baseline before claiming
that permanent internalization is needed. The interesting question is whether
the artifact ranking induced by source contextual execution predicts target
context use, compiled-adapter use, or permanent post-withdrawal acquisition.

## Limits

The evidence covers two benchmarks, one base model/configuration, and no
cross-size teacher or on-policy task-reward training. Uncertainty is not
reported. Composition is shown on only 31 ALFWorld Look episodes, and the
strength/merging failures show that “compiled” does not imply robustly
composable.

## Code custody

- Official repository: https://github.com/yuaofan0-oss/LatentSkill.
- Persistent EIT checkout pinned at
  `b5a91141a4c435fdf14272eee1d6f89ab20d2e5e` on 2026-07-17.
- PDF SHA-256:
  `6c9d364b937a1ca1303da01e038c8b2cb7369d0bbc3d3768d74bbb597dc6e5df`.

## Links

[[skill-lifecycle-research-snapshot-2026-07-17]] · [[skillopt]] ·
[[skill-zero-five]] · [[skill0]] · [[continual-facts-in-weights]]
