---
title: CRISP — Compressed Reasoning via Iterative Self-Policy Distillation
type: source
tags: [self-distillation, concise-reasoning, opd, efficiency]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2603.05433
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2603.05433.pdf
code: https://github.com/HJSang/OPSD_Reasoning_Compression
authors: Hejian Sang et al.
year: 2026
---

# CRISP

## TL;DR

CRISP prompts a frozen copy of the current model to reason concisely, then uses
on-policy self-distillation to transfer that behavior into the unprompted
policy. The teacher is periodically refreshed from the improved student.

## Bearing on our work

This occupies the simple “prompt a desired effort style, then teach the model
to do it without the prompt” mechanism for concise reasoning. It reinforces
that SKILL0-style prompt withdrawal plus OPD is not itself novel. CRISP is not
user-controllable across multiple tool prices, but it is a required
self-distillation baseline.

## Raw source

EIT PDF `papers/arxiv_2603.05433.pdf`; pinned repository
`repos/OPSD_Reasoning_Compression`.

## Links

[[compute-elasticity-distillation]] · [[bard-budget-aware-reasoning-distillation]] ·
[[rethinking-privileged-opd]]
