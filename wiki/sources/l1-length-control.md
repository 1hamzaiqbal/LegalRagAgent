---
title: L1 — Length Controlled Policy Optimization
type: source
tags: [reasoning-efficiency, controllable-effort, rl]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2503.04697
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2503.04697.pdf
authors: Pranjal Aggarwal and Sean Welleck
year: 2025
venue: COLM 2025
code: https://github.com/cmu-l3/l1
---

# L1: Controlling How Long a Reasoning Model Thinks

## TL;DR

Length Controlled Policy Optimization trains one reasoning model to follow a
token target supplied in the prompt. Target lengths are sampled per example
during RL, giving the model inference-time control over a smooth
accuracy/compute tradeoff.

## Method details that matter

- LCPO-Exact appends a target-length instruction and rewards correctness minus
  deviation from the requested length.
- LCPO-Max treats the prompt value as a soft maximum and rewards correct
  answers that stay within it.
- During training, target lengths are sampled uniformly from 100 to 4000
  tokens; the length-penalty coefficient itself is fixed.
- The paper evaluates adherence and accuracy across multiple requested budgets
  and reports weaker generalization beyond the trained length range.
- An additive length objective can collapse to trivially short reasoning,
  motivating the paper's multiplicative maximum-budget reward.

## Novelty consequence

This is the closest public predecessor to the generic interpretation of
Inkling's controllable effort. Varying the cost coefficient rather than the
target length may be an implementation distinction, but “one RL-trained model
obeys different effort requests” is not a new claim after LCPO.

## Bearing on our work

LCPO is the reasoning-only controllability baseline. Our potential difference
must be about choosing **which** resource to spend—internal reasoning,
retrieval, context, or verification—and grounding that choice in downstream
reader utility, as specified in [[effort-conditioned-resource-allocation]].

## Raw source

EIT PDF: `papers/arxiv_2503.04697.pdf`; pinned code: `repos/l1`.

## Links

[[inkling-controllable-effort]] ·
[[training-language-models-to-reason-efficiently]] · [[three-dial]]
