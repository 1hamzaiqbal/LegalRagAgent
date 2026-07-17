---
title: Training Language Models to Reason Efficiently
type: source
tags: [reasoning-efficiency, length-penalty, rl]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2502.04463
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2502.04463.pdf
authors: Daman Arora and Andrea Zanette
year: 2025
venue: NeurIPS 2025
code: https://github.com/Zanette-Labs/efficient-reasoning
---

# Training Language Models to Reason Efficiently

## TL;DR

Arora and Zanette add a length-sensitive reward to online RL so correct,
shorter responses receive higher reward. They train separate operating points
with fixed penalty coefficient `alpha`, producing a family of checkpoints on
the accuracy/token frontier.

## Key claims and method

- The reward is correctness multiplied by a bounded, normalized length term;
  increasing `alpha` favors shorter correct responses.
- Lengths are normalized within prompt rollouts, reducing the tendency to
  punish intrinsically harder prompts merely because their solutions are
  longer.
- The experiments train four fixed `alpha` settings rather than one policy
  conditioned on an inference-time effort request.
- The authors explicitly note that `alpha` controls average generation cost but
  does not target an exact requested length.

## Bearing on the Inkling comparison

The addendum's distinction is substantially correct: this paper chooses one
penalty coefficient for a training run, while Inkling reports varying effort
signals across samples. However, that extension is not unoccupied:
[[l1-length-control]] samples prompt-specified target lengths during RL and
trains one controllable model.

## Bearing on our work

Fixed-penalty separate policies are a required baseline for
[[effort-conditioned-resource-allocation]]. A new result must show more than a
different point on the accuracy/cost curve: it must demonstrate useful
inference-time control, cross-resource allocation, or reader-conditioned
evidence utility.

## Raw source

EIT PDF: `papers/arxiv_2502.04463.pdf`; pinned code: `repos/efficient-reasoning`.

## Links

[[inkling-controllable-effort]] · [[l1-length-control]] · [[three-dial]]
