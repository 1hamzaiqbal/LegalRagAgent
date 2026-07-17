---
title: Verbalizable Representations Form a Global Workspace in Language Models
type: source
tags: [interpretability, jacobian-lens, j-space, steering, latent-reasoning]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://transformer-circuits.pub/2026/workspace/index.html
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/web/anthropic-global-workspace.html
code: https://github.com/anthropics/jacobian-lens
authors: Wes Gurnee et al.
year: 2026
---

# Verbalizable Representations Form a Global Workspace

## TL;DR

Anthropic introduces the Jacobian lens (J-lens), which transports an
intermediate residual-stream activation into the final-layer basis using an
average activation-to-future-output Jacobian. Sparse nonnegative combinations
of token-indexed J-lens vectors define the J-space, a small representational
component that the paper associates causally with report, modulation,
reasoning, and flexible reuse.

## Method

For layer `l`, the paper estimates

`J_l = E[d h_final,t' / d h_l,t]`

over prompts, source positions, and current/future target positions, then
reads an activation with `softmax(W_U norm(J_l h_l))`. The reference code fits
the lens on open-weight decoder-only models and uses backward passes; the paper
uses 1,000 sequences of 128 tokens, while the README says roughly 100 prompts
is usable.

The J-space is not a fixed low-dimensional linear subspace. It is defined by
sparse nonnegative combinations of an overcomplete set of J-lens vectors. The
authors typically use at most 25 active vectors and report that the J-space
component explains under 10% of activation variance.

## Causal/training evidence

- Steering or swapping J-lens coordinates changes reported concepts and can
  redirect intermediate reasoning.
- Suppressing the J-space harms complex reasoning more than routine fluency.
- Counterfactual-reflection training supervises what the model should say if
  interrupted and asked to reflect. It changes behavior in the uninterrupted
  context, and J-space ablation reverses much of the improvement.

## Limitations and access requirements

- The lens only directly indexes concepts representable as vocabulary tokens;
  multi-token concepts are a limitation.
- The paper does not claim a complete neural global-workspace architecture.
- J-lens requires model weights, residual activations, and gradients/Jacobians.
  API output logits are insufficient.
- The public repository is a reference implementation, explicitly not
  maintained, and ships no model weights or fitting corpus.

## Bearing on our work

J-space is best used to diagnose whether cost conditions recruit concepts such
as uncertainty, verification, backtracking, or stopping, and to test causal
interventions when those modes collapse after distillation. It is not already
a teacher-to-student transfer method. Counterfactual reflection is a promising
white-box training arm, but [[lori]] makes a generic latent-alignment novelty
claim unsafe.

## Raw source

EIT snapshot `web/anthropic-global-workspace.html`; pinned code
`repos/jacobian-lens`.

## Links

[[compute-elasticity-distillation]] · [[lori]] · [[coconut]]
