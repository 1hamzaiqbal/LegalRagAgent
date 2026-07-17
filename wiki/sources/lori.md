---
title: LoRi — Low-Rank Distillation for Implicit Reasoning
type: source
tags: [implicit-cot, low-rank, hidden-state-distillation, reasoning]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2606.05315
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2606.05315.pdf
authors: Ryan Solgi, Jiayi Tian, Zheng Zhang
year: 2026
---

# LoRi

## TL;DR

LoRi aligns teacher and student hidden-state reasoning trajectories in a shared
low-rank tensor subspace using rationale-level and answer-anchor statistics.
The teacher representations are precomputed, making student training cheap.

## Evidence

- Evaluated across Qwen and Llama scales on GSM8K, GSM8K-Hard, and SVAMP.
- Llama-3.1-8B reaches 62.9% average implicit accuracy versus 64.0% explicit
  CoT in the reported table.
- Five latent reasoning steps are optimal in its ablation; more steps add no
  benefit.
- The paper reports roughly 5–7x lower latency than explicit CoT for the tested
  Llama models.
- The relation between low-rank geometry and reasoning ability remains
  empirical rather than theoretically established.

## Bearing on our work

This is the closest warning against claiming generic novelty for cross-model
latent reasoning alignment. A J-space method would need to show why
token-indexed workspace coordinates preserve cost-conditioned behaviors better
than LoRi-style low-rank trajectory statistics.

## Raw source

EIT PDF `papers/arxiv_2606.05315.pdf`. No primary code repository was located
during the 2026-07-17 pass.

## Links

[[compute-elasticity-distillation]] · [[jacobian-global-workspace]] ·
[[coconut]]
