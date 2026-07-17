---
title: TurnOPD — Turn-Aware On-Policy Distillation
type: source
tags: [opd, agents, long-horizon, training-efficiency]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2607.05804
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2607.05804.pdf
authors: Yuhang Zhou et al.
year: 2026
---

# TurnOPD

## TL;DR

TurnOPD targets two long-agent OPD inefficiencies: full-horizon rollouts spend
compute on low-yield tail turns, and token-normalized KL concentrates training
loss on shallow turns. It adapts rollout depth and progressively shifts loss
toward turn-balanced supervision.

## Evidence

- Evaluated on ALFWorld, WebShop, and Multi-Hop Search with Qwen students and
  task-specialized teachers.
- Under equal wall-clock protocols, it advances the reported accuracy–time
  frontier over vanilla OPD.
- ALFWorld Qwen3-1.7B improves from 73.52 to 85.60 Avg@4 while reducing 100-step
  wall time from 4.42h to 1.93h.
- The Qwen3-4B ALFWorld student slightly exceeds the paper's teacher reference
  on overall Avg@4; this is a task/protocol-specific result, not broad teacher
  domination.

## Bearing on our work

TurnOPD occupies budget-adaptive *training* for long-horizon OPD. Our proposed
deployment frontier must distinguish inference-time effort conditioning from
training-rollout efficiency. Turn-normalized supervision is a required agentic
baseline if the project advances beyond single-turn reasoning.

## Raw source

EIT PDF `papers/arxiv_2607.05804.pdf`.

## Links

[[compute-elasticity-distillation]] · [[rethinking-opd]] · [[opd-distillation]]
