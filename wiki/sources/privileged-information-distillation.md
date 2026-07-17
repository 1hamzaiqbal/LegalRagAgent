---
title: Privileged Information Distillation for Language Models
type: source
tags: [privileged-information, distillation, opd, agents]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2602.04942
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2602.04942.pdf
code: https://github.com/Emilianopp/Privileged-Information-Distillation
authors: Emiliano Penaloza et al.
year: 2026
venue: ICLR 2026 workshop paper
---

# Privileged Information Distillation

## TL;DR

The paper transfers training-time privileged information (PI) to an agent that
must act without it at inference. A PI-conditioned teacher and unconditioned
student share parameters. `pi-Distill` jointly trains them from teacher
trajectories; OPSD samples student trajectories and applies a reverse-KL
penalty toward the PI-conditioned teacher.

## What matters

- The motivating case is frontier-agent distillation when proprietary
  reasoning is hidden but actions/tool calls can be observed.
- Joint teacher/student training was the most robust configuration in the
  authors' analysis.
- PI must be useful and learnable. Low teacher–student KL helps transfer;
  high-KL or negative-utility PI can degrade it.
- The paper evaluates models up to 8B and notes that its factor analysis is
  observational rather than fully controlled.
- The cloned repository currently contains only a notice that code awaits
  legal approval; it is primary-source custody, not a runnable reproduction.

## Bearing on our work

This closes broad novelty for “give the teacher a skill/context that the
student lacks.” A compute-elasticity contribution must transfer the *conditional
frontier* and establish that effort modes survive withdrawal of PI. It also
supports an API-teacher lane: action-only PI can be useful when chain of
thought or dense logits are unavailable.

## Raw source

EIT PDF `papers/arxiv_2602.04942.pdf`; pinned placeholder repository
`repos/Privileged-Information-Distillation`.

## Links

[[compute-elasticity-distillation]] · [[opd-skill0-design]] ·
[[rethinking-privileged-opd]]
