---
title: Agent Lightning
type: source
tags: [agents, harness, reinforcement-learning, tracing]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2508.03680
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2508.03680.pdf
repo: /engrfs/project/jacobsn/hiqbal/literature/legalrag/repos/agent-lightning
year: 2025
---

# Agent Lightning

## TL;DR

Agent Lightning decouples arbitrary agent execution from RL training by
capturing prompts, tool calls, and rewards as structured spans and translating
agent trajectories into training transitions.

## Bearing

It is a strong reuse option for complex existing agents and a future training
bridge. The one-tool Phase 0 can be simpler; the project should own a
framework-neutral trace schema and add Agent Lightning only when its span store
or trainer integration removes real work.

## Raw source

EIT PDF `papers/arxiv_2508.03680.pdf`; pinned repo `repos/agent-lightning`.
