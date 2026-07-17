---
title: Three-Dial Retrieval Utility Track
type: hub
tags: [active-track, retrieval-utility, reader-conditioning, cost]
created: 2026-07-17
updated: 2026-07-17
status: active
---

# Three-dial retrieval utility

## Research question

Can we predict and control the marginal task utility of an evidence **set** for
a particular reader under a cost budget— including when the best action is to
stop, abstain, or reject evidence?

## Dials

1. **Evidence/set quality:** exposure, coverage, sufficiency, redundancy,
   contradiction, source authority.
2. **Reader conversion:** parametric competence and ability to use the set.
3. **Effort/cost:** retrieval actions, depth, tokens, latency, and the marginal
   value of another action.

## Contribution bar

The contribution is not “utility rather than relevance,” a static search-
budget grid, or generic conflict resolution. It must add a paired causal
reader-conditioned benchmark/estimator or a policy that improves a pre-
registered accuracy–cost frontier against the baselines mapped in
[[coverage-audit-2026-07-17]].

## Immediate build order

- Freeze the paired record schema and reconstruct the July master table.
- Add repeated outcomes to estimate policy/reader variance.
- Replicate the reader×task crossover on one non-legal dataset.
- Establish fixed-budget, RPP/QPP, intervention, and set-sufficiency baselines.
- Only then train a marginal-utility/stop policy.

## Existing evidence

[[judge-answer-conversion]] · [[offline-bandit-v0]] ·
[[alloc-internalization-rung2]] · [[helpfulness-benchmark]] ·
[[predicting-retrieval-utility]] · [[cue-r]] · [[sure-rag]]
