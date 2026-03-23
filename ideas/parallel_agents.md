# Parallel Agent Retrieval (ASMR-inspired)

## Concept

Instead of one retrieval call per step, spawn 3 parallel sub-agents that each search from a different angle:

```
query → spawn 3 parallel agents:
  Agent 1: "Find the governing rule/doctrine" (targets MBE/wex study material)
  Agent 2: "Find exceptions, defenses, or limitations" (targets edge cases)
  Agent 3: "Find case applications or analogous precedent" (targets caselaw)
→ each agent retrieves + synthesizes independently
→ aggregator combines the three perspectives into final sub-answer
```

## Why this could work

- **Specialization beats generalization** (from ASMR paper). Each agent can be optimized for its retrieval angle rather than one generic query.
- **Diversity of evidence** — currently we get 5 very similar passages. With 3 specialized agents, we'd get passages covering rule + exceptions + application.
- **Parallel execution** — LangGraph supports parallel node execution, so the 3 agents run concurrently (same wall-clock time as 1 agent).

## What we'd need to change

- Modify executor_node to spawn sub-agents (or run 3 retrieval calls with different query framings)
- Add an aggregation step that merges the 3 perspectives
- The query_rewriter could generate the 3 specialized queries instead of synonyms

## Lighter-weight version (try first)

Just change the query_rewriter prompt to generate queries targeting different ASPECTS rather than different TERMINOLOGY:
- Current: "negligence elements duty breach" / "tort liability prima facie case" / "fault liability elements"
- Proposed: "elements and definition of negligence" / "defenses and exceptions to negligence liability" / "negligence case application standard of care"

This is a prompt tweak, not an architecture change. Could test in 10 minutes.

## References

- ASMR blog post: Supermemory's ~99% SOTA memory system
- Key insight: parallel specialized agents > single general agent
- Their technique: 3 parallel reader agents + 3 parallel search agents + ensemble answering
