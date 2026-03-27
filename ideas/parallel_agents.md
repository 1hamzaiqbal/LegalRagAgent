# Aspect-Specialized Retrieval (ARCHIVED)

> **ARCHIVED**: The lightweight version (prompt-only aspect query rewrite) is in `RESEARCH.md` Tier 1 #2. The full parallel-agent design is deferred until simpler approaches are exhausted.

## Concept

Instead of one generic retrieval call per step, run 3 specialized retrieval workers that each search from a different angle:

```
query → spawn 3 parallel agents:
  Agent 1: "Find the governing rule/doctrine" (targets MBE/wex study material)
  Agent 2: "Find exceptions, defenses, or limitations" (targets edge cases)
  Agent 3: "Find case applications or analogous precedent" (targets caselaw)
→ each worker retrieves + synthesizes independently
→ aggregator combines the three perspectives into one stronger sub-answer
```

 ## Why this could work

- **Specialization beats generalization.** Each worker can be optimized for its retrieval angle rather than one generic query.
- **Diversity of evidence.** Current traces often show 5 passages that are topically related but redundant. Aspect-specialized retrieval should increase coverage of rule + exceptions + application.
- **Parallel execution is plausible.** LangGraph can support concurrent branches, so wall-clock time could stay close to a single retrieval pass if backend throughput allows.

## Why this idea fits the current traces

- In `logs/playtest/02_barqa_multihop.txt`, the system repeatedly finds passages about the existence of the implied warranty doctrine, but misses the specific "what must be proven" aspect.
- In `logs/playtest/08_housing_dense_only.txt`, the system often retrieves the right general topic but the wrong jurisdiction, which suggests that retrieval needs stronger structural guidance than synonym-only rewrites.

This points to a broader pattern: the current query rewriter is good at terminology variation, but weaker at aspect variation.

## What we'd need to change

- Modify `executor_node` to spawn sub-agents, or more simply run 3 retrieval calls with different query framings inside the existing node.
- Add an aggregation step that merges the 3 perspectives into one sub-answer.
- Update `skills/query_rewriter.md` so it can generate aspect-specialized queries instead of mostly synonym-style alternatives.

## Lighter-weight version (try first)

Just change the query rewriter so it generates queries targeting different ASPECTS rather than different TERMINOLOGY:
- Current: "negligence elements duty breach" / "tort liability prima facie case" / "fault liability elements"
- Proposed: "elements and definition of negligence" / "defenses and exceptions to negligence liability" / "negligence case application standard of care"

This is a prompt tweak, not an architecture change. It is the fastest way to test whether aspect diversity helps before adding any graph complexity.

## Highest-value adjacent idea

For the housing corpus, pair aspect specialization with jurisdiction control:

- one query specialized for the rule or elements
- one query specialized for procedure or notice requirements
- one query specialized for citations or code sections
- all three filtered or strongly biased by `state` when the question names a jurisdiction

## References

- ASMR blog post: Supermemory's memory-system writeup
- Key transferable insight: specialized parallel retrieval can outperform one general retrieval pass
