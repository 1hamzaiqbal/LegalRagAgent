# Architecture v2 — Planning Table as Shared State

Brainstorming notes. NOT implemented. Thinking through before building.

## Core Idea

Strip down to fewer nodes. The planning table becomes the central coordination mechanism — it lives "outside" any single agent's scope and is the shared state that all nodes read/write.

```
START → planner_node → [agent_1, agent_2] (parallel) → synthesizer_node
              ↑                                               |
              └───────────── (if incomplete) ─────────────────┘
```

## Key Differences from Current Setup

**No hardcoded escalation pipeline.** Currently: rag → rewrite → direct_answer is baked into `_execute_step_with_escalation`. Instead, each executor just does its thing, returns what it found, and the synthesizer decides what's missing. The "escalation" happens naturally through the planning table — if the synthesizer says "we don't have X", planner adds a new step for X.

**No separate replanner node.** The synthesizer handles both synthesis AND completeness checking. If incomplete, it kicks back to the planner with notes on what's missing. Planner updates the planning table. Simpler loop.

**2 parallel executor agents.** Each gets assigned steps from the planning table. They can see each other's previous-round output (n-1 round visibility). This lets them build on what the other found rather than working in isolation.

**Planning table as coordination layer.** Fields per step:
- `sub_question`
- `status` (pending / completed)
- `met_status` — did this step produce useful evidence? (set by synthesizer)
- `result` / `evidence`
- `round` — which iteration this step was created in

## Flow

1. **Planner** decomposes question → planning table with initial steps
2. **Executors** (2 parallel) each take pending steps, do retrieval + synthesis
3. **Synthesizer** looks at ALL completed steps:
   - Is the evidence sufficient to answer the question?
   - If yes → produce final answer, done
   - If no → annotate what's missing, kick back to planner
4. **Planner** reads synthesizer feedback, adds new steps to planning table
5. Repeat from step 2 (max 3 rounds)

## Open Questions

- **Can parallel agents see each other's output?** If agent_1 finishes first and agent_2 can read its results, agent_2 might adjust its approach. But this creates ordering dependencies. Maybe: agents only see PREVIOUS round results, not current round.
- **Write conflicts on shared state.** Two agents writing to the planning table simultaneously could conflict. LangGraph handles this via state reducers (append-only lists), but need to verify.
- **How many agents?** 2 feels right for now. 3+ adds coordination overhead without clear benefit for typical 2-4 step plans.
- **Should executors have different specializations?** e.g., one focused on corpus retrieval, one on web search? Or both generalist? Start generalist, specialize later if needed.

## What to Strip from Current System

- `_execute_step_with_escalation` (hardcoded escalation chain)
- Separate judge node / judge LLM call per step (let synthesizer handle quality assessment)
- `parallel_replanner_node` (synthesizer does this)
- `max_retries` per step (no retries — if evidence is bad, synthesizer requests a NEW step with different framing)
- Complexity field (let planner just decide naturally)

## What to Keep

- Planning table as a data structure
- Query rewriter (aspect-based variant)
- Retrieval stack (BM25 + dense + cross-encoder)
- Web scraper (trafilatura)
- Evidence store (accumulated across rounds)
- Router (collection selection)

## Discussion Notes

- "Intuitive and unintuitive things work — just try experimentally"
- Replanner nodes should check each others' previous round output — maybe n-1 only
- Reduce heaviness first, then add parallel nodes
- Planning table `met_status` field as coordination signal
- Consider: writing to same planning table from parallel agents could add conflicts — need state reducer or lock
- Alternative: replace synthesizer with replanner nodes that coordinate directly
