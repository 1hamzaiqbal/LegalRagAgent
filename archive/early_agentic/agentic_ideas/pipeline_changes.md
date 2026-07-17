# Pipeline Changes — Discussion Notes

Notes from codebase analysis (Feb 28 2026). Organized into quick wins vs deeper structural questions for team discussion.

---

## Part 1: Quick Simplifications (code cleanup, no accuracy risk)

### 1. Remove no-op router (~5 min)
`route_after_planner` always returns `"executor_node"` — replace the conditional edge with a fixed edge. Pure dead code.

### 2. Merge verify_answer + observability → single node (~20 min)
`verify_answer_node` does no verification (auto-pass since R1) — it just conditionally appends MC selection then always routes to observability. Every query (100/100 eval, 8/8 trace) takes the same path. Move MC selection into observability, delete the node. **8 → 7 nodes.**

### 3. Merge classifier + planner → single node (~30 min)
Classifier outputs 1 bit (simple/multi_hop) then planner re-reads the same objective. Combine into one "classify and plan" prompt. The classifier returns `multi_hop` 94%+ of the time anyway. **Saves 1 LLM call/query. 7 → 6 nodes.** Fallback already defaults to multi_hop + single step.

- Theoretical: prompt chaining vs prompt composition trade-off (Khattab et al., DSPy 2023) — when two sequential prompts share the same input and the first output is trivially small, composing into one call reduces latency without accuracy loss.

### Summary (Part 1)

| Change | LLM Calls Saved | Nodes | Risk |
|---|---|---|---|
| 1. No-op router | 0 | — | zero |
| 2. Merge verify+obs | 0 | -1 | zero |
| 3. Merge classify+plan | +1/query | -1 | low |

These are safe to do regardless of what we decide about the evaluator metric.

---

## Part 2: Changes Tied to Evaluator Metric Decision

These depend on whether we keep cosine similarity or switch to something else.

### 4. Retrieval quality gate (~15 min if we keep cosine sim)
Skip the synthesis LLM call when retrieval confidence < 0.45. Those steps always get marked `failed` and their output is discarded by `_aggregate_completed_answers()` anyway. **Saves ~0.6 LLM calls/query.**

- Empirical: ~20% of steps fail, each wastes a synthesis call whose output is never used.

### 5. Move stagnation detection into replanner skill (~20 min)
The router hardcodes stagnation check (3+ failures < 0.35, range < 0.1) that duplicates the same rule in `adaptive_replan.md`. Remove from router, let replanner handle it. Costs 1 extra LLM call on ~5% of queries, removes 15 lines of routing logic.

- Theoretical: separation of concerns — routing layer handles graph topology, replanner handles research quality decisions (Plan-and-Solve framework, Wang et al. 2023).

---

## Part 3: Structural Problems Identified from Eval Logs

### The core finding: RAG pipeline hurts accuracy vs baseline

| Setup | Accuracy | LLM Calls/Query | Time/Query |
|---|---|---|---|
| Gemma 27B + RAG | 45% | 13.1 avg | ~60s |
| Gemma 27B baseline (no RAG) | 57% | 1 | ~7s |
| DeepSeek-reasoner + RAG | 56% | 9.6 avg | ~413s |
| DeepSeek-reasoner baseline | 92% | 1 | ~85s |

### Problem A: Cosine similarity is fundamentally miscalibrated

Confidence score distribution (both models, 600+ steps):

| Bucket | Gemma (366 steps) | DeepSeek (257 steps) |
|---|---|---|
| < 0.40 | 7% | 3% |
| 0.40–0.49 | 28% | 27% |
| 0.50–0.59 | 40% | 48% |
| 0.60–0.69 | 22% | 19% |
| >= 0.70 | 4% | 3% |
| **Mean** | **0.535** | **0.543** |
| **Fail rate** | **77%** | **71%** |

The threshold was 0.50 in practice. Even at 0.50, 71-77% of steps fail. The distribution is tight (stdev 0.075-0.092) and centered well below any meaningful threshold.

**Why**: Cosine similarity between a legal question embedding and a passage embedding is inherently low — they use different language (questions vs. declarative statements about law). The metric has ~zero correlation with whether the retrieved passages are actually useful.

**Impact**: The evaluator is essentially a coin flip. It triggers replanning loops that burn LLM calls for no reason — **2.8 wasted steps per query on Gemma, 1.8 on DeepSeek**.

### Problem B: Cross-encoder scores exist but aren't used for confidence

The cross-encoder (`ms-marco-MiniLM-L-6-v2`) already computes relevance scores during reranking via `rerank_with_cross_encoder()`. These scores use full cross-attention and are calibrated for query-document relevance. But the scores are used only for ranking order, then thrown away. Confidence is recomputed from scratch using bi-encoder cosine similarity — a strictly worse signal.

**Fix option**: Return cross-encoder scores from `rerank_with_cross_encoder()` and use `mean(top_3_scores)` or `max(score)` as confidence. This is free — no extra computation.

### Problem C: Multi-query rewrite adds noise at 20K corpus scale

Query rewrite (primary + 2 alternatives) is designed to bridge terminological gaps in a large corpus. At 20K passages, the alternatives often retrieve the same or worse passages. The cross-encoder reranks against the primary only, so alternatives mainly widen the candidate pool — which at this corpus size adds more noise than signal.

### Problem D: 3-step cap forces retrieval of noise

Over 3 steps with cross-step dedup, the pipeline tries to find 15 unique relevant passages from 20K. Most queries have 2-3 genuinely relevant passages. By step 3, dedup forces retrieval of progressively worse passages. The 3rd step usually injects noise.

### Problem E: MC choice stripping hides useful retrieval signal

The pipeline strips answer choices before research to keep it "unbiased." But choices often contain the key legal concepts the retriever needs (e.g., choice A says "under res ipsa loquitur" — that's what the retriever should search for). The retriever has to infer the doctrine from the fact pattern alone, which is harder.

**Possible compromise**: Extract legal terms from MC choices as additional query keywords for retrieval, but keep them out of the synthesis prompt.

### Problem F: Classifier is a no-op in practice

100% of MC questions get classified as `multi_hop` (the skill says "when in doubt, multi_hop"). Every query pays the full multi_hop cost (11-17 LLM calls) even for single-concept questions that need 1 step.

---

## Part 4: Bigger Design Questions

### Option A: Replace cosine sim with cross-encoder confidence
Use cross-encoder scores (already computed during reranking) as the confidence signal. Better calibrated, no extra computation. Changes what the evaluator measures but keeps the loop structure.

### Option B: Make evaluator threshold-free
Always pass the step. Let the replanner judge "do I have enough evidence?" semantically from the actual answer content, not a number. The replanner already reads `accumulated_context`.

### Option C: Reduce default to 1 step, "try harder" fallback
Default to 1 research step. Only loop if replanner explicitly says step 1's answer is insufficient AND gives a meaningfully different question. Would cut LLM calls from ~13 to ~5 for most queries.

### Option D: Retrieval-only mode (no synthesis on bad retrieval)
If retrieval confidence is very low, skip synthesis entirely and fall back to direct LLM answer (baseline behavior). The pipeline gracefully degrades to baseline accuracy (57-92%) instead of injecting bad passages.

### Option E: Web search fallback (from ideadump.md notes)
When local retrieval fails, route to a web search subagent (Tavily/Exa) that searches authoritative legal sources (law.cornell.edu, courtlistener.com, justia.com). Addresses the "corpus doesn't cover it" problem directly. See `agentic_ideas/ideadump.md` for the Supervisor-Worker pattern notes.

### Option F: Feed MC choices into retrieval selectively
Extract legal terms from MC choices, use as additional retrieval queries. Keep synthesis unbiased (no choices in synthesis prompt). Best of both worlds — retriever gets signal, synthesis stays clean.

---

## Recommended Priority

1. **Do Part 1 now** — pure cleanup, no accuracy risk, no metric dependency
2. **Decide on evaluator metric** (cosine sim vs cross-encoder vs threshold-free) — this unblocks Part 2 and shapes the loop redesign
3. **Test single-pass mode** (Option C) — run 100-query eval with 1-step-only to see if it matches multi-step accuracy. If it does, the whole loop can be simplified dramatically
4. **Consider Option D** (fallback to baseline) — if RAG retrieval is bad, don't force it
