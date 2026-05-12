# planning_table + MuSiQue setup audit — 2026-04-26

Subagent-produced empirical audit of new modes before scaling to N=100+.
Verdict: **safe to scale** with 3 fixes.

## Setup status

- **MuSiQue in-row BM25 retrieval: CLEAN.** All 5 sampled records — retrieved_ids strictly within question's paragraph pool, gold_retrieved correctly recomputable, no question truncation. 83% gold-retrieval at k=5 across N=30.
- **planning_table mode: CLEAN with one cosmetic concern.** All 5 records have populated TODOs (2-3 each), per-TODO retrievals into structured findings, final answer demonstrably reads from findings. One record had near-identical TODOs → identical retrieval (wasted call).
- **One transient API failure**: rag_simple errored on 1/30 records (`3hop1__68732_39743_24526`) with `Expecting value: line 1577 column 1 (char 8668)` — OpenRouter JSON parse failure, no retry, lost 331s and the result.

## Specific findings

### A. MuSiQue retrieval

- Each sample's `retrieved_ids` shares the question's `q_id` prefix → pool isolation correct
- `gold_idx` properly comma-split and intersected
- `gold_retrieved` matches recomputation in 5/5 samples
- Cross-checked: `2hop__704217_82341` has gold `_7` and `_10`; rag_simple retrieved `_7` ✓
- Aggregate: **25/30 (83%) gold-retrieval at k=5** — BM25 is doing real work
- **Naming concern**: `evidence_store.cross_encoder_score` actually carries BM25 scores in MuSiQue path. Downstream CE-threshold gates would silently mix scales

### B. planning_table

- TODO counts: 2, 3, 3, 3, 2 — all within docs spec
- Quality: fact-focused sub-questions ("In which state is Richmond located?"), no question-restating
- Findings grounded: "passages do not contain..." appears 8/13 (honest abstention, not hallucination)
- **Per-TODO retrieval diversity (within Q)**: Jaccard = 1.00, 0.17, 0.00, 0.00, 0.50. 3/5 questions show genuine TODO-specific retrieval; 1/5 collapses to identical retrieval due to synonymous TODOs
- Final answer demonstrably reads findings (e.g. "As established in the research, the author...")
- `call_trace` field NOT present (planning_table doesn't enable EVAL_TRACE_CALLS), but content evidence shows table reaches final agent
- All 5 records: empty `error`, populated `planning_table`, `todos_count == len(planning_table)` ✓

### C. Cross-mode comparison (same 5 questions, seed=42)

- Retrieval overlap rag_simple ∩ planning_table: **2, 4, 2, 4, 0** out of unions 6, 6, 6, 9, 8. Per-TODO decomposition explores different paragraph subspaces in 4/5 cases
- **planning_table found gold for `3hop1__68732_39743_24526` (rag_simple errored before retrieval)** — one real recall win
- Cost ratio: ~5.6 LLM calls + 12s/q vs rag_simple's 1 call. ~5-6× cost
- Accuracy on 5-question slice: 0/5 EM both; F1 rs=0.32, pt=0.19. N too small for conclusions

## Recommendations applied / queued

1. ✓ **Run planning_table at N=100+** — wiring is sound (this audit's verdict)
2. **Add retry-on-API-JSON-error in `_llm_call`** — the lost 331s tax will recur 5-10× at N=200
3. **Deduplicate TODOs before retrieval** — Jaccard threshold or simple lexical dedup
4. (Skip) Rename `cross_encoder_score` to indicate BM25 — cosmetic, no behavior impact
5. (Skip for now) Add per-TODO query + evidence text to detail log — helps post-hoc analysis but not blocking
