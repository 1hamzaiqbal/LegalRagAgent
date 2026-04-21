### Phase 0: Foundation (March 22-24)
- Built eval harness, established baselines (DeepSeek, Scout, Llama 70B)
- Discovered: LLM-only (85% DeepSeek) beats RAG (70%) on well-known domains
- Established snap_hyde as best retrieval approach (82% Llama N=100)
- **Key insight:** prompt debiasing (+11pp) matters more than retrieval

### Phase 1: Method Exploration (March 25-27)
- Tested 15+ retrieval variants: arbitration, devil's advocate, decomposition, confidence gating, CE thresholding
- CE threshold (80.0% Llama N=200) became new BarExam best
- Confirmed: self-correction hurts (-3 to -10pp), counterevidence hurts (-6pp)
- Cross-dataset: RAG helps HousingQA (+9pp) but not CaseHOLD (-1.5pp)

### Phase 2: Small Model Audit (April 1-8)
- Full N=1195 BarExam baselines: Qwen3-32B (61.4%), Gemma-27B (58.0%), Qwen3-8B (52.1%)
- HPC cluster setup: vLLM serving Gemma 4 E4B and Qwen3-8B
- Gemma 4 E4B: 55.5% llm_only, 62.2% golden, 58.6% best full snap_hyde run (later rerun: 57.9%)

### Phase 3: Embedding Comparison (April 9-11)
- 7 embedders × 3 modes = 21 eval runs
- **Key finding:** Cross-encoder reranking dominates — all 6 non-gte-large embedders converge to exactly 65.0% with aligned reranking
- Embedding model choice barely matters when cross-encoder reranks against the question

### Phase 4: Gap Architecture (April 10-13)
- Designed snap → gap analysis → per-gap retrieval → final answer
- **BUG FOUND:** GAP_MIN_CE=1.0 filtered 90-95% of evidence. All gap results were llm_only (0% answer changes)
- **BUG FOUND:** 11-char HyDE outputs from prompt schema mismatch (Gemma merges system+user)
- After fixes and anchoring controls: gap_rag 63.5%, gap_rag_nosnap 64.5%, gap_hyde 62.0%, gap_hyde_nosnap 62.5%, gap_vectorless 61.5%
- **Key finding:** Showing snap answer in final call causes anchoring (0.5-2% changes vs 19-27% when hidden)

### Phase 5: Historical vectorless / parametric reasoning (April 12-13)
- LLM generates knowledge from parametric memory instead of searching the corpus
- vectorless_hybrid: 65.0% (N=200), vectorless_direct: 64.5% (N=200)
- Competitive with snap_hyde (65.5%) with ZERO vector infrastructure
- Naming caveat: "vectorless" is historical shorthand for multi-turn LLM reasoning / parametric knowledge exploitation, not real corpus search
- Full N=1195 "vectorless" validation was canceled after that naming issue was identified

### Phase 6: Anchoring Hypothesis (April 13, completed)
- gap_rag_nosnap reached 64.5% vs gap_rag 63.5%
- gap_hyde_nosnap (fixed) reached 62.5% vs fixed gap_hyde 62.0%
- gap_vectorless reached 61.5% and did not beat the plain historical vectorless baselines

### Phase 7: Paper-Core Controls and Scale Check (April 14)
- Snap/no-snap ablations completed: `rag_hyde` 62.5%, `vectorless_nosnap` 59.5%
- **Core result (later revised for HyDE):** snap adds +5.0pp to plain RAG and +5.0pp to parametric reasoning; the original `+3.0pp` HyDE row was later invalidated by the repaired prompt fix
- Cross-dataset follow-up: HousingQA `llm_only` 50.5%, `vectorless_direct` 50.0%, `vectorless_nosnap` 52.5%, `snap_hyde` 50.0%; CaseHOLD `llm_only` 69.5%, `vectorless_direct` 68.0%, `vectorless_nosnap` 67.5%
- Full N=1195 `subagent_rag` reached 56.9%, below `snap_hyde` 57.9%
- Infra: case-summary build job `44371` finished with 22K summaries; entity-graph rebuild job `44520` is 74% done

### Phase 8: Structured Search Follow-Up (April 15)
- Full N=1195 `entity_search` reached **53.2%** (`636/1195`) using real NLP entity-graph corpus search, zero embeddings, and 1 LLM call
- Scale robustness warning: `entity_search` drops from **60.0%** at N=200 to **53.2%** at N=1195 (`-6.8pp`), while vector `rag_simple` drops only from **57.0%** to **54.2%** (`-2.8pp`)
- New N=200 follow-ups did not move the frontier: `snap_entity_informed` = **59.5%**, `subagent_hyde` = **62.5%**
- The initial full N=1195 `rag_hyde` attempt was **broken** (100% 11-char HyDE outputs from the terse generic prompt); the later repaired rerun became the real source of truth

### Phase 9: Full Follow-Up Reruns (April 15-16)
- The first full `rag_hyde` rerun was later superseded by the repaired April 17 run
- Full `ce_threshold` completed at **55.9%** (`668/1195`)
- Full `gap_rag_nosnap` completed at **55.9%** (`668/1195`)
- Full `subagent_rag` 1-gap rerun improved to **57.2%** (`684/1195`)

## Historical Cluster Status (as of 2026-04-19)

| Job | Mode | N | Purpose |
|---|---|---|---|
| 44371 | case summaries build | — | Completed — 22K summaries built |
| 44394 | snap ablations | 200 | Completed — `rag_hyde` 62.5%, `vectorless_nosnap` 59.5% |
| 44395 | cross-dataset block | 200 | Completed — HousingQA and CaseHOLD follow-ups logged |
| 44520 | entity graph rebuild | — | Status unverified — last noted at 74% on 2026-04-14 |
| 45350 | `rag_hyde` + `ce_threshold` full | 1195 | Completed — `ce_threshold` 55.9%; the interim `rag_hyde` rerun was later superseded by `48555` |
| 45735 | `gap_rag_nosnap` + `subagent_rag` (1-gap) full | 1195 | Completed — 55.9%, 57.2% |
| 48393 | combo modes | 200 | Completed — `snap_hyde_report` 66.0%, `snap_hyde_report_snap` 64.0%, `subagent_rag_snap` 63.0%, `subagent_rag_full` 62.0% |
| 48555 | fixed `rag_hyde` full rerun | 1195 | Completed — `rag_hyde` 57.9%, matching the paired `snap_hyde` rerun |
| 43471 | vectorless_direct + vectorless_hybrid | 1195 | Cancelled — misnamed parametric-reasoning validation, not corpus search |
