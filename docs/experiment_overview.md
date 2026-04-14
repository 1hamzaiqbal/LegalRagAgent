# Experiment Overview

High-level summary of the LegalRagAgent experimental program. Source of truth: `logs/experiments.jsonl` (180 entries).

For individual experiment details: `EXPERIMENTS.md`. For research state: `RESEARCH.md`.

## Timeline

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
- Gemma 4 E4B: 55.5% llm_only, 62.2% golden, 57.9% latest snap_hyde full rerun (earlier clean run: 58.6%)

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
- **Core result:** snap adds +3.0pp to HyDE, +5.0pp to plain RAG, and +5.0pp to parametric reasoning on BarExam N=200
- Cross-dataset follow-up: HousingQA `llm_only` 50.5%, `vectorless_direct` 50.0%, `vectorless_nosnap` 52.5%, `snap_hyde` 50.0%; CaseHOLD `llm_only` 69.5%, `vectorless_direct` 68.0%, `vectorless_nosnap` 67.5%
- Full N=1195 `subagent_rag` reached 56.9%, below `snap_hyde` 57.9%
- Infra: case-summary build job `44371` finished with 22K summaries; entity-graph rebuild job `44520` is 74% done

## Paper Core Result (Gemma 4 E4B, BarExam N=200)

| Family | No-snap mode | No-snap acc | Snap mode | Snap acc | Snap lift |
|---|---|---|---|---|---|
| HyDE retrieval | `rag_hyde` | 62.5% | `snap_hyde` | 65.5% | **+3.0pp** |
| Plain RAG* | `rag_simple` | 57.0% | `snap_rag` | 62.0% | **+5.0pp** |
| Parametric reasoning | `vectorless_nosnap` | 59.5% | `vectorless_direct` | 64.5% | **+5.0pp** |

*Plain-RAG uses the existing `gte-large` April 10 reference pair so the comparison stays aligned with the paper's main ablation setting.

## Key Results (Gemma 4 E4B, BarExam)

### N=200 (seed=42) — Validated Modes

| Rank | Mode | Acc | Changed | Net | Calls | Vector Store? |
|---|---|---|---|---|---|---|
| 1 | subagent_rag | **66.0%** | — | — | 4.1 avg | yes |
| 2 | snap_hyde | **65.5%** | 27% | +37* | 3 | yes |
| 3 | vectorless_hybrid | **65.0%** | 18% | +7 | 4 | yes (k=3) |
| 4 | gap_rag_nosnap | **64.5%** | — | — | 3.0 avg | yes |
| 4 | vectorless_direct | **64.5%** | 19% | +6 | 3 | **no** |
| 4 | vectorless_choice_map | **64.5%** | — | — | 3 | **no** |
| 7 | ce_threshold | 64.0% | 10% | +5 | 2-3 | yes |
| 8 | subagent_hybrid | **63.5%** | — | — | 4.1 avg | yes |
| 8 | vectorless_role | **63.5%** | 7% | +4 | 3 | **no** |
| 8 | gap_rag FIXED | **63.5%** | 2% | +4 | 3-6 | yes |
| 11 | rag_arbitration | 63.0% | 6% | +3 | 3 | yes |
| 12 | rag_hyde | **62.5%** | — | — | 2 | yes |
| 12 | gap_hyde_nosnap FIXED | **62.5%** | — | — | 4.1 avg | yes |
| 14 | snap_rag | 62.0% | 1% | +2 | 2 | yes |
| 14 | gap_hyde FIXED | 62.0% | 0.5% | +1 | 4-8 | yes |
| 16 | gap_vectorless | **61.5%** | — | — | 4.1 avg | **no** |
| 17 | vectorless_elements | **61.0%** | — | — | 3 | **no** |
| 17 | subagent_rag_evidence | **61.0%** | — | — | 4.1 avg | yes |
| 19 | rag_rewrite | 59.5% | — | — | 3 | yes |
| 19 | vectorless_nosnap | **59.5%** | — | — | 2 | **no** |
| 21 | rag_simple | 57.0% | — | — | 1 | yes |
| 22 | llm_only | 55.5% | — | — | 1 | no |

*snap_hyde fix/break from N=1195 run

Note: the `vectorless_*` label is historical shorthand. `vectorless_direct`, `vectorless_role`, `vectorless_elements`, `vectorless_choice_map`, and `gap_vectorless` are multi-turn LLM reasoning / parametric-knowledge modes, not corpus search. `vectorless_hybrid` is the only one that still pools generated knowledge with vector retrieval.

### Full-Scale N=1195

| Mode | Accuracy | Detail Log |
|---|---|---|
| golden_passage | 62.2% | `logs/eval_golden_passage_cluster-vllm_*_detail.jsonl` |
| **snap_hyde** | **57.9%** | `logs/eval_rag_snap_hyde_cluster-vllm_20260413_*_detail.jsonl` |
| **subagent_rag** | **56.9%** | `logs/eval_subagent_rag_cluster-vllm_20260414_*_detail.jsonl` |
| llm_only | 55.5% | `logs/eval_llm_only_cluster-vllm_*_detail.jsonl` |
| rag_simple | 54.2% | `logs/eval_rag_simple_cluster-vllm_*_detail.jsonl` |
| vectorless_direct | **CANCELLED** | job `43471` canceled — mode is parametric reasoning, not real corpus search |
| vectorless_hybrid | **CANCELLED** | job `43471` canceled — same naming / validity issue |

Note: `subagent_rag` looked best at N=200 (66.0%) but falls behind `snap_hyde` at N=1195 (56.9% vs 57.9%). The planned full-scale "vectorless" runs were canceled because they would only validate extra reasoning steps, not corpus search.

### Cross-Dataset Follow-Up (Gemma 4 E4B, N=200)

| Dataset | llm_only | vectorless_direct | vectorless_nosnap | snap_hyde | Key take-away |
|---|---|---|---|---|---|
| HousingQA | **50.5%** | 50.0% | 52.5% | 50.0% | Parametric reasoning does not solve the unknown-domain problem here |
| CaseHOLD | **69.5%** | 68.0% | 67.5% | — | Parametric reasoning hurts citation-matching relative to `llm_only` |

## Top 10 Findings

1. **Snap reasoning is the biggest contributor** (+5pp). Forcing the model to reason before retrieval improves everything downstream.

2. **HyDE passage generation adds +3.5pp retrieval quality.** Passage-form queries match the doctrinal corpus better than question-form queries (genre mismatch between questions and passages).

3. **Cross-encoder reranking dominates embedding choice.** All 7 non-gte-large embedders converge to exactly 65.0% with question-based reranking. The embedding model barely matters.

4. **Subagent reports are the strongest current Gemma 4 E4B strategy at N=200, but not at full scale.** `subagent_rag` reached 66.0% on N=200, then 56.9% on N=1195 versus `snap_hyde` at 57.9%.

5. **"Vectorless" is competitive, but the name is misleading.** These modes are multi-turn parametric reasoning baselines, not corpus search.

6. **Showing the snap answer in the final call causes anchoring.** Modes that show snap: 0.5-2% answer changes. Modes that hide snap: 7-27% changes.

7. **GAP_MIN_CE=1.0 was a critical bug** that made all gap experiments into llm_only (0% answer changes). Discovered via fix/break analysis.

8. **11-char HyDE outputs** were caused by prompt schema mismatch — Gemma merges system+user, and gap-formatted input didn't match the system prompt's expectation.

9. **N=200 variance is ~5-7pp.** snap_hyde ranged 62.5%-67.5% across duplicate N=200 runs. Full N=1195 is essential for reliable comparison.

10. **The BarExam snap / parametric lift does not transfer cleanly off-domain.** New Gemma follow-ups are flat on HousingQA (50.0-52.5%) and negative on CaseHOLD (67.5-68.0% vs 69.5% `llm_only`).

## Validity Issues Encountered

| Issue | When Found | Impact | How Detected | Fix |
|---|---|---|---|---|
| GAP_MIN_CE=1.0 | April 13 | All gap experiments were llm_only | Fix/break analysis showed 0% answer changes | Set to -100 |
| 11-char HyDE | April 10 | 85% of gap HyDE passages truncated | Log char count analysis | Prompt schema fix (pass snap_answer, use Student's Answer format) |
| Mid-job SCP | April 10 | gap_hyde_nosnap/flat used wrong prompt | Call distribution analysis (97% NONE) | Mark results as tainted |
| ChromaDB corruption | April 8 | Qwen rag_simple (36.5%), snap_hyde (35.1%) | Accuracy far below baseline | Separate chroma dirs, local /tmp builds |
| N=200 variance | April 10 | snap_hyde_aligned: 62.5% vs 67.5% on duplicate runs | Running same config twice | Use N=1195 for final decisions |
| Snap anchoring | April 13 | snap_rag only changes 1% of answers | Fix/break analysis | Hide snap from final call |

## Validity Checklist (for future runs)

Before trusting any result, check:
- [ ] **Answer change rate > 0%** — if 0%, the mode is a no-op (just snap accuracy)
- [ ] **Evidence retrieval rate > 50%** — if low, evidence is being filtered/lost
- [ ] **LLM call count matches expected** — wrong count = wrong code path
- [ ] **No pred=None** — answer extraction working
- [ ] **Snap accuracy consistent** (~61.5% for N=200 seed=42 Gemma) — if different, something changed
- [ ] **Net improvement > 0** — if fixes = breaks, the mode changes answers randomly

## Data Locations

| Data | Path |
|---|---|
| All results (source of truth) | `logs/experiments.jsonl` |
| Per-question detail logs | `logs/eval_{mode}_{provider}_{date}_detail.jsonl` |
| SLURM job logs | `/engrfs/tmp/jacobsn/hiqbal_legalrag/logs/{jobid}.out` (cluster) |
| Experiment narratives | `EXPERIMENTS.md` |
| Research state + handoff | `RESEARCH.md` |
| HPC throughput data | `docs/hpc_throughput.md` |
| This overview | `docs/experiment_overview.md` |

## Current Cluster Status (as of 2026-04-14)

| Job | Mode | N | Purpose |
|---|---|---|---|
| 44371 | case summaries build | — | Completed — 22K summaries built |
| 44394 | snap ablations | 200 | Completed — `rag_hyde` 62.5%, `vectorless_nosnap` 59.5% |
| 44395 | cross-dataset block | 200 | Completed — HousingQA and CaseHOLD follow-ups logged |
| 44520 | entity graph rebuild | — | Running — 74% done |
| 43471 | vectorless_direct + vectorless_hybrid | 1195 | Cancelled — misnamed parametric-reasoning validation, not corpus search |
