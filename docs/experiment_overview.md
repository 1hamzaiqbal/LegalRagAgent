# Experiment Overview

High-level summary of the LegalRagAgent experimental program. Source of truth: `logs/experiments.jsonl` (164+ entries).

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
- Gemma 4 E4B: 55.5% llm_only, 62.2% golden, 58.6% snap_hyde (N=1195)

### Phase 3: Embedding Comparison (April 9-11)
- 7 embedders × 3 modes = 21 eval runs
- **Key finding:** Cross-encoder reranking dominates — all 7 non-gte-large embedders converge to exactly 65.0% with aligned reranking
- Embedding model choice barely matters when cross-encoder reranks against the question

### Phase 4: Gap Architecture (April 10-13)
- Designed snap → gap analysis → per-gap retrieval → final answer
- **BUG FOUND:** GAP_MIN_CE=1.0 filtered 90-95% of evidence. All gap results were llm_only (0% answer changes)
- **BUG FOUND:** 11-char HyDE outputs from prompt schema mismatch (Gemma merges system+user)
- After fixes: gap_rag 63.5%, gap_hyde 62.0% — evidence reaches model but anchoring limits changes to 0.5-2%
- **Key finding:** Showing snap answer in final call causes anchoring (0.5-2% changes vs 19-27% when hidden)

### Phase 5: Vectorless RAG (April 12-13)
- LLM generates knowledge from parametric memory instead of searching vector store
- vectorless_hybrid: 65.0% (N=200), vectorless_direct: 64.5% (N=200)
- Competitive with snap_hyde (65.5%) with ZERO vector infrastructure
- Full N=1195 validation running

### Phase 6: Anchoring Hypothesis (April 13, running)
- Testing gap_rag_nosnap, gap_vectorless, gap_hyde_nosnap — same retrieval but snap hidden from final call
- If these beat gap_rag (63.5%), anchoring is confirmed as the bottleneck

## Key Results (Gemma 4 E4B, BarExam)

### N=200 (seed=42) — Validated Modes

| Rank | Mode | Acc | Changed | Net | Calls | Vector Store? |
|---|---|---|---|---|---|---|
| 1 | snap_hyde | **65.5%** | 27% | +37* | 3 | yes |
| 2 | vectorless_hybrid | **65.0%** | 18% | +7 | 4 | yes (k=3) |
| 3 | vectorless_direct | **64.5%** | 19% | +6 | 3 | **no** |
| 3 | vectorless_choice_map | **64.5%** | — | — | 3 | **no** |
| 5 | ce_threshold | 64.0% | 10% | +5 | 2-3 | yes |
| 6 | vectorless_role | 63.5% | 7% | +4 | 3 | **no** |
| 6 | gap_rag FIXED | 63.5% | 2% | +4 | 3-6 | yes |
| 8 | rag_arbitration | 63.0% | 6% | +3 | 3 | yes |
| 9 | snap_rag | 62.0% | 1% | +2 | 2 | yes |
| 10 | gap_hyde FIXED | 62.0% | 0.5% | +1 | 4-8 | yes |
| 11 | vectorless_elements | 61.0% | — | — | 3 | **no** |
| 12 | rag_rewrite | 59.5% | — | — | 3 | yes |
| 13 | rag_simple | 57.0% | — | — | 1 | yes |
| 14 | llm_only | 55.5% | — | — | 1 | no |

*snap_hyde fix/break from N=1195 run

### Full-Scale N=1195

| Mode | Accuracy | Detail Log |
|---|---|---|
| golden_passage | 62.2% | `logs/eval_golden_passage_cluster-vllm_*_detail.jsonl` |
| **snap_hyde** | **57.9%** | `logs/eval_rag_snap_hyde_cluster-vllm_20260413_*_detail.jsonl` |
| llm_only | 55.5% | `logs/eval_llm_only_cluster-vllm_*_detail.jsonl` |
| rag_simple | 54.2% | `logs/eval_rag_simple_cluster-vllm_*_detail.jsonl` |
| vectorless_direct | **running** | — |
| vectorless_hybrid | **running** | — |

Note: N=200 → N=1195 drops ~7pp for snap_hyde (65.5% → 57.9%). Vectorless full-scale results pending.

## Top 10 Findings

1. **Snap reasoning is the biggest contributor** (+5pp). Forcing the model to reason before retrieval improves everything downstream.

2. **HyDE passage generation adds +3.5pp retrieval quality.** Passage-form queries match the doctrinal corpus better than question-form queries (genre mismatch between questions and passages).

3. **Cross-encoder reranking dominates embedding choice.** All 7 non-gte-large embedders converge to exactly 65.0% with question-based reranking. The embedding model barely matters.

4. **Vectorless RAG is competitive.** LLM-generated knowledge (64.5%) nearly matches vector retrieval (65.5%) on BarExam, eliminating the entire retrieval stack.

5. **Showing the snap answer in the final call causes anchoring.** Modes that show snap: 0.5-2% answer changes. Modes that hide snap: 7-27% changes.

6. **GAP_MIN_CE=1.0 was a critical bug** that made all gap experiments into llm_only (0% answer changes). Discovered via fix/break analysis.

7. **11-char HyDE outputs** were caused by prompt schema mismatch — Gemma merges system+user, and gap-formatted input didn't match the system prompt's expectation.

8. **N=200 variance is ~5-7pp.** snap_hyde ranged 62.5%-67.5% across duplicate N=200 runs. Full N=1195 is essential for reliable comparison.

9. **RAG helps only on unknown domains.** HousingQA (+9pp), BarExam (~0 net), CaseHOLD (-1.5pp).

10. **Self-correction consistently hurts.** self_verify -3pp, snap_debate -5pp, double_snap -2pp. Second-guessing without new information is destructive.

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

## Currently Running (as of 2026-04-13)

| Job | Mode | N | Purpose |
|---|---|---|---|
| 43471 | vectorless_direct + vectorless_hybrid | 1195 | Full-scale validation |
| 43458 | gap_rag_nosnap + gap_vectorless + gap_hyde_nosnap | 200 | Anchoring hypothesis test |
