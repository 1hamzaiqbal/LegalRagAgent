# Experiment Overview

High-level summary of the LegalRagAgent experimental program. Source of truth: `logs/experiments.jsonl` (**210** entries as of 2026-04-21).

For individual experiment details: `EXPERIMENTS.md`. For research state: `RESEARCH.md`.
<!-- Intentional overlap with docs/meeting_2026_04_17.md: this file is the cleaned synthesis, while the meeting notes preserve point-in-time discussion. Some duplicated summary content is expected and should not be removed automatically. -->

## Timeline

- Phases 0-3 (March 22 to April 11): harness/baseline setup, small-model audit, and embedding comparison are complete.
- Phases 4-6 (April 10-13): gap-family fixes, anchoring controls, and historical `vectorless_*` / parametric-reasoning work are complete.
- Phases 7-9 (April 14-16): cross-dataset checks, structured-search follow-up, and the first full rerun block are complete.
- → see `docs/archive/phases_early.md` for the full Phase 0-9 timeline and the archived 2026-04-19 cluster-status snapshot.

### Phase 10: HyDE Fix + Combo Modes (April 17)
- **Fixed `rag_hyde` full N=1195: 57.9%** — matches `snap_hyde` exactly. The +3pp snap lift for HyDE was a bug artifact.
- Fixed `rag_hyde` N=200: **66.0%** (validates the prompt fix)
- `snap_hyde_report` N=200: **66.0%** (snap_hyde + summarization = no gain)
- `snap_hyde_report_snap` N=200: **64.0%** (showing snap hurts -2pp)
- `subagent_rag_snap` N=200: **63.0%** (showing snap hurts -3pp vs 66.0%)
- `subagent_rag_full` N=200: **62.0%** (max info hurts -4pp vs 66.0%)
- **Key finding: showing snap to the final agent ALWAYS hurts (-2 to -4pp)**
- `logs/experiments.jsonl` reached **195** entries

## 2026-04-20: Leakage audit

A leakage audit on 2026-04-20 found answer-letter contamination in the HyDE-family retrieval query: **100%** of historical `rag_hyde` passages and **74%** of historical `rag_snap_hyde` passages began with `Answer: (X)`. `_sanitize_intermediate_text` was added in `02edbb7` on 2026-04-17, but it landed after the canonical N=1195 runs were logged and still had regex bugs, so every canonical HyDE-family leaderboard number in `logs/experiments.jsonl` is currently a pre-leak-fix reference.

Hardening landed on `hpc-setup` through `dfb6a9b`: `e508765`, `951729d`, `bf89b78`, `baef4d8`, `0b4e35d`, `71533fd`, `c85fe70`, `a377867`, `6118161`, `bab7cf5`, `a493491`. Smoke job `50812` confirmed generation-time cleanup with `rag_hyde` **19/30 (63.3%)** and `rag_snap_hyde` **21/30 (70.0%)**, both with `top_level_hyde_artifacts=0`; clean reruns are in flight via `50835`, `50836`, and pending job `50822`.

## 2026-04-21: Post-Fix Size-Comparison Wave

- Clean E4B mini-eval `50835` finished at `rag_simple` **60.5%**, `rag_hyde` **59.5%**, `rag_snap_hyde` **66.5%**, and `snap_only_in_final` **64.0%**, all with 0% leak.
- Narrative flip: the old "snap adds 0pp to HyDE" read was a leak artifact; clean E4B N=200 now shows snap adds **+7.0pp** over `rag_hyde` (**59.5% → 66.5%**).
- Landed full N=1195 post-fix rows already show monotonic `rag_simple` scaling: E2B **45.4%**, E4B **55.7%**, 26B-A4B **70.8%**, 31B **79.6%**; 26B-A4B `rag_hyde` has also landed at **74.2%**.

## Paper Core Result (Gemma 4 E4B, BarExam)

| Family | No-snap mode | No-snap acc | Snap mode | Snap acc | Snap lift |
|---|---|---|---|---|---|
| HyDE retrieval* | `rag_hyde` | 57.9% [pre-fix] | `snap_hyde` | 57.9% [pre-fix] | **0.0pp** (full N=1195 repaired comparison) |
| Plain RAG* | `rag_simple` | 57.0% | `snap_rag` | 62.0% | **+5.0pp** |
| Parametric reasoning | `vectorless_nosnap` | 59.5% | `vectorless_direct` | 64.5% | **+5.0pp** |

*HyDE uses the repaired full N=1195 comparison because the original April 14 N=200 `rag_hyde` row was prompt-tainted. Those canonical `rag_hyde` / `rag_snap_hyde` runs are still pre-leak-fix historical references after the 2026-04-20 audit; the earlier `rag_snap_hyde` **58.6%** peak was also pre-fix. Plain-RAG uses the existing `gte-large` April 10 reference pair so the comparison stays aligned with the paper's main ablation setting.

## Key Results (Gemma 4 E4B, BarExam)

### N=200 (seed=42) — Validated Modes

| Rank | Mode | Acc | Changed | Net | Calls | Vector Store? |
|---|---|---|---|---|---|---|
| 1 | subagent_rag | **66.0%** | — | — | 4.1 avg | yes |
| 1 | rag_hyde FIXED | **66.0%** | — | — | 2 | yes |
| 1 | snap_hyde_report | **66.0%** | — | — | 4 | yes |
| 4 | snap_hyde | **65.5%** | 27% | +37 | 3 | yes |
| 5 | vectorless_hybrid | **65.0%** | 18% | +7 | 4 | yes (k=3) |
| 6 | gap_rag_nosnap | **64.5%** | — | — | 3.0 avg | yes |
| 6 | vectorless_direct | **64.5%** | 19% | +6 | 3 | **no** |
| 6 | vectorless_choice_map | **64.5%** | — | — | 3 | **no** |
| 9 | ce_threshold | **64.0%** | 10% | +5 | 2-3 | yes |
| 9 | snap_hyde_report_snap | **64.0%** | — | — | 4 | yes |
| 11 | subagent_hybrid | **63.5%** | — | — | 4.1 avg | yes |
| 11 | vectorless_role | **63.5%** | 7% | +4 | 3 | **no** |
| 11 | gap_rag FIXED | **63.5%** | 2% | +4 | 3-6 | yes |
| 14 | rag_arbitration | **63.0%** | 6% | +3 | 3 | yes |
| 14 | subagent_rag_snap | **63.0%** | — | — | 4 | yes |
| 16 | gap_hyde_nosnap FIXED | **62.5%** | — | — | 4.1 avg | yes |
| 16 | subagent_hyde | **62.5%** | — | — | 5.2 avg | yes |
| 18 | snap_rag | **62.0%** | 1% | +2 | 2 | yes |
| 18 | gap_hyde FIXED | **62.0%** | 0.5% | +1 | 4-8 | yes |
| 18 | subagent_rag_full | **62.0%** | — | — | 4 | yes |
| 21 | gap_vectorless | **61.5%** | — | — | 4.1 avg | **no** |
| 22 | vectorless_elements | **61.0%** | — | — | 3 | **no** |
| 22 | subagent_rag_evidence | **61.0%** | — | — | 4.1 avg | yes |
| 24 | entity_search | **60.0%** | — | — | 1 | **no** |
| 25 | rag_rewrite | **59.5%** | — | — | 3 | yes |
| 25 | vectorless_nosnap | **59.5%** | — | — | 2 | **no** |
| 25 | snap_entity_informed | **59.5%** | — | — | 2 | **no** |
| 28 | rag_simple | **57.0%** | — | — | 1 | yes |
| 29 | llm_only | **55.5%** | — | — | 1 | no |

Note: the `vectorless_*` label is historical shorthand. `vectorless_direct`, `vectorless_role`, `vectorless_elements`, `vectorless_choice_map`, and `gap_vectorless` are multi-turn LLM reasoning / parametric-knowledge modes, not corpus search. `vectorless_hybrid` is the only one that still pools generated knowledge with vector retrieval.
The combo-mode additions reinforce the anchoring result: hiding snap from the final call preserves the 66.0% tier, while re-exposing snap or raw passages drops the score by 2-4pp.

### Full-Scale N=1195

| Mode | Accuracy | Detail Log |
|---|---|---|
| **31B rag_simple** | **79.6% [post-fix, full N=1195]** | `logs/eval_rag_simple_cluster-vllm_20260421_1203_detail.jsonl` |
| **26B-A4B rag_hyde** | **74.2% [post-fix, full N=1195]** | `logs/eval_rag_hyde_cluster-vllm_20260421_1112_detail.jsonl` |
| **26B-A4B rag_simple** | **70.8% [post-fix, full N=1195]** | `logs/eval_rag_simple_cluster-vllm_20260421_0857_detail.jsonl` |
| golden_passage | 62.2% | `logs/eval_golden_passage_cluster-vllm_20260408_1749_detail.jsonl` |
| **snap_hyde [E4B]** | **57.9% [pre-fix; clean rerun pending]** | `logs/eval_rag_snap_hyde_cluster-vllm_20260413_1102_detail.jsonl` |
| **rag_hyde (fixed) [E4B]** | **57.9% [pre-fix; clean rerun pending]** | `logs/eval_rag_hyde_cluster-vllm_20260417_2047_detail.jsonl` |
| **subagent_rag (1-gap)** | **57.2%** | `logs/eval_subagent_rag_cluster-vllm_20260416_1720_detail.jsonl` |
| **subagent_rag** | **56.9%** | `logs/eval_subagent_rag_cluster-vllm_20260414_1115_detail.jsonl` |
| **ce_threshold** | **55.9%** | `logs/eval_ce_threshold_cluster-vllm_20260415_2022_detail.jsonl` |
| **gap_rag_nosnap** | **55.9%** | `logs/eval_gap_rag_nosnap_cluster-vllm_20260416_0544_detail.jsonl` |
| **E4B rag_simple** | **55.7% [post-fix, full N=1195]** | `logs/eval_rag_simple_cluster-vllm_20260421_0812_detail.jsonl` |
| llm_only | 55.5% | `logs/eval_llm_only_cluster-vllm_20260408_1709_detail.jsonl` |
| rag_simple [pre-fix E4B] | 54.2% | `logs/eval_rag_simple_cluster-vllm_20260408_1813_detail.jsonl` |
| entity_search | 53.2% | `logs/eval_entity_search_cluster-vllm_20260415_0454_detail.jsonl` |
| **E2B rag_simple** | **45.4% [post-fix, full N=1195]** | `logs/eval_rag_simple_cluster-vllm_20260421_0802_detail.jsonl` |
| vectorless_direct | **CANCELLED** | job `43471` canceled — mode is parametric reasoning, not real corpus search |
| vectorless_hybrid | **CANCELLED** | job `43471` canceled — same naming / validity issue |

Note: the new E2B/E4B/26B/31B rows above are the landed post-fix size-comparison entries from 2026-04-21. The full E4B `snap_hyde` / `rag_hyde` rows remain pre-leak-fix historical references until the clean reruns in `docs/size_comparison_matrix.md` finish. The later 1-gap `subagent_rag` rerun improved to **57.2%**; and both `ce_threshold` and `gap_rag_nosnap` flatten at **55.9%**, barely above `llm_only` (**55.5%**). The planned full-scale "vectorless" runs were canceled because they would only validate extra reasoning steps, not corpus search.
Scale note: `entity_search` falls **6.8pp** from N=200 to N=1195 (`60.0% -> 53.2%`), while vector `rag_simple` falls only **2.8pp** (`57.0% -> 54.2%`). NLP entity matching is therefore less robust than vector search at scale in the current corpus setup.

### Cross-Dataset Follow-Up (Gemma 4 E4B, N=200)

| Dataset | llm_only | vectorless_direct | vectorless_nosnap | snap_hyde | Key take-away |
|---|---|---|---|---|---|
| HousingQA | **50.5%** | 50.0% | 52.5% | 50.0% | Parametric reasoning does not solve the unknown-domain problem here |
| CaseHOLD | **69.5%** | 68.0% | 67.5% | — | Parametric reasoning hurts citation-matching relative to `llm_only` |

## Top 10 Findings

Note: the full-scale HyDE-family findings below reference the pre-leak-fix canonical E4B runs until the clean reruns finish; the clean E4B mini-eval already flips snap-over-HyDE to **+7.0pp** (`59.5% -> 66.5%`).

1. **HyDE is the real driver.** Passage-form queries bridge the genre gap between question-form queries and doctrinal corpus passages. `rag_hyde` (fixed) = `snap_hyde` = **57.9%** at full N=1195. The previous +3pp snap lift for HyDE was a bug artifact.

2. **Snap helps plain RAG (+5pp) and parametric reasoning (+5pp), but adds zero to HyDE.** Snap is valuable when the retrieval query is the raw question (genre mismatch), but HyDE already solves that problem.

3. **Showing snap to the final agent always hurts (-2 to -4pp).** Confirmed across `snap_hyde_report_snap` (64.0%), `subagent_rag_snap` (63.0%), and `subagent_rag_full` (62.0%) — all worse than their no-snap counterparts.

4. **Cross-encoder reranking dominates embedding choice.** All 7 non-gte-large embedders converge to exactly 65.0% with question-based reranking. The embedding model barely matters.

5. **Subagent reports help at N=200 but not at full scale.** `subagent_rag` reached 66.0% on N=200; the best full rerun is the 1-gap variant at 57.2%, below `snap_hyde` / `rag_hyde` at 57.9%.

6. **"Vectorless" is competitive, but the name is misleading.** These modes are multi-turn parametric reasoning baselines, not corpus search.

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

## Historical Cluster Status (as of 2026-04-19)
Superseded by the 2026-04-20/21 leakage audit and size-comparison updates above.
→ see `docs/archive/phases_early.md` for the full 2026-04-19 job table.
