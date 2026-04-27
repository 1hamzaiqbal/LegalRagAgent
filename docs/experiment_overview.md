# Experiment Overview

High-level summary of the LegalRagAgent experimental program. Source of truth: `logs/experiments.jsonl` (**270+** entries as of 2026-04-26 night).

For individual experiment details: `EXPERIMENTS.md`. For research state: `RESEARCH.md`. **Live ground truth + meeting story: `docs/validation_log_2026-04-25.md` and `docs/methods_characterization_2026-04-26.md`.**

## Timeline

- Phases 0-9 (March 22 to April 16): harness/baseline setup, embedding comparison, gap-family, anchoring, vectorless, cross-dataset checks. → archived in `docs/archive/phases_early.md`.
- Phase 10 (April 17): HyDE fix + combo modes.
- Phase 11 (April 22): **Prompt-column bug discovered + patched** (commits `f95f316` + `3d5ff05`). BarExam dataset has a `prompt` column with shared fact pattern for 445/1195 questions (37%). The harness was silently dropping it from BOTH the model-facing prompt AND 11 retrieval call sites. Every BarExam number before commit `3d5ff05` is a pre-prompt-fix reference.
- **Phase 12 (April 25-26): Coverage wave + cross-family validation.**

### Phase 12 highlights (current)

- **15 of 17 cluster post-fix N=1195 cells landed** at commit `56bffc8` (E4B llm_only + golden_passage missing — 54173 wallclocked at 28h)
- **Bug-fix decomposition** validates clean: `llm_only` and `snap_only_in_final` both show **identical +5.44pp** lift at 26B (formatter-only, no retrieval). `rag_simple` shows +7.29pp = +5.44 formatter + +1.85pp marginal retrieval-query fix. Two no-retrieval modes producing identical lift is the strongest possible internal validation.
- **`rag_snap_hyde` is the proven winner on legal MC**: +3.09pp at 26B (78.08 → 81.17) and +3.69pp at E4B (58.49 → 62.18). Cross-size lift, not noise.
- **6-model BarExam llm_only N=100 board**: Llama 3.3 70b 81%, **Gemma 4 26B-A4B 79.75%**, Qwen3 30B MoE 70%, Qwen3 32b 68%, Gemma 3 27b 68%, Llama 4 Scout 17b 67%.
- **MuSiQue cross-method × cross-model (N=30 via API)**: NO method beats `rag_simple` on multi-hop. snap+HyDE breaks (-6.7pp on Gemma 26B, -6.7pp on Llama 70b). Cleanest snap-ablation: `ptable_no_snap_v2` 23.3% vs `ptable_with_snap_v2` 16.7% (-6.6pp from snap, same prompt + same retrieval).
- **Hardening** (commits `171c2c4`, `97c204a`): pre-flight smoke gate, per-question circuit breaker, summary-write guard for high-error-rate runs, think-tag stripping for Qwen3, `<span>` extractor fix for MuSiQue.

## Paper Core Result (post-prompt-fix N=1195, BarExam)

### Gemma 4 26B-A4B (3.8B active, MoE)

| Mode | EM | Δ vs `rag_simple` |
|---|---|---|
| `rag_snap_hyde` | **81.17%** | **+3.09pp** ← winner |
| `snap_only_in_final` | 80.59% | +2.51pp |
| `llm_only` | 79.75% | +1.67pp |
| `rag_hyde` | 78.91% | +0.83pp |
| `golden_passage` (oracle) | 78.66% | +0.58pp |
| `subagent_rag` | 78.16% | +0.08pp |
| `rag_simple` (baseline) | 78.08% | — |
| `subagent_hybrid` | 74.14% | -3.94pp |

### Gemma 4 E4B (8B effective)

| Mode | EM | Δ vs `rag_simple` |
|---|---|---|
| `rag_snap_hyde` | **62.18%** | **+3.69pp** ← same winner |
| `subagent_rag` | 60.92% | +2.43pp |
| `snap_hyde_report` | 60.75% | +2.26pp |
| `rag_hyde` | 60.59% | +2.10pp |
| `subagent_hyde` | 60.17% | +1.68pp |
| `subagent_hybrid` | 58.83% | +0.34pp |
| `rag_simple` | 58.49% | — |
| `snap_only_in_final` | 57.82% | -0.67pp |

`rag_snap_hyde` lifts +3-4pp over `rag_simple` consistently across both model sizes — **method effect is real and cross-size**, not a scaling artifact.

## Multi-hop ceiling story (MuSiQue)

| Mode | Gemma 4 26B | Llama 70b |
|---|---|---|
| `rag_simple` | **26.7%** (N=30) | 20.0% (N=30) / **21.0%** (N=100) |
| `rag_multi_query` | 23.3% (N=30) | 20.0% (N=30) |
| `planning_table_no_snap` v2 | 23.3% (N=30) | 20.0% (N=30) |
| `planning_table_with_snap` v2 | 16.7% (N=30) | n/a |
| `iterative_planning_table` | 20.0% (N=30) | 23.3% (N=30) |
| `advisor_planning_table` (cheap-plan/strong-synth) | 23.3% (N=30) | 23.3% (N=30) / **23.0% (N=100)** |
| `rag_snap_hyde` | 20.0% (N=30) | 13.3% (N=30) |
| `golden_passage` (oracle) | 62% (N=30) | n/a |

NO method significantly beats `rag_simple` on multi-hop across either model. Snap-bias hurts; multi-query and planning_table improve gold_retrieved (+3-4pp) but model can't translate retrieval recall into EM. **Bottleneck is composition over multiple passages, not retrieval coverage.**

**Advisor N=100 follow-up (Llama 70b)**: directional +2.0pp vs `rag_simple` (23.0% vs 21.0%), but McNemar paired p=0.824 — NOT statistically significant; 95% CI [-7pp, +11pp] crosses zero. The N=30 "advisor never loses to rag_simple" property dissolved at N=100 (b=11, c=9). Frame as **cost-parity** (86% strong-LLM input-token reduction, 43% output-token reduction vs `iter_ptable`) rather than accuracy-lift. Audit `a5bbd0b5840ac0da6`, validation_log § "advisor_planning_table on Llama 70b — N=100 follow-up".

## Historical (pre-prompt-fix) snapshots — kept for audit continuity

The Paper Core Result table previously cited HyDE-family numbers at 57.9% / +0pp snap lift (full N=1195 pre-fix) and 60.5% / 66.5% (E4B N=200 post-leak-fix-only). Both are superseded by the Phase 12 post-prompt-fix matrix above. See `docs/archive/research_legacy_blocks.md` for the older tables.

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
