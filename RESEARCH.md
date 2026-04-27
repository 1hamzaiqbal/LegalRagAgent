# Research Program

Persistent research state for the LegalRagAgent project. Read this first in any new session.

This project started as a heavy agentic RAG pipeline that hurt performance. We stripped it down, systematically tested each component, and found that simpler adaptive strategies beat complex ones. The long-term goal is still a strong full agentic pipeline, but we're rebuilding toward it intentionally and atomically — testing each element's effectiveness and documenting what works about the research process itself.

## Current execution status (2026-04-26 night, ~12h before Monday meeting)

**Live ground truth** is `docs/validation_log_2026-04-25.md`. Methods narrative is `docs/methods_characterization_2026-04-26.md`. Meeting summary is `docs/meeting_2026_04_27_summary.md`.

### Coverage wave landed (commit `56bffc8`)

15 of 17 cluster post-fix N=1195 cells landed clean. Missing: E4B `llm_only` + E4B `golden_passage` (job 54173 wallclocked at 28h, mode 3 partial / mode 4 never started).

**Gemma 4 26B BarExam (8/8 modes)** — `rag_snap_hyde` **81.17%** (+3.09pp vs `rag_simple` 78.08%) is the proven winner. Bug-fix decomposition lands cleanly: `llm_only` and `snap_only_in_final` both show identical +5.44pp formatter-only lift; `rag_simple` adds +1.85pp marginal retrieval-query fix.

**Gemma 4 E4B BarExam (6/8 modes)** — `rag_snap_hyde` **62.18%** (+3.69pp vs `rag_simple` 58.49%). Same winner as 26B; method lift +3-4pp cross-size, not a scaling artifact.

### Methods characterization (cross-model, multi-dataset)

**Cross-family BarExam llm_only N=100 board:**
- Llama 3.3 70b dense: 81%
- Gemma 4 26B-A4B MoE: 79.75% (cluster N=1195)
- Qwen3 30B MoE: 70% (N=100; +9.75pp Gemma 4 lead at the same MoE class)
- Qwen3 32b dense: 68%
- Gemma 3 27b dense: 68%
- Llama 4 Scout 17b MoE: 67%

**MuSiQue (multi-hop, N=30 via API) — methods DON'T LIFT cross-model:**

| Mode | Gemma 4 26B | Llama 70b |
|---|---|---|
| `rag_simple` | **26.7%** | **20.0%** |
| `rag_multi_query` | 23.3% | 20.0% |
| `planning_table_no_snap_v2` | 23.3% | 20.0% |
| `planning_table_with_snap_v2` | 16.7% | n/a |
| `rag_snap_hyde` | 20.0% | 13.3% |

NO method beats `rag_simple` on multi-hop across either model. snap-driven methods consistently underperform. **gold_retrieved is HIGHER for multi-query (87%) and ptable (90% on Llama)** but the model can't translate retrieval recall into EM. Bottleneck is composition over multiple passages.

**Cleanest snap-ablation:** `planning_table_no_snap_v2` 23.3% vs `planning_table_with_snap_v2` 16.7% on Gemma 4 26B — same prompt, same retrieval, only difference is snap-seeded plan-gen. Snap costs -6.6pp.

### Hardening (commits `171c2c4`, `97c204a`)

- Pre-flight smoke gate aborts on auth/404 in seconds
- Per-question circuit breaker (5 consecutive errors → SystemExit)
- Summary-write guard tags rows as `_FAILED-do-not-use` if error_rate > 50%
- Think-tag stripping unlocks Qwen3 family (was 1/5 → 4/5 on smoke)
- MuSiQue extractor strips `<span>...</span>` HTML wrappers (Llama 70b emitted these literally)

`logs/experiments.jsonl` now contains **270+** completed records across the cluster wave + cross-family API runs.

---

## Guiding Principles

Drawn from [Karpathy autoresearch](https://github.com/karpathy/autoresearch) and [Anthropic harness design](https://www.anthropic.com/engineering/harness-design).

### Experiment discipline (autoresearch)
1. **Single metric per experiment** — accuracy on fixed N=200, seed=42. No multi-variable sweeps.
2. **Fixed eval protocol** — don't change `eval/eval_harness.py` while experimenting on the pipeline.
3. **Keep improvements, discard regressions** — git commit on improvement, revert on regression.
4. **Simplicity criterion** — "removing something and getting equal results IS a great outcome." A 0.5% gain that adds ugly complexity is not worth it.
5. **Never stop the loop** — if stuck, re-read the data, try combinations, try more radical changes.
6. **Log failures too** — crashed/regressed experiments are information. Record them.
7. **One change at a time** — isolate variables. Don't combine untested ideas.

### Harness design (Anthropic)
8. **Generator-evaluator separation** — use `eval/eval_harness.py` as the objective judge, not self-assessment.
9. **Decompose into tractable chunks** — one experiment per hypothesis, not sweeps.
10. **Structured handoff** — every session ends by updating this file's Session Handoff section.
11. **Strip before adding** — "every component encodes an assumption about what the model can't do; stress-test those assumptions."
12. **Sprint contracts** — before each experiment, write: hypothesis + success criteria + keep/discard rule.
13. **Re-examine when capabilities change** — new models/providers should trigger reassessment of what's load-bearing.
14. **Evaluator calibration** — tune eval criteria carefully. Default LLM judgment is too lenient.

### Meta-principle (from this project)
15. **The project IS the harness** — we're not just using a research harness; we're researching what makes a harness effective. Document what works and what doesn't about the research process itself.

---

## Current Best Results

### Historical cross-dataset reference
Older Llama/Scout benchmark tables were moved out of the main research state so the active leak-fix and size-comparison front stays visible.
→ see `docs/archive/research_legacy_blocks.md` for the archived dataset and cross-model tables.

### Gemma 4 26B BarExam (post-fix N=1195, 8/8 modes — commit 56bffc8)

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

### Gemma 4 E4B BarExam (post-fix N=1195, 6/8 modes; missing llm_only + golden_passage)

| Mode | EM | Δ vs `rag_simple` |
|---|---|---|
| `rag_snap_hyde` | **62.18%** | **+3.69pp** ← same winner cross-size |
| `subagent_rag` | 60.92% | +2.43pp |
| `snap_hyde_report` | 60.75% | +2.26pp |
| `rag_hyde` | 60.59% | +2.10pp |
| `subagent_hyde` | 60.17% | +1.68pp |
| `subagent_hybrid` | 58.83% | +0.34pp |
| `rag_simple` (baseline) | 58.49% | — |
| `snap_only_in_final` | 57.82% | -0.67pp |

### Pre-fix N=200 numbers (historical, not directly comparable post-fix)

Older Gemma 4 E4B N=200 leaderboard (subagent_rag/fixed rag_hyde/snap_hyde_report all at 66.0%, vectorless_* family 61-65%, gap_* family 61-65%) is pre-leak-fix and pre-prompt-fix. Kept in `docs/archive/research_legacy_blocks.md` for audit continuity.

### Historical cross-model reference
Older cross-model comparison tables are archived alongside the legacy dataset table.
→ see `docs/archive/research_legacy_blocks.md` for the full reference block.

### Key findings (2026-04-26 update)

1. **`rag_snap_hyde` is the proven winner on legal MC across model sizes**: +3.09pp over rag_simple at 26B, +3.69pp at E4B. Cross-size lift is real, not a noise artifact.
2. **NO method beats `rag_simple` on MuSiQue multi-hop** — confirmed across Gemma 4 26B AND Llama 3.3 70b. Methods that lift on legal break on multi-hop entity composition.
3. **Snap-bias is real on multi-hop**: cleanest ablation = ptable_no_snap_v2 23.3% vs ptable_with_snap_v2 16.7% (same prompt, same retrieval). Snap costs -6.6pp.
4. **Retrieval recall is NOT the multi-hop bottleneck** — multi_query and ptable_no_snap both improve gold_retrieved (+3-4pp) but the model can't translate that into EM. Composition over multiple passages is the actual bottleneck.
5. **Bug-fix decomposition is rock-solid**: llm_only and snap_only_in_final both show identical +5.44pp formatter-only lift (no retrieval, so this is pure prompt-context recovery). `rag_simple` shows +7.29pp = +5.44 formatter + +1.85pp retrieval-query marginal.
6. **At 26B, `rag_simple` ≈ `golden_passage`** (78.08% vs 78.66%). The model has enough parametric coverage that retrieved evidence is barely net-positive; only `rag_snap_hyde` decisively wins.
7. **Showing snap answer letter to final agent always hurts** — regression-tested in `tests/test_sanitizer.py`. Strip the letter, keep the reasoning.
8. **Cross-family generation lift is real**: Gemma 4 26B-A4B beats Qwen3 30B MoE (direct architecture peer) by +9.75pp, beats Gemma 3 27b dense by +12pp.
9. **HyDE has a domain-specificity bound**: lifts on legal single-hop doctrine retrieval, but biases retrieval toward wrong-hop entities on multi-hop QA (gold_retrieved crashes from 83% → 50%). Synthesizer-prompt fix doesn't recover the snap+HyDE multi-hop loss.

### Historical (pre-2026-04-26) findings, kept for audit continuity

10. RAG value = f(LLM knowledge gap) — only HousingQA showed validated lift in cross-dataset reference (archived).
11. Counterevidence retrieval consistently hurts (devil -6pp, top-2 -3pp).
12. Self-consistency (3-vote) is a useful uncertainty signal — Scout disagrees more (40%) than Llama (23%).

---

## Experiment Queue

Each experiment follows the sprint contract format: hypothesis, change, success criteria, keep/discard rule.

### Tier 1 — Clear hypothesis, direct test

#### ~~1. Score thresholding~~ — COMPLETED (2026-03-27)
- **Result**: 80.0% BarExam (Llama 70B) — **NEW BEST**. KEPT.
- CE threshold < 4.0 → skip RAG, use snap answer directly.

#### ~~2. Aspect-based query rewrite~~ — COMPLETED (2026-03-27)
- **Result**: 76.0% BarExam — DISCARDED. Offline retrieval gains (CE 6.0 vs 3.0) did not translate end-to-end.

#### 3. Integrate confidence_gated into main.py
- **Hypothesis**: Making the full pipeline use confidence_gated routing by default will make the demo pipeline match eval performance.
- **Change**: Add self-consistency voting to the pipeline's router/executor. Route to RAG only on disagreement.
- **Success criteria**: Pipeline demo output matches eval harness accuracy for confidence_gated.
- **Keep/discard**: Keep unconditionally — this is integration, not experimentation.

### Tier 2 — Worth testing

#### 4. Adaptive k
- **Hypothesis**: Using snap confidence to choose retrieval depth (high confidence → k=3, low → k=7) improves over fixed k=5.
- **Change**: Add confidence-to-k mapping in snap_hyde flow.
- **Success criteria**: BarExam > 76.5%.
- **Keep/discard**: Keep if improvement; discard if neutral (principle 4: simplicity).

#### 5. MC choice-aware research
- **Hypothesis**: Making the planner and synthesizer aware of MC answer choices will let retrieval target distinguishing evidence rather than generic research.
- **Change**: Modify `skills/planner.md` (remove "don't structure steps around answer choices" line), update synthesizer prompt.
- **Success criteria**: BarExam improvement.
- **Keep/discard**: Keep if improvement. Watch for overfitting to MC format.

#### 6. State filtering for HousingQA
- **Hypothesis**: Extracting jurisdiction from the question and using it as a ChromaDB metadata filter will dramatically improve HousingQA retrieval quality.
- **Change**: Add state extraction to router_node, pass as metadata filter to retrieval.
- **Success criteria**: HousingQA > 56% (current snap_hyde).
- **Keep/discard**: Keep if improvement. Note: requires HousingQA eval which OOMs on 16GB when other processes run.

### Tier 3 — Speculative / deferred

#### 7. Context-Aware Decoding
- Contrast output probabilities with/without retrieved context. Research-heavy. May not be feasible with API-only models (need logprobs).

#### ~~8. Embedding model comparison~~ — COMPLETED (2026-04-11)
- **7 embedders tested** across 3 modes (rag_simple, snap_hyde, snap_hyde_aligned), N=200 each.
- **Key finding**: Cross-encoder reranking dominates — all 6 alternative embedders converge to exactly 65.0% with question-based reranking under `snap_hyde_aligned`. Embedding choice matters more on `rag_simple` than on aligned reranking.
- **Wave 1** (gte-large, legal-bert, stella-400m, bge-m3): snap_hyde 60% for non-gte, gte-large 65.5%.
- **Wave 2** (jina-v3, arctic-l-v2, nomic-v2-moe): snap_hyde 64.5% for all three — closer to gte-large.
- **Failed builds**: gte-qwen2-1.5b, stella-1.5b (transformers rope_theta compat).

#### 9. Domain-adaptive routing
- Automatically detect whether errors are random (→ confidence_gated) or systematic (→ snap_hyde). Requires characterizing the domain's error mode, which may need a calibration run.

#### ~~10. Gap-informed retrieval~~ — COMPLETED (2026-04-13)
- **Architecture tested**: `SNAP → ANALYZE GAPS → SUBAGENT RETRIEVAL (per gap) → FINAL REASONING`
- **Variants tested**: `gap_rag`, `gap_hyde`, `gap_hyde_ev`, `gap_hyde_nosnap`, `gap_hyde_flat`, plus follow-up anchoring ablations.
- **Result**: `gap_rag_nosnap` reached **64.5%**, fixed `gap_rag` reached **63.5%**, fixed `gap_hyde_nosnap` reached **62.5%**, and `gap_vectorless` reached **61.5%** on Gemma 4 E4B (N=200). Anchoring was confirmed, but the gap family still underperforms `snap_hyde` (**65.5%**) and `subagent_rag` (**66.0%**).
- **Keep/discard**: keep as an analyzed ablation family, discard as the main direction for now.

#### 11. Historical vectorless full-scale / keyword follow-up
- **Status**: baseline sweep complete at N=200; the planned full N=1195 "vectorless" jobs were canceled after we concluded the label was misleading and not testing real corpus search.
- **What is already done**: `vectorless_hybrid` **65.0%**, `vectorless_direct` / `vectorless_choice_map` **64.5%**, `vectorless_role` **63.5%**, `vectorless_elements` **61.0%**.
- **What remains**: define a real corpus-search control (`vectorless_keyword`, BM25 / structured index navigation, or similar) before any new full-scale follow-up.

#### ~~12. Combo-mode anchoring controls~~ — COMPLETED (2026-04-17)
- **Architecture tested**: `snap_hyde_report`, `snap_hyde_report_snap`, `subagent_rag_snap`, `subagent_rag_full`
- **Result**: `snap_hyde_report` matched the N=200 lead at **66.0%**, but every variant that exposed snap to the final agent underperformed: `snap_hyde_report_snap` **64.0%**, `subagent_rag_snap` **63.0%**, `subagent_rag_full` **62.0%**
- **Keep/discard**: keep report-only compression as a valid control; discard visible-snap combo variants as a main direction

#### ~~13. Full fixed HyDE rerun~~ — COMPLETED (2026-04-17)
- **Result**: fixed full `rag_hyde` reached **57.9%** (`692/1195`), matching the paired full `snap_hyde` rerun exactly
- **Keep/discard**: keep the HyDE prompt fix unconditionally; discard the old apparent `+3pp` snap lift in the HyDE family as a bug artifact

---

## Plan Snapshot (2026-04-17)
Historical phase-by-phase closure notes are archived; the live execution front is the 2026-04-20/21 leak-fix and size-comparison block at the top of this file.
→ see `docs/archive/research_legacy_blocks.md` for the full April 17 snapshot.

---

## Historical Reference Block
The 2026-03-27 CE-threshold reference table and its session learnings are archived to keep the main file focused on current work.
→ see `docs/archive/research_legacy_blocks.md` for the full historical block.

---

## Session Handoff

### Archived early handoff references
The 2026-04-03 audit marker and the audited April 17 meeting-action table are preserved in the archive.
→ see `docs/archive/research_legacy_blocks.md` for the verbatim reference blocks.

### Recent verified timeline
- 2026-04-03 audit: confirmed 5/7 full-set BarExam baselines, fixed stale full-run detection in `eval/run_experiment_queue.py` and `eval/monitor.py`, and recorded the cluster bring-up workflow.
- 2026-04-05 full-set comparisons: `or-gemma27b` baseline landed at **57.99%**; `golden_passage` materially beat plain retrieval on both `or-qwen3-32b` and `or-gemma27b`, confirming retrieval quality as the main bottleneck.
- 2026-04-07 through 2026-04-11 HPC block: full Qwen3-8B and Gemma 4 E4B runs completed; the focused 7-embedder sweep completed with 2 documented build failures.
- 2026-04-14 block: snap/no-snap ablations and cross-dataset follow-up both completed; full `subagent_rag` landed at **56.9%** (`680/1195`); and the misnamed full-vectorless jobs were formally canceled.
- 2026-04-15 block: `entity_search` full landed at **53.2%** (`636/1195`); `snap_entity_informed` reached **59.5%**; `subagent_hyde` reached **62.5%**; and the first full `rag_hyde` rerun was later superseded by the repaired April 17 rerun.
- 2026-04-16 block: full `ce_threshold` landed at **55.9%** (`668/1195`), full `gap_rag_nosnap` landed at **55.9%** (`668/1195`), and the full `subagent_rag` 1-gap rerun improved to **57.2%** (`684/1195`).
- 2026-04-17 block: fixed `rag_hyde` validated at **66.0%** (`132/200`), `snap_hyde_report` also reached **66.0%** (`132/200`), combo block `48393` closed with `snap_hyde_report_snap` **64.0%**, `subagent_rag_snap` **63.0%**, and `subagent_rag_full` **62.0%**, and the repaired full `rag_hyde` rerun `48555` finished at **57.9%** (`692/1195`). `logs/experiments.jsonl` now contains **195** entries.

### Current handoff
- Verified complete: Phase 1 small-model baselines, Gemma/Qwen HPC full runs, focused embedding sweep, historical vectorless baseline sweep, anchoring controls, subagent follow-up sweep, fixed gap reruns, snap ablations, cross-dataset follow-up, the April 15-16 full rerun block, the combo-mode controls, and the repaired April 17 full HyDE rerun. `logs/experiments.jsonl` now contains **195** recorded runs.
- Verified but still lower-priority historical findings: `golden_passage` consistently outperforms current plain retrieval on the strongest full-set models; `confidence_gated` remains the best Llama 70B adaptive baseline after `ce_threshold`.
- Full N=1195 comparison on the fixed HyDE path: `golden_passage` **62.2%**, `snap_hyde` **57.9%** (paired rerun; earlier best run **58.6%**), fixed `rag_hyde` **57.9%**, `subagent_rag` (1-gap) **57.2%**, `subagent_rag` **56.9%**, `ce_threshold` **55.9%**, `gap_rag_nosnap` **55.9%**, `llm_only` **55.5%**, `rag_simple` **54.2%**, `entity_search` **53.2%**.
- Newly closed: combo-mode job `48393` confirmed that showing snap to the final agent always hurts (`snap_hyde_report_snap` **64.0%**, `subagent_rag_snap` **63.0%**, `subagent_rag_full` **62.0%**) relative to the hidden-snap controls.
- Newly closed: fixed `rag_hyde` reached **66.0%** (`132/200`) at N=200 and **57.9%** (`692/1195`) at full scale, tying the paired `snap_hyde` rerun and invalidating the old apparent `+3pp` HyDE snap lift.
- Still pending: real corpus-search follow-ups that use the finished case-summary layer / rebuilt entity graph, `vectorless_keyword`, plus deferred OpenRouter baselines `or-nemotron` and `or-qwen35-9b`.
- Most likely next high-signal work: use the now-closed `48393` / `48555` results to tighten the paper narrative, then test real corpus-search controls or integrate `confidence_gated` into `main.py`.

### Blockers
- Cluster GPU availability (general-gpu partition, priority queue)
- `a100-2207` and `a100s-2307` are now known bad vLLM nodes; `r28-1801` is excluded for insufficient VRAM
- Full-set local inference is still expensive in wall-clock time; Gemma 4 E4B `rag_snap_hyde` reruns are roughly a 10-12h job
- Cerebras API still broken (empty responses)

---

## File Pointers

| File | Purpose |
|------|---------|
| `RESEARCH.md` (this file) | Research state, experiment queue, session handoff |
| `EXPERIMENTS.md` | Full experiment log (hypothesis → result → verdict) |
| `CLAUDE.md` | Operational source of truth (how to run, environment notes) |
| `logs/experiments.jsonl` | Machine-readable results (one JSON record per run) |
| `ideas/actionable_ideas.md` | Idea backlog archive (active queue is here) |
| `docs/experiment_summary.md` | Narrative experiment summary (generated 2026-03-30) |
| `docs/cluster_workflow.md` | Cluster bring-up plan for local inference + full evals |
