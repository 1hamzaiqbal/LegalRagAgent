# Research Program

Persistent research state for the LegalRagAgent project. Read this first in any new session.

This project started as a heavy agentic RAG pipeline that hurt performance. We stripped it down, systematically tested each component, and found that simpler adaptive strategies beat complex ones. The long-term goal is still a strong full agentic pipeline, but we're rebuilding toward it intentionally and atomically — testing each element's effectiveness and documenting what works about the research process itself.

## Current execution status
- 2026-04-20/21 leakage audit: every canonical HyDE-family leaderboard number previously cited from `logs/experiments.jsonl` is pre-leak-fix. Historical `rag_hyde` passages leaked `Answer: (X)` in **100%** of samples and historical `rag_snap_hyde` passages in **74%**.
- Hardening is on `hpc-setup` through `d0709bd` plus follow-on commits. Mini-eval smoke `50812` confirmed 0% leak at generation (`rag_hyde` 63.3%, `rag_snap_hyde` 70.0% at N=30); full mini-eval `50835` landed clean at N=200.
- **Narrative flip (2026-04-21)**: post-fix E4B N=200 shows `rag_simple` 60.5%, `rag_hyde` 59.5%, `rag_snap_hyde` 66.5%, `snap_only_in_final` 64.0%. **Snap now adds +7pp over plain HyDE** — the old "snap adds 0pp to HyDE" claim was a leak artifact.
- **Size-comparison wave in flight**: 12 full-N=1195 SLURM jobs dispatched across E2B/E4B/26B-A4B/31B covering a 10-mode matrix (llm_only, golden_passage, rag_simple, rag_hyde, rag_snap_hyde, snap_only_in_final, subagent_rag, subagent_hyde, subagent_hybrid, snap_hyde_report). Plan + live snapshot in `docs/size_comparison_matrix.md`.
- Landed full N=1195 rows so far (all post-fix, 0% leak): E2B `rag_simple` **45.4%**, E4B `rag_simple` **55.7%**, 26B-A4B `rag_simple` **70.8%** + `rag_hyde` **74.2%**, 31B `rag_simple` **79.6%**. Clean monotonic scaling on `rag_simple` across sizes. 31B N=200 matrix landed earlier at 79-85% across all 4 core modes.
- Active jobs: 50858/50859 (E4B P1a/P1b full), 50865 (31B full), 50868 (26B full), 50986 (E2B redo after 50867 wallclocked), 50990-50992 (26B expansion on a100s-2305 parallel), 50993-50995 (31B expansion queued for H100).
- The April 17 cluster follow-up is now complete: combo-mode block `48393` finished with `snap_hyde_report` **66.0%**, `snap_hyde_report_snap` **64.0%**, `subagent_rag_snap` **63.0%**, and `subagent_rag_full` **62.0%**; fixed full `rag_hyde` rerun `48555` finished at **57.9%** (`692/1195`), matching the paired full `snap_hyde` rerun and confirming snap adds **0pp** inside the HyDE family. Since the April 13 handoff, jobs `44394`, `44395`, `45350`, `45735`, `48393`, and `48555` all completed successfully. The case-summary build `44371` is done, and the entity-graph rebuild `44520` was last noted at 74%.
- Core Phase 1 small-model baseline block is complete; lower-priority OpenRouter extras (`or-nemotron`, `or-qwen35-9b`) remain explicitly deferred.
- Best Gemma 4 E4B N=200 result is now a three-way tie at **66.0%**: `subagent_rag`, fixed `rag_hyde`, and `snap_hyde_report`. The full-set Gemma 4 E4B comparison currently cites the pre-leak-fix HyDE numbers: `golden_passage` **62.2%**, `snap_hyde` **57.9%** (paired rerun; earlier best run **58.6%**), fixed `rag_hyde` **57.9%**, `subagent_rag` (1-gap) **57.2%**, `subagent_rag` **56.9%**, `ce_threshold` **55.9%**, `gap_rag_nosnap` **55.9%**, `llm_only` **55.5%**, `rag_simple` **54.2%**, `entity_search` **53.2%**.
- April 17 HyDE/combo validation is now closed: fixed `rag_hyde` reached **66.0%** at N=200, `snap_hyde_report` also reached **66.0%**, but every combo that exposed snap to the final agent underperformed: `snap_hyde_report_snap` **64.0%**, `subagent_rag_snap` **63.0%**, `subagent_rag_full` **62.0%**.
- Subagent follow-up sweep is complete: `subagent_rag` **66.0%**, `subagent_hybrid` **63.5%**, `subagent_rag_evidence` **61.0%** (Gemma 4 E4B, N=200).
- "Vectorless" baseline sweep is complete: `vectorless_hybrid` **65.0%**, `vectorless_direct` **64.5%**, `vectorless_choice_map` **64.5%**, `vectorless_role` **63.5%**, `vectorless_elements` **61.0%** (Gemma 4 E4B, N=200). Naming caveat: these modes are multi-turn LLM reasoning / parametric-knowledge exploitation, not real corpus search, so the full N=1195 vectorless jobs were canceled.
- Embedding comparison is complete for supported builds: **7 embedders tested**; `jina-v3`, `arctic-l-v2`, and `nomic-v2-moe` all finished at 61.5% `rag_simple` / 64.5% `rag_snap_hyde`; `gte-qwen2-1.5b` and `stella-1.5b` failed to build.
- Gap-family reruns are complete: `gap_rag_nosnap` **64.5%**, fixed `gap_rag` **63.5%**, fixed `gap_hyde_nosnap` **62.5%**, fixed `gap_hyde` **62.0%**, and `gap_vectorless` **61.5%** on Gemma 4 E4B (N=200). Anchoring is real, but the gap family still trails the best simpler baselines.
- `logs/experiments.jsonl` now contains **210** completed experiment records (cluster source of truth; growing with the size-comparison wave).

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

### Gemma 4 E4B + historical vectorless / parametric-reasoning snapshot (BarExam)

| Method | Accuracy | Scale | Notes |
|--------|----------|-------|-------|
| `subagent_rag` / fixed `rag_hyde` / `snap_hyde_report` | **66.0%** | N=200 | Current top tie on Gemma 4 E4B |
| `rag_snap_hyde` | **65.5%** | N=200 | Best 3-call retrieval baseline |
| `vectorless_hybrid` | **65.0%** | N=200 | Best vectorless result |
| `gap_rag_nosnap` | **64.5%** | N=200 | Best gap-family no-snap control |
| `vectorless_direct` / `vectorless_choice_map` | **64.5%** | N=200 | No vector store required |
| `snap_hyde_report_snap` | **64.0%** | N=200 | Showing snap in the final report-first call costs 2pp |
| `subagent_hybrid` / `vectorless_role` / fixed `gap_rag` | **63.5%** | N=200 | Second-tier follow-ups |
| `subagent_rag_snap` / `subagent_rag_full` | **63.0%** / **62.0%** | N=200 | Showing snap or raw passages to the final subagent hurts |
| `gap_hyde_nosnap` (fixed) | **62.5%** | N=200 | Anchoring-control improvement over fixed `gap_hyde` |
| fixed `rag_hyde` / `snap_hyde` (paired rerun) | **57.9%** | full N=1195 | Pre-leak-fix canonical run; clean reruns pending after the 2026-04-20 audit. The earlier `snap_hyde` **58.6%** run was also pre-fix |
| `subagent_rag` (1-gap) | **57.2%** | full N=1195 | Best non-HyDE combo full run |
| `llm_only` | **55.5%** | full N=1195 | Full small-model baseline |

Note: the `vectorless_*` label is historical shorthand. These are multi-turn LLM reasoning / parametric-knowledge modes, not real corpus search. `vectorless_hybrid` is the only one that still pools generated knowledge with vector retrieval.
The full N=1195 HyDE-family rows above are historical references until the clean reruns from `50835` and the follow-on full reruns finish.

### Historical cross-model reference
Older cross-model comparison tables are archived alongside the legacy dataset table.
→ see `docs/archive/research_legacy_blocks.md` for the full reference block.

### Key findings

1. **RAG value = f(LLM knowledge gap)** — only HousingQA shows validated lift
2. **Confidence gating works for random errors, not systematic bias** — BarExam: +2.5 over snap_hyde; HousingQA: -5.5 (model unanimously wrong, gating skips)
3. **Self-consistency (3-vote) is a good uncertainty signal** — Scout disagrees more (40%) than Llama (23%), correctly routing more to RAG
4. **Counterevidence retrieval consistently hurts** — devil -6, top-2 -3
5. **HyDE is the real driver in the strongest retrieval family** — fixed `rag_hyde` reached **66.0%** at N=200 and matches the paired full `snap_hyde` rerun at **57.9%**, implying that passage-form query generation, not the snap step itself, is doing the critical work.
6. **Snap helps plain RAG and parametric reasoning, not HyDE** — plain RAG and `vectorless_*` / parametric reasoning still show **+5pp** snap gains, but the HyDE family collapses to **0pp** once the prompt bug is fixed.
7. **Showing snap to the final agent consistently hurts** — `snap_hyde_report_snap` drops to **64.0%** vs `snap_hyde_report` **66.0%**, `subagent_rag_snap` drops to **63.0%**, and `subagent_rag_full` drops to **62.0%** vs `subagent_rag` **66.0%**.
8. **Full-scale Gemma is still retrieval-limited** — even the repaired HyDE tie at **57.9%** remains well below `golden_passage` **62.2%**, so retrieval quality is still the main bottleneck.

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
