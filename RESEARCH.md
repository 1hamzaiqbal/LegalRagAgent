# Research Program

Persistent research state for the LegalRagAgent project. Read this first in any new session.

This project started as a heavy agentic RAG pipeline that hurt performance. We stripped it down, systematically tested each component, and found that simpler adaptive strategies beat complex ones. The long-term goal is still a strong full agentic pipeline, but we're rebuilding toward it intentionally and atomically — testing each element's effectiveness and documenting what works about the research process itself.

## Current execution status
- Active cluster work on April 17 is the combo-mode follow-up block `48393` (`snap_hyde_report`, `snap_hyde_report_snap`, `subagent_rag_snap`, `subagent_rag_full`) plus the fixed full `rag_hyde` rerun submission `48555`; since the April 13 handoff, jobs `44394`, `44395`, `45350`, and `45735` all completed successfully. The case-summary build `44371` is done, and the entity-graph rebuild `44520` was last noted at 74%.
- Core Phase 1 small-model baseline block is complete; lower-priority OpenRouter extras (`or-nemotron`, `or-qwen35-9b`) remain explicitly deferred.
- Best Gemma 4 E4B N=200 result is now a three-way tie at **66.0%**: `subagent_rag`, fixed `rag_hyde`, and `snap_hyde_report`. Full-set Gemma 4 E4B leaderboard (N=1195): `golden_passage` **62.2%**, `snap_hyde` **58.6%**, `subagent_rag` (1-gap) **57.2%**, `subagent_rag` **56.9%**, `ce_threshold` **55.9%**, `gap_rag_nosnap` **55.9%**, `llm_only` **55.5%**, `rag_hyde` **54.3%**, `rag_simple` **54.2%**, `entity_search` **53.2%**.
- April 17 validation: fixed `rag_hyde` reached **66.0%** at N=200, validating the HyDE prompt fix; `snap_hyde_report` also reached **66.0%** at N=200 in block `48393`, and the fixed full rerun was submitted as job `48555`.
- Subagent follow-up sweep is complete: `subagent_rag` **66.0%**, `subagent_hybrid` **63.5%**, `subagent_rag_evidence` **61.0%** (Gemma 4 E4B, N=200).
- "Vectorless" baseline sweep is complete: `vectorless_hybrid` **65.0%**, `vectorless_direct` **64.5%**, `vectorless_choice_map` **64.5%**, `vectorless_role` **63.5%**, `vectorless_elements` **61.0%** (Gemma 4 E4B, N=200). Naming caveat: these modes are multi-turn LLM reasoning / parametric-knowledge exploitation, not real corpus search, so the full N=1195 vectorless jobs were canceled.
- Embedding comparison is complete for supported builds: **7 embedders tested**; `jina-v3`, `arctic-l-v2`, and `nomic-v2-moe` all finished at 61.5% `rag_simple` / 64.5% `rag_snap_hyde`; `gte-qwen2-1.5b` and `stella-1.5b` failed to build.
- Gap-family reruns are complete: `gap_rag_nosnap` **64.5%**, fixed `gap_rag` **63.5%**, fixed `gap_hyde_nosnap` **62.5%**, fixed `gap_hyde` **62.0%**, and `gap_vectorless` **61.5%** on Gemma 4 E4B (N=200). Anchoring is real, but the gap family still trails the best simpler baselines.
- `logs/experiments.jsonl` now contains **189** completed experiment records.

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

### By dataset (Llama 70B, N=200, seed=42)

| Dataset | Best Mode | Accuracy | vs llm_only | Key insight |
|---------|-----------|----------|-------------|-------------|
| BarExam | **ce_threshold** | **80.0%** | +16 | Skip RAG when CE<4.0, use snap answer instead |
| HousingQA | `rag_snap_hyde` | **56.0%** | +9 | Model unanimously wrong (Yes-bias), gating skips 90% |
| CaseHOLD | `llm_only` / `confidence_gated` | **72.5%** | 0 | RAG pulls similar-but-wrong holdings |

### Gemma 4 E4B + historical vectorless / parametric-reasoning snapshot (BarExam)

| Method | Accuracy | Scale | Notes |
|--------|----------|-------|-------|
| `subagent_rag` | **66.0%** | N=200 | Best current Gemma 4 E4B result |
| `rag_snap_hyde` | **65.5%** | N=200 | Best small-model retrieval baseline |
| `gap_rag_nosnap` | **64.5%** | N=200 | Best gap-family no-snap control |
| `vectorless_hybrid` | **65.0%** | N=200 | Best vectorless result |
| `vectorless_direct` / `vectorless_choice_map` | **64.5%** | N=200 | No vector store required |
| `subagent_hybrid` / `vectorless_role` | **63.5%** | N=200 | Second-tier follow-ups |
| `gap_hyde_nosnap` (fixed) | **62.5%** | N=200 | Anchoring-control improvement over fixed `gap_hyde` |
| `gap_rag` (fixed) | **63.5%** | N=200 | Improved after the prompt/schema fix, still below snap/vectorless |
| `rag_snap_hyde` | **58.6%** | full N=1195 | Best clean full run (`700/1195`); later rerun landed at 57.9% (`692/1195`) |
| `llm_only` | **55.5%** | full N=1195 | Full small-model baseline |

Note: the `vectorless_*` label is historical shorthand. These are multi-turn LLM reasoning / parametric-knowledge modes, not real corpus search. `vectorless_hybrid` is the only one that still pools generated knowledge with vector retrieval.

### Cross-model comparison

| Model | Dataset | `confidence_gated` | `rag_snap_hyde` | `llm_only` |
|-------|---------|------------|-----------|----------|
| Llama 70B | BarExam | **79.0%** | 76.5% | 64% |
| Scout 17B | BarExam | **71.5%** | 71.0%* | 69% |
| Llama 70B | HousingQA | 50.5% | **56.0%** | 47% |
| Scout 17B | HousingQA | 53.5% | **54.0%*** | 50% |

*N=100, not directly comparable

### Key findings

1. **RAG value = f(LLM knowledge gap)** — only HousingQA shows validated lift
2. **Confidence gating works for random errors, not systematic bias** — BarExam: +2.5 over snap_hyde; HousingQA: -5.5 (model unanimously wrong, gating skips)
3. **Self-consistency (3-vote) is a good uncertainty signal** — Scout disagrees more (40%) than Llama (23%), correctly routing more to RAG
4. **Counterevidence retrieval consistently hurts** — devil -6, top-2 -3
5. **Subagent reports are the strongest current small-model strategy** — `subagent_rag` reached 66.0%, beating `rag_snap_hyde` by 0.5pp
6. **"Vectorless" is competitive, but the name is misleading** — `vectorless_hybrid` (65.0%) nearly matches `rag_snap_hyde` (65.5%), but these runs test multi-turn parametric reasoning, not corpus search
7. **Gap variants improved after the fix, but not enough** — `gap_rag_nosnap` reached 64.5% and fixed `gap_hyde_nosnap` reached 62.5%, confirming anchoring without overtaking the best simpler baselines
8. **Full-scale Gemma is still retrieval-limited** — best full `snap_hyde` is 58.6% vs `golden_passage` 62.2%, and even the improved `subagent_rag` 1-gap rerun only reached 57.2%

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

---

## Plan Snapshot (2026-04-17)

### Phase 1: Alignment Testing
Completed. All major retrieval modes have now been run on the same N=200, seed=42, BarExam, Gemma 4 E4B setup for a clean comparison.

| Mode | LLM Calls | Accuracy | What It Tests |
|---|---|---|---|
| `snap_hyde_aligned` (gte-large run2) | 3 | **67.5%** | HyDE retrieval + question reranking (high variance) |
| `rag_snap_hyde` (gte-large) | 3 | **65.5%** | HyDE passage retrieval + HyDE reranking |
| `snap_hyde_aligned` (legal-bert) | 3 | 65.0% | Domain embedding + question reranking |
| `ce_threshold` | 2-3 | 64.0% | CE gating on snap_hyde |
| `rag_arbitration` | 3 | 63.0% | Snap → review with retrieved evidence |
| `snap_hyde_aligned` (gte-large run1) | 3 | 62.5% | HyDE retrieval + question reranking |
| `golden_passage` | 1 | 62.2% | Ceiling (perfect retrieval, N=1195) |
| `snap_rag` | 2 | 62.0% | Snap + plain retrieval + snap in final |
| `snap_rag_nosnap` | 2 | 61.5% | Snap + plain retrieval, no snap in final (control) |
| `gap_rag` | 3-6 | 61.5% | Gap analysis + sub-question retrieval |
| `gap_hyde` | 4-8 | 61.5% | Gap analysis + HyDE retrieval (broken 11-char HyDE) |
| `gap_hyde_ev` | 4-8 | 61.0% | Gap + evidence only in final (broken HyDE) |
| `rag_rewrite` | 3 | 59.5% | Multi-query rewritten retrieval |
| `rag_simple` (gte-large) | 1 | 57.0% | Raw question retrieval baseline |
| `llm_only` | 1 | 55.5% | No retrieval (N=1195) |

### Phase 2: Gap-Informed Retrieval (completed 2026-04-13)
Completed; the gap family topped out at **64.5%** (`gap_rag_nosnap`) after the anchoring controls, but still did not beat `rag_snap_hyde` or `subagent_rag`.

### Phase 3: Historical vectorless / parametric reasoning (completed 2026-04-13)
Completed initial sweep; `vectorless_hybrid` reached **65.0%** and `vectorless_direct` reached **64.5%** on Gemma 4 E4B (N=200). Naming caveat: these are multi-turn parametric-knowledge baselines, not real corpus-search baselines, so the full N=1195 vectorless jobs were canceled.

### Phase 4: April 17 handoff
1. Completed since the April 13 snapshot: case-summary build `44371`, snap ablations `44394`, cross-dataset block `44395`, full `rag_hyde` + `ce_threshold` block `45350`, full `gap_rag_nosnap` + `subagent_rag` 1-gap block `45735`, and the April 17 N=200 HyDE-fix validation.
2. Running now: combo-mode N=200 block `48393` (`snap_hyde_report`, `snap_hyde_report_snap`, `subagent_rag_snap`, `subagent_rag_full`).
3. Submitted now: fixed full `rag_hyde` rerun `48555` after the April 17 validation recovered `rag_hyde` to **66.0%** at N=200.
4. Still open: define and run a real non-vector corpus-search control (`vectorless_keyword`, BM25, or structured index navigation) instead of the misleading parametric "vectorless" label.
5. Still open: integrate `confidence_gated` or another validated routing policy into `main.py` once the eval-side direction is stable.

---

## Historical Reference Block

### Completed experiments in the 2026-03-27 CE-threshold block

| # | Experiment | Llama 70B | Scout 17B | Verdict |
|---|---|---|---|---|
| 1 | **CE threshold** (Tier 1 #1) | **80.0%** | 71.5% | **KEEP** — new BarExam best |
| 2 | CaseHOLD CE threshold | 71.0% | — | NEUTRAL |
| 3 | Combined conf+CE | 76.5% | — | **DISCARD** — destructive interference |
| 4 | Aspect queries (Tier 1 #2) | 76.0% | — | **DISCARD** — offline gains don't translate |
| 5 | CE threshold k=3 | 79.0% | — | **DISCARD** |
| 6 | Pipeline integration (HyDE+CE) | 76.0% | — | DIAGNOSTIC — planner/synth cost -4pts |
| 7 | Self-verification | 73.0% | 58.5% | **DISCARD** — second-guessing destroys accuracy |
| 8 | Double-snap | 74.0% | — | **DISCARD** |
| 9 | Snap-debate | 72.0% | 64.0% | **DISCARD** — adversarial review worst of all |
| 10 | GPT 5.4 Mini llm_only | — | — | 74.0% (N=100) — strong baseline |

### Key learnings (this session)
1. **CE threshold (80.0%)** is the new best — skip RAG when evidence quality is low
2. **Self-correction is destructive** — second-guessing without new info hurts both models (Llama -3pts, Scout -10pts)
3. **Pipeline overhead costs 4pts** — planner decomposition + synthesizer recombination = lossy pipeline
4. **Components interact** — conf+CE creates dead zones, validated blocks don't compose additively
5. **Build from atoms, not from architecture** — proven: ce_threshold (atomic) > full_pipeline (architectural)
6. **GPT 5.4 Mini baseline is 74%** — higher than Llama llm_only (64%), different model family for generalization testing

**Next: cross-model ce_threshold validation (GPT 5.4 Mini, others) when resources free up.**

---

## Session Handoff

### Historical audit reference (2026-04-03)
- This was the repo-state checkpoint that confirmed the initial 5/7 full-set baselines and identified `or-nemotron` / `or-qwen35-9b` as the remaining deferred OpenRouter runs.
- It also fixed stale full-run detection in `eval/run_experiment_queue.py` and `eval/monitor.py`, so `full` now resolves from `eval_config.py` instead of the old `>=1900` heuristic.
- Keep it as the point-in-time audit marker; the April 17 timeline below is the current handoff state.

### Meeting action items status (audited 2026-04-17)
| # | Item | Status |
|---|------|--------|
| 1 | Try smaller models | ✅ **Done** — Qwen3-8B, Gemma 4 E4B, plus 5 API models |
| 2 | Golden passage test | ✅ **Done** — Qwen3-8B 60.1%, Gemma4 62.2%, Qwen32B 66.7%, Gemma27B 65.5% |
| 3 | Case studies | ✅ Script built (`eval/case_studies.py`) |
| 4 | Token/cost analysis | ✅ Script built (`eval/token_analysis.py`) |
| 5 | RAG on small models | ✅ **Done** — rag_simple + snap_hyde on Gemma4 + Qwen3-8B |
| 6 | Devil RAG inversion | ⬜ Planned in Phase 4, not started |
| 7 | Self-consistency / confidence | ✅ **Done** — `confidence_gated` validated on BarExam, HousingQA, and CaseHOLD |
| 8 | Embedding model comparison | ✅ **Done** — 7 supported embedders tested across `rag_simple`, `rag_snap_hyde`, `snap_hyde_aligned`; 2 failed builds documented |
| 9 | MLEB benchmark | ❌ Not started |
| 10 | ENGR node local inference | ✅ **Done** — vLLM serving Gemma4+Qwen3-8B on A40/A6000 |
| 11 | SNAP-HyDE literature review | ❌ Not started |

### Recent verified timeline
- 2026-04-03 audit: confirmed 5/7 full-set BarExam baselines, fixed stale full-run detection in `eval/run_experiment_queue.py` and `eval/monitor.py`, and recorded the cluster bring-up workflow.
- 2026-04-05 full-set comparisons: `or-gemma27b` baseline landed at **57.99%**; `golden_passage` materially beat plain retrieval on both `or-qwen3-32b` and `or-gemma27b`, confirming retrieval quality as the main bottleneck.
- 2026-04-07 through 2026-04-11 HPC block: full Qwen3-8B and Gemma 4 E4B runs completed; the focused 7-embedder sweep completed with 2 documented build failures.
- 2026-04-14 block: snap/no-snap ablations and cross-dataset follow-up both completed; full `subagent_rag` landed at **56.9%** (`680/1195`); and the misnamed full-vectorless jobs were formally canceled.
- 2026-04-15 block: `entity_search` full landed at **53.2%** (`636/1195`); `snap_entity_informed` reached **59.5%**; `subagent_hyde` reached **62.5%**; and the fixed full `rag_hyde` rerun completed at **54.3%** (`649/1195`).
- 2026-04-16 block: full `ce_threshold` landed at **55.9%** (`668/1195`), full `gap_rag_nosnap` landed at **55.9%** (`668/1195`), and the full `subagent_rag` 1-gap rerun improved to **57.2%** (`684/1195`). `logs/experiments.jsonl` reached **189** entries.
- 2026-04-17 block: fixed `rag_hyde` validated at **66.0%** (`132/200`), `snap_hyde_report` also reached **66.0%** (`132/200`), combo modes `snap_hyde_report`, `snap_hyde_report_snap`, `subagent_rag_snap`, and `subagent_rag_full` were launched in job `48393`, and the repaired full `rag_hyde` rerun was submitted as job `48555`.

### Current handoff
- Verified complete: Phase 1 small-model baselines, Gemma/Qwen HPC full runs, focused embedding sweep, historical vectorless baseline sweep, anchoring controls, subagent follow-up sweep, fixed gap reruns, snap ablations, cross-dataset follow-up, the April 15-16 full rerun block, and the April 17 HyDE-fix validation. `logs/experiments.jsonl` remains at **189** recorded runs.
- Verified but still lower-priority historical findings: `golden_passage` consistently outperforms current plain retrieval on the strongest full-set models; `confidence_gated` remains the best Llama 70B adaptive baseline after `ce_threshold`.
- Full N=1195 leaderboard on April 17: `golden_passage` **62.2%**, `snap_hyde` **58.6%**, `subagent_rag` (1-gap) **57.2%**, `subagent_rag` **56.9%**, `ce_threshold` **55.9%**, `gap_rag_nosnap` **55.9%**, `llm_only` **55.5%**, `rag_hyde` **54.3%**, `rag_simple` **54.2%**, `entity_search` **53.2%**.
- Running now: combo-mode job `48393` (`snap_hyde_report`, `snap_hyde_report_snap`, `subagent_rag_snap`, `subagent_rag_full`).
- Submitted / newly validated: fixed `rag_hyde` reached **66.0%** (`132/200`) at N=200, `snap_hyde_report` matched **66.0%** (`132/200`), and the repaired full `rag_hyde` rerun is job `48555`.
- Still pending: real corpus-search follow-ups that use the finished case-summary layer / rebuilt entity graph, `vectorless_keyword`, plus deferred OpenRouter baselines `or-nemotron` and `or-qwen35-9b`.
- Most likely next high-signal work: analyze `48393` and `48555` once they finish, then test real corpus-search controls or integrate `confidence_gated` into `main.py`.

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
