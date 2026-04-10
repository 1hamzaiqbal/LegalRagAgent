# Research Program

Persistent research state for the LegalRagAgent project. Read this first in any new session.

This project started as a heavy agentic RAG pipeline that hurt performance. We stripped it down, systematically tested each component, and found that simpler adaptive strategies beat complex ones. The long-term goal is still a strong full agentic pipeline, but we're rebuilding toward it intentionally and atomically — testing each element's effectiveness and documenting what works about the research process itself.

## Current execution status
- No LegalRAG eval is actively running on the cluster right now.
- Phase 1 small-model baselines: **7/7 complete** (including HPC cluster runs).
- Best small model: **Gemma 4 E4B** at 55.5% llm_only / 58.6% snap_hyde (full N=1195).
- Embedding comparison (Phase 2 partial): **4/12 embedders tested** — gte-large, legal-bert, stella-400m, bge-m3.
- Remaining Phase 1 runs on OpenRouter: `or-nemotron`, `or-qwen35-9b` (deferred, lower priority than embedding work).

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
| HousingQA | snap_hyde | **56.0%** | +9 | Model unanimously wrong (Yes-bias), gating skips 90% |
| CaseHOLD | llm_only / conf_gated | **72.5%** | 0 | RAG pulls similar-but-wrong holdings |

### Full BarExam comparison (Llama 70B, N=200)

| Method | Accuracy | LLM calls/Q | Notes |
|--------|----------|-------------|-------|
| **ce_threshold** | **80.0%** | 2-3 | NEW BEST — skip RAG on low-CE |
| confidence_gated | 79.0% | 3-5 | Previous best |
| snap_hyde | 76.5% | 4 | Best always-on retrieval |
| decompose_structured | 76.0% | 3-5 | Best decomposition |
| decompose_natural | 75.0% | 3-5 | |
| decompose_rag | 71.5% | 4-8 | WORSE — synthesis loses signal |
| llm_only | 64.0% | 1 | Baseline |

### Cross-model comparison

| Model | Dataset | conf_gated | snap_hyde | llm_only |
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
5. **The real BarExam win is step-by-step prompting** (+12.5), not retrieval (net 0)
6. **decompose_rag is worse than simpler approaches** — synthesis step that merges sub-answers + evidence loses signal
7. **Optimal strategy is domain-dependent** — detect error mode (random → confidence_gated, systematic → snap_hyde)

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

#### 8. Embedding model comparison — IN PROGRESS (2026-04-09)
- **Phase 1 complete**: 4 embedders tested (gte-large, legal-bert, stella-400m, bge-m3) on rag_simple + snap_hyde, N=200, Gemma 4 E4B.
- **Finding**: Alternative embedders beat baseline on rag_simple (+4-5pp) but snap_hyde flattens differences (~60% for all non-baseline).
- **Phase 2 planned**: Test remaining models (stella-1.5b, gte-qwen2-1.5b, jina-v3, arctic-l-v2) and run full N=1195 for best embedder.
- Infrastructure ready: `slurm_build_embeds_local.sh` + `slurm_phase2_after_embeds.sh` pipeline.

#### 9. Domain-adaptive routing
- Automatically detect whether errors are random (→ confidence_gated) or systematic (→ snap_hyde). Requires characterizing the domain's error mode, which may need a calibration run.

---

## Active Experiment

### Completed experiments this session (2026-03-27)

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

### Last session (2026-04-03 audit)
- Re-audited repo state, logs, and hub notes against the March 31 meeting plan.
- Confirmed **5/7** Phase 1 full-set BarExam baselines are completed:
  - `or-qwen3-32b`: **61.42%** (`734/1195`) — best completed Phase 1 baseline
  - `groq-qwen`: **59.33%** (`709/1195`)
  - `or-qwen3-14b`: **57.74%** (`690/1195`)
  - `or-qwen3-8b`: **54.23%** (`648/1195`)
  - `groq-llama8b`: **52.97%** (`633/1195`)
- Confirmed the remaining Phase 1 baselines are still unrun: `or-nemotron`, `or-qwen35-9b`.
- Confirmed **Phases 2-6 have not started yet**.
- Fixed stale full-run detection in:
  - `eval/run_experiment_queue.py`
  - `eval/monitor.py`
  so the monitor now derives the true `full` question count from `eval_config.py` instead of using the stale `>=1900` threshold.
- Added `docs/cluster_workflow.md` to capture the next useful infra step: cluster-based local inference / full eval bring-up.
- Cleaned repo scratch artifacts by ignoring/removing transient session-export and inspection files.

### Meeting action items status (audited 2026-04-10)
| # | Item | Status |
|---|------|--------|
| 1 | Try smaller models | ✅ **Done** — Qwen3-8B, Gemma 4 E4B, plus 5 API models |
| 2 | Golden passage test | ✅ **Done** — Qwen3-8B 60.1%, Gemma4 62.2%, Qwen32B 66.7%, Gemma27B 65.5% |
| 3 | Case studies | ✅ Script built (`eval/case_studies.py`) |
| 4 | Token/cost analysis | ✅ Script built (`eval/token_analysis.py`) |
| 5 | RAG on small models | ✅ **Done** — rag_simple + snap_hyde on Gemma4 + Qwen3-8B |
| 6 | Devil RAG inversion | ⬜ Planned in Phase 4, not started |
| 7 | Self-consistency / confidence | ⬜ Planned in Phase 5/6, not started |
| 8 | Embedding model comparison | 🔶 **Phase 1 done** — 4/12 embedders tested, 8 remaining |
| 9 | MLEB benchmark | ❌ Not started |
| 10 | ENGR node local inference | ✅ **Done** — vLLM serving Gemma4+Qwen3-8B on A40/A6000 |
| 11 | SNAP-HyDE literature review | ❌ Not started |

### Latest meeting follow-up (2026-04-03)
- Priority 1: run a **same-scale non-Qwen comparison** against `or-qwen3-32b` on full BarQA. Best repo-available candidates: `or-gemma27b` and `or-mistral`.
- Priority 2: run **golden-passage vs RAG** comparisons on the strongest current small-model candidates to check whether retrieval quality is the bottleneck.
- Priority 3: test **one alternate embedding model** before branching into heavier retrieval frameworks.
- Priority 4: try **LazyGraphRAG first**, then decide whether heavier GraphRAG / RAGFlow exploration is justified.
- Langlin GPU inference access is still useful, but it is not the first blocker relative to the higher-signal model / golden / embedding comparisons above.

### Latest completed block (2026-04-05)
- `llm_only / or-gemma27b / full`: **57.99%** (`693/1195`) — valid lower-baseline non-Qwen control.
- `golden_passage / or-qwen3-32b / full`: **66.69%** vs qwen32 baseline **61.42%** → **+5.27pp**.
- `rag_simple / or-qwen3-32b / full`: **63.10%** vs qwen32 baseline **61.42%** → **+1.67pp**.
- `golden_passage / or-gemma27b / full`: **65.52%** vs gemma baseline **57.99%** → **+7.53pp**.
- `rag_simple / or-gemma27b / full`: **54.64%** vs gemma baseline **57.99%** → **-3.35pp**.
- Current read: retrieval quality is a real bottleneck. Golden helps both models much more than the current plain retriever, especially Gemma.

### Current active work (2026-04-10)

#### Completed HPC cluster runs (2026-04-07 through 2026-04-09)
- **Qwen3-8B full BarExam**: llm_only 52.1%, golden 60.1%, rag_simple 36.5% (corrupted)
- **Gemma 4 E4B full BarExam**: llm_only 55.5%, golden 62.2%, rag_simple 54.2%, snap_hyde 58.6%
- **Embedding comparison** (Gemma 4 E4B, N=200): 4 embedders x 2 modes = 8 eval runs

#### Embedding comparison results (Gemma 4 E4B, N=200, BarExam)

| Embedding | Params | Dim | rag_simple | snap_hyde |
|---|---|---|---|---|
| gte-large (baseline) | 434M | 1024 | 57.0% | **65.5%** |
| legal-bert | 110M | 768 | **62.0%** | 60.0% |
| stella-400m | 400M | 1024 | 61.0% | 60.0% |
| bge-m3 | 568M | 1024 | 61.0% | 60.0% |

**Key findings:**
1. All 3 alternative embedders beat baseline on rag_simple (+4-5pp).
2. legal-bert rag_simple (62.0%) nearly matches golden passage ceiling (62.2%).
3. snap_hyde flattens all differences to ~60% for non-baseline embedders, while baseline stays at 65.5%.
4. This asymmetry matters: rag_simple embeds the raw question (question→passage similarity), while snap_hyde embeds an LLM-generated hypothetical passage (passage→passage similarity). The baseline gte-large may be better at passage→passage matching, which is what snap_hyde needs.

#### Untested embedders (available in EMBEDDING_MODELS dict)
- `stella-1.5b`: 1.5B params, MTEB ~59.8 — previously failed due to transformers compat, may work with gemma4 venv
- `gte-qwen2-1.5b`: 1.5B instruct embedder, 1536d, 32k context
- `jina-v3`: 570M, task-specific LoRA adapters
- `arctic-l-v2`: 568M, retrieval-optimized, no trust_remote_code needed
- `nomic-v2-moe`: 475M MoE, 768d

### Next session: pick up here
1. Test remaining embedding models — prioritize `stella-1.5b` (strongest MTEB) and `jina-v3` (task-specific LoRA).
2. Investigate why snap_hyde flattens at 60% for non-baseline embedders — may be a cross-encoder bottleneck.
3. If a new embedder beats baseline snap_hyde (65.5%), run full N=1195 for it.
4. Consider running the best rag_simple embedder (legal-bert at 62.0%) at full N=1195 to validate.
5. Longer-term: LazyGraphRAG feasibility, confidence-gated integration into main.py.

### Blockers
- Cluster GPU availability (general-gpu partition, priority queue)
- stella-1.5b requires transformers>=5.5.0 (gemma4 venv has it, primary venv doesn't)
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
