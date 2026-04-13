# Research Program

Persistent research state for the LegalRagAgent project. Read this first in any new session.

This project started as a heavy agentic RAG pipeline that hurt performance. We stripped it down, systematically tested each component, and found that simpler adaptive strategies beat complex ones. The long-term goal is still a strong full agentic pipeline, but we're rebuilding toward it intentionally and atomically — testing each element's effectiveness and documenting what works about the research process itself.

## Current execution status
- No LegalRAG eval is actively running on the cluster right now.
- Core Phase 1 small-model baseline block is complete; lower-priority OpenRouter extras (`or-nemotron`, `or-qwen35-9b`) remain explicitly deferred.
- Best small-model full-set snapshot: **Gemma 4 E4B** at 55.5% `llm_only`, 62.2% `golden_passage`, and 57.9% `rag_snap_hyde` / snap_hyde-style full rerun (N=1195).
- Vectorless RAG sweep is complete: `vectorless_hybrid` **65.0%**, `vectorless_direct` **64.5%**, `vectorless_choice_map` **64.5%**, `vectorless_role` **63.5%**, `vectorless_elements` **61.0%** (Gemma 4 E4B, N=200).
- Embedding comparison is complete for supported builds: **7 embedders tested**; `jina-v3`, `arctic-l-v2`, and `nomic-v2-moe` all finished at 61.5% `rag_simple` / 64.5% `rag_snap_hyde`; `gte-qwen2-1.5b` and `stella-1.5b` failed to build.
- Gap architecture is fixed and revalidated: `gap_rag` **63.5%**, `gap_hyde` **62.0%** on Gemma 4 E4B (N=200), still below `snap_hyde` / vectorless.

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

#### ~~8. Embedding model comparison~~ — COMPLETED (2026-04-11)
- **7 embedders tested** across 3 modes (rag_simple, snap_hyde, snap_hyde_aligned), N=200 each.
- **Key finding**: Cross-encoder reranking dominates — all 7 non-gte-large embedders converge to exactly 65.0% with question-based reranking. Embedding model choice barely matters.
- **Wave 1** (gte-large, legal-bert, stella-400m, bge-m3): snap_hyde 60% for non-gte, gte-large 65.5%.
- **Wave 2** (jina-v3, arctic-l-v2, nomic-v2-moe): snap_hyde 64.5% for all three — closer to gte-large.
- **Failed builds**: gte-qwen2-1.5b, stella-1.5b (transformers rope_theta compat).

#### 9. Domain-adaptive routing
- Automatically detect whether errors are random (→ confidence_gated) or systematic (→ snap_hyde). Requires characterizing the domain's error mode, which may need a calibration run.

#### ~~10. Gap-informed retrieval~~ — COMPLETED (2026-04-13)
- **Architecture tested**: `SNAP → ANALYZE GAPS → SUBAGENT RETRIEVAL (per gap) → FINAL REASONING`
- **Variants tested**: `gap_rag`, `gap_hyde`, `gap_hyde_ev`, `gap_hyde_nosnap`, `gap_hyde_flat`, plus follow-up anchoring ablations.
- **Result**: fixed `gap_rag` reached **63.5%** and fixed `gap_hyde` reached **62.0%** on Gemma 4 E4B (N=200), but both still underperform `snap_hyde` (**65.5%**) and `vectorless_hybrid` (**65.0%**).
- **Keep/discard**: keep as an analyzed ablation family, discard as the main direction for now.

---

## Plan Snapshot (2026-04-10)

### Phase 1: Alignment Testing
Run all existing retrieval modes on the **same** N=200, seed=42, BarExam, Gemma 4 E4B for a clean comparison. Many of these have been tested before but on different models or sample sizes — need them all on the same setup.

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
Completed; the gap family topped out at **63.5%** (`gap_rag`) after the prompt/schema fix and did not beat `snap_hyde` or vectorless.

### Phase 3: Vectorless RAG (completed 2026-04-13)
Completed initial sweep; `vectorless_hybrid` reached **65.0%** and `vectorless_direct` reached **64.5%** on Gemma 4 E4B (N=200).

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

### Meeting action items status (audited 2026-04-13)
| # | Item | Status |
|---|------|--------|
| 1 | Try smaller models | ✅ **Done** — Qwen3-8B, Gemma 4 E4B, plus 5 API models |
| 2 | Golden passage test | ✅ **Done** — Qwen3-8B 60.1%, Gemma4 62.2%, Qwen32B 66.7%, Gemma27B 65.5% |
| 3 | Case studies | ✅ Script built (`eval/case_studies.py`) |
| 4 | Token/cost analysis | ✅ Script built (`eval/token_analysis.py`) |
| 5 | RAG on small models | ✅ **Done** — rag_simple + snap_hyde on Gemma4 + Qwen3-8B |
| 6 | Devil RAG inversion | ⬜ Planned in Phase 4, not started |
| 7 | Self-consistency / confidence | ⬜ Planned in Phase 5/6, not started |
| 8 | Embedding model comparison | ✅ **Done** — 7 supported embedders tested across `rag_simple`, `rag_snap_hyde`, `snap_hyde_aligned`; 2 failed builds documented |
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

### Historical active work snapshot (2026-04-10)

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

### Session summary (2026-04-13)

**VECTORLESS RAG COMPLETE — competitive with snap_hyde:**
- `vectorless_hybrid`: **65.0%** (18% changed, +7 net) — LLM knowledge + sparse RAG
- `vectorless_direct`: **64.5%** (19% changed, +6 net) — pure LLM knowledge, NO vector store
- `vectorless_choice_map`: **64.5%** — rule + distractor mapping
- `vectorless_role`: **63.5%** (7% changed, +4 net) — barprep tutor
- `vectorless_elements`: **61.0%** — structured legal elements

**GAP ARCHITECTURE FIXED AND VALIDATED:**
- `gap_rag FIXED`: **63.5%** (2% changed, +4 net) — evidence reaches model but barely changes answers
- `gap_hyde FIXED`: **62.0%** (0.5% changed, +1 net) — same anchoring issue

**CRITICAL FINDING — Anchoring hypothesis:**
- Modes that SHOW snap in final call: 0.5-2% answer changes (gap modes, snap_rag)
- Modes that HIDE snap from final call: 7-27% answer changes (snap_hyde, vectorless, snap_rag_nosnap)
- Follow-up `gap_hyde_nosnap` landed at **61.5%**; `gap_rag_nosnap` and `gap_vectorless` do not have completed runs logged in `logs/experiments.jsonl`

**11-char HyDE root cause FOUND AND FIXED:**
- Prompt schema mismatch: snap_hyde system prompt expects "Student's Answer" but gap code sent "Evidence Gap"
- Gemma merges system+user → schema mismatch causes short-circuit
- Fix: pass snap_answer through, use matching schema with gap focus injected inside

**Completed since the last update:**
- `rag_snap_hyde` full N=1195 finished at **57.9%** (`692/1195`) on Gemma 4 E4B
- No active cluster evals are being tracked here right now

---

### Previous session (2026-04-10)

**Phase 1 alignment complete.** All major retrieval modes tested on Gemma 4 E4B N=200 BarExam:
- snap_hyde (65.5%) is the clear winner — HyDE for both retrieval and reranking
- ce_threshold (64.0%), rag_arbitration (63.0%) are solid alternatives
- Gap architecture (61.5%) adds no value — gap analysis doesn't improve over simpler snap+retrieve

**Phase 2 gap architecture complete.** Implemented and tested:
- gap_hyde, gap_rag, gap_hyde_ev, snap_rag, snap_rag_nosnap
- Key finding: gap analysis (0-3 gaps → per-gap retrieval) performs at 61.5%, same as simpler snap_rag (62.0%)
- Snap context in final call is neutral (+0.5pp)
- HyDE per-gap is broken on Gemma (11-char outputs from `_system_prompt(config, "hyde")` with gap-formatted input)

**Embedding comparison (7/9 embedders tested):**
- Wave 1: gte-large, legal-bert, stella-400m, bge-m3
- Wave 2: jina-v3, arctic-l-v2, nomic-v2-moe (built, eval in progress)
- Failed: gte-qwen2-1.5b, stella-1.5b (transformers rope_theta compat)
- snap_hyde_aligned results: legal-bert 65.0%, all others 62.5%

**Key ablation findings:**
- Snap reasoning adds +5pp (snap_rag 62% vs rag_simple 57%)
- HyDE passage adds +3.5pp retrieval quality (snap_hyde 65.5% vs snap_rag 62%)
- HyDE reranking adds +3pp (snap_hyde 65.5% vs snap_hyde_aligned 62.5%)
- Gap analysis adds -0.5pp (gap_rag 61.5% vs snap_rag 62%)
- Snap visible in final call adds +0.5pp (neutral)

**Known issues:**
- 11-char HyDE bug: `_system_prompt(config, "hyde")` with gap-formatted user input produces truncated outputs on Gemma
- Gap analysis prompt sensitivity: old prompt → 0% NONE (always finds gaps), new prompt → 97% NONE (almost never finds gaps)
- N=200 variance: snap_hyde_aligned ranged 62.5%–67.5% across runs

### Next session at that time
1. Wave 2 embedding eval later completed — jina-v3, arctic-l-v2, and nomic-v2-moe all landed at 61.5% `rag_simple` / 64.5% `rag_snap_hyde`
2. Record all Phase 1 + Phase 2 experiments in EXPERIMENTS.md
3. Consider running snap_hyde at full N=1195 with legal-bert embedder (65.0% aligned result suggests potential)
4. 11-char HyDE bug was later fixed; if revisiting gap architecture, keep using the snap_hyde-style prompt schema
5. Longer-term: vectorless RAG (LLM-as-retriever), LazyGraphRAG, confidence-gated integration into main.py

### Blockers
- Cluster GPU availability (general-gpu partition, priority queue)
- 11-char HyDE bug blocks gap_hyde variants from producing valid results
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
