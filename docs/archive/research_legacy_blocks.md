### By dataset (Llama 70B, N=200, seed=42)

| Dataset | Best Mode | Accuracy | vs llm_only | Key insight |
|---------|-----------|----------|-------------|-------------|
| BarExam | **ce_threshold** | **80.0%** | +16 | Skip RAG when CE<4.0, use snap answer instead |
| HousingQA | `rag_snap_hyde` | **56.0%** | +9 | Model unanimously wrong (Yes-bias), gating skips 90% |
| CaseHOLD | `llm_only` / `confidence_gated` | **72.5%** | 0 | RAG pulls similar-but-wrong holdings |

### Cross-model comparison

| Model | Dataset | `confidence_gated` | `rag_snap_hyde` | `llm_only` |
|-------|---------|------------|-----------|----------|
| Llama 70B | BarExam | **79.0%** | 76.5% | 64% |
| Scout 17B | BarExam | **71.5%** | 71.0%* | 69% |
| Llama 70B | HousingQA | 50.5% | **56.0%** | 47% |
| Scout 17B | HousingQA | 53.5% | **54.0%*** | 50% |

*N=100, not directly comparable

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

### Phase 4: April 17 closure
1. Completed since the April 13 snapshot: case-summary build `44371`, snap ablations `44394`, cross-dataset block `44395`, full `rag_hyde` + `ce_threshold` block `45350`, full `gap_rag_nosnap` + `subagent_rag` 1-gap block `45735`, combo block `48393`, and fixed full `rag_hyde` rerun `48555`.
2. Combo-mode results: `snap_hyde_report` **66.0%**, `snap_hyde_report_snap` **64.0%**, `subagent_rag_snap` **63.0%**, `subagent_rag_full` **62.0%**.
3. Full fixed `rag_hyde` rerun `48555`: **57.9%** (`692/1195`), tying the paired full `snap_hyde` rerun and collapsing the HyDE snap lift to **0pp**.
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
