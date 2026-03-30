# Research Program

Persistent research state for the LegalRagAgent project. Read this first in any new session.

This project started as a heavy agentic RAG pipeline that hurt performance. We stripped it down, systematically tested each component, and found that simpler adaptive strategies beat complex ones. Now we're building back up intentionally — testing each element's effectiveness and documenting what works about the research process itself.

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

#### 1. Score thresholding
- **Hypothesis**: Dropping retrieved passages below a cross-encoder score threshold and falling back to llm_only will improve BarExam accuracy by avoiding cases where bad evidence misleads the model.
- **Change**: Add CE score threshold to snap_hyde. If all passages below threshold, discard and use snap answer directly.
- **Success criteria**: BarExam accuracy > 76.5% (current snap_hyde). Should not hurt HousingQA.
- **Keep/discard**: Keep if BarExam improves without HousingQA regression. Discard if neutral or negative.
- **Prior evidence**: Error analysis showed CE scores 3.70 (wrong) vs 4.14 (correct) — small but consistent signal. Gold passage retrieval only 10% in decompose_rag.

#### 2. Aspect-based query rewrite (prompt-only)
- **Hypothesis**: Rewriting queries as rule/exception/application aspects instead of synonym variants will improve retrieval quality and downstream accuracy.
- **Change**: Modify `skills/query_rewriter.md` prompt to generate aspect-based alternatives. No architecture change.
- **Success criteria**: Improvement on any dataset without regression on others.
- **Keep/discard**: Keep if any dataset improves. Discard if neutral across all.
- **Prior evidence**: Offline retrieval test showed CE scores 6.0 (aspect) vs 3.0 (synonym) — 2x improvement. Never tested end-to-end.

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

#### 8. Embedding model upgrade
- gte-Qwen2 family scores higher on MTEB. Would require re-embedding all corpora (~8+ hours GPU time). Defer until other improvements plateau.

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

### Last session (2026-03-27)
- **10 experiments** total: 1 keep, 8 discard, 1 diagnostic
- Set up research framework, cleaned code, rewrote README
- New best: ce_threshold 80.0% (Llama BarExam)
- Proved pipeline overhead costs -4pts (planner/synthesizer are net-negative)
- Proved self-correction/debate are destructive (-3 to -10pts across models)
- GPT 5.4 Mini baseline: 74% llm_only (strong, different model family)
- Cerebras API returns empty responses (not usable)

### Next session: pick up here
1. Check resources (`free -h`) — rl-on-rl monitor eating 11GB, RAG runs OOM
2. **Priority: cross-model ce_threshold** — GPT 5.4 Mini (74% baseline, will RAG help or hurt a strong model?)
3. **Priority: restructure main.py** — strip planner/synthesizer, make ce_threshold the base pipeline
4. Tier 2: MC choice-aware (prompt change, no architecture), state filtering for HousingQA
5. Consider: `groq-qwen` (Qwen3-32B) and `groq-kimi` (Kimi K2) as additional model diversity

### Blockers
- ~~rl-on-rl monitor now using 11GB RAM~~ — RESOLVED (KL training completed 2026-03-28, all processes exited). RAM should be free.
- Groq rate limits: may still be near daily limits depending on when session restarts
- Cerebras API broken (empty responses, as of 2026-03-27)

---

## File Pointers

| File | Purpose |
|------|---------|
| `RESEARCH.md` (this file) | Research state, experiment queue, session handoff |
| `EXPERIMENTS.md` | Full experiment log (hypothesis → result → verdict) |
| `CLAUDE.md` | Operational source of truth (how to run, environment notes) |
| `logs/experiments.jsonl` | Machine-readable results (one JSON record per run) |
| `ideas/actionable_ideas.md` | Idea backlog archive (active queue is here) |
| `memory/MEMORY.md` | Cross-session memory index |
| `memory/research_principles.md` | Autoresearch + harness design principles (persistent) |
