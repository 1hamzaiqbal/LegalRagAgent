# Experiment Log

Running record of hypotheses, experiments, and results. Add new entries at the top.

## Format

```
### YYYY-MM-DD — Short description
**Hypothesis:** What we expected to happen and why
**Change:** What was modified
**Config:** model, N queries, any relevant settings
**Result:** accuracy, key metrics
**Verdict:** CONFIRMED / REFUTED / MIXED — with evidence
**Commit:** hash
```

## Established baselines (N=100, seed=42)

| Mode | Scout 17B | Llama 70B | GPT 5.4-nano |
|---|---|---|---|
| llm_only | 64% | 64% | 57%* |
| golden_passage | 73% (+9) | 81% (+17) | — |
| golden_arb_conservative | 71% (+7) | 80% (+16) | — |
| rag_simple (debiased) | 69% (+5)† | 73% (+9) | 59% (+2) |
| rag_rewrite (debiased) | 69% (+5)† | — | — |
| rag_arbitration | 68% (+4)† | — | — |

*N=28 curated only. †N=200 validation showed llm_only=69%, rag_simple=68% — the +5pt lift at N=100 was sampling noise.

**N=200 validated baselines (Scout):** llm_only=69%, rag_simple=68%. Use N=200+ for final decisions.

### HousingQA baselines

| Mode | Llama 70B (N=100) | Llama 70B (N=200) | Scout 17B (N=100) |
|---|---|---|---|
| llm_only | 44% | 47% | 50% |
| rag_snap_hyde | **61% (+17)** | **56% (+9)** | **54% (+4)** |

Note: HousingQA is Yes/No format, 65% No / 35% Yes class imbalance. LLM has massive Yes-bias (80% pred Yes). Retrieval against housing_statutes (1.8M docs, dense-only) dramatically reduces this bias. N=200 validation shows +9pt lift (down from +17 at N=100, same pattern as BarExam).

### Cross-dataset generalizability (Llama 70B, seed=42)

| Dataset | Format | Corpus size | llm_only | rag_snap_hyde | RAG lift | N |
|---|---|---|---|---|---|---|
| BarExam QA | 4-way MC | 686K | 64% | 76.5% | +12.5* | 200 |
| HousingQA | Yes/No | 1.8M | 47% | 56% | **+9** | 200 |
| CaseHOLD | 5-way MC | 50K | 72.5% | 71.0% | **-1.5** | 200 |
| Legal-RAG-QA | Open-ended | 190 | 100%† | 99.3%† | -0.7 | 138 |
| Australian Legal QA | Open-ended | 2.1K | 100%† | 100%† | 0 | 200 |

*BarExam N=200 net RAG lift is ~zero (fixed 15, hurt 15). †LLM-as-judge scoring (too lenient — treats most answers as correct).

**Key finding:** RAG helps only when the LLM genuinely lacks domain knowledge (HousingQA: obscure state statutes). On well-known domains (BarExam, criminal law, Australian QA) or tasks requiring citation matching (CaseHOLD), retrieval is neutral-to-harmful.

---

### 2026-04-10 — Comprehensive Gemma 4 E4B retrieval ablation

**Hypothesis:** Systematic ablation of retrieval pipeline components (snap reasoning, HyDE generation, gap analysis, reranking alignment, final-call context) will identify which components actually contribute to accuracy.

**Change:** Implemented and tested 10+ modes on identical conditions: Gemma 4 E4B, N=200, seed=42, BarExam, gte-large embedding.

**Config:** Gemma 4 E4B (cluster-vllm), N=200, seed=42, BarExam.

**Results:**

| Mode | Acc | Calls | What It Tests |
|---|---|---|---|
| snap_hyde | **65.5%** | 3 | HyDE retrieval + HyDE reranking |
| ce_threshold | 64.0% | 2-3 | CE gating on snap_hyde |
| rag_arbitration | 63.0% | 3 | Snap → review with evidence |
| snap_rag | 62.0% | 2 | Snap + plain retrieval + snap in final |
| snap_rag_nosnap | 61.5% | 2 | Snap + plain retrieval, no snap in final |
| gap_rag | 61.5% | 3-6 | Gap analysis + sub-question retrieval |
| gap_hyde | 61.5% | 4-8 | Gap analysis + HyDE (broken 11-char outputs) |
| gap_hyde_ev | 61.0% | 4-8 | Gap + evidence only in final |
| rag_rewrite | 59.5% | 3 | Multi-query rewritten retrieval |
| rag_simple | 57.0% | 1 | Raw question retrieval |
| llm_only | 55.5% | 1 | No retrieval |

**Component contribution analysis:**

| Component | How Measured | Contribution |
|---|---|---|
| Snap reasoning | snap_rag (62%) vs rag_simple (57%) | **+5pp** |
| HyDE for retrieval | snap_hyde (65.5%) vs snap_rag (62%) | **+3.5pp** |
| HyDE for reranking | snap_hyde (65.5%) vs snap_hyde_aligned (62.5%) | **+3pp** |
| Snap in final call | snap_rag (62%) vs snap_rag_nosnap (61.5%) | **+0.5pp (neutral)** |
| Gap analysis | gap_rag (61.5%) vs snap_rag (62%) | **-0.5pp (hurts)** |

**Key findings:**
1. **snap_hyde (65.5%) is the clear winner** — HyDE for both retrieval and reranking.
2. **Snap reasoning is the biggest single contributor** (+5pp) — forcing the model to reason before retrieval improves targeting.
3. **HyDE passage generation adds +3.5pp retrieval quality** — passage-form queries match the doctrinal corpus better than question-form queries (genre mismatch).
4. **HyDE reranking adds another +3pp** — cross-encoder benefits from passage-form input vs question-form.
5. **Gap analysis adds no value** — gap_rag (61.5%) ≤ snap_rag (62.0%). The overhead of gap identification and per-gap retrieval doesn't improve over simple snap+retrieve.
6. **Snap in final call is neutral** — snap_rag (62.0%) ≈ snap_rag_nosnap (61.5%). The model doesn't anchor or benefit from seeing its prior answer.
7. **11-char HyDE bug**: `_system_prompt(config, "hyde")` with gap-formatted user input produces truncated outputs on Gemma, making all gap_hyde variants unreliable. gap_rag (clean) confirms gap architecture doesn't help.

**Verdict:** CONFIRMED that snap_hyde is optimal. Gap architecture DISCARDED. The retrieval pipeline should be: snap → HyDE passage → dense retrieval → HyDE-based cross-encoder reranking → fresh answer with evidence.

**Commit:** 45dde43 through 6c8aa1e

---

### 2026-04-10 — snap_hyde_aligned across multiple embedders

**Hypothesis:** Decoupling dense retrieval (HyDE passage) from cross-encoder reranking (raw question) isolates the embedding model's contribution and may favor domain-specific embedders.

**Change:** Added `snap_hyde_aligned` mode with `rerank_query` parameter. Tested across 4 wave-1 embedders.

**Config:** Gemma 4 E4B, N=200, seed=42, BarExam.

**Results:**

| Embedder | rag_simple | snap_hyde | snap_hyde_aligned |
|---|---|---|---|
| gte-large | 57.0% | **65.5%** | 62.5% / 67.5%* |
| legal-bert | **62.0%** | 60.0% | **65.0%** |
| stella-400m | 61.0% | 60.0% | 62.5% |
| bge-m3 | 61.0% | 60.0% | 62.5% |

*Two runs showed high variance (62.5% vs 67.5%).

**Key findings:**
1. snap_hyde_aligned (question reranking) drops 3pp vs snap_hyde (HyDE reranking) for gte-large.
2. legal-bert's snap_hyde_aligned (65.0%) beats its snap_hyde (60.0%) by +5pp — domain vocabulary helps at embedding stage, and question-based reranking is better for legal-bert.
3. N=200 variance is ~5pp (gte-large aligned: 62.5% vs 67.5% across runs).

**Verdict:** MIXED — alignment helps legal-bert but hurts gte-large. HyDE-based reranking is generally better, but domain embedders benefit from question-based reranking.

**Commit:** 7f924a3

---

### 2026-04-09 — Embedding model A/B comparison on HPC cluster

**Hypothesis:** Alternative embedding models (legal-domain, larger MTEB, multilingual) will improve retrieval quality and downstream accuracy compared to the gte-large-en-v1.5 baseline.

**Change:** Built 4 embedding collections on cluster (local /tmp → NFS copy to avoid ChromaDB corruption). Each model gets its own ChromaDB directory. Ran rag_simple + snap_hyde for each at N=200 via autonomous SLURM pipeline.

**Config:** Gemma 4 E4B (cluster-vllm), N=200, seed=42, BarExam. Cross-encoder reranking on for all.

**Results:**

| Embedding | Params | rag_simple | snap_hyde |
|---|---|---|---|
| gte-large (baseline) | 434M | 57.0% | **65.5%** |
| legal-bert | 110M | **62.0%** | 60.0% |
| stella-400m | 400M | 61.0% | 60.0% |
| bge-m3 | 568M | 61.0% | 60.0% |

Reference: golden_passage = 62.2%, llm_only = 55.5%.

**Key findings:**
1. **rag_simple: all alternatives beat baseline** (+4-5pp). legal-bert's 62.0% nearly matches golden ceiling (62.2%).
2. **snap_hyde: baseline dominates** at 65.5%. All alternatives converge to exactly 60.0%.
3. **Asymmetric response to HyDE**: rag_simple embeds the raw question (question→passage), where legal vocabulary helps. snap_hyde embeds an LLM-generated hypothetical passage (passage→passage), where gte-large's general quality wins.
4. **Cross-encoder may be the bottleneck for snap_hyde**: all 3 non-baseline embedders produce identical 60.0%.
5. **Domain pretraining > parameter count**: legal-bert (110M) beats stella-400m (400M) and bge-m3 (568M).

**Verdict:** MIXED — Embedding matters for rag_simple (+5pp with legal-bert) but not for snap_hyde (baseline still best). Best overall: gte-large + snap_hyde at 65.5%.

**Commit:** bc6e361 (scripts), this commit (results sync)

---

### 2026-04-08 — HPC cluster full BarExam: Gemma 4 E4B vs Qwen3-8B

**Hypothesis:** Gemma 4 E4B (8B total, 4.5B effective MoE) will be competitive with Qwen3-8B despite lower effective parameter count.

**Change:** Set up vLLM serving on WashU HPC with split-venv pattern. Ran full N=1195 BarExam across all modes.

**Config:** Gemma 4 E4B (bf16, A6000) and Qwen3-8B (fp16, A40), N=1195 full BarExam.

**Results:**

| Mode | Gemma 4 E4B | Qwen3-8B | Delta |
|---|---|---|---|
| llm_only | **55.5%** | 52.1% | +3.4pp |
| golden_passage | **62.2%** | 60.1% | +2.1pp |
| rag_simple | **54.2%** | 36.5%* | — |
| snap_hyde | **58.6%** | — | — |
| avg latency | **12.5s** | 82.6s | **6.6x faster** |

*Qwen rag_simple corrupted by ChromaDB concurrent writes.

**Verdict:** CONFIRMED — Gemma 4 E4B is the best small model. 6.6x faster, 3.7x fewer tokens, higher accuracy.

**Commit:** bc6e361

---

### 2026-03-27 — CE score thresholding: skip RAG when retrieval quality is low

**Hypothesis:** Discarding retrieved passages when the best cross-encoder score is below a threshold (falling back to snap answer) will improve accuracy by avoiding cases where bad evidence misleads the model.

**Change:** Added `ce_threshold` eval mode. Flow: snap answer → HyDE → retrieve k=5 → if max CE < 4.0, use snap answer directly; otherwise, answer with evidence. Threshold calibrated from offline analysis of N=200 snap_hyde log showing snap accuracy 78% below CE=4.0 vs RAG accuracy 78% above.

**Config:** Llama 70B, N=200, seed=42, BarExam. CE threshold=4.0.

**Results:**

| Model | ce_threshold | snap_hyde | confidence_gated | llm_only |
|---|---|---|---|---|
| Llama 70B BarExam | **80.0%** | 76.5% | 79.0% | 64% |
| Scout 17B BarExam | **71.5%** | 71%* | 71.5% | 69% |

*N=100

**Routing split:**

| Model | snap_only (CE<4.0) | RAG (CE>=4.0) |
|---|---|---|
| Llama 70B | 87/200 (44%) at 77.0% | 113/200 (56%) at 82.3% |
| Scout 17B | 132/200 (66%) at 69.7% | 68/200 (34%) at 75.0% |

Scout routes 66% to snap_only (vs 44% for Llama) because Scout's retrieval gets lower CE scores. But Scout's snap accuracy is also lower (69.7% vs 77.0%).

**Flip analysis (Llama BarExam, N=200):**

| Comparison | ce_threshold broke | ce_threshold fixed | Net |
|---|---|---|---|
| vs snap_hyde | 11 | 18 | **ce_threshold +7** |
| vs confidence_gated | 11 | 13 | **ce_threshold +2** |

**Cross-model stability analysis (CRITICAL):**

| Context | CE<4.0 snap accuracy | Thresholding effect |
|---|---|---|
| Llama BarExam (all Qs) | 78% | **Helps** (+3.5 vs snap_hyde) |
| Scout BarExam (all Qs) | 66% | **Hurts** (-4 simulated) |
| Llama conf_gated (uncertain Qs) | 0% | **Catastrophic** |

CE thresholding depends on snap answer quality. For strong models on well-known domains (Llama BarExam), skipping low-quality retrieval is beneficial. For weaker models (Scout) or uncertain questions (confidence_gated subset), the snap answer is too unreliable to fall back to.

**Key findings:**

1. **Llama 80.0% = new best for BarExam.** Beats confidence_gated (79.0%) by 1pt, snap_hyde (76.5%) by 3.5pt. Fixed 18 questions snap_hyde got wrong, broke 11 → net +7.

2. **Scout 71.5% = matches confidence_gated.** The simulation predicted 67% (would hurt), but the actual run matched Scout's previous best. The threshold works differently for Scout: it routes 66% to snap (vs 44% Llama) because Scout's retrieval quality is lower, but Scout's snap is also weaker (69.7% vs 77.0%).

3. **The threshold effect is model-dependent but doesn't hurt.** CE=4.0 improves Llama (+3.5 over snap_hyde), matches Scout (0 vs conf_gated). Unlike confidence_gated which *loses* on HousingQA (-5.5), CE thresholding's downside seems bounded.

4. **Simulations from existing logs were pessimistic for Scout.** The N=100 Scout log used for simulation may not have been representative — actual N=200 performance was better. This reinforces principle: always run the actual experiment rather than relying solely on offline simulation.

**Verdict:** CONFIRMED for Llama BarExam (new best). NEUTRAL for Scout BarExam (matches conf_gated). The CE threshold is a valid retrieval quality gate that avoids the worst retrievals. CaseHOLD and HousingQA validation deferred (resource constraints).

**Commit:** TBD

---

### 2026-03-27 — Atomic reasoning blocks: self-correction is destructive

**Hypothesis:** A second LLM call for self-verification, double-checking, or adversarial debate will catch errors and improve accuracy over single-shot answers.

**Change:** Added 3 eval modes: `self_verify` (snap → review for errors), `double_snap` (2 independent answers → agree=use, disagree=CE-threshold RAG), `snap_debate` (snap → adversarial critique).

**Config:** Llama 70B + Scout 17B, N=200, seed=42, BarExam.

**Results:**

| Block | Llama 70B | Scout 17B | Calls | Type |
|---|---|---|---|---|
| llm_only (baseline) | 64% | 69% | 1 | — |
| snap_hyde (baseline) | 76.5% | 71%* | 3 | w/ retrieval |
| **ce_threshold** | **80.0%** | **71.5%** | 2-3 | w/ retrieval |
| self_verify | 73.0% | **58.5%** | 2 | reasoning only |
| double_snap | 74.0% | — | 2-4 | reasoning + optional RAG |
| snap_debate | 72.0% | 64.0% | 2 | reasoning only |

*N=100

**Key findings:**

1. **Self-correction is destructive for both models.** Llama loses 3-4pts (73% vs 76.5% snap), Scout loses 10.5pts (58.5% vs 69%). The review step introduces doubt that flips correct answers more than it fixes wrong ones.

2. **Weaker models are hurt more.** Scout's 10.5pt drop from self-verify vs Llama's 3.5pt drop. Weaker models are more susceptible to second-guessing.

3. **Double-snap with CE-threshold RAG doesn't help either.** 74.0% is below snap_hyde (76.5%). The 2-vote confidence signal is weaker than 3-vote (confidence_gated: 79%), and the CE fallback path has the same dead-zone problem.

4. **Adversarial debate is the worst.** Framing the second call as a critical reviewer makes it even more likely to flip correct answers (72.0% Llama, 64.0% Scout).

**Why self-correction fails on MC questions:** The model's initial reasoning on well-structured MC questions is already its best calibration. A review step has no new information — it's the same model re-reading the same question. It tends to rationalize changes rather than catch genuine errors. This is unlike retrieval, which provides external information that can genuinely shift the answer.

**Verdict:** DISCARD all three. Second-guessing without new information is net-negative. The model's first-pass answer is better than any self-review variant. This strongly supports the ce_threshold approach: the improvement comes from *external information* (retrieved evidence above quality threshold), not from additional reasoning passes.

**Commit:** included below

---

### 2026-03-27 — Pipeline integration: planner+synthesizer overhead costs 4 points

**Hypothesis:** Integrating HyDE + CE threshold into the full pipeline (router → planner → executor → synthesizer) will match standalone ce_threshold performance.

**Change:** Modified `_execute_rag_search` in `main.py` to use snap answer → HyDE → retrieve → CE threshold gating instead of query rewrite → retrieve → synthesize.

**Config:** Llama 70B, N=200, seed=42, BarExam. Full pipeline via `full_pipeline` eval mode.

**Result:** **76.0%** (vs 80.0% standalone ce_threshold, vs 76.5% snap_hyde)

**Routing:** 94% of steps used snap answer (CE<4.0). Planner's sub-questions produce HyDE passages that retrieve worse evidence than the full question does.

**Root cause:** Same as decompose_rag finding (-5pts). The planner decomposes questions into sub-questions that are less specific, producing worse HyDE passages. The synthesizer that recombines sub-answers is a lossy step. Together: -4pts.

**Key data:**
- Standalone ce_threshold: full question → snap → HyDE → CE gate → 80.0%
- Pipeline: planner sub-Q → snap → HyDE → CE gate → synthesizer → 76.0%
- The gap is entirely from planning decomposition + synthesis recombination

**Verdict:** The pipeline's planner/synthesizer architecture is net-negative for BarExam MC questions. The correct approach is bottom-up: start from the proven atomic strategy (ce_threshold) and only add components that demonstrably help.

**Implication for pipeline design:** Don't assume multi-step planning helps. For single-question tasks, the overhead of decomposition + recombination costs more than it gains. Planning may still help for genuinely multi-hop questions, but BarExam MC is not that.

**Commit:** included below

---

### 2026-03-27 — CE threshold k=3: fewer passages don't help

**Hypothesis:** Using k=3 instead of k=5 with CE thresholding will reduce noise from lower-ranked passages and improve accuracy.

**Config:** Llama 70B, N=200, seed=42, BarExam, CE threshold=4.0, k=3.

**Result:** 79.0% (vs 80.0% at k=5). Routing: 49% snap_only (vs 44% at k=5). Flip analysis: broke 16, fixed 14 → net -2.

**Verdict:** DISCARD. k=3 is marginally worse. The 4th and 5th passages at k=5 are net-positive when they pass the CE threshold.

---

### 2026-03-27 — Aspect-based query rewrite: offline CE gains don't translate to accuracy

**Hypothesis:** Adding rule/exception aspect queries alongside the HyDE passage will improve retrieval diversity and accuracy. Prior offline test showed 2x CE improvement (6.0 vs 3.0).

**Change:** Added `snap_hyde_aspect` mode. Flow: snap → HyDE + aspect queries (rule, exception) → multi-query retrieval (pool + rerank) → answer.

**Config:** Llama 70B, N=200, seed=42, BarExam.

**Results:** **76.0%** (vs snap_hyde 76.5%)

**Analysis:**
- Max CE scores slightly better: 4.30 (aspect) vs 4.15 (snap_hyde) — but far less than the offline 2x
- Flip analysis: broke 16, fixed 15 → net -1
- 4 LLM calls (vs 3 for snap_hyde) for no accuracy gain

**Why the offline 2x didn't translate:** The offline test compared aspect queries vs raw synonym queries (no HyDE). When HyDE is already in the mix, the HyDE passage provides strong embedding-level similarity that aspect queries can't improve much. The 2x CE gain was relative to a weaker baseline (synonym rewrites alone).

**Verdict:** DISCARD. Per simplicity criterion: same result with more complexity and cost. Focused HyDE retrieval is already sufficient for BarExam.

**Commit:** included below

---

### 2026-03-27 — CaseHOLD CE threshold validation + combined confidence+CE approach

**Hypothesis 1:** CE threshold generalizes to CaseHOLD.
**Result:** 71.0% (vs 72.5% baseline). 96% routed to snap_only because CaseHOLD holdings get very low CE scores. Effectively reduces to llm_only. **NEUTRAL** — threshold correctly avoids bad retrieval, slight noise from unused HyDE step.

**Hypothesis 2:** Combining confidence gating (3-vote) with CE thresholding (skip bad evidence on RAG path) will beat either alone.
**Result:** **76.5%** — worse than both ce_threshold (80.0%) and confidence_gated (79.0%).

**Routing breakdown (Llama BarExam N=200):**

| Route | Questions | Accuracy |
|---|---|---|
| skip_rag (unanimous) | 155 (78%) | 83.9% |
| rag (CE >= 4.0) | 22 (11%) | 54.5% |
| snap_ce_fallback (CE < 4.0) | 23 (11%) | **47.8%** |

**Root cause:** The snap_ce_fallback path (uncertain snap + low CE → use snap anyway) is catastrophic at 47.8%. These are questions where the model disagrees AND retrieval is poor — falling back to snap when the model is uncertain is worse than using mediocre evidence. confidence_gated's strength is forcing RAG on uncertain questions; CE threshold removes that lifeline.

**Verdict:** DISCARD. The two strategies address different failure modes and conflict when composed. CE threshold helps when snap is reliable (Llama BarExam); confidence gating helps when snap is uncertain. They don't compose — double filtering leaves "uncertain + bad evidence" with no good option.

**Key lesson:** Not all validated components compose well. Each encodes a different assumption about what the model needs, and combining them can create dead zones where neither assumption holds. Test compositions, don't assume additivity.

**Commit:** included below

---

### 2026-03-27 — CE score thresholding: skip RAG when retrieval quality is low

(see above for full details)

---

### 2026-03-26 — Decompose+RAG: combining decomposition with retrieval hurts

**Hypothesis:** Decomposing a question into sub-questions and applying Snap-HyDE retrieval per sub-question will outperform either decomposition or RAG alone, by combining structured reasoning with targeted evidence.

**Change:** Added `decompose_rag` eval mode. Flow: decompose → for each sub-question: snap answer → HyDE passage → retrieve k=3 → synthesize all sub-answers + evidence.

**Config:** Llama 70B + Scout 17B, N=200, seed=42, BarExam + CaseHOLD. HousingQA OOMs on 16GB RAM (1.8M doc collection too large).

**Results:**

| Model | Dataset | decompose_rag | decompose_best | snap_hyde | llm_only | conf_gated |
|---|---|---|---|---|---|---|
| Scout 17B | BarExam | 70.5% | 75% (struct) | 71%* | 69% | 71.5% |
| Scout 17B | CaseHOLD | **73.5%** | 73% (nat) | 71%* | 72.5%* | 72.5%* |
| Llama 70B | BarExam | 71.5% | 76% (struct) | 76.5% | 64% | **79%** |
| Llama 70B | CaseHOLD | 70.5% | 71.5% (nat) | 71% | **72.5%** | **72.5%** |

*N=100 or estimated from Llama results.

**Sub-analysis — decomposition success vs accuracy:**

| Model | Dataset | 1 sub-q (fallback) | 3 sub-q (decomposed) |
|---|---|---|---|
| Llama 70B | BarExam | 72.9% (107 Qs) | 69.9% (93 Qs) |
| Scout 17B | BarExam | 69.2% (26 Qs) | 70.7% (174 Qs) |
| Scout 17B | CaseHOLD | 75.7% (107 Qs) | 71.0% (93 Qs) |

**Key finding:** Llama 70B fails to decompose BarExam questions 53% of the time (returns 11-char non-JSON, falls back to single question = basically snap_hyde with extra synthesis). When it does decompose, accuracy drops 3pts vs fallback.

**Key findings:**

1. **Decompose+RAG is strictly worse than simpler approaches.** On BarExam, it underperforms both decompose-only (by 4-5pts) and snap_hyde (by 5-6pts for Llama). Adding retrieval to decomposed sub-questions introduces noise without improving reasoning.

2. **The synthesis step loses signal.** When Llama falls back to 1 sub-question (basically snap_hyde), it scores 72.9% — still below snap_hyde's 76.5%. The additional synthesis prompt that combines sub-answers + evidence is a lossy step.

3. **CaseHOLD Scout is the one marginal win (+0.5%).** But this is within noise — and the fallback path (75.7%) outperforms the decomposed path (71.0%) even here.

4. **Llama can't decompose MC questions.** The natural decomposition prompt returns very short non-JSON responses for 53% of BarExam questions. Scout succeeds 87% of the time. The MC format with choices confuses Llama's decomposition.

**Per-question flip analysis (Llama BarExam, N=200):**

| Comparison | decompose_rag broke | decompose_rag fixed | Net |
|---|---|---|---|
| vs snap_hyde | 30 | 20 | snap_hyde +10 |
| vs confidence_gated | 26 | 11 | conf_gated +15 |

Of the 30 questions snap_hyde got right that decompose_rag broke: 19 were fallback (1 sub-q), 11 were decomposed (3 sub-q). The synthesis step is the main damage source — even the fallback path (which is basically snap_hyde + unnecessary synthesis) loses 10 questions vs direct snap_hyde.

**Cross-encoder analysis:**
- BarExam CE scores: mean 3.27 (correct: 3.22, wrong: 3.56 — wrong answers have *higher* CE, meaning retrieved evidence actively misleads)
- CaseHOLD CE scores: mean -2.78 (negative = mostly irrelevant passages)
- Gold passage retrieval: only 10% on BarExam — the per-sub-question k=3 retrieval gets different (worse) passages than the single-query k=5

**Verdict:** REFUTED — combining decomposition with retrieval is strictly worse than either alone. The added complexity doesn't help. The synthesis step is the main damage source. This reinforces the project's core finding: simpler is better, and components should be tested in isolation before combining.

**Commit:** TBD

---

### 2026-03-25 — Decomposition: helps weaker models more, prompt variant matters

**Hypothesis:** Breaking a question into sub-questions and answering each independently improves accuracy over single-shot answering (no retrieval, pure reasoning test).

**Change:** Added `decompose` eval mode. Two prompt variants: "natural" (model decides sub-issues) and "structured" (IRAC: rule/application/exception). 3-5 LLM calls per question.

**Config:** Llama 70B + Scout 17B, N=200, seed=42, BarExam + HousingQA + CaseHOLD

**Results (N=200, all models/datasets/variants):**

| Model | Dataset | llm_only | decomp_natural | decomp_structured | snap_hyde | conf_gated |
|---|---|---|---|---|---|---|
| Llama 70B | BarExam | 64% | 75% | **76%** | 76.5% | **79%** |
| Llama 70B | HousingQA | 47% | 48.5% | 46.5% | **56%** | 50.5% |
| Llama 70B | CaseHOLD | **72.5%** | 71.5% | 70.5% | 71% | **72.5%** |
| Scout 17B | BarExam | 69% | 70% | **75%** | 71%* | 71.5% |
| Scout 17B | HousingQA | 50% | **59%** | 56% | 54%* | 53.5% |
| Scout 17B | CaseHOLD | 72.5%† | **73%** | 72% | 71%† | 72.5%† |

*N=100 baseline. †Estimated from Llama results.

**Key findings:**

1. **Decomposition helps weaker models more.** Scout gains +6 (BarExam structured) and +9 (HousingQA natural) from decomposition. Llama gains are smaller (+12 BarExam but only +1.5 Housing). Weaker models benefit more from being guided through sub-issues.

2. **Best prompt variant is model × dataset dependent.** Structured helps Scout on BarExam (+5 over natural) but natural helps Scout on HousingQA (+3 over structured). No universal winner.

3. **Scout HousingQA: decompose_natural (59%) beats everything including snap_hyde (54%).** This is the first time any non-RAG approach beats RAG on a knowledge-gap dataset. Decomposition helps the model reason through state-specific statutes it doesn't know by breaking the problem into manageable pieces.

4. **Decomposition ≈ snap prompting for Llama on BarExam.** Both get ~76%. The step-by-step reasoning in a single call achieves the same as explicit decomposition — for a strong model, the decomposition overhead isn't worth it.

5. **CaseHOLD: decomposition is neutral.** Citation-matching isn't helped by sub-question decomposition.

**Verdict:** CONFIRMED that decomposition helps, but the effect is model-dependent. Most valuable for weaker models. The Scout HousingQA result (59%) is particularly notable — it suggests decomposition can substitute for retrieval on knowledge-gap tasks by helping the model reason more carefully with what it already knows.

**Commit:** TBD

---

### 2026-03-25 — Confidence-gated RAG: self-consistency routing beats always-on RAG

**Hypothesis:** Using 3 snap answers as a confidence signal (unanimous = skip RAG, disagreement = apply Snap-HyDE) will outperform always-on RAG by avoiding cases where retrieval hurts.

**Change:** Added `confidence_gated` eval mode. 3 snap answers vote; if unanimous, skip RAG. If disagreement, apply Snap-HyDE using majority snap's reasoning.

**Config:** Llama 70B + Scout 17B, N=200, seed=42, BarExam + HousingQA + CaseHOLD

**Results:**

| Model | Dataset | llm_only | snap_hyde | maj_vote | conf_gated | skip% | RAG on uncertain |
|---|---|---|---|---|---|---|---|
| Llama 70B | BarExam | 64% | 76.5% | 76.5% | **79.0%** | 77% | +11pts |
| Llama 70B | HousingQA | 47% | **56.0%** | 49.5% | 50.5% | 90% | +10pts |
| Llama 70B | CaseHOLD | **72.5%** | 71.0% | 72.0% | **72.5%** | 90% | +5pts |
| Scout 17B | BarExam | 69% | 71.0%* | 68.5% | **71.5%** | 60% | +7pts |
| Scout 17B | HousingQA | 50% | **54.0%*** | 55.0% | 53.5% | 79% | -7pts |

*N=100 baseline, not directly comparable.

**Key findings:**

1. **On BarExam (both models): confidence-gated is the best approach.** It beats always-on snap_hyde (+2.5 Llama, +0.5 Scout) and majority-vote-only (+2.5 Llama, +3.0 Scout). RAG provides +11pts on uncertain questions, correctly avoided on confident ones.

2. **On HousingQA: always-on snap_hyde wins.** Confidence gating routes 90% to skip_rag because the model is *unanimously confident but wrong* (Yes-bias). Self-consistency detects factual uncertainty but not systematic bias.

3. **On CaseHOLD: confidence gating matches llm_only, avoids RAG regression.** The gating correctly identifies that RAG doesn't help on this task (90% skip).

4. **Scout routes more to RAG (40% on BarExam vs 23% Llama).** Weaker model = more disagreement = more retrieval triggered. This is the correct adaptive behavior.

**Verdict:** CONFIRMED for factual uncertainty (BarExam), REFUTED for systematic bias (HousingQA). Self-consistency is a valid confidence signal that improves RAG routing on tasks where the model's errors are random rather than systematic. The optimal strategy depends on the error mode: random errors → confidence-gated, systematic bias → always-on RAG.

**Commit:** TBD

---

### 2026-03-25 — Cross-dataset generalizability: RAG helps only on unknown domains

**Hypothesis:** Snap-HyDE's retrieval benefit generalizes across legal domains and task formats. Tested on 3 new datasets: CaseHOLD (5-way MC holding identification, 50K holdings corpus), Legal-RAG-QA (open-ended US criminal law, 190 passage corpus), and Australian Legal QA (open-ended Australian law, 2.1K passages).

**Change:** Added 3 new datasets to eval harness with dataset-specific prompts, embeddings, and LLM-as-judge scoring for open-ended tasks.

**Config:** Llama 70B, seed=42. CaseHOLD N=200, Legal-RAG-QA N=138 (full), Australian N=200.

**Results:**

| Dataset | llm_only | rag_snap_hyde | RAG lift |
|---|---|---|---|
| CaseHOLD (5-way MC) | 72.5% | 71.0% | **-1.5** |
| Legal-RAG-QA (open) | 100%† | 99.3%† | -0.7 |
| Australian Legal (open) | 100%† | 100%† | 0 |

†LLM-as-judge scoring — likely too lenient for open-ended tasks.

**Verdict:** REFUTED for generalizability. RAG does not help on these three new datasets:
- **CaseHOLD**: Retrieval *hurts* (-1.5pts). The task is citation-to-holding matching, not factoid retrieval. The holdings corpus contains similar but wrong holdings that mislead the model.
- **Legal-RAG-QA / Australian**: LLM already at ceiling (100%). These datasets don't test retrieval because the LLM can answer from training data alone. The LLM-as-judge is also too lenient for fine-grained evaluation.

**Cross-dataset pattern (all 5 datasets):**
- RAG lift is inversely proportional to LLM baseline accuracy
- **HousingQA** (47% baseline): +9pt lift — LLM genuinely doesn't know state statutes
- **BarExam** (64% baseline): ~0 net lift at N=200 — LLM knows doctrinal law
- **CaseHOLD** (72.5% baseline): -1.5pt — task is matching, not knowledge retrieval
- **Legal-RAG-QA / Australian** (100% baseline): 0 — LLM already at ceiling

**Commit:** TBD

---

### 2026-03-25 — Devil's advocate HyDE: counterevidence hurts more than it helps (76% vs 82%)

**Hypothesis:** 50% of Snap-HyDE failures are confirmation bias loops. Generating a devil's advocate HyDE passage (opposing the snap answer) alongside the supporting passage should break this loop by introducing counterevidence.

**Change:** Added `rag_devil_hyde` mode — snap answer → supporting HyDE + opposing HyDE → pool both retrievals → answer with evidence.

**Config:** Llama 70B, N=100, seed=42, BarExam

**Result:** **76%** (vs snap_hyde 82%, llm_only 64%)

| Subject | Devil HyDE | Snap HyDE |
|---|---|---|
| CONST. LAW | 100% | 100%* |
| CONTRACTS | 83% | 83%* |
| EVIDENCE | 86% | 86%* |
| REAL PROP. | 50% | 75%* |
| TORTS | 75% | 83%* |
| CRIM. LAW | 67% | 67%* |

**Verdict:** REFUTED — devil's advocate retrieval made things worse (-6pts). The counterevidence doesn't selectively fix confirmation bias; it introduces noise that confuses the model on questions it was already getting right. The cure is worse than the disease. On well-known doctrinal law, the model doesn't need to be second-guessed — it needs confirmation of its correct reasoning.

**Commit:** TBD

---

### 2026-03-25 — Top-2 answer HyDE: slight improvement over devil's advocate but still below snap (79%)

**Hypothesis:** Instead of generic counterevidence, retrieving for the model's second-choice answer would provide more targeted alternative evidence, helping on cases where the snap answer is a near-miss.

**Change:** Added `rag_top2_hyde` mode — snap with top-2 reasoning → HyDE for primary answer + HyDE for second-choice → pool retrievals → answer.

**Config:** Llama 70B, N=100, seed=42, BarExam

**Result:** **79%** (vs snap_hyde 82%, devil_hyde 76%, llm_only 64%)

**Verdict:** REFUTED — top-2 retrieval is better than devil's advocate (+3) but still worse than focused snap_hyde (-3). Providing evidence for two competing answers splits the model's attention. The lesson is consistent: focused, single-hypothesis retrieval outperforms diverse retrieval on this task. The model does best when given evidence aligned with its primary reasoning.

**Commit:** TBD

---

### 2026-03-25 — HousingQA N=200 validation: 56% (down from 61% at N=100)

**Hypothesis:** The +17pt RAG lift at N=100 holds at N=200.

**Change:** Ran rag_snap_hyde and llm_only at N=200 on HousingQA.

**Config:** Llama 70B, N=200, seed=42, housing_statutes collection

**Result:**

| Mode | N=100 | N=200 |
|---|---|---|
| llm_only | 44% | 47% |
| rag_snap_hyde | 61% | **56%** |
| RAG lift | +17 | **+9** |

**Verdict:** MIXED — RAG still provides meaningful lift (+9pts at N=200), but the N=100 result was inflated. The +17pt figure was optimistic; +9 is the more reliable estimate. Same pattern as BarExam: N=100 samples tend to be easier. However, +9pts is still the largest validated RAG lift in the project, confirming that retrieval is most valuable when the LLM genuinely lacks domain knowledge.

**Commit:** TBD

---

### 2026-03-25 — N=200 BarExam Snap-HyDE validation: 76.5% (down from 82% at N=100)

**Hypothesis:** The 82% N=100 result holds at N=200 for stability confirmation.

**Change:** Ran rag_snap_hyde at N=200 on BarExam.

**Config:** Llama 70B, N=200, seed=42

**Result:** 153/200 = **76.5%** (vs 82% at N=100)

By subject: EVIDENCE 82%, CONTRACTS 81%, CONST. LAW 79%, REAL PROP. 76%, TORTS 75%, CRIM. LAW 75%

**Deep analysis:**

| Subset | Snap (=LLM-only) | Final (Snap-HyDE) | RAG lift |
|---|---|---|---|
| First 100 (N=100 overlap) | 79% | 82% | +3 |
| Extra 100 (harder questions) | 74% | 71% | **-3** |
| Full 200 | 76.5% | 76.5% | **0** |

Retrieval fixed 15 questions and hurt 15 — perfect net zero. The first 100 questions were an easier sample (snap=79%) than the extra 100 (snap=74%). On easier questions, retrieval helps (+3); on harder questions, it hurts (-3).

**Verdict:** The N=100 result (82%) was inflated by an easier question sample. At N=200, Snap-HyDE's net retrieval contribution is zero — it shuffles which questions it gets right without improving overall accuracy. The 76.5% figure entirely comes from the snap step (LLM-only reasoning with step-by-step prompting).

**Key implication:** The true RAG lift may be near zero for BarExam with strong models. The snap-step format itself (explicit step-by-step reasoning) provides the real value, not the retrieval. HousingQA (+17pt lift) is where retrieval genuinely helps — because the LLM doesn't know state-specific statutes.

**Commit:** TBD

---

### 2026-03-25 — HousingQA: first eval on second corpus, massive RAG lift

**Hypothesis:** Snap-HyDE's retrieval benefit generalizes from BarExam (MC, doctrinal law) to HousingQA (Yes/No, state-specific statute law). HousingQA should show even larger RAG lift because state-specific housing statutes are less likely to be in the LLM's training data.

**Change:** Added `--dataset housing` support to eval_harness.py. Adapted prompts for Yes/No format, pointed retrieval at `housing_statutes` collection (1.8M docs, dense-only).

**Config:** Llama 70B, N=100, seed=42, housing_statutes collection

**Results:**

| Mode | Accuracy | Yes Recall | No Recall | Yes-bias |
|---|---|---|---|---|
| llm_only | 44% | 85% | 23% | 80% pred Yes |
| rag_snap_hyde | **61% (+17)** | 79% | 52% | 59% pred Yes |

**Verdict:** CONFIRMED — Snap-HyDE generalizes to a completely different corpus and task format. The +17pt lift is the largest RAG improvement in the project (vs +7 on BarExam). Key finding: the LLM has a massive Yes-bias on housing questions (predicts Yes 80% of the time when only 34% are Yes). Retrieval grounds the model in actual statute text and more than doubles No recall (23% → 52%).

**Cross-model results:**

| Model | llm_only | rag_snap_hyde | Lift |
|---|---|---|---|
| Llama 70B | 44% | **61%** | **+17** |
| Scout 17B | 50% | 54% | +4 |

RAG lift scales with model capability (same pattern as BarExam). Scout's smaller lift may be because its snap reasoning is too shallow to generate targeted HyDE passages.

**Commit:** TBD

### 2026-03-25 — Error analysis: Snap-HyDE failure modes on BarExam

**Hypothesis:** Understanding why Snap-HyDE fails on 18% of questions reveals whether failures are fixable with better retrieval or are fundamental reasoning limits.

**Change:** Categorized all 18 failures from Snap-HyDE (Llama 70B, N=100, BarExam).

**Results:**

| Failure Category | Count | % | Description |
|---|---|---|---|
| Retrieval HURT | 4 | 22% | Snap had correct answer, evidence flipped it to wrong |
| Model STUCK | 9 | 50% | Same wrong answer before and after retrieval |
| Retrieval SHIFTED | 5 | 28% | Evidence changed answer but still wrong |

Key stats: Snap accuracy=75%, retrieval fixed 11 questions but hurt 4 (net +7). Gold passage retrieved in only 2/18 failures. Avg max CE score: 4.14 (correct) vs 3.70 (wrong).

**Verdict:** The dominant failure mode (50%) is a **confirmation bias loop** — wrong snap reasoning → wrong HyDE targeting wrong doctrine → retrieves confirming evidence → model doubles down. This is inherent to snap-informed retrieval. Potential mitigations: devil's advocate HyDE (retrieve for opposing answer) or top-2 answer retrieval.

**Commit:** TBD

---

### 2026-03-24 — Snap-HyDE cross-model: best approach across all models
**Hypothesis:** Snap-HyDE's outsized lift on Llama 70B generalizes to weaker models.
**Change:** Ran rag_snap_hyde on Scout and GPT 5.4-nano.
**Config:** Scout N=100, GPT 5.4-nano N=100, seed=42
**Results:**
| Model | llm_only | golden | rag_hyde | snap_hyde |
|---|---|---|---|---|
| Llama 70B | 64% | 81% | 75% | **82%** |
| Scout 17B | 64% | 73% | 69% | **71%** |
| GPT 5.4-nano | 57% | — | 64% | **64%** |
**Verdict:** CONFIRMED for strong models, MIXED for weak. Snap-hyde is best or tied-best across all models. The lift scales with model capability: the better the snap reasoning, the more targeted the HyDE passage, the better the retrieval. Nano's snap isn't detailed enough to improve over plain HyDE.
**Commit:** TBD

### 2026-03-24 — Snap-HyDE beats golden passage ceiling (82% vs 81%)
**Hypothesis:** If the LLM reasons through the question first (snap), then generates a HyDE passage targeted at its specific reasoning, retrieval becomes more precise than generic HyDE.
**Change:** Added `rag_snap_hyde` — snap answer → targeted HyDE generation → retrieve → direct answer with evidence. Also tested `rag_multi_hyde` — 3 aspect-targeted HyDE passages (rule/exception/application) pooled.
**Config:** Llama 70B, N=100, seed=42
**Results:**
| Mode | Accuracy | Calls/q |
|---|---|---|
| rag_hyde | 75% | 2 |
| rag_multi_hyde | 74% | 2 |
| **rag_snap_hyde** | **82%** | 3 |
| golden_passage | 81% | 1 |
**Verdict:** CONFIRMED — snap-informed HyDE is our best result, exceeding even golden passages. The snap reasoning makes HyDE generation surgically targeted at the specific doctrine. Multi-HyDE (aspect diversity) didn't help — focused retrieval beats diverse retrieval.
**Implication:** The snap+targeted-retrieval pattern is more valuable than having the "perfect" passage, because it retrieves evidence aligned with the model's actual reasoning chain. This is the approach to integrate into the main pipeline.
**Commit:** TBD

### 2026-03-24 — HyDE k=10: more passages doesn't help
**Hypothesis:** Now that HyDE retrieves relevant passages, more passages (k=10 vs k=5) might help by providing more evidence.
**Change:** Ran HyDE with k=10 on Llama 70B.
**Config:** Llama 70B, N=100, seed=42, k=10
**Result:** 74% (vs 75% at k=5).
**Verdict:** REFUTED — more passages adds noise from lower-ranked results. k=5 is optimal.
**Commit:** TBD

### 2026-03-24 — HyDE cross-model: consistent lift, scales with capability
**Hypothesis:** HyDE improvement generalizes across model architectures (not just Llama family).
**Change:** Ran rag_hyde on GPT 5.4-nano.
**Config:** GPT 5.4-nano N=100, seed=42
**Result:** 64% (+7 over llm_only 57%, +5 over rag_simple 59%). Biggest proportional RAG lift across all models.
**Verdict:** CONFIRMED — HyDE generalizes across Llama and GPT architectures. Lift scales with model capability: nano +7, Scout +5†, 70B +11.
**Commit:** TBD

### 2026-03-24 — HyDE + arbitration: helps weak models, hurts strong ones
**Hypothesis:** Combining HyDE retrieval with conservative arbitration (snap → HyDE passages → review) would improve on direct HyDE by protecting the snap answer from misleading passages.
**Change:** Added `rag_hyde_arb` eval mode.
**Config:** Scout N=100, Llama 70B N=100, seed=42
**Results:**
| Mode | Scout | Llama 70B |
|---|---|---|
| rag_hyde (direct) | 69% | **75%** |
| rag_hyde_arb | **71%** (+2) | 74% (-1) |
**Verdict:** MIXED — arbitration helps Scout (+2, snap provides reasoning anchor) but slightly hurts Llama 70B (-1, conservative bias prevents using good passages). Optimal strategy is model-dependent: weaker models benefit from arbitration's protection, stronger models are better off seeing evidence directly.
**Commit:** TBD

### 2026-03-24 — HyDE retrieval: best retrieval result, +11pts on Llama 70B
**Hypothesis:** Questions and doctrinal answers live in different semantic spaces. Generating a hypothetical answer passage (HyDE) and embedding that instead of the question should bridge the gap and improve retrieval quality.
**Change:** Added `rag_hyde` eval mode — LLM generates a textbook-style passage, embeds it for retrieval, then answers with retrieved evidence.
**Config:** Scout N=200, Llama 70B N=100, seed=42, unfiltered collection
**Results:**
| Metric | rag_simple (Scout) | rag_hyde (Scout) | rag_simple (70B) | rag_hyde (70B) |
|---|---|---|---|---|
| Accuracy | 68% | 69% (+1) | 73% | **75%** (+2) |
| Gold recall | 2% | 6% (3x) | — | — |
| Mean max CE | -1.35 | **+0.56** | — | — |
| % positive CE | 33% | **56%** | — | — |
**Verdict:** CONFIRMED — HyDE dramatically improves retrieval quality (CE scores flip from negative to positive, gold recall triples). Accuracy gain is model-dependent: flat on Scout (can't use evidence well), +2pts on Llama 70B (75% is best retrieval result, captures 65% of golden passage value). The query-document semantic gap was the core retrieval problem.
**Commit:** TBD

### 2026-03-24 — MBE source filter: better gold recall, worse accuracy
**Hypothesis:** Filtering retrieval to MBE-only (2.3K docs) would improve accuracy by finding more relevant study material passages instead of caselaw.
**Change:** Added `--source-filter` to eval harness, tested MBE-filtered retrieval.
**Config:** Scout 17B, N=200, seed=42, rag_simple with `where={"source": "mbe"}`
**Result:**
| Metric | Unfiltered (686K) | MBE filter (2.3K) |
|---|---|---|
| Accuracy | 68% | **66%** (worse) |
| Gold recall | 2% | 8% (4x better) |
| Mean max CE | -1.35 | -2.54 (worse) |
| % positive CE | 33% | 22% (worse) |
**Verdict:** REFUTED — better gold recall doesn't mean better accuracy. MBE pool is too small/narrow for the embedding model to match well. Caselaw passages, while not gold, provide more semantically relevant content (higher CE scores) that the LLM uses for reasoning.
**Implication:** The retrieval bottleneck is embedding quality, not corpus composition. The embedding model can't match legal questions to doctrinal answers even in a focused 2.3K collection. Need either better embeddings or a different retrieval approach.
**Commit:** TBD

### 2026-03-24 — Root cause: gold passages drowned by 680K caselaw docs
**Hypothesis:** Retrieval quality is bad because the collection mixes 2,318 MBE study passages with 678,612 caselaw passages (0.3% signal). Filtering to MBE-only should dramatically improve gold recall.
**Change:** Analysis only — tested metadata filter `where={"source": "mbe"}` on retrieval.
**Config:** 20 questions, dense retrieval, gte-large-en-v1.5 embeddings
**Result:**
| Filter | Recall@1 | Recall@5 | Recall@10 |
|---|---|---|---|
| None (686K docs) | 0% | 0% | 0% |
| MBE only (2.3K docs) | 5% | 10% | 15% |
**Verdict:** CONFIRMED — caselaw passages are drowning the signal. MBE filter takes recall from 0% to 15%. Still low but proves the mechanism.
**Implication:** Need to either (a) filter to MBE-only, (b) split collections, or (c) boost MBE results in reranking. This is the single biggest retrieval improvement available.
**Commit:** bd09427

### 2026-03-24 — N=200 validation reveals sampling noise
**Hypothesis:** The +5pt rag_simple lift over llm_only (69% vs 64%) would hold at larger sample size.
**Change:** Ran both modes at N=200 on Scout.
**Config:** Scout 17B, N=200, seed=42
**Result:** llm_only=69.0%, rag_simple=68.0%
**Verdict:** REFUTED — the N=100 llm_only baseline (64%) was unlucky. At N=200 both modes are tied. Retrieved passages are neutral, not helpful, on Scout with debiased prompts.
**Takeaway:** N=100 is insufficient for detecting small effects. Use N=200+ for decisions.
**Commit:** 8f85a0c

### 2026-03-24 — Cross-encoder scores are universally negative
**Hypothesis:** (Analysis, not experiment) CE scores should predict when retrieval helps vs hurts.
**Change:** Analyzed CE score distributions across RAG outcome categories (helps/hurts/neutral).
**Config:** Scout rag_simple N=100
**Result:** RAG helps: mean CE=-2.06. RAG hurts: mean CE=-2.89. Neutral: mean CE=-0.88. All negative — cross-encoder considers every retrieved passage irrelevant.
**Verdict:** CONFIRMED — retrieval quality is the bottleneck. The debiased prompt makes the LLM ignore bad passages (good), but the passages aren't contributing useful information. The +5pt at N=100 was the LLM reasoning well *despite* the passages, not *because of* them.
**Implication:** Score thresholding (drop passages below CE=0) would eliminate downside risk. But the real fix is improving retrieval quality.
**Commit:** 8f85a0c

### 2026-03-24 — Retrieved passages vs golden across models
**Hypothesis:** Models that integrate golden evidence well (Llama 70B +17) will also benefit more from retrieved passages.
**Change:** Ran rag_simple (debiased) across Scout, Llama 70B, and GPT 5.4-nano.
**Config:** N=100, seed=42, dense-only retrieval (no BM25), k=5
**Results:**
| Model | llm_only | golden_passage | rag_simple | golden lift | rag lift |
|---|---|---|---|---|---|
| Llama 70B | 64% | 81% | 73% | +17 | +9 |
| Scout 17B | 64% | 73% | 69% | +9 | +5† |
| GPT 5.4-nano | 57% | — | 59% | — | +2 |
†Not validated at N=200.
**Verdict:** CONFIRMED — larger models extract more value from both golden and retrieved evidence. Llama 70B rag_simple (73%) reaches Scout's golden ceiling. GPT 5.4-nano barely benefits (+2).
**Commit:** 8f85a0c

### 2026-03-24 — Prompt debiasing: "reason first" vs "ground in passages"
**Hypothesis:** The old "ground claims in evidence passages" prompt was causing the RAG regression by making the LLM defer to irrelevant passages. Changing to "reason first, evidence supports" would fix this.
**Change:** Rewrote synthesize_and_cite.md, synthesizer.md, golden_passage prompt, and RAG eval prompts. Unified all evidence-facing prompts to the same "reason independently first" pattern.
**Config:** Scout 17B, N=100
**Result:** rag_simple went from 58% (old prompt, skill-based) to 69% (debiased, consistent prompt). +11 point swing from prompt change alone.
**Verdict:** CONFIRMED — prompt framing was the primary cause of RAG regression, not retrieval quality. "Ground in passages" told the LLM to anchor on irrelevant evidence. "Reason first" lets it ignore bad passages.
**Caveat:** Also fixed a confound where RAG modes used a different system prompt (skill file) than llm_only. Part of the +11 may be from prompt consistency, not just debiasing.
**Commit:** 8f85a0c

### 2026-03-24 — Query rewriting adds no value for Scout
**Hypothesis:** LLM query rewriting would improve retrieval quality and downstream accuracy over raw questions.
**Change:** Compared rag_simple (raw question) vs rag_rewrite (LLM rewrite + alternatives) on Scout.
**Config:** Scout 17B, N=100, seed=42
**Result:** Both 69.0%. Rewrite uses 2 LLM calls vs 1, double the latency (3.2s vs 1.8s).
**Verdict:** REFUTED for Scout — query rewriting is wasted cost. The raw question retrieves equally well.
**Caveat:** Untested on Llama 70B which may benefit differently. Earlier aspect-based rewriting showed 2x better CE scores (from Mar 22 experiments).
**Commit:** 8f85a0c

### 2026-03-24 — Golden arbitration A/B: neutral vs conservative framing
**Hypothesis:** Two-phase approach (snap answer → review evidence) would outperform direct evidence injection, and conservative framing ("don't change unless strong reason") would beat neutral.
**Change:** Added golden_arbitration (neutral) and golden_arb_conservative eval modes.
**Config:** DeepSeek N=28 curated, then Scout/Llama 70B N=100
**Results (N=100):**
| Mode | Scout | Llama 70B |
|---|---|---|
| golden_passage | 73% | 81% |
| golden_arb_conservative | 71% | 80% |
| golden_arb_neutral (N=28 only) | 78.6% | — |
**Verdict:** MIXED — arbitration nearly matches golden_passage (1-2pts behind) but doesn't beat it. Conservative > neutral (confirmed on N=28). Architecture works but doesn't improve over simpler direct approach with golden evidence.
**Commit:** 8f85a0c

### 2026-03-24 — Model scouting for eval lineup
**Hypothesis:** Need models in the 50-70% range where evidence can make a measurable difference (DeepSeek too strong at 82%).
**Change:** Scouted 8 models across Groq, OpenAI, OpenRouter.
**Results:**
| Model | Size | llm_only | Viable? |
|---|---|---|---|
| DeepSeek | ~236B MoE | 82% | Too strong (already knows answers) |
| Llama 70B (Groq) | 70B | 64% | **Yes** — best evidence integration |
| Scout 17B (Groq) | 17B MoE | 64% | **Yes** — cheapest, fastest |
| GPT 5.4-nano (OpenAI) | ? | 57% | **Yes** — alt architecture |
| Qwen 32B (Groq) | 32B | 39-46% | Too weak + thinking model issues |
| Gemma 27B/4B | 27B/4B | 29-32% | Too weak |
| GPT OSS 120B (Groq) | 120B | 57% | No reasoning output (just letter) |
| Mistral Small (OR free) | 24B | ~18% | Broken (rate limits) |
**Verdict:** Locked in 3-model lineup: Scout (workhorse), GPT 5.4-nano (alt arch), Llama 70B (smart).
**Commit:** 8f85a0c

---

## Pre-lightweight-rebuild results (for reference)

### 2026-03-23 — Gemma 4B full pipeline
**Result:** 31% accuracy, avg 25 LLM calls/query. Small model can't do legal reasoning.
**Commit:** 3508b83

### 2026-03-22 — Agentic RAG (DeepSeek, parallel executor)
**Result:** 76% accuracy. Better than simple RAG (70%) but worse than LLM-only (85%).
**Commit:** 3e33801

### 2026-03-22 — RAG + query rewrite (DeepSeek)
**Result:** 80% accuracy, 8% gold recall@5. Best RAG approach at the time.
**Commit:** 2a8e46a

### 2026-03-22 — Baselines (DeepSeek, N=100)
**Result:** LLM-only 85%, Golden passage 77%, Simple RAG 70%, Retrieval recall 0%.
**Takeaway:** Golden passage below LLM-only on DeepSeek — model already knows the answers.

### 2026-03-22 — Query strategy comparison
**Result:** Aspect-based queries (rule/exception/application) get 2x better CE scores than raw queries. Gold recall 0% across all strategies.
