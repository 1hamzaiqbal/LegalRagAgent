# Experiment Log

Running record of system changes and their eval results. Add new entries at the top.

## Format

```
### YYYY-MM-DD — Short description
**Change:** What was modified
**Config:** model, N queries, any relevant settings
**Result:** accuracy, key metrics
**Comparison:** vs previous best / baseline
**Takeaway:** what we learned
**Commit:** hash
```

---

### 2026-03-24 — Multi-model golden arbitration A/B (N=100)
**Change:** Ran golden_passage and golden_arb_conservative on N=100 across two models to test if evidence helps weaker models and if the two-phase arbitration approach works at scale.
**Config:** Llama 3.3 70B (Groq) and Llama 4 Scout 17B (Groq), N=100, seed=42
**Results:**
| Mode | Llama 70B | Scout 17B |
|---|---|---|
| llm_only | 64.0% | 64.0% |
| golden_passage | **81.0%** (+17) | 73.0% (+9) |
| golden_arb_conservative | **80.0%** (+16) | 71.0% (+7) |
**Takeaway:** (1) Evidence massively helps models that don't already know the answers — +17 pts for 70B vs DeepSeek where evidence was neutral. (2) Larger models integrate evidence ~2x better than smaller ones (same baseline, double the lift). (3) Conservative arbitration closely tracks golden_passage (1-2 pts behind) for both model sizes — the two-phase architecture works. (4) Llama 70B is the right model for our A/B tests — strong enough to reason but weak enough for evidence to matter. (5) Scout is fast and cheap but less capable at evidence integration.
**Next:** Test with *retrieved* passages instead of golden — does conservative arbitration protect the snap answer when evidence is noisy?
**Commit:** TBD (uncommitted)

### 2026-03-24 — Golden arbitration A/B + prompt debiasing (curated, N=28)
**Change:** (1) Rewrote all evidence-facing prompts to "reason first, evidence supports" — removed deference bias from golden_passage, synthesize_and_cite.md, synthesizer.md. (2) Added two new eval modes: `golden_arbitration` (neutral) and `golden_arb_conservative` — LLM answers naively first, then reviews golden passage.
**Config:** deepseek-chat, N=28 curated diagnostic set, 4 modes compared
**Results:**
| Mode | Accuracy | Latency/q | Calls/q | Notes |
|---|---|---|---|---|
| llm_only | 82.1% | 18.7s | 1 | Baseline (no evidence) |
| golden_passage (reasoning-first) | 82.1% | 17.9s | 1 | New prompt — up from 77% on old "use passage" prompt (N=100) |
| golden_arb_conservative | 82.1% | 25.3s | 2 | Snap + "don't change unless strong reason" |
| golden_arb_neutral | 78.6% | 25.8s | 2 | Snap + neutral review |
**Per-question flip analysis:** Arbitration correctly rescued mbe_1055 and mbe_1109 (snap wrong → evidence helped). But evidence actively misled on mbe_897 (both arb modes) and neutral framing lost mbe_935 by flipping a correct snap. mbe_806 and the nan question universally wrong.
**Comparison:** All modes cluster at 82.1% on this small set (vs prior 85.7% golden / 78.6% llm_only on same set with old prompts). Conservative > neutral by 1 question.
**Takeaway:** (1) Reasoning-first prompt fixed the "lazy LLM" golden passage regression — now tied with llm_only instead of below it. (2) Conservative arbitration framing is better than neutral — neutral is too willing to flip correct answers. (3) N=28 is too small for real separation; need N=100+. (4) Two-phase approach shows promise on specific questions but no top-line lift yet.
**Commit:** TBD (uncommitted)

### 2026-03-23 — Gemma 4B full pipeline
**Change:** Ran full parallel pipeline with Gemma 3 4B (via OpenRouter free tier)
**Config:** gemma-3-4b-it, N=100, full_parallel profile
**Result:** 31% accuracy, avg 25 LLM calls/query
**Comparison:** vs DeepSeek 85% LLM-only, 76% agentic
**Takeaway:** Small model can't do legal reasoning — pipeline can't compensate for weak base model. High escalation rate (judge marks most responses insufficient).
**Commit:** 3508b83

### 2026-03-22 — Agentic RAG (DeepSeek, parallel executor)
**Change:** Full parallel pipeline with planner, judge, escalation, completeness loop
**Config:** deepseek-chat, N=100, parallel executor
**Result:** 76% accuracy, 15% gold recall@5, ~82s/query
**Comparison:** vs 85% LLM-only (worse), vs 70% simple RAG (better)
**Takeaway:** Pipeline improves over simple RAG but still below LLM-only. Retrieval adds noise that sometimes overrides correct LLM intuition.
**Commit:** 3e33801

### 2026-03-22 — RAG + query rewrite (no pipeline)
**Change:** Added LLM query rewriting before retrieval (aspect-based rewrite)
**Config:** deepseek-chat, N=100, single rewrite + retrieve + answer
**Result:** 80% accuracy, 8% gold recall@5, ~34s/query
**Comparison:** vs 70% simple RAG (+10 points), vs 85% LLM-only (-5 points)
**Takeaway:** Query rewriting is the single biggest RAG improvement. Closes half the gap between simple RAG and LLM-only.
**Commit:** 2a8e46a

### 2026-03-22 — Simple RAG (relaxed synthesis prompt)
**Change:** Relaxed anti-fabrication constraint in synthesize_and_cite.md — allow LLM to supplement passages with training knowledge
**Config:** deepseek-chat, N=100
**Result:** 70% accuracy, 0% gold recall@5
**Comparison:** Same as strict prompt (70%). Prompt change didn't help.
**Takeaway:** Retrieved passages themselves anchor the LLM regardless of prompt instructions. The problem is retrieval quality, not the synthesis constraint.

### 2026-03-22 — Baselines established
**Change:** Ran all baseline evals on same 100-query sample
**Config:** deepseek-chat, N=100, seed=42
**Results:**
| Method | Accuracy | Gold Recall@5 | Latency/q |
|---|---|---|---|
| LLM-only | 85% | n/a | 17s |
| Golden passage | 77% | n/a | 4s |
| Simple RAG | 70% | 0% | 21s |
| Retrieval recall | n/a | 0% | 27s |
**Takeaway:** LLM-only is a strong baseline. Golden passage ceiling (77%) is below LLM-only (85%), suggesting some questions are better answered from model knowledge than from a single passage. Retrieval recall is 0% — gold passages are never in top-5 with raw questions.

### 2026-03-22 — Query strategy comparison
**Change:** Tested 5 query rewriting strategies on retrieval quality
**Config:** 10 questions, cross-encoder score comparison
**Results:**
| Strategy | Avg CE Score | Gold Recall |
|---|---|---|
| raw (no rewrite) | -2.3 | 0% |
| current (synonyms) | 3.0 | 0% |
| aspect (rule/exc/app) | 6.0 | 0% |
| decompose | 3.9 | 0% |
| abstract | 2.5 | 10% |
**Takeaway:** Aspect-based queries (targeting rule/exception/application separately) produce 2x better cross-encoder scores. This strategy should be the default rewriter.
