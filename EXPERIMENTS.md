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
