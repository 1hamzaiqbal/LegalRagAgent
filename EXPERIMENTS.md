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

### HousingQA baselines (N=100, seed=42)

| Mode | Llama 70B | Scout 17B |
|---|---|---|
| llm_only | 44% | 50% |
| rag_snap_hyde | **61% (+17)** | **54% (+4)** |

Note: HousingQA is Yes/No format, 65% No / 35% Yes class imbalance. LLM has massive Yes-bias (80% pred Yes). Retrieval against housing_statutes (1.8M docs, dense-only) dramatically reduces this bias. RAG lift much larger for Llama (+17) than Scout (+4), paralleling BarExam pattern.

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
