# LegalRagAgent: Experiment Summary

Complete record of all approaches tested, results, and lessons. 97 experiments across 5 datasets, 10+ models, 20+ pipeline modes.

---

## Baseline: What Models Know Without RAG

First question: how much do models already know?

| Model | BarExam | HousingQA | CaseHOLD | Notes |
|-------|---------|-----------|----------|-------|
| DeepSeek V3 | 79-82%* | — | — | Best overall |
| Llama 70B (Groq) | 64% | 47% | 72.5% | Main model |
| gpt-5.4-mini | 74%† | — | — | Strong baseline |
| Scout 17B (Groq) | 69% | 50% | — | Cheap, weaker |
| gpt-5.4-nano | 57%* | — | — | |
| gpt-4.1-nano | 46%* | — | — | |
| groq-qwen | 39-46%* | — | — | Poor |
| gemma 27B | 29-32%* | — | — | Very poor |
| or-phi4 | 0%* | — | — | Broken |
| groq-maverick | 0%* | — | — | Broken |

*N=28-100 small sample. †N=100.

**Upper bound**: Giving the model the exact relevant passage (golden_passage): Scout 73%, Llama 81%. This is the retrieval ceiling — best possible with perfect retrieval.

**Key finding**: Llama 70B at 64% leaves 17pts of room to the ceiling. Scout at 69% has less room. gpt-5.4-mini at 74% (untested with RAG) may behave differently.

---

## Phase 1: Basic Retrieval — Does RAG Help?

### rag_simple
Retrieve k=5 docs by raw query, answer with them.

| Model | N | Accuracy | vs baseline |
|-------|---|----------|-------------|
| Scout | 200 | 66-69% | ~0 |
| Llama 70B | 100 | 73% | +9 |

Early N=100 showed +5pt Scout lift — **sampling noise**. N=200 validated: rag_simple=68%, llm_only=69%. **Zero net benefit.** Root cause: 680K caselaw docs drown 2.3K MBE study passages. The retrieval corpus has severe imbalance.

### rag_rewrite
Rewrite query for better search, then retrieve.

| Model | Accuracy | vs rag_simple |
|-------|----------|--------------|
| Scout | 69% | ~0 |
| DeepSeek | 68%* | worse than llm_only |

No gain. Query rewriting doesn't fix corpus imbalance.

### rag_arbitration
Retrieve, answer, then arbitrate between RAG answer and llm_only answer.

| Model | Accuracy |
|-------|----------|
| Scout | 68% |

Not better. Arbitration doesn't resolve bad evidence — the model can't tell which answer is right.

---

## Phase 2: HyDE — Better Retrieval via Hypothetical Documents

Key insight: embed a hypothetical answer rather than the question. Matches the semantic space of actual passages better.

### rag_hyde
Generate one HyDE document, retrieve, answer.

| Model | BarExam | vs llm_only |
|-------|---------|-------------|
| Scout | 69% N=200 | 0 |
| Llama 70B | 75% N=100 | +11 |
| gpt-5.4-nano | 64% N=100 | +7 |

HyDE helps stronger models. Hypothetical document quality depends on what the model already knows — circular but effective.

### rag_multi_hyde
Multiple HyDE docs with different perspectives, take best.

| Model | Accuracy | vs rag_hyde |
|-------|----------|------------|
| Llama 70B | 74%* | -1 |

No improvement. Multiple mediocre hypotheticals don't beat one good one.

### rag_hyde_arb
HyDE + arbitration between RAG answer and direct answer.

| Model | Accuracy | vs rag_hyde |
|-------|----------|------------|
| Llama 70B | 74%* | -1 |
| Scout | 71%* | +2 |

Hurts strong models, minor help for weak. Arbitration is net negative when retrieval quality varies.

---

## Phase 3: Snap-HyDE — The Key Breakthrough

**Insight**: generate a quick "snap" answer first using CoT, then use THAT as the HyDE document. The snap answer encodes the model's reasoning about the question — a far better retrieval query than the raw question.

### rag_snap_hyde

| Model | Dataset | Accuracy | vs llm_only |
|-------|---------|----------|-------------|
| Llama 70B | BarExam N=200 | **76.5%** | +12.5 |
| Llama 70B | BarExam N=100 | 82% | +18 |
| Scout 17B | BarExam N=200 | 71% | +2 |
| gpt-5.4-nano | BarExam N=100 | 64% | +7 |
| Llama 70B | HousingQA N=200 | **56.0%** | **+9** |
| Llama 70B | CaseHOLD N=200 | 71.0% | -1.5 |
| Llama 70B | Legal-RAG-QA | 99.3% | -0.7 |
| Llama 70B | Australian | 100% | 0 |

**snap_hyde beats the golden passage ceiling on Llama 70B at N=100** (82% vs 81%). The snap CoT itself contains the reasoning — retrieval reinforces an already-correct answer.

N=200 BarExam drops to 76.5% — the N=100 result was partially noise, but technique is clearly best retrieval-only method.

**HousingQA +9 lift validated at N=200.** This is where RAG genuinely helps — model lacks knowledge of state-specific housing statutes. HousingQA also shows a systematic Yes-bias (model predicts Yes 81% of the time; correct is No 65% of the time). snap_hyde retrieval partially corrects this bias.

**CaseHOLD -1.5pts.** RAG pulls similar-but-wrong legal holdings. The corpus of 50K holdings confuses the model more than it helps.

### rag_devil_hyde
Generate a "devil's advocate" HyDE — argue the opposite of the snap answer.

| Model | Accuracy | vs snap_hyde |
|-------|----------|-------------|
| Llama 70B | 76%* | **-6** |

Retrieves counterevidence that consistently misleads. Never use.

### rag_top2_hyde
Retrieve two different documents using two HyDE angles.

| Model | Accuracy | vs snap_hyde |
|-------|----------|-------------|
| Llama 70B | 79%* | **-3** |

Additional context from a second angle adds noise. Note: this N=100 result also includes snap answer quality gains vs N=100 snap_hyde.

---

## Phase 4: Adaptive Routing — When to Use RAG

Core problem: snap_hyde always retrieves, but sometimes retrieval hurts. Can we route selectively?

### confidence_gated
Run question 3 times (self-consistency vote). All 3 agree → skip RAG. Any disagreement → snap_hyde.

| Model | Dataset | Accuracy | vs snap_hyde | Skip% |
|-------|---------|----------|-------------|-------|
| Llama 70B | BarExam N=200 | **79.0%** | **+2.5** | 77% skip |
| Llama 70B | HousingQA N=200 | 50.5% | -5.5 | 90% skip |
| Llama 70B | CaseHOLD N=200 | 72.5% | +1.5 | ~? |
| Scout 17B | BarExam N=200 | 71.5% | +0.5 | 60% skip |
| Scout 17B | HousingQA N=200 | 53.5% | -0.5 | ~? |

**BarExam win**: Model is confident AND wrong on only 23% of questions. On those 23%, RAG helps (+11pts within that segment). The 77% skip doesn't hurt.

**HousingQA disaster**: Model has systematic Yes-bias — unanimously confident in wrong answers. 90% of questions skip RAG. Result is barely better than llm_only.

```
HousingQA answer distribution (N=200):
  llm_only:        predicts YES:163, NO:37  (ground truth YES:71, NO:129)
  snap_hyde:       predicts YES:119, NO:81  — retrieval helps de-bias
  confidence_gated: predicts YES:160, NO:40  — skips 90% → reverts to bias
```

Lesson: confidence routing works for **random** knowledge gaps. It fails for **systematic** error modes.

### ce_threshold (current best)
Generate snap answer, run HyDE retrieval, score docs with cross-encoder. If max CE score < 4.0 → discard retrieval, use snap. Otherwise → answer with evidence.

| Model | Dataset | Accuracy | vs conf_gated | Routing |
|-------|---------|----------|--------------|---------|
| Llama 70B | BarExam N=200 | **80.0%** | +1.0 | 87 snap, 113 RAG |
| Scout 17B | BarExam N=200 | 71.5% | 0 | 132 snap, 68 RAG |
| Llama 70B | CaseHOLD N=200 | 71.0% | -1.5 | 193 snap, 7 RAG |

**Why CE beats confidence_gated**: Cross-encoder score is a direct measure of retrieval quality. Self-consistency is a proxy for model confidence — which doesn't detect systematic biases. CE asks "is this evidence actually relevant?" not "does the model agree with itself?"

```
Accuracy by routing path (Llama BarExam, N=200):
  snap_only (CE<4.0): 87 questions at 77.0%
  RAG (CE>=4.0):     113 questions at 82.3%
```

When evidence is good (CE>=4.0), RAG gives 82.3% accuracy. When it's bad, snap gives 77%. Threshold routing captures both.

**Flip analysis vs confidence_gated** (200 shared questions):
- CE right, CG wrong: +13 questions (CE routes to RAG when CG would have skipped, and the evidence helps)
- CG right, CE wrong: -11 questions (CG skipped RAG correctly, CE retrieved and it hurt)
- Net: +2 questions = +1% accuracy

### conf_ce_threshold
Combined routing: require both high confidence AND good CE score.

| Model | Accuracy | vs ce_threshold |
|-------|----------|----------------|
| Llama 70B BarExam | 76.5% | **-3.5** |

Routing: 22 RAG, 155 skip_rag, 23 snap_ce_fallback.

**Worse**. The two signals create dead zones — questions where confidence says skip but CE says evidence is good (or vice versa). Validated components don't compose additively.

---

## Phase 5: Question Decomposition

Can breaking a legal question into sub-questions improve reasoning?

### decompose (structured)
LLM decomposes the question into 2-3 sub-questions, answers each, then synthesizes.

| Model | Dataset | Accuracy | vs llm_only |
|-------|---------|----------|-------------|
| Llama 70B | BarExam | **76.0%** | +12 |
| Llama 70B | HousingQA | 48.5% | +1.5 |
| Llama 70B | CaseHOLD | 71.5% | -1 |
| Scout 17B | BarExam | 75.0% | +6 |
| Scout 17B | HousingQA | 59.0% | +9 |
| Scout 17B | CaseHOLD | 73.0% | ~0 |

Decomposition helps — it's chain-of-thought at the question level. On par with snap_hyde for BarExam (76% vs 76.5%). Weaker on HousingQA. Scout benefits more.

### decompose_rag
Decompose → retrieve per sub-question with snap-HyDE → synthesize all retrieved evidence.

| Model | Dataset | Accuracy | vs decompose |
|-------|---------|----------|-------------|
| Llama 70B | BarExam | 71.5% | **-4.5** |
| Llama 70B | CaseHOLD | 70.5% | -1 |
| Scout 17B | BarExam | 70.5% | **-4.5** |
| Scout 17B | CaseHOLD | 73.5% | +0.5 |

**Dead end.** Adding retrieval to decomposed sub-questions makes everything worse.

Flip analysis (decompose vs decompose_rag, BarExam N=200):
- decompose right, decompose_rag wrong: 23 questions
- decompose_rag right, decompose wrong: 14 questions
- Net: -9 questions

Sub-question count: 107/200 questions (54%) decompose to just 1 sub-question — the decomposition adds no value. For these, synthesis step adds overhead with zero upside.

Example failure: mbe_463 (Evidence question)
- decompose_rag decomposes to 1 sub-question (the full question verbatim), retrieves hearsay exception docs, synthesis loses track of the correct answer (C) and outputs B.
- Plain decompose gets it right because the structured reasoning path doesn't get distracted by retrieved snippets.

---

## Phase 6: Post-Answer Verification

After getting an answer, can we verify or improve it?

### self_verify
Answer question → verify the answer → produce final answer. Model second-guesses itself.

| Model | Dataset | Accuracy | vs snap_hyde |
|-------|---------|----------|-------------|
| Llama 70B | BarExam | 72.5% | **-4** |
| Llama 70B | CaseHOLD | 72.5% | +1.5 |
| Scout 17B | BarExam | **58.5%** | **-12.5** |

**Destructive on BarExam.** Second-guessing without new information makes things worse. Scout collapses catastrophically — it has insufficient confidence to maintain correct answers through a verification step. Verification introduces doubt about correct answers more than it fixes wrong ones.

### snap_debate
Two adversarial perspectives argue for and against the snap answer, then reach a conclusion.

| Model | Dataset | Accuracy | vs snap_hyde |
|-------|---------|----------|-------------|
| Llama 70B | BarExam | 72.0% | **-4.5** |
| Scout 17B | BarExam | 64.0% | **-7** |

Worse than self_verify. Adversarial review makes the model actively argue against its correct reasoning. The debate format systematically introduces doubt.

### double_snap
Run snap answer twice, compare via CE score, route to RAG or use agreement.

| Model | Dataset | Accuracy | Notes |
|-------|---------|----------|-------|
| Llama 70B | BarExam | 74.0% | Works — snap_agree:172/200, rag:16 |
| Llama 70B | CaseHOLD | **3.5%** | **API spend-alert hit mid-run, 187/200 errors** |

BarExam run was functional (74%) but below ce_threshold (80%). The CaseHOLD 3.5% is an infrastructure failure — an API spend-alert blocked 187/200 calls. Not a pipeline failure.

---

## Phase 7: Architectural Experiments

### full_pipeline (planner + snap-HyDE + synthesizer)
Full LangGraph pipeline: planner decomposes query into steps, snap-HyDE retrieves per step, synthesizer recombines.

| Model | Dataset | Accuracy | vs ce_threshold |
|-------|---------|----------|----------------|
| Llama 70B | BarExam | 76.0% | **-4** |

**Pipeline overhead costs 4 points.** Each step introduces loss: planner decomposition loses context, synthesizer recombination loses signal. Individual atomic operations beat the full pipeline by 4%.

Lesson: "build from atoms, not architecture." ce_threshold (snap → retrieve → CE-gate → answer) is 4 LLM calls. full_pipeline is 8-12 calls and performs worse.

### snap_hyde_aspect
Aspect-based sub-queries (rule / exception / application) instead of a single HyDE document.

| Model | Dataset | Accuracy | vs snap_hyde |
|-------|---------|----------|-------------|
| Llama 70B | BarExam | 76.0% | -0.5 |

Offline retrieval tests showed 2x better CE scores for aspect queries vs synonym variants. End-to-end: no improvement. Better retrieval quality doesn't transfer when snap answer quality is already high.

### ce_threshold_k3 vs k=5

| k | Accuracy | Verdict |
|---|----------|---------|
| k=5 | 80.0% | Keep |
| k=3 | 79.0% | Discard |

k=5 marginally better.

---

## Cross-Dataset Summary

| Dataset | Format | Corpus | LLM error type | Best mode |
|---------|--------|--------|----------------|-----------|
| BarExam | 4-way MC | 686K caselaw+MBE | Random knowledge gaps | ce_threshold (80%) |
| HousingQA | Yes/No | 1.8M housing statutes | Systematic Yes-bias | snap_hyde always-on (56%) |
| CaseHOLD | 5-way MC | 50K case holdings | Model exceeds corpus | llm_only (72.5%) |
| Legal-RAG-QA | Open-ended | 190 docs | Trivial | llm_only (100%) |
| Australian | Open-ended | 2.1K | Trivial | llm_only (100%) |

**The central finding**: RAG value is entirely a function of whether the LLM has the knowledge. HousingQA is the only dataset with validated lift at N=200 — it's the only one with genuinely obscure state-specific information the model doesn't know.

BarExam lift is substantially from the snap-answer CoT process (+12.5pts), not retrieved documents. Ablation: snap_hyde vs llm_only shows roughly half the "RAG lift" is from the snap CoT itself, not retrieval.

---

## BarExam Subject Breakdown (ce_threshold, Llama 70B, N=200)

| Subject | Correct | Total | Accuracy |
|---------|---------|-------|----------|
| CONST. LAW | 13 | 14 | 93% |
| CRIM. LAW | 11 | 12 | 92% |
| CONTRACTS | 18 | 21 | 86% |
| EVIDENCE | 14 | 17 | 82% |
| REAL PROP. | 13 | 17 | 76% |
| TORTS | 14 | 20 | 70% |
| Unlabeled | 77 | 99 | 78% |

Torts and Real Property are weakest. Torts questions often involve fine-grained comparative/contributory negligence distinctions that retrieval doesn't resolve well.

17 questions that ALL 7 tested modes get wrong — no approach helps them. These require reasoning the caselaw corpus doesn't contain, or jurisdiction-specific rules that contradict model priors.

---

## What Worked

| Mode | Best Result | Why It Works |
|------|------------|--------------|
| snap_hyde | 76.5% BarExam, 56% Housing | Snap CoT makes better retrieval query |
| confidence_gated | 79.0% BarExam | Self-consistency routes random errors to RAG |
| ce_threshold | **80.0% BarExam** | CE score directly measures evidence quality |
| decompose_structured | 76.0% BarExam | Chain-of-thought at question level |

## What Didn't Work

| Approach | Degradation | Why It Failed |
|----------|-------------|--------------|
| rag_simple/rewrite | ~0, sometimes negative | Corpus imbalance drowns relevant docs |
| rag_devil_hyde | -6pts | Counterevidence actively misleads |
| rag_top2_hyde | -3pts | Second angle adds noise |
| decompose_rag | -4.5pts | Synthesis step loses signal; fallback path adds overhead with no benefit |
| self_verify | -4 to -12pts | Second-guessing without new info = wrong answers become uncertain |
| snap_debate | -4.5 to -7pts | Adversarial format introduces systematic doubt in correct answers |
| conf_ce_threshold | -3.5pts | Components don't compose; dead zones |
| full_pipeline | -4pts | Planner + synthesizer = lossy overhead |
| snap_hyde_aspect | -0.5pts | Offline retrieval gains don't transfer end-to-end |
| confidence_gated on HousingQA | -5.5pts | Systematic bias → 90% skip RAG → reverts to yes-bias |

---

## Key Principles

1. **RAG value = f(knowledge gap)** — test each domain; never assume RAG helps
2. **Atomic > architectural** — ce_threshold (4 calls) beats full_pipeline (10+ calls) by 4pts
3. **Self-correction without new info is destructive** — self_verify, snap_debate all hurt
4. **Match routing to error mode** — random errors → gated routing; systematic bias → always-on
5. **The snap answer IS the research** — snap CoT encodes model reasoning; retrieval reinforces, not replaces
6. **Components don't compose additively** — conf+CE creates dead zones; decompose+RAG adds overhead
7. **Always validate at N=200** — N=100 results are unreliable; N=28 is nearly meaningless
8. **Upper bound anchors expectations** — golden passage accuracy (~81% Llama) sets the ceiling

---

*Generated 2026-03-30. Source: logs/experiments.jsonl (97 runs). See EXPERIMENTS.md for per-run details.*
