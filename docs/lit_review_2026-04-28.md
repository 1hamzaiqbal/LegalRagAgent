# Literature Review — Snap-Conditioned Retrieval & Bottleneck-Typed RAG

**Date:** 2026-04-28
**Reviewer:** Claude (Opus 4.7) for the LegalRagAgent paper sprint
**Working narrative being tested:** "Snap-conditioning is a model-agnostic RAG primitive whose lift mechanism is dataset-bottleneck-dependent (retrieval-bottlenecked MuSiQue vs reasoning-bottlenecked BarExam)."

---

## 1. TL;DR

**The exact framing — "single architectural primitive (snap-then-HyDE), opposite mechanisms on multi-hop vs legal MC, bottleneck-typed"  — is novel as a packaged claim.** The two halves exist in the literature but have never been combined:

- The **snap-conditioned-retrieval primitive** has close cousins (FLARE 2023, Speculative RAG 2024, Iter-RetGen 2023, GenRead 2022, Query2Doc 2023), but none package "single LLM call producing snap + HyDE-style passage, then retrieval, then a 2nd synthesizing call" exactly. The closest is **FLARE (Jiang et al., EMNLP 2023)** which generates a draft sentence then retrieves on it, but it does so iteratively token-by-token, not as a one-shot snap+HyDE pair.
- The **"task-dependent retrieval helps/hurts" observation** is widely reported (Vaddi 2026 on CaseHOLD/ContractNLI; Zheng et al. 2025 on BarExam) but has **never been operationalized as a *bottleneck taxonomy* that PREDICTS which RAG primitive will win.** Existing taxonomies (GRADE 2025, Know-Your-RAG 2025) classify *questions* by hop count or semantic distance, not *datasets* by which bottleneck dominates.

**Bottom line:** The team has a genuine gap to exploit, but the framing must lead with the *bottleneck-as-predictor* angle, not the snap-HyDE primitive (which a sharp reviewer can compress to "FLARE-lite + answer-conditioned HyDE"). The paper is publishable if the empirical claim is "one architecture, two datasets, two mechanisms, one taxonomy" — not "we invented snap-HyDE."

---

## 2. The 3 closest priors

### Prior 1: FLARE — Active Retrieval Augmented Generation (Jiang, Xu, Gao, Sun, Liu, Dou, Wang, Vu, Wei, Callan; EMNLP 2023)

- **URL:** <https://aclanthology.org/2023.emnlp-main.495/> (arXiv 2305.06983)
- **One-sentence claim:** Generates a tentative *next sentence* of the answer and uses it as a retrieval query whenever it contains a low-confidence token, then regenerates.
- **What's DIFFERENT from snap_hyde_2call:**
  1. FLARE works at the **token / sentence level inside long-form generation** (essay-style outputs); the team's snap_hyde_2call is question-answer scoped (one snap per question).
  2. FLARE retrieves *only when* a low-probability token is detected — the team's design always snap-conditions retrieval.
  3. FLARE iterates an unbounded number of times; snap_hyde_2call fixes call count to 2 (snap+HyDE in one call, synth in the second).
  4. FLARE never claims a bottleneck taxonomy; it's pitched as efficiency for long-form, not as a primitive that wins on opposite-bottleneck benchmarks.
  5. FLARE has not been evaluated on legal MC.

### Prior 2: Speculative RAG — Enhancing Retrieval Augmented Generation through Drafting (Wang et al., ICLR 2025)

- **URL:** <https://arxiv.org/abs/2407.08223>
- **One-sentence claim:** A small specialist LM produces multiple drafts in parallel from disjoint document subsets, and a larger generalist verifies and picks the best draft.
- **What's DIFFERENT from snap_hyde_2call:**
  1. **Drafts come AFTER retrieval, not before.** The "draft" never conditions the retrieval query — Speculative RAG is fundamentally a verification framework, not a retrieval-conditioning one.
  2. Two-model architecture (specialist + generalist); snap_hyde_2call uses a single model.
  3. Tested on MuSiQue (yes), TriviaQA, PopQA, PubHealth, ARC — but no legal benchmark.
  4. No bottleneck framing; pitched as "drafts diversify perspectives + reduce verification cost."

### Prior 3: A Reasoning-Focused Legal Retrieval Benchmark (Zheng, Guha, Arifov, Zhang, Skreta, Manning, Henderson, Ho; CS&Law Symposium 2025)

- **URL:** <https://arxiv.org/abs/2505.03970> (Stanford reglab — same group whose dataset the team uses)
- **One-sentence claim:** Introduces the BarExam QA + Housing Statute QA benchmarks; finds standard retrieval gives near-zero gain over zero-shot LLM on BarExam (≈+0.5pp on GPT-4o-mini), motivating retrievers that also reason.
- **What's DIFFERENT from the team's claim:**
  1. They observe the "retrieval barely helps on BarExam" finding but explain it as a *retriever-quality* problem ("retrievers must be legal reasoners too") and propose query-expansion fixes — they do NOT frame it as "BarExam is reasoning-bottlenecked, expect retrieval ceiling."
  2. They never evaluate any HyDE / snap / draft variant.
  3. They never compare to multi-hop QA (MuSiQue) and have no cross-domain bottleneck argument.
  4. Their headline is a benchmark + diagnosis, not a method that wins on both.
  5. **Critically: the team builds directly on their benchmark and is exactly the audience for a method paper that ANSWERS Zheng et al.'s open question.** This is a positioning gift — cite them as the gap statement.

---

## 3. Per-axis findings

### Axis 1 — Snap / answer-conditioned retrieval / draft-then-retrieve

The lineage:

| Paper | Year/Venue | Core idea | How it differs from snap_hyde_2call |
|---|---|---|---|
| HyDE (Gao et al.) | ACL 2023 | LLM generates hypothetical document, embed it, retrieve | Generates a *passage*, not a *snap answer + passage*. Doesn't condition on a committed answer. |
| Query2Doc (Wang, Yang, Wei) | EMNLP 2023, Microsoft | LLM generates pseudo-doc concatenated to query for BM25/dense retrieval | Same core, different concatenation; no snap commitment. |
| GenRead (Yu et al.) | ICLR 2023 | Generate doc *instead of* retrieving; clustering for diversity | Skips retrieval entirely; not a snap-then-retrieve method. |
| Iter-RetGen (Shao et al.) | EMNLP Findings 2023 | Iteratively use prior generation to inform next retrieval | Iterative, no explicit "snap commitment" semantics, no HyDE-style passage. |
| FLARE (Jiang et al.) | EMNLP 2023 | Predict next sentence → if low confidence, retrieve on it | **Closest prior** — but token/sentence level, conditional, no legal eval. |
| Speculative RAG (Wang et al.) | ICLR 2025 | Drafts from retrieved subsets verified by larger model | Drafts AFTER retrieval; not retrieval-conditioning. |
| Adaptive HyDE / "Never Come Up Empty" (Lei et al.) | arXiv 2507.16754, 2025 | Dynamic threshold relaxation on HyDE | Pure HyDE+retrieval-tuning; no snap, no bottleneck claim. |
| CoRAG (Wang et al., Microsoft) | arXiv 2501.14342, 2025 | Iteratively generate sub-queries from CoT, retrieve, refine | Trained, multi-step; no single-call snap+HyDE. |
| FrugalRAG (Java et al.) | arXiv 2507.07634, 2025 | ReAct + light RL to minimize search calls | Different objective (cost), not bottleneck framing. |
| HopRAG | ACL Findings 2025 | Multi-hop neighbor walks with pseudo-queries | Different — graph-walk not snap-condition. |

**Verdict on Axis 1:** Snap-then-HyDE-in-one-call has no exact prior with that packaging. Closest is FLARE (different granularity) and Iter-RetGen (different iteration semantics). The team's "snap commit + HyDE passage in a SINGLE LLM call" specifically is novel as packaging, but a reviewer will reasonably ask "isn't this just FLARE + HyDE?" — answer that pre-emptively.

### Axis 2 — Multi-hop QA SOTA on MuSiQue 2024-2026

There is no widely-adopted public leaderboard for full-corpus open-domain MuSiQue. Recent 2025 numbers from the literature, all on the *answerable* (MuSiQue-Ans) split:

| Method | Reported MuSiQue EM | Setup notes |
|---|---|---|
| Vanilla RAG (DPR+LM) baselines | 15-25% range | Common baseline range across CoRAG, Iter-RetGen, FAIR-RAG papers |
| Iter-RetGen | low-30s F1 (≈22% EM) | Older 2023 baseline |
| CoRAG (Wang et al. Microsoft 2025) | "10+ EM points over fine-tuned baselines" — full SOTA numbers vary by retriever | Trained, iterative |
| Speculative RAG | up to +12.97pp on PubHealth; MuSiQue gains modest | Drafts post-retrieval |
| StepChain GraphRAG | claims "+2.57 EM avg" over prior SOTA; absolute MuSiQue EM not extracted | Knowledge-graph routing |
| FAIR-RAG | F1 0.453 on HotpotQA, MuSiQue similar | Iterative refinement, late 2025 |
| FrugalRAG | competitive document recall with 2 searches | Cost-focused, mid-30s EM regime |

The team's **35.5% EM with `multi_hyde_diverse` on Llama 70b N=200** is in the same competitive ballpark as published SOTA-ish methods. It is **not embarrassingly behind**, but it is also **not unambiguously SOTA**. Critically, none of the SOTA papers test a fixed Llama 3.3 70b at the same N=200, so direct comparison is loose. **Frame as competitive-with-published-methods, not "we beat SOTA."**

### Axis 3 — Legal QA RAG 2025-2026

| Paper | Year | Finding |
|---|---|---|
| LegalBench-RAG (Pipitone, Alami) | arXiv 2408.10343, Aug 2024 | First legal RAG benchmark focused on retrieval precision; 6,858 query-answer pairs over 79M chars. Retrieval-only metrics. |
| A Reasoning-Focused Legal Retrieval Benchmark (Zheng et al., reglab) | CS&Law 2025 | Introduces BarExam QA + Housing QA; standard RAG gives ≈+0.5pp gain on GPT-4o-mini for BarExam; +23pp on Housing. **This is exactly the team's "BarExam is reasoning-bottlenecked" finding.** |
| Can Small Models Reason About Legal Documents? (Vaddi) | arXiv 2603.25944, Mar 2026 | Comparison study, no new method. Shows BM25/dense **identical** retrieval performance, RAG **+5pp on ContractNLI but -8pp on CaseHOLD** — concludes "task-dependent, not retriever-dependent, bottleneck is utilization." **Closest "task-dependent retrieval" framing in the literature.** |
| L-MARS (multi-agent legal) | arXiv 2509.00761, 2025 | Multi-agent reasoning + agentic search for legal; orthogonal to snap. |
| LRAGE | arXiv 2504.01840, 2025 | Toolkit, not method. |
| CBR-RAG | arXiv 2404.04302, 2024 | Case-based reasoning over legal QA. |

**Critical for the team:** Zheng et al. (2025) and Vaddi (2026) both observe the "RAG barely helps / hurts on legal MC" pattern. **Neither proposes snap-conditioning as the answer.** The team's contribution can plausibly be framed as "we explain WHY (reasoning bottleneck) and provide a method (snap-conditioning) that recovers a small but significant lift on the reasoning-bottlenecked side AND a large lift on the retrieval-bottlenecked side."

### Axis 4 — Bottleneck-typed RAG analysis

This is where the team has the cleanest gap.

| Paper | Year/Venue | What it does | What it doesn't do |
|---|---|---|---|
| Know Your RAG (IBM) | COLING 2025 | Taxonomy of {fact_single, summary, reasoning, unanswerable} for question difficulty | Question-level taxonomy; no DATASET-level "bottleneck" prediction; no method tied to taxonomy |
| GRADE (Generating multi-hop QA + difficulty matrix) | EMNLP Findings 2025 | 2D difficulty matrix: reasoning depth × semantic distance | Matrix-cell error rates as diagnostic; not "method X wins on bottleneck Y" |
| CBDR / SEAKR / SKILL-RAG / Rethinking LLM Parametric Knowledge | Various 2025 | Confidence-based dynamic retrieval — only retrieve when LLM is uncertain | Per-question gating, not a *dataset* taxonomy that predicts which RAG variant wins |
| When Iterative RAG Beats Ideal Evidence (arXiv 2601.19827) | 2026 | Bottleneck IS reasoning, not retrieval, in scientific multi-hop QA | Single-domain framing, no cross-dataset taxonomy |
| Adaptive-RAG (Jeong et al.) | NAACL 2024 | Per-question complexity routing (single-step / multi-step / no-retrieval) | Method, not predictive taxonomy of datasets |

**Verdict on Axis 4:** The literature has **per-question** difficulty taxonomies and **per-question** adaptive retrieval gating, but **no one has framed entire benchmarks as "retrieval-bottlenecked vs reasoning-bottlenecked" and then shown a single architecture wins on both via different mechanisms.** This is the team's strongest novelty axis. Lead with it.

### Axis 5 — Anything subsuming snap_hyde_2call

After reviewing FLARE, Speculative RAG, Iter-RetGen, GenRead, Query2Doc, CoRAG, HopRAG, FrugalRAG, FAIR-RAG, Adaptive HyDE, RankRAG, BlendFilter, DRAGIN, Demonstrate-Search-Predict (DSP):

**No paper packages all four properties together:**
1. Single LLM call produces both a snap answer AND a HyDE-style passage stating the controlling rule
2. Retrieval is run on the passage (not the question)
3. A second LLM call synthesizes the final answer using snap + retrieved evidence
4. Total cost is exactly 2 LLM calls (no iteration)

The closest packaging is **FLARE** (1, 2, 3 in spirit but iterative and token-level) and **Speculative RAG** (multi-draft + verifier but drafts come AFTER retrieval).

---

## 4. Gap analysis & paper-positioning recommendation

### Where the contribution sits

The paper-grade contribution is **not** the snap_hyde_2call mode in isolation — that's vulnerable to "this is FLARE + answer-conditioned HyDE" pushback. The contribution **is**:

> A **dataset-bottleneck taxonomy** (retrieval-bottlenecked vs reasoning-bottlenecked) that **predicts** which RAG primitive wins, demonstrated via a **single architecture** that exploits BOTH bottlenecks through DIFFERENT mechanisms — the snap commits to controlling-rule reasoning (helping the reasoning-bottlenecked side) AND seeds a HyDE passage that recovers retrieval recall (helping the retrieval-bottlenecked side).

### Suggested abstract framing (2-3 sentences the team can lift)

> "Recent work shows that retrieval-augmented generation (RAG) helps some QA tasks and hurts others, but the literature has framed this asymmetry as a *retriever quality* problem to be fixed with better embeddings or query expansion. We argue instead that QA benchmarks fall on a **bottleneck spectrum** — *retrieval-bottlenecked* (multi-hop, MuSiQue) vs *reasoning-bottlenecked* (legal MC, BarExam) — and that this typology *predicts* which RAG primitives will win. We introduce **snap-conditioned retrieval**, a single two-call architecture in which the LLM commits to a tentative answer and a HyDE-style controlling-rule passage in one call, retrieves on the passage, and synthesizes in a second call. The same architecture lifts a Llama 3.3 70B baseline by **+8.0pp** on MuSiQue (p=0.0195) and **+3.1pp** on BarExam (cross-model-confirmed at Gemma 4 26B and E4B), and we show via ablation that the two lifts come from *opposite* mechanisms — retrieval recall on multi-hop, reasoning anchoring on legal MC."

### Why this works

- **Cites Zheng 2025 and Vaddi 2026 as the gap statement** (both noticed the asymmetry, neither explained or fixed it).
- **Differentiates from FLARE/Speculative RAG** (different granularity + the cross-domain bottleneck story is the actual contribution, not the primitive).
- **Differentiates from GRADE/Know-Your-RAG** (they classify questions, you classify benchmarks and *predict* method choice).
- **Survives the "is this just HyDE?" critique** (the answer is "no, the contribution is the bottleneck taxonomy and the cross-domain demonstration; the primitive is a vehicle").

---

## 5. Risks (most likely reviewer pushback)

1. **"This is FLARE + HyDE rebadged."** Most likely Tier-1 reviewer pushback. Mitigation: explicitly compare to FLARE in Section 2; show that snap_hyde_2call is fixed-cost (2 calls), question-scoped (not token-scoped), and unconditional (FLARE only retrieves on low-confidence tokens). Run an ablation against a faithful FLARE baseline if at all possible.

2. **"BarExam +3pp at N=1195 is small; +8pp on MuSiQue at N=200 is modest sample."** McNemar p=0.0195 saves the MuSiQue claim, but a reviewer will demand the **Gemma 3 27B N=200 NULL** be addressed honestly. The current CLAUDE.md notes this is +2.5pp p=0.5901 — that **kills** the universal claim and forces the framing to be "model-dependent on multi-hop, model-agnostic on legal." Either run the full-corpus replicate before submission OR caveat the title from "model-agnostic" to "demonstrated on Llama-class models."

3. **"You haven't beaten SOTA on MuSiQue."** True — CoRAG, FAIR-RAG, StepChain-GraphRAG all report stronger absolute numbers (in their paper setups). Frame the contribution as **mechanism-explanatory + cost-efficient**, not SOTA-chasing. 2 LLM calls is dramatically cheaper than CoRAG's iterative retrieval chains.

4. **"Bottleneck taxonomy is just two data points."** The paper has BarExam (reasoning) and MuSiQue (retrieval) — that's a 2x2 with one cell. Strongly recommend adding a third benchmark to fill the taxonomy: HousingQA (also legal, more retrieval-heavy → should fall closer to MuSiQue side) or HotpotQA (multi-hop but easier, intermediate). Without a third dataset the taxonomy looks like rationalized post-hoc storytelling around two experiments.

5. **"snap_only_in_final = 9.5% on MuSiQue is suspicious."** A reviewer who reads Table 1 will ask why pure snap collapses so badly on MuSiQue (compared to llm_only baseline). Be ready to explain — likely answer: snap's commitment is *wrong* on multi-hop because the model lacks the second-hop fact, and forcing the wrong commit hurts more than no commit. This is actually evidence FOR the bottleneck thesis (commit hurts when you need retrieval; commit helps when you need reasoning anchoring), but the team needs to articulate it.

6. **"Why two calls and not one?"** A reviewer aware of cost will ask if a single-call variant works. The team should be ready with at least an ablation: "we tried single-call snap+HyDE+synth; the model conflates the snap with the synth and ignores retrieved evidence." If untested, run this before submission.

7. **Vaddi 2026 (arXiv 2603.25944) just published in March 2026.** It is the closest "task-dependent retrieval" framing in print. The team **must** cite it and explicitly differentiate. Risk of looking like an unaware concurrent submission if missed.

---

## Sources / key references for the paper bibliography

- **HyDE:** Gao, Ma, Lin, Callan. "Precise Zero-Shot Dense Retrieval without Relevance Labels." ACL 2023. <https://arxiv.org/abs/2212.10496>
- **FLARE:** Jiang et al. "Active Retrieval Augmented Generation." EMNLP 2023. <https://aclanthology.org/2023.emnlp-main.495/>
- **Speculative RAG:** Wang et al. ICLR 2025. <https://arxiv.org/abs/2407.08223>
- **CoRAG:** Wang, Chen, Yang, Huang, Dou, Wei. arXiv 2501.14342, 2025. <https://arxiv.org/abs/2501.14342>
- **Iter-RetGen:** Shao et al. EMNLP Findings 2023. <https://arxiv.org/abs/2305.15294>
- **GenRead:** Yu et al. ICLR 2023. <https://arxiv.org/abs/2209.10063>
- **Query2Doc:** Wang, Yang, Wei. EMNLP 2023, Microsoft. <https://arxiv.org/abs/2303.07678>
- **DSP:** Khattab et al. arXiv 2212.14024, 2022.
- **Adaptive HyDE / "Never Come Up Empty":** Lei et al. arXiv 2507.16754, 2025.
- **DRAGIN:** Su et al. ACL 2024. <https://arxiv.org/abs/2403.10081>
- **HopRAG:** Liu et al. ACL Findings 2025. <https://arxiv.org/abs/2502.12442>
- **FrugalRAG:** Java, Koundinyan et al. arXiv 2507.07634, 2025.
- **RankRAG:** NVIDIA, NeurIPS 2024. <https://arxiv.org/abs/2407.02485>
- **BlendFilter:** EMNLP 2024. <https://aclanthology.org/2024.emnlp-main.58/>
- **A Reasoning-Focused Legal Retrieval Benchmark:** Zheng, Guha, Arifov, Zhang, Skreta, Manning, Henderson, Ho. CS&Law Symposium 2025 / arXiv 2505.03970 — **REQUIRED CITE.**
- **LegalBench-RAG:** Pipitone, Alami. arXiv 2408.10343, 2024. <https://arxiv.org/abs/2408.10343>
- **Can Small Models Reason About Legal Documents?:** Vaddi. arXiv 2603.25944, Mar 2026 — **REQUIRED CITE; closest "task-dependent" framing.**
- **L-MARS:** arXiv 2509.00761, 2025.
- **CBR-RAG:** arXiv 2404.04302, 2024.
- **Know Your RAG (taxonomy):** IBM, COLING 2025. <https://arxiv.org/abs/2411.19710>
- **GRADE (difficulty matrix):** EMNLP Findings 2025. <https://arxiv.org/abs/2508.16994>
- **Adaptive-RAG (per-question gating):** Jeong et al. NAACL 2024.
- **MuSiQue benchmark:** Trivedi et al. TACL 2022. <https://arxiv.org/abs/2108.00573>
