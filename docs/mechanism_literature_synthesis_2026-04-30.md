# Mechanism-Literature Synthesis - 2026-04-30

Purpose: connect landed results, recent RAG/legal-RAG literature, and the
remaining research gaps into a small set of mechanistic questions. This is not
a leaderboard or a method pitch.

## Short Thesis

The project should ask:

> When does retrieval help a legal or multi-hop QA system, and what observable
> evidence tells us whether the bottleneck is retrieval, evidence utilization,
> composition, or answer-option disambiguation?

This is stronger than "snap-HyDE is new." The literature already contains close
neighbors for pseudo-doc retrieval, active retrieval, adaptive retrieval, and
draft-verification RAG. The gap is that those methods mostly decide **how to
retrieve**, while our logs expose a prior question: **what kind of failure is
the task expressing?**

## What We Know Locally

Primary local evidence:

- `docs/evidence_matrix_2026-04-30.md`
- `docs/musique_disagreement_audit_2026-04-30.md`
- `docs/casehold_flatness_audit_2026-04-30.md`
- `docs/scalr_depth_disagreement_2026-04-30.md`
- `docs/search_space_consolidation_2026-04-30.md`
- `docs/router_probe_findings_2026-04-30.md`

Observed regimes:

| Regime | Local signal | Mechanism read |
|---|---|---|
| MuSiQue Llama 70B | top-1 collapse from 27.5% to 13.0%; `two_call` 37.0%; `mhd` 35.5%; four-arm oracle 57.0%. | Retrieval depth and query formulation matter, but methods rescue only partially overlapping rows. |
| BarExam Gemma 4 26B | top-1/top-5 flat at N=200; full-N `rag_snap_hyde` +3.09pp over `rag_simple`. | More documents are not the obvious bottleneck; useful effect is more plausibly answer-option anchoring or legal-rule formulation. |
| LegalBench-SCALR Llama 70B | top-1 59.5%, top-5 77.0%, top-10 77.0%; gold-hit rises with k but accuracy saturates at top-5. | Candidate depth matters up to a small set; after that, additional retrieved evidence is not converted into answer gain. |
| CaseHOLD Llama 70B | top-5 72.0%, top-1 70.5%, `two_call` 69.5%; all current logs show 0/200 `gold_retrieved`. | Answer-level flatness is real, but retrieval-recall instrumentation is currently untrustworthy for this dataset. |

The connective tissue: RAG is not one intervention. It bundles retrieval
coverage, query formulation, evidence ordering, evidence utilization, answer
calibration, and sometimes agent coordination. Our results look contradictory
only if those are collapsed into a single "RAG helps" axis.

## Literature Boundary

### 1. Pseudo-doc and active retrieval are established

- HyDE pivots through hypothetical documents for zero-shot dense retrieval:
  <https://arxiv.org/abs/2212.10496>.
- Query2Doc uses LLM-generated pseudo-documents for query expansion and reports
  BM25 gains on ad-hoc retrieval: <https://arxiv.org/abs/2303.07678>.
- FLARE uses a predicted upcoming sentence as the retrieval query during
  generation, with retrieval triggered by low-confidence tokens:
  <https://aclanthology.org/2023.emnlp-main.495/>.
- DRAGIN decides when and what to retrieve from the model's real-time
  information needs: <https://aclanthology.org/2024.acl-long.702/>.

Implication: do not sell `snap_hyde_2call` as the central novelty. It is a
useful probe because it is fixed-cost and exposes whether tentative answer
formation helps or hurts. The bigger contribution is the diagnostic frame.

### 2. Adaptive RAG exists, but usually routes by question complexity

- Adaptive-RAG routes among no retrieval, single-step retrieval, and multi-step
  retrieval based on question complexity:
  <https://arxiv.org/abs/2403.14403>.
- Self-RAG trains reflection tokens to retrieve on demand and critique
  retrieved passages/generation: <https://arxiv.org/abs/2310.11511>.
- Speculative RAG drafts from retrieved document subsets and verifies with a
  larger model: <https://proceedings.iclr.cc/paper_files/paper/2025/hash/2ea06b52f613716e67458f5ab3fb7558-Abstract-Conference.html>.

Implication: a router is not novel by itself. The sharper question is whether
cheap observables can identify a **bottleneck regime** rather than a generic
"complexity" label. Our current offline router has oracle headroom but weak
leave-one-dataset-out behavior, so the path forward is richer mechanism
features, not immediate online routing.

### 3. RAG evaluation is moving toward diagnostic matrices

- GRADE models multi-hop QA difficulty along reasoning depth and semantic
  distance to evidence: <https://aclanthology.org/2025.findings-emnlp.236/>.
- Know Your RAG argues that dataset composition affects RAG evaluation and
  proposes label-targeted data generation:
  <https://arxiv.org/abs/2411.19710>.

Implication: our natural contribution is not "another RAG mode"; it is a
diagnostic matrix for assigning tasks to failure regimes, then choosing the
lightest intervention that matches the regime.

### 4. Legal RAG literature is already asking for decomposition

- Zheng et al. introduce Bar Exam QA and Housing Statute QA and explicitly
  separate retriever reasoning from downstream LLM use of retrieved passages:
  <https://law.stanford.edu/wp-content/uploads/2025/03/3709025.3712219.pdf>.
- Legal RAG Bench uses factorial retrieval-model x generation-model evaluation
  and hierarchical error decomposition; its abstract says retrieval is often
  the primary driver of legal RAG performance:
  <https://arxiv.org/abs/2603.01710>.
- Vaddi 2026 compares prompting/RAG strategies across ContractNLI, CaseHOLD,
  and ECtHR, and reports near-identical BM25 vs dense RAG behavior, pointing
  toward context utilization rather than retrieval method alone:
  <https://arxiv.org/abs/2603.25944>.

Implication: the legal-RAG opening is real, but it is not "legal RAG needs
agents." It is: legal tasks disagree about whether the active bottleneck is
retrieval, reasoning over retrieved law, or selecting among legally plausible
answer options.

## Gap We Can Claim

The defensible gap:

> Existing adaptive RAG work routes retrieval based on uncertainty, generation
> state, or question complexity. Existing legal-RAG work shows that legal tasks
> stress both retrieval and reasoning. What is missing is a small, reproducible
> diagnostic that predicts **which intervention should help** before launching
> another expensive method variant.

This gives us three research questions.

### RQ1: Can top-k sensitivity diagnose retrieval-depth bottlenecks?

Evidence now supports this:

- MuSiQue and SCALR are top-k sensitive.
- BarExam and CaseHOLD are not, at least in the current slices.

The stronger version needs:

- HousingQA k=1/k=5/k=10, because Zheng et al. report Housing as a setting
  where retrieved passages substantially help downstream QA.
- CaseHOLD retrieval-only mapping, because current `gold_retrieved=0/200`
  prevents a true recall claim.

### RQ2: When does generated/pseudo evidence help versus distract?

Evidence now supports a split:

- MuSiQue benefits from answer/pseudo-doc query formulation under Llama 70B.
- SCALR does not benefit from `two_call` despite being top-k sensitive.
- BarExam shows a full-N snap-family lift, but top-k is flat.
- Zheng et al. report that a generated reasoning rollout can confuse BarExam
  while retrieved passages help HousingQA.

Hypothesis:

> Generated evidence helps when it improves retrieval targeting or legal-rule
> anchoring; it hurts when it adds distractor commitments in an option-selection
> or already-saturated candidate set.

Needed analysis:

- Per-record "same evidence, answer flip" buckets.
- Evidence overlap and answer-change metrics in the router table.
- A small manual audit of generated pseudo-docs in rescued vs harmed rows.

### RQ3: Can an agent improve by preserving evidence state instead of adding agents?

Evidence now argues against unstructured agenticity:

- `subagent_rag` is significantly negative on MuSiQue.
- Report/planning variants have not produced a durable legal MC lift over
  simpler controls.
- SCALR top-10 increases evidence/gold hits without increasing accuracy.

Hypothesis:

> The useful agentic primitive is not "more subagents"; it is a shared evidence
> ledger that records claims, source ids, support/contradiction, jurisdiction,
> and whether a retrieved item changed the answer.

This is where legal specificity can matter: legal answers need rule provenance,
jurisdiction/date discipline, and contradiction handling. A ledger is a
mechanism for preserving those constraints across retrieval and synthesis.

## Candidate Paper Frame

Title direction:

> Diagnosing When Retrieval Helps: Bottleneck-Aware RAG for Legal and Multi-Hop
> Question Answering

Core contribution:

1. A diagnostic matrix with observable probes:
   `llm_only`, k=1/k=5/k=10 RAG, single gold-passage control, pseudo-doc query
   formulation, answer-anchored controls, and disagreement buckets.
2. A cross-task mechanism study showing that the same intervention can help,
   be flat, or hurt depending on bottleneck regime.
3. A constrained agentic direction: evidence-budgeted routing plus shared
   evidence state, not unconstrained multi-agent decomposition.

Do not claim SOTA. Do claim that the project explains why a method family moves
some tasks and not others.

## Next Work That Directly Serves The Story

1. **Run manifest**: make every cited detail log carry `hypothesis`,
   `bottleneck_regime`, `intervention_axis`, `k`, `provider`, `cite_status`,
   and `known_caveat`.
2. **Router table v2**: add features from disagreement audits: answer change,
   gold newly retrieved at larger k, overlap with pseudo-doc retrieval, and
   "more evidence but harmed" flags.
3. **CaseHOLD retrieval-only mapping**: determine whether correct holding
   options are actually present in retrieved text; do this before more
   CaseHOLD LLM calls.
4. **HousingQA clean diagnostic slice**: k=1/k=5 plus `two_call`, on cluster
   only if `housing_statutes` is present.
5. **Manual mechanism audit**: 20 rescued/harmed rows across MuSiQue, SCALR,
   BarExam, CaseHOLD, labeling each as retrieval miss, evidence present but
   unused, distractor evidence, or answer-option ambiguity.

## What Would Make This Bigger

The most interesting version is an **adaptive diagnostic agent**:

1. It first runs cheap probes: no-context answer, k=1 retrieval, retrieval
   score/gold-hit proxies, answer uncertainty, and answer-change under small
   evidence perturbation.
2. It assigns a provisional bottleneck label.
3. It chooses the next intervention: deeper k, pseudo-doc query expansion,
   answer anchoring, contradiction search, or no further retrieval.
4. It writes all evidence and decisions into a ledger.

This would connect the literature threads:

- active/dynamic retrieval asks when/what to retrieve;
- difficulty taxonomies ask what kind of task we are facing;
- legal-RAG benchmarks ask whether retrieval or reasoning caused the error;
- our logs ask whether a cheap observable predicts the intervention that
  actually changes the answer.

The immediate paper can stop short of building the full agent. It only needs
to show that bottleneck labels are measurable, predictive enough to guide
evaluation, and necessary to interpret the mixed results that otherwise look
like method noise.
