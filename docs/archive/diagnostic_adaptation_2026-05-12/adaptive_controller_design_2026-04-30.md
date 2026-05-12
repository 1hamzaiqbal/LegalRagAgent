# Adaptive Controller Design - 2026-04-30

Question: can the system quickly evaluate the task and adapt the RAG method on
the fly?

Short answer: yes, but it should be a constrained controller, not a free-form
"agent decides what to do" prompt. The current `adaptive_snap_route` experiment
already shows why: a single self-routing prompt is easy to bias and hard to
calibrate. The feasible path is a small, observable bottleneck router trained or
validated from the detail logs.

## Literature Pattern

Several papers already make the "route before spending compute" move:

- [Adaptive-RAG](https://aclanthology.org/2024.naacl-long.389/) routes among
  no-retrieval, single-step retrieval, and multi-step retrieval based on query
  complexity using a smaller classifier. Its limitation is that query complexity
  is not the same as bottleneck type.
- [Self-RAG](https://arxiv.org/abs/2310.11511) learns reflection tokens for
  retrieval-on-demand, relevance, support, and utility. This is conceptually
  right but training-heavy; the paper also notes prompt-only reflection is
  costly and brittle.
- [To Retrieve or Not to Retrieve?](https://arxiv.org/abs/2501.09292) evaluates
  uncertainty methods for dynamic retrieval and reports that uncertainty metrics
  can cut retrieval calls substantially with only slight accuracy loss. This is
  useful for skip/escalate decisions.
- [MBA-RAG](https://aclanthology.org/2025.coling-main.218/) treats retrieval
  strategies as bandit arms and optimizes a reward that balances accuracy and
  retrieval cost. This is the closest algorithmic fit for our harness.
- [Auto-RAG](https://arxiv.org/abs/2411.19443) fine-tunes a model to plan
  retrievals and stop when enough evidence has been gathered. This supports an
  agentic version, but it is more expensive than we need right now.
- [RAGRouter](https://arxiv.org/abs/2505.23052) and
  [RouteLLM](https://arxiv.org/abs/2406.18665) frame routing as learning which
  model/system is best for a query under cost constraints. The useful takeaway
  is not the exact model router; it is the offline preference-labeling setup.
- Newer domain-routing work, for example
  [Adaptive Query Routing](https://arxiv.org/abs/2604.14222), argues that no
  single retrieval paradigm dominates across query tiers and document
  structures. That matches our local logs.

The gap for this project: existing adaptive RAG usually routes on "complexity"
or "uncertainty." Our better unit is **bottleneck type**: retrieval depth,
query formulation, evidence composition, answer-option disambiguation, or
retrieval noise.

## Local Feasibility Signal

I added `scripts/evaluate_routing_oracle.py` to estimate routing headroom from
completed logs. It joins arms by question ID and computes an offline oracle.

Current generated artifacts:

- `docs/router_oracle_musique_2026-04-30.md`
- `docs/router_oracle_barexam_2026-04-30.md`

Key read:

| Dataset slice | Static best | Accuracy-first oracle | Interpretation |
|---|---:|---:|---|
| MuSiQue Llama 70b N=200: `rag`, `two_call`, `mhd`, `iter` | 37.0% | 57.0% | Large per-question method complementarity. A router is worth testing. |
| BarExam Gemma 4 26B N=200: `rag_top5`, `rag_top1`, `two_call` | 85.5% | 93.5% | Huge complementarity, but likely harder to predict online because many flips are subtle option-disambiguation cases. |

These are upper bounds, not expected deployed performance. Still, they justify
building a router-training dataset before adding more hand-designed methods.

## Proposed System: Bottleneck Router

Use a small controller with explicit stages:

1. **Task profile**: inspect the prompt without retrieval.
2. **Retrieval probe**: run cheap retrieval at k=1 and k=5 without generation.
3. **Route**: choose a method arm using task features plus retrieval-probe
   features.
4. **Execute**: run the chosen arm under a call budget.
5. **Verify/escalate**: if the answer is unsupported, abstaining, or parse-bad,
   escalate one tier.

This is agentic in behavior, but deterministic in structure.

### Arms

Start with a small arm set:

| Arm | Existing mode | When it should win |
|---|---|---|
| Direct/snap | `llm_only` or `snap_only_in_final` | Candidate-answer tasks where retrieval adds noise and model priors are strong. |
| Simple RAG k=1 | `rag_simple --retrieval-k 1` | One high-confidence passage is enough; avoid context pollution. |
| Simple RAG k=5 | `rag_simple --retrieval-k 5` | Retrieval-depth sensitive factual QA. |
| Query expansion | `rag_multi_query` | Lexical mismatch or low top-k overlap. |
| Pseudo-doc | `rag_snap_hyde_2call` / `multi_hyde_diverse` | Need semantic bridge from question to retrievable passage. |
| Iterative | `iter_hyde` / `iterative_planning_table` | Multi-hop composition or missing-link search. |

Do not include every historical mode. Start with 5-6 arms or the router-label
problem gets noisy.

## Features Available Before Answering

Static prompt features:

- answer format: MC4, MC5, yes/no, short span, long-form
- choice count and average choice length
- question length and prompt/context length
- named entity count, date/number count, citation/case/statute tokens
- multi-hop cue count: "who/whose child of", "after", "before", "where was X
  born", "which entity that...", nested clauses
- option similarity: are choices lexically close to each other?
- domain/corpus id when known

Retrieval-probe features:

- top-1 and top-5 cross-encoder scores
- top1-top2 and top1-top5 score margins
- BM25/dense overlap if available
- unique source/document count in top-5
- retrieved text length and score entropy
- answer-choice lexical overlap with retrieved chunks for MC datasets
- whether k=1 and k=5 retrieve from the same source cluster

Optional cheap LLM features:

- direct answer
- one-sentence "missing fact that would change my answer"
- self-reported route reason
- answer stability under one paraphrase

Use LLM self-confidence only as a weak feature. The existing
`adaptive_snap_route` result is evidence that self-routing prompts can collapse
to a dataset-level bias.

## Offline Training / Evaluation Plan

1. Build a router table by joining detail logs on `idx`.
2. For each row and arm, compute:
   - correctness
   - calls
   - latency
   - parse health
   - retrieval/gold-hit fields when available
3. Label each row with a cost-aware acceptable arm:
   `reward = correct - lambda * llm_calls - mu * latency`.
4. Train a simple classifier first:
   - logistic regression or gradient-boosted trees on static + retrieval-probe
     features
   - no LLM in the classifier until a non-LLM baseline is measured
5. Evaluate against:
   - best static arm
   - cheapest arm
   - current `adaptive_snap_route`
   - oracle upper bound
   - leave-one-dataset-out generalization

The leave-one-dataset-out test matters. If a router only memorizes "MuSiQue ->
MHD" and "BarExam -> top1/simple," it is not a general method.

## Harness Work Needed

Minimal next patch:

- Add `scripts/build_router_training_set.py`:
  - input: `--arm label=detail.jsonl`
  - output: row-per-question table with static features, retrieval-probe
    features, per-arm correctness/cost, and oracle label
- Add a first `scripts/train_router_baseline.py`:
  - scikit-learn baseline
  - cross-validation by dataset
  - reports static-best vs learned-router vs oracle

Then, only if offline routing beats static best:

- Add an eval mode `bottleneck_router`.
- Implement it as a state machine, not a free-form agent:
  `profile -> retrieval_probe -> choose_arm -> execute -> verify/escalate`.
- Keep the route decision in every detail log:
  `route_features`, `chosen_arm`, `route_reason`, `fallback_arm`,
  `route_confidence`, `reward_model_version`.

## What This Could Say In A Paper

The publishable claim would not be "we made an agent." It would be:

> A bottleneck-aware controller can select among retrieval strategies using
> cheap task and retrieval-probe features, outperforming the best fixed RAG
> pipeline under a fixed cost budget.

That claim is credible because it builds directly on Adaptive-RAG/MBA-RAG, but
it is sharper than query-complexity routing: it routes among mechanisms based
on observed bottleneck signals.

## Immediate Recommendation

Do not build the online agent first. Build the offline router dataset first.
The oracle results show enough headroom to justify it, and the offline table
will tell us whether the route is learnable from cheap features. If it is not
learnable offline, an online agent will mostly add latency and nondeterminism.
