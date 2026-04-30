# SpecRAG-Lite Diagnostic Controller - 2026-04-30

Purpose: turn the current bottleneck-taxonomy work into an implementable method
proposal. The method should be interesting because it decides **when** to spend
RAG/agentic budget, not because it adds another heavy pipeline to every query.

## Thesis

GROUNDED: Speculative RAG shows that draft diversity plus verifier selection can
improve RAG when multiple evidence subsets produce meaningfully different
answer+rationale drafts. Adaptive-RAG, Self-RAG, MBA-RAG, and routing work show
that retrieval should be conditional rather than automatic.

HYPOTHESIZED: The missing piece is a diagnostic controller that decides whether
a question is worth escalating to draft/verifier machinery in the first place.
Legal and legal-adjacent benchmarks are a good stress test because they express
different bottlenecks:

- MuSiQue: retrieval depth and query formulation.
- BarExamQA: answer-option/legal-prior dominated on the current top-k slice.
- HousingQA: predicted statutory lookup / metadata-sensitive retrieval.
- CaseHOLD: holding-option disambiguation with misleading retrieval.
- LegalBench-SCALR: small candidate set needed, but top-5 saturates answer gain.
- MLEB-SCALR: retrieval-only calibration, no answer-generation confound.

Proposed contribution:

> A bottleneck-conditioned RAG controller uses cheap probes to choose among
> direct answering, shallow retrieval, deeper retrieval, pseudo-document
> retrieval, metadata filtering, and SpecRAG-lite draft verification.

This is stronger than claiming one method wins. It asks whether the system can
predict the failure mode before spending more calls.

## What The Method Is Not

- Not full Speculative RAG copied into legal QA.
- Not a general multi-agent framework.
- Not a prompt-only router that asks the model what to do.
- Not a SOTA leaderboard claim.

The method is a constrained state machine with logged features, fixed arms, and
cost-aware evaluation.

## Controller Loop

```text
question
  -> task profile
  -> cheap retrieval probes
  -> bottleneck label
  -> choose cheapest sufficient arm
  -> execute arm
  -> log evidence state + answer
  -> optional verifier/escalation if support is weak
```

The key design choice is that the controller can stop early. Expensive draft
verification only runs when probe features predict it should help.

## Bottleneck Labels

| Label | Observable signal | Preferred intervention |
|---|---|---|
| Parametric-sufficient | Direct answer stable; retrieval scores low or distractor-like; top-k does not change answer. | `llm_only` / `snap_only_in_final`; no retrieval. |
| Retrieval-depth limited | k=1 evidence weak or misses gold/proxy; k=5 improves support; multi-hop cues present. | `rag_simple k=5/10`, possibly diverse subsets. |
| Query-formulation limited | raw-query retrieval weak; pseudo-doc or answer-conditioned query changes retrieved evidence and answer. | `rag_snap_hyde_2call`, `multi_hyde_diverse`. |
| Metadata-limited | query includes jurisdiction/date/source constraints; dense retrieval retrieves right topic but wrong jurisdiction/time. | metadata filter, state/year constrained retrieval. |
| Evidence-utilization limited | gold/proxy evidence present, but answer still wrong or top-10 adds recall without accuracy. | verifier, rationale compression, evidence ledger. |
| Option-disambiguation limited | choices/holdings are semantically close; retrieval overlaps distractors; more context hurts or stays flat. | score displayed candidates, not free-form generation. |
| Bridge-limited | support requires discovering an intermediate entity/doctrine not named in query. | iterative missing-link search, not random evidence diversity. |

## SpecRAG-Lite Arm

This arm should run only for retrieval-depth/query-formulation/evidence-selection
cases. It is deliberately smaller than full Speculative RAG.

Inputs:

- question
- top-k retrieved documents, usually k=10 or k=15
- optional clusters over retrieved documents
- current evidence ledger entries, if any

Steps:

1. Cluster or partition retrieved documents into 3 diverse subsets.
2. Draft 3 answer+rationale candidates in parallel or sequential API calls.
3. Run one verifier call that sees:
   - question,
   - answer drafts,
   - rationales,
   - compact source ids/snippets.
4. Select one draft or request one escalation:
   - deeper retrieval,
   - metadata-filtered retrieval,
   - contradiction search,
   - abstain/unsupported.

Logged fields:

- `answer_drafts`
- `draft_rationales`
- `draft_doc_ids`
- `selected_draft_idx`
- `verifier_reason`
- `verifier_vote`
- `selection_regret_oracle` (offline only)
- `generated_context_tokens`
- `evidence_tokens`
- `draft_overlap`

Do not log Speculative RAG probability names unless the backend exposes the
actual logprobs. If using a judge prompt, call it `verifier_vote`, not
`rhoSelf-reflect`.

## Cheap Features

Static task features:

- dataset/domain
- answer type: MC4, MC5, yes/no, short span, long-form
- question length and prompt length
- choice count and choice similarity
- citation/statute/date/jurisdiction token counts
- multi-hop cue count
- negation and exception cue count

Retrieval probe features:

- top-1, top-5, top-10 retrieved ids
- top score and score margin
- score entropy
- source/jurisdiction diversity
- evidence token budget
- retrieved choice overlap for MC/holding tasks
- gold/proxy hit when available
- k=1/k=5 retrieved-id overlap

Cheap answer features:

- direct answer
- answer stability under one paraphrase or temperature sample
- answer changes under k=1 vs k=5 evidence
- yes/no bias or choice-letter distribution
- unsupported/abstention-like output flags

## Metrics

Answer quality:

- closed-set accuracy
- MuSiQue EM/F1 and answer containment
- open-ended judge score with groundedness caveats

Retrieval quality:

- Recall@k, Precision@k, Hit@k, MRR@k, nDCG@k
- gold/proxy hit delta from k=1 to k=5/k=10
- metadata-correct retrieval rate when jurisdiction/year exists

Cost and compression:

- LLM calls per question
- latency
- input/output tokens
- generated-context tokens / retrieved-evidence tokens
- draft count

Controller quality:

- best fixed arm vs controller
- oracle headroom
- cost-aware reward:
  `correct - lambda * calls - mu * latency - nu * evidence_tokens`
- selection regret:
  best available arm correctness minus chosen arm correctness
- escalation precision:
  fraction of escalations that improve over cheaper arm

## Ablation Plan

### A. Diagnostic Probe Ablations

| Variant | Purpose |
|---|---|
| static-only router | Tests whether dataset/prompt shape is enough. |
| + retrieval-probe features | Tests whether cheap retrieval observations identify bottlenecks. |
| + direct-answer stability | Tests whether uncertainty/answer change adds value. |
| + metadata features | Tests legal-specific jurisdiction/date value, especially HousingQA. |
| + ledger/support features | Tests whether structured evidence state predicts escalation. |

### B. Arm Ablations

| Arm set | Purpose |
|---|---|
| direct vs rag k=5 | Minimal RAG value. |
| k=1 vs k=5 vs k=10 | Retrieval-depth diagnosis. |
| rag k=5 vs two-call | Query-formulation diagnosis. |
| rag k=5 vs SpecRAG-lite | Draft diversity/verifier value. |
| two-call vs SpecRAG-lite | Whether multi-draft selection adds beyond pseudo-doc retrieval. |
| metadata filter vs HyDE | Legal-specific retrieval control. |

### C. Generalization Tests

- leave-one-dataset-out
- leave-one-answer-format-out
- legal-only train, MuSiQue test
- MuSiQue train, legal test
- train without HousingQA, test HousingQA

If leave-one-dataset-out fails, the paper should be honest: the diagnostic
taxonomy may still be valuable, but the controller is not yet general.

## Decision Gates

Do not implement online SpecRAG-lite until at least one of these is true:

1. HousingQA shows top-k sensitivity or metadata sensitivity.
2. Offline router features predict `two_call` / deeper retrieval better than
   fixed `rag` on held-out rows.
3. MLEB-SCALR embedding A/B shows retrieval variance large enough to justify
   retriever selection as a controller arm.
4. Manual rescued/harmed audits show draft diversity would have selected a
   better evidence subset.

Do implement metadata filtering before SpecRAG-lite if HousingQA top-k remains
weak and errors are jurisdiction/source mismatches.

## Implementation Path

### Phase 1: Finish Diagnostics

- Complete HousingQA k=1/k=5/k=10 + `two_call`.
- Add MLEB-SCALR embedding A/B.
- Build a small manual audit of rescued/harmed rows across MuSiQue, SCALR,
  HousingQA, and BarExam.

Output:

- one diagnostic table by dataset
- one bottleneck label per dataset/model slice
- explicit "escalate / do not escalate" recommendation

### Phase 2: Offline Controller Table

Extend `scripts/build_router_training_set.py` with:

- top-k retrieval probe features
- answer-change fields
- metadata/jurisdiction fields
- generated-context compression fields
- retrieval-only qrels fields when available

Output:

- controller report with fixed-arm, learned-router, and oracle comparisons
- leave-one-dataset-out results
- error buckets for false escalations and missed escalations

### Phase 3: SpecRAG-Lite Mode

Add one eval mode, not a broad system:

```text
specrag_lite
  retrieve k=10/15
  cluster/partition evidence
  draft 3 answer+rationale candidates
  verifier selects candidate or escalation
  final answer = selected candidate
```

Required log fields:

- all draft texts and source ids
- selected draft id
- verifier rationale
- cost fields
- generated/evidence token counts

### Phase 4: Bottleneck Controller Mode

Only after Phase 2/3:

```text
bottleneck_controller
  profile
  retrieval_probe
  route arm
  execute chosen arm
  optional verifier/escalate
```

Compare to:

- best fixed method
- fixed SpecRAG-lite
- fixed `two_call`
- fixed `rag k=5`
- oracle

## What Counts As A Paper-Worthy Win

Strong win:

- Controller beats best fixed arm on held-out questions at equal/lower cost,
  and the learned route aligns with interpretable bottleneck labels.

Moderate win:

- Controller does not beat best fixed arm globally, but identifies when
  SpecRAG-lite is wasteful and cuts cost without losing much accuracy.

Still useful negative result:

- Offline oracle has large headroom, but cheap features cannot predict it.
  This says current RAG failures are row-specific and need richer evidence
  state, which motivates the ledger path.

Do-not-claim:

- "Speculative RAG solves legal RAG."
- "Agents improve legal QA."
- "More retrieval is better."
- "Gold hit means answer should be correct."

## Current Recommendation

Wait for HousingQA before coding SpecRAG-lite. If HousingQA is depth-sensitive,
run SpecRAG-lite on MuSiQue + HousingQA first. If HousingQA is metadata-limited,
build metadata-filtered retrieval first and treat SpecRAG-lite as a later
evidence-selection arm. If HousingQA is flat, the project should lean harder on
the diagnostic-taxonomy claim and use MLEB-SCALR / LegalBench-RAG for pure
retrieval calibration.
