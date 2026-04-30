# Search-Space Consolidation - 2026-04-30

Purpose: organize the current result space into a small set of defensible
research directions. This is not a new leaderboard. It is a decision map for
where to spend the next API/cluster budget.

Primary sources for this pass:

- `docs/evidence_matrix_2026-04-30.md`
- `docs/musique_disagreement_audit_2026-04-30.md`
- `docs/casehold_flatness_audit_2026-04-30.md`
- `docs/scalr_depth_disagreement_2026-04-30.md`
- `docs/research_strategy_2026-04-30.md`
- `docs/router_probe_findings_2026-04-30.md`
- `docs/evidence_budgeted_ledger_router_plan.md`
- `docs/signoff_log.md`
- `logs/experiments.jsonl`

## Working Thesis

Keep the project centered on **bottleneck-aware legal RAG**, not a new named
RAG recipe. The strongest current story is:

> Retrieval interventions only help when they target the task's active
> bottleneck. The same method family can help multi-hop QA, be neutral on
> legal MC, and be actively unhelpful or saturated on holding-selection tasks.

The agentic version should be:

> A legal RAG agent should allocate evidence budget and preserve structured
> evidence state, rather than blindly adding more retrieval calls or more
> natural-language subagents.

## Result Buckets

### Keep: Bottleneck Taxonomy

This is the cleanest empirical result family.

| Dataset slice | Current evidence | Regime label | Read |
|---|---|---|---|
| MuSiQue x Llama 70B | `rag_simple` top-5 27.5%, top-1 13.0%; `snap_hyde_2call` 37.0%, +9.5pp p=0.0079; `multi_hyde_diverse` 35.5%, +8.0pp p=0.0195; four-arm oracle 57.0%. | retrieval-depth + query-formulation limited | Multiple passages matter, pseudo-doc/diverse query formation helps, and methods rescue overlapping but non-identical rows. |
| BarExam x Gemma 4 26B | top-5 82.5%, top-1 83.0%, p=1.0; `snap_hyde_2call` 85.5%, +3.0pp p=0.377 at N=200; full-N `rag_snap_hyde` is approved at +3.09pp. | answer-option / legal-prior dominated | More retrieved documents are not the bottleneck; snap/anchoring effects are more plausible than recall. |
| CaseHOLD x Llama 70B | top-5 72.0%, top-1 70.5%, p=0.678; `snap_hyde_2call` 69.5%, -2.5pp p=0.487; all current logs show 0/200 gold retrieved. | candidate-depth insensitive under current harness | Useful falsification of "legal MC always needs more retrieved text," but not yet retrieval-recall evidence because gold-option ids are not mapped to retrievable ids. |
| LegalBench-SCALR x Llama 70B | top-1 59.5%, top-5 77.0%, top-10 77.0%; top-5 vs top-1 +17.5pp p=1.05e-08; top-10 vs top-5 0.0pp p=1.0. | candidate-depth limited, saturated by top-5 | Needs a small candidate set; extra top-10 recall adds no net answer gain. |

Decision: **invest**. This is the spine of the paper/workshop story.

### Conditional Keep: Evidence-Budgeted Router

The router idea has real oracle headroom but weak current learnability.

| Evidence | Read |
|---|---|
| MuSiQue four-arm oracle: static best 37.0%, oracle 57.0%. | Per-question complementarity is real. |
| BarExam three-arm oracle: static best 85.5%, oracle 93.5%. | There are valuable rare flips, especially between k=1/k=5/two-call. |
| Common `rag` vs `two_call` router table: random split one-rule stump 72.0% at 1.30 calls/q vs fixed `two_call` 71.5% at 2 calls/q. | Some cost-aware signal exists. |
| Leave-one-dataset-out: learned models mostly collapse to the cheap baseline and fail to identify MuSiQue when held out. | Cheap prompt/retrieval features are not enough for a deployed router claim yet. |

Decision: **keep offline, pause online**. Build richer features or ledger state
before spending on a live `bottleneck_router` mode.

### Conditional Keep: Shared Evidence Ledger

This is the best agentic/legal-specific angle, but it is not yet empirically
proven.

Why it fits the results:

- `subagent_rag` is significantly negative on MuSiQue (-12.0pp, p=0.0007)
  because the current report/gap path over-abstains.
- BarExam subagent/report-style variants did not produce a step-function gain
  over simpler snap/RAG controls.
- SCALR shows extra context can distract: top-10 adds gold hits but no net
  accuracy over top-5.

Interpretation: the missing ingredient is not "more agents." It is auditable
shared evidence state: claim, source, quote/span, support/contradiction,
jurisdiction/date, agent role, and confidence.

Decision: **prototype one ledger arm only after finishing no-call analysis**.
Do not build a broad multi-agent system yet.

### Pause: Method Leaderboard Search

The following families should not receive more budget unless tied to a specific
bottleneck hypothesis:

- more HyDE variants without a diagnostic reason;
- more planning-table variants on BarExam/legal MC;
- more confidence/CE threshold variants;
- more vectorless/direct-role prompt variants;
- more report-only subagent variants.

Reason: the result space is already crowded, many rows are pre-pivot or
stale/missing detail logs, and the strongest current claim is not "one method
wins everywhere."

### Kill For Current Paper Claims

Do not build the paper around these:

- "snap_hyde_2call is novel" - too close to HyDE/Query2Doc/FLARE-style
  primitives.
- "N=200 winner" - too narrow and unstable.
- "agentic RAG helps legal QA" - contradicted by BarExam/CaseHOLD/SCALR
  asymmetries and L-MARS-like benchmark dependence.
- "golden_passage is an oracle" - existing audits show it is a single-passage
  context control, not an upper bound.

## Current Evidence Gaps

| Gap | Status | Why it matters |
|---|---|---|
| MuSiQue disagreement audit across `rag`, `two_call`, `mhd`, `iter` | Done in `docs/musique_disagreement_audit_2026-04-30.md`. | Shows complementarity: 57.0% any-correct oracle, only 16 baseline-wrong rows rescued by all three stronger arms. |
| CaseHOLD disagreement audit | Done in `docs/casehold_flatness_audit_2026-04-30.md`; retrieval instrumentation still missing. | Confirms answer-level flatness but also shows `gold_retrieved` is currently 0/200 for all arms, so retrieval-recall claims are not supported yet. |
| HousingQA modern clean diagnostic slice | Existing rows are older and many detail logs are missing locally. | Housing/current-law style tasks are important for legal-agent motivation. |
| Legal RAG Bench loader | Missing. | Best direct benchmark fit for legal long-form RAG and retrieval/generator decomposition. |
| MLEB-SCALR retrieval-only scoring | Missing. | Would separate retriever calibration from QA accuracy for the SCALR family. |
| Run manifest / hypothesis tags | Missing. | Current `experiments.jsonl` has 351 rows, but many are historical search noise. |

## Targeted Next Work

### No-API Work First

1. **Manifest the search space**: add a small YAML/JSON manifest mapping
   detail logs to `hypothesis`, `dataset_regime`, `arm_family`, `k`, `provider`,
   `cite_status`, and `known_caveat`.
2. **Router table v2 without new API**: add features from existing paired logs:
   k=1/k=5 answer-change, whether gold newly appears at k=5, and disagreement
   bucket labels from the audits.
3. **CaseHOLD retrieval mapping**: wire a retrieval-only diagnostic that can
   tell whether the correct holding option appears in retrieved text. The
   current `gold_retrieved` field is not usable for this dataset.

### One Small API/Cluster Slice If We Spend Calls

Run **HousingQA clean diagnostic N=200** only if the cluster `housing_statutes`
collection is confirmed present:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python eval/eval_harness.py \
  --mode rag_simple --provider groq-llama70b --questions 200 --dataset housing --retrieval-k 1

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python eval/eval_harness.py \
  --mode rag_simple --provider groq-llama70b --questions 200 --dataset housing --retrieval-k 5

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python eval/eval_harness.py \
  --mode rag_snap_hyde_2call --provider groq-llama70b --questions 200 --dataset housing
```

Reason: HousingQA is the missing legal/current-law style cell. It tests whether
retrieval is genuinely useful in a legal domain where parametric knowledge
should be weaker. Do not spend this locally; local Chroma currently only has
`legalbench_scalr_holdings` and an empty `legal_passages` collection.

### Defer

- Full-corpus MuSiQue until rate limit/provider constraints are solved.
- Online bottleneck router until offline v2 beats static policies under
  leave-one-dataset-out.
- Ledger subagent until the no-call disagreement audits identify which evidence
  fields the ledger must preserve.

## Recommended Near-Term Order

1. Search-space manifest with cite status and hypothesis tags.
2. Router table v2 using disagreement-bucket features from the MuSiQue,
   CaseHOLD, and SCALR audits.
3. CaseHOLD retrieval-only mapping before any more CaseHOLD LLM calls.
4. HousingQA clean diagnostic slice on cluster, if collection is ready.
5. Ledger schema v1 tied to observed disagreement buckets.

This order keeps capacity low, converts existing logs into mechanism, and
prevents another round of method-search sprawl.
