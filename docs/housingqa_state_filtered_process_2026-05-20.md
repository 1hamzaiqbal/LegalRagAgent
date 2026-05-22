# HousingQA State-Filtered Process - 2026-05-20

HousingQA is structurally different from the other exact-scored benchmarks in
the current matrix. The Chroma collection contains statutes for every state and
territory, while each question names one jurisdiction. Treating the whole
national corpus as the live search space makes retrieval mostly a
wrong-jurisdiction matching problem instead of only a legal-query-generation
problem.

## Process Decision

State filtering is a HousingQA-specific retrieval process, not a new Snap-HyRE
method. The state is present in the question, so using it as Chroma metadata is
not answer leakage. As of the 2026-05-20 hardening pass, the main HousingQA
retrieval matrix requires this filter. Unfiltered Housing retrieval is only a
provenance/ablation path and must be launched with the explicit opt-out
`EVAL_ALLOW_UNFILTERED_HOUSING_RETRIEVAL=1`.

The filter must be applied uniformly to every retrieval method in the HousingQA
comparison:

- `rag_simple`
- `golden_plus_neighbors`
- `rag_hyde`
- `snap_hyre`
- `rag_rewrite`
- probe variants such as `rag_hyde_exemplar` and `snap_hyre_exemplar`

`llm_only` and `golden_passage` do not perform corpus retrieval, so the filter
does not apply to those rows.

Existing unfiltered full HousingQA rows should remain labeled as unfiltered
provenance rows. Do not retroactively describe them as state-filtered, and do
not count them as complete main-matrix retrieval rows.

## Harness Contract

Enable the process explicitly:

```bash
EVAL_HOUSING_STATE_FILTER=1 \
NO_SILENT_FALLBACK=1 \
DATASET=housing \
QUESTIONS=full \
RETRIEVAL_K=5 \
scripts/local/run_answer_cell.sh
```

When enabled:

- the local runner appends `_statefilter` to the default `CACHE_SCOPE`;
- if a custom `CACHE_SCOPE` is supplied, the runner requires the string
  `statefilter` in that scope and fails before launch otherwise;
- the eval CLI receives `--housing-state-filter`;
- retrieval calls use `where={"state": "<lowercase question state>"}`;
- detail logs record `housing_state_filter=true` and retrieval rows record
  `retrieval_where`;
- retrieval caches record both `where` and `housing_state_filter`;
- cache replay fails closed if the cache was built with a different `where`
  key, so an unfiltered cache cannot silently stand in for a state-filtered
  run.

The local answer runner auto-enables `EVAL_HOUSING_STATE_FILTER=1` when
`DATASET=housing` and any retrieval mode is requested. Direct
`eval/eval_harness.py` Housing retrieval runs fail unless `--housing-state-filter`
is passed. Retrieval-cache builds also fail unless `--housing-state-filter` is
passed. The only accepted bypass is the explicit provenance/ablation opt-out:

```bash
EVAL_ALLOW_UNFILTERED_HOUSING_RETRIEVAL=1
```

## Cache Construction

Build state-filtered caches with the same explicit flag:

```bash
uv run python scripts/build_retrieval_cache.py \
  --dataset housing \
  --questions full \
  --query-type raw_question \
  --max-k 10 \
  --housing-state-filter \
  --out caches/retrieval/full/housing_qfull_seed42_statefilter_raw_question_k10.jsonl
```

HyDE/Snap-HyRE state-filtered caches require the matching generation cache:

```bash
uv run python scripts/build_retrieval_cache.py \
  --dataset housing \
  --questions full \
  --query-type hyre_cache \
  --label-prefix snap_hyre \
  --hyre-cache-path caches/hyre/full/housing_qfull_seed42_or-gemma4-26b_snap_hyre.jsonl \
  --max-k 10 \
  --housing-state-filter \
  --out caches/retrieval/full/housing_qfull_seed42_statefilter_or-gemma4-26b_snap_hyre_k10.jsonl
```

## Reporting

Report HousingQA in one of two clearly named ways:

1. `HousingQA, unfiltered national corpus`: the current comprehensive matrix
   rows unless the logs say otherwise.
2. `HousingQA, state-filtered jurisdiction corpus`: the fair Housing-specific
   process where every retrieval method receives the same metadata filter.

The state-filtered version is the cleaner diagnostic for whether Snap-HyRE
helps once the benchmark is not dominated by cross-jurisdiction statute noise.
