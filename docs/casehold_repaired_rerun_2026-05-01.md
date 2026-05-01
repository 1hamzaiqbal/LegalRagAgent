# CaseHOLD Repaired Rerun - 2026-05-01

Purpose: record the first CaseHOLD answer-quality rerun after repairing the
gold-option mapping and rebuilding the `casehold_holdings` Chroma collection on
cluster job `58283`.

## Run

| Field | Value |
|---|---|
| Dataset | CaseHOLD |
| Provider | `groq-llama70b` |
| N | 200 |
| Seed | 42 |
| Cluster job | `58283` (`embed-eval-casehold`) |
| Source logs | `logs/eval_rag_simple_groq-llama70b_20260430_1738_detail.jsonl`; `logs/eval_rag_snap_hyde_2call_groq-llama70b_20260430_1751_detail.jsonl` |
| Summary rows | `logs/experiments.jsonl` run ids `20260430_1738_rag_simple_groq-llama70b_snap-hyde-2call-pair-groq-llama70b-casehold-n200` and `20260430_1751_rag_snap_hyde_2call_groq-llama70b_snap-hyde-2call-pair-groq-llama70b-casehold-n200` |

## Results

| Mode | Accuracy | Gold retrieved | Empty retrieval | Calls/q | In tok/q | Out tok/q |
|---|---:|---:|---:|---:|---:|---:|
| `rag_simple` | 139/200 (69.5%) | 32/200 (16.0%) | 0/200 | 1.0 | 673 | 424 |
| `rag_snap_hyde_2call` | 144/200 (72.0%) | 94/200 (47.0%) | 0/200 | 2.0 | 1310 | 890 |

Paired test:

| Comparison | Baseline | Treatment | Delta | b/c | McNemar p | 95% bootstrap CI |
|---|---:|---:|---:|---:|---:|---:|
| `rag_simple` -> `rag_snap_hyde_2call` | 69.5% | 72.0% | +2.5pp | 16/11 | 0.4421 | [-2.5, +7.5] pp |

## Interpretation

- The repaired run changes the retrieval-read story: CaseHOLD no longer has
  meaningless 0/200 gold-hit under the repaired collection.
- `rag_snap_hyde_2call` substantially improves gold-option retrieval
  (16.0% -> 47.0%) but does not produce a statistically reliable answer lift
  at N=200 (+2.5pp, p=0.4421).
- The defensible meeting claim is therefore narrower than either old extreme:
  CaseHOLD is not retrieval-instrumentation blind anymore for this pair, but
  it still looks answer-flat under the tested arms. Better retrieval recall is
  not automatically converted into answer accuracy.
- The old `casehold_flatness_audit_2026-04-30.md` remains useful for its
  answer-disagreement patterns, but its gold-hit rows are superseded by this
  repaired rerun for `rag_simple` vs `rag_snap_hyde_2call`.

## Verification

Commands run locally after pulling the cluster logs:

```bash
UV_CACHE_DIR=/tmp/uv-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  HF_DATASETS_OFFLINE=1 uv run python scripts/compute_mcnemar.py \
  logs/eval_rag_simple_groq-llama70b_20260430_1738_detail.jsonl \
  logs/eval_rag_snap_hyde_2call_groq-llama70b_20260430_1751_detail.jsonl

UV_CACHE_DIR=/tmp/uv-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  HF_DATASETS_OFFLINE=1 uv run python scripts/analyze_detail_flags.py \
  logs/eval_rag_simple_groq-llama70b_20260430_1738_detail.jsonl

UV_CACHE_DIR=/tmp/uv-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  HF_DATASETS_OFFLINE=1 uv run python scripts/analyze_detail_flags.py \
  logs/eval_rag_snap_hyde_2call_groq-llama70b_20260430_1751_detail.jsonl
```

`analyze_detail_flags.py` reported 200 rows for each log, no top-level
HyDE/report/knowledge artifacts, and no empty-retrieval rows.
