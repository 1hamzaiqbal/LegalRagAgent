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

## Update: Repaired Top-1 Depth Leg

Cluster job `58871` completed the repaired `rag_simple` k=1 leg over the
cluster `casehold_holdings` collection. The detail log was pulled locally:

| Mode | k | Accuracy | Gold retrieved | Empty retrieval | Detail log |
|---|---:|---:|---:|---:|---|
| `rag_simple` | 1 | 129/200 (64.5%) | 18/200 (9.0%) | 0/200 | `logs/eval_rag_simple_groq-llama70b_20260501_1432_detail.jsonl` |

Paired against the repaired k=5 `rag_simple` row above:

| Comparison | Baseline | Treatment | Delta | b/c | McNemar p | 95% bootstrap CI |
|---|---:|---:|---:|---:|---:|---:|
| `rag_simple` k=1 -> k=5 | 64.5% | 69.5% | +5.0pp | 16/6 | 0.0525 | [0.0, +9.5] pp |

Interpretation: CaseHOLD now has a directional depth signal under repaired
retrieval, but it remains weaker than SCALR's top-1 -> top-5 jump and is
borderline at N=200. Treat this as a validated diagnostic row, not a final
benchmark claim.

Verification commands:

```bash
UV_CACHE_DIR=/tmp/uv-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  HF_DATASETS_OFFLINE=1 uv run python scripts/analyze_detail_flags.py \
  logs/eval_rag_simple_groq-llama70b_20260501_1432_detail.jsonl

UV_CACHE_DIR=/tmp/uv-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  HF_DATASETS_OFFLINE=1 uv run python scripts/compute_mcnemar.py \
  logs/eval_rag_simple_groq-llama70b_20260501_1432_detail.jsonl \
  logs/eval_rag_simple_groq-llama70b_20260430_1738_detail.jsonl \
  --bootstrap-samples 2000
```

## Update: Repaired k=10 And HyDE Legs

The same cluster job also completed repaired `rag_simple` k=10 and `rag_hyde`
k=5:

| Mode | k | Accuracy | Gold retrieved | Empty retrieval | Detail log |
|---|---:|---:|---:|---:|---|
| `rag_simple` | 10 | 136/200 (68.0%) | 38/200 (19.0%) | 0/200 | `logs/eval_rag_simple_groq-llama70b_20260501_1440_detail.jsonl` |
| `rag_hyde` | 5 | 144/200 (72.0%) | 104/200 (52.0%) | 0/200 | `logs/eval_rag_hyde_groq-llama70b_20260501_1449_detail.jsonl` |

Paired tests against repaired k=5 `rag_simple`:

| Comparison | Baseline | Treatment | Delta | b/c | McNemar p | 95% bootstrap CI |
|---|---:|---:|---:|---:|---:|---:|
| `rag_simple` k=5 -> k=10 | 69.5% | 68.0% | -1.5pp | 6/9 | 0.6072 | [-5.0, +2.5] pp |
| `rag_simple` k=5 -> `rag_hyde` k=5 | 69.5% | 72.0% | +2.5pp | 18/13 | 0.4731 | [-2.5, +8.0] pp |

Paired `rag_snap_hyde_2call` vs `rag_hyde` is exactly answer-flat at 72.0%
for both arms (b/c=12/12, p=1.0, 95% CI [-5.0, +5.0] pp). The repaired
CaseHOLD read is therefore: generated-query retrieval strongly improves
gold-option retrieval, but neither more depth nor snap-conditioning produces
a reliable answer-accuracy lift at N=200.
