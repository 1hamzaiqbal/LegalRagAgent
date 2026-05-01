# Housing State-Filter Followup - 2026-05-01

Purpose: record the status of the HousingQA state-filter gate. This is a
blocker/fix note, not a citeable method result.

## Failed Cluster Run

Cluster job `58282` completed both planned `rag_state_filter` runs:

| Mode | k | Logged accuracy | Detail log | Status |
|---|---:|---:|---|---|
| `rag_state_filter` | 5 | 107/200 (53.5%) | `logs/eval_rag_state_filter_or-gemma4-26b_20260430_1649_detail.jsonl` | FAILED-EMPTY-RETRIEVAL |
| `rag_state_filter` | 10 | 110/200 (55.0%) | `logs/eval_rag_state_filter_or-gemma4-26b_20260430_1720_detail.jsonl` | FAILED-EMPTY-RETRIEVAL |

The harness summary guard tagged both rows
`FAILED-EMPTY-RETRIEVAL`. All 200 detail rows in each log have
`retrieved_ids=[]` and `evidence_store=[]`. These accuracies are parametric
model behavior, not state-filtered retrieval performance.

## Root Cause

Housing statute metadata is embedded from `datasets/housing_qa/statutes.csv`
with lowercase state names, for example `california` and `new hampshire`.
Question rows store display-case names, for example `California` and
`New Hampshire`. The state filter previously emitted:

```python
{"state": "New Hampshire"}
```

That did not match the Chroma metadata:

```python
{"state": "new hampshire"}
```

## Fix

Patched `eval/eval_harness.py` so `_housing_state_where(...)` lowercases the
question state before constructing the Chroma filter. Added
`test_housing_state_filter_uses_chroma_metadata_case` in
`tests/test_formatter.py`.

Verification:

```bash
UV_CACHE_DIR=/tmp/uv-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  HF_DATASETS_OFFLINE=1 uv run python tests/test_formatter.py

UV_CACHE_DIR=/tmp/uv-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  HF_DATASETS_OFFLINE=1 uv run python tests/test_sanitizer.py
```

Results: `tests/test_formatter.py` 11/11 passed; `tests/test_sanitizer.py`
10/10 passed.

## Fixed k=5 Landing

The fixed harness was synced to the cluster checkout and the Housing state
filter job was resubmitted as SLURM `58799`. The job completed the k=5 leg
before timing out during the k=10 leg.

Clean landed k=5 result:

| Mode | k | Accuracy | Detail log | Empty retrieval | Gold retrieval |
|---|---:|---:|---|---:|---:|
| `rag_state_filter` | 5 | 123/200 (61.5%) | `logs/eval_rag_state_filter_or-gemma4-26b_20260501_1406_detail.jsonl` | 0/200 | 81/200 |

Paired checks:

| Pair | N | Baseline | Treatment | Delta | b/c | McNemar p | 95% bootstrap CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| top-5 `rag_simple` -> k=5 `rag_state_filter` | 200 | 53.5% | 61.5% | +8.0pp | 36/20 | 0.0440 | [+1.0, +15.5] pp |
| top-10 `rag_simple` -> k=5 `rag_state_filter` | 200 | 58.0% | 61.5% | +3.5pp | 33/26 | 0.4350 | [-4.0, +11.0] pp |

Interpretation: the casing fix converted the state filter from a parametric
fallback into real retrieval. The k=5 result suggests jurisdiction metadata can
beat generic k=5 retrieval and is directionally above generic top-10, but the
top-10 state-filter leg is still needed before claiming the metadata gate is
settled.

## Chunked k=10 Completion Run

The remaining k=10 leg was relaunched as chunked SLURM array `58937` using
`scripts/hpc/slurm_housing_state_filter_chunks.sh`. Duplicate k=5 chunks were
cancelled after the clean k=5 log above was validated; array tasks 4--7 cover
k=10 over the same deterministic N=200 sample in four 50-question slices.
