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

## Resubmission

The fixed harness was synced to the cluster checkout and the Housing state
filter job was resubmitted as SLURM `58799`.

Meeting guidance: state that Housing remains a directional statutory-depth
signal from the existing k-sweep, but explicit state-filtered retrieval is
blocked until job `58799` lands and passes the empty-retrieval guard.
