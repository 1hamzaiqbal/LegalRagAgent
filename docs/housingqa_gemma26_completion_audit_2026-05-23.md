# HousingQA Gemma 26B Completion Audit - 2026-05-23

This is the current source-gated checkpoint for HousingQA `or-gemma4-26b`
after the May 22/23 completion-cycle logs.

## Result

No new HousingQA Gemma 26B full-N answer row can be promoted beyond the two
already signed state-filtered rows.

| Mode | Current status | Citable as full-N answer? | Source |
|---|---:|---:|---|
| `rag_simple` | 4531/6853 = 66.1% | yes | `logs/merged/housing_or-gemma4-26b_rag_simple_statefilter_full_20260521_185315_detail.jsonl` |
| `rag_hyde` | 4456/6853 = 65.0% | yes | `logs/eval_rag_hyde_or-gemma4-26b_20260521_174454_housing_local-snap-hyre-or-gemma4-26b-housing-rag_hyde-nfull-k5_detail.jsonl` |
| `snap_hyre` | 2554/3942 = 64.8% partial | no | `logs/eval_snap_hyre_or-gemma4-26b_20260522_124028_housing_local-snap-hyre-or-gemma4-26b-housing-snap_hyre-nfull-k5_detail.jsonl` |
| `llm_only` | 2025/3680 = 55.0% partial | no | two partial logs listed below |
| `golden_passage` | no HousingQA Gemma detail log found | no | none |

The full-N state-filtered retrieval cache for `snap_hyre` is valid and remains
citable as retrieval exposure, not answer accuracy:

| Method | Rows | Hit@5 | Recall@5 | MRR@5 | Source |
|---|---:|---:|---:|---:|---|
| Raw question RAG | 6853 | 0.3695 | 0.2413 | 0.2330 | `caches/retrieval/full/housing_qfull_seed42_statefilter_raw_question_k10.jsonl` |
| HyDE | 6853 | 0.3063 | 0.2042 | 0.1964 | `caches/retrieval/full/housing_qfull_seed42_statefilter_or-gemma4-26b_rag_hyde_k10.jsonl` |
| Snap-HyRE | 6853 | 0.3807 | 0.2505 | 0.2452 | `caches/retrieval/full/housing_qfull_seed42_statefilter_or-gemma4-26b_snap_hyre_k10.jsonl` |

## Log Inventory

Relevant full or partial answer detail logs:

- `logs/merged/housing_or-gemma4-26b_rag_simple_statefilter_full_20260521_185315_detail.jsonl`
  - 6853 rows, 6853 unique labels, 4531 correct.
- `logs/eval_rag_hyde_or-gemma4-26b_20260521_174454_housing_local-snap-hyre-or-gemma4-26b-housing-rag_hyde-nfull-k5_detail.jsonl`
  - 6853 rows, 6853 unique labels, 4456 correct.
- `logs/eval_snap_hyre_or-gemma4-26b_20260522_124028_housing_local-snap-hyre-or-gemma4-26b-housing-snap_hyre-nfull-k5_detail.jsonl`
  - 3942 rows, 3942 unique labels, 2554 correct.
- `logs/eval_llm_only_or-gemma4-26b_20260520_060947_housing_local-snap-hyre-or-gemma4-26b-housing-llm_only-nfull-k5_detail.jsonl`
  - 10 rows, 10 unique labels, 6 correct.
- `logs/eval_llm_only_or-gemma4-26b_20260520_061243_housing_local-snap-hyre-or-gemma4-26b-housing-llm_only-nfull-k5_detail.jsonl`
  - 3670 rows, 3670 unique labels, 2019 correct.

Filesystem search found no HousingQA Gemma 26B `golden_passage` detail log.

## Audit Commands

Detail-log health:

```bash
python3 scripts/analyze_detail_flags.py \
  logs/merged/housing_or-gemma4-26b_rag_simple_statefilter_full_20260521_185315_detail.jsonl \
  logs/eval_rag_hyde_or-gemma4-26b_20260521_174454_housing_local-snap-hyre-or-gemma4-26b-housing-rag_hyde-nfull-k5_detail.jsonl \
  logs/eval_llm_only_or-gemma4-26b_20260520_060947_housing_local-snap-hyre-or-gemma4-26b-housing-llm_only-nfull-k5_detail.jsonl \
  logs/eval_llm_only_or-gemma4-26b_20260520_061243_housing_local-snap-hyre-or-gemma4-26b-housing-llm_only-nfull-k5_detail.jsonl \
  logs/eval_snap_hyre_or-gemma4-26b_20260522_124028_housing_local-snap-hyre-or-gemma4-26b-housing-snap_hyre-nfull-k5_detail.jsonl
```

Key output:

- `rag_simple`: rows 6853, accuracy 4531/6853, errors 0, missing predictions
  0, empty retrieval 0.
- `rag_hyde`: rows 6853, accuracy 4456/6853, errors 0, missing predictions 0,
  empty retrieval 0.
- `llm_only`: rows 10 + 3670, errors 0, missing predictions 0.
- `snap_hyre`: rows 3942, accuracy 2554/3942, errors 0, missing predictions 0,
  empty retrieval 0.

State-filter answer-row gates:

```bash
python3 scripts/audit_housing_statefilter_detail.py \
  --provider or-gemma4-26b --mode rag_simple --expected-rows 6853 \
  logs/merged/housing_or-gemma4-26b_rag_simple_statefilter_full_20260521_185315_detail.jsonl

python3 scripts/audit_housing_statefilter_detail.py \
  --provider or-gemma4-26b --mode rag_hyde --expected-rows 6853 --require-hyre-cache \
  logs/eval_rag_hyde_or-gemma4-26b_20260521_174454_housing_local-snap-hyre-or-gemma4-26b-housing-rag_hyde-nfull-k5_detail.jsonl

python3 scripts/audit_housing_statefilter_detail.py \
  --provider or-gemma4-26b --mode snap_hyre --expected-rows 6853 --require-hyre-cache \
  logs/eval_snap_hyre_or-gemma4-26b_20260522_124028_housing_local-snap-hyre-or-gemma4-26b-housing-snap_hyre-nfull-k5_detail.jsonl
```

Key output:

- `rag_simple`: pass; wrong provider/mode/dataset 0, missing prediction 0,
  error 0, missing state filter 0, retrieval/doc/HyRE cache misses 0,
  bad evidence length 0, missing exact final 0, fallback 0, think tags 0.
- `rag_hyde`: pass with the same zero-failure counters.
- `snap_hyre`: fails only the full-N gate: `expected 6853 rows, found 3942`.
  The 3942 present rows otherwise have zero structural failures.

Retrieval-cache gates:

```bash
python3 scripts/audit_retrieval_cache.py \
  --cache caches/retrieval/full/housing_qfull_seed42_statefilter_raw_question_k10.jsonl \
  --dataset housing --min-k 5 --ks 1,3,5,10

python3 scripts/audit_retrieval_cache.py \
  --cache caches/retrieval/full/housing_qfull_seed42_statefilter_or-gemma4-26b_rag_hyde_k10.jsonl \
  --dataset housing --min-k 5 --ks 1,3,5,10

python3 scripts/audit_retrieval_cache.py \
  --cache caches/retrieval/full/housing_qfull_seed42_statefilter_or-gemma4-26b_snap_hyre_k10.jsonl \
  --dataset housing --min-k 5 --ks 1,3,5,10
```

All three caches pass with 6853 rows, 0 duplicate keys, 0 missing indices, 0
empty retrieval rows, 0 short rows, and 0 rows without gold.

Provider readiness:

```bash
MODEL_LABEL=or-gemma4-26b python3 scripts/check_expected_provider_model.py \
  --provider or-gemma4-26b \
  --expected-model google/gemma-4-26b-a4b-it \
  --expected-label or-gemma4-26b

python3 scripts/check_openrouter_key_status.py --min-limit-remaining 0.01

OPENROUTER_PROVIDER_ONLY=Cloudflare python3 scripts/check_openrouter_chat_route.py \
  --provider or-gemma4-26b \
  --expected-model google/gemma-4-26b-a4b-it \
  --provider-only Cloudflare
```

These pass on 2026-05-23. The OpenRouter key status shows about 22.21 units
remaining. The Cloudflare route smoke returns `OK`.

## Coverage Gaps

Coverage was checked against the 6853-label order in the signed full
`rag_simple` log.

`llm_only` currently covers 3680 labels and is missing 3173 labels:

| sample start | sample end exclusive | first missing label | last missing label |
|---:|---:|---|---|
| 3479 | 4746 | `hqa_Ohio_734` | `hqa_Tennessee_7216` |
| 4793 | 4845 | `hqa_Texas_4531` | `hqa_Texas_9149` |
| 4999 | 6853 | `hqa_Nevada_5686` | `hqa_Wyoming_8829` |

`snap_hyre` currently covers 3942 labels and is missing 2911 labels:

| sample start | sample end exclusive | first missing label | last missing label |
|---:|---:|---|---|
| 3479 | 4746 | `hqa_Ohio_734` | `hqa_Tennessee_7216` |
| 4793 | 4845 | `hqa_Texas_4531` | `hqa_Texas_9149` |
| 5261 | 6853 | `hqa_New Jersey_3000` | `hqa_Wyoming_8829` |

The stale core queue lock remains at `/tmp/housing_gemma_core_queue.lock`, with
metadata PID `3819545`, but that PID is not live. `scripts/local/check_housing_gemma_readiness.sh`
correctly refuses launch while the lock is present.

## Continuation Commands

Use the same exact route:

```bash
export PROVIDER=or-gemma4-26b
export MODEL_LABEL=or-gemma4-26b
export OPENROUTER_PROVIDER_ONLY=Cloudflare
export NO_SILENT_FALLBACK=1
export LLM_MAX_COMPLETION_TOKENS=2048
export EVAL_MIN_COMPLETION_TOKENS=2048
```

For the missing `snap_hyre` answer spans, use the already-built full generation,
retrieval, and document caches:

```bash
for span in "3479 4746" "4793 4845" "5261"; do
  set -- $span
  SAMPLE_START="$1" SAMPLE_END="${2:-}" \
  MODE=snap_hyre QUESTIONS=full RETRIEVAL_K=5 \
  CACHE_SCOPE=qfull_seed42_statefilter \
  HYRE_CACHE_ROOT="$PWD/caches/hyre/full" \
  RETRIEVAL_CACHE_ROOT="$PWD/caches/retrieval/full" \
  scripts/local/run_housing_statefilter_rag_simple_with_doc_cache.sh
done
```

For the missing `llm_only` answer spans:

```bash
for span in "3479 4746" "4793 4845" "4999"; do
  set -- $span
  SAMPLE_START="$1" SAMPLE_END="${2:-}" \
  PROVIDER=or-gemma4-26b MODEL_LABEL=or-gemma4-26b \
  DATASET=housing QUESTIONS=full RETRIEVAL_K=5 MODES=llm_only \
  USE_CACHES=0 REQUIRE_RETRIEVAL_CACHES=0 \
  scripts/local/run_answer_cell.sh
done
```

For `golden_passage`, no HousingQA Gemma detail log exists. If that cell is
needed, launch it as a full answer row and audit it separately:

```bash
PROVIDER=or-gemma4-26b MODEL_LABEL=or-gemma4-26b \
DATASET=housing QUESTIONS=full RETRIEVAL_K=5 MODES=golden_passage \
USE_CACHES=0 REQUIRE_RETRIEVAL_CACHES=0 \
scripts/local/run_answer_cell.sh
```

After any continuation finishes, merge only clean non-overlapping logs:

```bash
python3 scripts/merge_detail_logs.py --on-duplicate last --output <merged-output.jsonl> <inputs...>
python3 scripts/analyze_detail_flags.py <merged-output.jsonl>
python3 scripts/audit_housing_statefilter_detail.py --provider or-gemma4-26b --mode snap_hyre --expected-rows 6853 --require-hyre-cache <merged-output.jsonl>
```

Then refresh signoff/status with the repo helpers rather than manually
promoting a partial row.

## Paper Impact

No Table 1 change is justified from the current logs. Keep HousingQA Gemma 26B
`LLM`, `Snap-HyRE`, and `Gold Evidence` blank in the answer matrix.

Table 2 remains valid: it is full-N state-filtered retrieval exposure, not
answer accuracy.

Do not describe the partial 64.8% Snap-HyRE answer log as a full HousingQA
Gemma 26B answer result.

## Prompt-To-Artifact Checklist

| Requirement | Evidence |
|---|---|
| Find all current HousingQA Gemma 26B detail logs for the named modes | Log inventory above from `find logs logs/merged ... *housing*or-gemma4-26b*detail.jsonl`. |
| Verify 6853 unique rows for candidate full rows | `rag_simple` and `rag_hyde` pass; `snap_hyre` and `llm_only` are partial; `golden_passage` absent. |
| Verify provider/model/mode/dataset | `audit_housing_statefilter_detail.py` reports wrong provider/mode/dataset 0 for citable rows and for the present `snap_hyre` partial. |
| Verify state filtering where applicable | `rag_simple`, `rag_hyde`, and present `snap_hyre` rows report missing state filter 0. |
| Verify no missing predictions/errors | `analyze_detail_flags.py` and state-filter audit report errors 0 and missing predictions 0 for present rows. |
| Verify exact `Answer: Yes/No` format | state-filter audit reports missing exact final 0 for citable rows and the present `snap_hyre` partial. |
| Verify no silent fallback/think tags | state-filter audit reports fallback 0 and think tags 0. |
| Verify retrieval/doc/HyRE cache hits where required | state-filter audit reports retrieval/doc/HyRE cache misses 0 for citable generated rows and present `snap_hyre` partial. |
| Merge partial logs only when clean | No merge was performed; coverage gaps show partial logs cannot yet produce full-N rows. |
| Update source-gated artifacts | This report, `docs/signoff_log.md`, paper lineage notes, and current dashboard notes now separate citable full-N rows from partial rows. |
| Regenerate paper if Table 1/Table 2 changed | No regeneration needed because no citable full-N answer value changed and retrieval tables already match the full caches. |
