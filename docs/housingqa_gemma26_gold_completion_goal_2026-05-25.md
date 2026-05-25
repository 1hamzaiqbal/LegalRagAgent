# HousingQA Gemma 26B Gold Completion Goal - 2026-05-25

## Current State

HousingQA `or-gemma4-26b` is complete locally for the deployable/core rows:

| Mode | Detail log | Result |
|---|---|---:|
| `llm_only` | `logs/merged/housing_or-gemma4-26b_llm_only_full_20260523_114720_detail.jsonl` | 3846/6853 = 56.1% |
| `rag_simple` | `logs/merged/housing_or-gemma4-26b_rag_simple_statefilter_full_20260521_185315_detail.jsonl` | 4531/6853 = 66.1% |
| `rag_hyde` | `logs/eval_rag_hyde_or-gemma4-26b_20260521_174454_housing_local-snap-hyre-or-gemma4-26b-housing-rag_hyde-nfull-k5_detail.jsonl` | 4456/6853 = 65.0% |
| `snap_hyre` | `logs/merged/housing_or-gemma4-26b_snap_hyre_statefilter_full_20260523_113019_detail.jsonl` | 4458/6853 = 65.1% |

Local audit found no completed or partial HousingQA `or-gemma4-26b`
`golden_passage` or `golden_plus_neighbors` detail log. The immediate
completion target is therefore `golden_passage`; `llm_only` does not need to be
rerun.

## Goal

Finish HousingQA Gemma 26B `golden_passage` full-N with strict guards, merge
any shards into one canonical detail log, audit health, update the result gate,
and regenerate the generated package/status summaries.

Success criteria:

- 6853 scored rows for `dataset=housing`, `provider=or-gemma4-26b`,
  `mode=golden_passage`.
- zero row errors, zero missing predictions, zero fallbacks, zero unclosed
  think tags, zero missing explicit final answer lines, and zero oracle-evidence
  missing rows.
- merged canonical log under `logs/merged/`.
- `docs/signoff_log.md`, `docs/compiled_results.md`, `current_status.md`, and
  `docs/generated/snap_hyre_package/` updated if the audit passes.

## Preflight

```bash
python3 scripts/check_expected_provider_model.py \
  --provider or-gemma4-26b \
  --expected-model google/gemma-4-26b-a4b-it \
  --expected-label or-gemma4-26b

OPENROUTER_PROVIDER_ONLY=Cloudflare \
python3 scripts/check_openrouter_chat_route.py \
  --provider or-gemma4-26b \
  --expected-model google/gemma-4-26b-a4b-it \
  --provider-only Cloudflare \
  --timeout 45 \
  --max-tokens 8 \
  --expected-content OK
```

## Sharded Run

Run shards rather than one serial full-N job. These shards are non-overlapping
after the deterministic full HousingQA ordering.

```bash
mkdir -p logs/golden_passage_housing_gemma26_20260525

for shard in \
  0:857 857:1714 1714:2571 2571:3428 \
  3428:4285 4285:5142 5142:5999 5999:
do
  start="${shard%%:*}"
  end="${shard#*:}"
  name="housing_gemma26_golden_passage_s${start}_e${end:-end}"
  (
    OPENROUTER_PROVIDER_ONLY=Cloudflare \
    NO_SILENT_FALLBACK=1 \
    EVAL_FINAL_FORMAT_RETRY=1 \
    EVAL_GENERATION_FORMAT_RETRY=1 \
    LLM_MAX_COMPLETION_TOKENS=2048 \
    EVAL_MIN_COMPLETION_TOKENS=2048 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    PROVIDER=or-gemma4-26b \
    MODEL_LABEL=or-gemma4-26b \
    DATASET=housing \
    QUESTIONS=full \
    SEED=42 \
    MODES=golden_passage \
    SAMPLE_START="$start" \
    SAMPLE_END="$end" \
    RETRIEVAL_K=5 \
    scripts/local/run_answer_cell.sh
  ) > "logs/golden_passage_housing_gemma26_20260525/${name}.out" 2>&1 &
done
wait
```

If OpenRouter rate limits are tight, launch the same shards in smaller batches
of two or four instead of all eight at once.

## Merge And Audit

After all shards complete, collect the eight `eval_golden_passage_or-gemma4-26b`
detail logs and merge them:

```bash
python3 scripts/merge_detail_logs.py \
  --output logs/merged/housing_or-gemma4-26b_golden_passage_full_$(date -u +%Y%m%d_%H%M%S)_detail.jsonl \
  --key idx \
  logs/eval_golden_passage_or-gemma4-26b_*_housing_*golden_passage-nfull-k5-s*_detail.jsonl

python3 scripts/analyze_detail_flags.py \
  logs/merged/housing_or-gemma4-26b_golden_passage_full_*_detail.jsonl
```

Then run the stricter oracle check through `scripts/local/run_answer_cell.sh`'s
postcheck logic or a dedicated audit script before citing the row.
