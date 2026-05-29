# Credibility A++ - Full-N Independent Factuality Judges

Status: **blocked by OpenRouter monthly key limit**. No `paper/` files were edited.

## Gate Result

The comprehensive A++ run requires two independent full-N judge passes over HyDE and SCOPE generations for BarExamQA, HousingQA state-filtered, SciFact, NFCorpus, FiQA, TREC-COVID, and SciDocs. The local OpenRouter key is currently exhausted:

- Key status check: `limit_remaining=0`, with a configured monthly limit of `50`.
- Claude route smoke, `anthropic/claude-sonnet-4.5`: HTTP 403, monthly key limit exceeded.
- GPT route smoke, `openai/gpt-4o`: HTTP 403, monthly key limit exceeded.
- Qwen fallback route smoke, `qwen/qwen3.6-max-preview`: HTTP 403, monthly key limit exceeded.

This is a hard provider-quota blocker, not a methodological failure. Running the full A++ cache now would only append failures.

## Existing Cache Coverage

Existing Gemma judge cache:

- Path: `docs/generated/factuality_judge_full_2026-05-28.jsonl`
- Rows: `13,956`
- Judge model: `or-gemma4-26b`
- BEIR coverage is full for all five BEIR datasets and both arms/premise kinds.
- Legal coverage is partial: BarExamQA has `968/1195` question rows per arm/premise kind; HousingQA state-filtered has `200/6853`.

Existing independent judge cache:

- Path: `docs/generated/factuality_judge_independent_q200_2026-05-29.jsonl`
- Rows: `3,848`
- Judge model: `openai/gpt-4o`
- Completed before the quota blocker: all five BEIR q200/q50 slices and `112` BarExamQA questions per arm/premise kind.
- HousingQA has no independent-judge rows in this cache.

The prior q200 independent analysis remains valid as a provisional result:

- Report: `docs/generated/credibility_A_independent_judge_2026-05-29.md`
- Pooled independent factuality AUC vs retrieval-hurt target: `0.586`
- Geometry AUC: `0.816`
- Independent factuality + geometry AUC: `0.816`
- Marginal lift after geometry: `0.000`

## A++ Verdict

No A++ substantive verdict can be issued until two independent judge passes are complete. The pre-stated A++ criteria require both independent judges at full N; current coverage is neither full-N nor two-judge.

Current status: **blocked/provisional**.

## Resume Commands

After OpenRouter quota is raised or a fresh approved key is available, run the two full-N caches separately and resume on interruption:

```bash
set -a
source .env
set +a

NO_SILENT_FALLBACK=1 \
LLM_PROVIDER=custom \
LLM_BASE_URL=https://openrouter.ai/api/v1 \
LLM_API_KEY="$OPENROUTER_API_KEY" \
LLM_MODEL=anthropic/claude-sonnet-4.5 \
EVAL_CONCURRENCY=8 \
uv run python scripts/build_factuality_judge_cache.py \
  --datasets all \
  --limit 0 \
  --provider custom \
  --resume \
  --output docs/generated/factuality_judge_full_claude_2026-05-29.jsonl
```

```bash
set -a
source .env
set +a

NO_SILENT_FALLBACK=1 \
LLM_PROVIDER=custom \
LLM_BASE_URL=https://openrouter.ai/api/v1 \
LLM_API_KEY="$OPENROUTER_API_KEY" \
LLM_MODEL=openai/gpt-4o \
EVAL_CONCURRENCY=8 \
uv run python scripts/build_factuality_judge_cache.py \
  --datasets all \
  --limit 0 \
  --provider custom \
  --resume \
  --output docs/generated/factuality_judge_full_gpt_2026-05-29.jsonl
```

If `openai/gpt-5` is available at resume time, replace `openai/gpt-4o` with that route after a successful route smoke. If either Claude or GPT remains unavailable, use `qwen/qwen3.6-max-preview` as the named fallback and record the substitution in this report.

