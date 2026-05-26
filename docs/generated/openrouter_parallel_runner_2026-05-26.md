# OpenRouter Parallel Runner - 2026-05-26

## Scope

This is the Q2 results-lane implementation note for bounded OpenRouter concurrency in the eval harness. No files under `paper/` were edited.

Implemented paths:

- `eval/eval_harness.py`: answer-run concurrency for OpenRouter providers, controlled by `--concurrency` or `EVAL_CONCURRENCY`.
- `scripts/build_generation_cache.py`: generation-cache concurrency for OpenRouter providers, controlled by the same knob.
- `main.py`: thread-local LLM call metrics plus a lock around provider pacing/cooldown.
- `eval/eval_harness.py`: thread-local call traces and event traces so parallel rows do not mix trace buffers.
- `llm_config.py`: per-thread OpenRouter `ChatOpenAI` client cache when `EVAL_CONCURRENCY > 1`.

The default is conservative: OpenRouter providers use 8 workers when no value is supplied; non-OpenRouter providers remain serial unless explicitly wired later. Detail logs are written in deterministic question order after parallel collection, not completion order.

## Guard Preservation

The worker runs the same row function used by the serial path:

- Same `NO_SILENT_FALLBACK` gate and fail-closed violations.
- Same exact-final-line extraction and scoring.
- Same final-answer format retry path.
- Same near-completion-cap checks.
- Same Snap-HyRE and rewrite parse/fallback gates.
- Same retrieval/hyre cache hit requirements when cache paths are supplied.
- Same summary guards for high error rate and empty retrieval.

Read-only caches are preloaded before worker launch to avoid lazy global cache races. The per-request retry/backoff behavior remains in `main._llm_call`: transient `429/5xx/connection/timeout/rate/overloaded/unavailable/temporarily` errors retry up to three attempts with backoff, while OpenRouter provider fallback remains disabled through the existing `extra_body.provider.allow_fallbacks=false` controls.

## Validation Run

Validation used BarExamQA q50, `llm_only`, `or-gemma4-26b`, `OPENROUTER_PROVIDER_ONLY=Cloudflare`, `NO_SILENT_FALLBACK=1`, `EVAL_FINAL_FORMAT_RETRY=1`, and `LLM_MAX_COMPLETION_TOKENS=2048`.

Commands:

```bash
NO_SILENT_FALLBACK=1 EVAL_FINAL_FORMAT_RETRY=1 LLM_MAX_COMPLETION_TOKENS=2048 \
OPENROUTER_PROVIDER_ONLY=Cloudflare HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run python eval/eval_harness.py \
  --mode llm_only --provider or-gemma4-26b --dataset barexam \
  --questions 50 --seed 42 --tag q2_openrouter_sync_q50 --concurrency 1

NO_SILENT_FALLBACK=1 EVAL_FINAL_FORMAT_RETRY=1 LLM_MAX_COMPLETION_TOKENS=2048 \
OPENROUTER_PROVIDER_ONLY=Cloudflare HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run python eval/eval_harness.py \
  --mode llm_only --provider or-gemma4-26b --dataset barexam \
  --questions 50 --seed 42 --tag q2_openrouter_parallel8_threadlocal_q50 --concurrency 8
```

Local detail logs:

- `logs/eval_llm_only_or-gemma4-26b_20260526_032308_barexam_q2_openrouter_sync_q50_detail.jsonl`
- `logs/eval_llm_only_or-gemma4-26b_20260526_033239_barexam_q2_openrouter_parallel8_threadlocal_q50_detail.jsonl`

These validation detail logs are local ignored artifacts; their summary rows were not left in `logs/experiments.jsonl`.

## Throughput

| Run | Workers | Rows | Wallclock | Sec/query wallclock | Sum row elapsed | Calls | Input tokens | Output tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Sync baseline | 1 | 50 | 345s | 6.9s | 344.6s | 50 | 16,889 | 29,547 |
| Parallel final | 8 | 50 | 54s | 1.1s | 337.9s | 50 | 16,889 | 30,047 |

Speedup by wallclock: `345 / 54 = 6.39x`.

## Correctness Diff

Harness health:

- Row count: 50 vs 50.
- Label order: identical.
- Errors: 0 vs 0.
- Missing required final answer line: 0 vs 0.
- `NO_SILENT_FALLBACK` violations: 0 vs 0.
- LLM calls: 50 vs 50.
- Input tokens: identical at 16,889.

Answer result comparison:

| Metric | Sync | Parallel |
|---|---:|---:|
| Correct | 43/50 | 44/50 |
| Accuracy | 86.0% | 88.0% |
| Prediction/correctness row diffs | -- | 4 rows |

Differing rows:

| Row | Label | Gold | Sync pred | Sync correct | Parallel pred | Parallel correct |
|---:|---|---|---|---:|---|---:|
| 15 | `qa_nan_mbe_508` | A | D | 0 | A | 1 |
| 25 | `qa_nan_mbe_532` | A | A | 1 | D | 0 |
| 39 | `qa_nan_mbe_400` | D | A | 0 | C | 0 |
| 44 | `qa_CONTRACTS_mbe_1045` | B | C | 0 | B | 1 |

Reading: the runner-level invariants held, and the output file order is deterministic. The live OpenRouter model did not produce bit-identical per-row answers across independent sync and parallel calls even with temperature zero and Cloudflare pinned. The final-code q50 validation therefore shows a throughput gain and clean harness gates, but it does **not** prove strict per-row result no-op under live model nondeterminism. Aggregate accuracy moved by +1/50.

## Recommendation

Use the parallel runner for throughput. For paper-facing or signoff rows where strict reproducibility matters, either compare against cached generation/answer artifacts or treat a fresh parallel OpenRouter call as a new model sample rather than a deterministic replay of a prior synchronous run.

The implementation itself preserves the methodology gates; the remaining non-identity is provider/model nondeterminism across separate calls, not label-order drift, cache fallback, parsing drift, or trace/metric mixing.
