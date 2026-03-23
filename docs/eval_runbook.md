# Eval Runbook

This file is for manually launching the next sanity checks and reruns. The agent should not auto-start these commands for you.

## Quick Sanity Pass

Run tests:

```bash
uv run pytest -q -s
```

Run curated playtests:

```bash
uv run python eval/run_playtests.py --profile full_parallel
```

If you specifically want to spot-check the aspect variant too:

```bash
uv run python eval/run_playtests.py --profile full_parallel_aspect
```

## Full 100-Question Bar Rerun

Replay the same label set used for comparison:

```bash
uv run python eval/eval_qa.py 100 --profile full_parallel --labels-file logs/eval_rag_rewrite_deepseek_20260322_20_detail.jsonl
```

Resume the latest matching run if it gets interrupted:

```bash
uv run python eval/eval_qa.py 100 --profile full_parallel --labels-file logs/eval_rag_rewrite_deepseek_20260322_20_detail.jsonl --continue
```

## Output Files

The rerun writes three main files:

- `logs/eval_qa_<provider>_<timestamp>.txt`
  - compact scorecard and per-label pass/fail summary
- `logs/eval_qa_<provider>_<timestamp>_detail.jsonl`
  - one JSON row per evaluated question
- `logs/eval_qa_<provider>_<timestamp>_labels.json`
  - the exact replay manifest

Each detail row includes:

- `artifact_path`
- `parallel_rounds`
- `collections`
- `completeness_verdict`
- `terminal_reason`
- `run_timings`
- `run_llm_metrics`

Per-question deep traces live in `logs/run_artifacts/*.json`.

## What To Inspect After The Rerun

Focus on:

1. Questions that still stop with `terminal_reason=max_rounds`
2. Questions that stop with `terminal_reason=stalled`
3. Whether fallback-direct steps are now showing up as `support=support_only`
4. Whether MC misses are shifting from "generic doctrine, wrong option" toward fewer answer-choice discrimination failures
