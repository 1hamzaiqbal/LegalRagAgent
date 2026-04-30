# AGENTS.md

Repository-local instructions for coding agents.

## Start Here

1. Read `CLAUDE.md` for current operational context, commands, environment notes,
   methodology gates, and the active result snapshot.
2. Read `docs/README.md` for the canonical documentation map and citation path.
3. For result claims, prefer the current citation gates in this order:
   `docs/signoff_log.md`, `docs/snap_hyde_2call_2026-04-28.md`,
   `docs/top1_ablation_2026-04-28.md`, `docs/compiled_results.md`, and
   `logs/experiments.jsonl`.

## Scope Rules

- Verify result claims against source logs or signoff docs before repeating them.
- Do not treat older narrative files as current unless `docs/README.md` says they
  are current.
- Preserve historical docs for traceability; archive or redirect stale working
  notes instead of deleting them.
- When running evals, keep `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` set and use
  `uv` or `~/.local/bin/uv` depending on PATH availability.

## Current Headline

As of the 2026-04-28 pivot, the live paper framing is the bottleneck taxonomy:
MuSiQue x Llama 70B is retrieval-depth sensitive, BarExam x Gemma 4 26B is
retrieval-depth flat, and LegalBench-SCALR/CaseHOLD are option-disambiguation
replicates. See `CLAUDE.md` and `docs/snap_hyde_2call_2026-04-28.md` for the
current numbers and caveats.
