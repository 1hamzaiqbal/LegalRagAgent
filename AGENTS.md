# AGENTS.md

Repository-local instructions for coding agents.

## Start Here

1. Read `CLAUDE.md` for current operational context, commands, environment notes,
   methodology gates, and the active result snapshot.
2. Read `docs/README.md` for the canonical documentation map and citation path.
3. For result claims, prefer the current citation gates in this order:
   `docs/signoff_log.md`, `docs/compiled_results.md`,
   `logs/experiments.jsonl`, and then older source-gated result docs listed in
   `docs/README.md`.

## Scope Rules

- Verify result claims against source logs or signoff docs before repeating them.
- Do not treat older narrative files as current unless `docs/README.md` says they
  are current.
- Preserve historical docs for traceability; archive or redirect stale working
  notes instead of deleting them.
- When running evals, keep `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` set and use
  `uv` or `~/.local/bin/uv` depending on PATH availability.

## Current Headline

As of the 2026-05-12 branch pivot, the live framing is a fixed-method
Snap-HyRE evaluation, not a diagnostic adaptive-controller story. The primary
harness mode is `snap_hyre`; `rag_snap_hyde_2call` remains a legacy alias for
older logs. The branch goal is to evaluate one straightforward Snap-HyRE method
across BarExamQA, HousingQA, Legal-Link-EU, and MASLegalBench, with retrieval
exposure metrics reported alongside downstream answer accuracy. CaseHOLD and
LegalBench-SCALR are historical/superseded for the active exact-scored main
matrix unless explicitly re-added under the current fixed-method contract. See
`CLAUDE.md`, `docs/README.md`, `docs/snap_hyre_comprehensive_plan_2026-05-12.md`,
and `docs/paper_iteration_signal_2026-05-20.md` for the current mission, method
ladder, caching workflow, launch gates, and paper-facing caveats.
