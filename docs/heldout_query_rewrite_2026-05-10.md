# Held-Out Query Rewrite Control - 2026-05-10

## Purpose

Run `rag_rewrite` on the same held-out rows 200-249 used by the compact
controller evaluation. This closes the remaining query-rewrite coverage gap:
BarExam already had an N=200 rewrite control, but the other legal datasets only
had N=50 calibration rewrite rows.

## Submission

- Provider: `or-gemma4-26b`
- Mode: `rag_rewrite`
- Slice: `--questions 250 --sample-start 200 --sample-end 250`
- Effective evaluated rows: 50
- Retrieval: `k=5`
- Manifest:
  `/engrfs/tmp/jacobsn/hiqbal_legalrag/logs/heldout_query_rewrite_20260510_174149.tsv`

| Dataset | Job |
|---|---:|
| BarExam | 67511 |
| HousingQA | 67512 |
| CaseHOLD | 67513 |
| LegalBench-SCALR | 67514 |

## Integration Gate

Before promoting results:

1. Confirm `sacct` completion and exit code for all jobs.
2. Inspect stdout for Tracebacks, API/rate-limit errors, parse failures, empty
   retrieval warnings, or timeout.
3. Run `scripts/analyze_detail_flags.py` on every landed detail log.
4. Compare query rewrite against matched held-out baselines and selected
   controller routes from `docs/heldout_controller_eval_2026-05-10.md`.
