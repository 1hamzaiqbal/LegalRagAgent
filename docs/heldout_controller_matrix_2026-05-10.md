# Held-Out Controller Matrix - 2026-05-10

## Purpose

Run a compact held-out check for the bottleneck-aware controller on deterministic
rows 200-249, separate from the seed-42 N=200 calibration slices used in the
current diagnostic tables.

This is intended to test whether the controller comparison remains plausible
off the calibration slice. It is not a full benchmark sweep.

## Submission

- Provider: `or-gemma4-26b`
- Slice: `N=50`, seed `42`, `sample_start=200`, `sample_end=250`
- Retrieval: `k=5`
- Checkout: `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre`
- Data / Chroma checkout: `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent`
- Manifest:
  `/engrfs/tmp/jacobsn/hiqbal_legalrag/logs/heldout_controller_matrix_20260510_163752.tsv`

| Dataset | Mode | Job |
|---|---|---:|
| BarExam | `rag_simple` | 67449 |
| BarExam | `adaptive_snap_hyre_v2` | 67450 |
| HousingQA | `rag_state_filter` | 67451 |
| HousingQA | `adaptive_snap_hyre_housing_verifier` | 67452 |
| CaseHOLD | `rag_simple` | 67453 |
| CaseHOLD | `adaptive_snap_hyre_diverse` | 67454 |
| LegalBench-SCALR | `rag_simple` | 67455 |
| LegalBench-SCALR | `rag_snap_hyde_2call` | 67456 |
| LegalBench-SCALR | `adaptive_snap_hyre_frontier` | 67457 |

SCALR's current selected controller route is the cached disagreement replay, so
these jobs land the held-out component rows first. A held-out disagreement replay
requires post-processing the component detail logs after they finish.

## Initial Queue Check

At the first poll, jobs 67449-67454 were running and 67455-67457 were pending.
No preflight failures had appeared in stdout.

## Integration Gate

Before promoting results:

1. Confirm `sacct` completion and exit code for all jobs.
2. Inspect stdout for Tracebacks, API/rate-limit errors, parse failures, empty
   retrieval warnings, or timeout.
3. Run `scripts/analyze_detail_flags.py` on every landed detail log.
4. Run `scripts/audit_adaptive_hyre_logs.py` on adaptive/detail logs where the
   adaptive audit applies.
5. Build a held-out diagnostic/controller comparison doc only from logs that
   pass these checks.
