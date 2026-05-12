# CaseHOLD Direct Option-Table Held-Out Result - 2026-05-11

## Purpose

Test the repaired `adaptive_snap_hyre_option_table` route on the same held-out
CaseHOLD rows 200-249 used by the diagnostic-controller validation.

The direct route scores the five displayed CaseHOLD answer holdings directly.
This avoids the candidate-conditioned Chroma query path that blocked the earlier
option-table attempts and isolates answer-option conversion rather than
retrieval recall.

## Run

- SLURM job: `67744`
- Status: completed, exit code `0:0`
- Dataset: `casehold`
- Slice: `--questions 250 --sample-start 200 --sample-end 250`
- Mode: `adaptive_snap_hyre_option_table`
- Provider: `or-gemma4-26b`
- Local detail log copied for analysis:
  `logs/eval_adaptive_snap_hyre_option_table_or-gemma4-26b_20260511_0028_casehold_casehold-option-table-direct-or-gemma4-26b-api-q250-start200-end250-k5-adaptive_snap_hyre_option_table_detail.jsonl`
- Cluster detail log:
  `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_adaptive_snap_hyre_option_table_or-gemma4-26b_20260511_0028_casehold_casehold-option-table-direct-or-gemma4-26b-api-q250-start200-end250-k5-adaptive_snap_hyre_option_table_detail.jsonl`

## Result

| Method | N | Correct | Accuracy | Avg. calls | Health |
|---|---:|---:|---:|---:|---|
| `adaptive_snap_hyre_option_table` | 50 | 35/50 | 70.0% | 2.00 | PASS |

Health checks:

- `analyze_detail_flags.py`: no top-level or nested answer-artifact flags.
- `audit_adaptive_hyre_logs.py`: 50 rows, 0 errors, 0 parse failures, 0
  missing predictions, 0 missing gold fields, 0 empty retrieval rows.
- `gold_retrieved=50/50` is expected by construction because the route scores
  displayed answer-option holdings directly; do not interpret it as corpus
  retrieval recall.

## Same-Slice Comparisons

All comparisons are on CaseHOLD held-out rows 200-249.

| Baseline | Treatment | N | Baseline acc | Treatment acc | Delta | b/c | McNemar p | Bootstrap CI pp |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `rag_simple` | `adaptive_snap_hyre_option_table` | 50 | 68.0% | 70.0% | +2.0 | 2/1 | 1.0000 | [-4, +10] |
| `rag_rewrite` | `adaptive_snap_hyre_option_table` | 50 | 76.0% | 70.0% | -6.0 | 1/4 | 0.3750 | [-14, +2] |
| `adaptive_snap_hyre_diverse` | `adaptive_snap_hyre_option_table` | 50 | 78.0% | 70.0% | -8.0 | 2/6 | 0.2891 | [-18, +2] |

## Interpretation

This fixes the implementation blocker but does not fix the CaseHOLD method
bottleneck. Direct option-table prompting is a clean targeted probe, not the
current selected route.

Meeting-safe read:

- The old option-table failures were implementation/indexing failures, not a
  conceptual impossibility.
- The repaired route executes cleanly and slightly beats the simple baseline on
  this held-out slice.
- It underperforms both legal query rewrite and diverse HyRE, so the current
  controller should keep `adaptive_snap_hyre_diverse` for CaseHOLD and treat
  direct option tables as a negative design point.
- The core CaseHOLD bottleneck remains answer-option conversion: exposing all
  options is not enough unless the selector can reason over fine legal
  distinctions.
