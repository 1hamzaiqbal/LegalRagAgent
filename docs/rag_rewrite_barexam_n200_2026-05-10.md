# BarExam Query Rewrite N=200 - 2026-05-10

## Question

Does ordinary legal query rewriting match the BarExam Snap-HyRE route once it is
run on the same N=200 slice?

## Setup

- Dataset: BarExam
- Provider: `or-gemma4-26b`
- Mode: `rag_rewrite`
- Slice: `N=200`, seed `42`
- Retrieval: `k=5`
- SLURM job: `67432`
- Submit manifest:
  `/engrfs/tmp/jacobsn/hiqbal_legalrag/logs/adaptive_hyre_mode_matrix_20260510_150427.tsv`
- Detail log:
  `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_rag_rewrite_or-gemma4-26b_20260510_1626_barexam_adaptive-hyre-or-gemma4-26b-barexam-n200-k5-rag_rewrite_detail.jsonl`

## Result

| Method | N | Accuracy | Gold retrieved | Gold retrieved but wrong | Gold missing but correct | Calls | Health |
|---|---:|---:|---:|---:|---:|---:|---|
| `rag_rewrite` | 200 | 164/200 = 82.0% | 29/200 | 4 | 139 | 2.00 | PASS |

Validation:

- `sacct` reported job `67432` as `COMPLETED` with exit code `0:0`.
- `scripts/analyze_detail_flags.py` loaded 200 rows and reported
  `accuracy=164/200`.
- Artifact checks reported zero top-level HyDE/report/knowledge artifacts and
  zero nested gap artifacts.
- Every row had `call_trace` and `trace_events`.
- No empty retrieval, parse failure, or row-level `error` was present in the
  diagnostic summary.

## Interpretation

The N=50 BarExam query-rewrite control reached 86.0%, but the same-slice N=200
run lands at 82.0%. That keeps query rewriting in the policy menu, but it does
not replace the current BarExam Snap-HyRE route: `adaptive_snap_hyre_v2` remains
the strongest N=200 BarExam row at 86.0%.

This is useful for the diagnostic adaptation story. It shows that legal query
formulation is a real route, but the controller should choose generated
reasoning on BarExam when the calibrated comparison includes the full N=200
slice.

Updated generated artifacts:

- `docs/legal_rag_diagnostic_table_with_rewrite_2026-05-10.md`
- `docs/legal_rag_diagnostic_table_with_rewrite_2026-05-10.json`
- `docs/diagnostic_controller_route_plan_with_rewrite_2026-05-10.md`
- `docs/diagnostic_controller_route_plan_with_rewrite_2026-05-10.json`
- `docs/diagnostic_controller_eval_with_rewrite_2026-05-10.md`
- `docs/diagnostic_controller_eval_with_rewrite_2026-05-10.json`
