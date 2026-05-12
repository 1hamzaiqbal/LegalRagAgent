# Diagnostic Controller Portfolio Comparison

Diagnostics: `docs/legal_rag_diagnostic_table_with_rewrite_2026-05-10.json`
Routes: `docs/diagnostic_controller_route_plan_with_rewrite_2026-05-10.json`

This is a source-summary comparison over available calibration evidence, not a fresh held-out benchmark.

| Portfolio | Datasets covered | Macro acc | Macro calls | Missing rows |
|---|---:|---:|---:|---:|
| `diagnostic_controller` | 4 | 77.9% | 1.30 | 0 |
| `baseline_retrieval` | 4 | 71.9% | 1.00 | 0 |
| `fixed_hyre_only` | 4 | 74.8% | 2.00 | 0 |
| `best_non_adaptive_same_n` | 4 | 72.9% | 1.50 | 0 |
| `query_rewrite_available` | 4 | 72.0% | 2.00 | 0 |

## Dataset Rows

| Dataset | Controller | Baseline retrieval | Fixed HyRE-only | Best non-adaptive same-N | Query rewrite |
|---|---|---|---|---|---|
| barexam | `adaptive_snap_hyre_v2` (200, 86.0%, 2.00 calls) | `rag_simple` (200, 80.0%, 1.00 calls) | `adaptive_snap_hyre_v2` (200, 86.0%, 2.00 calls) | `rag_rewrite` (200, 82.0%, 2.00 calls) | `rag_rewrite` (200, 82.0%, 2.00 calls) |
| casehold | `adaptive_snap_hyre_diverse` (200, 73.5%, 2.00 calls) | `rag_simple` (200, 73.0%, 1.00 calls) | `adaptive_snap_hyre_diverse` (200, 73.5%, 2.00 calls) | `rag_simple` (200, 73.0%, 1.00 calls) | `rag_rewrite` (50, 72.0%, 2.00 calls) |
| housing | `adaptive_snap_hyre_housing_verifier` (200, 74.5%, 1.00 calls) | `rag_state_filter` (200, 60.5%, 1.00 calls) | `adaptive_snap_hyre_diverse` (200, 63.5%, 2.00 calls) | `rag_state_filter` (200, 60.5%, 1.00 calls) | `rag_rewrite` (50, 58.0%, 2.00 calls) |
| legalbench_scalr | `adaptive_snap_hyre_disagreement_majority_prior` (200, 77.5%, 0.19 calls) | `rag_simple` (200, 74.0%, 1.00 calls) | `rag_snap_hyde_2call` (200, 76.0%, 2.00 calls) | `rag_snap_hyde_2call` (200, 76.0%, 2.00 calls) | `rag_rewrite` (50, 76.0%, 2.00 calls) |

## Reading

- `diagnostic_controller` is the current bottleneck-aware route plan.
- `baseline_retrieval` uses the simple retrieval baseline for each dataset, with HousingQA using the state-filter baseline because that is the current legal metadata baseline.
- `fixed_hyre_only` removes the targeted verifier/disagreement routes and asks how far a HyRE-style route gets without bottleneck-specific adaptation.
- `best_non_adaptive_same_n` only uses N=200 rows from non-adaptive methods in the diagnostic table.
- `query_rewrite_available` includes N=50 rows where N=200 rewrite rows are not yet available, so do not compare its macro score as a same-slice result.
