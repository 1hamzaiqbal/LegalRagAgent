# Diagnostic Controller Evaluation

Diagnostics: `docs/legal_rag_diagnostic_table_with_rewrite_2026-05-10.json`
Routes: `docs/diagnostic_controller_route_plan_with_rewrite_2026-05-10.json`

This report evaluates the route plan against existing diagnostic summaries. It is not a fresh held-out benchmark.

- Macro accuracy across selected routes: 77.9%
- Macro average LLM calls: 1.30
- Mixed-N caveat datasets: barexam

| Dataset | Bottleneck | Selected route | N | Acc | Calls | Baseline | Delta vs baseline pp | Best available | Delta vs best pp | Best same-N | Status |
|---|---|---|---:|---:|---:|---|---:|---|---:|---|---|
| barexam | `query_retrieval_gap` | `rag_rewrite` | 50 | 86.0% | 2.00 | `rag_simple` (200) | +6.0 | `adaptive_snap_hyre_v2` (200) | +0.0 | `rag_rewrite` (86.0%) | PASS_WITH_MIXED_N_CAVEAT |
| casehold | `answer_conversion_gap` | `adaptive_snap_hyre_diverse` | 200 | 73.5% | 2.00 | `rag_simple` (200) | +0.5 | `adaptive_snap_hyre_diverse` (200) | +0.0 | `adaptive_snap_hyre_diverse` (73.5%) | PASS |
| housing | `statutory_entailment_gap` | `adaptive_snap_hyre_housing_verifier` | 200 | 74.5% | 1.00 | `rag_state_filter` (200) | +14.0 | `adaptive_snap_hyre_housing_verifier` (200) | +0.0 | `adaptive_snap_hyre_housing_verifier` (74.5%) | PASS |
| legalbench_scalr | `method_disagreement_gap` | `adaptive_snap_hyre_disagreement_majority_prior` | 200 | 77.5% | 0.19 | `rag_simple` (200) | +3.5 | `adaptive_snap_hyre_disagreement_majority_prior` (200) | +0.0 | `adaptive_snap_hyre_disagreement_majority_prior` (77.5%) | PASS |

## Reading

- `PASS_WITH_MIXED_N_CAVEAT` means the selected route is present and scored, but at least one comparison row uses a different N. Treat the route as a policy hypothesis, not a paired claim.
- `Delta vs best pp` should be zero for a controller that selects the best available route in the diagnostic table. Negative values indicate that the controller intentionally chose a non-best route, usually for cost or calibration reasons.
- A paper-grade controller result still needs a same-slice or held-out evaluation where all candidate routes are available on the same questions.
