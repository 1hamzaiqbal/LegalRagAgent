# Diagnostic Framework Completion Audit - 2026-05-10

## Objective

Develop and evaluate a bottleneck-aware diagnostic adaptation framework for
legal RAG, where calibration traces determine when and how to apply Snap-HyRE /
HyRE, metadata filtering, option grounding, query rewriting, or verifier
policies across legal benchmarks.

## Requirement Checklist

| Requirement | Artifact / evidence | Status |
|---|---|---|
| Legal-only benchmark focus | `docs/legal_rag_diagnostic_table_2026-05-10.md` covers BarExam, HousingQA, CaseHOLD, and LegalBench-SCALR. | Done |
| Calibration traces include accuracy and retrieval metrics | `scripts/build_rag_diagnostic_table.py`; `docs/legal_rag_diagnostic_table_2026-05-10.md`; `docs/legal_rag_diagnostic_table_with_rewrite_2026-05-10.md`. | Done |
| Calibration traces include answer-conversion diagnostics | Diagnostic tables include gold retrieved but wrong, gold missing but correct, and conditional accuracy. | Done |
| Calibration traces include cost | Diagnostic tables include average LLM calls; controller evaluation reports macro average calls. | Done |
| Snap-HyRE / HyRE is represented as an intervention | N=200 rows for `adaptive_snap_hyre_v2`, `adaptive_snap_hyre_diverse`, `adaptive_snap_hyre_frontier`, and `rag_snap_hyde_2call`. | Done |
| Metadata filtering / statutory routing is represented | Housing baseline route uses `rag_state_filter`; controller routes HousingQA to `adaptive_snap_hyre_housing_verifier`. | Done |
| Option grounding / answer conversion is represented | CaseHOLD documents and controller route identify answer-option conversion; current route remains unresolved with `reject_or_escalate`. | Partially done |
| Query rewriting is represented as a non-HyRE control | `docs/rag_rewrite_baseline_n50_2026-05-10.md`; `docs/rag_rewrite_barexam_n200_2026-05-10.md`; with-rewrite diagnostic table and route plan. | Done |
| Same-slice BarExam query rewrite control | `docs/rag_rewrite_barexam_n200_2026-05-10.md`; `docs/legal_rag_diagnostic_table_with_rewrite_2026-05-10.md`; job `67432`. | Done |
| Verifier policy is represented | `docs/adaptive_hyre_housing_verifier_n200_2026-05-10.md`; controller routes HousingQA to verifier. | Done |
| Executable controller exists | `scripts/diagnostic_controller.py` consumes diagnostic JSON and emits route plan JSON/Markdown. | Done |
| Controller evaluation exists | `scripts/evaluate_diagnostic_controller.py`; `docs/diagnostic_controller_eval_with_rewrite_2026-05-10.md`. | Done as evidence-summary evaluation |
| Controller-vs-fixed comparison exists | `scripts/compare_diagnostic_controller.py`; `docs/diagnostic_controller_portfolio_comparison_2026-05-10.md`. | Done as calibration evidence |
| Held-out controller evaluation | Not yet complete. Current evaluation is over existing calibration slices, not a fresh held-out benchmark. | Missing |
| Source-gated result claims | Docs cite detail-log paths; query-rewrite logs were artifact-audited; existing N=200 rows are generated from source detail logs. | Done |

## Current Controller Evaluation

Current selected-route evidence from
`docs/diagnostic_controller_eval_with_rewrite_2026-05-10.md`:

| Dataset | Route | Evidence N | Accuracy | Calls | Caveat |
|---|---|---:|---:|---:|---|
| BarExam | `adaptive_snap_hyre_v2` | 200 | 86.0% | 2.00 | selected after N=200 query rewrite landed at 82.0% |
| CaseHOLD | `adaptive_snap_hyre_diverse` | 200 | 73.5% | 2.00 | answer-conversion unresolved |
| HousingQA | `adaptive_snap_hyre_housing_verifier` | 200 | 74.5% | 1.00 | strongest current targeted route |
| LegalBench-SCALR | `adaptive_snap_hyre_disagreement_majority_prior` | 200 | 77.5% | 0.19 | cached disagreement replay |

Macro selected-route accuracy is 77.9% with 1.30 average LLM calls in the
current evidence summary. The previous BarExam mixed-N caveat is removed:
`rag_rewrite` now has a same-slice N=200 result and no longer beats
`adaptive_snap_hyre_v2` on BarExam.

The portfolio comparison in
`docs/diagnostic_controller_portfolio_comparison_2026-05-10.md` reports:

| Portfolio | Macro accuracy | Macro calls | Caveat |
|---|---:|---:|---|
| `diagnostic_controller` | 77.9% | 1.30 | calibration-slice evidence |
| `baseline_retrieval` | 71.9% | 1.00 | same source table |
| `fixed_hyre_only` | 74.8% | 2.00 | HyRE-style routes without targeted verifier/disagreement |
| `best_non_adaptive_same_n` | 72.9% | 1.50 | N=200 only |
| `query_rewrite_available` | 72.0% | 2.00 | includes N=50 rows outside BarExam |

## Missing Work Before Calling The Objective Complete

1. Held-out controller evaluation:
   - current controller evaluation is an evidence-summary comparison over the
     available N=200 calibration slices, not a fresh held-out benchmark.
2. CaseHOLD answer-conversion intervention:
   - current diagnostics identify the bottleneck, but the policy route is still
     weak. The controller should either select a calibrated verifier/selector
     or explicitly route uncertain rows to `reject_or_escalate`.
3. Query rewrite same-slice coverage:
   - BarExam now has N=200 query rewrite, but HousingQA, CaseHOLD, and
     LegalBench-SCALR rewrite rows remain N=50 calibration controls.

## Next Concrete Experiment

Prioritize a compact fresh held-out controller-evaluation matrix, or a targeted
CaseHOLD answer-conversion policy that either improves accuracy or formalizes
`reject_or_escalate`. The goal is not to discover a new prompt; it is to make
the controller comparison fair enough to cite.
