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
| Query rewriting is represented as a non-HyRE control | `docs/rag_rewrite_baseline_n50_2026-05-10.md`; with-rewrite diagnostic table and route plan. | Done as N=50 calibration |
| Verifier policy is represented | `docs/adaptive_hyre_housing_verifier_n200_2026-05-10.md`; controller routes HousingQA to verifier. | Done |
| Executable controller exists | `scripts/diagnostic_controller.py` consumes diagnostic JSON and emits route plan JSON/Markdown. | Done |
| Controller evaluation exists | `scripts/evaluate_diagnostic_controller.py`; `docs/diagnostic_controller_eval_with_rewrite_2026-05-10.md`. | Done as evidence-summary evaluation |
| Held-out or same-slice controller evaluation | Not yet complete. Current evaluation mixes N=50 `rag_rewrite` with N=200 routes for BarExam. | Missing |
| Source-gated result claims | Docs cite detail-log paths; query-rewrite logs were artifact-audited; existing N=200 rows are generated from source detail logs. | Done |

## Current Controller Evaluation

Current selected-route evidence from
`docs/diagnostic_controller_eval_with_rewrite_2026-05-10.md`:

| Dataset | Route | Evidence N | Accuracy | Calls | Caveat |
|---|---|---:|---:|---:|---|
| BarExam | `rag_rewrite` | 50 | 86.0% | 2.00 | mixed-N route hypothesis only |
| CaseHOLD | `adaptive_snap_hyre_diverse` | 200 | 73.5% | 2.00 | answer-conversion unresolved |
| HousingQA | `adaptive_snap_hyre_housing_verifier` | 200 | 74.5% | 1.00 | strongest current targeted route |
| LegalBench-SCALR | `adaptive_snap_hyre_disagreement_majority_prior` | 200 | 77.5% | 0.19 | cached disagreement replay |

Macro selected-route accuracy is 77.9% with 1.30 average LLM calls in the
current evidence summary. This is not a paper-grade held-out result because
BarExam uses an N=50 rewrite calibration row while the competing HyRE row is
N=200.

## Missing Work Before Calling The Objective Complete

1. Same-slice controller evaluation:
   - either run `rag_rewrite` at N=200 for BarExam, or
   - evaluate all selected routes on a shared fresh N=200 held-out slice.
2. CaseHOLD answer-conversion intervention:
   - current diagnostics identify the bottleneck, but the policy route is still
     weak. The controller should either select a calibrated verifier/selector
     or explicitly route uncertain rows to `reject_or_escalate`.
3. Controller-vs-fixed comparisons:
   - report controller macro accuracy and calls against fixed `rag_simple`,
     fixed Snap-HyRE, fixed query rewrite, and the best single non-adaptive
     route on the same slice.

## Next Concrete Experiment

Prioritize a same-slice BarExam `rag_rewrite` N=200 run or a compact fresh N=200
controller-evaluation matrix. The goal is not to discover a new prompt; it is to
make the controller comparison fair enough to cite.
