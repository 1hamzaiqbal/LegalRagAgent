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
| Option grounding / answer conversion is represented | CaseHOLD documents and controller route identify answer-option conversion; `docs/casehold_reject_escalate_policy_2026-05-10.md` defines an auditable abstention/escalation policy. | Done as diagnostic policy |
| Query rewriting is represented as a non-HyRE control | `docs/rag_rewrite_baseline_n50_2026-05-10.md`; `docs/rag_rewrite_barexam_n200_2026-05-10.md`; with-rewrite diagnostic table and route plan. | Done |
| Same-slice BarExam query rewrite control | `docs/rag_rewrite_barexam_n200_2026-05-10.md`; `docs/legal_rag_diagnostic_table_with_rewrite_2026-05-10.md`; job `67432`. | Done |
| Verifier policy is represented | `docs/adaptive_hyre_housing_verifier_n200_2026-05-10.md`; controller routes HousingQA to verifier. | Done |
| Executable controller exists | `scripts/diagnostic_controller.py` consumes diagnostic JSON and emits route plan JSON/Markdown. | Done |
| Controller evaluation exists | `scripts/evaluate_diagnostic_controller.py`; `docs/diagnostic_controller_eval_with_rewrite_2026-05-10.md`. | Done as evidence-summary evaluation |
| Controller-vs-fixed comparison exists | `scripts/compare_diagnostic_controller.py`; `docs/diagnostic_controller_portfolio_comparison_2026-05-10.md`. | Done as calibration evidence |
| Held-out controller evaluation | `docs/heldout_controller_eval_2026-05-10.md` summarizes corrected jobs 67461-67469 over rows 200-249. | Done with caveats |
| Held-out query rewrite control | `docs/heldout_query_rewrite_2026-05-10.md` summarizes completed jobs 67511-67514 over rows 200-249. | Done |
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

The compact held-out retry in `docs/heldout_controller_eval_2026-05-10.md`
reports:

| Setting | Macro accuracy | Macro calls | Caveat |
|---|---:|---:|---|
| exact selected routes | 77.5% | 1.54 | SCALR disagreement replay ties baseline |
| matched baselines | 71.5% | 1.00 | same 50-row held-out slice |

Held-out dataset details: BarExam ties baseline at 76.0%; Housing verifier is
76.0% vs 62.0% baseline with one parse failure counted wrong; CaseHOLD diverse
is 78.0% vs 68.0% baseline; SCALR majority-prior disagreement replay is 80.0%,
tying baseline despite the frontier component reaching 84.0%.

The held-out query rewrite control in
`docs/heldout_query_rewrite_2026-05-10.md` reports 75.5% macro accuracy with
2.00 average calls on the same rows. It beats matched baseline retrieval
overall, but remains below the selected-route controller: BarExam benefits
strongly, CaseHOLD improves but does not pass the selected route, and HousingQA
and LegalBench-SCALR fall below their routed policies. This closes the
same-slice query-rewrite gap and strengthens the diagnostic routing argument.

## Validated Limitation

CaseHOLD answer conversion remains the main limitation, but it is now covered
as a diagnosed bottleneck rather than an uncovered framework requirement:

- current diagnostics define a `reject_or_escalate` policy for low-confidence
  CaseHOLD rows;
- `docs/casehold_option_table_heldout_2026-05-10.md` records the targeted
  option-table held-out attempts. The original candidate-conditioned path was
  blocked by embedding/query index errors, but the repaired direct option-table
  route in `docs/casehold_option_table_direct_heldout_2026-05-11.md` completed
  cleanly at 35/50 = 70.0%. That is a small lift over `rag_simple` but below
  `rag_rewrite` and `adaptive_snap_hyre_diverse`;
- `docs/casehold_replay_selector_heldout_2026-05-10.md` records a clean
  no-retrieval replay-selector negative result: 33/50 = 66.0%, below
  `rag_simple`, `rag_rewrite`, and the selected diverse route.

## Completion Decision

The objective is complete for the current framework stage. The repo now has:

1. a legal-only benchmark set;
2. generated calibration diagnostics with accuracy, retrieval exposure,
   conditional accuracy, cost, and health fields;
3. a bottleneck label and route plan;
4. represented intervention families: Snap-HyRE/HyRE, metadata/state filtering,
   option grounding, query rewriting, verifier routing, and reject/escalate;
5. controller evaluation on calibration evidence;
6. controller-vs-fixed portfolio comparison;
7. compact held-out controller validation;
8. held-out query rewrite controls;
9. source-gated docs for the unresolved CaseHOLD answer-conversion bottleneck.

Remaining CaseHOLD work is future research, not a blocker to saying the
diagnostic adaptation framework has been developed and evaluated.

## Future Work

The CaseHOLD option-table implementation blocker is fixed, but the direct
option-table route underperformed the stronger same-slice routes. The next
useful CaseHOLD move should change the option-conversion mechanism itself:
example directions include contrastive pairwise holding comparison,
rule-frame extraction before option selection, calibrated abstention, or a
selector trained/validated on disagreement traces.
