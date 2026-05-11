# Compact Meeting Packet - Diagnostic Legal RAG - 2026-05-11

Use this as the short entrypoint for the May 11 meeting. The longer source
gates remain `meeting_prep_2026-05-11_diagnostic_adaptation.md`,
`meeting_eval_expansion_status_2026-05-11.md`,
`meeting_package_audit_2026-05-11.md`, and `signoff_log.md`.

## One-Sentence Claim

Legal RAG does not have one universal failure mode. Calibration traces can
diagnose whether a task needs retrieval/query rewriting, Snap-HyRE/HyRE,
metadata filtering, option grounding, verifier policies, or
disagreement/reject policies; the current controller improves the verified
portfolio while exposing clear route-policy gaps for the paper.

## Meeting-Ready Result Tables

Verified calibration portfolio:

| Model & Method | BarExam | HousingQA | CaseHOLD | SCALR | Avg. | Calls |
|---|---:|---:|---:|---:|---:|---:|
| Gemma 4 26B + matched baseline route | 80.0 | 60.5 | 73.0 | 74.0 | 71.9 | 1.00 |
| + snap-only reasoning | 85.5 | 55.0 | 72.5 | 72.5 | 71.4 | 2.00 |
| + legal query rewrite control | 82.0 | 58.0* | 72.0* | 76.0* | 72.0 | 2.00 |
| + preselected HyRE-family route | 86.0 | 63.5 | 73.5 | 76.0 | 74.8 | 2.00 |
| + diagnostic controller routes | 86.0 | 74.5 | 73.5 | 77.5 | 77.9 | 1.30 |

`*` means N=50 calibration evidence; BarExam query rewrite is N=200.

Verified held-out slice, rows 200-249:

| Model & Method | BarExam | HousingQA | CaseHOLD | SCALR | Avg. | Calls |
|---|---:|---:|---:|---:|---:|---:|
| Gemma 4 26B + held-out baseline | 76.0 | 62.0 | 68.0 | 80.0 | 71.5 | 1.00 |
| + legal query rewrite | 90.0 | 58.0 | 76.0 | 78.0 | 75.5 | 2.00 |
| + selected diagnostic routes | 76.0 | 76.0 | 78.0 | 80.0 | 77.5 | 1.54 |

Use these as the main meeting numbers. Do not wait on the N>=500 jobs unless
they land cleanly before the meeting.

## N>=500 Canonical Scale-Up

These jobs were launched after the N=200 ladder to test the canonical routes at
larger scale. They are pending source gates and must not be promoted until
their stdout and detail logs validate.

Latest monitor refresh, 2026-05-11 16:30 CDT: the remaining monitored jobs
`67911`, `67912`, and `67913` all hit the 4-hour SLURM time limit. No partial
mode after cancellation is promoted. BarExam `67911` keeps only the copied
`rag_simple` mode at 400/500 = 80.0% with one missing prediction; partial
`rag_rewrite` stopped at `92/500`, and adaptive v2 was not reached. HousingQA
`67912` keeps only the copied `rag_state_filter` mode at 270/500 = 54.0% with
one missing prediction; partial `rag_rewrite` stopped at `116/500`, and the
verifier was not reached. CaseHOLD `67913` keeps copied clean `rag_simple`
359/500 = 71.8% and `rag_rewrite` 354/500 = 70.8%; partial diverse HyRE
stopped at `35/500`. The source stdout timeout logs are copied locally.
Earlier, `67912` HousingQA
`rag_state_filter` completed and was copied locally at 270/500 = 54.0%, with
errors 0, empty retrieval 0, no long-answer rows, and one missing prediction;
treat it as a verified baseline-mode log with that caveat while rewrite/verifier
modes continue. `67913` CaseHOLD `rag_rewrite` also completed cleanly at
354/500 = 70.8%, with errors 0, missing predictions 0, empty retrieval 0, and
no long-answer rows. Earlier, `67915` completed with exit `0:0`; the SCALR
`rag_rewrite` retry is clean at 423/571 = 74.1%, with errors 0, missing
predictions 0, empty retrieval 0, max output tokens 4,005, and no long-answer
rows. `67911` BarExam `rag_simple` also completed and was copied locally at
400/500 = 80.0%, with errors 0, empty retrieval 0, no long-answer rows, and one
missing prediction; treat it as a verified baseline-mode log with that caveat
while rewrite/adaptive modes continue. Earlier, `67897` completed with exit
`0:0`, but only the capped SCALR `rag_simple` half is clean. The paired
`adaptive_snap_hyre_frontier` half was copied locally and reaches 417/571 =
73.0%, but it fails the adaptive log audit because it has one missing
prediction and `analyze_detail_flags.py` also reports one long-answer row
(max final answer 20,480 chars). Do not promote `67897` as a clean paired
full-SCALR result.

| Job | Dataset | N | Modes | Status / purpose |
|---:|---|---:|---|---|
| 67897 | LegalBench-SCALR | 571 | `rag_simple`, `adaptive_snap_hyre_frontier` | Completed. `rag_simple` is clean at 419/571 = 73.4%; frontier is health-gated/rejected at 417/571 = 73.0% due one missing prediction and one long-answer row. |
| 67914 | LegalBench-SCALR | 571 | `rag_rewrite` | Rejected: CUDA/ECC failure on `a40-2206` before a detail log. |
| 67915 | LegalBench-SCALR | 571 | `rag_rewrite` | Completed clean retry at 423/571 = 74.1%, excluding `a40-2206`. |
| 67911 | BarExam | 500 | `rag_simple`, `rag_rewrite`, `adaptive_snap_hyre_v2` | Timed out. `rag_simple` mode copied and validated at 400/500 = 80.0% with one missing prediction; partial rewrite/adaptive are not promoted. |
| 67912 | HousingQA | 500 | `rag_state_filter`, `rag_rewrite`, `adaptive_snap_hyre_housing_verifier` | Timed out. `rag_state_filter` mode copied and validated at 270/500 = 54.0% with one missing prediction; partial rewrite/verifier are not promoted. |
| 67913 | CaseHOLD | 500 | `rag_simple`, `rag_rewrite`, `adaptive_snap_hyre_diverse` | Timed out. `rag_simple` clean at 359/500 = 71.8%; `rag_rewrite` clean at 354/500 = 70.8%; partial diverse HyRE is not promoted. |

The N=200 ladder remains the complete ablation table. The N>=500 runs are a
scale-up sanity layer for the most important routes, not a replacement for the
source-gated meeting package.

## Bottleneck Readout

| Dataset | Bottleneck | Current evidence | Meeting read |
|---|---|---|---|
| BarExam | Query/legal-reasoning formulation | Snap-HyRE v2 helps in calibration, but held-out rewrite wins. | Need a rewrite-vs-HyRE selector. |
| HousingQA | Statutory entailment and false-positive Yes behavior | Verifier routing beats state-filter baseline and fixed HyRE. | Keep state filtering plus conservative verifier. |
| CaseHOLD | Answer-option conversion | Diverse HyRE and rewrite help; direct option-table and replay selectors are negative. | Retrieval exposure alone is not enough. |
| SCALR | Method disagreement / candidate exposure | Controller/disagreement replay helps in calibration; held-out exact route is flat. | Refine disagreement arbitration. |

## Figures To Use

Generated by `uv run python scripts/build_meeting_package_figures.py`.

| Figure | Use |
|---|---|
| `docs/presentation/figures/12_diagnostic_adaptation_calibration_ablation.png` | Main inherited calibration table. |
| `docs/presentation/figures/13_diagnostic_adaptation_heldout_ablation.png` | Held-out table. |
| `docs/presentation/figures/14_diagnostic_controller_macro_lift.png` | Macro lift and call-cost comparison. |
| `docs/presentation/figures/15_bottleneck_diagnostic_route_map.png` | Evidence signal to route map. |
| `docs/presentation/figures/16_method_ladder_flowchart.png` | Method-family flowchart. |

## What Is Still Needed

Before the meeting:

1. No May 11 scale-up jobs remain active; use only copied, validated mode logs
   and explicit timeout/rejection notes.
2. If a job completes, copy detail JSONLs locally, run
   `scripts/analyze_detail_flags.py`, run `scripts/audit_adaptive_hyre_logs.py`
   for adaptive/HyRE-family rows, and update docs only for clean rows or
   explicit rejected rows.
3. If the jobs do not finish cleanly, present the verified N=200/N=50 package
   and say the N>=500 scale-up is running under the same gates.

For the paper:

1. Turn the current rule/evidence-summary controller into an automatic router.
2. Complete larger-scale runs for the top routes after output caps are stable.
3. Improve BarExam rewrite-vs-HyRE selection, SCALR disagreement arbitration,
   and CaseHOLD option conversion.
