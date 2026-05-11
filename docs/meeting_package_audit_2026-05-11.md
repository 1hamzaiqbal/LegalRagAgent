# Meeting Package Completion Audit - 2026-05-11

Objective audited:

> Produce a meeting-ready, source-gated diagnostic adaptation package for legal
> RAG by May 11 at 4pm: consolidate four legal benchmarks, verify logs/results,
> build inherited ablation tables, diagrams, and bottleneck summaries, and use
> targeted runs only where they clarify the controller story. The narrative
> should show that calibration traces can route among baseline RAG, query
> rewrite, Snap-HyRE/HyRE, metadata filtering, option grounding, verifier
> policies, and disagreement/reject policies, with clear evidence for what
> works, what fails, and what needs to be improved for a paper.

## Artifact Checklist

| Requirement | Evidence | Status |
|---|---|---|
| Consolidate four legal benchmarks | `docs/meeting_prep_2026-05-11_diagnostic_adaptation.md` uses BarExam, HousingQA, CaseHOLD, and LegalBench-SCALR; MuSiQue is explicitly excluded from the main table. | Done |
| Verify calibration results | `docs/diagnostic_controller_portfolio_comparison_2026-05-10.json` backs the calibration table: baseline macro 71.9%, fixed HyRE 74.8%, controller 77.9%. | Done |
| Verify held-out results | `docs/heldout_controller_eval_2026-05-10.json` and `docs/heldout_query_rewrite_2026-05-10.json` back the held-out table: baseline 71.5%, rewrite 75.5%, controller 77.5%. | Done |
| Verify the live CaseHOLD targeted run | SLURM job `67744` completed with exit `0:0`; `analyze_detail_flags.py` and `audit_adaptive_hyre_logs.py` passed on the copied detail log. | Done |
| Verify BarExam snap-only control | SLURM job `67773` completed with exit `0:0`; copied detail log has 200 rows, 171/200 = 85.5%, average calls 2.00, errors 0, and one missing prediction. | Done |
| Verify HousingQA snap-only control | SLURM job `67775` completed with exit `0:0`; copied detail log has 200 rows, 110/200 = 55.0%, average calls 2.00, errors 0, and one missing prediction. | Done, negative control |
| Verify CaseHOLD snap-only control | SLURM job `67777` completed with exit `0:0`; copied detail log has 200 rows, 148/200 = 74.0%, average calls 2.00, errors 0, and no missing predictions. | Done |
| Verify SCALR snap-only control | SLURM job `67779` completed with exit `0:0`; copied detail log has 200 rows, 145/200 = 72.5%, average calls 2.00, errors 0, and no missing predictions. | Done |
| Verify HousingQA HyRE-only control | SLURM job `67826` completed with exit `0:0`; copied detail log has 200 rows, 100/200 = 50.0%, average calls 2.00, errors 0, and no missing predictions. | Done, negative control |
| Verify CaseHOLD HyRE-only control | SLURM job `67827` completed with exit `0:0`; copied detail log has 200 rows, 143/200 = 71.5%, average calls 2.00, errors 0, and one missing prediction. | Done, weak/negative control |
| Repair retrieval-bearing launch blocker | `rag_utils.py` now reinitializes the GTE remote-code `position_ids` buffer; direct embedding smoke `67820` and `rag_hyde` smoke `67821` completed cleanly. | Done |
| Relaunch missing ladder/model-coverage jobs | Gemma N=200 retrieval controls `67825-67831` and Groq held-out sanity jobs `67832-67839` are queued/running with the repaired embedder. | In progress; not report numbers yet |
| Build inherited ablation tables | Markdown tables in `docs/meeting_prep_2026-05-11_diagnostic_adaptation.md`; slide-ready PNGs `12_diagnostic_adaptation_calibration_ablation.png` and `13_diagnostic_adaptation_heldout_ablation.png`. | Done |
| Build diagrams | Mermaid controller diagram in the meeting prep; PNG route-map and macro-lift figures in `docs/presentation/figures/`. | Done |
| Build bottleneck summaries | `docs/meeting_prep_2026-05-11_diagnostic_adaptation.md` has the benchmark table and bottleneck summary table; `15_bottleneck_diagnostic_route_map.png` visualizes it. | Done |
| Use targeted runs only | The main meeting table incorporates the targeted CaseHOLD option-table repair; the expansion status separately tracks snap-only controls and retrieval/model-coverage jobs. | Done |
| Show routes across all named intervention families | The meeting prep and route map cover baseline RAG, query rewrite, Snap-HyRE/HyRE, metadata/state filtering, option grounding, verifier policies, disagreement arbitration, and reject/escalate. | Done |
| Show what works | Housing verifier and CaseHOLD diverse HyRE have the clearest held-out gains; controller macro improves by +6.0pp in calibration and held-out summaries. Snap-only is now fully measured and shows why routing is needed. | Done |
| Show what fails | CaseHOLD direct option table is clean but weak: 70.0%, below query rewrite 76.0% and diverse HyRE 78.0%; replay selector is 66.0%. | Done |
| Show what needs improvement for a paper | Meeting prep identifies BarExam rewrite-vs-HyRE selection, SCALR disagreement arbitration, and deeper CaseHOLD option-conversion mechanisms. | Done |

## Full Objective Coverage Audit

This section audits the complete active goal, including the stricter asks that
go beyond the meeting-ready package. Passing the meeting package does not by
itself mean every long-horizon experiment has completed.

| Goal requirement | Concrete artifact or evidence | Coverage |
|---|---|---|
| Meeting-ready diagnostic adaptation package by May 11 at 4pm | `docs/meeting_prep_2026-05-11_diagnostic_adaptation.md`, `docs/meeting_eval_expansion_status_2026-05-11.md`, this audit, and pushed commit `e9aafe8`. | Covered for meeting use. |
| Four legal benchmarks only | BarExam, HousingQA, CaseHOLD, and LegalBench-SCALR are the only main-table datasets in the meeting prep. | Covered. |
| Source-gated result reporting | Source JSONs, copied detail logs, `signoff_log.md`, and validation commands are listed in this audit. | Covered for reported numbers. |
| Inherited ablation table | Calibration table now includes baseline retrieval, snap-only reasoning, query rewrite, fixed HyRE family, and diagnostic controller. | Covered at the portfolio level. |
| Snap-only across all four legal benchmarks | `docs/snap_only_controls_2026-05-11.json` and four copied detail logs; all pass `analyze_detail_flags.py`. | Covered. |
| HyRE-only across all four legal benchmarks | HousingQA job `67826` is complete and negative; CaseHOLD job `67827` is complete and weak/negative; BarExam and SCALR jobs `67825` and `67828` are still running under the repaired embedder. | Partial; HousingQA and CaseHOLD are citeable. |
| Fixed Snap-HyRE fill-in rows | Existing portfolio has fixed HyRE-family rows; missing provider-matched N=200 fill-ins are SLURM `67829-67831`. | In progress / pending; do not cite yet. |
| Adaptive/controller rows | `docs/diagnostic_controller_portfolio_comparison_2026-05-10.json` and `docs/heldout_controller_eval_2026-05-10.json`. | Covered for current controller story. |
| Cross-model coverage | Groq Llama 70B held-out sanity jobs `67832-67839` are queued. | Pending; not a report result yet. |
| Full-corpus evaluations where feasible | Harness full sizes are documented in `docs/meeting_eval_expansion_status_2026-05-11.md`; all-method/all-model full corpus is not feasible before the meeting; targeted full-SCALR sanity job `67863` is launched for `rag_simple` and `adaptive_snap_hyre_frontier`. | Launched / pending; no full-corpus result is promoted. |
| Diagrams and flowcharts | Figures 12-16 under `docs/presentation/figures/`, plus Mermaid framework diagram in the meeting prep. | Covered. |
| Bottleneck summaries | Meeting prep bottleneck table maps each dataset to evidence signal and route. | Covered. |
| Targeted runs only | Run list is scoped to snap-only fill-in, HyRE/Snap-HyRE ladder controls, CaseHOLD option-table repair, and cross-model sanity. | Covered. |
| Verification of reported logs | Commands in this audit rerun figure generation and detail-log checks. | Covered for reported rows. |
| No unverified claims promoted | Pending jobs are listed as pending; invalid runs are explicitly excluded in expansion status. | Covered. |

## Generated Figure Manifest

Generated command:

```bash
uv run python scripts/build_meeting_package_figures.py
```

Outputs:

- `docs/presentation/figures/12_diagnostic_adaptation_calibration_ablation.png`
- `docs/presentation/figures/13_diagnostic_adaptation_heldout_ablation.png`
- `docs/presentation/figures/14_diagnostic_controller_macro_lift.png`
- `docs/presentation/figures/15_bottleneck_diagnostic_route_map.png`
- `docs/presentation/figures/16_method_ladder_flowchart.png`

Figures 12-14 read source-gated JSON summaries:

- `docs/diagnostic_controller_portfolio_comparison_2026-05-10.json`
- `docs/snap_only_controls_2026-05-11.json`
- `docs/heldout_controller_eval_2026-05-10.json`
- `docs/heldout_query_rewrite_2026-05-10.json`

Figures 15-16 are scripted diagrams generated from the same meeting-prep source
claims and the linked source-gated result docs; they are not independent result
sources.

## Commands Re-Run During This Audit

```bash
uv run python scripts/build_meeting_package_figures.py
uv run python scripts/audit_adaptive_hyre_logs.py logs/eval_adaptive_snap_hyre_option_table_or-gemma4-26b_20260511_0028_casehold_casehold-option-table-direct-or-gemma4-26b-api-q250-start200-end250-k5-adaptive_snap_hyre_option_table_detail.jsonl
uv run python scripts/analyze_detail_flags.py logs/eval_adaptive_snap_hyre_option_table_or-gemma4-26b_20260511_0028_casehold_casehold-option-table-direct-or-gemma4-26b-api-q250-start200-end250-k5-adaptive_snap_hyre_option_table_detail.jsonl
uv run python scripts/analyze_detail_flags.py logs/eval_snap_only_in_final_or-gemma4-26b_20260511_0346_barexam_meeting-missing-ladder-retry-or-gemma4-26b-n200-k5-snap_only_in_final_detail.jsonl
uv run python scripts/analyze_detail_flags.py logs/eval_snap_only_in_final_or-gemma4-26b_20260511_0259_housing_meeting-missing-ladder-retry-or-gemma4-26b-n200-k5-snap_only_in_final_detail.jsonl
uv run python scripts/analyze_detail_flags.py logs/eval_snap_only_in_final_or-gemma4-26b_20260511_0418_casehold_meeting-missing-ladder-retry-or-gemma4-26b-n200-k5-snap_only_in_final_detail.jsonl
uv run python scripts/analyze_detail_flags.py logs/eval_snap_only_in_final_or-gemma4-26b_20260511_0411_legalbench_scalr_meeting-missing-ladder-retry-or-gemma4-26b-n200-k5-snap_only_in_final_detail.jsonl
uv run python scripts/analyze_detail_flags.py logs/eval_rag_hyde_or-gemma4-26b_20260511_0443_housing_meeting-missing-retrieval-fixed-or-gemma4-26b-n200-k5-rag_hyde_detail.jsonl
uv run python scripts/analyze_detail_flags.py logs/eval_rag_hyde_or-gemma4-26b_20260511_0511_casehold_meeting-missing-retrieval-fixed-or-gemma4-26b-n200-k5-rag_hyde_detail.jsonl
```

Health-check result for the direct option-table log:

- rows: 50
- correct: 35/50 = 70.0%
- average calls: 2.00
- errors: 0
- parse failures: 0
- missing predictions: 0
- empty retrieval: 0
- artifact flags: 0

Health-check result for BarExam snap-only:

- rows: 200
- correct: 171/200 = 85.5%
- average calls: 2.00
- errors: 0
- missing predictions: 1
- artifact flags: 0

Health-check result for CaseHOLD snap-only:

- rows: 200
- correct: 148/200 = 74.0%
- average calls: 2.00
- errors: 0
- missing predictions: 0
- artifact flags: 0

Health-check result for SCALR snap-only:

- rows: 200
- correct: 145/200 = 72.5%
- average calls: 2.00
- errors: 0
- missing predictions: 0
- artifact flags: 0

Health-check result for HousingQA snap-only:

- rows: 200
- correct: 110/200 = 55.0%
- average calls: 2.00
- errors: 0
- missing predictions: 1
- artifact flags: 0

Health-check result for HousingQA HyRE-only:

- rows: 200
- correct: 100/200 = 50.0%
- average calls: 2.00
- errors: 0
- missing predictions: 0
- empty retrieval: 0
- artifact flags: 0

Health-check result for CaseHOLD HyRE-only:

- rows: 200
- correct: 143/200 = 71.5%
- average calls: 2.00
- errors: 0
- missing predictions: 1
- empty retrieval: 0
- artifact flags: 0

## Remaining Risk

This package is meeting-ready, not a finished paper submission. The current
controller is still partly evidence-summary/rule-based rather than a fully
automatic learned router. BarExam and SCALR have route-policy nuance on the
held-out slice, and CaseHOLD still needs a better option-conversion mechanism.
The expanded ladder/model-coverage jobs are running under source gates and
should not be promoted unless they finish cleanly. Those are paper directions,
not blockers for the May 11 meeting package.
