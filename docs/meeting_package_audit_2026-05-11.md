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
| Verify BarExam HyRE-only control | SLURM job `67825` completed with exit `0:0`; copied detail log has 200 rows, 164/200 = 82.0%, average calls 2.00, errors 0, and no missing predictions. | Done, modest positive control |
| Verify HousingQA snap-only control | SLURM job `67775` completed with exit `0:0`; copied detail log has 200 rows, 110/200 = 55.0%, average calls 2.00, errors 0, and one missing prediction. | Done, negative control |
| Verify CaseHOLD snap-only control | SLURM job `67867` completed with exit `0:0`; copied detail log has 200 rows, 145/200 = 72.5%, average calls 2.00, errors 0, no missing predictions, and no long-answer rows. This supersedes health-caveated job `67777`. | Done |
| Verify SCALR snap-only control | SLURM job `67779` completed with exit `0:0`; copied detail log has 200 rows, 145/200 = 72.5%, average calls 2.00, errors 0, and no missing predictions. | Done |
| Verify HousingQA HyRE-only control | SLURM job `67826` completed with exit `0:0`; copied detail log has 200 rows, 100/200 = 50.0%, average calls 2.00, errors 0, and no missing predictions. | Done, negative control |
| Verify CaseHOLD HyRE-only control | SLURM job `67827` completed with exit `0:0`; copied detail log has 200 rows, 143/200 = 71.5%, average calls 2.00, errors 0, and one missing prediction. | Done, weak/negative control |
| Reject SCALR HyRE-only uncapped control | SLURM job `67828` completed with exit `0:0`, but copied detail log has one runaway final answer: 267,458 chars / 70,593 output tokens. | Done; not a clean report number |
| Verify HousingQA fixed Snap-HyRE control | SLURM job `67830` completed with exit `0:0`; copied detail log has 200 rows, 103/200 = 51.5%, average calls 2.00, errors 0, no missing predictions, and no long-answer rows. | Done, negative control |
| Verify CaseHOLD fixed Snap-HyRE control | SLURM job `67831` completed with exit `0:0`; copied detail log has 200 rows, 144/200 = 72.0%, average calls 2.00, errors 0, no missing predictions, and no long-answer rows. | Done, weak/negative control |
| Verify BarExam fixed Snap-HyRE control | SLURM job `67829` completed with exit `0:0`; copied detail log has 200 rows, 169/200 = 84.5%, average calls 2.00, errors 0, one missing prediction, and no long-answer rows. | Done |
| Verify Groq held-out sanity rows | Jobs `67832`, `67833`, `67834`, `67835`, `67836`, `67838`, and `67839` completed cleanly and were copied locally; job `67837` completed but is rejected due errors 2, empty retrieval 2, and missing predictions 2. | Done, partial model-coverage sanity |
| Verify capped SCALR HyRE-only rerun | SLURM job `67864` completed the eval loop at 148/200 = 74.0%; copied detail log has 200 rows, average calls 2.00, errors 0, one missing prediction, empty retrieval 0, and no long-answer rows. The SLURM wrapper failed after results because `scripts/postprocess_adaptive_hyre_sweep.py` was missing. | Done, wrapper-caveated |
| Relaunch capped CaseHOLD snap-only rerun | SLURM job `67866` was cancelled after row 12 produced a 157,678-character answer; `llm_config.py` now sends OpenRouter caps through `extra_body.max_tokens`, and replacement `67867` completed cleanly with `LLM_MAX_COMPLETION_TOKENS=4096`. | Done |
| Repair retrieval-bearing launch blocker | `rag_utils.py` now reinitializes the GTE remote-code `position_ids` buffer; direct embedding smoke `67820` and `rag_hyde` smoke `67821` completed cleanly. | Done |
| Harden detail-log validation | `scripts/analyze_detail_flags.py` now reports errors, missing predictions, parse failures, empty retrieval rows, average calls, max output tokens, max final-answer length, and long-answer outliers in addition to artifact leakage. | Done |
| Add optional runaway-output cap | `llm_config.py` supports `LLM_MAX_COMPLETION_TOKENS` for targeted reruns that need bounded final-answer generations. For OpenRouter, the cap is sent as `extra_body.max_tokens`; default behavior remains unchanged when the env var is unset. | Done |
| Reconcile missing ladder/model-coverage jobs | Gemma N=200 retrieval controls `67825-67831`, SCALR capped HyRE-only `67864`, capped CaseHOLD snap-only replacement `67867`, and Groq held-out sanity jobs `67832-67839` have landed, with invalid rows explicitly rejected. Full-SCALR probe `67863` is rejected/cancelled; capped replacement `67897` remains live. | Mostly done |
| Build inherited ablation tables | Markdown tables in `docs/meeting_prep_2026-05-11_diagnostic_adaptation.md`; slide-ready PNGs `12_diagnostic_adaptation_calibration_ablation.png` and `13_diagnostic_adaptation_heldout_ablation.png`. | Done |
| Build diagrams | Mermaid controller diagram in the meeting prep; PNG route-map and macro-lift figures in `docs/presentation/figures/`. | Done |
| Add figure captions | `docs/presentation/figures/captions.md` now includes captions and source notes for figures 12-16. | Done |
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
| Meeting-ready diagnostic adaptation package by May 11 at 4pm | `docs/meeting_prep_2026-05-11_diagnostic_adaptation.md`, `docs/meeting_eval_expansion_status_2026-05-11.md`, this audit, and the current pushed meeting branch. | Covered for meeting use. |
| Four legal benchmarks only | BarExam, HousingQA, CaseHOLD, and LegalBench-SCALR are the only main-table datasets in the meeting prep. | Covered. |
| Source-gated result reporting | Source JSONs, copied detail logs, `signoff_log.md`, and validation commands are listed in this audit. | Covered for reported numbers. |
| Inherited ablation table | Calibration table now includes matched baseline route, snap-only reasoning, query rewrite, preselected HyRE-family route, and diagnostic controller. | Covered at the portfolio level. |
| Snap-only across all four legal benchmarks | `docs/snap_only_controls_2026-05-11.json` and four copied detail logs; snap-only intentionally has empty retrieval payloads. CaseHOLD now uses clean capped replacement `67867`, which supersedes the health-caveated `67777` row. | Covered. |
| HyRE-only across all four legal benchmarks | BarExam job `67825` is complete and modestly positive; HousingQA job `67826` is complete and negative; CaseHOLD job `67827` is complete and weak/negative; SCALR uncapped job `67828` is rejected for runaway output, while capped rerun `67864` is detail-log clean at 74.0% with a postprocess wrapper caveat. | Covered with SCALR wrapper caveat. |
| Fixed Snap-HyRE fill-in rows | Existing SCALR row is already source-gated; BarExam job `67829`, HousingQA job `67830`, and CaseHOLD job `67831` are complete. | Covered for the Gemma N=200 ladder. |
| Adaptive/controller rows | `docs/diagnostic_controller_portfolio_comparison_2026-05-10.json` and `docs/heldout_controller_eval_2026-05-10.json`. | Covered for current controller story. |
| Cross-model coverage | Groq Llama 70B held-out sanity jobs `67832-67839` are complete; seven rows are clean and one CaseHOLD selected route row is rejected by health gates. | Covered as held-out sanity, not a main result table. |
| Full-corpus evaluations where feasible | Harness full sizes are documented in `docs/meeting_eval_expansion_status_2026-05-11.md`; all-method/all-model full corpus is not feasible before the meeting. Targeted full-SCALR job `67863` is rejected/cancelled because both the completed baseline half and partial frontier half hit runaway-output gates. Capped replacement `67897` is running. | Launched / pending; no full-corpus result is promoted. |
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

Refresh note, 2026-05-11 13:36 CDT: these validation commands were rerun for
the promoted May 11 detail logs. The rerun matches the documented promoted and
rejected rows: snap-only, HyRE-only, fixed Snap-HyRE, CaseHOLD option-table, and
Groq held-out rows remain source-consistent; the SCALR uncapped HyRE-only row
and full-SCALR `67863` baseline half remain rejected/health-gated for runaway
final-answer length; Groq CaseHOLD diverse remains rejected by the adaptive log
audit because it has errors, empty retrieval rows, missing predictions, and
unexpected call counts. The completed `67897` capped full-SCALR `rag_simple`
mode was also copied and checked: 571 rows, 419/571 = 73.4%, errors 0, missing
predictions 0, empty retrieval 0, max output tokens 4,405, and no long-answer
rows. Because the paired frontier mode is still running, this is tracked as a
verified baseline-half log rather than a promoted full replacement result. The
completed `67913` CaseHOLD `rag_simple` mode was copied and checked: 500 rows,
359/500 = 71.8%, errors 0, missing predictions 0, empty retrieval 0, max output
tokens 2,725, and no long-answer rows. Because CaseHOLD rewrite/diverse modes
are still running or pending, this is tracked as a verified baseline-mode log
rather than a completed scale-up ablation.

Refresh note, 2026-05-11 15:30 CDT: `67897` completed. The capped SCALR
`rag_simple` half remains a clean verified baseline-half log, but the paired
`adaptive_snap_hyre_frontier` half is rejected/health-gated: copied detail log
has 571 rows, 417/571 = 73.0%, errors 0, parse failures 0, empty retrieval 0,
average calls 2.00, max output tokens 8,454, max final-answer chars 20,480,
one missing prediction, and one long-answer row. `scripts/audit_adaptive_hyre_logs.py`
exits nonzero with `FAIL missing_prediction=1`.

Refresh note, 2026-05-11 15:50 CDT: `67915` completed and the SCALR
`rag_rewrite` retry was copied and checked: 571 rows, 423/571 = 74.1%, errors
0, missing predictions 0, empty retrieval 0, average calls 2.00, max output
tokens 4,005, and no long-answer rows. `67911` BarExam `rag_simple` mode also
completed and was copied and checked: 500 rows, 400/500 = 80.0%, errors 0, one
missing prediction, empty retrieval 0, average calls 1.00, max output tokens
2,260, and no long-answer rows. BarExam rewrite/adaptive modes remain running
or pending, so the BarExam row is tracked as a verified baseline-mode log.

```bash
uv run python scripts/build_meeting_package_figures.py
uv run python scripts/audit_adaptive_hyre_logs.py logs/eval_adaptive_snap_hyre_option_table_or-gemma4-26b_20260511_0028_casehold_casehold-option-table-direct-or-gemma4-26b-api-q250-start200-end250-k5-adaptive_snap_hyre_option_table_detail.jsonl
uv run python scripts/analyze_detail_flags.py logs/eval_adaptive_snap_hyre_option_table_or-gemma4-26b_20260511_0028_casehold_casehold-option-table-direct-or-gemma4-26b-api-q250-start200-end250-k5-adaptive_snap_hyre_option_table_detail.jsonl
uv run python scripts/analyze_detail_flags.py logs/eval_snap_only_in_final_or-gemma4-26b_20260511_0346_barexam_meeting-missing-ladder-retry-or-gemma4-26b-n200-k5-snap_only_in_final_detail.jsonl
uv run python scripts/analyze_detail_flags.py logs/eval_rag_hyde_or-gemma4-26b_20260511_0526_barexam_meeting-missing-retrieval-fixed-or-gemma4-26b-n200-k5-rag_hyde_detail.jsonl
uv run python scripts/analyze_detail_flags.py logs/eval_snap_only_in_final_or-gemma4-26b_20260511_0259_housing_meeting-missing-ladder-retry-or-gemma4-26b-n200-k5-snap_only_in_final_detail.jsonl
uv run python scripts/analyze_detail_flags.py logs/eval_snap_only_in_final_or-gemma4-26b_20260511_0943_casehold_meeting-capped-snap-casehold-v2-or-gemma4-26b-n200-k5-snap_only_in_final_detail.jsonl
uv run python scripts/analyze_detail_flags.py logs/eval_snap_only_in_final_or-gemma4-26b_20260511_0411_legalbench_scalr_meeting-missing-ladder-retry-or-gemma4-26b-n200-k5-snap_only_in_final_detail.jsonl
uv run python scripts/analyze_detail_flags.py logs/eval_rag_hyde_or-gemma4-26b_20260511_0443_housing_meeting-missing-retrieval-fixed-or-gemma4-26b-n200-k5-rag_hyde_detail.jsonl
uv run python scripts/analyze_detail_flags.py logs/eval_rag_hyde_or-gemma4-26b_20260511_0511_casehold_meeting-missing-retrieval-fixed-or-gemma4-26b-n200-k5-rag_hyde_detail.jsonl
uv run python scripts/analyze_detail_flags.py logs/eval_rag_hyde_or-gemma4-26b_20260511_0559_legalbench_scalr_meeting-missing-retrieval-fixed-or-gemma4-26b-n200-k5-rag_hyde_detail.jsonl
uv run python scripts/analyze_detail_flags.py logs/eval_rag_hyde_or-gemma4-26b_20260511_0734_detail.jsonl
uv run python scripts/analyze_detail_flags.py logs/eval_rag_simple_or-gemma4-26b_20260511_0731_legalbench_scalr_meeting-full-scalr-sanity-or-gemma4-26b-n571-k5-rag_simple_detail.jsonl
uv run python scripts/analyze_detail_flags.py logs/eval_rag_simple_or-gemma4-26b_20260511_1218_legalbench_scalr_meeting-full-scalr-capped-or-gemma4-26b-n571-k5-rag_simple_detail.jsonl
uv run python scripts/analyze_detail_flags.py logs/eval_adaptive_snap_hyre_frontier_or-gemma4-26b_20260511_1513_legalbench_scalr_meeting-full-scalr-capped-or-gemma4-26b-n571-k5-adaptive_snap_hyre_frontier_detail.jsonl
uv run python scripts/audit_adaptive_hyre_logs.py logs/eval_adaptive_snap_hyre_frontier_or-gemma4-26b_20260511_1513_legalbench_scalr_meeting-full-scalr-capped-or-gemma4-26b-n571-k5-adaptive_snap_hyre_frontier_detail.jsonl
uv run python scripts/analyze_detail_flags.py logs/eval_rag_simple_or-gemma4-26b_20260511_1334_casehold_meeting-n500-canonical-or-gemma4-26b-casehold-n500-k5-rag_simple_detail.jsonl
uv run python scripts/analyze_detail_flags.py logs/eval_rag_rewrite_or-gemma4-26b_20260511_1542_legalbench_scalr_meeting-n500-canonical-r2-or-gemma4-26b-legalbench_scalr-n571-k5-rag_rewrite_detail.jsonl
uv run python scripts/analyze_detail_flags.py logs/eval_rag_simple_or-gemma4-26b_20260511_1538_barexam_meeting-n500-canonical-or-gemma4-26b-barexam-n500-k5-rag_simple_detail.jsonl
rg -n "232797|CANCELLED|RESULTS" logs/slurm_67863_full_scalr_cancelled.out
uv run python scripts/analyze_detail_flags.py logs/eval_rag_snap_hyde_2call_or-gemma4-26b_20260511_0559_housing_meeting-missing-retrieval-fixed-or-gemma4-26b-n200-k5-rag_snap_hyde_2call_detail.jsonl
uv run python scripts/analyze_detail_flags.py logs/eval_rag_snap_hyde_2call_or-gemma4-26b_20260511_0602_casehold_meeting-missing-retrieval-fixed-or-gemma4-26b-n200-k5-rag_snap_hyde_2call_detail.jsonl
uv run python scripts/audit_adaptive_hyre_logs.py logs/eval_rag_hyde_or-gemma4-26b_20260511_0559_legalbench_scalr_meeting-missing-retrieval-fixed-or-gemma4-26b-n200-k5-rag_hyde_detail.jsonl logs/eval_rag_snap_hyde_2call_or-gemma4-26b_20260511_0559_housing_meeting-missing-retrieval-fixed-or-gemma4-26b-n200-k5-rag_snap_hyde_2call_detail.jsonl logs/eval_rag_snap_hyde_2call_or-gemma4-26b_20260511_0602_casehold_meeting-missing-retrieval-fixed-or-gemma4-26b-n200-k5-rag_snap_hyde_2call_detail.jsonl
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

Health-check result for BarExam HyRE-only:

- rows: 200
- correct: 164/200 = 82.0%
- average calls: 2.00
- errors: 0
- missing predictions: 0
- empty retrieval: 0
- artifact flags: 0

Health-check result for CaseHOLD snap-only:

- rows: 200
- correct: 145/200 = 72.5%
- average calls: 2.00
- errors: 0
- missing predictions: 0
- long final-answer rows: 0
- max final-answer chars: 6,081
- max output tokens: 4,646
- verdict: clean capped replacement; supersedes health-caveated job `67777`
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

Health-check result for SCALR HyRE-only uncapped:

- rows: 200
- correct: 142/200 = 71.0%
- average calls: 2.00
- errors: 0
- missing predictions: 0
- empty retrieval: 0
- long final-answer rows: 1
- max final-answer chars: 267,458
- max output tokens: 70,593
- verdict: rejected as a clean report number; capped rerun `67864` replaces it
  as wrapper-caveated evidence

Health-check result for SCALR HyRE-only capped:

- rows: 200
- correct: 148/200 = 74.0%
- average calls: 2.00
- errors: 0
- missing predictions: 1
- empty retrieval: 0
- long final-answer rows: 0
- max final-answer chars: 10,107
- max output tokens: 2,395
- verdict: detail-log clean but wrapper-caveated; SLURM failed after results
  because `scripts/postprocess_adaptive_hyre_sweep.py` was missing

Health-check result for full-SCALR `rag_simple` half of job `67863`:

- rows: 571
- correct: 424/571 = 74.3%
- average calls: 1.00
- errors: 0
- missing predictions: 0
- empty retrieval: 0
- long final-answer rows: 3
- max final-answer chars: 233,166
- max output tokens: 73,151
- verdict: structurally complete but health-gated; do not promote as a
  full-corpus report number unless the long-answer rows are resolved or
  explicitly accepted as a caveated sanity baseline

Full-SCALR frontier half of job `67863`:

- stdout source: `logs/slurm_67863_full_scalr_cancelled.out`
- progressed to row 300/571 before cancellation
- row 296 produced a 232,797-character final answer
- no clean frontier detail log was written before cancellation
- verdict: rejected; capped replacement `67897` launched with
  `LLM_MAX_COMPLETION_TOKENS=4096`

Health-check result for HousingQA fixed Snap-HyRE:

- rows: 200
- correct: 103/200 = 51.5%
- average calls: 2.00
- errors: 0
- missing predictions: 0
- empty retrieval: 0
- long final-answer rows: 0
- artifact flags: 0

Health-check result for CaseHOLD fixed Snap-HyRE:

- rows: 200
- correct: 144/200 = 72.0%
- average calls: 2.00
- errors: 0
- missing predictions: 0
- empty retrieval: 0
- long final-answer rows: 0
- artifact flags: 0

## Remaining Risk

This package is meeting-ready, not a finished paper submission. The current
controller is still partly evidence-summary/rule-based rather than a fully
automatic learned router. BarExam and SCALR have route-policy nuance on the
held-out slice, and CaseHOLD still needs a better option-conversion mechanism.
The remaining live row is capped full-SCALR replacement `67897`. Do not promote
it unless both halves finish and pass the health gates. That is a paper
direction, not a blocker for the May 11 meeting package.
