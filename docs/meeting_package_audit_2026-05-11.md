# Meeting Package Completion Audit - 2026-05-11

Objective audited:

> Produce a meeting-ready, source-gated diagnostic adaptation package for legal
> RAG by May 12 at 4pm: consolidate four legal benchmarks, verify logs/results,
> build inherited ablation tables, diagrams, and bottleneck summaries, and use
> targeted runs only where they clarify the controller story. The narrative
> should show that calibration traces can route among baseline RAG, query
> rewrite, Snap-HyRE/HyRE, metadata filtering, option grounding, verifier
> policies, and disagreement/reject policies, with clear evidence for what
> works, what fails, and what needs to be improved for a paper.

## Artifact Checklist

| Requirement | Evidence | Status |
|---|---|---|
| Consolidate four legal benchmarks | `docs/meeting_prep_2026-05-12_diagnostic_adaptation.md` uses BarExam, HousingQA, CaseHOLD, and LegalBench-SCALR; MuSiQue is explicitly excluded from the main table. | Done |
| Verify calibration results | `docs/diagnostic_controller_portfolio_comparison_2026-05-10.json` backs the calibration table: baseline macro 71.9%, fixed HyRE 74.8%, controller 77.9%. | Done |
| Verify held-out results | `docs/heldout_controller_eval_2026-05-10.json` and `docs/heldout_query_rewrite_2026-05-10.json` back the held-out table: baseline 71.5%, rewrite 75.5%, controller 77.5%. | Done |
| Verify the live CaseHOLD targeted run | SLURM job `67744` completed with exit `0:0`; `analyze_detail_flags.py` and `audit_adaptive_hyre_logs.py` passed on the copied detail log. | Done |
| Build inherited ablation tables | Markdown tables in `docs/meeting_prep_2026-05-12_diagnostic_adaptation.md`; slide-ready PNGs `12_diagnostic_adaptation_calibration_ablation.png` and `13_diagnostic_adaptation_heldout_ablation.png`. | Done |
| Build diagrams | Mermaid controller diagram in the meeting prep; PNG route-map and macro-lift figures in `docs/presentation/figures/`. | Done |
| Build bottleneck summaries | `docs/meeting_prep_2026-05-12_diagnostic_adaptation.md` has the benchmark table and bottleneck summary table; `15_bottleneck_diagnostic_route_map.png` visualizes it. | Done |
| Use targeted runs only | The only new run incorporated here is CaseHOLD direct option-table held-out job `67744`, launched to resolve the specific option-table blocker. | Done |
| Show routes across all named intervention families | The meeting prep and route map cover baseline RAG, query rewrite, Snap-HyRE/HyRE, metadata/state filtering, option grounding, verifier policies, disagreement arbitration, and reject/escalate. | Done |
| Show what works | Housing verifier and CaseHOLD diverse HyRE have the clearest held-out gains; controller macro improves by +6.0pp in calibration and held-out summaries. | Done |
| Show what fails | CaseHOLD direct option table is clean but weak: 70.0%, below query rewrite 76.0% and diverse HyRE 78.0%; replay selector is 66.0%. | Done |
| Show what needs improvement for a paper | Meeting prep identifies BarExam rewrite-vs-HyRE selection, SCALR disagreement arbitration, and deeper CaseHOLD option-conversion mechanisms. | Done |

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

The figure builder reads only source-gated JSON summaries:

- `docs/diagnostic_controller_portfolio_comparison_2026-05-10.json`
- `docs/heldout_controller_eval_2026-05-10.json`
- `docs/heldout_query_rewrite_2026-05-10.json`

## Commands Re-Run During This Audit

```bash
uv run python scripts/build_meeting_package_figures.py
uv run python scripts/audit_adaptive_hyre_logs.py logs/eval_adaptive_snap_hyre_option_table_or-gemma4-26b_20260511_0028_casehold_casehold-option-table-direct-or-gemma4-26b-api-q250-start200-end250-k5-adaptive_snap_hyre_option_table_detail.jsonl
uv run python scripts/analyze_detail_flags.py logs/eval_adaptive_snap_hyre_option_table_or-gemma4-26b_20260511_0028_casehold_casehold-option-table-direct-or-gemma4-26b-api-q250-start200-end250-k5-adaptive_snap_hyre_option_table_detail.jsonl
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

## Remaining Risk

This package is meeting-ready, not a finished paper submission. The current
controller is still partly evidence-summary/rule-based rather than a fully
automatic learned router. BarExam and SCALR have route-policy nuance on the
held-out slice, and CaseHOLD still needs a better option-conversion mechanism.
Those are paper directions, not blockers for the May 12 meeting package.

