# Documentation Index - LegalRagAgent

Updated 2026-05-01. This is the repo map. It keeps the current path short and
separates citeable state from historical working notes.

## Start Here

1. `../CLAUDE.md` - operational context for agents: active snapshot, commands,
   environment notes, methodology gates, and known cluster caveats.
2. `meeting_state_2026-05-01.md` - meeting-ready synthesis of current findings,
   open blockers, live jobs, and defensible interpretation.
3. `signoff_log.md` - cite-or-not gate. If a result is absent, `PENDING`, or
   explicitly caveated here, do not promote it to a paper-grade claim.
4. `compiled_results.md` plus `../logs/experiments.jsonl` - audited result
   ledger and machine-readable run summaries.
5. `benchmark_method_birdseye_2026-04-30.md` - compact map of benchmarks,
   methods, what each dataset tests, and harness gaps.
6. `final_class_report_2026-04-30.pdf` or `final_class_report_2026-04-30.tex`
   - current class-report draft with figures/tables.

Five-minute path: read `meeting_state_2026-05-01.md`, then use
`signoff_log.md` and `compiled_results.md` to verify any number before citing
it.

## Current Claim Gates

- Bottleneck-taxonomy pivot: `snap_hyde_2call_2026-04-28.md`.
- Retrieval-depth signature: `top1_ablation_2026-04-28.md`.
- MuSiQue disagreement buckets and golden-passage control:
  `musique_disagreement_audit_2026-04-30.md` and
  `musique_golden_passage_2026-04-30.md`.
- CaseHOLD repaired pair: `casehold_repaired_rerun_2026-05-01.md`.
- Housing state-filter caveat and resubmission: `housing_state_filter_followup_2026-05-01.md`.
- SCALR depth behavior: `scalr_depth_disagreement_2026-04-30.md`.
- BarExam post-fix audit truth: `audit_log.md`.

Use these after `signoff_log.md`; they explain mechanisms and caveats, but the
signoff log decides whether a claim is ready to cite.

## If You Need...

### Research Story And Literature

- `research_strategy_2026-04-30.md` - novelty boundary, grounded motivation,
  and recommended next experiments.
- `mechanism_literature_synthesis_2026-04-30.md` - how the observed bottlenecks
  connect to RAG/legal-RAG literature and likely gaps.
- `search_space_consolidation_2026-04-30.md` - keep/pause/kill decisions across
  current methods.
- `agentic_legal_rag_angles_2026-04-30.md` - workshop vs EMNLP positioning and
  multi-agent evidence-sharing angles.
- `paper_narrative_2026-04-28.md` - paper skeleton after the pivot. Prefer
  `meeting_state_2026-05-01.md` for current wording.

### Methods, Metrics, And Controllers

- `method_index.md` - mode taxonomy and local method names.
- `dataset_metric_consolidation_2026-04-30.md` - Speculative-RAG metric mapping,
  LegalBench/Legal RAG Bench dataset notes, and wiring gaps.
- `specrag_lite_diagnostic_controller_2026-04-30.md` - selective escalation
  proposal based on cheap bottleneck diagnostics.
- `adaptive_controller_design_2026-04-30.md` - feasible bottleneck-aware routing
  design.
- `evidence_budgeted_ledger_router_plan.md` - branch plan for a shared evidence
  ledger plus bottleneck-aware method routing.
- `router_probe_findings_2026-04-30.md`,
  `router_oracle_musique_2026-04-30.md`,
  `router_oracle_barexam_2026-04-30.md`,
  `router_baseline_report_2026-04-30.md`, and
  `router_baseline_housing_depth_2026-04-30.md` - offline routing headroom and
  failure modes.

### Dataset Evidence

- MuSiQue: `musique_disagreement_audit_2026-04-30.md`,
  `musique_golden_passage_2026-04-30.md`.
- BarExam: `audit_log.md`, `top1_ablation_2026-04-28.md`,
  `snap_hyde_2call_2026-04-28.md`.
- HousingQA: `housing_speculative_metrics_2026-04-30.md`,
  `housing_metadata_depth_audit_2026-04-30.md`,
  `housing_state_filter_followup_2026-05-01.md`.
- CaseHOLD: `casehold_flatness_audit_2026-04-30.md`,
  `casehold_gold_mapping_repair_2026-04-30.md`,
  `casehold_repaired_rerun_2026-05-01.md`.
- LegalBench-SCALR / MLEB-SCALR: `scalr_depth_disagreement_2026-04-30.md`,
  `mleb_scalr_retrieval_baseline_2026-04-30.md`,
  `mleb_scalr_embedding_ab_2026-04-30.md`.

### Generated Evidence Artifacts

- `evidence_matrix_2026-04-30.md` - generated hypothesis/result matrix.
- `run_manifest_2026-04-30.json` - machine-readable map from detail logs to
  hypotheses, interventions, bottleneck regimes, and caveats.
- `speculative_metrics_report_2026-04-30.md` - Speculative-RAG-aligned metrics
  generated from landed detail logs.
- `scripts/build_evidence_matrix.py`,
  `scripts/build_speculative_metrics_report.py`,
  `scripts/audit_disagreement_buckets.py`,
  `scripts/evaluate_routing_oracle.py`, and
  `scripts/score_retrieval_qrels.py` - reusable analysis scripts behind the
  reports.

### Presentation And Report Materials

- `final_class_report_2026-04-30.tex` and `final_class_report_2026-04-30.pdf`
  - current class-report draft.
- `report_adversarial_pass_2026-04-30.md` - reviewer-style critique and gaps to
  substantiate.
- `presentation/00_index.md` - presentation entrypoint.
- `presentation/01_results_tables.md` - talk-facing result tables.
- `presentation/02_methods_explained.md` - method descriptions.
- `presentation/03_takeaways.md` - talk takeaways.
- `presentation/04_datasets_and_models.md` - dataset/model context.
- `presentation/05_logs_index.md` - source logs behind presentation claims.
- `presentation/06_next_steps.md` - talk-facing next steps.

### Cluster And Reproducibility

- `hpc_setup_log.md` - cluster SSH, paths, venvs, model caches, and bad nodes.
- `cluster_workflow.md` - practical cluster run workflow.
- `hpc_throughput.md`, `hpc_qwen3_8b_eval.md`,
  `hpc_qwen3_8b_baseline_golden.md` - older cluster/model timing notes.
- `rigour_signoff.md` - methodology checklist before new claims.

## Historical Or Superseded

These files are retained for traceability. Do not use them as current claim
sources without re-checking `signoff_log.md` and the detail logs.

- `../RESEARCH.md` - historical running research log; useful for process, not
  the current entrypoint.
- `../EXPERIMENTS.md` - append-only experiment chronology; newest claim gates
  live in this docs index and the signoff log.
- `narrative_2026_04_27.md` - superseded by the bottleneck-taxonomy pivot.
- `experiment_overview.md` - high-level summary from an earlier consolidation.
- `meeting_2026_04_17.md`, `meeting_notes_042726.md`, and `action_items.md`
  - older meeting and sprint notes.
- `mcnemar_2026-04-27.md` - older paired-stat ledger for the multi-HyDE phase;
  use current dataset gates before quoting it.
- `golden_paradox_audit_2026-04-27.md`,
  `methods_vs_golden_audit_2026-04-27.md`, and
  `verification_2026-04-27.md` - useful mechanism context, but narrower than
  the current framing.
- `archive_2026-04-27/` and `archive/` - archived working docs retained for
  audit continuity.

## Validation Rule

Before citing a result:

1. Check `signoff_log.md`.
2. Confirm the source detail log or audit path in `compiled_results.md`,
   `snap_hyde_2call_2026-04-28.md`, `top1_ablation_2026-04-28.md`, or the
   dataset-specific gate above.
3. For paired claims, verify McNemar b/c counts where available.
4. If sources disagree, prefer the newest audit/signoff document and preserve
   the disagreement as a caveat.

Branch: `codex/evidence-ledger-router`.
