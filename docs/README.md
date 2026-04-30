# Documentation Index - LegalRagAgent

Updated 2026-04-30. This file is the consolidation layer for the paper sprint:
use it to decide which docs are current, which are evidence ledgers, and which
are historical context only.

## Current Reading Path

1. `../CLAUDE.md` - operational source of truth for agents: commands, env notes,
   current headline, and methodology gates.
2. `signoff_log.md` - cite-or-not gate for paper claims. If a result is not
   approved or explicitly caveated here, treat it as not paper-grade.
3. `snap_hyde_2call_2026-04-28.md` - current bottleneck-taxonomy pivot:
   `snap_hyde_2call`, MuSiQue mechanism, BarExam 2-call directional result,
   CaseHOLD/LegalBench-SCALR option-disambiguation replicate.
4. `top1_ablation_2026-04-28.md` - retrieval-depth signature: MuSiQue top-1
   collapse vs BarExam top-1/top-5 flatness.
5. `research_strategy_2026-04-30.md` - current strategy reset: novelty
   boundary, grounded motivation, harness gaps, and recommended next
   experiments.
6. `adaptive_controller_design_2026-04-30.md` - feasible path for fast
   bottleneck-aware method routing and agentic escalation.
7. `agentic_legal_rag_angles_2026-04-30.md` - research-angle memo for workshop
   vs EMNLP positioning, multi-agent evidence sharing, and legal-agent novelty.
8. `evidence_budgeted_ledger_router_plan.md` - branch plan combining
   bottleneck-aware routing with a shared evidence ledger for legal agents.
9. `paper_narrative_2026-04-28.md` - current paper skeleton after the pivot.
10. `compiled_results.md` - audited result ledger with direct log paths, caveats,
   and per-row provenance.

## Evidence Ledgers

- `logs/experiments.jsonl` - machine-readable run summaries.
- `evidence_matrix_2026-04-30.md` - reproducible matrix generated from landed
  detail logs by `scripts/build_evidence_matrix.py`.
- `router_oracle_musique_2026-04-30.md` and
  `router_oracle_barexam_2026-04-30.md` - offline routing headroom generated
  by `scripts/evaluate_routing_oracle.py`.
- `audit_log.md` - BarExam post-fix audit truth and historical guardrails.
- `mcnemar_2026-04-27.md` - paired tests from the pre-pivot MuSiQue matrix.
- `docs/audits/` - focused audit artifacts by dataset/model/date.
- `verification_2026-04-27.md` - older source-gating pass; useful for why some
  2026-04-27 claims stayed provisional.

## Current Planning

- `action_items.md` - paper-sprint task list. Some older sections remain for
  audit continuity; prefer dated update blocks at the top.
- `meeting_notes_042726.md` - meeting handoff that motivated golden-passage,
  top-k, 2-call, and dataset-matrix work.
- `datasets_frames_scalr_2026-04-28.md` - FRAMES and SCALR scoping. Important:
  LegalBench-SCALR 5-way MC has landed; MLEB-SCALR retrieval-only is still a
  separate unresolved benchmark.
- `lit_review_2026-04-28.md` - literature grounding for the bottleneck taxonomy.

## Historical Or Superseded Context

These files can still be useful, but do not use them as the first source for
current paper claims:

- `narrative_2026_04_27.md` - superseded by `paper_narrative_2026-04-28.md`.
- `experiment_overview.md` - high-level experiment summary; verify any numbers
  against current signoff and result ledgers before citing.
- `meeting_2026_04_17.md` - older meeting notes, now mostly historical.
- `validation_log_2026-04-25.md` - running validation log from an earlier phase.
- `golden_paradox_audit_2026-04-27.md` and
  `methods_vs_golden_audit_2026-04-27.md` - still useful for mechanism details,
  but narrower than the current paper framing.
- `archive_2026-04-27/` and `archive/` - archived working docs retained for
  traceability.

## Presentation Materials

- `presentation/00_index.md` - entrypoint for presentation docs.
- `presentation/01_results_tables.md` - presentation-facing result tables.
- `presentation/02_methods_explained.md` - method descriptions.
- `presentation/03_takeaways.md` - talk takeaways.
- `presentation/04_datasets_and_models.md` - dataset/model context.
- `presentation/05_logs_index.md` - source logs behind presentation claims.
- `presentation/06_next_steps.md` - presentation-facing next steps.

## Cluster / HPC

- `hpc_setup_log.md`
- `hpc_throughput.md`
- `hpc_qwen3_8b_eval.md`
- `hpc_qwen3_8b_baseline_golden.md`
- `cluster_workflow.md`

## Validation Rule

Before citing a result:

1. Find its signoff status in `signoff_log.md`.
2. Confirm the source log or audit path in `compiled_results.md`,
   `snap_hyde_2call_2026-04-28.md`, or `top1_ablation_2026-04-28.md`.
3. For paired claims, verify the McNemar row and b/c counts where available.
4. If sources disagree, prefer the newest audit/signoff document and preserve
   the disagreement as a caveat rather than silently reconciling it.

Branch: `hpc-setup`.
