# Documentation Index - LegalRagAgent

Updated 2026-05-12 for branch `snap_hyre_comprehensive`.

## Start Here

1. `../CLAUDE.md` - operational context and branch north star.
2. `snap_hyre_comprehensive_plan_2026-05-12.md` - active research plan:
   fixed Snap-HyRE across four legal benchmarks, three models, top-k
   calibration, retrieval caches, and validation gates.
3. `snap_hyre_experiment_runbook_2026-05-12.md` - concrete method ladder,
   cache workflow, validation gate, launch order, and open decisions.
4. `literature_snap_hyre_2026-05-12.md` - notes from the three downloaded
   related papers and the LegalSearchQA decision.
5. `signoff_log.md` - cite-or-not gate for any reported result.
6. `compiled_results.md` and `../logs/experiments.jsonl` - historical ledger
   and machine-readable summaries.
7. `method_index.md` - local harness mode names.
8. `cluster_workflow.md` and `hpc_setup_log.md` - cluster paths, environment
   notes, and operational caveats.

## Current Branch Narrative

The active direction is no longer a diagnostic adaptive controller. The current
target is a single, straightforward Snap-HyRE method that can be applied across
four legal benchmarks under the same comparison rules.

Primary method:

- `snap_hyre`: one call produces snap reasoning plus a HyRE passage;
  retrieval is conditioned on the HyRE passage; a second call answers using
  retrieved evidence and the original question. Legacy logs may call the same
  structure `rag_snap_hyde_2call`.

Primary comparison rows:

- `llm_only`
- `rag_simple`
- `rag_hyde`
- `snap_hyre`
- `golden_passage`
- `golden_plus_neighbors`
- `rag_rewrite`

Primary benchmarks:

- BarExamQA
- HousingQA
- CaseHOLD
- LegalBench-SCALR

MuSiQue and other non-legal datasets are not active main-report benchmarks on
this branch.

## Current Claim Gates

Use `signoff_log.md` first. If a number is absent, pending, rejected, or
caveated there, do not promote it as a clean claim.

Useful source-gated result docs still live at top level:

- `snap_hyde_2call_2026-04-28.md`
- `top1_ablation_2026-04-28.md`
- `casehold_repaired_rerun_2026-05-01.md`
- `housing_state_filter_followup_2026-05-01.md`
- `scalr_depth_disagreement_2026-04-30.md`
- `rag_rewrite_barexam_n200_2026-05-10.md`
- `rag_rewrite_baseline_n50_2026-05-10.md`
- `snap_only_controls_2026-05-11.json`

Older diagnostic-controller and adaptive-route artifacts were archived under
`archive/diagnostic_adaptation_2026-05-12/`. They remain provenance, not active
entrypoints.

## Active Analysis Helpers

- `../scripts/analyze_detail_flags.py` - detail-log health scan.
- `../scripts/score_retrieval_qrels.py` - Hit/Recall, MRR, nDCG scoring from
  retrieved ids and gold ids/qrels.
- `../scripts/build_hyre_cache.py` - build Snap-HyRE generation replay caches.
- `../scripts/build_retrieval_cache.py` - build deterministic passage-id
  retrieval caches for raw question, Snap-HyRE, and golden-neighbor queries.
- `../scripts/audit_retrieval_cache.py` - audit cache integrity and Hit/Recall,
  MRR at multiple k values before answer generation.
- `../scripts/merge_detail_logs.py` - merge chunked detail logs.
- `../scripts/compute_mcnemar.py` - paired significance tests.
- `../scripts/audit_golden_paradox.py` - BarExam golden-passage paradox audit.

## Archive Map

- `archive/diagnostic_adaptation_2026-05-12/` - May 9-11 adaptive/controller
  sprint docs and generated snapshots.
- `archive/legacy_working_notes_2026-05-12/` - older brainstorms, MuSiQue
  notes, pre-pivot paper drafts, router probes, and stale meeting notes.
- `archive/` and `archive_2026-04-27/` - earlier historical material.

Preserve archives for traceability. Prefer adding a current summary over
deleting old evidence.
