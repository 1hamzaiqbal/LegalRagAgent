# Documentation Index - LegalRagAgent

## Current map — 2026-07-17

The May Snap-HyRE index below is preserved for provenance, but its branch
narrative is superseded. Start here now:

1. `../wiki/snapshots/research-state-2026-07-17.md` — durable state snapshot,
   strongest findings, literature boundary, and decision gates.
2. `../wiki/tracks/three-dial.md` — primary research track.
3. `../wiki/tracks/opd-distillation.md` — gated engineering/distillation track.
4. `july_2026_completion_audit_2026-07-17.md` — local/EIT job and evidence
   reconciliation.
5. `signoff_log.md` — cite-or-not gate.
6. `../wiki/literature/index.md` — Obsidian navigation into the persistent EIT
   paper/repository vault.

Branches: `codex/three_dial`, `codex/opd_distillation`, and historical
`codex/scope_old`. The SCOPE/Snap-HyRE method is not the active framing.

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
5. `local_api_mirror_setup_2026-05-12.md` - optional local Chroma/API mirror
   setup to avoid SSH/SLURM for API-backed iteration.
6. `local_snap_hyre_handoff_2026-05-12.md` - pasteable local-machine handoff
   for API keys, embedding, retrieval caches, top-k diagnostics, full sweeps,
   and promotion gates.
7. `local_validation_goal_2026-05-12.md` - short-lived local validation goal
   to complete before launching the full comprehensive eval grid.
8. `snap_hyre_completion_audit_2026-05-12.md` - objective-to-artifact checklist
   showing what is present, missing, and still unverified.
9. `snap_hyre_prelaunch_readiness_2026-05-13.md` - pre-current benchmark-set
   launch-readiness provenance; useful for strict-provider/cache decisions, but
   not the active CaseHOLD/SCALR-inclusive launch plan.
10. `choice_aware_retrieval_probe_2026-05-13.md` - probe-only q20 retrieval
   diagnostics separating raw retrieval, choice exposure, generated legal
   query style, Snap-HyRE, and choice-conditioned Snap-HyRE.
11. `choice_aware_retrieval_q50_2026-05-14.md` - probe-only q50 follow-up on
   historical SCALR and CaseHOLD slices for blind/choice-aware HyDE, Snap-HyRE,
   diverse HyDE, and choice-conditioned Snap-HyRE.
12. `top_k_prelaunch_probe_2026-05-14.md` - q100 retrieval k=1..10 provenance
   plus limited BarExam downstream k=5 vs k=10 gate for the shared answer `k`;
   the source slice predates the current Legal-Link-EU / MASLegalBench matrix.
13. `comprehensive_run_status_2026-05-14.md` - superseded run-status ledger for
   the earlier CaseHOLD/SCALR-inclusive comprehensive queue; use
   `current_status.md`, `paper_iteration_signal_2026-05-20.md`, and
   `signoff_log.md` for current paper-facing status.
14. `candidate_benchmark_feasibility_2026-05-18.md` - feasibility note for
   Legal RAG Bench, LegalSearchQA, LEXam, MLEB retrieval sets,
   Legal-Link-EU, MASLegalBench, and other possible replacement or
   supplemental legal benchmarks.
15. `snap_hyre_failure_audit_2026-05-20.md` - current failure analysis for
   Snap-HyRE vs raw/simple retrieval, including anchor-loss, jurisdiction-loss,
   and harness-health checks.
16. `housingqa_state_filtered_process_2026-05-20.md` - HousingQA-specific
   state-filter process contract, cache naming, and reporting split for the
   national-corpus vs jurisdiction-corpus views.
17. `benchmark_paradigm_audit_2026-05-20.md` - fairness/paradigm audit
   confirming active benchmarks use real reference corpora, and explaining why
   CaseHOLD/LegalBench-SCALR remain excluded from the main matrix.
18. `paper_iteration_signal_2026-05-20.md` - paper/eval-agent handoff note for
   provisional rows, live tmux/log checks, and non-citable probe status.
19. `paper_meeting_handoff_2026-05-20.md` - meeting/writing handoff for the
   strongest current Snap-HyRE narrative, caveats, and provisional exemplar /
   state-filter context.
20. `snap_hyre_paper_agent_handoff_2026-05-20.md` - focused paper-agent handoff
   for positive Snap-HyRE claims, Gemma 26B emphasis, starred active/probe
   rows, and exact files to inspect.
21. `snap_hyre_good_example_handoff_2026-05-20.md` - concrete positive
   BarExamQA row where canonical Snap-HyRE fixes raw RAG/HyDE/rewrite, plus
   the current exemplar-worth-it read.
22. `barexam_housing_core_focus_2026-05-20.md` - narrowed operating note for
   finishing BarExamQA and state-filtered HousingQA on the three required core
   methods: `rag_simple`, `rag_hyde`, and `snap_hyre`.
23. `housingqa_statefilter_goal_checklist_2026-05-21.md` - explicit
   prompt-to-artifact checklist for the active HousingQA state-filter/exemplar
   goal, including required rows, queued jobs, validation gates, and exemplar
   promotion rule.
24. `housingqa_handoff_to_next_agent_2026-05-21.md` - concise HousingQA
   takeover message with current 6/9 core status, Gemma blockers, exact
   continuation commands, and files another agent should inspect.
25. `latest_results_handoff_2026-05-21.md` - current results/navigation
   handoff for another agent, including where to read first, what is signed,
   current Housing blockers, and continuation commands.
26. `signoff_log.md` - cite-or-not gate for any reported result.
27. `compiled_results.md` and `../logs/experiments.jsonl` - historical ledger
   and machine-readable summaries.
28. `method_index.md` - local harness mode names.
29. `cluster_workflow.md` and `hpc_setup_log.md` - cluster paths, environment
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
- Legal-Link-EU
- MASLegalBench

CaseHOLD and LegalBench-SCALR are historical/superseded for the active main
matrix unless explicitly re-added. Legal RAG Bench is tracked as a retrieval /
open-answer appendix candidate, not part of the exact-scored grid.

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
- `../scripts/build_generation_cache.py` - build full question-only HyDE and
  Snap-HyRE generation caches before answer sweeps.
- `../scripts/build_retrieval_cache.py` - build deterministic passage-id
  retrieval caches for raw question, HyDE, Snap-HyRE, and golden-neighbor
  queries.
- `../scripts/build_retrieval_doc_cache.py` - hydrate retrieval-cache passage
  IDs into a strict document-text cache for replaying large cached cells
  without reopening a large Chroma collection.
- `../scripts/local/` - API-first local runner scripts for provider smoke
  tests, retrieval-cache construction, and one-cell answer sweeps.
- `../scripts/check_expected_provider_model.py`,
  `../scripts/check_openrouter_key_status.py`, and
  `../scripts/check_openrouter_chat_route.py` - fail-closed provider/model,
  OpenRouter budget, and tiny exact-route chat-completion guards before long
  API-backed rows.
- `../scripts/local/watch_housing_gemma_until_ready.sh` - optional
  non-launching-by-default Housing Gemma watcher for OpenRouter reset windows;
  set `LAUNCH_ON_READY=1` only when the canonical continuation should start
  automatically after the exact route/budget preflight passes.
- `../scripts/local/housing_gemma_budget_watcher.sh` - status/start/stop
  manager for the detached Housing Gemma watcher, including stale lock/process
  checks; non-launching by default unless `LAUNCH_ON_READY=1` is set.
- `../scripts/audit_retrieval_cache.py` - audit cache integrity and Hit/Recall,
  MRR at multiple k values before answer generation.
- `../scripts/run_choice_aware_retrieval_probe.py` - probe-only retrieval
  diagnostics for blind/choice-aware HyDE, Snap-HyRE anchors, diverse HyDE, and
  choice-conditioned Snap-HyRE before promoting variants.
- `../scripts/audit_retrieval_id_alignment.py` - verify dataset gold ids are
  actual Chroma document ids before promoting Hit@k/MRR claims.
- `../scripts/compile_retrieval_cache_matrix.py` - compile cache audits into
  top-k selection tables.
- `../scripts/compile_efficiency_metrics.py` - compile offline token, latency,
  actual/logical-call, and cache-health efficiency snapshots from detail JSONL
  logs.
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
