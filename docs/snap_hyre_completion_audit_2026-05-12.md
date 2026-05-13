# Snap-HyRE Completion Audit - 2026-05-12

This audit maps the active branch goal to concrete artifacts. It is not a
result summary; it is a gate for deciding what remains before the goal is
complete.

## Objective Restated As Deliverables

Goal: produce a full-corpus, retrieval-first Snap-HyRE evaluation package across
BarExamQA, HousingQA, CaseHOLD, and LegalBench-SCALR using Gemma E4B, Gemma 4
26B, and Llama 3.3 70B. Select a universal top-k from cached retrieval
diagnostics, run the canonical ablation ladder with source-gated logs, and
produce verified tables, plots, and concise analysis showing where Snap-HyRE
improves retrieval exposure and whether that transfers to downstream accuracy.

Concrete success criteria:

1. Four legal benchmarks are set up with populated corpora/collections.
2. Three API-backed model labels are smoke-tested: `or-gemma3n-e4b`,
   `or-gemma4-26b`, and `groq-llama70b`.
3. Retrieval caches exist for full-corpus raw-question retrieval and relevant
   replayable generated-query methods.
4. Qrel alignment audits exist before any Hit@k/MRR claim is promoted.
5. A universal top-k is selected from cached diagnostics, or `k=5` is retained
   as the predeclared fallback with curve evidence.
6. The canonical ablation ladder is run full-corpus for each benchmark/model:
   `llm_only`, `rag_simple`, `rag_rewrite`, `rag_hyde`, `snap_hyre`,
   `golden_passage`, and `golden_plus_neighbors`.
7. Detail logs are validated with `scripts/analyze_detail_flags.py` and any
   failures/caveats are explicitly recorded.
8. Source-gated tables and plots are generated from validated logs/caches.
9. Concise analysis explains retrieval exposure lift and downstream transfer.
10. The branch is cleanly pushed with current docs and scripts.

## Prompt-to-Artifact Checklist

| Requirement | Current evidence | Status |
|---|---|---|
| Fixed-method Snap-HyRE branch narrative | `CLAUDE.md`, `docs/README.md`, `docs/snap_hyre_comprehensive_plan_2026-05-12.md`, `docs/snap_hyre_experiment_runbook_2026-05-12.md` | Present |
| Local pivot handoff | `docs/local_snap_hyre_handoff_2026-05-12.md` | Present |
| API-first provider labels | `llm_config.py`, `docs/snap_hyre_experiment_runbook_2026-05-12.md` | Present |
| Local provider/harness smoke scripts | `scripts/local/run_api_smoke.sh` | Present, not yet run locally |
| Local retrieval-cache runner | `scripts/local/build_retrieval_caches.sh` | Present, not yet run locally |
| Local answer-cell runner | `scripts/local/run_answer_cell.sh` | Present, not yet run locally |
| Four benchmark full embeddings available locally | `chroma_db/` expected under local checkout | Missing on this Mac; planned for local machine |
| Full raw/golden retrieval caches for all four datasets | `caches/retrieval/full/*.jsonl` | Incomplete/stale; must be regenerated or copied and audited |
| HyDE/Snap-HyRE generation caches for all three providers and four datasets | `caches/hyre/full/*.jsonl` | Missing/incomplete |
| Universal top-k selected from cache matrix | `docs/generated/retrieval_cache_matrix.md` | Missing |
| Full-corpus answer ladders for all dataset/model cells | `logs/eval_*detail.jsonl` plus `logs/experiments.jsonl` | Missing |
| Detail-log health validation for promoted rows | `scripts/analyze_detail_flags.py` output and `docs/signoff_log.md` | Missing for new full-corpus package |
| Retrieval exposure metrics | `scripts/audit_retrieval_cache.py`, `scripts/compile_retrieval_cache_matrix.py`, qrel alignment reports | Tooling present; full evidence missing |
| BarExam qrel policy | `docs/local_snap_hyre_handoff_2026-05-12.md` caveat | Decision recorded; full retrieval fix/selection pending |
| Tables and plots for final package | `docs/generated/` and figure outputs | Missing |
| Branch pushed | `snap_hyre_comprehensive` on `shrango` | Present as of the latest pushed commit before this audit |

## Current Completion Verdict

Not complete.

The branch now has a clear plan, handoff, provider naming, local runner scripts,
and validation gates. The actual full-corpus package is still missing because
the local machine has not yet produced the required embeddings/caches, full
answer logs, top-k matrix, tables, plots, and signoff entries.

## Next Concrete Gates

1. On the capable local machine, pull `snap_hyre_comprehensive`, set `.env`, and
   run `scripts/local/run_api_smoke.sh`.
2. Populate or rebuild `datasets/` and `chroma_db/`, then run
   `scripts/local/build_retrieval_caches.sh`.
3. Inspect `docs/generated/retrieval_cache_matrix.md` and
   `caches/retrieval/full/retrieval_id_alignment_*.txt`; decide whether BarExam
   retrieval is train-aligned only or uses an augmented qrel-complete collection.
4. Select universal `RETRIEVAL_K`.
5. Build full `rag_hyde` and `snap_hyre` generation caches one provider/dataset
   at a time.
6. Run `scripts/local/run_answer_cell.sh` for one dataset/model cell, validate
   logs, then scale across the remaining cells.
7. Add only validated rows to `docs/signoff_log.md`, then regenerate result
   tables and plots.

