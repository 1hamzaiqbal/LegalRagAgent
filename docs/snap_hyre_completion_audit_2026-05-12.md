# Snap-HyRE Completion Audit - 2026-05-12

> **Superseded benchmark-set note (2026-05-20):** this file is retained as the
> original completion-gate audit for the May 12 branch pivot. The active paper
> matrix is now BarExamQA, HousingQA, Legal-Link-EU, and MASLegalBench; CaseHOLD
> and LegalBench-SCALR are historical/provenance unless explicitly re-added. For
> current result claims, prefer `docs/signoff_log.md`,
> `docs/compiled_results.md`, `current_status.md`, and
> `docs/paper_iteration_signal_2026-05-20.md`.

This audit maps the active branch goal to concrete artifacts. It is not a
result summary; it is a gate for deciding what remains before the goal is
complete.

## Objective Restated As Deliverables

Goal: produce a full-corpus, retrieval-first Snap-HyRE evaluation package across
BarExamQA, HousingQA, Legal-Link-EU, and MASLegalBench using Groq Llama 3.1 8B,
Gemma 4 26B, and Llama 3.3 70B. Select a universal top-k from cached retrieval
diagnostics, run the canonical ablation ladder with source-gated logs, and
produce verified tables, plots, and concise analysis showing where Snap-HyRE
improves retrieval exposure and whether that transfers to downstream accuracy.

Concrete success criteria:

1. Four active legal benchmarks are set up with populated corpora/collections.
2. Three model labels are smoke-tested without substituting checkpoints:
   `groq-llama8b`, `or-gemma4-26b`, and `groq-llama70b`.
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
| Local generation-cache runner | `scripts/local/build_generation_caches.sh` | Present, not yet run locally |
| Local answer-cell runner | `scripts/local/run_answer_cell.sh` | Present, not yet run locally |
| Local result-package builder | `scripts/build_snap_hyre_package.py`, `scripts/local/build_result_package.sh` | Present, not yet populated with local full-corpus evidence |
| Four benchmark full embeddings available locally | `chroma_db/` expected under local checkout | Missing on this Mac; planned for local machine |
| Full raw/golden retrieval caches for all four datasets | `caches/retrieval/full/*.jsonl` | Incomplete/stale; must be regenerated or copied and audited |
| HyDE/Snap-HyRE generation caches for all three providers and four datasets | `caches/hyre/full/*.jsonl` | Missing/incomplete |
| Universal top-k selected from cache matrix | `docs/generated/retrieval_cache_matrix.md` | Missing |
| Full-corpus answer ladders for all dataset/model cells | `logs/eval_*detail.jsonl` plus `logs/experiments.jsonl` | Missing |
| Detail-log health validation for promoted rows | `scripts/analyze_detail_flags.py` output and `docs/signoff_log.md` | Missing for new full-corpus package |
| Retrieval exposure metrics | `scripts/audit_retrieval_cache.py`, `scripts/compile_retrieval_cache_matrix.py`, qrel alignment reports | Tooling present; full evidence missing |
| BarExam qrel policy | `docs/local_snap_hyre_handoff_2026-05-12.md` caveat | Decision recorded; full retrieval fix/selection pending |
| Tables and plots for final package | `scripts/build_snap_hyre_package.py` writes `docs/generated/snap_hyre_package/` | Tooling present; full evidence missing |
| Branch pushed | `snap_hyre_comprehensive` on `shrango` | Present as of the latest pushed commit before this audit |

## Current Completion Verdict

Not complete.

The branch now has a clear plan, handoff, provider naming, local runner scripts,
validation gates, a populated local WSL Chroma mirror, 100% qrel-aligned
corpora for the four comprehensive datasets, raw/golden retrieval-cache smokes,
one source-gated N=50 answer-cell validation, generated status tables, and a
signoff entry for the local validation gate.

The actual comprehensive package is still incomplete because most
dataset/model/method answer cells are missing, HyDE/Snap-HyRE generation caches
exist only for `legalbench_scalr` × `gemma4-26b` N=50 plus a partial BarExam
cache, and the replacement small-model API axis still needs strict smoke
validation.

## Local Validation Snapshot - 2026-05-13

- WSL/local setup verified at commit `4e6236d`.
- Canonical retrieval stack confirmed and cached locally:
  `Alibaba-NLP/gte-large-en-v1.5` plus
  `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- Local Chroma collections:
  `legal_passages` 856,835 docs, `housing_statutes` 1,837,403 docs,
  rebuilt `casehold_holdings` 51,296 docs, and
  `legalbench_scalr_holdings` 1,733 docs.
- Local runner defaults were hardened so `DISABLE_CROSS_ENCODER=0` unless a
  dense-only speed smoke is explicitly requested.
- API smoke passed for `or-gemma4-26b`.
- `or-gemma3n-e4b` was smoke-tested, but it is Gemma 3n E4B and should not be
  counted as the Gemma 4 E4B comprehensive axis. This older note predates the
  current `groq-llama8b` small-model row.
- `groq-llama70b` initially failed with a Groq 401 invalid-key preflight; after
  replacing the key on 2026-05-13, a one-question strict smoke passed.
- BarExamQA was repaired by appending 170,511 validation/test passages to the
  existing GTE-large `legal_passages` collection. Qrel alignment now passes at
  100% for the then-active BarExamQA, HousingQA, CaseHOLD, and
  LegalBench-SCALR set.
- Scoped cache filenames were added to prevent N=50/sample caches from being
  replayed as full-corpus caches. Full BarExam raw/golden caches and q5
  raw/golden cache smokes across all four datasets hydrate cleanly through the
  harness replay path.
- `legalbench_scalr` × `or-gemma4-26b` N=50 answer ladder at `k=5` completed:
  `llm_only` 76.0%, `rag_simple` 76.0%, `rag_rewrite` 74.0%,
  `rag_hyde` 78.0%, and `snap_hyre` 80.0%, all clean by
  `scripts/analyze_detail_flags.py`.
- SCALR N=50 `golden_passage` and `golden_plus_neighbors` remain rejected as
  oracle controls because they were run before oracle reference hydration was
  fixed. The follow-up q5 smoke confirms hydrated oracle controls now work.
- Generated package artifacts now live under
  `docs/generated/snap_hyre_package/`.

## Next Concrete Gates

1. Run strict API smokes for `groq-llama8b`, `or-gemma4-26b`, and
   `groq-llama70b` before launching broad answer cells.
2. Continue generation-cache construction one provider/dataset at a time,
   using `--resume`; the partial BarExam `gemma4-26b` `rag_hyde` cache can be
   resumed.
3. Expand answer ladders one dataset/model cell at a time after cache audits,
   keeping `RETRIEVAL_K=5` provisional until broader generated-cache/downstream
   evidence justifies changing it.
4. Add only validated rows to `docs/signoff_log.md`, then regenerate result
   tables and plots with `scripts/local/build_result_package.sh`.
