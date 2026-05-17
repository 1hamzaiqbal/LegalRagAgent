# Local Validation Goal - 2026-05-12

Use this as the short-lived goal for the local Codex instance before starting
the comprehensive full-corpus eval grid.

## Goal Text

Validate the local Snap-HyRE execution stack before comprehensive full-corpus
evals: pull `snap_hyre_comprehensive`, configure `OPENROUTER_API_KEY` and
`GROQ_API_KEY`, verify datasets and local Chroma collections for BarExamQA,
HousingQA, CaseHOLD, and LegalBench-SCALR, run API smoke tests, build
raw/golden retrieval caches, audit qrel alignment, select or justify the
universal top-k from the retrieval matrix, build N=50 HyDE/Snap-HyRE generation
caches, run one N=50 answer-cell ladder end-to-end, and update
`docs/signoff_log.md` plus `docs/snap_hyre_completion_audit_2026-05-12.md`
only with source-gated clean evidence.

## Pass Condition

Move to comprehensive full-corpus evals only after all of these are true:

- Provider smoke passes for the API-only three-model grid: `or-ministral-8b`,
  `or-gemma4-26b`, and `groq-llama70b`.
- `scripts/local/build_retrieval_caches.sh` writes a retrieval matrix and qrel
  alignment reports for the four legal datasets.
- The chosen `RETRIEVAL_K` is recorded with a short justification.
- One N=50 ladder cell completes through `scripts/local/run_answer_cell.sh`.
- `scripts/local/build_result_package.sh` produces a package status artifact
  that correctly reflects the clean rows and remaining missing cells.

If any condition fails, fix or document that blocker before launching broader
full-corpus runs.

## Status - 2026-05-13

Passed for local alignment/cache readiness; comprehensive model grid is now
API-only and no longer waits on a vLLM Gemma 4 E4B launch gate.

- Local WSL checkout, `.env`, datasets, Chroma collections, GTE embeddings, and
  MiniLM reranker are set up.
- API smoke passed for `or-gemma4-26b`.
- The earlier `or-gemma3n-e4b` smoke is not a comprehensive-model gate:
  `google/gemma-3n-e4b-it` is not the historical `google/gemma-4-E4B-it`
  checkpoint.
- The small-model API row is `or-ministral-8b`, not historical Gemma 4 E4B.
- `groq-llama70b` initially failed with a Groq 401 invalid-key preflight; after
  replacing the key on 2026-05-13, a one-question strict smoke passed.
- `legal_passages` was patched from 686,324 train passages to the full
  856,835-passage BarExam corpus by appending validation/test passages with the
  same GTE 1.5 large encoder.
- Qrel alignment now passes at 100% for all four datasets: BarExamQA,
  HousingQA, CaseHOLD, and LegalBench-SCALR.
- `scripts/local/build_retrieval_caches.sh` now writes question-scoped cache
  filenames and produced clean BarExam full caches plus q5 raw/golden cache
  smokes for all four datasets.
- Provisional `RETRIEVAL_K=5` was used for the answer-cell validation; broader
  generated-query/downstream evidence is still needed before changing it.
- `legalbench_scalr` × `or-gemma4-26b` N=50 generation caches and answer ladder
  completed at `k=5`; main rows passed detail-log validation.
- A follow-up `legalbench_scalr` × `or-gemma4-26b` q5 smoke validated the new
  scoped HyDE/Snap-HyRE cache filenames, strict retrieval-cache replay, and
  hydrated oracle controls across the full seven-row ladder.
- `scripts/local/build_result_package.sh` produced
  `docs/generated/snap_hyre_package/package_status.md` and companion CSV/plot
  artifacts.

Do not start broad comprehensive sweeps until `or-ministral-8b`, `or-gemma4-26b`,
and `groq-llama70b` all pass strict API smoke checks. BarExam alignment, Llama
callability, and golden-control hydration are resolved locally.
