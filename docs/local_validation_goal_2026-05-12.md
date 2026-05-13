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

- API smoke passes for `or-gemma3n-e4b`, `or-gemma4-26b`, and
  `groq-llama70b`.
- `scripts/local/build_retrieval_caches.sh` writes a retrieval matrix and qrel
  alignment reports for the four legal datasets.
- The chosen `RETRIEVAL_K` is recorded with a short justification.
- One N=50 ladder cell completes through `scripts/local/run_answer_cell.sh`.
- `scripts/local/build_result_package.sh` produces a package status artifact
  that correctly reflects the clean rows and remaining missing cells.

If any condition fails, fix or document that blocker before launching broader
full-corpus runs.
