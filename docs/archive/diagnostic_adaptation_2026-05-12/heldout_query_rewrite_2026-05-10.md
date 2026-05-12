# Held-Out Query Rewrite Control - 2026-05-10

## Purpose

Run `rag_rewrite` on the same held-out rows 200-249 used by the compact controller evaluation. This closes the query-rewrite same-slice coverage gap for the four legal datasets.

## Submission

- Provider: `or-gemma4-26b`
- Mode: `rag_rewrite`
- Slice: `--questions 250 --sample-start 200 --sample-end 250`
- Effective evaluated rows: 50
- Retrieval: `k=5`
- Manifest: `/engrfs/tmp/jacobsn/hiqbal_legalrag/logs/heldout_query_rewrite_20260510_174149.tsv`

## Results

- Query rewrite macro accuracy: 75.5%
- Baseline macro accuracy on same slice: 71.5%
- Exact selected-route macro accuracy on same slice: 77.5%
- Query rewrite macro calls: 2.00

| Dataset | Query rewrite | Baseline | Selected route | Delta vs baseline pp | Delta vs selected pp | Health |
|---|---:|---:|---:|---:|---:|---|
| barexam | 45/50 = 90.0% | 38/50 = 76.0% | `adaptive_snap_hyre_v2` 38/50 = 76.0% | +14.0 | +14.0 | PASS |
| casehold | 38/50 = 76.0% | 34/50 = 68.0% | `adaptive_snap_hyre_diverse` 39/50 = 78.0% | +8.0 | -2.0 | PASS |
| housing | 29/50 = 58.0% | 31/50 = 62.0% | `adaptive_snap_hyre_housing_verifier` 38/50 = 76.0% | -4.0 | -18.0 | PASS |
| legalbench_scalr | 39/50 = 78.0% | 40/50 = 80.0% | `adaptive_snap_hyre_disagreement_majority_prior` 40/50 = 80.0% | -2.0 | -2.0 | PASS |

## Interpretation

- Query rewrite is highly dataset-dependent, which supports the controller framing rather than a universal rewrite policy.
- BarExam is the strongest rewrite case on this held-out slice: 90.0%, beating both baseline and `adaptive_snap_hyre_v2` by +14 points.
- HousingQA remains a verifier/entailment task: query rewrite reaches only 58.0%, below state-filter baseline and far below the verifier route.
- CaseHOLD query rewrite reaches 76.0%, between baseline and `adaptive_snap_hyre_diverse`; it is useful but does not remove the answer-conversion caveat.
- LegalBench-SCALR query rewrite reaches 78.0%, below baseline and below the frontier component; query rewriting should not be the SCALR route.

## Validation

- All four jobs completed with `sacct` exit code `0:0`.
- `scripts/analyze_detail_flags.py` loaded 50 rows for each detail log.
- Artifact checks reported zero top-level HyDE/report/knowledge artifacts and zero nested gap artifacts for all four logs.
- Every row had `call_trace` and `trace_events`.

## Source Logs

- housing / `rag_rewrite`: `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_rag_rewrite_or-gemma4-26b_20260510_1759_housing_heldout-query-rewrite-or-gemma4-26b-housing-q250-start200-end250-k5-rag_rewrite_detail.jsonl`
- casehold / `rag_rewrite`: `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_rag_rewrite_or-gemma4-26b_20260510_1800_casehold_heldout-query-rewrite-or-gemma4-26b-casehold-q250-start200-end250-k5-rag_rewrite_detail.jsonl`
- legalbench_scalr / `rag_rewrite`: `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_rag_rewrite_or-gemma4-26b_20260510_1801_legalbench_scalr_heldout-query-rewrite-or-gemma4-26b-legalbench_scalr-q250-start200-end250-k5-rag_rewrite_detail.jsonl`
- barexam / `rag_rewrite`: `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_rag_rewrite_or-gemma4-26b_20260510_1802_barexam_heldout-query-rewrite-or-gemma4-26b-barexam-q250-start200-end250-k5-rag_rewrite_detail.jsonl`
