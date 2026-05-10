# Legal Query Rewrite Baseline N=50 - 2026-05-10

## Question

Does a normal legal query-rewrite RAG baseline explain the same gains we have
been attributing to Snap-HyRE, or does generated reasoning add something beyond
better search phrasing?

## Setup

- Provider: `or-gemma4-26b`
- Dataset slice: `N=50`, seed `42`
- Retrieval: `k=5`
- Mode: `rag_rewrite`
- Collections: cluster Chroma collections for the four legal datasets
- Submission manifest:
  `/engrfs/tmp/jacobsn/hiqbal_legalrag/logs/adaptive_hyre_mode_matrix_20260510_141643.tsv`
- SLURM jobs: `67424` BarExam, `67425` HousingQA, `67426` CaseHOLD,
  `67427` LegalBench-SCALR

The array completed detail logs successfully. The wrapper postprocess step
failed afterward because the jobs were launched from the older cluster checkout,
which does not contain `scripts/postprocess_adaptive_hyre_sweep.py`; this does
not invalidate the detail logs below.

## Results

| Dataset | Detail log | Accuracy | Gold retrieved | Calls | Health |
|---|---|---:|---:|---:|---|
| BarExam | `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent/logs/eval_rag_rewrite_or-gemma4-26b_20260510_1440_detail.jsonl` | 43/50 = 86.0% | 4/50 | 2.00 | PASS |
| HousingQA | `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent/logs/eval_rag_rewrite_or-gemma4-26b_20260510_1439_detail.jsonl` | 29/50 = 58.0% | 6/50 | 2.00 | parse_fail=1 |
| CaseHOLD | `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent/logs/eval_rag_rewrite_or-gemma4-26b_20260510_1445_detail.jsonl` | 36/50 = 72.0% | 34/50 | 2.00 | PASS |
| LegalBench-SCALR | `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent/logs/eval_rag_rewrite_or-gemma4-26b_20260510_1438_detail.jsonl` | 38/50 = 76.0% | 37/50 | 2.00 | PASS |

Artifact audit was clean for all four logs: zero top-level HyDE/report/knowledge
answer-artifact flags and zero nested gap artifact flags. Every row has
`call_trace` and `trace_events`.

## Interpretation

This is a useful control for the diagnostic-adaptation story.

- BarExam: query rewriting reaches 86.0% on this N=50 slice, so the BarExam
  intervention should not be described as HyRE-specific without a paired
  rewrite-vs-HyRE comparison on the same slice.
- HousingQA: query rewriting is not the right bottleneck intervention. The
  verifier route remains the strong result because the failure mode is statutory
  yes/no entailment, not generic search phrasing.
- CaseHOLD: query rewriting retrieves the gold option in 34/50 rows but only
  answers 36/50 correctly. This reinforces the answer-option conversion
  bottleneck: exposing the right holding is not sufficient.
- LegalBench-SCALR: query rewriting is competitive with fixed Snap-HyDE on
  this calibration slice, but cached disagreement arbitration remains the
  stronger current N=200 route.

## Controller Implication

The policy menu should include `query_rewrite_rag` as a first-class route. It is
not a replacement for Snap-HyRE; it is a diagnostic arm that helps decide whether
the active bottleneck is ordinary query formulation, generated reasoning,
metadata filtering, option grounding, or answer verification.

The updated generated artifacts are:

- `docs/legal_rag_diagnostic_table_with_rewrite_2026-05-10.md`
- `docs/legal_rag_diagnostic_table_with_rewrite_2026-05-10.json`
- `docs/diagnostic_controller_route_plan_with_rewrite_2026-05-10.md`
- `docs/diagnostic_controller_route_plan_with_rewrite_2026-05-10.json`
