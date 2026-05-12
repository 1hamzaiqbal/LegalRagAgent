# Adaptive HyRE Candidate Verifier N=50 - 2026-05-10

## Question

Can a candidate-first final verifier improve holding-selection tasks by reducing
overreliance on noisy retrieved passages?

This probe targeted CaseHOLD and LegalBench-SCALR after cached replay showed
that fixed HyRE retrieval did not uniformly solve option conversion.

## Method

Mode: `adaptive_snap_hyre_candidate_verifier`

For CaseHOLD and LegalBench-SCALR, the method keeps cached HyRE retrieval but
changes the final prompt:

- compare all displayed holdings directly against the citing context;
- use retrieved passages as support or tie-breakers, not replacement
  candidates;
- if retrieval is noisy, choose the displayed holding whose rule and fact
  pattern best fit the citing context.

The method uses the fixed HyRE replay cache, so both jobs should use one final
LLM call per row.

## Cluster Evidence

Manifest:
`/engrfs/tmp/jacobsn/hiqbal_legalrag/logs/adaptive_hyre_mode_matrix_20260510_1045_candidate_verifier.tsv`

Jobs:

| Job | Dataset | Status |
|---:|---|---|
| 67390 | CaseHOLD | PASS |
| 67391 | LegalBench-SCALR | PASS |

Detail logs:

- CaseHOLD:
  `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_adaptive_snap_hyre_candidate_verifier_or-gemma4-26b_20260510_1058_casehold_adaptive-hyre-or-gemma4-26b-casehold-n50-k5-adaptive_snap_hyre_candidate_verifier_detail.jsonl`
- LegalBench-SCALR:
  `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_adaptive_snap_hyre_candidate_verifier_or-gemma4-26b_20260510_1143_legalbench_scalr_adaptive-hyre-or-gemma4-26b-legalbench_scalr-n50-k5-adaptive_snap_hyre_candidate_verifier_detail.jsonl`

## Results

| Dataset | Method | N | Accuracy | Gold Retrieved | Avg Calls | Audit |
|---|---|---:|---:|---:|---:|---|
| CaseHOLD | `adaptive_snap_hyre_candidate_verifier` | 50 | 37/50 = 74.0% | 14/50 | 1.00 | PASS |
| LegalBench-SCALR | `adaptive_snap_hyre_candidate_verifier` | 50 | 36/50 = 72.0% | 30/50 | 1.00 | PASS |

Comparisons from the postprocess summaries:

| Dataset | Comparison | Delta | b/c | p |
|---|---|---:|---:|---:|
| CaseHOLD | frontier -> candidate verifier | +4.0pp | 3/1 | 0.625 |
| CaseHOLD | rag_simple -> candidate verifier | -2.0pp | 6/7 | 1 |
| SCALR | frontier -> candidate verifier | -6.0pp | 2/5 | 0.4531 |
| SCALR | rag_snap_hyde_2call -> candidate verifier | -8.0pp | 2/6 | 0.2891 |

## Interpretation

The candidate-first prompt is not a general fix.

CaseHOLD has a small positive signal over the cached frontier and stability
variants, but it does not beat the N=50 rag_simple slice and the lift is not
statistically reliable. This suggests candidate-first answer conversion may be
useful only if paired with a better CaseHOLD retrieval or option-reranking step.

SCALR should reject this variant. It underperforms the cached frontier and
produced two pathological long generations, including rows with answer outputs
over 150k characters. SCALR should stay on plain Snap-HyDE/frontier behavior or
a tightly bounded verifier, not this verbose candidate-first prompt.

## Decision

- CaseHOLD: keep as a possible component, but do not scale to N=200 yet.
- SCALR: reject this prompt variant.
- Next useful CaseHOLD probe: option/candidate reranking before final answer,
  not another longer final prompt.

