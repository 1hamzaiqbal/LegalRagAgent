# Adaptive HyRE CaseHOLD Option Score N=50 - 2026-05-10

## Question

Can CaseHOLD answer selection be solved without a final LLM verifier by scoring
each displayed holding against option-conditioned retrieval evidence?

This probe follows the option-reranker run, which improved gold retrieval but
did not improve accuracy beyond the candidate verifier. The goal here was to
test whether a cheap deterministic selector could replace the final answer
conversion prompt.

## Method

Mode: `adaptive_snap_hyre_option_score`

For CaseHOLD only:

- replay the cached snap/HyRE object;
- retrieve evidence once per displayed candidate holding using the question,
  candidate text, and cached HyRE text;
- select the candidate whose candidate-conditioned retrieval result has the
  strongest cross-encoder score.

With cached HyRE replay this path uses no LLM calls. It is a scoring-only
selector.

## Cluster Evidence

Job: `67403`

Detail log:
`/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_adaptive_snap_hyre_option_score_or-gemma4-26b_20260510_1211_casehold_casehold-option-score-cached-or-gemma4-26b-casehold-n50-k5-adaptive_snap_hyre_option_score_detail.jsonl`

Summary:
`/engrfs/tmp/jacobsn/hiqbal_legalrag/logs/adaptive_hyre_casehold_option_score_cached_n50.md`

Health:

- rows: 50
- average LLM calls: 0.00
- empty retrieval: 0
- missing predictions: 0
- parse failures: 0
- audit: PASS

## Result

| Method | N | Accuracy | Gold Retrieved | Calls | Health |
|---|---:|---:|---:|---:|---|
| `adaptive_snap_hyre_candidate_verifier` | 50 | 74.0% | 14/50 | 1.00 | PASS |
| `adaptive_snap_hyre_option_reranker` | 50 | 74.0% | 20/50 | 1.00 | PASS |
| `adaptive_snap_hyre_option_score` | 50 | 22.0% | 18/50 | 0.00 | PASS |
| `rag_simple` | 50 slice | 76.0% | source-paired slice | 1.00 | control |

Paired comparisons from the postprocess summary:

| Comparison | Delta | b/c | p |
|---|---:|---:|---:|
| `rag_simple` -> option score | -54.0pp | 3/30 | 1.401e-06 |
| `rag_snap_hyde_2call` -> option score | -44.0pp | 4/26 | 5.948e-05 |
| frontier -> option score | -48.0pp | 2/26 | 3.032e-06 |

## Interpretation

The non-generative selector is decisively rejected. It retrieves gold evidence
on 18/50 examples, but raw candidate-conditioned cross-encoder scores are badly
miscalibrated for choosing the correct CaseHOLD option.

This is useful because it separates two mechanisms:

- option-conditioned retrieval can improve evidence exposure;
- answer-option conversion still needs an LLM verifier, calibration layer, or
  learned selector.

The failed score-only path should not be scaled to N=200. Future CaseHOLD work
should keep the compact candidate evidence bundles but use a bounded verifier or
calibrated selector rather than max cross-encoder score.

## Decision

- Reject `adaptive_snap_hyre_option_score` as a deployment method.
- Keep the result as evidence that CaseHOLD is not only a retrieval problem.
- Next CaseHOLD probe should be either a bounded LLM selector over the compact
  score table or an offline-calibrated selector, not another uncalibrated
  cross-encoder max rule.
