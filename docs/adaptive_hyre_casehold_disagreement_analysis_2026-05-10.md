# Adaptive HyRE CaseHOLD Disagreement Analysis - 2026-05-10

## Question

Is there a cheap adaptive rule across existing CaseHOLD selector variants that
beats the current 74.0% frontier without launching more retrieval or prompt
sweeps?

This analysis joins the N=50 CaseHOLD detail logs by `label` for:

- `adaptive_snap_hyre_frontier`
- `adaptive_snap_hyre_candidate_verifier`
- `adaptive_snap_hyre_option_reranker`
- `adaptive_snap_hyre_option_score`
- `adaptive_snap_hyre_option_replay_minimal_rule`

Script:
`scripts/analyze_casehold_disagreements.py`

Cluster report:
`/engrfs/tmp/jacobsn/hiqbal_legalrag/logs/casehold_selector_disagreement_20260510.md`

## Method Accuracy

| Method | Correct | Accuracy |
|---|---:|---:|
| `frontier` | 35/50 | 70.0% |
| `candidate_verifier` | 37/50 | 74.0% |
| `option_reranker` | 37/50 | 74.0% |
| `option_score` | 11/50 | 22.0% |
| `replay_minimal_rule` | 31/50 | 62.0% |

## Headroom

- any-method oracle: 42/50 = 84.0%
- all methods correct: 7/50 = 14.0%
- no method correct: 8/50 = 16.0%

The oracle number matters: across the variants we already ran, 10 additional
rows are recoverable beyond the 74.0% candidate/reranker ceiling. But that
signal is not exposed by a trivial voting rule.

## Simple Ensemble Probes

| Rule | Correct | Accuracy |
|---|---:|---:|
| `majority_candidate_reranker_replay` | 37/50 | 74.0% |
| `prefer_candidate` | 37/50 | 74.0% |
| `prefer_reranker` | 37/50 | 74.0% |
| `score_agree_else_candidate` | 37/50 | 74.0% |
| `score_agree_else_reranker` | 37/50 | 74.0% |

## Correctness Patterns

| Correct methods | Rows |
|---|---:|
| `frontier,candidate,reranker,replay` | 22 |
| `none` | 8 |
| `frontier,candidate,reranker,score,replay` | 7 |
| `frontier,candidate,reranker,score` | 2 |
| `frontier,candidate,reranker` | 2 |
| `reranker` | 2 |
| `candidate` | 1 |
| `candidate,reranker,replay` | 1 |
| `frontier` | 1 |
| `frontier,candidate` | 1 |
| `candidate,reranker` | 1 |
| `score,replay` | 1 |
| `score` | 1 |

## Interpretation

The row-level picture is sharper than the aggregate result:

- CaseHOLD has real selector headroom: an oracle over existing methods reaches
  84.0%.
- The headroom is not captured by simple majority vote or by trusting score-only
  agreement.
- The 8 rows missed by every method likely need better evidence, better
  candidate normalization, or a different representation of the cited holding.
- The rows where only `reranker`, only `candidate`, only `frontier`, or only
  `score` succeeds are the most useful next inspection set. They can tell us
  which observable features predict when each selector should be trusted.

## Decision

Do not launch another broad CaseHOLD prompt sweep yet. The next adaptive step
should be feature analysis over disagreements:

- answer entropy among candidate/reranker/replay outputs;
- whether the snap answer agrees with candidate/reranker;
- candidate text length and specificity gaps;
- whether retrieved candidate evidence contains the gold option text;
- whether score-only is correct only on high-margin cases.

If these features separate the singleton-win rows, then a calibrated router or
selector is justified. If they do not, CaseHOLD should stay framed as a
remaining answer-conversion bottleneck rather than forced into the parity story.
