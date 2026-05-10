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

Escalation-row export:
`/engrfs/tmp/jacobsn/hiqbal_legalrag/logs/casehold_escalation_rows_20260510.jsonl`

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

## Selective Adaptive Policies

These policies answer only on high-confidence rows and mark the rest for
escalation. `Total if escalated solved` is an upper bound, not achieved
accuracy.

| Policy | Answered | Answered Accuracy | Escalated | Total If Escalated Solved |
|---|---:|---:|---:|---:|
| `accept_candidate_reranker_replay_unanimous` | 35/50 | 85.7% | 15/50 | 41/50 = 82.0% |
| `accept_candidate_reranker_agree` | 43/50 | 81.4% | 7/50 | 39/50 = 78.0% |
| `accept_reranker_snap_agree` | 34/50 | 85.3% | 16/50 | 41/50 = 82.0% |
| `accept_candidate_snap_agree` | 35/50 | 82.9% | 15/50 | 40/50 = 80.0% |
| `accept_candidate_reranker_agree_and_snap_agree` | 33/50 | 87.9% | 17/50 | 41/50 = 82.0% |

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
- Agreement among LLM selectors is a strong confidence signal: when candidate,
  reranker, and replay all emit one unique answer, candidate/reranker accuracy
  is 30/35 = 85.7%.
- Candidate/reranker disagreement is a strong danger signal: both methods are
  only 2/7 = 28.6% correct on those rows.
- Snap agreement is also useful: reranker accuracy is 29/34 = 85.3% when it
  agrees with the snap answer, but only 8/16 = 50.0% when it disagrees.
- Selective policies can identify high-precision accepted subsets, topping out
  at 33/50 answered with 87.9% accepted accuracy, but the upper bound with
  oracle escalation is still 82.0%.
- The 8 rows missed by every method likely need better evidence, better
  candidate normalization, or a different representation of the cited holding.
- The rows where only `reranker`, only `candidate`, only `frontier`, or only
  `score` succeeds are the most useful next inspection set. They can tell us
  which observable features predict when each selector should be trusted.

## Decision

Do not launch another broad CaseHOLD prompt sweep yet. The next adaptive step
should be feature analysis over disagreements:

- use answer entropy among candidate/reranker/replay outputs as a confidence
  gate;
- use snap agreement with candidate/reranker as a confidence gate;
- candidate text length and specificity gaps;
- whether retrieved candidate evidence contains the gold option text;
- whether score-only is correct only on high-margin cases.

These features separate high-confidence and low-confidence regions, but not into
a finished deployment rule that beats 74.0% without a stronger escalation path.
A calibrated router is justified only if the escalated rows receive a different
intervention than the existing candidate/reranker/replay prompts.

## Escalation Export Notes

Using `accept_candidate_reranker_agree_and_snap_agree` as the high-confidence
acceptance rule exports 17 escalation rows. The first-pass inspection separates
two cases:

- rejected-but-solved rows, where the confidence gate is too strict. Example:
  `ch_ch_test_120` has frontier/candidate/reranker/score all correct, but replay
  and snap choose `C`, so the strict gate rejects a row that the base selector
  handles.
- genuinely hard rows, where the existing methods converge on a distractor or
  split without a reliable signal. Example: `ch_ch_test_1465` has no method
  correct, and retrieved evidence supports candidate `C` while the gold is `E`.

This suggests the next escalation should not be another final prompt over the
same retrieved snippets. The most plausible intervention is candidate
normalization: rewrite each candidate into a compact rule frame, compare the
frames to the citing context, and only then use retrieved evidence as a
secondary signal. That targets the observed failure where the model is pulled
toward a retrieved distractor or toward a more verbose-but-wrong candidate.
