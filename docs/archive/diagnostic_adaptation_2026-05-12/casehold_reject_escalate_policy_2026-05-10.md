# CaseHOLD Reject-Or-Escalate Policy - 2026-05-10

## Question

Can the diagnostic controller do something defensible for CaseHOLD even though
the current selectors do not beat the 73-74% accuracy band?

## Evidence

Source report:
`/engrfs/tmp/jacobsn/hiqbal_legalrag/logs/casehold_selector_disagreement_20260510.md`

Escalation export:
`/engrfs/tmp/jacobsn/hiqbal_legalrag/logs/casehold_escalation_rows_20260510.jsonl`

Related local summaries:

- `docs/adaptive_hyre_casehold_disagreement_analysis_2026-05-10.md`
- `docs/adaptive_hyre_casehold_rule_frame_replay_2026-05-10.md`
- `docs/adaptive_hyre_casehold_selector_replay_n50_2026-05-10.md`

The N=50 CaseHOLD selector analysis compared five existing routes:

| Route | Accuracy |
|---|---:|
| `frontier` | 35/50 = 70.0% |
| `candidate_verifier` | 37/50 = 74.0% |
| `option_reranker` | 37/50 = 74.0% |
| `option_score` | 11/50 = 22.0% |
| `replay_minimal_rule` | 31/50 = 62.0% |

No simple ensemble beat 74.0%. The any-method oracle reached 42/50 = 84.0%,
so there is selector headroom, but the current observable rules cannot reliably
choose the right selector on every row.

## Confidence Signals

The useful CaseHOLD diagnostic is selective confidence, not unconditional
accuracy.

| Signal | Rows | Accuracy |
|---|---:|---:|
| candidate / reranker / replay all emit one answer | 35 | 85.7% |
| candidate and reranker agree | 43 | 81.4% |
| candidate and reranker disagree | 7 | 28.6% |
| reranker agrees with snap answer | 34 | 85.3% |
| reranker disagrees with snap answer | 16 | 50.0% |

The strict selective policy
`accept_candidate_reranker_agree_and_snap_agree` answered 33/50 rows at 87.9%
accepted accuracy and escalated 17/50 rows. If an oracle solved every escalated
row, the upper bound would be 41/50 = 82.0%; the actual rule-frame replay over
those 17 escalation rows solved only 8/17, yielding 37/50 = 74.0% overall.

## Controller Policy

For CaseHOLD, route selection should be:

1. Run the current strongest answer-conversion route:
   `adaptive_snap_hyre_diverse` for N=200 reporting, or the candidate/reranker
   selector family when those row-level traces are available.
2. If candidate, reranker, and snap-style outputs agree, accept the answer.
3. If candidate and reranker disagree, or if the selector family has high answer
   entropy, mark the row `reject_or_escalate`.
4. Do not spend another generic final-prompt call over the same evidence. The
   rule-frame replay was clean but did not improve overall accuracy.

This policy is not a solved CaseHOLD method. It is an auditable abstention
policy: it identifies rows where the current system is likely unreliable and
keeps the framework honest about the distinction between retrieval exposure and
answer-option conversion.

## Interpretation For The Paper

CaseHOLD is the strongest counterexample to "just add Snap-HyRE." Retrieval and
gold exposure can improve while final answer accuracy stays flat. The
bottleneck-aware controller should therefore treat CaseHOLD as an
answer-conversion / abstention regime:

- use HyRE-style retrieval to expose plausible holdings;
- use agreement features to accept high-confidence rows;
- escalate disagreement rows to a future calibrated option converter rather
  than overclaiming an accuracy lift from another prompt.

This supports the broader thesis: generated reasoning is useful only when
routed to the bottleneck it can actually address. On CaseHOLD, the current
diagnostic value is knowing when generated retrieval is insufficient.
