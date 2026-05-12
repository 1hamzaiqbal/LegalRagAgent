# CaseHOLD Rule-Frame Replay Probe

Date: 2026-05-10

Purpose: test whether CaseHOLD escalation rows can be recovered by asking the
LLM to normalize each answer option into a compact legal rule frame before
choosing. This is a no-retrieval replay probe over the previously exported
selective-policy escalation rows.

## Setup

- Script: `scripts/replay_casehold_selector.py`
- Variant: `rule_frame`
- Provider: `or-gemma4-26b`
- Source rows:
  `/engrfs/tmp/jacobsn/hiqbal_legalrag/logs/casehold_escalation_rows_20260510.jsonl`
- Full detail log:
  `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_adaptive_snap_hyre_option_replay_rule_frame_or-gemma4-26b_20260510_casehold_escalation_n17_detail.jsonl`

## Result

The full escalation replay completed without API, parse, or formatting errors:

| Split | Correct | Accuracy |
|---|---:|---:|
| 5-row smoke | 2 / 5 | 40.0% |
| 17-row escalation set | 8 / 17 | 47.1% |

The strict high-confidence policy from the disagreement analysis answered
33/50 rows at 29/33 correct and escalated 17 rows. Adding this rule-frame
selector for the escalated rows gives:

`29 + 8 = 37 / 50 = 74.0%`

That matches the strongest CaseHOLD N=50 candidate/reranker frontier rather
than improving it.

## Interpretation

This is a useful rejection result. The rule-frame prompt is better than raw
score selection on the difficult subset, but it does not recover enough
escalated rows to justify another layer in the current adaptive method.

For CaseHOLD, the next useful move is not another generic final selector over
the same evidence. The evidence still points to answer-option conversion as the
bottleneck: gold retrieval can improve, but the model must learn when a choice
is over-specific, under-specific, or keyed to the wrong procedural posture. A
stronger next probe should make those option-level contrasts explicit before the
final answer, or train/evaluate a lightweight calibrated option converter.
