# CaseHOLD Held-Out Replay Selector - 2026-05-10

## Purpose

Test a cheap CaseHOLD answer-conversion fallback without new retrieval,
embedding, or Chroma access. The replay selector consumes the already validated
held-out `adaptive_snap_hyre_diverse` detail log and asks one final selector
call per row to reconsider the same evidence.

This is a cleaner follow-up than the blocked option-table route because it
targets final answer conversion while avoiding the candidate-conditioned
embedding path that failed in `docs/casehold_option_table_heldout_2026-05-10.md`.

## Submission

- Active job: `67534`
- Script: `scripts/replay_casehold_selector.py`
- Source log:
  `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_adaptive_snap_hyre_diverse_or-gemma4-26b_20260510_1721_casehold_heldout-controller-retry-or-gemma4-26b-casehold-q250-start200-end250-k5-adaptive_snap_hyre_diverse_detail.jsonl`
- Output log:
  `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_adaptive_snap_hyre_option_replay_snap_guard_or-gemma4-26b_20260510_casehold_heldout_n50_detail.jsonl`
- Provider: `or-gemma4-26b`
- Variant: `snap_guard`
- Compute path: `general-cpu`; no GPU, no fresh retrieval.

## Integration Gate

Before promoting results:

1. Confirm `sacct` completion and exit code for job `67534`.
2. Inspect `/engrfs/tmp/jacobsn/hiqbal_legalrag/logs/67534.out` for Tracebacks,
   API/rate-limit errors, parse failures, null predictions, or timeout.
3. Run `scripts/analyze_detail_flags.py` on the output detail log.
4. Compare against the held-out CaseHOLD rows already validated:
   - `rag_simple`: 34/50 = 68.0%
   - `rag_rewrite`: 38/50 = 76.0%
   - `adaptive_snap_hyre_diverse`: 39/50 = 78.0%

