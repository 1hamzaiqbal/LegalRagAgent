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

- Job: `67534` completed with exit code `0:0`.
- Script: `scripts/replay_casehold_selector.py`
- Source log:
  `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_adaptive_snap_hyre_diverse_or-gemma4-26b_20260510_1721_casehold_heldout-controller-retry-or-gemma4-26b-casehold-q250-start200-end250-k5-adaptive_snap_hyre_diverse_detail.jsonl`
- Output log:
  `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_adaptive_snap_hyre_option_replay_snap_guard_or-gemma4-26b_20260510_casehold_heldout_n50_detail.jsonl`
- Provider: `or-gemma4-26b`
- Variant: `snap_guard`
- Compute path: `general-cpu`; no GPU, no fresh retrieval.

## Integration Gate

Completed:

1. `sacct` reports job `67534` completed with exit code `0:0`.
2. Stdout reports all 50 rows completed with no row-level errors.
3. `scripts/analyze_detail_flags.py` loaded 50 rows and reported no artifact
   flags.
4. Row comparison against the held-out source route covers all 50 overlapping
   labels.

## Results

| Method | Held-out CaseHOLD accuracy | Notes |
|---|---:|---|
| `rag_simple` | 34/50 = 68.0% | matched held-out baseline |
| `adaptive_snap_hyre_option_replay_snap_guard` | 33/50 = 66.0% | this replay selector |
| `rag_rewrite` | 38/50 = 76.0% | held-out query rewrite control |
| `adaptive_snap_hyre_diverse` | 39/50 = 78.0% | current selected route |

Replay-vs-source McNemar counts from row overlap:

| Comparison | b | c |
|---|---:|---:|
| replay correct / diverse wrong vs diverse correct / replay wrong | 4 | 10 |

## Interpretation

The replay selector is a clean negative result. It confirms that simply
re-asking a stricter final selector over the same diverse-route evidence does
not solve CaseHOLD answer-option conversion. It is worse than the held-out
baseline, worse than query rewrite, and worse than the current selected route.

For the diagnostic controller, keep CaseHOLD routed to
`adaptive_snap_hyre_diverse` plus the documented `reject_or_escalate` policy.
The remaining promising CaseHOLD work is not another generic replay prompt; it
is a more explicit option-conversion mechanism or a repaired option-table path.
