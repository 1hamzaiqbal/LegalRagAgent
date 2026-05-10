# CaseHOLD Option-Table Held-Out Probe - 2026-05-10

## Purpose

Test whether the existing compact option-conversion route,
`adaptive_snap_hyre_option_table`, improves the remaining CaseHOLD bottleneck on
the same held-out rows 200-249 used by the controller validation.

This is a targeted follow-up to the completion audit caveat: CaseHOLD retrieval
and candidate exposure can improve without reliably improving final answer
accuracy, so the unresolved question is whether an option-level selector can
convert retrieved evidence into the right displayed holding.

## Submission

- Active job: `67521`
- Superseded job: `67519` failed in preflight before method execution because
  the cluster launch did not see `adaptive_snap_hyre_option_table` in
  `EVAL_MODES`; the mode was confirmed present in the checkout and importable
  inside `.venv`, then resubmitted with explicit `sbatch --export`.
- Superseded job: `67520` failed in preflight for the same reason because the
  Slurm script defaulted `REPO` to the non-adaptive cluster checkout. Job
  `67521` explicitly exports `REPO=/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre`
  and `DATA_REPO=/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent`.
- Dataset: `casehold`
- Mode: `adaptive_snap_hyre_option_table`
- Provider path: cluster vLLM, Gemma 4 26B
- Slice: `--questions 250 --sample-start 200 --sample-end 250`
- Effective evaluated rows: 50
- Retrieval: `k=5`
- Tag: `casehold-option-table-heldout-or-gemma4-26b-casehold-q250-start200-end250-k5`

## Integration Gate

Before promoting results:

1. Confirm `sacct` completion and exit code for job `67521`.
2. Inspect `/engrfs/tmp/jacobsn/hiqbal_legalrag/logs/67521.out` and the vLLM
   log for Tracebacks, parsing failures, empty retrieval, or timeout.
3. Run `scripts/analyze_detail_flags.py` on the landed detail JSONL.
4. Run `scripts/audit_adaptive_hyre_logs.py` on the landed detail JSONL.
5. Compare against the held-out CaseHOLD rows already validated:
   - `rag_simple`: 34/50 = 68.0%
   - `rag_rewrite`: 38/50 = 76.0%
   - `adaptive_snap_hyre_diverse`: 39/50 = 78.0%
