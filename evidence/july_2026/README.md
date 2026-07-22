# July 2026 evidence package

This directory is the durable, tracked bridge between the July 2 local/EIT
experiments and later research decisions. It preserves compact result objects,
the relevant Slurm stdout, and manifests for larger local detail logs without
duplicating every raw artifact in Git.

## Contents

- `judge_results/`: stable JSON summaries for the BarExam, Housing, SciDocs,
  FiQA, and mixed-legal judge runs.
- `eit_job_logs/`: exact stdout for the ten historical EIT jobs inspected
  during the 2026-07-17 reconciliation plus the gated-OPD follow-up job 106078.
  A scheduler-level success is not assumed to be a semantic success; job 93598
  is the canonical counterexample.
- `manifests/local_july_detail_logs.tsv`: SHA-256, size, and source path for 71
  larger July detail logs retained outside this compact package.
- `manifests/tracked_evidence.sha256`: integrity manifest for the original
  compact evidence copied into this directory. Regenerate it after adding new
  tracked summaries rather than treating it as an automatic live manifest.
- `opd_teacher_evaluator_qualification_bd1ca8b_v1.json`: read-only
  reconstruction of teacher learning signal, truncation, paired teacher
  movement, and predecessor student baselines from sealed EIT traces.

The result-level interpretation and cite gates are in
[`docs/july_2026_completion_audit_2026-07-17.md`](../../docs/july_2026_completion_audit_2026-07-17.md)
and [`docs/signoff_log.md`](../../docs/signoff_log.md). The research-level
interpretation is in
[`wiki/snapshots/research-state-2026-07-17.md`](../../wiki/snapshots/research-state-2026-07-17.md).

## Integrity and recovery

Before cleanup, a complete Git bundle, tracked diff, untracked-file archive,
and checksums were written to the sibling recovery directory
`/Users/hamzaiqbal/grad/LegalRagAgent_recovery_20260717`. Old SCOPE submission
artifacts were also committed on `codex/scope_old` and zipped under
`/Users/hamzaiqbal/grad/LegalRagAgent_archive/`. Nothing in this package is a
substitute for those recovery copies.

## OPD teacher/evaluator qualification — audited 2026-07-22

CPU Slurm job `126821` ran
`scripts/opd_math/qualification_audit.py` without invoking Math-Verify or
changing a stored score. The tracked output is
`opd_teacher_evaluator_qualification_bd1ca8b_v1.json` (SHA-256
`2255b2f7a49404a9c47c26cf42f53a4a7b6d4448b55aa0d85b423b0aa400b9f1`).

The audit establishes a setup diagnosis, not a performance win: O teacher run
108609 sampled 100/4,322 selected prompt groups, had 18 mixed-reward groups,
16 nonzero-gradient steps, and 178/400 completions at its 1,024-token cap.
Nearly every student parse failure and most teacher parse failures occurred at
the completion cap. The old task-RL student results remain absolute pilot
scores because no raw-student evaluation exists on the identical held-out
records and decoding contract. No predecessor artifact demonstrates OPD
student improvement.
