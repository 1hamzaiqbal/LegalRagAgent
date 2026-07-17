# July 2026 evidence package

This directory is the durable, tracked bridge between the July 2 local/EIT
experiments and later research decisions. It preserves compact result objects,
the relevant Slurm stdout, and manifests for larger local detail logs without
duplicating every raw artifact in Git.

## Contents

- `judge_results/`: stable JSON summaries for the BarExam, Housing, SciDocs,
  FiQA, and mixed-legal judge runs.
- `eit_job_logs/`: exact stdout for the ten EIT jobs inspected during the
  2026-07-17 reconciliation. A scheduler-level success is not assumed to be a
  semantic success; job 93598 is the canonical counterexample.
- `manifests/local_july_detail_logs.tsv`: SHA-256, size, and source path for 71
  larger July detail logs retained outside this compact package.
- `manifests/tracked_evidence.sha256`: integrity manifest for the original
  compact evidence copied into this directory. Regenerate it after adding new
  tracked summaries rather than treating it as an automatic live manifest.

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
`/Users/hamzaiqbal/grad/LegalRagAgent_archives/`. Nothing in this package is a
substitute for those recovery copies.
