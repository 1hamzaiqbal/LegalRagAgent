# Documentation index — LegalRagAgent

Current map as of 2026-07-17.

## Read first

1. [`../wiki/snapshots/research-state-2026-07-17.md`](../wiki/snapshots/research-state-2026-07-17.md)
   — durable state, strongest findings, caveats, and decision gates.
2. [`../wiki/tracks/three-dial.md`](../wiki/tracks/three-dial.md) — primary
   science track.
3. [`../wiki/tracks/opd-distillation.md`](../wiki/tracks/opd-distillation.md)
   — gated engineering track.
4. [`july_2026_completion_audit_2026-07-17.md`](july_2026_completion_audit_2026-07-17.md)
   — local/EIT job and evidence reconciliation.
5. [`signoff_log.md`](signoff_log.md) — cite-or-not gate.
6. [`../wiki/literature/index.md`](../wiki/literature/index.md) — Obsidian map
   into the persistent paper/repository vault.
7. [`worktree_map_2026-07-17.md`](worktree_map_2026-07-17.md) — exact local/EIT
   branches, storage roles, and recovery paths.
8. [`archive_manifest_2026-07-17.md`](archive_manifest_2026-07-17.md) — what
   left the active branches, what stayed, checksums, and restore paths.

## Evidence path

Use result evidence in this order:

1. `signoff_log.md`
2. `july_2026_completion_audit_2026-07-17.md`
3. `../evidence/july_2026/`
4. `compiled_results.md` and `../logs/experiments.jsonl` for older ledger rows
5. the exact JSONL/stdout named by the preceding artifact

Do not cite `current_status.md`, old meeting notes, or historical narrative
files as final evidence.

## Historical material

The SCOPE/Snap-HyRE submission, reviews, class report, old paper tree, and old
root research diaries are preserved on `codex/scope_old`, named Git archive
branches, and verified ZIPs in persistent EIT storage. The Mac archive
directory contains only restore manifests. Source-gated historical audits
remain in `docs/` because current three-dial reconstruction still links to
their logs. A later evidence-compaction pass may replace those only after a new
master table has preserved every citable row.

## Maintenance rule

Append a new dated snapshot or completion audit when the state changes. Do not
silently rewrite the 2026-07-17 snapshot. Every new numeric claim needs a source
log and a signoff/audit pointer; every novelty-changing paper needs a wiki page
and persistent-vault record.
