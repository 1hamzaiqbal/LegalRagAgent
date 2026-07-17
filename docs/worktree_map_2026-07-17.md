# LegalRagAgent worktree map — 2026-07-17

## Active and historical surfaces

| Track | Local Mac | EIT persistent project space | Branch | State |
|---|---|---|---|---|
| Three-dial utility | `/Users/hamzaiqbal/grad/LegalRagAgent` | `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-three-dial` | `codex/three_dial` | active, clean, synced |
| OPD/distillation | `/Users/hamzaiqbal/grad/LegalRagAgent-opd-distillation` | `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-opd-distillation` | `codex/opd_distillation` | gated active track, clean, synced |
| SCOPE history | `/Users/hamzaiqbal/grad/LegalRagAgent-scope-old` | `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-scope-old` | `codex/scope_old` | historical, clean, synced |

All three branches are also pushed to the user-owned GitHub remote `origin` on
the Mac and fetched through the `hamza` remote on EIT.

## Historical EIT worktrees — preserve until a later deletion pass

- `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent` — dirty historical
  `codex/evidence-ledger-router` worktree. It is **not** the active worktree.
- `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre` — dirty,
  behind historical `codex/final-report-snap-hyde`. It is **not** active.

Their HEADs, statuses, binary worktree/index diffs, untracked tarballs, and a
full all-refs Git bundle are preserved at:

`/engrfs/project/jacobsn/hiqbal/archives/legalrag/2026-07-17/`

The archive is 252 MB, its Git bundle verifies, its tarballs pass gzip checks,
and `SHA256SUMS` covers the package. The original worktrees have not been
deleted or reset.

## Experiment and literature storage

- Persistent paper/repository vault:
  `/engrfs/project/jacobsn/hiqbal/literature/legalrag/`
- Historical experiment scratch:
  `/engrfs/tmp/jacobsn/hiqbal_legalrag/`
- Gated-OPD job 106078 artifacts and checksums:
  `/engrfs/tmp/jacobsn/hiqbal_legalrag/opd_gated_smoke_106078_artifacts/`
- Exact gated-OPD stdout:
  `/engrfs/tmp/jacobsn/hiqbal_legalrag/opd_gated_smoke_106078.out`

## Operating rule

New development should begin from the track-specific worktree above. Do not
run experiments from either historical EIT worktree. A later space-reclamation
pass may remove duplicate caches/checkpoints only after comparing them against
the manifests and confirming that no active Slurm script references them.
