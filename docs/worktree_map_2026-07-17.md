# LegalRagAgent worktree map — 2026-07-17

## Active and historical surfaces

| Track | Local Mac | EIT persistent project space | Branch | State |
|---|---|---|---|---|
| Three-dial utility | `/Users/hamzaiqbal/grad/LegalRagAgent` | `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-three-dial` | `codex/three_dial` | active, clean, synced |
| OPD/distillation | `/Users/hamzaiqbal/grad/LegalRagAgent-opd-distillation` | `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-opd-distillation` | `codex/opd_distillation` | gated active track, clean, synced |
| SCOPE history | branch only; no checked-out Mac worktree | `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-scope-old` | `codex/scope_old` | historical, clean, synced |

All three branches are also pushed to the user-owned GitHub remote `origin` on
the Mac and fetched through the `hamza` remote on EIT.

The redundant 738 MB Mac SCOPE worktree was removed after its clean HEAD was
verified against GitHub and EIT. Restore it only when needed:

```bash
git -C /Users/hamzaiqbal/grad/LegalRagAgent worktree add \
  /Users/hamzaiqbal/grad/LegalRagAgent-scope-old codex/scope_old
```

## Named archive branches

| Branch | Commit | Purpose |
|---|---|---|
| `codex/archive/pre_cleanup_20260717` | `cc872f88f04a9b703d6adccf68b04c42939e4e07` | Exact tracked state immediately before the active-surface cleanup commit |
| `codex/archive/early_agentic_20260717` | `461dff39c88e63759f8936bdb740b92501edab2c` | Formerly untracked February-March agentic course-project files, imported byte-for-byte under `archive/early_agentic/` |

Both branches are pushed to GitHub. They are preservation refs, not merge
targets for active research.

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

The Mac cleanup payloads and full recovery bundle are stored separately at:

- `/engrfs/project/jacobsn/hiqbal/archives/legalrag/2026-07-17/local-cleanup/archives/`
- `/engrfs/project/jacobsn/hiqbal/archives/legalrag/2026-07-17/local-cleanup/recovery-package/LegalRagAgent-recovery-20260717.zip`

The archive ZIPs passed their stored SHA-256 checks and `unzip -t`; the
recovery ZIP has SHA-256
`c917458a8f6409c9c00585edc0b7a658841b43afbca7d158a91689d20dca6f17`
and passed `unzip -t`. The nested untracked SCOPE tree also matched the Mac
source by relative path and SHA-256 before the local payload was removed.

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
