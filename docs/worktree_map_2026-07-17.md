# LegalRagAgent worktree map — 2026-07-17

## Active and historical surfaces

| Track | Local Mac | EIT persistent project space | Branch | State |
|---|---|---|---|---|
| Three-dial utility | `/Users/hamzaiqbal/grad/LegalRagAgent` | `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-three-dial` | `codex/three_dial` | active, clean, synced |
| OPD/distillation | `/Users/hamzaiqbal/grad/LegalRagAgent-opd-distillation` | `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-opd-distillation` | `codex/opd_distillation` | gated active track, clean, synced |
| SCOPE history | branch only; no checked-out Mac worktree | archived; restore only when needed | `codex/scope_old` | historical Git/archive state |

The two active EIT paths are independent partial clones with the user-owned
GitHub repository named `origin`. They no longer depend on a shared Git
directory inside a historical checkout.

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

## Archived EIT checkouts

`/engrfs/project/jacobsn/hiqbal/src/` now contains only the two active
LegalRagAgent clones. Seven predecessor checkouts—about 84 GB at move time—are
preserved under:

`/engrfs/project/jacobsn/hiqbal/archives/LegalRagAgent_archive/2026-07-17/legacy-worktrees/`

They include the 73 GB dirty common checkout, dirty adaptive-HyRE, HPC-setup,
and Snap-HyRE-comprehensive checkouts, the historical SCOPE checkout, and the
two old linked active worktrees. The linked-worktree pointers were repaired
after the move, and each directory has an `ARCHIVE_STATUS_2026-07-17.txt`.
They are inspectable preservation copies, not launch targets.

Their HEADs, statuses, binary worktree/index diffs, untracked tarballs, and a
full all-refs Git bundle are preserved at:

`/engrfs/project/jacobsn/hiqbal/archives/LegalRagAgent_archive/2026-07-17/`

The pre-move recovery package is 252 MB, its Git bundle verifies, its tarballs
pass gzip checks, and `SHA256SUMS` covers the package. The full moved
directories add a second recovery layer without occupying the source
namespace.

The Mac cleanup payloads and full recovery bundle are stored separately at:

- `/engrfs/project/jacobsn/hiqbal/archives/LegalRagAgent_archive/2026-07-17/local-cleanup/archives/`
- `/engrfs/project/jacobsn/hiqbal/archives/LegalRagAgent_archive/2026-07-17/local-cleanup/recovery-package/LegalRagAgent-recovery-20260717.zip`

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

New development begins from one of the two track-specific paths above. Never
submit from `archives/LegalRagAgent_archive/`. A later artifact-deduplication pass may remove
duplicate environments, Chroma data, caches, or checkpoints inside the legacy
archive only after mapping each payload to a canonical retained copy.
