# CLAUDE.md

Operational context for LegalRagAgent as of 2026-07-17.

## Source of truth

Read in this order:

1. `wiki/snapshots/research-state-2026-07-17.md`
2. `ACTIVE_TRACK.md`
3. `wiki/tracks/three-dial.md` or `wiki/tracks/opd-distillation.md`
4. `docs/july_2026_completion_audit_2026-07-17.md`
5. `docs/signoff_log.md`
6. `wiki/literature/index.md`

The old May Snap-HyRE operational narrative is preserved on
`codex/scope_old` and in verified external archives. It is not the active north
star and is intentionally absent from this file.

## Branches and worktrees

| Track | Mac | EIT persistent path | Branch |
|---|---|---|---|
| Three-dial | `/Users/hamzaiqbal/grad/LegalRagAgent` | `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-three-dial` | `codex/three_dial` |
| OPD/distillation | `/Users/hamzaiqbal/grad/LegalRagAgent-opd-distillation` | `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-opd-distillation` | `codex/opd_distillation` |
| SCOPE history | branch only; restore on demand | `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-scope-old` | `codex/scope_old` |

See `docs/worktree_map_2026-07-17.md` before touching EIT. The older
`LegalRagAgent` and `LegalRagAgent-adaptive-hyre` EIT worktrees are dirty
historical surfaces and must not run new experiments.

## Current research state

- SCOPE/Snap-HyRE is closed as the primary contribution.
- The primary object is reader-conditioned marginal evidence-set utility under
  cost, including help, no-effect, harm, stopping, abstention, and conflict.
- Existing July evidence shows reader/task sign reversals; exact values and
  caveats are in the snapshot and source-gated audit.
- Fixed-arm and cheap offline allocation policies did not beat the best fixed
  arm. Repeated outcomes and evidence/reader state are required before claiming
  learnable per-question effort control.
- EIT jobs 93802 and 106078 validate bare and negative-gap-gated OPD plumbing.
  They do not establish task learning.
- E2 teacher-with-skill versus teacher-without-skill is the gate for E3.

## Next build order

1. Freeze the paired `(question, reader, evidence set/action)` schema.
2. Recompute the July master table from row-level logs with uncertainty and
   evidence-role labels.
3. Repeat generations and replicate one reader × evidence crossover on one
   non-legal dataset.
4. Implement fixed-budget, RPP/QPP, intervention, and set-sufficiency baselines.
5. Train marginal-utility stopping only after those baselines are stable.
6. Run OPD E2; proceed to task-RL + gap-gated OPD only if the skill gap exists.

## Environment

```bash
uv sync
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run pytest -q
uv run python scripts/opd/test_opd_loss.py
```

Use `~/.local/bin/uv` if required. Keep API keys in `.env`; never print them.
The local `.venv`, datasets, Chroma store, caches, and logs are intentionally
retained because they are active/reusable substrate, not archival clutter.

## Storage and recovery

- Full-paper/repository vault:
  `/engrfs/project/jacobsn/hiqbal/literature/legalrag/`
- EIT experiment scratch: `/engrfs/tmp/jacobsn/hiqbal_legalrag/`
- Small local archive manifests: `/Users/hamzaiqbal/grad/LegalRagAgent_archive/`
- Verified cleanup ZIPs:
  `/engrfs/project/jacobsn/hiqbal/archives/legalrag/2026-07-17/local-cleanup/archives/`
- Full pre-cleanup recovery ZIP:
  `/engrfs/project/jacobsn/hiqbal/archives/legalrag/2026-07-17/local-cleanup/recovery-package/LegalRagAgent-recovery-20260717.zip`
- EIT historical-worktree recovery:
  `/engrfs/project/jacobsn/hiqbal/archives/legalrag/2026-07-17/`

## Claim rules

- Use `docs/signoff_log.md` as the citation gate.
- Distinguish source-log facts, reconciled inferences, and hypotheses.
- Never call a checkpoint or finite loss a successful policy.
- Never generalize a paired subset threshold as a universal law.
- Every materially relevant paper gets a wiki source page, links, an index/log
  entry, and a checksummed PDF or pinned repository in the EIT vault.
