# AGENTS.md

Repository-local instructions for coding agents.

## Start here

1. Read `CLAUDE.md` for the current branch, machine lanes, commands, and gates.
2. Read `wiki/snapshots/research-state-2026-07-17.md` for the durable research
   snapshot.
3. Read the track page for the branch: `wiki/tracks/three-dial.md` or
   `wiki/tracks/opd-distillation.md`.
4. For numeric claims, use `docs/signoff_log.md`, then
   `docs/july_2026_completion_audit_2026-07-17.md`, compact evidence under
   `evidence/july_2026/`, and finally the named source logs.

## Branch and machine lanes

- `codex/three_dial` is the primary science track for reader-conditioned
  marginal evidence-set utility under cost.
- `codex/opd_distillation` is the gated implementation track. E2 must establish
  a teacher skill gap before any scientific E3 distillation claim.
- `codex/scope_old` is immutable historical provenance for SCOPE/Snap-HyRE,
  reviews, class reports, and the old paper.
- The Mac worktrees and persistent EIT worktrees are mapped in
  `docs/worktree_map_2026-07-17.md`. Never run new work from the two dirty
  historical EIT worktrees named there.

The active branches do not contain a live submission paper. Do not recreate or
edit old paper material here; retrieve it from `codex/scope_old` or the verified
ZIPs in `/Users/hamzaiqbal/grad/LegalRagAgent_archive/`.

## Methodology gates

- Verify every reported result against source logs or the signoff/audit path.
- Do not promote plumbing success, finite loss, or checkpoint creation into a
  task-performance claim.
- Three-dial work must retain the paired reader × question × evidence/action
  unit, repeated outcomes where possible, and explicit calls/tokens/latency.
- Compare learned effort control with fixed-budget, utility-prediction,
  intervention, and set-sufficiency baselines. Generic “retrieval helpfulness”
  and generic agentic search are occupied claims.
- OPD work requires task reward plus gap gating for the main arm; bare OPD is a
  collapse diagnostic.
- Preserve provenance. Archive or redirect historical material; never silently
  rewrite old result records.

## Runtime rules

- Use `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` for cached evals unless a
  deliberate download is part of the task.
- Use `uv` or `~/.local/bin/uv`, depending on `PATH`.
- Keep secrets in `.env`; never print or commit them.
- Before committing, run the focused tests for changed code plus
  `git diff --check`.

## Current headline

SCOPE/Snap-HyRE is closed as the primary direction. The active object is
reader-conditioned marginal evidence-set utility: predict whether another
retrieval action will help, do nothing, or harm a particular reader, and stop,
abstain, or arbitrate under cost. OPD is a gated route for internalizing that
policy, not yet a result.
