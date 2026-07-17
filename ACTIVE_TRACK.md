# Active track: OPD and retrieval-skill distillation

This branch (`codex/opd_distillation`) is the clean implementation surface for
testing whether reader-conditioned retrieval-control behavior can be
internalized into a smaller policy.

Start with:

1. `wiki/snapshots/research-state-2026-07-17.md`
2. `wiki/tracks/opd-distillation.md`
3. `wiki/sources/sdar.md` and `wiki/sources/skill1.md`
4. `scripts/opd/README.md`

Current gate: run E2 (teacher with versus without
`scripts/opd/skills/allocation.md`) before scientific E3. The gated objective
implemented here is a tested building block, not task-RL integration and not a
performance result.
