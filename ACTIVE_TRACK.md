# Active track: OPD identifiability campaign

This branch (`codex/opd_identifiability_v1`) is a fresh successor to the
terminal OPD-math campaigns. Its first job is to reproduce the pinned upstream
OPSD positive control with exact code, data revisions, model revision, and
evaluation before any new project OPD claim or cross-scale run is allowed.

Start with:

1. `wiki/snapshots/research-state-2026-07-17.md`
2. `wiki/tracks/opd-distillation.md`
3. `wiki/snapshots/opd-identifiability-v1-2026-07-23.md`
4. `configs/opd_math/identifiability_v1.json`
5. `wiki/tracks/opd-math-source-transfer.md`
6. `scripts/opd_math/README.md`

Current gate: reproduce the upstream Qwen3-1.7B base AIME24 average@12, pass a
real one-step parameter-update diagnostic, then run the explicit 100-step OPSD
control and evaluate every registered checkpoint. Only a positive result after
independent reconstruction permits a separately preregistered raw-8B to 1.7B
O-only pilot. M's failed teacher gate and DeepMath's failed qualification stay
immutable. No finite loss, checkpoint, or scheduler success is a performance
result.
