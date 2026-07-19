# Active track: OPD math source transfer

This branch (`codex/opd_math_pipeline`) is an isolated child of the gated OPD
track. It tests how the teacher-training source interacts with the student's
on-policy math-rollout source before returning the method to retrieval control.

Start with:

1. `wiki/snapshots/research-state-2026-07-17.md`
2. `wiki/tracks/opd-distillation.md`
3. `wiki/tracks/opd-math-source-transfer.md`
4. `scripts/opd_math/README.md`
5. `configs/opd_math/source_manifest.json`
6. `configs/opd_math/teacher_training_plan.json`

Current gate: freeze one final code-and-documentation commit, rerun full M/O
raw-student support under the v2 exact-environment contract, retrain both
teachers, and establish the held-out teacher gaps before launching the two
task-RL baselines and four OPD arms. The campaign closes only with all six
held-out result gates and the paired matrix readout on that same commit. No
finite loss or checkpoint is a performance result.
