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

Current gate: validate the isolated environment and full collision-aware data
preparation, then run one-step teacher and task-reward-plus-score-function-OPD
plumbing smokes.
No finite loss or checkpoint is a performance result. A real main arm requires
a strictly positive held-out teacher gap, student rollout support, and the exact
tokenizer/server contract.
