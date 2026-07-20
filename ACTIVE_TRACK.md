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

Current gate: do not launch the unsealed four-arm successor. Before it launched,
the study was expanded into the objective-family design in
`wiki/snapshots/opd-objective-family-expansion-2026-07-20.md`. First finish and
commit the strict verifier-recovery substrate, then implement and validate the
new objective registry, cross-veRL fidelity ladder, generalized preregistration,
and analysis. Full M/O raw-student support and a newly strict O teacher/gap are
shared prerequisites, but no 100-step expanded student arm launches until the
complete three-seed matrix is sealed. M's failed gate is immutable; never
retrain it or launch `M_M`/`M_O`. No finite loss or checkpoint is a performance
result.
