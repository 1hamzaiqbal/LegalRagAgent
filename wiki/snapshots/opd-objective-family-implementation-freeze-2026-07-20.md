---
title: OPD Objective-Family Implementation Freeze
type: snapshot
tags: [opd, math, objective-family, verl, fidelity, preregistration]
created: 2026-07-20
updated: 2026-07-20
status: implementation complete locally; EIT environment and diagnostics not yet run
---

# OPD objective-family implementation freeze - 2026-07-20

## Decision boundary

The active scientific target is the O-teacher objective-family comparison on
two student distributions. `M` and `O` identify MATH and OpenR1 student
rollouts; they do not identify two teachers. Every teacher-scored arm uses only
a freshly qualified O teacher. The failed historical M teacher remains
permanently unavailable: no M retraining, merge, routing, `M_M`, or `M_O`.

DeepMath-103K is terminally unqualified as candidate `C` under its frozen
contract. It exceeded the common prompt bound and retained unresolved semantic
review edges. No C feasibility, teacher, O/C matrix, or routed multi-teacher
arm follows from this campaign. See [[deepmath-negative-qualification-2026-07-20]].

## Fixed comparison

The registered matrix is six objectives by two student sources by three seeds,
for 36 scientific arms:

1. task RL;
2. task RL plus ungated sampled K1, clip 5;
3. task RL plus ungated sampled K1, unclipped;
4. task RL plus positive-gap-gated sampled K1, beta 5 and clip 5;
5. local bare sampled K1 using the veRL-compatible clip-10 contract; and
6. native pinned-veRL bare sampled K1 at commit
   `6a6242f3d8ec7d9f8b4936f4905144707d91fe3b`.

The comparison fixes non-thinking Qwen3-1.7B, seed-specific shared LoRA
initialization, source-and-seed-specific prompt order, one prompt and four
rollouts per update, 100 optimizer steps, the same sampling bounds, and the
same O checkpoint. The local task-reward/K1 objectives and native veRL reference
have distinct execution implementations but are explicitly bound to a common
registered recipe. Bare K1 remains a collapse/control diagnostic, not the main
scientific claim.

## Implemented custody surface

The current local implementation adds:

- exact immutable prompt plans and shared initialized adapters;
- a local five-objective launcher with task-RL/no-teacher and O-only K1 routing;
- deterministic scientific preregistration and one scheduler-bound receipt per
  arm;
- exact native-veRL prompt materialization and a pinned upstream run plan;
- a fresh isolated veRL environment setup, exact full-package freeze, and
  two-GPU Slurm preflight;
- native-veRL preflight and post-run custody for its checkpoint, optimizer,
  rollouts, metrics, and LoRA delta;
- extraction of a real Qwen rollout with generation-time behavior scores for
  the stored-tensor local-versus-veRL comparison; and
- a post-job full-custody auditor that reopens each one-step run, binds the
  completed Slurm stdout, and requires all 12 objective-source receipts before
  it can seal the fidelity closure.

The fidelity closure authorizes only later outcome-blind preregistration. It is
not a task-performance result and cannot release held-out evaluation.

## Already passed

- direct imported-veRL analytic scalar and gradient comparison: job `108498`;
- synthetic stored-rollout scalar, gradient, trace, and AdamW update
  comparison: job `108501`;
- finite-state rejection and nonzero-update coverage: job `108548`.

The tracked receipt for the last item is
`evidence/july_2026/opd_finite_state_108548.json`. These are implementation
checks only.

## Remaining execution order

1. Freeze and sync one clean implementation commit.
2. Build and freeze the isolated pinned-veRL environment, then pass its two-GPU
   import/cache preflight.
3. Regenerate exact M/O raw-student support gates on that same commit.
4. Train one fresh O teacher, recompute the strict gap, merge only if it passes,
   and independently audit the merge.
5. Run the ten local and two native-veRL one-step diagnostics at seed 0.
6. Extract one real-model fixture from the local bare-K1 diagnostic and pass
   the stored-tensor local-versus-pinned-veRL comparison.
7. Post-audit every completed diagnostic, seal the 12-cell fidelity closure,
   and only then seal the 36-arm preregistration and launch plan.
8. Launch all three seeds without early held-out inspection; run exact held-out
   gates only after training completes; release and analyze only through the
   campaign-wide correction and stop rules.

No outcome-dependent objective deletion, rescue training, bound relaxation,
or replacement C dataset is allowed inside this preregistration boundary.

## Canonical files

- `configs/opd_math/objective_registry.json`
- `configs/opd_math/objective_family_student_plan.json`
- `configs/opd_math/objective_family_verl_plan.json`
- `configs/opd_math/fidelity_plan.json`
- `scripts/opd/objective_family_inputs.py`
- `scripts/opd/objective_family_fidelity.py`
- `scripts/opd/objective_family_preregistration.py`
- `scripts/opd/prepare_verl_objective_data.py`
- `scripts/opd/verl_objective_contract.py`
- `scripts/opd/verl_run_custody.py`
- `scripts/hpc/slurm_opd_math_objective_family_train.sh`
- `scripts/hpc/slurm_opd_math_objective_family_verl.sh`
- `scripts/hpc/setup_opd_math_verl_env.sh`
- `scripts/hpc/slurm_opd_math_verl_preflight.sh`

## Links

[[opd-program-goal-2026-07-20]] ·
[[opd-objective-family-expansion-2026-07-20]] ·
[[opd-math-source-transfer]] ·
[[deepmath-negative-qualification-2026-07-20]]
