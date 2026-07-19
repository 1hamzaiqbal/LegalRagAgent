---
title: OPD Math Scientific Cutover
type: snapshot
tags: [opd, math, eit, source-transfer, custody, scientific-campaign]
created: 2026-07-18
updated: 2026-07-18
status: predecessor jobs quiescing; final campaign not launched
supersedes: opd-math-eit-handoff-2026-07-18
---

# OPD math scientific cutover - 2026-07-18

## Bottom line and claim boundary

The OPD-math implementation has a fail-closed exact-environment successor
contract, but the final scientific campaign has not started. Commit `feacecb`
hardens evaluation publication, environment custody, v2-only authorization,
student-result closure, and O-source timing plans. It does not turn any
predecessor training or evaluation into a scientific result.

The campaign must use one final Git commit for fresh M/O raw-student support,
both 100-step teachers, teacher-gap evaluation and gates, two task-RL baselines,
four OPD arms, all held-out result gates, and the six-arm readout. A tracked
commit after that boundary reopens the campaign. Plumbing success, finite loss,
checkpoint creation, and manifest-level eligibility are not task-performance
claims.

## Canonical data identity

- prepared manifest SHA-256:
  `dc4cf7dc36ae5b5178b782bb9c9841e096fbf42dabcbebaa74fb0ed6afcdf430`;
- MATH frozen test: 5,000 rows, with shared-stem subquestions retained;
- matched teacher train: 4,322 rows per source;
- student train: 2,161 rows per source;
- matched teacher gap: 353 rows per source;
- source holdout: 370 rows per source;
- scientific own-source teacher-gap support: M 353 rows and O 4,585 rows.

## Contract cutover

| Surface | Successor requirement |
|---|---|
| Evaluation | `opd_math_evaluation_contract_v2_exact_environment` |
| Shard and merge publication | exact two-file output tree plus adjacent `<dir>.custody.json` commit companion |
| Student support | `student_support_v2_exact_environment` |
| Held-out arm result | `student_heldout_result_v2_exact_environment` |
| Six-arm readout | `opd_math_six_run_matrix_v2_exact_environment` |
| Gate/result publication | fsynced temporary file plus no-overwrite hard-link commit |

Every authorizing consumer independently rehashes task data, samples, summaries,
companions, environment freezes, and the relevant Git identity. Legacy v1
evaluations remain readable for diagnosis but cannot authorize support,
teacher-gap, held-out, or matrix claims.

## Demoted predecessor evidence

The following artifacts are preserved; none can populate the final matrix:

- legacy `f283c9c` raw-student support: M pass@4 0.6201 and mixed-group
  fraction 0.1981; O pass@4 0.1772 and mixed-group fraction 0.1273. These are
  numerical diagnostics only because the evaluations lack the successor exact
  environment contract;
- predecessor commit `6be96e6` task-RL jobs `107182` and `107183`: 100 steps,
  with respectively 22 and 12 mixed groups. They demonstrate usable training
  signal and parameter movement, not held-out improvement;
- predecessor commit `a3be35f` teachers and evaluation jobs: recipe and systems
  diagnostics only. Terminal hashes and accounting are recorded after all jobs
  quiesce; no artifact is deleted or overwritten.

## Exact-environment and same-commit invariants

The unchanged isolated environments are captured in new read-only,
commit-specific train and serve freezes. Each allocated job verifies its live
environment, freeze bytes, installed-distribution map, executable identity,
clean worktree, and full Git commit. Teacher evaluation must equal the teacher's
training commit; support must equal the current student-training commit;
held-out evaluation must equal its student run; all six arms must share the
same exact environment and Git identity.

The EIT checkout remains at predecessor `a3be35f` until all submitted
predecessor jobs are terminal and audited. It is then fast-forwarded once to
the final pushed commit and held there through both teachers and the downstream
campaign.

## O timing-plan contract

The primary O own-source evaluation plan is derived from a 32-record exact-v2
timing prefix for a 4,585-record total, safety factor 1.25, maximum shard wall
time 64,800 seconds, and concurrency four. The plan binds the raw `sacct` row,
commit, task file, exact environment freeze, model/revision, generation budget,
shard count, array index span, and adapter mode. The self-hash detects mutation;
the literal `sbatch --array=...%4` command and read-only launch ledger remain
part of operator custody because Slurm does not expose the original throttle
reliably per task.

## Final launch sequence

1. Quiesce and terminally audit every `a3be35f` teacher/evaluation job.
2. Push and freeze the final code-and-documentation commit and both exact
   environments.
3. Exercise one-record evaluator plus merge publication custody on persistent
   `/engrfs/project` storage.
4. Rerun full M/O raw Qwen3-1.7B support under v2 and publish fresh support
   gates.
5. Retrain both Qwen3-8B teachers for the predeclared 100-step recipe.
6. Evaluate base and trained teachers on the complete own-source gap surfaces
   and require the held-out teacher-gap gates.
7. Run the two matched task-RL baselines and four source-transfer OPD arms.
8. Publish all six held-out gates and the deterministic paired matrix readout.

## Validation and residual boundaries

At code commit `feacecb`, 256 local tests pass; workspace-contract validation,
Bash syntax, Python compilation, and `git diff --check` also pass. The final
campaign commit must repeat that ladder after documentation closes.

Two residual operational P2s remain explicit. A hard kill between directory
promotion and companion publication can require manual quarantine, but the
orphan cannot authorize downstream work. Timing provenance remains
operator-trusted rather than scheduler-signed, so the raw accounting capture,
read-only plan, and literal launch ledger are mandatory.

## Links

[[opd-math-source-transfer]] · [[opd-distillation]] ·
[[opd-math-eit-handoff-2026-07-18]] · [[research-state-2026-07-17]]
