---
title: OPD Objective-Family Live Handoff
type: snapshot
tags: [opd, math, objective-family, eit, custody, preregistration]
created: 2026-07-21
updated: 2026-07-21
status: active; fresh O qualification in progress; no scientific student arm launched
---

# OPD objective-family live handoff - 2026-07-21

## Bottom line

The active experiment is the fixed O-teacher objective-family comparison on
MATH and OpenR1 student rollouts. It contains six objectives, two student
sources, and three seeds, for 36 scientific arms. No scientific student arm or
held-out objective-family evaluation has launched.

The experiment checkout is clean at
`d89ba3d7be728d9ee3197f37d8a8836a4a9640c5` both locally and on EIT. The
external release controller is intentionally not part of that checkout. Its
audited bytes are versioned on `codex/opd_objective_family_release_v1` and
sealed on EIT at:

`/engrfs/project/jacobsn/hiqbal/artifacts/legalrag/opd_math/releases/objective_family_v1_58deff52`

## Immutable boundaries

- The historical M-trained teacher failed its original skill-gap gate and the
  teacher-favorable sensitivity analysis. It is permanently excluded: no M
  retraining, M merge, M routing, `M_M`, or `M_O`.
- MATH remains allowed as student-rollout and evaluation data for the qualified
  O teacher. The active cells are therefore O-to-M and O-to-O.
- DeepMath-103K failed its frozen candidate-C qualification because two prompts
  exceeded the common 1,536-token bound while zero truncations were allowed.
  DeepMath teacher training and the conditional O/C matrix remain closed.
- No gate relaxation, objective deletion, replacement seed, rescue training,
  or early held-out inspection is allowed.

## Fixed scientific matrix

The six registered objectives are:

1. matched task RL;
2. task RL plus ungated sampled K1, clip 5;
3. task RL plus ungated sampled K1, unclipped;
4. task RL plus positive-gap-gated sampled K1, beta 5 and clip 5;
5. local bare sampled K1 with the veRL-compatible clip-10 contract; and
6. native pinned-veRL bare sampled K1 at commit
   `6a6242f3d8ec7d9f8b4936f4905144707d91fe3b`.

Every arm uses non-thinking Qwen3-1.7B, a seed-specific shared LoRA
initialization, a source-and-seed-specific 100-prompt order, four rollouts per
update, and 100 optimizer steps. Teacher-scored objectives use only the fresh
O teacher after its strict same-commit gate passes. Bare K1 is a collapse and
implementation control, not the primary scientific claim.

## Passed prerequisites

- Direct imported-veRL analytic scalar/gradient check: job `108498`, completed
  `0:0`.
- Synthetic stored-rollout scalar/gradient/AdamW comparison: job `108501`,
  completed `0:0`.
- Finite-state rejection and nonzero-update check: job `108548`, completed
  `0:0`.
- Train and serve environment freezes: jobs `108551` and `108552`, completed
  `0:0`.
- Pinned veRL environment and corrected preflight: jobs `108574` and `108576`,
  completed `0:0`. The failed predecessor preflight `108575` remains negative
  provenance.
- Shared seed initializations: jobs `108583`--`108585`, completed `0:0`.
- Exact prompt plans and all six native-veRL scientific data files are present
  under the commit-specific objective-family input roots.
- M raw-student support passed and primary/independent files are byte-identical
  at SHA-256
  `23fde719ccc42f4ade03bec742a1014642937d1def02d59859c48a675d2a44e5`.
  Its pass-at-4 is `0.6200832947709394`, mixed-group fraction
  `0.1980564553447478`, and sample accuracy `0.5242943081906525`.
- O raw-student support passed and primary/independent files are byte-identical
  at SHA-256
  `9501b77fb717441dc6ebe8dcf92ed21d4f1161d5a5f5baa3049e38a578f48267`.
  Its pass-at-4 is `0.1772327626099028`, mixed-group fraction
  `0.12725590004627488`, and sample accuracy `0.10411846367422489`.

Student support is a rollout-feasibility gate. It does not rehabilitate the
failed M teacher and is not held-out task-performance evidence.

## Fresh O-teacher qualification in progress

- Fresh O teacher job `108609` completed `0:0`; its adapter remains at
  `teachers/O/run_108609/final_adapter` and is not yet a promoted checkpoint.
- Base/trained timing evaluations `108619` and `108621` and their CPU merges
  `108620` and `108622` completed `0:0`.
- Planner `108623` failed before geometry because it was given a merged timing
  summary instead of the required shard summary. It produced no full
  evaluation, gate, merge, or student result. Its immutable terminal record is
  under `evaluation_plans/O_gap_d89ba3d_v1/`.
- Corrected outcome-blind planner `108916` completed `0:0`. Its immutable v2
  plan is SHA-256
  `30ea4a013927dad397ae6071592225be91919b160c3494d9d92b0c76d96efb4e`
  and fixes identical five-shard `0-4%4` geometry for both arms.
- Full base array `108924` and trained array `108929` were submitted on
  2026-07-21 with that exact geometry over all 4,585 O teacher-gap records.
  Their canonical CPU merges are `108931` and `108932`, dependency-held on the
  corresponding complete array.

The outcome-blind launch intent and scheduler submission custody are sealed at
`ledgers/O_gap_full_d89ba3d_v1/` and
`ledgers/O_gap_full_d89ba3d_v1_submission/`. The intent SHA-256 is
`6270f660f1ee73595c583e81ec3e9091620cd0652be7382a8673507105e72993`;
the submission receipt SHA-256 is
`e8cdb22a1e5beee32898971dd5c5feb567e9231c91de96f55728998cb7cfac5e`.
No evaluation score or completion was inspected when these were created.

The timing plan and live arrays are not gap evidence. Both complete evaluations
must be CPU-merged and feed two distinct canonical deterministic teacher-gap
jobs. The primary and independent computations must agree byte-for-byte and
pass exact recomputation. Only that passing gate can authorize one fresh O
merge and its independent custody audit.

## Release-controller custody

Three independent read-only audits found the original release controller launch-ready
after fixing atomic publication, interrupted first-intent recovery, malformed
arm isolation, exact resumed external-evidence schemas, environment/cost
replay, and Slurm whole-second chronology handling. Before first use, the
controller was extended outcome-blind to make the exact fresh-O independent
audit receipt mandatory in the program manifest, release plan, and every
training authorization. The final controller and auditor were re-audited
against the prospective 108609 gate/merge chain.

Frozen SHA-256 values:

- controller:
  `b9a028350733813b64137224ab07de74088d01e35b5a227a51789c33fef134ab`;
- independent fresh-O auditor:
  `d13b8cdb29bcad21389c4b92d5a26c9fa0ce2cc088ca2ac801c44151e5dea3e2`;
- evaluation wrapper:
  `c94ccd50228d7597a2024d289265be2a12482faa05b2a886135d357a450e612a`;
- local training wrapper:
  `9bdc4830c5208041f8d5a20dcda3e7b17da794dc0e4265eda69cbde581b32dfd`;
- upstream-veRL training wrapper:
  `48429fae630daa21a67d31eb4318fa6cfe8599c35cbe1fc6311676103363fc70`.

These release bytes come from `codex/opd_objective_family_release_v1` at
`b25f4e4` and are sealed on EIT at
`/engrfs/project/jacobsn/hiqbal/artifacts/legalrag/opd_math/releases/objective_family_v1_b9a02835/`.
The predecessor `objective_family_v1_58deff52` release remains immutable
provenance but is not the launch program for this campaign.

The controller passes Python compilation, all wrapper `bash -n` checks, its
built-in self-test with payload SHA-256
`bcef8d90245a179f5a2314e8d89668eaf310c49299dadcb9d17a2e26dbfbf853`,
and focused atomic-copy/zero-intent recovery tests. `shellcheck` is unavailable
on the current Mac.

## EIT transport recovery note

The canonical `wustl` SSH alias may land on `ssh-shell-1.engr.wustl.edu`, whose
noninteractive module initialization hung during the live O full-gap launch.
The same authenticated EIT lane was verified through
`ssh-shell-2.engr.wustl.edu` using the existing `wustl` account/key and all
unchanged `/engrfs/project/jacobsn/hiqbal/...` paths. This is a login-node
transport fallback only: it does not change the experiment commit, Slurm
account, artifacts, environment, or any scientific boundary. Use it only when
the canonical endpoint is unhealthy, and continue to verify the exact
repository/commit and immutable artifact paths before submitting a job.

## Remaining launch order

1. Finish the fresh O teacher, timing prefixes, and immutable full-gap plan.
2. Run complete base/trained O gap arrays and CPU merges; independently
   recompute the strict gate; merge and audit O only if it passes.
3. Run ten local and two pinned-veRL one-step full-custody diagnostics at seed
   zero. These are plumbing checks, never task-performance results.
4. Extract the real-model rollout fixture, pass the local-versus-pinned-veRL
   tensor/update check, post-audit all 12 cells, and seal the fidelity closure.
5. Seal the exact preregistration, launch plan, external program manifest, and
   release plan before any scientific student arm.
6. Launch all 36 three-seed arms without early held-out inspection. Record
   terminal scheduler and deep artifact custody for every arm.
7. Only after all arms are terminal, authorize the globally capped held-out
   wave, produce exact per-arm gates, and release the campaign-wide corrected
   analysis.

## Frozen diagnostic operator choices

Before any one-step diagnostic launch, use this fresh root:

`/engrfs/project/jacobsn/hiqbal/artifacts/legalrag/opd_math/fidelity/full_custody_d89ba3d7be728d9ee3197f37d8a8836a4a9640c5`

The twelve run directories are
`runs/<objective_id>__<source>__seed0`; their post-job receipts are
`receipts/<objective_id>__<source>.json`. The real-model fixture, comparison
receipt, launch ledger, scheduler accounting, and closure live under the same
root. Every path is fresh and no predecessor artifact may be copied into it.

The real-model stored-rollout fixture is frozen to the local
`k1_bare_verl_compatible_clip10__O__seed0` diagnostic. This choice was made
before that run existed and before any diagnostic outcome was observed. O is
used because it is the qualified teacher's own source distribution; the choice
cannot be changed after launch.

The diagnostic M/O veRL JSONL files and manifests already exist under
`objective_family_verl_data/d89ba3d7be728d9ee3197f37d8a8836a4a9640c5/`.
Both seed-zero diagnostic pairs were deterministically reopened and validated
against the canonical task bytes and prompt plans on 2026-07-21.

## Frozen fresh-O promotion paths and scheduler shape

The fresh d89 O teacher may be promoted only through these new paths:

- primary gate:
  `/engrfs/project/jacobsn/hiqbal/artifacts/legalrag/opd_math/gates/teacher_gap/O_gap_d89ba3d_v1.json`
- independent gate:
  `/engrfs/project/jacobsn/hiqbal/artifacts/legalrag/opd_math/gates/teacher_gap/O_gap_d89ba3d_v1_independent.json`
- merged checkpoint:
  `/engrfs/project/jacobsn/hiqbal/artifacts/legalrag/opd_math/teachers/O/merged_d89ba3d_v1`
- independent audit receipt:
  `/engrfs/project/jacobsn/hiqbal/artifacts/legalrag/opd_math/audits/objective_family/O_teacher_d89ba3d_v1.json`

After both full-O merged evaluations exist, run two distinct canonical
`scripts/hpc/slurm_opd_math_quality_gate.sh` jobs with identical scientific
inputs and seed but the two output paths above. Do not override the tracked
job name, resources, time limit, stdout template, workdir, or launcher. Submit
each with exactly `sbatch --parsable --chdir=<d89 repo> --export=<frozen map>
<tracked relative launcher>`. Both jobs must complete `0:0`, produce
byte-identical gates, and pass exact recomputation. The producer-created gate
files may remain mode `0644` during the teacher merge: the tracked merge
reopens and exactly recomputes the primary gate and binds its physical hash in
checkpoint provenance. After merge completion, the external auditor must set
both gate files and all three completed stdout files to exact mode `0444`,
revalidate their bytes and provenance, and persistently archive the logs before
publishing its receipt. No preregistration or student launch object exists
before that receipt.

Only after both gate jobs have completed may one canonical tracked teacher
merge be submitted with the same exact three-option `sbatch` shape. It must
consume the primary gate, the gate-bound fresh-O adapter, and the frozen merged
checkpoint path. The sealed external auditor must then re-query all three
Slurm jobs, reproduce both gate computations, run the strongest d89 teacher
provenance validator, independently reproduce the checkpoint tree hash, seal
the checkpoint/stdout, archive hash-identical copies of all three Slurm logs
under the persistent audit namespace, and publish the fresh audit receipt
without clobbering.
The release program manifest, release plan, and every training authorization
must carry that exact receipt binding and teacher identity. An audit receipt is
therefore a mandatory launch dependency, not an optional bookkeeping record.

## Interpretation boundary

Passing support, environment, analytic, one-step, or finite-update checks means
the experiment is executable and the OPD implementation is faithful enough to
test. It does not mean OPD improves task performance. That claim can be made
only from the sealed held-out release after all 36 arms are terminal.
