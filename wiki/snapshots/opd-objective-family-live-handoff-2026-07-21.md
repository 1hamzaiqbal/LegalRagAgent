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

## Fresh O-teacher chain now running

- Teacher training job `108609` is the fresh d89 O teacher.
- Timing base/trained evaluations `108619` and `108621` depend on successful
  teacher completion.
- CPU timing merges `108620` and `108622` depend on their corresponding
  evaluations.
- Planner job `108623` depends on both merges and will seal the exact full-O
  shard count and array specification.

The timing prefix and planner are not gap evidence. After the plan exists, the
complete 4,585-record base and trained O evaluations must run with identical
planned geometry, be CPU-merged, and feed a fresh deterministic teacher-gap
gate. The primary and independent recomputations must agree exactly. Only a
passing gate can authorize one fresh O merge and its independent custody audit.

## Release-controller custody

Three independent read-only audits found the release controller launch-ready
after fixing atomic publication, interrupted first-intent recovery, malformed
arm isolation, exact resumed external-evidence schemas, environment/cost
replay, and Slurm whole-second chronology handling.

Frozen SHA-256 values:

- controller:
  `58deff52bc459963cc993abb0423de0002cda52e052d4499f58e786889fa8161`;
- evaluation wrapper:
  `c94ccd50228d7597a2024d289265be2a12482faa05b2a886135d357a450e612a`;
- local training wrapper:
  `9bdc4830c5208041f8d5a20dcda3e7b17da794dc0e4265eda69cbde581b32dfd`;
- upstream-veRL training wrapper:
  `48429fae630daa21a67d31eb4318fa6cfe8599c35cbe1fc6311676103363fc70`.

The controller passes Python compilation, all wrapper `bash -n` checks, its
built-in self-test with payload SHA-256
`bcef8d90245a179f5a2314e8d89668eaf310c49299dadcb9d17a2e26dbfbf853`,
and focused atomic-copy/zero-intent recovery tests. `shellcheck` is unavailable
on the current Mac.

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

## Interpretation boundary

Passing support, environment, analytic, one-step, or finite-update checks means
the experiment is executable and the OPD implementation is faithful enough to
test. It does not mean OPD improves task performance. That claim can be made
only from the sealed held-out release after all 36 arms are terminal.

