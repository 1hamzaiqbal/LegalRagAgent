---
title: OPD Program Goal and Execution Boundary
type: snapshot
tags: [opd, math, objective-family, deepmath, preregistration, goal]
created: 2026-07-20
updated: 2026-07-20
status: active implementation goal; no successor scientific arm launched
---

# OPD program goal - 2026-07-20

## Goal

Complete two rigorously separated, outcome-blind campaigns:

1. execute the O-teacher objective-family study on MATH and OpenR1 student
   rollouts (`O_M` and `O_O`) after its implementation, analytic/stored-
   rollout/full-custody fidelity ladder, fresh support gates, fresh strict O
   teacher gap, and three-seed preregistration all pass;
2. qualify DeepMath-103K as candidate source `C` using only data/provenance,
   collision, label/verifier, prompt-bound, and raw-model feasibility evidence;
   if and only if it passes, preregister a later fresh O/C source-transfer study
   (`C_C`, `C_O`, `O_C`, `O_O`) and optionally a routed multi-teacher extension.

These campaigns answer different questions. The first isolates objective
behavior across two student distributions with one qualified O teacher. The
second asks whether a genuinely separate teacher source can support a symmetric
source-transfer or multi-teacher study. DeepMath results cannot alter the first
campaign's objectives, seeds, gates, or analysis.

## Step 1 - O-to-M / O-to-O objective family

The committed registry contains six fixed identities:

- matched task RL;
- task RL plus sampled K1, ungated and clip 5;
- task RL plus sampled K1, ungated and unclipped;
- task RL plus sampled K1, beta-5 positive-gap gated and clip 5;
- local bare K1 with the veRL-compatible clip-10 surface; and
- a pinned upstream-veRL clip-10 reference.

The registry is an implementation identity, not launch permission. Scientific
launch remains fail-closed until one final clean commit binds exact raw-student
support, a fresh passing O teacher and merge, all fidelity receipts, one-step
objective-by-source diagnostics, three seeds, exact prompt plans and shared
initial adapters, evaluation paths, contrasts, correction family, stop rules,
and immutable EIT output roots.

## Step 2 - DeepMath qualification in parallel

DeepMath is provisional source `C`, not a qualified dataset or teacher. The
outcome-blind contract is
`configs/opd_math/deepmath_qualification_plan.json`. It pins the dataset bytes,
forbids all three R1 solution fields from training, and requires:

- a complete global collision inventory spanning C, O, Numina lineage, M
  train/test, MATH-500, AIME/AMC, MATH-Beyond, and every frozen evaluation set;
- no skipped collision buckets, unresolved semantic candidates, or unresolved
  label conflicts;
- at least 5,000 eligible unique clusters, at least 99% gold parseability, and
  zero prompt truncation under the common 1,536-token prompt bound; and
- a fixed disjoint 512-record raw-model surface for Qwen3-8B and Qwen3-1.7B,
  with predeclared non-floor/non-ceiling, student support, mixed-group, and
  verifier-error gates.

No C teacher training is allowed until those gates pass and deterministic
train, teacher-confirmation, student, and source-holdout roles are frozen. A
failed C teacher gap is terminal; another dataset cannot be selected after
seeing that outcome.

## Step 3 - conditional O/C successor

Only after C passes qualification do we create a new preregistration and train
fresh O and C teachers under the same commit, recipe, budget, seeds, and equal-
sized confirmation sets. Both teacher gaps must independently pass before the
2x2 matrix launches. Routed or multiple-teacher OPD is secondary to the four
single-teacher cells and must compare with the best single teacher at matched
student-update and inference budgets.

MATH remains useful here only as a frozen external transfer target. It is not
reintroduced as a teacher-training source.

## Permanent M boundary

The historical M-trained teacher remains failed under the original gate and
teacher-favorable sensitivity. It is never retrained, merged, sampled,
rescored to change the decision, or used to supervise a student. `M_M` and
`M_O` remain prohibited. This does not prohibit MATH data: MATH is a valid
student-rollout and evaluation distribution for the qualified O teacher in
the objective-family campaign, and later an external transfer target.

## Current checkpoint

The objective registry, direct-import analytic gate, and synthetic stored-
rollout gate are implemented. Jobs `108498` and `108501` matched the pinned
veRL scalar and gradient exactly; `108501` additionally matched trace
reconstruction and one AdamW update. A real-model stored rollout, finite-state
coverage, and Level-3 objective-by-source diagnostics remain. The DeepMath plan,
raw-byte/schema verifier, and restartable EIT intake are implemented. Job
`108481` verified all ten pinned Parquet shards (`2,136,106,260` bytes,
`103,022` rows); an independent reopen reproduced the exact manifest hash.
Inventory job `108510` then exposed an empty `problem` in the pinned NuminaMath
lineage source after sealing C and O. It is preserved as a failed data-contract
attempt in [[deepmath-inventory-failure-2026-07-20]]. The successor inventory
retains all upstream rows with an explicit `problem_missing` flag and gives
candidate C a zero-missing-prompt gate; it must use a new root and new plan
hash. The full collision/label audit, tokenizer surface, and raw-model
feasibility runs remain incomplete. No new scientific student arm, C teacher,
or C OPD arm has launched.

## Links

[[opd-objective-family-expansion-2026-07-20]] -
[[opd-m-teacher-clarification-and-source-options-2026-07-20]] -
[[deepmath-103k]] - [[opd-math-source-transfer]] - [[mopd-multi-teacher]]
