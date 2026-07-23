---
title: OPD Identifiability Campaign v1
date: 2026-07-23
status: setup only; no training authorized
tags: [opd, positive-control, identifiability, opsd, math, eit]
---

# OPD identifiability campaign v1

## Decision

The terminal [[opd-gated-campaign-v2-2026-07-22]] result is preserved. It did
not test student improvement: the trained O-teacher family failed its fixed
generation-length gate before any OPD arm launched. The successor therefore
separates three questions that the earlier large grid would have confounded:

1. Can the EIT environment recover a known-positive official OPSD result?
2. Can a raw, clearly stronger 8B teacher improve a 1.7B student on O beyond
   matched task reward and offline distillation?
3. Only after those pass, does additional O post-training make the 8B model a
   better teacher rather than merely a better solver?

The machine-readable source of truth is
`configs/opd_math/identifiability_v1.json`.

## Stage P1: known-positive control

The control uses the official OPSD checkout at commit `7448751f...3df`,
Qwen3-1.7B in both student and privileged fixed-teacher roles, the pinned
29,434-row `siyanzhao/Openthoughts_math_30k_opsd` corpus, and AIME 2024 as a
control-selection surface with 12 generations per problem. It is a setup
control and prior-art reproduction, not a project contribution or untouched
project holdout.

The upstream launcher contains an ambiguity: it specifies 30 epochs and omits
`max_steps`, while the README calls training a roughly 100-step run and its
evaluation launcher expects checkpoints 25, 50, 75, and 100. This campaign
adds the explicit `--max_steps 100` CLI override and records it as a harness
correction. Upstream objective code remains unchanged. Raw source, the applied
data-locality patch, environment, data, checkpoints, logs, and evaluations are
hash-bound.

P1 passes only when the base control lies inside the preregistered sanity band,
all four checkpoints are evaluated, the best checkpoint improves Average@12
by at least three percentage points over the locally measured base, training
is finite with a real parameter update, and an independent reconstruction
agrees. Scheduler completion alone never passes the stage.

## What becomes available after P1

A passing P1 authorizes preparation—not automatic launch—of a minimal O-only
cross-scale pilot with raw Qwen3-8B teaching Qwen3-1.7B. The first comparison
is raw versus task RL, offline distillation, bare sampled K1, task RL plus
gated K1, and a top-k or full-distribution positive control. MATH remains an
external transfer surface; the failed M teacher remains permanently excluded.

The previous 36-arm objective-family design becomes a confirmatory campaign
only after the minimal cross-scale pilot establishes a useful signal.

## First setup incident

Jobs 130641 and 130642 successfully built the exact pinned environment and
materialized the exact datasets. Slurm split the original four-GPU preflight
130643 across two nodes because the wrapper requested four GPUs but did not
require one node. The preflight and its untouched dependent base evaluation
130644 were cancelled. No evaluation or training ran. The terminal receipt is
`preflight/job_130643/terminal_failure.json` under the EIT campaign artifact
root. The corrected wrappers require one node and record the immutable setup
producer commit separately from the corrected execution commit.

## Links

[[opd-teacher-evaluator-baseline-qualification-2026-07-22]] ·
[[opd-objective-family-expansion-2026-07-20]] · [[opd-distillation]] ·
[[opsd-self-distilled-reasoner]]
