---
title: OPD Identifiability Campaign v1
date: 2026-07-23
status: base reproduced; one-step diagnostic preregistered; full training blocked
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

### Base reproduction result

EIT job `130650` completed on four A6000 GPUs at repository commit
`583fd6d641744eb48047ded32f5f727e141d8af0`. Independent reconstruction found
193 correct of 360 generations, or **53.6111% Average@12**, with 359/360
formatted outputs. This lies inside the preregistered 40--63% sanity band. The
sealed evaluation and gate hashes are recorded in
`configs/opd_math/identifiability_v1_one_step.json`.

This result releases only the one-step real-model update diagnostic. That
diagnostic inherits the official fixed-teacher, full-vocabulary OPSD recipe,
changes `max_steps`, `save_steps`, and `logging_steps` to exactly one, and must
prove a finite loss, a finite positive gradient norm, a step-1 checkpoint, and
finite nonzero LoRA-B parameters. The latter matrices have zero initialization
for the newly constructed PEFT adapter, so a nonzero step-1 value is direct
parameter-update evidence. The run remains plumbing evidence, never a student
task-performance claim. A post-Slurm terminal audit and log hash are required
before the 100-step control can even be preregistered; it is not queued
automatically.

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

The healthy single-node A100-SXM4 lane was not immediately available, while a
single A6000 node had four free identical Ampere GPUs. Positive-control base
and checkpoint evaluation may therefore use either four A100-SXM4 GPUs or four
A6000 GPUs, always on one node and always the same type for the base and every
checkpoint. Training hardware remains a separate memory-gated decision.

## First one-step training incident

One-step job `132150` passed launch custody and reached the pinned local
training-data loader, then failed before model loading or optimization. The
downloaded Parquet rows are physically readable, but their embedded Hugging
Face feature metadata uses `_type: List` for unused list-valued columns. The
pinned upstream `datasets==3.6.0` runtime does not recognize that serialized
feature name. No optimizer step, checkpoint, or OPD result was created.

The correction preserves every raw byte and projects only the upstream
trainer's required ordered `(problem, solution)` columns into a fresh
metadata-free Parquet namespace. A separate process must rehash all 29,434
ordered pairs and load every row with the exact pinned runtime before a new
one-step preregistration can name those normalized artifact hashes. This is a
data-locality compatibility repair, not a semantic dataset or objective
change; the failed job remains immutable.

## Links

[[opd-teacher-evaluator-baseline-qualification-2026-07-22]] ·
[[opd-objective-family-expansion-2026-07-20]] · [[opd-distillation]] ·
[[opsd-self-distilled-reasoner]]
