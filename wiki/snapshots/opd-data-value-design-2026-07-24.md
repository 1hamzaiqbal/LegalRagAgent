---
title: OPD Data Value and Epiplexity Experiment Design
type: snapshot
tags: [opd, data-value, source-transfer, epiplexity, experiment-design]
created: 2026-07-24
updated: 2026-07-24
status: design and analysis plumbing only; blocked on positive-control task gain
---

# OPD data value: what is worth distilling?

## Compact research question

Can measurements made before or early in OPD predict which training data will
produce transferable student gains, for a particular student, teacher, target,
and compute budget?

This is the same reader-conditioned marginal-utility object as the three-dial
track, with a training source in place of a retrieved evidence set:

- **reader:** the acting student checkpoint and inference mode;
- **action/source:** which teacher signal and data slice to train on;
- **budget:** examples, generated tokens, optimizer steps, latency, and FLOPs;
- **outcome:** paired task gain relative to a matched non-OPD control.

We do not preregister a scaling law, universal metric, or architecture as the
result. We preregister questions, interventions, measurements, and falsifiers.

## Release ladder

### P0: upstream positive control

The current branch must first establish that exact upstream same-size OPSD can
produce a real parameter update and then a task gain on its registered AIME24
control. One-step success proves plumbing only. The 100-step control must beat
the locally reconstructed base by its existing gate before this study may
claim OPD is an effective intervention.

### P1: measurement fidelity

On frozen prompts and checkpoints, independently establish three distinct
objects:

1. **Prequential proxy:** teacher-forced per-token NLL curves and the discrete
   area above terminal loss, converted from nats to bits.
2. **Requential-compatible code:** teacher-generated paths, unclipped
   full-vocabulary `KL(teacher || student)`, response-token counts, and
   cumulative bits.
3. **OPD-state proxy:** the same KL and sampled-token gaps on student-generated
   paths used by OPD. This is retained under that name and never promoted to a
   requential estimate.

Analytic fixtures must reconstruct every aggregate from per-token records.
An observation-only patch to a disposable upstream execution tree must pass a
one-step objective/gradient agreement test before any measured run.

### P2: low-cost within-source pilot

Create hash-seeded, disjoint, equal-prompt and approximately equal-token blocks
from the already audited O corpus. Block construction cannot use held-out
outcomes. For each block, measure the P1 vector, train short matched OPSD and
control arms from the same initialized adapter, and evaluate paired changes on
development-only target surfaces. Use at least three seeds.

This phase tests whether the measurements have enough variation and stability
to justify a larger source study. It does not establish cross-dataset transfer.

### P3: cross-source confirmation

Only after P2, qualify a genuinely different math source under a separate
boundary. The current failed M teacher, `M_M`, `M_O`, and terminal DeepMath
qualification remain closed. MATH may remain an external evaluation target;
using it as a target does not resurrect an M-trained teacher. A new candidate
source needs its own contamination, verifier, support, and teacher-skill gates.

Freeze source arms, target benchmarks, budgets, seeds, and the prediction rule
before confirmatory outcomes are inspected. Evaluate every trained checkpoint
on every registered target to form a source x target value matrix.

## Experimental unit and outcome

The atomic row is:

`source x student x teacher x objective x seed x budget x target`.

For target score `Y`, the primary causal outcome is a paired
difference-in-differences against a matched control:

`V = (Y_OPD,post - Y_OPD,pre) - (Y_control,post - Y_control,pre)`.

Also report absolute post-training score, calls, generated/scored tokens,
latency, peak memory, and GPU-hours. No conclusion may be based on scheduler
success, finite loss, or a checkpoint existing.

## Measurements to over-collect

- full per-token teacher and student log-probabilities on fixed teacher paths;
- full-vocabulary `KL(teacher || student)` in nats and bits, before clipping;
- the executed/clipped objective value separately;
- student-path sampled K1 gaps and positive/negative/zero fractions;
- teacher-forced source NLL at every registered checkpoint;
- task reward, strict parse state, pass@k, and mixed-reward support;
- prompt, teacher-reasoning, student-completion, and scored token counts;
- answer correctness, source/block IDs, skill/difficulty strata, and collision IDs;
- gradient norm, update norm, adapter norm, optimizer state, and numerical checks;
- generation, teacher-forward, backward, and wall-clock latency plus peak VRAM;
- matched target outcomes at each registered budget.

## Predictions and falsifiers

1. **Plumbing prediction:** redirecting compiler caches should remove the exact
   home-quota failure without changing the one-step training recipe.
2. **Non-monotonicity prediction:** raw teacher-student divergence alone will
   not monotonically predict OPD value. Both near-zero signal and inaccessible
   high divergence can have low value.
3. **Conditional-value prediction:** a combination of extractable structure,
   student support, and source-target alignment should predict value better
   than teacher accuracy, source NLL, or divergence alone.
4. **Observer prediction:** rankings may change with student scale or budget;
   a stable universal source ranking is not assumed.
5. **Falsifier:** if pretraining diagnostics fail leave-one-block-out prediction
   in P2, do not launch a large cross-source campaign on the claim that we can
   predict data value. Preserve the null and either narrow to characterization
   or stop.

## Analysis discipline

- P2 is exploratory for feature/form choice; P3 is confirmatory.
- Report every measured feature, but fit only a small preregistered model in P3.
- Compare against mean-only, teacher-accuracy-only, source-NLL-only,
  divergence-only, and random-block baselines.
- Use leave-one-source-out or leave-one-block-out prediction, paired bootstrap
  intervals for task deltas, and seed-level uncertainty. Individual generated
  samples are not independent replicates.
- Treat AIME24 as a control-selection surface after P0, not an untouched final
  benchmark.

## Current authority

This page authorizes design, schema, offline analysis tests, and future
observation-only instrumentation. It does not authorize a 100-step run, a new
teacher, DeepMath reuse, or held-out inspection. The machine-readable contract
is `configs/opd_math/data_value_v1.json`.

## Links

[[epiplexity]] · [[opd-math-source-transfer]] ·
[[opd-objective-family-expansion-2026-07-20]] · [[opd-distillation]]
