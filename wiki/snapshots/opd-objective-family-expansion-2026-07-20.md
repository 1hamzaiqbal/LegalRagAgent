---
title: OPD Objective-Family Expansion Design
type: snapshot
tags: [opd, math, objective-ablation, verl, preregistration, eit]
created: 2026-07-20
updated: 2026-07-20
status: implementation in progress; analytic and stored-rollout fidelity passed; no expanded arm launched or preregistered
---

# OPD objective-family expansion - 2026-07-20

## Decision and provenance boundary

The next scientific campaign will compare task reward, ungated OPD, clipping,
positive-gap gating, and bare OPD rather than treating the current gated
objective as if it were ordinary OPD. It will also attempt an independent
upstream-veRL reference at pinned veRL commit
`6a6242f3d8ec7d9f8b4936f4905144707d91fe3b`.

This page is a design record, not a sealed preregistration and not a result.
Nothing in the terminal `ae90bc7` namespace is modified or reinterpreted:

- The **M-trained teacher** remains a genuine failed teacher. Its adapter is
  never retrained, merged, or used; `M_M` and `M_O` remain prohibited. This is
  not a ban on the MATH dataset: M remains an allowed student-rollout and
  held-out evaluation source for `task_rl` and the passing O teacher (`O_M`).
- The failed-HOME diagnostic `107961`, successful plumbing-only diagnostic
  `108244`, old O gate, and old baseline artifacts remain immutable history.
- The verifier-recovery campaign described in
  [[opd-math-verifier-recovery-2026-07-20]] had not launched or sealed its
  student comparison when this expansion was requested. Its strict verifier,
  O-teacher recovery, support gates, and custody machinery remain common
  prerequisites. Its planned four-arm result is not silently repurposed.
- Expanded artifacts require a new campaign ID, sealed preregistration,
  immutable EIT root, and stable run IDs. No held-out result from a predecessor
  may authorize or select an expanded objective.

The working campaign name is `opd_math_objective_family_v1`. The exact
campaign root and Git commit will be written only when the implementation and
fidelity tests are complete.

Implementation checkpoint: the declarative registry now lives at
`configs/opd_math/objective_registry.json`, binds nullable clipping and exact
coefficients into the local trainer, and records both byte and canonical
hashes. Initial CPU tests execute and independently reconstruct all five local
objectives and distinguish clipped, unclipped, and gated gradients. The
dedicated local veRL-compatible arm now stores behavior log probabilities and
executes the pinned ratio-form PPO scalar rather than merely relabeling the
local score-function surrogate. Direct-import job `108498` matched the pinned
veRL scalar and gradient exactly; stored-rollout job `108501` also matched its
trace reconstruction and one AdamW update exactly. These complete the positive
analytic and synthetic stored-rollout calculations, not the full ladder.
Finite-state coverage, a real-model stored rollout, objective-by-source
one-step custody, and scientific preregistration remain incomplete; the
trainer explicitly rejects a registry-selected scientific launch.

## Scientific questions

1. Does adding an ungated sampled-K1 teacher auxiliary improve matched task RL?
2. Does the positive-gap gate improve on that ungated auxiliary?
3. Does clipping help, harm, or merely suppress large negative teacher gaps?
4. Does pure OPD without task reward improve, collapse, or simply imitate on
   the target task?
5. Does the local K1 implementation agree with the pinned upstream veRL
   calculation and, if systems matching is achievable, with an actual veRL
   training run?
6. Are the answers different when the passing O teacher supervises M student
   rollouts (`O_M`) versus O student rollouts (`O_O`)?
7. Which preregistered teacher-student gap regions predict helpful, null, or
   harmful updates?

This is an empirical characterization study. No metric, scaling law, or new
architecture is assumed in advance.

## Objective registry

Every scientific run will select one immutable objective ID from a committed
registry. Free-form combinations of command-line flags will not authorize a
scientific run.

| ID | Task reward | K1 signal | Gap clip | Positive-gap gate | K1 scale | Role |
|---|---:|---:|---:|---:|---:|---|
| `task_rl` | 1.0 | off | n/a | off | n/a | matched task-learning baseline |
| `task_rl_k1_ungated_clip5` | 1.0 | on | 5.0 | off | 0.01 | isolates removal of the gate |
| `task_rl_k1_ungated_unclipped` | 1.0 | on | none | off | 0.01 | clean sampled-K1 score-function auxiliary |
| `task_rl_k1_gated_clip5_beta5` | 1.0 | on | 5.0 | sigmoid beta 5.0 | 0.01 | current gated OPD variant |
| `k1_bare_verl_compatible_clip10` | off | on | 10.0 | off | 1.0 | pure local OPD/collapse diagnostic and local veRL-compatible objective |
| `k1_verl_upstream_clip10` | off | on | 10.0 | off | 1.0 | actual pinned-veRL reference, conditional on systems audit |

For the first three task-plus-teacher rows, disabling the named mechanism is
the only objective change: student, O teacher, coefficient, data order, seed,
rollout geometry, update budget, optimizer, decoding, LoRA rank, completion
bound, and gradient clipping remain matched. “Unclipped” removes only the
teacher-student-gap clamp; the shared global gradient-norm clip remains fixed.

`k1_bare_verl_compatible_clip10` is both the requested pure/bare OPD ablation
and the local side of the veRL agreement test. It is not coefficient-matched to
the task-plus-K1 auxiliary and is not a replacement for task RL. Combining
those roles avoids a redundant pure clip-5 arm. The two `clip10` rows are the
implementation-agreement pair.

The upstream row is called **parameter-matched veRL** only if the audit can
match all material fields: initial student and O teacher, tokenizer and
prompts, exact task rows/order, `n=4` rollouts, 100 updates, response-only mask,
non-thinking sampling, AdamW and learning rate, LoRA versus full-parameter
training, batch/reduction semantics, old-policy log-probabilities, K1 clamp,
gradient clipping, and effective sample/token budget. If a material mismatch
remains, the immutable ID stays the same but all reports call it an **upstream
veRL reference**, list the mismatches, and do not claim parameter matching.
The expected honest label is “declared-hyperparameter-matched veRL pure-K1
reference”: HF versus vLLM sampling, FSDP mixed-precision forward numerics, and
runtime packages are real backend differences even when the scientific recipe
is matched.

## Experiment matrix

The only teacher is the newly strict-gated O teacher. Each objective is trained
from the same immutable Qwen3-1.7B starting checkpoint on both allowed student
sources:

| Training source | Meaning | Objectives per seed | Primary evaluation |
|---|---|---:|---|
| `O_M` | O teacher scores M student rollouts | 5 local plus 1 upstream | frozen M source holdout |
| `O_O` | O teacher scores O student rollouts | 5 local plus 1 upstream | frozen O source holdout |

Thus one seed contains 12 arms: six objectives on M and six on O. The
`task_rl` row has no teacher request but is duplicated across sources as the
matched baseline. All arms use one prompt group per optimizer step, four
student samples per group, 100 optimizer steps, and the committed student
recipe unless the veRL systems audit proves that one field cannot be matched.

This campaign therefore already has two usable benchmark distributions. It
does not require a second qualified teacher and must not be delayed or silently
expanded while searching for one. A new teacher source is needed only for a
separate teacher-source x student-source interaction or multi-teacher study;
see [[opd-m-teacher-clarification-and-source-options-2026-07-20]].

The final comparison preregisters seeds `0`, `1`, and `2` before any 100-step
held-out outcome is inspected. Seed 0 may be operationally launched first, but
it is a final registered replicate, not a result-selection pilot; no heldout is
inspected until every seed completes or reaches a preregistered terminal
numerical failure. It cannot be used to drop objectives, tune coefficients,
change gates, or select replacement seeds. A failed numerical-safety arm is
preserved as a failure; it is not rescued. The only pilots are the twelve
one-step objective-by-source diagnostics.

Each source/seed receives an explicit 100-record prompt plan and one canonical
initialized LoRA adapter shared by all local and veRL arms. Implementations may
not reshuffle rows or independently initialize adapters. Realized token counts
are measured rather than claimed identical because different policies and
rollout backends can produce different completion lengths.

Primary evaluation is matched-source and paired by frozen record ID. A later
cross-source evaluation of every checkpoint is optional and must be separately
preregistered because it approximately doubles held-out inference cost.

## Common gates before the comparison

1. Commit, push, and freeze one clean recovery/expansion commit in the active
   local and EIT clones.
2. Produce the sealed negative-only legacy M audit. It may preserve the fact
   that M failed; it can never authorize M teacher use.
3. Regenerate strict raw-student support for M and O.
4. Train only one strict 100-step O teacher. Require informative task reward.
5. Run paired base/trained timing, then the complete 4,585-record O gap
   evaluation with at least five shared shards. Require the preregistered
   teacher-favorable worst-case confidence lower bound to remain positive.
6. Merge only the passing O teacher and bind its checkpoint tree, tokenizer,
   environment, process, and evaluation provenance.
7. Pass all three fidelity levels below.
8. Pass one-step full-custody diagnostics for every materially different
   objective on both sources. Diagnostics never inspect source-holdout results.
9. Seal the full three-seed arm matrix and analysis before launching any
   100-step expanded arm.

If the O gate fails, no teacher-based objective launches. `task_rl` may still
be preserved as a baseline experiment, but it is not evidence about OPD.

## Fidelity ladder

### Level 1: analytic tensors

CPU float64 tests will cover:

- raw K1 value and detached teacher-minus-student advantage;
- gradient sign for positive and negative gaps;
- response-only masking and zero gradient on masked positions;
- clip-5, no-clip, and clip-10 behavior at and beyond the boundary;
- beta-5 gate values and gradients;
- global token aggregation and veRL old/current-log-probability handling;
- teacher detachment and absence of gradient through the advantage; and
- fail-closed rejection of nonfinite loss, per-token surrogate, gradient,
  optimizer state, parameter, or update norm.

The veRL-compatible test imports the loss implementation from the exact pinned
checkout. A copied formula is not sufficient evidence of upstream agreement.

### Level 2: one shared stored rollout

Local and veRL-compatible calculations consume identical prompt/completion
token IDs, response masks, student behavior/current log-probabilities, and O
teacher log-probabilities. The fixture is hash-bound and retained on EIT. We
compare:

- raw and executed per-token advantages;
- gate values and K1 value;
- scalar veRL ratio-form objective and local score-function surrogate;
- gradient sign and cosine; and
- one deterministic optimizer update from the same starting parameters.

Scalar/tensor comparisons use predeclared absolute/relative tolerances.
Gradient and update agreement use a predeclared cosine threshold. The current
generic local score-function scalar is not required to equal veRL's PPO-ratio scalar;
their gradients must agree at an exactly on-policy ratio of one. The dedicated
local veRL-compatible calculation must match veRL's scalar, ratio, gradient,
and update. The existing schema-v2 `108244` trace may seed a provenance-linked
real-rollout fixture, but
it remains plumbing evidence and never becomes a performance result.

Synthetic Level-2 fixture SHA-256:
`e8a3469cfb90b6d5b8fc1ce0519efdbaac3e650fa306cf947f7910ae124e4ef5`.
Job `108501` passed all declared comparisons with zero scalar, gradient, and
AdamW-update error and on-policy gradient cosine `1.0`; its receipt SHA-256 is
`810ef012721d9555dd5dae5abf1c35989e6a5ca5327e63c4b0a41dc5e07cd601`.
This is synthetic stored-tensor evidence. A hash-bound real-model rollout and
Level-3 full-custody diagnostics remain required.

### Level 3: one-step full custody

Each objective/source pair starts from the exact registered student, samples
its own four-trajectory group, applies one optimizer step, and seals:

- clean Git/upstream commit and environment freezes;
- objective-registry and preregistration hashes;
- task row, prompt, token, and RNG identities;
- exact behavior, current-student, and teacher token log-probabilities;
- raw/clipped gap, gate, reward, loss, gradient, and update metrics;
- adapter/checkpoint tree and parameter delta; and
- Slurm job, GPU, wall-time, and stdout custody.

Every materially different objective must show finite arithmetic and a real
parameter update. That proves plumbing only. It does not authorize a claim of
task improvement.

## Implementation delta

The current code supports `task_rl` and the clipped beta-5 gated main arm, plus
legacy diagnostic K1 modes. The successor needs:

1. a declarative, hash-bound objective registry with `None` as a real no-gap-
   clip value rather than a magic large number;
2. scientific custody for all four local teacher-scored objectives, not the current
   `TASK_REWARD_MODES` shortcut;
3. explicit 100-record prompt plans and one hash-bound initialized LoRA adapter
   per seed, consumed without implementation-local reshuffling;
4. explicit behavior-log-probability storage and response-mask custody;
5. a separate local veRL-compatible PPO-ratio K1 objective with clamp 10,
   response-only token-mean aggregation, and old/current-logprob handling; the
   current detached score-function importance ratio is not a substitute;
6. per-token raw/executed advantage, gate, surrogate, and old/current ratio;
7. per-step finite checks before and after backward/optimizer, plus gradient
   and update norms and finite optimizer-state checks;
8. a generalized 12-arm-by-three-seed preregistration and result builder;
9. seed-stratified paired held-out analysis and gap-region analyses;
10. a no-shuffle Parquet adapter, digest-pinned environment, and launcher for
    the clean pinned veRL checkout; and
11. an upstream observation-only patch or callback whose bytes and purpose
    are recorded separately from objective code.

The upstream checkout itself remains pinned and clean. Any required data or
instrumentation patch is generated from this repository, checksummed, and
applied to a disposable execution tree. If objective code must be changed,
that run is not an actual upstream veRL reproduction.

## Measurements retained

For every sample and step, retain exact token IDs and masks; behavior, current
student, and teacher log-probabilities; task reward and strict verifier status;
raw and executed K1 advantages; gate values; score-function token terms; K1
value; task and total losses; positive/negative/zero-gap fractions; completion
length; realized tokens; gradient and update norms; optimizer finiteness;
rollout/training/teacher latency; peak GPU memory; Slurm elapsed time; and
teacher-serving request/token counts. Large traces and checkpoints remain on
EIT under the immutable campaign root.

Heldout records retain all four completions, strict verifier attempts and
uncertainty, paired record IDs, and worst-case bounds. No verifier failure is
silently converted into an ordinary wrong answer.

## Preregistered analysis

The four co-primary contrasts are:

1. `task_rl_k1_ungated_clip5 - task_rl` on M;
2. `task_rl_k1_ungated_clip5 - task_rl` on O;
3. `task_rl_k1_gated_clip5_beta5 - task_rl_k1_ungated_clip5` on M; and
4. `task_rl_k1_gated_clip5_beta5 - task_rl_k1_ungated_clip5` on O.

Use a paired hierarchical bootstrap—seed, then record—with 10,000 fixed-seed
draws and Bonferroni 98.75% intervals across those four contrasts. Secondary
95% analyses cover clipped ungated minus unclipped, pure K1 minus task RL and
raw student, local minus upstream veRL, source interactions, verifier-
uncertainty envelopes, and token/GPU-normalized effects. The three individual
seed effects, mean effect, minimum/maximum seed effect, task accuracy,
completion length, realized tokens, and GPU time are always shown. Three seeds
support a robustness check, not a scaling-law claim.

The local-versus-upstream contrast is an implementation-agreement analysis,
not a superiority test and not a statistical equivalence claim. Objective-
level tensor and one-step gradient/update agreement are the primary fidelity
evidence; final task performance is a secondary systems comparison because
distributed generation may not be bitwise identical.

Preregistered moderator bins are source, task correctness, teacher/student
correctness pattern where observable, raw gap sign, raw gap quantile fixed
from training traces without heldout labels, completion-length quantile, and
step quartile. Gap-region analyses are descriptive unless the preregistration
names a heldout prediction and correction family.

## Preregistration artifacts

The successor uses four immutable layers:

1. `common_prerequisites.json`: data/prepared manifest, M-negative audit,
   support run IDs, O teacher run, timing plan, strict O gap, merge identity,
   commits, environments, and all thresholds.
2. `fidelity_plan.json`: objective registry, analytic cases, stored-rollout
   fixture, tolerances, expected veRL API, and one-step diagnostic paths.
3. `experiment_preregistration.json`: all 36 training arms, stable run IDs,
   sources, seeds, exact configs, row order, starting checkpoints, evaluation
   paths, primary contrasts, multiplicity method, stop rules, and cost budget.
4. `launch_ledger.jsonl`: append-only submissions, dependencies, Slurm IDs,
   terminal states, and checksums. Operational retries receive new run IDs and
   preserve failed attempts; scientific settings do not change.

Each layer is written atomically, independently validated, made read-only, and
hash-bound by every downstream manifest. The preregistration is outcome-blind
operator custody, not a claim of cryptographic chronology.

## GPU cost estimate

Observed EIT A100-SXM4 usage gives the following planning anchors:

| Work | Observed or estimated A100-GPU hours | Basis |
|---|---:|---|
| strict M raw-student support | 6.1 | 34 historical shards |
| strict O raw-student support | 7.6 | 34 historical shards |
| strict 100-step O teacher | 1.6 | job `107419` |
| complete O base gap evaluation | 34.5 | three historical shards |
| complete O trained gap evaluation | 63.8 | three historical shards |
| one M heldout checkpoint | 1.9 | six historical shards |
| one O heldout checkpoint | 2.3 | six historical shards |
| one local 100-step task-RL arm | 0.6--0.8 | jobs `107420` and `107421` |
| one local 100-step teacher arm on M | 0.82 | baseline plus observed `108244` server overhead |
| one local 100-step teacher arm on O | 0.97 | O baseline plus observed server overhead |

The fixed strict prerequisite is therefore approximately **114 A100-hours**,
dominated by the full O teacher-gap evaluation. One 12-arm seed is estimated
at **38.3--41.0 GPU-hours** including matched heldouts and a provisional
two-GPU veRL reference. Three seeds, raw-student reference evaluation, and all
twelve one-step diagnostics add approximately **122.6--131.5 GPU-hours**;
budget **135--145 GPU-hours** with operational contingency. Because the strict
O authorization must be regenerated under the current same-commit boundary,
the initial end-to-end campaign is roughly **236--255 GPU-hours**. This is a
GPU-hour estimate, not queue or wall time.

The actual veRL cost is provisional until its EIT environment and minimum
resource topology pass a one-step diagnostic. Upstream's canonical script uses
eight actor GPUs plus four teacher GPUs. The audited minimum for the smaller
Qwen3-1.7B/Qwen3-8B pair is two full A40s—one actor/rollout and one teacher—on
the three-A40 Jacobs lane; a third GPU is a separately preregistered fallback,
not an improvised rescue. A40 hours and the historical A100-SXM4 anchors are
reported separately in the final cost ledger rather than silently treated as
identical hardware.

Optional cross-source evaluation for every three-seed checkpoint adds roughly
another 75--80 A100-hours. Large checkpoints and traces remain on persistent
EIT storage; only compact manifests, tables, and analyses enter Git.

Final-only LoRA adapters for 36 arms are expected to occupy about 5.4 GB; the
complete result namespace should remain near 6--8 GB before the separate veRL
runtime/cache. The merged O teacher is referenced by hash rather than copied.

## Launch order and stop rules

1. finish and commit the strict recovery substrate;
2. implement the registry, traces, analysis, and veRL bridge;
3. pass CPU tests and shared-rollout fidelity;
4. freeze one clean successor commit locally and on EIT;
5. run common O-only prerequisites;
6. run and seal all one-step diagnostics;
7. seal the complete three-seed preregistration;
8. launch seed 0, then seeds 1 and 2 without inspecting or adapting to
   heldout outcomes;
9. run the exact paired heldouts and outcome-blind result builder; and
10. publish all successes, nulls, harms, numerical failures, and systems
    mismatches.

Never relax the O gate, verifier cap, support gate, finite-safety checks, or
hyperparameters after observing an outcome. Never substitute a new seed for a
failed registered seed. Never promote finite loss, a checkpoint, or cross-
implementation tensor agreement into a task-performance claim.

## Multi-teacher follow-on

Pinned veRL MOPD uses hard per-sample `data_source` routing to one teacher; it
does not average teachers or arbitrate disagreement. A scientific multi-
teacher experiment currently has no valid second teacher because M failed and
is permanently excluded. The later route is to establish an independently
passing second teacher, preregister a sealed routing manifest, bind teacher
identity to every sample/token trace, and compare routed multi-teacher OPD to
the best single passing teacher. It is deliberately outside this objective-
family campaign.

## Links

[[opd-math-verifier-recovery-2026-07-20]] · [[opd-math-source-transfer]] ·
[[verl-opd-trainer]] · [[mopd-multi-teacher]] · [[mopd-multi-rollout]]
