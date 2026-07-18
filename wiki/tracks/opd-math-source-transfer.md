---
title: OPD Math Source-Transfer Track
type: track
tags: [opd, distillation, math, teacher-data, source-transfer]
created: 2026-07-17
updated: 2026-07-18
status: raw-student support passed; teacher science pending
---

# OPD math source transfer

## Research question

When does a teacher's relationship to the student-rollout distribution make it
a better teacher, beyond the teacher's aggregate task accuracy?

This is intentionally a question-and-experiment program rather than an a
priori promise of a new metric, architecture, or scaling law. The first probe
uses two math sources and one tokenizer-aligned Qwen family:

- `M`: MATH-lighteval, 7,500 train and 5,000 frozen test problems;
- `O`: OpenR1-Math-220k `default`, 93,733 train problems and no official test;
- teacher: Qwen3-8B trained separately on M or O with verifiable reward;
- student: Qwen3-1.7B;
- primary matrix: `(M,M)`, `(M,O)`, `(O,M)`, `(O,O)`, where the first
  coordinate is teacher-training source and the second is student-rollout
  source.

## Questions the matrix can answer

1. Does same-source teacher training improve student task learning after
   controlling teacher examples, updates, and rollout count while measuring
   source-dependent token use explicitly?
2. If it does, is the gain explained by higher teacher task accuracy, lower
   teacher NLL on student trajectories, more positive teacher-student token
   gaps, or exact item exposure?
3. Can a teacher with higher standalone accuracy be a worse distillation
   teacher for a particular student/source pair?
4. Which trajectory regions help: correct versus incorrect student samples,
   low versus high teacher NLL, short versus long completions, and positive
   versus negative teacher gaps?
5. Do any relationships transfer from M to O or from the 8B teacher to the 4B
   memory fallback? Only after the first matrix works should model scale become
   another factor.

These measurements could later yield a metric or transfer law, but the current
contribution target is the empirical characterization itself.

## Confounds removed up front

Same-source does not mean same item in the primary matrix. A global problem
cluster is built before splitting. Exact and conservative formatting-only
collisions are joined; clusters spanning M and O are quarantined; and every
training copy of an MATH-test problem is excluded. Remaining clusters are
deterministically allocated within source and problem stratum at 60/30/5/5 to
teacher training, student OPD, teacher-gap development, and source holdout. A
versioned math-token near-duplicate candidate audit runs before partitioning.
It globally orders token-5 shingles and indexes a Jaccard-recall prefix of each
record, so every pair at or above the declared candidate threshold is surfaced
unless an explicit bucket cap is hit; any cap hit fails the scan closed.
High-confidence edges are clustered and every candidate/edge is retained for
sensitivity analysis. The guarantee applies to token-shingle Jaccard, not
arbitrary paraphrases, which this semantic screen does not claim to detect.

`same_items` is retained as a positive-control ablation. It asks whether exact
teacher exposure changes the supervision signal; it cannot support a general
source-alignment claim.

OpenR1 is larger and contains selected DeepSeek-R1 traces. The primary teacher
comparison uses matched problem/update budgets, measures realized token use,
and uses only problem plus gold
answer for GRPO. Its `<think>` messages are not silently used. Full-corpus
OpenR1 is a secondary data-dose arm.

Qwen's pretraining corpus does not give us item-level exposure labels for these
benchmarks. This matrix therefore identifies the effect of our controlled
M-versus-O teacher post-training, conditional on a shared base checkpoint; it
cannot identify whether the base model had previously seen an item. Collision
quarantine removes experiment-introduced overlap, not unknown pretraining
contamination.

## Objective custody

Teacher stage: verifiable-reward GRPO with TRL's DAPO loss normalization. This
is not the full veRL DAPO recipe and will not be labeled `Qwen3-8B-DAPO`.
Both M and O teachers must match the same committed
[`teacher_training_plan.json`](../../configs/opd_math/teacher_training_plan.json):
100 optimizer steps, identical generation/update geometry, decoding, LoRA,
seed, and explicit prompt/completion bounds. Each scientific gate binds the
plan hash, normalized config hash, actual step count, and measured prompt
lengths. This controls our post-training recipe; it does not identify unknown
Qwen pretraining exposure or make realized source token counts equal.

Student stage:

\[
A_i^R = \frac{r_i-\mu_{G(i)}}{\sigma_{G(i)}+\epsilon},
\qquad
L_R=-\frac{1}{N}\sum_i A_i^R\frac{1}{T_i}\sum_t\log p_S(y_{it})
\]

and the dense teacher auxiliary is a token-local score-function surrogate

\[
\Delta_{it}=\log p_T(y_{it})-\operatorname{sg}\log p_S(y_{it}),
\]

\[
L_{SF-gap}=-\operatorname{mean}_{it}
\left[\sigma(\beta\Delta_{it})\,
\operatorname{clip}(\Delta_{it})\,\log p_S(y_{it})\right].
\]

The main loss is `L_R + lambda * L_SF-gap`; `task_rl` is the matched baseline.
The logged ratio is a sampled K1 reverse-KL **value** estimate. Direct autodiff
through K1 would average to zero; the detached ratio times student log
probability is instead a K4/r-trick-equivalent score-function gradient only
when on-policy, ungated, and unclipped. The executed main arm is clipped and
positive-gap-gated, so it is deliberately biased and is not labeled
K4-equivalent. This is not full-vocabulary KL and not an exact SDAR
implementation. See [[ema-policy-gradient]].

## Required gates

1. **Data:** exact pinned revisions, complete collision scan, gold parseability
   reported, semantic near-duplicate audit completed.
2. **Teacher:** trained teacher strictly improves over its own base checkpoint
   on the same frozen `teacher_skill_dev` records and decoding protocol, with a
   positive paired-bootstrap lower bound and a predeclared minimum record count.
   Ties fail. The gate requires completion text and recomputes every reward
   against the exact registered gold; self-reported reward fields cannot pass.
   Target-distribution performance is reported separately.
3. **Student support:** nonzero pass@k and a predeclared minimum fraction of
   mixed-reward groups. Otherwise stop: an identity-bound warm-start lane is
   not implemented, and an ungated smoke cannot substitute for this gate.
4. **Tokenizer:** exact vocabulary/token IDs, special IDs, chat template,
   rendered probes, and live vLLM tokenization match. A scientific main arm
   also binds and rechecks the same-host Linux PID, process start identity,
   command-line checkpoint, merge provenance, alias, port, and maximum context
   length. This is local process custody, not remote cryptographic attestation.
5. **Code custody:** teacher and student runs must begin and end on the same
   clean immutable Git commit; mid-run code drift prevents artifact promotion.
6. **Claim:** a finite step/checkpoint is plumbing only; task performance and
   uncertainty determine scientific promotion.

## Measurements to over-collect

- per-record repeated task reward and parse status;
- prompt/completion tokens, latency, memory, and update budget;
- student NLL and teacher NLL on each student trajectory;
- mean and sign distribution of teacher-student token gaps;
- gate value, task loss, score-function surrogate, sampled K1 value, and
  mixed-reward group fraction;
- answer correctness, completion length, source, role, and collision status;
- base/trained teacher accuracy and paired confidence interval;
- task-RL versus task-RL-plus-score-function-OPD accuracy and learning curves.

## Estimator ablation after the main matrix works

The minimal server returns only the teacher probability of each sampled
student token, so the first matrix is the `k=0` sampled path. A later,
predeclared variance ablation can compare Top-k reverse KL at `k=16` and
`k=32`, and optionally a biased truncated-head-only condition. Top-k is not a
novelty claim: EMA-PG already derives the exact-head plus sampled-tail
estimator, and current veRL exposes on-policy distillation loss modes. The
research object remains whether a teacher's signal helps this student/source,
not which KL implementation exists.

EMA anchoring is excluded from the initial matrix. Replacing the fixed trained
teacher with a lagged student changes the teacher identity and would create a
different three-policy stability experiment.

Perplexity bins are analysis strata, not a training rule assumed in advance.
In particular, low teacher NLL can describe familiar but wrong trajectories.

## Implementation and status

Machine-readable inputs live in
[`configs/opd_math/source_manifest.json`](../../configs/opd_math/source_manifest.json),
the matched teacher recipe lives in
[`configs/opd_math/teacher_training_plan.json`](../../configs/opd_math/teacher_training_plan.json),
and the runnable handoff is
[`scripts/opd_math/README.md`](../../scripts/opd_math/README.md).

The linked `One-Shot-RLVR-Qwen2.5-Math-1.5B-7.5k-MATH` object is a model, not
the MATH dataset. The named third-party Qwen3-8B-DAPO repository has no weights.
Both are recorded as provenance; neither is silently used.

The bounded EIT path is now validated through both physical student-source
role files. [[opd-math-eit-handoff-2026-07-18]] records the exact execution
commits, environments, audit manifest, jobs, update evidence, custody-label
correction, and remaining gates. The completed M->M and O->O smokes establish
exact-token teacher scoring and real parameter movement only. They used
partial non-scientific data, a raw teacher, and flat all-zero task-reward
groups, so no task-performance or source-transfer result is claimed.

The complete raw Qwen3-1.7B support campaign now passes on both canonical
2,161-row student sources: M pass@4 is 0.6201 with 0.1981 mixed-reward groups;
O pass@4 is 0.1772 with 0.1273 mixed-reward groups. The gates authorize an
attempt at scientific task-reward training, not a claim that training or OPD
works. [[opd-math-eit-handoff-2026-07-18]] records the exact gate hashes,
Slurm jobs, independent recomputation, and preserved login-lane anomaly. The
100-step teachers and their own-source skill-gap gates remain pending.

## Links

[[opd-distillation]] · [[self-distillation-cluster-update-2026-07-17]] ·
[[opd-math-eit-handoff-2026-07-18]] ·
[[sdar]] · [[ema-policy-gradient]] · [[verl-opd-trainer]] ·
[[opsd-self-distilled-reasoner]] · [[sdft-continual-learning]] ·
[[sdpo-rich-feedback]]
