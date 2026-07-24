---
title: OPD Identifiability Campaign v1
date: 2026-07-23
status: base and 1024-token one-step passed; 4096-token diagnostic preregistered; full training blocked
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

Job `132150` is sealed by a read-only terminal-failure receipt with SHA-256
`5798e721ff7a805311907d0f1d5ef336a9d85e7fd8f51804f878658d7facaa1b`.
Normalization job `134970` and independent pinned-runtime audit job `134971`
then reconstructed all 29,434 ordered pairs. Both source and normalized views
have ordered-pair digest
`70c846e88711954db47176e133aae7b938f28eeed0f5beac07ab5681ddf9df77`;
the audit loaded all rows with `datasets==3.6.0` and `pyarrow==25.0.0`. The
normalized manifest and audit hashes are, respectively,
`a46726161fb7edfe5a1c2cab90fbb8b2f5e4187ae00e4de96205f61aaebf5408`
and `b59ad7d806d6aa38f11b1cc51f00bec1f4996a887078d8c71286966c2a93afb6`.
These receipts authorize only a newly hash-bound one-step retry; 100-step
training remains blocked.

That retry, job `135003`, verified that the metadata repair worked: pinned
`datasets` loaded all 29,434 rows, all four ranks loaded the model, and TRL
entered preprocessing. It then failed before optimizer creation with
`KeyError: 'text'`. The first projection had removed both conversational
columns, but pinned TRL 0.26 uses source `conversations` to identify the dataset
as conversational and convert it to ChatML `messages`; without that field it
falls through to its default `text` column. No checkpoint, in-job pass gate,
or OPD result exists. A second compatibility correction must therefore retain
exact ordered `(problem, solution, conversations)` fields, independently run
all rows through pinned ChatML conversion and tokenization plus the upstream
custom collator on CPU, and bind those receipts before another one-step retry.

Job `135003` is sealed by terminal-failure receipt SHA-256
`259c197755dd1cdd325c2a2995376736a61c3474ff4eae4f864e6a72c23aadea`.
Trainer-data projection job `135009` and independent audit job `135010` then
passed. The audit converted and tokenized all 29,434 rows under
`datasets==3.6.0`, `transformers==4.57.1`, and `trl==0.26.0`, observed
94,142,111 tokens with maximum sequence length 7,066, and successfully ran a
four-example batch through the pinned upstream custom collator. The trainer
field sequence digest is
`6e89aa72cab0c23513c9cbb578412a65c159e8dd6052b06d287c4234aa545c35`;
the manifest and audit hashes are
`305d1b60260efb2ec3faca92fc8c4177dce82a6091662937dcf9e94c63b47978`
and `9c93c6e4ced4196a6a2f89fbbf76675e5af7a59b6a69e6f32fd5b72c1eba8203`.
Only a new hash-bound one-step retry is authorized; 100-step training remains
blocked.

## Third one-step training incident

Retry job `135015` preserved the audited trainer projection and progressed
through all-rank model loading into vLLM/TorchInductor profile compilation. It
then failed before optimizer creation because vLLM and Triton resolved compile
caches under the quota-full login home (`/home/compute/hiqbal`). No optimizer
step, checkpoint, in-job pass gate, or OPD result was created. The failure is
sealed by terminal-failure receipt SHA-256
`c7b871ed33961faba68f5cc659b7ddd18d3fad5779267d4d12d325392bbcc09a`.

Retry 3 is deliberately narrower than a training change. It preserves the
model revision, upstream objective, audited ordered data, hardware, and exact
one-step recipe. It only binds XDG, vLLM, TorchInductor, Triton, CUDA, Torch,
extension, and temporary caches to distinct directories beneath a per-Slurm-job
EIT scratch namespace. The training process validates and records every
resolved directory, rejects paths under the login home, and performs a write
probe before launching the trainer. This retry still cannot release 100-step
training without both the original in-job parameter-update gate and an
independent terminal audit.

## Fourth one-step training incident

Cache-corrected job `135079` passed every per-job EIT cache check, loaded the
model on all four A6000 ranks, generated the first four student continuations
(3,356 tokens total), and entered the exact full-vocabulary
`KL(teacher || student)` calculation. It then exhausted 48 GB device memory
while materializing the full-vocabulary divergence, before the loss returned
to the trainer for backward. No backward pass, optimizer step, checkpoint,
in-job pass gate, or OPD result was created. The failure is sealed by
terminal-failure receipt SHA-256
`9ad342a0dbd86b5d152f7e39ca5279a5b0307ca98143ab9cb8e4ce836163b2d2`.

The upstream 1.7B launch geometry is documented for four H100s and uses
microbatch four, gradient accumulation two, effective batch 32, and 1,024
completion tokens. Retry 4 keeps four A6000s, effective batch 32, and the full
1,024-token cap. Its only training-geometry change is microbatch `4 -> 2` and
gradient accumulation `2 -> 4`. This reduces the per-rank full-vocabulary
activation peak without shortening student solutions or changing the number
of examples per optimizer update. It remains a one-step plumbing diagnostic
and cannot release full training without its original parameter-update and
terminal-audit gates.

### Passing 1,024-token update and its boundary

Retry 4, job `135083`, completed on one node with four A6000 GPUs in 7 minutes
33 seconds. It returned finite loss `0.0114`, finite positive gradient norm
`0.06874745339155197`, and checkpoint 1. Independent inspection found 196
LoRA-B tensors containing 36,678,164 nonzero parameters; the adapter SHA-256 is
`d2472343d712d0bd1f2bd4010b109552bf4fd1a836250ca8d11d81dd22d8e50c`.
The independent terminal receipt is
`one_step/job_135083/terminal_audit.json` under the EIT campaign artifact root,
with SHA-256
`9bcaea19dc41ca906d9b320586b091f08181e7511e86129b53342f24a33c46bf`.

This proves that the pinned upstream full-vocabulary objective can make a real
parameter update in the qualified geometry. It does not show task improvement,
and its 1,024-token cap is not qualified for the project's scientific arms.
The separate audited length calibration had already found 16/128 raw-student
samples capped at 2,048 tokens but 0/128 capped at 4,096, selecting 4,096 as
the smallest passing student cap. Therefore the pending 1,024-token 100-step
launch was withdrawn before submission rather than treating upstream
reproduction settings as a no-truncation result.

## Length-qualified one-step successor

`configs/opd_math/identifiability_v1_one_step_long4096.json` preregisters a new
one-step diagnostic at the calibrated 4,096-token cap. It uses four
A100-SXM4-80GB GPUs, microbatch one, and gradient accumulation eight, preserving
effective batch 32. Model, data order, seeds, optimizer, full-vocabulary
`KL(teacher || student)`, and pointwise divergence clip remain unchanged.

The copied upstream harness receives two audited, objective-preserving changes.
It pads a rank only to that microbatch's longest observed completion instead of
materializing cap-length masked padding, and it writes every trajectory to a
rank-specific exclusive JSON file with prompt, completion, exact token count,
cap flag, rank, and local sequence. The final partial trajectory buffer is
flushed. A pass requires all 32 expected trajectories, at most one trajectory
at the 4,096 cap, a real finite update, a complete hash-bound tree, and an
independent terminal audit. Any failure stops before 100-step training.

## Links

[[opd-teacher-evaluator-baseline-qualification-2026-07-22]] ·
[[opd-objective-family-expansion-2026-07-20]] · [[opd-distillation]] ·
[[opsd-self-distilled-reasoner]]
