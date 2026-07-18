---
title: veRL On-Policy Distillation Trainer
type: source
tags: [opd, verl, implementation, k1, top-k, teacher-server]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://github.com/verl-project/verl/tree/main/examples/on_policy_distillation_trainer
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/repos/verl
authors: veRL contributors
year: 2026
---

# veRL on-policy distillation trainer

## TL;DR

Current veRL is a useful independent reference for our exact-token sampled
reverse-KL path, but it is not a drop-in implementation of our scientific arm.
Its canonical Qwen3 FSDP example performs bare K1-log-ratio policy-gradient
distillation with task reward disabled and one response per prompt. It uses
separate Ray actor and teacher GPU pools—8 actor plus 4 teacher GPUs in the
canonical script—so it cannot replace our deliberately lean single-GPU
teacher-server/student scaffold.

The right use is implementation custody and, later, a commit-pinned
cross-implementation comparison. The source-transfer question still requires
our held-out teacher gap, student task-support gate, task-RL baseline, source
provenance, non-thinking prompt contract, and model/tokenizer revision hashes.

## Sampled K1 and policy-gradient path

For each student-sampled token, veRL computes

\[
d_t=\log p_S(y_t\mid h_t)-\log p_T(y_t\mid h_t).
\]

With `loss_mode=k1` and `use_policy_gradient=true`, it detaches `d_t`, uses
`-d_t` as the advantage, and applies its clipped PPO policy loss. It explicitly
rejects K1 with direct backpropagation. This independently supports our naming:
the logged scalar is a K1 value estimate while the update is a score-function
surrogate, K4-equivalent only in the unclipped on-policy limit.

The generic `full` loss is unimplemented. K1, K2, K3, absolute-gap, and MSE
paths use the observed student token's two log probabilities rather than a
full-vocabulary distribution.

## veRL Top-k is a separate objective

`forward_kl_topk` asks the teacher for its top-k IDs and probabilities and
computes a sparse teacher-head contribution

\[
\sum_{i\in TopK(T)}p_T(i)[\log p_T(i)-\log p_S(i)].
\]

The truncated teacher mass is not renormalized, so the contribution can be
negative; veRL clamps it to zero. This is not EMA-PG's unbiased exact-head plus
sampled-tail estimator. veRL recommends K1 with policy gradient, and K3 or
top-k forward contribution with direct supervised backpropagation. We must not
collapse these different objectives into one “Top-k KL” label.

## Token and masking custody

The current implementation gets the important mechanics right:

1. student prompt and response token IDs are passed directly to the teacher;
2. the teacher scores prompt log probabilities at temperature one;
3. the first unscorable input position is discarded and logits are shifted;
4. only response-token positions survive the prompt/padding mask.

Upstream assumes, but does not explicitly verify, that returned teacher token
IDs equal student labels. It also does not pin model revisions or tokenizer
hashes. Our scaffold therefore keeps the stricter live tokenizer/server probe,
token-ID-keyed score requirement, and checkpoint provenance checks.

## Runtime and canonical configuration

The current integrated trainer allocates disjoint Ray pools, launches one or
more vLLM/SGLang teacher replicas, and routes student trajectories to those
replicas asynchronously. This is distinct from the obsolete documentation for
the removed `recipe/gkd` path.

The pinned FSDP Qwen3 launcher defaults to:

- Qwen3-8B student and Qwen3-32B teacher;
- 8 actor GPUs plus 4 teacher GPUs;
- one response per prompt, task reward off;
- 1,024 prompt and 2,048 response tokens;
- temperature 1, top-p 1, untruncated top-k; and
- K1 plus clipped policy gradient.

The Megatron example instead defaults to direct `forward_kl_topk` with
`topk=64`. A nominal 0.6B-to-1.7B example still asks for eight total GPUs.
The package's vLLM version declaration and current Docker image also differ;
any reproduction needs its own commit-pinned container rather than modifying
our lean TRL or serving environments.

## What to borrow and what not to borrow

Borrow:

- exact-token teacher scoring and response-only masks;
- separate K1 value and score-function-loss reporting;
- bounded gap/advantage diagnostics;
- dataset-source routing as a model for the M/O matrix; and
- later cross-implementation validation on adequate hardware.

Do not borrow:

- task-reward-off, `n=1` canonical settings as our main arm;
- unpinned model/data/tokenizer identity;
- the fail-open assumption that same-family token IDs align; or
- obsolete `recipe/gkd` commands.

Merely toggling `use_task_rewards=true` is insufficient because grouped math
reward requires multiple responses per prompt. veRL's canonical launcher is a
bare OPD diagnostic, not evidence for teacher usefulness.

## Version and custody

- Audited current main commit:
  `e003163181731412595257a72ec173071efb125f`, 2026-07-17.
- Official repository: https://github.com/verl-project/verl.
- Example directory:
  https://github.com/verl-project/verl/tree/e003163181731412595257a72ec173071efb125f/examples/on_policy_distillation_trainer.
- Persistent EIT checkout:
  `/engrfs/project/jacobsn/hiqbal/literature/legalrag/repos/verl`.

## Links

[[ema-policy-gradient]] · [[opd-math-source-transfer]] · [[sdar]] ·
[[opd-distillation]] · [[opsd-self-distilled-reasoner]]
