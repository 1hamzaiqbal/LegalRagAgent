---
title: veRL On-Policy Distillation Trainer
type: source
tags: [opd, verl, implementation, k1, top-k, teacher-server]
created: 2026-07-17
updated: 2026-07-20
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

## 2026-07-20 fidelity re-audit

The EIT checkout was re-audited at clean upstream commit
`6a6242f3d8ec7d9f8b4936f4905144707d91fe3b`, including the canonical example,
the actual teacher manager, response masking, K1/K3/Top-k loss registry, and the
policy-gradient integration. The conclusion is narrower than “we reproduced
veRL,” but positive:

- **P0 — none found in the underlying sampled-token K1 mechanics.** Our student
  generates a fresh trajectory, the frozen teacher scores those exact token
  IDs, the student recomputes the same response-token log probabilities before
  any update, and the detached teacher-minus-student gap multiplies the current
  student log probability. With the local helper's gate disabled and clip
  removed, one optimizer update per fresh rollout has the same gradient as
  veRL's `k1 + use_policy_gradient` path at importance ratio one. Our EOS and
  padding masks include exactly the generated completion positions.
- **P1 — the scientific main arm is not canonical bare OPD.** It deliberately
  adds grouped task reward, clips the teacher-student gap, applies a detached
  positive-gap gate, and scales the dense auxiliary by `0.01`. It must remain
  named `task_rl_k1_gap`; an unqualified “veRL OPD reproduction” claim would be
  false. Even the diagnostic named `k1_bare` fixes a gap clip of `5`, while
  veRL's canonical launcher clips the K1 value at `10`; neither is the exact
  unclipped limit used for the clean K4-equivalent gradient statement. If a
  parameter-matched cross-implementation replication becomes necessary, add it
  as a separate diagnostic after the current preregistered campaign rather than
  changing the live arm.
- **P2 — expected implementation differences.** Our task term gives each
  completion equal weight after taking its mean token log probability; veRL's
  default policy loss is a global response-token mean. We do not retain
  rollout-time behavior log probabilities because every rollout receives one
  immediate update; veRL retains `old_log_probs` and can reuse a batch with
  clipped importance ratios. We also lack veRL's K3 and sparse teacher-Top-k
  objectives and query the external teacher sequentially rather than through
  Ray-managed replicas. These differences affect scaling or later ablations,
  not the validity of the current one-update sampled-K1 calculation.

Local anchors are `scripts/opd/opd_train.py:366-457,487-638`,
`scripts/opd/opd_loss.py:60-140`, and `scripts/opd/teacher_client.py:144-209`.
Upstream anchors at the pinned commit are
`verl/experimental/agent_loop/agent_loop.py:727-859,1006-1029`,
`verl/experimental/teacher_loop/teacher_manager.py:35-56,99-141`,
`verl/trainer/distillation/losses.py:230-296,364-399`, and
`verl/trainer/ppo/core_algos.py:1279-1362,2147-2204`.

The local analytic gradient, exact-token, mask, and trace-reconstruction tests
passed on 2026-07-20 (`scripts/opd/test_opd_loss.py` plus
`tests/test_opd_reward_loss.py` and `tests/test_teacher_client_token_ids.py`:
11 pytest cases plus the standalone loss checks). This is code-level evidence,
not a held-out task-improvement result.

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

## What veRL's multi-teacher “MOPD” does

The merged veRL MOPD path is **hard routing, not aggregation**. Each sample
carries one routing value (by default `data_source`); the trainer resolves that
value to exactly one teacher, sends the full student prompt+response token
sequence to that teacher, and uses that one teacher's token distribution. The
manager allocates a separate replica pool per teacher and rejects missing,
unknown, or duplicate routing keys. The canonical example routes GSM8K rows to
a text teacher and Geometry3K rows to a vision-language teacher. It does not
average teachers, choose the best teacher per token, or measure disagreement on
the same trajectory.

For a later LegalRagAgent extension, the smallest faithful wiring is therefore
a sealed routing manifest from sample source to teacher URL/model/checkpoint,
with one quality gate and tokenizer contract per teacher, plus the selected
teacher key and immutable identity in every sample trace. The preregistration,
prelaunch receipt, and held-out readout would need to bind and stratify all of
those identities. The failed M teacher must not be reused; every future teacher
would need its own positive gate. If the scientific question instead needs
teacher conflict or “which teacher should the student disobey?”, all teachers
must score the same student trajectory and the aggregation/arbitration rule
must be explicit—that is beyond veRL's existing routed MOPD scaffold and should
remain a later campaign.

Naming warning: arXiv `2605.12652` also uses **MOPD** for *Multi-Rollout*
On-Policy Distillation, which conditions the teacher on peer successes and
failures. Use “multi-teacher routed OPD” for the veRL feature to avoid
conflating the two methods.

Multi-teacher anchors are
`examples/on_policy_distillation_trainer/run_qwen3_8b_mopd_fsdp.sh:109-139`,
`verl/workers/config/distillation.py:289-320`,
`verl/experimental/teacher_loop/teacher_model.py:154-203`, and
`verl/experimental/teacher_loop/teacher_manager.py:99-141`.

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
  `6a6242f3d8ec7d9f8b4936f4905144707d91fe3b`, 2026-07-20.
- Official repository: https://github.com/verl-project/verl.
- Example directory:
  https://github.com/verl-project/verl/tree/6a6242f3d8ec7d9f8b4936f4905144707d91fe3b/examples/on_policy_distillation_trainer.
- Persistent EIT checkout:
  `/engrfs/project/jacobsn/hiqbal/literature/legalrag/repos/verl`.

## Links

[[ema-policy-gradient]] · [[mopd-multi-teacher]] · [[opd-math-source-transfer]] · [[sdar]] ·
[[opd-distillation]] · [[opsd-self-distilled-reasoner]]
