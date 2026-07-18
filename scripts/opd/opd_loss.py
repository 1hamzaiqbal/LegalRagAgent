#!/usr/bin/env python
"""Pure-torch OPD and KD losses.

Token alignment requirement: `student_logprobs` and `teacher_logprobs` must be
the same shape and must refer to the same completion token positions. Upstream
code must use a teacher/student tokenizer-family pair such as Qwen3-to-Qwen3 or
Llama-3.x-to-Llama-3.x.

OPD score-function objective:

  A_t = log p_teacher(y_t | x, y_<t) - stopgrad(log p_student(y_t | x, y_<t))
  L   = - mean_t A_t * log p_student(y_t | x, y_<t)

The detached multiplier contains the K1 log-ratio value. The loss itself is a
score-function surrogate: without clipping or gating, its expected gradient is
equivalent to the K4/r-trick reverse-KL gradient described by Zhang and Ba
(2026). It is *not* direct autodiff through K1; that gradient averages to zero.
`A_t` can be clamped for stability. An optional detached, SDAR-inspired
positive-gap gate

  g_t = sigmoid(beta * A_t)

can attenuate tokens where the privileged teacher assigns lower probability
than the student. Because the K1 advantage remains in the gradient multiplier,
this is not an exact reproduction of SDAR's auxiliary loss. This dense building
block does not replace task reward. For slightly stale samples, an optional detached
importance weight

  rho_t = exp(stopgrad(logp_current_t - logp_behavior_t))

is clipped to `[1 - eps, 1 + eps]` and multiplied into the token loss.
"""
from __future__ import annotations

import torch


def _check_same_shape(name: str, a: torch.Tensor, b: torch.Tensor) -> None:
    if a.shape != b.shape:
        raise ValueError(f"{name} shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")


def _float_mask(mask: torch.Tensor | None, ref: torch.Tensor) -> torch.Tensor:
    if mask is None:
        return torch.ones_like(ref, dtype=ref.dtype)
    if mask.shape != ref.shape:
        raise ValueError(f"mask shape mismatch: {tuple(mask.shape)} vs {tuple(ref.shape)}")
    return mask.to(device=ref.device, dtype=ref.dtype)


def masked_mean(values: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Mean over unmasked values, raising if the mask selects no tokens."""
    weights = _float_mask(mask, values)
    denom = weights.sum()
    if denom.detach().item() <= 0:
        raise ValueError("masked_mean received an empty mask")
    return (values * weights).sum() / denom


def reverse_kl_score_function_loss(
    student_logprobs: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    behavior_logprobs: torch.Tensor | None = None,
    advantage_clip: float | None = 5.0,
    ratio_clip_eps: float | None = 0.2,
    gap_gate_beta: float | None = None,
) -> torch.Tensor:
    """Score-function reverse-KL surrogate for aligned sampled tokens.

    With on-policy samples, no gap gate, and no advantage clipping, the
    expected gradient matches the K4/r-trick reverse-KL gradient. The scalar
    returned here is a surrogate loss, not a KL value estimate. Use
    :func:`sampled_k1_estimate` for the sampled K1 value.

    Args:
        student_logprobs: Current student log p for sampled completion tokens.
        teacher_logprobs: Teacher log p for the same tokens.
        mask: Optional bool/float mask for valid completion-token positions.
        behavior_logprobs: Optional log p from the policy that sampled the
            tokens. If provided, a detached importance ratio corrects stale
            samples.
        advantage_clip: If not None, clamp `A_t` to this absolute value.
        ratio_clip_eps: If stale-sample ratios are used, clip to
            `[1 - eps, 1 + eps]`. Set None to disable clipping.
        gap_gate_beta: If positive, multiply each token update by
            `sigmoid(beta * (teacher_logp - student_logp))`. This strongly
            attenuates negative teacher gaps. Set None to use bare OPD.
    """
    _check_same_shape("student/teacher", student_logprobs, teacher_logprobs)
    weights = _float_mask(mask, student_logprobs)

    advantage = teacher_logprobs - student_logprobs.detach()
    if advantage_clip is not None:
        advantage = torch.clamp(advantage, min=-advantage_clip, max=advantage_clip)

    gap_gate = 1.0
    if gap_gate_beta is not None:
        if gap_gate_beta <= 0:
            raise ValueError("gap_gate_beta must be positive when provided")
        gap_gate = torch.sigmoid(gap_gate_beta * advantage.detach())

    ratio = 1.0
    if behavior_logprobs is not None:
        _check_same_shape("student/behavior", student_logprobs, behavior_logprobs)
        ratio = torch.exp((student_logprobs.detach() - behavior_logprobs.detach()).clamp(-30.0, 30.0))
        if ratio_clip_eps is not None:
            ratio = torch.clamp(ratio, min=1.0 - ratio_clip_eps, max=1.0 + ratio_clip_eps)

    token_loss = -ratio * gap_gate * advantage.detach() * student_logprobs
    return masked_mean(token_loss, weights)


def opd_policy_loss(*args, **kwargs) -> torch.Tensor:
    """Compatibility alias for :func:`reverse_kl_score_function_loss`."""
    return reverse_kl_score_function_loss(*args, **kwargs)


def reverse_kl_estimate(
    student_logprobs: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Deprecated name for the sampled K1 estimate; not a full-vocabulary KL."""
    _check_same_shape("student/teacher", student_logprobs, teacher_logprobs)
    return masked_mean(student_logprobs.detach() - teacher_logprobs.detach(), mask)


def sampled_k1_estimate(
    student_logprobs: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Monte Carlo K1 value on student-sampled tokens.

    A finite batch estimate may be negative even though the exact reverse KL is
    nonnegative. It must not be reported as a full-vocabulary KL measurement.
    """
    return reverse_kl_estimate(student_logprobs, teacher_logprobs, mask)


def group_reward_advantages(
    rewards: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standardize detached rewards within each prompt group.

    Returns `(advantages, informative_group_mask)`, where the second tensor has
    one boolean per unique group and is true exactly when that group has reward
    variation. Population standard deviation avoids NaNs for small groups.
    """
    if rewards.ndim != 1 or group_ids.ndim != 1 or rewards.shape != group_ids.shape:
        raise ValueError(
            f"rewards/group_ids must be matching vectors, got {tuple(rewards.shape)} and {tuple(group_ids.shape)}"
        )
    unique = torch.unique(group_ids, sorted=True)
    advantages = torch.zeros_like(rewards, dtype=torch.float32)
    informative = torch.zeros(unique.shape, dtype=torch.bool, device=rewards.device)
    detached = rewards.detach().float()
    for j, group_id in enumerate(unique):
        selected = group_ids.eq(group_id)
        values = detached[selected]
        if values.numel() < 2:
            raise ValueError(f"reward group {int(group_id.item())} has fewer than two samples")
        mean = values.mean()
        std = values.std(unbiased=False)
        if std.item() > eps:
            advantages[selected] = (values - mean) / (std + eps)
            informative[j] = True
    return advantages, informative


def task_reward_policy_loss(
    student_logprobs: torch.Tensor,
    rewards: torch.Tensor,
    group_ids: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Grouped task-reward score-function loss using mean sequence logprob.

    All-equal reward groups intentionally contribute zero task gradient. The
    returned advantages and informative-group mask are detached diagnostics.
    """
    if student_logprobs.ndim != 2:
        raise ValueError(f"student_logprobs must be [batch,time], got {tuple(student_logprobs.shape)}")
    if rewards.shape != (student_logprobs.shape[0],):
        raise ValueError("one reward is required per sampled completion")
    if group_ids.shape != rewards.shape:
        raise ValueError("one group ID is required per sampled completion")
    weights = _float_mask(mask, student_logprobs)
    lengths = weights.sum(dim=1)
    if torch.any(lengths <= 0):
        raise ValueError("each completion must contain at least one scored token")
    sequence_mean_logp = (student_logprobs * weights).sum(dim=1) / lengths
    advantages, informative = group_reward_advantages(rewards, group_ids, eps=eps)
    loss = (-advantages.detach() * sequence_mean_logp).mean()
    return loss, advantages.detach(), informative.detach()


def kd_forward_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Supervised NLL for teacher-provided text.

    `logits` has shape `[batch, time, vocab]` and `labels` has shape
    `[batch, time]`. Positions with `ignore_index` or a false `mask` are
    excluded. This is the closed-teacher fallback when token-level teacher
    logprobs are unavailable.
    """
    if logits.ndim != 3:
        raise ValueError(f"logits must be [batch, time, vocab], got {tuple(logits.shape)}")
    if labels.shape != logits.shape[:2]:
        raise ValueError(f"labels shape {tuple(labels.shape)} does not match logits {tuple(logits.shape[:2])}")
    active = labels.ne(ignore_index)
    if mask is not None:
        if mask.shape != labels.shape:
            raise ValueError(f"mask shape mismatch: {tuple(mask.shape)} vs {tuple(labels.shape)}")
        active = active & mask.bool()

    safe_labels = labels.masked_fill(~active, 0)
    log_probs = torch.log_softmax(logits, dim=-1)
    target_log_probs = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    return -masked_mean(target_log_probs, active)
