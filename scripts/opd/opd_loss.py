#!/usr/bin/env python
"""Pure-torch OPD and KD losses.

Token alignment requirement: `student_logprobs` and `teacher_logprobs` must be
the same shape and must refer to the same completion token positions. Upstream
code must use a teacher/student tokenizer-family pair such as Qwen3-to-Qwen3 or
Llama-3.x-to-Llama-3.x.

OPD objective:

  A_t = log p_teacher(y_t | x, y_<t) - stopgrad(log p_student(y_t | x, y_<t))
  L   = - mean_t A_t * log p_student(y_t | x, y_<t)

This is the policy-gradient form of minimizing reverse KL, KL(student ||
teacher), over completions sampled from the student. `A_t` can be clamped for
stability. For slightly stale samples, an optional detached importance weight

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


def opd_policy_loss(
    student_logprobs: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    behavior_logprobs: torch.Tensor | None = None,
    advantage_clip: float | None = 5.0,
    ratio_clip_eps: float | None = 0.2,
) -> torch.Tensor:
    """Policy-gradient OPD loss for token logprob tensors.

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
    """
    _check_same_shape("student/teacher", student_logprobs, teacher_logprobs)
    weights = _float_mask(mask, student_logprobs)

    advantage = teacher_logprobs - student_logprobs.detach()
    if advantage_clip is not None:
        advantage = torch.clamp(advantage, min=-advantage_clip, max=advantage_clip)

    ratio = 1.0
    if behavior_logprobs is not None:
        _check_same_shape("student/behavior", student_logprobs, behavior_logprobs)
        ratio = torch.exp((student_logprobs.detach() - behavior_logprobs.detach()).clamp(-30.0, 30.0))
        if ratio_clip_eps is not None:
            ratio = torch.clamp(ratio, min=1.0 - ratio_clip_eps, max=1.0 + ratio_clip_eps)

    token_loss = -ratio * advantage.detach() * student_logprobs
    return masked_mean(token_loss, weights)


def reverse_kl_estimate(
    student_logprobs: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample estimate of mean `logp_student - logp_teacher` on sampled tokens."""
    _check_same_shape("student/teacher", student_logprobs, teacher_logprobs)
    return masked_mean(student_logprobs.detach() - teacher_logprobs.detach(), mask)


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
