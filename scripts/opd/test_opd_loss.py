#!/usr/bin/env python
"""CPU-only tests for OPD loss helpers."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from opd_loss import (
    kd_forward_loss,
    reverse_kl_score_function_loss,
    task_reward_policy_loss,
)


def assert_close(a, b, name, tol=1e-6):
    if abs(float(a) - float(b)) > tol:
        raise AssertionError(f"{name}: {a} != {b}")


def test_opd_step_moves_toward_teacher():
    theta = torch.tensor([-3.0], requires_grad=True)
    teacher = torch.tensor([-1.0])
    before_gap = abs(float(theta.detach() - teacher))
    loss = reverse_kl_score_function_loss(theta, teacher)
    loss.backward()
    with torch.no_grad():
        theta -= 0.1 * theta.grad
    after_gap = abs(float(theta.detach() - teacher))
    if not after_gap < before_gap:
        raise AssertionError(f"OPD step did not reduce gap: before={before_gap} after={after_gap}")
    print("PASS opd_step_moves_toward_teacher", flush=True)


def test_clamp_and_ratio_paths():
    student = torch.tensor([[-5.0, -0.5, -2.0]], requires_grad=True)
    teacher = torch.tensor([[5.0, -3.0, -1.0]])
    behavior = torch.tensor([[-5.5, -0.1, -2.3]])
    mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
    loss = reverse_kl_score_function_loss(
        student,
        teacher,
        mask,
        behavior_logprobs=behavior,
        advantage_clip=1.0,
        ratio_clip_eps=0.2,
    )
    if not torch.isfinite(loss):
        raise AssertionError("clamp/ratio loss is not finite")
    loss.backward()
    if not torch.isfinite(student.grad).all():
        raise AssertionError("clamp/ratio grad is not finite")
    print("PASS clamp_and_ratio_paths", flush=True)


def test_gap_gate_attenuates_negative_teacher_gap():
    student = torch.tensor([[-2.0, -2.0]], requires_grad=True)
    teacher = torch.tensor([[-1.0, -3.0]])
    loss = reverse_kl_score_function_loss(
        student,
        teacher,
        advantage_clip=None,
        gap_gate_beta=5.0,
    )
    loss.backward()
    positive_gap_grad = abs(float(student.grad[0, 0]))
    negative_gap_grad = abs(float(student.grad[0, 1]))
    if not positive_gap_grad > 100.0 * negative_gap_grad:
        raise AssertionError(
            "gap gate did not attenuate the negative teacher gap: "
            f"positive={positive_gap_grad} negative={negative_gap_grad}"
        )
    print("PASS gap_gate_attenuates_negative_teacher_gap", flush=True)


def test_gap_gate_rejects_nonpositive_beta():
    student = torch.tensor([-2.0], requires_grad=True)
    teacher = torch.tensor([-1.0])
    try:
        reverse_kl_score_function_loss(student, teacher, gap_gate_beta=0.0)
    except ValueError as exc:
        if "positive" not in str(exc):
            raise
    else:
        raise AssertionError("gap_gate_beta=0 should fail")
    print("PASS gap_gate_rejects_nonpositive_beta", flush=True)


def test_kd_forward_loss_manual_nll():
    logits = torch.tensor(
        [[[2.0, 0.0, -1.0], [0.5, 1.5, -0.5]], [[0.0, 1.0, 2.0], [3.0, 1.0, 0.0]]],
        dtype=torch.float32,
    )
    labels = torch.tensor([[0, 1], [2, -100]])
    loss = kd_forward_loss(logits, labels)
    lp = torch.log_softmax(logits, dim=-1)
    manual = -(lp[0, 0, 0] + lp[0, 1, 1] + lp[1, 0, 2]) / 3.0
    assert_close(loss.item(), manual.item(), "kd_forward_loss")
    print("PASS kd_forward_loss_manual_nll", flush=True)


def test_score_function_gradient_matches_exact_reverse_kl():
    student_logits = torch.tensor([0.3, -0.2], requires_grad=True)
    teacher_logits = torch.tensor([-0.1, 0.4])
    student_logp = torch.log_softmax(student_logits, dim=0)
    teacher_logp = torch.log_softmax(teacher_logits, dim=0)
    student_p = student_logp.exp()

    exact = (student_p * (student_logp - teacher_logp)).sum()
    exact_grad = torch.autograd.grad(exact, student_logits, retain_graph=True)[0]

    expected_surrogate = (
        student_p.detach()
        * (student_logp.detach() - teacher_logp.detach())
        * student_logp
    ).sum()
    surrogate_grad = torch.autograd.grad(expected_surrogate, student_logits, retain_graph=True)[0]
    if not torch.allclose(exact_grad, surrogate_grad, atol=1e-6, rtol=1e-6):
        raise AssertionError(f"score-function gradient mismatch: {exact_grad} vs {surrogate_grad}")

    direct_k1 = (student_p.detach() * (student_logp - teacher_logp.detach())).sum()
    direct_k1_grad = torch.autograd.grad(direct_k1, student_logits)[0]
    if not torch.allclose(direct_k1_grad, torch.zeros_like(direct_k1_grad), atol=1e-6):
        raise AssertionError(f"direct K1 autodiff should average to zero: {direct_k1_grad}")
    print("PASS score_function_gradient_matches_exact_reverse_kl", flush=True)


def test_grouped_task_reward_zero_and_mixed_groups():
    logps = torch.tensor([[-1.0], [-2.0], [-3.0], [-4.0]], requires_grad=True)
    rewards = torch.tensor([0.0, 1.0, 1.0, 1.0])
    groups = torch.tensor([0, 0, 1, 1])
    loss, advantages, informative = task_reward_policy_loss(logps, rewards, groups)
    assert informative.tolist() == [True, False]
    assert_close(advantages[2], 0.0, "equal_group_advantage_2")
    assert_close(advantages[3], 0.0, "equal_group_advantage_3")
    loss.backward()
    assert_close(logps.grad[2], 0.0, "equal_group_grad_2")
    assert_close(logps.grad[3], 0.0, "equal_group_grad_3")
    print("PASS grouped_task_reward_zero_and_mixed_groups", flush=True)


def main():
    torch.manual_seed(0)
    test_opd_step_moves_toward_teacher()
    test_clamp_and_ratio_paths()
    test_gap_gate_attenuates_negative_teacher_gap()
    test_gap_gate_rejects_nonpositive_beta()
    test_kd_forward_loss_manual_nll()
    test_score_function_gradient_matches_exact_reverse_kl()
    test_grouped_task_reward_zero_and_mixed_groups()
    print("PASS all_opd_loss_tests", flush=True)


if __name__ == "__main__":
    main()
