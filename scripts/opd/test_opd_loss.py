#!/usr/bin/env python
"""CPU-only tests for OPD loss helpers."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from opd_loss import kd_forward_loss, opd_policy_loss


def assert_close(a, b, name, tol=1e-6):
    if abs(float(a) - float(b)) > tol:
        raise AssertionError(f"{name}: {a} != {b}")


def test_opd_step_moves_toward_teacher():
    theta = torch.tensor([-3.0], requires_grad=True)
    teacher = torch.tensor([-1.0])
    before_gap = abs(float(theta.detach() - teacher))
    loss = opd_policy_loss(theta, teacher)
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
    loss = opd_policy_loss(
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


def main():
    torch.manual_seed(0)
    test_opd_step_moves_toward_teacher()
    test_clamp_and_ratio_paths()
    test_kd_forward_loss_manual_nll()
    print("PASS all_opd_loss_tests", flush=True)


if __name__ == "__main__":
    main()
