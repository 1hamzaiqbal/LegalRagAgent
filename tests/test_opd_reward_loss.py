import torch
import pytest

from scripts.opd.opd_loss import (
    group_reward_advantages,
    reverse_kl_score_function_loss,
    task_reward_policy_loss,
)
from scripts.opd.opd_train import trainable_parameter_signature


def test_reward_advantages_are_isolated_by_prompt_group():
    rewards = torch.tensor([0.0, 1.0, 1.0, 1.0])
    groups = torch.tensor([0, 0, 1, 1])
    advantages, informative = group_reward_advantages(rewards, groups)
    assert advantages[0] < 0 < advantages[1]
    assert torch.equal(advantages[2:], torch.zeros(2))
    assert informative.tolist() == [True, False]


def test_all_equal_rewards_have_exactly_zero_task_gradient():
    logps = torch.tensor([[-1.0, -2.0], [-0.5, -0.75]], requires_grad=True)
    rewards = torch.ones(2)
    groups = torch.zeros(2, dtype=torch.long)
    loss, _, informative = task_reward_policy_loss(logps, rewards, groups)
    loss.backward()
    assert not informative.any()
    assert torch.equal(logps.grad, torch.zeros_like(logps.grad))


def test_combined_loss_decomposes_exactly():
    student = torch.tensor([[-2.0], [-1.0]], requires_grad=True)
    teacher = torch.tensor([[-1.0], [-0.5]])
    mask = torch.ones_like(student, dtype=torch.bool)
    rewards = torch.tensor([0.0, 1.0])
    groups = torch.zeros(2, dtype=torch.long)
    task, _, _ = task_reward_policy_loss(student, rewards, groups, mask)
    reverse_kl = reverse_kl_score_function_loss(student, teacher, mask, gap_gate_beta=5.0)
    combined = task + 0.01 * reverse_kl
    assert torch.allclose(combined, task + 0.01 * reverse_kl)
    combined.backward()
    assert torch.isfinite(student.grad).all()


def test_parameter_signature_fails_closed_on_nonfinite_weights():
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(float("nan"))
    with pytest.raises(RuntimeError, match="non-finite"):
        trainable_parameter_signature(model)
