import torch
import pytest

from scripts.opd.opd_loss import (
    group_reward_advantages,
    reverse_kl_score_function_loss,
    sampled_k1_estimate,
    task_reward_policy_loss,
)
from scripts.opd.opd_train import trainable_parameter_signature
from scripts.opd.trace_metrics import reconstruct_step_metrics


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


@pytest.mark.parametrize("mode", ["task_rl", "task_rl_k1_gap"])
def test_pure_python_trace_metrics_match_executed_torch_objective(mode):
    student = torch.tensor(
        [
            [-2.0, -1.0, -0.5],
            [-1.5, -3.0, 0.0],
            [-0.25, 0.0, 0.0],
            [-4.0, -2.0, -1.0],
        ],
        dtype=torch.float32,
    )
    teacher = torch.tensor(
        [
            [4.0, -2.0, -0.4],
            [-2.0, 3.5, 0.0],
            [0.25, 0.0, 0.0],
            [-5.0, -1.0, 5.0],
        ],
        dtype=torch.float32,
    )
    mask = torch.tensor(
        [
            [True, True, True],
            [True, True, False],
            [True, False, False],
            [True, True, True],
        ]
    )
    rewards = torch.tensor([1.0, 0.0, 1.0, 1.0])
    groups = torch.tensor([0, 0, 1, 1])
    task, _, informative = task_reward_policy_loss(student, rewards, groups, mask)
    rows = []
    for index in range(4):
        active = mask[index]
        student_values = student[index][active].tolist()
        teacher_values = teacher[index][active].tolist()
        rows.append(
            {
                "group_id": int(groups[index]),
                "completion_token_ids": list(range(1, len(student_values) + 1)),
                "student_token_logprobs": student_values,
                "teacher_token_logprobs_on_student_trajectory": (
                    teacher_values if mode == "task_rl_k1_gap" else None
                ),
                "reward": float(rewards[index]),
            }
        )
    reconstructed = reconstruct_step_metrics(
        rows,
        mode=mode,
        task_reward_coef=1.0,
        k1_coef=0.01,
        gap_gate_beta=5.0,
        advantage_clip=5.0,
    )

    assert reconstructed["task_loss"] == pytest.approx(float(task), abs=1e-5)
    assert reconstructed["reward_mean"] == pytest.approx(float(rewards.mean()))
    assert reconstructed["informative_group_fraction"] == pytest.approx(
        float(informative.float().mean())
    )
    assert reconstructed["tokens"] == int(mask.sum())
    if mode == "task_rl":
        assert reconstructed["total_loss"] == pytest.approx(float(task), abs=1e-5)
        assert reconstructed["sampled_k1_estimate"] is None
    else:
        reverse = reverse_kl_score_function_loss(
            student,
            teacher,
            mask,
            advantage_clip=5.0,
            ratio_clip_eps=None,
            gap_gate_beta=5.0,
        )
        k1 = sampled_k1_estimate(student, teacher, mask)
        assert reconstructed["reverse_kl_score_function_surrogate"] == pytest.approx(
            float(reverse), abs=1e-5
        )
        assert reconstructed["sampled_k1_estimate"] == pytest.approx(
            float(k1), abs=1e-5
        )
        assert reconstructed["total_loss"] == pytest.approx(
            float(task + 0.01 * reverse), abs=1e-5
        )
