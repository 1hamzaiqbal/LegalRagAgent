import copy
from argparse import Namespace

import pytest
import torch

from scripts.opd.objective_registry import (
    EXPECTED_OBJECTIVE_IDS,
    LOCAL_OBJECTIVE_IDS,
    REGISTRY_ID,
    UPSTREAM_VERL_COMMIT,
    load_objective_registry,
    validate_objective_registry,
)
from scripts.opd.opd_loss import (
    reverse_kl_score_function_loss,
    task_reward_policy_loss,
)
from scripts.opd.opd_train import (
    bind_registered_objective,
    objective_loss_from_logprobs,
    validate_run_contract,
)
from scripts.opd.trace_metrics import reconstruct_step_metrics


def _objective(registry, objective_id):
    return next(item for item in registry["objectives"] if item["id"] == objective_id)


def _fixed_inputs():
    student = torch.tensor(
        [
            [-2.0, -1.0, -0.5],
            [-1.5, -3.0, 0.0],
            [-0.25, 0.0, 0.0],
            [-4.0, -2.0, -1.0],
        ],
        dtype=torch.float32,
        requires_grad=True,
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
    return student, teacher, mask, rewards, groups


def _trace_rows(student, teacher, mask, rewards, groups, *, sampled_k1, task_reward):
    rows = []
    for index in range(student.shape[0]):
        selected = mask[index]
        student_values = student[index][selected].detach().tolist()
        teacher_values = teacher[index][selected].detach().tolist()
        rows.append(
            {
                "group_id": int(groups[index]),
                "completion_token_ids": list(range(1, len(student_values) + 1)),
                "student_token_logprobs": student_values,
                "teacher_token_logprobs_on_student_trajectory": (
                    teacher_values if sampled_k1 else None
                ),
                "reward": float(rewards[index]) if task_reward else None,
            }
        )
    return rows


def test_registry_is_exact_hash_bound_and_not_launch_authority():
    registry = load_objective_registry()
    assert registry["registry_id"] == REGISTRY_ID
    assert registry["upstream_verl_commit"] == UPSTREAM_VERL_COMMIT
    assert registry["registry_alone_authorizes_scientific_launch"] is False
    assert tuple(item["id"] for item in registry["objectives"]) == EXPECTED_OBJECTIVE_IDS
    assert len(registry["sha256"]) == 64
    assert len(registry["canonical_sha256"]) == 64


@pytest.mark.parametrize(
    ("objective_id", "field", "replacement"),
    [
        ("task_rl", "task_reward_coef", 0.5),
        ("task_rl_k1_ungated_clip5", "advantage_clip", 10.0),
        ("task_rl_k1_ungated_unclipped", "advantage_clip", 5.0),
        ("task_rl_k1_gated_clip5_beta5", "gap_gate_beta", 1.0),
        ("k1_bare_verl_compatible_clip10", "k1_coef", 0.01),
        ("k1_verl_upstream_clip10", "local_executable", True),
    ],
)
def test_registry_rejects_semantic_drift(objective_id, field, replacement):
    registry = load_objective_registry()
    payload = {
        key: copy.deepcopy(value)
        for key, value in registry.items()
        if key not in {"path", "sha256", "canonical_sha256"}
    }
    _objective(payload, objective_id)[field] = replacement
    with pytest.raises(ValueError, match="drifted|routing"):
        validate_objective_registry(payload)


def test_registry_binding_overrides_free_form_loss_flags_and_routes_upstream():
    args = Namespace(
        objective_id="task_rl_k1_gated_clip5_beta5",
        mode=None,
        task_reward_coef=99.0,
        k1_coef=99.0,
        advantage_clip=99.0,
        gap_gate_beta=99.0,
    )
    contract = bind_registered_objective(args)
    assert args.mode == "task_rl_k1_gated_clip5_beta5"
    assert args.task_reward_coef == 1.0
    assert args.k1_coef == 0.01
    assert args.advantage_clip == 5.0
    assert args.gap_gate_beta == 5.0
    assert contract["registry_alone_authorizes_scientific_launch"] is False

    upstream = Namespace(objective_id="k1_verl_upstream_clip10", mode=None)
    with pytest.raises(ValueError, match="pinned upstream veRL launcher"):
        bind_registered_objective(upstream)


@pytest.mark.parametrize("objective_id", sorted(LOCAL_OBJECTIVE_IDS))
def test_local_registered_objective_matches_manual_torch_and_trace_reconstruction(
    objective_id,
):
    registry = load_objective_registry()
    spec = _objective(registry, objective_id)
    student, teacher, mask, rewards, groups = _fixed_inputs()
    task_rewards = rewards if spec["task_reward"] else None
    task_groups = groups if spec["task_reward"] else None
    teacher_logprobs = teacher if spec["sampled_k1"] else None

    total, metrics = objective_loss_from_logprobs(
        student,
        teacher_logprobs,
        mask,
        mode=objective_id,
        task_reward_coef=spec["task_reward_coef"],
        k1_coef=spec["k1_coef"],
        advantage_clip=spec["advantage_clip"],
        gap_gate_beta=spec["gap_gate_beta"],
        rewards=task_rewards,
        group_ids=task_groups,
    )
    expected = student.sum() * 0.0
    if spec["task_reward"]:
        task_loss, _, _ = task_reward_policy_loss(student, rewards, groups, mask)
        expected = expected + spec["task_reward_coef"] * task_loss
    if spec["sampled_k1"]:
        reverse = reverse_kl_score_function_loss(
            student,
            teacher,
            mask,
            advantage_clip=spec["advantage_clip"],
            ratio_clip_eps=None,
            gap_gate_beta=spec["gap_gate_beta"],
        )
        expected = expected + spec["k1_coef"] * reverse
    assert torch.allclose(total, expected)

    rows = _trace_rows(
        student,
        teacher,
        mask,
        rewards,
        groups,
        sampled_k1=spec["sampled_k1"],
        task_reward=spec["task_reward"],
    )
    reconstructed = reconstruct_step_metrics(
        rows,
        mode=objective_id,
        task_reward_coef=spec["task_reward_coef"],
        k1_coef=spec["k1_coef"],
        advantage_clip=spec["advantage_clip"],
        gap_gate_beta=spec["gap_gate_beta"],
    )
    assert reconstructed["total_loss"] == pytest.approx(
        float(total.detach()), abs=1e-5
    )
    assert reconstructed["tokens"] == int(mask.sum())
    assert reconstructed["task_loss"] == pytest.approx(metrics["task_loss"], abs=1e-5)
    if spec["sampled_k1"]:
        assert reconstructed["reverse_kl_score_function_surrogate"] == pytest.approx(
            metrics["reverse_kl_score_function_surrogate"], abs=1e-5
        )


def test_unclipped_and_gated_objectives_have_distinct_gradients():
    rewards = torch.ones(2)
    groups = torch.zeros(2, dtype=torch.long)
    mask = torch.ones((2, 1), dtype=torch.bool)
    teacher = torch.tensor([[-10.0], [8.0]])

    def gradient(objective_id):
        student = torch.tensor([[-1.0], [-1.0]], requires_grad=True)
        spec = _objective(load_objective_registry(), objective_id)
        loss, _ = objective_loss_from_logprobs(
            student,
            teacher,
            mask,
            mode=objective_id,
            task_reward_coef=spec["task_reward_coef"],
            k1_coef=spec["k1_coef"],
            advantage_clip=spec["advantage_clip"],
            gap_gate_beta=spec["gap_gate_beta"],
            rewards=rewards,
            group_ids=groups,
        )
        loss.backward()
        return student.grad.detach().clone()

    clipped = gradient("task_rl_k1_ungated_clip5")
    unclipped = gradient("task_rl_k1_ungated_unclipped")
    gated = gradient("task_rl_k1_gated_clip5_beta5")
    assert not torch.allclose(clipped, unclipped)
    assert abs(float(gated[0, 0])) < abs(float(clipped[0, 0])) * 1e-6
    assert abs(float(gated[1, 0])) == pytest.approx(abs(float(clipped[1, 0])), rel=1e-5)


def test_registry_alone_cannot_authorize_scientific_training():
    args = Namespace(
        objective_id="task_rl",
        mode=None,
        task_reward_coef=1.0,
        k1_coef=0.0,
        advantage_clip=5.0,
        gap_gate_beta=5.0,
        steps=100,
        max_new_tokens=512,
        max_prompt_tokens=1536,
        lr=1e-5,
        grad_clip=1.0,
        lora=32,
        group_size=4,
        micro_prompts=1,
        min_informative_group_fraction=0.05,
        allow_ungated_smoke=False,
        teacher_connect_timeout=10.0,
        teacher_read_timeout=120.0,
        teacher_retries=3,
    )
    bind_registered_objective(args)
    with pytest.raises(ValueError, match="successor preregistration"):
        validate_run_contract(args, [])
