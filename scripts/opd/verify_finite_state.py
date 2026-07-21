#!/usr/bin/env python3
"""Exercise the fail-closed finite-state boundary without loading a model."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

import torch

try:
    from .objective_registry import load_objective_registry
    from .opd_train import (
        objective_loss_from_logprobs,
        optimizer_state_signature,
        parameter_update_l2,
        trainable_parameter_signature,
        trainable_parameter_snapshot,
    )
except ImportError:
    from objective_registry import load_objective_registry  # type: ignore
    from opd_train import (  # type: ignore
        objective_loss_from_logprobs,
        optimizer_state_signature,
        parameter_update_l2,
        trainable_parameter_signature,
        trainable_parameter_snapshot,
    )


ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expect_failure(label: str, fn, contains: str) -> dict[str, str]:
    try:
        fn()
    except (RuntimeError, ValueError) as exc:
        if contains not in str(exc):
            raise AssertionError(f"{label} failed for the wrong reason: {exc}") from exc
        return {"case": label, "status": "rejected", "exception": type(exc).__name__}
    raise AssertionError(f"{label} unexpectedly passed")


def _task_inputs():
    student = torch.tensor([[-2.0], [-1.0]], dtype=torch.float64, requires_grad=True)
    mask = torch.ones_like(student, dtype=torch.bool)
    rewards = torch.tensor([0.0, 1.0], dtype=torch.float64)
    groups = torch.zeros(2, dtype=torch.long)
    return student, mask, rewards, groups


def run() -> dict:
    registry = load_objective_registry()
    cases: list[dict] = []
    for value, label in (
        (float("nan"), "nan_student_logprob"),
        (float("inf"), "posinf_student_logprob"),
        (float("-inf"), "neginf_student_logprob"),
    ):
        student, mask, rewards, groups = _task_inputs()
        student = student.detach().clone()
        student[0, 0] = value
        cases.append(
            _expect_failure(
                label,
                lambda student=student: objective_loss_from_logprobs(
                    student,
                    None,
                    mask,
                    mode="task_rl",
                    task_reward_coef=1.0,
                    k1_coef=0.0,
                    advantage_clip=None,
                    gap_gate_beta=None,
                    rewards=rewards,
                    group_ids=groups,
                ),
                "student log-probabilities",
            )
        )

    student, mask, rewards, groups = _task_inputs()
    teacher = torch.tensor([[float("nan")], [-1.0]], dtype=torch.float64)
    cases.append(
        _expect_failure(
            "nan_teacher_logprob",
            lambda: objective_loss_from_logprobs(
                student,
                teacher,
                mask,
                mode="task_rl_k1_ungated_clip5",
                task_reward_coef=1.0,
                k1_coef=0.01,
                advantage_clip=5.0,
                gap_gate_beta=None,
                rewards=rewards,
                group_ids=groups,
            ),
            "teacher log-probabilities",
        )
    )
    behavior = student.detach().clone()
    behavior[0, 0] = float("inf")
    cases.append(
        _expect_failure(
            "inf_behavior_logprob",
            lambda: objective_loss_from_logprobs(
                student,
                torch.full_like(student, -1.0),
                mask,
                mode="k1_bare_verl_compatible_clip10",
                task_reward_coef=0.0,
                k1_coef=1.0,
                advantage_clip=10.0,
                gap_gate_beta=None,
                behavior_lps=behavior,
            ),
            "behavior log-probabilities",
        )
    )
    bad_rewards = rewards.clone()
    bad_rewards[0] = float("nan")
    cases.append(
        _expect_failure(
            "nan_task_reward",
            lambda: objective_loss_from_logprobs(
                student,
                None,
                mask,
                mode="task_rl",
                task_reward_coef=1.0,
                k1_coef=0.0,
                advantage_clip=None,
                gap_gate_beta=None,
                rewards=bad_rewards,
                group_ids=groups,
            ),
            "task rewards",
        )
    )

    model = torch.nn.Linear(2, 1, bias=False, dtype=torch.float64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    before = trainable_parameter_snapshot(model)
    loss = model(torch.ones((1, 2), dtype=torch.float64)).square().mean()
    loss.backward()
    gradient_norm = float(
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    )
    optimizer.step()
    update_norm = parameter_update_l2(model, before)
    optimizer_signature = optimizer_state_signature(optimizer)
    parameter_signature = trainable_parameter_signature(model)
    if not (math.isfinite(gradient_norm) and gradient_norm > 0 and update_norm > 0):
        raise AssertionError("finite optimizer case did not produce a real finite update")
    cases.append({"case": "finite_adamw_update", "status": "passed"})

    first_state = next(iter(optimizer.state.values()))
    state_tensor = next(value for value in first_state.values() if isinstance(value, torch.Tensor))
    state_tensor.fill_(float("nan"))
    cases.append(
        _expect_failure(
            "nan_optimizer_state",
            lambda: optimizer_state_signature(optimizer),
            "optimizer state tensor",
        )
    )
    with torch.no_grad():
        next(model.parameters()).fill_(float("inf"))
    cases.append(
        _expect_failure(
            "inf_parameter",
            lambda: trainable_parameter_signature(model),
            "trainable parameters",
        )
    )

    gradient_model = torch.nn.Linear(1, 1, bias=False)
    next(gradient_model.parameters()).grad = torch.full_like(
        next(gradient_model.parameters()), float("nan")
    )
    cases.append(
        _expect_failure(
            "nan_gradient",
            lambda: torch.nn.utils.clip_grad_norm_(
                gradient_model.parameters(), 1.0, error_if_nonfinite=True
            ),
            "non-finite",
        )
    )

    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()
    git_status = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    if git_status.strip():
        raise ValueError("finite-state verification requires a clean Git checkout")
    return {
        "schema_version": 1,
        "check_id": "opd_objective_finite_state_v1",
        "status": "passed",
        "git_commit": git_commit,
        "git_worktree_clean": True,
        "objective_registry_sha256": registry["sha256"],
        "script_sha256": sha256_file(Path(__file__)),
        "torch_version": torch.__version__,
        "dtype": "float64",
        "cases": cases,
        "finite_case": {
            "gradient_norm_before_clip": gradient_norm,
            "parameter_update_l2": update_norm,
            "optimizer_state_signature": optimizer_signature,
            "parameter_signature": parameter_signature,
        },
        "scientific_launch_authorized": False,
        "claim_boundary": "Finite-state rejection and one update only; no task-performance claim.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing to overwrite finite-state receipt: {args.output}")
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(args.output, 0o444)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
