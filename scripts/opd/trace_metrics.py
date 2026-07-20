#!/usr/bin/env python3
"""Pure-Python reconstruction of student training step metrics.

Scientific trace rows serialize the exact student token log-probabilities,
teacher log-probabilities on the same student trajectory, rewards, and group
IDs.  Those values are sufficient to reconstruct every scalar loss diagnostic
except the parameter-space gradient norm.  Keeping the reconstruction free of
Torch makes it an independent arithmetic audit of the recorded GPU reductions.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable, Mapping


TASK_REWARD_MODES = {"task_rl", "task_rl_k1_gap"}
K1_MODES = {"opd", "opd_gated", "k1_bare", "k1_gap_only", "task_rl_k1_gap"}
GATED_K1_MODES = {"opd_gated", "k1_gap_only", "task_rl_k1_gap"}
REWARD_ADVANTAGE_EPS = 1e-6
STEP_METRIC_ABS_TOLERANCE = 1e-5
STEP_METRIC_REL_TOLERANCE = 1e-6


def _finite_number(value: Any, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _token_values(row: Mapping[str, Any], field: str, label: str) -> list[float]:
    raw = row.get(field)
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{label} lacks {field}")
    values = [_finite_number(value, f"{label} {field}") for value in raw]
    completion_ids = row.get("completion_token_ids")
    if not isinstance(completion_ids, list) or len(values) != len(completion_ids):
        raise ValueError(f"{label} {field} is not aligned to completion token IDs")
    return values


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def reconstruct_step_metrics(
    sample_rows: Iterable[Mapping[str, Any]],
    *,
    mode: str,
    task_reward_coef: float,
    k1_coef: float,
    gap_gate_beta: float,
    advantage_clip: float,
) -> dict[str, float | int | None]:
    """Reconstruct one step's recorded scalar metrics from its sample rows."""

    rows = [dict(row) for row in sample_rows]
    if not rows:
        raise ValueError("cannot reconstruct an empty training step")
    if mode not in TASK_REWARD_MODES | K1_MODES:
        raise ValueError(f"unsupported trace-metric mode: {mode}")
    task_reward_coef = _finite_number(task_reward_coef, "task_reward_coef")
    k1_coef = _finite_number(k1_coef, "k1_coef")
    gap_gate_beta = _finite_number(gap_gate_beta, "gap_gate_beta")
    advantage_clip = _finite_number(advantage_clip, "advantage_clip")
    if advantage_clip <= 0 or (mode in GATED_K1_MODES and gap_gate_beta <= 0):
        raise ValueError("gap-gate and advantage-clip coefficients must be positive")

    student_by_sample: list[list[float]] = []
    teacher_by_sample: list[list[float] | None] = []
    groups: dict[int, list[int]] = defaultdict(list)
    rewards: list[float] = []
    for index, row in enumerate(rows):
        label = f"sample {index}"
        student = _token_values(row, "student_token_logprobs", label)
        student_by_sample.append(student)
        group_id = row.get("group_id")
        if type(group_id) is not int:
            raise ValueError(f"{label} lacks an integer group_id")
        groups[int(group_id)].append(index)
        if mode in TASK_REWARD_MODES:
            reward = _finite_number(row.get("reward"), f"{label} reward")
            rewards.append(reward)
        elif row.get("reward") is not None:
            raise ValueError(f"{label} unexpectedly has task reward in mode {mode}")
        if mode in K1_MODES:
            teacher = _token_values(
                row, "teacher_token_logprobs_on_student_trajectory", label
            )
            if len(teacher) != len(student):
                raise ValueError(f"{label} student/teacher token arrays differ in length")
            teacher_by_sample.append(teacher)
        else:
            if row.get("teacher_token_logprobs_on_student_trajectory") is not None:
                raise ValueError(f"{label} unexpectedly has teacher token values")
            teacher_by_sample.append(None)

    task_loss = 0.0
    reward_mean: float | None = None
    informative_group_fraction: float | None = None
    if mode in TASK_REWARD_MODES:
        advantages = [0.0] * len(rows)
        informative_groups = 0
        for group_id in sorted(groups):
            indices = groups[group_id]
            if len(indices) < 2:
                raise ValueError(f"reward group {group_id} has fewer than two samples")
            values = [rewards[index] for index in indices]
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            std = math.sqrt(variance)
            if std > REWARD_ADVANTAGE_EPS:
                informative_groups += 1
                for index in indices:
                    advantages[index] = (
                        rewards[index] - mean
                    ) / (std + REWARD_ADVANTAGE_EPS)
        sequence_means = [sum(values) / len(values) for values in student_by_sample]
        task_loss = -sum(
            advantage * sequence_mean
            for advantage, sequence_mean in zip(
                advantages, sequence_means, strict=True
            )
        ) / len(rows)
        reward_mean = sum(rewards) / len(rewards)
        informative_group_fraction = informative_groups / len(groups)

    tokens = sum(len(values) for values in student_by_sample)
    reverse_surrogate = 0.0
    sampled_k1: float | None = None
    gap_gate_mean: float | None = None
    positive_gap_fraction: float | None = None
    if mode in K1_MODES:
        k1_terms: list[float] = []
        surrogate_terms: list[float] = []
        gate_terms: list[float] = []
        positive = 0
        for student, teacher in zip(
            student_by_sample, teacher_by_sample, strict=True
        ):
            if teacher is None:  # Defensive; mode validation above requires it.
                raise ValueError("K1 trace lacks teacher token values")
            for student_logprob, teacher_logprob in zip(
                student, teacher, strict=True
            ):
                gap = teacher_logprob - student_logprob
                clipped_gap = min(max(gap, -advantage_clip), advantage_clip)
                gate = (
                    _sigmoid(gap_gate_beta * clipped_gap)
                    if mode in GATED_K1_MODES
                    else 1.0
                )
                k1_terms.append(student_logprob - teacher_logprob)
                surrogate_terms.append(-gate * clipped_gap * student_logprob)
                gate_terms.append(gate)
                positive += int(gap > 0)
        sampled_k1 = sum(k1_terms) / tokens
        reverse_surrogate = sum(surrogate_terms) / tokens
        gap_gate_mean = sum(gate_terms) / tokens
        positive_gap_fraction = positive / tokens

    if mode == "task_rl":
        total_loss = task_reward_coef * task_loss
    elif mode == "task_rl_k1_gap":
        total_loss = task_reward_coef * task_loss + k1_coef * reverse_surrogate
    else:
        total_loss = reverse_surrogate

    return {
        "task_loss": task_loss,
        "reverse_kl_score_function_surrogate": reverse_surrogate,
        "sampled_k1_estimate": sampled_k1,
        "gap_gate_mean": gap_gate_mean,
        "positive_gap_fraction": positive_gap_fraction,
        "reward_mean": reward_mean,
        "informative_group_fraction": informative_group_fraction,
        "tokens": tokens,
        "total_loss": total_loss,
    }


def validate_recorded_step_metrics(
    recorded: Mapping[str, Any],
    reconstructed: Mapping[str, float | int | None],
    *,
    label: str,
) -> None:
    """Reject stale step metrics while allowing bounded GPU reduction drift."""

    for field, expected in reconstructed.items():
        actual = recorded.get(field)
        if expected is None:
            if actual is not None:
                raise ValueError(f"{label} {field} must be null")
            continue
        actual_value = _finite_number(actual, f"{label} {field}")
        if field == "tokens":
            if type(actual) is not int or int(actual) != int(expected):
                raise ValueError(f"{label} {field} differs from trace reconstruction")
            continue
        if not math.isclose(
            actual_value,
            float(expected),
            rel_tol=STEP_METRIC_REL_TOLERANCE,
            abs_tol=STEP_METRIC_ABS_TOLERANCE,
        ):
            raise ValueError(f"{label} {field} differs from trace reconstruction")
