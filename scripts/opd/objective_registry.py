#!/usr/bin/env python3
"""Validated, hash-bound objective identities for the OPD math successor.

The registry defines objective semantics only.  It deliberately does not
authorize a scientific launch; that requires the later sealed campaign
preregistration, prerequisite identities, and one-step custody receipts.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
OBJECTIVE_REGISTRY_PATH = ROOT / "configs" / "opd_math" / "objective_registry.json"
REGISTRY_ID = "opd_math_objective_family_v1"
UPSTREAM_VERL_COMMIT = "6a6242f3d8ec7d9f8b4936f4905144707d91fe3b"
EXPECTED_OBJECTIVE_IDS = (
    "task_rl",
    "task_rl_k1_ungated_clip5",
    "task_rl_k1_ungated_unclipped",
    "task_rl_k1_gated_clip5_beta5",
    "k1_bare_verl_compatible_clip10",
    "k1_verl_upstream_clip10",
)
LOCAL_OBJECTIVE_IDS = frozenset(EXPECTED_OBJECTIVE_IDS[:-1])
UPSTREAM_OBJECTIVE_IDS = frozenset({EXPECTED_OBJECTIVE_IDS[-1]})
TASK_REWARD_OBJECTIVE_IDS = frozenset(
    {
        "task_rl",
        "task_rl_k1_ungated_clip5",
        "task_rl_k1_ungated_unclipped",
        "task_rl_k1_gated_clip5_beta5",
    }
)
K1_OBJECTIVE_IDS = frozenset(
    {
        "task_rl_k1_ungated_clip5",
        "task_rl_k1_ungated_unclipped",
        "task_rl_k1_gated_clip5_beta5",
        "k1_bare_verl_compatible_clip10",
        "k1_verl_upstream_clip10",
    }
)
GATED_K1_OBJECTIVE_IDS = frozenset({"task_rl_k1_gated_clip5_beta5"})
TASK_AND_K1_OBJECTIVE_IDS = TASK_REWARD_OBJECTIVE_IDS & K1_OBJECTIVE_IDS

_OBJECTIVE_KEYS = {
    "id",
    "implementation",
    "local_executable",
    "task_reward",
    "sampled_k1",
    "task_reward_coef",
    "k1_coef",
    "advantage_clip",
    "gap_gate_beta",
    "loss_aggregation",
    "objective_contract",
}
_LOSS_AGGREGATIONS = {
    "sequence_mean_task",
    "sequence_mean_task_plus_response_token_mean_k1",
    "response_token_mean",
}
_EXPECTED_OBJECTIVE_SEMANTICS: dict[str, dict[str, Any]] = {
    "task_rl": {
        "implementation": "local",
        "local_executable": True,
        "task_reward": True,
        "sampled_k1": False,
        "task_reward_coef": 1.0,
        "k1_coef": 0.0,
        "advantage_clip": None,
        "gap_gate_beta": None,
        "loss_aggregation": "sequence_mean_task",
        "objective_contract": "grouped_verifiable_math_task_reward_v2",
    },
    "task_rl_k1_ungated_clip5": {
        "implementation": "local",
        "local_executable": True,
        "task_reward": True,
        "sampled_k1": True,
        "task_reward_coef": 1.0,
        "k1_coef": 0.01,
        "advantage_clip": 5.0,
        "gap_gate_beta": None,
        "loss_aggregation": "sequence_mean_task_plus_response_token_mean_k1",
        "objective_contract": "grouped_task_reward_plus_clipped_ungated_sampled_k1_score_function_v1",
    },
    "task_rl_k1_ungated_unclipped": {
        "implementation": "local",
        "local_executable": True,
        "task_reward": True,
        "sampled_k1": True,
        "task_reward_coef": 1.0,
        "k1_coef": 0.01,
        "advantage_clip": None,
        "gap_gate_beta": None,
        "loss_aggregation": "sequence_mean_task_plus_response_token_mean_k1",
        "objective_contract": "grouped_task_reward_plus_unclipped_ungated_sampled_k1_score_function_v1",
    },
    "task_rl_k1_gated_clip5_beta5": {
        "implementation": "local",
        "local_executable": True,
        "task_reward": True,
        "sampled_k1": True,
        "task_reward_coef": 1.0,
        "k1_coef": 0.01,
        "advantage_clip": 5.0,
        "gap_gate_beta": 5.0,
        "loss_aggregation": "sequence_mean_task_plus_response_token_mean_k1",
        "objective_contract": "grouped_task_reward_plus_clipped_positive_gap_gated_sampled_k1_score_function_v1",
    },
    "k1_bare_verl_compatible_clip10": {
        "implementation": "local",
        "local_executable": True,
        "task_reward": False,
        "sampled_k1": True,
        "task_reward_coef": 0.0,
        "k1_coef": 1.0,
        "advantage_clip": 10.0,
        "gap_gate_beta": None,
        "loss_aggregation": "response_token_mean",
        "objective_contract": "bare_sampled_k1_policy_gradient_clip10_local_verl_compatible_v1",
    },
    "k1_verl_upstream_clip10": {
        "implementation": "upstream_verl",
        "local_executable": False,
        "task_reward": False,
        "sampled_k1": True,
        "task_reward_coef": 0.0,
        "k1_coef": 1.0,
        "advantage_clip": 10.0,
        "gap_gate_beta": None,
        "loss_aggregation": "response_token_mean",
        "objective_contract": "bare_sampled_k1_policy_gradient_clip10_upstream_verl_6a6242f_v1",
    },
}


def canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _finite_nonnegative(value: Any, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be finite")
    result = float(value)
    if result < 0:
        raise ValueError(f"{label} must be nonnegative")
    return result


def _optional_positive(value: Any, label: str) -> float | None:
    if value is None:
        return None
    result = _finite_nonnegative(value, label)
    if result <= 0:
        raise ValueError(f"{label} must be positive when present")
    return result


def _validate_objective(raw: Any, position: int) -> dict[str, Any]:
    label = f"objective[{position}]"
    if not isinstance(raw, dict) or set(raw) != _OBJECTIVE_KEYS:
        raise ValueError(f"{label} does not have the exact registry schema")
    objective = dict(raw)
    objective_id = objective["id"]
    if not isinstance(objective_id, str) or not objective_id:
        raise ValueError(f"{label} has an invalid id")
    if objective["implementation"] not in {"local", "upstream_verl"}:
        raise ValueError(f"{label} has an unsupported implementation")
    if type(objective["local_executable"]) is not bool:
        raise ValueError(f"{label} local_executable must be boolean")
    if type(objective["task_reward"]) is not bool or type(objective["sampled_k1"]) is not bool:
        raise ValueError(f"{label} task_reward and sampled_k1 must be boolean")
    task_coef = _finite_nonnegative(objective["task_reward_coef"], f"{label} task_reward_coef")
    k1_coef = _finite_nonnegative(objective["k1_coef"], f"{label} k1_coef")
    clip = _optional_positive(objective["advantage_clip"], f"{label} advantage_clip")
    gate = _optional_positive(objective["gap_gate_beta"], f"{label} gap_gate_beta")
    if objective["task_reward"] != (task_coef > 0):
        raise ValueError(f"{label} task-reward flag and coefficient disagree")
    if objective["sampled_k1"] != (k1_coef > 0):
        raise ValueError(f"{label} sampled-K1 flag and coefficient disagree")
    if not objective["sampled_k1"] and (clip is not None or gate is not None):
        raise ValueError(f"{label} configures K1 controls without sampled K1")
    if gate is not None and objective_id not in GATED_K1_OBJECTIVE_IDS:
        raise ValueError(f"{label} has a gate outside the registered gated objective")
    if objective["loss_aggregation"] not in _LOSS_AGGREGATIONS:
        raise ValueError(f"{label} has an unsupported loss aggregation")
    if not isinstance(objective["objective_contract"], str) or not objective["objective_contract"]:
        raise ValueError(f"{label} lacks an objective contract")
    objective["task_reward_coef"] = task_coef
    objective["k1_coef"] = k1_coef
    objective["advantage_clip"] = clip
    objective["gap_gate_beta"] = gate
    return objective


def validate_objective_registry(payload: Mapping[str, Any]) -> dict[str, Any]:
    expected_top_keys = {
        "schema_version",
        "registry_id",
        "status",
        "registry_alone_authorizes_scientific_launch",
        "upstream_verl_commit",
        "objectives",
    }
    if set(payload) != expected_top_keys:
        raise ValueError("objective registry does not have the exact top-level schema")
    if payload.get("schema_version") != 1 or payload.get("registry_id") != REGISTRY_ID:
        raise ValueError("objective registry identity is unsupported")
    if payload.get("status") != "implementation_only_not_launch_authorized":
        raise ValueError("objective registry status is not implementation-only")
    if payload.get("registry_alone_authorizes_scientific_launch") is not False:
        raise ValueError("objective registry must not authorize scientific launch")
    if payload.get("upstream_verl_commit") != UPSTREAM_VERL_COMMIT:
        raise ValueError("objective registry veRL commit drifted")
    raw_objectives = payload.get("objectives")
    if not isinstance(raw_objectives, list):
        raise ValueError("objective registry objectives must be a list")
    objectives = [
        _validate_objective(raw, position)
        for position, raw in enumerate(raw_objectives)
    ]
    ids = tuple(objective["id"] for objective in objectives)
    if ids != EXPECTED_OBJECTIVE_IDS:
        raise ValueError("objective registry IDs or ordering drifted")
    for objective in objectives:
        objective_id = objective["id"]
        observed_semantics = {
            key: value for key, value in objective.items() if key != "id"
        }
        if observed_semantics != _EXPECTED_OBJECTIVE_SEMANTICS[objective_id]:
            raise ValueError(f"objective {objective_id} semantics drifted")
        if objective_id in LOCAL_OBJECTIVE_IDS:
            if objective["implementation"] != "local" or objective["local_executable"] is not True:
                raise ValueError(f"local objective {objective_id} has invalid execution routing")
        else:
            if objective["implementation"] != "upstream_verl" or objective["local_executable"] is not False:
                raise ValueError(f"upstream objective {objective_id} has invalid execution routing")
    normalized = dict(payload)
    normalized["objectives"] = objectives
    return normalized


def load_objective_registry(path: str | Path = OBJECTIVE_REGISTRY_PATH) -> dict[str, Any]:
    registry_path = Path(path).resolve()
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("objective registry must contain a JSON object")
    registry = validate_objective_registry(payload)
    registry["path"] = str(registry_path)
    registry["sha256"] = hashlib.sha256(registry_path.read_bytes()).hexdigest()
    registry["canonical_sha256"] = canonical_json_sha256(payload)
    return registry


def resolve_objective(
    objective_id: str,
    *,
    registry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    active = load_objective_registry() if registry is None else dict(registry)
    objectives = active.get("objectives")
    if not isinstance(objectives, list):
        raise ValueError("validated objective registry lacks objectives")
    for objective in objectives:
        if isinstance(objective, dict) and objective.get("id") == objective_id:
            return dict(objective)
    raise ValueError(f"unknown registered objective: {objective_id}")
