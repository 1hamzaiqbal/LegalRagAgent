#!/usr/bin/env python3
"""Validate the pinned upstream-veRL objective-family execution contract."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

try:
    from .objective_family_inputs import EXPECTED_STUDENT, EXPECTED_STUDENT_REVISION
    from .objective_registry import (
        UPSTREAM_VERL_COMMIT,
        load_objective_registry,
        resolve_objective,
    )
except ImportError:
    from objective_family_inputs import EXPECTED_STUDENT, EXPECTED_STUDENT_REVISION  # type: ignore
    from objective_registry import (  # type: ignore
        UPSTREAM_VERL_COMMIT,
        load_objective_registry,
        resolve_objective,
    )


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "configs/opd_math/objective_family_verl_plan.json"
PLAN_ID = "opd_math_objective_family_verl_v1"
OBJECTIVE_ID = "k1_verl_upstream_clip10"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_plan(payload: Mapping[str, Any]) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "plan_id",
        "status",
        "scientific_launch_authorized",
        "objective_id",
        "upstream_verl_commit",
        "student",
        "student_revision",
        "teacher",
        "allowed_sources",
        "allowed_seeds",
        "diagnostic_seed",
        "diagnostic_optimizer_steps",
        "scientific_optimizer_steps",
        "fixed_config",
        "claim_boundary",
        "stage_rules",
    }
    if set(payload) != expected_keys:
        raise ValueError("upstream veRL plan schema drifted")
    expected = {
        "schema_version": 1,
        "plan_id": PLAN_ID,
        "status": "implementation_contract_not_launch_authorized",
        "scientific_launch_authorized": False,
        "objective_id": OBJECTIVE_ID,
        "upstream_verl_commit": UPSTREAM_VERL_COMMIT,
        "student": EXPECTED_STUDENT,
        "student_revision": EXPECTED_STUDENT_REVISION,
        "teacher": "Qwen/Qwen3-8B",
        "allowed_sources": ["M", "O"],
        "allowed_seeds": [0, 1, 2],
        "diagnostic_seed": 0,
        "diagnostic_optimizer_steps": 1,
        "scientific_optimizer_steps": 100,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(f"upstream veRL plan {field} drifted")
    config = payload.get("fixed_config")
    if not isinstance(config, dict) or config != {
        "actor_gpus": 1,
        "teacher_gpus": 1,
        "actor_learning_rate": 1e-5,
        "actor_weight_decay": 0.01,
        "actor_betas": [0.9, 0.999],
        "actor_gradient_clip": 1.0,
        "actor_ppo_epochs": 1,
        "actor_ppo_mini_batch_size": 1,
        "actor_ppo_micro_batch_size_per_gpu": 4,
        "actor_use_dynamic_batch_size": False,
        "actor_use_remove_padding": True,
        "actor_gradient_checkpointing": True,
        "actor_use_torch_compile": True,
        "actor_loss_aggregation": "token-mean",
        "actor_clip_ratio_low": 0.2,
        "actor_clip_ratio_high": 0.2,
        "actor_dual_clip_ratio": 3.0,
        "actor_lora_rank": 32,
        "actor_lora_alpha": 64,
        "actor_lora_targets": [
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
        ],
        "train_prompt_batch_size": 1,
        "filter_overlong_prompts": True,
        "prompt_truncation": "error",
        "rollouts_per_prompt": 4,
        "max_prompt_tokens": 1536,
        "max_response_tokens": 512,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "thinking": False,
        "data_shuffle": False,
        "rollout_gpu_memory_utilization": 0.4,
        "rollout_max_model_length": 2049,
        "rollout_max_num_batched_tokens": 8192,
        "distillation_loss_mode": "k1",
        "distillation_use_policy_gradient": True,
        "distillation_use_task_rewards": False,
        "distillation_loss_max_clamp": 10.0,
        "distillation_log_prob_min_clamp": -10.0,
        "distillation_policy_loss_mode": "vanilla",
        "task_reward_function": "constant_zero_unused",
        "teacher_tensor_parallel_size": 1,
        "teacher_gpu_memory_utilization": 0.55,
        "teacher_max_model_length": 2049,
        "rollout_tensor_parallel_size": 1,
        "checkpoint_save_lora_only": True,
        "validation_enabled": False,
        "logger": "console",
    }:
        raise ValueError("upstream veRL fixed recipe drifted")
    if payload.get("stage_rules") != {
        "plan_alone_authorizes_launch": False,
        "same_initial_adapter_as_local_arm_required": True,
        "same_prompt_plan_prefix_as_local_arm_required": True,
        "fresh_o_teacher_required": True,
        "m_teacher_prohibited": True,
        "full_custody_diagnostic_required": True,
        "sealed_preregistration_required_for_science": True,
    }:
        raise ValueError("upstream veRL stage rules drifted")
    registry = load_objective_registry()
    objective = resolve_objective(OBJECTIVE_ID, registry=registry)
    if (
        not isinstance(objective, dict)
        or objective.get("implementation") != "upstream_verl"
        or objective.get("local_executable") is not False
        or objective.get("sampled_k1") is not True
    ):
        raise ValueError("upstream veRL registry identity drifted")
    return {"payload": dict(payload), "registry": registry}


def load_plan() -> dict[str, Any]:
    payload = json.loads(PLAN.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("upstream veRL plan must be a JSON object")
    result = validate_plan(payload)
    result.update({"path": str(PLAN.resolve()), "sha256": sha256_file(PLAN)})
    return result
