#!/usr/bin/env python3
"""Extract a hash-bound real-model K1 fidelity fixture from a one-step run."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any

try:
    from .objective_family_inputs import sha256_file, sha256_tree
    from .objective_registry import load_objective_registry
except ImportError:
    from objective_family_inputs import sha256_file, sha256_tree  # type: ignore
    from objective_registry import load_objective_registry  # type: ignore


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ID = "real_model_rollout_k1_v1"
EXPECTED_OBJECTIVE = "k1_bare_verl_compatible_clip10"
HEX40 = re.compile(r"^[0-9a-f]{40}$")


def _json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("sample trace must be a regular file")
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"sample trace row {number} is not an object")
        rows.append(value)
    return rows


def _finite_list(value: Any, length: int, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{label} length drifted")
    result = []
    for item in value:
        if type(item) not in (int, float) or not math.isfinite(float(item)):
            raise ValueError(f"{label} contains a nonfinite value")
        result.append(float(item))
    return result


def build_fixture(run_root: Path) -> dict[str, Any]:
    run_root = run_root.resolve()
    trace_root = run_root / "traces"
    run_path = trace_root / "run_manifest.json"
    completion_path = trace_root / "completion_manifest.json"
    samples_path = trace_root / "samples.jsonl"
    run = _json(run_path, "run manifest")
    completion = _json(completion_path, "completion manifest")
    rows = _jsonl(samples_path)
    registry = load_objective_registry()
    objective = run.get("objective_registry") or {}
    objective_spec = objective.get("objective") or {}
    binding = run.get("binding") or {}
    if (
        run.get("objective") != "k1_bare"
        or objective_spec.get("id") != EXPECTED_OBJECTIVE
        or objective.get("registry_sha256") != registry["sha256"]
        or binding.get("objective_family_diagnostic") is not True
        or run.get("optimizer_steps_planned") != 1
        or run.get("micro_prompts_per_step") != 1
        or run.get("planned_rollout_samples") != 4
    ):
        raise ValueError("run is not the registered one-step bare-K1 diagnostic")
    commit = run.get("git_commit")
    if not isinstance(commit, str) or HEX40.fullmatch(commit) is None:
        raise ValueError("run lacks an immutable Git identity")
    if (
        completion.get("objective_family_diagnostic") is not True
        or completion.get("intended_scientific_run") is not False
        or completion.get("custody_required") is not True
        or completion.get("optimizer_steps_completed") != 1
        or completion.get("rollout_samples") != 4
        or completion.get("realized_training_geometry_observed") is not True
        or completion.get("finite_nonzero_gradient_observed") is not True
        or completion.get("parameter_update_observed") is not True
        or completion.get("clean_stable_code") is not True
        or completion.get("stable_environment_end") is not True
        or completion.get("live_local_server_process_binding_validated") is not True
        or completion.get("training_artifact_eligible_for_held_out_evaluation") is not False
        or completion.get("scientific_use_allowed") is not False
    ):
        raise ValueError("bare-K1 diagnostic did not complete full custody")
    if run.get("completion") != completion:
        raise ValueError("run/completion manifests disagree")
    if len(rows) != 4:
        raise ValueError("real-model fixture requires exactly four generated samples")

    max_width = max(len(row.get("completion_token_ids") or []) for row in rows)
    samples = []
    for index, row in enumerate(rows):
        completion_ids = row.get("completion_token_ids")
        prompt_ids = row.get("prompt_token_ids")
        if (
            row.get("schema_version") != 3
            or row.get("step") != 1
            or row.get("group_id") != 0
            or row.get("sample_idx") != index
            or not isinstance(prompt_ids, list)
            or not prompt_ids
            or not isinstance(completion_ids, list)
            or not completion_ids
            or any(type(token) is not int or token < 0 for token in prompt_ids + completion_ids)
        ):
            raise ValueError(f"sample trace identity drifted at index {index}")
        width = len(completion_ids)
        behavior = _finite_list(
            row.get("behavior_token_logprobs_on_student_trajectory"), width, "behavior"
        )
        current = _finite_list(row.get("student_token_logprobs"), width, "student")
        teacher = _finite_list(
            row.get("teacher_token_logprobs_on_student_trajectory"), width, "teacher"
        )
        padding = max_width - width
        samples.append(
            {
                "sample_id": f"{row.get('record_id')}:{index}",
                "prompt_token_ids": prompt_ids,
                "completion_token_ids": completion_ids + [0] * padding,
                "response_mask": [True] * width + [False] * padding,
                "behavior_logprobs": behavior + [0.0] * padding,
                "current_logprobs": current + [0.0] * padding,
                "teacher_logprobs": teacher + [0.0] * padding,
            }
        )
    final_adapter = completion.get("final_adapter")
    final_tree = completion.get("final_adapter_tree_sha256")
    if (
        not isinstance(final_adapter, str)
        or not Path(final_adapter).is_dir()
        or sha256_tree(final_adapter) != final_tree
    ):
        raise ValueError("run lacks a stable promoted diagnostic adapter")
    teacher = (run.get("gates") or {}).get("teacher_provenance") or {}
    teacher_checkpoint = run.get("teacher_checkpoint")
    teacher_tree = teacher.get("output_checkpoint_tree_sha256")
    if (
        not isinstance(teacher_checkpoint, str)
        or not Path(teacher_checkpoint).is_dir()
        or sha256_tree(
            teacher_checkpoint, exclude_relative_paths=("merge_provenance.json",)
        )
        != teacher_tree
    ):
        raise ValueError("run lacks a stable O-teacher checkpoint binding")
    return {
        "schema_version": 1,
        "fixture_id": FIXTURE_ID,
        "status": "real_model_stored_tensor_fidelity_only",
        "scientific_launch_authorized": False,
        "dtype": "float64",
        "samples": samples,
        "settings": {
            "loss_mode": "k1",
            "loss_max_clamp": 10.0,
            "policy_loss_mode": "vanilla",
            "clip_ratio_low": 0.2,
            "clip_ratio_high": 0.2,
            "dual_clip_ratio": 3.0,
            "loss_agg_mode": "token-mean",
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 0.001,
            "betas": [0.9, 0.999],
            "epsilon": 1e-08,
            "weight_decay": 0.0,
        },
        "tolerances": {
            "absolute": 1e-12,
            "relative": 1e-12,
            "gradient_cosine_minimum": 0.999999999999,
        },
        "provenance": {
            "source_samples": str(samples_path.resolve()),
            "source_samples_sha256": sha256_file(samples_path),
            "run_manifest": str(run_path.resolve()),
            "run_manifest_sha256": sha256_file(run_path),
            "completion_manifest": str(completion_path.resolve()),
            "completion_manifest_sha256": sha256_file(completion_path),
            "local_git_commit": commit,
            "objective_registry_sha256": registry["sha256"],
            "student": run.get("student"),
            "student_revision": run.get("student_revision"),
            "teacher_checkpoint": teacher_checkpoint,
            "teacher_checkpoint_tree_sha256": teacher_tree,
            "extractor_sha256": sha256_file(Path(__file__).resolve()),
            "behavior_logprobs_origin": "generation_transition_scores_before_update",
            "current_student_logprobs_origin": "pre_update_student_forward_on_generated_tokens",
            "teacher_logprobs_origin": "frozen_o_teacher_exact_generated_token_scores",
            "heldout_outcomes_inspected": False,
        },
    }


def write_new(path: Path, payload: dict[str, Any]) -> None:
    path = path.resolve()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite fixture: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(path, 0o444)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixture = build_fixture(args.run_root)
    write_new(args.output, fixture)
    print(json.dumps({"output": str(args.output.resolve()), "sha256": sha256_file(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
