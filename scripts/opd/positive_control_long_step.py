#!/usr/bin/env python3
"""Run and audit the length-qualified one-step OPSD diagnostic."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

try:
    from scripts.opd.positive_control_one_step import (
        audit_training,
        hash_tree,
        load_object,
        require_hash,
        sha256,
        training_command,
        validate_prerequisites as validate_upstream_one_step_prerequisites,
        validate_runtime_cache_environment,
        write_exclusive,
    )
except ModuleNotFoundError:  # Direct execution by absolute path on EIT.
    from positive_control_one_step import (  # type: ignore[no-redef]
        audit_training,
        hash_tree,
        load_object,
        require_hash,
        sha256,
        training_command,
        validate_prerequisites as validate_upstream_one_step_prerequisites,
        validate_runtime_cache_environment,
        write_exclusive,
    )


RUN_CONFIG = "one_step_length_qualified_4096"
ALLOWED_RECIPE_CHANGES = {
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "max_completion_tokens",
    "run_config",
}


def validate_prerequisites(config: dict, repository_commit: str) -> dict:
    if config.get("status") != "preregistered_diagnostic_only_100_step_training_blocked":
        raise RuntimeError("long-context one-step diagnostic is not preregistered")
    if config.get("stage_id") != "one_step_length_qualified_update_diagnostic":
        raise RuntimeError("unexpected long-context diagnostic stage")
    if not all(config.get("immutable_boundaries", {}).values()):
        raise RuntimeError("a long-context immutable boundary is not enforced")

    prerequisite = config["release_prerequisite"]
    records = {}
    for key in (
        "upstream_one_step_preregistration",
        "upstream_one_step_in_job_gate",
        "upstream_one_step_terminal_audit",
        "student_length_calibration",
    ):
        path = Path(prerequisite[key]).resolve()
        records[key] = {
            "path": str(path),
            "sha256": require_hash(path, prerequisite[f"{key}_sha256"], key),
        }

    one_step_config = load_object(
        Path(prerequisite["upstream_one_step_preregistration"]),
        "upstream one-step preregistration",
    )
    one_step_custody = validate_upstream_one_step_prerequisites(
        one_step_config, prerequisite["upstream_one_step_repository_commit"]
    )
    if one_step_config.get("retry", {}).get("attempt_id") != "microbatch_memory_retry_4":
        raise RuntimeError("long-context diagnostic is not descended from the passing geometry")
    for key in ("upstream", "training_data", "runtime_cache_policy"):
        if config.get(key) != one_step_config.get(key):
            raise RuntimeError(f"long-context {key} differs from the passing diagnostic")

    changed_recipe_keys = {
        key
        for key in one_step_config["recipe"]
        if one_step_config["recipe"].get(key) != config["recipe"].get(key)
    }
    if changed_recipe_keys != ALLOWED_RECIPE_CHANGES:
        raise RuntimeError(
            "long-context recipe changes are not exact: "
            f"observed {sorted(changed_recipe_keys)}"
        )
    expected_recipe = {
        **one_step_config["recipe"],
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "max_completion_tokens": 4096,
        "run_config": RUN_CONFIG,
    }
    if config["recipe"] != expected_recipe:
        raise RuntimeError("long-context recipe is not the exact released extension")
    if config["recipe"]["effective_batch_size"] != 32:
        raise RuntimeError("long-context diagnostic changed the effective batch size")

    hardware = config.get("hardware", {})
    if hardware != {
        "partition": "general-gpu",
        "gpu_type": "a100-sxm4",
        "gpu_count": 4,
        "minimum_vram_mib_per_gpu": 81000,
        "single_node_required": True,
        "purpose": "one-step 4096-token full-vocabulary memory and update gate",
    }:
        raise RuntimeError("long-context hardware contract drifted")

    in_job = load_object(
        Path(prerequisite["upstream_one_step_in_job_gate"]), "upstream in-job gate"
    )
    terminal = load_object(
        Path(prerequisite["upstream_one_step_terminal_audit"]),
        "upstream terminal audit",
    )
    if in_job.get("status") != "passed" or in_job.get("decision") != (
        "ONE_STEP_UPDATE_DIAGNOSTIC_PASSED_IN_JOB"
    ):
        raise RuntimeError("upstream one-step in-job gate is not passing")
    if terminal.get("status") != "passed" or terminal.get("decision") != (
        "ONE_STEP_FULL_CUSTODY_PASSED"
    ):
        raise RuntimeError("upstream one-step terminal audit is not passing")
    if terminal.get("slurm") != {
        "job_id": prerequisite["upstream_one_step_job_id"],
        "state": "COMPLETED",
        "exit_code": "0:0",
    }:
        raise RuntimeError("upstream one-step terminal custody drifted")

    calibration = load_object(
        Path(prerequisite["student_length_calibration"]), "student length calibration"
    )
    if calibration.get("artifact_type") != (
        "opd_math_gated_campaign_v2_stage1_terminal_receipt"
    ):
        raise RuntimeError("unexpected length-calibration artifact")
    decision = calibration.get("decisions", {}).get("student", {})
    surface = calibration.get("student_surfaces", {}).get("raw_4096", {})
    if decision != {"status": "QUALIFIED", "selected_max_completion_tokens": 4096}:
        raise RuntimeError("raw-student 4096-token calibration is not qualified")
    if surface.get("samples") != 128 or surface.get("at_cap_samples") != 0:
        raise RuntimeError("raw-student 4096-token calibration counts drifted")
    if prerequisite.get("calibrated_student_cap") != 4096:
        raise RuntimeError("preregistered student cap differs from calibration")

    return {
        "repository_commit": repository_commit,
        "release_prerequisite_files": records,
        "upstream_one_step_prerequisite_custody": one_step_custody,
        "length_calibration": {
            "status": decision["status"],
            "selected_max_completion_tokens": decision[
                "selected_max_completion_tokens"
            ],
            "samples": surface["samples"],
            "at_cap_samples": surface["at_cap_samples"],
        },
    }


def validate_execution_manifest(manifest: dict) -> None:
    if manifest.get("artifact_type") != "opsd_audited_harness_execution_tree":
        raise RuntimeError("long-context run lacks the audited harness execution tree")
    if manifest.get("semantic_edits") != []:
        raise RuntimeError("long-context execution tree contains a semantic objective edit")
    efficiency = manifest.get("semantic_preserving_efficiency_edits", [])
    if efficiency != [
        "pad each rank only to its observed batch maximum while retaining every generated token and active-token loss term"
    ]:
        raise RuntimeError("dynamic-padding execution contract drifted")
    changed = manifest.get("changed_files", {})
    if "opsd_trainer.py" not in changed or changed["opsd_trainer.py"].get(
        "replacement_count"
    ) != 4:
        raise RuntimeError("trajectory and dynamic-padding harness edits are missing")


def audit_trajectories(run_dir: Path, config: dict) -> dict:
    generation_dir = run_dir / "generations"
    files = sorted(generation_dir.glob("generations_step_*_rank_*.json"))
    if not files:
        raise RuntimeError("no rank-specific trajectory files were saved")
    trajectories = []
    file_records = []
    for path in files:
        payload = load_object(path, "trajectory file")
        rows = payload.get("generations")
        if not isinstance(rows, list) or payload.get("num_samples") != len(rows):
            raise RuntimeError(f"trajectory cardinality drifted in {path}")
        if any(row.get("rank") != payload.get("rank") for row in rows):
            raise RuntimeError(f"trajectory rank drifted in {path}")
        trajectories.extend(rows)
        file_records.append(
            {"path": str(path.resolve()), "sha256": sha256(path), "rows": len(rows)}
        )

    gate = config["pass_gate"]
    expected = gate["expected_trajectory_count"]
    if len(trajectories) != expected:
        raise RuntimeError(
            f"expected {expected} stored trajectories, observed {len(trajectories)}"
        )
    cap = config["recipe"]["max_completion_tokens"]
    by_rank: dict[int, list[int]] = defaultdict(list)
    at_cap = 0
    tokens = 0
    for row in trajectories:
        required = {
            "step",
            "rank",
            "local_sequence",
            "prompt",
            "completion",
            "completion_tokens",
            "at_cap",
            "max_completion_length",
        }
        if set(row) != required:
            raise RuntimeError("trajectory schema drifted")
        rank = row["rank"]
        length = row["completion_tokens"]
        if type(rank) is not int or rank not in range(4):
            raise RuntimeError("trajectory rank is invalid")
        if type(length) is not int or length <= 0 or length > cap:
            raise RuntimeError("trajectory completion length is outside the registered cap")
        if row["max_completion_length"] != cap:
            raise RuntimeError("trajectory cap field drifted")
        if row["at_cap"] is not (length >= cap):
            raise RuntimeError("trajectory cap flag disagrees with its token count")
        if not isinstance(row["prompt"], str) or not isinstance(row["completion"], str):
            raise RuntimeError("trajectory text is missing")
        by_rank[rank].append(row["local_sequence"])
        at_cap += int(row["at_cap"])
        tokens += length
    expected_per_rank = expected // 4
    if sorted(by_rank) != [0, 1, 2, 3]:
        raise RuntimeError("not every rank contributed trajectories")
    for rank, sequence in by_rank.items():
        if sorted(sequence) != list(range(expected_per_rank)):
            raise RuntimeError(f"rank {rank} local trajectory sequence is incomplete")
    if at_cap > gate["maximum_at_cap_trajectory_count"]:
        raise RuntimeError(
            f"long-context cap gate failed: {at_cap}/{expected} trajectories reached cap"
        )
    fraction = at_cap / expected
    if fraction > gate["maximum_at_cap_fraction"]:
        raise RuntimeError("long-context cap fraction exceeds its registered maximum")

    return {
        "status": "passed",
        "trajectory_count": expected,
        "completion_tokens": tokens,
        "mean_completion_tokens": tokens / expected,
        "at_cap_trajectory_count": at_cap,
        "at_cap_fraction": fraction,
        "completion_cap": cap,
        "trajectory_files": file_records,
    }


def run_diagnostic(args: argparse.Namespace) -> int:
    config_path = args.config.resolve()
    config = load_object(config_path, "long-context one-step preregistration")
    custody = validate_prerequisites(config, args.repository_commit)
    execution_manifest = args.execution_tree / "LEGALRAG_EXECUTION_MANIFEST.json"
    manifest = load_object(execution_manifest, "execution manifest")
    if manifest.get("repository_commit") != args.repository_commit:
        raise RuntimeError("execution tree is not bound to the launch commit")
    if manifest.get("upstream_commit") != config["upstream"]["repository_commit"]:
        raise RuntimeError("execution tree upstream commit drifted")
    validate_execution_manifest(manifest)
    model_dir = args.model_dir.resolve()
    if model_dir.name != config["upstream"]["model_revision"]:
        raise RuntimeError("resolved model snapshot revision drifted")
    runtime_cache = validate_runtime_cache_environment(config, args.slurm_job_id)
    recipe = config["recipe"]
    command = training_command(
        args.env_dir.resolve(),
        args.execution_tree.resolve(),
        model_dir,
        args.output_root.resolve(),
        args.port,
        per_device_train_batch_size=recipe["per_device_train_batch_size"],
        gradient_accumulation_steps=recipe["gradient_accumulation_steps"],
        run_config=recipe["run_config"],
        max_completion_tokens=recipe["max_completion_tokens"],
    )
    start = {
        "schema_version": 1,
        "artifact_type": "opd_positive_control_long_step_custody_start",
        "campaign_id": config["campaign_id"],
        "stage_id": config["stage_id"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit": args.repository_commit,
        "slurm_job_id": args.slurm_job_id,
        "preregistration": str(config_path),
        "preregistration_sha256": sha256(config_path),
        "execution_manifest": str(execution_manifest.resolve()),
        "execution_manifest_sha256": sha256(execution_manifest),
        "model_dir": str(model_dir),
        "command": command,
        "runtime_cache": runtime_cache,
        **custody,
    }
    custody_path = args.output_root / "custody_start.json"
    write_exclusive(custody_path, start)
    completed = subprocess.run(
        command, cwd=args.execution_tree, env=os.environ.copy(), check=False
    )
    exit_receipt = {
        "schema_version": 1,
        "artifact_type": "opd_positive_control_long_step_exit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit": args.repository_commit,
        "slurm_job_id": args.slurm_job_id,
        "returncode": completed.returncode,
        "custody_start": str(custody_path.resolve()),
        "custody_start_sha256": sha256(custody_path),
    }
    exit_path = args.output_root / "training_exit.json"
    write_exclusive(exit_path, exit_receipt)
    if completed.returncode != 0:
        return completed.returncode

    training = audit_training(args.output_root, config, run_config=RUN_CONFIG)
    trajectories = audit_trajectories(
        args.output_root / "training" / RUN_CONFIG, config
    )
    training["trajectory_audit"] = trajectories
    # Bind the trajectory files into the complete training-tree manifest.
    training["training_artifact_files"] = hash_tree(
        args.output_root / "training" / RUN_CONFIG
    )
    receipt = {
        "schema_version": 1,
        "artifact_type": "opd_positive_control_long_step_in_job_gate",
        "campaign_id": config["campaign_id"],
        "stage_id": config["stage_id"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit": args.repository_commit,
        "slurm_job_id": args.slurm_job_id,
        "status": "passed",
        "decision": "LONG4096_ONE_STEP_UPDATE_AND_TRAJECTORY_GATE_PASSED_IN_JOB",
        "scientific_claim": "none_plumbing_and_length_custody_only",
        "custody_start": str(custody_path.resolve()),
        "custody_start_sha256": sha256(custody_path),
        "training_exit": str(exit_path.resolve()),
        "training_exit_sha256": sha256(exit_path),
        "audit": training,
        "release_state": "terminal_audit_required_100_step_training_blocked",
    }
    write_exclusive(args.output_root / "in_job_gate.json", receipt)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repository-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--env-dir", type=Path, required=True)
    parser.add_argument("--execution-tree", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run_diagnostic(parse_args()))
