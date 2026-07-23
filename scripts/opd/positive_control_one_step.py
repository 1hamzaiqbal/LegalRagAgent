#!/usr/bin/env python3
"""Run and audit the preregistered one-step upstream OPSD diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


RUN_CONFIG = "one_step_upstream_opsd"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_object(path: Path, label: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return payload


def write_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def require_hash(path: Path, expected: str, label: str) -> str:
    observed = sha256(path)
    if observed != expected:
        raise RuntimeError(
            f"{label} hash mismatch: expected {expected}, observed {observed}"
        )
    return observed


def validate_prerequisites(config: dict, repository_commit: str) -> dict:
    if config.get("status") != "preregistered_diagnostic_only_100_step_training_blocked":
        raise RuntimeError("one-step diagnostic is not preregistered")
    if config.get("stage_id") != "one_step_real_model_update_diagnostic":
        raise RuntimeError("unexpected diagnostic stage")
    boundaries = config["immutable_boundaries"]
    if not all(boundaries.values()):
        raise RuntimeError("an immutable campaign boundary is not enforced")

    prereq = config["prerequisites"]
    records = {}
    for key in (
        "base_evaluation_json",
        "base_gate",
        "preflight_receipt",
        "preflight_independent_audit",
        "data_manifest",
        "environment_freeze",
    ):
        path = Path(prereq[key]).resolve()
        records[key] = {
            "path": str(path),
            "sha256": require_hash(path, prereq[f"{key}_sha256"], key),
        }

    base_gate = load_object(Path(prereq["base_gate"]), "base gate")
    reconstruction = base_gate.get("independent_reconstruction", {})
    if base_gate.get("status") != "passed" or base_gate.get("decision") != prereq["base_gate_decision"]:
        raise RuntimeError("base reproduction gate is not passing")
    if base_gate.get("repository_commit") != prereq["base_evaluation_repository_commit"]:
        raise RuntimeError("base gate producer commit drifted")
    if reconstruction.get("correct") != prereq["base_correct"]:
        raise RuntimeError("base correct-count custody drifted")
    if reconstruction.get("generations") != prereq["base_generations"]:
        raise RuntimeError("base generation-count custody drifted")

    preflight = load_object(Path(prereq["preflight_receipt"]), "preflight receipt")
    if preflight.get("status") != "passed":
        raise RuntimeError("preflight receipt is not passing")
    audit = load_object(
        Path(prereq["preflight_independent_audit"]), "preflight independent audit"
    )
    if audit.get("status") not in {"passed", "passed_with_serialization_defect_reconstructed"}:
        raise RuntimeError("independent preflight audit is not passing")

    return {
        "repository_commit": repository_commit,
        "prerequisite_files": records,
        "base_gate_reconstruction": reconstruction,
    }


def training_command(
    env_dir: Path,
    execution_tree: Path,
    model_dir: Path,
    output_root: Path,
    port: int,
) -> list[str]:
    return [
        str(env_dir / "bin" / "accelerate"),
        "launch",
        "--config_file",
        str(execution_tree / "accelerate.yaml"),
        "--num_processes",
        "4",
        "--gradient_accumulation_steps",
        "2",
        "--main_process_port",
        str(port),
        "opsd_train.py",
        "--model_name_or_path",
        str(model_dir),
        "--learning_rate",
        "5e-6",
        "--max_grad_norm",
        "0.1",
        "--per_device_train_batch_size",
        "4",
        "--gradient_checkpointing",
        "--gradient_accumulation_steps",
        "2",
        "--output_dir",
        str(output_root / "training"),
        "--run_config",
        RUN_CONFIG,
        "--num_train_epochs",
        "30",
        "--max_steps",
        "1",
        "--max_completion_length",
        "1024",
        "--save_strategy",
        "steps",
        "--save_steps",
        "1",
        "--logging_steps",
        "1",
        "--attn_implementation",
        "flash_attention_2",
        "--torch_dtype",
        "bfloat16",
        "--max_length",
        "20000",
        "--beta",
        "0",
        "--use_vllm",
        "--vllm_mode",
        "colocate",
        "--vllm_gpu_memory_utilization",
        "0.6",
        "--vllm_tensor_parallel_size",
        "1",
        "--use_peft",
        "--lora_r",
        "64",
        "--lora_alpha",
        "128",
        "--lora_target_modules",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "--temperature",
        "1.1",
        "--top_p",
        "0.95",
        "--top_k",
        "20",
        "--lmbda",
        "1",
        "--fixed_teacher",
        "--jsd_token_clip",
        "0.05",
        "--wandb_project",
        "OPSD",
        "--seed",
        "42",
        "--data_seed",
        "42",
        "--report_to",
        "none",
    ]


def hash_tree(root: Path) -> list[dict]:
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"training tree contains a symlink: {path}")
        if path.is_file():
            records.append(
                {
                    "path": str(path.relative_to(root)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    if not records:
        raise RuntimeError("training artifact tree is empty")
    return records


def audit_training(output_root: Path, config: dict) -> dict:
    import torch
    from safetensors.torch import load_file

    run_dir = output_root / "training" / RUN_CONFIG
    checkpoint = run_dir / "checkpoint-1"
    state_path = checkpoint / "trainer_state.json"
    adapter_path = checkpoint / "adapter_model.safetensors"
    if not checkpoint.is_dir():
        raise RuntimeError("checkpoint-1 was not created")
    if not state_path.is_file():
        raise RuntimeError("checkpoint-1 trainer_state.json is missing")
    if not adapter_path.is_file():
        raise RuntimeError("checkpoint-1 adapter_model.safetensors is missing")

    state = load_object(state_path, "trainer state")
    gate = config["pass_gate"]
    if state.get("global_step") != gate["global_step"]:
        raise RuntimeError("trainer global step is not exactly one")
    if state.get("max_steps") != gate["global_step"]:
        raise RuntimeError("trainer max_steps is not exactly one")

    step_logs = [row for row in state.get("log_history", []) if row.get("step") == 1]
    loss_values = [float(row["loss"]) for row in step_logs if "loss" in row]
    grad_values = [float(row["grad_norm"]) for row in step_logs if "grad_norm" in row]
    if not loss_values or not all(math.isfinite(value) for value in loss_values):
        raise RuntimeError("step-one logged loss is missing or non-finite")
    if not grad_values or not all(math.isfinite(value) and value > 0 for value in grad_values):
        raise RuntimeError("step-one gradient norm is missing, non-finite, or zero")

    tensors = load_file(str(adapter_path), device="cpu")
    lora_b = {name: tensor for name, tensor in tensors.items() if "lora_B" in name}
    if not lora_b:
        raise RuntimeError("checkpoint contains no LoRA-B tensors")
    nonzero = {
        name: int(torch.count_nonzero(tensor).item())
        for name, tensor in lora_b.items()
    }
    if not any(count > 0 for count in nonzero.values()):
        raise RuntimeError("all LoRA-B tensors remain at their zero initialization")
    if not all(torch.isfinite(tensor.float()).all().item() for tensor in tensors.values()):
        raise RuntimeError("checkpoint adapter contains non-finite parameters")

    records = hash_tree(run_dir)
    return {
        "status": "passed",
        "global_step": state["global_step"],
        "max_steps": state["max_steps"],
        "logged_loss": loss_values,
        "logged_grad_norm": grad_values,
        "checkpoint": str(checkpoint.resolve()),
        "adapter_sha256": sha256(adapter_path),
        "adapter_tensor_count": len(tensors),
        "lora_B_tensor_count": len(lora_b),
        "nonzero_lora_B_tensor_count": sum(count > 0 for count in nonzero.values()),
        "nonzero_lora_B_parameter_count": sum(nonzero.values()),
        "parameter_update_basis": (
            "The raw pinned base constructs a new PEFT adapter whose LoRA-B matrices "
            "start at zero; checkpoint-1 contains finite nonzero LoRA-B values and the "
            "trainer records a finite positive gradient norm at global_step=1."
        ),
        "training_artifact_files": records,
    }


def run_diagnostic(args: argparse.Namespace) -> int:
    config_path = args.config.resolve()
    config = load_object(config_path, "one-step preregistration")
    custody = validate_prerequisites(config, args.repository_commit)
    execution_manifest = args.execution_tree / "LEGALRAG_EXECUTION_MANIFEST.json"
    manifest = load_object(execution_manifest, "execution manifest")
    if manifest.get("repository_commit") != args.repository_commit:
        raise RuntimeError("execution tree is not bound to the launch commit")
    if manifest.get("upstream_commit") != config["upstream"]["repository_commit"]:
        raise RuntimeError("execution tree upstream commit drifted")
    model_dir = args.model_dir.resolve()
    if model_dir.name != config["upstream"]["model_revision"]:
        raise RuntimeError("resolved model snapshot revision drifted")

    command = training_command(
        args.env_dir.resolve(),
        args.execution_tree.resolve(),
        model_dir,
        args.output_root.resolve(),
        args.port,
    )
    start = {
        "schema_version": 1,
        "artifact_type": "opd_positive_control_one_step_custody_start",
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
        **custody,
    }
    custody_path = args.output_root / "custody_start.json"
    write_exclusive(custody_path, start)

    completed = subprocess.run(
        command,
        cwd=args.execution_tree,
        env=os.environ.copy(),
        check=False,
    )
    exit_receipt = {
        "schema_version": 1,
        "artifact_type": "opd_positive_control_one_step_training_exit",
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

    audit = audit_training(args.output_root, config)
    receipt = {
        "schema_version": 1,
        "artifact_type": "opd_positive_control_one_step_in_job_gate",
        "campaign_id": config["campaign_id"],
        "stage_id": config["stage_id"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit": args.repository_commit,
        "slurm_job_id": args.slurm_job_id,
        "status": "passed",
        "decision": "ONE_STEP_UPDATE_DIAGNOSTIC_PASSED_IN_JOB",
        "scientific_claim": "none_plumbing_only",
        "custody_start": str(custody_path.resolve()),
        "custody_start_sha256": sha256(custody_path),
        "training_exit": str(exit_path.resolve()),
        "training_exit_sha256": sha256(exit_path),
        "audit": audit,
        "release_state": "terminal_audit_required_100_step_still_blocked",
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
