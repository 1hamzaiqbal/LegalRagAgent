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
RUNTIME_CACHE_VARIABLES = (
    "XDG_CACHE_HOME",
    "VLLM_CACHE_ROOT",
    "TORCHINDUCTOR_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "CUDA_CACHE_PATH",
    "TORCH_HOME",
    "TORCH_EXTENSIONS_DIR",
    "TMPDIR",
)


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


def validate_runtime_cache_environment(config: dict, slurm_job_id: str) -> dict:
    policy = config.get("runtime_cache_policy")
    if not isinstance(policy, dict):
        raise RuntimeError("runtime-cache policy is missing")
    if policy.get("per_job_namespace") != "job_${SLURM_JOB_ID}":
        raise RuntimeError("runtime-cache namespace is not bound to the Slurm job")
    if policy.get("require_existing_writable_directories") is not True:
        raise RuntimeError("runtime-cache directories are not required to be writable")
    if policy.get("record_in_custody_start") is not True:
        raise RuntimeError("runtime-cache paths are not required in custody")

    root = Path(policy["root"]).resolve()
    job_root = (root / f"job_{slurm_job_id}").resolve()
    environment = policy.get("environment")
    if not isinstance(environment, dict) or tuple(environment) != RUNTIME_CACHE_VARIABLES:
        raise RuntimeError("runtime-cache environment keys or order drifted")
    forbidden = tuple(Path(value).resolve() for value in policy["forbidden_prefixes"])

    records = {}
    for variable in RUNTIME_CACHE_VARIABLES:
        suffix = environment[variable]
        if Path(suffix).is_absolute() or Path(suffix).parts != (suffix,):
            raise RuntimeError(f"{variable} cache suffix must be one relative component")
        expected = (job_root / suffix).resolve()
        observed_raw = os.environ.get(variable)
        if not observed_raw:
            raise RuntimeError(f"{variable} is not exported")
        observed = Path(observed_raw).resolve()
        if observed != expected:
            raise RuntimeError(
                f"{variable} path drifted: expected {expected}, observed {observed}"
            )
        try:
            observed.relative_to(job_root)
        except ValueError as error:
            raise RuntimeError(f"{variable} escaped the per-job cache root") from error
        for prefix in forbidden:
            try:
                observed.relative_to(prefix)
            except ValueError:
                continue
            raise RuntimeError(f"{variable} resolves under forbidden prefix {prefix}")
        if not observed.is_dir() or not os.access(observed, os.W_OK | os.X_OK):
            raise RuntimeError(f"{variable} cache directory is not writable: {observed}")
        records[variable] = str(observed)

    probe = job_root / ".legalrag_cache_write_probe"
    descriptor = os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    probe.unlink()
    return {
        "status": "passed",
        "decision": "PER_JOB_EIT_RUNTIME_CACHE_PATHS_VALIDATED",
        "root": str(root),
        "job_root": str(job_root),
        "environment": records,
        "forbidden_prefixes": [str(path) for path in forbidden],
        "write_probe_passed": True,
    }


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

    retry = config.get("retry")
    trainer_retry_ids = {
        "trainer_data_retry_2",
        "runtime_cache_retry_3",
        "microbatch_memory_retry_4",
    }
    if retry is not None and retry.get("attempt_id") in trainer_retry_ids:
        retry_id = retry.get("attempt_id")
        allowed_change = {
            "trainer_data_retry_2": "restore_pinned_trl_conversations_field_only",
            "runtime_cache_retry_3": "runtime_cache_placement_only",
            "microbatch_memory_retry_4": (
                "per_device_microbatch_and_gradient_accumulation_only"
            ),
        }[retry_id]
        if retry.get("allowed_change") != allowed_change:
            raise RuntimeError(f"{retry_id} changes more than its registered boundary")
        if retry.get("model_recipe_and_ordered_rows_unchanged") is not True:
            raise RuntimeError(f"{retry_id} does not preserve model, recipe, and row order")
        prerequisite_keys = [
            "metadata_failure",
            "trainer_schema_failure",
            "trainer_data_manifest",
            "trainer_data_audit",
        ]
        if retry_id in {"runtime_cache_retry_3", "microbatch_memory_retry_4"}:
            prerequisite_keys.append("runtime_cache_failure")
        if retry_id == "microbatch_memory_retry_4":
            prerequisite_keys.append("full_vocab_oom_failure")
        for key in prerequisite_keys:
            path = Path(prereq[key]).resolve()
            records[key] = {
                "path": str(path),
                "sha256": require_hash(path, prereq[f"{key}_sha256"], key),
            }

        metadata_failure = load_object(
            Path(prereq["metadata_failure"]), "metadata failure"
        )
        trainer_failure = load_object(
            Path(prereq["trainer_schema_failure"]), "trainer-schema failure"
        )
        expected_failures = (
            (
                metadata_failure,
                "132150",
                "failed_before_training",
                "PARQUET_FEATURE_METADATA_INCOMPATIBLE",
            ),
            (
                trainer_failure,
                "135003",
                "failed_before_optimization",
                "TRAINER_CHATML_SOURCE_FIELD_MISSING",
            ),
        )
        for failure, job_id, status, decision in expected_failures:
            if failure.get("status") != status or failure.get("decision") != decision:
                raise RuntimeError(f"predecessor {job_id} failure custody drifted")
            if failure.get("slurm", {}).get("job_id") != job_id:
                raise RuntimeError(f"predecessor {job_id} identity drifted")
            if any(
                (
                    failure.get("optimizer_steps") != 0,
                    failure.get("checkpoint_created") is not False,
                    failure.get("opd_result_created") is not False,
                )
            ):
                raise RuntimeError(f"predecessor {job_id} produced a training result")

        if retry_id in {"runtime_cache_retry_3", "microbatch_memory_retry_4"}:
            expected_predecessors = ["132150", "135003", "135015"]
            if retry_id == "microbatch_memory_retry_4":
                expected_predecessors.append("135079")
            if retry.get("predecessor_job_ids") != expected_predecessors:
                raise RuntimeError("runtime-cache retry predecessor order drifted")
            cache_failure = load_object(
                Path(prereq["runtime_cache_failure"]), "runtime-cache failure"
            )
            if cache_failure.get("status") != "failed_before_optimization" or cache_failure.get(
                "decision"
            ) != "RUNTIME_COMPILE_CACHE_HOME_QUOTA_EXCEEDED":
                raise RuntimeError("job 135015 is not the sealed runtime-cache failure")
            if cache_failure.get("slurm", {}).get("job_id") != "135015":
                raise RuntimeError("runtime-cache predecessor identity drifted")
            if any(
                (
                    cache_failure.get("optimizer_steps") != 0,
                    cache_failure.get("checkpoint_created") is not False,
                    cache_failure.get("opd_result_created") is not False,
                )
            ):
                raise RuntimeError("runtime-cache predecessor produced a training result")

        if retry_id == "microbatch_memory_retry_4":
            oom_failure = load_object(
                Path(prereq["full_vocab_oom_failure"]), "full-vocabulary OOM failure"
            )
            if oom_failure.get("status") != "failed_before_backward_or_optimization" or oom_failure.get(
                "decision"
            ) != "A6000_FULL_VOCAB_MICROBATCH4_OOM":
                raise RuntimeError("job 135079 is not the sealed full-vocabulary OOM")
            if oom_failure.get("slurm", {}).get("job_id") != "135079":
                raise RuntimeError("full-vocabulary OOM predecessor identity drifted")
            if any(
                (
                    oom_failure.get("backward_completed") is not False,
                    oom_failure.get("optimizer_steps") != 0,
                    oom_failure.get("checkpoint_created") is not False,
                    oom_failure.get("opd_result_created") is not False,
                )
            ):
                raise RuntimeError("full-vocabulary OOM predecessor produced an update")
            recipe = config.get("recipe", {})
            if retry.get("effective_batch_size_unchanged") is not True or recipe.get(
                "effective_batch_size"
            ) != 32:
                raise RuntimeError("memory retry changed the effective batch size")
            if retry.get("completion_cap_unchanged") is not True or recipe.get(
                "max_completion_tokens"
            ) != 1024:
                raise RuntimeError("memory retry changed the completion cap")
            if recipe.get("per_device_train_batch_size") != 2 or recipe.get(
                "gradient_accumulation_steps"
            ) != 4:
                raise RuntimeError("memory retry does not use the preregistered 2x4 geometry")

        manifest = load_object(
            Path(prereq["trainer_data_manifest"]), "trainer-data manifest"
        )
        trainer_audit = load_object(
            Path(prereq["trainer_data_audit"]), "trainer-data audit"
        )
        training_data = config.get("training_data", {})
        if manifest.get("artifact_type") != "opd_positive_control_trainer_data":
            raise RuntimeError("unexpected trainer-data manifest type")
        if trainer_audit.get("status") != "passed" or trainer_audit.get(
            "decision"
        ) != "PINNED_TRL026_TRAINER_DATA_COMPATIBLE":
            raise RuntimeError("trainer-data audit is not passing")
        expected_versions = {
            "datasets_version": "3.6.0",
            "transformers_version": "4.57.1",
            "trl_version": "0.26.0",
        }
        if any(trainer_audit.get(key) != value for key, value in expected_versions.items()):
            raise RuntimeError("trainer-data audit did not use the pinned runtime")
        if trainer_audit.get("upstream_commit") != config["upstream"][
            "repository_commit"
        ]:
            raise RuntimeError("trainer-data audit used the wrong upstream checkout")
        for key in ("rows", "trainer_field_sequence_sha256"):
            expected = training_data.get(key)
            if manifest.get(key) != expected or trainer_audit.get(key) != expected:
                raise RuntimeError(f"trainer-data {key} custody drifted")
        if trainer_audit.get("columns") != training_data.get("required_columns"):
            raise RuntimeError("trainer-data column custody drifted")
        if trainer_audit.get("token_sequence_sha256") != training_data.get(
            "token_sequence_sha256"
        ):
            raise RuntimeError("trainer-data token sequence custody drifted")
        if trainer_audit.get("tokenized_sequences") != training_data.get("rows"):
            raise RuntimeError("trainer-data audit did not tokenize every row")
        if trainer_audit.get("collator_batch_size") != 4:
            raise RuntimeError("trainer-data audit did not exercise the custom collator")
        if trainer_audit.get("manifest_sha256") != prereq[
            "trainer_data_manifest_sha256"
        ]:
            raise RuntimeError("trainer-data audit names the wrong manifest")
        manifest_shards = [row["sha256"] for row in manifest["trainer_shards"]]
        if trainer_audit.get("trainer_shard_sha256") != manifest_shards:
            raise RuntimeError("trainer-data shard hashes disagree")
        if training_data.get("trainer_shard_sha256") != manifest_shards:
            raise RuntimeError("preregistered trainer-data shard hashes disagree")
        train_glob = training_data.get("parquet_glob")
        trainer_root = Path(trainer_audit["trainer_root"]).resolve()
        if train_glob != str(trainer_root / "*.parquet"):
            raise RuntimeError("training Parquet path is not the audited trainer namespace")
        live_shards = sorted(trainer_root.glob("*.parquet"))
        if [sha256(path) for path in live_shards] != manifest_shards:
            raise RuntimeError("trainer-data shards changed after their audit")
        if os.environ.get("LEGALRAG_OPSD_TRAIN_PARQUET") != train_glob:
            raise RuntimeError("training Parquet environment is not preregistered")

    elif retry is not None:
        if retry.get("attempt_id") != "normalized_data_retry_1":
            raise RuntimeError("unexpected one-step retry identity")
        if retry.get("allowed_change") != "training_parquet_serialization_only":
            raise RuntimeError("retry changes more than Parquet serialization")
        if retry.get("model_recipe_and_ordered_rows_unchanged") is not True:
            raise RuntimeError("retry does not preserve the model, recipe, and rows")
        for key in (
            "terminal_failure",
            "normalized_data_manifest",
            "normalized_data_audit",
        ):
            path = Path(prereq[key]).resolve()
            records[key] = {
                "path": str(path),
                "sha256": require_hash(path, prereq[f"{key}_sha256"], key),
            }

        failure = load_object(Path(prereq["terminal_failure"]), "terminal failure")
        if failure.get("status") != "failed_before_training":
            raise RuntimeError("predecessor was not sealed as a pre-training failure")
        if failure.get("decision") != "PARQUET_FEATURE_METADATA_INCOMPATIBLE":
            raise RuntimeError("predecessor failure is not the registered metadata defect")
        if failure.get("slurm", {}).get("job_id") != retry.get("predecessor_job_id"):
            raise RuntimeError("predecessor job identity drifted")
        if any(
            (
                failure.get("optimizer_steps") != 0,
                failure.get("checkpoint_created") is not False,
                failure.get("opd_result_created") is not False,
            )
        ):
            raise RuntimeError("predecessor unexpectedly produced a training result")

        normalized = load_object(
            Path(prereq["normalized_data_manifest"]), "normalized-data manifest"
        )
        normalized_audit = load_object(
            Path(prereq["normalized_data_audit"]), "normalized-data audit"
        )
        training_data = config.get("training_data", {})
        if normalized.get("artifact_type") != "opd_positive_control_normalized_data":
            raise RuntimeError("unexpected normalized-data manifest type")
        if normalized_audit.get("status") != "passed" or normalized_audit.get(
            "decision"
        ) != "NORMALIZED_DATA_LOAD_COMPATIBLE":
            raise RuntimeError("normalized-data audit is not passing")
        if normalized_audit.get("datasets_version") != "3.6.0":
            raise RuntimeError("normalized data was not audited with datasets 3.6.0")
        for key in ("rows", "row_sequence_sha256"):
            expected = training_data.get(key)
            if normalized.get(key) != expected or normalized_audit.get(key) != expected:
                raise RuntimeError(f"normalized-data {key} custody drifted")
        if normalized_audit.get("columns") != training_data.get("required_columns"):
            raise RuntimeError("normalized-data column custody drifted")
        if normalized_audit.get("manifest_sha256") != prereq[
            "normalized_data_manifest_sha256"
        ]:
            raise RuntimeError("normalized-data audit names the wrong manifest")
        manifest_shards = [row["sha256"] for row in normalized["normalized_shards"]]
        if normalized_audit.get("normalized_shard_sha256") != manifest_shards:
            raise RuntimeError("normalized-data shard hashes disagree")
        if training_data.get("normalized_shard_sha256") != manifest_shards:
            raise RuntimeError("preregistered normalized-data shard hashes disagree")
        train_glob = training_data.get("parquet_glob")
        normalized_root = Path(normalized_audit["normalized_root"]).resolve()
        if train_glob != str(normalized_root / "*.parquet"):
            raise RuntimeError("training Parquet path is not the audited namespace")
        live_shards = sorted(normalized_root.glob("*.parquet"))
        if [sha256(path) for path in live_shards] != manifest_shards:
            raise RuntimeError("normalized training shards changed after their audit")
        if os.environ.get("LEGALRAG_OPSD_TRAIN_PARQUET") != train_glob:
            raise RuntimeError("training Parquet environment is not preregistered")

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
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 2,
    max_steps: int = 1,
    save_steps: int = 1,
    logging_steps: int = 1,
    run_config: str = RUN_CONFIG,
    max_completion_tokens: int = 1024,
) -> list[str]:
    return [
        str(env_dir / "bin" / "accelerate"),
        "launch",
        "--config_file",
        str(execution_tree / "accelerate.yaml"),
        "--num_processes",
        "4",
        "--gradient_accumulation_steps",
        str(gradient_accumulation_steps),
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
        str(per_device_train_batch_size),
        "--gradient_checkpointing",
        "--gradient_accumulation_steps",
        str(gradient_accumulation_steps),
        "--output_dir",
        str(output_root / "training"),
        "--run_config",
        run_config,
        "--num_train_epochs",
        "30",
        "--max_steps",
        str(max_steps),
        "--max_completion_length",
        str(max_completion_tokens),
        "--save_strategy",
        "steps",
        "--save_steps",
        str(save_steps),
        "--logging_steps",
        str(logging_steps),
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


def audit_training(
    output_root: Path, config: dict, *, run_config: str = RUN_CONFIG
) -> dict:
    import torch
    from safetensors.torch import load_file

    run_dir = output_root / "training" / run_config
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
    runtime_cache = None
    if config.get("retry", {}).get("attempt_id") in {
        "runtime_cache_retry_3",
        "microbatch_memory_retry_4",
    }:
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
        "runtime_cache": runtime_cache,
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
