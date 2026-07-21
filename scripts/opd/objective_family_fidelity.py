#!/usr/bin/env python3
"""Post-job audits and closure for objective-family fidelity diagnostics.

The one-step jobs are plumbing diagnostics, never scientific outcomes.  This
module runs only after Slurm has closed each stdout file.  It independently
reopens the local traces or native-veRL receipt, verifies a finite nonzero
optimizer update, binds the completed scheduler stdout, seals a diagnostic
receipt, and finally constructs the exact 12-cell fidelity closure.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import stat
import subprocess
from pathlib import Path
from typing import Any

try:
    from .objective_family_inputs import canonical_json_sha256, sha256_file, sha256_tree
    from .objective_registry import (
        EXPECTED_OBJECTIVE_IDS,
        K1_OBJECTIVE_IDS,
        LOCAL_OBJECTIVE_IDS,
        UPSTREAM_OBJECTIVE_IDS,
        UPSTREAM_VERL_COMMIT,
        load_objective_registry,
        resolve_objective,
    )
    from .verify_verl_stored_rollout import (
        REAL_MODEL_FIXTURE_ID,
        load_fixture,
    )
    from .verl_run_custody import PREFLIGHT_ID, RECEIPT_ID as VERL_RECEIPT_ID
except ImportError:
    from objective_family_inputs import (  # type: ignore
        canonical_json_sha256,
        sha256_file,
        sha256_tree,
    )
    from objective_registry import (  # type: ignore
        EXPECTED_OBJECTIVE_IDS,
        K1_OBJECTIVE_IDS,
        LOCAL_OBJECTIVE_IDS,
        UPSTREAM_OBJECTIVE_IDS,
        UPSTREAM_VERL_COMMIT,
        load_objective_registry,
        resolve_objective,
    )
    from verify_verl_stored_rollout import (  # type: ignore
        REAL_MODEL_FIXTURE_ID,
        load_fixture,
    )
    from verl_run_custody import (  # type: ignore
        PREFLIGHT_ID,
        RECEIPT_ID as VERL_RECEIPT_ID,
    )


ROOT = Path(__file__).resolve().parents[2]
FIDELITY_PLAN = ROOT / "configs/opd_math/fidelity_plan.json"
DIAGNOSTIC_RECEIPT_ID = "opd_math_objective_family_full_custody_diagnostic_v1"
FIDELITY_CLOSURE_ID = "opd_math_objective_family_fidelity_closure_v1"
REAL_RECEIPT_ID = "real_model_rollout_local_vs_pinned_verl_k1_v1"
SOURCES = ("M", "O")
EXPECTED_DIAGNOSTIC_KEYS = tuple(
    f"{objective_id}__{source}"
    for source in SOURCES
    for objective_id in EXPECTED_OBJECTIVE_IDS
)
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json(path: str | Path, label: str, *, readonly: bool = False) -> tuple[Path, dict[str, Any]]:
    raw = Path(path)
    _expect(raw.is_file() and not raw.is_symlink(), f"{label} must be a regular file")
    if readonly:
        _expect(
            raw.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH) == 0,
            f"{label} must be sealed read-only",
        )
    payload = json.loads(raw.read_text(encoding="utf-8"))
    _expect(isinstance(payload, dict), f"{label} must contain one JSON object")
    return raw.resolve(), payload


def _jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    _expect(path.is_file() and not path.is_symlink(), f"{label} must be a regular file")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        _expect(isinstance(value, dict), f"{label} row {line_number} is not an object")
        rows.append(value)
    return rows


def _finite_positive(value: Any, label: str) -> float:
    _expect(type(value) in (int, float), f"{label} is not numeric")
    result = float(value)
    _expect(math.isfinite(result) and result > 0, f"{label} is not finite and positive")
    return result


def _clean_commit() -> str:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    _expect(HEX40.fullmatch(commit) is not None and not status.strip(), "fidelity closure requires clean immutable Git")
    return commit


def _inside(root: Path, child: Path, label: str) -> Path:
    root = root.resolve()
    child = child.resolve()
    try:
        child.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escaped its run root") from exc
    return child


def _seal_tree(path: Path) -> None:
    for candidate in path.rglob("*"):
        _expect(not candidate.is_symlink(), f"cannot seal tree containing symlink: {candidate}")
        os.chmod(candidate, 0o555 if candidate.is_dir() else 0o444)
    os.chmod(path, 0o555)


def _bind_completed_stdout(
    path: str | Path, *, implementation: str, scheduler_job_id: str
) -> dict[str, Any]:
    raw = Path(path)
    _expect(raw.is_file() and not raw.is_symlink(), "Slurm stdout must be a regular file")
    expected_name = (
        f"opd_objective_family_{scheduler_job_id}.out"
        if implementation == "local"
        else f"opd_objective_verl_{scheduler_job_id}.out"
    )
    _expect(raw.name == expected_name, "Slurm stdout filename does not match scheduler custody")
    payload = raw.read_text(encoding="utf-8", errors="replace")
    end_marker = (
        "Objective-family run completed; held-out evaluation remains forbidden"
        if implementation == "local"
        else "Pinned veRL objective-family run completed; held-out release remains campaign-wide"
    )
    _expect(end_marker in payload, "Slurm stdout lacks the successful terminal marker")
    _expect(payload.strip(), "Slurm stdout is empty")
    os.chmod(raw, 0o444)
    return {
        "path": str(raw.resolve()),
        "sha256": sha256_file(raw),
        "bytes": raw.stat().st_size,
        "terminal_marker_observed": True,
    }


def _validate_trace_binding(completion: dict[str, Any], name: str, path: Path, rows: int) -> None:
    binding = (completion.get("trace_artifacts") or {}).get(name)
    _expect(
        isinstance(binding, dict)
        and binding.get("path") == str(path.resolve())
        and binding.get("rows") == rows
        and binding.get("sha256") == sha256_file(path),
        f"local diagnostic {name} trace binding drifted",
    )


def audit_local_diagnostic(
    *,
    run_root: str | Path,
    slurm_stdout: str | Path,
    objective_id: str,
    source: str,
    expected_commit: str,
) -> dict[str, Any]:
    _expect(objective_id in LOCAL_OBJECTIVE_IDS, "local diagnostic objective routing drifted")
    _expect(source in SOURCES, "local diagnostic source is invalid")
    _expect(HEX40.fullmatch(expected_commit) is not None, "expected commit is invalid")
    registry = load_objective_registry()
    objective = resolve_objective(objective_id, registry=registry)
    root = Path(run_root).resolve()
    _expect(root.is_dir() and not root.is_symlink(), "local diagnostic run root is missing")
    traces = _inside(root, root / "traces", "local trace root")
    run_path, run = _json(traces / "run_manifest.json", "local run manifest")
    completion_path, completion = _json(
        traces / "completion_manifest.json", "local completion manifest"
    )
    _expect(run.get("completion") == completion, "local run and completion manifests disagree")
    registry_binding = run.get("objective_registry") or {}
    registered = registry_binding.get("objective") or {}
    binding = run.get("binding") or {}
    expected_mode = {
        "task_rl": "task_rl",
        "task_rl_k1_ungated_clip5": "task_rl_k1_gap",
        "task_rl_k1_ungated_unclipped": "task_rl_k1_gap",
        "task_rl_k1_gated_clip5_beta5": "task_rl_k1_gap",
        "k1_bare_verl_compatible_clip10": "k1_bare",
    }[objective_id]
    _expect(
        run.get("objective") == expected_mode
        and run.get("objective_contract") == objective["objective_contract"]
        and registered == objective
        and registry_binding.get("registry_sha256") == registry["sha256"],
        "local diagnostic objective identity drifted",
    )
    _expect(
        run.get("git_commit") == expected_commit
        and run.get("git_worktree_clean") is True
        and run.get("seed") == 0
        and run.get("optimizer_steps_planned") == 1
        and run.get("micro_prompts_per_step") == 1
        and run.get("planned_rollout_samples") == 4,
        "local diagnostic fixed run geometry drifted",
    )
    _expect(
        binding.get("student_source") == source
        and binding.get("objective_family_diagnostic") is True
        and binding.get("budget_mode") == "primary_matched"
        and isinstance(binding.get("scheduler_job_id"), str)
        and re.fullmatch(r"[1-9][0-9]*", binding["scheduler_job_id"]) is not None,
        "local diagnostic source or scheduler binding drifted",
    )
    prompt = binding.get("objective_family_prompt_plan") or {}
    initialization = binding.get("objective_family_initialization") or {}
    _expect(
        prompt.get("source") == source
        and prompt.get("seed") == 0
        and prompt.get("consumed_prefix_rows") == 1
        and initialization.get("seed") == 0,
        "local diagnostic prompt or initialization binding drifted",
    )
    for selected, label in ((prompt, "prompt plan"), (initialization, "initialization manifest")):
        path = Path(str(selected.get("path")))
        _expect(path.is_file() and not path.is_symlink(), f"local {label} is missing")
        _expect(sha256_file(path) == selected.get("sha256"), f"local {label} hash drifted")
    sampled_k1 = objective_id in K1_OBJECTIVE_IDS
    _expect(
        binding.get("teacher_source") == ("O" if sampled_k1 else None)
        and binding.get("pair_id") == (f"O_{source}" if sampled_k1 else None),
        "local diagnostic teacher routing drifted",
    )
    gates = run.get("gates") or {}
    if sampled_k1:
        _expect(
            isinstance(gates.get("teacher_gap"), dict)
            and isinstance(gates.get("teacher_provenance"), dict)
            and isinstance(gates.get("server_scoring_contract"), dict)
            and isinstance(run.get("teacher_checkpoint"), str),
            "local K1 diagnostic lacks O-teacher custody",
        )
    else:
        _expect(
            run.get("teacher_checkpoint") is None
            and gates.get("teacher_gap") is None
            and gates.get("teacher_provenance") is None
            and gates.get("server_scoring_contract") is None,
            "task-RL diagnostic unexpectedly used a teacher",
        )
    _expect(
        completion.get("status") in {"completed", "completed_zero_task_signal_diagnostic"}
        and completion.get("objective_family_diagnostic") is True
        and completion.get("intended_scientific_run") is False
        and completion.get("custody_required") is True
        and completion.get("optimizer_steps_completed") == 1
        and completion.get("rollout_samples") == 4
        and completion.get("step_trace_rows") == 1
        and completion.get("sample_trace_rows") == 4
        and completion.get("realized_training_geometry_observed") is True
        and completion.get("finite_nonzero_gradient_observed") is True
        and completion.get("parameter_update_observed") is True
        and completion.get("clean_stable_code") is True
        and completion.get("stable_training_environment") is True
        and completion.get("stable_environment_end") is True
        and completion.get("stable_final_artifact_hash") is True
        and completion.get("training_artifact_eligible_for_held_out_evaluation") is False
        and completion.get("scientific_use_allowed") is False,
        "local diagnostic completion custody failed",
    )
    _expect(
        completion.get("local_server_process_binding_required") is sampled_k1
        and completion.get("live_local_server_process_binding_validated") is sampled_k1,
        "local diagnostic server-process custody drifted",
    )
    optimizer = completion.get("optimizer_state_signature_final") or {}
    _expect(
        isinstance(optimizer.get("tensors"), int)
        and optimizer["tensors"] > 0
        and isinstance(optimizer.get("elements"), int)
        and optimizer["elements"] > 0,
        "local diagnostic optimizer state is empty",
    )
    _finite_positive(optimizer.get("squared_l2"), "local optimizer squared L2")
    updates = completion.get("parameter_update_l2_by_step")
    _expect(isinstance(updates, list) and len(updates) == 1, "local update vector drifted")
    _finite_positive(updates[0], "local parameter update")
    steps_path = traces / "steps.jsonl"
    samples_path = traces / "samples.jsonl"
    steps = _jsonl(steps_path, "local step trace")
    samples = _jsonl(samples_path, "local sample trace")
    _expect(len(steps) == 1 and len(samples) == 4, "local diagnostic trace counts drifted")
    step = steps[0]
    _expect(step.get("step") == 1 and step.get("samples") == 4, "local diagnostic step identity drifted")
    for field in (
        "gradient_norm_before_clip",
        "parameter_update_l2",
        "optimizer_state_squared_l2",
    ):
        _finite_positive(step.get(field), f"local step {field}")
    _expect(
        isinstance(step.get("optimizer_state_tensors"), int)
        and step["optimizer_state_tensors"] > 0
        and isinstance(step.get("optimizer_state_elements"), int)
        and step["optimizer_state_elements"] > 0,
        "local step optimizer state is empty",
    )
    _validate_trace_binding(completion, "steps.jsonl", steps_path, 1)
    _validate_trace_binding(completion, "samples.jsonl", samples_path, 4)
    final = Path(str(completion.get("final_adapter"))).resolve()
    _inside(root, final, "local final adapter")
    _expect(
        final.is_dir()
        and not final.is_symlink()
        and sha256_tree(final) == completion.get("final_adapter_tree_sha256"),
        "local final adapter custody drifted",
    )
    stdout = _bind_completed_stdout(
        slurm_stdout,
        implementation="local",
        scheduler_job_id=binding["scheduler_job_id"],
    )
    run_tree_hash = sha256_tree(root)
    _seal_tree(root)
    return {
        "schema_version": 1,
        "receipt": DIAGNOSTIC_RECEIPT_ID,
        "status": "passed_plumbing",
        "scientific_use_allowed": False,
        "training_artifact_eligible_for_held_out_evaluation": False,
        "implementation": "local",
        "objective_id": objective_id,
        "source": source,
        "seed": 0,
        "git_commit": expected_commit,
        "objective_registry_sha256": registry["sha256"],
        "scheduler_job_id": binding["scheduler_job_id"],
        "run": {
            "root": str(root),
            "tree_sha256": run_tree_hash,
            "manifest": str(run_path),
            "manifest_sha256": sha256_file(run_path),
            "completion_manifest": str(completion_path),
            "completion_manifest_sha256": sha256_file(completion_path),
            "final_adapter_tree_sha256": completion["final_adapter_tree_sha256"],
        },
        "slurm_stdout": stdout,
        "checks": {
            "exact_one_step_four_rollouts": True,
            "finite_nonzero_gradient": True,
            "finite_nonzero_parameter_update": True,
            "finite_nonzero_optimizer_state": True,
            "exact_prompt_and_initialization_bound": True,
            "o_teacher_only_when_required": True,
            "heldout_outcomes_inspected": False,
        },
        "claim_boundary": "One-step local execution fidelity only; no task-performance inference.",
    }


def _binding_matches(value: Any, *, path_key: str, hash_key: str, tree: bool = False) -> None:
    _expect(isinstance(value, dict), f"native veRL {path_key} binding is invalid")
    path = Path(str(value.get(path_key))).resolve()
    expected = value.get(hash_key)
    observed = sha256_tree(path) if tree else sha256_file(path)
    _expect(observed == expected, f"native veRL {path_key} hash drifted")


def audit_upstream_diagnostic(
    *,
    native_receipt: str | Path,
    slurm_stdout: str | Path,
    source: str,
    expected_commit: str,
) -> dict[str, Any]:
    _expect(source in SOURCES, "upstream diagnostic source is invalid")
    receipt_path, native = _json(native_receipt, "native veRL receipt", readonly=True)
    _expect(
        native.get("schema_version") == 1
        and native.get("receipt") == VERL_RECEIPT_ID
        and native.get("status") == "passed_plumbing"
        and native.get("scientific_use_allowed") is False
        and native.get("training_artifact_eligible_for_held_out_evaluation") is False
        and native.get("objective_id") in UPSTREAM_OBJECTIVE_IDS
        and native.get("source") == source
        and native.get("seed") == 0
        and native.get("optimizer_steps") == 1
        and native.get("git_commit") == expected_commit
        and native.get("upstream_verl_commit") == UPSTREAM_VERL_COMMIT
        and native.get("finite_nonzero_gradient_observed") is True
        and native.get("parameter_update_observed") is True
        and native.get("optimizer_state_observed") is True
        and native.get("heldout_outcomes_inspected") is False,
        "native veRL diagnostic receipt failed",
    )
    preflight_binding = native.get("preflight") or {}
    preflight_path = Path(str(preflight_binding.get("path"))).resolve()
    _expect(
        preflight_path.is_file()
        and sha256_file(preflight_path) == preflight_binding.get("sha256"),
        "native veRL preflight binding drifted",
    )
    _, preflight = _json(preflight_path, "native veRL preflight", readonly=True)
    _expect(
        preflight.get("preflight") == PREFLIGHT_ID
        and preflight.get("campaign_kind") == "diagnostic"
        and preflight.get("source") == source
        and preflight.get("seed") == 0
        and preflight.get("optimizer_steps") == 1
        and preflight.get("git_commit") == expected_commit
        and preflight.get("heldout_outcomes_inspected") is False,
        "native veRL diagnostic preflight drifted",
    )
    scheduler_job_id = preflight.get("scheduler_job_id")
    _expect(
        isinstance(scheduler_job_id, str)
        and re.fullmatch(r"[1-9][0-9]*", scheduler_job_id) is not None,
        "native veRL scheduler identity drifted",
    )
    _binding_matches(native.get("run_log"), path_key="path", hash_key="sha256")
    _binding_matches(
        native.get("actor_checkpoint"),
        path_key="path",
        hash_key="tree_sha256",
        tree=True,
    )
    _binding_matches(
        native.get("final_adapter"),
        path_key="path",
        hash_key="tree_sha256",
        tree=True,
    )
    optimizer = native.get("optimizer") or {}
    _expect(
        isinstance(optimizer.get("tensors"), int)
        and optimizer["tensors"] > 0
        and isinstance(optimizer.get("elements"), int)
        and optimizer["elements"] > 0,
        "native veRL optimizer state is empty",
    )
    _finite_positive(optimizer.get("squared_l2"), "native veRL optimizer squared L2")
    _binding_matches(optimizer, path_key="path", hash_key="sha256")
    update = native.get("adapter_update") or {}
    _finite_positive(update.get("delta_l2"), "native veRL adapter delta")
    _finite_positive(update.get("delta_max_abs"), "native veRL maximum adapter delta")
    _expect(
        isinstance(update.get("changed_tensors"), int) and update["changed_tensors"] > 0,
        "native veRL adapter did not change",
    )
    metrics = native.get("metrics")
    _expect(isinstance(metrics, list) and len(metrics) == 1, "native veRL metric coverage drifted")
    _finite_positive(metrics[0].get("gradient_norm"), "native veRL gradient norm")
    _expect(
        type(metrics[0].get("distillation_loss")) in (int, float)
        and math.isfinite(float(metrics[0]["distillation_loss"])),
        "native veRL distillation loss is nonfinite",
    )
    rollouts = native.get("rollouts") or {}
    _expect(rollouts.get("rows") == 4, "native veRL rollout count drifted")
    for binding in rollouts.get("files") or []:
        _expect(
            isinstance(binding, dict)
            and Path(str(binding.get("path"))).is_file()
            and sha256_file(binding["path"]) == binding.get("sha256")
            and binding.get("rows") == 4,
            "native veRL rollout binding drifted",
        )
    rollout_files = rollouts.get("files") or []
    _expect(len(rollout_files) == 1, "native veRL one-step rollout file coverage drifted")
    rollout_root = Path(str(rollout_files[0]["path"])).resolve().parent
    _expect(
        sha256_tree(rollout_root) == rollouts.get("tree_sha256"),
        "native veRL rollout tree drifted",
    )
    output_root = Path(str(preflight.get("output_root"))).resolve()
    _expect(output_root.is_dir() and not output_root.is_symlink(), "native veRL run root is missing")
    stdout = _bind_completed_stdout(
        slurm_stdout,
        implementation="upstream_verl",
        scheduler_job_id=scheduler_job_id,
    )
    return {
        "schema_version": 1,
        "receipt": DIAGNOSTIC_RECEIPT_ID,
        "status": "passed_plumbing",
        "scientific_use_allowed": False,
        "training_artifact_eligible_for_held_out_evaluation": False,
        "implementation": "upstream_verl",
        "objective_id": next(iter(UPSTREAM_OBJECTIVE_IDS)),
        "source": source,
        "seed": 0,
        "git_commit": expected_commit,
        "objective_registry_sha256": load_objective_registry()["sha256"],
        "scheduler_job_id": scheduler_job_id,
        "run": {
            "root": str(output_root),
            "tree_sha256": sha256_tree(output_root),
            "native_receipt": str(receipt_path),
            "native_receipt_sha256": sha256_file(receipt_path),
            "final_adapter_tree_sha256": native["final_adapter"]["tree_sha256"],
        },
        "slurm_stdout": stdout,
        "checks": {
            "exact_one_step_four_rollouts": True,
            "finite_nonzero_gradient": True,
            "finite_nonzero_parameter_update": True,
            "finite_nonzero_optimizer_state": True,
            "exact_prompt_and_initialization_bound": True,
            "o_teacher_only_when_required": True,
            "heldout_outcomes_inspected": False,
        },
        "claim_boundary": "One-step native pinned-veRL execution fidelity only; no task-performance inference.",
    }


def validate_diagnostic_receipt(
    path: str | Path, *, objective_id: str, source: str, commit: str
) -> dict[str, Any]:
    receipt_path, payload = _json(path, "full-custody diagnostic receipt", readonly=True)
    expected_keys = {
        "schema_version",
        "receipt",
        "status",
        "scientific_use_allowed",
        "training_artifact_eligible_for_held_out_evaluation",
        "implementation",
        "objective_id",
        "source",
        "seed",
        "git_commit",
        "objective_registry_sha256",
        "scheduler_job_id",
        "run",
        "slurm_stdout",
        "checks",
        "claim_boundary",
    }
    _expect(set(payload) == expected_keys, "full-custody diagnostic receipt schema drifted")
    expected_implementation = "local" if objective_id in LOCAL_OBJECTIVE_IDS else "upstream_verl"
    _expect(
        payload.get("schema_version") == 1
        and payload.get("receipt") == DIAGNOSTIC_RECEIPT_ID
        and payload.get("status") == "passed_plumbing"
        and payload.get("scientific_use_allowed") is False
        and payload.get("training_artifact_eligible_for_held_out_evaluation") is False
        and payload.get("implementation") == expected_implementation
        and payload.get("objective_id") == objective_id
        and payload.get("source") == source
        and payload.get("seed") == 0
        and payload.get("git_commit") == commit
        and payload.get("objective_registry_sha256") == load_objective_registry()["sha256"],
        "full-custody diagnostic receipt identity drifted",
    )
    checks = payload.get("checks")
    _expect(
        isinstance(checks, dict)
        and set(checks)
        == {
            "exact_one_step_four_rollouts",
            "finite_nonzero_gradient",
            "finite_nonzero_parameter_update",
            "finite_nonzero_optimizer_state",
            "exact_prompt_and_initialization_bound",
            "o_teacher_only_when_required",
            "heldout_outcomes_inspected",
        }
        and all(checks[key] is True for key in checks if key != "heldout_outcomes_inspected")
        and checks["heldout_outcomes_inspected"] is False,
        "full-custody diagnostic checks drifted",
    )
    run = payload.get("run") or {}
    run_root = Path(str(run.get("root"))).resolve()
    _expect(
        run_root.is_dir()
        and not run_root.is_symlink()
        and sha256_tree(run_root) == run.get("tree_sha256"),
        "full-custody diagnostic run tree drifted",
    )
    stdout = payload.get("slurm_stdout") or {}
    stdout_path = Path(str(stdout.get("path")))
    _expect(
        stdout_path.is_file()
        and not stdout_path.is_symlink()
        and sha256_file(stdout_path) == stdout.get("sha256")
        and stdout.get("terminal_marker_observed") is True,
        "full-custody diagnostic Slurm stdout drifted",
    )
    return {"path": str(receipt_path), "sha256": sha256_file(receipt_path), "payload": payload}


def validate_real_model_receipt(path: str | Path, *, commit: str) -> dict[str, Any]:
    receipt_path, payload = _json(path, "real-model fidelity receipt", readonly=True)
    _expect(
        payload.get("schema_version") == 1
        and payload.get("check_id") == REAL_RECEIPT_ID
        and payload.get("status") == "pass"
        and payload.get("scientific_launch_authorized") is False,
        "real-model fidelity receipt identity drifted",
    )
    coverage = payload.get("coverage") or {}
    _expect(
        coverage.get("real_model_generated_rollout") is True
        and coverage.get("behavior_scores_from_generation") is True
        and isinstance(coverage.get("samples"), int)
        and coverage["samples"] == 4
        and isinstance(coverage.get("valid_tokens"), int)
        and coverage["valid_tokens"] > 0,
        "real-model fidelity coverage drifted",
    )
    comparison = payload.get("comparison") or {}
    required_checks = (
        "local_upstream_scalar_matches",
        "local_upstream_gradient_matches",
        "local_upstream_adamw_update_matches",
        "trace_reconstruction_matches",
        "on_policy_score_function_gradient_matches",
        "on_policy_score_function_gradient_cosine_pass",
        "masked_gradient_zero",
    )
    _expect(
        all(comparison.get(key) is True for key in required_checks),
        "real-model fidelity comparison did not fully pass",
    )
    custody = payload.get("custody") or {}
    local = custody.get("local") or {}
    upstream = custody.get("upstream_verl") or {}
    _expect(
        local.get("commit") == commit
        and local.get("tracked_status") == "clean"
        and upstream.get("commit") == UPSTREAM_VERL_COMMIT
        and upstream.get("tracked_status") == "clean",
        "real-model fidelity code custody drifted",
    )
    fixture_path = Path(str(local.get("fixture_path"))).resolve()
    _expect(
        fixture_path.is_file()
        and sha256_file(fixture_path) == local.get("fixture_sha256"),
        "real-model fidelity fixture binding drifted",
    )
    fixture = load_fixture(fixture_path)
    provenance = fixture.get("provenance") or {}
    _expect(
        fixture.get("fixture_id") == REAL_MODEL_FIXTURE_ID
        and provenance.get("local_git_commit") == commit
        and provenance.get("objective_registry_sha256") == load_objective_registry()["sha256"]
        and provenance.get("heldout_outcomes_inspected") is False,
        "real-model fidelity fixture provenance drifted",
    )
    return {"path": str(receipt_path), "sha256": sha256_file(receipt_path), "payload": payload}


def write_new(path: str | Path, payload: dict[str, Any]) -> Path:
    raw = Path(path).resolve()
    if raw.exists() or raw.is_symlink():
        raise FileExistsError(f"refusing to overwrite fidelity artifact: {raw}")
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(raw, 0o444)
    return raw


def build_closure(
    *,
    real_model_receipt: str | Path,
    diagnostic_receipts: dict[str, str | Path],
) -> dict[str, Any]:
    commit = _clean_commit()
    _expect(
        set(diagnostic_receipts) == set(EXPECTED_DIAGNOSTIC_KEYS),
        "fidelity closure requires exactly all 12 objective-source diagnostics",
    )
    registry = load_objective_registry()
    real = validate_real_model_receipt(real_model_receipt, commit=commit)
    diagnostics: dict[str, Any] = {}
    for key in EXPECTED_DIAGNOSTIC_KEYS:
        objective_id, source = key.rsplit("__", 1)
        receipt = validate_diagnostic_receipt(
            diagnostic_receipts[key],
            objective_id=objective_id,
            source=source,
            commit=commit,
        )
        diagnostics[key] = {
            "path": receipt["path"],
            "sha256": receipt["sha256"],
            "objective_id": objective_id,
            "source": source,
            "status": "passed_plumbing",
            "scientific_use_allowed": False,
        }
    return {
        "schema_version": 1,
        "closure": FIDELITY_CLOSURE_ID,
        "status": "passed",
        "all_levels_passed": True,
        "scientific_launch_authorized": False,
        "git_commit": commit,
        "objective_registry_sha256": registry["sha256"],
        "fidelity_plan": {"path": str(FIDELITY_PLAN.resolve()), "sha256": sha256_file(FIDELITY_PLAN)},
        "stored_real_model": {
            "path": real["path"],
            "sha256": real["sha256"],
            "status": "passed",
            "behavior_scores_from_generation": True,
        },
        "full_custody_diagnostics": diagnostics,
        "heldout_outcomes_inspected": False,
        "claim_boundary": (
            "All implementation-fidelity levels passed. This closure permits later "
            "outcome-blind preregistration only and is not a task-performance result."
        ),
    }


def _parse_receipt_bindings(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        key, separator, path = value.partition("=")
        _expect(separator == "=" and key and path and key not in result, "invalid diagnostic receipt binding")
        result[key] = path
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    local = commands.add_parser("audit-local")
    local.add_argument("--run-root", required=True)
    local.add_argument("--slurm-stdout", required=True)
    local.add_argument("--objective-id", choices=sorted(LOCAL_OBJECTIVE_IDS), required=True)
    local.add_argument("--source", choices=SOURCES, required=True)
    local.add_argument("--expected-commit", required=True)
    local.add_argument("--output", required=True)
    upstream = commands.add_parser("audit-upstream")
    upstream.add_argument("--native-receipt", required=True)
    upstream.add_argument("--slurm-stdout", required=True)
    upstream.add_argument("--source", choices=SOURCES, required=True)
    upstream.add_argument("--expected-commit", required=True)
    upstream.add_argument("--output", required=True)
    close = commands.add_parser("close")
    close.add_argument("--real-model-receipt", required=True)
    close.add_argument("--diagnostic-receipt", action="append", default=[])
    close.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "audit-local":
        payload = audit_local_diagnostic(
            run_root=args.run_root,
            slurm_stdout=args.slurm_stdout,
            objective_id=args.objective_id,
            source=args.source,
            expected_commit=args.expected_commit,
        )
    elif args.command == "audit-upstream":
        payload = audit_upstream_diagnostic(
            native_receipt=args.native_receipt,
            slurm_stdout=args.slurm_stdout,
            source=args.source,
            expected_commit=args.expected_commit,
        )
    else:
        payload = build_closure(
            real_model_receipt=args.real_model_receipt,
            diagnostic_receipts=_parse_receipt_bindings(args.diagnostic_receipt),
        )
    output = write_new(args.output, payload)
    print(json.dumps({"output": str(output), "sha256": sha256_file(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
