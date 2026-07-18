#!/usr/bin/env python3
"""Fail-closed held-out result gates and the primary six-run OPD matrix.

Training completion is only permission to evaluate an adapter.  This module
separately proves that a held-out evaluation belongs to an eligible scientific
student run, then combines exactly two task-RL baselines and four teacher-source
transfer arms.  Result authorization is deliberately independent of the sign
of an observed effect: harms and nulls are scientific results when custody is
valid.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import subprocess
from argparse import Namespace
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    from .data_contract import iter_jsonl
    from .math_reward import verify_completion
    from .quality_gates import (
        EVALUATION_MERGED_KIND,
        EXPECTED_TEACHER_TRAIN_PACKAGES,
        _prepared_role_binding,
        canonical_json_sha256,
        checked_evaluation,
        recompute_student_gate,
        recompute_teacher_gate,
        sha256_file,
        sha256_tree,
    )
except ImportError:
    from data_contract import iter_jsonl  # type: ignore
    from math_reward import verify_completion  # type: ignore
    from quality_gates import (  # type: ignore
        EVALUATION_MERGED_KIND,
        EXPECTED_TEACHER_TRAIN_PACKAGES,
        _prepared_role_binding,
        canonical_json_sha256,
        checked_evaluation,
        recompute_student_gate,
        recompute_teacher_gate,
        sha256_file,
        sha256_tree,
    )


ROOT = Path(__file__).resolve().parents[2]
CANONICAL_STUDENT_TRAINING_PLAN = (
    ROOT / "configs" / "opd_math" / "student_training_plan.json"
)
SCHEMA_VERSION = 1
STUDENT_HELDOUT_GATE = "student_heldout_result_v1"
MATRIX_READOUT = "opd_math_six_run_matrix_v1"
SOURCE_HOLDOUT_ROLE = "source_holdout"
TRAIN_ROLE = "student_opd"
SAMPLES_PER_PROBLEM = 4
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 0
HELDOUT_DECODING = {
    "thinking": False,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,
    "max_new_tokens": 512,
    "seed": 0,
}
EXPECTED_TRAIN_PACKAGES = dict(EXPECTED_TEACHER_TRAIN_PACKAGES)
EXPECTED_SERVE_PACKAGES = {
    "torch": "2.11.0",
    "transformers": "5.12.1",
    "peft": "0.19.1",
    "accelerate": "1.14.0",
    "requests": "2.32.5",
    "vllm": "0.24.0",
}
MATRIX_CONTRACT = {
    "baseline_M": {"objective": "task_rl", "student_source": "M", "teacher_source": None},
    "baseline_O": {"objective": "task_rl", "student_source": "O", "teacher_source": None},
    "M_M": {"objective": "task_rl_k1_gap", "student_source": "M", "teacher_source": "M"},
    "M_O": {"objective": "task_rl_k1_gap", "student_source": "O", "teacher_source": "M"},
    "O_M": {"objective": "task_rl_k1_gap", "student_source": "M", "teacher_source": "O"},
    "O_O": {"objective": "task_rl_k1_gap", "student_source": "O", "teacher_source": "O"},
}
PAIR_BY_KEY = {key: key for key in ("M_M", "M_O", "O_M", "O_O")}
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")


def _task_prompt_sha256(row: Mapping[str, Any]) -> str:
    prompt = row.get("prompt")
    if prompt is not None:
        if not isinstance(prompt, list):
            raise ValueError("registered training prompt must be conversational messages")
        return canonical_json_sha256(prompt)
    prompt_text = row.get("prompt_text")
    if not isinstance(prompt_text, str) or not prompt_text:
        raise ValueError("registered training row lacks a stable prompt identity")
    return canonical_json_sha256(prompt_text)


def _json_object(path: Path, label: str) -> dict[str, Any]:
    path = Path(path)
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _expect(payload: Mapping[str, Any], key: str, expected: Any, label: str) -> None:
    actual = payload.get(key)
    if actual != expected:
        raise ValueError(
            f"{label} {key} mismatch: expected={expected!r}, actual={actual!r}"
        )


def _absolute(raw: Any, anchor: Path, label: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{label} must be a non-empty path")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = anchor.parent / path
    return path.resolve()


def _hash_identity(value: Any, label: str, pattern: re.Pattern[str] = HEX64) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(f"{label} lacks a valid immutable hash identity")
    return value


def _clean_state(state: Any, commit: str, label: str) -> None:
    if not isinstance(state, dict):
        raise ValueError(f"{label} lacks a Git state")
    _expect(state, "commit", commit, label)
    _expect(state, "dirty", False, label)


def git_state() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain=v1"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return {"commit": commit, "dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": True}


def _result_builder_custody(expected_commit: str) -> dict[str, Any]:
    state = git_state()
    _clean_state(state, expected_commit, "student result builder")
    builder_path = Path(__file__).resolve()
    return {
        "git_state": state,
        "builder_relative_path": builder_path.relative_to(ROOT).as_posix(),
        "builder_file_sha256": sha256_file(builder_path),
    }


def _file_binding(path: Path, expected_hash: Any, label: str) -> str:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    actual = sha256_file(path)
    if actual != expected_hash:
        raise ValueError(f"{label} hash has drifted: {path}")
    return actual


def _manifest_augmented_payload(
    payload: Mapping[str, Any], path: Path, label: str
) -> dict[str, Any]:
    """Prove that an in-run payload is the exact content of its source file."""

    path = Path(path).resolve()
    declared_hash = _hash_identity(payload.get("manifest_sha256"), f"{label} manifest")
    _file_binding(path, declared_hash, f"{label} manifest")
    disk = _json_object(path, f"{label} manifest")
    embedded = dict(payload)
    embedded.pop("manifest_sha256", None)
    if disk != embedded:
        raise ValueError(f"embedded {label} differs from its exact manifest file")
    return disk


def _canonical_plan_binding(binding: Any) -> dict[str, Any]:
    plan = _json_object(CANONICAL_STUDENT_TRAINING_PLAN, "student training plan")
    if (
        plan.get("schema_version") != 1
        or plan.get("plan_id") != "opd_math_student_primary_pilot_v1"
        or plan.get("objectives") != ["task_rl", "task_rl_k1_gap"]
    ):
        raise ValueError("canonical student training plan has an unsupported identity")
    fixed = plan.get("fixed_config")
    if not isinstance(fixed, dict) or not fixed:
        raise ValueError("canonical student training plan lacks fixed_config")
    config_hash = canonical_json_sha256(fixed)
    expected = {
        "path": str(CANONICAL_STUDENT_TRAINING_PLAN.resolve()),
        "sha256": sha256_file(CANONICAL_STUDENT_TRAINING_PLAN),
        "plan_id": plan["plan_id"],
        "plan_config_sha256": config_hash,
        "actual_config_sha256": config_hash,
        "config": fixed,
        "compliant": True,
    }
    if binding != expected:
        raise ValueError("student run is not bound to the exact canonical training plan")
    return expected


def _freeze_versions(path: Path) -> dict[str, str]:
    versions: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        name, version = line.split("==", 1)
        versions[name.lower().replace("_", "-")] = version
    return versions


def _validate_freeze(
    binding: Any,
    *,
    commit: str,
    filename: str,
    expected_packages: Mapping[str, str],
    label: str,
) -> dict[str, Any]:
    if not isinstance(binding, dict):
        raise ValueError(f"{label} lacks an environment-freeze binding")
    path = _absolute(binding.get("path"), CANONICAL_STUDENT_TRAINING_PLAN, label)
    if path.name != filename or path.parent.name != commit:
        raise ValueError(f"{label} is not the commit-specific {filename}")
    digest = _file_binding(path, binding.get("sha256"), label)
    _expect(binding, "required_packages", dict(expected_packages), label)
    frozen = _freeze_versions(path)
    mismatch = {
        name: {"expected": version, "actual": frozen.get(name)}
        for name, version in expected_packages.items()
        if frozen.get(name) != version
    }
    if mismatch:
        raise ValueError(f"{label} package pins have drifted: {mismatch}")
    return {"path": str(path), "sha256": digest}


def _validate_environment(
    environment: Any, *, commit: str, requires_teacher: bool
) -> dict[str, Any]:
    if not isinstance(environment, dict):
        raise ValueError("student run lacks its scientific environment contract")
    _expect(environment, "schema_version", 1, "student environment")
    _expect(environment, "git_commit", commit, "student environment")
    _expect(
        environment,
        "train_runtime_packages",
        EXPECTED_TRAIN_PACKAGES,
        "student environment",
    )
    train = _validate_freeze(
        environment.get("train_freeze"),
        commit=commit,
        filename="train.freeze.txt",
        expected_packages=EXPECTED_TRAIN_PACKAGES,
        label="student train freeze",
    )
    serve_binding = environment.get("serve_freeze")
    if requires_teacher:
        serve = _validate_freeze(
            serve_binding,
            commit=commit,
            filename="serve.freeze.txt",
            expected_packages=EXPECTED_SERVE_PACKAGES,
            label="student serve freeze",
        )
    else:
        if serve_binding is not None:
            raise ValueError("task-RL baseline unexpectedly carries a teacher serve freeze")
        serve = None
    return {"train_freeze": train, "serve_freeze": serve}


def _gate_without_file_hash(gate: Mapping[str, Any]) -> dict[str, Any]:
    clean = dict(gate)
    clean.pop("manifest_sha256", None)
    return clean


def _validate_support_gate(
    gate: Any,
    *,
    source: str,
    model: str,
    revision: str,
    prepared_path: Path,
    prepared_hash: str,
) -> dict[str, Any]:
    if not isinstance(gate, dict):
        raise ValueError("student run lacks a student-support gate")
    _expect(gate, "schema_version", 3, "student-support gate")
    _expect(gate, "gate", "student_support_v1", "student-support gate")
    _expect(gate, "gate_strength", "scientific", "student-support gate")
    _expect(gate, "passed", True, "student-support gate")
    _expect(
        gate,
        "authorizes_scientific_training",
        True,
        "student-support gate",
    )
    _expect(gate, "student_model", model, "student-support gate")
    _expect(gate, "student_model_revision", revision, "student-support gate")
    _expect(gate, "task_sources", [source], "student-support gate")
    _expect(gate, "task_roles", [TRAIN_ROLE], "student-support gate")
    _expect(gate, "prepared_manifest", str(prepared_path), "student-support gate")
    _expect(gate, "prepared_manifest_sha256", prepared_hash, "student-support gate")
    _expect(gate, "decoding", HELDOUT_DECODING, "student-support gate")
    _expect(gate, "samples_per_problem", SAMPLES_PER_PROBLEM, "student-support gate")
    manifest_hash = _hash_identity(
        gate.get("manifest_sha256"), "student-support gate manifest"
    )
    original = _gate_without_file_hash(gate)
    recomputed = recompute_student_gate(original)
    if recomputed != original:
        changed = sorted(
            key
            for key in set(original) | set(recomputed)
            if original.get(key) != recomputed.get(key)
        )
        raise ValueError(
            "student-support gate differs from deterministic recomputation: "
            f"changed_fields={changed[:20]}"
        )
    return {
        "manifest_sha256": manifest_hash,
        "payload_sha256": canonical_json_sha256(original),
        "source": source,
    }


def _validate_teacher_identity(
    *,
    run: Mapping[str, Any],
    teacher_gate: Any,
    provenance: Any,
    tokenizer_contract: Any,
    server_contract: Any,
    teacher_source: str,
    student_model: str,
    student_revision: str,
    prepared_path: Path,
    prepared_hash: str,
    commit: str,
) -> dict[str, Any]:
    if not isinstance(teacher_gate, dict):
        raise ValueError("main arm lacks a teacher-gap gate")
    _expect(teacher_gate, "schema_version", 3, "teacher-gap gate")
    _expect(teacher_gate, "gate", "teacher_gap_v1", "teacher-gap gate")
    _expect(teacher_gate, "gate_strength", "scientific", "teacher-gap gate")
    _expect(teacher_gate, "passed", True, "teacher-gap gate")
    _expect(
        teacher_gate,
        "authorizes_scientific_merge",
        True,
        "teacher-gap gate",
    )
    _expect(teacher_gate, "task_sources", [teacher_source], "teacher-gap gate")
    _expect(teacher_gate, "task_roles", ["teacher_gap_dev"], "teacher-gap gate")
    _expect(teacher_gate, "prepared_manifest", str(prepared_path), "teacher-gap gate")
    _expect(teacher_gate, "prepared_manifest_sha256", prepared_hash, "teacher-gap gate")
    gate_file_hash = _hash_identity(
        teacher_gate.get("manifest_sha256"), "teacher-gap gate manifest"
    )
    gate_disk_payload = _gate_without_file_hash(teacher_gate)
    if recompute_teacher_gate(gate_disk_payload) != gate_disk_payload:
        raise ValueError("teacher-gap gate differs from deterministic recomputation")

    if not isinstance(provenance, dict):
        raise ValueError("main arm lacks merged-teacher provenance")
    _expect(provenance, "schema_version", 1, "teacher provenance")
    _expect(provenance, "schema", "opd_math_merged_teacher_v2", "teacher provenance")
    _expect(provenance, "status", "completed", "teacher provenance")
    checkpoint = _absolute(
        provenance.get("output_checkpoint"), CANONICAL_STUDENT_TRAINING_PLAN, "teacher checkpoint"
    )
    _expect(run, "teacher_checkpoint", str(checkpoint), "student run")
    provenance_path = checkpoint / "merge_provenance.json"
    provenance_disk = _manifest_augmented_payload(
        provenance, provenance_path, "teacher provenance"
    )
    checkpoint_hash = sha256_tree(
        checkpoint, exclude_relative_paths=("merge_provenance.json",)
    )
    _expect(
        provenance,
        "output_checkpoint_tree_sha256",
        checkpoint_hash,
        "teacher provenance",
    )
    teacher_gate_path = _absolute(
        provenance.get("teacher_gap_manifest"), provenance_path, "teacher-gap manifest"
    )
    _expect(
        provenance,
        "teacher_gap_manifest_sha256",
        gate_file_hash,
        "teacher provenance",
    )
    _manifest_augmented_payload(teacher_gate, teacher_gate_path, "teacher-gap gate")
    _expect(provenance, "prepared_manifest", str(prepared_path), "teacher provenance")
    _expect(
        provenance,
        "prepared_manifest_sha256",
        prepared_hash,
        "teacher provenance",
    )
    _expect(provenance, "base_model", run.get("teacher_base_model"), "teacher provenance")
    _expect(
        provenance,
        "base_revision",
        run.get("teacher_base_revision"),
        "teacher provenance",
    )
    _expect(
        provenance,
        "adapter_tree_sha256",
        teacher_gate.get("trained_adapter_tree_sha256"),
        "teacher provenance",
    )
    merge_code = provenance.get("merge_code")
    if not isinstance(merge_code, dict):
        raise ValueError("teacher provenance lacks merge-code custody")
    for field in (
        "git_state_start",
        "git_state_after_merge",
        "git_state_before_promotion",
        "git_state_end",
    ):
        _clean_state(merge_code.get(field), commit, f"teacher merge {field}")
    _expect(merge_code, "clean_stable_code", True, "teacher merge")
    _expect(
        merge_code,
        "packages",
        {name: EXPECTED_TRAIN_PACKAGES[name] for name in ("torch", "transformers", "peft")},
        "teacher merge",
    )

    if not isinstance(tokenizer_contract, dict):
        raise ValueError("main arm lacks a tokenizer contract")
    _hash_identity(
        tokenizer_contract.get("manifest_sha256"), "tokenizer-contract manifest"
    )
    _expect(tokenizer_contract, "gate", "tokenizer_contract_v1", "tokenizer contract")
    _expect(tokenizer_contract, "passed", True, "tokenizer contract")
    _expect(tokenizer_contract, "exact_contract_match", True, "tokenizer contract")
    _expect(
        tokenizer_contract.get("student") or {},
        "model",
        student_model,
        "tokenizer contract student",
    )
    _expect(
        tokenizer_contract.get("student") or {},
        "revision",
        student_revision,
        "tokenizer contract student",
    )
    _expect(
        tokenizer_contract.get("teacher") or {},
        "model",
        str(checkpoint),
        "tokenizer contract teacher",
    )
    if not (tokenizer_contract.get("server_probe") or {}).get("matches"):
        raise ValueError("tokenizer contract lacks a passing server probe")

    if not isinstance(server_contract, dict):
        raise ValueError("main arm lacks an exact-token server-scoring contract")
    _hash_identity(server_contract.get("manifest_sha256"), "server-scoring manifest")
    _expect(server_contract, "schema_version", 2, "server-scoring contract")
    _expect(
        server_contract,
        "probe",
        "exact_token_teacher_scoring_v1",
        "server-scoring contract",
    )
    _expect(server_contract, "passed", True, "server-scoring contract")
    _expect(server_contract, "tokenizer", student_model, "server-scoring contract")
    _expect(
        server_contract,
        "tokenizer_revision",
        student_revision,
        "server-scoring contract",
    )
    _expect(server_contract, "server_model", run.get("teacher_model"), "server-scoring contract")
    _expect(
        server_contract,
        "local_process_binding_validated",
        True,
        "server-scoring contract",
    )
    local = server_contract.get("local_process_binding")
    if not isinstance(local, dict):
        raise ValueError("server-scoring contract lacks local process binding")
    _expect(
        local,
        "scope",
        "local_linux_proc_process_binding_not_remote_cryptographic_attestation",
        "server local-process binding",
    )
    _expect(local, "validated", True, "server local-process binding")
    _expect(
        local,
        "teacher_checkpoint",
        str(checkpoint),
        "server local-process binding",
    )
    _expect(
        local,
        "teacher_provenance_manifest",
        str(provenance_path),
        "server local-process binding",
    )
    _expect(
        local,
        "teacher_checkpoint_tree_sha256",
        checkpoint_hash,
        "server local-process binding",
    )
    _expect(
        local,
        "teacher_provenance_manifest_sha256",
        provenance.get("manifest_sha256"),
        "server local-process binding",
    )
    tokenizer_server = tokenizer_contract.get("server")
    if not isinstance(tokenizer_server, dict):
        raise ValueError("tokenizer contract lacks its live server identity")
    _expect(
        tokenizer_server,
        "url",
        server_contract.get("server_url"),
        "tokenizer/server-scoring identity",
    )
    _expect(
        tokenizer_server,
        "model",
        server_contract.get("server_model"),
        "tokenizer/server-scoring identity",
    )

    return {
        "teacher_source": teacher_source,
        "base_model": run.get("teacher_base_model"),
        "base_revision": run.get("teacher_base_revision"),
        "teacher_gap_manifest_sha256": gate_file_hash,
        "teacher_gap_payload_sha256": canonical_json_sha256(gate_disk_payload),
        "merged_checkpoint": str(checkpoint),
        "merged_checkpoint_tree_sha256": checkpoint_hash,
        "merge_provenance_manifest_sha256": provenance.get("manifest_sha256"),
        "merge_provenance_payload_sha256": canonical_json_sha256(provenance_disk),
        "tokenizer_contract_manifest_sha256": tokenizer_contract.get("manifest_sha256"),
        "tokenizer_contract_payload_sha256": canonical_json_sha256(
            _gate_without_file_hash(tokenizer_contract)
        ),
        "server_scoring_manifest_sha256": server_contract.get("manifest_sha256"),
        "server_scoring_payload_sha256": canonical_json_sha256(
            _gate_without_file_hash(server_contract)
        ),
    }


def _validate_trace_artifacts(
    *,
    run: Mapping[str, Any],
    completion: Mapping[str, Any],
    completion_path: Path,
    training_rows: list[dict[str, Any]],
    objective: str,
    source: str,
    fixed: Mapping[str, Any],
) -> dict[str, Any]:
    artifacts = completion.get("trace_artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {"steps.jsonl", "samples.jsonl"}:
        raise ValueError("completion manifest lacks the exact two scientific trace artifacts")
    paths: dict[str, Path] = {}
    for name in ("steps.jsonl", "samples.jsonl"):
        item = artifacts[name]
        if not isinstance(item, dict):
            raise ValueError(f"trace artifact binding is invalid: {name}")
        path = _absolute(item.get("path"), completion_path, f"trace {name}")
        if path != (completion_path.parent / name).resolve():
            raise ValueError(f"trace artifact is not the canonical sibling: {name}")
        _file_binding(path, item.get("sha256"), f"trace {name}")
        physical_rows = sum(1 for _ in iter_jsonl(path))
        _expect(item, "rows", physical_rows, f"trace {name}")
        paths[name] = path

    steps = list(iter_jsonl(paths["steps.jsonl"]))
    expected_steps = int(fixed["optimizer_steps"])
    if len(steps) != expected_steps:
        raise ValueError("step trace does not contain the exact optimizer-step budget")
    gradients: list[float] = []
    for expected_step, row in enumerate(steps, 1):
        _expect(row, "schema_version", 1, "step trace")
        _expect(row, "step", expected_step, "step trace")
        _expect(row, "mode", objective, "step trace")
        _expect(row, "prompts", int(fixed["micro_prompts"]), "step trace")
        _expect(
            row,
            "samples",
            int(fixed["micro_prompts"]) * int(fixed["group_size"]),
            "step trace",
        )
        for field in ("total_loss", "gradient_norm_before_clip"):
            value = row.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"step trace has non-finite {field}")
        gradients.append(float(row["gradient_norm_before_clip"]))
    if not any(value > 0 for value in gradients):
        raise ValueError("step trace contains no finite nonzero gradient")

    selected_rows: dict[str, dict[str, str]] = {}
    for row in training_rows:
        record_id = row.get("record_id")
        solution = row.get("solution")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("selected training row lacks record_id")
        if record_id in selected_rows:
            raise ValueError("selected training rows contain duplicate record IDs")
        if not isinstance(solution, str) or not solution.strip():
            raise ValueError("selected training row lacks a gold solution")
        selected_rows[record_id] = {
            "solution": solution,
            "prompt_sha256": _task_prompt_sha256(row),
        }

    sample_rows = list(iter_jsonl(paths["samples.jsonl"]))
    expected_samples = (
        expected_steps * int(fixed["micro_prompts"]) * int(fixed["group_size"])
    )
    if len(sample_rows) != expected_samples:
        raise ValueError("sample trace does not contain the exact rollout-sample budget")
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    scored_tokens = 0
    sample_expanded_prompt_tokens = 0
    teacher_latency = 0.0
    for row_number, row in enumerate(sample_rows, 1):
        _expect(row, "schema_version", 1, "sample trace")
        step = row.get("step")
        group_id = row.get("group_id")
        sample_idx = row.get("sample_idx")
        if not isinstance(step, int) or not 1 <= step <= expected_steps:
            raise ValueError(f"sample trace has invalid step at row {row_number}")
        if not isinstance(group_id, int) or not 0 <= group_id < int(fixed["micro_prompts"]):
            raise ValueError(f"sample trace has invalid group_id at row {row_number}")
        if not isinstance(sample_idx, int) or not 0 <= sample_idx < int(fixed["group_size"]):
            raise ValueError(f"sample trace has invalid sample_idx at row {row_number}")
        _expect(row, "source", source, "sample trace")
        record_id = row.get("record_id")
        if record_id not in selected_rows:
            raise ValueError(f"sample trace uses an unregistered training record: {record_id!r}")
        registered = selected_rows[str(record_id)]
        _expect(
            row,
            "prompt_sha256",
            registered["prompt_sha256"],
            "sample trace",
        )
        completion_text = row.get("completion_text")
        if not isinstance(completion_text, str):
            raise ValueError("sample trace lacks completion_text")
        _expect(
            row,
            "completion_sha256",
            hashlib.sha256(completion_text.encode("utf-8")).hexdigest(),
            "sample trace",
        )
        prompt_ids = row.get("prompt_token_ids")
        completion_ids = row.get("completion_token_ids")
        if (
            not isinstance(prompt_ids, list)
            or not prompt_ids
            or any(type(x) is not int or x < 0 for x in prompt_ids)
            or len(prompt_ids) > int(fixed["max_prompt_tokens"])
        ):
            raise ValueError("sample trace lacks exact prompt token IDs")
        if (
            not isinstance(completion_ids, list)
            or not completion_ids
            or any(type(x) is not int or x < 0 for x in completion_ids)
            or len(completion_ids) > int(fixed["max_new_tokens"])
        ):
            raise ValueError("sample trace lacks exact completion token IDs")
        _expect(row, "prompt_tokens", len(prompt_ids), "sample trace")
        _expect(row, "completion_tokens", len(completion_ids), "sample trace")
        sample_expanded_prompt_tokens += len(prompt_ids)
        verdict = verify_completion(completion_text, registered["solution"])
        if verdict.get("status") in {"gold_parse_failed", "verifier_error"}:
            raise RuntimeError(f"training trace verifier failure: {verdict}")
        _expect(row, "reward_status", verdict.get("status"), "sample trace")
        _expect(row, "reward", float(verdict.get("reward")), "sample trace")
        if objective == "task_rl":
            for field in (
                "teacher_nll_on_student_trajectory",
                "mean_teacher_student_gap",
                "mean_abs_k1_log_ratio",
                "min_teacher_student_gap",
                "max_teacher_student_gap",
                "positive_teacher_gap_fraction",
            ):
                _expect(row, field, None, "task-RL sample trace")
        else:
            for field in (
                "teacher_nll_on_student_trajectory",
                "mean_teacher_student_gap",
                "mean_abs_k1_log_ratio",
            ):
                value = row.get(field)
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    raise ValueError(f"main-arm sample trace lacks finite {field}")
        scored_tokens += len(completion_ids)
        latency = row.get("teacher_scoring_latency_seconds")
        if latency is not None:
            if not isinstance(latency, (int, float)) or not math.isfinite(float(latency)):
                raise ValueError("sample trace has invalid teacher-scoring latency")
            teacher_latency += float(latency)
        grouped[(step, group_id)].append(row)

    expected_groups = expected_steps * int(fixed["micro_prompts"])
    if len(grouped) != expected_groups:
        raise ValueError("sample trace has missing or duplicate prompt groups")
    informative_groups = 0
    informative_steps: set[int] = set()
    realized_ids: list[str] = []
    realized_prompt_sequence: list[dict[str, str]] = []
    prompt_tokens = 0
    rollout_latency = 0.0
    for key in sorted(grouped):
        rows = sorted(grouped[key], key=lambda row: row["sample_idx"])
        if [row["sample_idx"] for row in rows] != list(range(int(fixed["group_size"]))):
            raise ValueError(f"sample trace group has duplicate or missing samples: {key}")
        if len({row["record_id"] for row in rows}) != 1:
            raise ValueError(f"sample trace group mixes record IDs: {key}")
        if len({tuple(row["prompt_token_ids"]) for row in rows}) != 1:
            raise ValueError(f"sample trace group mixes prompt token IDs: {key}")
        if len({row["prompt_sha256"] for row in rows}) != 1:
            raise ValueError(f"sample trace group mixes prompt identity: {key}")
        rewards = [float(row["reward"]) for row in rows]
        if any(value != rewards[0] for value in rewards[1:]):
            informative_groups += 1
            informative_steps.add(key[0])
        realized_ids.append(rows[0]["record_id"])
        realized_prompt_sequence.append(
            {
                "record_id": rows[0]["record_id"],
                "prompt_sha256": rows[0]["prompt_sha256"],
            }
        )
        prompt_tokens += len(rows[0]["prompt_token_ids"])
        latency = rows[0].get("rollout_batch_latency_seconds")
        if not isinstance(latency, (int, float)) or not math.isfinite(float(latency)):
            raise ValueError("sample trace has invalid rollout latency")
        rollout_latency += float(latency)

    expected_fraction = informative_groups / expected_groups
    _expect(
        completion,
        "minimum_informative_group_fraction",
        fixed["min_informative_group_fraction"],
        "student completion",
    )
    if expected_fraction < float(fixed["min_informative_group_fraction"]):
        raise ValueError("training trace does not satisfy the predeclared task-signal gate")
    checks = {
        "optimizer_steps_completed": expected_steps,
        "rollout_samples": expected_samples,
        "scored_completion_tokens": scored_tokens,
        "prompt_group_tokens": prompt_tokens,
        "sample_expanded_prompt_tokens": sample_expanded_prompt_tokens,
        "prompt_groups_seen": expected_groups,
        "step_trace_rows": expected_steps,
        "sample_trace_rows": expected_samples,
        "realized_training_geometry_observed": True,
        "unique_training_records": len(set(realized_ids)),
        "realized_record_ids_sha256": canonical_json_sha256(realized_ids),
        "realized_prompt_sequence_sha256": canonical_json_sha256(
            realized_prompt_sequence
        ),
        "informative_task_steps": len(informative_steps),
        "informative_task_groups": informative_groups,
        "total_task_groups": expected_groups,
        "informative_group_fraction": expected_fraction,
    }
    for field, expected in checks.items():
        actual = completion.get(field)
        if isinstance(expected, float):
            if not isinstance(actual, (int, float)) or not math.isclose(
                float(actual), expected, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(f"completion {field} differs from trace recomputation")
        else:
            _expect(completion, field, expected, "student completion")
    for field, expected in (
        ("total_rollout_latency_seconds", rollout_latency),
        ("total_teacher_scoring_latency_seconds", teacher_latency),
    ):
        actual = completion.get(field)
        if not isinstance(actual, (int, float)) or not math.isclose(
            float(actual), expected, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError(f"completion {field} differs from trace recomputation")
    return {
        "steps": expected_steps,
        "samples": expected_samples,
        "prompt_groups": expected_groups,
        "prompt_group_tokens": prompt_tokens,
        "sample_expanded_prompt_tokens": sample_expanded_prompt_tokens,
        "realized_record_ids_sha256": canonical_json_sha256(realized_ids),
        "realized_prompt_sequence_sha256": canonical_json_sha256(
            realized_prompt_sequence
        ),
        "informative_group_fraction": expected_fraction,
        "steps_sha256": artifacts["steps.jsonl"]["sha256"],
        "samples_sha256": artifacts["samples.jsonl"]["sha256"],
    }


def _validate_student_run(
    *,
    matrix_key: str,
    run_path: Path,
    completion_path: Path,
    adapter: Path,
    prepared_path: Path,
    prepared: Mapping[str, Any],
    prepared_hash: str,
    student_model: str,
    student_revision: str,
    evaluation_git_commit: str,
) -> dict[str, Any]:
    contract = MATRIX_CONTRACT[matrix_key]
    objective = str(contract["objective"])
    source = str(contract["student_source"])
    teacher_source = contract["teacher_source"]
    run_path = Path(run_path).resolve()
    completion_path = Path(completion_path).resolve()
    if run_path.name != "run_manifest.json" or completion_path.name != "completion_manifest.json":
        raise ValueError("student result gate requires canonical run/completion filenames")
    if run_path.parent != completion_path.parent:
        raise ValueError("student run and completion manifests must be siblings")
    run = _json_object(run_path, "student run manifest")
    completion = _json_object(completion_path, "student completion manifest")
    if run.get("completion") != completion:
        raise ValueError("student run does not embed the exact completion manifest")
    for payload, label in ((run, "student run"), (completion, "student completion")):
        _expect(payload, "schema_version", 1, label)
        _expect(payload, "status", "completed", label)
        _expect(payload, "objective", objective, label)
        _expect(payload, "intended_scientific_run", True, label)
        _expect(
            payload,
            "training_artifact_eligible_for_held_out_evaluation",
            True,
            label,
        )
    for field in (
        "task_signal_observed",
        "finite_nonzero_gradient_observed",
        "parameter_update_observed",
        "clean_stable_code",
        "stable_training_environment",
        "stable_environment_after_candidate_save",
        "stable_environment_end",
        "stable_final_artifact_hash",
    ):
        _expect(completion, field, True, "student completion")
    if completion.get("initial_parameter_signature") == completion.get(
        "final_parameter_signature"
    ):
        raise ValueError("student completion does not show a parameter-signature change")

    commit = run.get("git_commit")
    _hash_identity(commit, "student training Git commit", HEX40)
    _expect(run, "git_worktree_clean", True, "student run")
    _clean_state(run.get("git_state_start"), commit, "student run git_state_start")
    for field in (
        "git_state_start",
        "git_state_training_end",
        "git_state_after_candidate_save",
        "git_state_end",
    ):
        _clean_state(completion.get(field), commit, f"student completion {field}")
    if commit != evaluation_git_commit:
        raise ValueError("held-out evaluation and student training use different Git commits")

    _expect(run, "student", student_model, "student run")
    _expect(run, "student_revision", student_revision, "student run")
    binding = run.get("binding")
    if not isinstance(binding, dict):
        raise ValueError("student run lacks its normalized binding")
    plan_binding = _canonical_plan_binding(binding.get("student_training_plan"))
    fixed = plan_binding["config"]
    _expect(run, "optimizer_steps_planned", fixed["optimizer_steps"], "student run")
    _expect(run, "micro_prompts_per_step", fixed["micro_prompts"], "student run")
    _expect(
        run,
        "planned_rollout_samples",
        fixed["optimizer_steps"] * fixed["micro_prompts"] * fixed["group_size"],
        "student run",
    )
    _expect(run, "seed", fixed["seed"], "student run")
    expected_generation = {
        "group_size": fixed["group_size"],
        "temperature": fixed["temperature"],
        "top_p": fixed["top_p"],
        "top_k": fixed["top_k"],
        "max_new_tokens": fixed["max_new_tokens"],
        "enable_thinking": fixed["enable_thinking"],
    }
    _expect(run, "generation", expected_generation, "student run")
    expected_optimization = {
        "attn_implementation": fixed["attn_implementation"],
        "gradient_checkpointing": fixed["gradient_checkpointing"],
        "learning_rate": fixed["learning_rate"],
        "lora_r": fixed["lora_r"],
    }
    _expect(run, "optimization", expected_optimization, "student run")
    expected_loss = {
        "task_reward_coef": fixed["task_reward_coef"],
        "k1_coef": fixed["k1_coef"],
        "gap_gate_beta": fixed["gap_gate_beta"],
        "advantage_clip": fixed["advantage_clip"],
    }
    _expect(run, "loss", expected_loss, "student run")

    _expect(binding, "student_source", source, "student run binding")
    _expect(binding, "teacher_source", teacher_source, "student run binding")
    _expect(binding, "budget_mode", "primary_matched", "student run binding")
    expected_pair = PAIR_BY_KEY.get(matrix_key)
    _expect(binding, "pair_id", expected_pair, "student run binding")
    train_relative = f"roles/{source}/{TRAIN_ROLE}.jsonl"
    _expect(binding, "task_role_file", train_relative, "student run binding")
    matched_budget = prepared.get("primary_matched_budgets", {}).get(TRAIN_ROLE)
    if not isinstance(matched_budget, int) or matched_budget <= 0:
        raise ValueError("prepared manifest lacks a positive matched student budget")
    _expect(binding, "matched_task_limit", matched_budget, "student run binding")
    _expect(run, "task_limit", matched_budget, "student run")
    _expect(run, "selected_task_rows", matched_budget, "student run")
    train_path = (prepared_path.parent / train_relative).resolve()
    _expect(run, "task_file", str(train_path), "student run")
    train_entry = prepared.get("files", {}).get(train_relative)
    if not isinstance(train_entry, dict):
        raise ValueError("prepared manifest lacks the student training role file")
    train_hash = _file_binding(train_path, train_entry.get("sha256"), "student train role")
    _expect(run, "task_file_sha256", train_hash, "student run")
    full_training_rows = list(iter_jsonl(train_path))
    _expect(binding, "task_file_rows", len(full_training_rows), "student run binding")
    if matched_budget > len(full_training_rows):
        raise ValueError("matched student budget exceeds the registered role file")
    training_rows = full_training_rows[:matched_budget]
    if any(row.get("source") != source or row.get("role") != TRAIN_ROLE for row in training_rows):
        raise ValueError("selected student training rows violate source/role custody")

    gates = run.get("gates")
    if not isinstance(gates, dict):
        raise ValueError("student run lacks scientific gate identities")
    prepared_gate = gates.get("prepared_data")
    expected_prepared_gate = {
        "path": str(prepared_path),
        "sha256": prepared_hash,
        "task_role_file": train_relative,
        "task_file_sha256": train_hash,
        "scientific_use_allowed": True,
    }
    if prepared_gate != expected_prepared_gate:
        raise ValueError("student run prepared-data binding has drifted")
    support_identity = _validate_support_gate(
        gates.get("student_support"),
        source=source,
        model=student_model,
        revision=student_revision,
        prepared_path=prepared_path,
        prepared_hash=prepared_hash,
    )
    environment_identity = _validate_environment(
        binding.get("environment_contract"),
        commit=commit,
        requires_teacher=teacher_source is not None,
    )

    if teacher_source is None:
        for field in ("teacher_gap", "teacher_provenance", "server_scoring_contract", "tokenizer_contract"):
            _expect(gates, field, None, "task-RL baseline gates")
        for field in (
            "teacher_model",
            "teacher_checkpoint",
            "teacher_base_model",
            "teacher_base_revision",
        ):
            _expect(run, field, None, "task-RL baseline")
        _expect(
            completion,
            "local_server_process_binding_required",
            False,
            "task-RL completion",
        )
        teacher_identity = None
    else:
        _expect(
            completion,
            "local_server_process_binding_required",
            True,
            "main-arm completion",
        )
        _expect(
            completion,
            "live_local_server_process_binding_validated",
            True,
            "main-arm completion",
        )
        _expect(
            binding,
            "local_checkpoint_custody_validated",
            True,
            "main-arm binding",
        )
        _expect(
            binding,
            "server_alias_and_token_contract_validated",
            True,
            "main-arm binding",
        )
        _expect(
            completion,
            "local_server_process_binding_error",
            None,
            "main-arm completion",
        )
        server_contract = gates.get("server_scoring_contract")
        if not isinstance(server_contract, dict) or not isinstance(
            server_contract.get("local_process_binding"), dict
        ):
            raise ValueError("main-arm run lacks its initial local process binding")
        _expect(
            completion,
            "local_server_process_binding_end",
            server_contract["local_process_binding"],
            "main-arm completion",
        )
        teacher_identity = _validate_teacher_identity(
            run=run,
            teacher_gate=gates.get("teacher_gap"),
            provenance=gates.get("teacher_provenance"),
            tokenizer_contract=gates.get("tokenizer_contract"),
            server_contract=gates.get("server_scoring_contract"),
            teacher_source=str(teacher_source),
            student_model=student_model,
            student_revision=student_revision,
            prepared_path=prepared_path,
            prepared_hash=prepared_hash,
            commit=commit,
        )

    adapter = Path(adapter).resolve()
    final_adapter = _absolute(
        completion.get("final_adapter"), completion_path, "student final adapter"
    )
    if final_adapter != adapter:
        raise ValueError("supplied student adapter differs from the eligible final adapter")
    adapter_hash = sha256_tree(adapter)
    _expect(
        completion,
        "final_adapter_tree_sha256",
        adapter_hash,
        "student completion",
    )
    trace_identity = _validate_trace_artifacts(
        run=run,
        completion=completion,
        completion_path=completion_path,
        training_rows=training_rows,
        objective=objective,
        source=source,
        fixed=fixed,
    )
    return {
        "run_manifest": str(run_path),
        "run_manifest_sha256": sha256_file(run_path),
        "completion_manifest": str(completion_path),
        "completion_manifest_sha256": sha256_file(completion_path),
        "objective": objective,
        "student_source": source,
        "teacher_source": teacher_source,
        "git_commit": commit,
        "student_training_plan_sha256": plan_binding["sha256"],
        "student_training_config_sha256": plan_binding["actual_config_sha256"],
        "student_adapter": str(adapter),
        "student_adapter_tree_sha256": adapter_hash,
        "student_support": support_identity,
        "teacher": teacher_identity,
        "environment": environment_identity,
        "trace": trace_identity,
    }


def student_heldout_result(args: Namespace) -> dict[str, Any]:
    if args.matrix_key not in MATRIX_CONTRACT:
        raise ValueError(f"unknown matrix key: {args.matrix_key!r}")
    expected = MATRIX_CONTRACT[args.matrix_key]
    if args.task_source != expected["student_source"]:
        raise ValueError("matrix key and held-out task source disagree")
    prepared_path = Path(args.prepared_manifest).resolve()
    prepared = _json_object(prepared_path, "prepared manifest")
    if prepared.get("schema_version") != 1 or prepared.get("scientific_use_allowed") is not True:
        raise ValueError("held-out results require scientific prepared data")
    prepared_state = prepared.get("code_git_state")
    if not isinstance(prepared_state, dict) or prepared_state.get("dirty") is not False:
        raise ValueError("held-out results require clean prepared-data Git custody")
    _hash_identity(prepared_state.get("commit"), "prepared-data Git commit", HEX40)
    prepared_hash = sha256_file(prepared_path)

    summary, grouped, evaluation_binding = checked_evaluation(
        args.student_summary,
        args.student_samples,
        expected_model=args.student_model,
        expected_revision=args.student_revision,
        expected_source=args.task_source,
        expected_role=SOURCE_HOLDOUT_ROLE,
    )
    if evaluation_binding.get("evaluation_artifact_kind") != EVALUATION_MERGED_KIND:
        raise ValueError(
            "scientific held-out evaluation requires a schema-v2 merged artifact"
        )
    if evaluation_binding["samples_per_problem"] != SAMPLES_PER_PROBLEM:
        raise ValueError("scientific held-out evaluation requires exactly four samples per record")
    if summary.get("decoding") != HELDOUT_DECODING:
        raise ValueError(
            "scientific held-out decoding differs from the predeclared 4x non-thinking contract"
        )
    task_path = Path(evaluation_binding["task_file"])
    prepared_checked, prepared_binding = _prepared_role_binding(
        prepared_path,
        source=args.task_source,
        role=SOURCE_HOLDOUT_ROLE,
        task_file=task_path,
        selected_records=len(grouped),
        strength="scientific",
        model_kind="student",
        model=args.student_model,
        revision=args.student_revision,
    )
    if prepared_checked != prepared:
        raise ValueError("prepared manifest changed during held-out gate construction")

    adapter = Path(args.trained_adapter).resolve()
    evaluated_adapter = _absolute(
        summary.get("adapter"), Path(args.student_summary).resolve(), "evaluation adapter"
    )
    if evaluated_adapter != adapter:
        raise ValueError("held-out evaluation used a different student adapter")
    adapter_hash = sha256_tree(adapter)
    _expect(summary, "adapter_tree_sha256", adapter_hash, "held-out evaluation")
    run_binding = _validate_student_run(
        matrix_key=args.matrix_key,
        run_path=Path(args.student_run_manifest),
        completion_path=Path(args.student_completion_manifest),
        adapter=adapter,
        prepared_path=prepared_path,
        prepared=prepared,
        prepared_hash=prepared_hash,
        student_model=args.student_model,
        student_revision=args.student_revision,
        evaluation_git_commit=evaluation_binding["evaluation_git_commit"],
    )
    result_builder = _result_builder_custody(run_binding["git_commit"])
    record_rewards = {key: list(grouped[key]) for key in sorted(grouped)}
    record_accuracy = {
        key: sum(record_rewards[key]) / len(record_rewards[key]) for key in record_rewards
    }
    accuracy = sum(record_accuracy.values()) / len(record_accuracy)
    if not math.isclose(float(summary["accuracy"]), accuracy, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("held-out accuracy differs from record-level reward recomputation")
    inputs = {
        "matrix_key": args.matrix_key,
        "student_run_manifest": str(Path(args.student_run_manifest).resolve()),
        "student_run_manifest_sha256": run_binding["run_manifest_sha256"],
        "student_completion_manifest": str(Path(args.student_completion_manifest).resolve()),
        "student_completion_manifest_sha256": run_binding["completion_manifest_sha256"],
        "student_summary": str(Path(args.student_summary).resolve()),
        "student_summary_sha256": sha256_file(Path(args.student_summary)),
        "student_samples": str(Path(args.student_samples).resolve()),
        "student_samples_sha256": sha256_file(Path(args.student_samples)),
        "trained_adapter": str(adapter),
        "trained_adapter_tree_sha256": adapter_hash,
        "prepared_manifest": str(prepared_path),
        "prepared_manifest_sha256": prepared_hash,
        "student_model": args.student_model,
        "student_revision": args.student_revision,
        "task_source": args.task_source,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "gate": STUDENT_HELDOUT_GATE,
        "matrix_key": args.matrix_key,
        "passed": True,
        "authorizes_scientific_matrix_readout": True,
        "authorization_is_independent_of_effect_sign": True,
        "objective": run_binding["objective"],
        "student_source": args.task_source,
        "teacher_source": run_binding["teacher_source"],
        "student_model": args.student_model,
        "student_model_revision": args.student_revision,
        "records": len(record_rewards),
        "samples_per_problem": SAMPLES_PER_PROBLEM,
        "samples": len(record_rewards) * SAMPLES_PER_PROBLEM,
        "accuracy": accuracy,
        "record_rewards": record_rewards,
        "record_accuracy": record_accuracy,
        "record_ids_sha256": canonical_json_sha256(sorted(record_rewards)),
        "decoding": HELDOUT_DECODING,
        "prepared_binding": prepared_binding,
        "evaluation_binding": evaluation_binding,
        "student_run_binding": run_binding,
        "result_builder": result_builder,
        "inputs": inputs,
        "requirements": {
            "eligible_scientific_training_artifact": True,
            "exact_adapter_identity": True,
            "exact_matched_source_holdout": True,
            "pinned_student_identity": True,
            "four_sample_nonthinking_decoding": True,
            "recomputed_binary_math_rewards": True,
            "exact_training_plan_config_trace_environment_git_and_gate_custody": True,
        },
        "claim_boundary": (
            "This gate authorizes inclusion in the predeclared held-out matrix regardless of "
            "whether the run helps, harms, or ties its baseline. It does not itself establish "
            "an improvement or training-seed robustness."
        ),
    }


def recompute_student_heldout_result(gate: Mapping[str, Any]) -> dict[str, Any]:
    inputs = gate.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("student held-out gate lacks deterministic inputs")
    for field, hash_field, tree in (
        ("student_run_manifest", "student_run_manifest_sha256", False),
        ("student_completion_manifest", "student_completion_manifest_sha256", False),
        ("student_summary", "student_summary_sha256", False),
        ("student_samples", "student_samples_sha256", False),
        ("prepared_manifest", "prepared_manifest_sha256", False),
        ("trained_adapter", "trained_adapter_tree_sha256", True),
    ):
        path = Path(str(inputs.get(field)))
        actual = sha256_tree(path) if tree else sha256_file(path)
        if actual != inputs.get(hash_field):
            raise ValueError(f"held-out input identity drifted: {field}")
    return student_heldout_result(
        Namespace(
            matrix_key=inputs["matrix_key"],
            student_run_manifest=Path(inputs["student_run_manifest"]),
            student_completion_manifest=Path(inputs["student_completion_manifest"]),
            student_summary=Path(inputs["student_summary"]),
            student_samples=Path(inputs["student_samples"]),
            trained_adapter=Path(inputs["trained_adapter"]),
            prepared_manifest=Path(inputs["prepared_manifest"]),
            student_model=inputs["student_model"],
            student_revision=inputs["student_revision"],
            task_source=inputs["task_source"],
        )
    )


def _percentile_interval(values: list[float]) -> list[float]:
    if len(values) != BOOTSTRAP_DRAWS:
        raise ValueError(f"matrix requires exactly {BOOTSTRAP_DRAWS} bootstrap draws")
    ordered = sorted(values)
    return [
        ordered[int(0.025 * (len(ordered) - 1))],
        ordered[int(0.975 * (len(ordered) - 1))],
    ]


def _effect_label(interval: Iterable[float]) -> str:
    low, high = list(interval)
    if low > 0:
        return "helps"
    if high < 0:
        return "harms"
    return "inconclusive"


def _contrast(estimate: float, draws: list[float], formula: str) -> dict[str, Any]:
    interval = _percentile_interval(draws)
    return {
        "estimate": estimate,
        "bootstrap_95_ci": interval,
        "classification": _effect_label(interval),
        "formula": formula,
    }


def _mean_for_indices(values: list[float], indices: list[int]) -> float:
    return sum(values[index] for index in indices) / len(indices)


def matrix_readout(
    gate_paths: Mapping[str, Path], *, seed: int = BOOTSTRAP_SEED, draws: int = BOOTSTRAP_DRAWS
) -> dict[str, Any]:
    if set(gate_paths) != set(MATRIX_CONTRACT):
        raise ValueError(
            "matrix inputs must be exactly baseline_M, baseline_O, M_M, M_O, O_M, O_O"
        )
    if seed != BOOTSTRAP_SEED or draws != BOOTSTRAP_DRAWS:
        raise ValueError(
            f"primary matrix requires seed={BOOTSTRAP_SEED} and {BOOTSTRAP_DRAWS} draws"
        )
    gates: dict[str, dict[str, Any]] = {}
    inputs: dict[str, dict[str, Any]] = {}
    for key in MATRIX_CONTRACT:
        path = Path(gate_paths[key]).resolve()
        gate = _json_object(path, f"student held-out gate {key}")
        _expect(gate, "schema_version", SCHEMA_VERSION, f"held-out gate {key}")
        _expect(gate, "gate", STUDENT_HELDOUT_GATE, f"held-out gate {key}")
        _expect(gate, "matrix_key", key, f"held-out gate {key}")
        _expect(gate, "passed", True, f"held-out gate {key}")
        _expect(
            gate,
            "authorizes_scientific_matrix_readout",
            True,
            f"held-out gate {key}",
        )
        recomputed = recompute_student_heldout_result(gate)
        if recomputed != gate:
            changed = sorted(
                field
                for field in set(gate) | set(recomputed)
                if gate.get(field) != recomputed.get(field)
            )
            raise ValueError(
                f"held-out gate {key} differs from deterministic recomputation: "
                f"changed_fields={changed[:20]}"
            )
        contract = MATRIX_CONTRACT[key]
        for field in ("objective", "student_source", "teacher_source"):
            _expect(gate, field, contract[field], f"held-out gate {key}")
        gates[key] = gate
        inputs[key] = {"path": str(path), "sha256": sha256_file(path)}

    reference = gates["baseline_M"]
    result_builder = _result_builder_custody(
        reference["student_run_binding"]["git_commit"]
    )
    for key, gate in gates.items():
        _expect(gate, "student_model", reference["student_model"], f"held-out gate {key}")
        _expect(
            gate,
            "student_model_revision",
            reference["student_model_revision"],
            f"held-out gate {key}",
        )
        _expect(
            gate["prepared_binding"],
            "prepared_manifest_sha256",
            reference["prepared_binding"]["prepared_manifest_sha256"],
            f"held-out gate {key}",
        )
        _expect(gate, "decoding", HELDOUT_DECODING, f"held-out gate {key}")
        _expect(gate, "samples_per_problem", SAMPLES_PER_PROBLEM, f"held-out gate {key}")
        _expect(
            gate["student_run_binding"],
            "student_training_plan_sha256",
            reference["student_run_binding"]["student_training_plan_sha256"],
            f"held-out gate {key}",
        )
        _expect(
            gate,
            "result_builder",
            result_builder,
            f"held-out gate {key}",
        )
        _expect(
            gate["student_run_binding"],
            "student_training_config_sha256",
            reference["student_run_binding"]["student_training_config_sha256"],
            f"held-out gate {key}",
        )
        _expect(
            gate["student_run_binding"],
            "git_commit",
            reference["student_run_binding"]["git_commit"],
            f"held-out gate {key}",
        )
        for field in (
            "evaluation_git_commit",
            "evaluator_file_sha256",
            "evaluation_packages",
            "tokenizer_contract_sha256",
        ):
            _expect(
                gate["evaluation_binding"],
                field,
                reference["evaluation_binding"][field],
                f"held-out gate {key} evaluation",
            )

    for source, keys in (("M", ("baseline_M", "M_M", "O_M")), ("O", ("baseline_O", "M_O", "O_O"))):
        expected_ids = sorted(gates[keys[0]]["record_rewards"])
        expected_task_hash = gates[keys[0]]["evaluation_binding"]["task_file_sha256"]
        expected_support = gates[keys[0]]["student_run_binding"]["student_support"]
        expected_training_sequence = {
            field: gates[keys[0]]["student_run_binding"]["trace"].get(field)
            for field in (
                "realized_record_ids_sha256",
                "realized_prompt_sequence_sha256",
            )
        }
        if any(value is None for value in expected_training_sequence.values()):
            raise ValueError(f"{source} baseline lacks realized training-sequence custody")
        for key in keys[1:]:
            if sorted(gates[key]["record_rewards"]) != expected_ids:
                raise ValueError(f"{source} held-out arms do not share the exact record set")
            _expect(
                gates[key]["evaluation_binding"],
                "task_file_sha256",
                expected_task_hash,
                f"{source} held-out arm {key}",
            )
            if gates[key]["student_run_binding"]["student_support"] != expected_support:
                raise ValueError(f"{source} arms do not share the exact student-support identity")
            actual_training_sequence = {
                field: gates[key]["student_run_binding"]["trace"].get(field)
                for field in expected_training_sequence
            }
            if actual_training_sequence != expected_training_sequence:
                raise ValueError(
                    f"{source} arms do not share the exact realized training sequence"
                )

    teacher_artifact_fields = (
        "teacher_source",
        "base_model",
        "base_revision",
        "teacher_gap_manifest_sha256",
        "teacher_gap_payload_sha256",
        "merged_checkpoint",
        "merged_checkpoint_tree_sha256",
        "merge_provenance_manifest_sha256",
        "merge_provenance_payload_sha256",
    )
    for teacher_source, keys in (("M", ("M_M", "M_O")), ("O", ("O_M", "O_O"))):
        first_full = gates[keys[0]]["student_run_binding"]["teacher"]
        second_full = gates[keys[1]]["student_run_binding"]["teacher"]
        first = {field: first_full.get(field) for field in teacher_artifact_fields}
        second = {field: second_full.get(field) for field in teacher_artifact_fields}
        if first != second:
            raise ValueError(
                f"teacher-{teacher_source} arms do not share one exact teacher identity"
            )

    run_paths = [gate["student_run_binding"]["run_manifest"] for gate in gates.values()]
    adapter_paths = [gate["student_run_binding"]["student_adapter"] for gate in gates.values()]
    if len(set(run_paths)) != len(run_paths) or len(set(adapter_paths)) != len(adapter_paths):
        raise ValueError("matrix arms accidentally reuse a run manifest or student adapter path")

    per_record: dict[str, dict[str, float]] = {
        key: {
            record_id: sum(values) / len(values)
            for record_id, values in gate["record_rewards"].items()
        }
        for key, gate in gates.items()
    }
    arm_accuracy = {
        key: sum(values.values()) / len(values) for key, values in per_record.items()
    }
    observed_delta = {
        "M_M": arm_accuracy["M_M"] - arm_accuracy["baseline_M"],
        "O_M": arm_accuracy["O_M"] - arm_accuracy["baseline_M"],
        "M_O": arm_accuracy["M_O"] - arm_accuracy["baseline_O"],
        "O_O": arm_accuracy["O_O"] - arm_accuracy["baseline_O"],
    }
    observed_same_cross_m = arm_accuracy["M_M"] - arm_accuracy["O_M"]
    observed_same_cross_o = arm_accuracy["O_O"] - arm_accuracy["M_O"]
    observed_stratified = (observed_same_cross_m + observed_same_cross_o) / 2
    observed_factorial = observed_same_cross_m + observed_same_cross_o

    record_ids = {
        "M": sorted(per_record["baseline_M"]),
        "O": sorted(per_record["baseline_O"]),
    }
    vectors = {
        key: [per_record[key][record_id] for record_id in record_ids[gate["student_source"]]]
        for key, gate in gates.items()
    }
    rng = random.Random(seed)
    bootstrap: dict[str, list[float]] = defaultdict(list)
    for _ in range(draws):
        m_idx = [rng.randrange(len(record_ids["M"])) for _ in record_ids["M"]]
        o_idx = [rng.randrange(len(record_ids["O"])) for _ in record_ids["O"]]
        sample_means = {
            "baseline_M": _mean_for_indices(vectors["baseline_M"], m_idx),
            "M_M": _mean_for_indices(vectors["M_M"], m_idx),
            "O_M": _mean_for_indices(vectors["O_M"], m_idx),
            "baseline_O": _mean_for_indices(vectors["baseline_O"], o_idx),
            "M_O": _mean_for_indices(vectors["M_O"], o_idx),
            "O_O": _mean_for_indices(vectors["O_O"], o_idx),
        }
        for key, baseline in (
            ("M_M", "baseline_M"),
            ("O_M", "baseline_M"),
            ("M_O", "baseline_O"),
            ("O_O", "baseline_O"),
        ):
            bootstrap[f"delta:{key}"].append(sample_means[key] - sample_means[baseline])
        same_m = sample_means["M_M"] - sample_means["O_M"]
        same_o = sample_means["O_O"] - sample_means["M_O"]
        bootstrap["same_cross:M"].append(same_m)
        bootstrap["same_cross:O"].append(same_o)
        bootstrap["stratified"].append((same_m + same_o) / 2)
        bootstrap["factorial"].append(same_m + same_o)

    baseline_deltas = {
        key: _contrast(
            observed_delta[key],
            bootstrap[f"delta:{key}"],
            f"accuracy({key}) - accuracy(baseline_{MATRIX_CONTRACT[key]['student_source']})",
        )
        for key in ("M_M", "O_M", "M_O", "O_O")
    }
    same_vs_cross = {
        "M": _contrast(
            observed_same_cross_m,
            bootstrap["same_cross:M"],
            "accuracy(M_M) - accuracy(O_M)",
        ),
        "O": _contrast(
            observed_same_cross_o,
            bootstrap["same_cross:O"],
            "accuracy(O_O) - accuracy(M_O)",
        ),
        "equal_stratum_mean": _contrast(
            observed_stratified,
            bootstrap["stratified"],
            "0.5 * ((M_M - O_M) + (O_O - M_O))",
        ),
    }
    stratified_interaction = _contrast(
        observed_factorial,
        bootstrap["factorial"],
        "(M_M - O_M) - (M_O - O_O)",
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "readout": MATRIX_READOUT,
        "scientific_readout_authorized": True,
        "authorization_is_independent_of_effect_sign": True,
        "matrix_keys": list(MATRIX_CONTRACT),
        "student_model": reference["student_model"],
        "student_model_revision": reference["student_model_revision"],
        "arm_accuracy": arm_accuracy,
        "arm_records": {key: gate["records"] for key, gate in gates.items()},
        "baseline_deltas": baseline_deltas,
        "same_vs_cross": same_vs_cross,
        "stratified_interaction": stratified_interaction,
        "result_builder": result_builder,
        "bootstrap": {
            "unit": "paired record within held-out source stratum",
            "draws": draws,
            "seed": seed,
            "confidence_interval": "percentile_95",
            "classification_rule": {
                "helps": "lower_bound > 0",
                "harms": "upper_bound < 0",
                "inconclusive": "interval includes 0",
            },
        },
        "inputs": inputs,
        "claim_boundary": (
            "The readout reports the matched seed-0 pilot and paired record-bootstrap "
            "uncertainty. It does not estimate training-seed variance or prove a universal "
            "source-transfer law. A negative or null effect remains an authorized result."
        ),
    }


def recompute_matrix_readout(payload: Mapping[str, Any]) -> dict[str, Any]:
    inputs = payload.get("inputs")
    bootstrap = payload.get("bootstrap")
    if not isinstance(inputs, dict) or set(inputs) != set(MATRIX_CONTRACT):
        raise ValueError("matrix readout lacks the exact six deterministic inputs")
    if not isinstance(bootstrap, dict):
        raise ValueError("matrix readout lacks its bootstrap contract")
    paths: dict[str, Path] = {}
    for key, binding in inputs.items():
        if not isinstance(binding, dict):
            raise ValueError(f"matrix input binding is invalid: {key}")
        path = Path(str(binding.get("path")))
        _file_binding(path, binding.get("sha256"), f"matrix gate {key}")
        paths[key] = path
    return matrix_readout(
        paths,
        seed=int(bootstrap.get("seed")),
        draws=int(bootstrap.get("draws")),
    )


def matrix_markdown(payload: Mapping[str, Any]) -> str:
    if payload.get("readout") != MATRIX_READOUT:
        raise ValueError("cannot render an unknown matrix readout")
    lines = [
        "# OPD math six-run held-out readout",
        "",
        "All six arms passed artifact and evaluation custody. Authorization is independent of effect sign.",
        "",
        "| Arm | Held-out accuracy | Delta vs matched task-RL | 95% record-bootstrap CI | Readout |",
        "|---|---:|---:|---:|---|",
    ]
    deltas = payload["baseline_deltas"]
    for key in payload["matrix_keys"]:
        accuracy = float(payload["arm_accuracy"][key])
        if key.startswith("baseline_"):
            lines.append(f"| `{key}` | {accuracy:.6f} | — | — | baseline |")
        else:
            contrast = deltas[key]
            low, high = contrast["bootstrap_95_ci"]
            lines.append(
                f"| `{key}` | {accuracy:.6f} | {contrast['estimate']:+.6f} | "
                f"[{low:+.6f}, {high:+.6f}] | {contrast['classification']} |"
            )
    lines.extend(
        [
            "",
            "## Same-source versus cross-source teacher",
            "",
            "| Contrast | Estimate | 95% record-bootstrap CI | Readout |",
            "|---|---:|---:|---|",
        ]
    )
    for label, key in (("Target M", "M"), ("Target O", "O"), ("Equal-stratum mean", "equal_stratum_mean")):
        contrast = payload["same_vs_cross"][key]
        low, high = contrast["bootstrap_95_ci"]
        lines.append(
            f"| {label} | {contrast['estimate']:+.6f} | "
            f"[{low:+.6f}, {high:+.6f}] | {contrast['classification']} |"
        )
    interaction = payload["stratified_interaction"]
    low, high = interaction["bootstrap_95_ci"]
    lines.extend(
        [
            "",
            "## Stratified interaction",
            "",
            f"`{interaction['formula']}` = **{interaction['estimate']:+.6f}** "
            f"(95% CI [{low:+.6f}, {high:+.6f}]; {interaction['classification']}).",
            "",
            str(payload["claim_boundary"]),
            "",
        ]
    )
    return "\n".join(lines)


def _write_new(path: Path, content: str) -> None:
    path = Path(path)
    if path.is_symlink() or path.exists():
        raise FileExistsError(f"refusing to overwrite result artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _preflight_result_outputs(
    paths: Iterable[Path], *, protected_trees: Iterable[Path] = ()
) -> list[Path]:
    raw_paths = [Path(path) for path in paths]
    resolved_paths = [path.resolve() for path in raw_paths]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("result output paths must be distinct")
    protected = [(Path(path).resolve()) for path in protected_trees]
    for path, resolved in zip(raw_paths, resolved_paths, strict=True):
        if path.is_symlink() or path.exists():
            raise FileExistsError(f"refusing to overwrite result artifact: {path}")
        if resolved.is_relative_to(ROOT):
            raise ValueError("scientific result outputs must be outside the Git worktree")
        for tree in protected:
            if resolved == tree or resolved.is_relative_to(tree):
                raise ValueError(
                    f"result output would mutate a protected input tree: {tree}"
                )
    return resolved_paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    heldout = subparsers.add_parser("heldout", help="gate one trained student held-out evaluation")
    heldout.add_argument("--matrix-key", choices=tuple(MATRIX_CONTRACT), required=True)
    heldout.add_argument("--student-run-manifest", type=Path, required=True)
    heldout.add_argument("--student-completion-manifest", type=Path, required=True)
    heldout.add_argument("--student-summary", type=Path, required=True)
    heldout.add_argument("--student-samples", type=Path, required=True)
    heldout.add_argument("--trained-adapter", type=Path, required=True)
    heldout.add_argument("--prepared-manifest", type=Path, required=True)
    heldout.add_argument("--student-model", required=True)
    heldout.add_argument("--student-revision", required=True)
    heldout.add_argument("--task-source", choices=("M", "O"), required=True)
    heldout.add_argument("--output", type=Path, required=True)

    matrix = subparsers.add_parser("matrix", help="combine the exact six primary result gates")
    for key in MATRIX_CONTRACT:
        matrix.add_argument(f"--{key.replace('_', '-').lower()}", type=Path, required=True)
    matrix.add_argument("--output-json", type=Path, required=True)
    matrix.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "heldout":
        _preflight_result_outputs(
            [args.output],
            protected_trees=[
                args.trained_adapter,
                args.prepared_manifest.resolve().parent,
            ],
        )
        payload = student_heldout_result(args)
        teacher = payload["student_run_binding"].get("teacher")
        if isinstance(teacher, dict):
            _preflight_result_outputs(
                [args.output],
                protected_trees=[Path(teacher["merged_checkpoint"])],
            )
        if recompute_student_heldout_result(payload) != payload:
            raise RuntimeError("held-out gate failed deterministic self-recomputation")
        _write_new(args.output, json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"output": str(args.output.resolve()), "passed": True}, sort_keys=True))
        return 0

    _preflight_result_outputs([args.output_json, args.output_markdown])
    gate_paths = {
        key: getattr(args, key.lower())
        for key in MATRIX_CONTRACT
    }
    payload = matrix_readout(gate_paths)
    if recompute_matrix_readout(payload) != payload:
        raise RuntimeError("matrix readout failed deterministic self-recomputation")
    _write_new(args.output_json, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _write_new(args.output_markdown, matrix_markdown(payload))
    print(
        json.dumps(
            {
                "output_json": str(args.output_json.resolve()),
                "output_markdown": str(args.output_markdown.resolve()),
                "authorized": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
