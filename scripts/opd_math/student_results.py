#!/usr/bin/env python3
"""Fail-closed held-out result gates and conditional OPD readouts.

Training completion is only permission to evaluate an adapter.  This module
separately proves that a held-out evaluation belongs to an eligible scientific
student run.  The active successor combines two task-RL baselines with the two
allowed O-teacher arms; the historical six-run builder remains available only
for predecessor provenance.  Result authorization is deliberately independent
of the sign of an observed effect: harms and nulls are scientific results when
custody is valid.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import stat
import subprocess
import sys
from argparse import Namespace
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    from scripts.opd.trace_metrics import (
        reconstruct_step_metrics,
        validate_recorded_step_metrics,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.opd.trace_metrics import (  # type: ignore
        reconstruct_step_metrics,
        validate_recorded_step_metrics,
    )

try:
    from .data_contract import iter_jsonl
    from .math_reward import (
        EVALUATION_VERIFIER_ERROR_POLICY,
        EVALUATION_VERIFIER_MAX_ATTEMPTS,
        verify_completion,
        verify_evaluation_completion,
    )
    from .quality_gates import (
        EVALUATION_CONTRACT,
        EVALUATION_MERGED_KIND,
        EXPECTED_TEACHER_TRAIN_PACKAGES,
        STUDENT_GATE_TYPE,
        _prepared_role_binding,
        bootstrap_delta,
        canonical_json_sha256,
        checked_evaluation,
        recompute_student_gate,
        recompute_teacher_gate,
        sha256_file,
        sha256_tree,
        write_text_exclusive_fsync,
    )
    from .verify_environment import (
        SCHEMA as ENVIRONMENT_VERIFICATION_SCHEMA,
        reverify_recorded_environment,
    )
    from .server_scoring_probe import expected_serve_environment_launcher
except ImportError:
    from data_contract import iter_jsonl  # type: ignore
    from math_reward import (  # type: ignore
        EVALUATION_VERIFIER_ERROR_POLICY,
        EVALUATION_VERIFIER_MAX_ATTEMPTS,
        verify_completion,
        verify_evaluation_completion,
    )
    from quality_gates import (  # type: ignore
        EVALUATION_CONTRACT,
        EVALUATION_MERGED_KIND,
        EXPECTED_TEACHER_TRAIN_PACKAGES,
        STUDENT_GATE_TYPE,
        _prepared_role_binding,
        bootstrap_delta,
        canonical_json_sha256,
        checked_evaluation,
        recompute_student_gate,
        recompute_teacher_gate,
        sha256_file,
        sha256_tree,
        write_text_exclusive_fsync,
    )
    from verify_environment import (  # type: ignore
        SCHEMA as ENVIRONMENT_VERIFICATION_SCHEMA,
        reverify_recorded_environment,
    )
    from server_scoring_probe import (  # type: ignore
        expected_serve_environment_launcher,
    )


ROOT = Path(__file__).resolve().parents[2]
CANONICAL_STUDENT_TRAINING_PLAN = (
    ROOT / "configs" / "opd_math" / "student_training_plan.json"
)
ENVIRONMENT_VERIFIER = ROOT / "scripts" / "opd_math" / "verify_environment.py"
SCHEMA_VERSION = 1
STUDENT_HELDOUT_GATE = "student_heldout_result_v2_exact_environment"
MATRIX_READOUT = "opd_math_six_run_matrix_v2_exact_environment"
O_TEACHER_READOUT = "opd_math_o_teacher_conditional_readout_v2_exact_environment"
O_TEACHER_PREREGISTRATION = (
    "opd_math_o_teacher_conditional_preregistration_v4_diagnostic_eval_custody"
)
O_TEACHER_LAUNCH_LEDGER = "opd_math_o_teacher_launch_ledger_v3"
O_TEACHER_PRELAUNCH_RECEIPT = "opd_math_o_teacher_student_prelaunch_receipt_v1"
O_M_ONE_STEP_DIAGNOSTIC_AUDIT = (
    "opd_math_o_m_one_step_full_custody_diagnostic_v1"
)
M_TEACHER_NEGATIVE_SELECTION_AUDIT = (
    "opd_math_m_teacher_negative_legacy_strict_replay_v2"
)
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
O_TEACHER_CONTRACT = {
    key: MATRIX_CONTRACT[key]
    for key in ("baseline_M", "O_M", "baseline_O", "O_O")
}
O_TEACHER_PAIRS = {
    "M": ("baseline_M", "O_M"),
    "O": ("baseline_O", "O_O"),
}
O_TEACHER_STABLE_IDENTITY_FIELDS = (
    "teacher_source",
    "base_model",
    "base_revision",
    "teacher_gap_manifest",
    "teacher_gap_manifest_sha256",
    "teacher_gap_payload_sha256",
    "merged_checkpoint",
    "merged_checkpoint_tree_sha256",
    "merge_provenance_manifest_sha256",
    "merge_provenance_payload_sha256",
)
O_TEACHER_SUPPORT_IDENTITY_FIELDS = (
    "manifest_sha256",
    "payload_sha256",
    "source",
)
O_TEACHER_ARM_IDENTITY_FIELDS = (
    "heldout_gate",
    "run_manifest",
    "student_completion_manifest",
    "student_adapter",
    "student_eval_summary",
    "student_eval_samples",
    "student_eval_custody",
    "prelaunch_receipt",
)
O_M_DIAGNOSTIC_IDENTITY_FIELDS = (
    "terminal_audit",
    "terminal_audit_sha256",
    "run_manifest",
    "run_manifest_sha256",
    "completion_manifest",
    "completion_manifest_sha256",
    "student_adapter",
    "student_adapter_tree_sha256",
    "diagnostic_clean_before_preregistration",
    "plumbing_only",
    "scientific_result",
)
O_TEACHER_CLAIM_BOUNDARY = (
    "Conditional on selecting the preregistered O teacher because it passed its "
    "scientific skill-gap gate, this four-arm seed-0 pilot estimates whether "
    "adding its gap-gated OPD signal to matched task RL helped, harmed, or was "
    "inconclusive on each frozen source. The M teacher failed and its M_M/M_O "
    "arms are prohibited. This readout cannot establish universal OPD benefit, "
    "same-source superiority, training-seed robustness, a scaling or transfer "
    "law, or any result for a rescued M teacher."
)
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


def _parse_utc_timestamp(value: Any, label: str) -> datetime:
    """Parse a strict UTC ISO-8601 custody timestamp."""

    if not isinstance(value, str) or re.fullmatch(
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?Z",
        value,
    ) is None:
        raise ValueError(f"{label} must be a strict UTC ISO-8601 timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{label} is not a valid UTC timestamp") from exc
    if parsed.tzinfo != timezone.utc:
        raise ValueError(f"{label} is not UTC")
    return parsed


def _sealed_json_file(
    path: Path,
    expected_sha256: Any,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    """Open one immutable JSON custody artifact and verify its file identity."""

    raw = Path(path).expanduser()
    if raw.is_symlink() or not raw.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    if raw.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise ValueError(f"{label} must be sealed read-only")
    resolved = raw.resolve()
    expected = _hash_identity(expected_sha256, label)
    if sha256_file(resolved) != expected:
        raise ValueError(f"{label} hash differs from the sealed identity")
    return resolved, _json_object(resolved, label)


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
    if (
        path.name != filename
        or path.parent.name != commit
        or path.parent.parent.name != "environment_freezes"
    ):
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


def _validate_environment_verification(
    recorded: Any,
    *,
    freeze: Mapping[str, Any],
    commit: str,
    kind: str,
) -> dict[str, Any]:
    if not isinstance(recorded, dict):
        raise ValueError(f"student {kind} environment lacks exact live verification")
    _expect(recorded, "schema_version", 1, f"student {kind} verification")
    _expect(
        recorded,
        "schema",
        ENVIRONMENT_VERIFICATION_SCHEMA,
        f"student {kind} verification",
    )
    _expect(recorded, "status", "passed", f"student {kind} verification")
    _expect(recorded, "expected_commit", commit, f"student {kind} verification")
    _expect(recorded, "freeze_kind", kind, f"student {kind} verification")
    root = _absolute(
        recorded.get("environment_root"),
        CANONICAL_STUDENT_TRAINING_PLAN,
        f"student {kind} environment root",
    )
    _expect(
        recorded,
        "live_python",
        str(root / "bin" / "python"),
        f"student {kind} verification",
    )
    distribution_count = recorded.get("installed_distribution_count")
    if not isinstance(distribution_count, int) or distribution_count <= 0:
        raise ValueError(f"student {kind} verification lacks a full distribution count")
    _hash_identity(
        recorded.get("installed_distribution_map_sha256"),
        f"student {kind} installed distribution map",
    )

    commit_freeze = recorded.get("commit_freeze")
    expected_commit_freeze = {
        "path": freeze["path"],
        "sha256": freeze["sha256"],
        "byte_identical_to_requirements_freeze": True,
    }
    if commit_freeze != expected_commit_freeze:
        raise ValueError(f"student {kind} verification commit-freeze identity drifted")
    requirements = recorded.get("requirements_freeze")
    if not isinstance(requirements, dict):
        raise ValueError(f"student {kind} verification lacks requirements.freeze.txt")
    requirements_path = root / "requirements.freeze.txt"
    _expect(
        requirements,
        "path",
        str(requirements_path),
        f"student {kind} requirements freeze",
    )
    requirements_hash = _file_binding(
        requirements_path,
        requirements.get("sha256"),
        f"student {kind} requirements freeze",
    )
    if requirements_hash != freeze["sha256"]:
        raise ValueError(
            f"student {kind} requirements freeze differs from the commit freeze"
        )

    executable = recorded.get("expected_executable")
    if kind == "train":
        if executable is not None:
            raise ValueError("student train verification unexpectedly binds vLLM")
    else:
        if not isinstance(executable, dict):
            raise ValueError("student serve verification lacks bin/vllm custody")
        vllm = root / "bin" / "vllm"
        _expect(executable, "path", str(vllm), "student serve executable")
        _file_binding(vllm, executable.get("sha256"), "student serve executable")
        _expect(
            executable,
            "shebang",
            f"#!{root}/bin/python",
            "student serve executable",
        )

    try:
        current = reverify_recorded_environment(recorded, in_process=kind == "train")
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"student {kind} live environment no longer verifies") from exc
    if current != recorded:
        raise ValueError(f"student {kind} live environment identity changed")
    return dict(recorded)


def _validate_environment(
    environment: Any, *, commit: str, requires_teacher: bool
) -> dict[str, Any]:
    if not isinstance(environment, dict):
        raise ValueError("student run lacks its scientific environment contract")
    _expect(environment, "schema_version", 2, "student environment")
    _expect(environment, "git_commit", commit, "student environment")
    verifier = {
        "path": str(ENVIRONMENT_VERIFIER.resolve()),
        "sha256": sha256_file(ENVIRONMENT_VERIFIER),
    }
    _expect(environment, "verifier", verifier, "student environment")
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
    train_verification = _validate_environment_verification(
        environment.get("train_verification"),
        freeze=train,
        commit=commit,
        kind="train",
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
        serve_verification = _validate_environment_verification(
            environment.get("serve_verification"),
            freeze=serve,
            commit=commit,
            kind="serve",
        )
    else:
        if serve_binding is not None or environment.get("serve_verification") is not None:
            raise ValueError(
                "task-RL baseline unexpectedly carries a teacher serve environment"
            )
        serve = None
        serve_verification = None
    return {
        "verifier": verifier,
        "train_freeze": train,
        "train_verification": train_verification,
        "serve_freeze": serve,
        "serve_verification": serve_verification,
    }


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
    _expect(gate, "gate", STUDENT_GATE_TYPE, "student-support gate")
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
    _expect(provenance, "schema", "opd_math_merged_teacher_v3", "teacher provenance")
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
    teacher_training_environment = teacher_gate.get("teacher_training_environment")
    if not isinstance(teacher_training_environment, dict):
        raise ValueError("teacher-gap gate lacks exact teacher train-environment custody")
    _expect(
        provenance,
        "teacher_training_environment",
        teacher_training_environment,
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
        "teacher_gap_manifest": str(teacher_gate_path),
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
    require_task_signal: bool = True,
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
        _expect(row, "schema_version", 2, "sample trace")
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
        student_logprobs = row.get("student_token_logprobs")
        if (
            not isinstance(student_logprobs, list)
            or len(student_logprobs) != len(completion_ids)
            or any(
                type(value) not in (int, float) or not math.isfinite(float(value))
                for value in student_logprobs
            )
        ):
            raise ValueError("sample trace lacks exact student token log-probabilities")
        recomputed_student_nll = -sum(float(value) for value in student_logprobs) / len(
            student_logprobs
        )
        if not math.isclose(
            float(row.get("student_nll", math.nan)),
            recomputed_student_nll,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError("sample trace student-NLL summary does not match exact token values")
        sample_expanded_prompt_tokens += len(prompt_ids)
        verdict = verify_completion(completion_text, registered["solution"])
        if verdict.get("reward") is None:
            raise RuntimeError(f"training trace verifier failure: {verdict}")
        _expect(row, "reward_status", verdict.get("status"), "sample trace")
        _expect(row, "reward", float(verdict.get("reward")), "sample trace")
        if objective == "task_rl":
            _expect(
                row,
                "teacher_token_logprobs_on_student_trajectory",
                None,
                "task-RL sample trace",
            )
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
            teacher_logprobs = row.get("teacher_token_logprobs_on_student_trajectory")
            if (
                not isinstance(teacher_logprobs, list)
                or len(teacher_logprobs) != len(completion_ids)
                or any(
                    type(value) not in (int, float) or not math.isfinite(float(value))
                    for value in teacher_logprobs
                )
            ):
                raise ValueError("main-arm sample trace lacks exact teacher token log-probabilities")
            recomputed_teacher_nll = -sum(float(value) for value in teacher_logprobs) / len(
                teacher_logprobs
            )
            if not math.isclose(
                float(row.get("teacher_nll_on_student_trajectory", math.nan)),
                recomputed_teacher_nll,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise ValueError(
                    "main-arm teacher-NLL summary does not match exact token values"
                )
            token_gaps = [
                float(teacher_logprob) - float(student_logprob)
                for teacher_logprob, student_logprob in zip(
                    teacher_logprobs,
                    student_logprobs,
                    strict=True,
                )
            ]
            expected_gap_metrics = {
                "mean_teacher_student_gap": sum(token_gaps) / len(token_gaps),
                "mean_abs_k1_log_ratio": sum(abs(value) for value in token_gaps)
                / len(token_gaps),
                "min_teacher_student_gap": min(token_gaps),
                "max_teacher_student_gap": max(token_gaps),
                "positive_teacher_gap_fraction": sum(value > 0 for value in token_gaps)
                / len(token_gaps),
            }
            for field, expected_metric in expected_gap_metrics.items():
                if not math.isclose(
                    float(row.get(field, math.nan)),
                    expected_metric,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                ):
                    raise ValueError(
                        f"main-arm sample trace {field} does not match exact token values"
                    )
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

    samples_by_step: dict[int, list[dict[str, Any]]] = {
        step: [] for step in range(1, expected_steps + 1)
    }
    for row in sample_rows:
        samples_by_step[int(row["step"])].append(row)
    for step, recorded in enumerate(steps, 1):
        reconstructed = reconstruct_step_metrics(
            samples_by_step[step],
            mode=objective,
            task_reward_coef=float(fixed["task_reward_coef"]),
            k1_coef=float(fixed["k1_coef"]),
            gap_gate_beta=float(fixed["gap_gate_beta"]),
            advantage_clip=float(fixed["advantage_clip"]),
        )
        validate_recorded_step_metrics(
            recorded,
            reconstructed,
            label=f"student step trace {step}",
        )

    expected_fraction = informative_groups / expected_groups
    _expect(
        completion,
        "minimum_informative_group_fraction",
        fixed["min_informative_group_fraction"],
        "student completion",
    )
    if require_task_signal and expected_fraction < float(
        fixed["min_informative_group_fraction"]
    ):
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


def _validate_student_prelaunch_receipt(
    value: Any,
    *,
    matrix_key: str,
    objective: str,
    source: str,
    campaign_run_id: str,
    scheduler_job_id: str,
    commit: str,
    run_path: Path,
    completion_path: Path,
    adapter: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(value, dict):
        raise ValueError("student run lacks its prelaunch receipt binding")
    path_value = value.get("path")
    if not isinstance(path_value, str) or not Path(path_value).is_absolute():
        raise ValueError("student prelaunch receipt path must be absolute")
    path, receipt = _sealed_json_file(
        Path(path_value), value.get("sha256"), "student prelaunch receipt"
    )
    _expect(
        value,
        "payload_sha256",
        canonical_json_sha256(receipt),
        "student prelaunch receipt binding",
    )
    _expect(receipt, "schema_version", 1, "student prelaunch receipt")
    _expect(
        receipt,
        "receipt",
        O_TEACHER_PRELAUNCH_RECEIPT,
        "student prelaunch receipt",
    )
    for field, expected in (
        ("sealed_before_optimizer_start", True),
        ("run_key", matrix_key),
        ("run_id", campaign_run_id),
        ("scheduler_job_id", scheduler_job_id),
        ("mode", objective),
        ("student_source", source),
        ("git_commit", commit),
    ):
        _expect(receipt, field, expected, "student prelaunch receipt")
    _parse_utc_timestamp(receipt.get("created_utc"), "student prelaunch receipt created_utc")
    expected_artifacts = {
        "run_manifest": str(run_path),
        "student_completion_manifest": str(completion_path),
        "student_adapter": str(adapter),
        "prelaunch_receipt": str(path),
    }
    _expect(
        receipt,
        "expected_artifacts",
        expected_artifacts,
        "student prelaunch receipt",
    )
    preregistration = receipt.get("preregistration")
    launch_ledger = receipt.get("launch_ledger")
    for label, item in (
        ("preregistration", preregistration),
        ("launch ledger", launch_ledger),
    ):
        if not isinstance(item, dict):
            raise ValueError(f"student prelaunch receipt lacks {label} custody")
        bound_path = Path(str(item.get("path")))
        _file_binding(bound_path, item.get("sha256"), f"student prelaunch {label}")
    return receipt, {
        "path": str(path),
        "sha256": sha256_file(path),
        "payload_sha256": canonical_json_sha256(receipt),
        "campaign_id": receipt.get("campaign_id"),
        "run_key": matrix_key,
        "sealed_before_optimizer_start": True,
        "preregistration": preregistration,
        "launch_ledger": launch_ledger,
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
    adapter = Path(adapter).resolve()
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
    _expect(run, "normalized_training_config", fixed, "student run")
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
    campaign_run_id = binding.get("campaign_run_id")
    if not isinstance(campaign_run_id, str) or re.fullmatch(
        r"[A-Za-z0-9._-]+", campaign_run_id
    ) is None:
        raise ValueError("student run lacks a safe preregistered campaign run ID")
    scheduler_job_id = binding.get("scheduler_job_id")
    if not isinstance(scheduler_job_id, str) or re.fullmatch(
        r"[1-9][0-9]*", scheduler_job_id
    ) is None:
        raise ValueError("student run lacks its positive Slurm job ID custody")
    prelaunch_receipt, prelaunch_identity = _validate_student_prelaunch_receipt(
        binding.get("prelaunch_receipt"),
        matrix_key=matrix_key,
        objective=objective,
        source=source,
        campaign_run_id=campaign_run_id,
        scheduler_job_id=scheduler_job_id,
        commit=commit,
        run_path=run_path,
        completion_path=completion_path,
        adapter=adapter,
    )
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
        _expect(
            binding,
            "serve_environment_process_binding_validated",
            False,
            "task-RL binding",
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
            binding,
            "serve_environment_process_binding_validated",
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
        serve_verification = environment_identity.get("serve_verification")
        if not isinstance(serve_verification, dict):
            raise ValueError("main-arm run lacks exact serve-environment verification")
        expected_launcher = expected_serve_environment_launcher(
            Path(serve_verification["environment_root"])
        )
        _expect(
            server_contract["local_process_binding"],
            "serve_environment_launcher",
            expected_launcher,
            "main-arm server/serve-environment binding",
        )
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

    if prelaunch_receipt.get("student_support") != support_identity:
        raise ValueError("student run support differs from its prelaunch receipt")
    prelaunch_teacher = prelaunch_receipt.get("o_teacher")
    if teacher_identity is None:
        if prelaunch_teacher is not None:
            raise ValueError("baseline prelaunch receipt unexpectedly binds a teacher")
    else:
        stable_teacher = {
            field: teacher_identity.get(field)
            for field in O_TEACHER_STABLE_IDENTITY_FIELDS
        }
        if prelaunch_teacher != stable_teacher:
            raise ValueError("student run teacher differs from its prelaunch receipt")

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
        "campaign_run_id": campaign_run_id,
        "scheduler_job_id": scheduler_job_id,
        "git_commit": commit,
        "student_training_plan_sha256": plan_binding["sha256"],
        "student_training_config_sha256": plan_binding["actual_config_sha256"],
        "student_adapter": str(adapter),
        "student_adapter_tree_sha256": adapter_hash,
        "student_support": support_identity,
        "teacher": teacher_identity,
        "prelaunch_receipt": prelaunch_identity,
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
    if evaluation_binding.get("evaluation_contract") != EVALUATION_CONTRACT:
        raise ValueError(
            "scientific held-out evaluation requires the exact-environment v2 contract"
        )
    if not isinstance(evaluation_binding.get("evaluation_environment"), dict):
        raise ValueError(
            "scientific held-out evaluation requires exact train-environment custody"
        )
    if not isinstance(
        evaluation_binding.get("evaluation_post_promotion_custody"), dict
    ):
        raise ValueError(
            "scientific held-out evaluation requires post-promotion custody"
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
    evaluation_environment = evaluation_binding["evaluation_environment"]
    run_environment = run_binding["environment"]
    _expect(
        evaluation_environment,
        "verifier",
        run_environment["verifier"],
        "held-out evaluation/student-run train environment",
    )
    for field in ("path", "sha256"):
        _expect(
            evaluation_environment["train_freeze"],
            field,
            run_environment["train_freeze"][field],
            "held-out evaluation/student-run train freeze",
        )
    _expect(
        evaluation_environment,
        "train_verification",
        run_environment["train_verification"],
        "held-out evaluation/student-run train environment",
    )
    result_builder = _result_builder_custody(run_binding["git_commit"])
    record_rewards = {key: list(grouped[key]) for key in sorted(grouped)}
    record_accuracy = {
        key: sum(record_rewards[key]) / len(record_rewards[key]) for key in record_rewards
    }
    error_counts_by_record: dict[str, int] = defaultdict(int)
    for item in evaluation_binding["verifier_error_sample_keys"]:
        error_counts_by_record[item["record_id"]] += 1
    record_accuracy_bounds = {
        key: [
            record_accuracy[key],
            (
                sum(record_rewards[key]) + error_counts_by_record.get(key, 0)
            )
            / len(record_rewards[key]),
        ]
        for key in record_rewards
    }
    accuracy = sum(record_accuracy.values()) / len(record_accuracy)
    accuracy_upper_bound = sum(
        bounds[1] for bounds in record_accuracy_bounds.values()
    ) / len(record_accuracy_bounds)
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
        "accuracy_bounds_under_verifier_uncertainty": [
            accuracy,
            accuracy_upper_bound,
        ],
        "verifier_error_samples": evaluation_binding["verifier_error_samples"],
        "verifier_error_sample_keys_sha256": evaluation_binding[
            "verifier_error_sample_keys_sha256"
        ],
        "record_rewards": record_rewards,
        "record_accuracy": record_accuracy,
        "record_accuracy_bounds_under_verifier_uncertainty": record_accuracy_bounds,
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
            "exact_evaluation_environment_and_post_promotion_custody": True,
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
    return _percentile_interval_at(values, 0.025, 0.975)


def _percentile_interval_at(
    values: list[float], lower_probability: float, upper_probability: float
) -> list[float]:
    if len(values) != BOOTSTRAP_DRAWS:
        raise ValueError(f"readout requires exactly {BOOTSTRAP_DRAWS} bootstrap draws")
    if not 0.0 <= lower_probability < upper_probability <= 1.0:
        raise ValueError("bootstrap percentile probabilities are invalid")
    ordered = sorted(values)
    return [
        ordered[int(lower_probability * (len(ordered) - 1))],
        ordered[int(upper_probability * (len(ordered) - 1))],
    ]


def _effect_label(interval: Iterable[float]) -> str:
    low, high = list(interval)
    if low > 0:
        return "helps"
    if high < 0:
        return "harms"
    return "inconclusive"


def _contrast(
    estimate: float,
    draws: list[float],
    formula: str,
    *,
    uncertainty_estimate_bounds: list[float] | None = None,
    uncertainty_lower_draws: list[float] | None = None,
    uncertainty_upper_draws: list[float] | None = None,
) -> dict[str, Any]:
    interval = _percentile_interval(draws)
    result = {
        "estimate": estimate,
        "bootstrap_95_ci": interval,
        "classification": _effect_label(interval),
        "formula": formula,
    }
    if (
        uncertainty_estimate_bounds is not None
        or uncertainty_lower_draws is not None
        or uncertainty_upper_draws is not None
    ):
        if (
            uncertainty_estimate_bounds is None
            or uncertainty_lower_draws is None
            or uncertainty_upper_draws is None
            or len(uncertainty_estimate_bounds) != 2
        ):
            raise ValueError("verifier-uncertainty contrast inputs must be complete")
        lower_interval = _percentile_interval(uncertainty_lower_draws)
        upper_interval = _percentile_interval(uncertainty_upper_draws)
        robust_interval = [lower_interval[0], upper_interval[1]]
        result["classification_without_verifier_uncertainty"] = result[
            "classification"
        ]
        result["classification"] = _effect_label(robust_interval)
        result["verifier_uncertainty_sensitivity"] = {
            "policy": "binary_worst_case_bootstrap_envelope_v1",
            "estimate_bounds": uncertainty_estimate_bounds,
            "bootstrap_95_envelope": robust_interval,
            "pessimistic_bootstrap_95_ci": lower_interval,
            "optimistic_bootstrap_95_ci": upper_interval,
        }
    return result


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
            "evaluation_contract",
            "evaluation_environment",
        ):
            _expect(
                gate["evaluation_binding"],
                field,
                reference["evaluation_binding"][field],
                f"held-out gate {key} evaluation",
            )
        if not isinstance(
            gate["evaluation_binding"].get("evaluation_post_promotion_custody"),
            dict,
        ):
            raise ValueError(
                f"held-out gate {key} lacks post-promotion evaluation custody"
            )

    reference_environment = reference["student_run_binding"].get("environment")
    if not isinstance(reference_environment, dict):
        raise ValueError("baseline_M lacks validated environment custody")
    reference_verifier = reference_environment.get("verifier")
    if not isinstance(reference_verifier, dict):
        raise ValueError("baseline_M lacks environment-verifier code custody")
    reference_train_freeze = reference_environment.get("train_freeze")
    if not isinstance(reference_train_freeze, dict):
        raise ValueError("baseline_M lacks a validated train environment freeze")
    reference_train_verification = reference_environment.get("train_verification")
    if not isinstance(reference_train_verification, dict):
        raise ValueError("baseline_M lacks exact live train environment verification")
    for key, gate in gates.items():
        environment = gate["student_run_binding"].get("environment")
        if not isinstance(environment, dict):
            raise ValueError(f"held-out gate {key} lacks validated environment custody")
        _expect(
            environment,
            "verifier",
            reference_verifier,
            f"held-out gate {key} environment",
        )
        _expect(
            environment,
            "train_freeze",
            reference_train_freeze,
            f"held-out gate {key} environment",
        )
        _expect(
            environment,
            "train_verification",
            reference_train_verification,
            f"held-out gate {key} environment",
        )

    for key in ("baseline_M", "baseline_O"):
        _expect(
            gates[key]["student_run_binding"]["environment"],
            "serve_freeze",
            None,
            f"held-out gate {key} baseline environment",
        )
        _expect(
            gates[key]["student_run_binding"]["environment"],
            "serve_verification",
            None,
            f"held-out gate {key} baseline environment",
        )
    reference_serve_freeze = gates["M_M"]["student_run_binding"]["environment"].get(
        "serve_freeze"
    )
    if not isinstance(reference_serve_freeze, dict):
        raise ValueError("M_M lacks a validated teacher serve environment freeze")
    reference_serve_verification = gates["M_M"]["student_run_binding"][
        "environment"
    ].get("serve_verification")
    if not isinstance(reference_serve_verification, dict):
        raise ValueError("M_M lacks exact live teacher serve environment verification")
    for key in ("M_M", "M_O", "O_M", "O_O"):
        _expect(
            gates[key]["student_run_binding"]["environment"],
            "serve_freeze",
            reference_serve_freeze,
            f"held-out gate {key} environment",
        )
        _expect(
            gates[key]["student_run_binding"]["environment"],
            "serve_verification",
            reference_serve_verification,
            f"held-out gate {key} environment",
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
        "teacher_gap_manifest",
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
    per_record_bounds: dict[str, dict[str, list[float]]] = {
        key: {
            record_id: list(bounds)
            for record_id, bounds in gate[
                "record_accuracy_bounds_under_verifier_uncertainty"
            ].items()
        }
        for key, gate in gates.items()
    }
    for key in gates:
        if set(per_record_bounds[key]) != set(per_record[key]) or any(
            len(bounds) != 2
            or bounds[0] != per_record[key][record_id]
            or not 0.0 <= bounds[0] <= bounds[1] <= 1.0
            for record_id, bounds in per_record_bounds[key].items()
        ):
            raise ValueError(f"held-out gate {key} has invalid verifier uncertainty bounds")
    arm_accuracy = {
        key: sum(values.values()) / len(values) for key, values in per_record.items()
    }
    arm_accuracy_bounds = {
        key: [
            sum(bounds[0] for bounds in values.values()) / len(values),
            sum(bounds[1] for bounds in values.values()) / len(values),
        ]
        for key, values in per_record_bounds.items()
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
    observed_delta_bounds = {
        key: [
            arm_accuracy_bounds[key][0] - arm_accuracy_bounds[baseline][1],
            arm_accuracy_bounds[key][1] - arm_accuracy_bounds[baseline][0],
        ]
        for key, baseline in (
            ("M_M", "baseline_M"),
            ("O_M", "baseline_M"),
            ("M_O", "baseline_O"),
            ("O_O", "baseline_O"),
        )
    }
    observed_same_cross_bounds = {
        "M": [
            arm_accuracy_bounds["M_M"][0] - arm_accuracy_bounds["O_M"][1],
            arm_accuracy_bounds["M_M"][1] - arm_accuracy_bounds["O_M"][0],
        ],
        "O": [
            arm_accuracy_bounds["O_O"][0] - arm_accuracy_bounds["M_O"][1],
            arm_accuracy_bounds["O_O"][1] - arm_accuracy_bounds["M_O"][0],
        ],
    }
    observed_stratified_bounds = [
        (observed_same_cross_bounds["M"][0] + observed_same_cross_bounds["O"][0])
        / 2,
        (observed_same_cross_bounds["M"][1] + observed_same_cross_bounds["O"][1])
        / 2,
    ]
    observed_factorial_bounds = [
        observed_same_cross_bounds["M"][0] + observed_same_cross_bounds["O"][0],
        observed_same_cross_bounds["M"][1] + observed_same_cross_bounds["O"][1],
    ]

    record_ids = {
        "M": sorted(per_record["baseline_M"]),
        "O": sorted(per_record["baseline_O"]),
    }
    vectors = {
        key: [per_record[key][record_id] for record_id in record_ids[gate["student_source"]]]
        for key, gate in gates.items()
    }
    lower_vectors = {
        key: [
            per_record_bounds[key][record_id][0]
            for record_id in record_ids[gate["student_source"]]
        ]
        for key, gate in gates.items()
    }
    upper_vectors = {
        key: [
            per_record_bounds[key][record_id][1]
            for record_id in record_ids[gate["student_source"]]
        ]
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
        sample_lower = {
            "baseline_M": _mean_for_indices(lower_vectors["baseline_M"], m_idx),
            "M_M": _mean_for_indices(lower_vectors["M_M"], m_idx),
            "O_M": _mean_for_indices(lower_vectors["O_M"], m_idx),
            "baseline_O": _mean_for_indices(lower_vectors["baseline_O"], o_idx),
            "M_O": _mean_for_indices(lower_vectors["M_O"], o_idx),
            "O_O": _mean_for_indices(lower_vectors["O_O"], o_idx),
        }
        sample_upper = {
            "baseline_M": _mean_for_indices(upper_vectors["baseline_M"], m_idx),
            "M_M": _mean_for_indices(upper_vectors["M_M"], m_idx),
            "O_M": _mean_for_indices(upper_vectors["O_M"], m_idx),
            "baseline_O": _mean_for_indices(upper_vectors["baseline_O"], o_idx),
            "M_O": _mean_for_indices(upper_vectors["M_O"], o_idx),
            "O_O": _mean_for_indices(upper_vectors["O_O"], o_idx),
        }
        for key, baseline in (
            ("M_M", "baseline_M"),
            ("O_M", "baseline_M"),
            ("M_O", "baseline_O"),
            ("O_O", "baseline_O"),
        ):
            bootstrap[f"delta:{key}"].append(sample_means[key] - sample_means[baseline])
            bootstrap[f"uncertainty_lower:delta:{key}"].append(
                sample_lower[key] - sample_upper[baseline]
            )
            bootstrap[f"uncertainty_upper:delta:{key}"].append(
                sample_upper[key] - sample_lower[baseline]
            )
        same_m = sample_means["M_M"] - sample_means["O_M"]
        same_o = sample_means["O_O"] - sample_means["M_O"]
        bootstrap["same_cross:M"].append(same_m)
        bootstrap["same_cross:O"].append(same_o)
        bootstrap["stratified"].append((same_m + same_o) / 2)
        bootstrap["factorial"].append(same_m + same_o)
        lower_same_m = sample_lower["M_M"] - sample_upper["O_M"]
        upper_same_m = sample_upper["M_M"] - sample_lower["O_M"]
        lower_same_o = sample_lower["O_O"] - sample_upper["M_O"]
        upper_same_o = sample_upper["O_O"] - sample_lower["M_O"]
        bootstrap["uncertainty_lower:same_cross:M"].append(lower_same_m)
        bootstrap["uncertainty_upper:same_cross:M"].append(upper_same_m)
        bootstrap["uncertainty_lower:same_cross:O"].append(lower_same_o)
        bootstrap["uncertainty_upper:same_cross:O"].append(upper_same_o)
        bootstrap["uncertainty_lower:stratified"].append(
            (lower_same_m + lower_same_o) / 2
        )
        bootstrap["uncertainty_upper:stratified"].append(
            (upper_same_m + upper_same_o) / 2
        )
        bootstrap["uncertainty_lower:factorial"].append(
            lower_same_m + lower_same_o
        )
        bootstrap["uncertainty_upper:factorial"].append(
            upper_same_m + upper_same_o
        )

    baseline_deltas = {
        key: _contrast(
            observed_delta[key],
            bootstrap[f"delta:{key}"],
            f"accuracy({key}) - accuracy(baseline_{MATRIX_CONTRACT[key]['student_source']})",
            uncertainty_estimate_bounds=observed_delta_bounds[key],
            uncertainty_lower_draws=bootstrap[
                f"uncertainty_lower:delta:{key}"
            ],
            uncertainty_upper_draws=bootstrap[
                f"uncertainty_upper:delta:{key}"
            ],
        )
        for key in ("M_M", "O_M", "M_O", "O_O")
    }
    same_vs_cross = {
        "M": _contrast(
            observed_same_cross_m,
            bootstrap["same_cross:M"],
            "accuracy(M_M) - accuracy(O_M)",
            uncertainty_estimate_bounds=observed_same_cross_bounds["M"],
            uncertainty_lower_draws=bootstrap["uncertainty_lower:same_cross:M"],
            uncertainty_upper_draws=bootstrap["uncertainty_upper:same_cross:M"],
        ),
        "O": _contrast(
            observed_same_cross_o,
            bootstrap["same_cross:O"],
            "accuracy(O_O) - accuracy(M_O)",
            uncertainty_estimate_bounds=observed_same_cross_bounds["O"],
            uncertainty_lower_draws=bootstrap["uncertainty_lower:same_cross:O"],
            uncertainty_upper_draws=bootstrap["uncertainty_upper:same_cross:O"],
        ),
        "equal_stratum_mean": _contrast(
            observed_stratified,
            bootstrap["stratified"],
            "0.5 * ((M_M - O_M) + (O_O - M_O))",
            uncertainty_estimate_bounds=observed_stratified_bounds,
            uncertainty_lower_draws=bootstrap["uncertainty_lower:stratified"],
            uncertainty_upper_draws=bootstrap["uncertainty_upper:stratified"],
        ),
    }
    stratified_interaction = _contrast(
        observed_factorial,
        bootstrap["factorial"],
        "(M_M - O_M) - (M_O - O_O)",
        uncertainty_estimate_bounds=observed_factorial_bounds,
        uncertainty_lower_draws=bootstrap["uncertainty_lower:factorial"],
        uncertainty_upper_draws=bootstrap["uncertainty_upper:factorial"],
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
        "arm_accuracy_bounds_under_verifier_uncertainty": arm_accuracy_bounds,
        "arm_records": {key: gate["records"] for key, gate in gates.items()},
        "baseline_deltas": baseline_deltas,
        "same_vs_cross": same_vs_cross,
        "stratified_interaction": stratified_interaction,
        "environment_freezes": {
            "train": reference_train_freeze,
            "serve": reference_serve_freeze,
        },
        "evaluation_contract": EVALUATION_CONTRACT,
        "evaluation_environment": reference["evaluation_binding"][
            "evaluation_environment"
        ],
        "evaluation_contract_sha256_by_arm": {
            key: gate["evaluation_binding"]["evaluation_contract_sha256"]
            for key, gate in gates.items()
        },
        "evaluation_post_promotion_custody_by_arm": {
            key: gate["evaluation_binding"]["evaluation_post_promotion_custody"]
            for key, gate in gates.items()
        },
        "result_builder": result_builder,
        "bootstrap": {
            "unit": "paired record within held-out source stratum",
            "draws": draws,
            "seed": seed,
            "confidence_interval": "percentile_95",
            "verifier_uncertainty": "binary_worst_case_bootstrap_envelope_v1",
            "classification_rule": {
                "helps": "worst_case_envelope_lower_bound > 0",
                "harms": "worst_case_envelope_upper_bound < 0",
                "inconclusive": "worst_case_envelope includes 0",
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
    def displayed_interval(contrast: Mapping[str, Any]) -> list[float]:
        sensitivity = contrast.get("verifier_uncertainty_sensitivity")
        if isinstance(sensitivity, dict):
            return list(sensitivity["bootstrap_95_envelope"])
        return list(contrast["bootstrap_95_ci"])

    for key in payload["matrix_keys"]:
        accuracy = float(payload["arm_accuracy"][key])
        if key.startswith("baseline_"):
            lines.append(f"| `{key}` | {accuracy:.6f} | — | — | baseline |")
        else:
            contrast = deltas[key]
            low, high = displayed_interval(contrast)
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
        low, high = displayed_interval(contrast)
        lines.append(
            f"| {label} | {contrast['estimate']:+.6f} | "
            f"[{low:+.6f}, {high:+.6f}] | {contrast['classification']} |"
        )
    interaction = payload["stratified_interaction"]
    low, high = displayed_interval(interaction)
    lines.extend(
        [
            "",
            "## Stratified interaction",
            "",
            f"`{interaction['formula']}` = **{interaction['estimate']:+.6f}** "
            f"(95% CI [{low:+.6f}, {high:+.6f}]; {interaction['classification']}).",
            "",
            "Displayed intervals and classifications include the worst-case binary "
            "envelope for every bounded verifier-uncertain sample.",
            "",
            str(payload["claim_boundary"]),
            "",
        ]
    )
    return "\n".join(lines)


def _validate_o_m_diagnostic_external_custody(
    *,
    run: Mapping[str, Any],
    gates: Mapping[str, Any],
    binding: Mapping[str, Any],
    student_model: str,
    student_revision: str,
    prepared_path: Path,
    prepared_hash: str,
    commit: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Recompute every pre-existing gate/environment used by the diagnostic."""

    support_identity = _validate_support_gate(
        gates.get("student_support"),
        source="M",
        model=student_model,
        revision=student_revision,
        prepared_path=prepared_path,
        prepared_hash=prepared_hash,
    )
    environment_identity = _validate_environment(
        binding.get("environment_contract"), commit=commit, requires_teacher=True
    )
    teacher_identity = _validate_teacher_identity(
        run=run,
        teacher_gate=gates.get("teacher_gap"),
        provenance=gates.get("teacher_provenance"),
        tokenizer_contract=gates.get("tokenizer_contract"),
        server_contract=gates.get("server_scoring_contract"),
        teacher_source="O",
        student_model=student_model,
        student_revision=student_revision,
        prepared_path=prepared_path,
        prepared_hash=prepared_hash,
        commit=commit,
    )
    return support_identity, environment_identity, teacher_identity


def _validate_o_m_diagnostic_identity(
    identity: Any,
    *,
    commit: str,
) -> dict[str, Any]:
    """Validate the clean one-step O_M plumbing gate sealed before preregistration."""

    if not isinstance(identity, dict) or set(identity) != set(
        O_M_DIAGNOSTIC_IDENTITY_FIELDS
    ):
        raise ValueError(
            "O-teacher preregistration lacks the exact one-step diagnostic identity"
        )
    for field, expected in (
        ("diagnostic_clean_before_preregistration", True),
        ("plumbing_only", True),
        ("scientific_result", False),
    ):
        _expect(identity, field, expected, "O_M one-step diagnostic")
    for field in (
        "terminal_audit",
        "run_manifest",
        "completion_manifest",
        "student_adapter",
    ):
        value = identity.get(field)
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ValueError(f"O_M one-step diagnostic {field} must be absolute")
    for field in (
        "terminal_audit_sha256",
        "run_manifest_sha256",
        "completion_manifest_sha256",
        "student_adapter_tree_sha256",
    ):
        _hash_identity(identity.get(field), f"O_M one-step diagnostic {field}")

    audit_path, audit = _sealed_json_file(
        Path(identity["terminal_audit"]),
        identity["terminal_audit_sha256"],
        "O_M one-step diagnostic terminal audit",
    )
    _expect(audit, "schema_version", 1, "O_M one-step diagnostic terminal audit")
    _expect(
        audit,
        "audit",
        O_M_ONE_STEP_DIAGNOSTIC_AUDIT,
        "O_M one-step diagnostic terminal audit",
    )
    for field, expected in (
        ("passed", True),
        ("diagnostic_clean_before_preregistration", True),
        ("plumbing_only", True),
        ("scientific_result", False),
        ("matrix_key", "O_M"),
        ("git_commit", commit),
    ):
        _expect(audit, field, expected, "O_M one-step diagnostic terminal audit")

    audit_inputs = audit.get("inputs")
    if not isinstance(audit_inputs, dict) or set(audit_inputs) != {
        "run_manifest",
        "run_manifest_sha256",
        "completion_manifest",
        "completion_manifest_sha256",
        "student_adapter",
        "student_adapter_tree_sha256",
    }:
        raise ValueError("O_M one-step diagnostic terminal audit inputs are invalid")
    for field in audit_inputs:
        _expect(
            audit_inputs,
            field,
            identity[field],
            "O_M one-step diagnostic terminal audit",
        )

    run_path = Path(identity["run_manifest"]).resolve()
    completion_path = Path(identity["completion_manifest"]).resolve()
    if (
        run_path.name != "run_manifest.json"
        or completion_path.name != "completion_manifest.json"
        or run_path.parent != completion_path.parent
    ):
        raise ValueError("O_M one-step diagnostic run/completion paths are not canonical siblings")
    _file_binding(
        run_path,
        identity["run_manifest_sha256"],
        "O_M one-step diagnostic run manifest",
    )
    _file_binding(
        completion_path,
        identity["completion_manifest_sha256"],
        "O_M one-step diagnostic completion manifest",
    )
    run = _json_object(run_path, "O_M one-step diagnostic run manifest")
    completion = _json_object(
        completion_path, "O_M one-step diagnostic completion manifest"
    )
    if run.get("completion") != completion:
        raise ValueError("O_M one-step diagnostic run does not embed its completion")
    for payload, label in ((run, "diagnostic run"), (completion, "diagnostic completion")):
        _expect(payload, "schema_version", 1, label)
        _expect(payload, "objective", "task_rl_k1_gap", label)
        _expect(payload, "intended_scientific_run", False, label)
    if completion.get("status") not in {
        "completed",
        "completed_zero_task_signal_smoke",
    }:
        raise ValueError("O_M one-step diagnostic did not complete its plumbing run")
    for field, expected in (
        ("optimizer_steps_planned", 1),
        ("micro_prompts_per_step", 1),
        ("planned_rollout_samples", 4),
    ):
        _expect(run, field, expected, "O_M one-step diagnostic run")
    for field, expected in (
        ("optimizer_steps_completed", 1),
        ("rollout_samples", 4),
        ("step_trace_rows", 1),
        ("sample_trace_rows", 4),
        ("realized_training_geometry_observed", True),
        ("finite_nonzero_gradient_observed", True),
        ("parameter_update_observed", True),
        ("clean_stable_code", True),
        ("stable_training_environment", True),
        ("stable_environment_after_candidate_save", True),
        ("stable_environment_end", True),
        ("stable_final_artifact_hash", True),
        ("live_local_server_process_binding_validated", True),
        ("training_artifact_eligible_for_held_out_evaluation", False),
        ("scientific_use_allowed", False),
    ):
        _expect(completion, field, expected, "O_M one-step diagnostic completion")
    if completion.get("initial_parameter_signature") == completion.get(
        "final_parameter_signature"
    ):
        raise ValueError("O_M one-step diagnostic lacks a parameter-signature change")
    binding = run.get("binding")
    if not isinstance(binding, dict):
        raise ValueError("O_M one-step diagnostic lacks its run binding")
    for field, expected in (
        ("pair_id", "O_M"),
        ("student_source", "M"),
        ("teacher_source", "O"),
        ("budget_mode", "dose_response"),
        ("local_checkpoint_custody_validated", True),
        ("server_alias_and_token_contract_validated", True),
        ("live_local_server_process_binding_validated", True),
        ("serve_environment_process_binding_validated", True),
    ):
        _expect(binding, field, expected, "O_M one-step diagnostic binding")
    _expect(run, "git_commit", commit, "O_M one-step diagnostic run")
    _expect(run, "git_worktree_clean", True, "O_M one-step diagnostic run")
    _clean_state(run.get("git_state_start"), commit, "O_M diagnostic git start")
    for field in (
        "git_state_start",
        "git_state_training_end",
        "git_state_after_candidate_save",
        "git_state_end",
    ):
        _clean_state(
            completion.get(field),
            commit,
            f"O_M diagnostic completion {field}",
        )

    canonical_plan = _json_object(
        CANONICAL_STUDENT_TRAINING_PLAN, "canonical student training plan"
    )
    canonical_fixed = canonical_plan.get("fixed_config")
    if not isinstance(canonical_fixed, dict):
        raise ValueError("canonical student training plan lacks fixed_config")
    diagnostic_fixed = dict(canonical_fixed)
    diagnostic_fixed.update({"budget_mode": "dose_response", "optimizer_steps": 1})
    _expect(
        run,
        "normalized_training_config",
        diagnostic_fixed,
        "O_M one-step diagnostic run",
    )
    _expect(run, "seed", diagnostic_fixed["seed"], "O_M one-step diagnostic run")
    _expect(
        run,
        "generation",
        {
            "group_size": diagnostic_fixed["group_size"],
            "temperature": diagnostic_fixed["temperature"],
            "top_p": diagnostic_fixed["top_p"],
            "top_k": diagnostic_fixed["top_k"],
            "max_new_tokens": diagnostic_fixed["max_new_tokens"],
            "enable_thinking": diagnostic_fixed["enable_thinking"],
        },
        "O_M one-step diagnostic run",
    )
    _expect(
        run,
        "optimization",
        {
            "attn_implementation": diagnostic_fixed["attn_implementation"],
            "gradient_checkpointing": diagnostic_fixed["gradient_checkpointing"],
            "learning_rate": diagnostic_fixed["learning_rate"],
            "lora_r": diagnostic_fixed["lora_r"],
        },
        "O_M one-step diagnostic run",
    )
    _expect(
        run,
        "loss",
        {
            "task_reward_coef": diagnostic_fixed["task_reward_coef"],
            "k1_coef": diagnostic_fixed["k1_coef"],
            "gap_gate_beta": diagnostic_fixed["gap_gate_beta"],
            "advantage_clip": diagnostic_fixed["advantage_clip"],
        },
        "O_M one-step diagnostic run",
    )

    gates = run.get("gates")
    if not isinstance(gates, dict):
        raise ValueError("O_M one-step diagnostic lacks scientific gate custody")
    prepared_gate = gates.get("prepared_data")
    if not isinstance(prepared_gate, dict):
        raise ValueError("O_M one-step diagnostic lacks prepared-data custody")
    prepared_path = Path(str(prepared_gate.get("path"))).resolve()
    prepared_hash = _file_binding(
        prepared_path,
        prepared_gate.get("sha256"),
        "O_M one-step diagnostic prepared manifest",
    )
    prepared = _json_object(prepared_path, "O_M diagnostic prepared manifest")
    _expect(prepared, "scientific_use_allowed", True, "O_M diagnostic prepared manifest")
    train_relative = "roles/M/student_opd.jsonl"
    _expect(
        prepared_gate,
        "task_role_file",
        train_relative,
        "O_M diagnostic prepared gate",
    )
    train_path = (prepared_path.parent / train_relative).resolve()
    train_entry = prepared.get("files", {}).get(train_relative)
    if not isinstance(train_entry, dict):
        raise ValueError("O_M diagnostic prepared manifest lacks the M student role")
    train_hash = _file_binding(
        train_path, train_entry.get("sha256"), "O_M diagnostic training role"
    )
    _expect(run, "task_file", str(train_path), "O_M one-step diagnostic run")
    _expect(run, "task_file_sha256", train_hash, "O_M one-step diagnostic run")
    _expect(
        prepared_gate,
        "task_file_sha256",
        train_hash,
        "O_M diagnostic prepared gate",
    )
    task_limit = run.get("task_limit")
    selected_rows = run.get("selected_task_rows")
    if (
        not isinstance(task_limit, int)
        or task_limit <= 0
        or selected_rows != task_limit
        or task_limit > int(train_entry.get("rows", 0))
    ):
        raise ValueError("O_M diagnostic task-limit custody is invalid")
    training_rows = list(iter_jsonl(train_path))[:task_limit]
    if len(training_rows) != task_limit:
        raise ValueError("O_M diagnostic training role is shorter than its task limit")

    student_model = run.get("student")
    student_revision = run.get("student_revision")
    if not isinstance(student_model, str) or not student_model:
        raise ValueError("O_M diagnostic lacks its student model")
    _hash_identity(student_revision, "O_M diagnostic student revision", HEX40)
    support_identity, environment_identity, teacher_identity = (
        _validate_o_m_diagnostic_external_custody(
            run=run,
            gates=gates,
            binding=binding,
            student_model=student_model,
            student_revision=student_revision,
            prepared_path=prepared_path,
            prepared_hash=prepared_hash,
            commit=commit,
        )
    )

    adapter = Path(identity["student_adapter"]).resolve()
    adapter_hash = sha256_tree(adapter)
    _expect(
        identity,
        "student_adapter_tree_sha256",
        adapter_hash,
        "O_M one-step diagnostic",
    )
    _expect(
        completion,
        "final_adapter",
        str(adapter),
        "O_M one-step diagnostic completion",
    )
    _expect(
        completion,
        "final_adapter_tree_sha256",
        adapter_hash,
        "O_M one-step diagnostic completion",
    )
    trace_identity = _validate_trace_artifacts(
        run=run,
        completion=completion,
        completion_path=completion_path,
        training_rows=training_rows,
        objective="task_rl_k1_gap",
        source="M",
        fixed=diagnostic_fixed,
        require_task_signal=False,
    )
    expected_task_signal = trace_identity["informative_group_fraction"] >= float(
        diagnostic_fixed["min_informative_group_fraction"]
    )
    _expect(
        completion,
        "task_signal_observed",
        expected_task_signal,
        "O_M one-step diagnostic completion",
    )
    return {
        "terminal_audit": str(audit_path),
        **{field: identity[field] for field in O_M_DIAGNOSTIC_IDENTITY_FIELDS if field != "terminal_audit"},
        "validated_student_support": support_identity,
        "validated_teacher": teacher_identity,
        "validated_environment": environment_identity,
        "validated_trace": trace_identity,
    }


def _absolute_regular_file(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError(f"{label} path must be absolute")
    raw = Path(value).expanduser()
    if raw.is_symlink() or not raw.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    return raw.resolve()


def _absolute_regular_directory(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError(f"{label} path must be absolute")
    raw = Path(value).expanduser()
    if raw.is_symlink() or not raw.is_dir():
        raise ValueError(f"{label} must be a regular non-symlink directory")
    return raw.resolve()


def _legacy_m_file_binding(value: Any, label: str) -> tuple[Path, str]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError(f"M-negative compatibility audit has invalid {label} binding")
    path = _absolute_regular_file(value.get("path"), label)
    expected = _hash_identity(value.get("sha256"), f"{label} hash")
    return path, _file_binding(path, expected, label)


def _legacy_m_json_binding(value: Any, label: str) -> tuple[Path, dict[str, Any]]:
    if not isinstance(value, dict) or set(value) != {
        "path",
        "sha256",
        "payload_sha256",
    }:
        raise ValueError(f"M-negative compatibility audit has invalid {label} binding")
    path = _absolute_regular_file(value.get("path"), label)
    _file_binding(path, _hash_identity(value.get("sha256"), f"{label} hash"), label)
    payload = _json_object(path, label)
    _expect(
        value,
        "payload_sha256",
        canonical_json_sha256(payload),
        f"{label} binding",
    )
    return path, payload


def _legacy_m_strict_verdict(completion: str, solution: str) -> dict[str, Any]:
    """Replay one historical completion with the current strict eval scorer."""

    return verify_evaluation_completion(completion, solution)


def _legacy_m_sample_surfaces(
    path: Path,
    *,
    label: str,
    task_rows: Mapping[str, dict[str, Any]],
    task_order: Mapping[str, int],
    samples_per_problem: int,
) -> tuple[
    dict[str, list[float]],
    dict[str, list[float]],
    list[dict[str, Any]],
]:
    stored_by_key: dict[tuple[str, int], float] = {}
    strict_by_key: dict[tuple[str, int], float] = {}
    verifier_errors: list[dict[str, Any]] = []
    for row_number, row in enumerate(iter_jsonl(path), 1):
        if row.get("schema_version") != 2:
            raise ValueError(f"{label} row {row_number} is not legacy evaluation schema 2")
        record_id = row.get("record_id")
        sample_idx = row.get("sample_idx")
        if (
            not isinstance(record_id, str)
            or record_id not in task_rows
            or not isinstance(sample_idx, int)
            or not 0 <= sample_idx < samples_per_problem
        ):
            raise ValueError(f"{label} row {row_number} has an invalid sample identity")
        key = (record_id, sample_idx)
        if key in stored_by_key:
            raise ValueError(f"{label} contains duplicate sample identity {key}")
        _expect(row, "source", "M", f"{label} row {row_number}")
        _expect(
            row,
            "global_record_index",
            task_order[record_id],
            f"{label} row {row_number}",
        )
        completion = row.get("completion_text")
        if not isinstance(completion, str):
            raise ValueError(f"{label} row {row_number} lacks completion text")
        _expect(
            row,
            "completion_sha256",
            hashlib.sha256(completion.encode("utf-8")).hexdigest(),
            f"{label} row {row_number}",
        )
        reward = row.get("reward")
        if type(reward) not in (int, float) or float(reward) not in {0.0, 1.0}:
            raise ValueError(f"{label} row {row_number} has a non-binary stored reward")
        if not isinstance(row.get("reward_status"), str):
            raise ValueError(f"{label} row {row_number} lacks its legacy reward status")
        stored_by_key[key] = float(reward)

        verdict = _legacy_m_strict_verdict(
            completion, str(task_rows[record_id]["solution"])
        )
        strict_reward = verdict.get("reward")
        if strict_reward is None:
            raise RuntimeError(
                f"current strict scorer cannot score {label} sample {key}: {verdict}"
            )
        if type(strict_reward) not in (int, float) or float(strict_reward) not in {
            0.0,
            1.0,
        }:
            raise ValueError(f"current strict scorer returned a non-binary reward for {key}")
        strict_by_key[key] = float(strict_reward)
        if verdict.get("status") == "verifier_error_zeroed":
            verifier_errors.append(
                {
                    "record_id": record_id,
                    "sample_idx": sample_idx,
                    "stage": verdict.get("verifier_stage"),
                    "error_type": verdict.get("verifier_error_type"),
                }
            )

    expected_keys = {
        (record_id, sample_idx)
        for record_id in task_rows
        for sample_idx in range(samples_per_problem)
    }
    if set(stored_by_key) != expected_keys:
        missing = sorted(expected_keys - set(stored_by_key))[:10]
        extra = sorted(set(stored_by_key) - expected_keys)[:10]
        raise ValueError(
            f"{label} does not cover the exact task/sample surface; "
            f"missing={missing}, extra={extra}"
        )
    stored = {
        record_id: [stored_by_key[(record_id, idx)] for idx in range(samples_per_problem)]
        for record_id in task_rows
    }
    strict = {
        record_id: [strict_by_key[(record_id, idx)] for idx in range(samples_per_problem)]
        for record_id in task_rows
    }
    verifier_errors.sort(key=lambda item: (item["record_id"], item["sample_idx"]))
    return stored, strict, verifier_errors


def _legacy_m_accuracy(surface: Mapping[str, list[float]]) -> float:
    return sum(sum(values) / len(values) for values in surface.values()) / len(surface)


def _legacy_m_assign_errors(
    surface: Mapping[str, list[float]],
    errors: Iterable[Mapping[str, Any]],
    value: float,
) -> dict[str, list[float]]:
    assigned = {record_id: list(values) for record_id, values in surface.items()}
    for error in errors:
        assigned[str(error["record_id"])][int(error["sample_idx"])] = value
    return assigned


def _legacy_m_bootstrap_result(
    base: dict[str, list[float]],
    trained: dict[str, list[float]],
    *,
    seed: int,
    draws: int,
    min_delta: float,
    min_records: int,
) -> dict[str, Any]:
    keys, delta, low, high = bootstrap_delta(base, trained, seed, draws)
    base_accuracy = _legacy_m_accuracy(base)
    trained_accuracy = _legacy_m_accuracy(trained)
    requirements = {
        "minimum_records_met": len(keys) >= min_records,
        "strict_delta_met": delta > min_delta,
        "positive_bootstrap_lower_bound_met": low > 0,
    }
    return {
        "records": len(keys),
        "base_accuracy": base_accuracy,
        "trained_accuracy": trained_accuracy,
        "paired_delta": delta,
        "bootstrap_95_ci": [low, high],
        "requirements": requirements,
        "passed": all(requirements.values()),
    }


def _legacy_m_validate_summary_and_custody(
    *,
    label: str,
    summary_path: Path,
    summary: Mapping[str, Any],
    samples_path: Path,
    custody_path: Path,
    custody: Mapping[str, Any],
    gate: Mapping[str, Any],
    task_path: Path,
    task_rows: int,
    adapter: Path | None,
    adapter_tree_sha256: str | None,
) -> None:
    expected_samples = task_rows * 4
    for field, expected in (
        ("schema_version", 2),
        ("artifact_kind", EVALUATION_MERGED_KIND),
        ("model", gate["base_model"]),
        ("model_revision", gate["base_model_revision"]),
        ("task_file", str(task_path)),
        ("task_file_sha256", gate["task_file_sha256"]),
        ("task_sources", ["M"]),
        ("task_roles", ["teacher_gap_dev"]),
        ("records", task_rows),
        ("samples", expected_samples),
        ("samples_per_problem", 4),
        ("decoding", gate["decoding"]),
        ("samples_file", "samples.jsonl"),
        ("samples_file_sha256", sha256_file(samples_path)),
        ("completion_text_in_samples", True),
        ("adapter", None if adapter is None else str(adapter)),
        ("adapter_tree_sha256", adapter_tree_sha256),
    ):
        _expect(summary, field, expected, f"legacy M {label} summary")
    if summary_path.parent != samples_path.parent:
        raise ValueError(f"legacy M {label} summary and samples are not one merged tree")
    for field, expected in (
        ("schema_version", 1),
        ("artifact_kind", EVALUATION_MERGED_KIND),
        ("model", gate["base_model"]),
        ("model_revision", gate["base_model_revision"]),
        ("task_file_sha256", gate["task_file_sha256"]),
        ("summary", str(summary_path)),
        ("summary_sha256", sha256_file(summary_path)),
        ("samples", str(samples_path)),
        ("samples_sha256", sha256_file(samples_path)),
        ("output_dir", str(summary_path.parent)),
        ("adapter_tree_sha256", adapter_tree_sha256),
        ("publication_commit_point", True),
        ("stable_environment_after_promotion", True),
        ("stable_final_artifact_hash", True),
    ):
        _expect(custody, field, expected, f"legacy M {label} custody")
    expected_tree = sha256_tree(summary_path.parent)
    _expect(custody, "output_tree_sha256", expected_tree, f"legacy M {label} custody")
    gate_custody = gate[f"{label}_evaluation_post_promotion_custody"]
    if not isinstance(gate_custody, dict):
        raise ValueError(f"legacy M gate lacks {label} evaluation custody")
    for field, expected in (
        ("path", str(custody_path)),
        ("sha256", sha256_file(custody_path)),
        ("tree_sha256", expected_tree),
    ):
        _expect(gate_custody, field, expected, f"legacy M {label} gate custody")


def _replay_m_negative_compatibility(
    inputs: Any,
    *,
    legacy_commit: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_input_keys = {
        "teacher_gap_manifest",
        "terminal_audit",
        "base_summary",
        "base_samples",
        "base_custody",
        "trained_summary",
        "trained_samples",
        "trained_custody",
        "task_file",
        "teacher_run_manifest",
        "trained_adapter",
    }
    if not isinstance(inputs, dict) or set(inputs) != expected_input_keys:
        raise ValueError("M-negative compatibility audit has an invalid exact input set")
    gate_path, gate = _legacy_m_json_binding(
        inputs["teacher_gap_manifest"], "legacy M teacher-gap manifest"
    )
    terminal_path, terminal = _legacy_m_json_binding(
        inputs["terminal_audit"], "legacy M terminal audit"
    )
    _expect(gate, "schema_version", 3, "legacy M teacher-gap manifest")
    _expect(gate, "gate", "teacher_gap_v1", "legacy M teacher-gap manifest")
    _expect(gate, "gate_strength", "scientific", "legacy M teacher-gap manifest")
    _expect(gate, "passed", False, "legacy M teacher-gap manifest")
    _expect(
        gate,
        "authorizes_scientific_merge",
        False,
        "legacy M teacher-gap manifest",
    )
    _expect(gate, "task_sources", ["M"], "legacy M teacher-gap manifest")
    _expect(gate, "task_roles", ["teacher_gap_dev"], "legacy M teacher-gap manifest")
    _expect(
        gate,
        "evaluation_git_commit",
        legacy_commit,
        "legacy M teacher-gap manifest",
    )
    _expect(
        gate,
        "teacher_training_git_commit",
        legacy_commit,
        "legacy M teacher-gap manifest",
    )
    if gate.get("require_positive_ci") is not True:
        raise ValueError("legacy M teacher gap did not require a positive bootstrap lower bound")
    draws = gate.get("bootstrap_draws")
    seed = gate.get("bootstrap_seed")
    min_records = gate.get("min_records")
    min_delta = gate.get("min_delta")
    if (
        not isinstance(draws, int)
        or draws < 10_000
        or not isinstance(seed, int)
        or not isinstance(min_records, int)
        or min_records <= 0
        or type(min_delta) not in (int, float)
        or float(min_delta) < 0
    ):
        raise ValueError("legacy M teacher gap has invalid preregistered statistics")

    _expect(terminal, "schema_version", 1, "legacy M terminal audit")
    _expect(
        terminal,
        "classification",
        "scientific_teacher_gap_negative_inconclusive_result",
        "legacy M terminal audit",
    )
    _expect(terminal, "git_commit", legacy_commit, "legacy M terminal audit")
    independent = terminal.get("independent_recompute")
    if not isinstance(independent, dict):
        raise ValueError("legacy M terminal audit lacks independent recomputation")
    for field, expected in (
        ("state", "COMPLETED"),
        ("exit_code", "0:0"),
        ("recompute_teacher_gate_equal_to_disk", True),
        ("gate_sha256", sha256_file(gate_path)),
    ):
        _expect(independent, field, expected, "legacy M independent recomputation")
    downstream = terminal.get("downstream_authorization")
    if not isinstance(downstream, dict):
        raise ValueError("legacy M terminal audit lacks downstream authorization")
    for field in ("M_teacher_merge", "M_M_scientific_arm", "M_O_scientific_arm"):
        _expect(downstream, field, False, "legacy M downstream authorization")
    _expect(
        downstream,
        "six_arm_matrix_under_current_campaign",
        False,
        "legacy M downstream authorization",
    )
    forbidden = terminal.get("forbidden_actions_observed")
    if not isinstance(forbidden, dict) or any(forbidden.values()):
        raise ValueError("legacy M terminal audit observed a prohibited M action")

    terminal_artifacts = terminal.get("artifacts")
    if not isinstance(terminal_artifacts, dict):
        raise ValueError("legacy M terminal audit lacks exact artifacts")
    terminal_gate = terminal_artifacts.get("gate")
    if not isinstance(terminal_gate, dict):
        raise ValueError("legacy M terminal audit lacks its gate binding")
    for field, expected in (
        ("path", str(gate_path)),
        ("sha256", sha256_file(gate_path)),
    ):
        _expect(terminal_gate, field, expected, "legacy M terminal gate binding")

    paths: dict[str, Path] = {}
    for key in (
        "base_summary",
        "base_samples",
        "base_custody",
        "trained_summary",
        "trained_samples",
        "trained_custody",
        "task_file",
        "teacher_run_manifest",
    ):
        paths[key], _ = _legacy_m_file_binding(inputs[key], f"legacy M {key}")
    for key in (
        "base_summary",
        "base_samples",
        "base_custody",
        "trained_summary",
        "trained_samples",
        "trained_custody",
    ):
        terminal_binding = terminal_artifacts.get(key)
        if not isinstance(terminal_binding, dict):
            raise ValueError(f"legacy M terminal audit lacks {key}")
        for field, expected in (
            ("path", str(paths[key])),
            ("sha256", sha256_file(paths[key])),
        ):
            _expect(terminal_binding, field, expected, f"legacy M terminal {key}")
    for prefix in ("base", "trained"):
        for suffix in ("summary", "samples"):
            field = f"{prefix}_{suffix}"
            _expect(gate, field, str(paths[field]), "legacy M teacher-gap manifest")
            _expect(
                gate,
                f"{field}_sha256",
                sha256_file(paths[field]),
                "legacy M teacher-gap manifest",
            )
    _expect(gate, "task_file", str(paths["task_file"]), "legacy M teacher-gap manifest")
    _expect(
        gate,
        "task_file_sha256",
        sha256_file(paths["task_file"]),
        "legacy M teacher-gap manifest",
    )
    _expect(
        gate,
        "teacher_run_manifest",
        str(paths["teacher_run_manifest"]),
        "legacy M teacher-gap manifest",
    )
    _expect(
        gate,
        "teacher_run_manifest_sha256",
        sha256_file(paths["teacher_run_manifest"]),
        "legacy M teacher-gap manifest",
    )

    adapter_binding = inputs["trained_adapter"]
    if not isinstance(adapter_binding, dict) or set(adapter_binding) != {
        "path",
        "tree_sha256",
    }:
        raise ValueError("M-negative compatibility audit has invalid adapter binding")
    adapter = _absolute_regular_directory(
        adapter_binding.get("path"), "legacy M trained adapter"
    )
    adapter_hash = sha256_tree(adapter)
    _expect(
        adapter_binding,
        "tree_sha256",
        adapter_hash,
        "legacy M trained adapter binding",
    )
    _expect(gate, "trained_adapter", str(adapter), "legacy M teacher-gap manifest")
    _expect(
        gate,
        "trained_adapter_tree_sha256",
        adapter_hash,
        "legacy M teacher-gap manifest",
    )
    run = _json_object(paths["teacher_run_manifest"], "legacy M teacher run manifest")
    for field, expected in (
        ("schema_version", 1),
        ("status", "completed"),
        ("source", "M"),
        ("role", "teacher_train"),
        ("model", gate["base_model"]),
        ("model_revision", gate["base_model_revision"]),
        ("final_adapter", str(adapter)),
        ("final_adapter_tree_sha256", adapter_hash),
        ("intended_scientific_run", True),
        ("scientific_use_allowed", True),
        ("optimizer_progress_complete", True),
        ("stable_final_artifact_hash", True),
    ):
        _expect(run, field, expected, "legacy M teacher run manifest")
    for state_field in ("git_state_start", "git_state_end"):
        _clean_state(run.get(state_field), legacy_commit, f"legacy M {state_field}")

    task_list = list(iter_jsonl(paths["task_file"]))
    task_rows: dict[str, dict[str, Any]] = {}
    task_order: dict[str, int] = {}
    for index, row in enumerate(task_list):
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or record_id in task_rows:
            raise ValueError("legacy M task file has invalid or duplicate record IDs")
        _expect(row, "source", "M", "legacy M task row")
        _expect(row, "role", "teacher_gap_dev", "legacy M task row")
        if not isinstance(row.get("solution"), str) or not row["solution"]:
            raise ValueError("legacy M task row lacks a solution")
        task_rows[record_id] = row
        task_order[record_id] = index
    if len(task_rows) != gate.get("shared_records"):
        raise ValueError("legacy M task rows differ from the gate record count")

    summaries = {
        prefix: _json_object(paths[f"{prefix}_summary"], f"legacy M {prefix} summary")
        for prefix in ("base", "trained")
    }
    custodies = {
        prefix: _json_object(paths[f"{prefix}_custody"], f"legacy M {prefix} custody")
        for prefix in ("base", "trained")
    }
    _legacy_m_validate_summary_and_custody(
        label="base",
        summary_path=paths["base_summary"],
        summary=summaries["base"],
        samples_path=paths["base_samples"],
        custody_path=paths["base_custody"],
        custody=custodies["base"],
        gate=gate,
        task_path=paths["task_file"],
        task_rows=len(task_rows),
        adapter=None,
        adapter_tree_sha256=None,
    )
    _legacy_m_validate_summary_and_custody(
        label="trained",
        summary_path=paths["trained_summary"],
        summary=summaries["trained"],
        samples_path=paths["trained_samples"],
        custody_path=paths["trained_custody"],
        custody=custodies["trained"],
        gate=gate,
        task_path=paths["task_file"],
        task_rows=len(task_rows),
        adapter=adapter,
        adapter_tree_sha256=adapter_hash,
    )

    base_stored, base_strict, base_errors = _legacy_m_sample_surfaces(
        paths["base_samples"],
        label="legacy M base samples",
        task_rows=task_rows,
        task_order=task_order,
        samples_per_problem=4,
    )
    trained_stored, trained_strict, trained_errors = _legacy_m_sample_surfaces(
        paths["trained_samples"],
        label="legacy M trained samples",
        task_rows=task_rows,
        task_order=task_order,
        samples_per_problem=4,
    )
    legacy_result = _legacy_m_bootstrap_result(
        base_stored,
        trained_stored,
        seed=seed,
        draws=draws,
        min_delta=float(min_delta),
        min_records=min_records,
    )
    for field, expected in (
        ("records", gate["shared_records"]),
        ("base_accuracy", gate["base_accuracy"]),
        ("trained_accuracy", gate["trained_accuracy"]),
        ("paired_delta", gate["paired_delta"]),
        ("bootstrap_95_ci", gate["bootstrap_95_ci"]),
        ("requirements", gate["requirements"]),
        ("passed", False),
    ):
        _expect(legacy_result, field, expected, "legacy M stored-reward recomputation")
    for prefix in ("base", "trained"):
        _expect(
            summaries[prefix],
            "accuracy",
            legacy_result[f"{prefix}_accuracy"],
            f"legacy M {prefix} summary",
        )
    terminal_gate_result = terminal.get("gate_result")
    expected_terminal_gate_result = {
        field: gate[field]
        for field in (
            "schema_version",
            "gate",
            "gate_strength",
            "passed",
            "authorizes_scientific_merge",
            "shared_records",
            "base_accuracy",
            "trained_accuracy",
            "paired_delta",
            "bootstrap_95_ci",
            "min_delta",
            "min_records",
            "require_positive_ci",
            "bootstrap_draws",
            "bootstrap_seed",
            "requirements",
        )
    }
    if terminal_gate_result != expected_terminal_gate_result:
        raise ValueError("legacy M terminal gate result differs from the sealed gate")

    point = _legacy_m_bootstrap_result(
        base_strict,
        trained_strict,
        seed=seed,
        draws=draws,
        min_delta=float(min_delta),
        min_records=min_records,
    )
    pessimistic = _legacy_m_bootstrap_result(
        _legacy_m_assign_errors(base_strict, base_errors, 1.0),
        _legacy_m_assign_errors(trained_strict, trained_errors, 0.0),
        seed=seed,
        draws=draws,
        min_delta=float(min_delta),
        min_records=min_records,
    )
    teacher_favorable = _legacy_m_bootstrap_result(
        _legacy_m_assign_errors(base_strict, base_errors, 0.0),
        _legacy_m_assign_errors(trained_strict, trained_errors, 1.0),
        seed=seed,
        draws=draws,
        min_delta=float(min_delta),
        min_records=min_records,
    )
    if point["passed"]:
        raise ValueError("current strict scorer no longer confirms the M-negative gate")
    error_count = len(base_errors) + len(trained_errors)
    if error_count and teacher_favorable["passed"]:
        raise ValueError(
            "M-negative claim is not robust to teacher-favorable verifier-error assignment"
        )
    strict_replay = {
        "scorer": "current_verify_evaluation_completion",
        "verifier_error_policy": EVALUATION_VERIFIER_ERROR_POLICY,
        "verifier_max_attempts": EVALUATION_VERIFIER_MAX_ATTEMPTS,
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
        "min_delta": float(min_delta),
        "min_records": min_records,
        "base_verifier_errors": base_errors,
        "trained_verifier_errors": trained_errors,
        "verifier_error_count": error_count,
        "negative_proof_policy": "zero_errors_or_teacher_favorable_assignment_v1",
        "point_strict_replay": point,
        "worst_case_for_improvement": {
            "assignment": "base_errors_correct_trained_errors_incorrect",
            **pessimistic,
        },
        "teacher_favorable_error_assignment": {
            "assignment": "base_errors_incorrect_trained_errors_correct",
            **teacher_favorable,
        },
        "negative_gate_confirmed": True,
        "error_policy_satisfied": error_count == 0 or not teacher_favorable["passed"],
    }
    legacy_recomputation = {
        "scorer": "sealed_legacy_stored_rewards",
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
        "result": legacy_result,
        "equal_to_original_gate": True,
        "terminal_audit_equal_to_original_gate": True,
    }
    return legacy_recomputation, strict_replay


def _derive_m_negative_compatibility_inputs(args: Namespace) -> tuple[dict[str, Any], str]:
    gate_path = _absolute_regular_file(
        str(Path(args.teacher_gap_manifest).resolve()), "legacy M teacher-gap manifest"
    )
    terminal_path = _absolute_regular_file(
        str(Path(args.terminal_audit).resolve()), "legacy M terminal audit"
    )
    gate = _json_object(gate_path, "legacy M teacher-gap manifest")
    terminal = _json_object(terminal_path, "legacy M terminal audit")
    legacy_commit = _hash_identity(
        gate.get("evaluation_git_commit"), "legacy M producer commit", HEX40
    )
    terminal_artifacts = terminal.get("artifacts")
    if not isinstance(terminal_artifacts, dict):
        raise ValueError("legacy M terminal audit lacks exact artifacts")
    gate_terminal = terminal_artifacts.get("gate")
    if not isinstance(gate_terminal, dict):
        raise ValueError("legacy M terminal audit lacks its gate")
    if gate_terminal != {"path": str(gate_path), "sha256": sha256_file(gate_path)}:
        raise ValueError("supplied legacy M gate differs from the terminal audit")
    inputs: dict[str, Any] = {
        "teacher_gap_manifest": {
            "path": str(gate_path),
            "sha256": sha256_file(gate_path),
            "payload_sha256": canonical_json_sha256(gate),
        },
        "terminal_audit": {
            "path": str(terminal_path),
            "sha256": sha256_file(terminal_path),
            "payload_sha256": canonical_json_sha256(terminal),
        },
    }
    for key in (
        "base_summary",
        "base_samples",
        "base_custody",
        "trained_summary",
        "trained_samples",
        "trained_custody",
    ):
        value = terminal_artifacts.get(key)
        if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
            raise ValueError(f"legacy M terminal audit lacks exact {key}")
        inputs[key] = dict(value)
    task_path = _absolute_regular_file(gate.get("task_file"), "legacy M task file")
    run_path = _absolute_regular_file(
        str(Path(args.teacher_run_manifest).resolve()), "legacy M teacher run manifest"
    )
    if str(run_path) != gate.get("teacher_run_manifest"):
        raise ValueError("supplied legacy M run manifest differs from the teacher gate")
    inputs["task_file"] = {
        "path": str(task_path),
        "sha256": _hash_identity(gate.get("task_file_sha256"), "legacy M task hash"),
    }
    inputs["teacher_run_manifest"] = {
        "path": str(run_path),
        "sha256": _hash_identity(
            gate.get("teacher_run_manifest_sha256"), "legacy M run manifest hash"
        ),
    }
    adapter = _absolute_regular_directory(
        str(Path(args.trained_adapter).resolve()), "legacy M trained adapter"
    )
    if str(adapter) != gate.get("trained_adapter"):
        raise ValueError("supplied legacy M adapter differs from the teacher gate")
    inputs["trained_adapter"] = {
        "path": str(adapter),
        "tree_sha256": _hash_identity(
            gate.get("trained_adapter_tree_sha256"), "legacy M adapter tree hash"
        ),
    }
    return inputs, legacy_commit


def m_teacher_negative_compatibility_audit(args: Namespace) -> dict[str, Any]:
    """Build a negative-only current-scorer replay of the historical M gate."""

    inputs, legacy_commit = _derive_m_negative_compatibility_inputs(args)
    state = git_state()
    audit_commit = _hash_identity(state.get("commit"), "compatibility audit commit", HEX40)
    _clean_state(state, audit_commit, "M-negative compatibility audit builder")
    legacy_recomputation, strict_replay = _replay_m_negative_compatibility(
        inputs, legacy_commit=legacy_commit
    )
    created_utc = getattr(args, "created_utc", None)
    if created_utc is None:
        created_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    _parse_utc_timestamp(created_utc, "M-negative compatibility audit created_utc")
    return {
        "schema_version": 1,
        "audit": M_TEACHER_NEGATIVE_SELECTION_AUDIT,
        "created_utc": created_utc,
        "selection_context_only": True,
        "scientific_authorization": False,
        "m_teacher_gate_passed": False,
        "merge_authorized": False,
        "m_arms_prohibited": True,
        "m_m_arm_authorized": False,
        "m_o_arm_authorized": False,
        "task_source": "M",
        "legacy_producer_git_commit": legacy_commit,
        "audit_git_commit": audit_commit,
        "audit_builder": _result_builder_custody(audit_commit),
        "inputs": inputs,
        "legacy_gate_recomputation": legacy_recomputation,
        "strict_replay": strict_replay,
        "claim_boundary": (
            "This artifact only proves that the sealed historical M teacher remains "
            "ineligible selection context under the current strict scorer. It cannot "
            "authorize an M teacher merge, M_M, M_O, rescue training, or any scientific arm."
        ),
    }


def _validate_m_negative_selection_context(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError("O-teacher preregistration lacks M-negative selection context")
    path_value = value.get("path")
    if not isinstance(path_value, str) or not Path(path_value).is_absolute():
        raise ValueError("M-negative selection-context path must be absolute")
    path, audit = _sealed_json_file(
        Path(path_value), value.get("sha256"), "M-negative selection-context audit"
    )
    expected_keys = {
        "schema_version",
        "audit",
        "created_utc",
        "selection_context_only",
        "scientific_authorization",
        "m_teacher_gate_passed",
        "merge_authorized",
        "m_arms_prohibited",
        "m_m_arm_authorized",
        "m_o_arm_authorized",
        "task_source",
        "legacy_producer_git_commit",
        "audit_git_commit",
        "audit_builder",
        "inputs",
        "legacy_gate_recomputation",
        "strict_replay",
        "claim_boundary",
    }
    if set(audit) != expected_keys:
        raise ValueError("M-negative selection-context audit has an invalid schema")
    _expect(audit, "schema_version", 1, "M-negative selection-context audit")
    _expect(
        audit,
        "audit",
        M_TEACHER_NEGATIVE_SELECTION_AUDIT,
        "M-negative selection-context audit",
    )
    for field, expected in (
        ("selection_context_only", True),
        ("scientific_authorization", False),
        ("m_teacher_gate_passed", False),
        ("merge_authorized", False),
        ("m_arms_prohibited", True),
        ("m_m_arm_authorized", False),
        ("m_o_arm_authorized", False),
        ("task_source", "M"),
    ):
        _expect(audit, field, expected, "M-negative selection-context audit")
    _parse_utc_timestamp(audit.get("created_utc"), "M-negative audit created_utc")
    legacy_commit = _hash_identity(
        audit.get("legacy_producer_git_commit"), "legacy M producer commit", HEX40
    )
    audit_commit = _hash_identity(
        audit.get("audit_git_commit"), "M-negative audit commit", HEX40
    )
    builder = audit.get("audit_builder")
    if not isinstance(builder, dict):
        raise ValueError("M-negative selection-context audit lacks builder custody")
    _clean_state(builder.get("git_state"), audit_commit, "M-negative audit builder")
    _clean_state(git_state(), audit_commit, "M-negative compatibility validator")
    _expect(
        builder,
        "builder_relative_path",
        Path(__file__).resolve().relative_to(ROOT).as_posix(),
        "M-negative audit builder",
    )
    _expect(
        builder,
        "builder_file_sha256",
        sha256_file(Path(__file__).resolve()),
        "M-negative audit builder",
    )
    legacy_recomputation, strict_replay = _replay_m_negative_compatibility(
        audit.get("inputs"), legacy_commit=legacy_commit
    )
    _expect(
        audit,
        "legacy_gate_recomputation",
        legacy_recomputation,
        "M-negative selection-context audit",
    )
    _expect(
        audit,
        "strict_replay",
        strict_replay,
        "M-negative selection-context audit",
    )
    inputs = audit["inputs"]
    teacher_gate_binding = inputs["teacher_gap_manifest"]
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "git_commit": legacy_commit,
        "audit_git_commit": audit_commit,
        "teacher_gap_manifest": dict(teacher_gate_binding),
        "legacy_gate_recomputed_exactly": True,
        "strict_negative_replay_confirmed": True,
        "verifier_error_count": strict_replay["verifier_error_count"],
        "m_arms_prohibited": True,
    }


def _load_o_teacher_preregistration(
    path: Path,
    *,
    launch_ledger_path: Path,
    gate_paths: Mapping[str, Path],
) -> tuple[dict[str, Any], dict[str, Any]]:
    ledger_raw = Path(launch_ledger_path).expanduser()
    if ledger_raw.is_symlink() or not ledger_raw.is_file():
        raise ValueError("O-teacher launch ledger must be a regular non-symlink file")
    if ledger_raw.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise ValueError("O-teacher launch ledger must be sealed read-only")
    ledger_path = ledger_raw.resolve()
    ledger_sha256 = sha256_file(ledger_path)
    ledger = _json_object(ledger_path, "O-teacher launch ledger")
    _expect(ledger, "schema_version", 1, "O-teacher launch ledger")
    _expect(ledger, "ledger", O_TEACHER_LAUNCH_LEDGER, "O-teacher launch ledger")
    for field, expected in (
        ("sealed_before_student_arm_launch", True),
        ("student_arm_outcomes_inspected_before_sealing", False),
        ("teacher_selection_condition_known_before_sealing", True),
        ("diagnostic_clean_before_preregistration", True),
    ):
        _expect(ledger, field, expected, "O-teacher launch ledger")
    ledger_campaign_id = ledger.get("campaign_id")
    if not isinstance(ledger_campaign_id, str) or not ledger_campaign_id.strip():
        raise ValueError("O-teacher launch ledger lacks campaign_id")
    ledger_created_utc = ledger.get("created_utc")
    ledger_created_at = _parse_utc_timestamp(
        ledger_created_utc, "O-teacher launch ledger created_utc"
    )
    preregistered = ledger.get("preregistration")
    if not isinstance(preregistered, dict):
        raise ValueError("O-teacher launch ledger lacks preregistration custody")
    expected_sha256 = _hash_identity(
        preregistered.get("sha256"), "O-teacher launch-ledger preregistration"
    )
    raw = Path(path).expanduser()
    if raw.is_symlink() or not raw.is_file():
        raise ValueError("O-teacher preregistration must be a regular non-symlink file")
    resolved = raw.resolve()
    _expect(
        preregistered,
        "path",
        str(resolved),
        "O-teacher launch ledger",
    )
    if raw.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise ValueError("O-teacher preregistration must be sealed read-only")
    _hash_identity(expected_sha256, "O-teacher preregistration")
    actual_sha256 = sha256_file(resolved)
    if actual_sha256 != expected_sha256:
        raise ValueError("O-teacher preregistration hash differs from the launch ledger")
    payload = _json_object(resolved, "O-teacher preregistration")
    _expect(payload, "schema_version", 1, "O-teacher preregistration")
    _expect(
        payload,
        "preregistration",
        O_TEACHER_PREREGISTRATION,
        "O-teacher preregistration",
    )
    for field, expected in (
        ("student_outcome_blind", True),
        ("sealed_before_student_arm_launch", True),
        ("student_arm_outcomes_inspected_before_sealing", False),
        ("teacher_selection_condition_known_before_sealing", True),
        ("diagnostic_clean_before_preregistration", True),
        ("operational_retry_requires_new_preregistration", True),
        ("arm_keys", list(O_TEACHER_CONTRACT)),
        ("claim_boundary", O_TEACHER_CLAIM_BOUNDARY),
    ):
        _expect(payload, field, expected, "O-teacher preregistration")
    campaign_id = payload.get("campaign_id")
    if not isinstance(campaign_id, str) or not campaign_id.strip():
        raise ValueError("O-teacher preregistration lacks a campaign_id")
    if campaign_id != ledger_campaign_id:
        raise ValueError("O-teacher launch ledger and preregistration campaign differ")
    created_utc = payload.get("created_utc")
    created_at = _parse_utc_timestamp(
        created_utc, "O-teacher preregistration created_utc"
    )
    if ledger_created_at < created_at:
        raise ValueError(
            "O-teacher launch ledger predates the preregistration it seals"
        )
    commit = _hash_identity(
        payload.get("git_commit"), "O-teacher preregistration commit", HEX40
    )
    prepared = payload.get("prepared_manifest")
    if not isinstance(prepared, dict):
        raise ValueError("O-teacher preregistration lacks prepared-manifest custody")
    prepared_path = prepared.get("path")
    if not isinstance(prepared_path, str) or not Path(prepared_path).is_absolute():
        raise ValueError("O-teacher preregistration prepared path must be absolute")
    _hash_identity(prepared.get("sha256"), "O-teacher preregistration prepared manifest")
    _hash_identity(
        payload.get("student_training_plan_sha256"),
        "O-teacher preregistration student plan",
    )
    diagnostic_identity = _validate_o_m_diagnostic_identity(
        payload.get("one_step_diagnostic"), commit=commit
    )
    selection_context = _validate_m_negative_selection_context(
        payload.get("selection_context")
    )
    teacher_identity = payload.get("o_teacher_stable_identity")
    if not isinstance(teacher_identity, dict) or set(teacher_identity) != set(
        O_TEACHER_STABLE_IDENTITY_FIELDS
    ):
        raise ValueError(
            "O-teacher preregistration lacks the exact stable teacher identity"
        )
    _expect(
        teacher_identity,
        "teacher_source",
        "O",
        "O-teacher preregistration stable teacher",
    )
    for field in ("base_model", "base_revision"):
        value = teacher_identity.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"O-teacher preregistration stable teacher lacks {field}"
            )
    _hash_identity(
        teacher_identity.get("base_revision"),
        "O-teacher preregistration teacher revision",
        HEX40,
    )
    for field in ("teacher_gap_manifest", "merged_checkpoint"):
        value = teacher_identity.get(field)
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ValueError(
                f"O-teacher preregistration stable teacher {field} must be absolute"
            )
    for field in O_TEACHER_STABLE_IDENTITY_FIELDS:
        if field.endswith("_sha256"):
            _hash_identity(
                teacher_identity.get(field),
                f"O-teacher preregistration stable teacher {field}",
            )
    support_identities = payload.get("student_support_identities")
    if not isinstance(support_identities, dict) or set(support_identities) != {
        "M",
        "O",
    }:
        raise ValueError(
            "O-teacher preregistration lacks exact M/O student-support identities"
        )
    for source in ("M", "O"):
        identity = support_identities[source]
        if not isinstance(identity, dict) or set(identity) != set(
            O_TEACHER_SUPPORT_IDENTITY_FIELDS
        ):
            raise ValueError(
                f"O-teacher preregistration {source} support identity is invalid"
            )
        _expect(
            identity,
            "source",
            source,
            f"O-teacher preregistration {source} support identity",
        )
        for field in ("manifest_sha256", "payload_sha256"):
            _hash_identity(
                identity.get(field),
                f"O-teacher preregistration {source} support {field}",
            )
    if diagnostic_identity["validated_student_support"] != support_identities["M"]:
        raise ValueError(
            "O_M one-step diagnostic used a different M student-support gate"
        )
    diagnostic_teacher = {
        field: diagnostic_identity["validated_teacher"].get(field)
        for field in O_TEACHER_STABLE_IDENTITY_FIELDS
    }
    if diagnostic_teacher != teacher_identity:
        raise ValueError("O_M one-step diagnostic used a different O teacher")
    arms = payload.get("arms")
    if not isinstance(arms, dict) or set(arms) != set(O_TEACHER_CONTRACT):
        raise ValueError("O-teacher preregistration lacks the exact four arm bindings")
    for key in O_TEACHER_CONTRACT:
        arm = arms[key]
        if not isinstance(arm, dict) or set(arm) != set(
            O_TEACHER_ARM_IDENTITY_FIELDS
        ):
            raise ValueError(f"O-teacher preregistration arm binding is invalid: {key}")
        for field in O_TEACHER_ARM_IDENTITY_FIELDS:
            value = arm.get(field)
            if not isinstance(value, str) or not Path(value).is_absolute():
                raise ValueError(
                    f"O-teacher preregistration {key} {field} must be absolute"
                )
        if Path(arm["heldout_gate"]).resolve() != Path(gate_paths[key]).resolve():
            raise ValueError(
                f"O-teacher gate {key} was not fixed by the sealed preregistration"
            )
    outputs = payload.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != {
        "json",
        "markdown",
        "manifest",
    }:
        raise ValueError("O-teacher preregistration lacks exact output bindings")
    for field in ("json", "markdown", "manifest"):
        value = outputs[field]
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ValueError(f"O-teacher preregistration output {field} must be absolute")
    expected_inference = {
        "bootstrap_unit": "paired_record_within_source",
        "draws": BOOTSTRAP_DRAWS,
        "seed": BOOTSTRAP_SEED,
        "resampling_order": "M_then_O_single_random.Random_stream",
        "record_order": "lexicographic_record_id",
        "co_primary_contrasts": ["delta_M", "delta_O"],
        "familywise_alpha": 0.05,
        "familywise_interval": "Bonferroni_percentile_97.5",
        "verifier_uncertainty": "binary_worst_case_bootstrap_envelope_v1",
    }
    _expect(payload, "inference", expected_inference, "O-teacher preregistration")
    return payload, {
        "path": str(resolved),
        "sha256": actual_sha256,
        "campaign_id": campaign_id,
        "created_utc": created_utc,
        "outputs": dict(outputs),
        "sealed_read_only": True,
        "sealed_before_student_arm_launch": True,
        "student_arm_outcomes_inspected_before_sealing": False,
        "teacher_selection_condition_known_before_sealing": True,
        "diagnostic_clean_before_preregistration": True,
        "one_step_diagnostic": diagnostic_identity,
        "selection_context": selection_context,
        "launch_ledger": {
            "path": str(ledger_path),
            "sha256": ledger_sha256,
            "created_utc": ledger_created_utc,
            "sealed_read_only": True,
        },
    }


def o_teacher_prelaunch_receipt(args: Namespace) -> dict[str, Any]:
    """Validate the sealed four-arm boundary before one optimizer starts."""

    prereg_raw = _json_object(Path(args.preregistration), "O-teacher preregistration")
    prereg_arms = prereg_raw.get("arms")
    if not isinstance(prereg_arms, dict) or set(prereg_arms) != set(O_TEACHER_CONTRACT):
        raise ValueError("O-teacher preregistration lacks the exact four arm bindings")
    gate_paths = {
        key: Path(str(prereg_arms[key].get("heldout_gate")))
        for key in O_TEACHER_CONTRACT
    }
    preregistration, prereg_binding = _load_o_teacher_preregistration(
        Path(args.preregistration),
        launch_ledger_path=Path(args.launch_ledger),
        gate_paths=gate_paths,
    )
    run_key = args.run_key
    if run_key not in O_TEACHER_CONTRACT:
        raise ValueError("prelaunch run key is not one of the four allowed arms")
    contract = O_TEACHER_CONTRACT[run_key]
    _expect(
        {"mode": args.mode},
        "mode",
        contract["objective"],
        "O-teacher prelaunch",
    )
    _expect(
        {"source": args.student_source},
        "source",
        contract["student_source"],
        "O-teacher prelaunch",
    )
    if not isinstance(args.run_id, str) or re.fullmatch(
        r"[A-Za-z0-9._-]+", args.run_id
    ) is None:
        raise ValueError("prelaunch run ID is not filesystem-safe")
    if not isinstance(args.scheduler_job_id, str) or re.fullmatch(
        r"[1-9][0-9]*", args.scheduler_job_id
    ) is None:
        raise ValueError("prelaunch scheduler job ID is invalid")

    state = git_state()
    commit = preregistration["git_commit"]
    _clean_state(state, commit, "O-teacher prelaunch Git state")
    prepared = preregistration["prepared_manifest"]
    _file_binding(
        Path(prepared["path"]),
        prepared["sha256"],
        "O-teacher prelaunch prepared manifest",
    )
    _file_binding(
        CANONICAL_STUDENT_TRAINING_PLAN,
        preregistration["student_training_plan_sha256"],
        "O-teacher prelaunch student plan",
    )

    arm = preregistration["arms"][run_key]
    out_dir = Path(args.out_dir).resolve()
    expected_paths = {
        "run_manifest": str((out_dir / "traces" / "run_manifest.json").resolve()),
        "student_completion_manifest": str(
            (out_dir / "traces" / "completion_manifest.json").resolve()
        ),
        "student_adapter": str((out_dir / "final").resolve()),
        "prelaunch_receipt": str(Path(args.output).resolve()),
    }
    for field, expected in expected_paths.items():
        _expect(arm, field, expected, f"O-teacher preregistration arm {run_key}")

    support_path = Path(args.student_support_manifest).resolve()
    support_raw = support_path.expanduser()
    if support_raw.is_symlink() or not support_raw.is_file():
        raise ValueError("prelaunch student-support gate must be a regular file")
    if support_raw.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise ValueError("prelaunch student-support gate must be sealed read-only")
    support = _json_object(support_raw, "prelaunch student-support gate")
    _expect(support, "gate", STUDENT_GATE_TYPE, "prelaunch student-support gate")
    _expect(support, "passed", True, "prelaunch student-support gate")
    _expect(
        support,
        "authorizes_scientific_training",
        True,
        "prelaunch student-support gate",
    )
    _expect(
        support,
        "task_sources",
        [args.student_source],
        "prelaunch student-support gate",
    )
    recomputed_support = recompute_student_gate(support)
    if recomputed_support != support:
        raise ValueError("prelaunch student-support gate failed exact recomputation")
    support_identity = {
        "manifest_sha256": sha256_file(support_raw),
        "payload_sha256": canonical_json_sha256(support),
        "source": args.student_source,
    }
    if support_identity != preregistration["student_support_identities"][
        args.student_source
    ]:
        raise ValueError("prelaunch student-support identity differs from preregistration")

    teacher_identity: dict[str, Any] | None = None
    teacher_paths = (
        args.teacher_gap_manifest,
        args.teacher_checkpoint,
        args.teacher_provenance_manifest,
    )
    if contract["teacher_source"] is None:
        if any(value is not None for value in teacher_paths):
            raise ValueError("task-RL baseline prelaunch unexpectedly supplied a teacher")
    else:
        if any(value is None for value in teacher_paths):
            raise ValueError("O-teacher prelaunch requires gate/checkpoint/provenance paths")
        expected_teacher = preregistration["o_teacher_stable_identity"]
        teacher_gate_path = Path(args.teacher_gap_manifest).resolve()
        checkpoint = Path(args.teacher_checkpoint).resolve()
        provenance_path = Path(args.teacher_provenance_manifest).resolve()
        _expect(
            expected_teacher,
            "teacher_gap_manifest",
            str(teacher_gate_path),
            "O-teacher prelaunch",
        )
        _expect(
            expected_teacher,
            "merged_checkpoint",
            str(checkpoint),
            "O-teacher prelaunch",
        )
        if provenance_path != (checkpoint / "merge_provenance.json").resolve():
            raise ValueError("prelaunch teacher provenance is not canonical in checkpoint")
        gate = _json_object(teacher_gate_path, "prelaunch O teacher-gap manifest")
        _file_binding(
            teacher_gate_path,
            expected_teacher["teacher_gap_manifest_sha256"],
            "prelaunch O teacher-gap manifest",
        )
        _expect(
            expected_teacher,
            "teacher_gap_payload_sha256",
            canonical_json_sha256(gate),
            "O-teacher prelaunch",
        )
        if recompute_teacher_gate(gate) != gate:
            raise ValueError("prelaunch O teacher gap failed exact recomputation")
        checkpoint_hash = sha256_tree(
            checkpoint, exclude_relative_paths=("merge_provenance.json",)
        )
        _expect(
            expected_teacher,
            "merged_checkpoint_tree_sha256",
            checkpoint_hash,
            "O-teacher prelaunch",
        )
        provenance = _json_object(provenance_path, "prelaunch teacher provenance")
        _expect(
            expected_teacher,
            "merge_provenance_manifest_sha256",
            sha256_file(provenance_path),
            "O-teacher prelaunch",
        )
        _expect(
            expected_teacher,
            "merge_provenance_payload_sha256",
            canonical_json_sha256(provenance),
            "O-teacher prelaunch",
        )
        teacher_identity = dict(expected_teacher)

    receipt_created_at = datetime.now(timezone.utc)
    if receipt_created_at < _parse_utc_timestamp(
        prereg_binding["created_utc"], "O-teacher preregistration created_utc"
    ) or receipt_created_at < _parse_utc_timestamp(
        prereg_binding["launch_ledger"]["created_utc"],
        "O-teacher launch ledger created_utc",
    ):
        raise ValueError("student prelaunch receipt would predate its sealed custody")
    receipt = {
        "schema_version": 1,
        "receipt": O_TEACHER_PRELAUNCH_RECEIPT,
        "created_utc": receipt_created_at.isoformat().replace("+00:00", "Z"),
        "sealed_before_optimizer_start": True,
        "campaign_id": preregistration["campaign_id"],
        "run_key": run_key,
        "run_id": args.run_id,
        "scheduler_job_id": args.scheduler_job_id,
        "mode": args.mode,
        "student_source": args.student_source,
        "git_commit": commit,
        "out_dir": str(out_dir),
        "expected_artifacts": expected_paths,
        "preregistration": {
            "path": prereg_binding["path"],
            "sha256": prereg_binding["sha256"],
        },
        "launch_ledger": {
            "path": prereg_binding["launch_ledger"]["path"],
            "sha256": prereg_binding["launch_ledger"]["sha256"],
        },
        "student_support": support_identity,
        "o_teacher": teacher_identity,
        "claim_boundary": (
            "This receipt proves that the sealed path and artifact identities were "
            "validated before this wrapper invoked the optimizer. It is local operator "
            "custody, not remote cryptographic attestation or a performance result."
        ),
    }
    output = Path(args.output).resolve()
    protected = [
        Path(prepared["path"]).resolve().parent,
        support_path.parent,
        Path(prereg_binding["path"]).resolve().parent,
    ]
    if teacher_identity is not None:
        protected.extend(
            [
                Path(teacher_identity["teacher_gap_manifest"]).resolve().parent,
                Path(teacher_identity["merged_checkpoint"]).resolve(),
            ]
        )
    _preflight_result_outputs([output], protected_trees=protected)
    try:
        _write_new(output, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        written = _json_object(output, "student prelaunch receipt")
        if written != receipt:
            raise RuntimeError("student prelaunch receipt changed during publication")
        output.chmod(0o444)
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return receipt


def _o_teacher_primary_contrast(
    *,
    estimate: float,
    estimate_bounds: list[float],
    point_draws: list[float],
    lower_draws: list[float],
    upper_draws: list[float],
    formula: str,
) -> dict[str, Any]:
    ordinary = _percentile_interval_at(point_draws, 0.025, 0.975)
    familywise = _percentile_interval_at(point_draws, 0.0125, 0.9875)
    pessimistic = _percentile_interval_at(lower_draws, 0.0125, 0.9875)
    optimistic = _percentile_interval_at(upper_draws, 0.0125, 0.9875)
    robust = [pessimistic[0], optimistic[1]]
    return {
        "formula": formula,
        "estimate": estimate,
        "bootstrap_95_ci": ordinary,
        "bootstrap_97_5_fwer_ci": familywise,
        "classification_without_verifier_uncertainty": _effect_label(familywise),
        "classification": _effect_label(robust),
        "verifier_uncertainty_sensitivity": {
            "policy": "binary_worst_case_bootstrap_envelope_v1",
            "estimate_bounds": estimate_bounds,
            "bootstrap_97_5_fwer_envelope": robust,
            "pessimistic_bootstrap_97_5_fwer_ci": pessimistic,
            "optimistic_bootstrap_97_5_fwer_ci": optimistic,
        },
    }


def _o_teacher_secondary_contrast(
    *,
    estimate: float,
    estimate_bounds: list[float],
    point_draws: list[float],
    lower_draws: list[float],
    upper_draws: list[float],
    formula: str,
    not_a_same_source_effect: bool = False,
) -> dict[str, Any]:
    pessimistic = _percentile_interval_at(lower_draws, 0.025, 0.975)
    optimistic = _percentile_interval_at(upper_draws, 0.025, 0.975)
    result = {
        "formula": formula,
        "estimate": estimate,
        "bootstrap_95_ci": _percentile_interval_at(point_draws, 0.025, 0.975),
        "verifier_uncertainty_sensitivity": {
            "policy": "binary_worst_case_bootstrap_envelope_v1",
            "estimate_bounds": estimate_bounds,
            "bootstrap_95_envelope": [pessimistic[0], optimistic[1]],
            "pessimistic_bootstrap_95_ci": pessimistic,
            "optimistic_bootstrap_95_ci": optimistic,
        },
        "confirmatory": False,
    }
    if not_a_same_source_effect:
        result["not_a_same_source_effect"] = True
    return result


def o_teacher_readout(
    gate_paths: Mapping[str, Path],
    *,
    preregistration_path: Path,
    launch_ledger_path: Path,
    seed: int = BOOTSTRAP_SEED,
    draws: int = BOOTSTRAP_DRAWS,
) -> dict[str, Any]:
    """Build the preregistered four-arm readout conditional on the O teacher.

    This deliberately excludes M_M and M_O. The M teacher failed its scientific
    skill-gap gate, so requiring those arms would convert a negative gate into a
    hidden rescue. Each input gate is exactly recomputed before any outcome is
    inspected.
    """

    if set(gate_paths) != set(O_TEACHER_CONTRACT):
        raise ValueError(
            "O-teacher readout inputs must be exactly baseline_M, O_M, "
            "baseline_O, O_O"
        )
    if seed != BOOTSTRAP_SEED or draws != BOOTSTRAP_DRAWS:
        raise ValueError(
            f"O-teacher readout requires seed={BOOTSTRAP_SEED} and "
            f"{BOOTSTRAP_DRAWS} draws"
        )
    preregistration, preregistration_binding = _load_o_teacher_preregistration(
        preregistration_path,
        launch_ledger_path=launch_ledger_path,
        gate_paths=gate_paths,
    )

    gates: dict[str, dict[str, Any]] = {}
    inputs: dict[str, dict[str, Any]] = {}
    for key, contract in O_TEACHER_CONTRACT.items():
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
        for field in ("objective", "student_source", "teacher_source"):
            _expect(gate, field, contract[field], f"held-out gate {key}")
        gates[key] = gate
        inputs[key] = {"path": str(path), "sha256": sha256_file(path)}

    reference = gates["baseline_M"]
    _expect(
        reference["student_run_binding"],
        "git_commit",
        preregistration["git_commit"],
        "O-teacher preregistration",
    )
    _expect(
        reference["prepared_binding"],
        "prepared_manifest_sha256",
        preregistration["prepared_manifest"]["sha256"],
        "O-teacher preregistration",
    )
    _expect(
        reference["student_run_binding"],
        "student_training_plan_sha256",
        preregistration["student_training_plan_sha256"],
        "O-teacher preregistration",
    )
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
        for field in (
            "student_training_plan_sha256",
            "student_training_config_sha256",
            "git_commit",
        ):
            _expect(
                gate["student_run_binding"],
                field,
                reference["student_run_binding"][field],
                f"held-out gate {key}",
            )
        _expect(gate, "result_builder", result_builder, f"held-out gate {key}")
        preregistered_arm = preregistration["arms"][key]
        _expect(
            gate["student_run_binding"],
            "run_manifest",
            str(Path(preregistered_arm["run_manifest"]).resolve()),
            f"O-teacher preregistration arm {key}",
        )
        _expect(
            gate["student_run_binding"],
            "student_adapter",
            str(Path(preregistered_arm["student_adapter"]).resolve()),
            f"O-teacher preregistration arm {key}",
        )
        prelaunch = gate["student_run_binding"].get("prelaunch_receipt")
        if not isinstance(prelaunch, dict):
            raise ValueError(f"held-out gate {key} lacks prelaunch receipt custody")
        _expect(
            prelaunch,
            "path",
            str(Path(preregistered_arm["prelaunch_receipt"]).resolve()),
            f"O-teacher preregistration arm {key}",
        )
        _expect(
            prelaunch,
            "preregistration",
            {
                "path": preregistration_binding["path"],
                "sha256": preregistration_binding["sha256"],
            },
            f"O-teacher preregistration arm {key}",
        )
        _expect(
            prelaunch,
            "launch_ledger",
            {
                "path": preregistration_binding["launch_ledger"]["path"],
                "sha256": preregistration_binding["launch_ledger"]["sha256"],
            },
            f"O-teacher preregistration arm {key}",
        )
        deterministic_inputs = gate.get("inputs")
        if not isinstance(deterministic_inputs, dict):
            raise ValueError(f"held-out gate {key} lacks deterministic inputs")
        for prereg_field, input_field in (
            ("run_manifest", "student_run_manifest"),
            ("student_completion_manifest", "student_completion_manifest"),
            ("student_adapter", "trained_adapter"),
            ("student_eval_summary", "student_summary"),
            ("student_eval_samples", "student_samples"),
        ):
            _expect(
                deterministic_inputs,
                input_field,
                str(Path(preregistered_arm[prereg_field]).resolve()),
                f"O-teacher preregistration arm {key}",
            )
        evaluation_custody = gate["evaluation_binding"].get(
            "evaluation_post_promotion_custody"
        )
        if not isinstance(evaluation_custody, dict):
            raise ValueError(
                f"held-out gate {key} lacks post-promotion evaluation custody"
            )
        _expect(
            evaluation_custody,
            "path",
            str(Path(preregistered_arm["student_eval_custody"]).resolve()),
            f"O-teacher preregistration arm {key}",
        )
        _expect(
            deterministic_inputs,
            "prepared_manifest",
            str(Path(preregistration["prepared_manifest"]["path"]).resolve()),
            f"O-teacher preregistration arm {key}",
        )
        for field in (
            "evaluation_git_commit",
            "evaluator_file_sha256",
            "evaluation_packages",
            "tokenizer_contract_sha256",
            "evaluation_contract",
            "evaluation_environment",
        ):
            _expect(
                gate["evaluation_binding"],
                field,
                reference["evaluation_binding"][field],
                f"held-out gate {key} evaluation",
            )
        if not isinstance(
            gate["evaluation_binding"].get("evaluation_post_promotion_custody"),
            dict,
        ):
            raise ValueError(
                f"held-out gate {key} lacks post-promotion evaluation custody"
            )

    reference_environment = reference["student_run_binding"].get("environment")
    if not isinstance(reference_environment, dict):
        raise ValueError("baseline_M lacks validated environment custody")
    common_train_environment: dict[str, Any] = {}
    for field in ("verifier", "train_freeze", "train_verification"):
        value = reference_environment.get(field)
        if not isinstance(value, dict):
            raise ValueError(f"baseline_M lacks validated {field} custody")
        common_train_environment[field] = value
    for key, gate in gates.items():
        environment = gate["student_run_binding"].get("environment")
        if not isinstance(environment, dict):
            raise ValueError(f"held-out gate {key} lacks validated environment custody")
        for field, expected in common_train_environment.items():
            _expect(environment, field, expected, f"held-out gate {key} environment")

    for key in ("baseline_M", "baseline_O"):
        environment = gates[key]["student_run_binding"]["environment"]
        _expect(environment, "serve_freeze", None, f"held-out gate {key} baseline environment")
        _expect(
            environment,
            "serve_verification",
            None,
            f"held-out gate {key} baseline environment",
        )
        _expect(
            gates[key]["student_run_binding"],
            "teacher",
            None,
            f"held-out gate {key} baseline",
        )

    o_m_environment = gates["O_M"]["student_run_binding"]["environment"]
    o_serve_freeze = o_m_environment.get("serve_freeze")
    o_serve_verification = o_m_environment.get("serve_verification")
    if not isinstance(o_serve_freeze, dict):
        raise ValueError("O_M lacks a validated teacher serve environment freeze")
    if not isinstance(o_serve_verification, dict):
        raise ValueError("O_M lacks exact live teacher serve environment verification")
    for key in ("O_M", "O_O"):
        environment = gates[key]["student_run_binding"]["environment"]
        _expect(environment, "serve_freeze", o_serve_freeze, f"held-out gate {key} environment")
        _expect(
            environment,
            "serve_verification",
            o_serve_verification,
            f"held-out gate {key} environment",
        )

    pair_custody: dict[str, dict[str, Any]] = {}
    for source, (baseline_key, opd_key) in O_TEACHER_PAIRS.items():
        baseline = gates[baseline_key]
        opd = gates[opd_key]
        baseline_ids = sorted(baseline["record_rewards"])
        if sorted(opd["record_rewards"]) != baseline_ids:
            raise ValueError(f"{source} held-out arms do not share the exact record set")
        _expect(opd, "record_ids_sha256", baseline["record_ids_sha256"], f"{source} held-out pair")
        _expect(
            opd["evaluation_binding"],
            "task_file_sha256",
            baseline["evaluation_binding"]["task_file_sha256"],
            f"{source} held-out pair",
        )
        if opd["student_run_binding"]["student_support"] != baseline["student_run_binding"][
            "student_support"
        ]:
            raise ValueError(f"{source} arms do not share the exact student-support identity")
        if baseline["student_run_binding"]["student_support"] != preregistration[
            "student_support_identities"
        ][source]:
            raise ValueError(
                f"{source} student-support identity differs from the sealed preregistration"
            )
        expected_sequence = {
            field: baseline["student_run_binding"]["trace"].get(field)
            for field in (
                "realized_record_ids_sha256",
                "realized_prompt_sequence_sha256",
            )
        }
        if any(value is None for value in expected_sequence.values()):
            raise ValueError(f"{source} baseline lacks realized training-sequence custody")
        actual_sequence = {
            field: opd["student_run_binding"]["trace"].get(field)
            for field in expected_sequence
        }
        if actual_sequence != expected_sequence:
            raise ValueError(
                f"{source} arms do not share the exact realized training sequence"
            )
        pair_custody[source] = {
            "record_ids_sha256": baseline["record_ids_sha256"],
            "task_file_sha256": baseline["evaluation_binding"]["task_file_sha256"],
            "student_support": baseline["student_run_binding"]["student_support"],
            "realized_training_sequence": expected_sequence,
        }

    process_teacher_fields = (
        "tokenizer_contract_manifest_sha256",
        "tokenizer_contract_payload_sha256",
        "server_scoring_manifest_sha256",
        "server_scoring_payload_sha256",
    )
    stable_teacher_identities: list[dict[str, Any]] = []
    process_teacher_custody: dict[str, dict[str, Any]] = {}
    for key in ("O_M", "O_O"):
        teacher = gates[key]["student_run_binding"].get("teacher")
        if not isinstance(teacher, dict):
            raise ValueError(f"held-out gate {key} lacks validated O-teacher identity")
        _expect(teacher, "teacher_source", "O", f"held-out gate {key} teacher")
        stable = {
            field: teacher.get(field)
            for field in O_TEACHER_STABLE_IDENTITY_FIELDS
        }
        process = {field: teacher.get(field) for field in process_teacher_fields}
        if any(value is None for value in stable.values()):
            raise ValueError(f"held-out gate {key} lacks complete stable teacher custody")
        if any(value is None for value in process.values()):
            raise ValueError(f"held-out gate {key} lacks complete teacher process custody")
        stable_teacher_identities.append(stable)
        process_teacher_custody[key] = process
    if stable_teacher_identities[0] != stable_teacher_identities[1]:
        raise ValueError("O-teacher arms do not share one exact teacher identity")
    preregistered_teacher_identity = preregistration[
        "o_teacher_stable_identity"
    ]
    if stable_teacher_identities[0] != preregistered_teacher_identity:
        raise ValueError(
            "O-teacher stable identity differs from the sealed preregistration"
        )
    preregistered_teacher_gate = Path(
        preregistered_teacher_identity["teacher_gap_manifest"]
    ).resolve()
    _file_binding(
        preregistered_teacher_gate,
        preregistered_teacher_identity["teacher_gap_manifest_sha256"],
        "preregistered O teacher-gap manifest",
    )

    run_paths = [gate["student_run_binding"]["run_manifest"] for gate in gates.values()]
    adapter_paths = [gate["student_run_binding"]["student_adapter"] for gate in gates.values()]
    if len(set(run_paths)) != len(run_paths) or len(set(adapter_paths)) != len(adapter_paths):
        raise ValueError(
            "O-teacher readout arms accidentally reuse a run manifest or student adapter path"
        )

    per_record = {
        key: {
            record_id: sum(values) / len(values)
            for record_id, values in gate["record_rewards"].items()
        }
        for key, gate in gates.items()
    }
    per_record_bounds = {
        key: {
            record_id: list(bounds)
            for record_id, bounds in gate[
                "record_accuracy_bounds_under_verifier_uncertainty"
            ].items()
        }
        for key, gate in gates.items()
    }
    for key in gates:
        if set(per_record_bounds[key]) != set(per_record[key]) or any(
            len(bounds) != 2
            or bounds[0] != per_record[key][record_id]
            or not 0.0 <= bounds[0] <= bounds[1] <= 1.0
            for record_id, bounds in per_record_bounds[key].items()
        ):
            raise ValueError(f"held-out gate {key} has invalid verifier uncertainty bounds")

    arm_accuracy = {
        key: sum(values.values()) / len(values) for key, values in per_record.items()
    }
    arm_accuracy_bounds = {
        key: [
            sum(bounds[0] for bounds in values.values()) / len(values),
            sum(bounds[1] for bounds in values.values()) / len(values),
        ]
        for key, values in per_record_bounds.items()
    }
    observed = {
        source: arm_accuracy[opd_key] - arm_accuracy[baseline_key]
        for source, (baseline_key, opd_key) in O_TEACHER_PAIRS.items()
    }
    observed_bounds = {
        source: [
            arm_accuracy_bounds[opd_key][0] - arm_accuracy_bounds[baseline_key][1],
            arm_accuracy_bounds[opd_key][1] - arm_accuracy_bounds[baseline_key][0],
        ]
        for source, (baseline_key, opd_key) in O_TEACHER_PAIRS.items()
    }

    record_ids = {
        source: sorted(per_record[baseline_key])
        for source, (baseline_key, _) in O_TEACHER_PAIRS.items()
    }
    vectors = {
        key: [
            per_record[key][record_id]
            for record_id in record_ids[str(gates[key]["student_source"])]
        ]
        for key in gates
    }
    lower_vectors = {
        key: [
            per_record_bounds[key][record_id][0]
            for record_id in record_ids[str(gates[key]["student_source"])]
        ]
        for key in gates
    }
    upper_vectors = {
        key: [
            per_record_bounds[key][record_id][1]
            for record_id in record_ids[str(gates[key]["student_source"])]
        ]
        for key in gates
    }

    bootstrap: dict[str, list[float]] = defaultdict(list)
    rng = random.Random(seed)
    for _ in range(draws):
        indices = {
            "M": [rng.randrange(len(record_ids["M"])) for _ in record_ids["M"]],
            "O": [rng.randrange(len(record_ids["O"])) for _ in record_ids["O"]],
        }
        point_delta: dict[str, float] = {}
        lower_delta: dict[str, float] = {}
        upper_delta: dict[str, float] = {}
        for source, (baseline_key, opd_key) in O_TEACHER_PAIRS.items():
            idx = indices[source]
            point_delta[source] = _mean_for_indices(
                vectors[opd_key], idx
            ) - _mean_for_indices(vectors[baseline_key], idx)
            lower_delta[source] = _mean_for_indices(
                lower_vectors[opd_key], idx
            ) - _mean_for_indices(upper_vectors[baseline_key], idx)
            upper_delta[source] = _mean_for_indices(
                upper_vectors[opd_key], idx
            ) - _mean_for_indices(lower_vectors[baseline_key], idx)
            bootstrap[f"delta:{source}"].append(point_delta[source])
            bootstrap[f"lower:delta:{source}"].append(lower_delta[source])
            bootstrap[f"upper:delta:{source}"].append(upper_delta[source])
        bootstrap["equal_weight"].append(
            (point_delta["M"] + point_delta["O"]) / 2
        )
        bootstrap["lower:equal_weight"].append(
            (lower_delta["M"] + lower_delta["O"]) / 2
        )
        bootstrap["upper:equal_weight"].append(
            (upper_delta["M"] + upper_delta["O"]) / 2
        )
        bootstrap["heterogeneity"].append(point_delta["O"] - point_delta["M"])
        bootstrap["lower:heterogeneity"].append(
            lower_delta["O"] - upper_delta["M"]
        )
        bootstrap["upper:heterogeneity"].append(
            upper_delta["O"] - lower_delta["M"]
        )

    primary_results = {
        f"delta_{source}": _o_teacher_primary_contrast(
            estimate=observed[source],
            estimate_bounds=observed_bounds[source],
            point_draws=bootstrap[f"delta:{source}"],
            lower_draws=bootstrap[f"lower:delta:{source}"],
            upper_draws=bootstrap[f"upper:delta:{source}"],
            formula=f"accuracy({opd_key}) - accuracy({baseline_key})",
        )
        for source, (baseline_key, opd_key) in O_TEACHER_PAIRS.items()
    }
    equal_weight_bounds = [
        (observed_bounds["M"][0] + observed_bounds["O"][0]) / 2,
        (observed_bounds["M"][1] + observed_bounds["O"][1]) / 2,
    ]
    heterogeneity_bounds = [
        observed_bounds["O"][0] - observed_bounds["M"][1],
        observed_bounds["O"][1] - observed_bounds["M"][0],
    ]
    secondary = {
        "equal_weight_source_average": _o_teacher_secondary_contrast(
            estimate=(observed["M"] + observed["O"]) / 2,
            estimate_bounds=equal_weight_bounds,
            point_draws=bootstrap["equal_weight"],
            lower_draws=bootstrap["lower:equal_weight"],
            upper_draws=bootstrap["upper:equal_weight"],
            formula="0.5 * (delta_M + delta_O)",
        ),
        "source_heterogeneity": _o_teacher_secondary_contrast(
            estimate=observed["O"] - observed["M"],
            estimate_bounds=heterogeneity_bounds,
            point_draws=bootstrap["heterogeneity"],
            lower_draws=bootstrap["lower:heterogeneity"],
            upper_draws=bootstrap["upper:heterogeneity"],
            formula="delta_O - delta_M",
            not_a_same_source_effect=True,
        ),
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "readout": O_TEACHER_READOUT,
        "scientific_readout_authorized": True,
        "authorization_is_independent_of_effect_sign": True,
        "conditional_on_passing_o_teacher_gate": True,
        "not_a_six_arm_matrix": True,
        "preregistration": preregistration_binding,
        "arm_keys": list(O_TEACHER_CONTRACT),
        "student_model": reference["student_model"],
        "student_model_revision": reference["student_model_revision"],
        "arm_accuracy": arm_accuracy,
        "arm_accuracy_bounds_under_verifier_uncertainty": arm_accuracy_bounds,
        "arm_records": {key: gate["records"] for key, gate in gates.items()},
        "primary_results": primary_results,
        "secondary_cross_source": secondary,
        "custody": {
            "within_source": pair_custody,
            "o_teacher_stable_artifact_identity": stable_teacher_identities[0],
            "o_teacher_per_arm_process_custody": process_teacher_custody,
            "selection_condition_recomputed_in_each_o_arm": True,
        },
        "environment_freezes": {
            "train": common_train_environment["train_freeze"],
            "serve": o_serve_freeze,
        },
        "evaluation_contract": EVALUATION_CONTRACT,
        "evaluation_environment": reference["evaluation_binding"][
            "evaluation_environment"
        ],
        "evaluation_contract_sha256_by_arm": {
            key: gate["evaluation_binding"]["evaluation_contract_sha256"]
            for key, gate in gates.items()
        },
        "evaluation_post_promotion_custody_by_arm": {
            key: gate["evaluation_binding"]["evaluation_post_promotion_custody"]
            for key, gate in gates.items()
        },
        "result_builder": result_builder,
        "bootstrap": {
            "unit": "paired record within held-out source",
            "draws": draws,
            "seed": seed,
            "record_order": "lexicographic_record_id",
            "ordinary_interval": "percentile_95",
            "familywise_interval": "Bonferroni_percentile_97.5_over_two_co_primary_contrasts",
            "familywise_alpha": 0.05,
            "co_primary_contrasts": ["delta_M", "delta_O"],
            "verifier_uncertainty": "binary_worst_case_bootstrap_envelope_v1",
            "classification_rule": {
                "helps": "robust_Bonferroni_envelope_lower_bound > 0",
                "harms": "robust_Bonferroni_envelope_upper_bound < 0",
                "inconclusive": "robust_Bonferroni_envelope includes 0",
            },
        },
        "inputs": inputs,
        "claim_boundary": O_TEACHER_CLAIM_BOUNDARY,
    }


def recompute_o_teacher_readout(payload: Mapping[str, Any]) -> dict[str, Any]:
    inputs = payload.get("inputs")
    bootstrap = payload.get("bootstrap")
    preregistration = payload.get("preregistration")
    if not isinstance(inputs, dict) or set(inputs) != set(O_TEACHER_CONTRACT):
        raise ValueError("O-teacher readout lacks the exact four deterministic inputs")
    if not isinstance(bootstrap, dict):
        raise ValueError("O-teacher readout lacks its bootstrap contract")
    if not isinstance(preregistration, dict):
        raise ValueError("O-teacher readout lacks preregistration custody")
    launch_ledger = preregistration.get("launch_ledger")
    if not isinstance(launch_ledger, dict):
        raise ValueError("O-teacher readout lacks launch-ledger custody")
    paths: dict[str, Path] = {}
    for key, binding in inputs.items():
        if not isinstance(binding, dict):
            raise ValueError(f"O-teacher input binding is invalid: {key}")
        path = Path(str(binding.get("path")))
        _file_binding(path, binding.get("sha256"), f"O-teacher gate {key}")
        paths[key] = path
    return o_teacher_readout(
        paths,
        preregistration_path=Path(str(preregistration.get("path"))),
        launch_ledger_path=Path(str(launch_ledger.get("path"))),
        seed=int(bootstrap.get("seed")),
        draws=int(bootstrap.get("draws")),
    )


def o_teacher_markdown(payload: Mapping[str, Any]) -> str:
    if payload.get("readout") != O_TEACHER_READOUT:
        raise ValueError("cannot render an unknown O-teacher readout")
    lines = [
        "# OPD math O-teacher conditional held-out readout",
        "",
        "This readout is conditional on the preregistered O teacher passing its scientific skill-gap gate.",
        "The M teacher failed; `M_M` and `M_O` were prohibited and are not hidden missing arms.",
        "",
        "| Source | Baseline | O-teacher OPD | Delta | Robust Bonferroni 97.5% envelope | Readout |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for source, (baseline_key, opd_key) in O_TEACHER_PAIRS.items():
        contrast = payload["primary_results"][f"delta_{source}"]
        low, high = contrast["verifier_uncertainty_sensitivity"][
            "bootstrap_97_5_fwer_envelope"
        ]
        lines.append(
            f"| {source} | {float(payload['arm_accuracy'][baseline_key]):.6f} | "
            f"{float(payload['arm_accuracy'][opd_key]):.6f} | "
            f"{float(contrast['estimate']):+.6f} | [{low:+.6f}, {high:+.6f}] | "
            f"{contrast['classification']} |"
        )
    lines.extend(
        [
            "",
            "The displayed primary intervals use paired record bootstrap, Bonferroni "
            "multiplicity control over the two source contrasts, and the worst-case "
            "binary envelope for every bounded verifier-uncertain sample.",
            "",
            "## Secondary cross-source diagnostics",
            "",
            "| Diagnostic | Estimate | Robust 95% verifier envelope |",
            "|---|---:|---:|",
        ]
    )
    for label, key in (
        ("Equal-weight average across M and O", "equal_weight_source_average"),
        ("O-minus-M heterogeneity (not a same-source effect)", "source_heterogeneity"),
    ):
        diagnostic = payload["secondary_cross_source"][key]
        low, high = diagnostic["verifier_uncertainty_sensitivity"][
            "bootstrap_95_envelope"
        ]
        lines.append(
            f"| {label} | {float(diagnostic['estimate']):+.6f} | "
            f"[{low:+.6f}, {high:+.6f}] |"
        )
    lines.extend(
        [
            "",
            str(payload["claim_boundary"]),
            "",
        ]
    )
    return "\n".join(lines)


def _write_new(path: Path, content: str) -> None:
    write_text_exclusive_fsync(path, content, label="result artifact")


def _write_readout_bundle(
    *,
    output_json: Path,
    json_content: str,
    output_markdown: Path,
    markdown_content: str,
    output_manifest: Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish and seal one checksummed readout bundle or leave no output."""

    created: list[Path] = []
    try:
        _write_new(output_json, json_content)
        created.append(output_json.resolve())
        _write_new(output_markdown, markdown_content)
        created.append(output_markdown.resolve())
        manifest = {
            "schema_version": 1,
            "manifest": "opd_math_readout_output_bundle_v1",
            "readout": payload.get("readout"),
            "result_json": {
                "path": str(output_json.resolve()),
                "sha256": sha256_file(output_json),
            },
            "result_markdown": {
                "path": str(output_markdown.resolve()),
                "sha256": sha256_file(output_markdown),
            },
            "preregistration": payload.get("preregistration"),
        }
        _write_new(
            output_manifest,
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        )
        created.append(output_manifest.resolve())
        for path in created:
            path.chmod(0o444)
        return manifest
    except BaseException:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise


_TREE_PATH_KEYS = {
    "adapter",
    "student_adapter",
    "trained_adapter",
    "final_adapter",
    "merged_checkpoint",
    "teacher_checkpoint",
    "environment_root",
    "train_environment_root",
    "serve_environment_root",
}
_FILE_PATH_KEYS = {
    "path",
    "task_file",
    "student_run_manifest",
    "run_manifest",
    "student_completion_manifest",
    "completion_manifest",
    "student_summary",
    "summary",
    "student_samples",
    "samples",
    "base_summary",
    "base_samples",
    "trained_summary",
    "trained_samples",
    "prepared_manifest",
    "teacher_run_manifest",
    "teacher_gap_manifest",
    "teacher_provenance_manifest",
    "tokenizer_contract",
    "server_scoring_contract",
    "terminal_audit",
    "student_eval_custody",
    "prelaunch_receipt",
}


def _collect_bound_artifact_paths(value: Any, protected: set[Path]) -> None:
    """Collect absolute artifact/env paths from a validated custody payload."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in _TREE_PATH_KEYS and isinstance(item, str) and Path(item).is_absolute():
                protected.add(Path(item).resolve())
            elif key in _FILE_PATH_KEYS and isinstance(item, str) and Path(item).is_absolute():
                path = Path(item).resolve()
                protected.add(path if path.exists() and path.is_dir() else path.parent)
            else:
                _collect_bound_artifact_paths(item, protected)
    elif isinstance(value, list):
        for item in value:
            _collect_bound_artifact_paths(item, protected)


def _student_gate_protected_trees(
    gate: Mapping[str, Any],
    *,
    gate_path: Path | None = None,
) -> list[Path]:
    """Protect every direct and transitive input of one held-out gate."""

    protected: set[Path] = set()
    if gate_path is not None:
        protected.add(Path(gate_path).resolve().parent)
    _collect_bound_artifact_paths(gate.get("inputs"), protected)
    _collect_bound_artifact_paths(gate.get("evaluation_binding"), protected)
    _collect_bound_artifact_paths(gate.get("student_run_binding"), protected)

    run_binding = gate.get("student_run_binding")
    if not isinstance(run_binding, Mapping):
        raise ValueError("held-out gate lacks student-run custody")
    run_path_value = run_binding.get("run_manifest")
    if not isinstance(run_path_value, str) or not Path(run_path_value).is_absolute():
        raise ValueError("held-out gate lacks an absolute student run manifest")
    run_path = Path(run_path_value).resolve()
    if run_path.is_file():
        run_disk = _json_object(run_path, "student run manifest for output protection")
        _collect_bound_artifact_paths(run_disk, protected)
        gates = run_disk.get("gates")
        if isinstance(gates, Mapping):
            for gate_payload in gates.values():
                _collect_bound_artifact_paths(gate_payload, protected)

    teacher = run_binding.get("teacher")
    if isinstance(teacher, Mapping):
        teacher_gate_value = teacher.get("teacher_gap_manifest")
        if isinstance(teacher_gate_value, str):
            teacher_gate_path = Path(teacher_gate_value).resolve()
            protected.add(teacher_gate_path.parent)
            if teacher_gate_path.is_file():
                _collect_bound_artifact_paths(
                    _json_object(teacher_gate_path, "teacher gate for output protection"),
                    protected,
                )
    return sorted(protected, key=str)


def _readout_protected_trees(payload: Mapping[str, Any]) -> list[Path]:
    """Return every direct/transitive artifact tree a readout must not mutate."""

    inputs = payload.get("inputs")
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError("readout lacks bound held-out gate inputs")
    protected: set[Path] = set()
    preregistration = payload.get("preregistration")
    if isinstance(preregistration, dict):
        _collect_bound_artifact_paths(
            {
                "path": preregistration.get("path"),
                "launch_ledger": preregistration.get("launch_ledger"),
                "one_step_diagnostic": preregistration.get("one_step_diagnostic"),
                "selection_context": preregistration.get("selection_context"),
            },
            protected,
        )
        selection_context = preregistration.get("selection_context")
        if isinstance(selection_context, Mapping):
            teacher_gate_binding = selection_context.get("teacher_gap_manifest")
            if isinstance(teacher_gate_binding, Mapping):
                teacher_gate_path = Path(
                    str(teacher_gate_binding.get("path"))
                ).resolve()
                _file_binding(
                    teacher_gate_path,
                    teacher_gate_binding.get("sha256"),
                    "M-negative teacher gate for output protection",
                )
                _collect_bound_artifact_paths(
                    _json_object(
                        teacher_gate_path,
                        "M-negative teacher gate for output protection",
                    ),
                    protected,
                )
    for key, binding in inputs.items():
        if not isinstance(binding, dict):
            raise ValueError(f"readout input binding is invalid: {key}")
        gate_path = Path(str(binding.get("path"))).resolve()
        _file_binding(gate_path, binding.get("sha256"), f"readout gate {key}")
        gate = _json_object(gate_path, f"readout gate {key}")
        protected.update(
            _student_gate_protected_trees(gate, gate_path=gate_path)
        )
    return sorted(protected, key=str)


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
    m_negative = subparsers.add_parser(
        "m-negative-compatibility-audit",
        help="seal a negative-only current-scorer replay of the historical M gate",
    )
    m_negative.add_argument("--teacher-gap-manifest", type=Path, required=True)
    m_negative.add_argument("--terminal-audit", type=Path, required=True)
    m_negative.add_argument("--teacher-run-manifest", type=Path, required=True)
    m_negative.add_argument("--trained-adapter", type=Path, required=True)
    m_negative.add_argument("--output", type=Path, required=True)
    prelaunch = subparsers.add_parser(
        "prelaunch", help="seal one primary student launch against the preregistration"
    )
    prelaunch.add_argument("--run-key", choices=tuple(O_TEACHER_CONTRACT), required=True)
    prelaunch.add_argument("--run-id", required=True)
    prelaunch.add_argument("--scheduler-job-id", required=True)
    prelaunch.add_argument("--mode", choices=("task_rl", "task_rl_k1_gap"), required=True)
    prelaunch.add_argument("--student-source", choices=("M", "O"), required=True)
    prelaunch.add_argument("--out-dir", type=Path, required=True)
    prelaunch.add_argument("--student-support-manifest", type=Path, required=True)
    prelaunch.add_argument("--teacher-gap-manifest", type=Path)
    prelaunch.add_argument("--teacher-checkpoint", type=Path)
    prelaunch.add_argument("--teacher-provenance-manifest", type=Path)
    prelaunch.add_argument("--preregistration", type=Path, required=True)
    prelaunch.add_argument("--launch-ledger", type=Path, required=True)
    prelaunch.add_argument("--output", type=Path, required=True)
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

    o_teacher = subparsers.add_parser(
        "o-teacher-readout",
        help="combine the four allowed gates conditional on the passing O teacher",
    )
    for key in O_TEACHER_CONTRACT:
        o_teacher.add_argument(
            f"--{key.replace('_', '-').lower()}", type=Path, required=True
        )
    o_teacher.add_argument("--preregistration", type=Path, required=True)
    o_teacher.add_argument("--launch-ledger", type=Path, required=True)
    o_teacher.add_argument("--output-json", type=Path, required=True)
    o_teacher.add_argument("--output-markdown", type=Path, required=True)
    o_teacher.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "m-negative-compatibility-audit":
        _preflight_result_outputs([args.output])
        payload = m_teacher_negative_compatibility_audit(args)
        output = args.output.resolve()
        try:
            _write_new(output, json.dumps(payload, indent=2, sort_keys=True) + "\n")
            output.chmod(0o444)
            validated = _validate_m_negative_selection_context(
                {"path": str(output), "sha256": sha256_file(output)}
            )
            if not validated["strict_negative_replay_confirmed"]:
                raise RuntimeError("M-negative compatibility audit failed strict replay")
        except BaseException:
            output.unlink(missing_ok=True)
            raise
        print(
            json.dumps(
                {
                    "output": str(output),
                    "selection_context_only": True,
                    "m_arms_prohibited": True,
                    "strict_negative_replay_confirmed": True,
                },
                sort_keys=True,
            )
        )
        return 0

    if args.command == "prelaunch":
        receipt = o_teacher_prelaunch_receipt(args)
        print(
            json.dumps(
                {
                    "output": str(args.output.resolve()),
                    "run_key": receipt["run_key"],
                    "sealed_before_optimizer_start": True,
                },
                sort_keys=True,
            )
        )
        return 0

    if args.command == "heldout":
        _preflight_result_outputs([args.output])
        payload = student_heldout_result(args)
        _preflight_result_outputs(
            [args.output], protected_trees=_student_gate_protected_trees(payload)
        )
        if recompute_student_heldout_result(payload) != payload:
            raise RuntimeError("held-out gate failed deterministic self-recomputation")
        output = args.output.resolve()
        try:
            _write_new(output, json.dumps(payload, indent=2, sort_keys=True) + "\n")
            written = _json_object(output, "published held-out gate")
            if written != payload or recompute_student_heldout_result(written) != written:
                raise RuntimeError(
                    "held-out gate failed deterministic recomputation after publication"
                )
            output.chmod(0o444)
        except BaseException:
            output.unlink(missing_ok=True)
            raise
        print(json.dumps({"output": str(args.output.resolve()), "passed": True}, sort_keys=True))
        return 0

    initial_outputs = [args.output_json, args.output_markdown]
    if args.command == "o-teacher-readout":
        initial_outputs.append(args.output_manifest)
    _preflight_result_outputs(initial_outputs)
    if args.command == "matrix":
        gate_paths = {
            key: getattr(args, key.lower())
            for key in MATRIX_CONTRACT
        }
        payload = matrix_readout(gate_paths)
        if recompute_matrix_readout(payload) != payload:
            raise RuntimeError("matrix readout failed deterministic self-recomputation")
        markdown = matrix_markdown(payload)
    elif args.command == "o-teacher-readout":
        gate_paths = {
            key: getattr(args, key.lower())
            for key in O_TEACHER_CONTRACT
        }
        payload = o_teacher_readout(
            gate_paths,
            preregistration_path=args.preregistration,
            launch_ledger_path=args.launch_ledger,
        )
        if recompute_o_teacher_readout(payload) != payload:
            raise RuntimeError(
                "O-teacher readout failed deterministic self-recomputation"
            )
        expected_outputs = payload["preregistration"]["outputs"]
        if {
            "json": str(args.output_json.resolve()),
            "markdown": str(args.output_markdown.resolve()),
            "manifest": str(args.output_manifest.resolve()),
        } != expected_outputs:
            raise ValueError(
                "O-teacher output paths differ from the sealed preregistration"
            )
        markdown = o_teacher_markdown(payload)
    else:  # pragma: no cover - argparse requires a registered subcommand.
        raise RuntimeError(f"unsupported result command: {args.command}")
    final_outputs = [args.output_json, args.output_markdown]
    if args.command == "o-teacher-readout":
        final_outputs.append(args.output_manifest)
    _preflight_result_outputs(
        final_outputs,
        protected_trees=_readout_protected_trees(payload),
    )
    if args.command == "o-teacher-readout":
        _write_readout_bundle(
            output_json=args.output_json,
            json_content=json.dumps(payload, indent=2, sort_keys=True) + "\n",
            output_markdown=args.output_markdown,
            markdown_content=markdown,
            output_manifest=args.output_manifest,
            payload=payload,
        )
    else:
        _write_new(
            args.output_json,
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
        )
        _write_new(args.output_markdown, markdown)
    print(
        json.dumps(
            {
                "output_json": str(args.output_json.resolve()),
                "output_markdown": str(args.output_markdown.resolve()),
                "output_manifest": (
                    str(args.output_manifest.resolve())
                    if args.command == "o-teacher-readout"
                    else None
                ),
                "authorized": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
