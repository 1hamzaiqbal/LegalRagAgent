#!/usr/bin/env python3
"""Seal and validate the objective-family scientific launch boundary.

The tracked objective and fidelity plans describe required work but cannot
authorize training.  This module validates an outcome-blind external
preregistration after every prerequisite exists, seals a launch plan, and
creates one scheduler-bound receipt immediately before each optimizer run.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import stat
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

_REPO_ROOT = Path(__file__).resolve().parents[2]

try:
    from .objective_family_inputs import (
        EXPECTED_STUDENT,
        EXPECTED_STUDENT_REVISION,
        canonical_json_sha256,
        sha256_file,
        sha256_tree,
        validate_initialization_manifest,
    )
    from .objective_registry import (
        EXPECTED_OBJECTIVE_IDS,
        LOCAL_OBJECTIVE_IDS,
        UPSTREAM_OBJECTIVE_IDS,
        load_objective_registry,
    )
    from ..opd_math.quality_gates import (
        recompute_student_gate,
        recompute_teacher_gate,
    )
except ImportError:
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from objective_family_inputs import (  # type: ignore
        EXPECTED_STUDENT,
        EXPECTED_STUDENT_REVISION,
        canonical_json_sha256,
        sha256_file,
        sha256_tree,
        validate_initialization_manifest,
    )
    from objective_registry import (  # type: ignore
        EXPECTED_OBJECTIVE_IDS,
        LOCAL_OBJECTIVE_IDS,
        UPSTREAM_OBJECTIVE_IDS,
        load_objective_registry,
    )
    from scripts.opd_math.quality_gates import (  # type: ignore
        recompute_student_gate,
        recompute_teacher_gate,
    )


ROOT = _REPO_ROOT
STUDENT_PLAN = ROOT / "configs/opd_math/objective_family_student_plan.json"
FIDELITY_PLAN = ROOT / "configs/opd_math/fidelity_plan.json"
PREREGISTRATION_ID = "opd_math_objective_family_preregistration_v1"
LAUNCH_PLAN_ID = "opd_math_objective_family_launch_plan_v1"
PRELAUNCH_RECEIPT_ID = "opd_math_objective_family_prelaunch_receipt_v1"
FIDELITY_CLOSURE_ID = "opd_math_objective_family_fidelity_closure_v1"
SOURCES = ("M", "O")
SEEDS = (0, 1, 2)
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
SAFE = re.compile(r"^[A-Za-z0-9._-]+$")


def arm_key(objective_id: str, source: str, seed: int) -> str:
    return f"{objective_id}__{source}__seed{seed}"


EXPECTED_ARM_KEYS = tuple(
    arm_key(objective_id, source, seed)
    for seed in SEEDS
    for source in SOURCES
    for objective_id in EXPECTED_OBJECTIVE_IDS
)
EXPECTED_DIAGNOSTIC_KEYS = tuple(
    f"{objective_id}__{source}"
    for source in SOURCES
    for objective_id in EXPECTED_OBJECTIVE_IDS
)


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _git_state() -> dict[str, Any]:
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
    return {"commit": commit, "clean": not status.strip()}


def _parse_utc(value: Any, label: str) -> datetime:
    _expect(isinstance(value, str) and value.endswith("Z"), f"{label} must be UTC Z time")
    try:
        return datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{label} is invalid") from exc


def _regular_readonly_file(path: str | Path, label: str) -> Path:
    raw = Path(path)
    _expect(not raw.is_symlink() and raw.is_file(), f"{label} must be a regular file")
    _expect(
        raw.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH) == 0,
        f"{label} must be sealed read-only",
    )
    return raw.resolve()


def _json(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _expect(isinstance(payload, dict), f"{label} must be a JSON object")
    return payload


def _binding(value: Any, label: str, *, readonly: bool = True) -> tuple[Path, dict[str, Any]]:
    _expect(
        isinstance(value, dict) and set(value) == {"path", "sha256"},
        f"{label} binding schema drifted",
    )
    path = (
        _regular_readonly_file(value["path"], label)
        if readonly
        else Path(value["path"]).resolve()
    )
    _expect(sha256_file(path) == value["sha256"], f"{label} hash drifted")
    return path, _json(path, label)


def validate_fidelity_closure(value: Any, *, commit: str) -> dict[str, Any]:
    try:
        from .objective_family_fidelity import (
            validate_diagnostic_receipt,
            validate_real_model_receipt,
        )
    except ImportError:
        from objective_family_fidelity import (  # type: ignore
            validate_diagnostic_receipt,
            validate_real_model_receipt,
        )

    path, payload = _binding(value, "objective-family fidelity closure")
    expected_keys = {
        "schema_version",
        "closure",
        "status",
        "all_levels_passed",
        "scientific_launch_authorized",
        "git_commit",
        "objective_registry_sha256",
        "fidelity_plan",
        "stored_real_model",
        "full_custody_diagnostics",
        "heldout_outcomes_inspected",
        "claim_boundary",
    }
    _expect(set(payload) == expected_keys, "fidelity closure schema drifted")
    registry = load_objective_registry()
    for field, expected in (
        ("schema_version", 1),
        ("closure", FIDELITY_CLOSURE_ID),
        ("status", "passed"),
        ("all_levels_passed", True),
        ("scientific_launch_authorized", False),
        ("git_commit", commit),
        ("objective_registry_sha256", registry["sha256"]),
        ("heldout_outcomes_inspected", False),
    ):
        _expect(payload.get(field) == expected, f"fidelity closure {field} drifted")
    fidelity_path, _ = _binding(payload["fidelity_plan"], "tracked fidelity plan", readonly=False)
    _expect(fidelity_path == FIDELITY_PLAN.resolve(), "fidelity closure plan path drifted")
    real = payload.get("stored_real_model")
    _expect(
        isinstance(real, dict)
        and set(real) == {"path", "sha256", "status", "behavior_scores_from_generation"}
        and real.get("status") == "passed"
        and real.get("behavior_scores_from_generation") is True,
        "real-model fidelity binding drifted",
    )
    real_path = _regular_readonly_file(real["path"], "real-model fidelity receipt")
    _expect(sha256_file(real_path) == real["sha256"], "real-model fidelity hash drifted")
    validate_real_model_receipt(real_path, commit=commit)
    diagnostics = payload.get("full_custody_diagnostics")
    _expect(
        isinstance(diagnostics, dict) and set(diagnostics) == set(EXPECTED_DIAGNOSTIC_KEYS),
        "fidelity closure diagnostic matrix drifted",
    )
    for key, item in diagnostics.items():
        _expect(
            isinstance(item, dict)
            and set(item)
            == {
                "path",
                "sha256",
                "objective_id",
                "source",
                "status",
                "scientific_use_allowed",
            },
            f"fidelity diagnostic binding schema drifted: {key}",
        )
        objective_id, source = key.rsplit("__", 1)
        _expect(
            item.get("objective_id") == objective_id
            and item.get("source") == source
            and item.get("status") == "passed_plumbing"
            and item.get("scientific_use_allowed") is False,
            f"fidelity diagnostic identity drifted: {key}",
        )
        receipt_path = _regular_readonly_file(item["path"], f"fidelity diagnostic {key}")
        _expect(sha256_file(receipt_path) == item["sha256"], f"diagnostic hash drifted: {key}")
        validate_diagnostic_receipt(
            receipt_path,
            objective_id=objective_id,
            source=source,
            commit=commit,
        )
    return {"path": str(path), "sha256": sha256_file(path), "payload": payload}


def _validate_freeze(value: Any, *, commit: str, kind: str) -> dict[str, str]:
    _expect(
        isinstance(value, dict) and set(value) == {"path", "sha256"},
        f"{kind} freeze binding drifted",
    )
    path = _regular_readonly_file(value["path"], f"{kind} environment freeze")
    _expect(
        path.name == f"{kind}.freeze.txt"
        and path.parent.name == commit
        and path.parent.parent.name == "environment_freezes",
        f"{kind} freeze is not commit-specific",
    )
    _expect(sha256_file(path) == value["sha256"], f"{kind} freeze hash drifted")
    return {"path": str(path), "sha256": value["sha256"]}


def _validate_support(value: Any, *, source: str, commit: str) -> dict[str, Any]:
    expected_keys = {"path", "sha256", "payload_sha256", "source"}
    _expect(isinstance(value, dict) and set(value) == expected_keys, "support binding drifted")
    _expect(value.get("source") == source, f"{source} support source drifted")
    path = _regular_readonly_file(value["path"], f"{source} support gate")
    _expect(sha256_file(path) == value["sha256"], f"{source} support hash drifted")
    payload = _json(path, f"{source} support gate")
    _expect(canonical_json_sha256(payload) == value["payload_sha256"], f"{source} support payload drifted")
    _expect(
        payload.get("passed") is True
        and payload.get("authorizes_scientific_training") is True
        and payload.get("task_sources") == [source]
        and payload.get("evaluation_git_commit") == commit,
        f"{source} support gate is not a passing same-commit scientific gate",
    )
    original = dict(payload)
    original.pop("manifest_sha256", None)
    _expect(
        recompute_student_gate(original) == original,
        f"{source} support gate differs from deterministic recomputation",
    )
    return dict(value)


def _validate_teacher(value: Any, *, commit: str) -> dict[str, Any]:
    expected_keys = {
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
    }
    _expect(isinstance(value, dict) and set(value) == expected_keys, "O teacher identity drifted")
    _expect(value.get("teacher_source") == "O", "objective-family teacher must be O")
    _expect(
        value.get("base_model") == "Qwen/Qwen3-8B"
        and isinstance(value.get("base_revision"), str)
        and HEX40.fullmatch(value["base_revision"]),
        "O teacher base identity drifted",
    )
    gap_path = _regular_readonly_file(value["teacher_gap_manifest"], "O teacher gap")
    gap = _json(gap_path, "O teacher gap")
    _expect(sha256_file(gap_path) == value["teacher_gap_manifest_sha256"], "O gap hash drifted")
    _expect(canonical_json_sha256(gap) == value["teacher_gap_payload_sha256"], "O gap payload drifted")
    _expect(
        gap.get("passed") is True
        and gap.get("authorizes_scientific_merge") is True
        and gap.get("task_sources") == ["O"]
        and gap.get("evaluation_git_commit") == commit,
        "O teacher gap is not a same-commit passing O gate",
    )
    original_gap = dict(gap)
    original_gap.pop("manifest_sha256", None)
    _expect(
        recompute_teacher_gate(original_gap) == original_gap,
        "O teacher gap differs from deterministic recomputation",
    )
    checkpoint = Path(value["merged_checkpoint"]).resolve()
    _expect(checkpoint.is_dir() and not checkpoint.is_symlink(), "O checkpoint is missing")
    provenance_path = checkpoint / "merge_provenance.json"
    _expect(provenance_path.is_file() and not provenance_path.is_symlink(), "O provenance is missing")
    _expect(
        sha256_tree(checkpoint, exclude_relative_paths=("merge_provenance.json",))
        == value["merged_checkpoint_tree_sha256"],
        "O merged checkpoint tree drifted",
    )
    provenance = _json(provenance_path, "O merge provenance")
    _expect(
        sha256_file(provenance_path) == value["merge_provenance_manifest_sha256"]
        and canonical_json_sha256(provenance) == value["merge_provenance_payload_sha256"],
        "O merge provenance drifted",
    )
    _expect(
        provenance.get("schema") == "opd_math_merged_teacher_v3"
        and provenance.get("status") == "completed"
        and provenance.get("base_model") == value["base_model"]
        and provenance.get("base_revision") == value["base_revision"]
        and provenance.get("teacher_gap_manifest") == str(gap_path)
        and provenance.get("teacher_gap_manifest_sha256")
        == value["teacher_gap_manifest_sha256"]
        and provenance.get("output_checkpoint") == str(checkpoint)
        and provenance.get("output_checkpoint_tree_sha256")
        == value["merged_checkpoint_tree_sha256"],
        "O merge provenance does not bind the preregistered teacher identity",
    )
    return dict(value)


def _validate_prompt_bindings(value: Any, *, commit: str) -> dict[str, Any]:
    expected = {f"{source}_seed{seed}" for source in SOURCES for seed in SEEDS}
    _expect(isinstance(value, dict) and set(value) == expected, "prompt-plan matrix drifted")
    result: dict[str, Any] = {}
    for key in sorted(expected):
        source, seed_text = key.split("_seed")
        seed = int(seed_text)
        item = value[key]
        _expect(
            isinstance(item, dict) and set(item) == {"path", "sha256", "sequence_sha256"},
            f"prompt-plan binding schema drifted: {key}",
        )
        path = _regular_readonly_file(item["path"], f"prompt plan {key}")
        payload = _json(path, f"prompt plan {key}")
        _expect(
            sha256_file(path) == item["sha256"]
            and payload.get("git_commit") == commit
            and payload.get("source") == source
            and payload.get("seed") == seed
            and payload.get("sequence_sha256") == item["sequence_sha256"]
            and isinstance(payload.get("sequence"), list)
            and len(payload["sequence"]) == 100,
            f"prompt plan identity drifted: {key}",
        )
        result[key] = dict(item)
    return result


def _validate_initializations(value: Any, *, commit: str) -> dict[str, Any]:
    expected = {f"seed{seed}" for seed in SEEDS}
    _expect(isinstance(value, dict) and set(value) == expected, "initial adapter matrix drifted")
    result: dict[str, Any] = {}
    for key in sorted(expected):
        seed = int(key.removeprefix("seed"))
        item = value[key]
        _expect(
            isinstance(item, dict)
            and set(item) == {"manifest_path", "manifest_sha256", "adapter_path", "adapter_tree_sha256"},
            f"initial adapter binding schema drifted: {key}",
        )
        contract = validate_initialization_manifest(
            item["manifest_path"],
            student=EXPECTED_STUDENT,
            student_revision=EXPECTED_STUDENT_REVISION,
            seed=seed,
            lora_r=32,
            git_commit=commit,
        )
        _expect(
            contract["sha256"] == item["manifest_sha256"]
            and contract["adapter_path"] == item["adapter_path"]
            and contract["adapter_tree_sha256"] == item["adapter_tree_sha256"],
            f"initial adapter identity drifted: {key}",
        )
        result[key] = dict(item)
    return result


def validate_preregistration_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "preregistration",
        "status",
        "scientific_launch_authorized",
        "student_outcome_blind",
        "sealed_before_student_arm_launch",
        "student_arm_outcomes_inspected_before_sealing",
        "campaign_id",
        "created_utc",
        "git_commit",
        "objective_registry",
        "student_training_plan",
        "fidelity_closure",
        "prepared_manifest",
        "environment_freezes",
        "m_teacher_boundary",
        "student_support",
        "o_teacher",
        "prompt_plans",
        "initial_adapters",
        "sources",
        "seeds",
        "objective_ids",
        "arm_keys",
        "arms",
        "analysis",
        "stop_rules",
    }
    _expect(set(payload) == expected_keys, "objective-family preregistration schema drifted")
    for field, expected in (
        ("schema_version", 1),
        ("preregistration", PREREGISTRATION_ID),
        ("status", "sealed_before_student_launch"),
        ("scientific_launch_authorized", True),
        ("student_outcome_blind", True),
        ("sealed_before_student_arm_launch", True),
        ("student_arm_outcomes_inspected_before_sealing", False),
        ("sources", list(SOURCES)),
        ("seeds", list(SEEDS)),
        ("objective_ids", list(EXPECTED_OBJECTIVE_IDS)),
        ("arm_keys", list(EXPECTED_ARM_KEYS)),
    ):
        _expect(payload.get(field) == expected, f"preregistration {field} drifted")
    _expect(isinstance(payload.get("campaign_id"), str) and SAFE.fullmatch(payload["campaign_id"]), "campaign ID is invalid")
    _parse_utc(payload.get("created_utc"), "preregistration created_utc")
    commit = payload.get("git_commit")
    _expect(isinstance(commit, str) and HEX40.fullmatch(commit), "preregistration commit is invalid")
    state = _git_state()
    _expect(state == {"commit": commit, "clean": True}, "preregistration Git state is not current and clean")
    registry = load_objective_registry()
    for field, path in (
        ("objective_registry", Path(registry["path"])),
        ("student_training_plan", STUDENT_PLAN),
    ):
        binding = payload.get(field)
        _expect(isinstance(binding, dict) and set(binding) == {"path", "sha256"}, f"{field} binding drifted")
        _expect(
            Path(binding["path"]).resolve() == path.resolve()
            and sha256_file(path) == binding["sha256"],
            f"{field} bytes drifted",
        )
    fidelity = validate_fidelity_closure(payload["fidelity_closure"], commit=commit)
    prepared_path, prepared = _binding(payload["prepared_manifest"], "prepared manifest", readonly=False)
    _expect(prepared.get("scientific_use_allowed") is True, "prepared manifest is not scientific")
    freezes = payload.get("environment_freezes")
    _expect(
        isinstance(freezes, dict)
        and set(freezes) == {"train", "serve", "upstream_verl"},
        "environment freeze matrix drifted",
    )
    validated_freezes = {
        kind: _validate_freeze(freezes[kind], commit=commit, kind=kind)
        for kind in ("train", "serve", "upstream_verl")
    }
    _expect(
        payload.get("m_teacher_boundary")
        == {
            "m_teacher_gate_passed": False,
            "m_teacher_permanently_excluded": True,
            "m_retraining_allowed": False,
            "m_merge_allowed": False,
            "m_m_allowed": False,
            "m_o_allowed": False,
            "math_student_and_evaluation_use_allowed": True,
        },
        "permanent M-teacher boundary drifted",
    )
    support = payload.get("student_support")
    _expect(isinstance(support, dict) and set(support) == set(SOURCES), "support matrix drifted")
    validated_support = {
        source: _validate_support(support[source], source=source, commit=commit)
        for source in SOURCES
    }
    teacher = _validate_teacher(payload.get("o_teacher"), commit=commit)
    prompts = _validate_prompt_bindings(payload.get("prompt_plans"), commit=commit)
    initializations = _validate_initializations(payload.get("initial_adapters"), commit=commit)
    arms = payload.get("arms")
    _expect(isinstance(arms, dict) and set(arms) == set(EXPECTED_ARM_KEYS), "preregistered arm matrix drifted")
    run_ids: set[str] = set()
    for key in EXPECTED_ARM_KEYS:
        arm = arms[key]
        _expect(
            isinstance(arm, dict)
            and set(arm)
            == {
                "objective_id",
                "implementation",
                "source",
                "seed",
                "run_id",
                "prompt_plan_key",
                "initial_adapter_key",
                "training_out",
                "prelaunch_receipt",
                "heldout_gate",
            },
            f"arm schema drifted: {key}",
        )
        objective_id, source, seed_text = key.rsplit("__", 2)
        seed = int(seed_text.removeprefix("seed"))
        implementation = "local" if objective_id in LOCAL_OBJECTIVE_IDS else "upstream_verl"
        _expect(
            arm.get("objective_id") == objective_id
            and arm.get("implementation") == implementation
            and arm.get("source") == source
            and arm.get("seed") == seed
            and arm.get("prompt_plan_key") == f"{source}_seed{seed}"
            and arm.get("initial_adapter_key") == f"seed{seed}",
            f"arm identity drifted: {key}",
        )
        run_id = arm.get("run_id")
        _expect(isinstance(run_id, str) and SAFE.fullmatch(run_id) and run_id not in run_ids, f"arm run ID drifted: {key}")
        run_ids.add(run_id)
        for path_field in ("training_out", "prelaunch_receipt", "heldout_gate"):
            value = arm.get(path_field)
            _expect(isinstance(value, str) and Path(value).is_absolute(), f"arm {key} {path_field} must be absolute")
    _expect(
        payload.get("analysis")
        == {
            "primary_contrasts": [
                "task_rl_k1_ungated_clip5-minus-task_rl@M",
                "task_rl_k1_ungated_clip5-minus-task_rl@O",
                "task_rl_k1_gated_clip5_beta5-minus-task_rl_k1_ungated_clip5@M",
                "task_rl_k1_gated_clip5_beta5-minus-task_rl_k1_ungated_clip5@O",
            ],
            "bootstrap": "paired_hierarchical_seed_then_record",
            "bootstrap_draws": 10000,
            "bootstrap_seed": 0,
            "familywise_interval": 0.9875,
            "multiplicity": "bonferroni_four_co_primary",
            "inspect_heldout_only_after_all_terminal": True,
        },
        "preregistered analysis drifted",
    )
    _expect(
        payload.get("stop_rules")
        == {
            "no_gate_relaxation": True,
            "no_objective_dropping": True,
            "no_replacement_seeds": True,
            "no_rescue_training": True,
            "numerical_failure_is_terminal_per_arm": True,
            "m_teacher_use_prohibited": True,
            "heldout_outcome_may_not_change_training": True,
        },
        "preregistered stop rules drifted",
    )
    return {
        "payload": dict(payload),
        "commit": commit,
        "registry": registry,
        "fidelity": fidelity,
        "prepared_manifest": {"path": str(prepared_path), "sha256": sha256_file(prepared_path)},
        "environment_freezes": validated_freezes,
        "student_support": validated_support,
        "o_teacher": teacher,
        "prompt_plans": prompts,
        "initial_adapters": initializations,
    }


def validate_preregistration(path: str | Path) -> dict[str, Any]:
    resolved = _regular_readonly_file(path, "objective-family preregistration")
    validated = validate_preregistration_payload(_json(resolved, "objective-family preregistration"))
    validated.update({"path": str(resolved), "sha256": sha256_file(resolved)})
    return validated


def validate_launch_plan(path: str | Path, *, preregistration: dict[str, Any]) -> dict[str, Any]:
    resolved = _regular_readonly_file(path, "objective-family launch plan")
    payload = _json(resolved, "objective-family launch plan")
    _expect(
        set(payload)
        == {
            "schema_version",
            "launch_plan",
            "campaign_id",
            "created_utc",
            "sealed_before_student_arm_launch",
            "student_arm_outcomes_inspected_before_sealing",
            "preregistration",
            "arm_run_ids",
        },
        "objective-family launch plan schema drifted",
    )
    for field, expected in (
        ("schema_version", 1),
        ("launch_plan", LAUNCH_PLAN_ID),
        ("campaign_id", preregistration["payload"]["campaign_id"]),
        ("sealed_before_student_arm_launch", True),
        ("student_arm_outcomes_inspected_before_sealing", False),
    ):
        _expect(payload.get(field) == expected, f"launch plan {field} drifted")
    _parse_utc(payload.get("created_utc"), "launch plan created_utc")
    _expect(
        payload.get("preregistration")
        == {"path": preregistration["path"], "sha256": preregistration["sha256"]},
        "launch plan preregistration binding drifted",
    )
    expected_run_ids = {
        key: preregistration["payload"]["arms"][key]["run_id"]
        for key in EXPECTED_ARM_KEYS
    }
    _expect(payload.get("arm_run_ids") == expected_run_ids, "launch plan run IDs drifted")
    return {"path": str(resolved), "sha256": sha256_file(resolved), "payload": payload}


def validate_prelaunch_receipt(args) -> dict[str, Any]:
    receipt_path = _regular_readonly_file(args.prelaunch_receipt, "objective-family prelaunch receipt")
    receipt = _json(receipt_path, "objective-family prelaunch receipt")
    _expect(
        set(receipt)
        == {
            "schema_version",
            "receipt",
            "sealed_before_optimizer_start",
            "campaign_id",
            "run_key",
            "run_id",
            "scheduler_job_id",
            "objective_id",
            "source",
            "seed",
            "git_commit",
            "out_dir",
            "expected_artifacts",
            "preregistration",
            "launch_plan",
            "student_support",
            "o_teacher",
            "prompt_plan",
            "initial_adapter",
        },
        "objective-family prelaunch receipt schema drifted",
    )
    prereg_path, _ = _binding(receipt["preregistration"], "receipt preregistration")
    prereg = validate_preregistration(prereg_path)
    launch_path, _ = _binding(receipt["launch_plan"], "receipt launch plan")
    launch = validate_launch_plan(launch_path, preregistration=prereg)
    registry_contract = getattr(args, "objective_registry_contract", None)
    _expect(isinstance(registry_contract, dict), "prelaunch lacks objective registry")
    objective = registry_contract["objective"]
    key = arm_key(objective["id"], args.student_source, args.seed)
    arm = prereg["payload"]["arms"].get(key)
    _expect(isinstance(arm, dict) and arm.get("implementation") == "local", "prelaunch arm is not local")
    out_dir = Path(args.out_dir).resolve()
    expected_artifacts = {
        "run_manifest": str((out_dir / "traces" / "run_manifest.json").resolve()),
        "student_completion_manifest": str((out_dir / "traces" / "completion_manifest.json").resolve()),
        "student_adapter": str((out_dir / "final").resolve()),
        "prelaunch_receipt": str(receipt_path),
    }
    for field, expected in (
        ("schema_version", 1),
        ("receipt", PRELAUNCH_RECEIPT_ID),
        ("sealed_before_optimizer_start", True),
        ("campaign_id", prereg["payload"]["campaign_id"]),
        ("run_key", key),
        ("run_id", args.campaign_run_id),
        ("scheduler_job_id", args.scheduler_job_id),
        ("objective_id", objective["id"]),
        ("source", args.student_source),
        ("seed", args.seed),
        ("git_commit", prereg["commit"]),
        ("out_dir", str(out_dir)),
        ("expected_artifacts", expected_artifacts),
        ("student_support", prereg["student_support"][args.student_source]),
        ("o_teacher", prereg["o_teacher"] if objective["sampled_k1"] else None),
        ("prompt_plan", prereg["prompt_plans"][arm["prompt_plan_key"]]),
        ("initial_adapter", prereg["initial_adapters"][arm["initial_adapter_key"]]),
    ):
        _expect(receipt.get(field) == expected, f"objective-family prelaunch {field} drifted")
    _expect(arm["run_id"] == args.campaign_run_id, "prelaunch run ID differs from preregistration")
    _expect(arm["training_out"] == str(out_dir), "prelaunch output differs from preregistration")
    _expect(arm["prelaunch_receipt"] == str(receipt_path), "prelaunch receipt path differs from preregistration")
    _expect(
        launch["payload"]["arm_run_ids"][key] == args.campaign_run_id,
        "prelaunch run ID differs from launch plan",
    )
    return {
        "path": str(receipt_path),
        "sha256": sha256_file(receipt_path),
        "sealed_before_optimizer_start": True,
        "run_key": key,
        "preregistration": receipt["preregistration"],
        "launch_plan": receipt["launch_plan"],
    }


def validate_upstream_prelaunch_receipt(
    path: str | Path,
    *,
    objective_id: str,
    source: str,
    seed: int,
    out_dir: str | Path,
    run_id: str,
    scheduler_job_id: str,
) -> dict[str, Any]:
    receipt_path = _regular_readonly_file(path, "upstream veRL prelaunch receipt")
    receipt = _json(receipt_path, "upstream veRL prelaunch receipt")
    prereg_path, _ = _binding(receipt.get("preregistration"), "receipt preregistration")
    prereg = validate_preregistration(prereg_path)
    launch_path, _ = _binding(receipt.get("launch_plan"), "receipt launch plan")
    launch = validate_launch_plan(launch_path, preregistration=prereg)
    key = arm_key(objective_id, source, seed)
    arm = prereg["payload"]["arms"].get(key)
    _expect(
        isinstance(arm, dict)
        and arm.get("implementation") == "upstream_verl"
        and objective_id in UPSTREAM_OBJECTIVE_IDS,
        "prelaunch arm is not pinned upstream veRL",
    )
    resolved_out = Path(out_dir).resolve()
    expected_artifacts = {
        "upstream_preflight": str(Path(str(resolved_out) + ".preflight.json")),
        "upstream_run_receipt": str(Path(str(resolved_out) + ".receipt.json")),
        "student_adapter": str((resolved_out / "final").resolve()),
        "prelaunch_receipt": str(receipt_path),
    }
    expected = {
        "schema_version": 1,
        "receipt": PRELAUNCH_RECEIPT_ID,
        "sealed_before_optimizer_start": True,
        "campaign_id": prereg["payload"]["campaign_id"],
        "run_key": key,
        "run_id": run_id,
        "scheduler_job_id": scheduler_job_id,
        "objective_id": objective_id,
        "source": source,
        "seed": seed,
        "git_commit": prereg["commit"],
        "out_dir": str(resolved_out),
        "expected_artifacts": expected_artifacts,
        "preregistration": {"path": prereg["path"], "sha256": prereg["sha256"]},
        "launch_plan": {"path": launch["path"], "sha256": launch["sha256"]},
        "student_support": prereg["student_support"][source],
        "o_teacher": prereg["o_teacher"],
        "prompt_plan": prereg["prompt_plans"][arm["prompt_plan_key"]],
        "initial_adapter": prereg["initial_adapters"][arm["initial_adapter_key"]],
    }
    _expect(receipt == expected, "upstream veRL prelaunch receipt drifted")
    _expect(
        arm["run_id"] == run_id
        and arm["training_out"] == str(resolved_out)
        and arm["prelaunch_receipt"] == str(receipt_path)
        and launch["payload"]["arm_run_ids"][key] == run_id,
        "upstream veRL prelaunch differs from sealed arm/launch plan",
    )
    return {"path": str(receipt_path), "sha256": sha256_file(receipt_path), "run_key": key}


def _write_new(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite sealed artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(path, 0o444)


def seal_preregistration(draft: Path, output: Path) -> dict[str, Any]:
    payload = _json(draft.resolve(), "objective-family preregistration draft")
    validate_preregistration_payload(payload)
    _write_new(output.resolve(), payload)
    return validate_preregistration(output.resolve())


def seal_launch_plan(preregistration_path: Path, output: Path, created_utc: str) -> dict[str, Any]:
    prereg = validate_preregistration(preregistration_path)
    payload = {
        "schema_version": 1,
        "launch_plan": LAUNCH_PLAN_ID,
        "campaign_id": prereg["payload"]["campaign_id"],
        "created_utc": created_utc,
        "sealed_before_student_arm_launch": True,
        "student_arm_outcomes_inspected_before_sealing": False,
        "preregistration": {"path": prereg["path"], "sha256": prereg["sha256"]},
        "arm_run_ids": {
            key: prereg["payload"]["arms"][key]["run_id"] for key in EXPECTED_ARM_KEYS
        },
    }
    _parse_utc(created_utc, "launch plan created_utc")
    _write_new(output.resolve(), payload)
    return validate_launch_plan(output.resolve(), preregistration=prereg)


def write_prelaunch(args: argparse.Namespace) -> dict[str, Any]:
    prereg = validate_preregistration(args.preregistration)
    launch = validate_launch_plan(args.launch_plan, preregistration=prereg)
    key = args.run_key
    _expect(key in EXPECTED_ARM_KEYS, "prelaunch run key is not registered")
    arm = prereg["payload"]["arms"][key]
    _expect(
        arm["implementation"] in {"local", "upstream_verl"},
        "prelaunch implementation is unsupported",
    )
    _expect(args.run_id == arm["run_id"], "prelaunch run ID drifted")
    _expect(isinstance(args.scheduler_job_id, str) and re.fullmatch(r"[1-9][0-9]*", args.scheduler_job_id), "scheduler job ID is invalid")
    output = Path(args.output).resolve()
    _expect(str(output) == arm["prelaunch_receipt"], "prelaunch output path was not preregistered")
    out_dir = Path(arm["training_out"]).resolve()
    objective = arm["objective_id"]
    expected_artifacts = (
        {
            "run_manifest": str((out_dir / "traces" / "run_manifest.json").resolve()),
            "student_completion_manifest": str(
                (out_dir / "traces" / "completion_manifest.json").resolve()
            ),
            "student_adapter": str((out_dir / "final").resolve()),
            "prelaunch_receipt": str(output),
        }
        if arm["implementation"] == "local"
        else {
            "upstream_preflight": str(Path(str(out_dir) + ".preflight.json")),
            "upstream_run_receipt": str(Path(str(out_dir) + ".receipt.json")),
            "student_adapter": str((out_dir / "final").resolve()),
            "prelaunch_receipt": str(output),
        }
    )
    payload = {
        "schema_version": 1,
        "receipt": PRELAUNCH_RECEIPT_ID,
        "sealed_before_optimizer_start": True,
        "campaign_id": prereg["payload"]["campaign_id"],
        "run_key": key,
        "run_id": arm["run_id"],
        "scheduler_job_id": args.scheduler_job_id,
        "objective_id": objective,
        "source": arm["source"],
        "seed": arm["seed"],
        "git_commit": prereg["commit"],
        "out_dir": str(out_dir),
        "expected_artifacts": expected_artifacts,
        "preregistration": {"path": prereg["path"], "sha256": prereg["sha256"]},
        "launch_plan": {"path": launch["path"], "sha256": launch["sha256"]},
        "student_support": prereg["student_support"][arm["source"]],
        "o_teacher": prereg["o_teacher"] if objective in set(EXPECTED_OBJECTIVE_IDS) - {"task_rl"} else None,
        "prompt_plan": prereg["prompt_plans"][arm["prompt_plan_key"]],
        "initial_adapter": prereg["initial_adapters"][arm["initial_adapter_key"]],
    }
    _write_new(output, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    seal = subparsers.add_parser("seal-preregistration")
    seal.add_argument("--draft", type=Path, required=True)
    seal.add_argument("--output", type=Path, required=True)
    launch = subparsers.add_parser("seal-launch-plan")
    launch.add_argument("--preregistration", type=Path, required=True)
    launch.add_argument("--created-utc", required=True)
    launch.add_argument("--output", type=Path, required=True)
    prelaunch = subparsers.add_parser("prelaunch")
    prelaunch.add_argument("--preregistration", type=Path, required=True)
    prelaunch.add_argument("--launch-plan", type=Path, required=True)
    prelaunch.add_argument("--run-key", required=True)
    prelaunch.add_argument("--run-id", required=True)
    prelaunch.add_argument("--scheduler-job-id", required=True)
    prelaunch.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "seal-preregistration":
        result = seal_preregistration(args.draft, args.output)
    elif args.command == "seal-launch-plan":
        result = seal_launch_plan(args.preregistration, args.output, args.created_utc)
    else:
        result = write_prelaunch(args)
    print(json.dumps({key: value for key, value in result.items() if key != "payload"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
