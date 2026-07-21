#!/usr/bin/env python3
"""Validate the fail-closed OPD objective-family fidelity ladder."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

try:
    from .objective_registry import EXPECTED_OBJECTIVE_IDS, UPSTREAM_VERL_COMMIT
except ImportError:
    from objective_registry import EXPECTED_OBJECTIVE_IDS, UPSTREAM_VERL_COMMIT  # type: ignore


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN = ROOT / "configs/opd_math/fidelity_plan.json"
REGISTRY = ROOT / "configs/opd_math/objective_registry.json"
SYNTHETIC_FIXTURE = ROOT / "configs/opd_math/fidelity/shared_rollout_k1_v1.json"
FINITE_STATE_SCRIPT = ROOT / "scripts/opd/verify_finite_state.py"
FINITE_STATE_TRACKED_RECEIPT = (
    ROOT / "evidence/july_2026/opd_finite_state_108548.json"
)
FIDELITY_ID = "opd_math_objective_family_fidelity_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_fidelity_plan(payload: Mapping[str, Any]) -> dict[str, Any]:
    expected_top = {
        "schema_version",
        "fidelity_id",
        "status",
        "scientific_launch_authorized",
        "objective_registry_sha256",
        "upstream_verl_commit",
        "sources",
        "objective_ids",
        "levels",
        "stage_rules",
    }
    _require(set(payload) == expected_top, "fidelity plan top-level schema drifted")
    _require(payload["schema_version"] == 1, "fidelity plan schema version drifted")
    _require(payload["fidelity_id"] == FIDELITY_ID, "fidelity plan identity drifted")
    _require(
        payload["status"] == "implementation_fidelity_in_progress_not_launch_authorized",
        "fidelity plan status drifted",
    )
    _require(payload["scientific_launch_authorized"] is False, "fidelity plan authorized science")
    _require(payload["objective_registry_sha256"] == sha256_file(REGISTRY), "registry bytes drifted")
    _require(payload["upstream_verl_commit"] == UPSTREAM_VERL_COMMIT, "veRL commit drifted")
    _require(payload["sources"] == ["M", "O"], "fidelity source order drifted")
    _require(tuple(payload["objective_ids"]) == EXPECTED_OBJECTIVE_IDS, "objective IDs drifted")

    levels = payload["levels"]
    _require(
        set(levels)
        == {
            "analytic_imported_verl",
            "finite_state",
            "stored_synthetic",
            "stored_real_model",
            "full_custody_one_step",
        },
        "fidelity levels drifted",
    )
    analytic = levels["analytic_imported_verl"]
    _require(analytic["status"] == "passed" and analytic["slurm_job_id"] == "108498", "analytic receipt drifted")
    _require(
        analytic["receipt_sha256"]
        == "9f4a93fbb75d7ddcc4ca5abe9e9f3b5ed7ebd336197f6dc0e0e5e0b4a5a39d47",
        "analytic receipt hash drifted",
    )
    finite = levels["finite_state"]
    _require(
        set(finite)
        == {
            "status",
            "slurm_job_id",
            "receipt_path",
            "tracked_receipt_path",
            "receipt_sha256",
            "required_checks",
        },
        "finite-state receipt schema drifted",
    )
    _require(
        finite["status"] == "passed"
        and finite["slurm_job_id"] == "108548"
        and finite["receipt_path"]
        == "/engrfs/project/jacobsn/hiqbal/artifacts/legalrag/opd_math/fidelity/finite_state_f3a3222/receipt.json"
        and finite["tracked_receipt_path"]
        == "evidence/july_2026/opd_finite_state_108548.json",
        "finite-state receipt identity drifted",
    )
    _require(
        finite["receipt_sha256"] == sha256_file(FINITE_STATE_TRACKED_RECEIPT),
        "finite-state receipt bytes drifted",
    )
    finite_receipt = json.loads(FINITE_STATE_TRACKED_RECEIPT.read_text(encoding="utf-8"))
    _require(
        finite_receipt.get("schema_version") == 1
        and finite_receipt.get("check_id") == "opd_objective_finite_state_v1"
        and finite_receipt.get("status") == "passed"
        and finite_receipt.get("git_worktree_clean") is True
        and finite_receipt.get("objective_registry_sha256") == sha256_file(REGISTRY)
        and finite_receipt.get("script_sha256") == sha256_file(FINITE_STATE_SCRIPT)
        and finite_receipt.get("scientific_launch_authorized") is False,
        "finite-state receipt contract drifted",
    )
    expected_finite_cases = {
        "nan_student_logprob": "rejected",
        "posinf_student_logprob": "rejected",
        "neginf_student_logprob": "rejected",
        "nan_teacher_logprob": "rejected",
        "inf_behavior_logprob": "rejected",
        "nan_task_reward": "rejected",
        "finite_adamw_update": "passed",
        "nan_optimizer_state": "rejected",
        "inf_parameter": "rejected",
        "nan_gradient": "rejected",
    }
    observed_finite_cases = {
        item.get("case"): item.get("status")
        for item in finite_receipt.get("cases", [])
        if isinstance(item, dict)
    }
    _require(observed_finite_cases == expected_finite_cases, "finite-state cases drifted")
    finite_case = finite_receipt.get("finite_case")
    _require(isinstance(finite_case, dict), "finite-state positive case is missing")
    for field in ("gradient_norm_before_clip", "parameter_update_l2"):
        value = finite_case.get(field)
        _require(
            type(value) in (int, float) and math.isfinite(float(value)) and float(value) > 0,
            f"finite-state {field} is not finite and positive",
        )
    synthetic = levels["stored_synthetic"]
    _require(synthetic["status"] == "passed" and synthetic["slurm_job_id"] == "108501", "synthetic receipt drifted")
    _require(synthetic["fixture_sha256"] == sha256_file(SYNTHETIC_FIXTURE), "stored fixture drifted")
    _require(
        synthetic["receipt_sha256"]
        == "810ef012721d9555dd5dae5abf1c35989e6a5ca5327e63c4b0a41dc5e07cd601",
        "stored receipt hash drifted",
    )
    real = levels["stored_real_model"]
    _require(real["status"] == "pending" and real["accepted_fixture"] is None, "real-model status drifted")
    _require(
        "behavior_token_logprobs_from_generation_scores" in real["required_fields"],
        "real-model fixture permits reconstructed behavior scores",
    )
    full = levels["full_custody_one_step"]
    _require(
        full["status"] == "pending"
        and full["expected_runs"] == 12
        and full["local_runs"] == 10
        and full["upstream_verl_runs"] == 2,
        "full-custody matrix drifted",
    )
    _require(
        payload["stage_rules"]
        == {
            "all_levels_must_pass_before_preregistration": True,
            "diagnostic_outcomes_may_select_objectives": False,
            "diagnostics_may_inspect_source_holdout": False,
            "diagnostics_authorize_scientific_launch": False,
            "real_model_fixture_may_substitute_current_scoring_for_behavior_scores": False,
            "failed_fidelity_case_may_be_dropped": False,
        },
        "fidelity stage rules drifted",
    )
    return dict(payload)


def load_fidelity_plan(path: Path = DEFAULT_PLAN) -> dict[str, Any]:
    path = path.resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "fidelity plan must be a JSON object")
    result = validate_fidelity_plan(payload)
    result["path"] = str(path)
    result["sha256"] = sha256_file(path)
    return result
