#!/usr/bin/env python3
"""Outcome-blind held-out gate and terminal release for OPD objective-family v1.

This program deliberately lives outside the tracked experiment checkout.  Its
bytes and custody wrappers are sealed before any scientific student arm starts,
so the experiment can retain the exact d89ba3d code boundary while adding an
executable launch and release implementation for the already-declared 36-arm
analysis.

The program launches only preregistered, held scheduler jobs and validates and
seals:

* one program manifest before preregistration;
* one release plan after preregistration and before student launch;
* one post-training authorization and submission barrier for all held-out jobs;
* one held-out gate for each of the 36 preregistered arms; and
* one campaign-wide result bundle only after every arm is terminal.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
PROGRAM_ID = "opd_math_objective_family_release_program_v1"
PLAN_ID = "opd_math_objective_family_release_plan_v1"
ARM_GATE_ID = "opd_math_objective_family_heldout_gate_v1"
RESULT_ID = "opd_math_objective_family_terminal_readout_v1"
BUNDLE_ID = "opd_math_objective_family_terminal_bundle_v1"
EXPECTED_COMMIT = "d89ba3d7be728d9ee3197f37d8a8836a4a9640c5"
EXPECTED_O_PRIMARY_GATE = Path("/engrfs/project/jacobsn/hiqbal/artifacts/legalrag/opd_math/gates/teacher_gap/O_gap_d89ba3d_v1.json")
EXPECTED_O_INDEPENDENT_GATE = Path("/engrfs/project/jacobsn/hiqbal/artifacts/legalrag/opd_math/gates/teacher_gap/O_gap_d89ba3d_v1_independent.json")
EXPECTED_O_CHECKPOINT = Path("/engrfs/project/jacobsn/hiqbal/artifacts/legalrag/opd_math/teachers/O/merged_d89ba3d_v1")
EXPECTED_O_AUDIT_RECEIPT = Path("/engrfs/project/jacobsn/hiqbal/artifacts/legalrag/opd_math/audits/objective_family/O_teacher_d89ba3d_v1.json")
EXPECTED_O_ADAPTER = Path("/engrfs/project/jacobsn/hiqbal/artifacts/legalrag/opd_math/teachers/O/run_108609/final_adapter")
EXPECTED_O_RUN_MANIFEST = Path("/engrfs/project/jacobsn/hiqbal/artifacts/legalrag/opd_math/teachers/O/run_108609/run_manifest.json")
EXPECTED_STUDENT = "Qwen/Qwen3-1.7B"
EXPECTED_STUDENT_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
SOURCES = ("M", "O")
SEEDS = (0, 1, 2)
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 0
PRIMARY_COVERAGE = 0.9875
PRIMARY_LOWER_Q = 0.00625
PRIMARY_UPPER_Q = 0.99375
SECONDARY_LOWER_Q = 0.025
SECONDARY_UPPER_Q = 0.975
HELDOUT_DECODING = {
    "thinking": False,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,
    "max_new_tokens": 512,
    "seed": 0,
}
SELECTED_HOLDOUT_RECORDS = 370
EVALUATION_SHARDS = 6
EVALUATION_ARRAY_SPEC = "0-5%4"
EVALUATION_RUN_ID = "holdout370_6sh_evalseed0_attempt0"
PRIMARY_CONTRASTS = (
    ("task_rl_k1_ungated_clip5-minus-task_rl@M", "M", "task_rl_k1_ungated_clip5", "task_rl"),
    ("task_rl_k1_ungated_clip5-minus-task_rl@O", "O", "task_rl_k1_ungated_clip5", "task_rl"),
    (
        "task_rl_k1_gated_clip5_beta5-minus-task_rl_k1_ungated_clip5@M",
        "M",
        "task_rl_k1_gated_clip5_beta5",
        "task_rl_k1_ungated_clip5",
    ),
    (
        "task_rl_k1_gated_clip5_beta5-minus-task_rl_k1_ungated_clip5@O",
        "O",
        "task_rl_k1_gated_clip5_beta5",
        "task_rl_k1_ungated_clip5",
    ),
)
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
PROGRAM_CLAIM_BOUNDARY = (
    "This program implements the already-declared objective-family release. "
    "Its existence does not authorize training, held-out inspection, or an improvement claim."
)
TERMINAL_CLAIM_BOUNDARY = (
    "Scheduler terminality and training eligibility only; no held-out outcomes were read."
)
_PINNED_TOKENIZER: Any | None = None


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: str | Path, *, exclude: Iterable[str] = ()) -> str:
    raw_root = Path(path)
    expect(not raw_root.is_symlink(), f"tree root is a symlink: {raw_root}")
    root = raw_root.resolve()
    expect(root.is_dir(), f"tree is missing: {root}")
    excluded = {Path(item).as_posix() for item in exclude}
    digest = hashlib.sha256()
    digest.update(b"opd-math-tree-v1\0")
    files: list[tuple[str, Path]] = []
    for item in root.rglob("*"):
        mode = item.lstat().st_mode
        relative = item.relative_to(root).as_posix()
        expect(not stat.S_ISLNK(mode), f"tree contains symlink: {item}")
        expect(stat.S_ISREG(mode) or stat.S_ISDIR(mode), f"tree contains unbound special node: {item}")
        if stat.S_ISREG(mode) and relative not in excluded:
            files.append((relative, item))
    files.sort(key=lambda value: value[0])
    expect(files, f"tree is empty: {root}")
    for relative, item in files:
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def load_json(path: str | Path, label: str) -> dict[str, Any]:
    resolved = Path(path)
    expect(resolved.is_file() and not resolved.is_symlink(), f"{label} is not regular: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    expect(isinstance(payload, dict), f"{label} must be a JSON object")
    return payload


def load_jsonl(path: str | Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            expect(isinstance(value, dict), f"{label}:{line_number} is not an object")
            rows.append(value)
    expect(rows, f"{label} is empty")
    return rows


def regular_readonly(path: str | Path, label: str) -> Path:
    raw = Path(path)
    expect(raw.is_file() and not raw.is_symlink(), f"{label} must be a regular file")
    expect(
        raw.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH) == 0,
        f"{label} must be sealed read-only",
    )
    return raw.resolve()


def wait_for_regular_readonly(path: str | Path, label: str, *, attempts: int = 900) -> Path:
    for attempt in range(attempts):
        try:
            return regular_readonly(path, label)
        except ValueError:
            if attempt + 1 == attempts:
                raise
            time.sleep(1)
    raise AssertionError("unreachable")


def file_binding(path: str | Path, *, readonly: bool = True) -> dict[str, str]:
    resolved = regular_readonly(path, "bound file") if readonly else Path(path).resolve()
    expect(resolved.is_file() and not resolved.is_symlink(), f"bound file is missing: {resolved}")
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def copy_new_readonly(source: str | Path, destination: str | Path) -> Path:
    source_path = regular_readonly(source, "copy source")
    destination_path = Path(destination).resolve()
    expect(not destination_path.exists() and not destination_path.is_symlink(), f"copy destination already exists: {destination_path}")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination_path.name}.partial.", dir=destination_path.parent
    )
    temporary = Path(temporary_name)
    try:
        with source_path.open("rb") as reader, os.fdopen(descriptor, "wb") as writer:
            shutil.copyfileobj(reader, writer)
            writer.flush()
            os.fsync(writer.fileno())
            os.fchmod(writer.fileno(), 0o444)
        expect(sha256_file(temporary) == sha256_file(source_path), "staged copy hash drifted")
        os.link(temporary, destination_path, follow_symlinks=False)
        directory_fd = os.open(destination_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return destination_path


def validate_binding(value: Any, label: str, *, readonly: bool = True) -> Path:
    expect(isinstance(value, dict) and set(value) == {"path", "sha256"}, f"{label} binding drifted")
    path = regular_readonly(value["path"], label) if readonly else Path(value["path"]).resolve()
    expect(sha256_file(path) == value["sha256"], f"{label} hash drifted")
    return path


def write_new(path: str | Path, payload: Mapping[str, Any] | str) -> Path:
    target = Path(path).resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to overwrite sealed artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    text = payload if isinstance(payload, str) else json.dumps(payload, indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.partial.", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
            os.fchmod(handle.fileno(), 0o444)
        # A same-filesystem hard link is an atomic no-clobber publication: a
        # complete staged inode becomes visible at the final name, or the call
        # fails if any artifact already owns that name.
        os.link(temporary, target, follow_symlinks=False)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def write_new_or_exact(path: str | Path, payload: Mapping[str, Any] | str) -> Path:
    """Create one immutable artifact, or accept an already-identical artifact.

    This is deliberately narrower than an overwrite: it makes controller
    commit phases resumable after interruption without permitting mutation of
    a sealed custody record.
    """

    target = Path(path).resolve()
    expected = payload if isinstance(payload, str) else json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if target.exists() or target.is_symlink():
        resolved = regular_readonly(target, "resumable sealed artifact")
        expect(resolved.read_text(encoding="utf-8") == expected, f"existing sealed artifact differs: {target}")
        return resolved
    return write_new(target, payload)


def repo_git_state(repo: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True, check=True, capture_output=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=repo,
        text=True,
        check=True,
        capture_output=True,
    ).stdout
    return {"commit": commit, "clean": not status.strip()}


def parse_utc(value: Any, label: str) -> datetime:
    expect(isinstance(value, str) and value.endswith("Z"), f"{label} must be UTC Z time")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError(f"{label} is invalid") from error
    expect(parsed.tzinfo is not None, f"{label} lacks timezone")
    return parsed


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def parse_slurm_time(value: Any, label: str) -> datetime:
    expect(isinstance(value, str) and value and value != "Unknown", f"{label} is missing")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{label} is invalid") from error
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo("America/Chicago"))
    return parsed.astimezone(timezone.utc)


def configure_repo(repo: str | Path) -> Path:
    resolved = Path(repo).resolve()
    expect((resolved / "scripts/opd/objective_family_preregistration.py").is_file(), "repo lacks objective-family implementation")
    expect(repo_git_state(resolved) == {"commit": EXPECTED_COMMIT, "clean": True}, "release requires clean d89ba3d checkout")
    if str(resolved) not in sys.path:
        sys.path.insert(0, str(resolved))
    return resolved


def pinned_tokenizer():
    global _PINNED_TOKENIZER
    if _PINNED_TOKENIZER is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            EXPECTED_STUDENT,
            revision=EXPECTED_STUDENT_REVISION,
            local_files_only=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        _PINNED_TOKENIZER = tokenizer
    return _PINNED_TOKENIZER


def adapter_delta_l2(initial_root: str | Path, final_root: str | Path) -> dict[str, Any]:
    from safetensors import safe_open

    initial_file = Path(initial_root).resolve() / "adapter_model.safetensors"
    final_file = Path(final_root).resolve() / "adapter_model.safetensors"
    expect(initial_file.is_file() and final_file.is_file(), "adapter delta lacks safetensors files")
    squared_l2 = 0.0
    tensors = 0
    elements = 0
    with safe_open(initial_file, framework="pt", device="cpu") as initial, safe_open(
        final_file, framework="pt", device="cpu"
    ) as final:
        expect(set(initial.keys()) == set(final.keys()), "adapter tensor keys drifted")
        for key in sorted(initial.keys()):
            left = initial.get_tensor(key).float()
            right = final.get_tensor(key).float()
            expect(tuple(left.shape) == tuple(right.shape), f"adapter tensor shape drifted: {key}")
            delta = right - left
            value = float(delta.square().sum().item())
            expect(math.isfinite(value) and value >= 0.0, f"adapter tensor delta is nonfinite: {key}")
            squared_l2 += value
            tensors += 1
            elements += int(delta.numel())
    expect(tensors > 0 and elements > 0 and squared_l2 > 0.0, "adapter did not change from initialization")
    return {"tensors": tensors, "elements": elements, "squared_l2": squared_l2, "delta_l2": math.sqrt(squared_l2)}


def percentile(values: Sequence[float], probability: float) -> float:
    expect(values, "percentile input is empty")
    expect(0.0 <= probability <= 1.0, "percentile probability is invalid")
    ordered = sorted(float(value) for value in values)
    expect(all(math.isfinite(value) for value in ordered), "percentile input is nonfinite")
    index = int(probability * (len(ordered) - 1))
    return ordered[index]


def interval(values: Sequence[float], lower: float, upper: float) -> list[float]:
    return [percentile(values, lower), percentile(values, upper)]


def effect_label(bounds: Sequence[float]) -> str:
    expect(len(bounds) == 2 and bounds[0] <= bounds[1], "invalid effect interval")
    if bounds[0] > 0.0:
        return "helps"
    if bounds[1] < 0.0:
        return "harms"
    return "inconclusive"


def arm_key(objective_id: str, source: str, seed: int) -> str:
    return f"{objective_id}__{source}__seed{seed}"


def vector_mean(values: Sequence[float], indices: Sequence[int]) -> float:
    expect(indices, "bootstrap selected no records")
    return sum(values[index] for index in indices) / len(indices)


def contrast_result(
    *,
    name: str,
    formula: str,
    estimate: float,
    estimate_bounds: Sequence[float],
    point_draws: Sequence[float],
    lower_draws: Sequence[float],
    upper_draws: Sequence[float],
    confirmatory: bool,
) -> dict[str, Any]:
    lower_q, upper_q = (
        (PRIMARY_LOWER_Q, PRIMARY_UPPER_Q)
        if confirmatory
        else (SECONDARY_LOWER_Q, SECONDARY_UPPER_Q)
    )
    point_interval = interval(point_draws, lower_q, upper_q)
    pessimistic = interval(lower_draws, lower_q, upper_q)
    optimistic = interval(upper_draws, lower_q, upper_q)
    robust = [pessimistic[0], optimistic[1]]
    point_direction = effect_label(point_interval)
    robust_direction = effect_label(robust)
    return {
        "name": name,
        "formula": formula,
        "estimate": estimate,
        "estimate_bounds_under_verifier_uncertainty": list(estimate_bounds),
        "interval": point_interval,
        "interval_coverage": PRIMARY_COVERAGE if confirmatory else 0.95,
        "interval_quantiles": [lower_q, upper_q],
        "classification_without_verifier_uncertainty": (
            point_direction if confirmatory else None
        ),
        "classification": robust_direction if confirmatory else None,
        "verifier_uncertainty": {
            "policy": "binary_worst_case_hierarchical_bootstrap_envelope_v1",
            "pessimistic_interval": pessimistic,
            "optimistic_interval": optimistic,
            "robust_envelope": robust,
        },
        "confirmatory": confirmatory,
    }


def expected_analysis_contract() -> dict[str, Any]:
    return {
        "primary_contrasts": [name for name, _, _, _ in PRIMARY_CONTRASTS],
        "bootstrap": "paired_hierarchical_seed_then_record",
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "seed_sampling": "sample_three_of_[0,1,2]_with_replacement_per_draw",
        "record_sampling": "sample_one_M_and_one_O_record_vector_with_replacement_per_draw",
        "pairing": "reuse_each_source_record_vector_across_all_sampled_seeds_objectives_and_contrasts",
        "crossed_design": "same_heldout_records_are_crossed_with_all_three_training_seeds",
        "point_estimate": "equal_weight_mean_of_three_seed_specific_record_means",
        "multiplicity": "bonferroni_four_co_primary",
        "familywise_alpha": 0.05,
        "primary_interval_coverage": PRIMARY_COVERAGE,
        "primary_interval_quantiles": [PRIMARY_LOWER_Q, PRIMARY_UPPER_Q],
        "secondary_interval_coverage": 0.95,
        "secondary_interval_quantiles": [SECONDARY_LOWER_Q, SECONDARY_UPPER_Q],
        "verifier_uncertainty": "binary_worst_case_hierarchical_bootstrap_envelope_v1",
        "inspect_heldout_only_after_all_training_terminal": True,
        "release_only_after_all_36_gates_sealed": True,
    }


def expected_heldout_contract() -> dict[str, Any]:
    return {
        "role": "source_holdout",
        "selected_records_per_source": SELECTED_HOLDOUT_RECORDS,
        "selection": "first_370_records_in_registered_source_holdout",
        "samples_per_problem": 4,
        "decoding": HELDOUT_DECODING,
        "same_record_set_required_within_source": True,
        "evaluation_git_commit": EXPECTED_COMMIT,
        "raw_student_auxiliary_evaluation_required": True,
        "raw_student_auxiliary_is_secondary_only": True,
        "shards": EVALUATION_SHARDS,
        "array_spec": EVALUATION_ARRAY_SPEC,
        "shard_strategy": "contiguous_balanced_v1",
        "merge_strategy": "ordered_contiguous_shards_v1",
    }


def expected_terminal_policy() -> dict[str, Any]:
    return {
        "all_36_training_arms_terminal_before_any_heldout_launch": True,
        "completed_arm_may_be_gated": True,
        "failed_arm_has_no_replacement_seed": True,
        "failed_arm_is_retained_as_terminal_missing_outcome": True,
        "all_36_gate_paths_must_exist_before_outcome_release": True,
        "failed_training_arm_uses_failure_gate_not_model_evaluation": True,
        "no_objective_dropping": True,
        "no_rescue_training": True,
    }


def expected_m_teacher_boundary() -> dict[str, Any]:
    return {
        "m_teacher_permanently_excluded": True,
        "m_retraining_allowed": False,
        "m_merge_allowed": False,
        "m_m_allowed": False,
        "m_o_allowed": False,
        "math_student_and_evaluation_use_allowed": True,
    }


def expected_release_sequence() -> list[str]:
    return [
        "authorize_each_training_arm_before_held_submission",
        "record_each_held_scheduler_submission_then_release_job",
        "consume_external_authorization_inside_job_before_d89_launcher",
        "all_36_training_runs_terminal",
        "deep_audit_each_training_arm_once_and_seal_terminal_snapshot_without_heldout_outcomes",
        "seal_one_all_target_evaluation_authorization_before_any_heldout_release",
        "submit_all_valid_evaluation_chains_held_and_seal_one_submission_index",
        "release_the_globally_serialized_evaluation_wave_only_after_the_index_is_sealed_and_record_observed_release_result",
        "run_afterany_supervisors_and_terminalize_every_evaluation_target",
        "seal_global_evaluation_wave_and_finalizer_receipt_without_releasing_numeric_results",
        "seal_all_arm_and_raw_student_gates",
        "release_campaign_bundle_once_only",
    ]


def evaluation_target_paths(
    *,
    evaluation_root: Path,
    control_root: Path,
    campaign_id: str,
    target_key: str,
) -> dict[str, Any]:
    label = f"{campaign_id}--{target_key}"
    artifact_root = (
        evaluation_root / "source_holdout" / label / EVALUATION_RUN_ID
    ).resolve()
    custody_root = (control_root / "evaluation" / target_key).resolve()
    return {
        "label": label,
        "run_id": EVALUATION_RUN_ID,
        "artifact_root": str(artifact_root),
        "evaluation_summary": str((artifact_root / "merged/summary.json").resolve()),
        "evaluation_samples": str((artifact_root / "merged/samples.jsonl").resolve()),
        "evaluation_companion": str((artifact_root / "merged.custody.json").resolve()),
        "evaluation_authorization": str((custody_root / "authorization.json").resolve()),
        "evaluation_submission_receipt": str((custody_root / "submission.json").resolve()),
        "evaluation_merge_consumption_receipt": str((custody_root / "merge_consumption.json").resolve()),
        "evaluation_merge_supervisor_receipt": str((custody_root / "merge_supervisor.json").resolve()),
        "evaluation_seal_supervisor_receipt": str((custody_root / "seal_supervisor.json").resolve()),
        "evaluation_terminal_failure_receipt": str((custody_root / "terminal_failure.json").resolve()),
        "evaluation_array_accounting_receipt": str((custody_root / "array_accounting.json").resolve()),
        "evaluation_array_accounting_raw": str((custody_root / "array.sacct.txt").resolve()),
        "evaluation_consumption_receipt": str((custody_root / "consumption.json").resolve()),
        "evaluation_seal_receipt": str((custody_root / "seal.json").resolve()),
        "evaluation_shard_consumption_root": str((custody_root / "shard_consumption").resolve()),
        "evaluation_log_root": str((custody_root / "slurm").resolve()),
        "evaluation_private_log_root": str((custody_root / "private_logs").resolve()),
        "shards": EVALUATION_SHARDS,
        "array_spec": EVALUATION_ARRAY_SPEC,
    }


def tracked_contract(repo: Path) -> dict[str, Any]:
    configure_repo(repo)
    from scripts.opd.objective_family_preregistration import (  # type: ignore
        EXPECTED_ARM_KEYS,
        EXPECTED_DIAGNOSTIC_KEYS,
    )
    from scripts.opd.objective_registry import load_objective_registry  # type: ignore

    registry = load_objective_registry()
    objective_ids = [row["id"] for row in registry["objectives"]]
    expect(len(objective_ids) == 6 and len(set(objective_ids)) == 6, "objective registry is not six unique objectives")
    expect(len(EXPECTED_ARM_KEYS) == 36 and len(set(EXPECTED_ARM_KEYS)) == 36, "tracked arm set is not exactly 36")
    return {
        "objective_registry": {
            "path": registry["path"],
            "sha256": registry["sha256"],
        },
        "objective_ids": objective_ids,
        "arm_keys": list(EXPECTED_ARM_KEYS),
        "diagnostic_keys": list(EXPECTED_DIAGNOSTIC_KEYS),
    }


O_TEACHER_AUDIT_CLAIM_BOUNDARY = (
    "Teacher gate and merged-checkpoint custody only; no OPD student-performance result."
)


def _validate_scheduler_attestation(value: Any, label: str, repo: Path) -> dict[str, Any]:
    expected = {
        "job_id", "job_name", "state", "state_raw", "exit_code", "elapsed_seconds",
        "submit", "start", "end", "alloc_tres", "req_tres", "stdout_template", "stdout",
        "workdir", "submit_line", "partition", "account", "time_limit_minutes", "ncpus",
        "nnodes", "sacct_raw_sha256",
    }
    expect(isinstance(value, dict) and set(value) == expected, f"{label} scheduler schema drifted")
    expect(
        isinstance(value.get("job_id"), str)
        and value["job_id"].isdigit()
        and value.get("state") == "COMPLETED"
        and value.get("exit_code") == "0:0"
        and isinstance(value.get("elapsed_seconds"), int)
        and value["elapsed_seconds"] >= 0,
        f"{label} scheduler terminal identity drifted",
    )
    for field in ("submit", "start", "end"):
        parse_slurm_time(value.get(field), f"{label} scheduler {field}")
    expect(
        value.get("workdir") == str(repo.resolve())
        and value.get("partition") == "general-cpu"
        and value.get("account") == "engr-lab-jacobsn"
        and isinstance(value.get("sacct_raw_sha256"), str)
        and HEX64.fullmatch(value["sacct_raw_sha256"]),
        f"{label} scheduler lane drifted",
    )
    return dict(value)


def _assert_tree_readonly(root: Path, label: str) -> None:
    expect(root.is_dir() and not root.is_symlink(), f"{label} tree is missing")
    for path in (root, *root.rglob("*")):
        mode = path.lstat().st_mode
        expect(not stat.S_ISLNK(mode), f"{label} tree contains symlink: {path}")
        expect(
            stat.S_ISREG(mode) or stat.S_ISDIR(mode),
            f"{label} tree contains unbound special node: {path}",
        )
        expect(
            mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH) == 0,
            f"{label} tree contains writable path: {path}",
        )


def validate_o_teacher_audit_receipt_sealed(
    path: str | Path, repo: Path
) -> tuple[Path, dict[str, Any]]:
    configure_repo(repo)
    resolved = regular_readonly(path, "O teacher audit receipt")
    expect(resolved == EXPECTED_O_AUDIT_RECEIPT, "O teacher audit receipt path differs from frozen boundary")
    payload = load_json(resolved, "O teacher audit receipt")
    expected = {
        "schema_version", "receipt", "status", "created_utc", "git", "auditor",
        "primary_gate", "independent_gate", "gates_byte_identical",
        "gates_distinct_paths_inodes_jobs", "gates_recomputed_exactly", "teacher_identity",
        "checkpoint_tree_hash_independently_reproduced", "strong_teacher_provenance_validator_passed",
        "tracked_teacher_validator_passed", "merge_scheduler_terminal",
        "merge_submitted_after_both_gates_completed", "merge_stdout",
        "checkpoint_sealed_read_only", "all_bound_gate_and_stdout_artifacts_sealed_read_only",
        "stable_custody_revalidated_before_publication", "heldout_student_outcomes_inspected",
        "claim_boundary",
    }
    expect(set(payload) == expected, "O teacher audit receipt schema drifted")
    for field, value in (
        ("schema_version", 1),
        ("receipt", "opd_math_objective_family_o_teacher_independent_audit_v1"),
        ("status", "passed_and_sealed"),
        ("gates_byte_identical", True),
        ("gates_distinct_paths_inodes_jobs", True),
        ("gates_recomputed_exactly", True),
        ("checkpoint_tree_hash_independently_reproduced", True),
        ("strong_teacher_provenance_validator_passed", True),
        ("tracked_teacher_validator_passed", True),
        ("merge_submitted_after_both_gates_completed", True),
        ("checkpoint_sealed_read_only", True),
        ("all_bound_gate_and_stdout_artifacts_sealed_read_only", True),
        ("stable_custody_revalidated_before_publication", True),
        ("heldout_student_outcomes_inspected", False),
        ("claim_boundary", O_TEACHER_AUDIT_CLAIM_BOUNDARY),
    ):
        expect(payload.get(field) == value, f"O teacher audit {field} drifted")
    audit_created = parse_utc(payload.get("created_utc"), "O teacher audit created_utc")
    expect(payload.get("git") == {"commit": EXPECTED_COMMIT, "tracked_clean": True}, "O teacher audit Git custody drifted")
    auditor = payload.get("auditor")
    expect(isinstance(auditor, dict) and set(auditor) == {"path", "sha256"}, "O teacher auditor binding drifted")
    auditor_path = validate_binding(auditor, "O teacher auditor")
    expect(auditor_path == Path(__file__).with_name("audit_o_teacher.py").resolve(), "O teacher audit used an untrusted auditor")
    gates: list[tuple[Path, dict[str, Any], dict[str, Any], Path]] = []
    for field, expected_gate_path in (
        ("primary_gate", EXPECTED_O_PRIMARY_GATE),
        ("independent_gate", EXPECTED_O_INDEPENDENT_GATE),
    ):
        binding = payload.get(field)
        expect(
            isinstance(binding, dict)
            and set(binding) == {"path", "sha256", "stdout", "scheduler"},
            f"{field} audit binding drifted",
        )
        gate_path = regular_readonly(binding["path"], field)
        expect(gate_path == expected_gate_path, f"{field} path differs from frozen boundary")
        expect(stat.S_IMODE(gate_path.lstat().st_mode) == 0o444, f"{field} mode is not exactly 0444")
        expect(sha256_file(gate_path) == binding["sha256"], f"{field} audit hash drifted")
        scheduler = _validate_scheduler_attestation(binding["scheduler"], field, repo)
        stdout_binding = binding["stdout"]
        expect(
            isinstance(stdout_binding, dict)
            and set(stdout_binding) == {"scheduler_path", "archive_path", "sha256"},
            f"{field} stdout binding drifted",
        )
        stdout = regular_readonly(stdout_binding["archive_path"], f"{field} archived stdout")
        expect(
            stdout_binding["scheduler_path"] == scheduler["stdout"]
            and sha256_file(stdout) == stdout_binding["sha256"],
            f"{field} stdout path/hash drifted",
        )
        expect(stat.S_IMODE(stdout.lstat().st_mode) == 0o444, f"{field} archived stdout mode drifted")
        expect(
            scheduler["job_name"] == "opd_math_gate"
            and scheduler["time_limit_minutes"] == 240
            and scheduler["ncpus"] == 2
            and scheduler["nnodes"] == 1,
            f"{field} scheduler resources drifted",
        )
        marker = f"PASS gate computation completed; inspect passed/strength before use: {gate_path}"
        expect(
            sum(line == marker for line in stdout.read_text(encoding="utf-8", errors="replace").splitlines()) == 1,
            f"{field} stdout marker drifted",
        )
        gate = load_json(gate_path, field)
        expect(
            Path(str(gate.get("trained_adapter"))).resolve() == EXPECTED_O_ADAPTER
            and Path(str(gate.get("teacher_run_manifest"))).resolve()
            == EXPECTED_O_RUN_MANIFEST,
            f"{field} is not bound to fresh teacher job 108609",
        )
        gates.append((gate_path, gate, scheduler, stdout))
    expect(gates[0][0] != gates[1][0], "O teacher audit gate paths collapsed")
    expect(
        (gates[0][0].stat().st_dev, gates[0][0].stat().st_ino)
        != (gates[1][0].stat().st_dev, gates[1][0].stat().st_ino),
        "O teacher audit gate inodes collapsed",
    )
    expect(gates[0][0].read_bytes() == gates[1][0].read_bytes(), "O teacher audit gate bytes diverged")
    expect(gates[0][2]["job_id"] != gates[1][2]["job_id"], "O teacher audit gate jobs collapsed")
    from scripts.opd_math.quality_gates import recompute_teacher_gate  # type: ignore
    for label, (_, gate, _, _) in zip(("primary", "independent"), gates):
        original = dict(gate)
        original.pop("manifest_sha256", None)
        expect(recompute_teacher_gate(original) == original, f"{label} O teacher gate no longer recomputes")
    merge_scheduler = _validate_scheduler_attestation(payload.get("merge_scheduler_terminal"), "O teacher merge", repo)
    expect(
        merge_scheduler.get("job_name") == "opd_math_merge"
        and merge_scheduler["time_limit_minutes"] == 120
        and merge_scheduler["ncpus"] == 8
        and merge_scheduler["nnodes"] == 1,
        "O teacher merge scheduler resources drifted",
    )
    latest_gate_end = max(
        parse_slurm_time(gates[0][2]["end"], "primary O gate end"),
        parse_slurm_time(gates[1][2]["end"], "independent O gate end"),
    )
    expect(
        latest_gate_end <= parse_slurm_time(merge_scheduler["submit"], "O teacher merge submit"),
        "O teacher merge predates an audited gate",
    )
    expect(
        parse_slurm_time(merge_scheduler["end"], "O teacher merge end") <= audit_created,
        "O teacher audit predates merge completion",
    )
    merge_stdout = payload.get("merge_stdout")
    expect(
        isinstance(merge_stdout, dict)
        and set(merge_stdout) == {"scheduler_path", "archive_path", "sha256"},
        "O teacher merge stdout binding drifted",
    )
    merge_stdout_path = regular_readonly(merge_stdout["archive_path"], "O teacher archived merge stdout")
    expect(
        sha256_file(merge_stdout_path) == merge_stdout["sha256"]
        and merge_stdout["scheduler_path"] == merge_scheduler["stdout"],
        "O teacher merge stdout drifted",
    )
    expect(stat.S_IMODE(merge_stdout_path.lstat().st_mode) == 0o444, "O teacher archived merge stdout mode drifted")
    stdout_inodes = {
        (path.stat().st_dev, path.stat().st_ino)
        for path in (gates[0][3], gates[1][3], merge_stdout_path)
    }
    expect(len(stdout_inodes) == 3, "O teacher audit stdout inodes collapsed")
    archive_parents = {path.parent for path in (gates[0][3], gates[1][3], merge_stdout_path)}
    expect(len(archive_parents) == 1, "O teacher audit stdout archives are not colocated")
    archive_root = next(iter(archive_parents))
    expected_archive_root = resolved.with_name(resolved.name + ".logs")
    expect(archive_root == expected_archive_root, "O teacher audit stdout archive root drifted")
    expect(
        gates[0][3] == expected_archive_root / "primary_gate_stdout.log"
        and gates[1][3] == expected_archive_root / "independent_gate_stdout.log"
        and merge_stdout_path == expected_archive_root / "merge_stdout.log",
        "O teacher audit stdout archive filenames drifted",
    )
    expect(
        archive_root.is_dir()
        and not archive_root.is_symlink()
        and stat.S_IMODE(archive_root.lstat().st_mode) == 0o555,
        "O teacher audit stdout archive root is not sealed",
    )
    identity = payload.get("teacher_identity")
    expect(isinstance(identity, dict), "O teacher audit lacks teacher identity")
    primary_gate_path, primary_gate, _, _ = gates[0]
    expect(
        identity.get("teacher_gap_manifest") == str(primary_gate_path)
        and identity.get("teacher_gap_manifest_sha256") == sha256_file(primary_gate_path)
        and identity.get("teacher_gap_payload_sha256") == canonical_json_sha256(primary_gate),
        "O teacher identity is not bound to the audited primary gate",
    )
    checkpoint = Path(str(identity.get("merged_checkpoint")))
    expect(not checkpoint.is_symlink(), "audited O checkpoint may not be a symlink")
    checkpoint = checkpoint.resolve()
    expect(checkpoint == EXPECTED_O_CHECKPOINT, "audited O checkpoint path differs from frozen boundary")
    _assert_tree_readonly(checkpoint, "audited O checkpoint")
    marker = f"PASS scientifically gated teacher merge: {checkpoint}"
    expect(sum(line == marker for line in merge_stdout_path.read_text(encoding="utf-8", errors="replace").splitlines()) == 1, "O teacher merge stdout marker drifted")
    return resolved, payload


def validate_o_teacher_audit_receipt_full(
    path: str | Path, repo: Path
) -> tuple[Path, dict[str, Any]]:
    resolved, payload = validate_o_teacher_audit_receipt_sealed(path, repo)
    configure_repo(repo)
    from scripts.opd.objective_family_preregistration import _validate_teacher  # type: ignore
    from scripts.opd.opd_train import _validate_teacher_provenance  # type: ignore

    auditor_path = validate_binding(payload["auditor"], "O teacher auditor")
    spec = importlib.util.spec_from_file_location("sealed_o_teacher_auditor", auditor_path)
    expect(spec is not None and spec.loader is not None, "cannot load sealed O teacher auditor")
    auditor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auditor)
    for field in ("primary_gate", "independent_gate"):
        binding = payload[field]
        gate_path = regular_readonly(binding["path"], field)
        gate = load_json(gate_path, field)
        stdout_path = regular_readonly(binding["stdout"]["scheduler_path"], f"{field} scheduler stdout")
        expect(
            sha256_file(stdout_path) == binding["stdout"]["sha256"]
            and stat.S_IMODE(stdout_path.lstat().st_mode) == 0o444,
            f"{field} live scheduler stdout differs from archived audit bytes",
        )
        observed = auditor.query_terminal_job(
            job_id=binding["scheduler"]["job_id"],
            stdout_path=stdout_path,
            expected_name="opd_math_gate",
            stdout_template=auditor.EXPECTED_GATE_STDOUT,
            repo=repo,
            launcher=auditor.GATE_LAUNCHER,
            expected_exports=auditor.expected_gate_exports(gate, gate_path, repo),
            expected_cpu=2,
            expected_mem="8G",
            expected_minutes=240,
        )
        expect(observed == binding["scheduler"], f"{field} live scheduler evidence differs from audit receipt")
    identity = payload["teacher_identity"]
    expect(_validate_teacher(identity, commit=EXPECTED_COMMIT) == identity, "audited O teacher identity failed tracked validation")
    checkpoint = Path(identity["merged_checkpoint"]).resolve()
    expect(
        sha256_tree(checkpoint, exclude=("merge_provenance.json",))
        == identity["merged_checkpoint_tree_sha256"],
        "audited O teacher checkpoint differs from independent tree hash",
    )
    primary_binding = payload["primary_gate"]
    primary_path = regular_readonly(primary_binding["path"], "primary O gate")
    primary = load_json(primary_path, "primary O gate")
    adapter = Path(primary["trained_adapter"]).resolve()
    merge_stdout = regular_readonly(payload["merge_stdout"]["scheduler_path"], "O teacher scheduler merge stdout")
    expect(
        sha256_file(merge_stdout) == payload["merge_stdout"]["sha256"]
        and stat.S_IMODE(merge_stdout.lstat().st_mode) == 0o444,
        "live O teacher merge stdout differs from archived audit bytes",
    )
    observed_merge = auditor.query_terminal_job(
        job_id=payload["merge_scheduler_terminal"]["job_id"],
        stdout_path=merge_stdout,
        expected_name="opd_math_merge",
        stdout_template=auditor.EXPECTED_MERGE_STDOUT,
        repo=repo,
        launcher=auditor.MERGE_LAUNCHER,
        expected_exports=auditor.expected_merge_exports(
            repo=repo,
            gate_path=primary_path,
            adapter=adapter,
            checkpoint=checkpoint,
        ),
        expected_cpu=8,
        expected_mem="64G",
        expected_minutes=120,
    )
    expect(observed_merge == payload["merge_scheduler_terminal"], "live merge scheduler evidence differs from audit receipt")
    gate_for_provenance = dict(primary)
    gate_for_provenance["manifest_sha256"] = sha256_file(primary_path)
    strong = _validate_teacher_provenance(
        str(checkpoint / "merge_provenance.json"),
        gate_for_provenance,
        SimpleNamespace(
            teacher_base_model=identity["base_model"],
            teacher_base_revision=identity["base_revision"],
            teacher_checkpoint=str(checkpoint),
        ),
    )
    expect(
        strong.get("manifest_sha256") == identity["merge_provenance_manifest_sha256"],
        "full O teacher validation disagrees with merge provenance hash",
    )
    return resolved, payload


def validate_o_teacher_release_lineage_values(
    *,
    audit_binding: Mapping[str, Any],
    program_audit_binding: Mapping[str, Any],
    plan_audit_binding: Mapping[str, Any],
    audit_teacher_identity: Mapping[str, Any],
    prereg_teacher_identity: Mapping[str, Any],
    audit_created: datetime,
    program_created: datetime,
    prereg_created: datetime,
    launch_created: datetime,
    release_created: datetime,
    student_outcomes_inspected: bool,
    heldout_outcomes_inspected: bool,
) -> None:
    expect(
        dict(audit_binding) == dict(program_audit_binding) == dict(plan_audit_binding),
        "O teacher audit binding is not identical across audit/program/release lineage",
    )
    expect(
        dict(audit_teacher_identity) == dict(prereg_teacher_identity),
        "O teacher audit identity differs from preregistration",
    )
    expect(
        audit_created <= program_created <= prereg_created <= launch_created <= release_created,
        "O teacher audit/program/preregistration/launch/release chronology drifted",
    )
    expect(not student_outcomes_inspected and not heldout_outcomes_inspected, "O teacher lineage was sealed after outcome inspection")


def create_program_manifest(repo: Path, output: Path, o_teacher_audit_receipt: Path) -> dict[str, Any]:
    repo = configure_repo(repo)
    contract = tracked_contract(repo)
    audit_path, audit = validate_o_teacher_audit_receipt_full(o_teacher_audit_receipt, repo)
    script = Path(__file__).resolve()
    expect(script.is_file() and not script.is_symlink(), "release program must be a regular file")
    evaluation_wrapper = script.with_name("objective_family_evaluation_job.sh")
    expect(evaluation_wrapper.is_file() and not evaluation_wrapper.is_symlink(), "evaluation custody wrapper must be a regular file")
    local_training_wrapper = script.with_name("objective_family_local_training_job.sh")
    upstream_training_wrapper = script.with_name("objective_family_upstream_training_job.sh")
    expect(local_training_wrapper.is_file() and not local_training_wrapper.is_symlink(), "local training custody wrapper must be a regular file")
    expect(upstream_training_wrapper.is_file() and not upstream_training_wrapper.is_symlink(), "upstream training custody wrapper must be a regular file")
    created_utc = utc_now()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "program": PROGRAM_ID,
        "status": "sealed_outcome_blind_before_student_launch",
        "created_utc": created_utc,
        "git_commit": EXPECTED_COMMIT,
        "student_model": EXPECTED_STUDENT,
        "student_revision": EXPECTED_STUDENT_REVISION,
        "sources": list(SOURCES),
        "seeds": list(SEEDS),
        "objective_registry": contract["objective_registry"],
        "objective_ids": contract["objective_ids"],
        "arm_keys": contract["arm_keys"],
        "program_file": file_binding(script, readonly=False),
        "evaluation_wrapper": file_binding(evaluation_wrapper, readonly=False),
        "local_training_wrapper": file_binding(local_training_wrapper, readonly=False),
        "upstream_training_wrapper": file_binding(upstream_training_wrapper, readonly=False),
        "o_teacher_audit_receipt": file_binding(audit_path),
        "analysis": expected_analysis_contract(),
        "heldout": expected_heldout_contract(),
        "terminal_policy": expected_terminal_policy(),
        "m_teacher_boundary": expected_m_teacher_boundary(),
        "claim_boundary": PROGRAM_CLAIM_BOUNDARY,
    }
    expect(
        parse_utc(audit["created_utc"], "O teacher audit created_utc") <= parse_utc(created_utc, "program created_utc"),
        "program manifest predates the O teacher audit",
    )
    write_new(output, payload)
    return payload


def validate_program_manifest(path: str | Path, repo: Path) -> tuple[Path, dict[str, Any]]:
    resolved = regular_readonly(path, "release program manifest")
    payload = load_json(resolved, "release program manifest")
    contract = tracked_contract(repo)
    expected_keys = {
        "schema_version",
        "program",
        "status",
        "created_utc",
        "git_commit",
        "student_model",
        "student_revision",
        "sources",
        "seeds",
        "objective_registry",
        "objective_ids",
        "arm_keys",
        "program_file",
        "evaluation_wrapper",
        "local_training_wrapper",
        "upstream_training_wrapper",
        "o_teacher_audit_receipt",
        "analysis",
        "heldout",
        "terminal_policy",
        "m_teacher_boundary",
        "claim_boundary",
    }
    expect(set(payload) == expected_keys, "release program manifest schema drifted")
    for field, expected in (
        ("schema_version", SCHEMA_VERSION),
        ("program", PROGRAM_ID),
        ("status", "sealed_outcome_blind_before_student_launch"),
        ("git_commit", EXPECTED_COMMIT),
        ("student_model", EXPECTED_STUDENT),
        ("student_revision", EXPECTED_STUDENT_REVISION),
        ("sources", list(SOURCES)),
        ("seeds", list(SEEDS)),
        ("objective_registry", contract["objective_registry"]),
        ("objective_ids", contract["objective_ids"]),
        ("arm_keys", contract["arm_keys"]),
        ("analysis", expected_analysis_contract()),
        ("heldout", expected_heldout_contract()),
        ("terminal_policy", expected_terminal_policy()),
        ("m_teacher_boundary", expected_m_teacher_boundary()),
        ("claim_boundary", PROGRAM_CLAIM_BOUNDARY),
    ):
        expect(payload.get(field) == expected, f"release program {field} drifted")
    parse_utc(payload.get("created_utc"), "release program created_utc")
    program_path = validate_binding(payload["program_file"], "release program", readonly=True)
    expect(program_path == Path(__file__).resolve(), "running release program differs from sealed program")
    wrapper_path = validate_binding(payload["evaluation_wrapper"], "evaluation custody wrapper", readonly=True)
    expect(wrapper_path == Path(__file__).with_name("objective_family_evaluation_job.sh").resolve(), "running evaluation wrapper differs from sealed program")
    local_wrapper_path = validate_binding(payload["local_training_wrapper"], "local training custody wrapper", readonly=True)
    expect(local_wrapper_path == Path(__file__).with_name("objective_family_local_training_job.sh").resolve(), "running local training wrapper differs from sealed program")
    upstream_wrapper_path = validate_binding(payload["upstream_training_wrapper"], "upstream training custody wrapper", readonly=True)
    expect(upstream_wrapper_path == Path(__file__).with_name("objective_family_upstream_training_job.sh").resolve(), "running upstream training wrapper differs from sealed program")
    audit_path = validate_binding(payload["o_teacher_audit_receipt"], "O teacher audit receipt")
    _, audit = validate_o_teacher_audit_receipt_sealed(audit_path, repo)
    expect(
        parse_utc(audit["created_utc"], "O teacher audit created_utc")
        <= parse_utc(payload["created_utc"], "release program created_utc"),
        "release program predates O teacher audit",
    )
    return resolved, payload


def create_release_plan(args: argparse.Namespace) -> dict[str, Any]:
    repo = configure_repo(args.repo)
    from scripts.opd.objective_family_preregistration import (  # type: ignore
        validate_launch_plan,
        validate_preregistration,
    )

    program_path, program_payload = validate_program_manifest(args.program_manifest, repo)
    prereg = validate_preregistration(args.preregistration)
    launch = validate_launch_plan(args.launch_plan, preregistration=prereg)
    created_utc = utc_now()
    release_created = parse_utc(created_utc, "release plan created_utc")
    audit_path = validate_binding(program_payload["o_teacher_audit_receipt"], "O teacher audit receipt")
    _, audit = validate_o_teacher_audit_receipt_sealed(audit_path, repo)
    expect(audit["teacher_identity"] == prereg["o_teacher"], "preregistration O teacher differs from the independently audited teacher")
    audit_created = parse_utc(audit["created_utc"], "O teacher audit created_utc")
    program_created = parse_utc(program_payload["created_utc"], "program created_utc")
    prereg_created = parse_utc(prereg["payload"]["created_utc"], "preregistration created_utc")
    launch_created = parse_utc(launch["payload"]["created_utc"], "launch plan created_utc")
    validate_o_teacher_release_lineage_values(
        audit_binding=program_payload["o_teacher_audit_receipt"],
        program_audit_binding=program_payload["o_teacher_audit_receipt"],
        plan_audit_binding=program_payload["o_teacher_audit_receipt"],
        audit_teacher_identity=audit["teacher_identity"],
        prereg_teacher_identity=prereg["o_teacher"],
        audit_created=audit_created,
        program_created=program_created,
        prereg_created=prereg_created,
        launch_created=launch_created,
        release_created=release_created,
        student_outcomes_inspected=False,
        heldout_outcomes_inspected=False,
    )
    expect(
        audit_created <= program_created <= prereg_created <= launch_created <= release_created,
        "release chronology must be audit <= program <= preregistration <= launch plan <= release plan",
    )
    expect(prereg["commit"] == EXPECTED_COMMIT, "release plan preregistration commit drifted")
    evaluation_root = Path(args.evaluation_root).resolve()
    result_root = Path(args.result_root).resolve()
    control_root = Path(args.control_root).resolve()
    train_environment_root = Path(args.train_environment_root).resolve()
    hf_home = Path(args.hf_home).resolve()
    expect(
        train_environment_root.is_dir()
        and not train_environment_root.is_symlink()
        and (train_environment_root / "bin/python").is_file(),
        "release train environment root is invalid",
    )
    expect(hf_home.is_dir() and not hf_home.is_symlink(), "release HF home is invalid")
    terminal_snapshot = Path(args.terminal_snapshot).resolve()
    expect(
        terminal_snapshot == (control_root / "terminal_training_snapshot.json").resolve(),
        "terminal snapshot must use the frozen control-root path",
    )
    expect(len({evaluation_root, result_root, control_root}) == 3, "release roots must be distinct")
    expect(
        evaluation_root.parent == result_root.parent == control_root.parent
        and evaluation_root.name == "evaluations"
        and result_root.name == "results"
        and control_root.name == "control",
        "release roots must be evaluations/results/control siblings under one campaign root",
    )
    prepared_path = Path(prereg["prepared_manifest"]["path"]).resolve()
    prepared = load_json(prepared_path, "prepared manifest")
    holdout_selection: dict[str, dict[str, Any]] = {}
    for source in SOURCES:
        relative = f"roles/{source}/source_holdout.jsonl"
        entry = (prepared.get("files") or {}).get(relative)
        expect(isinstance(entry, dict), f"prepared manifest lacks {relative}")
        task_path = (prepared_path.parent / relative).resolve()
        rows = load_jsonl(task_path, f"source holdout {source}")
        expect(
            len(rows) == entry.get("rows")
            and len(rows) >= SELECTED_HOLDOUT_RECORDS
            and entry.get("sha256") == sha256_file(task_path),
            f"source holdout physical bytes/rows drifted: {source}",
        )
        selected_ids = [row.get("record_id") for row in rows[:SELECTED_HOLDOUT_RECORDS]]
        expect(all(isinstance(value, str) and value for value in selected_ids), f"source holdout prefix IDs invalid: {source}")
        expect(len(set(selected_ids)) == SELECTED_HOLDOUT_RECORDS, f"source holdout prefix IDs duplicate: {source}")
        holdout_selection[source] = {
            "task_file": str(task_path),
            "task_file_sha256": sha256_file(task_path),
            "physical_rows": len(rows),
            "selected_records": SELECTED_HOLDOUT_RECORDS,
            "selected_record_ids_sha256": canonical_json_sha256(selected_ids),
            "selection": "first_370_records_in_registered_source_holdout",
        }
    arm_paths: dict[str, dict[str, Any]] = {}
    campaign_id = prereg["payload"]["campaign_id"]
    for key in prereg["payload"]["arm_keys"]:
        arm = prereg["payload"]["arms"][key]
        evaluation_paths = evaluation_target_paths(
            evaluation_root=evaluation_root,
            control_root=control_root,
            campaign_id=campaign_id,
            target_key=key,
        )
        arm_paths[key] = {
            "training_out": arm["training_out"],
            "prelaunch_receipt": arm["prelaunch_receipt"],
            "heldout_gate": arm["heldout_gate"],
            "training_authorization": str((control_root / "training" / key / "authorization.json").resolve()),
            "submission_receipt": str((control_root / "training" / key / "submission.json").resolve()),
            "training_consumption_receipt": str((control_root / "training" / key / "consumption.json").resolve()),
            "terminal_audit_receipt": str((control_root / "training_audits" / f"{key}.json").resolve()),
            **evaluation_paths,
        }
    raw_student = {}
    for source in SOURCES:
        target_key = f"raw_student__{source}"
        evaluation_paths = evaluation_target_paths(
            evaluation_root=evaluation_root,
            control_root=control_root,
            campaign_id=campaign_id,
            target_key=target_key,
        )
        raw_student[source] = {
            "summary": evaluation_paths.pop("evaluation_summary"),
            "samples": evaluation_paths.pop("evaluation_samples"),
            "companion": evaluation_paths.pop("evaluation_companion"),
            "authorization": evaluation_paths.pop("evaluation_authorization"),
            "submission_receipt": evaluation_paths.pop("evaluation_submission_receipt"),
            "merge_consumption_receipt": evaluation_paths.pop("evaluation_merge_consumption_receipt"),
            "merge_supervisor_receipt": evaluation_paths.pop("evaluation_merge_supervisor_receipt"),
            "seal_supervisor_receipt": evaluation_paths.pop("evaluation_seal_supervisor_receipt"),
            "terminal_failure_receipt": evaluation_paths.pop("evaluation_terminal_failure_receipt"),
            "array_accounting_receipt": evaluation_paths.pop("evaluation_array_accounting_receipt"),
            "array_accounting_raw": evaluation_paths.pop("evaluation_array_accounting_raw"),
            "consumption_receipt": evaluation_paths.pop("evaluation_consumption_receipt"),
            "seal_receipt": evaluation_paths.pop("evaluation_seal_receipt"),
            "shard_consumption_root": evaluation_paths.pop("evaluation_shard_consumption_root"),
            "log_root": evaluation_paths.pop("evaluation_log_root"),
            "private_log_root": evaluation_paths.pop("evaluation_private_log_root"),
            "gate": str((result_root / "gates" / f"{target_key}.json").resolve()),
            **evaluation_paths,
        }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "release_plan": PLAN_ID,
        "status": "sealed_before_student_arm_launch",
        "created_utc": created_utc,
        "student_arm_outcomes_inspected_before_sealing": False,
        "heldout_outcomes_inspected_before_sealing": False,
        "program_manifest": file_binding(program_path),
        "o_teacher_audit_receipt": program_payload["o_teacher_audit_receipt"],
        "preregistration": {"path": prereg["path"], "sha256": prereg["sha256"]},
        "launch_plan": {"path": launch["path"], "sha256": launch["sha256"]},
        "campaign_id": campaign_id,
        "git_commit": EXPECTED_COMMIT,
        "evaluation_root": str(evaluation_root),
        "result_root": str(result_root),
        "control_root": str(control_root),
        "train_environment_root": str(train_environment_root),
        "hf_home": str(hf_home),
        "arm_paths": arm_paths,
        "raw_student_auxiliary": raw_student,
        "holdout_selection": holdout_selection,
        "terminal_snapshot": str(terminal_snapshot),
        "evaluation_wave_authorization": str((control_root / "evaluation_wave_authorization.json").resolve()),
        "evaluation_wave_submission_index": str((control_root / "evaluation_wave_submission_index.json").resolve()),
        "evaluation_wave_submission_journal_root": str((control_root / "evaluation_submission_journal").resolve()),
        "evaluation_wave_release_intent": str((control_root / "evaluation_wave_release_intent.json").resolve()),
        "evaluation_wave_release_result": str((control_root / "evaluation_wave_release_result.json").resolve()),
        "evaluation_wave_release_failure": str((control_root / "evaluation_wave_release_failure.json").resolve()),
        "evaluation_wave_submission_failure": str((control_root / "evaluation_wave_submission_failure.json").resolve()),
        "evaluation_wave_finalizer_receipt": str((control_root / "evaluation_wave_finalizer.json").resolve()),
        "evaluation_wave_finalizer_private_log": str((control_root / "evaluation_wave_finalizer.controller.private.log").resolve()),
        "evaluation_wave_seal": str((control_root / "evaluation_wave_seal.json").resolve()),
        "outputs": {
            "json": str((result_root / "readout.json").resolve()),
            "markdown": str((result_root / "readout.md").resolve()),
            "manifest": str((result_root / "bundle_manifest.json").resolve()),
        },
        "analysis": expected_analysis_contract(),
        "release_sequence": expected_release_sequence(),
    }
    planned_absent: list[Path] = [
        terminal_snapshot,
        Path(payload["evaluation_wave_authorization"]),
        Path(payload["evaluation_wave_submission_index"]),
        Path(payload["evaluation_wave_submission_journal_root"]),
        Path(payload["evaluation_wave_release_intent"]),
        Path(payload["evaluation_wave_release_result"]),
        Path(payload["evaluation_wave_release_failure"]),
        Path(payload["evaluation_wave_submission_failure"]),
        Path(payload["evaluation_wave_finalizer_receipt"]),
        Path(payload["evaluation_wave_finalizer_private_log"]),
        Path(payload["evaluation_wave_seal"]),
        *(Path(value) for value in payload["outputs"].values()),
    ]
    for item in arm_paths.values():
        planned_absent.extend(Path(item[field]) for field in (
            "training_out", "prelaunch_receipt", "heldout_gate",
            "artifact_root", "evaluation_summary", "evaluation_samples",
            "evaluation_companion", "evaluation_authorization",
            "evaluation_submission_receipt", "evaluation_merge_consumption_receipt",
            "evaluation_merge_supervisor_receipt", "evaluation_seal_supervisor_receipt",
            "evaluation_terminal_failure_receipt", "evaluation_array_accounting_receipt",
            "evaluation_array_accounting_raw",
            "evaluation_consumption_receipt",
            "evaluation_seal_receipt", "evaluation_shard_consumption_root",
            "evaluation_log_root", "evaluation_private_log_root",
            "training_authorization", "submission_receipt", "training_consumption_receipt",
            "terminal_audit_receipt",
        ))
    for item in raw_student.values():
        planned_absent.extend(Path(item[field]) for field in (
            "artifact_root", "summary", "samples", "companion", "authorization",
            "submission_receipt", "merge_consumption_receipt", "merge_supervisor_receipt",
            "seal_supervisor_receipt", "terminal_failure_receipt", "array_accounting_receipt",
            "array_accounting_raw", "consumption_receipt", "seal_receipt",
            "shard_consumption_root", "log_root", "private_log_root", "gate",
        ))
    runtime_train_freeze = (
        evaluation_root.parent / "environment_freezes" / EXPECTED_COMMIT / "train.freeze.txt"
    ).resolve()
    planned_absent.append(runtime_train_freeze)
    existing = sorted(str(path) for path in planned_absent if path.exists() or path.is_symlink())
    expect(not existing, f"release plan paths are not fresh: {existing[:10]}")
    source_train_freeze = validate_binding(prereg["environment_freezes"]["train"], "preregistered train freeze")
    copy_new_readonly(source_train_freeze, runtime_train_freeze)
    payload["runtime_train_freeze"] = {
        "source": file_binding(source_train_freeze),
        "runtime": file_binding(runtime_train_freeze),
    }
    write_new(args.output, payload)
    return payload


def validate_release_plan(path: str | Path, repo: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    from scripts.opd.objective_family_preregistration import (  # type: ignore
        validate_launch_plan,
        validate_preregistration,
    )

    resolved = regular_readonly(path, "release plan")
    payload = load_json(resolved, "release plan")
    expected_keys = {
        "schema_version", "release_plan", "status", "created_utc",
        "student_arm_outcomes_inspected_before_sealing",
        "heldout_outcomes_inspected_before_sealing", "program_manifest",
        "o_teacher_audit_receipt",
        "preregistration", "launch_plan", "campaign_id", "git_commit",
        "evaluation_root", "result_root", "control_root",
        "train_environment_root", "hf_home", "runtime_train_freeze",
        "arm_paths", "raw_student_auxiliary", "holdout_selection",
        "terminal_snapshot", "evaluation_wave_authorization",
        "evaluation_wave_submission_index", "evaluation_wave_submission_journal_root", "evaluation_wave_release_intent",
        "evaluation_wave_release_result", "evaluation_wave_release_failure",
        "evaluation_wave_submission_failure", "evaluation_wave_finalizer_receipt",
        "evaluation_wave_finalizer_private_log", "evaluation_wave_seal",
        "outputs", "analysis", "release_sequence",
    }
    expect(set(payload) == expected_keys, "release plan schema drifted")
    expect(payload.get("schema_version") == SCHEMA_VERSION and payload.get("release_plan") == PLAN_ID, "release plan identity drifted")
    expect(payload.get("status") == "sealed_before_student_arm_launch", "release plan status drifted")
    expect(payload.get("student_arm_outcomes_inspected_before_sealing") is False, "release plan admits student outcome inspection")
    expect(payload.get("heldout_outcomes_inspected_before_sealing") is False, "release plan admits heldout inspection")
    release_created = parse_utc(payload.get("created_utc"), "release plan created_utc")
    program_path = validate_binding(payload.get("program_manifest"), "program manifest")
    _, program_payload = validate_program_manifest(program_path, repo)
    expect(payload.get("o_teacher_audit_receipt") == program_payload["o_teacher_audit_receipt"], "release plan O teacher audit binding differs from program manifest")
    audit_path = validate_binding(payload.get("o_teacher_audit_receipt"), "O teacher audit receipt")
    _, audit = validate_o_teacher_audit_receipt_sealed(audit_path, repo)
    audit_created = parse_utc(audit["created_utc"], "O teacher audit created_utc")
    program_created = parse_utc(program_payload["created_utc"], "program created_utc")
    prereg_path = validate_binding(payload.get("preregistration"), "preregistration")
    prereg = validate_preregistration(prereg_path)
    expect(audit["teacher_identity"] == prereg["o_teacher"], "release plan preregistration teacher differs from O teacher audit")
    launch_path = validate_binding(payload.get("launch_plan"), "launch plan")
    launch = validate_launch_plan(launch_path, preregistration=prereg)
    prereg_created = parse_utc(prereg["payload"]["created_utc"], "preregistration created_utc")
    launch_created = parse_utc(launch["payload"]["created_utc"], "launch plan created_utc")
    validate_o_teacher_release_lineage_values(
        audit_binding=payload["o_teacher_audit_receipt"],
        program_audit_binding=program_payload["o_teacher_audit_receipt"],
        plan_audit_binding=payload["o_teacher_audit_receipt"],
        audit_teacher_identity=audit["teacher_identity"],
        prereg_teacher_identity=prereg["o_teacher"],
        audit_created=audit_created,
        program_created=program_created,
        prereg_created=prereg_created,
        launch_created=launch_created,
        release_created=release_created,
        student_outcomes_inspected=payload["student_arm_outcomes_inspected_before_sealing"],
        heldout_outcomes_inspected=payload["heldout_outcomes_inspected_before_sealing"],
    )
    expect(
        audit_created <= program_created <= prereg_created <= launch_created <= release_created,
        "release chronology drifted",
    )
    expect(payload.get("campaign_id") == prereg["payload"]["campaign_id"], "release plan campaign drifted")
    expect(payload.get("git_commit") == EXPECTED_COMMIT == prereg["commit"], "release plan commit drifted")
    expect(payload.get("analysis") == expected_analysis_contract(), "release analysis drifted")
    expect(payload.get("release_sequence") == expected_release_sequence(), "release sequence drifted")
    evaluation_root = Path(str(payload.get("evaluation_root"))).resolve()
    result_root = Path(str(payload.get("result_root"))).resolve()
    control_root = Path(str(payload.get("control_root"))).resolve()
    train_environment_root = Path(str(payload.get("train_environment_root"))).resolve()
    hf_home = Path(str(payload.get("hf_home"))).resolve()
    expect(
        train_environment_root.is_dir()
        and not train_environment_root.is_symlink()
        and (train_environment_root / "bin/python").is_file(),
        "release train environment root drifted",
    )
    expect(hf_home.is_dir() and not hf_home.is_symlink(), "release HF home drifted")
    for label, root in (("evaluation", evaluation_root), ("result", result_root), ("control", control_root)):
        expect(root.is_absolute(), f"release {label} root is not absolute")
    expect(
        evaluation_root.parent == result_root.parent == control_root.parent
        and evaluation_root.name == "evaluations"
        and result_root.name == "results"
        and control_root.name == "control",
        "release root topology drifted",
    )
    runtime_freeze = payload.get("runtime_train_freeze")
    expect(isinstance(runtime_freeze, dict) and set(runtime_freeze) == {"source", "runtime"}, "runtime train-freeze binding drifted")
    source_freeze_path = validate_binding(runtime_freeze["source"], "runtime train-freeze source")
    runtime_freeze_path = validate_binding(runtime_freeze["runtime"], "runtime train-freeze copy")
    expect(runtime_freeze["source"] == prereg["environment_freezes"]["train"], "runtime train-freeze source differs from preregistration")
    expect(
        runtime_freeze_path == (evaluation_root.parent / "environment_freezes" / EXPECTED_COMMIT / "train.freeze.txt").resolve()
        and sha256_file(runtime_freeze_path) == sha256_file(source_freeze_path),
        "runtime train-freeze copy drifted",
    )
    arm_paths = payload.get("arm_paths")
    expect(isinstance(arm_paths, dict) and set(arm_paths) == set(prereg["payload"]["arm_keys"]), "release arm path matrix drifted")
    for key, item in arm_paths.items():
        arm = prereg["payload"]["arms"][key]
        expected_item = {
            "training_out": arm["training_out"],
            "prelaunch_receipt": arm["prelaunch_receipt"],
            "heldout_gate": arm["heldout_gate"],
            "training_authorization": str((control_root / "training" / key / "authorization.json").resolve()),
            "submission_receipt": str((control_root / "training" / key / "submission.json").resolve()),
            "training_consumption_receipt": str((control_root / "training" / key / "consumption.json").resolve()),
            "terminal_audit_receipt": str((control_root / "training_audits" / f"{key}.json").resolve()),
            **evaluation_target_paths(
                evaluation_root=evaluation_root,
                control_root=control_root,
                campaign_id=payload["campaign_id"],
                target_key=key,
            ),
        }
        expect(item == expected_item, f"release arm paths/config drifted: {key}")
    raw = payload.get("raw_student_auxiliary")
    expect(isinstance(raw, dict) and set(raw) == set(SOURCES), "raw-student auxiliary matrix drifted")
    for source, item in raw.items():
        target_key = f"raw_student__{source}"
        paths = evaluation_target_paths(
            evaluation_root=evaluation_root,
            control_root=control_root,
            campaign_id=payload["campaign_id"],
            target_key=target_key,
        )
        expected_item = {
            "summary": paths.pop("evaluation_summary"),
            "samples": paths.pop("evaluation_samples"),
            "companion": paths.pop("evaluation_companion"),
            "authorization": paths.pop("evaluation_authorization"),
            "submission_receipt": paths.pop("evaluation_submission_receipt"),
            "merge_consumption_receipt": paths.pop("evaluation_merge_consumption_receipt"),
            "merge_supervisor_receipt": paths.pop("evaluation_merge_supervisor_receipt"),
            "seal_supervisor_receipt": paths.pop("evaluation_seal_supervisor_receipt"),
            "terminal_failure_receipt": paths.pop("evaluation_terminal_failure_receipt"),
            "array_accounting_receipt": paths.pop("evaluation_array_accounting_receipt"),
            "array_accounting_raw": paths.pop("evaluation_array_accounting_raw"),
            "consumption_receipt": paths.pop("evaluation_consumption_receipt"),
            "seal_receipt": paths.pop("evaluation_seal_receipt"),
            "shard_consumption_root": paths.pop("evaluation_shard_consumption_root"),
            "log_root": paths.pop("evaluation_log_root"),
            "private_log_root": paths.pop("evaluation_private_log_root"),
            "gate": str((result_root / "gates" / f"{target_key}.json").resolve()),
            **paths,
        }
        expect(item == expected_item, f"raw-student paths/config drifted: {source}")
    prepared_path = Path(prereg["prepared_manifest"]["path"]).resolve()
    prepared = load_json(prepared_path, "prepared manifest")
    expected_selection: dict[str, dict[str, Any]] = {}
    for source in SOURCES:
        relative = f"roles/{source}/source_holdout.jsonl"
        entry = (prepared.get("files") or {}).get(relative)
        expect(isinstance(entry, dict), f"prepared manifest lacks {relative}")
        task_path = (prepared_path.parent / relative).resolve()
        rows = load_jsonl(task_path, f"source holdout {source}")
        selected_ids = [row.get("record_id") for row in rows[:SELECTED_HOLDOUT_RECORDS]]
        expect(
            len(rows) == entry.get("rows")
            and len(rows) >= SELECTED_HOLDOUT_RECORDS
            and entry.get("sha256") == sha256_file(task_path),
            f"release holdout physical bytes/rows drifted: {source}",
        )
        expect(
            all(isinstance(value, str) and value for value in selected_ids)
            and len(set(selected_ids)) == SELECTED_HOLDOUT_RECORDS,
            f"release holdout selected IDs drifted: {source}",
        )
        expected_selection[source] = {
            "task_file": str(task_path),
            "task_file_sha256": sha256_file(task_path),
            "physical_rows": len(rows),
            "selected_records": SELECTED_HOLDOUT_RECORDS,
            "selected_record_ids_sha256": canonical_json_sha256(selected_ids),
            "selection": "first_370_records_in_registered_source_holdout",
        }
    expect(payload.get("holdout_selection") == expected_selection, "release holdout selection drifted")
    terminal_snapshot = Path(payload.get("terminal_snapshot", "")).resolve()
    expect(terminal_snapshot.is_absolute(), "terminal snapshot path is not absolute")
    expect(
        terminal_snapshot == (control_root / "terminal_training_snapshot.json").resolve(),
        "terminal snapshot is not at the frozen control-root path",
    )
    expect(
        Path(payload.get("evaluation_wave_authorization", "")).resolve()
        == (control_root / "evaluation_wave_authorization.json").resolve(),
        "evaluation-wave authorization path drifted",
    )
    expect(
        Path(payload.get("evaluation_wave_submission_index", "")).resolve()
        == (control_root / "evaluation_wave_submission_index.json").resolve(),
        "evaluation-wave submission-index path drifted",
    )
    expect(
        Path(payload.get("evaluation_wave_submission_journal_root", "")).resolve()
        == (control_root / "evaluation_submission_journal").resolve(),
        "evaluation-wave submission-journal path drifted",
    )
    expect(
        Path(payload.get("evaluation_wave_release_intent", "")).resolve()
        == (control_root / "evaluation_wave_release_intent.json").resolve(),
        "evaluation-wave release-intent path drifted",
    )
    expect(
        Path(payload.get("evaluation_wave_release_result", "")).resolve()
        == (control_root / "evaluation_wave_release_result.json").resolve(),
        "evaluation-wave release-result path drifted",
    )
    expect(
        Path(payload.get("evaluation_wave_release_failure", "")).resolve()
        == (control_root / "evaluation_wave_release_failure.json").resolve(),
        "evaluation-wave release-failure path drifted",
    )
    expect(
        Path(payload.get("evaluation_wave_submission_failure", "")).resolve()
        == (control_root / "evaluation_wave_submission_failure.json").resolve(),
        "evaluation-wave submission-failure path drifted",
    )
    expect(
        Path(payload.get("evaluation_wave_finalizer_receipt", "")).resolve()
        == (control_root / "evaluation_wave_finalizer.json").resolve(),
        "evaluation-wave finalizer-receipt path drifted",
    )
    expect(
        Path(payload.get("evaluation_wave_finalizer_private_log", "")).resolve()
        == (control_root / "evaluation_wave_finalizer.controller.private.log").resolve(),
        "evaluation-wave finalizer-log path drifted",
    )
    expect(
        Path(payload.get("evaluation_wave_seal", "")).resolve()
        == (control_root / "evaluation_wave_seal.json").resolve(),
        "evaluation-wave seal path drifted",
    )
    outputs = payload.get("outputs")
    expect(isinstance(outputs, dict) and set(outputs) == {"json", "markdown", "manifest"}, "release outputs drifted")
    expect(all(Path(value).is_absolute() for value in outputs.values()), "release output path is not absolute")
    expect(outputs == {
        "json": str((result_root / "readout.json").resolve()),
        "markdown": str((result_root / "readout.md").resolve()),
        "manifest": str((result_root / "bundle_manifest.json").resolve()),
    }, "release output derivation drifted")
    return resolved, payload, prereg


TERMINAL_STATES = {
    "COMPLETED",
    "FAILED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "CANCELLED",
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
}


def training_authorization_payload(
    *, repo: Path, release_plan: str | Path, arm_key_value: str, created_utc: str
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    plan_path, plan, prereg = validate_release_plan(release_plan, repo)
    expect(arm_key_value in plan["arm_paths"], "training arm is not preregistered")
    created = parse_utc(created_utc, "training authorization created_utc")
    expect(parse_utc(plan["created_utc"], "release plan created_utc") <= created, "training authorization predates release plan")
    arm = prereg["payload"]["arms"][arm_key_value]
    paths = plan["arm_paths"][arm_key_value]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "authorization": "opd_math_objective_family_training_launch_authorization_v1",
        "status": "authorized_before_held_scheduler_submission",
        "created_utc": created_utc,
        "arm_key": arm_key_value,
        "run_id": arm["run_id"],
        "objective_id": arm["objective_id"],
        "implementation": arm["implementation"],
        "source": arm["source"],
        "seed": arm["seed"],
        "release_plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
        "program_manifest": plan["program_manifest"],
        "o_teacher_audit_receipt": plan["o_teacher_audit_receipt"],
        "preregistration": plan["preregistration"],
        "launch_plan": plan["launch_plan"],
        "expected_training_out": paths["training_out"],
        "expected_prelaunch_receipt": paths["prelaunch_receipt"],
        "expected_submission_receipt": paths["submission_receipt"],
        "expected_consumption_receipt": paths["training_consumption_receipt"],
        "student_and_heldout_outcomes_inspected": False,
        "submission_protocol": "sbatch_held_then_record_submission_then_scontrol_release",
    }
    return plan_path, plan, payload


def authorize_training(args: argparse.Namespace) -> dict[str, Any]:
    repo = configure_repo(args.repo)
    created_utc = utc_now()
    _, plan, payload = training_authorization_payload(
        repo=repo,
        release_plan=args.release_plan,
        arm_key_value=args.arm_key,
        created_utc=created_utc,
    )
    paths = plan["arm_paths"][args.arm_key]
    expect(Path(args.output).resolve() == Path(paths["training_authorization"]).resolve(), "training authorization output drifted")
    forbidden = [
        Path(paths[field])
        for field in (
            "training_out", "prelaunch_receipt", "submission_receipt",
            "training_consumption_receipt", "heldout_gate", "evaluation_summary",
            "evaluation_samples", "evaluation_authorization",
        )
    ]
    existing = [str(path) for path in forbidden if path.exists() or path.is_symlink()]
    expect(not existing, f"training authorization is not prelaunch-fresh: {existing}")
    write_new(args.output, payload)
    return payload


def validate_training_authorization(
    *, repo: Path, release_plan: str | Path, arm_key_value: str
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    plan_path, plan, _ = validate_release_plan(release_plan, repo)
    path = regular_readonly(plan["arm_paths"][arm_key_value]["training_authorization"], f"training authorization {arm_key_value}")
    observed = load_json(path, f"training authorization {arm_key_value}")
    _, _, expected_payload = training_authorization_payload(
        repo=repo,
        release_plan=plan_path,
        arm_key_value=arm_key_value,
        created_utc=observed.get("created_utc"),
    )
    expect(observed == expected_payload, f"training authorization drifted: {arm_key_value}")
    return path, observed, plan


def query_one_sacct(job_id: str) -> tuple[str, dict[str, Any]]:
    expect(re.fullmatch(r"[1-9][0-9]*", job_id) is not None, "scheduler job ID is invalid")
    command = [
        "sacct", "-X", "-j", job_id,
        "--format=JobIDRaw,JobName,State,ExitCode,ElapsedRaw,Submit,Start,End,AllocTRES,NodeList",
        "-n", "-P",
    ]
    raw = subprocess.run(command, check=True, text=True, capture_output=True).stdout
    matches = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        expect(len(parts) >= 10, f"unexpected sacct row: {line!r}")
        if parts[0] == job_id:
            matches.append(parts[:10])
    expect(len(matches) == 1, f"sacct lacks one exact top-level row for {job_id}")
    job, name, state, exit_code, elapsed, submit, start, end, alloc_tres, node_list = matches[0]
    normalized = state.split()[0].split("+")[0]
    expect(elapsed.isdigit(), f"invalid scheduler elapsed for {job_id}")
    return raw, {
        "job_id": job,
        "job_name": name,
        "state": normalized,
        "state_raw": state,
        "exit_code": exit_code,
        "elapsed_seconds": int(elapsed),
        "submit": submit,
        "start": start,
        "end": end,
        "alloc_tres": alloc_tres,
        "node_list": node_list,
    }


def query_held_job(job_id: str) -> dict[str, str]:
    expect(re.fullmatch(r"[1-9][0-9]*", job_id) is not None, "held scheduler job ID is invalid")
    raw = subprocess.run(
        ["squeue", "--noheader", "--jobs", job_id, "--format=%i|%T|%r"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    rows = [line.strip().split("|", 2) for line in raw.splitlines() if line.strip()]
    expect(len(rows) == 1 and len(rows[0]) == 3, f"held scheduler query is not one job: {job_id}")
    observed_job, state, reason = rows[0]
    expect(observed_job == job_id, f"held scheduler identity drifted: {job_id}")
    expect(state == "PENDING" and reason == "JobHeldUser", f"job is not user-held: {job_id} {state} {reason}")
    return {"job_id": observed_job, "state": state, "reason": reason}


def record_submission(args: argparse.Namespace) -> dict[str, Any]:
    repo = configure_repo(args.repo)
    auth_path, auth, plan = validate_training_authorization(
        repo=repo, release_plan=args.release_plan, arm_key_value=args.arm_key
    )
    paths = plan["arm_paths"][args.arm_key]
    expect(Path(args.output).resolve() == Path(paths["submission_receipt"]).resolve(), "submission receipt output drifted")
    expect(not Path(paths["training_consumption_receipt"]).exists(), "training authorization was consumed before submission receipt")
    raw, scheduler = query_one_sacct(args.scheduler_job_id)
    expect(scheduler["state"] == "PENDING", "submission must be recorded while job is pending")
    held_before = query_held_job(args.scheduler_job_id)
    control = _scontrol_snapshot(args.scheduler_job_id)
    program = load_json(plan["program_manifest"]["path"], "program manifest")
    implementation = auth["implementation"]
    wrapper_binding = program[
        "local_training_wrapper" if implementation == "local" else "upstream_training_wrapper"
    ]
    expect(Path(str(control["command"])).resolve() == Path(wrapper_binding["path"]).resolve(), "training submission did not use the sealed custody wrapper")
    expect(control["job_state"] == "PENDING" and control["reason"] == "JobHeldUser", "training submission was not held in scontrol")
    expect(control["partition"] == "general-gpu" and control["account"] == "engr-lab-jacobsn", "training scheduler lane drifted")
    req_tres = _tres_tokens(str(control["req_tres"]))
    expected_resources = (
        {"cpu": "8", "mem": "96G", "gres/gpu": "1", "gres/gpu:a100-sxm4": "1"}
        if implementation == "local"
        else {"cpu": "16", "mem": "160G", "gres/gpu": "2", "gres/gpu:a100-sxm4": "2"}
    )
    expect(all(req_tres.get(field) == value for field, value in expected_resources.items()), "training scheduler resources drifted")
    submit_utc = parse_slurm_time(scheduler["submit"], "scheduler submit")
    # Slurm reports Submit only to whole-second precision, while controller
    # receipts retain microseconds. Compare at the scheduler's precision so an
    # authorization followed by sbatch in the same second is not rejected.
    expect(parse_utc(plan["created_utc"], "release plan created_utc").replace(microsecond=0) <= submit_utc, "scheduler submission predates release plan")
    expect(parse_utc(auth["created_utc"], "training auth created_utc").replace(microsecond=0) <= submit_utc, "scheduler submission predates training authorization")
    created_utc = utc_now()
    created = parse_utc(created_utc, "submission receipt created_utc")
    expect(created >= submit_utc, "submission receipt predates scheduler submission")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "receipt": "opd_math_objective_family_scheduler_submission_v1",
        "status": "held_submission_recorded_before_optimizer_start",
        "created_utc": created_utc,
        "arm_key": args.arm_key,
        "run_id": auth["run_id"],
        "scheduler_job_id": args.scheduler_job_id,
        "training_authorization": {"path": str(auth_path), "sha256": sha256_file(auth_path)},
        "release_plan": auth["release_plan"],
        "scheduler": scheduler,
        "held_queue": held_before,
        "scheduler_control": control,
        "custody_wrapper": wrapper_binding,
        "sacct_raw_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "job_was_held_or_pending_when_recorded": True,
        "optimizer_started_before_receipt": False,
        "student_and_heldout_outcomes_inspected": False,
    }
    write_new(args.output, payload)
    return payload


def validate_submission_receipt(
    *, repo: Path, release_plan: str | Path, arm_key_value: str, requery: bool
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    auth_path, auth, plan = validate_training_authorization(
        repo=repo, release_plan=release_plan, arm_key_value=arm_key_value
    )
    path = regular_readonly(plan["arm_paths"][arm_key_value]["submission_receipt"], f"submission receipt {arm_key_value}")
    payload = load_json(path, f"submission receipt {arm_key_value}")
    expected_keys = {
        "schema_version", "receipt", "status", "created_utc", "arm_key", "run_id",
        "scheduler_job_id", "training_authorization", "release_plan", "scheduler", "held_queue",
        "scheduler_control", "custody_wrapper",
        "sacct_raw_sha256", "job_was_held_or_pending_when_recorded",
        "optimizer_started_before_receipt", "student_and_heldout_outcomes_inspected",
    }
    expect(set(payload) == expected_keys, f"submission receipt schema drifted: {arm_key_value}")
    expect(payload.get("schema_version") == SCHEMA_VERSION and payload.get("receipt") == "opd_math_objective_family_scheduler_submission_v1", f"submission receipt identity drifted: {arm_key_value}")
    expect(payload.get("status") == "held_submission_recorded_before_optimizer_start", f"submission receipt status drifted: {arm_key_value}")
    expect(payload.get("arm_key") == arm_key_value and payload.get("run_id") == auth["run_id"], f"submission arm identity drifted: {arm_key_value}")
    expect(payload.get("training_authorization") == {"path": str(auth_path), "sha256": sha256_file(auth_path)}, f"submission auth binding drifted: {arm_key_value}")
    expect(payload.get("release_plan") == auth["release_plan"], f"submission plan binding drifted: {arm_key_value}")
    expect(payload.get("job_was_held_or_pending_when_recorded") is True and payload.get("optimizer_started_before_receipt") is False and payload.get("student_and_heldout_outcomes_inspected") is False, f"submission chronology flags drifted: {arm_key_value}")
    created = parse_utc(payload.get("created_utc"), "submission created_utc")
    scheduler = payload.get("scheduler")
    expect(isinstance(scheduler, dict) and set(scheduler) == {
        "job_id", "job_name", "state", "state_raw", "exit_code", "elapsed_seconds",
        "submit", "start", "end", "alloc_tres", "node_list",
    }, f"submission scheduler schema drifted: {arm_key_value}")
    expect(scheduler.get("job_id") == payload.get("scheduler_job_id") and scheduler.get("state") == "PENDING", f"submission scheduler binding drifted: {arm_key_value}")
    expect(payload.get("held_queue") == {
        "job_id": payload.get("scheduler_job_id"), "state": "PENDING", "reason": "JobHeldUser"
    }, f"submission held-state binding drifted: {arm_key_value}")
    program = load_json(plan["program_manifest"]["path"], "program manifest")
    implementation = auth["implementation"]
    expected_wrapper = program["local_training_wrapper" if implementation == "local" else "upstream_training_wrapper"]
    expect(payload.get("custody_wrapper") == expected_wrapper, f"submission custody wrapper drifted: {arm_key_value}")
    control = payload.get("scheduler_control") or {}
    expect(control.get("job_id") == payload.get("scheduler_job_id") and Path(str(control.get("command"))).resolve() == Path(expected_wrapper["path"]).resolve(), f"submission scontrol custody drifted: {arm_key_value}")
    submit_utc = parse_slurm_time(scheduler.get("submit"), "scheduler submit")
    expect(parse_utc(plan["created_utc"], "release plan created_utc").replace(microsecond=0) <= submit_utc <= created, f"submission chronology drifted: {arm_key_value}")
    expect(parse_utc(auth["created_utc"], "training auth created_utc").replace(microsecond=0) <= submit_utc, f"authorization chronology drifted: {arm_key_value}")
    if requery:
        _, current = query_one_sacct(str(payload["scheduler_job_id"]))
        expect(current["job_id"] == scheduler["job_id"] and current["submit"] == scheduler["submit"] and current["job_name"] == scheduler["job_name"], f"scheduler immutable identity drifted: {arm_key_value}")
    return path, payload, plan


def consume_training_authorization(args: argparse.Namespace) -> dict[str, Any]:
    repo = configure_repo(args.repo)
    submission_path, submission, plan = validate_submission_receipt(
        repo=repo, release_plan=args.release_plan, arm_key_value=args.arm_key, requery=True
    )
    current_job = os.environ.get("SLURM_JOB_ID")
    expect(current_job == args.scheduler_job_id == submission["scheduler_job_id"], "training authorization must be consumed inside the registered Slurm job")
    paths = plan["arm_paths"][args.arm_key]
    expect(Path(args.output).resolve() == Path(paths["training_consumption_receipt"]).resolve(), "training consumption output drifted")
    expect(not Path(paths["training_out"]).exists() and not Path(paths["prelaunch_receipt"]).exists(), "training artifacts exist before external authorization consumption")
    created_utc = utc_now()
    created = parse_utc(created_utc, "training consumption created_utc")
    expect(created >= parse_utc(submission["created_utc"], "submission created_utc"), "training consumption predates submission receipt")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "receipt": "opd_math_objective_family_training_authorization_consumption_v1",
        "status": "consumed_inside_registered_job_before_d89_launcher",
        "created_utc": created_utc,
        "arm_key": args.arm_key,
        "run_id": submission["run_id"],
        "scheduler_job_id": args.scheduler_job_id,
        "submission_receipt": {"path": str(submission_path), "sha256": sha256_file(submission_path)},
        "release_plan": submission["release_plan"],
        "expected_training_out": paths["training_out"],
        "expected_prelaunch_receipt": paths["prelaunch_receipt"],
        "optimizer_started_before_consumption": False,
        "student_and_heldout_outcomes_inspected": False,
    }
    write_new(args.output, payload)
    return payload


def validate_consumption_receipt(
    *, repo: Path, release_plan: str | Path, arm_key_value: str, requery: bool = False
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    submission_path, submission, plan = validate_submission_receipt(
        repo=repo, release_plan=release_plan, arm_key_value=arm_key_value, requery=requery
    )
    path = regular_readonly(plan["arm_paths"][arm_key_value]["training_consumption_receipt"], f"training consumption {arm_key_value}")
    payload = load_json(path, f"training consumption {arm_key_value}")
    expected = {
        "schema_version": SCHEMA_VERSION,
        "receipt": "opd_math_objective_family_training_authorization_consumption_v1",
        "status": "consumed_inside_registered_job_before_d89_launcher",
        "created_utc": payload.get("created_utc"),
        "arm_key": arm_key_value,
        "run_id": submission["run_id"],
        "scheduler_job_id": submission["scheduler_job_id"],
        "submission_receipt": {"path": str(submission_path), "sha256": sha256_file(submission_path)},
        "release_plan": submission["release_plan"],
        "expected_training_out": plan["arm_paths"][arm_key_value]["training_out"],
        "expected_prelaunch_receipt": plan["arm_paths"][arm_key_value]["prelaunch_receipt"],
        "optimizer_started_before_consumption": False,
        "student_and_heldout_outcomes_inspected": False,
    }
    parse_utc(payload.get("created_utc"), "training consumption created_utc")
    expect(payload == expected, f"training consumption receipt drifted: {arm_key_value}")
    return path, payload, plan


def validate_training_control_shallow(
    *, plan_path: Path, plan: Mapping[str, Any], prereg: Mapping[str, Any], key: str,
    require_consumption: bool,
) -> tuple[Path, dict[str, Any], Path | None, dict[str, Any] | None]:
    """Validate one arm's sealed control receipts without replaying the release plan."""

    arm = prereg["payload"]["arms"][key]
    paths = plan["arm_paths"][key]
    auth_path = regular_readonly(paths["training_authorization"], f"training authorization {key}")
    auth = load_json(auth_path, f"training authorization {key}")
    expect(set(auth) == {
        "schema_version", "authorization", "status", "created_utc", "arm_key", "run_id",
        "objective_id", "implementation", "source", "seed", "release_plan",
        "program_manifest", "o_teacher_audit_receipt", "preregistration", "launch_plan", "expected_training_out",
        "expected_prelaunch_receipt", "expected_submission_receipt", "expected_consumption_receipt",
        "student_and_heldout_outcomes_inspected", "submission_protocol",
    }, f"training authorization shallow schema drifted: {key}")
    expect(auth.get("schema_version") == SCHEMA_VERSION and auth.get("authorization") == "opd_math_objective_family_training_launch_authorization_v1" and auth.get("status") == "authorized_before_held_scheduler_submission", f"training authorization shallow identity drifted: {key}")
    for field, expected in (
        ("arm_key", key), ("run_id", arm["run_id"]), ("objective_id", arm["objective_id"]),
        ("implementation", arm["implementation"]), ("source", arm["source"]), ("seed", arm["seed"]),
        ("release_plan", {"path": str(plan_path), "sha256": sha256_file(plan_path)}),
        ("program_manifest", plan["program_manifest"]),
        ("o_teacher_audit_receipt", plan["o_teacher_audit_receipt"]),
        ("preregistration", plan["preregistration"]),
        ("launch_plan", plan["launch_plan"]), ("expected_training_out", paths["training_out"]),
        ("expected_prelaunch_receipt", paths["prelaunch_receipt"]),
        ("expected_submission_receipt", paths["submission_receipt"]),
        ("expected_consumption_receipt", paths["training_consumption_receipt"]),
        ("student_and_heldout_outcomes_inspected", False),
        ("submission_protocol", "sbatch_held_then_record_submission_then_scontrol_release"),
    ):
        expect(auth.get(field) == expected, f"training authorization shallow field drifted: {key} {field}")
    parse_utc(auth["created_utc"], f"training authorization created_utc {key}")
    submission_path = regular_readonly(paths["submission_receipt"], f"submission receipt {key}")
    submission = load_json(submission_path, f"submission receipt {key}")
    expect(set(submission) == {
        "schema_version", "receipt", "status", "created_utc", "arm_key", "run_id",
        "scheduler_job_id", "training_authorization", "release_plan", "scheduler", "held_queue",
        "scheduler_control", "custody_wrapper", "sacct_raw_sha256",
        "job_was_held_or_pending_when_recorded", "optimizer_started_before_receipt",
        "student_and_heldout_outcomes_inspected",
    }, f"submission shallow schema drifted: {key}")
    expect(submission.get("schema_version") == SCHEMA_VERSION and submission.get("receipt") == "opd_math_objective_family_scheduler_submission_v1" and submission.get("status") == "held_submission_recorded_before_optimizer_start", f"submission shallow identity drifted: {key}")
    expect(submission.get("arm_key") == key and submission.get("run_id") == arm["run_id"], f"submission shallow arm drifted: {key}")
    expect(submission.get("training_authorization") == file_binding(auth_path) and submission.get("release_plan") == auth["release_plan"], f"submission shallow custody drifted: {key}")
    expect(submission.get("job_was_held_or_pending_when_recorded") is True and submission.get("optimizer_started_before_receipt") is False and submission.get("student_and_heldout_outcomes_inspected") is False, f"submission shallow boundary drifted: {key}")
    scheduler = submission.get("scheduler")
    expect(isinstance(scheduler, dict) and set(scheduler) == {
        "job_id", "job_name", "state", "state_raw", "exit_code", "elapsed_seconds",
        "submit", "start", "end", "alloc_tres", "node_list",
    }, f"submission shallow scheduler schema drifted: {key}")
    expect(scheduler.get("job_id") == submission.get("scheduler_job_id") and scheduler.get("state") == "PENDING", f"submission shallow scheduler drifted: {key}")
    expect(submission.get("held_queue") == {"job_id": submission.get("scheduler_job_id"), "state": "PENDING", "reason": "JobHeldUser"}, f"submission shallow held queue drifted: {key}")
    program = load_json(plan["program_manifest"]["path"], "program manifest")
    expected_wrapper = program["local_training_wrapper" if arm["implementation"] == "local" else "upstream_training_wrapper"]
    expect(submission.get("custody_wrapper") == expected_wrapper, f"submission shallow wrapper drifted: {key}")
    control = submission.get("scheduler_control") or {}
    expect(set(control) == {"job_id", "job_state", "reason", "user_held", "row_count", "array_expanded", "dependency", "array_task_id", "submit_time", "work_dir", "std_out", "command", "partition", "account", "req_tres", "array_job_id", "raw_sha256", "raw_text"}, f"submission shallow scheduler-control schema drifted: {key}")
    expect(control.get("job_id") == submission.get("scheduler_job_id") and Path(str(control.get("command"))).resolve() == Path(expected_wrapper["path"]).resolve(), f"submission shallow scheduler control drifted: {key}")
    expect(control.get("job_state") == "PENDING" and control.get("reason") == "JobHeldUser" and control.get("user_held") is True and control.get("row_count") == 1 and control.get("array_expanded") is False and control.get("partition") == "general-gpu" and control.get("account") == "engr-lab-jacobsn", f"submission shallow scheduler lane/held state drifted: {key}")
    resources = _tres_tokens(str(control.get("req_tres")))
    expected_resources = ({"cpu": "8", "mem": "96G", "gres/gpu": "1", "gres/gpu:a100-sxm4": "1"} if arm["implementation"] == "local" else {"cpu": "16", "mem": "160G", "gres/gpu": "2", "gres/gpu:a100-sxm4": "2"})
    expect(all(resources.get(field) == value for field, value in expected_resources.items()), f"submission shallow scheduler resources drifted: {key}")
    auth_created = parse_utc(auth["created_utc"], f"training authorization created_utc {key}")
    submit_time = parse_slurm_time(scheduler["submit"], f"scheduler submit {key}")
    submission_created = parse_utc(submission["created_utc"], f"submission created_utc {key}")
    plan_created = parse_utc(plan["created_utc"], "release plan created_utc")
    expect(plan_created <= auth_created, f"training authorization predates release plan: {key}")
    expect(plan_created.replace(microsecond=0) <= submit_time and auth_created.replace(microsecond=0) <= submit_time <= submission_created, f"training control chronology drifted: {key}")
    consumption_path = Path(paths["training_consumption_receipt"]).resolve()
    if not require_consumption and not consumption_path.exists():
        return submission_path, submission, None, None
    consumption_path = regular_readonly(consumption_path, f"training consumption {key}")
    consumption = load_json(consumption_path, f"training consumption {key}")
    expected_consumption = {
        "schema_version": SCHEMA_VERSION,
        "receipt": "opd_math_objective_family_training_authorization_consumption_v1",
        "status": "consumed_inside_registered_job_before_d89_launcher",
        "created_utc": consumption.get("created_utc"),
        "arm_key": key,
        "run_id": arm["run_id"],
        "scheduler_job_id": submission["scheduler_job_id"],
        "submission_receipt": file_binding(submission_path),
        "release_plan": submission["release_plan"],
        "expected_training_out": paths["training_out"],
        "expected_prelaunch_receipt": paths["prelaunch_receipt"],
        "optimizer_started_before_consumption": False,
        "student_and_heldout_outcomes_inspected": False,
    }
    consumption_created = parse_utc(consumption.get("created_utc"), f"training consumption created_utc {key}")
    expect(consumption_created >= submission_created, f"training consumption predates submission: {key}")
    expect(consumption == expected_consumption, f"training consumption shallow drifted: {key}")
    return submission_path, submission, consumption_path, consumption


def _prelaunch_job_identity(path: str | Path, key: str) -> dict[str, str]:
    resolved = regular_readonly(path, f"prelaunch receipt {key}")
    payload = load_json(resolved, f"prelaunch receipt {key}")
    expect(payload.get("run_key") == key, f"prelaunch run key drifted: {key}")
    job_id = payload.get("scheduler_job_id")
    run_id = payload.get("run_id")
    expect(isinstance(job_id, str) and re.fullmatch(r"[1-9][0-9]*", job_id), f"prelaunch job ID invalid: {key}")
    expect(isinstance(run_id, str) and run_id, f"prelaunch run ID invalid: {key}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "job_id": job_id,
        "run_id": run_id,
    }


def sacct_command(job_ids: Sequence[str]) -> list[str]:
    expect(len(job_ids) == 36 and len(set(job_ids)) == 36, "terminal snapshot requires 36 unique scheduler jobs")
    return [
        "sacct",
        "-X",
        "-j",
        ",".join(job_ids),
        "--format=JobIDRaw,JobName,State,ExitCode,ElapsedRaw,Submit,Start,End,AllocTRES,NodeList",
        "-n",
        "-P",
    ]


def parse_sacct_rows(raw: str, job_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
    expected_ids = set(job_ids)
    rows: dict[str, dict[str, Any]] = {}
    for line in raw.splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        expect(len(parts) >= 10, f"unexpected sacct row: {line!r}")
        job_id, name, state, exit_code, elapsed, submit, start, end, alloc_tres, node_list = parts[:10]
        if job_id not in expected_ids:
            continue
        expect(job_id not in rows, f"duplicate top-level sacct row: {job_id}")
        normalized_state = state.split()[0].split("+")[0]
        expect(normalized_state in TERMINAL_STATES, f"job is not terminal: {job_id} {state}")
        expect(elapsed.isdigit(), f"sacct elapsed is not numeric: {job_id}")
        rows[job_id] = {
            "job_id": job_id,
            "job_name": name,
            "state": normalized_state,
            "state_raw": state,
            "exit_code": exit_code,
            "elapsed_seconds": int(elapsed),
            "submit": submit,
            "start": start,
            "end": end,
            "alloc_tres": alloc_tres,
            "node_list": node_list,
        }
    expect(set(rows) == expected_ids, f"sacct lacks terminal rows: {sorted(expected_ids - set(rows))}")
    return rows


def query_sacct(job_ids: Sequence[str]) -> tuple[str, dict[str, dict[str, Any]]]:
    command = sacct_command(job_ids)
    raw = subprocess.run(command, check=True, text=True, capture_output=True).stdout
    rows = parse_sacct_rows(raw, job_ids)
    return raw, rows


def forbidden_after_training_paths(plan: Mapping[str, Any]) -> list[Path]:
    paths: list[Path] = [
        *(Path(value) for value in plan["outputs"].values()),
        Path(plan["evaluation_wave_authorization"]),
        Path(plan["evaluation_wave_submission_index"]),
        Path(plan["evaluation_wave_submission_journal_root"]),
        Path(plan["evaluation_wave_release_intent"]),
        Path(plan["evaluation_wave_release_result"]),
        Path(plan["evaluation_wave_release_failure"]),
        Path(plan["evaluation_wave_submission_failure"]),
        Path(plan["evaluation_wave_finalizer_receipt"]),
        Path(plan["evaluation_wave_finalizer_private_log"]),
        Path(plan["evaluation_wave_seal"]),
    ]
    for item in plan["arm_paths"].values():
        paths.extend(Path(item[field]) for field in (
            "heldout_gate", "artifact_root", "evaluation_summary", "evaluation_samples",
            "evaluation_companion", "evaluation_authorization",
            "evaluation_submission_receipt", "evaluation_merge_consumption_receipt",
            "evaluation_merge_supervisor_receipt", "evaluation_seal_supervisor_receipt",
            "evaluation_terminal_failure_receipt", "evaluation_array_accounting_receipt",
            "evaluation_array_accounting_raw", "evaluation_consumption_receipt",
            "evaluation_seal_receipt", "evaluation_shard_consumption_root",
            "evaluation_log_root", "evaluation_private_log_root",
        ))
    for item in plan["raw_student_auxiliary"].values():
        paths.extend(Path(item[field]) for field in (
            "artifact_root", "summary", "samples", "companion", "authorization",
            "submission_receipt", "merge_consumption_receipt", "merge_supervisor_receipt",
            "seal_supervisor_receipt", "terminal_failure_receipt", "array_accounting_receipt",
            "array_accounting_raw", "consumption_receipt", "seal_receipt",
            "shard_consumption_root", "log_root", "private_log_root", "gate",
        ))
    return paths


def seal_terminal_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    repo = configure_repo(args.repo)
    plan_path, plan, prereg = validate_release_plan(args.release_plan, repo)
    expected_output = Path(plan["terminal_snapshot"]).resolve()
    expect(Path(args.output).resolve() == expected_output, "terminal snapshot output differs from release plan")
    if expected_output.exists() or expected_output.is_symlink():
        _, existing = validate_terminal_snapshot(expected_output, plan_path, plan, repo=repo, prereg=prereg)
        return existing
    existing_heldout = sorted(
        str(path) for path in forbidden_after_training_paths(plan)
        if path.exists() or path.is_symlink()
    )
    expect(not existing_heldout, f"heldout/result artifacts existed before terminal snapshot: {existing_heldout[:10]}")
    validator_runtime = require_train_python(plan, prereg)
    submissions: dict[str, dict[str, Any]] = {}
    submission_paths: dict[str, Path] = {}
    for key in prereg["payload"]["arm_keys"]:
        path, payload, _, _ = validate_training_control_shallow(
            plan_path=plan_path, plan=plan, prereg=prereg, key=key,
            require_consumption=False,
        )
        _, current = query_one_sacct(str(payload["scheduler_job_id"]))
        expect(current["job_id"] == payload["scheduler"]["job_id"] and current["submit"] == payload["scheduler"]["submit"] and current["job_name"] == payload["scheduler"]["job_name"], f"scheduler immutable identity drifted: {key}")
        submission_paths[key] = path
        submissions[key] = payload
    job_ids = [submissions[key]["scheduler_job_id"] for key in prereg["payload"]["arm_keys"]]
    raw, sacct = query_sacct(job_ids)
    arms: dict[str, dict[str, Any]] = {}
    for key in prereg["payload"]["arm_keys"]:
        submission = submissions[key]
        scheduler = sacct[submission["scheduler_job_id"]]
        consumption_binding = None
        prelaunch_binding = None
        training_validation = None
        training_audit_binding = None
        failure = None
        if scheduler["state"] != "COMPLETED" or scheduler["exit_code"] != "0:0":
            failure = {
                "classification": "terminal_scheduler_failure",
                "error_type": None,
                "error_sha256": None,
            }
        else:
            try:
                _, _, consumption_path, consumption_payload = validate_training_control_shallow(
                    plan_path=plan_path, plan=plan, prereg=prereg, key=key,
                    require_consumption=True,
                )
                expect(consumption_path is not None, f"training consumption is missing: {key}")
                expect(consumption_payload is not None, f"training consumption payload is missing: {key}")
                consumed = parse_utc(consumption_payload["created_utc"], f"training consumption created_utc {key}")
                expect(parse_slurm_time(scheduler["start"], f"terminal scheduler start {key}") <= consumed <= parse_slurm_time(scheduler["end"], f"terminal scheduler end {key}"), f"training consumption is outside scheduler runtime: {key}")
                consumption_binding = {"path": str(consumption_path), "sha256": sha256_file(consumption_path)}
            except (ValueError, RuntimeError) as error:
                failure = {
                    "classification": "terminal_prelauncher_or_consumption_failure",
                    "error_type": type(error).__name__,
                    "error_sha256": hashlib.sha256(str(error).encode("utf-8")).hexdigest(),
                }
        if failure is None:
            audit_target = Path(plan["arm_paths"][key]["terminal_audit_receipt"])
            if audit_target.exists() or audit_target.is_symlink():
                audit_path, audit = validate_training_audit_receipt(
                    plan_path=plan_path, plan=plan, prereg=prereg, key=key,
                    terminal_arm=None, mode="selected_root",
                )
                prelaunch = _prelaunch_job_identity(plan["arm_paths"][key]["prelaunch_receipt"], key)
                expect(prelaunch["job_id"] == submission["scheduler_job_id"] and prelaunch["run_id"] == submission["run_id"], f"tracked prelaunch differs from submission: {key}")
                expect(audit["scheduler_terminal"] == scheduler and audit["submission_receipt"] == file_binding(submission_paths[key]) and audit["training_consumption_receipt"] == file_binding(consumption_path) and audit["prelaunch_receipt"] == {"path": prelaunch["path"], "sha256": prelaunch["sha256"]}, f"existing training audit differs from terminal custody: {key}")
                prelaunch_binding = {"path": prelaunch["path"], "sha256": prelaunch["sha256"]}
                training_audit_binding = file_binding(audit_path)
                training_validation = audit["deep_validation"]
            else:
                try:
                    prelaunch = _prelaunch_job_identity(plan["arm_paths"][key]["prelaunch_receipt"], key)
                    expect(prelaunch["job_id"] == submission["scheduler_job_id"] and prelaunch["run_id"] == submission["run_id"], f"tracked prelaunch differs from submission: {key}")
                    prelaunch_binding = {"path": prelaunch["path"], "sha256": prelaunch["sha256"]}
                    prepared_validation = validate_completed_training_for_arm(
                        repo=repo, key=key, plan=plan, prereg=prereg
                    )
                except (
                    ValueError, RuntimeError, FileNotFoundError, PermissionError,
                    IsADirectoryError, KeyError, TypeError, IndexError,
                ) as error:
                    failure = {
                        "classification": "terminal_training_custody_failure",
                        "error_type": type(error).__name__,
                        "error_sha256": hashlib.sha256(str(error).encode("utf-8")).hexdigest(),
                    }
                if failure is None:
                    audit_path, audit = seal_training_audit_receipt(
                        repo=repo, plan_path=plan_path, plan=plan, prereg=prereg,
                        key=key, scheduler=scheduler, submission_path=submission_paths[key],
                        consumption_path=consumption_path, prelaunch=prelaunch,
                        validator_runtime=validator_runtime,
                        deep_validation=prepared_validation,
                    )
                    training_audit_binding = file_binding(audit_path)
                    training_validation = audit["deep_validation"]
        eligible = failure is None and training_validation is not None
        arms[key] = {
            "submission_receipt": {"path": str(submission_paths[key]), "sha256": sha256_file(submission_paths[key])},
            "consumption_receipt": consumption_binding,
            "prelaunch_receipt": prelaunch_binding,
            "training_audit_receipt": training_audit_binding,
            "run_id": submission["run_id"],
            "scheduler": scheduler,
            "training_status": "completed_eligible" if eligible else failure["classification"],
            "training_eligible_for_heldout": eligible,
            "training_validation": training_validation,
            "terminal_failure": failure,
        }
    snapshot_created_utc = utc_now()
    audit_times = [
        parse_utc(load_json(value["training_audit_receipt"]["path"], f"training audit {key}")["created_utc"], f"training audit created_utc {key}")
        for key, value in arms.items() if value["training_audit_receipt"] is not None
    ]
    expect(all(parse_utc(snapshot_created_utc, "terminal snapshot created_utc") >= value for value in audit_times), "terminal snapshot predates a training audit")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "snapshot": "opd_math_objective_family_terminal_training_snapshot_v1",
        "status": "all_36_training_jobs_terminal_before_heldout",
        "created_utc": snapshot_created_utc,
        "release_plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
        "git_commit": EXPECTED_COMMIT,
        "arm_keys": prereg["payload"]["arm_keys"],
        "arms": arms,
        "all_training_jobs_terminal": True,
        "eligible_arms": sum(value["training_eligible_for_heldout"] for value in arms.values()),
        "terminal_failed_arms": [key for key, value in arms.items() if not value["training_eligible_for_heldout"]],
        "heldout_outcomes_inspected": False,
        "sacct": {
            "command": sacct_command(job_ids),
            "raw_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            "raw_text": raw,
        },
        "claim_boundary": TERMINAL_CLAIM_BOUNDARY,
    }
    write_new_or_exact(args.output, payload)
    return payload


def validate_terminal_snapshot(
    path: str | Path,
    plan_path: Path,
    plan: Mapping[str, Any],
    *,
    repo: Path,
    prereg: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    resolved = regular_readonly(path, "terminal snapshot")
    expect(resolved == Path(plan["terminal_snapshot"]).resolve(), "terminal snapshot path drifted")
    payload = load_json(resolved, "terminal snapshot")
    expected_keys = {
        "schema_version", "snapshot", "status", "created_utc", "release_plan", "git_commit",
        "arm_keys", "arms", "all_training_jobs_terminal", "eligible_arms",
        "terminal_failed_arms", "heldout_outcomes_inspected", "sacct", "claim_boundary",
    }
    expect(set(payload) == expected_keys, "terminal snapshot schema drifted")
    expect(payload.get("schema_version") == SCHEMA_VERSION and payload.get("snapshot") == "opd_math_objective_family_terminal_training_snapshot_v1", "terminal snapshot identity drifted")
    expect(payload.get("status") == "all_36_training_jobs_terminal_before_heldout", "terminal snapshot status drifted")
    created = parse_utc(payload.get("created_utc"), "terminal snapshot created_utc")
    expect(created >= parse_utc(plan["created_utc"], "release plan created_utc"), "terminal snapshot predates release plan")
    expect(payload.get("release_plan") == {"path": str(plan_path), "sha256": sha256_file(plan_path)}, "terminal snapshot release-plan binding drifted")
    expect(payload.get("git_commit") == EXPECTED_COMMIT, "terminal snapshot commit drifted")
    expect(payload.get("all_training_jobs_terminal") is True and payload.get("heldout_outcomes_inspected") is False, "terminal snapshot boundary drifted")
    expect(payload.get("arm_keys") == prereg["payload"]["arm_keys"], "terminal snapshot arm order drifted")
    expect(payload.get("claim_boundary") == TERMINAL_CLAIM_BOUNDARY, "terminal claim boundary drifted")
    arms = payload.get("arms")
    expect(isinstance(arms, dict) and set(arms) == set(plan["arm_paths"]), "terminal snapshot arm matrix drifted")
    job_ids: list[str] = []
    for key, value in arms.items():
        expected_arm_keys = {
            "submission_receipt", "consumption_receipt", "prelaunch_receipt", "training_audit_receipt",
            "run_id", "scheduler", "training_status",
            "training_eligible_for_heldout", "training_validation", "terminal_failure",
        }
        expect(isinstance(value, dict) and set(value) == expected_arm_keys, f"terminal snapshot arm schema drifted: {key}")
        scheduler = value["scheduler"]
        expect(set(scheduler) == {
            "job_id", "job_name", "state", "state_raw", "exit_code",
            "elapsed_seconds", "submit", "start", "end", "alloc_tres", "node_list",
        }, f"terminal scheduler schema drifted: {key}")
        expect(scheduler.get("state") in TERMINAL_STATES, f"terminal snapshot state invalid: {key}")
        scheduler_submit = parse_slurm_time(scheduler["submit"], f"terminal scheduler submit {key}")
        scheduler_end = parse_slurm_time(scheduler["end"], f"terminal scheduler end {key}")
        expect(scheduler_submit <= scheduler_end <= created, f"terminal snapshot scheduler chronology drifted: {key}")
        if scheduler.get("start") != "Unknown":
            scheduler_start = parse_slurm_time(scheduler["start"], f"terminal scheduler start {key}")
            expect(scheduler_submit <= scheduler_start <= scheduler_end, f"terminal scheduler start chronology drifted: {key}")
        else:
            scheduler_start = None
            expect(scheduler.get("state") != "COMPLETED", f"completed terminal job lacks start time: {key}")
        job_ids.append(str(scheduler["job_id"]))
        submission_path, observed_submission, consumption_path_shallow, consumption_payload_shallow = validate_training_control_shallow(
            plan_path=plan_path, plan=plan, prereg=prereg, key=key,
            require_consumption=value.get("consumption_receipt") is not None,
        )
        expect(value.get("submission_receipt") == {
            "path": str(submission_path), "sha256": sha256_file(submission_path)
        }, f"terminal submission binding drifted: {key}")
        expect(observed_submission["scheduler_job_id"] == scheduler["job_id"], f"terminal scheduler/submission drifted: {key}")
        expect(value.get("run_id") == observed_submission["run_id"], f"terminal run ID drifted: {key}")
        eligible = value.get("training_eligible_for_heldout") is True
        if consumption_payload_shallow is not None:
            expect(scheduler_start is not None, f"consumed training job lacks scheduler start: {key}")
            consumed = parse_utc(consumption_payload_shallow["created_utc"], f"terminal consumption created_utc {key}")
            expect(scheduler_start <= consumed <= scheduler_end, f"terminal consumption outside scheduler runtime: {key}")
        expect((value.get("training_status") == "completed_eligible") is eligible, f"terminal eligibility/status drifted: {key}")
        if eligible:
            expect(scheduler["state"] == "COMPLETED" and scheduler["exit_code"] == "0:0", f"eligible terminal arm did not complete successfully: {key}")
            expect(value.get("terminal_failure") is None and isinstance(value.get("training_validation"), dict), f"eligible terminal arm lacks validation: {key}")
            expect(consumption_path_shallow is not None, f"terminal consumption is missing: {key}")
            consumption_path = consumption_path_shallow
            expect(value.get("consumption_receipt") == {
                "path": str(consumption_path), "sha256": sha256_file(consumption_path)
            }, f"terminal consumption binding drifted: {key}")
            prelaunch = _prelaunch_job_identity(plan["arm_paths"][key]["prelaunch_receipt"], key)
            expect(value.get("prelaunch_receipt") == {
                "path": prelaunch["path"], "sha256": prelaunch["sha256"]
            }, f"terminal prelaunch binding drifted: {key}")
            expect(prelaunch["job_id"] == scheduler["job_id"] and prelaunch["run_id"] == value["run_id"], f"terminal prelaunch identity drifted: {key}")
            validate_training_audit_receipt(
                plan_path=plan_path,
                plan=plan,
                prereg=prereg,
                key=key,
                terminal_arm=value,
                mode="metadata",
            )
            audit_created = parse_utc(
                load_json(value["training_audit_receipt"]["path"], f"terminal training audit {key}")["created_utc"],
                f"training audit created_utc {key}",
            )
            expect(created >= audit_created, f"terminal snapshot predates training audit: {key}")
        else:
            expect(value.get("training_validation") is None, f"failed terminal arm retained training validation: {key}")
            expect(value.get("training_audit_receipt") is None, f"failed terminal arm retained a training audit: {key}")
            expect(not Path(plan["arm_paths"][key]["terminal_audit_receipt"]).exists(), f"failed terminal arm has an audit file: {key}")
            failure = value.get("terminal_failure")
            expect(isinstance(failure, dict) and set(failure) == {
                "classification", "error_type", "error_sha256"
            }, f"failed terminal arm lacks exact failure: {key}")
            if scheduler["state"] != "COMPLETED" or scheduler["exit_code"] != "0:0":
                expected_failure = {
                    "classification": "terminal_scheduler_failure",
                    "error_type": None,
                    "error_sha256": None,
                }
                expect(value.get("consumption_receipt") is None and value.get("prelaunch_receipt") is None, f"scheduler failure retained downstream receipts: {key}")
            else:
                expect(failure["classification"] in {"terminal_prelauncher_or_consumption_failure", "terminal_training_custody_failure"}, f"completed failed arm has invalid classification: {key}")
                expect(isinstance(failure.get("error_type"), str) and isinstance(failure.get("error_sha256"), str) and HEX64.fullmatch(failure["error_sha256"]), f"completed failed arm lacks sealed error identity: {key}")
                if value.get("consumption_receipt") is not None:
                    validate_binding(value["consumption_receipt"], f"failed-arm consumption {key}")
                if value.get("prelaunch_receipt") is not None:
                    validate_binding(value["prelaunch_receipt"], f"failed-arm prelaunch {key}")
                expected_failure = failure
            expect(failure == expected_failure, f"terminal failure classification drifted: {key}")
            expect(value.get("training_status") == expected_failure["classification"], f"terminal failure status drifted: {key}")
    sacct = payload.get("sacct")
    expect(isinstance(sacct, dict) and set(sacct) == {"command", "raw_sha256", "raw_text"}, "terminal sacct schema drifted")
    expect(sacct.get("command") == sacct_command(job_ids), "terminal sacct command drifted")
    expect(hashlib.sha256(str(sacct["raw_text"]).encode("utf-8")).hexdigest() == sacct["raw_sha256"], "terminal sacct raw hash drifted")
    expect(parse_sacct_rows(str(sacct["raw_text"]), job_ids) == {
        value["scheduler"]["job_id"]: value["scheduler"] for value in arms.values()
    }, "terminal sacct raw rows drifted")
    # Slurm does not guarantee stable row ordering across equivalent sacct
    # queries.  The normalized per-job rows above are the durable equality
    # check; retain the original raw text only as immutable creation evidence.
    expect(payload.get("eligible_arms") == sum(value["training_eligible_for_heldout"] for value in arms.values()), "terminal eligible count drifted")
    expect(payload.get("terminal_failed_arms") == [key for key in plan["arm_paths"] if not arms[key]["training_eligible_for_heldout"]], "terminal failed list drifted")
    return resolved, payload


LOCAL_MODE_BY_OBJECTIVE = {
    "task_rl": "task_rl",
    "task_rl_k1_ungated_clip5": "task_rl_k1_ungated_clip5",
    "task_rl_k1_ungated_unclipped": "task_rl_k1_ungated_unclipped",
    "task_rl_k1_gated_clip5_beta5": "task_rl_k1_gated_clip5_beta5",
    "k1_bare_verl_compatible_clip10": "k1_bare_verl_compatible_clip10",
}


def _finite_number(value: Any, label: str, *, positive: bool = False) -> float:
    expect(type(value) in (int, float) and math.isfinite(float(value)), f"{label} is nonfinite")
    observed = float(value)
    if positive:
        expect(observed > 0.0, f"{label} is not positive")
    return observed


def _trace_binding(completion: Mapping[str, Any], name: str, expected_rows: int) -> tuple[Path, list[dict[str, Any]]]:
    binding = (completion.get("trace_artifacts") or {}).get(name)
    expect(isinstance(binding, dict), f"completion lacks {name} binding")
    path = Path(str(binding.get("path"))).resolve()
    expect(path.is_file() and not path.is_symlink(), f"trace is missing: {path}")
    expect(binding.get("sha256") == sha256_file(path), f"trace hash drifted: {name}")
    rows = load_jsonl(path, name)
    expect(binding.get("rows") == expected_rows == len(rows), f"trace row count drifted: {name}")
    return path, rows


def validate_local_training(
    *,
    repo: Path,
    key: str,
    arm: Mapping[str, Any],
    prereg: Mapping[str, Any],
) -> dict[str, Any]:
    from scripts.opd.objective_family_preregistration import validate_prelaunch_receipt  # type: ignore
    from scripts.opd.objective_family_inputs import (  # type: ignore
        task_prompt_sha256,
        validate_initialization_manifest,
        validate_prompt_plan,
    )
    from scripts.opd.objective_registry import load_objective_registry, resolve_objective  # type: ignore
    from scripts.opd.opd_train import recompute_student_trace_geometry, render_prompt  # type: ignore
    from scripts.opd_math.math_reward import verify_completion  # type: ignore
    from scripts.opd_math.verify_environment import reverify_recorded_environment  # type: ignore

    root = Path(arm["training_out"]).resolve()
    expect(root.is_dir() and not root.is_symlink(), f"local training root is missing: {key}")
    run_path = root / "traces/run_manifest.json"
    completion_path = root / "traces/completion_manifest.json"
    run = load_json(run_path, f"local run {key}")
    completion = load_json(completion_path, f"local completion {key}")
    expect(run.get("completion") == completion, f"local run/completion disagree: {key}")
    objective_id = str(arm["objective_id"])
    source = str(arm["source"])
    seed = int(arm["seed"])
    registry = load_objective_registry()
    objective = resolve_objective(objective_id, registry=registry)
    registry_binding = run.get("objective_registry") or {}
    binding = run.get("binding") or {}
    expect(run.get("objective") == LOCAL_MODE_BY_OBJECTIVE[objective_id], f"local mode drifted: {key}")
    expected_registry_binding = {
        "registry_id": registry["registry_id"],
        "registry_path": registry["path"],
        "registry_sha256": registry["sha256"],
        "registry_canonical_sha256": registry["canonical_sha256"],
        "registry_status": registry["status"],
        "registry_alone_authorizes_scientific_launch": registry["registry_alone_authorizes_scientific_launch"],
        "objective": objective,
    }
    expect(registry_binding == expected_registry_binding, f"local objective registry drifted: {key}")
    expect(run.get("objective_contract") == objective["objective_contract"], f"local objective contract drifted: {key}")
    expect(run.get("git_commit") == EXPECTED_COMMIT and run.get("git_worktree_clean") is True, f"local Git custody failed: {key}")
    expect(run.get("student") == EXPECTED_STUDENT and run.get("student_revision") == EXPECTED_STUDENT_REVISION, f"local student identity drifted: {key}")
    expect(run.get("seed") == seed, f"local seed drifted: {key}")
    expect(run.get("optimizer_steps_planned") == 100 and run.get("planned_rollout_samples") == 400, f"local planned geometry drifted: {key}")
    expect(run.get("status") == "completed" and run.get("intended_scientific_run") is True, f"local run status drifted: {key}")
    expect(run.get("scientific_use_allowed") is False and run.get("training_artifact_eligible_for_held_out_evaluation") is True, f"local run claim boundary drifted: {key}")
    expect(binding.get("student_source") == source and binding.get("objective_family_diagnostic") is False, f"local source/diagnostic drifted: {key}")
    expect(binding.get("campaign_run_id") == arm["run_id"], f"local run ID drifted: {key}")
    scheduler_job_id = binding.get("scheduler_job_id")
    expect(isinstance(scheduler_job_id, str) and re.fullmatch(r"[1-9][0-9]*", scheduler_job_id), f"local scheduler ID drifted: {key}")
    student_plan_path = Path(prereg["payload"]["student_training_plan"]["path"]).resolve()
    student_plan = load_json(student_plan_path, "objective-family student plan")
    expected_config = dict(student_plan["common_fixed_config"])
    for field in student_plan["objective_fields_from_registry"]:
        expected_config[field] = objective[field]
    expected_config["seed"] = seed
    expect(run.get("normalized_training_config") == expected_config, f"local normalized training config drifted: {key}")
    plan_binding = binding.get("student_training_plan") or {}
    expect(plan_binding.get("path") == str(student_plan_path) and plan_binding.get("sha256") == sha256_file(student_plan_path), f"local student-plan bytes drifted: {key}")
    expect(plan_binding.get("config") == expected_config and plan_binding.get("actual_config_sha256") == canonical_json_sha256(expected_config), f"local student-plan config drifted: {key}")
    expect(plan_binding.get("compliant") is True and plan_binding.get("diagnostic") is False and plan_binding.get("scientific_launch_authorized") is False, f"local student-plan flags drifted: {key}")
    prepared_path = Path(prereg["prepared_manifest"]["path"]).resolve()
    prepared_payload = load_json(prepared_path, "prepared manifest")
    task_path = (prepared_path.parent / "roles" / source / "student_opd.jsonl").resolve()
    task_rows = load_jsonl(task_path, f"local task rows {key}")
    task_entry = (prepared_payload.get("files") or {}).get(f"roles/{source}/student_opd.jsonl") or {}
    matched_budget = (prepared_payload.get("primary_matched_budgets") or {}).get("student_opd")
    expect(
        isinstance(task_entry.get("rows"), int)
        and task_entry["rows"] > 0
        and task_entry.get("sha256") == sha256_file(task_path)
        and len(task_rows) == task_entry["rows"]
        and matched_budget == task_entry["rows"],
        f"local prepared task budget drifted: {key}",
    )
    expected_task_pool_rows = int(task_entry["rows"])
    expect(run.get("task_file") == str(task_path) and run.get("task_file_sha256") == sha256_file(task_path), f"local task-file custody drifted: {key}")
    expect(
        (
            run.get("eligible_task_pool_rows"),
            run.get("selected_task_rows"),
            run.get("task_limit"),
        )
        == (expected_task_pool_rows, 100, matched_budget),
        f"local matched task budget drifted: {key}",
    )
    prompt_key = arm["prompt_plan_key"]
    prompt_binding = prereg["prompt_plans"][prompt_key]
    prompt, ordered_rows = validate_prompt_plan(
        prompt_binding["path"],
        rows=task_rows,
        source=source,
        seed=seed,
        task_file=task_path,
        prepared_manifest=prepared_path,
        git_commit=EXPECTED_COMMIT,
        steps=100,
        diagnostic=False,
    )
    expect(prompt == binding.get("objective_family_prompt_plan") and prompt_binding == {
        "path": prompt["path"], "sha256": prompt["sha256"], "sequence_sha256": prompt["sequence_sha256"]
    }, f"local prompt custody drifted: {key}")
    init_binding = prereg["initial_adapters"][arm["initial_adapter_key"]]
    initialization = validate_initialization_manifest(
        init_binding["manifest_path"],
        student=EXPECTED_STUDENT,
        student_revision=EXPECTED_STUDENT_REVISION,
        seed=seed,
        lora_r=32,
        git_commit=EXPECTED_COMMIT,
    )
    expect(initialization == binding.get("objective_family_initialization"), f"local initialization custody drifted: {key}")
    expect(init_binding == {
        "manifest_path": init_binding["manifest_path"],
        "manifest_sha256": sha256_file(init_binding["manifest_path"]),
        "adapter_path": initialization["adapter_path"],
        "adapter_tree_sha256": initialization["adapter_tree_sha256"],
    }, f"local preregistered initialization drifted: {key}")
    sampled_k1 = bool(objective["sampled_k1"])
    expect(binding.get("teacher_source") == ("O" if sampled_k1 else None), f"local teacher routing drifted: {key}")
    expect(binding.get("pair_id") == (f"O_{source}" if sampled_k1 else None), f"local pair routing drifted: {key}")
    launcher = (repo / "scripts/hpc/slurm_opd_math_objective_family_train.sh").resolve()
    expect(binding.get("objective_family_launcher") == {
        "path": str(launcher), "sha256": sha256_file(launcher)
    }, f"local objective-family launcher drifted: {key}")
    expect(run.get("gates", {}).get("prepared_data") == {
        "path": str(prepared_path),
        "sha256": sha256_file(prepared_path),
        "task_role_file": f"roles/{source}/student_opd.jsonl",
        "task_file_sha256": sha256_file(task_path),
        "scientific_use_allowed": True,
    }, f"local prepared-data gate drifted: {key}")
    support_path = Path(prereg["student_support"][source]["path"]).resolve()
    support_payload = load_json(support_path, f"local support gate {key}")
    support_payload["manifest_sha256"] = sha256_file(support_path)
    expect(run.get("gates", {}).get("student_support") == support_payload, f"local support gate drifted: {key}")
    environment = binding.get("environment_contract") or {}
    expect(environment.get("schema_version") == 2 and environment.get("git_commit") == EXPECTED_COMMIT, f"local environment contract drifted: {key}")
    train_freeze = prereg["environment_freezes"]["train"]
    expect((environment.get("train_freeze") or {}).get("path") == train_freeze["path"] and (environment.get("train_freeze") or {}).get("sha256") == train_freeze["sha256"], f"local train freeze drifted: {key}")
    expect(reverify_recorded_environment(environment["train_verification"], in_process=False) == environment["train_verification"], f"local train environment revalidation drifted: {key}")
    if sampled_k1:
        tokenizer_contract_path = Path(f"{root}.tokenizer_contract.json").resolve()
        server_contract_path = Path(f"{root}.server_scoring_contract.json").resolve()
        server_models_path = Path(f"{root}.server_models.json").resolve()
        vllm_log_path = Path(f"{root}.vllm.log").resolve()
        tokenizer_contract_payload = load_json(tokenizer_contract_path, f"local tokenizer contract {key}")
        server_contract_payload = load_json(server_contract_path, f"local server scoring contract {key}")
        tokenizer_gate = dict(tokenizer_contract_payload)
        tokenizer_gate["manifest_sha256"] = sha256_file(tokenizer_contract_path)
        server_gate = dict(server_contract_payload)
        server_gate["manifest_sha256"] = sha256_file(server_contract_path)
        expect(run.get("gates", {}).get("tokenizer_contract") == tokenizer_gate, f"local tokenizer-contract bytes drifted: {key}")
        expect(run.get("gates", {}).get("server_scoring_contract") == server_gate, f"local server-contract bytes drifted: {key}")
        expect(
            completion.get("local_server_process_binding_end")
            == server_contract_payload.get("local_process_binding"),
            f"local server-process end binding drifted: {key}",
        )
        expect(server_models_path.is_file() and not server_models_path.is_symlink(), f"local server-model evidence missing: {key}")
        expect(vllm_log_path.is_file() and not vllm_log_path.is_symlink(), f"local vLLM evidence missing: {key}")
        serve_freeze = prereg["environment_freezes"]["serve"]
        expect((environment.get("serve_freeze") or {}).get("path") == serve_freeze["path"] and (environment.get("serve_freeze") or {}).get("sha256") == serve_freeze["sha256"], f"local serve freeze drifted: {key}")
        expect(reverify_recorded_environment(environment["serve_verification"], in_process=False) == environment["serve_verification"], f"local serve environment revalidation drifted: {key}")
        teacher_path = Path(prereg["o_teacher"]["teacher_gap_manifest"]).resolve()
        teacher_payload = load_json(teacher_path, f"local O teacher gap {key}")
        teacher_payload["manifest_sha256"] = sha256_file(teacher_path)
        expect(run.get("gates", {}).get("teacher_gap") == teacher_payload, f"local O teacher gap drifted: {key}")
        expect(Path(str(run.get("teacher_checkpoint"))).resolve() == Path(prereg["o_teacher"]["merged_checkpoint"]).resolve(), f"local teacher checkpoint drifted: {key}")
        expect(run.get("teacher_base_model") == prereg["o_teacher"]["base_model"] and run.get("teacher_base_revision") == prereg["o_teacher"]["base_revision"], f"local teacher base identity drifted: {key}")
        expect(run.get("gates", {}).get("teacher_provenance", {}).get("manifest_sha256") == prereg["o_teacher"]["merge_provenance_manifest_sha256"], f"local teacher provenance drifted: {key}")
        expect(completion.get("local_server_process_binding_required") is True and completion.get("live_local_server_process_binding_validated") is True and completion.get("local_server_process_binding_error") is None, f"local teacher process custody failed: {key}")
    else:
        tokenizer_contract_path = None
        server_contract_path = None
        server_models_path = None
        vllm_log_path = None
        expect(environment.get("serve_freeze") is None and environment.get("serve_verification") is None, f"task-RL bound a serve environment: {key}")
        for field in ("teacher_gap", "teacher_provenance", "server_scoring_contract", "tokenizer_contract"):
            expect(run.get("gates", {}).get(field) is None, f"task-RL bound {field}: {key}")
        expect(run.get("teacher_model") is None and run.get("teacher_checkpoint") is None and run.get("teacher_base_model") is None and run.get("teacher_base_revision") is None, f"task-RL bound teacher identity: {key}")
    prelaunch_args = SimpleNamespace(
        prelaunch_receipt=arm["prelaunch_receipt"],
        objective_registry_contract=registry_binding,
        student_source=source,
        seed=seed,
        out_dir=str(root),
        campaign_run_id=arm["run_id"],
        scheduler_job_id=scheduler_job_id,
    )
    prelaunch = validate_prelaunch_receipt(prelaunch_args)
    expect(prelaunch.get("run_key") == key, f"local prelaunch validation drifted: {key}")
    for field, expected in (
        ("status", "completed"),
        ("objective_family_diagnostic", False),
        ("intended_scientific_run", True),
        ("optimizer_steps_completed", 100),
        ("rollout_samples", 400),
        ("step_trace_rows", 100),
        ("sample_trace_rows", 400),
        ("realized_training_geometry_observed", True),
        ("finite_nonzero_gradient_observed", True),
        ("parameter_update_observed", True),
        ("clean_stable_code", True),
        ("stable_training_environment", True),
        ("stable_environment_end", True),
        ("stable_final_artifact_hash", True),
        ("training_artifact_eligible_for_held_out_evaluation", True),
    ):
        expect(completion.get(field) == expected, f"local completion {field} drifted: {key}")
    expect(completion.get("git_state_start") == {"commit": EXPECTED_COMMIT, "dirty": False}, f"local start Git state drifted: {key}")
    for field in ("git_state_training_end", "git_state_after_candidate_save", "git_state_end"):
        expect(completion.get(field) == {"commit": EXPECTED_COMMIT, "dirty": False}, f"local {field} drifted: {key}")
    expect(completion.get("stable_environment_after_candidate_save") is True, f"local post-save environment drifted: {key}")
    expect(completion.get("initial_parameter_signature") == initialization["trainable_parameter_signature"], f"local initial parameter signature drifted: {key}")
    updates = completion.get("parameter_update_l2_by_step")
    expect(isinstance(updates, list) and len(updates) == 100, f"local update vector drifted: {key}")
    update_values = [_finite_number(value, f"local update {key}") for value in updates]
    expect(all(value >= 0.0 for value in update_values) and any(value > 0.0 for value in update_values), f"local update signal invalid: {key}")
    expect(completion.get("initial_parameter_signature") != completion.get("final_parameter_signature"), f"local final parameter signature did not change: {key}")
    optimizer = completion.get("optimizer_state_signature_final") or {}
    expect(isinstance(optimizer.get("tensors"), int) and optimizer["tensors"] > 0, f"local optimizer state missing: {key}")
    _finite_number(optimizer.get("squared_l2"), f"local optimizer state {key}", positive=True)
    final = Path(str(completion.get("final_adapter"))).resolve()
    expect(final == root / "final" and final.is_dir() and not final.is_symlink(), f"local final adapter path drifted: {key}")
    final_hash = sha256_tree(final)
    expect(completion.get("final_adapter_tree_sha256") == final_hash, f"local final adapter hash drifted: {key}")
    adapter_delta = adapter_delta_l2(initialization["adapter_path"], final)
    steps_path, steps = _trace_binding(completion, "steps.jsonl", 100)
    samples_path, samples = _trace_binding(completion, "samples.jsonl", 400)
    expect([row.get("step") for row in steps] == list(range(1, 101)), f"local step sequence drifted: {key}")
    expect(all(row.get("source") == source for row in samples), f"local sample source drifted: {key}")
    gradient_values = []
    step_update_values = []
    for row in steps:
        gradient_values.append(_finite_number(row.get("gradient_norm_before_clip"), f"local gradient {key}"))
        step_update_values.append(_finite_number(row.get("parameter_update_l2"), f"local step update {key}"))
    expect(all(value >= 0.0 for value in gradient_values) and any(value > 0.0 for value in gradient_values), f"local gradient signal invalid: {key}")
    expect(all(value >= 0.0 for value in step_update_values) and any(value > 0.0 for value in step_update_values), f"local step-update signal invalid: {key}")
    tokenizer = pinned_tokenizer()
    expected_groups: dict[tuple[int, int], dict[str, Any]] = {}
    for step, row in enumerate(ordered_rows, 1):
        _, prompt_token_ids = render_prompt(
            tokenizer,
            row,
            int(expected_config["max_prompt_tokens"]),
            False,
        )
        expected_groups[(step, 0)] = {
            "record_id": row["record_id"],
            "source": source,
            "prompt_sha256": task_prompt_sha256(row),
            "prompt_token_ids": prompt_token_ids,
        }
    replay = recompute_student_trace_geometry(
        steps_path=steps_path,
        samples_path=samples_path,
        mode=objective_id,
        expected_steps=100,
        micro_prompts=1,
        group_size=int(expected_config["group_size"]),
        max_prompt_tokens=int(expected_config["max_prompt_tokens"]),
        max_completion_tokens=int(expected_config["max_new_tokens"]),
        expected_groups=expected_groups,
        tokenizer=tokenizer,
        require_behavior_logprobs=True,
        loss_config={
            "task_reward_coef": objective["task_reward_coef"],
            "k1_coef": objective["k1_coef"],
            "gap_gate_beta": objective["gap_gate_beta"],
            "advantage_clip": objective["advantage_clip"],
        },
    )
    for field in (
        "step_trace_rows", "sample_trace_rows", "rollout_samples",
        "unique_training_records", "realized_record_ids_sha256",
        "realized_prompt_sequence_sha256", "prompt_group_tokens",
        "sample_expanded_prompt_tokens", "completion_tokens",
    ):
        completion_field = "scored_completion_tokens" if field == "completion_tokens" else field
        expect(completion.get(completion_field) == replay[field], f"local replay aggregate {field} drifted: {key}")
    expect(completion.get("prompt_groups_seen") == replay["prompt_groups"] == 100, f"local prompt-group replay drifted: {key}")
    rows_by_id = {row["record_id"]: row for row in task_rows}
    if bool(objective["task_reward"]):
        for row_number, sample in enumerate(samples, 1):
            source_row = rows_by_id[sample["record_id"]]
            verified = verify_completion(sample["completion_text"], source_row["solution"])
            expect(verified.get("reward") is not None, f"local reward revalidation failed: {key} row {row_number}")
            expect(float(verified["reward"]) == float(sample["reward"]) and verified["status"] == sample["reward_status"], f"local reward drifted: {key} row {row_number}")
    else:
        expect(all(sample.get("reward") is None and sample.get("reward_status") is None for sample in samples), f"local bare-K1 trace contains task rewards: {key}")
    if bool(objective["task_reward"]):
        fraction = _finite_number(completion.get("informative_group_fraction"), f"local informative fraction {key}")
        expect(fraction >= 0.05 and completion.get("minimum_informative_group_fraction") == 0.05, f"local informative-group gate failed: {key}")
        expect(completion.get("total_task_groups") == 100 and completion.get("task_signal_observed") is True, f"local task-signal geometry drifted: {key}")
    else:
        expect(completion.get("total_task_groups") == 0 and completion.get("informative_group_fraction") == 1.0, f"local bare-K1 task counters drifted: {key}")
    return {
        "implementation": "local",
        "run_manifest": file_binding(run_path, readonly=False),
        "completion_manifest": file_binding(completion_path, readonly=False),
        "prelaunch_receipt": prelaunch["path"] if isinstance(prelaunch.get("path"), str) else arm["prelaunch_receipt"],
        "final_adapter": {"path": str(final), "tree_sha256": final_hash},
        "adapter_delta": adapter_delta,
        "step_trace": {"path": str(steps_path), "sha256": sha256_file(steps_path), "rows": len(steps)},
        "sample_trace": {"path": str(samples_path), "sha256": sha256_file(samples_path), "rows": len(samples)},
        "scheduler_job_id": scheduler_job_id,
        "external_artifacts": {
            "local_k1": (
                None
                if not sampled_k1
                else {
                    "tokenizer_contract": file_binding(tokenizer_contract_path, readonly=False),
                    "server_scoring_contract": file_binding(server_contract_path, readonly=False),
                    "server_models": file_binding(server_models_path, readonly=False),
                    "vllm_log": file_binding(vllm_log_path, readonly=False),
                }
            ),
            "upstream_verl": None,
        },
        "validation_method": {
            "kind": "d89_same_code_integrity_replay_v1",
            "independent_implementation": False,
            "local_trace_geometry_replayed": True,
            "local_math_rewards_reverified": bool(objective["task_reward"]),
        },
        "cost": {
            "training_elapsed_seconds": _finite_number(completion.get("total_training_elapsed_seconds"), f"training elapsed {key}", positive=True),
            "rollout_latency_seconds": _finite_number(completion.get("total_rollout_latency_seconds"), f"rollout latency {key}", positive=True),
            "teacher_scoring_latency_seconds": _finite_number(completion.get("total_teacher_scoring_latency_seconds"), f"teacher latency {key}"),
            "scored_completion_tokens": int(completion.get("scored_completion_tokens")),
            "prompt_group_tokens": int(completion.get("prompt_group_tokens")),
            "peak_cuda_memory_bytes": completion.get("peak_cuda_memory_bytes"),
        },
    }


def validate_upstream_training(
    *, repo: Path, key: str, arm: Mapping[str, Any], prereg: Mapping[str, Any]
) -> dict[str, Any]:
    from scripts.opd.objective_family_preregistration import validate_upstream_prelaunch_receipt  # type: ignore
    from scripts.opd.objective_family_inputs import validate_initialization_manifest  # type: ignore
    from scripts.opd.prepare_verl_objective_data import validate_dataset  # type: ignore
    from scripts.opd.verl_objective_contract import load_plan  # type: ignore
    from scripts.opd import verl_run_custody as verl_custody  # type: ignore
    from scripts.opd.verl_run_custody import RECEIPT_ID  # type: ignore
    from scripts.opd.objective_registry import UPSTREAM_VERL_COMMIT  # type: ignore
    from scripts.opd_math.verify_environment import run_external_environment_verification  # type: ignore

    root = Path(arm["training_out"]).resolve()
    preflight_path = Path(str(root) + ".preflight.json")
    receipt_path = Path(str(root) + ".receipt.json")
    preflight_file = regular_readonly(preflight_path, f"upstream preflight {key}")
    preflight = load_json(preflight_file, f"upstream preflight {key}")
    receipt = load_json(receipt_path, f"upstream receipt {key}")
    expect(set(receipt) == {
        "schema_version", "receipt", "status", "scientific_use_allowed",
        "training_artifact_eligible_for_heldout_evaluation", "objective_id",
        "source", "seed", "optimizer_steps", "git_commit",
        "upstream_verl_commit", "preflight", "run_log", "actor_checkpoint",
        "optimizer", "rollouts", "adapter_update", "final_adapter", "metrics",
        "finite_nonzero_gradient_observed", "parameter_update_observed",
        "optimizer_state_observed", "heldout_outcomes_inspected", "claim_boundary",
    }, f"upstream receipt schema drifted: {key}")
    objective_id = str(arm["objective_id"])
    source = str(arm["source"])
    seed = int(arm["seed"])
    prelaunch_identity = _prelaunch_job_identity(arm["prelaunch_receipt"], key)
    prelaunch = validate_upstream_prelaunch_receipt(
        arm["prelaunch_receipt"],
        objective_id=objective_id,
        source=source,
        seed=seed,
        out_dir=root,
        run_id=arm["run_id"],
        scheduler_job_id=prelaunch_identity["job_id"],
    )
    expected_preflight_keys = {
        "schema_version", "preflight", "status", "scientific_launch_authorized",
        "campaign_kind", "objective_id", "source", "seed", "optimizer_steps",
        "scheduler_job_id", "run_id", "git_commit", "git_tracked_clean",
        "objective_plan", "objective_registry_sha256", "launcher", "upstream_verl",
        "environment", "student", "student_revision", "initialization", "data",
        "student_support", "o_teacher", "output_root", "prelaunch_receipt",
        "heldout_outcomes_inspected", "claim_boundary",
    }
    expect(set(preflight) == expected_preflight_keys, f"upstream preflight schema drifted: {key}")
    objective_plan = load_plan()
    for field, expected in (
        ("schema_version", 1),
        ("preflight", verl_custody.PREFLIGHT_ID),
        ("status", "validated_before_optimizer_start"),
        ("scientific_launch_authorized", True),
        ("campaign_kind", "scientific"),
        ("objective_id", objective_id),
        ("source", source),
        ("seed", seed),
        ("optimizer_steps", 100),
        ("scheduler_job_id", prelaunch_identity["job_id"]),
        ("run_id", arm["run_id"]),
        ("git_commit", EXPECTED_COMMIT),
        ("git_tracked_clean", True),
        ("objective_plan", {"path": objective_plan["path"], "sha256": objective_plan["sha256"]}),
        ("objective_registry_sha256", objective_plan["registry"]["sha256"]),
        ("student", EXPECTED_STUDENT),
        ("student_revision", EXPECTED_STUDENT_REVISION),
        ("output_root", str(root)),
        ("prelaunch_receipt", prelaunch),
        ("heldout_outcomes_inspected", False),
    ):
        expect(preflight.get(field) == expected, f"upstream preflight {field} drifted: {key}")
    launcher = (repo / "scripts/hpc/slurm_opd_math_objective_family_verl.sh").resolve()
    expect(preflight.get("launcher") == {"path": str(launcher), "sha256": sha256_file(launcher)}, f"upstream launcher drifted: {key}")
    upstream = preflight.get("upstream_verl") or {}
    upstream_checkout = Path(str(upstream.get("checkout"))).resolve()
    expect(verl_custody._git_state(upstream_checkout) == {"commit": UPSTREAM_VERL_COMMIT, "tracked_clean": True}, f"upstream veRL checkout drifted: {key}")
    expect(upstream.get("commit") == UPSTREAM_VERL_COMMIT and upstream.get("tracked_clean") is True, f"upstream veRL identity drifted: {key}")
    expect(upstream.get("core_files") == {
        relative: sha256_file(upstream_checkout / relative)
        for relative in verl_custody.UPSTREAM_CORE_FILES
    }, f"upstream veRL core-file hashes drifted: {key}")
    environment = preflight.get("environment") or {}
    upstream_freeze = prereg["environment_freezes"]["upstream_verl"]
    expect(environment.get("freeze") == upstream_freeze["path"] and environment.get("freeze_sha256") == upstream_freeze["sha256"], f"upstream environment freeze drifted: {key}")
    run_external_environment_verification(
        environment_root=environment["root"],
        commit_freeze=environment["freeze"],
        expected_commit=EXPECTED_COMMIT,
        freeze_kind="upstream_verl",
    )
    init_binding = prereg["initial_adapters"][arm["initial_adapter_key"]]
    initialization = validate_initialization_manifest(
        init_binding["manifest_path"],
        student=EXPECTED_STUDENT,
        student_revision=EXPECTED_STUDENT_REVISION,
        seed=seed,
        lora_r=32,
        git_commit=EXPECTED_COMMIT,
    )
    expect(preflight.get("initialization") == initialization, f"upstream initialization drifted: {key}")
    data = preflight.get("data") or {}
    prompt_binding = prereg["prompt_plans"][arm["prompt_plan_key"]]
    recomputed_data = validate_dataset(
        task_file=Path(prereg["prepared_manifest"]["path"]).parent / "roles" / source / "student_opd.jsonl",
        prepared_manifest=Path(prereg["prepared_manifest"]["path"]),
        prompt_plan=Path(prompt_binding["path"]),
        source=source,
        seed=seed,
        git_commit=EXPECTED_COMMIT,
        diagnostic=False,
        output=Path(data["path"]),
        manifest_path=Path(data["manifest_path"]),
    )
    expect(data == recomputed_data, f"upstream materialized data drifted: {key}")
    expect(preflight.get("student_support") == prereg["student_support"][source], f"upstream support identity drifted: {key}")
    expect(preflight.get("o_teacher") == prereg["o_teacher"], f"upstream O-teacher identity drifted: {key}")
    for field, expected in (
        ("schema_version", 1),
        ("receipt", RECEIPT_ID),
        ("status", "completed_training_pending_heldout"),
        ("scientific_use_allowed", False),
        ("training_artifact_eligible_for_heldout_evaluation", True),
        ("objective_id", objective_id),
        ("source", source),
        ("seed", seed),
        ("optimizer_steps", 100),
        ("git_commit", EXPECTED_COMMIT),
        ("upstream_verl_commit", UPSTREAM_VERL_COMMIT),
        ("finite_nonzero_gradient_observed", True),
        ("parameter_update_observed", True),
        ("optimizer_state_observed", True),
        ("heldout_outcomes_inspected", False),
    ):
        expect(receipt.get(field) == expected, f"upstream receipt {field} drifted: {key}")
    final = receipt.get("final_adapter") or {}
    final_path = Path(str(final.get("path"))).resolve()
    expect(final_path == root / "final" and final_path.is_dir() and not final_path.is_symlink(), f"upstream final path drifted: {key}")
    final_hash = sha256_tree(final_path)
    expect(final.get("tree_sha256") == final_hash, f"upstream final hash drifted: {key}")
    expect(receipt.get("preflight") == {"path": str(preflight_file), "sha256": sha256_file(preflight_file)}, f"upstream receipt preflight binding drifted: {key}")
    run_log = Path(str((receipt.get("run_log") or {}).get("path"))).resolve()
    actor_checkpoint = Path(str((receipt.get("actor_checkpoint") or {}).get("path"))).resolve()
    rollout_dir = (root / "rollouts").resolve()
    expect(run_log == (root / "run.log").resolve() and actor_checkpoint == (root / "checkpoints/global_step_100/actor").resolve(), f"upstream artifact paths drifted: {key}")
    expect(receipt.get("run_log") == {"path": str(run_log), "sha256": sha256_file(run_log)}, f"upstream run-log binding drifted: {key}")
    expect(receipt.get("actor_checkpoint") == {"path": str(actor_checkpoint), "tree_sha256": sha256_tree(actor_checkpoint)}, f"upstream actor checkpoint drifted: {key}")
    metrics = verl_custody._parse_metrics(run_log, 100)
    expect(receipt.get("metrics") == metrics, f"upstream metrics replay drifted: {key}")
    rollouts = verl_custody._rollout_custody(rollout_dir, 100)
    expect(receipt.get("rollouts") == rollouts and rollouts.get("rows") == 400, f"upstream rollout replay drifted: {key}")
    optimizer = verl_custody._optimizer_custody(actor_checkpoint)
    expect(receipt.get("optimizer") == optimizer, f"upstream optimizer replay drifted: {key}")
    update = verl_custody._adapter_delta(Path(initialization["adapter_path"]), final_path)
    expect(receipt.get("adapter_update") == update, f"upstream adapter-delta replay drifted: {key}")
    expect(receipt.get("final_adapter") == {"path": str(final_path), "tree_sha256": final_hash}, f"upstream final-adapter binding drifted: {key}")
    return {
        "implementation": "upstream_verl",
        "preflight": file_binding(preflight_file, readonly=True),
        "native_receipt": file_binding(receipt_path, readonly=True),
        "prelaunch_receipt": {"path": prelaunch_identity["path"], "sha256": prelaunch_identity["sha256"]},
        "final_adapter": {"path": str(final_path), "tree_sha256": final_hash},
        "adapter_delta": update,
        "scheduler_job_id": prelaunch_identity["job_id"],
        "external_artifacts": {
            "local_k1": None,
            "upstream_verl": {
                "preflight": file_binding(preflight_file, readonly=True),
                "native_receipt": file_binding(receipt_path, readonly=True),
            },
        },
        "validation_method": {
            "kind": "d89_same_code_integrity_replay_v1",
            "independent_implementation": False,
            "upstream_private_helpers_replayed": True,
        },
        "cost": {
            "training_elapsed_seconds": None,
            "rollout_latency_seconds": None,
            "teacher_scoring_latency_seconds": None,
            "scored_completion_tokens": None,
            "prompt_group_tokens": None,
            "peak_cuda_memory_bytes": None,
        },
    }


def _evaluation_scope(plan: Mapping[str, Any], *, arm_key_value: str | None, raw_source: str | None) -> tuple[str, dict[str, str]]:
    expect((arm_key_value is None) != (raw_source is None), "select exactly one arm or raw source")
    if arm_key_value is not None:
        expect(arm_key_value in plan["arm_paths"], "evaluation arm is not preregistered")
        paths = plan["arm_paths"][arm_key_value]
        return arm_key_value, {
            "summary": paths["evaluation_summary"],
            "samples": paths["evaluation_samples"],
            "companion": paths["evaluation_companion"],
            "authorization": paths["evaluation_authorization"],
            "submission_receipt": paths["evaluation_submission_receipt"],
            "merge_consumption_receipt": paths["evaluation_merge_consumption_receipt"],
            "merge_supervisor_receipt": paths["evaluation_merge_supervisor_receipt"],
            "seal_supervisor_receipt": paths["evaluation_seal_supervisor_receipt"],
            "terminal_failure_receipt": paths["evaluation_terminal_failure_receipt"],
            "array_accounting_receipt": paths["evaluation_array_accounting_receipt"],
            "array_accounting_raw": paths["evaluation_array_accounting_raw"],
            "consumption_receipt": paths["evaluation_consumption_receipt"],
            "seal_receipt": paths["evaluation_seal_receipt"],
            "shard_consumption_root": paths["evaluation_shard_consumption_root"],
            "log_root": paths["evaluation_log_root"],
            "private_log_root": paths["evaluation_private_log_root"],
            "artifact_root": paths["artifact_root"],
        }
    assert raw_source is not None
    expect(raw_source in SOURCES, "raw evaluation source is invalid")
    paths = plan["raw_student_auxiliary"][raw_source]
    return f"raw_student__{raw_source}", {
        "summary": paths["summary"],
        "samples": paths["samples"],
        "companion": paths["companion"],
        "authorization": paths["authorization"],
        "submission_receipt": paths["submission_receipt"],
        "merge_consumption_receipt": paths["merge_consumption_receipt"],
        "merge_supervisor_receipt": paths["merge_supervisor_receipt"],
        "seal_supervisor_receipt": paths["seal_supervisor_receipt"],
        "terminal_failure_receipt": paths["terminal_failure_receipt"],
        "array_accounting_receipt": paths["array_accounting_receipt"],
        "array_accounting_raw": paths["array_accounting_raw"],
        "consumption_receipt": paths["consumption_receipt"],
        "seal_receipt": paths["seal_receipt"],
        "shard_consumption_root": paths["shard_consumption_root"],
        "log_root": paths["log_root"],
        "private_log_root": paths["private_log_root"],
        "artifact_root": paths["artifact_root"],
    }


def expected_evaluation_contract(
    *,
    repo: Path,
    plan: Mapping[str, Any],
    prereg: Mapping[str, Any],
    scope: str,
    paths: Mapping[str, str],
    training_custody: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if scope in plan["arm_paths"]:
        arm = prereg["payload"]["arms"][scope]
        source = arm["source"]
        target = plan["arm_paths"][scope]
        adapter = training_custody["final_adapter"] if training_custody is not None else None
    else:
        expect(scope in {"raw_student__M", "raw_student__O"}, "evaluation scope is invalid")
        source = scope.rsplit("__", 1)[1]
        target = plan["raw_student_auxiliary"][source]
        adapter = None
    program_manifest = load_json(plan["program_manifest"]["path"], "program manifest")
    prepared_path = Path(prereg["prepared_manifest"]["path"]).resolve()
    return {
        "scope": scope,
        "source": source,
        "role": "source_holdout",
        "model": EXPECTED_STUDENT,
        "model_revision": EXPECTED_STUDENT_REVISION,
        "adapter": adapter,
        "task": plan["holdout_selection"][source],
        "selected_records": SELECTED_HOLDOUT_RECORDS,
        "samples_per_problem": 4,
        "decoding": HELDOUT_DECODING,
        "shards": EVALUATION_SHARDS,
        "array_spec": EVALUATION_ARRAY_SPEC,
        "shard_strategy": "contiguous_balanced_v1",
        "merge_strategy": "ordered_contiguous_shards_v1",
        "label": target["label"],
        "run_id": target["run_id"],
        "artifact_root": paths["artifact_root"],
        "expected_summary": paths["summary"],
        "expected_samples": paths["samples"],
        "expected_companion": paths["companion"],
        "submission_receipt": paths["submission_receipt"],
        "merge_consumption_receipt": paths["merge_consumption_receipt"],
        "merge_supervisor_receipt": paths["merge_supervisor_receipt"],
        "seal_supervisor_receipt": paths["seal_supervisor_receipt"],
        "terminal_failure_receipt": paths["terminal_failure_receipt"],
        "array_accounting_receipt": paths["array_accounting_receipt"],
        "array_accounting_raw": paths["array_accounting_raw"],
        "consumption_receipt": paths["consumption_receipt"],
        "seal_receipt": paths["seal_receipt"],
        "shard_consumption_root": paths["shard_consumption_root"],
        "log_root": paths["log_root"],
        "private_log_root": paths["private_log_root"],
        "environment": {
            "root": plan["train_environment_root"],
            "preregistered_freeze": prereg["environment_freezes"]["train"],
            "runtime_freeze": plan["runtime_train_freeze"]["runtime"],
            "hf_home": plan["hf_home"],
        },
        "code": {
            "release_program": program_manifest["program_file"],
            "evaluation_wrapper": program_manifest["evaluation_wrapper"],
            "tracked_evaluator_wrapper": file_binding(repo / "scripts/hpc/slurm_opd_math_evaluate.sh", readonly=False),
            "tracked_merge_wrapper": file_binding(repo / "scripts/hpc/slurm_opd_math_merge_evaluation.sh", readonly=False),
            "evaluator": file_binding(repo / "scripts/opd_math/evaluate_math.py", readonly=False),
            "merger": file_binding(repo / "scripts/opd_math/merge_evaluations.py", readonly=False),
        },
        "exports": {
            "OPD_MATH_REPO": str(repo),
            "OPD_MATH_TRAIN_ENV": plan["train_environment_root"],
            "OPD_MATH_RUN_ROOT": str(Path(plan["evaluation_root"]).parent),
            "OPD_MATH_HF_HOME": plan["hf_home"],
            "OPD_MATH_DATA_ROOT": str(prepared_path.parent),
            "OPD_MATH_EVAL_SOURCE": source,
            "OPD_MATH_EVAL_ROLE": "source_holdout",
            "OPD_MATH_EVAL_MODEL": EXPECTED_STUDENT,
            "OPD_MATH_EVAL_MODEL_REVISION": EXPECTED_STUDENT_REVISION,
            "OPD_MATH_EVAL_MAX_RECORDS": "370",
            "OPD_MATH_EVAL_LABEL": target["label"],
            "OPD_MATH_EVAL_RUN_ID": target["run_id"],
            "OPD_MATH_EVAL_SHARDS": "6",
            "OPD_MATH_EVAL_SAMPLES_PER_PROBLEM": "4",
            "OPD_MATH_EVAL_TEMPERATURE": "1.0",
            "OPD_MATH_EVAL_TOP_P": "1.0",
            "OPD_MATH_EVAL_TOP_K": "0",
            "OPD_MATH_EVAL_MAX_NEW_TOKENS": "512",
            "OPD_MATH_SEED": "0",
            "OPD_MATH_EVAL_ADAPTER": (
                None if adapter is None else adapter["path"]
            ),
        },
        "sanitized_scheduler_environment": {
            "HOME": "/home/compute/hiqbal",
            "SLURM_CONF": "/project/compute/slurm/etc/slurm.conf",
            "LANG": "C",
            "LC_ALL": "C",
            "TZ": "UTC",
        },
    }


def seal_tree_readonly(path: str | Path) -> None:
    raw_root = Path(path)
    expect(not raw_root.is_symlink(), f"cannot seal symlinked tree: {raw_root}")
    root = raw_root.resolve()
    expect(root.is_dir(), f"cannot seal missing tree: {root}")
    for item in root.rglob("*"):
        mode = item.lstat().st_mode
        expect(not stat.S_ISLNK(mode), f"cannot seal tree containing symlink: {item}")
        expect(stat.S_ISREG(mode) or stat.S_ISDIR(mode), f"cannot seal tree containing special node: {item}")
        current = stat.S_IMODE(mode)
        os.chmod(item, current & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    root_mode = stat.S_IMODE(root.stat().st_mode)
    os.chmod(root, root_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def require_train_python(plan: Mapping[str, Any], prereg: Mapping[str, Any]) -> dict[str, Any]:
    environment_root = Path(plan["train_environment_root"]).resolve()
    expected_python = environment_root / "bin/python"
    expect(expected_python.is_file(), "train Python is missing")
    observed_python = Path(sys.executable)
    expect(
        Path(sys.prefix).resolve() == environment_root,
        f"deep training validation has wrong sys.prefix: expected {environment_root}, observed {sys.prefix}",
    )
    expect(
        observed_python.absolute() == expected_python.absolute(),
        f"deep training validation must run under {expected_python}, observed {observed_python}",
    )
    from scripts.opd_math.verify_environment import verify_environment  # type: ignore

    live_verification = verify_environment(
        environment_root=environment_root,
        commit_freeze=plan["runtime_train_freeze"]["runtime"]["path"],
        expected_commit=EXPECTED_COMMIT,
        freeze_kind="train",
    )
    return {
        "environment_root": str(environment_root),
        "expected_python": str(expected_python),
        "expected_python_resolved": str(expected_python.resolve()),
        "observed_python": str(observed_python),
        "observed_python_resolved": str(observed_python.resolve()),
        "observed_sys_prefix": str(Path(sys.prefix).resolve()),
        "train_environment_freeze": prereg["environment_freezes"]["train"],
        "live_environment_verification": live_verification,
    }


def validate_completed_training_for_arm(
    *, repo: Path, key: str, plan: Mapping[str, Any], prereg: Mapping[str, Any]
) -> dict[str, Any]:
    arm = prereg["payload"]["arms"][key]
    if arm["implementation"] == "local":
        result = validate_local_training(repo=repo, key=key, arm=arm, prereg=prereg)
    else:
        result = validate_upstream_training(repo=repo, key=key, arm=arm, prereg=prereg)
    expect(Path(result["final_adapter"]["path"]).resolve() == Path(plan["arm_paths"][key]["training_out"]).resolve() / "final", f"training final path differs from release plan: {key}")
    return result


def _seal_bound_file(binding: Mapping[str, Any], label: str) -> dict[str, str]:
    expect(isinstance(binding, Mapping) and set(binding) == {"path", "sha256"}, f"{label} binding drifted")
    path = Path(str(binding["path"])).resolve()
    expect(path.is_file() and not path.is_symlink(), f"{label} is missing")
    expect(sha256_file(path) == binding["sha256"], f"{label} hash drifted before sealing")
    os.chmod(path, 0o444)
    return file_binding(path)


def seal_training_audit_receipt(
    *,
    repo: Path,
    plan_path: Path,
    plan: Mapping[str, Any],
    prereg: Mapping[str, Any],
    key: str,
    scheduler: Mapping[str, Any],
    submission_path: Path,
    consumption_path: Path,
    prelaunch: Mapping[str, Any],
    validator_runtime: Mapping[str, Any],
    deep_validation: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    arm = prereg["payload"]["arms"][key]
    validation = deep_validation if deep_validation is not None else validate_completed_training_for_arm(
        repo=repo, key=key, plan=plan, prereg=prereg
    )
    external = validation.get("external_artifacts")
    expect(isinstance(external, dict) and set(external) == {"local_k1", "upstream_verl"}, f"training external-artifact schema drifted: {key}")
    sealed_external: dict[str, Any] = {"local_k1": None, "upstream_verl": None}
    if external["local_k1"] is not None:
        expect(set(external["local_k1"]) == {"tokenizer_contract", "server_scoring_contract", "server_models", "vllm_log"}, f"local K1 external-artifact schema drifted: {key}")
        sealed_external["local_k1"] = {
            label: _seal_bound_file(binding, f"{key} {label}")
            for label, binding in external["local_k1"].items()
        }
    if external["upstream_verl"] is not None:
        expect(set(external["upstream_verl"]) == {"preflight", "native_receipt"}, f"upstream external-artifact schema drifted: {key}")
        sealed_external["upstream_verl"] = {
            label: _seal_bound_file(binding, f"{key} {label}")
            for label, binding in external["upstream_verl"].items()
        }
    validation["external_artifacts"] = sealed_external
    training_root = Path(plan["arm_paths"][key]["training_out"]).resolve()
    training_tree_sha256 = sha256_tree(training_root)
    seal_tree_readonly(training_root)
    expect(sha256_tree(training_root) == training_tree_sha256, f"training tree changed while sealing: {key}")
    validation["training_root"] = {
        "path": str(training_root),
        "tree_sha256": training_tree_sha256,
        "sealed_read_only": True,
    }
    audit = {
        "schema_version": SCHEMA_VERSION,
        "receipt": "opd_math_objective_family_training_audit_v1",
        "status": "deep_validated_and_sealed_before_heldout",
        "created_utc": utc_now(),
        "campaign_id": plan["campaign_id"],
        "arm_key": key,
        "objective_id": arm["objective_id"],
        "implementation": arm["implementation"],
        "source": arm["source"],
        "seed": arm["seed"],
        "run_id": arm["run_id"],
        "git_commit": EXPECTED_COMMIT,
        "release_plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
        "program_manifest": plan["program_manifest"],
        "preregistration": plan["preregistration"],
        "launch_plan": plan["launch_plan"],
        "submission_receipt": {"path": str(submission_path), "sha256": sha256_file(submission_path)},
        "training_consumption_receipt": {"path": str(consumption_path), "sha256": sha256_file(consumption_path)},
        "prelaunch_receipt": {"path": prelaunch["path"], "sha256": prelaunch["sha256"]},
        "scheduler_terminal": dict(scheduler),
        "validator_runtime": dict(validator_runtime),
        "validation_method": validation["validation_method"],
        "training_root": validation["training_root"],
        "consumable_adapter": validation["final_adapter"],
        "external_artifacts": sealed_external,
        "deep_validation": validation,
        "heldout_outcomes_inspected": False,
        "scientific_result_claimed": False,
        "claim_boundary": "Deep training integrity validation only; held-out task performance is not established.",
    }
    audit_path = write_new_or_exact(plan["arm_paths"][key]["terminal_audit_receipt"], audit)
    return audit_path, audit


def validate_training_audit_receipt(
    *,
    plan_path: Path,
    plan: Mapping[str, Any],
    prereg: Mapping[str, Any],
    key: str,
    terminal_arm: Mapping[str, Any] | None = None,
    mode: str = "metadata",
) -> tuple[Path, dict[str, Any]]:
    expect(mode in {"metadata", "selected_root", "adapter"}, "training-audit validation mode is invalid")
    path = regular_readonly(plan["arm_paths"][key]["terminal_audit_receipt"], f"training audit {key}")
    payload = load_json(path, f"training audit {key}")
    expected_keys = {
        "schema_version", "receipt", "status", "created_utc", "campaign_id", "arm_key",
        "objective_id", "implementation", "source", "seed", "run_id", "git_commit",
        "release_plan", "program_manifest", "preregistration", "launch_plan",
        "submission_receipt", "training_consumption_receipt", "prelaunch_receipt",
        "scheduler_terminal", "validator_runtime", "validation_method", "training_root",
        "consumable_adapter", "external_artifacts", "deep_validation",
        "heldout_outcomes_inspected", "scientific_result_claimed", "claim_boundary",
    }
    expect(set(payload) == expected_keys, f"training audit schema drifted: {key}")
    arm = prereg["payload"]["arms"][key]
    for field, expected in (
        ("schema_version", SCHEMA_VERSION),
        ("receipt", "opd_math_objective_family_training_audit_v1"),
        ("status", "deep_validated_and_sealed_before_heldout"),
        ("campaign_id", plan["campaign_id"]),
        ("arm_key", key),
        ("objective_id", arm["objective_id"]),
        ("implementation", arm["implementation"]),
        ("source", arm["source"]),
        ("seed", arm["seed"]),
        ("run_id", arm["run_id"]),
        ("git_commit", EXPECTED_COMMIT),
        ("release_plan", {"path": str(plan_path), "sha256": sha256_file(plan_path)}),
        ("program_manifest", plan["program_manifest"]),
        ("preregistration", plan["preregistration"]),
        ("launch_plan", plan["launch_plan"]),
        ("heldout_outcomes_inspected", False),
        ("scientific_result_claimed", False),
    ):
        expect(payload.get(field) == expected, f"training audit {field} drifted: {key}")
    parse_utc(payload.get("created_utc"), f"training audit created_utc {key}")
    exact_control_paths = {
        "submission_receipt": plan["arm_paths"][key]["submission_receipt"],
        "training_consumption_receipt": plan["arm_paths"][key]["training_consumption_receipt"],
        "prelaunch_receipt": plan["arm_paths"][key]["prelaunch_receipt"],
    }
    for field, expected_path in exact_control_paths.items():
        resolved_binding = validate_binding(payload[field], f"training audit {key} {field}")
        expect(resolved_binding == Path(expected_path).resolve(), f"training audit control path drifted: {key} {field}")
    scheduler = payload.get("scheduler_terminal")
    expect(isinstance(scheduler, dict) and set(scheduler) == {
        "job_id", "job_name", "state", "state_raw", "exit_code", "elapsed_seconds",
        "submit", "start", "end", "alloc_tres", "node_list",
    }, f"training audit scheduler schema drifted: {key}")
    expect(scheduler.get("state") == "COMPLETED" and scheduler.get("exit_code") == "0:0", f"training audit scheduler is not successful: {key}")
    expect(parse_utc(payload["created_utc"], f"training audit created_utc {key}") >= parse_slurm_time(scheduler["end"], f"training audit scheduler end {key}"), f"training audit predates scheduler end: {key}")
    runtime = payload.get("validator_runtime")
    expect(isinstance(runtime, dict) and set(runtime) == {
        "environment_root", "expected_python", "expected_python_resolved", "observed_python",
        "observed_python_resolved", "observed_sys_prefix", "train_environment_freeze",
        "live_environment_verification",
    }, f"training audit validator-runtime schema drifted: {key}")
    environment_root = Path(plan["train_environment_root"]).resolve()
    expected_python = environment_root / "bin/python"
    expect(runtime["environment_root"] == str(environment_root) and runtime["observed_sys_prefix"] == str(environment_root), f"training audit validator prefix drifted: {key}")
    expect(runtime["expected_python"] == str(expected_python) and runtime["observed_python"] == str(expected_python), f"training audit validator executable drifted: {key}")
    expect(runtime["expected_python_resolved"] == str(expected_python.resolve()) and runtime["observed_python_resolved"] == str(expected_python.resolve()), f"training audit resolved executable drifted: {key}")
    expect(runtime["train_environment_freeze"] == prereg["environment_freezes"]["train"], f"training audit validator freeze drifted: {key}")
    live = runtime["live_environment_verification"]
    expect(isinstance(live, dict) and live.get("schema") == "opd_math_environment_verification_v1" and live.get("status") == "passed", f"training audit live environment record drifted: {key}")
    expect(set(live) == {
        "schema_version", "schema", "status", "environment_root", "live_python",
        "expected_commit", "freeze_kind", "installed_distribution_count",
        "installed_distribution_map_sha256", "requirements_freeze", "commit_freeze",
        "expected_executable",
    }, f"training audit live environment schema drifted: {key}")
    expect(live.get("schema_version") == 1 and live.get("environment_root") == str(environment_root) and live.get("live_python") == str(expected_python.absolute()), f"training audit live environment identity drifted: {key}")
    expect(live.get("expected_commit") == EXPECTED_COMMIT and live.get("freeze_kind") == "train", f"training audit live environment commit drifted: {key}")
    expect(type(live.get("installed_distribution_count")) is int and live["installed_distribution_count"] > 0, f"training audit live distribution count drifted: {key}")
    expect(HEX64.fullmatch(str(live.get("installed_distribution_map_sha256"))) is not None, f"training audit live distribution hash drifted: {key}")
    expect(live.get("requirements_freeze") == {
        "path": str(environment_root / "requirements.freeze.txt"),
        "sha256": plan["runtime_train_freeze"]["runtime"]["sha256"],
    }, f"training audit live requirements freeze drifted: {key}")
    expect(live.get("commit_freeze") == {
        "path": plan["runtime_train_freeze"]["runtime"]["path"],
        "sha256": plan["runtime_train_freeze"]["runtime"]["sha256"],
        "byte_identical_to_requirements_freeze": True,
    }, f"training audit live environment freeze binding drifted: {key}")
    expect(live.get("expected_executable") is None, f"training audit train environment unexpectedly binds a serve executable: {key}")
    expect(payload.get("validation_method") == payload.get("deep_validation", {}).get("validation_method"), f"training audit validation method drifted: {key}")
    expect(payload.get("training_root") == payload.get("deep_validation", {}).get("training_root"), f"training-audit root binding drifted: {key}")
    expect(payload.get("consumable_adapter") == payload.get("deep_validation", {}).get("final_adapter"), f"training-audit adapter binding drifted: {key}")
    expect(payload.get("external_artifacts") == payload.get("deep_validation", {}).get("external_artifacts"), f"training-audit external binding drifted: {key}")
    training_root = payload.get("training_root")
    expect(isinstance(training_root, dict) and set(training_root) == {"path", "tree_sha256", "sealed_read_only"}, f"training audit root schema drifted: {key}")
    expect(Path(training_root["path"]).resolve() == Path(plan["arm_paths"][key]["training_out"]).resolve() and training_root["sealed_read_only"] is True and HEX64.fullmatch(str(training_root["tree_sha256"])) is not None, f"training audit root identity drifted: {key}")
    adapter = payload.get("consumable_adapter")
    expect(isinstance(adapter, dict) and Path(str(adapter.get("path"))).resolve() == Path(plan["arm_paths"][key]["training_out"]).resolve() / "final", f"training audit adapter path drifted: {key}")
    expect(HEX64.fullmatch(str(adapter.get("tree_sha256"))) is not None, f"training audit adapter hash drifted: {key}")
    external = payload.get("external_artifacts")
    expect(isinstance(external, dict) and set(external) == {"local_k1", "upstream_verl"}, f"training audit external schema drifted: {key}")
    if arm["implementation"] == "local" and arm["objective_id"] == "task_rl":
        expect(external["local_k1"] is None and external["upstream_verl"] is None, f"training audit task-RL external branch drifted: {key}")
    elif arm["implementation"] == "local":
        expect(isinstance(external["local_k1"], dict) and set(external["local_k1"]) == {
            "tokenizer_contract", "server_scoring_contract", "server_models", "vllm_log",
        } and external["upstream_verl"] is None, f"training audit local K1 external branch drifted: {key}")
    else:
        expect(external["local_k1"] is None and isinstance(external["upstream_verl"], dict) and set(external["upstream_verl"]) == {
            "preflight", "native_receipt",
        }, f"training audit upstream external branch drifted: {key}")
    deep = payload.get("deep_validation")
    expect(isinstance(deep, dict), f"training audit deep validation is invalid: {key}")
    local_deep_keys = {
        "implementation", "run_manifest", "completion_manifest", "prelaunch_receipt",
        "final_adapter", "adapter_delta", "step_trace", "sample_trace", "scheduler_job_id",
        "external_artifacts", "validation_method", "cost", "training_root",
    }
    upstream_deep_keys = {
        "implementation", "preflight", "native_receipt", "prelaunch_receipt",
        "final_adapter", "adapter_delta", "scheduler_job_id", "external_artifacts",
        "validation_method", "cost", "training_root",
    }
    expect(set(deep) == (local_deep_keys if arm["implementation"] == "local" else upstream_deep_keys), f"training audit deep-validation schema drifted: {key}")
    expected_method = (
        {
            "kind": "d89_same_code_integrity_replay_v1",
            "independent_implementation": False,
            "local_trace_geometry_replayed": True,
            "local_math_rewards_reverified": arm["objective_id"] != "k1_bare_verl_compatible_clip10",
        }
        if arm["implementation"] == "local"
        else {
            "kind": "d89_same_code_integrity_replay_v1",
            "independent_implementation": False,
            "upstream_private_helpers_replayed": True,
        }
    )
    expect(deep.get("validation_method") == expected_method and payload.get("validation_method") == expected_method, f"training audit validation-method schema drifted: {key}")
    cost = deep.get("cost")
    cost_keys = {"training_elapsed_seconds", "rollout_latency_seconds", "teacher_scoring_latency_seconds", "scored_completion_tokens", "prompt_group_tokens", "peak_cuda_memory_bytes"}
    expect(isinstance(cost, dict) and set(cost) == cost_keys, f"training audit cost schema drifted: {key}")
    if arm["implementation"] == "local":
        for field in ("training_elapsed_seconds", "rollout_latency_seconds"):
            expect(type(cost[field]) in (int, float) and math.isfinite(float(cost[field])) and float(cost[field]) > 0.0, f"training audit local positive cost drifted: {key} {field}")
        expect(type(cost["teacher_scoring_latency_seconds"]) in (int, float) and math.isfinite(float(cost["teacher_scoring_latency_seconds"])) and float(cost["teacher_scoring_latency_seconds"]) >= 0.0, f"training audit local teacher cost drifted: {key}")
        for field in ("scored_completion_tokens", "prompt_group_tokens"):
            expect(type(cost[field]) is int and cost[field] > 0, f"training audit local token cost drifted: {key} {field}")
        expect(cost["peak_cuda_memory_bytes"] is None or (type(cost["peak_cuda_memory_bytes"]) is int and cost["peak_cuda_memory_bytes"] >= 0), f"training audit local memory cost drifted: {key}")
        completion_path = validate_binding(deep["completion_manifest"], f"training audit completion manifest {key}")
        completion = load_json(completion_path, f"training audit completion manifest {key}")
        expect(cost == {
            "training_elapsed_seconds": completion.get("total_training_elapsed_seconds"),
            "rollout_latency_seconds": completion.get("total_rollout_latency_seconds"),
            "teacher_scoring_latency_seconds": completion.get("total_teacher_scoring_latency_seconds"),
            "scored_completion_tokens": completion.get("scored_completion_tokens"),
            "prompt_group_tokens": completion.get("prompt_group_tokens"),
            "peak_cuda_memory_bytes": completion.get("peak_cuda_memory_bytes"),
        }, f"training audit cost differs from completion manifest: {key}")
    else:
        expect(all(cost[field] is None for field in cost_keys), f"training audit upstream unknown cost fields drifted: {key}")
    expect(deep.get("scheduler_job_id") == scheduler.get("job_id"), f"training audit deep scheduler identity drifted: {key}")
    expect(payload.get("claim_boundary") == "Deep training integrity validation only; held-out task performance is not established.", f"training audit claim boundary drifted: {key}")
    if terminal_arm is not None:
        expect(terminal_arm.get("training_validation") == payload["deep_validation"], f"terminal/audit validation drifted: {key}")
        expect(terminal_arm.get("training_audit_receipt") == {"path": str(path), "sha256": sha256_file(path)}, f"terminal audit binding drifted: {key}")
        expect(payload["scheduler_terminal"] == terminal_arm.get("scheduler"), f"terminal/audit scheduler drifted: {key}")
        expect(payload["submission_receipt"] == terminal_arm.get("submission_receipt"), f"terminal/audit submission drifted: {key}")
        expect(payload["training_consumption_receipt"] == terminal_arm.get("consumption_receipt"), f"terminal/audit consumption drifted: {key}")
        expect(payload["prelaunch_receipt"] == terminal_arm.get("prelaunch_receipt"), f"terminal/audit prelaunch drifted: {key}")
    if mode == "selected_root":
        root = Path(payload["training_root"]["path"]).resolve()
        expect(payload["training_root"]["tree_sha256"] == sha256_tree(root), f"selected training root hash drifted: {key}")
        expect(all(not (item.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)) for item in [root, *root.rglob("*")]), f"selected training root is writable: {key}")
        for family in ("local_k1", "upstream_verl"):
            if payload["external_artifacts"][family] is not None:
                for label, binding in payload["external_artifacts"][family].items():
                    validate_binding(binding, f"selected training external {key} {label}")
    if mode in {"selected_root", "adapter"}:
        adapter = payload["consumable_adapter"]
        expect(adapter["tree_sha256"] == sha256_tree(adapter["path"]), f"selected adapter hash drifted: {key}")
    return path, payload


def authorize_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    repo = configure_repo(args.repo)
    plan_path, plan, prereg = validate_release_plan(args.release_plan, repo)
    terminal_path, terminal = validate_terminal_snapshot(
        plan["terminal_snapshot"], plan_path, plan, repo=repo, prereg=prereg
    )
    scope, paths = _evaluation_scope(
        plan,
        arm_key_value=getattr(args, "arm_key", None),
        raw_source=getattr(args, "raw_source", None),
    )
    expect(Path(args.output).resolve() == Path(paths["authorization"]).resolve(), "evaluation authorization output drifted")
    created_utc = utc_now()
    training_custody = None
    if scope in terminal["arms"]:
        expect(
            terminal["arms"][scope]["training_eligible_for_heldout"] is True,
            "cannot evaluate terminal failed training arm",
        )
        _, audit = validate_training_audit_receipt(
            plan_path=plan_path,
            plan=plan,
            prereg=prereg,
            key=scope,
            terminal_arm=terminal["arms"][scope],
            mode="selected_root",
        )
        training_custody = audit["deep_validation"]
    forbidden = [
        Path(paths[field])
        for field in (
            "artifact_root", "summary", "samples", "companion", "submission_receipt",
            "merge_consumption_receipt", "merge_supervisor_receipt", "seal_supervisor_receipt",
            "terminal_failure_receipt", "array_accounting_receipt", "array_accounting_raw",
            "consumption_receipt", "seal_receipt", "shard_consumption_root", "log_root",
            "private_log_root",
        )
    ]
    expect(not any(path.exists() or path.is_symlink() for path in forbidden), "evaluation artifacts existed before authorization")
    contract = expected_evaluation_contract(
        repo=repo,
        plan=plan,
        prereg=prereg,
        scope=scope,
        paths=paths,
        training_custody=training_custody,
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "authorization": "opd_math_objective_family_heldout_launch_authorization_v1",
        "status": "authorized_after_all_36_training_jobs_terminal",
        "created_utc": created_utc,
        "scope": scope,
        "release_plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
        "terminal_snapshot": {"path": str(terminal_path), "sha256": sha256_file(terminal_path)},
        "expected_summary": paths["summary"],
        "expected_samples": paths["samples"],
        "expected_companion": paths["companion"],
        "training_custody": training_custody,
        "evaluation_contract": contract,
        "heldout_outcomes_inspected_before_authorization": False,
        "evaluation_retry_policy": {
            "incomplete_infrastructure_attempt_may_retry_identically": False,
            "completed_evaluation_may_not_be_resampled": True,
            "verifier_cap_failure_may_not_be_resampled": True,
            "any_evaluation_failure_is_terminal_for_this_campaign": True,
        },
    }
    write_new(args.output, payload)
    return payload


def validate_evaluation_authorization(
    *,
    repo: Path,
    release_plan: str | Path,
    arm_key_value: str | None,
    raw_source: str | None,
) -> dict[str, Any]:
    plan_path, plan, prereg = validate_release_plan(release_plan, repo)
    terminal_path, terminal = validate_terminal_snapshot(
        plan["terminal_snapshot"], plan_path, plan, repo=repo, prereg=prereg
    )
    scope, paths = _evaluation_scope(plan, arm_key_value=arm_key_value, raw_source=raw_source)
    auth_path = regular_readonly(paths["authorization"], f"evaluation authorization {scope}")
    payload = load_json(auth_path, f"evaluation authorization {scope}")
    training_custody = None
    if scope in plan["arm_paths"]:
        _, audit = validate_training_audit_receipt(
            plan_path=plan_path,
            plan=plan,
            prereg=prereg,
            key=scope,
            terminal_arm=terminal["arms"][scope],
            mode="metadata",
        )
        training_custody = audit["deep_validation"]
    expected = {
        "schema_version": SCHEMA_VERSION,
        "authorization": "opd_math_objective_family_heldout_launch_authorization_v1",
        "status": "authorized_after_all_36_training_jobs_terminal",
        "created_utc": payload.get("created_utc"),
        "scope": scope,
        "release_plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
        "terminal_snapshot": {"path": str(terminal_path), "sha256": sha256_file(terminal_path)},
        "expected_summary": paths["summary"],
        "expected_samples": paths["samples"],
        "expected_companion": paths["companion"],
        "training_custody": training_custody,
        "evaluation_contract": expected_evaluation_contract(
            repo=repo,
            plan=plan,
            prereg=prereg,
            scope=scope,
            paths=paths,
            training_custody=training_custody,
        ),
        "heldout_outcomes_inspected_before_authorization": False,
        "evaluation_retry_policy": {
            "incomplete_infrastructure_attempt_may_retry_identically": False,
            "completed_evaluation_may_not_be_resampled": True,
            "verifier_cap_failure_may_not_be_resampled": True,
            "any_evaluation_failure_is_terminal_for_this_campaign": True,
        },
    }
    created = parse_utc(payload.get("created_utc"), f"evaluation authorization created_utc {scope}")
    terminal_created = parse_utc(load_json(terminal_path, "terminal snapshot")["created_utc"], "terminal snapshot created_utc")
    expect(created >= terminal_created, f"evaluation authorization predates terminal snapshot: {scope}")
    expect(payload == expected, f"evaluation authorization drifted: {scope}")
    _, wave = validate_evaluation_wave_authorization(
        repo=repo, plan_path=plan_path, plan=plan, prereg=prereg
    )
    expect(
        wave["targets"][scope]["status"] == "evaluate"
        and wave["targets"][scope]["authorization"] == {"path": str(auth_path), "sha256": sha256_file(auth_path)}
        and wave["targets"][scope]["evaluation_contract"] == payload["evaluation_contract"],
        f"evaluation authorization is not wave-bound: {scope}",
    )
    return {"path": str(auth_path), "sha256": sha256_file(auth_path), "scope": scope}


def validate_evaluation_authorization_command(args: argparse.Namespace) -> dict[str, Any]:
    repo = configure_repo(args.repo)
    return validate_evaluation_authorization(
        repo=repo,
        release_plan=args.release_plan,
        arm_key_value=getattr(args, "arm_key", None),
        raw_source=getattr(args, "raw_source", None),
    )


def evaluation_scheduler_policy() -> dict[str, Any]:
    return {
        "array_spec": EVALUATION_ARRAY_SPEC,
        "global_gpu_limit": 4,
        "serialization": "target_array_afterany_previous_target_seal",
        "no_requeue": True,
        "no_retry": True,
        "shard_resources": {
            "partition": "general-gpu",
            "account": "engr-lab-jacobsn",
            "gpu": "a100-sxm4:1",
            "cpus": 8,
            "memory": "96G",
            "time": "24:00:00",
            "exclude": "a100s-2307,a100-2207,r28-1801",
        },
        "supervisor_resources": {
            "partition": "general-cpu",
            "account": "engr-lab-jacobsn",
            "cpus": 4,
            "memory": "32G",
            "time": "02:00:00",
        },
    }


def authorize_evaluation_wave(args: argparse.Namespace) -> dict[str, Any]:
    repo = configure_repo(args.repo)
    plan_path, plan, prereg = validate_release_plan(args.release_plan, repo)
    terminal_path, terminal = validate_terminal_snapshot(
        plan["terminal_snapshot"], plan_path, plan, repo=repo, prereg=prereg
    )
    output = Path(args.output).resolve()
    expect(output == Path(plan["evaluation_wave_authorization"]).resolve(), "evaluation-wave authorization output drifted")
    if output.exists() or output.is_symlink():
        _, existing = validate_evaluation_wave_authorization(
            repo=repo, plan_path=plan_path, plan=plan, prereg=prereg
        )
        return existing
    target_order = ["raw_student__M", "raw_student__O", *prereg["payload"]["arm_keys"]]
    checked_paths: list[str] = []
    targets: dict[str, dict[str, Any]] = {}
    evaluated_order: list[str] = []
    created_utc = utc_now()
    resumed_authorizations = 0
    for ordinal, scope in enumerate(target_order):
        if scope.startswith("raw_student__"):
            arm_key_value = None
            raw_source = scope.rsplit("__", 1)[1]
            source = raw_source
            training_failed = False
        else:
            arm_key_value = scope
            raw_source = None
            source = prereg["payload"]["arms"][scope]["source"]
            training_failed = not terminal["arms"][scope]["training_eligible_for_heldout"]
        _, paths = _evaluation_scope(plan, arm_key_value=arm_key_value, raw_source=raw_source)
        scoped_paths = sorted({str(Path(value).resolve()) for value in paths.values()})
        checked_paths.extend(scoped_paths)
        if training_failed:
            expect(not any(Path(value).exists() or Path(value).is_symlink() for value in paths.values()), f"failed training target has evaluation artifacts: {scope}")
            targets[scope] = {
                "ordinal": ordinal,
                "status": "terminal_training_failure_no_evaluation",
                "source": source,
                "terminal_training_binding": terminal["arms"][scope],
                "authorization": None,
                "evaluation_contract": None,
                "evaluation_contract_sha256": None,
                "expected_paths": paths,
            }
            continue
        training_custody = None
        if arm_key_value is not None:
            _, audit = validate_training_audit_receipt(
                plan_path=plan_path,
                plan=plan,
                prereg=prereg,
                key=scope,
                terminal_arm=terminal["arms"][scope],
                mode="selected_root",
            )
            training_custody = audit["deep_validation"]
        forbidden = [Path(value) for key, value in paths.items() if key != "authorization"]
        expect(not any(path.exists() or path.is_symlink() for path in forbidden), f"heldout path existed before wave authorization: {scope}")
        contract = expected_evaluation_contract(
            repo=repo,
            plan=plan,
            prereg=prereg,
            scope=scope,
            paths=paths,
            training_custody=training_custody,
        )
        existing_authorization = None
        authorization_path = Path(paths["authorization"])
        if authorization_path.exists() or authorization_path.is_symlink():
            existing_authorization = load_json(regular_readonly(authorization_path, f"resumed authorization {scope}"), f"resumed authorization {scope}")
            resumed_authorizations += 1
        authorization_payload = {
            "schema_version": SCHEMA_VERSION,
            "authorization": "opd_math_objective_family_heldout_launch_authorization_v1",
            "status": "authorized_after_all_36_training_jobs_terminal",
            "created_utc": created_utc if existing_authorization is None else existing_authorization.get("created_utc"),
            "scope": scope,
            "release_plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
            "terminal_snapshot": {"path": str(terminal_path), "sha256": sha256_file(terminal_path)},
            "expected_summary": paths["summary"],
            "expected_samples": paths["samples"],
            "expected_companion": paths["companion"],
            "training_custody": training_custody,
            "evaluation_contract": contract,
            "heldout_outcomes_inspected_before_authorization": False,
            "evaluation_retry_policy": {
                "incomplete_infrastructure_attempt_may_retry_identically": False,
                "completed_evaluation_may_not_be_resampled": True,
                "verifier_cap_failure_may_not_be_resampled": True,
                "any_evaluation_failure_is_terminal_for_this_campaign": True,
            },
        }
        parse_utc(authorization_payload["created_utc"], f"evaluation authorization created_utc {scope}")
        auth_path = write_new_or_exact(paths["authorization"], authorization_payload)
        auth_binding = file_binding(auth_path)
        targets[scope] = {
            "ordinal": ordinal,
            "status": "evaluate",
            "source": source,
            "terminal_training_binding": None,
            "authorization": auth_binding,
            "evaluation_contract": contract,
            "evaluation_contract_sha256": canonical_json_sha256(contract),
            "expected_paths": paths,
        }
        evaluated_order.append(scope)
    program = load_json(plan["program_manifest"]["path"], "program manifest")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "authorization": "opd_math_objective_family_evaluation_wave_authorization_v1",
        "status": "authorized_after_all_36_training_terminal_before_any_heldout",
        "created_utc": created_utc,
        "controller": program["program_file"],
        "wrapper": program["evaluation_wrapper"],
        "release_plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
        "program_manifest": plan["program_manifest"],
        "terminal_snapshot": {"path": str(terminal_path), "sha256": sha256_file(terminal_path)},
        "git_commit": EXPECTED_COMMIT,
        "target_order": target_order,
        "evaluated_target_order": evaluated_order,
        "scheduler_policy": evaluation_scheduler_policy(),
        "targets": targets,
        "target_entry_sha256s": {scope: canonical_json_sha256(targets[scope]) for scope in target_order},
        "heldout_absence": {
            "all_planned_paths_absent_before_authorization": True,
            "checked_paths": len(set(checked_paths)),
            "canonical_path_list_sha256": canonical_json_sha256(sorted(set(checked_paths))),
            "resumed_exact_target_authorizations": resumed_authorizations,
        },
        "heldout_outcomes_inspected": False,
    }
    write_new_or_exact(output, payload)
    return payload


def validate_evaluation_wave_authorization(
    *, repo: Path, plan_path: Path, plan: Mapping[str, Any], prereg: Mapping[str, Any]
) -> tuple[Path, dict[str, Any]]:
    path = regular_readonly(plan["evaluation_wave_authorization"], "evaluation-wave authorization")
    payload = load_json(path, "evaluation-wave authorization")
    expected_keys = {
        "schema_version", "authorization", "status", "created_utc", "controller", "wrapper",
        "release_plan", "program_manifest", "terminal_snapshot", "git_commit", "target_order",
        "evaluated_target_order", "scheduler_policy", "targets", "target_entry_sha256s",
        "heldout_absence", "heldout_outcomes_inspected",
    }
    expect(set(payload) == expected_keys, "evaluation-wave authorization schema drifted")
    expect(payload.get("schema_version") == SCHEMA_VERSION and payload.get("authorization") == "opd_math_objective_family_evaluation_wave_authorization_v1", "evaluation-wave authorization identity drifted")
    expect(payload.get("status") == "authorized_after_all_36_training_terminal_before_any_heldout", "evaluation-wave authorization status drifted")
    expect(payload.get("release_plan") == {"path": str(plan_path), "sha256": sha256_file(plan_path)}, "evaluation-wave authorization plan drifted")
    expect(payload.get("program_manifest") == plan["program_manifest"] and payload.get("git_commit") == EXPECTED_COMMIT, "evaluation-wave authorization program drifted")
    expect(payload.get("terminal_snapshot") == file_binding(plan["terminal_snapshot"]), "evaluation-wave authorization terminal drifted")
    expected_order = ["raw_student__M", "raw_student__O", *prereg["payload"]["arm_keys"]]
    expect(payload.get("target_order") == expected_order and set(payload.get("targets", {})) == set(expected_order), "evaluation-wave target order drifted")
    expect(payload.get("scheduler_policy") == evaluation_scheduler_policy(), "evaluation-wave scheduler policy drifted")
    absence = payload.get("heldout_absence", {})
    expect(payload.get("heldout_outcomes_inspected") is False and absence.get("all_planned_paths_absent_before_authorization") is True, "evaluation-wave outcome boundary drifted")
    expect(isinstance(absence.get("resumed_exact_target_authorizations"), int) and 0 <= absence["resumed_exact_target_authorizations"] <= len(expected_order), "evaluation-wave authorization resume count drifted")
    program = load_json(plan["program_manifest"]["path"], "program manifest")
    expect(payload.get("controller") == program["program_file"] and payload.get("wrapper") == program["evaluation_wrapper"], "evaluation-wave code binding drifted")
    evaluated = []
    for ordinal, scope in enumerate(expected_order):
        target = payload["targets"][scope]
        expect(target.get("ordinal") == ordinal and target.get("source") in SOURCES, f"evaluation-wave target identity drifted: {scope}")
        expect(payload["target_entry_sha256s"].get(scope) == canonical_json_sha256(target), f"evaluation-wave target hash drifted: {scope}")
        if target.get("status") == "evaluate":
            validate_binding(target.get("authorization"), f"evaluation-wave target authorization {scope}")
            expect(target.get("evaluation_contract_sha256") == canonical_json_sha256(target.get("evaluation_contract")), f"evaluation-wave contract hash drifted: {scope}")
            evaluated.append(scope)
        else:
            expect(target.get("status") == "terminal_training_failure_no_evaluation" and target.get("authorization") is None and target.get("evaluation_contract") is None, f"evaluation-wave failure target drifted: {scope}")
    expect(payload.get("evaluated_target_order") == evaluated, "evaluation-wave evaluated order drifted")
    parse_utc(payload.get("created_utc"), "evaluation-wave authorization created_utc")
    return path, payload


def _scontrol_snapshot(job_id: str) -> dict[str, Any]:
    expect(re.fullmatch(r"[1-9][0-9]*", job_id) is not None, "scontrol job ID is invalid")
    raw = subprocess.run(
        ["scontrol", "show", "job", "-o", job_id],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    rows = []
    for line in raw.splitlines():
        fields = {}
        for token in shlex.split(line):
            if "=" in token:
                name, value = token.split("=", 1)
                fields[name] = value
        if fields.get("JobId") == job_id or fields.get("ArrayJobId") == job_id:
            rows.append(fields)
    expect(rows, f"scontrol lacks exact job/array rows: {job_id}")
    exact = [row for row in rows if row.get("JobId") == job_id]
    fields = exact[0] if len(exact) == 1 else rows[0]
    array_expanded = not exact and len(rows) >= 1
    if array_expanded:
        for field in ("Command", "Partition", "Account", "WorkDir"):
            expect(len({row.get(field) for row in rows}) == 1, f"expanded array {field} drifted: {job_id}")
    user_held = any(row.get("JobState") == "PENDING" and row.get("Reason") == "JobHeldUser" for row in rows)
    return {
        "job_id": job_id,
        "job_state": "ARRAY_EXPANDED" if array_expanded else fields.get("JobState"),
        "reason": "JobHeldUser" if user_held else ("ArrayExpanded" if array_expanded else fields.get("Reason")),
        "user_held": user_held,
        "row_count": len(rows),
        "array_expanded": array_expanded,
        "dependency": fields.get("Dependency"),
        "array_task_id": fields.get("ArrayTaskId"),
        "submit_time": fields.get("SubmitTime"),
        "work_dir": fields.get("WorkDir"),
        "std_out": fields.get("StdOut"),
        "command": fields.get("Command"),
        "partition": fields.get("Partition"),
        "account": fields.get("Account"),
        "req_tres": fields.get("ReqTRES"),
        "array_job_id": fields.get("ArrayJobId"),
        "raw_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "raw_text": raw,
    }


def _parse_sbatch_id(raw: str, label: str) -> str:
    value = raw.strip().split(";", 1)[0]
    expect(re.fullmatch(r"[1-9][0-9]*", value) is not None, f"{label} returned invalid job ID")
    return value


def _evaluation_selector(scope: str) -> tuple[str, str]:
    if scope in {"raw_student__M", "raw_student__O"}:
        return "raw_source", scope.rsplit("__", 1)[1]
    return "arm_key", scope


def _best_effort_scontrol(job_id: str) -> dict[str, Any]:
    try:
        return {"available": True, "snapshot": _scontrol_snapshot(job_id), "error_type": None, "error_sha256": None}
    except BaseException as error:
        return {
            "available": False,
            "snapshot": None,
            "error_type": type(error).__name__,
            "error_sha256": hashlib.sha256(str(error).encode("utf-8")).hexdigest(),
        }


def complete_evaluation_wave_release(
    *, plan: Mapping[str, Any], wave_path: Path, index_path: Path,
    index: Mapping[str, Any], clean_environment: Mapping[str, str],
) -> dict[str, Any]:
    """Finish or recover the one canonical held-job release transaction."""

    failure_path = Path(plan["evaluation_wave_release_failure"]).resolve()
    expect(not failure_path.exists() and not failure_path.is_symlink(), "evaluation wave has a terminal release failure")
    result_path = Path(plan["evaluation_wave_release_result"]).resolve()
    if result_path.exists() or result_path.is_symlink():
        result = load_json(regular_readonly(result_path, "evaluation-wave release result"), "evaluation-wave release result")
        expect(result.get("submission_index") == file_binding(index_path) and result.get("status") == "all_held_jobs_released_and_observed", "evaluation-wave release result drifted")
        return result
    all_jobs = index.get("all_job_ids")
    expect(isinstance(all_jobs, list) and all_jobs, "evaluation-wave release lacks jobs")
    release_command = ["scontrol", "release", ",".join(all_jobs)]
    intent_payload = {
        "schema_version": SCHEMA_VERSION,
        "authorization": "opd_math_objective_family_evaluation_wave_release_intent_v1",
        "status": "all_held_jobs_authorized_for_one_canonical_release",
        "created_utc": utc_now(),
        "wave_authorization": file_binding(wave_path),
        "submission_index": file_binding(index_path),
        "all_job_ids": all_jobs,
        "release_command": release_command,
        "heldout_outcomes_inspected": False,
    }
    intent_path = Path(plan["evaluation_wave_release_intent"]).resolve()
    if intent_path.exists() or intent_path.is_symlink():
        existing_intent = load_json(regular_readonly(intent_path, "evaluation-wave release intent"), "evaluation-wave release intent")
        intent_payload["created_utc"] = existing_intent.get("created_utc")
    intent_path = write_new_or_exact(intent_path, intent_payload)
    before = {job_id: _best_effort_scontrol(job_id) for job_id in all_jobs}
    available = [entry["snapshot"] for entry in before.values() if entry["available"]]
    expect(len(available) == len(all_jobs), "not all held jobs were queryable before release")
    held = [row for row in available if row.get("user_held") is True]
    command_executed = False
    stdout = ""
    stderr = ""
    return_code = 0
    if len(held) == len(all_jobs):
        process = subprocess.run(release_command, check=False, text=True, capture_output=True, env=dict(clean_environment))
        command_executed = True
        stdout, stderr, return_code = process.stdout, process.stderr, process.returncode
    elif held:
        return_code = -1
        stderr = "mixed held/released state before resumable release"
    after: dict[str, Any] = {}
    for attempt in range(30):
        after = {job_id: _best_effort_scontrol(job_id) for job_id in all_jobs}
        if all(entry["available"] and entry["snapshot"].get("user_held") is False for entry in after.values()):
            break
        if attempt < 29:
            time.sleep(1)
    released = return_code == 0 and all(
        entry["available"] and entry["snapshot"].get("user_held") is False
        for entry in after.values()
    )
    common = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": utc_now(),
        "wave_authorization": file_binding(wave_path),
        "submission_index": file_binding(index_path),
        "release_intent": file_binding(intent_path),
        "all_job_ids": all_jobs,
        "release_command": release_command,
        "command_executed_this_invocation": command_executed,
        "command_return_code": return_code,
        "command_stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
        "command_stderr_sha256": hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
        "pre_release_snapshots": before,
        "post_release_snapshots": after,
        "heldout_outcomes_inspected": False,
    }
    if not released:
        failure = {
            **common,
            "receipt": "opd_math_objective_family_evaluation_wave_release_failure_v1",
            "status": "terminal_partial_or_failed_release_no_retry",
            "retry_authorized": False,
        }
        write_new_or_exact(failure_path, failure)
        raise RuntimeError("evaluation-wave release did not produce an all-released observable state")
    result = {
        **common,
        "receipt": "opd_math_objective_family_evaluation_wave_release_result_v1",
        "status": "all_held_jobs_released_and_observed",
    }
    write_new_or_exact(result_path, result)
    return result


def scheduler_jobs_for_comments(comments: Sequence[str], *, start_date: str) -> dict[str, list[dict[str, str]]]:
    wanted = set(comments)
    found: dict[str, list[dict[str, str]]] = {comment: [] for comment in comments}
    commands = [
        ["squeue", "--noheader", "--user", "hiqbal", "--format=%i|%k"],
        ["sacct", "-X", "-n", "-P", "-u", "hiqbal", "--starttime", start_date, "--format=JobIDRaw,Comment"],
    ]
    for command in commands:
        process = subprocess.run(command, check=True, text=True, capture_output=True)
        for line in process.stdout.splitlines():
            parts = line.strip().split("|", 1)
            if len(parts) != 2 or parts[1] not in wanted:
                continue
            entry = {"job_id": parts[0], "comment": parts[1], "source": command[0]}
            if entry not in found[parts[1]]:
                found[parts[1]].append(entry)
    return found


def terminalize_incomplete_submission_journal(
    *, plan: Mapping[str, Any], wave_path: Path,
) -> None:
    root = Path(plan["evaluation_wave_submission_journal_root"]).resolve()
    intents = sorted(root.glob("*.intent.json")) if root.is_dir() else []
    intent_payloads = [load_json(regular_readonly(path, "submission journal intent"), "submission journal intent") for path in intents]
    comments = [str(payload["scheduler_comment"]) for payload in intent_payloads]
    matches = (
        scheduler_jobs_for_comments(
            comments, start_date=parse_utc(plan["created_utc"], "release plan created_utc").date().isoformat(),
        )
        if comments
        else {}
    )
    failure = {
        "schema_version": SCHEMA_VERSION,
        "receipt": "opd_math_objective_family_evaluation_wave_submission_failure_v1",
        "status": "terminal_incomplete_submission_journal_no_retry",
        "created_utc": utc_now(),
        "authorization": file_binding(wave_path),
        "submitted_job_ids": sorted({entry["job_id"] for values in matches.values() for entry in values}),
        "target_receipts": {},
        "journal_intents": {path.name: file_binding(path) for path in intents},
        "scheduler_comment_matches": matches,
        "error_type": "InterruptedSubmissionTransaction",
        "error_sha256": hashlib.sha256(canonical_json_bytes(matches)).hexdigest(),
        "heldout_outcomes_inspected": False,
    }
    seal_tree_readonly(root)
    write_new_or_exact(plan["evaluation_wave_submission_failure"], failure)
    raise RuntimeError("evaluation-wave submission was interrupted; discovered jobs are sealed in the terminal failure receipt")


def submit_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    repo = configure_repo(args.repo)
    plan_path, plan, _ = validate_release_plan(args.release_plan, repo)
    scope, paths = _evaluation_scope(
        plan,
        arm_key_value=getattr(args, "arm_key", None),
        raw_source=getattr(args, "raw_source", None),
    )
    authorization = validate_evaluation_authorization(
        repo=repo,
        release_plan=plan_path,
        arm_key_value=getattr(args, "arm_key", None),
        raw_source=getattr(args, "raw_source", None),
    )
    auth_payload = load_json(authorization["path"], f"evaluation authorization {scope}")
    contract = auth_payload["evaluation_contract"]
    output = Path(args.output).resolve()
    expect(output == Path(paths["submission_receipt"]).resolve(), "evaluation submission output drifted")
    expect(not output.exists() and not output.is_symlink(), "evaluation submission receipt already exists")
    log_root = Path(paths["log_root"]).resolve()
    log_root.mkdir(parents=True, exist_ok=False)
    wrapper = Path(contract["code"]["evaluation_wrapper"]["path"]).resolve()
    selector_kind, selector_value = _evaluation_selector(scope)
    external_exports = {
        **{key: value for key, value in contract["exports"].items() if value is not None},
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        **contract["sanitized_scheduler_environment"],
        "OPD_RELEASE_PROGRAM": contract["code"]["release_program"]["path"],
        "OPD_RELEASE_PLAN": str(plan_path),
        "OPD_RELEASE_REPO": str(repo),
        "OPD_RELEASE_SCOPE": scope,
        "OPD_RELEASE_SELECTOR_KIND": selector_kind,
        "OPD_RELEASE_SELECTOR_VALUE": selector_value,
        "OPD_RELEASE_SHARD_CONSUMPTION_ROOT": paths["shard_consumption_root"],
        "OPD_RELEASE_MERGE_CONSUMPTION": paths["merge_consumption_receipt"],
        "OPD_RELEASE_FINAL_CONSUMPTION": paths["consumption_receipt"],
        "OPD_RELEASE_EVALUATION_SEAL": paths["seal_receipt"],
    }
    clean_environment = {
        "HOME": "/home/compute/hiqbal",
        "USER": "hiqbal",
        "LOGNAME": "hiqbal",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "SLURM_CONF": "/project/compute/slurm/etc/slurm.conf",
        "LANG": "C",
        "LC_ALL": "C",
        "TZ": "UTC",
    }

    def command_for(*, phase: str, output_pattern: Path, dependency: str | None, gpu: bool, hold: bool) -> list[str]:
        exports = dict(external_exports)
        exports["OPD_RELEASE_PHASE"] = phase
        command = [
            "/usr/bin/sbatch", "--parsable", f"--chdir={repo}",
            f"--output={output_pattern}",
            "--partition=general-g" if gpu else "--partition=general-c",
            "--cpus-per-task=8" if gpu else "--cpus-per-task=4",
            "--mem=64G" if gpu else "--mem=32G",
            "--time=08:00:00" if gpu else "--time=02:00:00",
        ]
        if gpu:
            command.extend(("--gres=gpu:a100:1", f"--array={EVALUATION_ARRAY_SPEC}"))
        if hold:
            command.append("--hold")
        if dependency is not None:
            command.append(f"--dependency=afterok:{dependency}")
        export_value = ",".join(f"{key}={value}" for key, value in sorted(exports.items()))
        command.extend((f"--export={export_value}", str(wrapper)))
        return command

    array_command = command_for(
        phase="shard",
        output_pattern=log_root / "eval-%A_%a.out",
        dependency=None,
        gpu=True,
        hold=True,
    )
    array_raw = subprocess.run(array_command, check=True, text=True, capture_output=True, env=clean_environment).stdout
    array_job = _parse_sbatch_id(array_raw, "evaluation array submission")
    merge_command = command_for(
        phase="merge",
        output_pattern=log_root / "merge-%j.out",
        dependency=array_job,
        gpu=False,
        hold=False,
    )
    merge_raw = subprocess.run(merge_command, check=True, text=True, capture_output=True, env=clean_environment).stdout
    merge_job = _parse_sbatch_id(merge_raw, "evaluation merge submission")
    seal_command = command_for(
        phase="seal",
        output_pattern=log_root / "seal-%j.out",
        dependency=merge_job,
        gpu=False,
        hold=False,
    )
    seal_raw = subprocess.run(seal_command, check=True, text=True, capture_output=True, env=clean_environment).stdout
    seal_job = _parse_sbatch_id(seal_raw, "evaluation seal submission")
    array_snapshot = _scontrol_snapshot(array_job)
    merge_snapshot = _scontrol_snapshot(merge_job)
    seal_snapshot = _scontrol_snapshot(seal_job)
    expect(array_snapshot["job_state"] == "PENDING" and array_snapshot["reason"] == "JobHeldUser", "evaluation array was not submitted held")
    expect(merge_snapshot["job_state"] == "PENDING" and str(merge_snapshot["dependency"]).startswith(f"afterok:{array_job}"), "evaluation merge dependency drifted")
    expect(seal_snapshot["job_state"] == "PENDING" and str(seal_snapshot["dependency"]).startswith(f"afterok:{merge_job}"), "evaluation seal dependency drifted")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "receipt": "opd_math_objective_family_evaluation_submission_v1",
        "status": "held_array_and_dependent_merge_seal_recorded",
        "created_utc": utc_now(),
        "scope": scope,
        "authorization": {"path": authorization["path"], "sha256": authorization["sha256"]},
        "release_plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
        "commands": {"array": array_command, "merge": merge_command, "seal": seal_command},
        "exports": external_exports,
        "sbatch": {
            "array_raw": array_raw,
            "array_job_id": array_job,
            "merge_raw": merge_raw,
            "merge_job_id": merge_job,
            "seal_raw": seal_raw,
            "seal_job_id": seal_job,
        },
        "scheduler_at_receipt": {
            "array": array_snapshot,
            "merge": merge_snapshot,
            "seal": seal_snapshot,
        },
        "heldout_outcomes_inspected": False,
    }
    write_new(output, payload)
    subprocess.run(["scontrol", "release", array_job], check=True, text=True, capture_output=True, env=clean_environment)
    return payload


def submit_evaluation_wave(args: argparse.Namespace) -> dict[str, Any]:
    repo = configure_repo(args.repo)
    plan_path, plan, prereg = validate_release_plan(args.release_plan, repo)
    wave_path, wave = validate_evaluation_wave_authorization(
        repo=repo, plan_path=plan_path, plan=plan, prereg=prereg
    )
    output = Path(args.output).resolve()
    expect(output == Path(plan["evaluation_wave_submission_index"]).resolve(), "evaluation-wave submission output drifted")
    expect(not Path(plan["evaluation_wave_submission_failure"]).exists(), "evaluation-wave has a terminal submission failure")
    program = load_json(plan["program_manifest"]["path"], "program manifest")
    wrapper = Path(program["evaluation_wrapper"]["path"]).resolve()
    clean_environment = {
        "HOME": "/home/compute/hiqbal",
        "USER": "hiqbal",
        "LOGNAME": "hiqbal",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "SLURM_CONF": "/project/compute/slurm/etc/slurm.conf",
        "LANG": "C",
        "LC_ALL": "C",
        "TZ": "UTC",
    }
    if output.exists() or output.is_symlink():
        index_path = regular_readonly(output, "resumable evaluation-wave submission index")
        existing_index = load_json(index_path, "resumable evaluation-wave submission index")
        expect(existing_index.get("authorization") == file_binding(wave_path), "resumable evaluation-wave index authorization drifted")
        expect(existing_index.get("release_plan") == file_binding(plan_path), "resumable evaluation-wave index plan drifted")
        expect(existing_index.get("evaluated_target_order") == wave["evaluated_target_order"], "resumable evaluation-wave index order drifted")
        for scope, binding in (existing_index.get("target_submission_receipts") or {}).items():
            validate_binding(binding, f"resumable evaluation-wave target submission {scope}")
        complete_evaluation_wave_release(
            plan=plan, wave_path=wave_path, index_path=index_path,
            index=existing_index, clean_environment=clean_environment,
        )
        return existing_index
    journal_root = Path(plan["evaluation_wave_submission_journal_root"]).resolve()
    if journal_root.exists() or journal_root.is_symlink():
        expect(journal_root.is_dir() and not journal_root.is_symlink(), "submission journal root is invalid")
        if any(journal_root.iterdir()):
            terminalize_incomplete_submission_journal(plan=plan, wave_path=wave_path)
    else:
        journal_root.mkdir(parents=True, exist_ok=False)
    submitted_job_ids: list[str] = []
    target_receipts: dict[str, dict[str, Any]] = {}
    job_chains: dict[str, dict[str, Any]] = {}
    previous_scope: str | None = None
    previous_seal_job: str | None = None

    journal_entries: list[dict[str, Any]] = []

    def submit(command: list[str], label: str, *, scope: str, phase: str) -> tuple[str, str]:
        ordinal = len(journal_entries)
        comment_tokens = [token for token in command if token.startswith("--comment=")]
        expect(len(comment_tokens) == 1, f"submission command lacks one scheduler comment: {label}")
        comment = comment_tokens[0].split("=", 1)[1]
        intent = {
            "schema_version": SCHEMA_VERSION,
            "intent": "opd_math_objective_family_sbatch_intent_v1",
            "status": "sealed_before_sbatch_call",
            "created_utc": utc_now(),
            "ordinal": ordinal,
            "scope": scope,
            "phase": phase,
            "scheduler_comment": comment,
            "command": command,
            "wave_authorization": file_binding(wave_path),
            "heldout_outcomes_inspected": False,
        }
        intent_path = write_new(journal_root / f"{ordinal:03d}.intent.json", intent)
        raw = subprocess.run(command, check=True, text=True, capture_output=True, env=clean_environment).stdout
        job_id = _parse_sbatch_id(raw, label)
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "receipt": "opd_math_objective_family_sbatch_result_v1",
            "status": "scheduler_job_id_recorded",
            "created_utc": utc_now(),
            "ordinal": ordinal,
            "scope": scope,
            "phase": phase,
            "scheduler_comment": comment,
            "intent": file_binding(intent_path),
            "job_id": job_id,
            "sbatch_stdout_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            "heldout_outcomes_inspected": False,
        }
        receipt_path = write_new(journal_root / f"{ordinal:03d}.result.json", receipt)
        journal_entries.append({"intent": file_binding(intent_path), "result": file_binding(receipt_path)})
        submitted_job_ids.append(job_id)
        return job_id, raw

    try:
        for scope in wave["evaluated_target_order"]:
            target = wave["targets"][scope]
            if scope.startswith("raw_student__"):
                arm_key_value = None
                raw_source = scope.rsplit("__", 1)[1]
            else:
                arm_key_value = scope
                raw_source = None
            _, paths = _evaluation_scope(plan, arm_key_value=arm_key_value, raw_source=raw_source)
            authorization_path = validate_binding(target["authorization"], f"evaluation authorization {scope}")
            authorization = file_binding(authorization_path)
            auth_payload = load_json(authorization_path, f"evaluation authorization {scope}")
            expect(auth_payload.get("scope") == scope and auth_payload.get("evaluation_contract") == target["evaluation_contract"], f"evaluation authorization target drifted: {scope}")
            contract = auth_payload["evaluation_contract"]
            log_root = Path(paths["log_root"]).resolve()
            log_root.mkdir(parents=True, exist_ok=False)
            external_exports = {
                **{key: value for key, value in contract["exports"].items() if value is not None},
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                **contract["sanitized_scheduler_environment"],
                "OPD_RELEASE_PROGRAM": contract["code"]["release_program"]["path"],
                "OPD_RELEASE_PLAN": str(plan_path),
                "OPD_RELEASE_REPO": str(repo),
                "OPD_RELEASE_SCOPE": scope,
                "OPD_RELEASE_SELECTOR_KIND": _evaluation_selector(scope)[0],
                "OPD_RELEASE_SELECTOR_VALUE": _evaluation_selector(scope)[1],
                "OPD_RELEASE_SHARD_CONSUMPTION_ROOT": paths["shard_consumption_root"],
                "OPD_RELEASE_MERGE_CONSUMPTION": paths["merge_consumption_receipt"],
                "OPD_RELEASE_MERGE_SUPERVISOR": paths["merge_supervisor_receipt"],
                "OPD_RELEASE_SEAL_SUPERVISOR": paths["seal_supervisor_receipt"],
                "OPD_RELEASE_EVALUATION_SEAL": paths["seal_receipt"],
                "OPD_RELEASE_PRIVATE_LOG_ROOT": paths["private_log_root"],
                "OPD_RELEASE_EVALUATION_WAVE_SEAL": plan["evaluation_wave_seal"],
            }

            def command_for(*, phase: str, output_pattern: Path, dependency: str | None, shard: bool) -> list[str]:
                exports = dict(external_exports)
                exports["OPD_RELEASE_PHASE"] = phase
                command = [
                    "/usr/bin/sbatch", "--parsable", f"--chdir={repo}", "--hold", "--no-requeue",
                    "--account=engr-lab-jacobsn",
                    f"--comment=opd_obj_{plan['campaign_id']}_{len(journal_entries):03d}_{phase}",
                    f"--output={output_pattern}",
                ]
                if shard:
                    command.extend((
                        "--job-name=opd_obj_eval_shard", "--partition=general-gpu",
                        "--gpus=a100-sxm4:1", "--cpus-per-task=8", "--mem=96G",
                        "--time=24:00:00", "--exclude=a100s-2307,a100-2207,r28-1801",
                        f"--array={EVALUATION_ARRAY_SPEC}",
                    ))
                else:
                    command.extend((
                        f"--job-name=opd_obj_eval_{phase}", "--partition=general-cpu",
                        "--cpus-per-task=4", "--mem=32G", "--time=02:00:00",
                    ))
                if dependency is not None:
                    command.append(f"--dependency=afterany:{dependency}")
                command.append("--export=" + ",".join(f"{key}={value}" for key, value in sorted(exports.items())))
                command.append(str(wrapper))
                return command

            array_command = command_for(
                phase="shard",
                output_pattern=log_root / "shard-%A_%a.public.out",
                dependency=previous_seal_job,
                shard=True,
            )
            array_job, array_raw = submit(array_command, f"evaluation array {scope}", scope=scope, phase="shard")
            merge_command = command_for(
                phase="merge_supervisor",
                output_pattern=log_root / "merge-supervisor-%j.public.out",
                dependency=array_job,
                shard=False,
            )
            merge_job, merge_raw = submit(merge_command, f"evaluation merge supervisor {scope}", scope=scope, phase="merge_supervisor")
            seal_command = command_for(
                phase="seal_supervisor",
                output_pattern=log_root / "seal-supervisor-%j.public.out",
                dependency=merge_job,
                shard=False,
            )
            seal_job, seal_raw = submit(seal_command, f"evaluation seal supervisor {scope}", scope=scope, phase="seal_supervisor")
            snapshots = {
                "array": _scontrol_snapshot(array_job),
                "merge_supervisor": _scontrol_snapshot(merge_job),
                "seal_supervisor": _scontrol_snapshot(seal_job),
            }
            for label, snapshot in snapshots.items():
                expect(snapshot["job_state"] == "PENDING" and snapshot["reason"] == "JobHeldUser", f"evaluation {scope} {label} was not submitted held")
                expect(Path(str(snapshot["command"])).resolve() == wrapper, f"evaluation {scope} {label} did not submit the sealed wrapper")
                expect(snapshot["partition"] == ("general-gpu" if label == "array" else "general-cpu") and snapshot["account"] == "engr-lab-jacobsn", f"evaluation {scope} {label} resource lane drifted")
            expect(snapshots["array"]["array_task_id"] == EVALUATION_ARRAY_SPEC, f"evaluation array geometry drifted: {scope}")
            receipt = {
                "schema_version": SCHEMA_VERSION,
                "receipt": "opd_math_objective_family_evaluation_target_submission_v1",
                "status": "complete_target_chain_held_before_wave_release",
                "created_utc": utc_now(),
                "scope": scope,
                "authorization": {"path": authorization["path"], "sha256": authorization["sha256"]},
                "wave_authorization": {"path": str(wave_path), "sha256": sha256_file(wave_path)},
                "target_entry_sha256": wave["target_entry_sha256s"][scope],
                "release_plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
                "previous_evaluated_scope": previous_scope,
                "commands": {"array": array_command, "merge_supervisor": merge_command, "seal_supervisor": seal_command},
                "exports": external_exports,
                "jobs": {
                    "array": {"job_id": array_job, "raw": array_raw, "array_spec": EVALUATION_ARRAY_SPEC},
                    "merge_supervisor": {"job_id": merge_job, "raw": merge_raw, "dependency": f"afterany:{array_job}"},
                    "seal_supervisor": {"job_id": seal_job, "raw": seal_raw, "dependency": f"afterany:{merge_job}"},
                },
                "held_snapshots": snapshots,
                "heldout_outcomes_inspected": False,
            }
            receipt_path = write_new(paths["submission_receipt"], receipt)
            target_receipts[scope] = file_binding(receipt_path)
            job_chains[scope] = receipt["jobs"]
            previous_scope = scope
            previous_seal_job = seal_job
        expect(previous_seal_job is not None, "evaluation wave has no targets")
        finalizer_exports = {
            "HOME": "/home/compute/hiqbal",
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "SLURM_CONF": "/project/compute/slurm/etc/slurm.conf",
            "LANG": "C", "LC_ALL": "C", "TZ": "UTC",
            "OPD_MATH_TRAIN_ENV": plan["train_environment_root"],
            "OPD_RELEASE_PROGRAM": program["program_file"]["path"],
            "OPD_RELEASE_PLAN": str(plan_path),
            "OPD_RELEASE_REPO": str(repo),
            "OPD_RELEASE_SCOPE": "evaluation_wave",
            "OPD_RELEASE_SELECTOR_KIND": "raw_source",
            "OPD_RELEASE_SELECTOR_VALUE": "M",
            "OPD_RELEASE_PHASE": "wave_finalizer",
            "OPD_RELEASE_EVALUATION_WAVE_SEAL": plan["evaluation_wave_seal"],
            "OPD_RELEASE_FINALIZER_RECEIPT": plan["evaluation_wave_finalizer_receipt"],
        }
        finalizer_command = [
            "/usr/bin/sbatch", "--parsable", f"--chdir={repo}", "--hold", "--no-requeue",
            "--account=engr-lab-jacobsn", f"--comment=opd_obj_{plan['campaign_id']}_{len(journal_entries):03d}_wave_finalizer", "--output=/dev/null",
            "--job-name=opd_eval_wave_final", "--partition=general-cpu", "--cpus-per-task=4",
            "--mem=32G", "--time=02:00:00", f"--dependency=afterany:{previous_seal_job}",
            "--export=" + ",".join(f"{key}={value}" for key, value in sorted(finalizer_exports.items())),
            str(wrapper),
        ]
        finalizer_job, finalizer_raw = submit(finalizer_command, "evaluation wave finalizer", scope="evaluation_wave", phase="wave_finalizer")
        finalizer_snapshot = _scontrol_snapshot(finalizer_job)
        expect(finalizer_snapshot["job_state"] == "PENDING" and finalizer_snapshot["reason"] == "JobHeldUser", "evaluation wave finalizer was not held")
        expect(Path(str(finalizer_snapshot["command"])).resolve() == wrapper, "evaluation wave finalizer wrapper drifted")
    except BaseException as error:
        intent_paths = sorted(journal_root.glob("*.intent.json"))
        intent_payloads = [load_json(regular_readonly(path, "submission journal intent"), "submission journal intent") for path in intent_paths]
        comment_matches = scheduler_jobs_for_comments(
            [str(payload["scheduler_comment"]) for payload in intent_payloads],
            start_date=parse_utc(plan["created_utc"], "release plan created_utc").date().isoformat(),
        )
        discovered_job_ids = sorted({entry["job_id"] for values in comment_matches.values() for entry in values})
        failure = {
            "schema_version": SCHEMA_VERSION,
            "receipt": "opd_math_objective_family_evaluation_wave_submission_failure_v1",
            "status": "partial_submission_left_held_no_jobs_released",
            "created_utc": utc_now(),
            "authorization": {"path": str(wave_path), "sha256": sha256_file(wave_path)},
            "submitted_job_ids": sorted(set(submitted_job_ids) | set(discovered_job_ids)),
            "target_receipts": target_receipts,
            "journal_intents": {path.name: file_binding(path) for path in intent_paths},
            "scheduler_comment_matches": comment_matches,
            "error_type": type(error).__name__,
            "error_sha256": hashlib.sha256(str(error).encode("utf-8")).hexdigest(),
            "heldout_outcomes_inspected": False,
        }
        seal_tree_readonly(journal_root)
        write_new(plan["evaluation_wave_submission_failure"], failure)
        raise
    journal_tree_sha256 = sha256_tree(journal_root)
    seal_tree_readonly(journal_root)
    expect(sha256_tree(journal_root) == journal_tree_sha256, "submission journal changed while sealing")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "receipt": "opd_math_objective_family_evaluation_wave_submission_v1",
        "status": "all_jobs_held_and_indexed_before_release",
        "created_utc": utc_now(),
        "authorization": {"path": str(wave_path), "sha256": sha256_file(wave_path)},
        "release_plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
        "target_order": wave["target_order"],
        "evaluated_target_order": wave["evaluated_target_order"],
        "target_submission_receipts": target_receipts,
        "job_chains": job_chains,
        "submission_journal": {"path": str(journal_root), "tree_sha256": journal_tree_sha256, "entries": journal_entries},
        "wave_finalizer": {
            "job_id": finalizer_job,
            "raw": finalizer_raw,
            "dependency": f"afterany:{previous_seal_job}",
            "command": finalizer_command,
            "held_snapshot": finalizer_snapshot,
        },
        "all_job_ids": submitted_job_ids,
        "all_jobs_verified_held": True,
        "global_gpu_limit": 4,
        "heldout_outcomes_inspected": False,
    }
    index_path = write_new(output, payload)
    complete_evaluation_wave_release(
        plan=plan, wave_path=wave_path, index_path=index_path,
        index=payload, clean_environment=clean_environment,
    )
    return payload


def validate_evaluation_wave_submission(
    *, repo: Path, plan_path: Path, plan: Mapping[str, Any], prereg: Mapping[str, Any]
) -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
    wave_path, wave = validate_evaluation_wave_authorization(
        repo=repo, plan_path=plan_path, plan=plan, prereg=prereg
    )
    expect(not Path(plan["evaluation_wave_submission_failure"]).exists(), "evaluation wave has a terminal submission failure")
    expect(not Path(plan["evaluation_wave_release_failure"]).exists(), "evaluation wave has a terminal release failure")
    index_path = regular_readonly(plan["evaluation_wave_submission_index"], "evaluation-wave submission index")
    index = load_json(index_path, "evaluation-wave submission index")
    expected_keys = {
        "schema_version", "receipt", "status", "created_utc", "authorization", "release_plan",
        "target_order", "evaluated_target_order", "target_submission_receipts", "job_chains", "submission_journal",
        "wave_finalizer", "all_job_ids", "all_jobs_verified_held", "global_gpu_limit",
        "heldout_outcomes_inspected",
    }
    expect(set(index) == expected_keys, "evaluation-wave submission index schema drifted")
    expect(index.get("schema_version") == SCHEMA_VERSION and index.get("receipt") == "opd_math_objective_family_evaluation_wave_submission_v1", "evaluation-wave submission index identity drifted")
    expect(index.get("status") == "all_jobs_held_and_indexed_before_release" and index.get("authorization") == {"path": str(wave_path), "sha256": sha256_file(wave_path)}, "evaluation-wave submission authorization drifted")
    expect(index.get("release_plan") == {"path": str(plan_path), "sha256": sha256_file(plan_path)}, "evaluation-wave submission plan drifted")
    expect(index.get("target_order") == wave["target_order"] and index.get("evaluated_target_order") == wave["evaluated_target_order"], "evaluation-wave submission target order drifted")
    expect(index.get("all_jobs_verified_held") is True and index.get("global_gpu_limit") == 4 and index.get("heldout_outcomes_inspected") is False, "evaluation-wave submission boundary drifted")
    journal = index.get("submission_journal") or {}
    journal_root = Path(plan["evaluation_wave_submission_journal_root"]).resolve()
    expect(Path(str(journal.get("path"))).resolve() == journal_root and journal.get("tree_sha256") == sha256_tree(journal_root), "evaluation-wave submission journal drifted")
    expect(isinstance(journal.get("entries"), list) and len(journal["entries"]) == 3 * len(wave["evaluated_target_order"]) + 1, "evaluation-wave submission journal entry count drifted")
    for ordinal, entry in enumerate(journal["entries"]):
        intent_path = validate_binding(entry.get("intent"), f"evaluation-wave journal intent {ordinal}")
        result_path = validate_binding(entry.get("result"), f"evaluation-wave journal result {ordinal}")
        expect(load_json(intent_path, f"journal intent {ordinal}").get("ordinal") == ordinal and load_json(result_path, f"journal result {ordinal}").get("ordinal") == ordinal, f"evaluation-wave journal ordinal drifted: {ordinal}")
    expect(set(index.get("target_submission_receipts", {})) == set(wave["evaluated_target_order"]), "evaluation-wave target receipt matrix drifted")
    graph_job_ids: list[str] = []
    previous_scope = None
    previous_seal_job = None
    wrapper = Path(load_json(plan["program_manifest"]["path"], "program manifest")["evaluation_wrapper"]["path"]).resolve()
    for scope in wave["evaluated_target_order"]:
        binding = index["target_submission_receipts"][scope]
        receipt_path = validate_binding(binding, f"evaluation-wave target submission {scope}")
        receipt = load_json(receipt_path, f"evaluation-wave target submission {scope}")
        expect(receipt.get("scope") == scope and receipt.get("previous_evaluated_scope") == previous_scope, f"evaluation-wave target chain order drifted: {scope}")
        expect(receipt.get("wave_authorization") == file_binding(wave_path) and receipt.get("release_plan") == file_binding(plan_path), f"evaluation-wave target chain custody drifted: {scope}")
        jobs = receipt.get("jobs") or {}
        commands = receipt.get("commands") or {}
        expect(set(jobs) == {"array", "merge_supervisor", "seal_supervisor"} and set(commands) == {"array", "merge_supervisor", "seal_supervisor"}, f"evaluation-wave target graph schema drifted: {scope}")
        array_id = jobs["array"]["job_id"]
        merge_id = jobs["merge_supervisor"]["job_id"]
        seal_id = jobs["seal_supervisor"]["job_id"]
        graph_job_ids.extend((array_id, merge_id, seal_id))
        array_command = commands["array"]
        merge_command = commands["merge_supervisor"]
        seal_command = commands["seal_supervisor"]
        expect(array_command[-1] == str(wrapper) and merge_command[-1] == str(wrapper) and seal_command[-1] == str(wrapper), f"evaluation-wave wrapper graph drifted: {scope}")
        for token in ("--hold", "--no-requeue", "--partition=general-gpu", "--gpus=a100-sxm4:1", "--cpus-per-task=8", "--mem=96G", "--time=24:00:00", f"--array={EVALUATION_ARRAY_SPEC}"):
            expect(token in array_command, f"evaluation-wave array resource graph drifted: {scope} {token}")
        expected_array_dependency = None if previous_seal_job is None else f"--dependency=afterany:{previous_seal_job}"
        observed_array_dependencies = [value for value in array_command if value.startswith("--dependency=")]
        expect(observed_array_dependencies == ([] if expected_array_dependency is None else [expected_array_dependency]), f"evaluation-wave array dependency graph drifted: {scope}")
        expect(f"--dependency=afterany:{array_id}" in merge_command and f"--dependency=afterany:{merge_id}" in seal_command, f"evaluation-wave supervisor dependency graph drifted: {scope}")
        for command in (merge_command, seal_command):
            for token in ("--hold", "--no-requeue", "--partition=general-cpu", "--cpus-per-task=4", "--mem=32G", "--time=02:00:00"):
                expect(token in command, f"evaluation-wave supervisor resource graph drifted: {scope} {token}")
        snapshots = receipt.get("held_snapshots") or {}
        array_output_tokens = [token for token in array_command if token.startswith("--output=")]
        expect(len(array_output_tokens) == 1 and Path(str(snapshots.get("array", {}).get("std_out"))).resolve() == Path(array_output_tokens[0].split("=", 1)[1]).resolve(), f"evaluation-wave array stdout snapshot drifted: {scope}")
        expect(jobs["merge_supervisor"].get("dependency") == f"afterany:{array_id}" and jobs["seal_supervisor"].get("dependency") == f"afterany:{merge_id}", f"evaluation-wave stored dependency drifted: {scope}")
        previous_scope = scope
        previous_seal_job = seal_id
    all_jobs = index.get("all_job_ids")
    expect(isinstance(all_jobs, list) and len(all_jobs) == 3 * len(wave["evaluated_target_order"]) + 1 and len(set(all_jobs)) == len(all_jobs), "evaluation-wave job ID ledger drifted")
    expect(all(isinstance(value, str) and re.fullmatch(r"[1-9][0-9]*", value) for value in all_jobs), "evaluation-wave job ID invalid")
    finalizer = index.get("wave_finalizer") or {}
    expect(previous_seal_job is not None and finalizer.get("dependency") == f"afterany:{previous_seal_job}", "evaluation-wave finalizer dependency graph drifted")
    finalizer_command = finalizer.get("command") or []
    for token in ("--hold", "--no-requeue", "--partition=general-cpu", "--cpus-per-task=4", "--mem=32G", "--time=02:00:00", f"--dependency=afterany:{previous_seal_job}"):
        expect(token in finalizer_command, f"evaluation-wave finalizer resource graph drifted: {token}")
    expect(finalizer_command[-1] == str(wrapper), "evaluation-wave finalizer wrapper drifted")
    graph_job_ids.append(finalizer["job_id"])
    expect(graph_job_ids == all_jobs, "evaluation-wave all-job ordering drifted")
    parse_utc(index.get("created_utc"), "evaluation-wave submission created_utc")
    intent_path = regular_readonly(plan["evaluation_wave_release_intent"], "evaluation-wave release intent")
    intent = load_json(intent_path, "evaluation-wave release intent")
    expect(intent == {
        "schema_version": SCHEMA_VERSION,
        "authorization": "opd_math_objective_family_evaluation_wave_release_intent_v1",
        "status": "all_held_jobs_authorized_for_one_canonical_release",
        "created_utc": intent.get("created_utc"),
        "wave_authorization": {"path": str(wave_path), "sha256": sha256_file(wave_path)},
        "submission_index": {"path": str(index_path), "sha256": sha256_file(index_path)},
        "all_job_ids": all_jobs,
        "release_command": ["scontrol", "release", ",".join(all_jobs)],
        "heldout_outcomes_inspected": False,
    }, "evaluation-wave release intent drifted")
    parse_utc(intent.get("created_utc"), "evaluation-wave release-intent created_utc")
    result_path = regular_readonly(plan["evaluation_wave_release_result"], "evaluation-wave release result")
    result = load_json(result_path, "evaluation-wave release result")
    expect(set(result) == {
        "schema_version", "receipt", "status", "created_utc", "wave_authorization",
        "submission_index", "release_intent", "all_job_ids", "release_command",
        "command_executed_this_invocation", "command_return_code",
        "command_stdout_sha256", "command_stderr_sha256", "pre_release_snapshots",
        "post_release_snapshots", "heldout_outcomes_inspected",
    }, "evaluation-wave release-result schema drifted")
    expect(result.get("schema_version") == SCHEMA_VERSION and result.get("receipt") == "opd_math_objective_family_evaluation_wave_release_result_v1" and result.get("status") == "all_held_jobs_released_and_observed", "evaluation-wave release-result identity drifted")
    expect(result.get("wave_authorization") == file_binding(wave_path) and result.get("submission_index") == file_binding(index_path) and result.get("release_intent") == file_binding(intent_path), "evaluation-wave release-result custody drifted")
    expect(result.get("all_job_ids") == all_jobs and result.get("release_command") == ["scontrol", "release", ",".join(all_jobs)], "evaluation-wave release-result jobs drifted")
    expect(result.get("command_return_code") == 0 and isinstance(result.get("command_executed_this_invocation"), bool), "evaluation-wave release-result execution drifted")
    expect(all(HEX64.fullmatch(str(result.get(field))) is not None for field in ("command_stdout_sha256", "command_stderr_sha256")), "evaluation-wave release-result stream hashes drifted")
    expect(set(result.get("post_release_snapshots", {})) == set(all_jobs), "evaluation-wave post-release snapshot matrix drifted")
    expect(all(entry.get("available") is True and entry["snapshot"].get("user_held") is False for entry in result["post_release_snapshots"].values()), "evaluation-wave release-result retains held jobs")
    expect(result.get("heldout_outcomes_inspected") is False, "evaluation-wave release-result admits outcome inspection")
    parse_utc(result.get("created_utc"), "evaluation-wave release-result created_utc")
    return index_path, index, intent_path, intent


def validate_evaluation_submission(
    *, repo: Path, release_plan: str | Path, arm_key_value: str | None, raw_source: str | None
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, str]]:
    plan_path, plan, prereg = validate_release_plan(release_plan, repo)
    _, wave_index, _, _ = validate_evaluation_wave_submission(
        repo=repo, plan_path=plan_path, plan=plan, prereg=prereg
    )
    scope, paths = _evaluation_scope(plan, arm_key_value=arm_key_value, raw_source=raw_source)
    authorization = validate_evaluation_authorization(
        repo=repo, release_plan=plan_path, arm_key_value=arm_key_value, raw_source=raw_source
    )
    path = regular_readonly(paths["submission_receipt"], f"evaluation submission {scope}")
    payload = load_json(path, f"evaluation submission {scope}")
    expect(set(payload) == {
        "schema_version", "receipt", "status", "created_utc", "scope", "authorization",
        "wave_authorization", "target_entry_sha256", "release_plan", "previous_evaluated_scope",
        "commands", "exports", "jobs", "held_snapshots", "heldout_outcomes_inspected",
    }, f"evaluation submission schema drifted: {scope}")
    expect(payload.get("schema_version") == SCHEMA_VERSION and payload.get("receipt") == "opd_math_objective_family_evaluation_target_submission_v1", f"evaluation submission identity drifted: {scope}")
    expect(payload.get("status") == "complete_target_chain_held_before_wave_release" and payload.get("scope") == scope, f"evaluation submission status drifted: {scope}")
    expect(payload.get("authorization") == {"path": authorization["path"], "sha256": authorization["sha256"]}, f"evaluation submission authorization drifted: {scope}")
    expect(payload.get("release_plan") == {"path": str(plan_path), "sha256": sha256_file(plan_path)}, f"evaluation submission plan drifted: {scope}")
    parse_utc(payload.get("created_utc"), f"evaluation submission created_utc {scope}")
    jobs = payload.get("jobs") or {}
    expect(set(jobs) == {"array", "merge_supervisor", "seal_supervisor"}, f"evaluation submission job schema drifted: {scope}")
    for label in jobs:
        expect(_parse_sbatch_id(str(jobs[label]["raw"]), f"stored {scope} {label}") == jobs[label]["job_id"], f"evaluation stored job response drifted: {scope} {label}")
    expect(wave_index["target_submission_receipts"].get(scope) == {"path": str(path), "sha256": sha256_file(path)}, f"evaluation target submission is not wave-indexed: {scope}")
    expect(payload.get("heldout_outcomes_inspected") is False, f"evaluation submission admits outcome inspection: {scope}")
    return path, payload, plan, paths


def legacy_consume_evaluation_authorization(args: argparse.Namespace) -> dict[str, Any]:
    repo = configure_repo(args.repo)
    plan_path, plan, _ = validate_release_plan(args.release_plan, repo)
    scope, paths = _evaluation_scope(
        plan,
        arm_key_value=getattr(args, "arm_key", None),
        raw_source=getattr(args, "raw_source", None),
    )
    auth_path = regular_readonly(paths["authorization"], f"evaluation authorization {scope}")
    auth = load_json(auth_path, f"evaluation authorization {scope}")
    submission_path = regular_readonly(paths["submission_receipt"], f"evaluation submission {scope}")
    submission = load_json(submission_path, f"evaluation submission {scope}")
    expect(submission.get("authorization") == {"path": str(auth_path), "sha256": sha256_file(auth_path)}, f"evaluation consumption auth binding drifted: {scope}")
    expect(submission.get("release_plan") == {"path": str(plan_path), "sha256": sha256_file(plan_path)}, f"evaluation consumption plan binding drifted: {scope}")
    expect(auth.get("scope") == submission.get("scope") == scope, f"evaluation consumption scope drifted: {scope}")
    jobs = submission.get("sbatch") or {}
    phase = args.phase
    created_utc = utc_now()
    common = {
        "schema_version": SCHEMA_VERSION,
        "receipt": "opd_math_objective_family_evaluation_authorization_consumption_v1",
        "status": "consumed_inside_registered_job_before_tracked_evaluator",
        "created_utc": created_utc,
        "scope": scope,
        "phase": phase,
        "authorization": {"path": str(auth_path), "sha256": sha256_file(auth_path)},
        "submission": {"path": str(submission_path), "sha256": sha256_file(submission_path)},
        "release_plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
        "evaluation_contract_sha256": canonical_json_sha256(auth["evaluation_contract"]),
        "heldout_outcomes_inspected": False,
    }
    if phase == "shard":
        expect(os.environ.get("SLURM_ARRAY_JOB_ID") == jobs.get("array_job_id"), "shard consumption ran in wrong Slurm array")
        observed_index = os.environ.get("SLURM_ARRAY_TASK_ID")
        expect(observed_index is not None and observed_index.isdigit(), "shard consumption lacks array index")
        shard_index = int(observed_index)
        expect(shard_index == args.shard_index and 0 <= shard_index < EVALUATION_SHARDS, "shard consumption index drifted")
        output = Path(paths["shard_consumption_root"]) / f"shard_{shard_index:05d}.json"
        expect(Path(args.output).resolve() == output.resolve(), "shard consumption output drifted")
        shard_root = Path(paths["artifact_root"]) / "shards" / f"shard_{shard_index:05d}"
        shard_companion = Path(str(shard_root) + ".custody.json")
        expect(not shard_root.exists() and not shard_companion.exists(), "shard artifacts existed before authorization consumption")
        payload = {
            **common,
            "scheduler_job_id": os.environ.get("SLURM_JOB_ID"),
            "array_job_id": jobs["array_job_id"],
            "shard_index": shard_index,
            "expected_shard_root": str(shard_root.resolve()),
            "expected_shard_companion": str(shard_companion.resolve()),
        }
    else:
        expect(phase == "merge", "evaluation consumption phase is invalid")
        expect(os.environ.get("SLURM_JOB_ID") == jobs.get("merge_job_id"), "merge consumption ran in wrong Slurm job")
        expect(Path(args.output).resolve() == Path(paths["merge_consumption_receipt"]).resolve(), "merge consumption output drifted")
        shard_bindings = {}
        for index in range(EVALUATION_SHARDS):
            receipt_path = regular_readonly(
                Path(paths["shard_consumption_root"]) / f"shard_{index:05d}.json",
                f"evaluation shard consumption {scope} {index}",
            )
            shard_root = Path(paths["artifact_root"]) / "shards" / f"shard_{index:05d}"
            shard_companion = Path(str(shard_root) + ".custody.json")
            expect(shard_root.is_dir() and shard_companion.is_file(), f"merge consumption lacks completed shard {scope} {index}")
            shard_bindings[str(index)] = {
                "consumption": file_binding(receipt_path),
                "artifact_tree_sha256": sha256_tree(shard_root),
                "companion": file_binding(shard_companion, readonly=False),
            }
        expect(not Path(paths["summary"]).exists() and not Path(paths["samples"]).exists() and not Path(paths["companion"]).exists(), "merged artifacts existed before merge consumption")
        payload = {
            **common,
            "scheduler_job_id": jobs["merge_job_id"],
            "array_job_id": jobs["array_job_id"],
            "shards": shard_bindings,
            "expected_summary": paths["summary"],
            "expected_samples": paths["samples"],
            "expected_companion": paths["companion"],
        }
    write_new(args.output, payload)
    return payload


def load_evaluation_worker_context(
    *, repo: Path, release_plan: str | Path, arm_key_value: str | None,
    raw_source: str | None,
) -> tuple[Path, dict[str, Any], str, dict[str, str], Path, dict[str, Any], Path, dict[str, Any]]:
    """Load the sealed worker token chain without replaying preregistration trees."""

    repo = configure_repo(repo)
    plan_path = regular_readonly(release_plan, "worker release plan")
    plan = load_json(plan_path, "worker release plan")
    expect(plan.get("schema_version") == SCHEMA_VERSION and plan.get("release_plan") == PLAN_ID and plan.get("git_commit") == EXPECTED_COMMIT, "worker release-plan identity drifted")
    program_path = validate_binding(plan.get("program_manifest"), "worker program manifest")
    program = load_json(program_path, "worker program manifest")
    expect(validate_binding(program["program_file"], "worker controller") == Path(__file__).resolve(), "worker controller differs from sealed program")
    expect(validate_binding(program["evaluation_wrapper"], "worker wrapper") == Path(__file__).with_name("objective_family_evaluation_job.sh").resolve(), "worker wrapper differs from sealed program")
    scope, paths = _evaluation_scope(plan, arm_key_value=arm_key_value, raw_source=raw_source)
    wave_path = regular_readonly(plan["evaluation_wave_authorization"], "worker wave authorization")
    wave = load_json(wave_path, "worker wave authorization")
    expect(wave.get("release_plan") == file_binding(plan_path) and wave.get("controller") == program["program_file"] and wave.get("wrapper") == program["evaluation_wrapper"], "worker wave custody drifted")
    target = (wave.get("targets") or {}).get(scope)
    expect(isinstance(target, dict) and target.get("status") == "evaluate" and wave.get("target_entry_sha256s", {}).get(scope) == canonical_json_sha256(target), f"worker wave target drifted: {scope}")
    auth_path = validate_binding(target["authorization"], f"worker authorization {scope}")
    auth = load_json(auth_path, f"worker authorization {scope}")
    expect(auth.get("scope") == scope and auth.get("release_plan") == file_binding(plan_path) and auth.get("evaluation_contract") == target.get("evaluation_contract"), f"worker authorization payload drifted: {scope}")
    expect(canonical_json_sha256(auth["evaluation_contract"]) == target.get("evaluation_contract_sha256"), f"worker contract hash drifted: {scope}")
    index_path = regular_readonly(plan["evaluation_wave_submission_index"], "worker wave submission index")
    index = load_json(index_path, "worker wave submission index")
    expect(index.get("authorization") == file_binding(wave_path) and index.get("release_plan") == file_binding(plan_path), "worker wave index custody drifted")
    wait_for_regular_readonly(plan["evaluation_wave_release_result"], "worker wave release result")
    release_result = load_json(plan["evaluation_wave_release_result"], "worker wave release result")
    expect(release_result.get("status") == "all_held_jobs_released_and_observed" and release_result.get("submission_index") == file_binding(index_path), "worker release result drifted")
    submission_path = regular_readonly(paths["submission_receipt"], f"worker submission {scope}")
    submission = load_json(submission_path, f"worker submission {scope}")
    expect(submission.get("scope") == scope and submission.get("authorization") == file_binding(auth_path) and submission.get("wave_authorization") == file_binding(wave_path) and submission.get("release_plan") == file_binding(plan_path), f"worker submission custody drifted: {scope}")
    expect(index.get("target_submission_receipts", {}).get(scope) == file_binding(submission_path), f"worker submission is not wave-indexed: {scope}")
    expect(set(submission.get("jobs") or {}) == {"array", "merge_supervisor", "seal_supervisor"}, f"worker submission jobs drifted: {scope}")
    if auth.get("training_custody") is not None:
        adapter = auth["training_custody"]["final_adapter"]
        expect(adapter.get("tree_sha256") == sha256_tree(adapter["path"]), f"worker adapter hash drifted: {scope}")
    return plan_path, plan, scope, paths, auth_path, auth, submission_path, submission


def consume_evaluation_authorization(args: argparse.Namespace) -> dict[str, Any]:
    repo = Path(args.repo).resolve()
    plan_path, plan, scope, paths, auth_path, auth, submission_path, submission = load_evaluation_worker_context(
        repo=repo, release_plan=args.release_plan,
        arm_key_value=getattr(args, "arm_key", None),
        raw_source=getattr(args, "raw_source", None),
    )
    expect(args.phase == "shard", "only shard jobs consume evaluation authorization directly")
    jobs = submission["jobs"]
    expect(os.environ.get("SLURM_ARRAY_JOB_ID") == jobs["array"]["job_id"], "shard consumption ran in wrong Slurm array")
    observed_index = os.environ.get("SLURM_ARRAY_TASK_ID")
    expect(observed_index is not None and observed_index.isdigit(), "shard consumption lacks array index")
    shard_index = int(observed_index)
    expect(shard_index == args.shard_index and 0 <= shard_index < EVALUATION_SHARDS, "shard consumption index drifted")
    output = Path(paths["shard_consumption_root"]) / f"shard_{shard_index:05d}.json"
    expect(Path(args.output).resolve() == output.resolve(), "shard consumption output drifted")
    shard_root = Path(paths["artifact_root"]) / "shards" / f"shard_{shard_index:05d}"
    shard_companion = Path(str(shard_root) + ".custody.json")
    expect(not shard_root.exists() and not shard_companion.exists(), "shard artifacts existed before authorization consumption")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "receipt": "opd_math_objective_family_evaluation_authorization_consumption_v1",
        "status": "consumed_inside_registered_job_before_tracked_evaluator",
        "created_utc": utc_now(),
        "scope": scope,
        "phase": "shard",
        "authorization": {"path": str(auth_path), "sha256": sha256_file(auth_path)},
        "submission": {"path": str(submission_path), "sha256": sha256_file(submission_path)},
        "release_plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
        "evaluation_contract_sha256": canonical_json_sha256(auth["evaluation_contract"]),
        "scheduler_job_id": os.environ.get("SLURM_JOB_ID"),
        "array_job_id": jobs["array"]["job_id"],
        "shard_index": shard_index,
        "expected_shard_root": str(shard_root.resolve()),
        "expected_shard_companion": str(shard_companion.resolve()),
        "heldout_outcomes_inspected": False,
    }
    write_new(args.output, payload)
    return payload


def _tres_tokens(value: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for token in value.split(","):
        if not token:
            continue
        name, separator, raw = token.partition("=")
        expect(separator and name and raw, f"invalid TRES token: {token}")
        result[name] = raw
    return result


def _memory_mebibytes(value: str | None) -> int | None:
    if not value:
        return None
    match = re.fullmatch(r"([0-9]+)([KMGT])", value)
    if match is None:
        return None
    amount = int(match.group(1))
    multiplier = {"K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024}[match.group(2)]
    return int(amount * multiplier)


def query_evaluation_array(
    array_job_id: str, *, raw_output_path: str | Path | None = None,
    expected_log_root: str | Path | None = None,
) -> tuple[list[str], str, dict[str, dict[str, Any]]]:
    expect(re.fullmatch(r"[1-9][0-9]*", array_job_id) is not None, "evaluation array job ID is invalid")
    command = [
        "sacct", "-X", "--array", "-n", "-P", "-j", array_job_id,
        "--format=JobID,JobIDRaw,JobName,Partition,Account,State,ExitCode,ElapsedRaw,ReqTRES,AllocTRES,Submit,Start,End,NodeList,StdOut",
    ]
    raw = ""
    parsed: dict[str, dict[str, Any]] = {}
    for attempt in range(30):
        raw = subprocess.run(command, check=True, text=True, capture_output=True).stdout
        parsed = {}
        for line in raw.splitlines():
            if not line.strip():
                continue
            parts = line.split("|")
            if len(parts) < 15:
                continue
            logical, raw_id, name, partition, account, state, exit_code, elapsed, req_tres, alloc_tres, submit, start, end, node, stdout = parts[:15]
            match = re.fullmatch(re.escape(array_job_id) + r"_([0-5])", logical)
            if match is None:
                continue
            index = match.group(1)
            if index in parsed:
                parsed[index]["duplicate"] = True
                continue
            normalized = state.split()[0].split("+")[0]
            parsed[index] = {
                "logical_job_id": logical,
                "job_id_raw": raw_id,
                "job_name": name,
                "partition": partition,
                "account": account,
                "state": normalized,
                "state_raw": state,
                "exit_code": exit_code,
                "elapsed_seconds": int(elapsed) if elapsed.isdigit() else None,
                "req_tres": req_tres,
                "alloc_tres": alloc_tres,
                "submit": submit,
                "start": start,
                "end": end,
                "node_list": node,
                "stdout": stdout,
            }
        if set(parsed) == {str(index) for index in range(EVALUATION_SHARDS)} and all(row["state"] in TERMINAL_STATES for row in parsed.values()):
            break
        if attempt < 29:
            time.sleep(2)
    if raw_output_path is not None:
        write_new_or_exact(raw_output_path, raw)
    expect(set(parsed) == {str(index) for index in range(EVALUATION_SHARDS)}, "evaluation array accounting lacks exact indices 0-5")
    seen_raw_ids: set[str] = set()
    for index, row in parsed.items():
        expect(not row.get("duplicate"), f"duplicate evaluation array index: {array_job_id}_{index}")
        expect(row["logical_job_id"] == f"{array_job_id}_{index}", f"evaluation array logical identity drifted: {array_job_id}_{index}")
        expect(isinstance(row["job_id_raw"], str) and row["job_id_raw"] and row["job_id_raw"] not in seen_raw_ids, f"evaluation array raw identity drifted: {array_job_id}_{index}")
        seen_raw_ids.add(row["job_id_raw"])
        expect(row["job_name"] == "opd_obj_eval_shard", f"evaluation array job name drifted: {array_job_id}_{index}")
        expect(row["state"] in TERMINAL_STATES, f"evaluation array task is not terminal: {array_job_id}_{index}")
        expect(row["partition"] == "general-gpu" and row["account"] == "engr-lab-jacobsn", f"evaluation array lane drifted: {array_job_id}_{index}")
        req = _tres_tokens(row["req_tres"])
        expect(req.get("cpu") == "8" and _memory_mebibytes(req.get("mem")) == 96 * 1024, f"evaluation array requested CPU/memory drifted: {array_job_id}_{index}")
        expect(req.get("gres/gpu") == "1" and req.get("gres/gpu:a100-sxm4") == "1", f"evaluation array requested GPU drifted: {array_job_id}_{index}")
        if row["alloc_tres"]:
            alloc = _tres_tokens(row["alloc_tres"])
            expect(alloc.get("cpu") == "8" and _memory_mebibytes(alloc.get("mem")) == 96 * 1024, f"evaluation array allocated CPU/memory drifted: {array_job_id}_{index}")
            expect(alloc.get("gres/gpu") == "1" and alloc.get("gres/gpu:a100-sxm4") == "1", f"evaluation array allocated GPU drifted: {array_job_id}_{index}")
        else:
            expect(row["state"] != "COMPLETED", f"completed evaluation array task lacks allocations: {array_job_id}_{index}")
        if expected_log_root is not None:
            expected_stdout = Path(expected_log_root).resolve() / "shard-%A_%a.public.out"
            expect(Path(row["stdout"]).resolve() == expected_stdout, f"evaluation array stdout path drifted: {array_job_id}_{index}")
    return command, raw, parsed


def validate_shard_consumption_receipt(
    *, path: Path, scope: str, index: int, submission_path: Path, authorization_path: Path,
    array_job_id: str, plan_path: Path, paths: Mapping[str, str],
    accounting_row: Mapping[str, Any], evaluation_contract_sha256: str,
) -> dict[str, Any]:
    resolved = regular_readonly(path, f"shard consumption {scope} {index}")
    payload = load_json(resolved, f"shard consumption {scope} {index}")
    expected_keys = {
        "schema_version", "receipt", "status", "created_utc", "scope", "phase", "authorization",
        "submission", "release_plan", "evaluation_contract_sha256", "scheduler_job_id",
        "array_job_id", "shard_index", "expected_shard_root", "expected_shard_companion",
        "heldout_outcomes_inspected",
    }
    expect(set(payload) == expected_keys, f"shard consumption schema drifted: {scope} {index}")
    expect(payload.get("schema_version") == SCHEMA_VERSION and payload.get("receipt") == "opd_math_objective_family_evaluation_authorization_consumption_v1", f"shard consumption identity drifted: {scope} {index}")
    expect(payload.get("scope") == scope and payload.get("phase") == "shard" and payload.get("shard_index") == index and payload.get("array_job_id") == array_job_id, f"shard consumption routing drifted: {scope} {index}")
    expect(payload.get("authorization") == file_binding(authorization_path) and payload.get("submission") == file_binding(submission_path), f"shard consumption custody drifted: {scope} {index}")
    expect(payload.get("release_plan") == file_binding(plan_path), f"shard consumption release-plan drifted: {scope} {index}")
    expect(payload.get("evaluation_contract_sha256") == evaluation_contract_sha256, f"shard consumption contract drifted: {scope} {index}")
    expect(payload.get("scheduler_job_id") in {accounting_row.get("logical_job_id"), accounting_row.get("job_id_raw")}, f"shard consumption scheduler task drifted: {scope} {index}")
    expected_root = (Path(paths["artifact_root"]) / "shards" / f"shard_{index:05d}").resolve()
    expect(Path(str(payload.get("expected_shard_root"))).resolve() == expected_root and Path(str(payload.get("expected_shard_companion"))).resolve() == Path(str(expected_root) + ".custody.json").resolve(), f"shard consumption output paths drifted: {scope} {index}")
    expect(payload.get("heldout_outcomes_inspected") is False, f"shard consumption outcome boundary drifted: {scope} {index}")
    parse_utc(payload.get("created_utc"), f"shard consumption created_utc {scope} {index}")
    return payload


def _write_evaluation_failure(
    *, plan_path: Path, paths: Mapping[str, str], scope: str, stage: str, failure_class: str,
    authorization_path: Path, submission_path: Path, scheduler_evidence: Mapping[str, Any],
    partial_artifacts: Mapping[str, Any], merge_invoked: bool,
    machine_validation_may_have_read_outcome_bytes: bool = False,
) -> tuple[Path, dict[str, Any]]:
    failure_path = Path(paths["terminal_failure_receipt"]).resolve()
    if failure_path.exists():
        return regular_readonly(failure_path, f"terminal failure {scope}"), load_json(failure_path, f"terminal failure {scope}")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "receipt": "opd_math_objective_family_evaluation_target_terminal_failure_v1",
        "status": "terminal_evaluation_failure_no_scientific_outcome",
        "created_utc": utc_now(),
        "scope": scope,
        "failure_stage": stage,
        "failure_class": failure_class,
        "authorization": file_binding(authorization_path),
        "submission": file_binding(submission_path),
        "release_plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
        "scheduler_evidence": dict(scheduler_evidence),
        "partial_artifacts": dict(partial_artifacts),
        "merge_invoked": merge_invoked,
        "merged_evaluation_eligible": False,
        "retry_authorized": False,
        "numeric_result_fields_copied": False,
        "machine_validation_may_have_read_outcome_bytes": machine_validation_may_have_read_outcome_bytes,
        "human_outcomes_inspected": False,
    }
    return write_new(failure_path, payload), payload


def supervise_evaluation_merge(args: argparse.Namespace) -> dict[str, Any]:
    repo = Path(args.repo).resolve()
    plan_path, plan, scope, paths, authorization_path, _, submission_path, submission = load_evaluation_worker_context(
        repo=repo, release_plan=args.release_plan,
        arm_key_value=getattr(args, "arm_key", None), raw_source=getattr(args, "raw_source", None),
    )
    jobs = submission["jobs"]
    expect(os.environ.get("SLURM_JOB_ID") == jobs["merge_supervisor"]["job_id"], "merge supervisor ran in wrong Slurm job")
    expect(Path(args.output).resolve() == Path(paths["merge_supervisor_receipt"]).resolve(), "merge supervisor output drifted")
    command, raw, rows = query_evaluation_array(
        jobs["array"]["job_id"], raw_output_path=paths["array_accounting_raw"],
        expected_log_root=paths["log_root"],
    )
    raw_path = regular_readonly(paths["array_accounting_raw"], f"array accounting raw {scope}")
    accounting = {
        "schema_version": SCHEMA_VERSION,
        "receipt": "opd_math_objective_family_evaluation_array_accounting_v1",
        "array_job_id": jobs["array"]["job_id"],
        "command": command,
        "raw": file_binding(raw_path),
        "expected_indices": list(range(EVALUATION_SHARDS)),
        "rows": rows,
        "all_terminal": True,
        "all_successful": all(row["state"] == "COMPLETED" and row["exit_code"] == "0:0" for row in rows.values()),
    }
    accounting_path = write_new(paths["array_accounting_receipt"], accounting)
    shard_bindings: dict[str, Any] = {}
    partial: dict[str, Any] = {}
    for index in range(EVALUATION_SHARDS):
        receipt_path = Path(paths["shard_consumption_root"]) / f"shard_{index:05d}.json"
        shard_root = Path(paths["artifact_root"]) / "shards" / f"shard_{index:05d}"
        companion = Path(str(shard_root) + ".custody.json")
        if receipt_path.is_file() and not receipt_path.is_symlink():
            validate_shard_consumption_receipt(
                path=receipt_path, scope=scope, index=index, submission_path=submission_path,
                authorization_path=authorization_path, array_job_id=jobs["array"]["job_id"],
                plan_path=plan_path, paths=paths, accounting_row=rows[str(index)],
                evaluation_contract_sha256=canonical_json_sha256(
                    load_json(authorization_path, f"merge authorization {scope}")["evaluation_contract"]
                ),
            )
            partial[str(index)] = {"consumption": file_binding(receipt_path)}
        if shard_root.is_dir() and not shard_root.is_symlink():
            partial.setdefault(str(index), {})["artifact_tree_sha256"] = sha256_tree(shard_root)
        if companion.is_file() and not companion.is_symlink():
            partial.setdefault(str(index), {})["companion"] = file_binding(companion, readonly=False)
        if accounting["all_successful"]:
            expect(receipt_path.is_file() and shard_root.is_dir() and companion.is_file(), f"successful array lacks shard artifacts: {scope} {index}")
            shard_bindings[str(index)] = partial[str(index)]
    if not accounting["all_successful"]:
        failure_path, _ = _write_evaluation_failure(
            plan_path=plan_path, paths=paths, scope=scope, stage="shard_array",
            failure_class="scheduler_nonzero", authorization_path=authorization_path,
            submission_path=submission_path, scheduler_evidence={"array_accounting": file_binding(accounting_path)},
            partial_artifacts=partial, merge_invoked=False,
        )
        payload = {
            "schema_version": SCHEMA_VERSION, "receipt": "opd_math_objective_family_evaluation_merge_supervisor_v1",
            "status": "skipped_due_to_shard_failure", "created_utc": utc_now(), "scope": scope,
            "submission": file_binding(submission_path), "array_accounting": file_binding(accounting_path),
            "terminal_failure": file_binding(failure_path), "merge_private_log": None,
            "machine_outcomes_inspected": False, "human_outcomes_inspected": False,
        }
        write_new(args.output, payload)
        return payload
    merge_consumption = {
        "schema_version": SCHEMA_VERSION,
        "receipt": "opd_math_objective_family_evaluation_merge_consumption_v1",
        "status": "all_six_shards_validated_before_tracked_merge",
        "created_utc": utc_now(),
        "scope": scope,
        "authorization": file_binding(authorization_path),
        "submission": file_binding(submission_path),
        "array_accounting": file_binding(accounting_path),
        "shards": shard_bindings,
        "expected_summary": paths["summary"],
        "expected_samples": paths["samples"],
        "expected_companion": paths["companion"],
        "human_outcomes_inspected": False,
    }
    merge_consumption_path = write_new(paths["merge_consumption_receipt"], merge_consumption)
    private_root = Path(paths["private_log_root"]).resolve()
    private_root.mkdir(parents=True, exist_ok=True)
    private_partial = private_root / f"merge.private.log.partial.{os.environ.get('SLURM_JOB_ID')}"
    private_log = private_root / "merge.private.log"
    expect(not private_partial.exists() and not private_log.exists(), "merge private log already exists")
    tracked = repo / "scripts/hpc/slurm_opd_math_merge_evaluation.sh"
    with private_partial.open("xb") as handle:
        completed = subprocess.run(["/bin/bash", str(tracked)], stdout=handle, stderr=subprocess.STDOUT, env=os.environ.copy())
        handle.flush()
        os.fsync(handle.fileno())
    private_partial.rename(private_log)
    os.chmod(private_log, 0o400)
    if completed.returncode != 0:
        failure_path, _ = _write_evaluation_failure(
            plan_path=plan_path, paths=paths, scope=scope, stage="merge_execution",
            failure_class="tracked_merge_nonzero", authorization_path=authorization_path,
            submission_path=submission_path, scheduler_evidence={"return_code": completed.returncode},
            partial_artifacts={"merge_private_log": file_binding(private_log)}, merge_invoked=True,
        )
        status = "handled_merge_execution_failure"
    else:
        expect(Path(paths["summary"]).is_file() and Path(paths["samples"]).is_file() and Path(paths["companion"]).is_file(), f"tracked merge did not create complete outputs: {scope}")
        failure_path = None
        status = "tracked_merge_completed_pending_seal"
    payload = {
        "schema_version": SCHEMA_VERSION, "receipt": "opd_math_objective_family_evaluation_merge_supervisor_v1",
        "status": status, "created_utc": utc_now(), "scope": scope,
        "submission": file_binding(submission_path), "array_accounting": file_binding(accounting_path),
        "merge_consumption": file_binding(merge_consumption_path),
        "terminal_failure": None if failure_path is None else file_binding(failure_path),
        "merge_private_log": file_binding(private_log),
        "machine_outcomes_inspected": False, "human_outcomes_inspected": False,
    }
    write_new(args.output, payload)
    return payload


def supervise_evaluation_seal(args: argparse.Namespace) -> dict[str, Any]:
    repo = configure_repo(args.repo)
    plan_path, plan, prereg = validate_release_plan(args.release_plan, repo)
    scope, paths = _evaluation_scope(plan, arm_key_value=getattr(args, "arm_key", None), raw_source=getattr(args, "raw_source", None))
    submission_path, submission, _, _ = validate_evaluation_submission(
        repo=repo, release_plan=plan_path, arm_key_value=getattr(args, "arm_key", None), raw_source=getattr(args, "raw_source", None)
    )
    jobs = submission["jobs"]
    expect(os.environ.get("SLURM_JOB_ID") == jobs["seal_supervisor"]["job_id"], "seal supervisor ran in wrong Slurm job")
    expect(Path(args.output).resolve() == Path(paths["seal_supervisor_receipt"]).resolve(), "seal supervisor output drifted")
    authorization_path = regular_readonly(paths["authorization"], f"seal authorization {scope}")
    auth = load_json(authorization_path, f"seal authorization {scope}")
    merge_path = regular_readonly(paths["merge_supervisor_receipt"], f"merge supervisor {scope}")
    merge = load_json(merge_path, f"merge supervisor {scope}")
    artifact_root = Path(paths["artifact_root"]).resolve()
    failure_path = Path(paths["terminal_failure_receipt"]).resolve()
    validation_status: str
    machine_read = False
    if failure_path.exists():
        regular_readonly(failure_path, f"terminal failure {scope}")
        if artifact_root.is_dir() and not artifact_root.is_symlink():
            seal_tree_readonly(artifact_root)
        validation_status = "terminal_evaluation_failure"
    else:
        summary_path = Path(paths["summary"]).resolve()
        samples_path = Path(paths["samples"]).resolve()
        companion_path = Path(paths["companion"]).resolve()
        expect(merge.get("status") == "tracked_merge_completed_pending_seal", f"seal supervisor lacks successful merge: {scope}")
        expect(artifact_root.is_dir() and summary_path.is_file() and samples_path.is_file() and companion_path.is_file(), f"evaluation merged artifact is incomplete: {scope}")
        seal_tree_readonly(artifact_root)
        source = auth["evaluation_contract"]["source"]
        expected_adapter = None if auth.get("training_custody") is None else auth["training_custody"]["final_adapter"]
        machine_read = True
        try:
            checked_holdout(
                repo=repo, prereg=prereg, plan=plan, source=source,
                summary_path=summary_path, samples_path=samples_path,
                expected_adapter=expected_adapter,
            )
        except RuntimeError as error:
            expect(str(error).startswith("evaluation verifier-error fraction exceeds the registered cap"), f"unexpected heldout runtime failure: {scope}")
            validation_status = "terminal_verifier_cap_failure"
        except BaseException as error:
            failure_path, _ = _write_evaluation_failure(
                plan_path=plan_path, paths=paths, scope=scope, stage="heldout_validation",
                failure_class=type(error).__name__, authorization_path=authorization_path,
                submission_path=submission_path,
                scheduler_evidence={"error_sha256": hashlib.sha256(str(error).encode("utf-8")).hexdigest()},
                partial_artifacts={
                    "summary": file_binding(summary_path), "samples": file_binding(samples_path),
                    "companion": file_binding(companion_path),
                },
                merge_invoked=True,
                machine_validation_may_have_read_outcome_bytes=True,
            )
            validation_status = "terminal_evaluation_failure"
        else:
            validation_status = "valid_merged_evaluation"
    consumption = {
        "schema_version": SCHEMA_VERSION,
        "receipt": "opd_math_objective_family_evaluation_consumption_v1",
        "status": validation_status,
        "created_utc": utc_now(),
        "scope": scope,
        "authorization": file_binding(authorization_path),
        "submission": file_binding(submission_path),
        "merge_supervisor": file_binding(merge_path),
        "terminal_failure": None if not failure_path.exists() else file_binding(failure_path),
        "artifact_root": None if not artifact_root.is_dir() else {"path": str(artifact_root), "tree_sha256": sha256_tree(artifact_root)},
        "numeric_result_fields_copied": False,
        "machine_validation_read_outcome_artifacts": machine_read,
        "human_outcomes_inspected": False,
    }
    consumption_path = write_new(paths["consumption_receipt"], consumption)
    seal = {
        "schema_version": SCHEMA_VERSION,
        "seal": "opd_math_objective_family_evaluation_target_seal_v1",
        "status": validation_status,
        "created_utc": utc_now(),
        "scope": scope,
        "authorization": file_binding(authorization_path),
        "submission": file_binding(submission_path),
        "consumption": file_binding(consumption_path),
        "artifact_tree_sha256": None if not artifact_root.is_dir() else sha256_tree(artifact_root),
        "machine_validation_read_outcome_artifacts": machine_read,
        "human_outcomes_inspected": False,
        "scientific_result_claimed": False,
    }
    seal_path = write_new(paths["seal_receipt"], seal)
    supervisor = {
        "schema_version": SCHEMA_VERSION,
        "receipt": "opd_math_objective_family_evaluation_seal_supervisor_v1",
        "status": "target_terminal_seal_written",
        "created_utc": utc_now(),
        "scope": scope,
        "submission": file_binding(submission_path),
        "merge_supervisor": file_binding(merge_path),
        "target_seal": file_binding(seal_path),
        "terminal_status": validation_status,
        "machine_validation_read_outcome_artifacts": machine_read,
        "human_outcomes_inspected": False,
    }
    write_new(args.output, supervisor)
    return supervisor


def legacy_seal_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    repo = configure_repo(args.repo)
    plan_path, plan, prereg = validate_release_plan(args.release_plan, repo)
    scope, paths = _evaluation_scope(
        plan,
        arm_key_value=getattr(args, "arm_key", None),
        raw_source=getattr(args, "raw_source", None),
    )
    auth_path = regular_readonly(paths["authorization"], f"evaluation authorization {scope}")
    auth = load_json(auth_path, f"evaluation authorization {scope}")
    submission_path = regular_readonly(paths["submission_receipt"], f"evaluation submission {scope}")
    submission = load_json(submission_path, f"evaluation submission {scope}")
    merge_consumption_path = regular_readonly(paths["merge_consumption_receipt"], f"evaluation merge consumption {scope}")
    merge_consumption = load_json(merge_consumption_path, f"evaluation merge consumption {scope}")
    jobs = submission.get("sbatch") or {}
    expect(os.environ.get("SLURM_JOB_ID") == jobs.get("seal_job_id"), "evaluation seal ran in wrong Slurm job")
    expect(Path(args.output).resolve() == Path(paths["seal_receipt"]).resolve(), "evaluation seal output drifted")
    expect(merge_consumption.get("scheduler_job_id") == jobs.get("merge_job_id"), f"evaluation merge-consumption scheduler drifted: {scope}")
    for label, job_id in (("array", jobs.get("array_job_id")), ("merge", jobs.get("merge_job_id"))):
        _, scheduler = query_one_sacct(str(job_id))
        expect(scheduler["state"] == "COMPLETED" and scheduler["exit_code"] == "0:0", f"evaluation {label} did not complete: {scope}")
    artifact_root = Path(paths["artifact_root"]).resolve()
    summary_path = Path(paths["summary"]).resolve()
    samples_path = Path(paths["samples"]).resolve()
    companion_path = Path(paths["companion"]).resolve()
    expect(artifact_root.is_dir() and summary_path.is_file() and samples_path.is_file() and companion_path.is_file(), f"evaluation merged artifact is incomplete: {scope}")
    for path in (artifact_root, summary_path, samples_path, companion_path):
        expect(not path.is_symlink(), f"evaluation artifact is a symlink: {scope} {path}")
    seal_tree_readonly(artifact_root)
    source = auth["evaluation_contract"]["source"]
    expected_adapter = (
        None
        if auth.get("training_custody") is None
        else auth["training_custody"]["final_adapter"]
    )
    validation_status = "valid_merged_evaluation"
    try:
        checked_holdout(
            repo=repo,
            prereg=prereg,
            plan=plan,
            source=source,
            summary_path=summary_path,
            samples_path=samples_path,
            expected_adapter=expected_adapter,
        )
    except RuntimeError as error:
        message = str(error)
        expect(message.startswith("evaluation verifier-error fraction exceeds the registered cap"), f"unexpected evaluation validation failure: {scope}")
        validation_status = "terminal_verifier_cap_failure"
    log_root = Path(paths["log_root"]).resolve()
    log_bindings: dict[str, dict[str, str]] = {}
    expected_logs = [
        *(log_root / f"eval-{jobs['array_job_id']}_{index}.out" for index in range(EVALUATION_SHARDS)),
        log_root / f"merge-{jobs['merge_job_id']}.out",
    ]
    for log_path in expected_logs:
        expect(log_path.is_file() and not log_path.is_symlink(), f"evaluation log is missing: {log_path}")
        os.chmod(log_path, 0o444)
        log_bindings[log_path.name] = file_binding(log_path)
    accounting_command = [
        "sacct", "-X", "-n", "-P", "-j",
        f"{jobs['array_job_id']},{jobs['merge_job_id']}",
        "--format=JobIDRaw,JobName,State,ExitCode,ElapsedRaw,AllocTRES,Submit,Start,End,NodeList,StdOut",
    ]
    accounting_raw = subprocess.run(accounting_command, check=True, text=True, capture_output=True).stdout
    consumption = {
        "schema_version": SCHEMA_VERSION,
        "receipt": "opd_math_objective_family_evaluation_consumption_v1",
        "status": validation_status,
        "created_utc": utc_now(),
        "scope": scope,
        "authorization": {"path": str(auth_path), "sha256": sha256_file(auth_path)},
        "submission": {"path": str(submission_path), "sha256": sha256_file(submission_path)},
        "merge_consumption": {"path": str(merge_consumption_path), "sha256": sha256_file(merge_consumption_path)},
        "release_plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
        "scheduler_job_ids": {
            "array": jobs["array_job_id"],
            "merge": jobs["merge_job_id"],
            "seal": jobs["seal_job_id"],
        },
        "artifact_root": {"path": str(artifact_root), "tree_sha256": sha256_tree(artifact_root)},
        "summary": file_binding(summary_path),
        "samples": file_binding(samples_path),
        "companion": file_binding(companion_path),
        "logs": log_bindings,
        "accounting": {
            "command": accounting_command,
            "raw_sha256": hashlib.sha256(accounting_raw.encode("utf-8")).hexdigest(),
            "raw_text": accounting_raw,
        },
        "numeric_result_fields_copied": False,
        "outcomes_exposed_to_operator": False,
    }
    consumption_path = write_new(paths["consumption_receipt"], consumption)
    seal = {
        "schema_version": SCHEMA_VERSION,
        "seal": "opd_math_objective_family_evaluation_target_seal_v1",
        "status": validation_status,
        "created_utc": utc_now(),
        "scope": scope,
        "authorization": {"path": str(auth_path), "sha256": sha256_file(auth_path)},
        "submission": {"path": str(submission_path), "sha256": sha256_file(submission_path)},
        "consumption": {"path": str(consumption_path), "sha256": sha256_file(consumption_path)},
        "artifact_tree_sha256": sha256_tree(artifact_root),
        "heldout_outcomes_inspected": False,
        "scientific_result_claimed": False,
    }
    write_new(args.output, seal)
    return seal


def legacy_seal_evaluation_wave(args: argparse.Namespace) -> dict[str, Any]:
    repo = configure_repo(args.repo)
    plan_path, plan, prereg = validate_release_plan(args.release_plan, repo)
    _, terminal = validate_terminal_snapshot(
        plan["terminal_snapshot"], plan_path, plan, repo=repo, prereg=prereg
    )
    output = Path(args.output).resolve()
    expect(output == Path(plan["evaluation_wave_seal"]).resolve(), "evaluation-wave seal output drifted")
    target_order = ["raw_student__M", "raw_student__O", *prereg["payload"]["arm_keys"]]
    targets: dict[str, dict[str, Any]] = {}
    for scope in target_order:
        if scope.startswith("raw_student__"):
            arm_key_value = None
            raw_source = scope.rsplit("__", 1)[1]
            training_failed = False
        else:
            arm_key_value = scope
            raw_source = None
            training_failed = not terminal["arms"][scope]["training_eligible_for_heldout"]
        _, paths = _evaluation_scope(
            plan, arm_key_value=arm_key_value, raw_source=raw_source
        )
        if training_failed:
            forbidden = [
                Path(paths[field])
                for field in (
                    "artifact_root", "authorization", "submission_receipt",
                    "merge_consumption_receipt", "consumption_receipt", "seal_receipt",
                    "shard_consumption_root", "log_root",
                )
            ]
            expect(not any(path.exists() or path.is_symlink() for path in forbidden), f"failed training arm has evaluation artifacts: {scope}")
            targets[scope] = {
                "status": "terminal_training_failure_no_evaluation",
                "training_terminal": terminal["arms"][scope]["terminal_failure"],
                "submission": None,
                "consumption": None,
                "target_seal": None,
                "scheduler": None,
                "seal_log": None,
            }
            continue
        auth_path = regular_readonly(paths["authorization"], f"wave authorization {scope}")
        submission_path = regular_readonly(paths["submission_receipt"], f"wave submission {scope}")
        submission = load_json(submission_path, f"wave submission {scope}")
        expect(submission.get("authorization") == {"path": str(auth_path), "sha256": sha256_file(auth_path)}, f"wave authorization binding drifted: {scope}")
        jobs = submission.get("sbatch") or {}
        scheduler = {}
        all_completed = True
        for label, job_field in (("array", "array_job_id"), ("merge", "merge_job_id"), ("seal", "seal_job_id")):
            _, row = query_one_sacct(str(jobs.get(job_field)))
            expect(row["state"] in TERMINAL_STATES, f"evaluation target is not terminal: {scope} {label}")
            scheduler[label] = row
            all_completed = all_completed and row["state"] == "COMPLETED" and row["exit_code"] == "0:0"
        seal_log_path = Path(paths["log_root"]) / f"seal-{jobs['seal_job_id']}.out"
        seal_log_binding = None
        if seal_log_path.is_file() and not seal_log_path.is_symlink():
            os.chmod(seal_log_path, 0o444)
            seal_log_binding = file_binding(seal_log_path)
        if all_completed:
            expect(seal_log_binding is not None, f"successful wave target lacks seal log: {scope}")
            consumption_path = regular_readonly(paths["consumption_receipt"], f"wave consumption {scope}")
            target_seal_path = regular_readonly(paths["seal_receipt"], f"wave target seal {scope}")
            consumption = load_json(consumption_path, f"wave consumption {scope}")
            target_seal = load_json(target_seal_path, f"wave target seal {scope}")
            expect(target_seal.get("scope") == consumption.get("scope") == scope, f"wave target scope drifted: {scope}")
            expect(target_seal.get("consumption") == {"path": str(consumption_path), "sha256": sha256_file(consumption_path)}, f"wave target consumption binding drifted: {scope}")
            artifact_root = Path(paths["artifact_root"]).resolve()
            expect(
                all(
                    not (item.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
                    for item in [artifact_root, *artifact_root.rglob("*")]
                ),
                f"wave target artifact is not sealed: {scope}",
            )
            expect(target_seal.get("artifact_tree_sha256") == sha256_tree(artifact_root), f"wave target artifact hash drifted: {scope}")
            status = target_seal.get("status")
            expect(status in {"valid_merged_evaluation", "terminal_verifier_cap_failure"}, f"wave target status invalid: {scope}")
            targets[scope] = {
                "status": status,
                "training_terminal": None,
                "submission": file_binding(submission_path),
                "consumption": file_binding(consumption_path),
                "target_seal": file_binding(target_seal_path),
                "scheduler": scheduler,
                "seal_log": seal_log_binding,
            }
        else:
            targets[scope] = {
                "status": "terminal_evaluation_scheduler_failure",
                "training_terminal": None,
                "submission": file_binding(submission_path),
                "consumption": None,
                "target_seal": None,
                "scheduler": scheduler,
                "seal_log": seal_log_binding,
            }
    evaluation_control_root = Path(plan["control_root"]) / "evaluation"
    seal_tree_readonly(evaluation_control_root)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "seal": "opd_math_objective_family_evaluation_wave_seal_v1",
        "status": "all_38_targets_terminal_before_scientific_gate_release",
        "created_utc": utc_now(),
        "release_plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
        "terminal_snapshot": file_binding(plan["terminal_snapshot"]),
        "target_order": target_order,
        "targets": targets,
        "all_training_arms_terminal": True,
        "all_evaluation_targets_terminal": True,
        "all_valid_evaluations_consumed": True,
        "campaign_wide_consumer_authorized": True,
        "heldout_outcomes_inspected": False,
        "scientific_result_claimed": False,
    }
    write_new(output, payload)
    return payload


def seal_evaluation_wave(args: argparse.Namespace) -> dict[str, Any]:
    repo = configure_repo(args.repo)
    plan_path, plan, prereg = validate_release_plan(args.release_plan, repo)
    terminal_path, terminal = validate_terminal_snapshot(
        plan["terminal_snapshot"], plan_path, plan, repo=repo, prereg=prereg
    )
    wave_path, wave = validate_evaluation_wave_authorization(
        repo=repo, plan_path=plan_path, plan=plan, prereg=prereg
    )
    index_path, index, intent_path, _ = validate_evaluation_wave_submission(
        repo=repo, plan_path=plan_path, plan=plan, prereg=prereg
    )
    output = Path(args.output).resolve()
    expect(output == Path(plan["evaluation_wave_seal"]).resolve(), "evaluation-wave seal output drifted")
    expect(os.environ.get("SLURM_JOB_ID") == index["wave_finalizer"]["job_id"], "evaluation-wave finalizer ran in wrong Slurm job")
    if output.exists() or output.is_symlink():
        _, existing = validate_evaluation_wave_seal(
            repo=repo, plan_path=plan_path, plan=plan, prereg=prereg
        )
        return existing
    targets: dict[str, dict[str, Any]] = {}
    valid_count = 0
    verifier_cap_count = 0
    evaluation_failure_count = 0
    training_failure_count = 0
    for scope in wave["target_order"]:
        target = wave["targets"][scope]
        if target["status"] == "terminal_training_failure_no_evaluation":
            if scope.startswith("raw_student__"):
                raise ValueError("raw student target cannot have a training failure")
            _, paths = _evaluation_scope(plan, arm_key_value=scope, raw_source=None)
            expect(not any(Path(value).exists() or Path(value).is_symlink() for value in paths.values()), f"failed training target has evaluation artifacts: {scope}")
            targets[scope] = {
                "status": "terminal_training_failure_no_evaluation",
                "training_terminal": terminal["arms"][scope]["terminal_failure"],
                "submission": None,
                "array_accounting": None,
                "merge_supervisor": None,
                "seal_supervisor": None,
                "terminal_failure": None,
                "consumption": None,
                "target_seal": None,
                "public_logs": None,
            }
            training_failure_count += 1
            continue
        if scope.startswith("raw_student__"):
            arm_key_value = None
            raw_source = scope.rsplit("__", 1)[1]
        else:
            arm_key_value = scope
            raw_source = None
        _, paths = _evaluation_scope(plan, arm_key_value=arm_key_value, raw_source=raw_source)
        submission_path, submission, _, _ = validate_evaluation_submission(
            repo=repo, release_plan=plan_path, arm_key_value=arm_key_value, raw_source=raw_source
        )
        jobs = submission["jobs"]
        if not Path(paths["seal_receipt"]).exists() or not Path(paths["seal_supervisor_receipt"]).exists():
            synthetic = argparse.Namespace(
                repo=repo, release_plan=plan_path, arm_key=arm_key_value,
                raw_source=raw_source,
            )
            terminalize_supervisor_exception(
                synthetic, phase="seal_supervisor",
                error=RuntimeError("wave finalizer synthesized a missing target seal"),
            )
        accounting_path = None
        accounting_binding = None
        if Path(paths["array_accounting_receipt"]).exists():
            accounting_path = regular_readonly(paths["array_accounting_receipt"], f"wave array accounting {scope}")
            accounting = load_json(accounting_path, f"wave array accounting {scope}")
            expect(accounting.get("array_job_id") == jobs["array"]["job_id"], f"wave array accounting identity drifted: {scope}")
            if accounting.get("raw") is not None:
                validate_binding(accounting["raw"], f"wave array accounting raw {scope}")
            accounting_binding = file_binding(accounting_path)
        merge_scheduler = _best_effort_scontrol(jobs["merge_supervisor"]["job_id"])
        seal_scheduler = _best_effort_scontrol(jobs["seal_supervisor"]["job_id"])
        merge_path = regular_readonly(paths["merge_supervisor_receipt"], f"wave merge supervisor {scope}")
        seal_supervisor_path = regular_readonly(paths["seal_supervisor_receipt"], f"wave seal supervisor {scope}")
        consumption_path = regular_readonly(paths["consumption_receipt"], f"wave consumption {scope}")
        target_seal_path = regular_readonly(paths["seal_receipt"], f"wave target seal {scope}")
        target_seal = load_json(target_seal_path, f"wave target seal {scope}")
        terminal_failure_binding = None
        if Path(paths["terminal_failure_receipt"]).exists():
            terminal_failure_path = regular_readonly(paths["terminal_failure_receipt"], f"wave terminal failure {scope}")
            terminal_failure = load_json(terminal_failure_path, f"wave terminal failure {scope}")
            expect(terminal_failure.get("scope") == scope and terminal_failure.get("merged_evaluation_eligible") is False and terminal_failure.get("retry_authorized") is False, f"wave terminal failure payload drifted: {scope}")
            terminal_failure_binding = file_binding(terminal_failure_path)
            status = "terminal_evaluation_failure"
        else:
            status = target_seal.get("status")
        expect(status in {"valid_merged_evaluation", "terminal_verifier_cap_failure", "terminal_evaluation_failure"}, f"wave target status invalid: {scope}")
        if status == "valid_merged_evaluation":
            valid_count += 1
        elif status == "terminal_verifier_cap_failure":
            verifier_cap_count += 1
        else:
            evaluation_failure_count += 1
        log_root = Path(paths["log_root"]).resolve()
        expected_public_logs = [
            *(log_root / f"shard-{jobs['array']['job_id']}_{index}.public.out" for index in range(EVALUATION_SHARDS)),
            log_root / f"merge-supervisor-{jobs['merge_supervisor']['job_id']}.public.out",
            log_root / f"seal-supervisor-{jobs['seal_supervisor']['job_id']}.public.out",
        ]
        public_logs: dict[str, dict[str, str]] = {}
        missing_public_logs: list[str] = []
        for log_path in expected_public_logs:
            if log_path.is_file() and not log_path.is_symlink():
                expect(log_path.read_text(encoding="utf-8") == "evaluation phase terminal\n", f"evaluation public log content drifted: {log_path}")
                os.chmod(log_path, 0o444)
                public_logs[log_path.name] = file_binding(log_path)
            else:
                missing_public_logs.append(log_path.name)
        targets[scope] = {
            "status": status,
            "training_terminal": None,
            "submission": file_binding(submission_path),
            "array_accounting": accounting_binding,
            "merge_supervisor": file_binding(merge_path),
            "seal_supervisor": file_binding(seal_supervisor_path),
            "terminal_failure": terminal_failure_binding,
            "consumption": file_binding(consumption_path),
            "target_seal": file_binding(target_seal_path),
            "public_logs": public_logs,
            "missing_public_logs": missing_public_logs,
            "supervisor_scheduler": {
                "merge": merge_scheduler,
                "seal": seal_scheduler,
            },
        }
    required_primary_objectives = {
        "task_rl", "task_rl_k1_ungated_clip5", "task_rl_k1_gated_clip5_beta5"
    }
    required_primary_scopes = {
        key for key, arm in prereg["payload"]["arms"].items()
        if arm["objective_id"] in required_primary_objectives
    }
    primary_complete = all(targets[scope]["status"] == "valid_merged_evaluation" for scope in required_primary_scopes)
    evaluation_control_root = (Path(plan["control_root"]) / "evaluation").resolve()
    seal_tree_readonly(evaluation_control_root)
    private_log_files = [
        item for item in evaluation_control_root.rglob("*")
        if item.is_file() and "private_logs" in item.parts
    ]
    expect(all(stat.S_IMODE(item.stat().st_mode) & (stat.S_IRWXG | stat.S_IRWXO) == 0 for item in private_log_files), "sealed evaluation private log became group/world accessible")
    evaluation_control_tree_sha256 = sha256_tree(evaluation_control_root)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "seal": "opd_math_objective_family_evaluation_wave_seal_v2",
        "status": "all_38_targets_terminal_before_scientific_gate_release",
        "created_utc": utc_now(),
        "release_plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
        "program_manifest": plan["program_manifest"],
        "terminal_snapshot": {"path": str(terminal_path), "sha256": sha256_file(terminal_path)},
        "wave_authorization": {"path": str(wave_path), "sha256": sha256_file(wave_path)},
        "submission_index": {"path": str(index_path), "sha256": sha256_file(index_path)},
        "release_intent": {"path": str(intent_path), "sha256": sha256_file(intent_path)},
        "release_result": file_binding(plan["evaluation_wave_release_result"]),
        "evaluation_control_tree_sha256": evaluation_control_tree_sha256,
        "private_logs_owner_only": True,
        "target_order": wave["target_order"],
        "targets": targets,
        "all_targets_terminal": True,
        "all_successful_evaluations_consumed": True,
        "terminal_training_failure_count": training_failure_count,
        "terminal_evaluation_failure_count": evaluation_failure_count,
        "terminal_verifier_cap_failure_count": verifier_cap_count,
        "valid_evaluation_count": valid_count,
        "terminal_bundle_release_authorized": True,
        "primary_complete_case_analysis_authorized": primary_complete,
        "machine_validation_read_some_outcome_artifacts": True,
        "human_outcomes_inspected": False,
        "scientific_result_claimed": False,
    }
    write_new_or_exact(output, payload)
    return payload


def legacy_validate_evaluation_wave_seal(
    *, repo: Path, plan_path: Path, plan: Mapping[str, Any], prereg: Mapping[str, Any]
) -> tuple[Path, dict[str, Any]]:
    path = regular_readonly(plan["evaluation_wave_seal"], "evaluation-wave seal")
    payload = load_json(path, "evaluation-wave seal")
    expected_keys = {
        "schema_version", "seal", "status", "created_utc", "release_plan",
        "terminal_snapshot", "target_order", "targets", "all_training_arms_terminal",
        "all_evaluation_targets_terminal", "all_valid_evaluations_consumed",
        "campaign_wide_consumer_authorized", "heldout_outcomes_inspected",
        "scientific_result_claimed",
    }
    expect(set(payload) == expected_keys, "evaluation-wave seal schema drifted")
    expect(payload.get("schema_version") == SCHEMA_VERSION and payload.get("seal") == "opd_math_objective_family_evaluation_wave_seal_v1", "evaluation-wave seal identity drifted")
    expect(payload.get("status") == "all_38_targets_terminal_before_scientific_gate_release", "evaluation-wave seal status drifted")
    expect(payload.get("release_plan") == {"path": str(plan_path), "sha256": sha256_file(plan_path)}, "evaluation-wave plan binding drifted")
    expect(payload.get("terminal_snapshot") == file_binding(plan["terminal_snapshot"]), "evaluation-wave terminal binding drifted")
    expected_order = ["raw_student__M", "raw_student__O", *prereg["payload"]["arm_keys"]]
    expect(payload.get("target_order") == expected_order and set(payload.get("targets", {})) == set(expected_order), "evaluation-wave target matrix drifted")
    expect(payload.get("all_training_arms_terminal") is True and payload.get("all_evaluation_targets_terminal") is True and payload.get("all_valid_evaluations_consumed") is True and payload.get("campaign_wide_consumer_authorized") is True, "evaluation-wave authorization flags drifted")
    expect(payload.get("heldout_outcomes_inspected") is False and payload.get("scientific_result_claimed") is False, "evaluation-wave claim boundary drifted")
    parse_utc(payload.get("created_utc"), "evaluation-wave seal created_utc")
    for scope, target in payload["targets"].items():
        expect(isinstance(target, dict) and set(target) == {
            "status", "training_terminal", "submission", "consumption",
            "target_seal", "scheduler", "seal_log",
        }, f"evaluation-wave target schema drifted: {scope}")
        if target["target_seal"] is not None:
            validate_binding(target["target_seal"], f"evaluation-wave target seal {scope}")
            validate_binding(target["consumption"], f"evaluation-wave target consumption {scope}")
            validate_binding(target["submission"], f"evaluation-wave target submission {scope}")
            validate_binding(target["seal_log"], f"evaluation-wave target log {scope}")
    return path, payload


def validate_evaluation_wave_seal(
    *, repo: Path, plan_path: Path, plan: Mapping[str, Any], prereg: Mapping[str, Any]
) -> tuple[Path, dict[str, Any]]:
    path = regular_readonly(plan["evaluation_wave_seal"], "evaluation-wave seal")
    payload = load_json(path, "evaluation-wave seal")
    expected_keys = {
        "schema_version", "seal", "status", "created_utc", "release_plan", "program_manifest",
        "terminal_snapshot", "wave_authorization", "submission_index", "release_intent", "release_result",
        "evaluation_control_tree_sha256", "private_logs_owner_only",
        "target_order", "targets", "all_targets_terminal", "all_successful_evaluations_consumed",
        "terminal_training_failure_count", "terminal_evaluation_failure_count",
        "terminal_verifier_cap_failure_count", "valid_evaluation_count",
        "terminal_bundle_release_authorized", "primary_complete_case_analysis_authorized",
        "machine_validation_read_some_outcome_artifacts", "human_outcomes_inspected",
        "scientific_result_claimed",
    }
    expect(set(payload) == expected_keys, "evaluation-wave seal schema drifted")
    expect(payload.get("schema_version") == SCHEMA_VERSION and payload.get("seal") == "opd_math_objective_family_evaluation_wave_seal_v2", "evaluation-wave seal identity drifted")
    expect(payload.get("status") == "all_38_targets_terminal_before_scientific_gate_release", "evaluation-wave seal status drifted")
    expect(payload.get("release_plan") == {"path": str(plan_path), "sha256": sha256_file(plan_path)} and payload.get("program_manifest") == plan["program_manifest"], "evaluation-wave seal plan drifted")
    expect(payload.get("terminal_snapshot") == file_binding(plan["terminal_snapshot"]), "evaluation-wave terminal binding drifted")
    expect(payload.get("wave_authorization") == file_binding(plan["evaluation_wave_authorization"]), "evaluation-wave authorization binding drifted")
    expect(payload.get("submission_index") == file_binding(plan["evaluation_wave_submission_index"]), "evaluation-wave submission binding drifted")
    expect(payload.get("release_intent") == file_binding(plan["evaluation_wave_release_intent"]), "evaluation-wave release-intent binding drifted")
    expect(payload.get("release_result") == file_binding(plan["evaluation_wave_release_result"]), "evaluation-wave release-result binding drifted")
    evaluation_control_root = (Path(plan["control_root"]) / "evaluation").resolve()
    expect(payload.get("evaluation_control_tree_sha256") == sha256_tree(evaluation_control_root), "evaluation-wave control-tree hash drifted")
    private_log_files = [item for item in evaluation_control_root.rglob("*") if item.is_file() and "private_logs" in item.parts]
    expect(payload.get("private_logs_owner_only") is True and all(stat.S_IMODE(item.stat().st_mode) & (stat.S_IRWXG | stat.S_IRWXO) == 0 for item in private_log_files), "evaluation-wave private-log permissions drifted")
    expected_order = ["raw_student__M", "raw_student__O", *prereg["payload"]["arm_keys"]]
    expect(payload.get("target_order") == expected_order and set(payload.get("targets", {})) == set(expected_order), "evaluation-wave target matrix drifted")
    counts = {
        "terminal_training_failure_count": 0,
        "terminal_evaluation_failure_count": 0,
        "terminal_verifier_cap_failure_count": 0,
        "valid_evaluation_count": 0,
    }
    for scope, target in payload["targets"].items():
        status = target.get("status")
        if status == "terminal_training_failure_no_evaluation":
            counts["terminal_training_failure_count"] += 1
            expect(target.get("submission") is None and target.get("target_seal") is None, f"training-failure target retained evaluation custody: {scope}")
            continue
        expect(status in {"valid_merged_evaluation", "terminal_verifier_cap_failure", "terminal_evaluation_failure"}, f"evaluation-wave target status drifted: {scope}")
        for field in ("submission", "merge_supervisor", "seal_supervisor", "consumption", "target_seal"):
            validate_binding(target.get(field), f"evaluation-wave {scope} {field}")
        if status == "terminal_evaluation_failure":
            validate_binding(target.get("terminal_failure"), f"evaluation-wave {scope} terminal failure")
        else:
            expect(target.get("terminal_failure") is None, f"nonfailure target retained terminal failure: {scope}")
        if target.get("array_accounting") is not None:
            validate_binding(target.get("array_accounting"), f"evaluation-wave {scope} array accounting")
        for binding in (target.get("public_logs") or {}).values():
            validate_binding(binding, f"evaluation-wave {scope} public log")
        if status == "valid_merged_evaluation":
            counts["valid_evaluation_count"] += 1
        elif status == "terminal_verifier_cap_failure":
            counts["terminal_verifier_cap_failure_count"] += 1
        else:
            counts["terminal_evaluation_failure_count"] += 1
    for field, value in counts.items():
        expect(payload.get(field) == value, f"evaluation-wave count drifted: {field}")
    expect(sum(counts.values()) == len(expected_order), "evaluation-wave terminal count does not cover all targets")
    expect(payload.get("all_targets_terminal") is True and payload.get("all_successful_evaluations_consumed") is True and payload.get("terminal_bundle_release_authorized") is True, "evaluation-wave terminal authorization flags drifted")
    expect(payload.get("machine_validation_read_some_outcome_artifacts") is True and payload.get("human_outcomes_inspected") is False and payload.get("scientific_result_claimed") is False, "evaluation-wave inspection boundary drifted")
    expect(isinstance(payload.get("primary_complete_case_analysis_authorized"), bool), "evaluation-wave primary-analysis flag drifted")
    parse_utc(payload.get("created_utc"), "evaluation-wave seal created_utc")
    return path, payload


def _prepared_holdout_binding(prereg: Mapping[str, Any], plan: Mapping[str, Any], *, source: str, task_path: Path, records: int) -> dict[str, Any]:
    prepared_path = Path(prereg["prepared_manifest"]["path"]).resolve()
    prepared = load_json(prepared_path, "prepared manifest")
    relative = f"roles/{source}/source_holdout.jsonl"
    entry = (prepared.get("files") or {}).get(relative)
    expect(isinstance(entry, dict), f"prepared manifest lacks {relative}")
    expected_task = (prepared_path.parent / relative).resolve()
    expect(task_path == expected_task, f"heldout task path drifted: {source}")
    expect(entry.get("sha256") == sha256_file(expected_task), f"heldout task hash drifted: {source}")
    expect(records == SELECTED_HOLDOUT_RECORDS, f"heldout must evaluate exact 370-row prefix: {source}")
    rows = load_jsonl(expected_task, f"heldout task {source}")
    expect(len(rows) == entry.get("rows") and len(rows) >= records, f"heldout physical row count drifted: {source}")
    record_ids = [row.get("record_id") for row in rows[:records]]
    expect(all(isinstance(value, str) and value for value in record_ids), f"heldout record IDs invalid: {source}")
    expect(len(set(record_ids)) == len(record_ids), f"heldout record IDs duplicate: {source}")
    result = {
        "prepared_manifest": str(prepared_path),
        "prepared_manifest_sha256": sha256_file(prepared_path),
        "task_file": str(expected_task),
        "task_file_sha256": entry["sha256"],
        "records": records,
        "record_ids_sha256": canonical_json_sha256(record_ids),
    }
    expected = plan["holdout_selection"][source]
    expect(result["task_file"] == expected["task_file"], f"heldout selected task path drifted: {source}")
    expect(result["task_file_sha256"] == expected["task_file_sha256"], f"heldout selected task hash drifted: {source}")
    expect(result["records"] == expected["selected_records"], f"heldout selected count drifted: {source}")
    expect(result["record_ids_sha256"] == expected["selected_record_ids_sha256"], f"heldout selected IDs drifted: {source}")
    return result


def checked_holdout(
    *,
    repo: Path,
    prereg: Mapping[str, Any],
    plan: Mapping[str, Any],
    source: str,
    summary_path: str | Path,
    samples_path: str | Path,
    expected_adapter: Mapping[str, Any] | None,
) -> dict[str, Any]:
    from scripts.opd_math.quality_gates import (  # type: ignore
        EVALUATION_CONTRACT,
        EVALUATION_MERGED_KIND,
        MAX_EVALUATION_VERIFIER_ERROR_FRACTION,
        checked_evaluation,
    )

    summary_file = regular_readonly(summary_path, "heldout summary")
    samples_file = regular_readonly(samples_path, "heldout samples")
    summary, grouped, binding = checked_evaluation(
        summary_file,
        samples_file,
        expected_model=EXPECTED_STUDENT,
        expected_revision=EXPECTED_STUDENT_REVISION,
        expected_source=source,
        expected_role="source_holdout",
    )
    expect(binding.get("evaluation_artifact_kind") == EVALUATION_MERGED_KIND, "heldout is not a merged evaluation")
    expect(binding.get("evaluation_contract") == EVALUATION_CONTRACT, "heldout lacks exact-environment contract")
    expect(binding.get("evaluation_git_commit") == EXPECTED_COMMIT, "heldout evaluation commit drifted")
    expect(binding.get("samples_per_problem") == 4, "heldout must have four samples per record")
    expect(summary.get("decoding") == HELDOUT_DECODING, "heldout decoding drifted")
    expect(binding.get("verifier_error_fraction") <= MAX_EVALUATION_VERIFIER_ERROR_FRACTION, "heldout verifier cap failed")
    prepared = _prepared_holdout_binding(
        prereg,
        plan,
        source=source,
        task_path=Path(binding["task_file"]).resolve(),
        records=int(binding["records"]),
    )
    adapter_value = summary.get("adapter")
    adapter_hash = summary.get("adapter_tree_sha256")
    if expected_adapter is None:
        expect(adapter_value is None and adapter_hash is None, "raw-student heldout unexpectedly used an adapter")
    else:
        adapter_path = Path(str(adapter_value)).resolve()
        expect(adapter_path == Path(expected_adapter["path"]).resolve(), "heldout evaluated wrong adapter")
        expect(adapter_hash == expected_adapter["tree_sha256"] == sha256_tree(adapter_path), "heldout adapter hash drifted")
    record_rewards = {record_id: list(grouped[record_id]) for record_id in sorted(grouped)}
    errors_by_record: dict[str, int] = defaultdict(int)
    for item in binding["verifier_error_sample_keys"]:
        errors_by_record[str(item["record_id"])] += 1
    record_accuracy = {
        record_id: sum(values) / len(values) for record_id, values in record_rewards.items()
    }
    record_bounds = {
        record_id: [
            record_accuracy[record_id],
            (sum(record_rewards[record_id]) + errors_by_record.get(record_id, 0))
            / len(record_rewards[record_id]),
        ]
        for record_id in record_rewards
    }
    accuracy = sum(record_accuracy.values()) / len(record_accuracy)
    accuracy_bounds = [
        sum(value[0] for value in record_bounds.values()) / len(record_bounds),
        sum(value[1] for value in record_bounds.values()) / len(record_bounds),
    ]
    expect(math.isclose(float(summary["accuracy"]), accuracy, rel_tol=0.0, abs_tol=1e-12), "heldout accuracy recomputation drifted")
    return {
        "summary": file_binding(summary_file),
        "samples": file_binding(samples_file),
        "evaluation_binding": binding,
        "prepared_binding": prepared,
        "records": len(record_rewards),
        "samples_per_problem": 4,
        "accuracy": accuracy,
        "accuracy_bounds_under_verifier_uncertainty": accuracy_bounds,
        "record_rewards": record_rewards,
        "record_accuracy": record_accuracy,
        "record_accuracy_bounds_under_verifier_uncertainty": record_bounds,
        "record_ids_sha256": canonical_json_sha256(sorted(record_rewards)),
        "verifier_error_samples": binding["verifier_error_samples"],
        "verifier_error_fraction": binding["verifier_error_fraction"],
    }


def verifier_cap_failure_binding(summary_path: str | Path, samples_path: str | Path, error: RuntimeError) -> dict[str, Any]:
    message = str(error)
    expect(message.startswith("evaluation verifier-error fraction exceeds the registered cap"), "unexpected heldout runtime failure")
    summary_file = regular_readonly(summary_path, "verifier-cap summary")
    samples_file = regular_readonly(samples_path, "verifier-cap samples")
    rows = load_jsonl(samples_file, "verifier-cap samples")
    errors = sum(row.get("reward_status") == "verifier_error_zeroed" for row in rows)
    fraction = errors / len(rows)
    expect(fraction > 0.001, "verifier-cap failure does not exceed frozen cap")
    return {
        "reason": "verifier_error_cap_exceeded",
        "maximum_verifier_error_fraction": 0.001,
        "observed_verifier_error_samples": errors,
        "observed_verifier_error_fraction": fraction,
        "summary": file_binding(summary_file),
        "samples": file_binding(samples_file),
        "error_message_sha256": hashlib.sha256(message.encode("utf-8")).hexdigest(),
        "no_effect_estimate_released": True,
    }


def arm_gate_payload(args: argparse.Namespace) -> dict[str, Any]:
    repo = configure_repo(args.repo)
    plan_path, plan, prereg = validate_release_plan(args.release_plan, repo)
    terminal_path, terminal = validate_terminal_snapshot(
        plan["terminal_snapshot"], plan_path, plan, repo=repo, prereg=prereg
    )
    wave_path, wave = validate_evaluation_wave_seal(
        repo=repo, plan_path=plan_path, plan=plan, prereg=prereg
    )
    key = args.arm_key
    expect(key in plan["arm_paths"], "arm gate key is not preregistered")
    paths = plan["arm_paths"][key]
    expect(Path(args.output).resolve() == Path(paths["heldout_gate"]).resolve(), "arm gate output path drifted")
    arm = prereg["payload"]["arms"][key]
    terminal_arm = terminal["arms"][key]
    common = {
        "schema_version": SCHEMA_VERSION,
        "gate": ARM_GATE_ID,
        "arm_key": key,
        "objective_id": arm["objective_id"],
        "implementation": arm["implementation"],
        "source": arm["source"],
        "seed": arm["seed"],
        "run_id": arm["run_id"],
        "release_plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
        "terminal_snapshot": {"path": str(terminal_path), "sha256": sha256_file(terminal_path)},
        "evaluation_wave_seal": {"path": str(wave_path), "sha256": sha256_file(wave_path)},
        "authorization_is_independent_of_effect_sign": True,
    }
    if not terminal_arm["training_eligible_for_heldout"]:
        expect(not Path(paths["evaluation_summary"]).exists() and not Path(paths["evaluation_samples"]).exists(), "terminal failed training arm has heldout outputs")
        payload = {
            **common,
            "status": "terminal_training_failure_no_heldout",
            "eligible_for_primary_or_secondary_inference": False,
            "training_terminal": terminal_arm,
            "evaluation_authorization": None,
            "training_custody": None,
            "heldout": None,
            "claim_boundary": "Terminal training failure retained without replacement, imputation, or held-out evaluation.",
        }
    elif wave["targets"][key]["status"] == "terminal_evaluation_failure":
        authorization = validate_evaluation_authorization(
            repo=repo,
            release_plan=plan_path,
            arm_key_value=key,
            raw_source=None,
        )
        payload = {
            **common,
            "status": "terminal_evaluation_failure_no_heldout",
            "eligible_for_primary_or_secondary_inference": False,
            "training_terminal": terminal_arm,
            "evaluation_authorization": authorization,
            "training_custody": terminal_arm["training_validation"],
            "heldout": None,
            "technical_failure": wave["targets"][key],
            "claim_boundary": "Terminal evaluation scheduler failure retained without replacement, imputation, or resampling.",
        }
    else:
        authorization = validate_evaluation_authorization(
            repo=repo,
            release_plan=plan_path,
            arm_key_value=key,
            raw_source=None,
        )
        auth_payload = load_json(authorization["path"], f"authorization {key}")
        try:
            heldout = checked_holdout(
                repo=repo,
                prereg=prereg,
                plan=plan,
                source=arm["source"],
                summary_path=paths["evaluation_summary"],
                samples_path=paths["evaluation_samples"],
                expected_adapter=auth_payload["training_custody"]["final_adapter"],
            )
        except RuntimeError as error:
            technical_failure = verifier_cap_failure_binding(
                paths["evaluation_summary"], paths["evaluation_samples"], error
            )
            payload = {
                **common,
                "status": "terminal_heldout_verifier_cap_failure",
                "eligible_for_primary_or_secondary_inference": False,
                "training_terminal": terminal_arm,
                "evaluation_authorization": authorization,
                "training_custody": auth_payload["training_custody"],
                "heldout": None,
                "technical_failure": technical_failure,
                "claim_boundary": "Verifier-cap failure is terminal and cannot be rescued by uncertainty envelopes or resampling.",
            }
        else:
            payload = {
                **common,
                "status": "eligible_heldout_gate",
                "eligible_for_primary_or_secondary_inference": True,
                "training_terminal": terminal_arm,
                "evaluation_authorization": authorization,
                "training_custody": auth_payload["training_custody"],
                "heldout": heldout,
                "claim_boundary": "Eligible campaign cell; no effect claim is valid until the 36-arm terminal bundle is released.",
            }
    return payload


def seal_arm_gate(args: argparse.Namespace) -> dict[str, Any]:
    payload = arm_gate_payload(args)
    write_new(args.output, payload)
    return payload


def raw_student_gate_payload(args: argparse.Namespace) -> dict[str, Any]:
    repo = configure_repo(args.repo)
    plan_path, plan, prereg = validate_release_plan(args.release_plan, repo)
    terminal_path, _ = validate_terminal_snapshot(
        plan["terminal_snapshot"], plan_path, plan, repo=repo, prereg=prereg
    )
    wave_path, wave = validate_evaluation_wave_seal(
        repo=repo, plan_path=plan_path, plan=plan, prereg=prereg
    )
    source = args.raw_source
    expect(source in SOURCES, "raw student gate source is invalid")
    paths = plan["raw_student_auxiliary"][source]
    expect(Path(args.output).resolve() == Path(paths["gate"]).resolve(), "raw student gate output path drifted")
    authorization = validate_evaluation_authorization(
        repo=repo,
        release_plan=plan_path,
        arm_key_value=None,
        raw_source=source,
    )
    common = {
        "schema_version": SCHEMA_VERSION,
        "gate": "opd_math_objective_family_raw_student_heldout_gate_v1",
        "source": source,
        "release_plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
        "terminal_snapshot": {"path": str(terminal_path), "sha256": sha256_file(terminal_path)},
        "evaluation_wave_seal": {"path": str(wave_path), "sha256": sha256_file(wave_path)},
        "evaluation_authorization": authorization,
        "secondary_only": True,
        "raw_baseline_has_no_training_seed_variance": True,
    }
    wave_target = wave["targets"][f"raw_student__{source}"]
    if wave_target["status"] == "terminal_evaluation_failure":
        return {
            **common,
            "status": "terminal_auxiliary_evaluation_failure",
            "heldout": None,
            "technical_failure": wave_target,
            "claim_boundary": "Auxiliary raw baseline evaluation failed terminally and is unavailable for secondary comparisons.",
        }
    try:
        heldout = checked_holdout(
            repo=repo,
            prereg=prereg,
            plan=plan,
            source=source,
            summary_path=paths["summary"],
            samples_path=paths["samples"],
            expected_adapter=None,
        )
    except RuntimeError as error:
        payload = {
            **common,
            "status": "terminal_auxiliary_verifier_cap_failure",
            "heldout": None,
            "technical_failure": verifier_cap_failure_binding(paths["summary"], paths["samples"], error),
            "claim_boundary": "Auxiliary raw baseline failed its verifier cap and is unavailable for secondary comparisons.",
        }
    else:
        payload = {
            **common,
            "status": "eligible_auxiliary_baseline",
            "heldout": heldout,
            "claim_boundary": "Auxiliary raw-student source-holdout baseline; not the prelaunch support gate.",
        }
    return payload


def seal_raw_student_gate(args: argparse.Namespace) -> dict[str, Any]:
    payload = raw_student_gate_payload(args)
    write_new(args.output, payload)
    return payload


def _load_and_recompute_gates(
    *, repo: Path, plan_path: Path, plan: Mapping[str, Any], prereg: Mapping[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    arm_gates: dict[str, dict[str, Any]] = {}
    for key in prereg["payload"]["arm_keys"]:
        gate_path = regular_readonly(plan["arm_paths"][key]["heldout_gate"], f"arm gate {key}")
        observed = load_json(gate_path, f"arm gate {key}")
        expected = arm_gate_payload(
            SimpleNamespace(repo=repo, release_plan=plan_path, arm_key=key, output=gate_path)
        )
        expect(observed == expected, f"arm gate differs from deterministic recomputation: {key}")
        arm_gates[key] = observed
    raw_gates: dict[str, dict[str, Any]] = {}
    for source in SOURCES:
        gate_path = regular_readonly(plan["raw_student_auxiliary"][source]["gate"], f"raw gate {source}")
        observed = load_json(gate_path, f"raw gate {source}")
        expected = raw_student_gate_payload(
            SimpleNamespace(repo=repo, release_plan=plan_path, raw_source=source, output=gate_path)
        )
        expect(observed == expected, f"raw gate differs from deterministic recomputation: {source}")
        raw_gates[source] = observed
    return arm_gates, raw_gates


def _gate_vectors(gate: Mapping[str, Any]) -> tuple[list[str], list[float], list[float], list[float]]:
    heldout = gate.get("heldout")
    expect(isinstance(heldout, dict), "eligible gate lacks heldout payload")
    point = heldout.get("record_accuracy")
    bounds = heldout.get("record_accuracy_bounds_under_verifier_uncertainty")
    expect(isinstance(point, dict) and isinstance(bounds, dict) and set(point) == set(bounds), "gate record vectors drifted")
    record_ids = sorted(point)
    expect(record_ids, "gate record vector is empty")
    points = [float(point[key]) for key in record_ids]
    lowers = [float(bounds[key][0]) for key in record_ids]
    uppers = [float(bounds[key][1]) for key in record_ids]
    expect(all(0.0 <= lower <= value <= upper <= 1.0 for value, lower, upper in zip(points, lowers, uppers)), "gate verifier bounds invalid")
    return record_ids, points, lowers, uppers


def _contrast_catalog(objective_ids: Sequence[str]) -> dict[str, tuple[str, str, str]]:
    catalog: dict[str, tuple[str, str, str]] = {}
    for name, source, treatment, baseline in PRIMARY_CONTRASTS:
        catalog[name] = (source, treatment, baseline)
    for source in SOURCES:
        for name, treatment, baseline in (
            ("clip5_minus_unclipped", "task_rl_k1_ungated_clip5", "task_rl_k1_ungated_unclipped"),
            ("gated_minus_task_rl", "task_rl_k1_gated_clip5_beta5", "task_rl"),
            ("unclipped_minus_task_rl", "task_rl_k1_ungated_unclipped", "task_rl"),
            ("local_bare_minus_task_rl", "k1_bare_verl_compatible_clip10", "task_rl"),
            ("upstream_bare_minus_task_rl", "k1_verl_upstream_clip10", "task_rl"),
            ("local_bare_minus_upstream_bare", "k1_bare_verl_compatible_clip10", "k1_verl_upstream_clip10"),
        ):
            catalog[f"{name}@{source}"] = (source, treatment, baseline)
    expect(set(objective_ids) == {
        "task_rl",
        "task_rl_k1_ungated_clip5",
        "task_rl_k1_ungated_unclipped",
        "task_rl_k1_gated_clip5_beta5",
        "k1_bare_verl_compatible_clip10",
        "k1_verl_upstream_clip10",
    }, "contrast catalog objective set drifted")
    return catalog


def allocated_gpu_count(value: Any) -> int | None:
    """Parse Slurm AllocTRES GPU allocation without assuming launcher geometry."""
    if not isinstance(value, str) or not value.strip():
        return None
    generic_gpu: list[int] = []
    generic_gres: list[int] = []
    typed_gres: list[int] = []
    for token in value.split(","):
        name, separator, raw_count = token.partition("=")
        normalized = name.strip().lower()
        if not (normalized == "gpu" or normalized.startswith("gres/gpu")):
            continue
        expect(separator and raw_count.isdigit(), f"ambiguous GPU AllocTRES token: {token}")
        count = int(raw_count)
        expect(count >= 0, f"negative GPU AllocTRES token: {token}")
        if normalized == "gpu":
            generic_gpu.append(count)
        elif normalized == "gres/gpu":
            generic_gres.append(count)
        else:
            typed_gres.append(count)
    if not (generic_gpu or generic_gres or typed_gres):
        return None
    for counts in (generic_gpu, generic_gres):
        if counts:
            expect(len(set(counts)) == 1, "conflicting aggregate GPU AllocTRES counts")
            aggregate = counts[0]
            if typed_gres:
                expect(sum(typed_gres) == aggregate, "typed and aggregate GPU AllocTRES disagree")
            return aggregate
    return sum(typed_gres)


def build_readout(
    *,
    repo: Path,
    plan_path: Path,
    plan: Mapping[str, Any],
    prereg: Mapping[str, Any],
    arm_gates: Mapping[str, Mapping[str, Any]],
    raw_gates: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    objective_ids = list(prereg["payload"]["objective_ids"])
    terminal_failed = [key for key, gate in arm_gates.items() if gate.get("status") != "eligible_heldout_gate"]
    eligible = {key: gate for key, gate in arm_gates.items() if gate.get("status") == "eligible_heldout_gate"}
    vectors: dict[tuple[str, str, int], dict[str, Any]] = {}
    source_records: dict[str, list[str]] = {}
    source_task_hash: dict[str, str] = {}
    source_record_seed_contract: dict[str, Any] = {}
    for key, gate in eligible.items():
        record_ids, point, lower, upper = _gate_vectors(gate)
        source = str(gate["source"])
        source_records.setdefault(source, record_ids)
        expect(source_records[source] == record_ids, f"heldout record pairing drifted within source {source}")
        heldout = gate["heldout"]
        task_hash = heldout["prepared_binding"]["task_file_sha256"]
        source_task_hash.setdefault(source, task_hash)
        expect(source_task_hash[source] == task_hash, f"heldout task hash drifted within source {source}")
        seed_contract = heldout["evaluation_binding"]["record_seed_contract"]
        source_record_seed_contract.setdefault(source, seed_contract)
        expect(source_record_seed_contract[source] == seed_contract, f"record RNG contract drifted within source {source}")
        vectors[(gate["objective_id"], source, int(gate["seed"]))] = {
            "point": point,
            "lower": lower,
            "upper": upper,
        }
    raw_vectors: dict[str, dict[str, Any]] = {}
    for source, gate in raw_gates.items():
        if gate.get("status") != "eligible_auxiliary_baseline":
            continue
        record_ids, point, lower, upper = _gate_vectors(gate)
        if source in source_records:
            expect(source_records[source] == record_ids, f"raw/student heldout record pairing drifted: {source}")
            expect(gate["heldout"]["prepared_binding"]["task_file_sha256"] == source_task_hash[source], f"raw/student task hash drifted: {source}")
            expect(gate["heldout"]["evaluation_binding"]["record_seed_contract"] == source_record_seed_contract[source], f"raw/student record RNG contract drifted: {source}")
        raw_vectors[source] = {"point": point, "lower": lower, "upper": upper}

    primary_objectives = {
        "task_rl",
        "task_rl_k1_ungated_clip5",
        "task_rl_k1_gated_clip5_beta5",
    }
    primary_cells = {
        (objective, source, seed)
        for objective in primary_objectives
        for source in SOURCES
        for seed in SEEDS
    }
    missing_primary = sorted(arm_key(*cell) for cell in primary_cells if cell not in vectors)
    primary_estimable = not missing_primary

    catalog = _contrast_catalog(objective_ids)
    estimable_catalog = {
        name: coordinates
        for name, coordinates in catalog.items()
        if all((objective, coordinates[0], seed) in vectors for objective in coordinates[1:] for seed in SEEDS)
    }
    observed: dict[str, dict[str, Any]] = {}
    for name, (source, treatment, baseline) in estimable_catalog.items():
        seed_effects = []
        seed_lower = []
        seed_upper = []
        for seed in SEEDS:
            treatment_vector = vectors[(treatment, source, seed)]
            baseline_vector = vectors[(baseline, source, seed)]
            seed_effects.append(sum(a - b for a, b in zip(treatment_vector["point"], baseline_vector["point"])) / len(source_records[source]))
            seed_lower.append(sum(a - b for a, b in zip(treatment_vector["lower"], baseline_vector["upper"])) / len(source_records[source]))
            seed_upper.append(sum(a - b for a, b in zip(treatment_vector["upper"], baseline_vector["lower"])) / len(source_records[source]))
        observed[name] = {
            "source": source,
            "treatment": treatment,
            "baseline": baseline,
            "seed_effects": {str(seed): seed_effects[seed] for seed in SEEDS},
            "seed_effect_min": min(seed_effects),
            "seed_effect_max": max(seed_effects),
            "estimate": sum(seed_effects) / len(seed_effects),
            "estimate_bounds": [sum(seed_lower) / len(seed_lower), sum(seed_upper) / len(seed_upper)],
        }

    raw_comparisons: dict[str, dict[str, Any]] = {}
    for source in SOURCES:
        if source not in source_records or source not in raw_vectors:
            continue
        raw = raw_vectors[source]
        for objective in objective_ids:
            if not all((objective, source, seed) in vectors for seed in SEEDS):
                continue
            name = f"{objective}_minus_raw@{source}"
            seed_effects = []
            seed_lower = []
            seed_upper = []
            for seed in SEEDS:
                trained = vectors[(objective, source, seed)]
                seed_effects.append(sum(a - b for a, b in zip(trained["point"], raw["point"])) / len(source_records[source]))
                seed_lower.append(sum(a - b for a, b in zip(trained["lower"], raw["upper"])) / len(source_records[source]))
                seed_upper.append(sum(a - b for a, b in zip(trained["upper"], raw["lower"])) / len(source_records[source]))
            raw_comparisons[name] = {
                "source": source,
                "treatment": objective,
                "baseline": "raw_student",
                "seed_effects": {str(seed): seed_effects[seed] for seed in SEEDS},
                "seed_effect_min": min(seed_effects),
                "seed_effect_max": max(seed_effects),
                "estimate": sum(seed_effects) / len(seed_effects),
                "estimate_bounds": [sum(seed_lower) / len(seed_lower), sum(seed_upper) / len(seed_upper)],
            }

    draw_names = list(observed) + list(raw_comparisons)
    draws = {
        name: {"point": [], "lower": [], "upper": []}
        for name in draw_names
    }
    rng = random.Random(BOOTSTRAP_SEED)
    for _ in range(BOOTSTRAP_DRAWS):
        sampled_seeds = [rng.randrange(len(SEEDS)) for _ in SEEDS]
        indices = {
            source: [rng.randrange(len(source_records[source])) for _ in source_records[source]]
            for source in SOURCES
            if source in source_records
        }
        for name, meta in observed.items():
            source = meta["source"]
            treatment = meta["treatment"]
            baseline = meta["baseline"]
            point_values = []
            lower_values = []
            upper_values = []
            for seed in sampled_seeds:
                treatment_vector = vectors[(treatment, source, seed)]
                baseline_vector = vectors[(baseline, source, seed)]
                idx = indices[source]
                point_values.append(vector_mean(treatment_vector["point"], idx) - vector_mean(baseline_vector["point"], idx))
                lower_values.append(vector_mean(treatment_vector["lower"], idx) - vector_mean(baseline_vector["upper"], idx))
                upper_values.append(vector_mean(treatment_vector["upper"], idx) - vector_mean(baseline_vector["lower"], idx))
            draws[name]["point"].append(sum(point_values) / len(point_values))
            draws[name]["lower"].append(sum(lower_values) / len(lower_values))
            draws[name]["upper"].append(sum(upper_values) / len(upper_values))
        for name, meta in raw_comparisons.items():
            source = meta["source"]
            treatment = meta["treatment"]
            raw = raw_vectors[source]
            point_values = []
            lower_values = []
            upper_values = []
            idx = indices[source]
            for seed in sampled_seeds:
                trained = vectors[(treatment, source, seed)]
                point_values.append(vector_mean(trained["point"], idx) - vector_mean(raw["point"], idx))
                lower_values.append(vector_mean(trained["lower"], idx) - vector_mean(raw["upper"], idx))
                upper_values.append(vector_mean(trained["upper"], idx) - vector_mean(raw["lower"], idx))
            draws[name]["point"].append(sum(point_values) / len(point_values))
            draws[name]["lower"].append(sum(lower_values) / len(lower_values))
            draws[name]["upper"].append(sum(upper_values) / len(upper_values))

    results: dict[str, dict[str, Any]] = {}
    for name, meta in {**observed, **raw_comparisons}.items():
        confirmatory = primary_estimable and name in {item[0] for item in PRIMARY_CONTRASTS}
        results[name] = contrast_result(
            name=name,
            formula=f"accuracy({meta['treatment']}) - accuracy({meta['baseline']})",
            estimate=meta["estimate"],
            estimate_bounds=meta["estimate_bounds"],
            point_draws=draws[name]["point"],
            lower_draws=draws[name]["lower"],
            upper_draws=draws[name]["upper"],
            confirmatory=confirmatory,
        )
        results[name].update({
            "source": meta["source"],
            "seed_effects": meta["seed_effects"],
            "seed_effect_min": meta["seed_effect_min"],
            "seed_effect_max": meta["seed_effect_max"],
        })
        if name.startswith("local_bare_minus_upstream_bare"):
            results[name]["systems_agreement_only"] = True
            results[name]["not_equivalence_or_superiority_test"] = True

    paired_source_effects = {
        "clip5_minus_task_rl": (
            "task_rl_k1_ungated_clip5-minus-task_rl@M",
            "task_rl_k1_ungated_clip5-minus-task_rl@O",
        ),
        "gated_minus_clip5": (
            "task_rl_k1_gated_clip5_beta5-minus-task_rl_k1_ungated_clip5@M",
            "task_rl_k1_gated_clip5_beta5-minus-task_rl_k1_ungated_clip5@O",
        ),
        "clip5_minus_unclipped": ("clip5_minus_unclipped@M", "clip5_minus_unclipped@O"),
        "gated_minus_task_rl": ("gated_minus_task_rl@M", "gated_minus_task_rl@O"),
        "unclipped_minus_task_rl": ("unclipped_minus_task_rl@M", "unclipped_minus_task_rl@O"),
        "local_bare_minus_task_rl": ("local_bare_minus_task_rl@M", "local_bare_minus_task_rl@O"),
        "upstream_bare_minus_task_rl": ("upstream_bare_minus_task_rl@M", "upstream_bare_minus_task_rl@O"),
        "local_bare_minus_upstream_bare": (
            "local_bare_minus_upstream_bare@M",
            "local_bare_minus_upstream_bare@O",
        ),
    }
    for objective in objective_ids:
        paired_source_effects[f"{objective}_minus_raw"] = (
            f"{objective}_minus_raw@M",
            f"{objective}_minus_raw@O",
        )
    cross_source_results: dict[str, dict[str, Any]] = {}
    all_meta = {**observed, **raw_comparisons}
    for base, (m_name, o_name) in paired_source_effects.items():
        if not primary_estimable and base in {"clip5_minus_task_rl", "gated_minus_clip5"}:
            continue
        if m_name not in all_meta or o_name not in all_meta:
            continue
        m_meta = all_meta[m_name]
        o_meta = all_meta[o_name]
        for kind in ("equal_weight_source_average", "O_minus_M_interaction"):
            name = f"{base}__{kind}"
            if kind == "equal_weight_source_average":
                estimate = (m_meta["estimate"] + o_meta["estimate"]) / 2
                estimate_bounds = [
                    (m_meta["estimate_bounds"][0] + o_meta["estimate_bounds"][0]) / 2,
                    (m_meta["estimate_bounds"][1] + o_meta["estimate_bounds"][1]) / 2,
                ]
                point_draws = [
                    (m_value + o_value) / 2
                    for m_value, o_value in zip(draws[m_name]["point"], draws[o_name]["point"])
                ]
                lower_draws = [
                    (m_value + o_value) / 2
                    for m_value, o_value in zip(draws[m_name]["lower"], draws[o_name]["lower"])
                ]
                upper_draws = [
                    (m_value + o_value) / 2
                    for m_value, o_value in zip(draws[m_name]["upper"], draws[o_name]["upper"])
                ]
                formula = f"0.5 * ({m_name} + {o_name})"
            else:
                estimate = o_meta["estimate"] - m_meta["estimate"]
                estimate_bounds = [
                    o_meta["estimate_bounds"][0] - m_meta["estimate_bounds"][1],
                    o_meta["estimate_bounds"][1] - m_meta["estimate_bounds"][0],
                ]
                point_draws = [
                    o_value - m_value
                    for m_value, o_value in zip(draws[m_name]["point"], draws[o_name]["point"])
                ]
                lower_draws = [
                    o_lower - m_upper
                    for m_upper, o_lower in zip(draws[m_name]["upper"], draws[o_name]["lower"])
                ]
                upper_draws = [
                    o_upper - m_lower
                    for m_lower, o_upper in zip(draws[m_name]["lower"], draws[o_name]["upper"])
                ]
                formula = f"{o_name} - {m_name}"
            cross_source_results[name] = contrast_result(
                name=name,
                formula=formula,
                estimate=estimate,
                estimate_bounds=estimate_bounds,
                point_draws=point_draws,
                lower_draws=lower_draws,
                upper_draws=upper_draws,
                confirmatory=False,
            )
            cross_source_results[name]["exploratory"] = True
            if base == "local_bare_minus_upstream_bare":
                cross_source_results[name]["systems_agreement_only"] = True
                cross_source_results[name]["not_equivalence_or_superiority_test"] = True

    arm_accuracy = {
        key: gate["heldout"]["accuracy"]
        for key, gate in eligible.items()
    }
    arm_accuracy_bounds = {
        key: gate["heldout"]["accuracy_bounds_under_verifier_uncertainty"]
        for key, gate in eligible.items()
    }
    cost: dict[str, Any] = {}
    for key, gate in arm_gates.items():
        terminal = gate["training_terminal"]["scheduler"]
        implementation = gate["implementation"]
        gpu_count = allocated_gpu_count(terminal.get("alloc_tres"))
        custody_cost = (gate.get("training_custody") or {}).get("cost")
        cost[key] = {
            "implementation": implementation,
            "scheduler_elapsed_seconds": terminal["elapsed_seconds"],
            "scheduler_alloc_tres": terminal.get("alloc_tres"),
            "scheduler_node_list": terminal.get("node_list"),
            "allocated_gpu_count": gpu_count,
            "allocation_bound_gpu_hours": (
                terminal["elapsed_seconds"] * gpu_count / 3600.0
                if gpu_count is not None
                else None
            ),
            "training_trace_cost": custody_cost,
        }
    all_primary_names = {name for name, _, _, _ in PRIMARY_CONTRASTS}
    primary_results = {
        name: results[name]
        for name, _, _, _ in PRIMARY_CONTRASTS
        if name in results
    }
    expect(primary_estimable == (len(primary_results) == 4), "primary estimability logic drifted")
    partial_primary_descriptive = {}
    if not primary_estimable:
        for name, _, _, _ in PRIMARY_CONTRASTS:
            meta = observed.get(name)
            if meta is None:
                continue
            partial_primary_descriptive[name] = {
                "status": "descriptive_partial_primary_only",
                "source": meta["source"],
                "treatment": meta["treatment"],
                "baseline": meta["baseline"],
                "estimate": meta["estimate"],
                "estimate_bounds_under_verifier_uncertainty": meta["estimate_bounds"],
                "seed_effects": meta["seed_effects"],
                "seed_effect_min": meta["seed_effect_min"],
                "seed_effect_max": meta["seed_effect_max"],
                "interval_released": False,
                "classification_released": False,
                "confirmatory": False,
            }
    return {
        "schema_version": SCHEMA_VERSION,
        "readout": RESULT_ID,
        "status": "released_all_36_terminal",
        "scientific_readout_authorized": True,
        "authorization_is_independent_of_effect_sign": True,
        "release_plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
        "git_commit": EXPECTED_COMMIT,
        "arm_keys": prereg["payload"]["arm_keys"],
        "eligible_heldout_arms": sorted(eligible),
        "terminal_failed_arms": terminal_failed,
        "primary_family_status": "estimable" if primary_estimable else "primary_family_not_estimable",
        "missing_primary_cells": missing_primary,
        "primary_results": primary_results if primary_estimable else {},
        "partial_primary_descriptive": partial_primary_descriptive,
        "secondary_results": {
            **{name: result for name, result in results.items() if name not in all_primary_names},
            **cross_source_results,
        },
        "composite_primary_summary": (
            {
                "K1_helps_across_both_sources": all(
                    results[name]["classification"] == "helps"
                    for name in (
                        "task_rl_k1_ungated_clip5-minus-task_rl@M",
                        "task_rl_k1_ungated_clip5-minus-task_rl@O",
                    )
                ),
                "gating_helps_across_both_sources": all(
                    results[name]["classification"] == "helps"
                    for name in (
                        "task_rl_k1_gated_clip5_beta5-minus-task_rl_k1_ungated_clip5@M",
                        "task_rl_k1_gated_clip5_beta5-minus-task_rl_k1_ungated_clip5@O",
                    )
                ),
                "one_source_cannot_compensate_for_the_other": True,
            }
            if primary_estimable
            else None
        ),
        "arm_accuracy": arm_accuracy,
        "arm_accuracy_bounds_under_verifier_uncertainty": arm_accuracy_bounds,
        "raw_student_accuracy": {
            source: raw_gates[source]["heldout"]["accuracy"]
            for source in SOURCES
            if raw_gates[source].get("status") == "eligible_auxiliary_baseline"
        },
        "raw_student_accuracy_bounds_under_verifier_uncertainty": {
            source: raw_gates[source]["heldout"]["accuracy_bounds_under_verifier_uncertainty"]
            for source in SOURCES
            if raw_gates[source].get("status") == "eligible_auxiliary_baseline"
        },
        "raw_student_auxiliary_status": {
            source: raw_gates[source]["status"] for source in SOURCES
        },
        "pairing": {
            source: {
                "records": len(source_records[source]),
                "record_ids_sha256": canonical_json_sha256(source_records[source]),
                "task_file_sha256": source_task_hash[source],
                "record_seed_contract": source_record_seed_contract[source],
            }
            for source in source_records
        },
        "bootstrap": expected_analysis_contract(),
        "cost": cost,
        "inputs": {
            "arm_gates": {
                key: file_binding(plan["arm_paths"][key]["heldout_gate"])
                for key in prereg["payload"]["arm_keys"]
            },
            "raw_student_gates": {
                source: file_binding(plan["raw_student_auxiliary"][source]["gate"])
                for source in SOURCES
            },
        },
        "claim_boundary": (
            "Three training seeds provide a robustness check, not a scaling law. "
            "Primary classifications use the four-contrast Bonferroni verifier envelope. "
            "Secondary intervals are descriptive and uncorrected."
        ),
    }


def readout_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# OPD objective-family terminal readout",
        "",
        f"Primary family: **{payload['primary_family_status']}**.",
        "",
        "## Four preregistered co-primary contrasts",
        "",
        "| Contrast | Estimate | Robust 98.75% envelope | Classification | Seed effects |",
        "|---|---:|---:|---|---|",
    ]
    if payload["primary_family_status"] != "estimable":
        lines.append(f"| Not estimable | — | — | — | Missing: {', '.join(payload['missing_primary_cells'])} |")
    else:
        for name, _, _, _ in PRIMARY_CONTRASTS:
            result = payload["primary_results"][name]
            low, high = result["verifier_uncertainty"]["robust_envelope"]
            seeds = ", ".join(f"{key}:{value:+.4f}" for key, value in result["seed_effects"].items())
            lines.append(f"| `{name}` | {result['estimate']:+.6f} | [{low:+.6f}, {high:+.6f}] | {result['classification']} | {seeds} |")
    lines.extend([
        "",
        "All 36 registered arms reached immutable terminal states before held-out authorization. "
        "Nulls, harms, and failures are retained. Secondary intervals are descriptive 95% intervals.",
        "",
        str(payload["claim_boundary"]),
        "",
    ])
    return "\n".join(lines)


def release_results(args: argparse.Namespace) -> dict[str, Any]:
    repo = configure_repo(args.repo)
    plan_path, plan, prereg = validate_release_plan(args.release_plan, repo)
    validate_terminal_snapshot(
        plan["terminal_snapshot"], plan_path, plan, repo=repo, prereg=prereg
    )
    finalizer_path = regular_readonly(plan["evaluation_wave_finalizer_receipt"], "evaluation-wave finalizer receipt")
    finalizer = load_json(finalizer_path, "evaluation-wave finalizer receipt")
    expect(finalizer.get("status") == "wave_seal_written" and finalizer.get("controller_return_code") == 0, "evaluation-wave finalizer did not complete successfully")
    expect(finalizer.get("wave_seal") == file_binding(plan["evaluation_wave_seal"]), "evaluation-wave finalizer seal binding drifted")
    arm_gates, raw_gates = _load_and_recompute_gates(
        repo=repo, plan_path=plan_path, plan=plan, prereg=prereg
    )
    payload = build_readout(
        repo=repo,
        plan_path=plan_path,
        plan=plan,
        prereg=prereg,
        arm_gates=arm_gates,
        raw_gates=raw_gates,
    )
    outputs = plan["outputs"]
    expect(Path(args.output_json).resolve() == Path(outputs["json"]).resolve(), "readout JSON output drifted")
    expect(Path(args.output_markdown).resolve() == Path(outputs["markdown"]).resolve(), "readout Markdown output drifted")
    expect(Path(args.output_manifest).resolve() == Path(outputs["manifest"]).resolve(), "readout manifest output drifted")
    json_text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    markdown = readout_markdown(payload)
    created: list[Path] = []
    try:
        json_path = write_new(args.output_json, json_text)
        created.append(json_path)
        markdown_path = write_new(args.output_markdown, markdown)
        created.append(markdown_path)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "bundle": BUNDLE_ID,
            "release_plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
            "json": file_binding(json_path),
            "markdown": file_binding(markdown_path),
            "readout_payload_sha256": canonical_json_sha256(payload),
        }
        manifest_path = write_new(args.output_manifest, manifest)
        created.append(manifest_path)
    except BaseException:
        for path in reversed(created):
            os.chmod(path, 0o644)
            path.unlink(missing_ok=True)
        raise
    return payload


def terminalize_supervisor_exception(
    args: argparse.Namespace, *, phase: str, error: BaseException
) -> dict[str, Any]:
    """Best-effort durable target terminalization for controller exceptions."""

    repo = configure_repo(args.repo)
    plan_path = regular_readonly(args.release_plan, "supervisor release plan")
    plan = load_json(plan_path, "supervisor release plan")
    scope, paths = _evaluation_scope(
        plan, arm_key_value=getattr(args, "arm_key", None),
        raw_source=getattr(args, "raw_source", None),
    )
    authorization_path = regular_readonly(paths["authorization"], f"emergency authorization {scope}")
    submission_path = regular_readonly(paths["submission_receipt"], f"emergency submission {scope}")
    error_evidence = {
        "phase": phase,
        "error_type": type(error).__name__,
        "error_sha256": hashlib.sha256(str(error).encode("utf-8")).hexdigest(),
        "scheduler_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    partial: dict[str, Any] = {}
    for label, field in (
        ("array_accounting_raw", "array_accounting_raw"),
        ("array_accounting", "array_accounting_receipt"),
        ("merge_consumption", "merge_consumption_receipt"),
    ):
        candidate = Path(paths[field])
        if candidate.is_file() and not candidate.is_symlink():
            os.chmod(candidate, 0o444)
            partial[label] = file_binding(candidate)
    private_root = Path(paths["private_log_root"])
    if private_root.is_dir() and not private_root.is_symlink():
        partial["private_log_files_present"] = sorted(
            item.name for item in private_root.iterdir() if item.is_file() and not item.is_symlink()
        )
    failure_path, _ = _write_evaluation_failure(
        plan_path=plan_path, paths=paths, scope=scope, stage=f"{phase}_controller",
        failure_class=type(error).__name__, authorization_path=authorization_path,
        submission_path=submission_path, scheduler_evidence=error_evidence,
        partial_artifacts=partial, merge_invoked=phase == "seal_supervisor",
        machine_validation_may_have_read_outcome_bytes=phase == "seal_supervisor",
    )
    merge_path = Path(paths["merge_supervisor_receipt"])
    if not merge_path.exists():
        merge_payload = {
            "schema_version": SCHEMA_VERSION,
            "receipt": "opd_math_objective_family_evaluation_merge_supervisor_v1",
            "status": "terminalized_controller_exception",
            "created_utc": utc_now(),
            "scope": scope,
            "submission": file_binding(submission_path),
            "array_accounting": partial.get("array_accounting"),
            "merge_consumption": partial.get("merge_consumption"),
            "terminal_failure": file_binding(failure_path),
            "merge_private_log": None,
            "machine_outcomes_inspected": phase == "seal_supervisor",
            "human_outcomes_inspected": False,
        }
        write_new_or_exact(merge_path, merge_payload)
    if phase == "merge_supervisor":
        return load_json(regular_readonly(merge_path, f"emergency merge supervisor {scope}"), f"emergency merge supervisor {scope}")
    artifact_root = Path(paths["artifact_root"])
    if artifact_root.is_dir() and not artifact_root.is_symlink():
        seal_tree_readonly(artifact_root)
    consumption_path = Path(paths["consumption_receipt"])
    if not consumption_path.exists():
        write_new_or_exact(consumption_path, {
            "schema_version": SCHEMA_VERSION,
            "receipt": "opd_math_objective_family_evaluation_consumption_v1",
            "status": "terminal_evaluation_failure",
            "created_utc": utc_now(),
            "scope": scope,
            "authorization": file_binding(authorization_path),
            "submission": file_binding(submission_path),
            "merge_supervisor": file_binding(merge_path),
            "terminal_failure": file_binding(failure_path),
            "artifact_root": None if not artifact_root.is_dir() else {"path": str(artifact_root.resolve()), "tree_sha256": sha256_tree(artifact_root)},
            "numeric_result_fields_copied": False,
            "machine_validation_read_outcome_artifacts": True,
            "human_outcomes_inspected": False,
        })
    seal_path = Path(paths["seal_receipt"])
    if not seal_path.exists():
        write_new_or_exact(seal_path, {
            "schema_version": SCHEMA_VERSION,
            "seal": "opd_math_objective_family_evaluation_target_seal_v1",
            "status": "terminal_evaluation_failure",
            "created_utc": utc_now(),
            "scope": scope,
            "authorization": file_binding(authorization_path),
            "submission": file_binding(submission_path),
            "consumption": file_binding(consumption_path),
            "artifact_tree_sha256": None if not artifact_root.is_dir() else sha256_tree(artifact_root),
            "machine_validation_read_outcome_artifacts": True,
            "human_outcomes_inspected": False,
            "scientific_result_claimed": False,
        })
    supervisor_path = Path(paths["seal_supervisor_receipt"])
    if not supervisor_path.exists():
        write_new_or_exact(supervisor_path, {
            "schema_version": SCHEMA_VERSION,
            "receipt": "opd_math_objective_family_evaluation_seal_supervisor_v1",
            "status": "terminalized_controller_exception",
            "created_utc": utc_now(),
            "scope": scope,
            "submission": file_binding(submission_path),
            "merge_supervisor": file_binding(merge_path),
            "target_seal": file_binding(seal_path),
            "terminal_status": "terminal_evaluation_failure",
            "machine_validation_read_outcome_artifacts": True,
            "human_outcomes_inspected": False,
        })
    return load_json(regular_readonly(supervisor_path, f"emergency seal supervisor {scope}"), f"emergency seal supervisor {scope}")


def record_wave_finalizer(args: argparse.Namespace) -> dict[str, Any]:
    repo = configure_repo(args.repo)
    plan_path = regular_readonly(args.release_plan, "finalizer release plan")
    plan = load_json(plan_path, "finalizer release plan")
    output = Path(args.output).resolve()
    expect(output == Path(plan["evaluation_wave_finalizer_receipt"]).resolve(), "finalizer receipt output drifted")
    private_log = regular_readonly(args.private_log, "finalizer private log")
    expect(private_log == Path(plan["evaluation_wave_finalizer_private_log"]).resolve(), "finalizer private log path drifted")
    controller_rc = int(args.controller_rc)
    wave_seal = None
    if Path(plan["evaluation_wave_seal"]).is_file():
        wave_seal = file_binding(plan["evaluation_wave_seal"])
    expect((controller_rc == 0) == (wave_seal is not None), "finalizer exit/seal consistency drifted")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "receipt": "opd_math_objective_family_evaluation_wave_finalizer_v1",
        "status": "wave_seal_written" if controller_rc == 0 else "terminal_finalizer_controller_failure",
        "created_utc": utc_now(),
        "scheduler_job_id": os.environ.get("SLURM_JOB_ID"),
        "release_plan": file_binding(plan_path),
        "submission_index": file_binding(plan["evaluation_wave_submission_index"]),
        "release_result": file_binding(plan["evaluation_wave_release_result"]),
        "controller_return_code": controller_rc,
        "private_log": file_binding(private_log),
        "wave_seal": wave_seal,
        "retry_authorized": False,
        "human_outcomes_inspected": False,
    }
    return_payload_path = write_new_or_exact(output, payload)
    return load_json(return_payload_path, "finalizer receipt")


def self_test() -> dict[str, Any]:
    if (Path.cwd() / "scripts/opd/objective_family_inputs.py").is_file() and str(Path.cwd()) not in sys.path:
        sys.path.insert(0, str(Path.cwd()))
    values = [float(index) for index in range(10_000)]
    expect(percentile(values, PRIMARY_LOWER_Q) == 62.0, "Bonferroni lower index drifted")
    expect(percentile(values, PRIMARY_UPPER_Q) == 9936.0, "Bonferroni upper index drifted")
    expect(percentile(values, SECONDARY_LOWER_Q) == 249.0, "95% lower index drifted")
    expect(percentile(values, SECONDARY_UPPER_Q) == 9749.0, "95% upper index drifted")
    expect(effect_label([0.01, 0.02]) == "helps", "helps classification drifted")
    expect(effect_label([-0.02, -0.01]) == "harms", "harms classification drifted")
    expect(effect_label([-0.01, 0.01]) == "inconclusive", "tie classification drifted")
    # Freeze the exact crossed RNG order: seed draw, then one M vector, then one O vector.
    rng = random.Random(BOOTSTRAP_SEED)
    first_seed_draw = [rng.randrange(3) for _ in range(3)]
    first_m_draw = [rng.randrange(4) for _ in range(4)]
    first_o_draw = [rng.randrange(5) for _ in range(5)]
    expect(first_seed_draw == [1, 1, 0], "crossed bootstrap seed RNG order drifted")
    expect(first_m_draw == [2, 3, 3, 2], "crossed bootstrap M RNG order drifted")
    expect(first_o_draw == [3, 2, 4, 1, 4], "crossed bootstrap O RNG order drifted")
    expect(canonical_json_sha256({"b": 2, "a": 1}) == canonical_json_sha256({"a": 1, "b": 2}), "canonical JSON drifted")
    expect(allocated_gpu_count("billing=8,cpu=8,gres/gpu=1,gres/gpu:a100=1") == 1, "typed/aggregate GPU parsing drifted")
    expect(allocated_gpu_count("cpu=8,gres/gpu:a100=2") == 2, "typed-only GPU parsing drifted")
    expect(allocated_gpu_count("cpu=4,mem=32G") is None, "no-GPU allocation parsing drifted")
    try:
        allocated_gpu_count("gres/gpu=1,gres/gpu:a100=2")
    except ValueError:
        pass
    else:
        raise ValueError("ambiguous GPU allocation was accepted")
    descriptive = contrast_result(
        name="descriptive",
        formula="a-b",
        estimate=0.1,
        estimate_bounds=[0.1, 0.1],
        point_draws=[0.1, 0.2],
        lower_draws=[0.1, 0.2],
        upper_draws=[0.1, 0.2],
        confirmatory=False,
    )
    expect(descriptive["classification"] is None and descriptive["classification_without_verifier_uncertainty"] is None, "secondary contrast emitted a decision")
    objective_values = {
        "task_rl": 0.40,
        "task_rl_k1_ungated_clip5": 0.60,
        "task_rl_k1_ungated_unclipped": 0.50,
        "task_rl_k1_gated_clip5_beta5": 0.70,
        "k1_bare_verl_compatible_clip10": 0.30,
        "k1_verl_upstream_clip10": 0.30,
    }
    with tempfile.TemporaryDirectory(prefix="opd_release_selftest_") as temporary:
        root = Path(temporary)
        arm_keys = [
            arm_key(objective, source, seed)
            for seed in SEEDS
            for source in SOURCES
            for objective in objective_values
        ]
        release_plan_path = root / "release_plan.json"
        release_plan_path.write_text("{}\n", encoding="utf-8")
        os.chmod(release_plan_path, 0o444)
        plan = {"arm_paths": {}, "raw_student_auxiliary": {}}
        prereg = {"payload": {"objective_ids": list(objective_values), "arm_keys": arm_keys}}
        arm_gates: dict[str, dict[str, Any]] = {}
        for key in arm_keys:
            objective, source, seed_text = key.rsplit("__", 2)
            seed = int(seed_text.removeprefix("seed"))
            gate_path = root / f"{key}.json"
            gate_path.write_text("{}\n", encoding="utf-8")
            os.chmod(gate_path, 0o444)
            plan["arm_paths"][key] = {"heldout_gate": str(gate_path)}
            value = objective_values[objective]
            arm_gates[key] = {
                "status": "eligible_heldout_gate",
                "objective_id": objective,
                "source": source,
                "seed": seed,
                "implementation": "upstream_verl" if objective == "k1_verl_upstream_clip10" else "local",
                "training_terminal": {"scheduler": {"elapsed_seconds": 3600}},
                "training_custody": {"cost": {}},
                "heldout": {
                    "accuracy": value,
                    "accuracy_bounds_under_verifier_uncertainty": [value, value],
                    "record_accuracy": {"r0": value, "r1": value},
                    "record_accuracy_bounds_under_verifier_uncertainty": {
                        "r0": [value, value],
                        "r1": [value, value],
                    },
                    "prepared_binding": {"task_file_sha256": f"task-{source}"},
                    "evaluation_binding": {"record_seed_contract": {"kind": "fixed"}},
                },
            }
        raw_gates = {
            source: {
                "status": "eligible_auxiliary_baseline",
                "heldout": {
                    "accuracy": 0.20,
                    "accuracy_bounds_under_verifier_uncertainty": [0.20, 0.20],
                    "record_accuracy": {"r0": 0.20, "r1": 0.20},
                    "record_accuracy_bounds_under_verifier_uncertainty": {
                        "r0": [0.20, 0.20],
                        "r1": [0.20, 0.20],
                    },
                    "prepared_binding": {"task_file_sha256": f"task-{source}"},
                    "evaluation_binding": {"record_seed_contract": {"kind": "fixed"}},
                }
            }
            for source in SOURCES
        }
        for source in SOURCES:
            raw_gate_path = root / f"raw_{source}.json"
            raw_gate_path.write_text("{}\n", encoding="utf-8")
            os.chmod(raw_gate_path, 0o444)
            plan["raw_student_auxiliary"][source] = {"gate": str(raw_gate_path)}
        synthetic = build_readout(
            repo=root,
            plan_path=release_plan_path,
            plan=plan,
            prereg=prereg,
            arm_gates=arm_gates,
            raw_gates=raw_gates,
        )
        expect(synthetic["primary_family_status"] == "estimable", "synthetic primary family is not estimable")
        for name in (
            "task_rl_k1_ungated_clip5-minus-task_rl@M",
            "task_rl_k1_ungated_clip5-minus-task_rl@O",
        ):
            expect(math.isclose(synthetic["primary_results"][name]["estimate"], 0.20, abs_tol=1e-12), "synthetic K1 contrast drifted")
            expect(synthetic["primary_results"][name]["classification"] == "helps", "synthetic K1 classification drifted")
        for name in (
            "task_rl_k1_gated_clip5_beta5-minus-task_rl_k1_ungated_clip5@M",
            "task_rl_k1_gated_clip5_beta5-minus-task_rl_k1_ungated_clip5@O",
        ):
            expect(math.isclose(synthetic["primary_results"][name]["estimate"], 0.10, abs_tol=1e-12), "synthetic gating contrast drifted")
        expect(all(
            value["classification"] is None
            for value in synthetic["secondary_results"].values()
        ), "synthetic secondary result emitted a decision")
        missing_key = arm_key("task_rl", "M", 0)
        missing_gate = dict(arm_gates[missing_key])
        missing_gate["status"] = "terminal_training_failure_no_heldout"
        incomplete_gates = dict(arm_gates)
        incomplete_gates[missing_key] = missing_gate
        incomplete = build_readout(
            repo=root,
            plan_path=release_plan_path,
            plan=plan,
            prereg=prereg,
            arm_gates=incomplete_gates,
            raw_gates=raw_gates,
        )
        expect(incomplete["primary_family_status"] == "primary_family_not_estimable", "missing primary cell did not close family")
        expect(incomplete["primary_results"] == {}, "partial primary result leaked into confirmatory output")
        primary_names = {name for name, _, _, _ in PRIMARY_CONTRASTS}
        expect(primary_names.isdisjoint(incomplete["secondary_results"]), "partial primary result leaked into secondary output")
        expect(not any(
            name.startswith("clip5_minus_task_rl__") or name.startswith("gated_minus_clip5__")
            for name in incomplete["secondary_results"]
        ), "derived partial-primary result leaked into secondary output")
        expect(incomplete["partial_primary_descriptive"], "available partial-primary estimates were not retained descriptively")
        tree = root / "tree_hash"
        tree.mkdir()
        (tree / "a.txt").write_text("a", encoding="utf-8")
        (tree / "b.txt").write_text("b", encoding="utf-8")
        from scripts.opd.objective_family_inputs import sha256_tree as tracked_tree_hash  # type: ignore

        expect(sha256_tree(tree) == tracked_tree_hash(tree), "external tree hash differs from tracked producer")
        prepared_root = root / "prepared"
        task_dir = prepared_root / "roles/O"
        task_dir.mkdir(parents=True)
        task_file = task_dir / "source_holdout.jsonl"
        holdout_rows = [
            {"record_id": f"o-{index:04d}", "source": "O", "role": "source_holdout"}
            for index in range(400)
        ]
        task_file.write_text("".join(json.dumps(row) + "\n" for row in holdout_rows), encoding="utf-8")
        prepared_manifest = prepared_root / "manifest.json"
        prepared_manifest.write_text(json.dumps({
            "files": {
                "roles/O/source_holdout.jsonl": {
                    "rows": 400,
                    "sha256": sha256_file(task_file),
                }
            }
        }), encoding="utf-8")
        selected_ids = [row["record_id"] for row in holdout_rows[:SELECTED_HOLDOUT_RECORDS]]
        holdout_plan = {"holdout_selection": {"O": {
            "task_file": str(task_file.resolve()),
            "task_file_sha256": sha256_file(task_file),
            "physical_rows": 400,
            "selected_records": SELECTED_HOLDOUT_RECORDS,
            "selected_record_ids_sha256": canonical_json_sha256(selected_ids),
            "selection": "first_370_records_in_registered_source_holdout",
        }}}
        selected = _prepared_holdout_binding(
            {"prepared_manifest": {"path": str(prepared_manifest)}},
            holdout_plan,
            source="O",
            task_path=task_file.resolve(),
            records=SELECTED_HOLDOUT_RECORDS,
        )
        expect(selected["records"] == 370 and selected["record_ids_sha256"] == canonical_json_sha256(selected_ids), "larger O holdout prefix contract drifted")
    lineage_binding = {"path": "/sealed/o_teacher_audit.json", "sha256": "a" * 64}
    lineage_identity = {"teacher_source": "O", "merged_checkpoint_tree_sha256": "b" * 64}
    lineage_times = [
        parse_utc(f"2026-07-21T00:00:0{index}Z", f"lineage test time {index}")
        for index in range(5)
    ]
    lineage = {
        "audit_binding": lineage_binding,
        "program_audit_binding": lineage_binding,
        "plan_audit_binding": lineage_binding,
        "audit_teacher_identity": lineage_identity,
        "prereg_teacher_identity": lineage_identity,
        "audit_created": lineage_times[0],
        "program_created": lineage_times[1],
        "prereg_created": lineage_times[2],
        "launch_created": lineage_times[3],
        "release_created": lineage_times[4],
        "student_outcomes_inspected": False,
        "heldout_outcomes_inspected": False,
    }
    validate_o_teacher_release_lineage_values(**lineage)
    invalid_lineages = []
    for field, value in (
        ("plan_audit_binding", {"path": "/swapped.json", "sha256": "a" * 64}),
        ("prereg_teacher_identity", {"teacher_source": "O", "merged_checkpoint_tree_sha256": "c" * 64}),
        ("audit_created", lineage_times[4]),
        ("student_outcomes_inspected", True),
        ("heldout_outcomes_inspected", True),
    ):
        candidate = dict(lineage)
        candidate[field] = value
        invalid_lineages.append(candidate)
    for candidate in invalid_lineages:
        try:
            validate_o_teacher_release_lineage_values(**candidate)
        except ValueError:
            pass
        else:
            raise AssertionError("O teacher lineage self-test accepted a tampered contract")
    return {
        "status": "passed",
        "bonferroni_indices": [62, 9936],
        "ordinary_95_indices": [249, 9749],
        "first_seed_draw": first_seed_draw,
        "first_M_draw": first_m_draw,
        "first_O_draw": first_o_draw,
    }


def add_scope_selector(parser: argparse.ArgumentParser) -> None:
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--arm-key")
    selector.add_argument("--raw-source", choices=SOURCES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    manifest = commands.add_parser("program-manifest")
    manifest.add_argument("--repo", type=Path, required=True)
    manifest.add_argument("--o-teacher-audit-receipt", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    plan = commands.add_parser("release-plan")
    plan.add_argument("--repo", type=Path, required=True)
    plan.add_argument("--program-manifest", type=Path, required=True)
    plan.add_argument("--preregistration", type=Path, required=True)
    plan.add_argument("--launch-plan", type=Path, required=True)
    plan.add_argument("--evaluation-root", type=Path, required=True)
    plan.add_argument("--result-root", type=Path, required=True)
    plan.add_argument("--control-root", type=Path, required=True)
    plan.add_argument("--train-environment-root", type=Path, required=True)
    plan.add_argument("--hf-home", type=Path, required=True)
    plan.add_argument("--terminal-snapshot", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    authorize_train = commands.add_parser("authorize-training")
    authorize_train.add_argument("--repo", type=Path, required=True)
    authorize_train.add_argument("--release-plan", type=Path, required=True)
    authorize_train.add_argument("--arm-key", required=True)
    authorize_train.add_argument("--output", type=Path, required=True)
    submission = commands.add_parser("record-submission")
    submission.add_argument("--repo", type=Path, required=True)
    submission.add_argument("--release-plan", type=Path, required=True)
    submission.add_argument("--arm-key", required=True)
    submission.add_argument("--scheduler-job-id", required=True)
    submission.add_argument("--output", type=Path, required=True)
    consume_train = commands.add_parser("consume-training-authorization")
    consume_train.add_argument("--repo", type=Path, required=True)
    consume_train.add_argument("--release-plan", type=Path, required=True)
    consume_train.add_argument("--arm-key", required=True)
    consume_train.add_argument("--scheduler-job-id", required=True)
    consume_train.add_argument("--output", type=Path, required=True)
    terminal = commands.add_parser("terminal-snapshot")
    terminal.add_argument("--repo", type=Path, required=True)
    terminal.add_argument("--release-plan", type=Path, required=True)
    terminal.add_argument("--output", type=Path, required=True)
    authorize_wave = commands.add_parser("authorize-evaluation-wave")
    authorize_wave.add_argument("--repo", type=Path, required=True)
    authorize_wave.add_argument("--release-plan", type=Path, required=True)
    authorize_wave.add_argument("--output", type=Path, required=True)
    authorize = commands.add_parser("authorize-evaluation")
    authorize.add_argument("--repo", type=Path, required=True)
    authorize.add_argument("--release-plan", type=Path, required=True)
    add_scope_selector(authorize)
    authorize.add_argument("--output", type=Path, required=True)
    validate_auth = commands.add_parser("validate-evaluation-authorization")
    validate_auth.add_argument("--repo", type=Path, required=True)
    validate_auth.add_argument("--release-plan", type=Path, required=True)
    add_scope_selector(validate_auth)
    submit_eval = commands.add_parser("submit-evaluation")
    submit_eval.add_argument("--repo", type=Path, required=True)
    submit_eval.add_argument("--release-plan", type=Path, required=True)
    add_scope_selector(submit_eval)
    submit_eval.add_argument("--output", type=Path, required=True)
    submit_wave = commands.add_parser("submit-evaluation-wave")
    submit_wave.add_argument("--repo", type=Path, required=True)
    submit_wave.add_argument("--release-plan", type=Path, required=True)
    submit_wave.add_argument("--output", type=Path, required=True)
    consume_eval = commands.add_parser("consume-evaluation-authorization")
    consume_eval.add_argument("--repo", type=Path, required=True)
    consume_eval.add_argument("--release-plan", type=Path, required=True)
    add_scope_selector(consume_eval)
    consume_eval.add_argument("--phase", choices=("shard",), required=True)
    consume_eval.add_argument("--shard-index", type=int)
    consume_eval.add_argument("--output", type=Path, required=True)
    merge_supervisor = commands.add_parser("supervise-evaluation-merge")
    merge_supervisor.add_argument("--repo", type=Path, required=True)
    merge_supervisor.add_argument("--release-plan", type=Path, required=True)
    add_scope_selector(merge_supervisor)
    merge_supervisor.add_argument("--output", type=Path, required=True)
    seal_supervisor = commands.add_parser("supervise-evaluation-seal")
    seal_supervisor.add_argument("--repo", type=Path, required=True)
    seal_supervisor.add_argument("--release-plan", type=Path, required=True)
    add_scope_selector(seal_supervisor)
    seal_supervisor.add_argument("--output", type=Path, required=True)
    seal_eval = commands.add_parser("seal-evaluation")
    seal_eval.add_argument("--repo", type=Path, required=True)
    seal_eval.add_argument("--release-plan", type=Path, required=True)
    add_scope_selector(seal_eval)
    seal_eval.add_argument("--output", type=Path, required=True)
    seal_wave = commands.add_parser("seal-evaluation-wave")
    seal_wave.add_argument("--repo", type=Path, required=True)
    seal_wave.add_argument("--release-plan", type=Path, required=True)
    seal_wave.add_argument("--output", type=Path, required=True)
    finalizer_record = commands.add_parser("record-wave-finalizer")
    finalizer_record.add_argument("--repo", type=Path, required=True)
    finalizer_record.add_argument("--release-plan", type=Path, required=True)
    finalizer_record.add_argument("--controller-rc", type=int, required=True)
    finalizer_record.add_argument("--private-log", type=Path, required=True)
    finalizer_record.add_argument("--output", type=Path, required=True)
    arm_gate = commands.add_parser("arm-gate")
    arm_gate.add_argument("--repo", type=Path, required=True)
    arm_gate.add_argument("--release-plan", type=Path, required=True)
    arm_gate.add_argument("--arm-key", required=True)
    arm_gate.add_argument("--output", type=Path, required=True)
    raw_gate = commands.add_parser("raw-gate")
    raw_gate.add_argument("--repo", type=Path, required=True)
    raw_gate.add_argument("--release-plan", type=Path, required=True)
    raw_gate.add_argument("--raw-source", choices=SOURCES, required=True)
    raw_gate.add_argument("--output", type=Path, required=True)
    release = commands.add_parser("release")
    release.add_argument("--repo", type=Path, required=True)
    release.add_argument("--release-plan", type=Path, required=True)
    release.add_argument("--output-json", type=Path, required=True)
    release.add_argument("--output-markdown", type=Path, required=True)
    release.add_argument("--output-manifest", type=Path, required=True)
    commands.add_parser("self-test")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "program-manifest":
        result = create_program_manifest(args.repo, args.output, args.o_teacher_audit_receipt)
    elif args.command == "release-plan":
        result = create_release_plan(args)
    elif args.command == "authorize-training":
        result = authorize_training(args)
    elif args.command == "record-submission":
        result = record_submission(args)
    elif args.command == "consume-training-authorization":
        result = consume_training_authorization(args)
    elif args.command == "terminal-snapshot":
        result = seal_terminal_snapshot(args)
    elif args.command == "authorize-evaluation-wave":
        result = authorize_evaluation_wave(args)
    elif args.command == "authorize-evaluation":
        raise ValueError("individual evaluation authorization is disabled; use authorize-evaluation-wave")
    elif args.command == "validate-evaluation-authorization":
        result = validate_evaluation_authorization_command(args)
    elif args.command == "submit-evaluation":
        raise ValueError("individual evaluation submission is disabled; use submit-evaluation-wave")
    elif args.command == "submit-evaluation-wave":
        result = submit_evaluation_wave(args)
    elif args.command == "consume-evaluation-authorization":
        result = consume_evaluation_authorization(args)
    elif args.command == "supervise-evaluation-merge":
        try:
            result = supervise_evaluation_merge(args)
        except BaseException as error:
            result = terminalize_supervisor_exception(args, phase="merge_supervisor", error=error)
    elif args.command == "supervise-evaluation-seal":
        try:
            result = supervise_evaluation_seal(args)
        except BaseException as error:
            result = terminalize_supervisor_exception(args, phase="seal_supervisor", error=error)
    elif args.command == "seal-evaluation":
        raise ValueError("individual evaluation sealing is disabled; use the seal supervisor")
    elif args.command == "seal-evaluation-wave":
        result = seal_evaluation_wave(args)
    elif args.command == "record-wave-finalizer":
        result = record_wave_finalizer(args)
    elif args.command == "arm-gate":
        result = seal_arm_gate(args)
    elif args.command == "raw-gate":
        result = seal_raw_student_gate(args)
    elif args.command == "release":
        result = release_results(args)
    else:
        result = self_test()
    print(json.dumps({
        "command": args.command,
        "status": result.get("status", "completed") if isinstance(result, dict) else "completed",
        "payload_sha256": canonical_json_sha256(result),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
