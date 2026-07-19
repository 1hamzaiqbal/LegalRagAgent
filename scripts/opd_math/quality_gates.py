#!/usr/bin/env python3
"""Create identity-bound teacher-gap, teacher-target, and student-support manifests.

Scientific gates are deliberately harder to create than smoke gates.  A smoke
gate can establish that the plumbing works, but its distinct type can never be
used to authorize a merged teacher checkpoint or a scientific OPD run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

try:
    from .data_contract import iter_jsonl
    from .math_reward import verify_completion, verify_trl_accuracy_completion
    from .verify_environment import reverify_recorded_environment
except ImportError:
    from data_contract import iter_jsonl  # type: ignore
    from math_reward import verify_completion, verify_trl_accuracy_completion  # type: ignore
    from verify_environment import reverify_recorded_environment  # type: ignore


SCHEMA_VERSION = 3
TEACHER_GATE_TYPE = "teacher_gap_v1"
TEACHER_SMOKE_GATE_TYPE = "teacher_gap_smoke_v1"
TEACHER_TARGET_REPORT_TYPE = "teacher_target_report_v1"
STUDENT_GATE_TYPE = "student_support_v1"
STUDENT_SMOKE_GATE_TYPE = "student_support_smoke_v1"
DEFAULT_TEACHER_MIN_RECORDS = 200
DEFAULT_STUDENT_MIN_RECORDS = 100
DEFAULT_MIN_PASS_AT_K = 0.01
DEFAULT_MIN_MIXED_GROUP_FRACTION = 0.01
DEFAULT_SCIENTIFIC_BOOTSTRAP_DRAWS = 10_000
MIN_SCIENTIFIC_BOOTSTRAP_DRAWS = 1_000
TEACHER_TARGET_REPORT_RECORDS = 353
SCIENTIFIC_SAMPLES_PER_PROBLEM = 4
TEACHER_GAP_DECODING = {
    "thinking": False,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "max_new_tokens": 1024,
    "seed": 0,
}
STUDENT_SUPPORT_DECODING = {
    "thinking": False,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,
    "max_new_tokens": 512,
    "seed": 0,
}
TEACHER_TRAIN_ROLE = "teacher_train"
TEACHER_GAP_ROLE = "teacher_gap_dev"
STUDENT_SUPPORT_ROLE = "student_opd"
EXPECTED_EVALUATION_PACKAGES = {
    "torch": "2.11.0",
    "transformers": "4.57.6",
    "peft": "0.19.1",
    "math-verify": "0.9.0",
}
EXPECTED_TEACHER_TRAIN_PACKAGES = {
    "torch": "2.11.0",
    "transformers": "4.57.6",
    "trl": "1.8.0",
    "datasets": "4.8.5",
    "peft": "0.19.1",
    "accelerate": "1.14.0",
    "huggingface-hub": "0.36.2",
    "requests": "2.32.5",
    "math-verify": "0.9.0",
}
EVALUATION_SCHEMA_VERSION = 2
EVALUATION_SHARD_KIND = "opd_math_evaluation_shard_v1"
EVALUATION_MERGED_KIND = "opd_math_evaluation_merged_v1"
EVALUATION_CONTRACT = "opd_math_evaluation_contract_v1"
RECORD_SEED_STRATEGY = "task_hash_global_index_record_id_sha256_v1"
SHARD_STRATEGY = "contiguous_balanced_v1"
MERGE_STRATEGY = "ordered_contiguous_shards_v1"
ROOT = Path(__file__).resolve().parents[2]
CANONICAL_TEACHER_TRAINING_PLAN = (
    ROOT / "configs" / "opd_math" / "teacher_training_plan.json"
)
ENVIRONMENT_VERIFIER = ROOT / "scripts" / "opd_math" / "verify_environment.py"


def sha256_file(path: Path) -> str:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_tree(path: Path, *, exclude_relative_paths: Iterable[str] = ()) -> str:
    """Hash a directory tree by sorted relative path and file content.

    Symlinks are rejected so the identity cannot silently change when a target
    outside the adapter/checkpoint directory is replaced.  Directory mtimes,
    owners, and host-specific absolute paths are intentionally excluded.
    """

    root = Path(path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"not a directory tree: {root}")
    excluded = {Path(item).as_posix() for item in exclude_relative_paths}
    files: list[tuple[str, Path]] = []
    for candidate in root.rglob("*"):
        relative = candidate.relative_to(root).as_posix()
        if candidate.is_symlink():
            raise ValueError(f"symlink is not permitted in an identity-bound tree: {candidate}")
        if candidate.is_file() and relative not in excluded:
            files.append((relative, candidate))
    files.sort(key=lambda item: item[0])
    if not files:
        raise ValueError(f"cannot hash an empty directory tree: {root}")

    digest = hashlib.sha256()
    digest.update(b"opd-math-tree-v1\0")
    for relative, candidate in files:
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(bytes.fromhex(sha256_file(candidate)))
    return digest.hexdigest()


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _trainer_log_max_step(path: Path) -> int | None:
    try:
        rows = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"trainer log history is not valid JSON: {path}") from exc
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"trainer log history must be a list of objects: {path}")
    steps: list[int] = []
    for row in rows:
        value = row.get("step")
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"trainer log contains a nonnumeric step: {value!r}")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0 or not numeric.is_integer():
            raise ValueError(f"trainer log contains an invalid optimizer step: {value!r}")
        steps.append(int(numeric))
    return max(steps) if steps else None


def _teacher_reward_signal_from_log(path: Path) -> dict[str, Any]:
    try:
        rows = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"trainer log history is not valid JSON: {path}") from exc
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"trainer log history must be a list of objects: {path}")
    zero_std_fractions: list[float] = []
    clipped_ratios: list[float] = []
    reward_stds: list[float] = []
    for row in rows:
        for field, target in (
            ("frac_reward_zero_std", zero_std_fractions),
            ("completions/clipped_ratio", clipped_ratios),
            ("reward_std", reward_stds),
        ):
            value = row.get(field)
            if value is None:
                continue
            if type(value) not in (int, float) or not math.isfinite(float(value)):
                raise ValueError(f"trainer log contains invalid {field}: {value!r}")
            target.append(float(value))
    informative = bool(zero_std_fractions) and any(
        value < 1.0 for value in zero_std_fractions
    )
    return {
        "informative_reward_observed": informative,
        "reward_log_entries": len(zero_std_fractions),
        "frac_reward_zero_std": zero_std_fractions,
        "max_mixed_reward_sample_fraction": (
            max(1.0 - value for value in zero_std_fractions)
            if zero_std_fractions
            else None
        ),
        "reward_std": reward_stds,
        "completion_clipped_ratio": clipped_ratios,
    }


def _path_from_manifest(raw: Any, manifest_path: Path, field: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{manifest_path} lacks a non-empty {field}")
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = manifest_path.parent / candidate
    return candidate.resolve()


def _immutable_revision(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError(f"{label} must be an immutable 40-character lowercase commit")
    return value


def _prepared_role_binding(
    prepared_manifest_path: Path,
    *,
    source: str,
    role: str,
    task_file: Path,
    selected_records: int,
    strength: str,
    model_kind: str,
    model: str,
    revision: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind a gate to a prepared role file, budget, and pinned model identity."""

    prepared_manifest_path = Path(prepared_manifest_path).resolve()
    prepared = _json_object(prepared_manifest_path, "prepared manifest")
    if prepared.get("schema_version") != 1:
        raise ValueError(f"unsupported prepared manifest schema: {prepared_manifest_path}")
    if strength == "scientific" and prepared.get("scientific_use_allowed") is not True:
        raise ValueError("scientific gates require prepared data authorized for scientific use")
    if strength == "scientific":
        code_state = prepared.get("code_git_state")
        if not isinstance(code_state, dict) or code_state.get("dirty") is not False:
            raise ValueError("scientific gates require prepared data from a clean Git state")
        _immutable_revision(code_state.get("commit"), "prepared-data Git commit")

    relative_task = f"roles/{source}/{role}.jsonl"
    expected_task = (prepared_manifest_path.parent / relative_task).resolve()
    actual_task = Path(task_file).resolve()
    if actual_task != expected_task:
        raise ValueError(
            f"gate task is not the registered {relative_task}: expected={expected_task}, "
            f"actual={actual_task}"
        )
    file_entry = prepared.get("files", {}).get(relative_task)
    if not isinstance(file_entry, dict):
        raise ValueError(f"prepared manifest does not register {relative_task}")
    task_hash = sha256_file(actual_task)
    if file_entry.get("sha256") != task_hash:
        raise ValueError(f"prepared manifest hash drift for {relative_task}")
    registered_rows = file_entry.get("rows")
    if not isinstance(registered_rows, int) or registered_rows <= 0:
        raise ValueError(f"prepared manifest has invalid row count for {relative_task}")
    actual_rows = sum(1 for _ in iter_jsonl(actual_task))
    if actual_rows != registered_rows:
        raise ValueError(
            f"prepared role row-count drift for {relative_task}: "
            f"manifest={registered_rows}, actual={actual_rows}"
        )

    primary_budget = prepared.get("primary_matched_budgets", {}).get(role)
    if not isinstance(primary_budget, int) or primary_budget <= 0:
        raise ValueError(f"prepared manifest lacks a positive primary budget for {role}")
    if strength == "scientific":
        expected_records = registered_rows if role == TEACHER_GAP_ROLE else primary_budget
        if selected_records != expected_records:
            raise ValueError(
                f"scientific {role} evaluation must use exactly {expected_records} registered "
                f"records; got {selected_records}"
            )
    elif selected_records > registered_rows:
        raise ValueError(
            f"smoke evaluation selected {selected_records} records from {registered_rows}-row role"
        )

    source_manifest_path = _path_from_manifest(
        prepared.get("source_manifest_path"), prepared_manifest_path, "source_manifest_path"
    )
    source_manifest_hash = sha256_file(source_manifest_path)
    if prepared.get("source_manifest_sha256") != source_manifest_hash:
        raise ValueError("prepared manifest source-manifest binding has drifted")
    source_manifest = _json_object(source_manifest_path, "source manifest")
    pinned_model = source_manifest.get("models", {}).get(model_kind)
    if not isinstance(pinned_model, dict):
        raise ValueError(f"source manifest lacks models.{model_kind}")
    pinned_revision = _immutable_revision(
        pinned_model.get("revision"), f"source manifest models.{model_kind}.revision"
    )
    if pinned_model.get("id") != model or pinned_revision != revision:
        raise ValueError(
            f"gate model is not pinned models.{model_kind}: "
            f"pinned={pinned_model.get('id')}@{pinned_revision}, requested={model}@{revision}"
        )

    return prepared, {
        "prepared_manifest": str(prepared_manifest_path),
        "prepared_manifest_sha256": sha256_file(prepared_manifest_path),
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": source_manifest_hash,
        "registered_task_file": relative_task,
        "registered_task_rows": registered_rows,
        "primary_matched_role_budget": primary_budget,
        "pinned_model_kind": model_kind,
        "pinned_model": model,
        "pinned_model_revision": revision,
    }


def _prepared_target_pair_binding(
    prepared_manifest_path: Path,
    *,
    teacher_source: str,
    target_source: str,
    task_file: Path,
    selected_records: int,
    model: str,
    revision: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind a cross-source report to its exact prepared pair and target prefix."""

    if teacher_source not in {"M", "O"} or target_source not in {"M", "O"}:
        raise ValueError("teacher and target sources must each be M or O")
    if teacher_source == target_source:
        raise ValueError("teacher-target reports require distinct teacher and target sources")
    prepared, prepared_binding = _prepared_role_binding(
        prepared_manifest_path,
        source=target_source,
        role=TEACHER_GAP_ROLE,
        task_file=task_file,
        selected_records=selected_records,
        strength="target_report",
        model_kind="teacher",
        model=model,
        revision=revision,
    )
    if prepared.get("scientific_use_allowed") is not True:
        raise ValueError("teacher-target reports require prepared data authorized for scientific use")
    code_state = prepared.get("code_git_state")
    if not isinstance(code_state, dict) or code_state.get("dirty") is not False:
        raise ValueError("teacher-target reports require prepared data from a clean Git state")
    _immutable_revision(code_state.get("commit"), "prepared-data Git commit")

    primary_budget = prepared.get("primary_matched_budgets", {}).get(TEACHER_GAP_ROLE)
    if primary_budget != TEACHER_TARGET_REPORT_RECORDS:
        raise ValueError(
            "teacher-target reports require the registered 353-record matched target budget"
        )
    if selected_records != TEACHER_TARGET_REPORT_RECORDS:
        raise ValueError(
            f"teacher-target reports require exactly {TEACHER_TARGET_REPORT_RECORDS} "
            f"target records; got {selected_records}"
        )

    pairs = prepared.get("pairs")
    if not isinstance(pairs, list) or any(not isinstance(pair, dict) for pair in pairs):
        raise ValueError("prepared manifest lacks a valid primary-pair registry")
    matches = [
        pair
        for pair in pairs
        if pair.get("teacher_source") == teacher_source
        and pair.get("opd_source") == target_source
    ]
    if len(matches) != 1:
        raise ValueError(
            "prepared manifest must register exactly one requested cross-source pair"
        )
    pair = matches[0]
    expected_pair_id = f"{teacher_source}_{target_source}"
    if pair.get("id") != expected_pair_id or pair.get("same_items") is not False:
        raise ValueError("prepared cross-source pair has an invalid identity or reuse policy")

    files = prepared.get("files")
    if not isinstance(files, dict):
        raise ValueError("prepared manifest lacks its file registry")

    def require_pair_file(field: str, relative: str) -> dict[str, Any]:
        entry = files.get(relative)
        if not isinstance(entry, dict):
            raise ValueError(f"prepared manifest does not register {relative}")
        if (
            pair.get(field) != relative
            or pair.get(f"{field}_rows") != entry.get("rows")
            or pair.get(f"{field}_sha256") != entry.get("sha256")
        ):
            raise ValueError(f"prepared pair binding has drifted for {field}")
        return entry

    target_relative = f"roles/{target_source}/{TEACHER_GAP_ROLE}.jsonl"
    target_entry = require_pair_file("target_gap_dev_file", target_relative)
    teacher_relative = f"roles/{teacher_source}/{TEACHER_GAP_ROLE}.jsonl"
    require_pair_file("teacher_skill_dev_file", teacher_relative)
    if pair.get("target_gap_dev_limit") != TEACHER_TARGET_REPORT_RECORDS:
        raise ValueError("prepared pair target-gap limit is not exactly 353 records")
    if pair.get("teacher_skill_dev_limit") != TEACHER_TARGET_REPORT_RECORDS:
        raise ValueError("prepared pair teacher-skill limit is not exactly 353 records")
    if target_entry.get("sha256") != sha256_file(task_file):
        raise ValueError("prepared pair target-gap task hash has drifted")

    return prepared, {
        **prepared_binding,
        "pair_id": expected_pair_id,
        "teacher_source": teacher_source,
        "target_source": target_source,
        "target_gap_dev_file": target_relative,
        "target_gap_dev_file_rows": target_entry.get("rows"),
        "target_gap_dev_file_sha256": target_entry.get("sha256"),
        "target_gap_dev_limit": TEACHER_TARGET_REPORT_RECORDS,
    }


def recompute_teacher_trace(
    path: Path,
    *,
    expected_steps: int,
    num_generations: int,
    source: str,
    selected_training_rows: list[dict[str, Any]],
    max_prompt_tokens: int,
    max_completion_tokens: int,
) -> dict[str, Any]:
    """Recompute teacher prompt-group geometry from the immutable sample trace."""

    registered: dict[str, dict[str, Any]] = {}
    registered_index: dict[str, int] = {}
    for row_number, row in enumerate(selected_training_rows, 1):
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"teacher training row {row_number} lacks a stable record_id")
        if record_id in registered:
            raise ValueError(f"teacher training rows repeat record_id {record_id!r}")
        if row.get("source") != source or row.get("role") != TEACHER_TRAIN_ROLE:
            raise ValueError("teacher training rows violate source/role custody")
        if not isinstance(row.get("prompt"), list) or not isinstance(row.get("solution"), str):
            raise ValueError("teacher training row lacks prompt/solution custody")
        registered[record_id] = row
        registered_index[record_id] = row_number - 1

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    completion_token_total = 0
    for row_number, row in enumerate(iter_jsonl(path), 1):
        if row.get("schema_version") != 1:
            raise ValueError(f"teacher trace row {row_number} has unsupported schema")
        batch_index = row.get("reward_batch_index")
        sample_idx = row.get("sample_idx")
        if type(batch_index) is not int or not 0 <= batch_index < expected_steps:
            raise ValueError(f"teacher trace row {row_number} has invalid reward_batch_index")
        if type(sample_idx) is not int or not 0 <= sample_idx < num_generations:
            raise ValueError(f"teacher trace row {row_number} has invalid sample_idx")
        record_id = row.get("record_id")
        if record_id not in registered:
            raise ValueError(
                f"teacher trace row {row_number} uses an unregistered training record: "
                f"{record_id!r}"
            )
        registered_row = registered[str(record_id)]
        if row.get("source") != source:
            raise ValueError(f"teacher trace row {row_number} has the wrong source")
        if row.get("solution") != registered_row["solution"]:
            raise ValueError(f"teacher trace row {row_number} has solution drift")
        if row.get("prompt_sha256") != canonical_json_sha256(registered_row["prompt"]):
            raise ValueError(f"teacher trace row {row_number} has prompt identity drift")
        prompt_tokens = row.get("prompt_tokens")
        prompt_ids = row.get("prompt_token_ids")
        if (
            type(prompt_tokens) is not int
            or prompt_tokens <= 0
            or prompt_tokens > max_prompt_tokens
            or not isinstance(prompt_ids, list)
            or not prompt_ids
            or any(type(value) is not int or value < 0 for value in prompt_ids)
            or len(prompt_ids) != prompt_tokens
        ):
            raise ValueError(f"teacher trace row {row_number} has invalid prompt token IDs")
        completion_ids = row.get("completion_token_ids")
        if (
            not isinstance(completion_ids, list)
            or not completion_ids
            or any(type(value) is not int or value < 0 for value in completion_ids)
            or len(completion_ids) > max_completion_tokens
        ):
            raise ValueError(f"teacher trace row {row_number} has invalid completion token IDs")
        if row.get("completion_tokens") != len(completion_ids):
            raise ValueError(f"teacher trace row {row_number} has completion-token drift")
        completion_text = row.get("completion_text")
        if not isinstance(completion_text, str):
            raise ValueError(f"teacher trace row {row_number} lacks completion text")
        if row.get("completion_sha256") != hashlib.sha256(
            completion_text.encode("utf-8")
        ).hexdigest():
            raise ValueError(f"teacher trace row {row_number} has completion identity drift")
        reward = row.get("reward")
        if (
            type(reward) not in (int, float)
            or not math.isfinite(float(reward))
            or float(reward) not in {0.0, 1.0}
        ):
            raise ValueError(f"teacher trace row {row_number} has invalid binary reward")
        verdict = verify_trl_accuracy_completion(
            completion_text, str(registered_row["solution"])
        )
        recomputed_reward = verdict.get("reward")
        if recomputed_reward is None:
            raise ValueError(
                f"teacher trace row {row_number} cannot be independently verified: {verdict}"
            )
        if float(reward) != float(recomputed_reward):
            raise ValueError(
                f"teacher trace row {row_number} reward disagrees with TRL accuracy "
                f"recomputation: recorded={reward}, recomputed={recomputed_reward}"
            )
        completion_token_total += len(completion_ids)
        grouped[batch_index].append(row)

    if sorted(grouped) != list(range(expected_steps)):
        raise ValueError("teacher trace does not contain every expected reward batch")
    realized_record_ids: list[str] = []
    realized_training_indices: list[int] = []
    prompt_group_tokens = 0
    informative_reward_groups = 0
    reward_sum = 0.0
    for batch_index in sorted(grouped):
        batch = sorted(grouped[batch_index], key=lambda row: row["sample_idx"])
        if [row["sample_idx"] for row in batch] != list(range(num_generations)):
            raise ValueError(f"teacher trace batch {batch_index} has missing/duplicate samples")
        for field in (
            "record_id",
            "source",
            "solution",
            "prompt_sha256",
            "prompt_tokens",
        ):
            if len({row[field] for row in batch}) != 1:
                raise ValueError(f"teacher trace batch {batch_index} mixes {field}")
        if len({tuple(row["prompt_token_ids"]) for row in batch}) != 1:
            raise ValueError(f"teacher trace batch {batch_index} mixes prompt token IDs")
        realized_record_ids.append(str(batch[0]["record_id"]))
        realized_training_indices.append(
            registered_index[realized_record_ids[-1]]
        )
        prompt_group_tokens += int(batch[0]["prompt_tokens"])
        rewards = [float(row["reward"]) for row in batch]
        reward_sum += sum(rewards)
        if len(set(rewards)) > 1:
            informative_reward_groups += 1

    return {
        "reward_batches": len(grouped),
        "completion_samples": sum(len(batch) for batch in grouped.values()),
        "unique_training_records": len(set(realized_record_ids)),
        "realized_record_ids_sha256": canonical_json_sha256(realized_record_ids),
        "realized_training_indices_sha256": canonical_json_sha256(
            realized_training_indices
        ),
        "prompt_group_tokens": prompt_group_tokens,
        "sample_expanded_prompt_tokens": prompt_group_tokens * num_generations,
        "total_completion_tokens": completion_token_total,
        "reward_sum": reward_sum,
        "reward_mean": reward_sum / (expected_steps * num_generations),
        "informative_reward_groups": informative_reward_groups,
        "informative_reward_group_fraction": informative_reward_groups / expected_steps,
        "expected_geometry_observed": (
            len(grouped) == expected_steps
            and sum(len(batch) for batch in grouped.values())
            == expected_steps * num_generations
        ),
    }


def _teacher_environment_binding(
    run: dict[str, Any], *, training_commit: str
) -> dict[str, Any]:
    """Validate and live-reverify the scientific teacher train environment."""

    environment = run.get("environment_contract")
    if not isinstance(environment, dict) or environment.get("schema_version") != 2:
        raise ValueError("scientific teacher run lacks exact train-environment custody")
    if environment.get("git_commit") != training_commit:
        raise ValueError("teacher train-environment commit differs from training Git custody")
    expected_verifier = {
        "path": str(ENVIRONMENT_VERIFIER.resolve()),
        "sha256": sha256_file(ENVIRONMENT_VERIFIER),
    }
    if environment.get("verifier") != expected_verifier:
        raise ValueError("teacher train-environment verifier identity has drifted")
    if environment.get("train_runtime_packages") != EXPECTED_TEACHER_TRAIN_PACKAGES:
        raise ValueError("teacher train-environment runtime package subset has drifted")
    if environment.get("serve_freeze") is not None or environment.get("serve_verification") is not None:
        raise ValueError("teacher training must not bind a serve environment")

    freeze = environment.get("train_freeze")
    recorded = environment.get("train_verification")
    if not isinstance(freeze, dict) or not isinstance(recorded, dict):
        raise ValueError("teacher train-environment binding is incomplete")
    freeze_path = _path_from_manifest(
        freeze.get("path"), CANONICAL_TEACHER_TRAINING_PLAN, "teacher train freeze"
    )
    if (
        freeze_path.name != "train.freeze.txt"
        or freeze_path.parent.name != training_commit
        or freeze_path.parent.parent.name != "environment_freezes"
    ):
        raise ValueError("teacher train freeze is not commit-specific")
    if freeze_path.is_symlink() or not freeze_path.is_file():
        raise ValueError("teacher train freeze is not a regular non-symlink file")
    freeze_hash = sha256_file(freeze_path)
    if freeze.get("sha256") != freeze_hash:
        raise ValueError("teacher train freeze hash has drifted")
    if freeze.get("required_packages") != EXPECTED_TEACHER_TRAIN_PACKAGES:
        raise ValueError("teacher train freeze lacks the pinned package subset")
    if recorded.get("expected_commit") != training_commit or recorded.get("freeze_kind") != "train":
        raise ValueError("teacher live-environment verification has the wrong commit or kind")
    if recorded.get("commit_freeze") != {
        "path": str(freeze_path),
        "sha256": freeze_hash,
        "byte_identical_to_requirements_freeze": True,
    }:
        raise ValueError("teacher live-environment verification does not bind the freeze")
    if reverify_recorded_environment(recorded, in_process=True) != recorded:
        raise ValueError("teacher live train environment differs from its recorded identity")
    for field in (
        "stable_environment_before_candidate_save",
        "stable_environment_after_candidate_save",
        "stable_environment_end",
        "stable_final_artifact_hash",
    ):
        if run.get(field) is not True:
            raise ValueError(f"scientific teacher run lacks {field}")
    return {
        "schema_version": 2,
        "git_commit": training_commit,
        "verifier": expected_verifier,
        "train_runtime_packages": EXPECTED_TEACHER_TRAIN_PACKAGES,
        "train_freeze": {
            "path": str(freeze_path),
            "sha256": freeze_hash,
            "required_packages": EXPECTED_TEACHER_TRAIN_PACKAGES,
        },
        "train_verification": recorded,
        "serve_freeze": None,
        "serve_verification": None,
    }


def _teacher_run_binding(
    run_manifest_path: Path,
    *,
    prepared: dict[str, Any],
    prepared_binding: dict[str, Any],
    source: str,
    model: str,
    revision: str,
    adapter: Path,
    adapter_hash: str,
    strength: str,
) -> dict[str, Any]:
    """Validate that the evaluated adapter is the exact eligible teacher run artifact."""

    run_manifest_path = Path(run_manifest_path).resolve()
    run = _json_object(run_manifest_path, "teacher run manifest")
    if run.get("schema_version") != 1 or run.get("stage") != "teacher_grpo":
        raise ValueError("teacher run manifest has an unsupported schema or stage")
    if run.get("status") != "completed":
        raise ValueError("teacher run did not complete")
    if run.get("model") != model or run.get("model_revision") != revision:
        raise ValueError("teacher run model identity differs from the gate")
    if run.get("source") != source or run.get("role") != TEACHER_TRAIN_ROLE:
        raise ValueError("teacher run source/role differs from the gate")
    if run.get("prepared_manifest") != prepared_binding["prepared_manifest"]:
        raise ValueError("teacher run is bound to a different prepared manifest")
    if run.get("prepared_manifest_sha256") != prepared_binding["prepared_manifest_sha256"]:
        raise ValueError("teacher run prepared-manifest hash has drifted")
    if run.get("source_manifest") != prepared_binding["source_manifest"]:
        raise ValueError("teacher run is bound to a different source manifest")
    if run.get("source_manifest_sha256") != prepared_binding["source_manifest_sha256"]:
        raise ValueError("teacher run source-manifest hash has drifted")
    if run.get("pinned_teacher_model") != model:
        raise ValueError("teacher run pinned-teacher model differs from the source manifest")
    if run.get("pinned_teacher_revision") != revision:
        raise ValueError("teacher run pinned-teacher revision differs from the source manifest")

    training_plan_path = _path_from_manifest(
        run.get("training_plan"), run_manifest_path, "training_plan"
    )
    if training_plan_path != CANONICAL_TEACHER_TRAINING_PLAN.resolve():
        raise ValueError("teacher run is not bound to the canonical tracked training plan")
    training_plan_hash = sha256_file(training_plan_path)
    if run.get("training_plan_sha256") != training_plan_hash:
        raise ValueError("teacher run training-plan hash has drifted")
    training_plan = _json_object(training_plan_path, "teacher training plan")
    if (
        training_plan.get("schema_version") != 1
        or training_plan.get("plan_id") != "opd_math_teacher_primary_v2"
        or training_plan.get("sources") != ["M", "O"]
    ):
        raise ValueError("teacher run training plan has an unsupported identity")
    fixed_config = training_plan.get("fixed_config")
    if not isinstance(fixed_config, dict) or not fixed_config:
        raise ValueError("teacher run training plan lacks fixed_config")
    plan_config_hash = canonical_json_sha256(fixed_config)
    actual_config = run.get("config")
    if not isinstance(actual_config, dict) or not actual_config:
        raise ValueError("teacher run lacks its normalized training config")
    actual_config_hash = canonical_json_sha256(actual_config)
    if run.get("training_plan_config_sha256") != plan_config_hash:
        raise ValueError("teacher run predeclared training-config hash has drifted")
    if run.get("teacher_training_config_sha256") != actual_config_hash:
        raise ValueError("teacher run actual training-config hash has drifted")

    final_adapter = run.get("final_adapter")
    if not isinstance(final_adapter, str) or Path(final_adapter).resolve() != adapter:
        raise ValueError("teacher run final-adapter path differs from the evaluated adapter")
    if run.get("final_adapter_tree_sha256") != adapter_hash:
        raise ValueError("teacher run final-adapter tree hash differs from the evaluated adapter")

    train_relative = f"roles/{source}/{TEACHER_TRAIN_ROLE}.jsonl"
    train_entry = prepared.get("files", {}).get(train_relative)
    if not isinstance(train_entry, dict):
        raise ValueError(f"prepared manifest does not register {train_relative}")
    expected_train_path = (
        Path(prepared_binding["prepared_manifest"]).parent / train_relative
    ).resolve()
    if sha256_file(expected_train_path) != train_entry.get("sha256"):
        raise ValueError("prepared teacher_train role file has drifted")
    if run.get("task_file") != str(expected_train_path):
        raise ValueError("teacher run did not use the registered teacher_train role file")
    if run.get("task_file_sha256") != train_entry.get("sha256"):
        raise ValueError("teacher run teacher_train hash differs from prepared data")
    registered_training_rows = list(iter_jsonl(expected_train_path))
    if len(registered_training_rows) != train_entry.get("rows"):
        raise ValueError("prepared teacher_train role row count has drifted")
    train_budget = prepared.get("primary_matched_budgets", {}).get(TEACHER_TRAIN_ROLE)
    if not isinstance(train_budget, int) or train_budget <= 0:
        raise ValueError("prepared manifest lacks a positive teacher_train budget")

    training_artifact_binding: dict[str, Any] = {}
    training_environment_binding: dict[str, Any] | None = None
    if strength == "scientific":
        if run.get("scientific_use_allowed") is not True:
            raise ValueError("scientific teacher gate requires an eligible teacher run")
        if run.get("intended_scientific_run") is not True:
            raise ValueError("teacher run was not declared as a primary scientific run")
        if run.get("budget_mode") != "primary_matched":
            raise ValueError("scientific teacher gate requires primary_matched teacher training")
        if run.get("selected_rows") != train_budget:
            raise ValueError("teacher run did not use the exact primary teacher_train budget")
        if run.get("training_plan_compliant") is not True or actual_config != fixed_config:
            raise ValueError(
                "scientific teacher gate requires the exact source-independent training plan"
            )
        if actual_config_hash != plan_config_hash:
            raise ValueError("scientific teacher training config differs from the predeclared hash")
        if run.get("packages") != EXPECTED_TEACHER_TRAIN_PACKAGES:
            raise ValueError(
                "scientific teacher training packages differ from the pinned environment: "
                f"expected={EXPECTED_TEACHER_TRAIN_PACKAGES}, actual={run.get('packages')}"
            )
        if run.get("actual_optimizer_steps") != fixed_config.get("max_steps"):
            raise ValueError("teacher run did not complete the plan's exact optimizer-step budget")
        if run.get("optimizer_progress_complete") is not True:
            raise ValueError("teacher run lacks complete optimizer-step progress")
        artifact_specs = {
            "trainer_state": "trainer_state.json",
            "trainer_log_history": "trainer_log_history.json",
            "train_metrics": "train_metrics.json",
        }
        artifact_paths: dict[str, Path] = {}
        for field, filename in artifact_specs.items():
            artifact_path = _path_from_manifest(
                run.get(field), run_manifest_path, field
            )
            expected_path = (run_manifest_path.parent / filename).resolve()
            if artifact_path != expected_path:
                raise ValueError(
                    f"teacher run {field} is not the canonical sibling artifact"
                )
            expected_hash = run.get(f"{field}_sha256")
            if not isinstance(expected_hash, str) or sha256_file(artifact_path) != expected_hash:
                raise ValueError(f"teacher run {field} hash has drifted")
            artifact_paths[field] = artifact_path

        trainer_state = _json_object(artifact_paths["trainer_state"], "trainer state")
        train_metrics = _json_object(artifact_paths["train_metrics"], "train metrics")
        expected_steps = fixed_config.get("max_steps")
        log_max_step = _trainer_log_max_step(artifact_paths["trainer_log_history"])
        reward_signal = _teacher_reward_signal_from_log(
            artifact_paths["trainer_log_history"]
        )
        if trainer_state.get("global_step") != expected_steps:
            raise ValueError("trainer state does not confirm the exact optimizer-step budget")
        if (
            train_metrics.get("actual_optimizer_steps") != expected_steps
            or train_metrics.get("optimizer_progress_complete") is not True
        ):
            raise ValueError("train metrics do not confirm the exact optimizer-step budget")
        if log_max_step != expected_steps or run.get("trainer_log_max_step") != expected_steps:
            raise ValueError("trainer log does not confirm the exact optimizer-step budget")
        if (
            reward_signal.get("informative_reward_observed") is not True
            or run.get("reward_signal") != reward_signal
            or train_metrics.get("reward_signal") != reward_signal
        ):
            raise ValueError(
                "teacher run/metrics do not bind an informative trainer reward signal"
            )
        training_artifact_binding = {
            "teacher_trainer_state": str(artifact_paths["trainer_state"]),
            "teacher_trainer_state_sha256": run["trainer_state_sha256"],
            "teacher_trainer_log_history": str(artifact_paths["trainer_log_history"]),
            "teacher_trainer_log_history_sha256": run["trainer_log_history_sha256"],
            "teacher_train_metrics": str(artifact_paths["train_metrics"]),
            "teacher_train_metrics_sha256": run["train_metrics_sha256"],
            "teacher_trainer_log_max_step": log_max_step,
            "teacher_reward_signal": reward_signal,
        }
        prompt_diagnostics = run.get("prompt_token_diagnostics")
        if not isinstance(prompt_diagnostics, dict):
            raise ValueError("teacher run lacks prompt-token diagnostics")
        if (
            prompt_diagnostics.get("max_prompt_tokens_allowed")
            != fixed_config.get("max_prompt_tokens")
            or not isinstance(prompt_diagnostics.get("max_rendered_prompt_tokens"), int)
            or prompt_diagnostics["max_rendered_prompt_tokens"]
            > fixed_config["max_prompt_tokens"]
            or prompt_diagnostics.get("implicit_truncation_allowed") is not False
        ):
            raise ValueError("teacher run does not satisfy the plan's prompt-length contract")
        if run.get("clean_stable_code") is not True:
            raise ValueError("scientific teacher gate requires clean, stable training code")
        teacher_samples = _path_from_manifest(
            run.get("teacher_samples"), run_manifest_path, "teacher_samples"
        )
        if teacher_samples != (run_manifest_path.parent / "teacher_samples.jsonl").resolve():
            raise ValueError("teacher run samples are not the canonical sibling artifact")
        teacher_samples_hash = sha256_file(teacher_samples)
        if run.get("teacher_samples_sha256") != teacher_samples_hash:
            raise ValueError("teacher run sample-trace hash has drifted")
        expected_sample_rows = fixed_config["max_steps"] * fixed_config["num_generations"]
        expected_unique_records = min(train_budget, fixed_config["max_steps"])
        recomputed_realized = recompute_teacher_trace(
            teacher_samples,
            expected_steps=fixed_config["max_steps"],
            num_generations=fixed_config["num_generations"],
            source=source,
            selected_training_rows=registered_training_rows[:train_budget],
            max_prompt_tokens=fixed_config["max_prompt_tokens"],
            max_completion_tokens=fixed_config["max_completion_length"],
        )
        teacher_sample_rows = recomputed_realized["completion_samples"]
        realized = run.get("realized_training")
        if (
            realized != recomputed_realized
            or train_metrics.get("realized_training") != recomputed_realized
            or run.get("teacher_samples_rows") != expected_sample_rows
            or teacher_sample_rows != expected_sample_rows
            or recomputed_realized.get("expected_geometry_observed") is not True
            or recomputed_realized["unique_training_records"]
            != expected_unique_records
            or recomputed_realized["informative_reward_groups"] <= 0
        ):
            raise ValueError(
                "teacher run/metrics do not equal trace-recomputed prompt geometry"
            )
        training_artifact_binding.update(
            {
                "teacher_samples": str(teacher_samples),
                "teacher_samples_sha256": teacher_samples_hash,
                "teacher_samples_rows": teacher_sample_rows,
                "teacher_realized_training": recomputed_realized,
                "teacher_expected_unique_training_records": expected_unique_records,
            }
        )
        git_states = [
            run.get("git_state_start"),
            run.get("git_state_before_candidate_save"),
            run.get("git_state_after_candidate_save"),
            run.get("git_state_end"),
        ]
        if any(not isinstance(state, dict) for state in git_states):
            raise ValueError("teacher run lacks complete candidate-promotion Git custody")
        start = git_states[0]
        assert isinstance(start, dict)
        training_commit = start.get("commit")
        if (
            start.get("dirty") is not False
            or not isinstance(training_commit, str)
            or re.fullmatch(r"[0-9a-f]{40}", training_commit) is None
            or any(
                state.get("dirty") is not False or state.get("commit") != training_commit
                for state in git_states[1:]
                if isinstance(state, dict)
            )
        ):
            raise ValueError("teacher run Git custody is not clean and stable")
        training_environment_binding = _teacher_environment_binding(
            run, training_commit=training_commit
        )

    return {
        "teacher_run_manifest": str(run_manifest_path),
        "teacher_run_manifest_sha256": sha256_file(run_manifest_path),
        "teacher_training_task_file": str(expected_train_path),
        "teacher_training_task_file_sha256": str(train_entry.get("sha256")),
        "teacher_training_budget_mode": run.get("budget_mode"),
        "teacher_training_selected_rows": run.get("selected_rows"),
        "teacher_training_primary_matched_budget": train_budget,
        "teacher_training_plan": str(training_plan_path),
        "teacher_training_plan_sha256": training_plan_hash,
        "teacher_training_plan_id": training_plan["plan_id"],
        "teacher_training_plan_config_sha256": plan_config_hash,
        "teacher_training_config_sha256": actual_config_hash,
        "teacher_training_packages": run.get("packages"),
        "teacher_training_environment": training_environment_binding,
        "teacher_training_actual_optimizer_steps": run.get("actual_optimizer_steps"),
        "teacher_training_prompt_token_diagnostics": run.get("prompt_token_diagnostics"),
        "teacher_training_git_commit": (
            run.get("git_state_end", {}).get("commit")
            if isinstance(run.get("git_state_end"), dict)
            else None
        ),
        **training_artifact_binding,
    }


def reward_by_record(
    path: Path, *, gold_by_record: dict[str, str]
) -> dict[str, list[float]]:
    """Recompute binary math rewards and reject malformed/duplicate samples."""

    grouped_rows: dict[str, list[tuple[int, float]]] = defaultdict(list)
    seen: set[tuple[str, int]] = set()
    for row_number, row in enumerate(iter_jsonl(path), start=1):
        status = row.get("reward_status")
        if status in {"gold_parse_failed", "verifier_error"}:
            raise RuntimeError(f"verifier failure in {path} at row {row_number}: {status}")
        if status not in {"correct", "incorrect", "prediction_parse_failed"}:
            raise ValueError(f"unknown reward_status in {path} at row {row_number}: {status!r}")
        record_id = row.get("record_id")
        sample_idx = row.get("sample_idx")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"invalid record_id in {path} at row {row_number}")
        if record_id not in gold_by_record:
            raise ValueError(
                f"record_id is not in the selected registered task rows in {path} "
                f"at row {row_number}: {record_id!r}"
            )
        if not isinstance(sample_idx, int) or sample_idx < 0:
            raise ValueError(f"invalid sample_idx in {path} at row {row_number}")
        key = (record_id, sample_idx)
        if key in seen:
            raise ValueError(f"duplicate record/sample identity in {path}: {key}")
        seen.add(key)
        try:
            reward = float(row["reward"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid reward in {path} at row {row_number}") from exc
        if not math.isfinite(reward) or reward not in {0.0, 1.0}:
            raise ValueError(
                f"math gate requires a finite binary reward in {path} at row {row_number}"
            )
        if (status == "correct") != (reward == 1.0):
            raise ValueError(
                f"reward/status disagreement in {path} at row {row_number}: "
                f"reward={reward}, status={status!r}"
            )
        completion = row.get("completion_text")
        if not isinstance(completion, str):
            raise ValueError(
                f"evaluation sample requires completion_text in {path} at row {row_number}"
            )
        verdict = verify_completion(completion, gold_by_record[record_id])
        recomputed_status = verdict.get("status")
        if recomputed_status in {"gold_parse_failed", "verifier_error"}:
            raise RuntimeError(
                f"verifier failure while recomputing {path} at row {row_number}: {verdict}"
            )
        if recomputed_status not in {"correct", "incorrect", "prediction_parse_failed"}:
            raise RuntimeError(
                f"unknown recomputed verifier status in {path} at row {row_number}: "
                f"{recomputed_status!r}"
            )
        try:
            recomputed_reward = float(verdict["reward"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"invalid recomputed verifier reward in {path} at row {row_number}: {verdict}"
            ) from exc
        if (
            not math.isfinite(recomputed_reward)
            or recomputed_reward not in {0.0, 1.0}
        ):
            raise RuntimeError(
                f"recomputed verifier reward is not finite binary in {path} "
                f"at row {row_number}: {verdict}"
            )
        if reward != recomputed_reward or status != recomputed_status:
            raise ValueError(
                f"reported reward/status disagree with verifier recomputation in {path} "
                f"at row {row_number}: reported=({reward}, {status!r}), "
                f"recomputed=({recomputed_reward}, {recomputed_status!r})"
            )
        grouped_rows[record_id].append((sample_idx, recomputed_reward))

    grouped: dict[str, list[float]] = {}
    for record_id, indexed in grouped_rows.items():
        indexed.sort()
        indices = [sample_idx for sample_idx, _ in indexed]
        if indices != list(range(len(indices))):
            raise ValueError(f"sample_idx values are not contiguous for {record_id} in {path}")
        grouped[record_id] = [reward for _, reward in indexed]
    if not grouped:
        raise ValueError(f"evaluation sample file is empty: {path}")
    return grouped


def _record_sampling_seed_v1(
    base_seed: int, task_hash: str, global_record_index: int, record_id: str
) -> int:
    payload = {
        "strategy": RECORD_SEED_STRATEGY,
        "base_seed": base_seed,
        "task_file_sha256": task_hash,
        "global_record_index": global_record_index,
        "record_id": record_id,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def _evaluation_canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _checked_merged_evaluation_provenance(
    summary_path: Path,
    samples_path: Path,
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Independently reconstruct a schema-v2 merged evaluation from its shards."""

    kind = summary.get("artifact_kind")
    if kind == EVALUATION_SHARD_KIND:
        raise ValueError(
            f"an incomplete evaluation shard cannot feed a quality gate: {summary_path}"
        )
    if kind != EVALUATION_MERGED_KIND:
        raise ValueError(f"unsupported schema-v2 evaluation kind in {summary_path}: {kind!r}")

    contract = summary.get("evaluation_contract")
    if not isinstance(contract, dict):
        raise ValueError(f"merged evaluation lacks an evaluation contract: {summary_path}")
    if contract.get("schema_version") != 1 or contract.get("contract") != EVALUATION_CONTRACT:
        raise ValueError(f"unsupported merged evaluation contract: {summary_path}")
    contract_hash = _evaluation_canonical_sha256(contract)
    if summary.get("evaluation_contract_sha256") != contract_hash:
        raise ValueError(f"merged evaluation contract hash mismatch: {summary_path}")

    task_path = _path_from_manifest(contract.get("task_file"), summary_path, "task_file")
    if task_path.is_symlink() or not task_path.is_file():
        raise ValueError(f"merged evaluation task must be a regular file: {task_path}")
    task_hash = sha256_file(task_path)
    if contract.get("task_file_sha256") != task_hash:
        raise ValueError(f"merged evaluation task changed after generation: {task_path}")
    task_rows = list(iter_jsonl(task_path))
    eligible = contract.get("eligible_records")
    if type(eligible) is not int or eligible <= 0 or eligible > len(task_rows):
        raise ValueError(f"merged evaluation has invalid eligible_records: {summary_path}")
    selected_rows = task_rows[:eligible]
    selected_ids = [row.get("record_id") for row in selected_rows]
    if any(not isinstance(record_id, str) or not record_id for record_id in selected_ids):
        raise ValueError(f"merged evaluation task prefix lacks record IDs: {task_path}")
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError(f"merged evaluation task prefix has duplicate record IDs: {task_path}")
    selected_ids_hash = _evaluation_canonical_sha256(selected_ids)
    if contract.get("eligible_record_ids_sha256") != selected_ids_hash:
        raise ValueError(f"merged evaluation selected-record hash mismatch: {summary_path}")

    seed_contract = contract.get("record_seed_contract")
    decoding = contract.get("decoding")
    if not isinstance(decoding, dict):
        raise ValueError(f"merged evaluation contract lacks decoding: {summary_path}")
    expected_seed_contract = {
        "strategy": RECORD_SEED_STRATEGY,
        "base_seed": decoding.get("seed"),
    }
    if (
        type(expected_seed_contract["base_seed"]) is not int
        or expected_seed_contract["base_seed"] < 0
        or seed_contract != expected_seed_contract
    ):
        raise ValueError(f"merged evaluation has an invalid record-seed contract: {summary_path}")
    shard_contract = contract.get("shard")
    if not isinstance(shard_contract, dict) or shard_contract.get("strategy") != SHARD_STRATEGY:
        raise ValueError(f"merged evaluation has an invalid shard contract: {summary_path}")
    shard_count = shard_contract.get("shard_count")
    if type(shard_count) is not int or shard_count <= 0 or shard_count > eligible:
        raise ValueError(f"merged evaluation has an invalid shard count: {summary_path}")

    mirrored_fields = (
        "model",
        "model_revision",
        "adapter",
        "adapter_tree_sha256",
        "task_file",
        "task_file_sha256",
        "tokenizer_contract_sha256",
        "samples_per_problem",
        "decoding",
        "record_seed_contract",
    )
    for field in mirrored_fields:
        if summary.get(field) != contract.get(field):
            raise ValueError(f"merged evaluation {field} differs from its contract")
    if summary.get("records") != eligible or summary.get("eligible_records") != eligible:
        raise ValueError(f"merged evaluation record count differs from its contract")

    code = contract.get("code")
    if not isinstance(code, dict):
        raise ValueError(f"merged evaluation contract lacks code identity: {summary_path}")
    expected_code = {
        "git": {"commit": code.get("git_commit"), "worktree_clean": True},
        "evaluator_file_sha256": code.get("evaluator_file_sha256"),
        "packages": code.get("packages"),
    }
    if summary.get("code") != expected_code:
        raise ValueError(f"merged evaluation code identity differs from its contract")

    merge_custody = summary.get("merge_custody")
    if not isinstance(merge_custody, dict) or merge_custody.get("stable") is not True:
        raise ValueError(f"merged evaluation lacks stable merge custody: {summary_path}")
    merger_path = Path(__file__).resolve().parent / "merge_evaluations.py"
    merger_hash = sha256_file(merger_path)
    for position in ("start", "end"):
        git = merge_custody.get(f"git_{position}")
        if git != expected_code["git"]:
            raise ValueError(f"merged evaluation Git {position} custody mismatch")
        if merge_custody.get(f"merger_file_sha256_{position}") != merger_hash:
            raise ValueError(f"merged evaluation merger-code {position} custody mismatch")
        if merge_custody.get(f"packages_{position}") != code.get("packages"):
            raise ValueError(f"merged evaluation package {position} custody mismatch")
        if merge_custody.get(f"task_file_sha256_{position}") != task_hash:
            raise ValueError(f"merged evaluation task {position} custody mismatch")
        if (
            merge_custody.get(f"adapter_tree_sha256_{position}")
            != contract.get("adapter_tree_sha256")
        ):
            raise ValueError(f"merged evaluation adapter {position} custody mismatch")
    if merge_custody.get("evaluator_file_sha256") != code.get("evaluator_file_sha256"):
        raise ValueError(f"merged evaluation evaluator-code custody mismatch")
    adapter = contract.get("adapter")
    if adapter is None:
        if contract.get("adapter_tree_sha256") is not None:
            raise ValueError("merged raw-model evaluation has an unexpected adapter hash")
    else:
        adapter_path = _path_from_manifest(adapter, summary_path, "adapter")
        if adapter_path.is_symlink() or not adapter_path.is_dir():
            raise ValueError(f"merged evaluation adapter is not a regular directory: {adapter_path}")
        if sha256_tree(adapter_path) != contract.get("adapter_tree_sha256"):
            raise ValueError(f"merged evaluation adapter changed after generation: {adapter_path}")

    merge = summary.get("merge")
    if not isinstance(merge, dict):
        raise ValueError(f"merged evaluation lacks shard provenance: {summary_path}")
    expected_merge_header = {
        "strategy": MERGE_STRATEGY,
        "shard_count": shard_count,
        "global_records": eligible,
        "selected_record_ids_sha256": selected_ids_hash,
    }
    for field, expected in expected_merge_header.items():
        if merge.get(field) != expected:
            raise ValueError(f"merged evaluation {field} provenance mismatch")
    shard_bindings = merge.get("shards")
    if not isinstance(shard_bindings, list) or len(shard_bindings) != shard_count:
        raise ValueError(f"merged evaluation lacks the exact shard set: {summary_path}")

    sample_count = contract.get("samples_per_problem")
    if type(sample_count) is not int or sample_count <= 0:
        raise ValueError(f"merged evaluation has invalid samples_per_problem: {summary_path}")
    reconstructed = bytearray()
    prior_stop = 0
    for expected_index, binding in enumerate(shard_bindings):
        if not isinstance(binding, dict) or binding.get("shard_index") != expected_index:
            raise ValueError(f"merged evaluation shard ordering is not canonical")
        record_start = eligible * expected_index // shard_count
        record_stop = eligible * (expected_index + 1) // shard_count
        if record_start != prior_stop:
            raise ValueError("merged evaluation shard coverage has a gap or overlap")
        prior_stop = record_stop
        expected_slice = selected_rows[record_start:record_stop]
        expected_slice_ids = selected_ids[record_start:record_stop]
        expected_slice_hash = _evaluation_canonical_sha256(expected_slice_ids)
        for field, expected in (
            ("record_start", record_start),
            ("record_stop", record_stop),
            ("selected_record_ids_sha256", expected_slice_hash),
        ):
            if binding.get(field) != expected:
                raise ValueError(f"merged evaluation shard {expected_index} {field} mismatch")

        shard_summary_path = _path_from_manifest(
            binding.get("summary"), summary_path, f"shard {expected_index} summary"
        )
        shard_samples_path = _path_from_manifest(
            binding.get("samples"), summary_path, f"shard {expected_index} samples"
        )
        if shard_summary_path.is_symlink() or shard_samples_path.is_symlink():
            raise ValueError("evaluation shard provenance may not use symlinks")
        if sha256_file(shard_summary_path) != binding.get("summary_sha256"):
            raise ValueError(f"evaluation shard {expected_index} summary changed after merge")
        if sha256_file(shard_samples_path) != binding.get("samples_sha256"):
            raise ValueError(f"evaluation shard {expected_index} samples changed after merge")
        shard_summary = _json_object(shard_summary_path, "evaluation shard summary")
        if (
            shard_summary.get("schema_version") != EVALUATION_SCHEMA_VERSION
            or shard_summary.get("artifact_kind") != EVALUATION_SHARD_KIND
            or shard_summary.get("evaluation_contract") != contract
            or shard_summary.get("evaluation_contract_sha256") != contract_hash
        ):
            raise ValueError(f"evaluation shard {expected_index} contract mismatch")
        expected_shard = {
            "strategy": SHARD_STRATEGY,
            "shard_count": shard_count,
            "shard_index": expected_index,
            "global_records": eligible,
            "record_start": record_start,
            "record_stop": record_stop,
            "selected_record_ids_sha256": expected_slice_hash,
        }
        if shard_summary.get("shard") != expected_shard:
            raise ValueError(f"evaluation shard {expected_index} task slice mismatch")
        declared_shard_samples = _path_from_manifest(
            shard_summary.get("samples_file"), shard_summary_path, "samples_file"
        )
        if declared_shard_samples != shard_samples_path:
            raise ValueError(f"evaluation shard {expected_index} sample path mismatch")
        if shard_summary.get("samples_file_sha256") != binding.get("samples_sha256"):
            raise ValueError(f"evaluation shard {expected_index} sample hash mismatch")

        rows = list(iter_jsonl(shard_samples_path))
        if len(rows) != len(expected_slice) * sample_count:
            raise ValueError(f"evaluation shard {expected_index} sample count mismatch")
        cursor = 0
        for offset, task_row in enumerate(expected_slice):
            global_index = record_start + offset
            record_id = str(task_row["record_id"])
            expected_seed = _record_sampling_seed_v1(
                seed_contract["base_seed"], task_hash, global_index, record_id
            )
            for sample_idx in range(sample_count):
                row = rows[cursor]
                cursor += 1
                if (
                    row.get("schema_version") != EVALUATION_SCHEMA_VERSION
                    or row.get("record_id") != record_id
                    or row.get("global_record_index") != global_index
                    or row.get("record_seed") != expected_seed
                    or row.get("sample_idx") != sample_idx
                ):
                    raise ValueError(
                        f"evaluation shard {expected_index} sample order/seed mismatch"
                    )
        shard_bytes = shard_samples_path.read_bytes()
        if not shard_bytes or not shard_bytes.endswith(b"\n"):
            raise ValueError(f"evaluation shard {expected_index} samples are incomplete")
        reconstructed.extend(shard_bytes)
    if prior_stop != eligible:
        raise ValueError("merged evaluation shard coverage is incomplete")
    if samples_path.read_bytes() != bytes(reconstructed):
        raise ValueError("merged samples are not the exact ordered concatenation of shards")

    return {
        "evaluation_artifact_kind": EVALUATION_MERGED_KIND,
        "evaluation_contract_sha256": contract_hash,
        "record_seed_contract": seed_contract,
        "selected_record_ids_sha256": selected_ids_hash,
        "evaluation_shard_count": shard_count,
        "evaluation_shard_strategy": SHARD_STRATEGY,
        "evaluation_merge_strategy": MERGE_STRATEGY,
        "evaluation_merge_provenance_sha256": _evaluation_canonical_sha256(merge),
        "evaluation_merge_custody_sha256": _evaluation_canonical_sha256(merge_custody),
        "evaluation_merger_file_sha256": merger_hash,
    }


def checked_evaluation(
    summary_path: Path,
    samples_path: Path,
    *,
    expected_model: str,
    expected_revision: str,
    expected_source: str,
    expected_role: str,
) -> tuple[dict[str, Any], dict[str, list[float]], dict[str, Any]]:
    """Validate a summary against its exact samples and task-file inputs."""

    summary_path = Path(summary_path).resolve()
    samples_path = Path(samples_path).resolve()
    summary = _json_object(summary_path, "evaluation summary")
    schema_version = summary.get("schema_version")
    if schema_version == 1:
        evaluation_provenance = {
            "evaluation_artifact_kind": "legacy_monolithic_v1",
            "evaluation_contract_sha256": None,
            "record_seed_contract": None,
            "selected_record_ids_sha256": None,
            "evaluation_shard_count": 1,
            "evaluation_shard_strategy": None,
            "evaluation_merge_strategy": None,
            "evaluation_merge_provenance_sha256": None,
            "evaluation_merge_custody_sha256": None,
            "evaluation_merger_file_sha256": None,
        }
    elif schema_version == EVALUATION_SCHEMA_VERSION:
        evaluation_provenance = _checked_merged_evaluation_provenance(
            summary_path, samples_path, summary
        )
    else:
        raise ValueError(f"unsupported evaluation summary schema: {summary_path}")
    if summary.get("model") != expected_model or summary.get("model_revision") != expected_revision:
        raise ValueError(
            f"evaluation model identity mismatch in {summary_path}: "
            f"expected {expected_model}@{expected_revision}"
        )
    _immutable_revision(summary.get("model_revision"), "evaluation model_revision")
    code = summary.get("code")
    if not isinstance(code, dict):
        raise ValueError(f"evaluation summary lacks code identity: {summary_path}")
    git = code.get("git")
    if not isinstance(git, dict):
        raise ValueError(f"evaluation summary lacks Git identity: {summary_path}")
    _immutable_revision(git.get("commit"), "evaluation Git commit")
    if git.get("worktree_clean") is not True:
        raise ValueError(f"evaluation requires a clean Git worktree: {summary_path}")
    evaluator_hash = code.get("evaluator_file_sha256")
    local_evaluator = Path(__file__).resolve().parent / "evaluate_math.py"
    if not isinstance(evaluator_hash, str) or evaluator_hash != sha256_file(local_evaluator):
        raise ValueError(f"evaluation code hash differs from the current evaluator: {summary_path}")
    packages = code.get("packages")
    if not isinstance(packages, dict) or any(
        packages.get(name) != version
        for name, version in EXPECTED_EVALUATION_PACKAGES.items()
    ):
        raise ValueError(
            f"evaluation package identity differs from the pinned evaluation environment: "
            f"expected={EXPECTED_EVALUATION_PACKAGES}, actual={packages}, summary={summary_path}"
        )
    tokenizer_hash = summary.get("tokenizer_contract_sha256")
    if not isinstance(tokenizer_hash, str) or re.fullmatch(r"[0-9a-f]{64}", tokenizer_hash) is None:
        raise ValueError(f"evaluation summary lacks a tokenizer contract hash: {summary_path}")

    declared_samples = _path_from_manifest(
        summary.get("samples_file"), summary_path, "samples_file"
    )
    if declared_samples != samples_path:
        raise ValueError(
            f"summary samples path mismatch: declared={declared_samples}, supplied={samples_path}"
        )
    samples_hash = sha256_file(samples_path)
    if summary.get("samples_file_sha256") != samples_hash:
        raise ValueError(f"summary/sample hash mismatch: {summary_path} vs {samples_path}")

    task_path = _path_from_manifest(summary.get("task_file"), summary_path, "task_file")
    task_hash = sha256_file(task_path)
    if summary.get("task_file_sha256") != task_hash:
        raise ValueError(f"summary/task hash mismatch: {summary_path} vs {task_path}")
    task_rows = list(iter_jsonl(task_path))
    if not task_rows:
        raise ValueError(f"evaluation task file is empty: {task_path}")
    actual_sources = sorted({str(row.get("source")) for row in task_rows})
    actual_roles = sorted({str(row.get("role")) for row in task_rows})
    if actual_sources != [expected_source] or summary.get("task_sources") != actual_sources:
        raise ValueError(
            f"task source mismatch for {summary_path}: expected={[expected_source]}, "
            f"task={actual_sources}, summary={summary.get('task_sources')}"
        )
    if actual_roles != [expected_role] or summary.get("task_roles") != actual_roles:
        raise ValueError(
            f"task role mismatch for {summary_path}: expected={[expected_role]}, "
            f"task={actual_roles}, summary={summary.get('task_roles')}"
        )

    records = summary.get("records")
    samples_per_problem = summary.get("samples_per_problem")
    if not isinstance(records, int) or records <= 0 or records > len(task_rows):
        raise ValueError(f"invalid evaluation record count in {summary_path}: {records!r}")
    if not isinstance(samples_per_problem, int) or samples_per_problem <= 0:
        raise ValueError(f"invalid samples_per_problem in {summary_path}: {samples_per_problem!r}")
    selected_rows = task_rows[:records]
    selected_ids = [row.get("record_id") for row in selected_rows]
    if any(not isinstance(record_id, str) or not record_id for record_id in selected_ids):
        raise ValueError(f"selected task rows lack record IDs: {task_path}")
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError(f"selected task rows contain duplicate record IDs: {task_path}")
    gold_by_record: dict[str, str] = {}
    for row_number, row in enumerate(selected_rows, start=1):
        solution = row.get("solution")
        if not isinstance(solution, str) or not solution.strip():
            raise ValueError(
                f"selected registered task row lacks a non-empty solution in {task_path} "
                f"at row {row_number}"
            )
        gold_by_record[str(row["record_id"])] = solution

    grouped = reward_by_record(samples_path, gold_by_record=gold_by_record)
    if set(grouped) != set(selected_ids):
        raise ValueError(
            f"sample record set does not equal the first {records} records in {task_path}"
        )
    bad_counts = {
        record_id: len(values)
        for record_id, values in grouped.items()
        if len(values) != samples_per_problem
    }
    if bad_counts:
        preview = dict(list(bad_counts.items())[:10])
        raise ValueError(f"samples_per_problem mismatch in {samples_path}: {preview}")
    total_samples = sum(len(values) for values in grouped.values())
    if summary.get("samples") != total_samples:
        raise ValueError(f"summary sample count mismatch in {summary_path}")
    accuracy = sum(sum(values) for values in grouped.values()) / total_samples
    try:
        reported_accuracy = float(summary["accuracy"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"evaluation summary lacks a valid accuracy: {summary_path}") from exc
    if not math.isclose(reported_accuracy, accuracy, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"summary accuracy mismatch in {summary_path}: "
            f"reported={reported_accuracy}, actual={accuracy}"
        )
    if not isinstance(summary.get("decoding"), dict) or not summary["decoding"]:
        raise ValueError(f"evaluation summary lacks a decoding contract: {summary_path}")
    required_decoding_fields = {
        "thinking",
        "temperature",
        "top_p",
        "top_k",
        "max_new_tokens",
        "seed",
    }
    if not required_decoding_fields.issubset(summary["decoding"]):
        missing = sorted(required_decoding_fields - set(summary["decoding"]))
        raise ValueError(f"evaluation decoding contract lacks {missing}: {summary_path}")
    if summary["decoding"]["thinking"] is not False:
        raise ValueError(f"OPD-math quality gates require non-thinking evaluation: {summary_path}")

    binding = {
        "summary": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "samples": str(samples_path),
        "samples_sha256": samples_hash,
        "task_file": str(task_path),
        "task_file_sha256": task_hash,
        "records": records,
        "samples_per_problem": samples_per_problem,
        "evaluation_git_commit": git["commit"],
        "evaluator_file_sha256": evaluator_hash,
        "evaluation_packages": packages,
        "tokenizer_contract_sha256": tokenizer_hash,
        **evaluation_provenance,
    }
    return summary, grouped, binding


def bootstrap_delta(
    base: dict[str, list[float]], trained: dict[str, list[float]], seed: int, draws: int
) -> tuple[list[str], float, float, float]:
    if draws <= 0:
        raise ValueError("bootstrap_draws must be positive")
    if set(base) != set(trained):
        raise ValueError("base and trained evaluations must contain the identical record set")
    keys = sorted(base)
    if not keys:
        raise ValueError("base and trained evaluations have no shared record IDs")
    mismatched = [key for key in keys if len(base[key]) != len(trained[key])]
    if mismatched:
        raise ValueError(f"base/trained sample counts differ for records: {mismatched[:10]}")
    per_record = [sum(trained[k]) / len(trained[k]) - sum(base[k]) / len(base[k]) for k in keys]
    delta = sum(per_record) / len(per_record)
    rng = random.Random(seed)
    samples = []
    for _ in range(draws):
        sampled_delta = sum(
            per_record[rng.randrange(len(per_record))] for _ in per_record
        ) / len(per_record)
        samples.append(sampled_delta)
    samples.sort()
    low = samples[int(0.025 * (draws - 1))]
    high = samples[int(0.975 * (draws - 1))]
    return keys, delta, low, high


def _gate_strength(args: argparse.Namespace) -> str:
    return "smoke" if bool(getattr(args, "smoke_gate", False)) else "scientific"


def _require_scientific_evaluation_contract(
    summary: dict[str, Any], binding: dict[str, Any], *, decoding: dict[str, Any], label: str
) -> None:
    if binding.get("evaluation_artifact_kind") != EVALUATION_MERGED_KIND:
        raise ValueError(
            f"scientific {label} evaluation requires a schema-v2 merged artifact"
        )
    if binding.get("samples_per_problem") != SCIENTIFIC_SAMPLES_PER_PROBLEM:
        raise ValueError(
            f"scientific {label} evaluation requires exactly "
            f"{SCIENTIFIC_SAMPLES_PER_PROBLEM} samples per problem"
        )
    if summary.get("decoding") != decoding:
        raise ValueError(
            f"scientific {label} evaluation decoding differs from the predeclared contract: "
            f"expected={decoding}, actual={summary.get('decoding')}"
        )


def _minimum_records(args: argparse.Namespace, scientific_default: int) -> int:
    strength = _gate_strength(args)
    requested = getattr(args, "min_records", None)
    default = scientific_default if strength == "scientific" else 1
    minimum = default if requested is None else requested
    if not isinstance(minimum, int) or minimum <= 0:
        raise ValueError("min_records must be a positive integer")
    if strength == "scientific" and minimum < scientific_default:
        raise ValueError(
            f"scientific min_records cannot be lowered below {scientific_default}; use --smoke-gate"
        )
    return minimum


def teacher_gap(args: argparse.Namespace) -> dict[str, Any]:
    strength = _gate_strength(args)
    minimum_records = _minimum_records(args, DEFAULT_TEACHER_MIN_RECORDS)
    if args.min_delta < 0:
        raise ValueError("min_delta cannot be negative")
    if strength == "scientific" and args.bootstrap_draws < MIN_SCIENTIFIC_BOOTSTRAP_DRAWS:
        raise ValueError(
            f"scientific gates require at least {MIN_SCIENTIFIC_BOOTSTRAP_DRAWS} bootstrap draws"
        )

    adapter = Path(args.trained_adapter).resolve()
    adapter_hash = sha256_tree(adapter)
    base_summary, base, base_binding = checked_evaluation(
        args.base_summary,
        args.base_samples,
        expected_model=args.base_model,
        expected_revision=args.base_revision,
        expected_source=args.task_source,
        expected_role=args.task_role,
    )
    trained_summary, trained, trained_binding = checked_evaluation(
        args.trained_summary,
        args.trained_samples,
        expected_model=args.base_model,
        expected_revision=args.base_revision,
        expected_source=args.task_source,
        expected_role=args.task_role,
    )
    if base_summary.get("adapter") is not None:
        raise ValueError("teacher gap base evaluation must not use an adapter")
    if base_summary.get("adapter_tree_sha256") is not None:
        raise ValueError("teacher gap base evaluation has unexpected adapter identity")
    trained_adapter = _path_from_manifest(
        trained_summary.get("adapter"), Path(args.trained_summary).resolve(), "adapter"
    )
    if trained_adapter != adapter:
        raise ValueError(f"trained adapter mismatch: summary={trained_adapter}, supplied={adapter}")
    if trained_summary.get("adapter_tree_sha256") != adapter_hash:
        raise ValueError("trained adapter tree differs from the identity recorded during evaluation")
    if base_binding["task_file_sha256"] != trained_binding["task_file_sha256"]:
        raise ValueError("base and trained teacher evaluations use different task files")
    if base_summary["decoding"] != trained_summary["decoding"]:
        raise ValueError("base and trained teacher evaluations use different decoding contracts")
    if strength == "scientific":
        _require_scientific_evaluation_contract(
            base_summary,
            base_binding,
            decoding=TEACHER_GAP_DECODING,
            label="teacher-gap",
        )
        _require_scientific_evaluation_contract(
            trained_summary,
            trained_binding,
            decoding=TEACHER_GAP_DECODING,
            label="teacher-gap",
        )
    for field in (
        "evaluation_git_commit",
        "evaluator_file_sha256",
        "evaluation_packages",
        "tokenizer_contract_sha256",
        "record_seed_contract",
        "selected_record_ids_sha256",
        "evaluation_shard_count",
        "evaluation_shard_strategy",
        "evaluation_merge_strategy",
    ):
        if base_binding[field] != trained_binding[field]:
            raise ValueError(f"base and trained teacher evaluations differ in {field}")

    keys, delta, low, high = bootstrap_delta(base, trained, args.seed, args.bootstrap_draws)
    prepared, prepared_binding = _prepared_role_binding(
        args.prepared_manifest,
        source=args.task_source,
        role=TEACHER_GAP_ROLE,
        task_file=Path(base_binding["task_file"]),
        selected_records=len(keys),
        strength=strength,
        model_kind="teacher",
        model=args.base_model,
        revision=args.base_revision,
    )
    run_binding = _teacher_run_binding(
        args.teacher_run_manifest,
        prepared=prepared,
        prepared_binding=prepared_binding,
        source=args.task_source,
        model=args.base_model,
        revision=args.base_revision,
        adapter=adapter,
        adapter_hash=adapter_hash,
        strength=strength,
    )
    base_accuracy = sum(sum(base[key]) / len(base[key]) for key in keys) / len(keys)
    trained_accuracy = sum(sum(trained[key]) / len(trained[key]) for key in keys) / len(keys)
    record_requirement_met = len(keys) >= minimum_records
    strict_delta_met = delta > args.min_delta
    positive_ci_met = low > 0
    passed = record_requirement_met and strict_delta_met and (
        positive_ci_met if strength == "scientific" else True
    )
    scientific_authorization = passed and strength == "scientific"
    gate_type = TEACHER_GATE_TYPE if strength == "scientific" else TEACHER_SMOKE_GATE_TYPE
    return {
        "schema_version": SCHEMA_VERSION,
        "gate": gate_type,
        "gate_strength": strength,
        "passed": passed,
        "authorizes_scientific_merge": scientific_authorization,
        "shared_records": len(keys),
        "base_accuracy": base_accuracy,
        "trained_accuracy": trained_accuracy,
        "paired_delta": delta,
        "bootstrap_95_ci": [low, high],
        "min_delta": args.min_delta,
        "min_records": minimum_records,
        "require_positive_ci": strength == "scientific",
        "bootstrap_draws": args.bootstrap_draws,
        "bootstrap_seed": args.seed,
        "requirements": {
            "minimum_records_met": record_requirement_met,
            "strict_delta_met": strict_delta_met,
            "positive_bootstrap_lower_bound_met": positive_ci_met,
        },
        "base_model": args.base_model,
        "base_model_revision": args.base_revision,
        "trained_adapter": str(adapter),
        "trained_adapter_tree_sha256": adapter_hash,
        "task_file": base_binding["task_file"],
        "task_file_sha256": base_binding["task_file_sha256"],
        "task_sources": [args.task_source],
        "task_roles": [args.task_role],
        "decoding": base_summary["decoding"],
        "base_summary": base_binding["summary"],
        "base_summary_sha256": base_binding["summary_sha256"],
        "trained_summary": trained_binding["summary"],
        "trained_summary_sha256": trained_binding["summary_sha256"],
        "base_samples": base_binding["samples"],
        "base_samples_sha256": base_binding["samples_sha256"],
        "trained_samples": trained_binding["samples"],
        "trained_samples_sha256": trained_binding["samples_sha256"],
        "evaluation_git_commit": base_binding["evaluation_git_commit"],
        "evaluator_file_sha256": base_binding["evaluator_file_sha256"],
        "evaluation_packages": base_binding["evaluation_packages"],
        "tokenizer_contract_sha256": base_binding["tokenizer_contract_sha256"],
        "base_evaluation_artifact_kind": base_binding["evaluation_artifact_kind"],
        "trained_evaluation_artifact_kind": trained_binding["evaluation_artifact_kind"],
        "base_evaluation_contract_sha256": base_binding[
            "evaluation_contract_sha256"
        ],
        "trained_evaluation_contract_sha256": trained_binding[
            "evaluation_contract_sha256"
        ],
        "record_seed_contract": base_binding["record_seed_contract"],
        "selected_record_ids_sha256": base_binding["selected_record_ids_sha256"],
        "evaluation_shard_count": base_binding["evaluation_shard_count"],
        "evaluation_shard_strategy": base_binding["evaluation_shard_strategy"],
        "evaluation_merge_strategy": base_binding["evaluation_merge_strategy"],
        "base_evaluation_merge_provenance_sha256": base_binding[
            "evaluation_merge_provenance_sha256"
        ],
        "trained_evaluation_merge_provenance_sha256": trained_binding[
            "evaluation_merge_provenance_sha256"
        ],
        "base_evaluation_merge_custody_sha256": base_binding[
            "evaluation_merge_custody_sha256"
        ],
        "trained_evaluation_merge_custody_sha256": trained_binding[
            "evaluation_merge_custody_sha256"
        ],
        "evaluation_merger_file_sha256": base_binding[
            "evaluation_merger_file_sha256"
        ],
        **prepared_binding,
        **run_binding,
    }


def teacher_target_report(args: argparse.Namespace) -> dict[str, Any]:
    """Create a scientific-strength but strictly non-authorizing target report."""

    if args.teacher_source not in {"M", "O"} or args.target_source not in {"M", "O"}:
        raise ValueError("teacher and target sources must each be M or O")
    if args.teacher_source == args.target_source:
        raise ValueError("teacher-target reports require distinct teacher and target sources")
    if args.task_role != TEACHER_GAP_ROLE:
        raise ValueError("teacher-target reports require task_role=teacher_gap_dev")
    if args.bootstrap_draws != DEFAULT_SCIENTIFIC_BOOTSTRAP_DRAWS:
        raise ValueError(
            "teacher-target reports require exactly 10000 paired bootstrap draws"
        )
    if args.seed != 0:
        raise ValueError("teacher-target reports require bootstrap seed zero")

    adapter = Path(args.trained_adapter).resolve()
    adapter_hash = sha256_tree(adapter)
    base_summary, base, base_binding = checked_evaluation(
        args.base_summary,
        args.base_samples,
        expected_model=args.base_model,
        expected_revision=args.base_revision,
        expected_source=args.target_source,
        expected_role=args.task_role,
    )
    trained_summary, trained, trained_binding = checked_evaluation(
        args.trained_summary,
        args.trained_samples,
        expected_model=args.base_model,
        expected_revision=args.base_revision,
        expected_source=args.target_source,
        expected_role=args.task_role,
    )
    if base_summary.get("adapter") is not None:
        raise ValueError("teacher-target base evaluation must not use an adapter")
    if base_summary.get("adapter_tree_sha256") is not None:
        raise ValueError("teacher-target base evaluation has unexpected adapter identity")
    trained_adapter = _path_from_manifest(
        trained_summary.get("adapter"), Path(args.trained_summary).resolve(), "adapter"
    )
    if trained_adapter != adapter:
        raise ValueError(
            f"trained adapter mismatch: summary={trained_adapter}, supplied={adapter}"
        )
    if trained_summary.get("adapter_tree_sha256") != adapter_hash:
        raise ValueError("trained adapter tree differs from the target evaluation identity")
    if base_binding["task_file_sha256"] != trained_binding["task_file_sha256"]:
        raise ValueError("base and trained target evaluations use different task files")
    if base_summary["decoding"] != trained_summary["decoding"]:
        raise ValueError("base and trained target evaluations use different decoding contracts")
    _require_scientific_evaluation_contract(
        base_summary,
        base_binding,
        decoding=TEACHER_GAP_DECODING,
        label="teacher-target",
    )
    _require_scientific_evaluation_contract(
        trained_summary,
        trained_binding,
        decoding=TEACHER_GAP_DECODING,
        label="teacher-target",
    )
    for field in (
        "evaluation_git_commit",
        "evaluator_file_sha256",
        "evaluation_packages",
        "tokenizer_contract_sha256",
        "record_seed_contract",
        "selected_record_ids_sha256",
        "evaluation_shard_count",
        "evaluation_shard_strategy",
        "evaluation_merge_strategy",
    ):
        if base_binding[field] != trained_binding[field]:
            raise ValueError(f"base and trained target evaluations differ in {field}")

    keys, delta, low, high = bootstrap_delta(
        base, trained, args.seed, args.bootstrap_draws
    )
    prepared, prepared_binding = _prepared_target_pair_binding(
        args.prepared_manifest,
        teacher_source=args.teacher_source,
        target_source=args.target_source,
        task_file=Path(base_binding["task_file"]),
        selected_records=len(keys),
        model=args.base_model,
        revision=args.base_revision,
    )
    run_binding = _teacher_run_binding(
        args.teacher_run_manifest,
        prepared=prepared,
        prepared_binding=prepared_binding,
        source=args.teacher_source,
        model=args.base_model,
        revision=args.base_revision,
        adapter=adapter,
        adapter_hash=adapter_hash,
        strength="scientific",
    )
    base_accuracy = sum(sum(base[key]) / len(base[key]) for key in keys) / len(keys)
    trained_accuracy = (
        sum(sum(trained[key]) / len(trained[key]) for key in keys) / len(keys)
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "report": TEACHER_TARGET_REPORT_TYPE,
        "report_strength": "scientific_measurement",
        "valid": True,
        "authorizes_scientific_merge": False,
        "authorizes_scientific_training": False,
        "claim_boundary": (
            "Cross-source target-distribution measurement only; this report cannot "
            "authorize a teacher merge or student training."
        ),
        "shared_records": len(keys),
        "base_accuracy": base_accuracy,
        "trained_accuracy": trained_accuracy,
        "paired_delta": delta,
        "bootstrap_95_ci": [low, high],
        "bootstrap_draws": args.bootstrap_draws,
        "bootstrap_seed": args.seed,
        "base_model": args.base_model,
        "base_model_revision": args.base_revision,
        "trained_adapter": str(adapter),
        "trained_adapter_tree_sha256": adapter_hash,
        "task_file": base_binding["task_file"],
        "task_file_sha256": base_binding["task_file_sha256"],
        "task_sources": [args.target_source],
        "task_roles": [args.task_role],
        "decoding": base_summary["decoding"],
        "base_summary": base_binding["summary"],
        "base_summary_sha256": base_binding["summary_sha256"],
        "trained_summary": trained_binding["summary"],
        "trained_summary_sha256": trained_binding["summary_sha256"],
        "base_samples": base_binding["samples"],
        "base_samples_sha256": base_binding["samples_sha256"],
        "trained_samples": trained_binding["samples"],
        "trained_samples_sha256": trained_binding["samples_sha256"],
        "evaluation_git_commit": base_binding["evaluation_git_commit"],
        "evaluator_file_sha256": base_binding["evaluator_file_sha256"],
        "evaluation_packages": base_binding["evaluation_packages"],
        "tokenizer_contract_sha256": base_binding["tokenizer_contract_sha256"],
        "base_evaluation_artifact_kind": base_binding["evaluation_artifact_kind"],
        "trained_evaluation_artifact_kind": trained_binding[
            "evaluation_artifact_kind"
        ],
        "base_evaluation_contract_sha256": base_binding[
            "evaluation_contract_sha256"
        ],
        "trained_evaluation_contract_sha256": trained_binding[
            "evaluation_contract_sha256"
        ],
        "record_seed_contract": base_binding["record_seed_contract"],
        "selected_record_ids_sha256": base_binding["selected_record_ids_sha256"],
        "evaluation_shard_count": base_binding["evaluation_shard_count"],
        "evaluation_shard_strategy": base_binding["evaluation_shard_strategy"],
        "evaluation_merge_strategy": base_binding["evaluation_merge_strategy"],
        "base_evaluation_merge_provenance_sha256": base_binding[
            "evaluation_merge_provenance_sha256"
        ],
        "trained_evaluation_merge_provenance_sha256": trained_binding[
            "evaluation_merge_provenance_sha256"
        ],
        "base_evaluation_merge_custody_sha256": base_binding[
            "evaluation_merge_custody_sha256"
        ],
        "trained_evaluation_merge_custody_sha256": trained_binding[
            "evaluation_merge_custody_sha256"
        ],
        "evaluation_merger_file_sha256": base_binding[
            "evaluation_merger_file_sha256"
        ],
        **prepared_binding,
        **run_binding,
    }


def student_support(args: argparse.Namespace) -> dict[str, Any]:
    strength = _gate_strength(args)
    minimum_records = _minimum_records(args, DEFAULT_STUDENT_MIN_RECORDS)
    for name in ("min_pass_at_k", "min_mixed_group_fraction"):
        value = getattr(args, name)
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between zero and one")
    if strength == "scientific" and args.min_pass_at_k < DEFAULT_MIN_PASS_AT_K:
        raise ValueError(
            f"scientific min_pass_at_k cannot be lowered below {DEFAULT_MIN_PASS_AT_K}; "
            "use --smoke-gate"
        )
    if (
        strength == "scientific"
        and args.min_mixed_group_fraction < DEFAULT_MIN_MIXED_GROUP_FRACTION
    ):
        raise ValueError(
            "scientific min_mixed_group_fraction cannot be lowered below "
            f"{DEFAULT_MIN_MIXED_GROUP_FRACTION}; use --smoke-gate"
        )

    summary, grouped, binding = checked_evaluation(
        args.student_summary,
        args.student_samples,
        expected_model=args.student_model,
        expected_revision=args.student_revision,
        expected_source=args.task_source,
        expected_role=args.task_role,
    )
    if summary.get("adapter") is not None:
        raise ValueError("student-support gate currently requires the raw, no-adapter student")
    if summary.get("adapter_tree_sha256") is not None:
        raise ValueError("raw student evaluation has unexpected adapter identity")
    if strength == "scientific":
        _require_scientific_evaluation_contract(
            summary,
            binding,
            decoding=STUDENT_SUPPORT_DECODING,
            label="student-support",
        )
    _, prepared_binding = _prepared_role_binding(
        args.prepared_manifest,
        source=args.task_source,
        role=STUDENT_SUPPORT_ROLE,
        task_file=Path(binding["task_file"]),
        selected_records=len(grouped),
        strength=strength,
        model_kind="student",
        model=args.student_model,
        revision=args.student_revision,
    )
    rewards = [value for values in grouped.values() for value in values]
    mixed = sum(len(set(values)) > 1 for values in grouped.values())
    pass_at_k = sum(
        any(value > 0 for value in values) for values in grouped.values()
    ) / len(grouped)
    mixed_fraction = mixed / len(grouped)
    record_requirement_met = len(grouped) >= minimum_records
    pass_at_k_met = pass_at_k >= args.min_pass_at_k
    mixed_fraction_met = mixed_fraction >= args.min_mixed_group_fraction
    passed = record_requirement_met and pass_at_k_met and mixed_fraction_met
    gate_type = STUDENT_GATE_TYPE if strength == "scientific" else STUDENT_SMOKE_GATE_TYPE
    return {
        "schema_version": SCHEMA_VERSION,
        "gate": gate_type,
        "gate_strength": strength,
        "passed": passed,
        "authorizes_scientific_training": passed and strength == "scientific",
        "records": len(grouped),
        "samples_per_problem": binding["samples_per_problem"],
        "sample_accuracy": sum(rewards) / len(rewards),
        "pass_at_k": pass_at_k,
        "mixed_reward_group_fraction": mixed_fraction,
        "min_pass_at_k": args.min_pass_at_k,
        "min_mixed_group_fraction": args.min_mixed_group_fraction,
        "min_records": minimum_records,
        "requirements": {
            "minimum_records_met": record_requirement_met,
            "minimum_pass_at_k_met": pass_at_k_met,
            "minimum_mixed_group_fraction_met": mixed_fraction_met,
        },
        "student_model": args.student_model,
        "student_model_revision": args.student_revision,
        "student_adapter": None,
        "task_file": binding["task_file"],
        "task_file_sha256": binding["task_file_sha256"],
        "task_sources": [args.task_source],
        "task_roles": [args.task_role],
        "decoding": summary["decoding"],
        "student_summary": binding["summary"],
        "student_summary_sha256": binding["summary_sha256"],
        "student_samples": binding["samples"],
        "student_samples_sha256": binding["samples_sha256"],
        "evaluation_git_commit": binding["evaluation_git_commit"],
        "evaluator_file_sha256": binding["evaluator_file_sha256"],
        "evaluation_packages": binding["evaluation_packages"],
        "tokenizer_contract_sha256": binding["tokenizer_contract_sha256"],
        "evaluation_artifact_kind": binding["evaluation_artifact_kind"],
        "evaluation_contract_sha256": binding["evaluation_contract_sha256"],
        "record_seed_contract": binding["record_seed_contract"],
        "selected_record_ids_sha256": binding["selected_record_ids_sha256"],
        "evaluation_shard_count": binding["evaluation_shard_count"],
        "evaluation_shard_strategy": binding["evaluation_shard_strategy"],
        "evaluation_merge_strategy": binding["evaluation_merge_strategy"],
        "evaluation_merge_provenance_sha256": binding[
            "evaluation_merge_provenance_sha256"
        ],
        "evaluation_merge_custody_sha256": binding[
            "evaluation_merge_custody_sha256"
        ],
        "evaluation_merger_file_sha256": binding[
            "evaluation_merger_file_sha256"
        ],
        **prepared_binding,
    }


def recompute_teacher_gate(gate: dict[str, Any]) -> dict[str, Any]:
    """Recompute a scientific teacher gate from its bound raw artifacts."""

    required_paths = (
        "base_summary",
        "base_samples",
        "trained_summary",
        "trained_samples",
        "trained_adapter",
        "prepared_manifest",
        "teacher_run_manifest",
    )
    paths: dict[str, Path] = {}
    for field in required_paths:
        value = gate.get(field)
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ValueError(f"teacher gate lacks an absolute {field} recomputation binding")
        paths[field] = Path(value)
    sources = gate.get("task_sources")
    roles = gate.get("task_roles")
    if sources not in (["M"], ["O"]) or roles != [TEACHER_GAP_ROLE]:
        raise ValueError("teacher gate lacks a canonical source/role recomputation contract")
    return teacher_gap(
        argparse.Namespace(
            base_summary=paths["base_summary"],
            base_samples=paths["base_samples"],
            trained_summary=paths["trained_summary"],
            trained_samples=paths["trained_samples"],
            base_model=gate.get("base_model"),
            base_revision=gate.get("base_model_revision"),
            trained_adapter=paths["trained_adapter"],
            prepared_manifest=paths["prepared_manifest"],
            teacher_run_manifest=paths["teacher_run_manifest"],
            task_source=sources[0],
            task_role=roles[0],
            min_delta=gate.get("min_delta"),
            min_records=gate.get("min_records"),
            bootstrap_draws=gate.get("bootstrap_draws"),
            seed=gate.get("bootstrap_seed"),
            smoke_gate=False,
        )
    )


def recompute_teacher_target_report(report: dict[str, Any]) -> dict[str, Any]:
    """Recompute a teacher-target report from its identity-bound artifacts."""

    if report.get("report") != TEACHER_TARGET_REPORT_TYPE:
        raise ValueError("not a canonical teacher-target report")
    required_paths = (
        "base_summary",
        "base_samples",
        "trained_summary",
        "trained_samples",
        "trained_adapter",
        "prepared_manifest",
        "teacher_run_manifest",
    )
    paths: dict[str, Path] = {}
    for field in required_paths:
        value = report.get(field)
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ValueError(
                f"teacher-target report lacks an absolute {field} recomputation binding"
            )
        paths[field] = Path(value)
    teacher_source = report.get("teacher_source")
    target_source = report.get("target_source")
    roles = report.get("task_roles")
    if (
        teacher_source not in {"M", "O"}
        or target_source not in {"M", "O"}
        or teacher_source == target_source
        or roles != [TEACHER_GAP_ROLE]
    ):
        raise ValueError("teacher-target report lacks a canonical cross-source contract")
    return teacher_target_report(
        argparse.Namespace(
            base_summary=paths["base_summary"],
            base_samples=paths["base_samples"],
            trained_summary=paths["trained_summary"],
            trained_samples=paths["trained_samples"],
            base_model=report.get("base_model"),
            base_revision=report.get("base_model_revision"),
            trained_adapter=paths["trained_adapter"],
            prepared_manifest=paths["prepared_manifest"],
            teacher_run_manifest=paths["teacher_run_manifest"],
            teacher_source=teacher_source,
            target_source=target_source,
            task_role=roles[0],
            bootstrap_draws=report.get("bootstrap_draws"),
            seed=report.get("bootstrap_seed"),
        )
    )


def recompute_student_gate(gate: dict[str, Any]) -> dict[str, Any]:
    """Recompute a scientific student-support gate from its bound raw artifacts."""

    required_paths = ("student_summary", "student_samples", "prepared_manifest")
    paths: dict[str, Path] = {}
    for field in required_paths:
        value = gate.get(field)
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ValueError(f"student gate lacks an absolute {field} recomputation binding")
        paths[field] = Path(value)
    sources = gate.get("task_sources")
    roles = gate.get("task_roles")
    if sources not in (["M"], ["O"]) or roles != [STUDENT_SUPPORT_ROLE]:
        raise ValueError("student gate lacks a canonical source/role recomputation contract")
    return student_support(
        argparse.Namespace(
            student_summary=paths["student_summary"],
            student_samples=paths["student_samples"],
            student_model=gate.get("student_model"),
            student_revision=gate.get("student_model_revision"),
            prepared_manifest=paths["prepared_manifest"],
            task_source=sources[0],
            task_role=roles[0],
            min_pass_at_k=gate.get("min_pass_at_k"),
            min_mixed_group_fraction=gate.get("min_mixed_group_fraction"),
            min_records=gate.get("min_records"),
            smoke_gate=False,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="gate_command", required=True)

    teacher = sub.add_parser("teacher-gap")
    teacher.add_argument("--base-summary", type=Path, required=True)
    teacher.add_argument("--base-samples", type=Path, required=True)
    teacher.add_argument("--trained-summary", type=Path, required=True)
    teacher.add_argument("--trained-samples", type=Path, required=True)
    teacher.add_argument("--base-model", required=True)
    teacher.add_argument("--base-revision", required=True)
    teacher.add_argument("--trained-adapter", type=Path, required=True)
    teacher.add_argument("--prepared-manifest", type=Path, required=True)
    teacher.add_argument("--teacher-run-manifest", type=Path, required=True)
    teacher.add_argument("--task-source", choices=("M", "O"), required=True)
    teacher.add_argument("--task-role", default="teacher_gap_dev")
    teacher.add_argument("--min-delta", type=float, default=0.0)
    teacher.add_argument("--min-records", type=int)
    teacher.add_argument("--bootstrap-draws", type=int, default=DEFAULT_SCIENTIFIC_BOOTSTRAP_DRAWS)
    teacher.add_argument("--seed", type=int, default=0)
    teacher.add_argument(
        "--smoke-gate",
        action="store_true",
        help="emit a non-scientific smoke-gate type that cannot authorize checkpoint merging",
    )
    teacher.add_argument("--output", type=Path, required=True)

    target = sub.add_parser("teacher-target-report")
    target.add_argument("--base-summary", type=Path, required=True)
    target.add_argument("--base-samples", type=Path, required=True)
    target.add_argument("--trained-summary", type=Path, required=True)
    target.add_argument("--trained-samples", type=Path, required=True)
    target.add_argument("--base-model", required=True)
    target.add_argument("--base-revision", required=True)
    target.add_argument("--trained-adapter", type=Path, required=True)
    target.add_argument("--prepared-manifest", type=Path, required=True)
    target.add_argument("--teacher-run-manifest", type=Path, required=True)
    target.add_argument("--teacher-source", choices=("M", "O"), required=True)
    target.add_argument("--target-source", choices=("M", "O"), required=True)
    target.add_argument("--task-role", default="teacher_gap_dev")
    target.add_argument(
        "--bootstrap-draws",
        type=int,
        default=DEFAULT_SCIENTIFIC_BOOTSTRAP_DRAWS,
    )
    target.add_argument("--seed", type=int, default=0)
    target.add_argument("--output", type=Path, required=True)

    student = sub.add_parser("student-support")
    student.add_argument("--student-summary", type=Path, required=True)
    student.add_argument("--student-samples", type=Path, required=True)
    student.add_argument("--student-model", required=True)
    student.add_argument("--student-revision", required=True)
    student.add_argument("--prepared-manifest", type=Path, required=True)
    student.add_argument("--task-source", choices=("M", "O"), required=True)
    student.add_argument("--task-role", default="student_opd")
    student.add_argument("--min-pass-at-k", type=float, default=DEFAULT_MIN_PASS_AT_K)
    student.add_argument(
        "--min-mixed-group-fraction",
        type=float,
        default=DEFAULT_MIN_MIXED_GROUP_FRACTION,
    )
    student.add_argument("--min-records", type=int)
    student.add_argument(
        "--smoke-gate",
        action="store_true",
        help="emit a non-scientific smoke-gate type that cannot authorize scientific training",
    )
    student.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.gate_command == "teacher-gap":
        result = teacher_gap(args)
    elif args.gate_command == "teacher-target-report":
        result = teacher_target_report(args)
    else:
        result = student_support(args)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing to overwrite an existing gate manifest: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    if args.gate_command == "teacher-target-report":
        return 0
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
