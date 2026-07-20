#!/usr/bin/env python3
"""Fail-closed merger for immutable OPD-math evaluation shards."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

try:
    from .data_contract import iter_jsonl
    from .evaluate_math import (
        EVALUATION_CONTRACT,
        LEGACY_EVALUATION_CONTRACT,
        EVALUATION_MERGED_KIND,
        EVALUATION_SHARD_KIND,
        MERGE_STRATEGY,
        POST_PROMOTION_CUSTODY_SCHEMA_VERSION,
        POST_PROMOTION_TREE_ALGORITHM,
        RECORD_SEED_STRATEGY,
        ROOT,
        SAMPLE_SCHEMA_VERSION,
        SHARD_STRATEGY,
        SUMMARY_SCHEMA_VERSION,
        balanced_shard_bounds,
        begin_transactional_directory,
        git_identity,
        package_versions,
        post_promotion_custody_path,
        promote_transactional_directory,
        publish_transactional_artifact,
        record_sampling_seed,
        sha256_file,
        sha256_tree,
        validate_evaluation_environment_contract,
        evaluation_environment_contract_unchanged,
        write_text_fsync,
        _is_complete_o_teacher_gap,
    )
    from .math_reward import (
        EVALUATION_VERIFIER_ERROR_POLICY,
        EVALUATION_VERIFIER_MAX_ATTEMPTS,
        MAX_EVALUATION_VERIFIER_ERROR_FRACTION,
        verify_evaluation_completion,
    )
    from .tokenizer_contract import canonical_sha256
except ImportError:
    from data_contract import iter_jsonl  # type: ignore
    from evaluate_math import (  # type: ignore
        EVALUATION_CONTRACT,
        LEGACY_EVALUATION_CONTRACT,
        EVALUATION_MERGED_KIND,
        EVALUATION_SHARD_KIND,
        MERGE_STRATEGY,
        POST_PROMOTION_CUSTODY_SCHEMA_VERSION,
        POST_PROMOTION_TREE_ALGORITHM,
        RECORD_SEED_STRATEGY,
        ROOT,
        SAMPLE_SCHEMA_VERSION,
        SHARD_STRATEGY,
        SUMMARY_SCHEMA_VERSION,
        balanced_shard_bounds,
        begin_transactional_directory,
        git_identity,
        package_versions,
        post_promotion_custody_path,
        promote_transactional_directory,
        publish_transactional_artifact,
        record_sampling_seed,
        sha256_file,
        sha256_tree,
        validate_evaluation_environment_contract,
        evaluation_environment_contract_unchanged,
        write_text_fsync,
        _is_complete_o_teacher_gap,
    )
    from math_reward import (  # type: ignore
        EVALUATION_VERIFIER_ERROR_POLICY,
        EVALUATION_VERIFIER_MAX_ATTEMPTS,
        MAX_EVALUATION_VERIFIER_ERROR_FRACTION,
        verify_evaluation_completion,
    )
    from tokenizer_contract import canonical_sha256  # type: ignore


HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
MERGER_PATH = Path(__file__).resolve()
EVALUATOR_PATH = MERGER_PATH.parent / "evaluate_math.py"


def _json_object(path: Path, label: str) -> dict[str, Any]:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _expect(payload: Mapping[str, Any], key: str, expected: Any, label: str) -> None:
    if payload.get(key) != expected:
        raise ValueError(
            f"{label} {key} mismatch: expected={expected!r}, actual={payload.get(key)!r}"
        )


def _absolute(raw: Any, anchor: Path, label: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{label} must be a non-empty path")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = anchor.parent / path
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError(f"{label} may not traverse a symlinked artifact leaf: {path}")
    return path.resolve()


def _hash(value: Any, label: str, pattern: re.Pattern[str] = HEX64) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(f"{label} must be an immutable hash")
    return value


def _finite_number(value: Any, label: str, *, nonnegative: bool = False) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be finite")
    numeric = float(value)
    if nonnegative and numeric < 0:
        raise ValueError(f"{label} must be nonnegative")
    return numeric


def _same_float(actual: Any, expected: float, label: str) -> None:
    numeric = _finite_number(actual, label)
    if not math.isclose(numeric, expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label} mismatch: expected={expected}, actual={numeric}")


def _checked_task_rows(task_file: Path) -> tuple[list[dict[str, Any]], str]:
    task_file = Path(task_file).expanduser()
    if task_file.is_symlink() or not task_file.is_file():
        raise ValueError(f"task file must be a regular non-symlink file: {task_file}")
    task_file = task_file.resolve()
    rows = list(iter_jsonl(task_file))
    if not rows:
        raise ValueError("evaluation task file is empty")
    record_ids: list[str] = []
    for index, row in enumerate(rows):
        record_id = row.get("record_id")
        solution = row.get("solution")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"task row {index} lacks a stable record_id")
        if not isinstance(solution, str) or not solution.strip():
            raise ValueError(f"task row {index} lacks a non-empty solution")
        record_ids.append(record_id)
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("evaluation task file contains duplicate record IDs")
    return rows, sha256_file(task_file)


def _validate_contract(
    contract: Any,
    *,
    task_file: Path,
    task_rows: list[dict[str, Any]],
    task_hash: str,
) -> dict[str, Any]:
    if not isinstance(contract, dict):
        raise ValueError("evaluation shard lacks an evaluation_contract")
    _expect(contract, "schema_version", 1, "evaluation contract")
    contract_name = contract.get("contract")
    if contract_name not in {EVALUATION_CONTRACT, LEGACY_EVALUATION_CONTRACT}:
        raise ValueError("evaluation contract has an unsupported version")
    _expect(contract, "task_file", str(task_file.resolve()), "evaluation contract")
    _expect(contract, "task_file_sha256", task_hash, "evaluation contract")
    eligible = contract.get("eligible_records")
    if type(eligible) is not int or eligible <= 0 or eligible > len(task_rows):
        raise ValueError("evaluation contract has an invalid eligible_records count")
    selected_rows = task_rows[:eligible]
    selected_ids = [row["record_id"] for row in selected_rows]
    _expect(
        contract,
        "eligible_record_ids_sha256",
        canonical_sha256(selected_ids),
        "evaluation contract",
    )
    if not isinstance(contract.get("model"), str) or not contract["model"]:
        raise ValueError("evaluation contract lacks a model")
    _hash(contract.get("model_revision"), "evaluation model revision", HEX40)
    adapter = contract.get("adapter")
    adapter_hash = contract.get("adapter_tree_sha256")
    if adapter is None:
        if adapter_hash is not None:
            raise ValueError("base evaluation contract has an adapter hash without an adapter")
    else:
        adapter_path = Path(str(adapter)).expanduser()
        if not adapter_path.is_absolute() or adapter_path.is_symlink() or not adapter_path.is_dir():
            raise ValueError("evaluation contract adapter is not an absolute regular directory")
        _hash(adapter_hash, "evaluation adapter tree")
    samples_per_problem = contract.get("samples_per_problem")
    if type(samples_per_problem) is not int or samples_per_problem <= 0:
        raise ValueError("evaluation contract has invalid samples_per_problem")
    expected_reward_verifier = {
        "candidate_error_policy": EVALUATION_VERIFIER_ERROR_POLICY,
        "maximum_attempts": EVALUATION_VERIFIER_MAX_ATTEMPTS,
        "maximum_error_fraction": MAX_EVALUATION_VERIFIER_ERROR_FRACTION,
        "training_policy": "abort",
    }
    if contract.get("reward_verifier") != expected_reward_verifier:
        raise ValueError("evaluation contract has an unsupported reward-verifier policy")
    decoding = contract.get("decoding")
    if not isinstance(decoding, dict) or decoding.get("thinking") is not False:
        raise ValueError("evaluation contract lacks non-thinking decoding")
    base_seed = decoding.get("seed")
    if type(base_seed) is not int or base_seed < 0:
        raise ValueError("evaluation contract has an invalid base seed")
    seed_contract = contract.get("record_seed_contract")
    expected_seed_contract = {"strategy": RECORD_SEED_STRATEGY, "base_seed": base_seed}
    if seed_contract != expected_seed_contract:
        raise ValueError("evaluation contract has an unsupported record-seed contract")
    shard = contract.get("shard")
    if not isinstance(shard, dict) or shard.get("strategy") != SHARD_STRATEGY:
        raise ValueError("evaluation contract has an unsupported shard strategy")
    shard_count = shard.get("shard_count")
    if type(shard_count) is not int or shard_count <= 0 or shard_count > eligible:
        raise ValueError("evaluation contract has an invalid shard count")
    _hash(contract.get("tokenizer_contract_sha256"), "tokenizer contract")
    code = contract.get("code")
    if not isinstance(code, dict):
        raise ValueError("evaluation contract lacks code custody")
    _hash(code.get("git_commit"), "evaluation Git commit", HEX40)
    _hash(code.get("evaluator_file_sha256"), "evaluation code")
    if not isinstance(code.get("packages"), dict) or not code["packages"]:
        raise ValueError("evaluation contract lacks package custody")
    environment_contract = code.get("environment_contract")
    if environment_contract is not None and not isinstance(environment_contract, dict):
        raise ValueError("evaluation contract has invalid train-environment custody")
    if contract_name == EVALUATION_CONTRACT and environment_contract is None:
        raise ValueError("exact-environment evaluation contract lacks environment custody")
    if contract_name == LEGACY_EVALUATION_CONTRACT and environment_contract is not None:
        raise ValueError("legacy evaluation contract cannot claim exact environment custody")
    if environment_contract is not None and (
        code.get("packages") != environment_contract.get("train_runtime_packages")
    ):
        raise ValueError("evaluation package custody differs from the exact environment")
    expected_sources = sorted({str(row.get("source")) for row in selected_rows})
    expected_roles = sorted({str(row.get("role")) for row in selected_rows})
    _expect(contract, "task_sources", expected_sources, "evaluation contract")
    _expect(contract, "task_roles", expected_roles, "evaluation contract")
    complete_o_gap = _is_complete_o_teacher_gap(
        selected_rows, physical_record_count=len(task_rows)
    )
    evaluation_plan = contract.get("evaluation_plan")
    if complete_o_gap:
        if not isinstance(evaluation_plan, dict):
            raise ValueError(
                "complete O teacher_gap_dev evaluation contract lacks the canonical v2 plan"
            )
        try:
            from .plan_evaluation_shards import revalidate_plan_binding_for_contract
        except ImportError:
            from plan_evaluation_shards import (  # type: ignore
                revalidate_plan_binding_for_contract,
            )
        revalidate_plan_binding_for_contract(evaluation_plan, contract)
    elif evaluation_plan is not None:
        raise ValueError(
            "evaluation plan binding is forbidden outside complete O teacher_gap_dev"
        )
    return contract


def _validate_shard_custody(summary: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    custody = summary.get("custody")
    if not isinstance(custody, dict) or custody.get("stable") is not True:
        raise ValueError("evaluation shard lacks stable start/end custody")
    for position in ("start", "end"):
        git = custody.get(f"git_{position}")
        if not isinstance(git, dict):
            raise ValueError(f"evaluation shard lacks git_{position}")
        _expect(git, "commit", contract["code"]["git_commit"], f"shard git_{position}")
        _expect(git, "worktree_clean", True, f"shard git_{position}")
        _expect(
            custody,
            f"evaluator_file_sha256_{position}",
            contract["code"]["evaluator_file_sha256"],
            "shard custody",
        )
        if contract.get("evaluation_plan") is not None:
            _expect(
                custody,
                f"evaluation_plan_{position}",
                contract["evaluation_plan"],
                "shard custody",
            )
        _expect(
            custody,
            f"packages_{position}",
            contract["code"]["packages"],
            "shard custody",
        )
        if contract["code"].get("environment_contract") is not None:
            _expect(
                custody,
                f"environment_contract_{position}",
                contract["code"]["environment_contract"],
                "shard custody",
            )
            _expect(
                custody,
                f"stable_environment_{position}",
                True,
                "shard custody",
            )
        _expect(
            custody,
            f"task_file_sha256_{position}",
            contract["task_file_sha256"],
            "shard custody",
        )
        _expect(
            custody,
            f"adapter_tree_sha256_{position}",
            contract["adapter_tree_sha256"],
            "shard custody",
        )


def validate_post_promotion_companion(
    summary_path: Path,
    summary: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    producer: str,
    expected_state: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Recompute the atomic authorization companion for one v2 artifact."""

    if contract.get("contract") == LEGACY_EVALUATION_CONTRACT:
        return None
    output_dir = Path(summary_path).parent
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise ValueError("published evaluation output must be a regular directory")
    expected_files = {"summary.json", "samples.jsonl"}
    children = list(output_dir.iterdir())
    if {path.name for path in children} != expected_files or any(
        path.is_symlink() or not path.is_file() for path in children
    ):
        raise ValueError(
            "published evaluation output must contain exactly summary.json and samples.jsonl"
        )
    companion_path = post_promotion_custody_path(output_dir)
    companion = _json_object(companion_path, "post-promotion custody companion")
    summary_file = output_dir / "summary.json"
    samples_file = output_dir / "samples.jsonl"
    tree_hash = sha256_tree(output_dir)
    stable_state = dict(expected_state)
    expected = {
        "schema_version": POST_PROMOTION_CUSTODY_SCHEMA_VERSION,
        "custody_kind": f"opd_math_{producer}_post_promotion_v2",
        "artifact_kind": summary.get("artifact_kind"),
        "evaluation_contract": contract.get("contract"),
        "evaluation_contract_sha256": canonical_sha256(contract),
        "output_dir": str(output_dir.resolve()),
        "tree_hash_algorithm": POST_PROMOTION_TREE_ALGORITHM,
        "output_tree_sha256": tree_hash,
        "summary": str(summary_file.resolve()),
        "summary_sha256": sha256_file(summary_file),
        "samples": str(samples_file.resolve()),
        "samples_sha256": sha256_file(samples_file),
        "model": summary.get("model"),
        "model_revision": summary.get("model_revision"),
        "adapter_tree_sha256": summary.get("adapter_tree_sha256"),
        "task_file_sha256": summary.get("task_file_sha256"),
        "selected_record_ids_sha256": contract.get("eligible_record_ids_sha256"),
        "shard": summary.get("shard"),
        "merge": summary.get("merge"),
        "producer_custody_start": stable_state,
        "post_promotion_custody_a": stable_state,
        "post_promotion_custody_b": stable_state,
        "post_promotion_custody_c": stable_state,
        "stable_environment_after_promotion": True,
        "stable_final_artifact_hash": True,
        "publication_commit_point": True,
    }
    if companion != expected:
        raise ValueError("post-promotion custody companion does not match the artifact")
    return {
        "path": companion_path,
        "sha256": sha256_file(companion_path),
        "tree_sha256": tree_hash,
        "payload": companion,
    }


def validate_sample_rows(
    sample_rows: list[dict[str, Any]],
    *,
    task_rows: list[dict[str, Any]],
    record_start: int,
    samples_per_problem: int,
    task_hash: str,
    base_seed: int,
) -> dict[str, Any]:
    expected_count = len(task_rows) * samples_per_problem
    if len(sample_rows) != expected_count:
        raise ValueError(
            f"sample row count mismatch: expected={expected_count}, actual={len(sample_rows)}"
        )
    correct = parse_failed = verifier_errors = completion_tokens = prompt_tokens = 0
    generation_latency = 0.0
    cursor = 0
    for local_index, task_row in enumerate(task_rows):
        global_index = record_start + local_index
        record_id = str(task_row["record_id"])
        expected_seed = record_sampling_seed(base_seed, task_hash, global_index, record_id)
        group_prompt_tokens: int | None = None
        group_latency: float | None = None
        for expected_sample_idx in range(samples_per_problem):
            row = sample_rows[cursor]
            cursor += 1
            _expect(row, "schema_version", SAMPLE_SCHEMA_VERSION, "evaluation sample")
            _expect(row, "record_id", record_id, "evaluation sample")
            _expect(row, "global_record_index", global_index, "evaluation sample")
            _expect(row, "record_seed", expected_seed, "evaluation sample")
            _expect(row, "sample_idx", expected_sample_idx, "evaluation sample")
            _expect(row, "source", task_row.get("source"), "evaluation sample")
            _expect(row, "cluster_id", task_row.get("cluster_id"), "evaluation sample")
            completion = row.get("completion_text")
            if not isinstance(completion, str):
                raise ValueError("merge requires completion_text for reward recomputation")
            _expect(
                row,
                "completion_sha256",
                hashlib.sha256(completion.encode("utf-8")).hexdigest(),
                "evaluation sample",
            )
            verdict = verify_evaluation_completion(
                completion, str(task_row["solution"])
            )
            if verdict.get("reward") is None:
                raise RuntimeError(
                    f"verifier failure while merging record {record_id}: {verdict}"
                )
            reward = _finite_number(row.get("reward"), "evaluation sample reward")
            if reward not in {0.0, 1.0}:
                raise ValueError("evaluation sample reward must be binary")
            stored_status = row.get("reward_status")
            stored_error = stored_status == "verifier_error_zeroed"
            if stored_error:
                if reward != 0.0:
                    raise ValueError("evaluation verifier-error sample must map to zero")
                if (
                    row.get("verifier_error_policy")
                    != EVALUATION_VERIFIER_ERROR_POLICY
                    or not isinstance(row.get("verifier_error_type"), str)
                    or not isinstance(row.get("verifier_error"), str)
                    or row.get("verifier_stage")
                    not in {"prediction_parse", "symbolic_verify"}
                    or row.get("verifier_attempts")
                    != EVALUATION_VERIFIER_MAX_ATTEMPTS
                ):
                    raise ValueError(
                        "evaluation verifier-error sample lacks exact bounded policy custody"
                    )
                history = row.get("verifier_error_history")
                if (
                    not isinstance(history, list)
                    or len(history) != EVALUATION_VERIFIER_MAX_ATTEMPTS
                    or any(
                        not isinstance(item, dict)
                        or item.get("status")
                        not in {"verifier_error", "prediction_parser_error"}
                        or not isinstance(item.get("error_type"), str)
                        or not isinstance(item.get("error"), str)
                        for item in history
                    )
                ):
                    raise ValueError(
                        "evaluation verifier-error sample lacks its exact retry history"
                    )
                # A later bounded replay may resolve a load-sensitive timeout.
                # The stored sample remains an unknown in [0, 1], and all
                # authorizing gates use worst-case sensitivity over that set.
            else:
                if verdict.get("status") == "verifier_error_zeroed":
                    raise RuntimeError(
                        "bounded verifier replay remained inconclusive for a determinate "
                        f"evaluation sample in record {record_id}"
                    )
                recomputed_reward = float(verdict.get("reward"))
                if reward != recomputed_reward or stored_status != verdict.get("status"):
                    raise ValueError(
                        f"evaluation sample reward disagrees with recomputation for {record_id}"
                    )
            completion_count = row.get("completion_tokens")
            prompt_count = row.get("prompt_tokens")
            if type(completion_count) is not int or completion_count <= 0:
                raise ValueError("evaluation sample completion_tokens must be positive")
            if type(prompt_count) is not int or prompt_count <= 0:
                raise ValueError("evaluation sample prompt_tokens must be positive")
            latency = _finite_number(
                row.get("generation_batch_latency_seconds"),
                "evaluation sample generation latency",
                nonnegative=True,
            )
            if group_prompt_tokens is None:
                group_prompt_tokens = prompt_count
                group_latency = latency
            elif prompt_count != group_prompt_tokens or not math.isclose(
                latency, float(group_latency), rel_tol=0.0, abs_tol=0.0
            ):
                raise ValueError("samples from one record disagree on prompt tokens or latency")
            correct += int(reward)
            parse_failed += int(stored_status == "prediction_parse_failed")
            verifier_errors += int(stored_error)
            completion_tokens += completion_count
        prompt_tokens += int(group_prompt_tokens)
        generation_latency += float(group_latency)
    samples = len(sample_rows)
    verifier_error_fraction = verifier_errors / samples
    return {
        "records": len(task_rows),
        "samples": samples,
        "accuracy": correct / samples,
        "accuracy_excluding_verifier_errors": (
            correct / (samples - verifier_errors) if verifier_errors < samples else None
        ),
        "accuracy_if_all_verifier_errors_correct": (correct + verifier_errors) / samples,
        "prediction_parse_failure_fraction": parse_failed / samples,
        "verifier_error_policy": EVALUATION_VERIFIER_ERROR_POLICY,
        "verifier_error_samples": verifier_errors,
        "verifier_error_fraction": verifier_error_fraction,
        "maximum_verifier_error_fraction": MAX_EVALUATION_VERIFIER_ERROR_FRACTION,
        "unique_prompt_tokens": prompt_tokens,
        "expanded_prompt_tokens": prompt_tokens * samples_per_problem,
        "total_completion_tokens": completion_tokens,
        "total_generation_latency_seconds": generation_latency,
    }


def _validate_summary_metrics(summary: Mapping[str, Any], metrics: Mapping[str, Any]) -> None:
    for field in (
        "records",
        "samples",
        "verifier_error_policy",
        "verifier_error_samples",
        "maximum_verifier_error_fraction",
        "unique_prompt_tokens",
        "expanded_prompt_tokens",
        "total_completion_tokens",
    ):
        _expect(summary, field, metrics[field], "evaluation summary")
    for field in (
        "accuracy",
        "accuracy_excluding_verifier_errors",
        "accuracy_if_all_verifier_errors_correct",
        "prediction_parse_failure_fraction",
        "verifier_error_fraction",
        "total_generation_latency_seconds",
    ):
        if metrics[field] is None:
            _expect(summary, field, None, "evaluation summary")
        else:
            _same_float(summary.get(field), float(metrics[field]), f"evaluation summary {field}")
    peak = summary.get("peak_cuda_memory_bytes")
    if peak is not None and (type(peak) is not int or peak < 0):
        raise ValueError("evaluation summary has invalid peak_cuda_memory_bytes")


def validate_shard_artifact(
    summary_path: Path,
    *,
    task_file: Path,
    task_rows: list[dict[str, Any]] | None = None,
    task_hash: str | None = None,
    expected_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    raw_summary_path = Path(summary_path).expanduser()
    if raw_summary_path.is_symlink() or raw_summary_path.parent.is_symlink():
        raise ValueError("shard summary or output directory may not be a symlink")
    summary_path = raw_summary_path.resolve()
    if summary_path.name != "summary.json":
        raise ValueError("shard summary must use the canonical summary.json filename")
    summary = _json_object(summary_path, "evaluation shard summary")
    _expect(summary, "schema_version", SUMMARY_SCHEMA_VERSION, "evaluation shard")
    _expect(summary, "artifact_kind", EVALUATION_SHARD_KIND, "evaluation shard")
    if task_rows is None or task_hash is None:
        task_rows, task_hash = _checked_task_rows(task_file)
    task_file = Path(task_file).resolve()
    contract = _validate_contract(
        summary.get("evaluation_contract"),
        task_file=task_file,
        task_rows=task_rows,
        task_hash=task_hash,
    )
    _expect(
        summary,
        "evaluation_contract_sha256",
        canonical_sha256(contract),
        "evaluation shard",
    )
    if expected_contract is not None and dict(contract) != dict(expected_contract):
        raise ValueError("evaluation shards do not share one exact evaluation contract")
    _validate_shard_custody(summary, contract)

    for field in (
        "model",
        "model_revision",
        "adapter",
        "adapter_tree_sha256",
        "task_file",
        "task_file_sha256",
        "tokenizer_contract_sha256",
        "decoding",
        "record_seed_contract",
        "samples_per_problem",
        "evaluation_plan",
    ):
        contract_field = "record_seed_contract" if field == "record_seed_contract" else field
        _expect(summary, field, contract.get(contract_field), "evaluation shard")
    plan_launch_validation = summary.get("plan_launch_validation")
    if contract.get("evaluation_plan") is None:
        if plan_launch_validation is not None:
            raise ValueError("unplanned evaluation shard has launch-plan metadata")
    elif plan_launch_validation != {
        "schema_version": 1,
        "validation_kind": "opd_math_o_primary_plan_launch_validation_v1",
        "phase": "shard",
        "array_spec_source": "predeclared_OPD_MATH_EVAL_ARRAY_SPEC_v1",
        "declared_array_spec": contract["evaluation_plan"]["array_spec"],
        "declared_slurm_array_argument": contract["evaluation_plan"][
            "slurm_array_argument"
        ],
        "array_task_count": contract["evaluation_plan"]["shard_count"],
        "array_task_min": 0,
        "array_task_max": contract["evaluation_plan"]["shard_count"] - 1,
        "validated": True,
    }:
        raise ValueError("planned evaluation shard lacks exact launch validation")
    _expect(summary, "eligible_records", contract["eligible_records"], "evaluation shard")
    expected_code = {
        "git": {"commit": contract["code"]["git_commit"], "worktree_clean": True},
        "evaluator_file_sha256": contract["code"]["evaluator_file_sha256"],
        "packages": contract["code"]["packages"],
    }
    if contract["code"].get("environment_contract") is not None:
        expected_code["environment_contract"] = contract["code"]["environment_contract"]
    _expect(summary, "code", expected_code, "evaluation shard")
    _expect(summary, "completion_text_in_samples", True, "evaluation shard")
    expected_producer_state = {
        "git": expected_code["git"],
        "evaluator_file_sha256": contract["code"]["evaluator_file_sha256"],
        "packages": contract["code"]["packages"],
        "task_file": contract["task_file"],
        "task_file_sha256": contract["task_file_sha256"],
        "adapter": contract["adapter"],
        "adapter_tree_sha256": contract["adapter_tree_sha256"],
    }
    if contract["code"].get("environment_contract") is not None:
        expected_producer_state.update(
            {
                "environment_contract": contract["code"]["environment_contract"],
                "stable_environment": True,
            }
        )
    if contract.get("evaluation_plan") is not None:
        expected_producer_state["evaluation_plan"] = contract["evaluation_plan"]
    companion = validate_post_promotion_companion(
        summary_path,
        summary,
        contract,
        producer="evaluation_shard",
        expected_state=expected_producer_state,
    )

    shard = summary.get("shard")
    if not isinstance(shard, dict):
        raise ValueError("evaluation shard lacks shard metadata")
    shard_count = contract["shard"]["shard_count"]
    shard_index = shard.get("shard_index")
    if type(shard_index) is not int:
        raise ValueError("evaluation shard has an invalid shard index")
    record_start, record_stop = balanced_shard_bounds(
        contract["eligible_records"], shard_count, shard_index
    )
    expected_shard = {
        "strategy": SHARD_STRATEGY,
        "shard_count": shard_count,
        "shard_index": shard_index,
        "global_records": contract["eligible_records"],
        "record_start": record_start,
        "record_stop": record_stop,
        "selected_record_ids_sha256": canonical_sha256(
            [row["record_id"] for row in task_rows[record_start:record_stop]]
        ),
    }
    if shard != expected_shard:
        raise ValueError("evaluation shard metadata does not equal its exact task slice")
    selected_rows = task_rows[record_start:record_stop]
    _expect(
        summary,
        "task_sources",
        sorted({str(row.get("source")) for row in selected_rows}),
        "evaluation shard",
    )
    _expect(
        summary,
        "task_roles",
        sorted({str(row.get("role")) for row in selected_rows}),
        "evaluation shard",
    )

    samples_path = _absolute(summary.get("samples_file"), summary_path, "samples_file")
    expected_samples_path = (summary_path.parent / "samples.jsonl").resolve()
    if samples_path != expected_samples_path:
        raise ValueError("evaluation shard samples are not the canonical sibling samples.jsonl")
    samples_hash = sha256_file(samples_path)
    _expect(summary, "samples_file_sha256", samples_hash, "evaluation shard")
    raw_samples = samples_path.read_bytes()
    if not raw_samples or not raw_samples.endswith(b"\n"):
        raise ValueError("evaluation shard samples must be non-empty newline-terminated JSONL")
    sample_rows = list(iter_jsonl(samples_path))
    metrics = validate_sample_rows(
        sample_rows,
        task_rows=selected_rows,
        record_start=record_start,
        samples_per_problem=contract["samples_per_problem"],
        task_hash=task_hash,
        base_seed=contract["record_seed_contract"]["base_seed"],
    )
    _validate_summary_metrics(summary, metrics)
    return {
        "summary": summary,
        "summary_path": summary_path,
        "summary_sha256": sha256_file(summary_path),
        "samples_path": samples_path,
        "samples_sha256": samples_hash,
        "samples_bytes": raw_samples,
        "sample_rows": sample_rows,
        "metrics": metrics,
        "contract": contract,
        "record_start": record_start,
        "record_stop": record_stop,
        "shard_index": shard_index,
        "post_promotion_companion": companion,
    }


def capture_merge_custody(
    task_file: Path,
    adapter: str | None,
    environment_contract: Mapping[str, Any] | None = None,
    evaluation_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    adapter_hash = None
    if adapter is not None:
        adapter_path = Path(adapter)
        if adapter_path.is_symlink() or not adapter_path.is_dir():
            raise ValueError("merge adapter is not a regular directory")
        adapter_hash = sha256_tree(adapter_path)
    stable_environment = evaluation_environment_contract_unchanged(environment_contract)
    if environment_contract is not None and not stable_environment:
        raise RuntimeError("merge train environment changed during execution")
    custody = {
        "git": git_identity(),
        "merger_file_sha256": sha256_file(MERGER_PATH),
        "evaluator_file_sha256": sha256_file(EVALUATOR_PATH),
        "packages": package_versions(),
        "task_file_sha256": sha256_file(task_file),
        "adapter_tree_sha256": adapter_hash,
    }
    if environment_contract is not None:
        custody.update(
            {
                "environment_contract": dict(environment_contract),
                "stable_environment": stable_environment,
            }
        )
    if evaluation_plan is not None:
        plan_path = Path(str(evaluation_plan.get("plan", ""))).expanduser()
        if (
            plan_path.is_symlink()
            or not plan_path.is_file()
            or str(plan_path.resolve()) != evaluation_plan.get("plan")
            or sha256_file(plan_path) != evaluation_plan.get("plan_file_sha256")
        ):
            raise RuntimeError("evaluation shard plan changed during merge")
        custody["evaluation_plan"] = dict(evaluation_plan)
    return custody


def require_clean_stable_merge_custody(
    start: Mapping[str, Any], end: Mapping[str, Any], contract: Mapping[str, Any]
) -> None:
    for state, label in ((start, "start"), (end, "end")):
        git = state.get("git")
        if not isinstance(git, dict):
            raise RuntimeError(f"merge lacks Git {label} custody")
        _expect(git, "commit", contract["code"]["git_commit"], f"merge Git {label}")
        _expect(git, "worktree_clean", True, f"merge Git {label}")
        _expect(
            state,
            "evaluator_file_sha256",
            contract["code"]["evaluator_file_sha256"],
            f"merge {label}",
        )
        if contract.get("evaluation_plan") is not None:
            _expect(
                state,
                "evaluation_plan",
                contract["evaluation_plan"],
                f"merge {label}",
            )
        _expect(state, "packages", contract["code"]["packages"], f"merge {label}")
        if contract["code"].get("environment_contract") is not None:
            _expect(
                state,
                "environment_contract",
                contract["code"]["environment_contract"],
                f"merge {label}",
            )
            _expect(state, "stable_environment", True, f"merge {label}")
        _expect(
            state,
            "task_file_sha256",
            contract["task_file_sha256"],
            f"merge {label}",
        )
        _expect(
            state,
            "adapter_tree_sha256",
            contract["adapter_tree_sha256"],
            f"merge {label}",
        )
    if dict(start) != dict(end):
        changed = sorted(
            key for key in set(start) | set(end) if start.get(key) != end.get(key)
        )
        raise RuntimeError(f"merge custody changed during execution: {changed}")


def merge_custody_manifest(
    start: Mapping[str, Any], end: Mapping[str, Any]
) -> dict[str, Any]:
    manifest = {
        "git_start": start["git"],
        "git_end": end["git"],
        "merger_file_sha256_start": start["merger_file_sha256"],
        "merger_file_sha256_end": end["merger_file_sha256"],
        "evaluator_file_sha256": end["evaluator_file_sha256"],
        "packages_start": start["packages"],
        "packages_end": end["packages"],
        "task_file_sha256_start": start["task_file_sha256"],
        "task_file_sha256_end": end["task_file_sha256"],
        "adapter_tree_sha256_start": start["adapter_tree_sha256"],
        "adapter_tree_sha256_end": end["adapter_tree_sha256"],
        "stable": True,
    }
    if start.get("environment_contract") is not None or end.get("environment_contract") is not None:
        manifest.update(
            {
                "environment_contract_start": start.get("environment_contract"),
                "environment_contract_end": end.get("environment_contract"),
                "stable_environment_start": start.get("stable_environment"),
                "stable_environment_end": end.get("stable_environment"),
            }
        )
    if start.get("evaluation_plan") is not None or end.get("evaluation_plan") is not None:
        manifest.update(
            {
                "evaluation_plan_start": start.get("evaluation_plan"),
                "evaluation_plan_end": end.get("evaluation_plan"),
            }
        )
    return manifest


def _expected_shard_paths(shard_root: Path, shard_count: int) -> list[Path]:
    shard_root = Path(shard_root).expanduser()
    if shard_root.is_symlink() or not shard_root.is_dir():
        raise ValueError(f"shard root must be a regular directory: {shard_root}")
    shard_root = shard_root.resolve()
    if type(shard_count) is not int or shard_count <= 0:
        raise ValueError("shard_count must be positive")
    expected_names = {f"shard_{index:05d}" for index in range(shard_count)}
    actual_names = {
        path.name
        for path in shard_root.iterdir()
        if path.name.startswith("shard_") and (path.is_dir() or path.is_symlink())
    }
    if actual_names != expected_names:
        raise ValueError(
            "shard root does not contain the exact completed shard set: "
            f"missing={sorted(expected_names - actual_names)}, "
            f"extra={sorted(actual_names - expected_names)}"
        )
    paths = [shard_root / f"shard_{index:05d}" for index in range(shard_count)]
    if any(path.is_symlink() or not path.is_dir() for path in paths):
        raise ValueError("completed shard directories must not be symlinks")
    return paths


def merge_shards(
    *,
    shard_root: Path,
    shard_count: int,
    task_file: Path,
    output_dir: Path,
    train_environment_root: Path | None = None,
    train_environment_freeze: Path | None = None,
    shard_plan: Path | None = None,
    plan_arm: str | None = None,
    array_spec: str | None = None,
    samples_per_problem: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    max_new_tokens: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    task_file = Path(task_file).expanduser().resolve()
    task_rows, task_hash = _checked_task_rows(task_file)
    shard_dirs = _expected_shard_paths(shard_root, shard_count)
    first_payload = _json_object(shard_dirs[0] / "summary.json", "first shard summary")
    contract = _validate_contract(
        first_payload.get("evaluation_contract"),
        task_file=task_file,
        task_rows=task_rows,
        task_hash=task_hash,
    )
    _expect(contract["shard"], "shard_count", shard_count, "merge contract")
    complete_o_gap = _is_complete_o_teacher_gap(
        task_rows[: contract["eligible_records"]],
        physical_record_count=len(task_rows),
    )
    supplied_plan_inputs = (
        shard_plan,
        plan_arm,
        array_spec,
        samples_per_problem,
        temperature,
        top_p,
        top_k,
        max_new_tokens,
        seed,
    )
    merge_plan_validation = None
    if complete_o_gap:
        if any(value is None for value in supplied_plan_inputs):
            raise ValueError(
                "complete O teacher_gap_dev merge requires explicit canonical plan custody"
            )
        try:
            from .plan_evaluation_shards import validate_launch_against_plan
        except ImportError:
            from plan_evaluation_shards import validate_launch_against_plan  # type: ignore
        adapter = contract.get("adapter")
        merge_plan_validation = validate_launch_against_plan(
            plan_path=Path(shard_plan),  # type: ignore[arg-type]
            arm=str(plan_arm),
            phase="merge",
            source="O",
            role="teacher_gap_dev",
            model=contract["model"],
            model_revision=contract["model_revision"],
            task_file=task_file,
            max_records=0,
            shard_count=shard_count,
            git_commit=contract["code"]["git_commit"],
            train_freeze=Path(train_environment_freeze),  # type: ignore[arg-type]
            adapter=None if adapter is None else Path(adapter),
            array_spec=str(array_spec),
            samples_per_problem=samples_per_problem,  # type: ignore[arg-type]
            temperature=temperature,  # type: ignore[arg-type]
            top_p=top_p,  # type: ignore[arg-type]
            top_k=top_k,  # type: ignore[arg-type]
            max_new_tokens=max_new_tokens,  # type: ignore[arg-type]
            seed=seed,  # type: ignore[arg-type]
        )
        if merge_plan_validation["plan_binding"] != contract.get("evaluation_plan"):
            raise ValueError(
                "merge plan/arm/geometry differs from the exact shard contract"
            )
    elif any(value is not None for value in supplied_plan_inputs):
        raise ValueError(
            "evaluation plan inputs are forbidden outside complete O teacher_gap_dev"
        )
    environment_contract = contract["code"].get("environment_contract")
    if environment_contract is None:
        if train_environment_root is not None or train_environment_freeze is not None:
            raise ValueError(
                "merge inputs specify an exact train environment but shards lack its custody"
            )
    else:
        live_git = git_identity()
        live_environment = validate_evaluation_environment_contract(
            argparse.Namespace(
                train_environment_root=train_environment_root,
                train_environment_freeze=train_environment_freeze,
            ),
            live_git,
            required=True,
        )
        if live_environment != environment_contract:
            raise ValueError(
                "merge live train environment differs from the exact shard contract"
            )
    custody_start = capture_merge_custody(
        task_file,
        contract.get("adapter"),
        environment_contract,
        contract.get("evaluation_plan"),
    )
    require_clean_stable_merge_custody(custody_start, custody_start, contract)

    validated = [
        validate_shard_artifact(
            directory / "summary.json",
            task_file=task_file,
            task_rows=task_rows,
            task_hash=task_hash,
            expected_contract=contract,
        )
        for directory in shard_dirs
    ]
    if [item["shard_index"] for item in validated] != list(range(shard_count)):
        raise ValueError("validated shard indices are incomplete or out of order")
    if validated[0]["record_start"] != 0:
        raise ValueError("merged shard coverage does not start at record zero")
    for left, right in zip(validated, validated[1:]):
        if left["record_stop"] != right["record_start"]:
            raise ValueError("merged shard coverage has a gap or overlap")
    if validated[-1]["record_stop"] != contract["eligible_records"]:
        raise ValueError("merged shard coverage does not reach the selected record boundary")

    final_output, partial_output = begin_transactional_directory(output_dir)
    merged_samples_path = partial_output / "samples.jsonl"
    with merged_samples_path.open("xb") as handle:
        for item in validated:
            handle.write(item["samples_bytes"])
        handle.flush()
        import os

        os.fsync(handle.fileno())
    merged_sample_rows = [row for item in validated for row in item["sample_rows"]]
    selected_rows = task_rows[: contract["eligible_records"]]
    metrics = validate_sample_rows(
        merged_sample_rows,
        task_rows=selected_rows,
        record_start=0,
        samples_per_problem=contract["samples_per_problem"],
        task_hash=task_hash,
        base_seed=contract["record_seed_contract"]["base_seed"],
    )
    if metrics["verifier_error_fraction"] > MAX_EVALUATION_VERIFIER_ERROR_FRACTION:
        raise RuntimeError(
            "merged evaluation exceeds the predeclared candidate verifier-error cap: "
            f"observed={metrics['verifier_error_fraction']}, "
            f"maximum={MAX_EVALUATION_VERIFIER_ERROR_FRACTION}"
        )
    custody_end = capture_merge_custody(
        task_file,
        contract.get("adapter"),
        environment_contract,
        contract.get("evaluation_plan"),
    )
    require_clean_stable_merge_custody(custody_start, custody_end, contract)

    shard_bindings = [
        {
            "shard_index": item["shard_index"],
            "summary": str(item["summary_path"]),
            "summary_sha256": item["summary_sha256"],
            "samples": str(item["samples_path"]),
            "samples_sha256": item["samples_sha256"],
            "record_start": item["record_start"],
            "record_stop": item["record_stop"],
            "selected_record_ids_sha256": item["summary"]["shard"][
                "selected_record_ids_sha256"
            ],
            **(
                {}
                if item["post_promotion_companion"] is None
                else {
                    "post_promotion_custody": str(
                        item["post_promotion_companion"]["path"]
                    ),
                    "post_promotion_custody_sha256": item[
                        "post_promotion_companion"
                    ]["sha256"],
                    "output_tree_sha256": item["post_promotion_companion"][
                        "tree_sha256"
                    ],
                }
            ),
        }
        for item in validated
    ]
    peak_values = [
        item["summary"].get("peak_cuda_memory_bytes")
        for item in validated
        if item["summary"].get("peak_cuda_memory_bytes") is not None
    ]
    summary_code = {
        "git": {
            "commit": contract["code"]["git_commit"],
            "worktree_clean": True,
        },
        "evaluator_file_sha256": contract["code"]["evaluator_file_sha256"],
        "packages": contract["code"]["packages"],
    }
    if environment_contract is not None:
        summary_code["environment_contract"] = environment_contract
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "artifact_kind": EVALUATION_MERGED_KIND,
        "evaluation_contract": contract,
        "evaluation_contract_sha256": canonical_sha256(contract),
        **(
            {}
            if merge_plan_validation is None
            else {
                "evaluation_plan": contract["evaluation_plan"],
                "plan_merge_validation": merge_plan_validation[
                    "launch_validation"
                ],
            }
        ),
        "model": contract["model"],
        "model_revision": contract["model_revision"],
        "code": summary_code,
        "merge_custody": merge_custody_manifest(custody_start, custody_end),
        "tokenizer_contract_sha256": contract["tokenizer_contract_sha256"],
        "adapter": contract["adapter"],
        "adapter_tree_sha256": contract["adapter_tree_sha256"],
        "task_file": contract["task_file"],
        "task_file_sha256": contract["task_file_sha256"],
        "records": metrics["records"],
        "eligible_records": contract["eligible_records"],
        "task_sources": contract["task_sources"],
        "task_roles": contract["task_roles"],
        "samples_per_problem": contract["samples_per_problem"],
        "samples": metrics["samples"],
        "accuracy": metrics["accuracy"],
        "accuracy_excluding_verifier_errors": metrics[
            "accuracy_excluding_verifier_errors"
        ],
        "accuracy_if_all_verifier_errors_correct": metrics[
            "accuracy_if_all_verifier_errors_correct"
        ],
        "prediction_parse_failure_fraction": metrics[
            "prediction_parse_failure_fraction"
        ],
        "verifier_error_policy": metrics["verifier_error_policy"],
        "verifier_error_samples": metrics["verifier_error_samples"],
        "verifier_error_fraction": metrics["verifier_error_fraction"],
        "maximum_verifier_error_fraction": metrics[
            "maximum_verifier_error_fraction"
        ],
        "unique_prompt_tokens": metrics["unique_prompt_tokens"],
        "expanded_prompt_tokens": metrics["expanded_prompt_tokens"],
        "total_completion_tokens": metrics["total_completion_tokens"],
        "total_generation_latency_seconds": metrics[
            "total_generation_latency_seconds"
        ],
        "peak_cuda_memory_bytes": max(peak_values) if peak_values else None,
        "decoding": contract["decoding"],
        "record_seed_contract": contract["record_seed_contract"],
        "completion_text_in_samples": True,
        "merge": {
            "strategy": MERGE_STRATEGY,
            "shard_count": shard_count,
            "global_records": contract["eligible_records"],
            "selected_record_ids_sha256": contract[
                "eligible_record_ids_sha256"
            ],
            "shards": shard_bindings,
        },
        "samples_file": "samples.jsonl",
        "samples_file_sha256": sha256_file(merged_samples_path),
    }
    _validate_summary_metrics(summary, metrics)
    write_text_fsync(
        partial_output / "summary.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    custody_before_promotion = capture_merge_custody(
        task_file,
        contract.get("adapter"),
        environment_contract,
        contract.get("evaluation_plan"),
    )
    require_clean_stable_merge_custody(
        custody_start, custody_before_promotion, contract
    )
    if contract.get("contract") == EVALUATION_CONTRACT:
        publish_transactional_artifact(
            partial_output,
            final_output,
            summary=summary,
            producer="evaluation_merge",
            custody_start=custody_start,
            capture_custody=lambda: capture_merge_custody(
                task_file,
                contract.get("adapter"),
                environment_contract,
                contract.get("evaluation_plan"),
            ),
            require_stable_custody=lambda start, end: require_clean_stable_merge_custody(
                start, end, contract
            ),
        )
    else:
        promote_transactional_directory(partial_output, final_output)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--task-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-environment-root", type=Path, required=True)
    parser.add_argument("--train-environment-freeze", type=Path, required=True)
    parser.add_argument("--shard-plan", type=Path)
    parser.add_argument("--plan-arm", choices=("base", "trained"))
    parser.add_argument("--array-spec")
    parser.add_argument("--samples-per-problem", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = merge_shards(
        shard_root=args.shard_root,
        shard_count=args.shard_count,
        task_file=args.task_file,
        output_dir=args.output_dir,
        train_environment_root=args.train_environment_root,
        train_environment_freeze=args.train_environment_freeze,
        shard_plan=args.shard_plan,
        plan_arm=args.plan_arm,
        array_spec=args.array_spec,
        samples_per_problem=args.samples_per_problem,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
