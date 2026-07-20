"""Synthetic schema-v2 merged evaluation artifacts for gate tests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.opd_math import evaluate_math as evaluation
from scripts.opd_math import merge_evaluations as merger
from scripts.opd_math.data_contract import iter_jsonl
from scripts.opd_math.quality_gates import sha256_tree


def _synthetic_evaluation_environment(
    directory: Path,
    *,
    git_commit: str,
    packages: dict[str, str],
) -> dict:
    """Create the exact-environment identity expected by scientific eval gates."""

    if packages != evaluation.EXPECTED_EVALUATION_PACKAGES:
        raise ValueError(
            "exact synthetic evaluations require the pinned evaluation packages"
        )
    environment_root = (Path(directory) / "synthetic-evaluation-env").resolve()
    freeze = (
        Path(directory)
        / "environment_freezes"
        / git_commit
        / "train.freeze.txt"
    )
    freeze.parent.mkdir(parents=True, exist_ok=True)
    freeze_text = "synthetic exact evaluation environment\n"
    if not freeze.exists():
        freeze.write_text(freeze_text)
    freeze = freeze.resolve()
    freeze_hash = evaluation.sha256_file(freeze)
    verification = {
        "schema_version": 1,
        "schema": "opd_math_environment_verification_v1",
        "status": "passed",
        "environment_root": str(environment_root),
        "live_python": str(environment_root / "bin" / "python"),
        "expected_commit": git_commit,
        "freeze_kind": "train",
        "installed_distribution_count": len(packages),
        "installed_distribution_map_sha256": evaluation.canonical_sha256(packages),
        "requirements_freeze": {"path": str(freeze), "sha256": freeze_hash},
        "commit_freeze": {
            "path": str(freeze),
            "sha256": freeze_hash,
            "byte_identical_to_requirements_freeze": True,
        },
        "expected_executable": None,
    }
    return {
        "schema_version": 2,
        "git_commit": git_commit,
        "verifier": {
            "path": str(evaluation.ENVIRONMENT_VERIFIER.resolve()),
            "sha256": evaluation.sha256_file(evaluation.ENVIRONMENT_VERIFIER),
        },
        "train_runtime_packages": dict(packages),
        "train_environment_root": str(environment_root),
        "train_freeze": {
            "path": str(freeze),
            "sha256": freeze_hash,
            "required_packages": dict(packages),
        },
        "train_verification": verification,
        "serve_freeze": None,
        "serve_verification": None,
    }


def _write_post_promotion_companion(
    output_dir: Path,
    *,
    summary: dict,
    contract: dict,
    producer: str,
    producer_state: dict,
) -> dict:
    """Write the authorization companion emitted after atomic v2 promotion."""

    output_dir = Path(output_dir).resolve()
    summary_path = output_dir / "summary.json"
    samples_path = output_dir / "samples.jsonl"
    tree_hash = sha256_tree(output_dir)
    payload = {
        "schema_version": evaluation.POST_PROMOTION_CUSTODY_SCHEMA_VERSION,
        "custody_kind": f"opd_math_{producer}_post_promotion_v2",
        "artifact_kind": summary.get("artifact_kind"),
        "evaluation_contract": contract.get("contract"),
        "evaluation_contract_sha256": evaluation.canonical_sha256(contract),
        "output_dir": str(output_dir),
        "tree_hash_algorithm": evaluation.POST_PROMOTION_TREE_ALGORITHM,
        "output_tree_sha256": tree_hash,
        "summary": str(summary_path),
        "summary_sha256": evaluation.sha256_file(summary_path),
        "samples": str(samples_path),
        "samples_sha256": evaluation.sha256_file(samples_path),
        "model": summary.get("model"),
        "model_revision": summary.get("model_revision"),
        "adapter_tree_sha256": summary.get("adapter_tree_sha256"),
        "task_file_sha256": summary.get("task_file_sha256"),
        "selected_record_ids_sha256": contract.get(
            "eligible_record_ids_sha256"
        ),
        "shard": summary.get("shard"),
        "merge": summary.get("merge"),
        "producer_custody_start": dict(producer_state),
        "post_promotion_custody_a": dict(producer_state),
        "post_promotion_custody_b": dict(producer_state),
        "post_promotion_custody_c": dict(producer_state),
        "stable_environment_after_promotion": True,
        "stable_final_artifact_hash": True,
        "publication_commit_point": True,
    }
    companion_path = evaluation.post_promotion_custody_path(output_dir)
    companion_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return {
        "path": companion_path.resolve(),
        "sha256": evaluation.sha256_file(companion_path),
        "tree_sha256": tree_hash,
    }


def write_merged_evaluation(
    directory: Path,
    name: str,
    task_path: Path,
    rewards_by_record: dict[str, list[float]],
    *,
    model: str,
    revision: str,
    adapter: Path | None,
    packages: dict[str, str],
    git_commit: str = "e" * 40,
    tokenizer_contract_sha256: str = "d" * 64,
    decoding: dict,
    exact_environment: bool = False,
    environment_contract: dict | None = None,
) -> tuple[Path, Path]:
    """Write a one-shard artifact plus its independently bound merged view."""

    task_path = Path(task_path).resolve()
    task_rows = list(iter_jsonl(task_path))
    record_count = len(rewards_by_record)
    selected_rows = task_rows[:record_count]
    selected_ids = [str(row["record_id"]) for row in selected_rows]
    if set(selected_ids) != set(rewards_by_record) or len(selected_ids) != len(
        rewards_by_record
    ):
        raise ValueError("synthetic evaluation rewards must cover the exact task prefix")
    sample_counts = {len(rewards_by_record[record_id]) for record_id in selected_ids}
    if len(sample_counts) != 1 or next(iter(sample_counts)) <= 0:
        raise ValueError("synthetic evaluation requires one positive sample count")
    samples_per_problem = next(iter(sample_counts))
    task_hash = evaluation.sha256_file(task_path)
    adapter_path = None if adapter is None else str(Path(adapter).resolve())
    adapter_hash = None if adapter is None else sha256_tree(Path(adapter))
    evaluator_hash = evaluation.sha256_file(Path(evaluation.__file__))
    if environment_contract is not None and not exact_environment:
        raise ValueError("an explicit evaluation environment requires exact_environment=True")
    environment = None
    if exact_environment:
        environment = (
            json.loads(json.dumps(environment_contract))
            if environment_contract is not None
            else _synthetic_evaluation_environment(
                Path(directory), git_commit=git_commit, packages=packages
            )
        )
    code = {
        "git_commit": git_commit,
        "evaluator_file_sha256": evaluator_hash,
        "packages": dict(packages),
    }
    if environment is not None:
        code["environment_contract"] = environment
    contract = {
        "schema_version": 1,
        "contract": (
            evaluation.EVALUATION_CONTRACT
            if environment is not None
            else evaluation.LEGACY_EVALUATION_CONTRACT
        ),
        "model": model,
        "model_revision": revision,
        "adapter": adapter_path,
        "adapter_tree_sha256": adapter_hash,
        "task_file": str(task_path),
        "task_file_sha256": task_hash,
        "eligible_records": record_count,
        "eligible_record_ids_sha256": evaluation.canonical_sha256(selected_ids),
        "task_sources": sorted({str(row.get("source")) for row in selected_rows}),
        "task_roles": sorted({str(row.get("role")) for row in selected_rows}),
        "samples_per_problem": samples_per_problem,
        "decoding": dict(decoding),
        "record_seed_contract": {
            "strategy": evaluation.RECORD_SEED_STRATEGY,
            "base_seed": decoding["seed"],
        },
        "shard": {"strategy": evaluation.SHARD_STRATEGY, "shard_count": 1},
        "tokenizer_contract_sha256": tokenizer_contract_sha256,
        "reward_verifier": {
            "candidate_error_policy": evaluation.EVALUATION_VERIFIER_ERROR_POLICY,
            "maximum_attempts": evaluation.EVALUATION_VERIFIER_MAX_ATTEMPTS,
            "maximum_error_fraction": evaluation.MAX_EVALUATION_VERIFIER_ERROR_FRACTION,
            "training_policy": "abort",
        },
        "code": code,
    }
    contract_hash = evaluation.canonical_sha256(contract)
    root = Path(directory) / f"{name}-evaluation-v2"
    if root.exists() or root.is_symlink():
        raise FileExistsError(root)
    shard_dir = root / "shards" / "shard_00000"
    merged_dir = root / "merged"
    shard_dir.mkdir(parents=True)
    merged_dir.mkdir()

    sample_rows = []
    correct = 0
    for global_index, task_row in enumerate(selected_rows):
        record_id = str(task_row["record_id"])
        seed = evaluation.record_sampling_seed(
            decoding["seed"], task_hash, global_index, record_id
        )
        for sample_idx, raw_reward in enumerate(rewards_by_record[record_id]):
            reward = float(raw_reward)
            completion = (
                f"Final answer: {task_row['solution']}."
                if reward == 1.0
                else r"Final answer: \boxed{-999999}."
            )
            correct += int(reward)
            sample_rows.append(
                {
                    "schema_version": 2,
                    "record_id": record_id,
                    "global_record_index": global_index,
                    "record_seed": seed,
                    "cluster_id": task_row.get("cluster_id"),
                    "source": task_row.get("source"),
                    "sample_idx": sample_idx,
                    "reward": reward,
                    "reward_status": "correct" if reward == 1.0 else "incorrect",
                    "completion_tokens": 8,
                    "prompt_tokens": 12,
                    "generation_batch_latency_seconds": 0.1 + global_index,
                    "completion_text": completion,
                    "completion_sha256": hashlib.sha256(
                        completion.encode("utf-8")
                    ).hexdigest(),
                }
            )
    sample_text = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in sample_rows
    )
    shard_samples = shard_dir / "samples.jsonl"
    shard_samples.write_text(sample_text)
    total_samples = len(sample_rows)
    metrics = {
        "records": record_count,
        "samples": total_samples,
        "accuracy": correct / total_samples,
        "accuracy_excluding_verifier_errors": correct / total_samples,
        "accuracy_if_all_verifier_errors_correct": correct / total_samples,
        "prediction_parse_failure_fraction": 0.0,
        "verifier_error_policy": evaluation.EVALUATION_VERIFIER_ERROR_POLICY,
        "verifier_error_samples": 0,
        "verifier_error_fraction": 0.0,
        "maximum_verifier_error_fraction": evaluation.MAX_EVALUATION_VERIFIER_ERROR_FRACTION,
        "unique_prompt_tokens": 12 * record_count,
        "expanded_prompt_tokens": 12 * total_samples,
        "total_completion_tokens": 8 * total_samples,
        "total_generation_latency_seconds": sum(
            0.1 + index for index in range(record_count)
        ),
    }
    git = {"commit": git_commit, "worktree_clean": True}
    shard_summary = {
        "schema_version": 2,
        "artifact_kind": evaluation.EVALUATION_SHARD_KIND,
        "evaluation_contract": contract,
        "evaluation_contract_sha256": contract_hash,
        "model": model,
        "model_revision": revision,
        "code": {
            "git": git,
            "evaluator_file_sha256": evaluator_hash,
            "packages": dict(packages),
            **(
                {"environment_contract": environment}
                if environment is not None
                else {}
            ),
        },
        "custody": {
            "git_start": git,
            "git_end": git,
            "evaluator_file_sha256_start": evaluator_hash,
            "evaluator_file_sha256_end": evaluator_hash,
            "packages_start": dict(packages),
            "packages_end": dict(packages),
            "task_file_sha256_start": task_hash,
            "task_file_sha256_end": task_hash,
            "adapter_tree_sha256_start": adapter_hash,
            "adapter_tree_sha256_end": adapter_hash,
            **(
                {
                    "environment_contract_start": environment,
                    "environment_contract_end": environment,
                    "stable_environment_start": True,
                    "stable_environment_end": True,
                }
                if environment is not None
                else {}
            ),
            "stable": True,
        },
        "tokenizer_contract_sha256": tokenizer_contract_sha256,
        "adapter": adapter_path,
        "adapter_tree_sha256": adapter_hash,
        "task_file": str(task_path),
        "task_file_sha256": task_hash,
        "records": record_count,
        "eligible_records": record_count,
        "task_sources": contract["task_sources"],
        "task_roles": contract["task_roles"],
        "samples_per_problem": samples_per_problem,
        **metrics,
        "peak_cuda_memory_bytes": 1024,
        "decoding": dict(decoding),
        "record_seed_contract": contract["record_seed_contract"],
        "shard": {
            "strategy": evaluation.SHARD_STRATEGY,
            "shard_count": 1,
            "shard_index": 0,
            "global_records": record_count,
            "record_start": 0,
            "record_stop": record_count,
            "selected_record_ids_sha256": evaluation.canonical_sha256(selected_ids),
        },
        "completion_text_in_samples": True,
        "samples_file": "samples.jsonl",
        "samples_file_sha256": evaluation.sha256_file(shard_samples),
    }
    shard_summary_path = shard_dir / "summary.json"
    shard_summary_path.write_text(
        json.dumps(shard_summary, indent=2, sort_keys=True) + "\n"
    )
    shard_companion = None
    if environment is not None:
        evaluator_state = {
            "git": git,
            "evaluator_file_sha256": evaluator_hash,
            "packages": dict(packages),
            "task_file": str(task_path),
            "task_file_sha256": task_hash,
            "adapter": adapter_path,
            "adapter_tree_sha256": adapter_hash,
            "environment_contract": environment,
            "stable_environment": True,
        }
        shard_companion = _write_post_promotion_companion(
            shard_dir,
            summary=shard_summary,
            contract=contract,
            producer="evaluation_shard",
            producer_state=evaluator_state,
        )
    merged_samples = merged_dir / "samples.jsonl"
    merged_samples.write_text(sample_text)
    merger_hash = evaluation.sha256_file(Path(merger.__file__))
    merge_custody = {
        "git_start": git,
        "git_end": git,
        "merger_file_sha256_start": merger_hash,
        "merger_file_sha256_end": merger_hash,
        "evaluator_file_sha256": evaluator_hash,
        "packages_start": dict(packages),
        "packages_end": dict(packages),
        "task_file_sha256_start": task_hash,
        "task_file_sha256_end": task_hash,
        "adapter_tree_sha256_start": adapter_hash,
        "adapter_tree_sha256_end": adapter_hash,
        **(
            {
                "environment_contract_start": environment,
                "environment_contract_end": environment,
                "stable_environment_start": True,
                "stable_environment_end": True,
            }
            if environment is not None
            else {}
        ),
        "stable": True,
    }
    merged_summary = {
        "schema_version": 2,
        "artifact_kind": evaluation.EVALUATION_MERGED_KIND,
        "evaluation_contract": contract,
        "evaluation_contract_sha256": contract_hash,
        "model": model,
        "model_revision": revision,
        "code": {
            "git": git,
            "evaluator_file_sha256": evaluator_hash,
            "packages": dict(packages),
            **(
                {"environment_contract": environment}
                if environment is not None
                else {}
            ),
        },
        "merge_custody": merge_custody,
        "tokenizer_contract_sha256": tokenizer_contract_sha256,
        "adapter": adapter_path,
        "adapter_tree_sha256": adapter_hash,
        "task_file": str(task_path),
        "task_file_sha256": task_hash,
        "records": record_count,
        "eligible_records": record_count,
        "task_sources": contract["task_sources"],
        "task_roles": contract["task_roles"],
        "samples_per_problem": samples_per_problem,
        **metrics,
        "peak_cuda_memory_bytes": 1024,
        "decoding": dict(decoding),
        "record_seed_contract": contract["record_seed_contract"],
        "completion_text_in_samples": True,
        "merge": {
            "strategy": evaluation.MERGE_STRATEGY,
            "shard_count": 1,
            "global_records": record_count,
            "selected_record_ids_sha256": evaluation.canonical_sha256(selected_ids),
            "shards": [
                {
                    "shard_index": 0,
                    "summary": str(shard_summary_path.resolve()),
                    "summary_sha256": evaluation.sha256_file(shard_summary_path),
                    "samples": str(shard_samples.resolve()),
                    "samples_sha256": evaluation.sha256_file(shard_samples),
                    "record_start": 0,
                    "record_stop": record_count,
                    "selected_record_ids_sha256": evaluation.canonical_sha256(
                        selected_ids
                    ),
                    **(
                        {
                            "post_promotion_custody": str(
                                shard_companion["path"]
                            ),
                            "post_promotion_custody_sha256": shard_companion[
                                "sha256"
                            ],
                            "output_tree_sha256": shard_companion["tree_sha256"],
                        }
                        if shard_companion is not None
                        else {}
                    ),
                }
            ],
        },
        "samples_file": "samples.jsonl",
        "samples_file_sha256": evaluation.sha256_file(merged_samples),
    }
    merged_summary_path = merged_dir / "summary.json"
    merged_summary_path.write_text(
        json.dumps(merged_summary, indent=2, sort_keys=True) + "\n"
    )
    if environment is not None:
        merge_state = {
            "git": git,
            "merger_file_sha256": merger_hash,
            "evaluator_file_sha256": evaluator_hash,
            "packages": dict(packages),
            "task_file_sha256": task_hash,
            "adapter_tree_sha256": adapter_hash,
            "environment_contract": environment,
            "stable_environment": True,
        }
        _write_post_promotion_companion(
            merged_dir,
            summary=merged_summary,
            contract=contract,
            producer="evaluation_merge",
            producer_state=merge_state,
        )
    return merged_summary_path, merged_samples
