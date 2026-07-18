"""Synthetic schema-v2 merged evaluation artifacts for gate tests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.opd_math import evaluate_math as evaluation
from scripts.opd_math import merge_evaluations as merger
from scripts.opd_math.data_contract import iter_jsonl
from scripts.opd_math.quality_gates import sha256_tree


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
    code = {
        "git_commit": git_commit,
        "evaluator_file_sha256": evaluator_hash,
        "packages": dict(packages),
    }
    contract = {
        "schema_version": 1,
        "contract": evaluation.EVALUATION_CONTRACT,
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
        "prediction_parse_failure_fraction": 0.0,
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
    return merged_summary_path, merged_samples
