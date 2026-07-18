#!/usr/bin/env python3
"""Merge a scientifically gated teacher adapter into a serving checkpoint."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any

try:
    from .quality_gates import (
        DEFAULT_TEACHER_MIN_RECORDS,
        SCHEMA_VERSION as GATE_SCHEMA_VERSION,
        TEACHER_GATE_TYPE,
        recompute_teacher_gate,
        sha256_file,
        sha256_tree,
    )
except ImportError:
    from quality_gates import (  # type: ignore
        DEFAULT_TEACHER_MIN_RECORDS,
        SCHEMA_VERSION as GATE_SCHEMA_VERSION,
        TEACHER_GATE_TYPE,
        recompute_teacher_gate,
        sha256_file,
        sha256_tree,
    )


PROVENANCE_SCHEMA = "opd_math_merged_teacher_v2"
PROVENANCE_FILENAME = "merge_provenance.json"
ROOT = Path(__file__).resolve().parents[2]


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"teacher-gap manifest is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"teacher-gap manifest must be a JSON object: {path}")
    return payload


def git_state() -> dict[str, str | bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain=v1"],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        ).stdout
        return {"commit": commit, "dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def clean_stable_git_custody(start: dict[str, Any], end: dict[str, Any]) -> bool:
    commit = start.get("commit")
    return bool(
        isinstance(commit, str)
        and re.fullmatch(r"[0-9a-f]{40}", commit)
        and start.get("dirty") is False
        and end.get("dirty") is False
        and end.get("commit") == commit
    )


def require_same_custody(before: dict[str, Any], after: dict[str, Any]) -> None:
    for field in ("manifest", "manifest_sha256", "adapter", "adapter_tree_sha256"):
        if after.get(field) != before.get(field):
            raise RuntimeError(f"teacher merge custody changed during merge: {field}")


def _require_bound_file(gate: dict[str, Any], path_key: str, hash_key: str) -> None:
    raw_path = gate.get(path_key)
    expected_hash = gate.get(hash_key)
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
        raise ValueError(f"teacher gap lacks an absolute {path_key} binding")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError(f"teacher gap lacks a valid {hash_key} binding")
    actual_hash = sha256_file(Path(raw_path))
    if actual_hash != expected_hash:
        raise ValueError(
            f"teacher gap bound artifact changed after evaluation: {path_key}={raw_path}"
        )


def validate_teacher_gate_for_merge(
    manifest_path: Path,
    *,
    base_model: str,
    base_revision: str,
    adapter: Path,
) -> dict[str, Any]:
    """Validate scientific gate custody without importing torch/PEFT/Transformers."""

    manifest_path = Path(manifest_path).resolve()
    gate = _read_json_object(manifest_path)
    if gate.get("schema_version") != GATE_SCHEMA_VERSION:
        raise ValueError(
            f"teacher gap has schema_version={gate.get('schema_version')!r}; "
            f"expected {GATE_SCHEMA_VERSION}"
        )
    if gate.get("gate") != TEACHER_GATE_TYPE:
        raise ValueError(
            f"merge requires gate={TEACHER_GATE_TYPE!r}; got {gate.get('gate')!r}"
        )
    if gate.get("gate_strength") != "scientific":
        raise ValueError("merge requires a scientific-strength teacher gap")
    if gate.get("passed") is not True or gate.get("authorizes_scientific_merge") is not True:
        raise ValueError("teacher gap did not authorize a scientific checkpoint merge")
    if gate.get("base_model") != base_model or gate.get("base_model_revision") != base_revision:
        raise ValueError(
            "teacher gap base identity does not match the requested merge: "
            f"gate={gate.get('base_model')}@{gate.get('base_model_revision')}, "
            f"requested={base_model}@{base_revision}"
        )

    minimum_records = gate.get("min_records")
    shared_records = gate.get("shared_records")
    if not isinstance(minimum_records, int) or minimum_records < DEFAULT_TEACHER_MIN_RECORDS:
        raise ValueError("teacher gap does not retain the scientific minimum-record contract")
    if not isinstance(shared_records, int) or shared_records < minimum_records:
        raise ValueError("teacher gap does not contain enough paired held-out records")
    try:
        paired_delta = float(gate["paired_delta"])
        min_delta = float(gate["min_delta"])
        bootstrap_ci = gate["bootstrap_95_ci"]
        lower_ci = float(bootstrap_ci[0])
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ValueError("teacher gap lacks valid paired-delta/bootstrap evidence") from exc
    if not all(math.isfinite(value) for value in (paired_delta, min_delta, lower_ci)):
        raise ValueError("teacher gap contains non-finite paired-delta/bootstrap evidence")
    if not paired_delta > min_delta:
        raise ValueError("teacher gap no longer satisfies its strict delta threshold")
    if not lower_ci > 0:
        raise ValueError("teacher gap lacks a positive bootstrap lower confidence bound")
    requirements = gate.get("requirements")
    if not isinstance(requirements, dict) or any(
        requirements.get(name) is not True
        for name in (
            "minimum_records_met",
            "strict_delta_met",
            "positive_bootstrap_lower_bound_met",
        )
    ):
        raise ValueError("teacher gap requirement attestations are incomplete")
    if gate.get("task_roles") != ["teacher_gap_dev"]:
        raise ValueError("teacher gap is not bound to the teacher_gap_dev role")
    if gate.get("task_sources") not in (["M"], ["O"]):
        raise ValueError("teacher gap is not bound to exactly one registered math source")

    adapter = Path(adapter).resolve()
    bound_adapter = gate.get("trained_adapter")
    if not isinstance(bound_adapter, str) or Path(bound_adapter).resolve() != adapter:
        raise ValueError(
            f"teacher gap adapter path mismatch: gate={bound_adapter!r}, requested={str(adapter)!r}"
        )
    adapter_hash = sha256_tree(adapter)
    if gate.get("trained_adapter_tree_sha256") != adapter_hash:
        raise ValueError("teacher adapter tree changed after the held-out evaluation")

    for path_key, hash_key in (
        ("task_file", "task_file_sha256"),
        ("base_summary", "base_summary_sha256"),
        ("trained_summary", "trained_summary_sha256"),
        ("base_samples", "base_samples_sha256"),
        ("trained_samples", "trained_samples_sha256"),
        ("prepared_manifest", "prepared_manifest_sha256"),
        ("source_manifest", "source_manifest_sha256"),
        ("teacher_run_manifest", "teacher_run_manifest_sha256"),
        ("teacher_training_task_file", "teacher_training_task_file_sha256"),
        ("teacher_training_plan", "teacher_training_plan_sha256"),
        ("teacher_trainer_state", "teacher_trainer_state_sha256"),
        ("teacher_trainer_log_history", "teacher_trainer_log_history_sha256"),
        ("teacher_train_metrics", "teacher_train_metrics_sha256"),
    ):
        _require_bound_file(gate, path_key, hash_key)

    recomputed = recompute_teacher_gate(gate)
    if recomputed != gate:
        changed = sorted(
            key for key in set(gate) | set(recomputed) if gate.get(key) != recomputed.get(key)
        )
        raise ValueError(
            "teacher gap does not equal deterministic recomputation from bound artifacts; "
            f"changed_fields={changed[:20]}"
        )

    return {
        "gate": gate,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "adapter": str(adapter),
        "adapter_tree_sha256": adapter_hash,
    }

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacher-gap-manifest", type=Path, required=True)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    code_state_start = git_state()
    if not clean_stable_git_custody(code_state_start, code_state_start):
        raise RuntimeError("scientific teacher merge requires a clean immutable Git start state")

    custody = validate_teacher_gate_for_merge(
        args.teacher_gap_manifest,
        base_model=args.base_model,
        base_revision=args.base_revision,
        adapter=args.adapter,
    )
    output_dir = args.output_dir.resolve()
    candidate_dir = output_dir.with_name(output_dir.name + ".candidate")
    if args.output_dir.is_symlink() or args.output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite teacher checkpoint output: {args.output_dir}"
        )
    if candidate_dir.is_symlink() or candidate_dir.exists():
        raise FileExistsError(f"refusing to reuse teacher merge candidate: {candidate_dir}")

    # Expensive and optional ML imports occur only after the pure custody check.
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        revision=args.base_revision,
        local_files_only=args.local_files_only,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    custody_before_adapter_load = validate_teacher_gate_for_merge(
        args.teacher_gap_manifest,
        base_model=args.base_model,
        base_revision=args.base_revision,
        adapter=args.adapter,
    )
    require_same_custody(custody, custody_before_adapter_load)
    model = PeftModel.from_pretrained(base, custody_before_adapter_load["adapter"])
    merged = model.merge_and_unload(safe_merge=True)
    candidate_dir.mkdir(parents=True, exist_ok=False)
    merged.save_pretrained(candidate_dir, safe_serialization=True, max_shard_size="5GB")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        revision=args.base_revision,
        local_files_only=args.local_files_only,
    )
    tokenizer.save_pretrained(candidate_dir)

    custody_after_merge = validate_teacher_gate_for_merge(
        args.teacher_gap_manifest,
        base_model=args.base_model,
        base_revision=args.base_revision,
        adapter=args.adapter,
    )
    require_same_custody(custody, custody_after_merge)
    code_state_after_merge = git_state()
    if not clean_stable_git_custody(code_state_start, code_state_after_merge):
        raise RuntimeError(
            "Git commit or cleanliness changed during teacher merge; candidate was not promoted"
        )

    checkpoint_hash = sha256_tree(
        candidate_dir, exclude_relative_paths=(PROVENANCE_FILENAME,)
    )
    gate = custody["gate"]
    provenance = {
        "schema_version": 1,
        "schema": PROVENANCE_SCHEMA,
        "base_model": args.base_model,
        "base_revision": args.base_revision,
        "adapter": custody["adapter"],
        "adapter_tree_sha256": custody["adapter_tree_sha256"],
        "teacher_gap_manifest": custody["manifest"],
        "teacher_gap_manifest_sha256": custody["manifest_sha256"],
        "prepared_manifest": gate["prepared_manifest"],
        "prepared_manifest_sha256": gate["prepared_manifest_sha256"],
        "teacher_run_manifest": gate["teacher_run_manifest"],
        "teacher_run_manifest_sha256": gate["teacher_run_manifest_sha256"],
        "teacher_training_plan": gate["teacher_training_plan"],
        "teacher_training_plan_sha256": gate["teacher_training_plan_sha256"],
        "teacher_training_plan_config_sha256": gate[
            "teacher_training_plan_config_sha256"
        ],
        "teacher_training_config_sha256": gate["teacher_training_config_sha256"],
        "teacher_training_packages": gate["teacher_training_packages"],
        "teacher_trainer_state": gate["teacher_trainer_state"],
        "teacher_trainer_state_sha256": gate["teacher_trainer_state_sha256"],
        "teacher_trainer_log_history": gate["teacher_trainer_log_history"],
        "teacher_trainer_log_history_sha256": gate[
            "teacher_trainer_log_history_sha256"
        ],
        "teacher_train_metrics": gate["teacher_train_metrics"],
        "teacher_train_metrics_sha256": gate["teacher_train_metrics_sha256"],
        "teacher_trainer_log_max_step": gate["teacher_trainer_log_max_step"],
        "source_manifest": gate["source_manifest"],
        "source_manifest_sha256": gate["source_manifest_sha256"],
        "task_file_sha256": gate["task_file_sha256"],
        "task_sources": gate["task_sources"],
        "task_roles": gate["task_roles"],
        "decoding": gate["decoding"],
        "merge_code": {
            "git_state_start": code_state_start,
            "git_state_after_merge": code_state_after_merge,
            "merger_file_sha256": sha256_file(Path(__file__)),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "peft")
            },
        },
        "output_checkpoint": str(output_dir),
        "output_checkpoint_tree_sha256": checkpoint_hash,
        "tree_hash_algorithm": "sha256_path_content_v1",
        "tree_hash_excludes": [PROVENANCE_FILENAME],
    }
    (candidate_dir / PROVENANCE_FILENAME).write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    code_state_before_promotion = git_state()
    if not clean_stable_git_custody(code_state_start, code_state_before_promotion):
        raise RuntimeError(
            "Git commit or cleanliness changed while writing merge provenance; "
            "candidate was not promoted"
        )
    provenance["merge_code"]["git_state_before_promotion"] = code_state_before_promotion
    candidate_dir.rename(output_dir)
    try:
        promoted_checkpoint_hash = sha256_tree(
            output_dir, exclude_relative_paths=(PROVENANCE_FILENAME,)
        )
    except (OSError, ValueError) as exc:
        rejected_dir = output_dir.with_name(output_dir.name + ".rejected_artifact_custody")
        output_dir.rename(rejected_dir)
        raise RuntimeError(
            "promoted teacher checkpoint could not be rehashed; "
            f"checkpoint moved to {rejected_dir}"
        ) from exc
    if promoted_checkpoint_hash != checkpoint_hash:
        rejected_dir = output_dir.with_name(output_dir.name + ".rejected_artifact_custody")
        output_dir.rename(rejected_dir)
        raise RuntimeError(
            "teacher checkpoint changed between candidate hashing and promotion; "
            f"checkpoint moved to {rejected_dir}"
        )
    code_state_after_promotion = git_state()
    if not clean_stable_git_custody(code_state_start, code_state_after_promotion):
        rejected_dir = output_dir.with_name(output_dir.name + ".rejected_code_custody")
        output_dir.rename(rejected_dir)
        raise RuntimeError(
            "Git commit or cleanliness changed during checkpoint promotion; "
            f"checkpoint moved to {rejected_dir}"
        )
    provenance["merge_code"].update(
        {
            "git_state_end": code_state_after_promotion,
            "clean_stable_code": True,
        }
    )
    provenance["status"] = "completed"
    (output_dir / PROVENANCE_FILENAME).write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    code_state_after_provenance = git_state()
    if not clean_stable_git_custody(code_state_start, code_state_after_provenance):
        rejected_dir = output_dir.with_name(output_dir.name + ".rejected_code_custody")
        output_dir.rename(rejected_dir)
        raise RuntimeError(
            "Git commit or cleanliness changed while finalizing merge provenance; "
            f"checkpoint moved to {rejected_dir}"
        )
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
