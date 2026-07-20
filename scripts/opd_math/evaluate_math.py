#!/usr/bin/env python3
"""Evaluate one immutable contiguous shard of a pinned math task.

Every record receives a deterministic seed derived from the task-file hash,
global record index, and record ID.  Consequently, changing shard geometry or
retrying a failed shard cannot change the random stream assigned to a record.
Outputs are written to a fresh partial directory and atomically promoted only
after code, task, adapter, package, and Git custody remain stable.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

try:
    from .data_contract import iter_jsonl
    from .math_reward import (
        EVALUATION_VERIFIER_ERROR_POLICY,
        EVALUATION_VERIFIER_MAX_ATTEMPTS,
        MAX_EVALUATION_VERIFIER_ERROR_FRACTION,
        verify_evaluation_completion,
    )
    from .quality_gates import EXPECTED_EVALUATION_PACKAGES, sha256_tree
    from .tokenizer_contract import canonical_sha256, tokenizer_fingerprint
    from .verify_environment import reverify_recorded_environment, verify_environment
except ImportError:
    from data_contract import iter_jsonl  # type: ignore
    from math_reward import (  # type: ignore
        EVALUATION_VERIFIER_ERROR_POLICY,
        EVALUATION_VERIFIER_MAX_ATTEMPTS,
        MAX_EVALUATION_VERIFIER_ERROR_FRACTION,
        verify_evaluation_completion,
    )
    from quality_gates import EXPECTED_EVALUATION_PACKAGES, sha256_tree  # type: ignore
    from tokenizer_contract import canonical_sha256, tokenizer_fingerprint  # type: ignore
    from verify_environment import (  # type: ignore
        reverify_recorded_environment,
        verify_environment,
    )


ROOT = Path(__file__).resolve().parents[2]
SUMMARY_SCHEMA_VERSION = 2
SAMPLE_SCHEMA_VERSION = 2
LEGACY_EVALUATION_CONTRACT = "opd_math_evaluation_contract_v1"
EVALUATION_CONTRACT = "opd_math_evaluation_contract_v2_exact_environment"
EVALUATION_SHARD_KIND = "opd_math_evaluation_shard_v1"
EVALUATION_MERGED_KIND = "opd_math_evaluation_merged_v1"
RECORD_SEED_STRATEGY = "task_hash_global_index_record_id_sha256_v1"
SHARD_STRATEGY = "contiguous_balanced_v1"
MERGE_STRATEGY = "ordered_contiguous_shards_v1"
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
ENVIRONMENT_VERIFIER = ROOT / "scripts" / "opd_math" / "verify_environment.py"
POST_PROMOTION_CUSTODY_SCHEMA_VERSION = 1
POST_PROMOTION_TREE_ALGORITHM = "opd-math-tree-v1"


def sha256_file(path: Path) -> str:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected a regular non-symlink file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_identity() -> dict[str, Any]:
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
        return {"commit": commit, "worktree_clean": not status.strip()}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "worktree_clean": False}


def package_versions() -> dict[str, str]:
    try:
        return {
            name: importlib.metadata.version(name)
            for name in EXPECTED_EVALUATION_PACKAGES
        }
    except importlib.metadata.PackageNotFoundError as exc:
        raise ValueError(f"required evaluation distribution is unavailable: {exc}") from exc


def validate_evaluation_environment_contract(
    args: argparse.Namespace, git: Mapping[str, Any], *, required: bool = True
) -> dict[str, Any] | None:
    """Bind one evaluator/merger process to the exact commit-specific train env."""

    environment_root = getattr(args, "train_environment_root", None)
    environment_freeze = getattr(args, "train_environment_freeze", None)
    if environment_root is None and environment_freeze is None:
        if required:
            raise ValueError(
                "evaluation requires --train-environment-root and "
                "--train-environment-freeze"
            )
        return None
    if environment_root is None or environment_freeze is None:
        raise ValueError(
            "evaluation environment custody requires both --train-environment-root and "
            "--train-environment-freeze"
        )

    commit = git.get("commit")
    if (
        git.get("worktree_clean") is not True
        or not isinstance(commit, str)
        or HEX40.fullmatch(commit) is None
    ):
        raise ValueError("evaluation environment custody requires one clean 40-hex Git commit")
    raw_root = Path(environment_root).expanduser()
    if raw_root.is_symlink() or not raw_root.is_dir():
        raise ValueError(
            f"evaluation environment root must be a regular non-symlink directory: {raw_root}"
        )
    root = raw_root.resolve(strict=True)
    freeze = Path(environment_freeze)
    if freeze.is_symlink() or not freeze.is_file():
        raise ValueError(
            f"evaluation environment freeze must be a regular non-symlink file: {freeze}"
        )
    freeze = freeze.resolve(strict=True)
    if (
        freeze.name != "train.freeze.txt"
        or freeze.parent.name != commit
        or freeze.parent.parent.name != "environment_freezes"
    ):
        raise ValueError(
            "evaluation environment freeze must be the commit-specific "
            f"environment_freezes/{commit}/train.freeze.txt"
        )
    runtime_packages = package_versions()
    if runtime_packages != EXPECTED_EVALUATION_PACKAGES:
        raise ValueError(
            "live evaluation packages differ from the pinned environment: "
            f"expected={EXPECTED_EVALUATION_PACKAGES}, actual={runtime_packages}"
        )
    verification = verify_environment(
        environment_root=root,
        commit_freeze=freeze,
        expected_commit=commit,
        freeze_kind="train",
    )
    commit_freeze = verification.get("commit_freeze")
    if not isinstance(commit_freeze, dict) or commit_freeze != {
        "path": str(freeze),
        "sha256": sha256_file(freeze),
        "byte_identical_to_requirements_freeze": True,
    }:
        raise ValueError("evaluation environment verification did not bind the selected freeze")
    return {
        "schema_version": 2,
        "git_commit": commit,
        "verifier": {
            "path": str(ENVIRONMENT_VERIFIER.resolve()),
            "sha256": sha256_file(ENVIRONMENT_VERIFIER),
        },
        "train_runtime_packages": runtime_packages,
        "train_environment_root": str(root),
        "train_freeze": {
            "path": str(freeze),
            "sha256": commit_freeze["sha256"],
            "required_packages": EXPECTED_EVALUATION_PACKAGES,
        },
        "train_verification": verification,
        "serve_freeze": None,
        "serve_verification": None,
    }


def evaluation_environment_contract_unchanged(
    contract: Mapping[str, Any] | None,
) -> bool:
    if contract is None:
        return True
    try:
        if contract.get("schema_version") != 2:
            return False
        if contract.get("verifier") != {
            "path": str(ENVIRONMENT_VERIFIER.resolve()),
            "sha256": sha256_file(ENVIRONMENT_VERIFIER),
        }:
            return False
        if contract.get("train_runtime_packages") != EXPECTED_EVALUATION_PACKAGES:
            return False
        if package_versions() != EXPECTED_EVALUATION_PACKAGES:
            return False
        if contract.get("serve_freeze") is not None or contract.get("serve_verification") is not None:
            return False
        freeze = contract.get("train_freeze")
        recorded = contract.get("train_verification")
        if not isinstance(freeze, dict) or not isinstance(recorded, dict):
            return False
        if contract.get("train_environment_root") != recorded.get("environment_root"):
            return False
        freeze_path = Path(str(freeze.get("path")))
        if (
            freeze_path.is_symlink()
            or not freeze_path.is_file()
            or sha256_file(freeze_path) != freeze.get("sha256")
            or freeze.get("required_packages") != EXPECTED_EVALUATION_PACKAGES
        ):
            return False
        if recorded.get("expected_commit") != contract.get("git_commit"):
            return False
        if recorded.get("freeze_kind") != "train":
            return False
        if recorded.get("commit_freeze") != {
            "path": str(freeze_path.resolve()),
            "sha256": freeze.get("sha256"),
            "byte_identical_to_requirements_freeze": True,
        }:
            return False
        if reverify_recorded_environment(recorded, in_process=True) != recorded:
            return False
    except (ImportError, OSError, TypeError, ValueError):
        return False
    return True


def record_sampling_seed(
    base_seed: int,
    task_file_sha256: str,
    global_record_index: int,
    record_id: str,
) -> int:
    """Derive a stable record seed that is independent of shard geometry."""

    if type(base_seed) is not int or base_seed < 0:
        raise ValueError("base seed must be a nonnegative integer")
    if not isinstance(task_file_sha256, str) or HEX64.fullmatch(task_file_sha256) is None:
        raise ValueError("task_file_sha256 must be a lowercase SHA-256 digest")
    if type(global_record_index) is not int or global_record_index < 0:
        raise ValueError("global_record_index must be a nonnegative integer")
    if not isinstance(record_id, str) or not record_id:
        raise ValueError("record_id must be a non-empty string")
    payload = {
        "strategy": RECORD_SEED_STRATEGY,
        "base_seed": base_seed,
        "task_file_sha256": task_file_sha256,
        "global_record_index": global_record_index,
        "record_id": record_id,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def balanced_shard_bounds(total: int, shard_count: int, shard_index: int) -> tuple[int, int]:
    """Return the exact contiguous slice assigned to one shard."""

    if type(total) is not int or total <= 0:
        raise ValueError("total must be a positive integer")
    if type(shard_count) is not int or shard_count <= 0 or shard_count > total:
        raise ValueError("shard count must be between one and the eligible record count")
    if type(shard_index) is not int or shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard index must be in [0, shard_count)")
    return total * shard_index // shard_count, total * (shard_index + 1) // shard_count


def _checked_record_ids(rows: list[dict[str, Any]], label: str) -> list[str]:
    record_ids: list[str] = []
    for index, row in enumerate(rows):
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"{label} row {index} lacks a stable record_id")
        record_ids.append(record_id)
    if len(record_ids) != len(set(record_ids)):
        raise ValueError(f"{label} contains duplicate record IDs")
    return record_ids


def _checked_adapter(path: Path | None) -> tuple[str | None, str | None]:
    if path is None:
        return None, None
    raw = Path(path).expanduser()
    if raw.is_symlink() or not raw.is_dir():
        raise ValueError(f"adapter must be a regular non-symlink directory: {raw}")
    resolved = raw.resolve()
    return str(resolved), sha256_tree(resolved)


def capture_evaluator_custody(
    task_file: Path,
    adapter: Path | None,
    environment_contract: Mapping[str, Any] | None = None,
    evaluation_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    adapter_path, adapter_hash = _checked_adapter(adapter)
    stable_environment = evaluation_environment_contract_unchanged(environment_contract)
    if environment_contract is not None and not stable_environment:
        raise RuntimeError("evaluation train environment changed during execution")
    custody = {
        "git": git_identity(),
        "evaluator_file_sha256": sha256_file(Path(__file__).resolve()),
        "packages": package_versions(),
        "task_file": str(Path(task_file).resolve()),
        "task_file_sha256": sha256_file(Path(task_file)),
        "adapter": adapter_path,
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
            raise RuntimeError("evaluation shard plan changed during execution")
        custody["evaluation_plan"] = dict(evaluation_plan)
    return custody


def require_clean_stable_custody(
    start: Mapping[str, Any], end: Mapping[str, Any], *, label: str
) -> None:
    start_git = start.get("git")
    end_git = end.get("git")
    if not isinstance(start_git, dict) or not isinstance(end_git, dict):
        raise RuntimeError(f"{label} lacks Git custody")
    commit = start_git.get("commit")
    if not isinstance(commit, str) or HEX40.fullmatch(commit) is None:
        raise RuntimeError(f"{label} lacks an immutable Git commit")
    if start_git.get("worktree_clean") is not True or end_git.get("worktree_clean") is not True:
        raise RuntimeError(f"{label} requires a clean Git worktree at start and end")
    if dict(start) != dict(end):
        changed = sorted(
            key for key in set(start) | set(end) if start.get(key) != end.get(key)
        )
        raise RuntimeError(f"{label} custody changed during execution: {changed}")


def evaluation_contract(
    *,
    model: str,
    model_revision: str,
    adapter: str | None,
    adapter_tree_sha256: str | None,
    task_file: str,
    task_file_sha256: str,
    eligible_record_ids: list[str],
    task_sources: list[str],
    task_roles: list[str],
    samples_per_problem: int,
    decoding: Mapping[str, Any],
    shard_count: int,
    tokenizer_contract_sha256: str,
    custody: Mapping[str, Any],
    environment_contract: Mapping[str, Any] | None = None,
    evaluation_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    git = custody.get("git")
    if not isinstance(git, dict):
        raise ValueError("evaluation contract lacks Git custody")
    code = {
        "git_commit": git["commit"],
        "evaluator_file_sha256": custody["evaluator_file_sha256"],
        "packages": custody["packages"],
    }
    if environment_contract is not None:
        code["environment_contract"] = dict(environment_contract)
    return {
        "schema_version": 1,
        "contract": (
            EVALUATION_CONTRACT
            if environment_contract is not None
            else LEGACY_EVALUATION_CONTRACT
        ),
        "model": model,
        "model_revision": model_revision,
        "adapter": adapter,
        "adapter_tree_sha256": adapter_tree_sha256,
        "task_file": task_file,
        "task_file_sha256": task_file_sha256,
        "eligible_records": len(eligible_record_ids),
        "eligible_record_ids_sha256": canonical_sha256(eligible_record_ids),
        "task_sources": task_sources,
        "task_roles": task_roles,
        "samples_per_problem": samples_per_problem,
        "reward_verifier": {
            "candidate_error_policy": EVALUATION_VERIFIER_ERROR_POLICY,
            "maximum_attempts": EVALUATION_VERIFIER_MAX_ATTEMPTS,
            "maximum_error_fraction": MAX_EVALUATION_VERIFIER_ERROR_FRACTION,
            "training_policy": "abort",
        },
        "decoding": dict(decoding),
        "record_seed_contract": {
            "strategy": RECORD_SEED_STRATEGY,
            "base_seed": decoding["seed"],
        },
        "shard": {"strategy": SHARD_STRATEGY, "shard_count": shard_count},
        **(
            {}
            if evaluation_plan is None
            else {"evaluation_plan": dict(evaluation_plan)}
        ),
        "tokenizer_contract_sha256": tokenizer_contract_sha256,
        "code": code,
    }


def custody_manifest(start: Mapping[str, Any], end: Mapping[str, Any]) -> dict[str, Any]:
    manifest = {
        "git_start": start["git"],
        "git_end": end["git"],
        "evaluator_file_sha256_start": start["evaluator_file_sha256"],
        "evaluator_file_sha256_end": end["evaluator_file_sha256"],
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


def _is_complete_o_teacher_gap(
    rows: list[dict[str, Any]], *, physical_record_count: int
) -> bool:
    """Identify the untruncated O teacher-gap surface independent of wrapper aliases."""

    return (
        bool(rows)
        and len(rows) == physical_record_count
        and {str(row.get("source")) for row in rows} == {"O"}
        and {str(row.get("role")) for row in rows} == {"teacher_gap_dev"}
    )


def validate_evaluation_plan(
    args: argparse.Namespace,
    *,
    task_file: Path,
    task_rows: list[dict[str, Any]],
    physical_record_count: int,
    adapter: Path | None,
    git: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Fail closed on the full O surface and reject plans everywhere else."""

    plan_path = getattr(args, "shard_plan", None)
    plan_arm = getattr(args, "plan_arm", None)
    array_spec = getattr(args, "array_spec", None)
    array_fields = (
        getattr(args, "array_task_count", None),
        getattr(args, "array_task_min", None),
        getattr(args, "array_task_max", None),
    )
    supplied = plan_path is not None or plan_arm is not None or array_spec is not None or any(
        value is not None for value in array_fields
    )
    required = _is_complete_o_teacher_gap(
        task_rows, physical_record_count=physical_record_count
    )
    if required and not supplied:
        raise ValueError("complete O teacher_gap_dev evaluation requires the canonical v2 plan")
    if not required and supplied:
        raise ValueError("an O primary shard plan may only be supplied to complete O teacher_gap_dev")
    if not required:
        return None, None
    if plan_path is None or plan_arm is None or array_spec is None or any(
        value is None for value in array_fields
    ):
        raise ValueError("complete O teacher_gap_dev evaluation lacks full plan/array custody")
    commit = git.get("commit")
    if not isinstance(commit, str):
        raise ValueError("complete O teacher_gap_dev evaluation lacks Git custody")
    try:
        from .plan_evaluation_shards import validate_launch_against_plan
    except ImportError:
        from plan_evaluation_shards import validate_launch_against_plan  # type: ignore

    validated = validate_launch_against_plan(
        plan_path=Path(plan_path),
        arm=str(plan_arm),
        phase="shard",
        source="O",
        role="teacher_gap_dev",
        model=args.model,
        model_revision=args.model_revision,
        task_file=task_file,
        max_records=args.max_records,
        shard_count=args.shard_count,
        git_commit=commit,
        train_freeze=Path(args.train_environment_freeze),
        adapter=adapter,
        array_spec=str(array_spec),
        samples_per_problem=args.samples_per_problem,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        array_task_count=array_fields[0],
        array_task_min=array_fields[1],
        array_task_max=array_fields[2],
    )
    return dict(validated["plan_binding"]), dict(validated["launch_validation"])


def begin_transactional_directory(final_path: Path) -> tuple[Path, Path]:
    """Create a fresh sibling partial directory for an absent final path."""

    raw = Path(final_path).expanduser()
    if raw.is_symlink() or raw.exists():
        raise FileExistsError(f"refusing to overwrite evaluation output: {raw}")
    parent = raw.parent.resolve()
    parent.mkdir(parents=True, exist_ok=True)
    final = parent / raw.name
    if final.is_symlink() or final.exists():
        raise FileExistsError(f"refusing to overwrite evaluation output: {final}")
    companion = post_promotion_custody_path(final)
    if companion.is_symlink() or companion.exists():
        raise FileExistsError(
            f"refusing to reuse output with an existing custody companion: {companion}"
        )
    partial = Path(tempfile.mkdtemp(prefix=f".{final.name}.partial.", dir=parent))
    return final, partial


def promote_transactional_directory(partial: Path, final: Path) -> None:
    """Atomically promote one completed directory without replacing a peer."""

    partial = Path(partial)
    final = Path(final)
    if partial.is_symlink() or not partial.is_dir():
        raise ValueError(f"partial output is not a regular directory: {partial}")
    lock = final.parent / f".{final.name}.promotion.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise FileExistsError(f"another process is promoting {final}") from exc
    try:
        if final.is_symlink() or final.exists():
            raise FileExistsError(f"refusing to replace completed output: {final}")
        partial.rename(final)
    finally:
        os.close(fd)
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def post_promotion_custody_path(output_dir: Path) -> Path:
    output = Path(output_dir)
    return output.parent / f"{output.name}.custody.json"


def _require_exact_published_files(output_dir: Path) -> None:
    output = Path(output_dir)
    if output.is_symlink() or not output.is_dir():
        raise ValueError(f"published output must be a regular directory: {output}")
    children = list(output.iterdir())
    expected_files = {"samples.jsonl", "summary.json"}
    if {path.name for path in children} != expected_files or any(
        path.is_symlink() or not path.is_file() for path in children
    ):
        raise ValueError(
            f"evaluation output must contain exactly regular files {sorted(expected_files)}"
        )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(Path(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def quarantine_published_artifact(output_dir: Path, identity: str) -> Path | None:
    """Revoke a companion first, then move a failed canonical output aside."""

    output = Path(output_dir)
    companion = post_promotion_custody_path(output)
    if not output.exists() and not companion.exists():
        return None
    rejected_root = output.parent / "rejected"
    rejected_root.mkdir(parents=True, exist_ok=True)
    stem = f"{output.name}_{identity[:12]}"
    suffix = 0
    while True:
        label = stem if suffix == 0 else f"{stem}_{suffix}"
        rejected_output = rejected_root / label
        rejected_companion = rejected_root / f"{label}.custody.json"
        if not rejected_output.exists() and not rejected_companion.exists():
            break
        suffix += 1
    if companion.exists() or companion.is_symlink():
        companion.rename(rejected_companion)
    if output.exists() or output.is_symlink():
        output.rename(rejected_output)
    _fsync_directory(rejected_root)
    _fsync_directory(output.parent)
    return rejected_output


def publish_transactional_artifact(
    partial: Path,
    final: Path,
    *,
    summary: Mapping[str, Any],
    producer: str,
    custody_start: Mapping[str, Any],
    capture_custody: Callable[[], Mapping[str, Any]],
    require_stable_custody: Callable[[Mapping[str, Any], Mapping[str, Any]], None],
) -> dict[str, Any]:
    """Publish output then atomically commit a post-promotion custody companion."""

    partial = Path(partial)
    final = Path(final)
    companion = post_promotion_custody_path(final)
    if partial.is_symlink() or not partial.is_dir():
        raise ValueError(f"partial output is not a regular directory: {partial}")
    if final.exists() or final.is_symlink() or companion.exists() or companion.is_symlink():
        raise FileExistsError(f"refusing to replace published evaluation artifact: {final}")
    _require_exact_published_files(partial)

    lock = final.parent / f".{final.name}.promotion.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    summary_identity = sha256_file(partial / "summary.json")
    companion_candidate = final.parent / f".{final.name}.custody.partial.{os.getpid()}"
    promoted = False
    try:
        if final.exists() or final.is_symlink() or companion.exists() or companion.is_symlink():
            raise FileExistsError(
                f"refusing to replace published evaluation artifact: {final}"
            )
        partial.rename(final)
        promoted = True
        _fsync_directory(final.parent)
        _require_exact_published_files(final)
        post_a = dict(capture_custody())
        require_stable_custody(custody_start, post_a)
        tree_a = sha256_tree(final)
        post_b = dict(capture_custody())
        require_stable_custody(custody_start, post_b)
        _require_exact_published_files(final)
        tree_b = sha256_tree(final)
        if post_a != post_b or tree_a != tree_b:
            raise RuntimeError("evaluation custody or final artifact changed after promotion")
        post_c = dict(capture_custody())
        require_stable_custody(custody_start, post_c)
        _require_exact_published_files(final)
        if post_c != post_a or sha256_tree(final) != tree_a:
            raise RuntimeError("evaluation changed before custody companion creation")
        summary_path = final / "summary.json"
        samples_path = final / "samples.jsonl"
        payload = {
            "schema_version": POST_PROMOTION_CUSTODY_SCHEMA_VERSION,
            "custody_kind": f"opd_math_{producer}_post_promotion_v2",
            "artifact_kind": summary.get("artifact_kind"),
            "evaluation_contract": summary.get("evaluation_contract", {}).get("contract"),
            "evaluation_contract_sha256": summary.get("evaluation_contract_sha256"),
            "output_dir": str(final.resolve()),
            "tree_hash_algorithm": POST_PROMOTION_TREE_ALGORITHM,
            "output_tree_sha256": tree_a,
            "summary": str(summary_path.resolve()),
            "summary_sha256": sha256_file(summary_path),
            "samples": str(samples_path.resolve()),
            "samples_sha256": sha256_file(samples_path),
            "model": summary.get("model"),
            "model_revision": summary.get("model_revision"),
            "adapter_tree_sha256": summary.get("adapter_tree_sha256"),
            "task_file_sha256": summary.get("task_file_sha256"),
            "selected_record_ids_sha256": summary.get(
                "evaluation_contract", {}
            ).get("eligible_record_ids_sha256"),
            "shard": summary.get("shard"),
            "merge": summary.get("merge"),
            "producer_custody_start": dict(custody_start),
            "post_promotion_custody_a": post_a,
            "post_promotion_custody_b": post_b,
            "post_promotion_custody_c": post_c,
            "stable_environment_after_promotion": True,
            "stable_final_artifact_hash": True,
            "publication_commit_point": True,
        }
        write_text_fsync(
            companion_candidate,
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
        )
        post_d = dict(capture_custody())
        require_stable_custody(custody_start, post_d)
        _require_exact_published_files(final)
        if post_d != post_a or sha256_tree(final) != tree_a:
            raise RuntimeError("evaluation changed before custody companion publication")
        os.link(companion_candidate, companion, follow_symlinks=False)
        companion_candidate.unlink()
        _fsync_directory(final.parent)
        return payload
    except Exception as original_error:
        try:
            if companion_candidate.exists() or companion_candidate.is_symlink():
                rejected_candidate = final.parent / (
                    f".{final.name}.custody.rejected.{summary_identity[:12]}"
                )
                if not rejected_candidate.exists() and not rejected_candidate.is_symlink():
                    companion_candidate.rename(rejected_candidate)
            if promoted:
                quarantine_published_artifact(final, summary_identity)
        except Exception as quarantine_error:
            raise RuntimeError(
                "publication failed and diagnostic quarantine also failed: "
                f"publication={original_error!r}; quarantine={quarantine_error!r}"
            ) from original_error
        raise
    finally:
        os.close(descriptor)
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def write_text_fsync(path: Path, content: str) -> None:
    with Path(path).open("x", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_records < 0:
        raise ValueError("--max-records must be nonnegative")
    if args.seed < 0:
        raise ValueError("--seed must be nonnegative")
    if args.samples_per_problem <= 0 or args.max_new_tokens <= 0:
        raise ValueError("sample count and completion length must be positive")
    if args.temperature <= 0 or not 0 < args.top_p <= 1 or args.top_k < 0:
        raise ValueError("invalid sampling contract")

    task_file = Path(args.task_file).expanduser()
    if task_file.is_symlink() or not task_file.is_file():
        raise ValueError(f"task file must be a regular non-symlink file: {task_file}")
    task_file = task_file.resolve()
    adapter = None if args.adapter is None else Path(args.adapter).expanduser().resolve()
    custody_start = capture_evaluator_custody(task_file, adapter)
    require_clean_stable_custody(custody_start, custody_start, label="evaluation start")
    environment_contract = validate_evaluation_environment_contract(
        args, custody_start["git"], required=True
    )
    custody_start = capture_evaluator_custody(
        task_file, adapter, environment_contract
    )
    require_clean_stable_custody(custody_start, custody_start, label="evaluation start")

    all_rows = list(iter_jsonl(task_file))
    physical_record_count = len(all_rows)
    if args.max_records > 0:
        all_rows = all_rows[: args.max_records]
    if not all_rows:
        raise ValueError("evaluation task file is empty")
    evaluation_plan, plan_launch_validation = validate_evaluation_plan(
        args,
        task_file=task_file,
        task_rows=all_rows,
        physical_record_count=physical_record_count,
        adapter=adapter,
        git=custody_start["git"],
    )
    if evaluation_plan is not None:
        custody_start = capture_evaluator_custody(
            task_file,
            adapter,
            environment_contract,
            evaluation_plan,
        )
        require_clean_stable_custody(
            custody_start, custody_start, label="planned evaluation start"
        )
    eligible_record_ids = _checked_record_ids(all_rows, "selected evaluation task")
    global_records = len(all_rows)
    record_start, record_stop = balanced_shard_bounds(
        global_records, args.shard_count, args.shard_index
    )
    rows = all_rows[record_start:record_stop]
    shard_record_ids = eligible_record_ids[record_start:record_stop]

    final_output, partial_output = begin_transactional_directory(Path(args.output_dir))

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.model_revision,
        local_files_only=args.local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.model_revision,
        local_files_only=args.local_files_only,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="sdpa",
    )
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    decoding = {
        "thinking": False,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
    }
    tokenizer_hash = canonical_sha256(tokenizer_fingerprint(tokenizer))
    task_sources = sorted({str(row.get("source")) for row in all_rows})
    task_roles = sorted({str(row.get("role")) for row in all_rows})
    contract = evaluation_contract(
        model=args.model,
        model_revision=args.model_revision,
        adapter=custody_start["adapter"],
        adapter_tree_sha256=custody_start["adapter_tree_sha256"],
        task_file=str(task_file),
        task_file_sha256=custody_start["task_file_sha256"],
        eligible_record_ids=eligible_record_ids,
        task_sources=task_sources,
        task_roles=task_roles,
        samples_per_problem=args.samples_per_problem,
        decoding=decoding,
        shard_count=args.shard_count,
        tokenizer_contract_sha256=tokenizer_hash,
        custody=custody_start,
        environment_contract=environment_contract,
        evaluation_plan=evaluation_plan,
    )
    contract_hash = canonical_sha256(contract)

    sample_path = partial_output / "samples.jsonl"
    correct = attempted = parse_failed = verifier_errors = 0
    unique_prompt_tokens = total_completion_tokens = 0
    total_generation_latency = 0.0
    with sample_path.open("x", encoding="utf-8") as handle, torch.inference_mode():
        for local_row_index, row in enumerate(rows):
            global_record_index = record_start + local_row_index
            messages = row.get("prompt")
            if not isinstance(messages, list):
                raise ValueError(f"row {global_record_index} lacks conversational prompt")
            record_id = shard_record_ids[local_row_index]
            record_seed = record_sampling_seed(
                args.seed,
                custody_start["task_file_sha256"],
                global_record_index,
                record_id,
            )
            random.seed(record_seed)
            torch.manual_seed(record_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(record_seed)
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to(device)
            prompt_width = int(inputs["input_ids"].shape[1])
            generation_started = time.perf_counter()
            generated = model.generate(
                **inputs,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=args.max_new_tokens,
                num_return_sequences=args.samples_per_problem,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            generation_latency = time.perf_counter() - generation_started
            total_generation_latency += generation_latency
            unique_prompt_tokens += prompt_width
            for sample_idx, output_ids in enumerate(generated):
                completion_ids = output_ids[prompt_width:].detach().cpu().tolist()
                if tokenizer.eos_token_id in completion_ids:
                    completion_ids = completion_ids[: completion_ids.index(tokenizer.eos_token_id) + 1]
                if not completion_ids:
                    raise RuntimeError(f"empty completion for record {record_id}")
                completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
                verdict = verify_evaluation_completion(completion, row["solution"])
                if verdict.get("reward") is None:
                    raise RuntimeError(f"evaluation verifier failure for {record_id}: {verdict}")
                reward = float(verdict["reward"])
                attempted += 1
                correct += int(reward)
                parse_failed += int(verdict["status"] == "prediction_parse_failed")
                verifier_errors += int(verdict["status"] == "verifier_error_zeroed")
                total_completion_tokens += len(completion_ids)
                result = {
                    "schema_version": SAMPLE_SCHEMA_VERSION,
                    "record_id": record_id,
                    "global_record_index": global_record_index,
                    "record_seed": record_seed,
                    "cluster_id": row.get("cluster_id"),
                    "source": row.get("source"),
                    "sample_idx": sample_idx,
                    "reward": reward,
                    "reward_status": verdict["status"],
                    "completion_tokens": len(completion_ids),
                    "prompt_tokens": prompt_width,
                    "generation_batch_latency_seconds": generation_latency,
                    "completion_sha256": hashlib.sha256(
                        completion.encode("utf-8")
                    ).hexdigest(),
                }
                if verdict["status"] == "verifier_error_zeroed":
                    result.update(
                        {
                            "verifier_error_type": verdict.get("verifier_error_type"),
                            "verifier_error": verdict.get("verifier_error"),
                            "verifier_stage": verdict.get("verifier_stage"),
                            "verifier_error_policy": verdict.get("policy"),
                            "verifier_attempts": verdict.get("verifier_attempts"),
                            "verifier_error_history": verdict.get(
                                "verifier_error_history"
                            ),
                        }
                    )
                if args.write_completions:
                    result["completion_text"] = completion
                handle.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    custody_end = capture_evaluator_custody(
        task_file, adapter, environment_contract, evaluation_plan
    )
    require_clean_stable_custody(
        custody_start, custody_end, label="evaluation start/end"
    )
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "artifact_kind": EVALUATION_SHARD_KIND,
        "evaluation_contract": contract,
        "evaluation_contract_sha256": contract_hash,
        **(
            {}
            if evaluation_plan is None
            else {
                "evaluation_plan": evaluation_plan,
                "plan_launch_validation": plan_launch_validation,
            }
        ),
        "model": args.model,
        "model_revision": args.model_revision,
        "code": {
            "git": custody_end["git"],
            "evaluator_file_sha256": custody_end["evaluator_file_sha256"],
            "packages": custody_end["packages"],
            "environment_contract": environment_contract,
        },
        "custody": custody_manifest(custody_start, custody_end),
        "tokenizer_contract_sha256": tokenizer_hash,
        "adapter": custody_end["adapter"],
        "adapter_tree_sha256": custody_end["adapter_tree_sha256"],
        "task_file": str(task_file),
        "task_file_sha256": custody_end["task_file_sha256"],
        "records": len(rows),
        "eligible_records": global_records,
        "task_sources": sorted({str(row.get("source")) for row in rows}),
        "task_roles": sorted({str(row.get("role")) for row in rows}),
        "samples_per_problem": args.samples_per_problem,
        "samples": attempted,
        "accuracy": correct / attempted,
        "accuracy_excluding_verifier_errors": (
            correct / (attempted - verifier_errors)
            if verifier_errors < attempted
            else None
        ),
        "accuracy_if_all_verifier_errors_correct": (correct + verifier_errors) / attempted,
        "prediction_parse_failure_fraction": parse_failed / attempted,
        "verifier_error_policy": EVALUATION_VERIFIER_ERROR_POLICY,
        "verifier_error_samples": verifier_errors,
        "verifier_error_fraction": verifier_errors / attempted,
        "maximum_verifier_error_fraction": MAX_EVALUATION_VERIFIER_ERROR_FRACTION,
        "unique_prompt_tokens": unique_prompt_tokens,
        "expanded_prompt_tokens": unique_prompt_tokens * args.samples_per_problem,
        "total_completion_tokens": total_completion_tokens,
        "total_generation_latency_seconds": total_generation_latency,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
        ),
        "decoding": decoding,
        "record_seed_contract": contract["record_seed_contract"],
        "shard": {
            "strategy": SHARD_STRATEGY,
            "shard_count": args.shard_count,
            "shard_index": args.shard_index,
            "global_records": global_records,
            "record_start": record_start,
            "record_stop": record_stop,
            "selected_record_ids_sha256": canonical_sha256(shard_record_ids),
        },
        "completion_text_in_samples": bool(args.write_completions),
        "samples_file": "samples.jsonl",
        "samples_file_sha256": sha256_file(sample_path),
    }
    write_text_fsync(
        partial_output / "summary.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    custody_before_promotion = capture_evaluator_custody(
        task_file, adapter, environment_contract, evaluation_plan
    )
    require_clean_stable_custody(
        custody_start,
        custody_before_promotion,
        label="evaluation pre-promotion",
    )
    publish_transactional_artifact(
        partial_output,
        final_output,
        summary=summary,
        producer="evaluation_shard",
        custody_start=custody_start,
        capture_custody=lambda: capture_evaluator_custody(
            task_file, adapter, environment_contract, evaluation_plan
        ),
        require_stable_custody=lambda start, end: require_clean_stable_custody(
            start, end, label="evaluation post-promotion"
        ),
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--task-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-environment-root", type=Path, required=True)
    parser.add_argument("--train-environment-freeze", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--samples-per-problem", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-plan", type=Path)
    parser.add_argument("--plan-arm", choices=("base", "trained"))
    parser.add_argument("--array-spec")
    parser.add_argument("--array-task-count", type=int)
    parser.add_argument("--array-task-min", type=int)
    parser.add_argument("--array-task-max", type=int)
    parser.add_argument("--write-completions", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    summary = evaluate(parse_args())
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
