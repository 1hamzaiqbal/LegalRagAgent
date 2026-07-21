#!/usr/bin/env python3
"""Materialize the exact prompt-plan prefix for the pinned upstream veRL arm."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

try:
    from .objective_family_inputs import (
        canonical_json_sha256,
        sha256_file,
        validate_prompt_plan,
    )
except ImportError:
    from objective_family_inputs import (  # type: ignore
        canonical_json_sha256,
        sha256_file,
        validate_prompt_plan,
    )


DATASET_ID = "opd_math_objective_family_verl_dataset_v1"


def _rows(path: Path) -> list[dict[str, Any]]:
    values = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"task row {number} is not an object")
            values.append(value)
    return values


def build_dataset(
    *,
    task_file: Path,
    prepared_manifest: Path,
    prompt_plan: Path,
    source: str,
    seed: int,
    git_commit: str,
    diagnostic: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    task_file = task_file.resolve()
    prepared_manifest = prepared_manifest.resolve()
    prompt_plan = prompt_plan.resolve()
    rows = _rows(task_file)
    steps = 1 if diagnostic else 100
    prompt_contract, ordered = validate_prompt_plan(
        prompt_plan,
        rows=rows,
        source=source,
        seed=seed,
        task_file=task_file,
        prepared_manifest=prepared_manifest,
        git_commit=git_commit,
        steps=steps,
        diagnostic=diagnostic,
    )
    output_rows = []
    for position, row in enumerate(ordered):
        prompt = row.get("prompt")
        solution = row.get("solution")
        record_id = row.get("record_id")
        if (
            not isinstance(prompt, list)
            or not prompt
            or not isinstance(solution, str)
            or not solution.strip()
            or not isinstance(record_id, str)
            or not record_id
        ):
            raise ValueError(f"registered veRL row {position} lacks prompt/solution identity")
        output_rows.append(
            {
                "data_source": f"legalrag_opd_math_{source}",
                "prompt": prompt,
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "record_id": record_id,
                    "source": source,
                    "prompt_plan_position": position,
                    "prompt_sha256": canonical_json_sha256(prompt),
                },
            }
        )
    manifest = {
        "schema_version": 1,
        "dataset": DATASET_ID,
        "status": "fixed_upstream_input_not_launch_authorization",
        "scientific_launch_authorized": False,
        "git_commit": git_commit,
        "source": source,
        "seed": seed,
        "diagnostic": diagnostic,
        "optimizer_steps": steps,
        "rows": len(output_rows),
        "task_file": {"path": str(task_file), "sha256": sha256_file(task_file)},
        "prepared_manifest": {
            "path": str(prepared_manifest),
            "sha256": sha256_file(prepared_manifest),
        },
        "prompt_plan": prompt_contract,
        "ordered_record_ids_sha256": canonical_json_sha256(
            [row["extra_info"]["record_id"] for row in output_rows]
        ),
        "output_rows_sha256": canonical_json_sha256(output_rows),
    }
    return output_rows, manifest


def write_new(output: Path, manifest_path: Path, rows, manifest) -> None:
    output = output.resolve()
    manifest_path = manifest_path.resolve()
    for path in (output, manifest_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to overwrite veRL input: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = dict(manifest)
    manifest["output"] = {
        "path": str(output),
        "sha256": sha256_file(output),
        "rows": len(rows),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.chmod(output, 0o444)
    os.chmod(manifest_path, 0o444)


def validate_dataset(
    *,
    task_file: Path,
    prepared_manifest: Path,
    prompt_plan: Path,
    source: str,
    seed: int,
    git_commit: str,
    diagnostic: bool,
    output: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    expected_rows, expected_manifest = build_dataset(
        task_file=task_file,
        prepared_manifest=prepared_manifest,
        prompt_plan=prompt_plan,
        source=source,
        seed=seed,
        git_commit=git_commit,
        diagnostic=diagnostic,
    )
    output = output.resolve()
    manifest_path = manifest_path.resolve()
    if output.is_symlink() or not output.is_file() or manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("veRL input bindings must be regular files")
    actual_rows = []
    for line in output.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("veRL materialized row is not an object")
            actual_rows.append(value)
    if actual_rows != expected_rows:
        raise ValueError("veRL materialized prompt order or content drifted")
    expected_manifest = dict(expected_manifest)
    expected_manifest["output"] = {
        "path": str(output),
        "sha256": sha256_file(output),
        "rows": len(actual_rows),
    }
    actual_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if actual_manifest != expected_manifest:
        raise ValueError("veRL dataset manifest differs from deterministic reconstruction")
    return {
        "path": str(output),
        "sha256": sha256_file(output),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "rows": len(actual_rows),
        "source": source,
        "seed": seed,
        "diagnostic": diagnostic,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "validate"):
        command = commands.add_parser(name)
        command.add_argument("--task-file", type=Path, required=True)
        command.add_argument("--prepared-manifest", type=Path, required=True)
        command.add_argument("--prompt-plan", type=Path, required=True)
        command.add_argument("--source", choices=("M", "O"), required=True)
        command.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
        command.add_argument("--git-commit", required=True)
        command.add_argument("--diagnostic", action="store_true")
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    common = {
        "task_file": args.task_file,
        "prepared_manifest": args.prepared_manifest,
        "prompt_plan": args.prompt_plan,
        "source": args.source,
        "seed": args.seed,
        "git_commit": args.git_commit,
        "diagnostic": args.diagnostic,
    }
    if args.command == "build":
        rows, manifest = build_dataset(**common)
        write_new(args.output, args.manifest, rows, manifest)
        result = {"output": str(args.output.resolve()), "rows": len(rows)}
    else:
        result = validate_dataset(
            **common, output=args.output, manifest_path=args.manifest
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
