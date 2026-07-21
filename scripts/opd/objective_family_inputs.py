#!/usr/bin/env python3
"""Build and validate immutable prompt and initialized-adapter inputs.

These artifacts make the objective-family comparison differ only in its
registered loss.  They do not authorize a scientific launch; the later
experiment preregistration binds their exact paths and hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping

try:
    from .objective_registry import load_objective_registry
except ImportError:
    from objective_registry import load_objective_registry  # type: ignore


ROOT = Path(__file__).resolve().parents[2]
STUDENT_PLAN = ROOT / "configs/opd_math/objective_family_student_plan.json"
PROMPT_PLAN_ID = "opd_math_objective_family_prompt_sequence_v1"
INITIALIZATION_ID = "opd_math_objective_family_initial_adapter_v1"
EXPECTED_STUDENT = "Qwen/Qwen3-1.7B"
EXPECTED_STUDENT_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def sha256_tree(
    root: str | Path, *, exclude_relative_paths: tuple[str, ...] = ()
) -> str:
    root = Path(root).resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"adapter tree must be a regular directory: {root}")
    digest = hashlib.sha256()
    digest.update(b"opd-math-tree-v1\0")
    excluded = {Path(item).as_posix() for item in exclude_relative_paths}
    files = []
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError(f"adapter tree contains a symlink: {path}")
        if path.is_file() and relative not in excluded:
            files.append(path)
    files.sort(key=lambda path: path.relative_to(root).as_posix())
    if not files:
        raise ValueError(f"adapter tree is empty: {root}")
    for path in files:
        relative = path.relative_to(root).as_posix().encode()
        payload_hash = bytes.fromhex(sha256_file(path))
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(payload_hash)
    return digest.hexdigest()


def task_prompt_sha256(row: Mapping[str, Any]) -> str:
    prompt = row.get("prompt")
    if prompt is not None:
        if not isinstance(prompt, list):
            raise ValueError("conversational prompt must be a list")
        return canonical_json_sha256(prompt)
    prompt_text = row.get("prompt_text")
    if not isinstance(prompt_text, str) or not prompt_text:
        raise ValueError("task row lacks a stable prompt")
    return canonical_json_sha256(prompt_text)


def _clean_git_state() -> str:
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
    if HEX40.fullmatch(commit) is None or status.strip():
        raise ValueError("objective-family inputs require a clean immutable Git checkout")
    return commit


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(row)
    if not rows:
        raise ValueError(f"task role is empty: {path}")
    return rows


def _student_plan() -> dict[str, Any]:
    plan = json.loads(STUDENT_PLAN.read_text(encoding="utf-8"))
    if (
        not isinstance(plan, dict)
        or plan.get("plan_id") != "opd_math_objective_family_student_v1"
        or plan.get("scientific_launch_authorized") is not False
        or plan.get("allowed_seeds") != [0, 1, 2]
        or plan.get("common_fixed_config", {}).get("optimizer_steps") != 100
        or plan.get("common_fixed_config", {}).get("micro_prompts") != 1
        or plan.get("diagnostic_fixed_overrides") != {"optimizer_steps": 1}
        or plan.get("diagnostic_seed") != 0
    ):
        raise ValueError("objective-family student plan is not the supported fixed plan")
    return plan


def build_prompt_plan(
    *,
    rows: list[dict[str, Any]],
    source: str,
    seed: int,
    task_file: Path,
    prepared_manifest: Path,
    git_commit: str,
) -> dict[str, Any]:
    plan = _student_plan()
    if source not in {"M", "O"} or seed not in plan["allowed_seeds"]:
        raise ValueError("prompt-plan source or seed is not registered")
    expected_rows = [
        row
        for row in rows
        if row.get("source") == source and row.get("role") == "student_opd"
    ]
    if len(expected_rows) != len(rows):
        raise ValueError("prompt-plan task file mixes sources or roles")
    record_ids = [row.get("record_id") for row in rows]
    if any(not isinstance(value, str) or not value for value in record_ids):
        raise ValueError("prompt-plan task rows require stable record IDs")
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("prompt-plan task role has duplicate record IDs")
    optimizer_steps = int(plan["common_fixed_config"]["optimizer_steps"])
    if len(rows) < optimizer_steps:
        raise ValueError("prompt-plan task role is smaller than the fixed step count")
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    selected = shuffled[:optimizer_steps]
    sequence = [
        {
            "position": position,
            "record_id": row["record_id"],
            "prompt_sha256": task_prompt_sha256(row),
        }
        for position, row in enumerate(selected)
    ]
    registry = load_objective_registry()
    payload = {
        "schema_version": 1,
        "plan": PROMPT_PLAN_ID,
        "status": "fixed_input_not_launch_authorization",
        "scientific_launch_authorized": False,
        "git_commit": git_commit,
        "objective_registry_sha256": registry["sha256"],
        "student_training_plan_sha256": sha256_file(STUDENT_PLAN),
        "prepared_manifest": {
            "path": str(prepared_manifest.resolve()),
            "sha256": sha256_file(prepared_manifest),
        },
        "task_file": {
            "path": str(task_file.resolve()),
            "sha256": sha256_file(task_file),
            "rows": len(rows),
        },
        "source": source,
        "seed": seed,
        "optimizer_steps": optimizer_steps,
        "micro_prompts": 1,
        "sequence": sequence,
        "sequence_sha256": canonical_json_sha256(sequence),
    }
    return payload


def validate_prompt_plan(
    path: str | Path,
    *,
    rows: list[dict[str, Any]],
    source: str,
    seed: int,
    task_file: str | Path,
    prepared_manifest: str | Path,
    git_commit: str,
    steps: int,
    diagnostic: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw = Path(path)
    if raw.is_symlink() or not raw.is_file():
        raise ValueError("objective-family prompt plan must be a regular file")
    payload = json.loads(raw.read_text(encoding="utf-8"))
    expected_keys = {
        "schema_version",
        "plan",
        "status",
        "scientific_launch_authorized",
        "git_commit",
        "objective_registry_sha256",
        "student_training_plan_sha256",
        "prepared_manifest",
        "task_file",
        "source",
        "seed",
        "optimizer_steps",
        "micro_prompts",
        "sequence",
        "sequence_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ValueError("objective-family prompt plan schema drifted")
    registry = load_objective_registry()
    expected = {
        "schema_version": 1,
        "plan": PROMPT_PLAN_ID,
        "status": "fixed_input_not_launch_authorization",
        "scientific_launch_authorized": False,
        "git_commit": git_commit,
        "objective_registry_sha256": registry["sha256"],
        "student_training_plan_sha256": sha256_file(STUDENT_PLAN),
        "source": source,
        "seed": seed,
        "optimizer_steps": 100,
        "micro_prompts": 1,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(f"objective-family prompt plan {field} drifted")
    for field, selected_path in (
        ("prepared_manifest", Path(prepared_manifest)),
        ("task_file", Path(task_file)),
    ):
        binding = payload.get(field)
        if not isinstance(binding, dict):
            raise ValueError(f"objective-family prompt plan lacks {field} custody")
        resolved = selected_path.resolve()
        if binding.get("path") != str(resolved) or binding.get("sha256") != sha256_file(resolved):
            raise ValueError(f"objective-family prompt plan {field} custody drifted")
    if payload["task_file"].get("rows") != len(rows):
        raise ValueError("objective-family prompt plan task-row count drifted")
    sequence = payload.get("sequence")
    if not isinstance(sequence, list) or len(sequence) != 100:
        raise ValueError("objective-family prompt sequence must contain exactly 100 rows")
    if payload.get("sequence_sha256") != canonical_json_sha256(sequence):
        raise ValueError("objective-family prompt sequence hash drifted")
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id or record_id in by_id:
            raise ValueError("selected task rows lack unique stable record IDs")
        if row.get("source") != source or row.get("role") != "student_opd":
            raise ValueError("selected task rows mix source or role")
        by_id[record_id] = row
    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for position, item in enumerate(sequence):
        if not isinstance(item, dict) or set(item) != {
            "position",
            "record_id",
            "prompt_sha256",
        }:
            raise ValueError("objective-family prompt-sequence row schema drifted")
        record_id = item.get("record_id")
        if item.get("position") != position or record_id not in by_id or record_id in seen:
            raise ValueError("objective-family prompt sequence has an invalid row identity")
        row = by_id[record_id]
        if item.get("prompt_sha256") != task_prompt_sha256(row):
            raise ValueError("objective-family prompt sequence prompt hash drifted")
        seen.add(record_id)
        ordered.append(row)
    required_steps = 1 if diagnostic else 100
    if steps != required_steps:
        raise ValueError(
            f"objective-family {'diagnostic' if diagnostic else 'scientific'} run "
            f"requires exactly {required_steps} optimizer steps"
        )
    return {
        "path": str(raw.resolve()),
        "sha256": sha256_file(raw),
        "sequence_sha256": payload["sequence_sha256"],
        "source": source,
        "seed": seed,
        "consumed_prefix_rows": required_steps,
        "full_sequence_rows": len(sequence),
    }, ordered[:required_steps]


def validate_initialization_manifest(
    path: str | Path,
    *,
    student: str,
    student_revision: str,
    seed: int,
    lora_r: int,
    git_commit: str,
) -> dict[str, Any]:
    raw = Path(path)
    if raw.is_symlink() or not raw.is_file():
        raise ValueError("objective-family initialization manifest must be a regular file")
    payload = json.loads(raw.read_text(encoding="utf-8"))
    expected_keys = {
        "schema_version",
        "initialization",
        "status",
        "scientific_launch_authorized",
        "git_commit",
        "objective_registry_sha256",
        "student_training_plan_sha256",
        "student",
        "student_revision",
        "seed",
        "lora",
        "adapter_path",
        "adapter_tree_sha256",
        "trainable_parameter_signature",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ValueError("objective-family initialization manifest schema drifted")
    registry = load_objective_registry()
    for field, expected in (
        ("schema_version", 1),
        ("initialization", INITIALIZATION_ID),
        ("status", "fixed_input_not_launch_authorization"),
        ("scientific_launch_authorized", False),
        ("git_commit", git_commit),
        ("objective_registry_sha256", registry["sha256"]),
        ("student_training_plan_sha256", sha256_file(STUDENT_PLAN)),
        ("student", student),
        ("student_revision", student_revision),
        ("seed", seed),
    ):
        if payload.get(field) != expected:
            raise ValueError(f"objective-family initialization {field} drifted")
    if payload.get("lora") != {
        "r": lora_r,
        "alpha": 2 * lora_r,
        "dropout": 0.0,
        "bias": "none",
        "target_modules": list(LORA_TARGETS),
        "task_type": "CAUSAL_LM",
    }:
        raise ValueError("objective-family initialization LoRA recipe drifted")
    adapter_value = payload.get("adapter_path")
    if not isinstance(adapter_value, str) or not Path(adapter_value).is_absolute():
        raise ValueError("objective-family initialization adapter path is not absolute")
    adapter_path = Path(adapter_value).resolve()
    tree_hash = sha256_tree(adapter_path)
    if payload.get("adapter_tree_sha256") != tree_hash:
        raise ValueError("objective-family initialization adapter tree drifted")
    signature = payload.get("trainable_parameter_signature")
    if not isinstance(signature, dict) or set(signature) != {"elements", "sum", "squared_l2"}:
        raise ValueError("objective-family initialization signature drifted")
    for key in ("elements", "sum", "squared_l2"):
        if type(signature[key]) not in (int, float):
            raise ValueError("objective-family initialization signature is nonnumeric")
    return {
        "path": str(raw.resolve()),
        "sha256": sha256_file(raw),
        "adapter_path": str(adapter_path),
        "adapter_tree_sha256": tree_hash,
        "seed": seed,
        "trainable_parameter_signature": signature,
    }


def _write_prompt_plans(args: argparse.Namespace) -> dict[str, Any]:
    commit = _clean_git_state()
    prepared_path = args.prepared_manifest.resolve()
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    if prepared.get("scientific_use_allowed") is not True:
        raise ValueError("prepared manifest is not scientific")
    output_root = args.output_root.resolve()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"refusing to overwrite prompt-plan root: {output_root}")
    output_root.mkdir(parents=True)
    written: dict[str, Any] = {}
    for source in ("M", "O"):
        relative = f"roles/{source}/student_opd.jsonl"
        task_file = (prepared_path.parent / relative).resolve()
        entry = prepared.get("files", {}).get(relative)
        if not isinstance(entry, dict) or entry.get("sha256") != sha256_file(task_file):
            raise ValueError(f"prepared role binding drifted: {relative}")
        rows = _load_rows(task_file)
        if entry.get("rows") != len(rows):
            raise ValueError(f"prepared role row count drifted: {relative}")
        for seed in (0, 1, 2):
            payload = build_prompt_plan(
                rows=rows,
                source=source,
                seed=seed,
                task_file=task_file,
                prepared_manifest=prepared_path,
                git_commit=commit,
            )
            target = output_root / f"{source}_seed{seed}.json"
            target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            os.chmod(target, 0o444)
            written[f"{source}_seed{seed}"] = {
                "path": str(target),
                "sha256": sha256_file(target),
                "sequence_sha256": payload["sequence_sha256"],
            }
    index = {
        "schema_version": 1,
        "index": "opd_math_objective_family_prompt_plan_index_v1",
        "git_commit": commit,
        "prepared_manifest": {"path": str(prepared_path), "sha256": sha256_file(prepared_path)},
        "plans": written,
        "scientific_launch_authorized": False,
    }
    index_path = output_root / "index.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(index_path, 0o444)
    os.chmod(output_root, 0o555)
    return index


def _write_initial_adapter(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    commit = _clean_git_state()
    plan = _student_plan()
    if args.seed not in plan["allowed_seeds"]:
        raise ValueError("initialization seed is not registered")
    if args.student != EXPECTED_STUDENT or args.student_revision != EXPECTED_STUDENT_REVISION:
        raise ValueError("initialization student identity drifted")
    lora_r = int(plan["common_fixed_config"]["lora_r"])
    output_root = args.output_root.resolve()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"refusing to overwrite initialization root: {output_root}")
    adapter_path = output_root / "adapter"
    adapter_path.mkdir(parents=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        args.student,
        revision=args.student_revision,
        local_files_only=True,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        attn_implementation=plan["common_fixed_config"]["attn_implementation"],
    ).to(device)
    model = get_peft_model(
        model,
        LoraConfig(
            r=lora_r,
            lora_alpha=2 * lora_r,
            lora_dropout=0.0,
            bias="none",
            target_modules=list(LORA_TARGETS),
            task_type="CAUSAL_LM",
        ),
    )
    total = squared = 0.0
    elements = 0
    for parameter in model.parameters():
        if parameter.requires_grad:
            values = parameter.detach().float()
            total += float(values.sum().item())
            squared += float(values.square().sum().item())
            elements += values.numel()
    if elements <= 0:
        raise RuntimeError("initialized adapter has no trainable parameters")
    model.save_pretrained(adapter_path, safe_serialization=True)
    tree_hash = sha256_tree(adapter_path)
    registry = load_objective_registry()
    payload = {
        "schema_version": 1,
        "initialization": INITIALIZATION_ID,
        "status": "fixed_input_not_launch_authorization",
        "scientific_launch_authorized": False,
        "git_commit": commit,
        "objective_registry_sha256": registry["sha256"],
        "student_training_plan_sha256": sha256_file(STUDENT_PLAN),
        "student": args.student,
        "student_revision": args.student_revision,
        "seed": args.seed,
        "lora": {
            "r": lora_r,
            "alpha": 2 * lora_r,
            "dropout": 0.0,
            "bias": "none",
            "target_modules": list(LORA_TARGETS),
            "task_type": "CAUSAL_LM",
        },
        "adapter_path": str(adapter_path),
        "adapter_tree_sha256": tree_hash,
        "trainable_parameter_signature": {
            "elements": elements,
            "sum": total,
            "squared_l2": squared,
        },
    }
    manifest = output_root / "initialization_manifest.json"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for path in adapter_path.rglob("*"):
        if path.is_file():
            os.chmod(path, 0o444)
    for path in sorted((p for p in adapter_path.rglob("*") if p.is_dir()), reverse=True):
        os.chmod(path, 0o555)
    os.chmod(adapter_path, 0o555)
    os.chmod(manifest, 0o444)
    os.chmod(output_root, 0o555)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prompts = subparsers.add_parser("prompt-plans")
    prompts.add_argument("--prepared-manifest", type=Path, required=True)
    prompts.add_argument("--output-root", type=Path, required=True)
    adapter = subparsers.add_parser("initial-adapter")
    adapter.add_argument("--student", default=EXPECTED_STUDENT)
    adapter.add_argument("--student-revision", default=EXPECTED_STUDENT_REVISION)
    adapter.add_argument("--seed", type=int, required=True)
    adapter.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "prompt-plans":
        result = _write_prompt_plans(args)
    else:
        result = _write_initial_adapter(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
