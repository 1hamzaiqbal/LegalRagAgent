#!/usr/bin/env python3
"""Preflight and audit pinned-veRL objective-family runs."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import stat
import subprocess
from pathlib import Path
from typing import Any

try:
    from .objective_family_inputs import (
        EXPECTED_STUDENT,
        EXPECTED_STUDENT_REVISION,
        canonical_json_sha256,
        sha256_file,
        sha256_tree,
        validate_initialization_manifest,
    )
    from .objective_family_preregistration import (
        _validate_support,
        _validate_teacher,
        validate_upstream_prelaunch_receipt,
    )
    from .prepare_verl_objective_data import validate_dataset
    from .verl_objective_contract import OBJECTIVE_ID, load_plan
except ImportError:
    from objective_family_inputs import (  # type: ignore
        EXPECTED_STUDENT,
        EXPECTED_STUDENT_REVISION,
        canonical_json_sha256,
        sha256_file,
        sha256_tree,
        validate_initialization_manifest,
    )
    from objective_family_preregistration import (  # type: ignore
        _validate_support,
        _validate_teacher,
        validate_upstream_prelaunch_receipt,
    )
    from prepare_verl_objective_data import validate_dataset  # type: ignore
    from verl_objective_contract import OBJECTIVE_ID, load_plan  # type: ignore


ROOT = Path(__file__).resolve().parents[2]
PREFLIGHT_ID = "opd_math_objective_family_verl_preflight_v1"
RECEIPT_ID = "opd_math_objective_family_verl_run_receipt_v1"
UPSTREAM_CORE_FILES = (
    "examples/on_policy_distillation_trainer/run_qwen3_8b_fsdp.sh",
    "verl/trainer/distillation/losses.py",
    "verl/trainer/ppo/core_algos.py",
    "verl/experimental/teacher_loop/teacher_manager.py",
    "verl/workers/engine/fsdp/transformer_impl.py",
)
HEX40 = re.compile(r"^[0-9a-f]{40}$")


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json(path: str | Path, label: str) -> dict[str, Any]:
    path = Path(path)
    _expect(path.is_file() and not path.is_symlink(), f"{label} must be a regular file")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _expect(isinstance(payload, dict), f"{label} must be a JSON object")
    return payload


def _git_state(path: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain=v1", "--untracked-files=no"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    return {"commit": commit, "tracked_clean": not status.strip()}


def _readonly(path: Path, label: str) -> Path:
    _expect(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    _expect(
        path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH) == 0,
        f"{label} is not sealed read-only",
    )
    return path.resolve()


def _teacher_identity(args, commit: str) -> dict[str, Any]:
    gap_path = Path(args.teacher_gap_manifest).resolve()
    gap = _json(gap_path, "O teacher gap")
    checkpoint = Path(args.teacher_checkpoint).resolve()
    provenance_path = Path(args.teacher_provenance_manifest).resolve()
    _expect(
        provenance_path == checkpoint / "merge_provenance.json",
        "veRL teacher provenance is not canonical inside its checkpoint",
    )
    provenance = _json(provenance_path, "O teacher merge provenance")
    identity = {
        "teacher_source": "O",
        "base_model": gap.get("base_model"),
        "base_revision": gap.get("base_model_revision"),
        "teacher_gap_manifest": str(gap_path),
        "teacher_gap_manifest_sha256": sha256_file(gap_path),
        "teacher_gap_payload_sha256": canonical_json_sha256(gap),
        "merged_checkpoint": str(checkpoint),
        "merged_checkpoint_tree_sha256": sha256_tree(
            checkpoint, exclude_relative_paths=("merge_provenance.json",)
        ),
        "merge_provenance_manifest_sha256": sha256_file(provenance_path),
        "merge_provenance_payload_sha256": canonical_json_sha256(provenance),
    }
    return _validate_teacher(identity, commit=commit)


def build_preflight(args) -> dict[str, Any]:
    plan = load_plan()
    local_state = _git_state(ROOT)
    commit = local_state["commit"]
    _expect(local_state["tracked_clean"] is True, "veRL preflight requires clean local Git")
    _expect(HEX40.fullmatch(commit) is not None, "local Git commit is invalid")
    upstream = Path(args.verl_checkout).resolve()
    upstream_state = _git_state(upstream)
    _expect(
        upstream_state
        == {"commit": plan["payload"]["upstream_verl_commit"], "tracked_clean": True},
        "veRL checkout is not the clean pinned commit",
    )
    source = args.source
    seed = args.seed
    diagnostic = args.campaign_kind == "diagnostic"
    expected_seed = plan["payload"]["diagnostic_seed"] if diagnostic else seed
    _expect(seed == expected_seed, "veRL diagnostic/scientific seed contract drifted")
    steps = (
        plan["payload"]["diagnostic_optimizer_steps"]
        if diagnostic
        else plan["payload"]["scientific_optimizer_steps"]
    )
    initialization = validate_initialization_manifest(
        args.initialization_manifest,
        student=EXPECTED_STUDENT,
        student_revision=EXPECTED_STUDENT_REVISION,
        seed=seed,
        lora_r=32,
        git_commit=commit,
    )
    data = validate_dataset(
        task_file=Path(args.task_file),
        prepared_manifest=Path(args.prepared_manifest),
        prompt_plan=Path(args.prompt_plan),
        source=source,
        seed=seed,
        git_commit=commit,
        diagnostic=diagnostic,
        output=Path(args.data_file),
        manifest_path=Path(args.data_manifest),
    )
    support_path = Path(args.student_support_manifest).resolve()
    support_payload = _json(support_path, f"{source} support gate")
    support = {
        "path": str(support_path),
        "sha256": sha256_file(support_path),
        "payload_sha256": canonical_json_sha256(support_payload),
        "source": source,
    }
    support = _validate_support(support, source=source, commit=commit)
    teacher = _teacher_identity(args, commit)
    freeze = _readonly(Path(args.environment_freeze), "upstream veRL environment freeze")
    _expect(
        freeze.name == "upstream_verl.freeze.txt" and freeze.parent.name == commit,
        "upstream veRL freeze is not commit-specific",
    )
    launcher = Path(args.launcher).resolve()
    _expect(
        launcher == (ROOT / "scripts/hpc/slurm_opd_math_objective_family_verl.sh").resolve()
        and launcher.is_file()
        and not launcher.is_symlink(),
        "upstream veRL launcher identity drifted",
    )
    output_root = Path(args.output_root).resolve()
    _expect(not output_root.exists() and not output_root.is_symlink(), "veRL output is not fresh")
    _expect(
        isinstance(args.scheduler_job_id, str)
        and re.fullmatch(r"[1-9][0-9]*", args.scheduler_job_id) is not None,
        "veRL scheduler job ID is invalid",
    )
    if diagnostic:
        _expect(args.prelaunch_receipt is None, "diagnostic unexpectedly binds science prelaunch")
        prelaunch = None
    else:
        prelaunch = validate_upstream_prelaunch_receipt(
            args.prelaunch_receipt,
            objective_id=OBJECTIVE_ID,
            source=source,
            seed=seed,
            out_dir=output_root,
            run_id=args.run_id,
            scheduler_job_id=args.scheduler_job_id,
        )
    core_files = {
        relative: sha256_file(upstream / relative) for relative in UPSTREAM_CORE_FILES
    }
    return {
        "schema_version": 1,
        "preflight": PREFLIGHT_ID,
        "status": "validated_before_optimizer_start",
        "scientific_launch_authorized": not diagnostic,
        "campaign_kind": args.campaign_kind,
        "objective_id": OBJECTIVE_ID,
        "source": source,
        "seed": seed,
        "optimizer_steps": steps,
        "scheduler_job_id": args.scheduler_job_id,
        "run_id": args.run_id,
        "git_commit": commit,
        "git_tracked_clean": True,
        "objective_plan": {"path": plan["path"], "sha256": plan["sha256"]},
        "objective_registry_sha256": plan["registry"]["sha256"],
        "launcher": {"path": str(launcher), "sha256": sha256_file(launcher)},
        "upstream_verl": {
            "checkout": str(upstream),
            "commit": upstream_state["commit"],
            "tracked_clean": True,
            "core_files": core_files,
        },
        "environment": {
            "root": str(Path(args.environment_root).resolve()),
            "freeze": str(freeze),
            "freeze_sha256": sha256_file(freeze),
        },
        "student": EXPECTED_STUDENT,
        "student_revision": EXPECTED_STUDENT_REVISION,
        "initialization": initialization,
        "data": data,
        "student_support": support,
        "o_teacher": teacher,
        "output_root": str(output_root),
        "prelaunch_receipt": prelaunch,
        "heldout_outcomes_inspected": False,
        "claim_boundary": (
            "Preflight validates native pinned-veRL bare K1 execution inputs only; "
            "it does not establish an optimizer update or held-out improvement."
        ),
    }


def write_new(path: Path, payload: dict[str, Any], *, readonly: bool = True) -> None:
    path = path.resolve()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite custody artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if readonly:
        os.chmod(path, 0o444)


def _tensor_file(path: Path) -> dict[str, Any]:
    from safetensors.torch import load_file

    candidates = sorted(path.glob("*.safetensors"))
    _expect(len(candidates) == 1, f"adapter must contain exactly one safetensors file: {path}")
    state = load_file(str(candidates[0]), device="cpu")
    _expect(state, f"adapter has no tensors: {path}")
    elements = 0
    total = 0.0
    squared = 0.0
    for name, tensor in state.items():
        values = tensor.detach().float()
        _expect(bool(values.isfinite().all()), f"adapter tensor is nonfinite: {name}")
        elements += values.numel()
        total += float(values.sum().item())
        squared += float(values.square().sum().item())
    return {
        "path": str(candidates[0].resolve()),
        "sha256": sha256_file(candidates[0]),
        "state": state,
        "signature": {"elements": elements, "sum": total, "squared_l2": squared},
    }


def _adapter_delta(initial_path: Path, final_path: Path) -> dict[str, Any]:
    initial = _tensor_file(initial_path)
    final = _tensor_file(final_path)
    _expect(set(initial["state"]) == set(final["state"]), "initial/final LoRA tensor keys differ")
    delta_squared = 0.0
    max_abs = 0.0
    changed_tensors = 0
    for key in sorted(initial["state"]):
        left = initial["state"][key].detach().float()
        right = final["state"][key].detach().float()
        _expect(left.shape == right.shape, f"initial/final LoRA shape differs: {key}")
        delta = right - left
        _expect(bool(delta.isfinite().all()), f"LoRA delta is nonfinite: {key}")
        value = float(delta.square().sum().item())
        delta_squared += value
        max_abs = max(max_abs, float(delta.abs().max().item()))
        changed_tensors += int(value > 0)
    _expect(delta_squared > 0 and max_abs > 0 and changed_tensors > 0, "veRL LoRA did not update")
    return {
        "initial_adapter_tree_sha256": sha256_tree(initial_path),
        "final_adapter_tree_sha256": sha256_tree(final_path),
        "initial_signature": initial["signature"],
        "final_signature": final["signature"],
        "delta_l2": math.sqrt(delta_squared),
        "delta_max_abs": max_abs,
        "changed_tensors": changed_tensors,
        "tensor_count": len(initial["state"]),
    }


def _walk_tensors(value: Any):
    import torch

    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _walk_tensors(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_tensors(child)


def _optimizer_custody(actor_checkpoint: Path) -> dict[str, Any]:
    import torch

    optimizers = sorted(actor_checkpoint.glob("optim_world_size_*_rank_*.pt"))
    _expect(len(optimizers) == 1, "single-actor veRL checkpoint lacks one optimizer shard")
    state = torch.load(optimizers[0], map_location="cpu", weights_only=False)
    tensors = list(_walk_tensors(state))
    _expect(tensors, "veRL optimizer checkpoint has no tensor state")
    elements = 0
    squared = 0.0
    for tensor in tensors:
        values = tensor.detach().float()
        _expect(bool(values.isfinite().all()), "veRL optimizer state is nonfinite")
        elements += values.numel()
        squared += float(values.square().sum().item())
    _expect(elements > 0 and squared > 0, "veRL optimizer state is empty or zero")
    return {
        "path": str(optimizers[0].resolve()),
        "sha256": sha256_file(optimizers[0]),
        "tensors": len(tensors),
        "elements": elements,
        "squared_l2": squared,
    }


def _parse_metrics(path: Path, steps: int) -> list[dict[str, Any]]:
    observed: dict[int, dict[str, float]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.search(r"step:(\d+)\s+-\s+(.+)", line)
        if match is None:
            continue
        step = int(match.group(1))
        metrics = {}
        for part in match.group(2).split(" - "):
            if ":" not in part:
                continue
            key, raw = part.split(":", 1)
            raw = raw.strip()
            wrapper = re.fullmatch(r"(?:np\.)?float(?:32|64)?\(([-+0-9.eE]+)\)", raw)
            if wrapper:
                raw = wrapper.group(1)
            try:
                value = float(raw)
            except ValueError:
                continue
            if math.isfinite(value):
                metrics[key.strip()] = value
        if metrics:
            observed[step] = metrics
    _expect(set(observed) == set(range(1, steps + 1)), "veRL console metric step coverage drifted")
    result = []
    for step in range(1, steps + 1):
        metrics = observed[step]
        gradients = [value for key, value in metrics.items() if key.endswith("grad_norm")]
        losses = [
            value
            for key, value in metrics.items()
            if key.endswith("distillation/loss") or key.endswith("distillation_loss")
        ]
        _expect(gradients and all(value > 0 for value in gradients), f"veRL step {step} lacks positive gradient")
        _expect(losses, f"veRL step {step} lacks finite distillation loss")
        result.append(
            {
                "step": step,
                "gradient_norm": gradients[-1],
                "distillation_loss": losses[-1],
                "metrics_sha256": canonical_json_sha256(metrics),
            }
        )
    return result


def _rollout_custody(path: Path, steps: int) -> dict[str, Any]:
    files = sorted(path.glob("*.jsonl"), key=lambda item: int(item.stem))
    _expect([item.name for item in files] == [f"{step}.jsonl" for step in range(1, steps + 1)], "veRL rollout dump coverage drifted")
    bindings = []
    total = 0
    for step, file_path in enumerate(files, 1):
        rows = []
        for line in file_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                _expect(isinstance(value, dict), "veRL rollout row is not an object")
                rows.append(value)
        _expect(len(rows) == 4, f"veRL rollout step {step} does not have four trajectories")
        _expect(all(row.get("step") == step for row in rows), "veRL rollout step identity drifted")
        _expect(len({row.get("input") for row in rows}) == 1, "veRL rollout group mixes prompts")
        _expect(all(isinstance(row.get("output"), str) for row in rows), "veRL rollout output is missing")
        total += len(rows)
        bindings.append({"path": str(file_path.resolve()), "sha256": sha256_file(file_path), "rows": len(rows)})
    return {"files": bindings, "rows": total, "tree_sha256": sha256_tree(path)}


def _seal_tree(path: Path) -> None:
    for candidate in path.rglob("*"):
        if candidate.is_symlink():
            raise ValueError(f"refusing to seal a tree containing symlink: {candidate}")
        os.chmod(candidate, 0o555 if candidate.is_dir() else 0o444)
    os.chmod(path, 0o555)


def audit_run(args) -> dict[str, Any]:
    preflight_path = _readonly(Path(args.preflight), "veRL preflight")
    preflight = _json(preflight_path, "veRL preflight")
    _expect(
        preflight.get("schema_version") == 1
        and preflight.get("preflight") == PREFLIGHT_ID
        and preflight.get("status") == "validated_before_optimizer_start"
        and preflight.get("heldout_outcomes_inspected") is False,
        "veRL preflight identity drifted",
    )
    local_state = _git_state(ROOT)
    _expect(
        local_state == {"commit": preflight.get("git_commit"), "tracked_clean": True},
        "local code changed during veRL execution",
    )
    upstream = Path(preflight["upstream_verl"]["checkout"])
    _expect(
        _git_state(upstream)
        == {"commit": preflight["upstream_verl"]["commit"], "tracked_clean": True},
        "pinned veRL checkout changed during execution",
    )
    output_root = Path(preflight["output_root"]).resolve()
    _expect(output_root.is_dir() and not output_root.is_symlink(), "veRL run output is missing")
    run_log = Path(args.run_log).resolve()
    checkpoint = Path(args.actor_checkpoint).resolve()
    rollouts = Path(args.rollout_dir).resolve()
    final = Path(args.final_adapter).resolve()
    for path in (run_log, checkpoint, rollouts, final):
        _expect(path == output_root / path.relative_to(output_root), "veRL artifact escaped output root")
    _expect(run_log.is_file() and not run_log.is_symlink(), "veRL run log is missing")
    _expect(checkpoint.is_dir() and rollouts.is_dir() and final.is_dir(), "veRL output tree is incomplete")
    steps = int(preflight["optimizer_steps"])
    metrics = _parse_metrics(run_log, steps)
    rollout = _rollout_custody(rollouts, steps)
    delta = _adapter_delta(Path(preflight["initialization"]["adapter_path"]), final)
    optimizer = _optimizer_custody(checkpoint)
    _expect(
        delta["initial_adapter_tree_sha256"]
        == preflight["initialization"]["adapter_tree_sha256"],
        "veRL initial adapter changed after preflight",
    )
    receipt = {
        "schema_version": 1,
        "receipt": RECEIPT_ID,
        "status": (
            "passed_plumbing"
            if preflight["campaign_kind"] == "diagnostic"
            else "completed_training_pending_heldout"
        ),
        "scientific_use_allowed": False,
        "training_artifact_eligible_for_heldout_evaluation": preflight["campaign_kind"] == "scientific",
        "objective_id": OBJECTIVE_ID,
        "source": preflight["source"],
        "seed": preflight["seed"],
        "optimizer_steps": steps,
        "git_commit": preflight["git_commit"],
        "upstream_verl_commit": preflight["upstream_verl"]["commit"],
        "preflight": {"path": str(preflight_path), "sha256": sha256_file(preflight_path)},
        "run_log": {"path": str(run_log), "sha256": sha256_file(run_log)},
        "actor_checkpoint": {"path": str(checkpoint), "tree_sha256": sha256_tree(checkpoint)},
        "optimizer": optimizer,
        "rollouts": rollout,
        "adapter_update": delta,
        "final_adapter": {"path": str(final), "tree_sha256": sha256_tree(final)},
        "metrics": metrics,
        "finite_nonzero_gradient_observed": True,
        "parameter_update_observed": True,
        "optimizer_state_observed": True,
        "heldout_outcomes_inspected": False,
        "claim_boundary": (
            "This receipt proves native pinned-veRL execution and a finite nonzero LoRA update. "
            "It does not establish held-out task improvement."
        ),
    }
    write_new(Path(args.output), receipt)
    _seal_tree(output_root)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--campaign-kind", choices=("diagnostic", "scientific"), required=True)
    preflight.add_argument("--source", choices=("M", "O"), required=True)
    preflight.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    preflight.add_argument("--run-id", required=True)
    preflight.add_argument("--scheduler-job-id", required=True)
    preflight.add_argument("--task-file", required=True)
    preflight.add_argument("--prepared-manifest", required=True)
    preflight.add_argument("--prompt-plan", required=True)
    preflight.add_argument("--initialization-manifest", required=True)
    preflight.add_argument("--data-file", required=True)
    preflight.add_argument("--data-manifest", required=True)
    preflight.add_argument("--student-support-manifest", required=True)
    preflight.add_argument("--teacher-checkpoint", required=True)
    preflight.add_argument("--teacher-gap-manifest", required=True)
    preflight.add_argument("--teacher-provenance-manifest", required=True)
    preflight.add_argument("--verl-checkout", required=True)
    preflight.add_argument("--environment-root", required=True)
    preflight.add_argument("--environment-freeze", required=True)
    preflight.add_argument("--launcher", required=True)
    preflight.add_argument("--output-root", required=True)
    preflight.add_argument("--prelaunch-receipt")
    preflight.add_argument("--output", type=Path, required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--preflight", required=True)
    audit.add_argument("--run-log", required=True)
    audit.add_argument("--actor-checkpoint", required=True)
    audit.add_argument("--rollout-dir", required=True)
    audit.add_argument("--final-adapter", required=True)
    audit.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "preflight":
        payload = build_preflight(args)
        write_new(args.output, payload)
    else:
        payload = audit_run(args)
    print(json.dumps({"output": str(args.output.resolve()), "sha256": sha256_file(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
