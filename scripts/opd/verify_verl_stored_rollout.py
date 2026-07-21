#!/usr/bin/env python3
"""Run Level-2 stored-rollout fidelity against a pinned veRL checkout."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
PINNED_VERL_COMMIT = "6a6242f3d8ec7d9f8b4936f4905144707d91fe3b"
SYNTHETIC_FIXTURE_ID = "shared_rollout_k1_v1"
REAL_MODEL_FIXTURE_ID = "real_model_rollout_k1_v1"
LOCAL_LOSS = ROOT / "scripts/opd/opd_loss.py"
LOCAL_TRACE_METRICS = ROOT / "scripts/opd/trace_metrics.py"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(checkout: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(checkout), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _validate_number_list(value: Any, label: str, length: int) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{label} must contain exactly {length} values")
    result = []
    for item in value:
        if type(item) not in (int, float) or not math.isfinite(float(item)):
            raise ValueError(f"{label} contains a nonfinite or nonnumeric value")
        result.append(float(item))
    return result


def load_fixture(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("stored-rollout fixture must be a JSON object")
    fixture_id = payload.get("fixture_id")
    expected_keys = {
        "schema_version",
        "fixture_id",
        "status",
        "scientific_launch_authorized",
        "dtype",
        "samples",
        "settings",
        "optimizer",
        "tolerances",
    }
    if fixture_id == REAL_MODEL_FIXTURE_ID:
        expected_keys.add("provenance")
    if set(payload) != expected_keys:
        raise ValueError("stored-rollout fixture has an invalid top-level schema")
    if payload["schema_version"] != 1 or fixture_id not in {
        SYNTHETIC_FIXTURE_ID,
        REAL_MODEL_FIXTURE_ID,
    }:
        raise ValueError("stored-rollout fixture identity is unsupported")
    expected_status = (
        "synthetic_stored_tensor_fidelity_only"
        if fixture_id == SYNTHETIC_FIXTURE_ID
        else "real_model_stored_tensor_fidelity_only"
    )
    if payload["status"] != expected_status:
        raise ValueError("stored-rollout fixture status drifted")
    if payload["scientific_launch_authorized"] is not False:
        raise ValueError("stored-rollout fixture must not authorize scientific launch")
    if payload["dtype"] != "float64":
        raise ValueError("stored-rollout fixture must use float64")
    if fixture_id == REAL_MODEL_FIXTURE_ID:
        provenance = payload["provenance"]
        expected_provenance_keys = {
            "source_samples",
            "source_samples_sha256",
            "run_manifest",
            "run_manifest_sha256",
            "completion_manifest",
            "completion_manifest_sha256",
            "local_git_commit",
            "objective_registry_sha256",
            "student",
            "student_revision",
            "teacher_checkpoint",
            "teacher_checkpoint_tree_sha256",
            "extractor_sha256",
            "behavior_logprobs_origin",
            "current_student_logprobs_origin",
            "teacher_logprobs_origin",
            "heldout_outcomes_inspected",
        }
        if not isinstance(provenance, dict) or set(provenance) != expected_provenance_keys:
            raise ValueError("real-model fixture provenance schema drifted")
        expected_origins = {
            "behavior_logprobs_origin": "generation_transition_scores_before_update",
            "current_student_logprobs_origin": "pre_update_student_forward_on_generated_tokens",
            "teacher_logprobs_origin": "frozen_o_teacher_exact_generated_token_scores",
        }
        for field, expected in expected_origins.items():
            if provenance.get(field) != expected:
                raise ValueError(f"real-model fixture {field} drifted")
        if provenance.get("heldout_outcomes_inspected") is not False:
            raise ValueError("real-model fixture inspected held-out outcomes")
        for path_field, hash_field in (
            ("source_samples", "source_samples_sha256"),
            ("run_manifest", "run_manifest_sha256"),
            ("completion_manifest", "completion_manifest_sha256"),
        ):
            bound = Path(str(provenance.get(path_field))).resolve()
            if not bound.is_file() or _sha256(bound) != provenance.get(hash_field):
                raise ValueError(f"real-model fixture {path_field} binding drifted")

    expected_settings = {
        "loss_mode": "k1",
        "loss_max_clamp": 10.0,
        "policy_loss_mode": "vanilla",
        "clip_ratio_low": 0.2,
        "clip_ratio_high": 0.2,
        "dual_clip_ratio": 3.0,
        "loss_agg_mode": "token-mean",
    }
    if payload["settings"] != expected_settings:
        raise ValueError("stored-rollout veRL settings drifted")
    expected_optimizer = {
        "name": "AdamW",
        "learning_rate": 0.001,
        "betas": [0.9, 0.999],
        "epsilon": 1e-08,
        "weight_decay": 0.0,
    }
    if payload["optimizer"] != expected_optimizer:
        raise ValueError("stored-rollout optimizer settings drifted")
    expected_tolerances = {
        "absolute": 1e-12,
        "relative": 1e-12,
        "gradient_cosine_minimum": 0.999999999999,
    }
    if payload["tolerances"] != expected_tolerances:
        raise ValueError("stored-rollout tolerances drifted")

    samples = payload["samples"]
    if not isinstance(samples, list) or len(samples) < 2:
        raise ValueError("stored-rollout fixture requires at least two samples")
    width = None
    seen_ids: set[str] = set()
    normalized_samples = []
    for index, sample in enumerate(samples):
        label = f"sample[{index}]"
        if not isinstance(sample, dict) or set(sample) != {
            "sample_id",
            "prompt_token_ids",
            "completion_token_ids",
            "response_mask",
            "behavior_logprobs",
            "current_logprobs",
            "teacher_logprobs",
        }:
            raise ValueError(f"{label} has an invalid schema")
        sample_id = sample["sample_id"]
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen_ids:
            raise ValueError(f"{label} has an invalid or duplicate sample_id")
        seen_ids.add(sample_id)
        prompt_ids = sample["prompt_token_ids"]
        completion_ids = sample["completion_token_ids"]
        response_mask = sample["response_mask"]
        if (
            not isinstance(prompt_ids, list)
            or not prompt_ids
            or any(type(item) is not int or item < 0 for item in prompt_ids)
        ):
            raise ValueError(f"{label} has invalid prompt token IDs")
        if (
            not isinstance(completion_ids, list)
            or not completion_ids
            or any(type(item) is not int or item < 0 for item in completion_ids)
        ):
            raise ValueError(f"{label} has invalid completion token IDs")
        if not isinstance(response_mask, list) or any(type(item) is not bool for item in response_mask):
            raise ValueError(f"{label} has an invalid response mask")
        if len(response_mask) != len(completion_ids) or not any(response_mask):
            raise ValueError(f"{label} response mask length or support is invalid")
        if width is None:
            width = len(completion_ids)
        elif len(completion_ids) != width:
            raise ValueError("stored-rollout samples must have a common padded width")
        behavior = _validate_number_list(
            sample["behavior_logprobs"], f"{label} behavior_logprobs", len(completion_ids)
        )
        current = _validate_number_list(
            sample["current_logprobs"], f"{label} current_logprobs", len(completion_ids)
        )
        teacher = _validate_number_list(
            sample["teacher_logprobs"], f"{label} teacher_logprobs", len(completion_ids)
        )
        normalized_samples.append(
            {
                **sample,
                "prompt_token_ids": list(prompt_ids),
                "completion_token_ids": list(completion_ids),
                "response_mask": list(response_mask),
                "behavior_logprobs": behavior,
                "current_logprobs": current,
                "teacher_logprobs": teacher,
            }
        )
    return {**payload, "samples": normalized_samples}


def _checkout_custody(checkout: Path, fixture_path: Path) -> dict[str, Any]:
    local_commit = _git(ROOT, "rev-parse", "HEAD")
    local_status = _git(ROOT, "status", "--porcelain", "--untracked-files=no")
    if local_status:
        raise ValueError("local OPD checkout has tracked modifications")
    upstream_commit = _git(checkout, "rev-parse", "HEAD")
    upstream_status = _git(checkout, "status", "--porcelain", "--untracked-files=no")
    if upstream_commit != PINNED_VERL_COMMIT or upstream_status:
        raise ValueError("veRL checkout is not the clean pinned commit")
    return {
        "local": {
            "checkout": str(ROOT),
            "commit": local_commit,
            "tracked_status": "clean",
            "opd_loss_sha256": _sha256(LOCAL_LOSS),
            "trace_metrics_sha256": _sha256(LOCAL_TRACE_METRICS),
            "harness_sha256": _sha256(Path(__file__).resolve()),
            "fixture_path": str(fixture_path.resolve()),
            "fixture_sha256": _sha256(fixture_path),
        },
        "upstream_verl": {
            "checkout": str(checkout.resolve()),
            "commit": upstream_commit,
            "tracked_status": "clean",
            "core_algos_sha256": _sha256(checkout / "verl/trainer/ppo/core_algos.py"),
            "distillation_losses_sha256": _sha256(
                checkout / "verl/trainer/distillation/losses.py"
            ),
        },
    }


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    flat_left = left.flatten()
    flat_right = right.flatten()
    denominator = torch.linalg.vector_norm(flat_left) * torch.linalg.vector_norm(flat_right)
    if float(denominator) == 0.0:
        raise ValueError("cannot compute cosine for a zero gradient")
    return float(torch.dot(flat_left, flat_right) / denominator)


def _upstream_loss(core_loss, kl_penalty, student, teacher, behavior, mask, config):
    k1 = kl_penalty(student, teacher, "k1")
    advantage = -torch.clamp(k1, min=-10.0, max=10.0).detach()
    loss, _ = core_loss(
        old_log_prob=behavior,
        log_prob=student,
        advantages=advantage,
        response_mask=mask,
        loss_agg_mode="token-mean",
        config=config,
        rollout_is_weights=None,
    )
    return loss, k1.detach(), advantage


def run_fidelity(fixture_path: Path, checkout: Path) -> dict[str, Any]:
    fixture_path = fixture_path.resolve()
    fixture = load_fixture(fixture_path)
    checkout = checkout.resolve()
    custody = _checkout_custody(checkout, fixture_path)
    for path in (str(checkout), str(ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)

    from opd_loss import reverse_kl_score_function_loss, verl_k1_policy_gradient_loss
    from omegaconf import OmegaConf
    from trace_metrics import reconstruct_step_metrics
    from verl.trainer.ppo.core_algos import (  # type: ignore[import-not-found]
        compute_policy_loss_vanilla,
        kl_penalty,
    )

    behavior = torch.tensor(
        [sample["behavior_logprobs"] for sample in fixture["samples"]],
        dtype=torch.float64,
    )
    current = torch.tensor(
        [sample["current_logprobs"] for sample in fixture["samples"]],
        dtype=torch.float64,
    )
    teacher = torch.tensor(
        [sample["teacher_logprobs"] for sample in fixture["samples"]],
        dtype=torch.float64,
    )
    mask = torch.tensor(
        [sample["response_mask"] for sample in fixture["samples"]], dtype=torch.bool
    )
    config = OmegaConf.create(
        {
            "clip_ratio": 0.2,
            "clip_ratio_low": 0.2,
            "clip_ratio_high": 0.2,
            "clip_ratio_c": 3.0,
            "global_batch_info": {},
        }
    )
    atol = float(fixture["tolerances"]["absolute"])
    rtol = float(fixture["tolerances"]["relative"])

    upstream_parameter = current.clone().requires_grad_(True)
    upstream_loss, raw_k1, advantage = _upstream_loss(
        compute_policy_loss_vanilla,
        kl_penalty,
        upstream_parameter,
        teacher,
        behavior,
        mask,
        config,
    )
    upstream_loss.backward()
    upstream_gradient = upstream_parameter.grad.detach().clone()

    local_parameter = current.clone().requires_grad_(True)
    local_loss = verl_k1_policy_gradient_loss(
        local_parameter, teacher, behavior, mask, loss_max_clamp=10.0
    )
    local_loss.backward()
    local_gradient = local_parameter.grad.detach().clone()

    optimizer_config = fixture["optimizer"]
    upstream_update_parameter = current.clone().requires_grad_(True)
    local_update_parameter = current.clone().requires_grad_(True)
    upstream_optimizer = torch.optim.AdamW(
        [upstream_update_parameter],
        lr=optimizer_config["learning_rate"],
        betas=tuple(optimizer_config["betas"]),
        eps=optimizer_config["epsilon"],
        weight_decay=optimizer_config["weight_decay"],
    )
    local_optimizer = torch.optim.AdamW(
        [local_update_parameter],
        lr=optimizer_config["learning_rate"],
        betas=tuple(optimizer_config["betas"]),
        eps=optimizer_config["epsilon"],
        weight_decay=optimizer_config["weight_decay"],
    )
    upstream_update_loss, _, _ = _upstream_loss(
        compute_policy_loss_vanilla,
        kl_penalty,
        upstream_update_parameter,
        teacher,
        behavior,
        mask,
        config,
    )
    local_update_loss = verl_k1_policy_gradient_loss(
        local_update_parameter, teacher, behavior, mask, loss_max_clamp=10.0
    )
    upstream_update_loss.backward()
    local_update_loss.backward()
    upstream_optimizer.step()
    local_optimizer.step()

    projected_upstream = behavior.clone().requires_grad_(True)
    projected_upstream_loss, _, _ = _upstream_loss(
        compute_policy_loss_vanilla,
        kl_penalty,
        projected_upstream,
        teacher,
        behavior,
        mask,
        config,
    )
    projected_upstream_loss.backward()
    projected_local = behavior.clone().requires_grad_(True)
    projected_local_loss = reverse_kl_score_function_loss(
        projected_local,
        teacher,
        mask,
        behavior_logprobs=behavior,
        advantage_clip=10.0,
        ratio_clip_eps=0.2,
        gap_gate_beta=None,
    )
    projected_local_loss.backward()
    projected_cosine = _cosine(projected_local.grad, projected_upstream.grad)

    trace_rows = []
    for index, sample in enumerate(fixture["samples"]):
        valid = sum(sample["response_mask"])
        trace_rows.append(
            {
                "group_id": index,
                "completion_token_ids": sample["completion_token_ids"][:valid],
                "student_token_logprobs": sample["current_logprobs"][:valid],
                "behavior_token_logprobs_on_student_trajectory": sample[
                    "behavior_logprobs"
                ][:valid],
                "teacher_token_logprobs_on_student_trajectory": sample[
                    "teacher_logprobs"
                ][:valid],
                "reward": None,
            }
        )
    reconstructed = reconstruct_step_metrics(
        trace_rows,
        mode="k1_bare_verl_compatible_clip10",
        task_reward_coef=0.0,
        k1_coef=1.0,
        advantage_clip=10.0,
        gap_gate_beta=None,
    )

    scalar_matches = bool(torch.allclose(local_loss, upstream_loss, atol=atol, rtol=rtol))
    gradient_matches = bool(
        torch.allclose(local_gradient, upstream_gradient, atol=atol, rtol=rtol)
    )
    update_matches = bool(
        torch.allclose(
            local_update_parameter,
            upstream_update_parameter,
            atol=atol,
            rtol=rtol,
        )
    )
    trace_matches = math.isclose(
        float(reconstructed["total_loss"]),
        float(local_loss.detach()),
        abs_tol=atol,
        rel_tol=rtol,
    )
    projected_gradient_matches = bool(
        torch.allclose(
            projected_local.grad,
            projected_upstream.grad,
            atol=atol,
            rtol=rtol,
        )
    )
    projected_cosine_pass = (
        projected_cosine >= fixture["tolerances"]["gradient_cosine_minimum"]
    )
    masked_gradient_zero = bool(
        torch.equal(local_gradient[~mask], torch.zeros_like(local_gradient[~mask]))
        and torch.equal(
            upstream_gradient[~mask], torch.zeros_like(upstream_gradient[~mask])
        )
    )
    checks = {
        "local_upstream_scalar_matches": scalar_matches,
        "local_upstream_gradient_matches": gradient_matches,
        "local_upstream_adamw_update_matches": update_matches,
        "trace_reconstruction_matches": trace_matches,
        "on_policy_score_function_gradient_matches": projected_gradient_matches,
        "on_policy_score_function_gradient_cosine_pass": projected_cosine_pass,
        "masked_gradient_zero": masked_gradient_zero,
    }
    if not all(checks.values()):
        raise AssertionError(f"stored-rollout fidelity failed: {checks}")

    ratios = torch.exp(current - behavior)
    return {
        "schema_version": 1,
        "check_id": (
            "stored_rollout_local_vs_pinned_verl_k1_v1"
            if fixture["fixture_id"] == SYNTHETIC_FIXTURE_ID
            else "real_model_rollout_local_vs_pinned_verl_k1_v1"
        ),
        "status": "pass",
        "scientific_launch_authorized": False,
        "custody": custody,
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "dtype": "float64",
        },
        "coverage": {
            "real_model_generated_rollout": fixture["fixture_id"] == REAL_MODEL_FIXTURE_ID,
            "behavior_scores_from_generation": fixture["fixture_id"]
            == REAL_MODEL_FIXTURE_ID,
            "samples": len(fixture["samples"]),
            "valid_tokens": int(mask.sum().item()),
            "masked_tokens": int((~mask).sum().item()),
            "ratios_below_clip": int(((ratios < 0.8) & mask).sum().item()),
            "ratios_inside_clip": int(
                ((ratios >= 0.8) & (ratios <= 1.2) & mask).sum().item()
            ),
            "ratios_above_clip": int(((ratios > 1.2) & mask).sum().item()),
            "negative_advantage_tokens": int(((advantage < 0) & mask).sum().item()),
            "positive_advantage_tokens": int(((advantage > 0) & mask).sum().item()),
        },
        "comparison": {
            **checks,
            "upstream_loss": float(upstream_loss.detach()),
            "local_loss": float(local_loss.detach()),
            "scalar_abs_error": abs(float(local_loss.detach() - upstream_loss.detach())),
            "gradient_max_abs_error": float(
                torch.max(torch.abs(local_gradient - upstream_gradient)).item()
            ),
            "adamw_update_max_abs_error": float(
                torch.max(
                    torch.abs(local_update_parameter - upstream_update_parameter)
                ).item()
            ),
            "on_policy_score_function_scalar": float(projected_local_loss.detach()),
            "on_policy_verl_ratio_scalar": float(projected_upstream_loss.detach()),
            "on_policy_gradient_cosine": projected_cosine,
            "on_policy_gradient_max_abs_error": float(
                torch.max(
                    torch.abs(projected_local.grad - projected_upstream.grad)
                ).item()
            ),
            "trace_reconstructed_total_loss": float(reconstructed["total_loss"]),
            "raw_k1_on_valid_tokens": raw_k1[mask].tolist(),
            "executed_advantage_on_valid_tokens": advantage[mask].tolist(),
            "behavior_current_ratio_on_valid_tokens": ratios[mask].tolist(),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--verl-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_fidelity(args.fixture, args.verl_checkout)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(args.output)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
