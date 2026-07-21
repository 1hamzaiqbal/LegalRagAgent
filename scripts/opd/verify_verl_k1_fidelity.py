#!/usr/bin/env python3
"""Compare the local veRL-compatible K1 loss with a pinned veRL checkout.

This harness imports veRL's actual ``kl_penalty`` and
``compute_policy_loss_vanilla`` implementations.  It is deliberately separate
from the training launcher: passing this check is implementation evidence, not
scientific launch authorization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
PINNED_VERL_COMMIT = "6a6242f3d8ec7d9f8b4936f4905144707d91fe3b"
CORE_ALGOS_RELATIVE = Path("verl/trainer/ppo/core_algos.py")
DISTILLATION_LOSSES_RELATIVE = Path("verl/trainer/distillation/losses.py")
LOCAL_LOSS_RELATIVE = Path("scripts/opd/opd_loss.py")


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


def verify_checkout(checkout: Path) -> dict[str, str]:
    checkout = checkout.resolve()
    if not checkout.is_dir():
        raise ValueError(f"veRL checkout does not exist: {checkout}")
    commit = _git(checkout, "rev-parse", "HEAD")
    if commit != PINNED_VERL_COMMIT:
        raise ValueError(
            f"veRL checkout is {commit}, expected pinned {PINNED_VERL_COMMIT}"
        )
    status = _git(checkout, "status", "--porcelain", "--untracked-files=no")
    if status:
        raise ValueError("veRL checkout has tracked modifications")
    core_algos = checkout / CORE_ALGOS_RELATIVE
    distillation_losses = checkout / DISTILLATION_LOSSES_RELATIVE
    if not core_algos.is_file() or not distillation_losses.is_file():
        raise ValueError("veRL checkout lacks required loss source files")
    return {
        "checkout": str(checkout),
        "commit": commit,
        "tracked_status": "clean",
        "core_algos_sha256": _sha256(core_algos),
        "distillation_losses_sha256": _sha256(distillation_losses),
    }


def verify_local_checkout() -> dict[str, str]:
    commit = _git(ROOT, "rev-parse", "HEAD")
    status = _git(ROOT, "status", "--porcelain", "--untracked-files=no")
    if status:
        raise ValueError("local OPD checkout has tracked modifications")
    local_loss = ROOT / LOCAL_LOSS_RELATIVE
    if not local_loss.is_file():
        raise ValueError("local OPD checkout lacks opd_loss.py")
    return {
        "checkout": str(ROOT),
        "commit": commit,
        "tracked_status": "clean",
        "opd_loss_sha256": _sha256(local_loss),
    }


def _install_import_roots(checkout: Path) -> None:
    for path in (str(checkout.resolve()), str(ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)


def run_fidelity(checkout: Path) -> dict[str, object]:
    upstream_custody = verify_checkout(checkout)
    local_custody = verify_local_checkout()
    _install_import_roots(checkout)

    # Import the local helper through the script directory so a third-party
    # top-level ``scripts`` package cannot shadow this checkout.
    from opd_loss import verl_k1_policy_gradient_loss
    from omegaconf import OmegaConf
    from verl.trainer.ppo.core_algos import (  # type: ignore[import-not-found]
        compute_policy_loss_vanilla,
        kl_penalty,
    )

    teacher = torch.tensor(
        [[-2.0, -2.5, -6.0, -0.1], [-4.0, -1.0, -0.5, -3.0]],
        dtype=torch.float64,
    )
    behavior = torch.tensor(
        [[-4.0, -0.8, -2.0, -3.0], [-2.0, -3.0, -4.0, -1.0]],
        dtype=torch.float64,
    )
    initial_student = torch.tensor(
        [[-3.0, -0.1, -3.5, -5.0], [-3.5, -1.3, -1.0, -2.5]],
        dtype=torch.float64,
    )
    mask = torch.tensor(
        [[True, True, True, False], [True, True, False, True]], dtype=torch.bool
    )

    upstream_student = initial_student.clone().requires_grad_(True)
    upstream_k1 = kl_penalty(upstream_student, teacher, "k1")
    expected_k1 = upstream_student - teacher
    if not torch.equal(upstream_k1, expected_k1):
        raise AssertionError("pinned veRL K1 no longer equals student minus teacher")
    upstream_advantage = -torch.clamp(upstream_k1, min=-10.0, max=10.0).detach()
    config = OmegaConf.create(
        {
            "clip_ratio": 0.2,
            "clip_ratio_low": 0.2,
            "clip_ratio_high": 0.2,
            "clip_ratio_c": 3.0,
        }
    )
    upstream_loss, _ = compute_policy_loss_vanilla(
        old_log_prob=behavior,
        log_prob=upstream_student,
        advantages=upstream_advantage,
        response_mask=mask,
        loss_agg_mode="token-mean",
        config=config,
        rollout_is_weights=None,
    )
    upstream_loss.backward()
    upstream_gradient = upstream_student.grad.detach().clone()

    local_student = initial_student.clone().requires_grad_(True)
    local_loss = verl_k1_policy_gradient_loss(
        local_student,
        teacher,
        behavior,
        mask,
        loss_max_clamp=10.0,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        dual_clip_ratio=3.0,
    )
    local_loss.backward()
    local_gradient = local_student.grad.detach().clone()

    scalar_abs_error = abs(float(local_loss.detach()) - float(upstream_loss.detach()))
    gradient_max_abs_error = float(
        torch.max(torch.abs(local_gradient - upstream_gradient)).item()
    )
    scalar_matches = bool(torch.allclose(local_loss, upstream_loss, atol=1e-12, rtol=1e-12))
    gradient_matches = bool(
        torch.allclose(local_gradient, upstream_gradient, atol=1e-12, rtol=1e-12)
    )
    masked_gradient_zero = bool(
        torch.equal(
            local_gradient[~mask], torch.zeros_like(local_gradient[~mask])
        )
        and torch.equal(
            upstream_gradient[~mask], torch.zeros_like(upstream_gradient[~mask])
        )
    )
    if not (scalar_matches and gradient_matches and masked_gradient_zero):
        raise AssertionError(
            "local K1 ratio-form loss differs from the pinned veRL implementation"
        )

    return {
        "schema_version": 1,
        "check_id": "local_vs_pinned_verl_k1_policy_gradient_v1",
        "status": "pass",
        "scientific_launch_authorized": False,
        "custody": {
            "local": local_custody,
            "upstream_verl": upstream_custody,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "dtype": "float64",
        },
        "settings": {
            "loss_mode": "k1",
            "loss_max_clamp": 10.0,
            "policy_loss_mode": "vanilla",
            "clip_ratio_low": 0.2,
            "clip_ratio_high": 0.2,
            "dual_clip_ratio": 3.0,
            "loss_agg_mode": "token-mean",
        },
        "coverage": {
            "valid_tokens": int(mask.sum().item()),
            "masked_tokens": int((~mask).sum().item()),
            "ratios_below_clip": int(
                ((torch.exp(initial_student - behavior) < 0.8) & mask).sum().item()
            ),
            "ratios_inside_clip": int(
                (
                    (torch.exp(initial_student - behavior) >= 0.8)
                    & (torch.exp(initial_student - behavior) <= 1.2)
                    & mask
                ).sum().item()
            ),
            "ratios_above_clip": int(
                ((torch.exp(initial_student - behavior) > 1.2) & mask).sum().item()
            ),
            "negative_advantage_tokens": int(
                ((upstream_advantage < 0) & mask).sum().item()
            ),
            "positive_advantage_tokens": int(
                ((upstream_advantage > 0) & mask).sum().item()
            ),
        },
        "comparison": {
            "upstream_loss": float(upstream_loss.detach()),
            "local_loss": float(local_loss.detach()),
            "scalar_abs_error": scalar_abs_error,
            "gradient_max_abs_error": gradient_max_abs_error,
            "scalar_matches": scalar_matches,
            "gradient_matches": gradient_matches,
            "masked_gradient_zero": masked_gradient_zero,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verl-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_fidelity(args.verl_checkout)
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
