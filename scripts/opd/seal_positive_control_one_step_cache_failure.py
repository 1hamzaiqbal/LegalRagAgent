#!/usr/bin/env python3
"""Seal the terminal pre-optimization runtime-cache quota failure of a retry job."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

try:
    from scripts.opd.positive_control_one_step import load_object, sha256, write_exclusive
    from scripts.opd.positive_control_one_step_terminal_audit import exact_job_accounting
except ModuleNotFoundError:  # Direct execution by absolute path on EIT.
    from positive_control_one_step import load_object, sha256, write_exclusive
    from positive_control_one_step_terminal_audit import exact_job_accounting


SIGNATURES = (
    "Disk quota exceeded",
    "/home/compute/hiqbal/.cache/vllm/torch_compile_cache",
    "/home/compute/hiqbal/.triton/autotune",
)


def build_receipt(
    *,
    job_id: str,
    run_dir: Path,
    slurm_log: Path,
    auditor_commit: str,
    accounting: dict,
) -> dict:
    expected_accounting = {"job_id": job_id, "state": "FAILED", "exit_code": "1:0"}
    if accounting != expected_accounting:
        raise RuntimeError(f"unexpected terminal accounting: {accounting}")

    run_dir = run_dir.resolve()
    slurm_log = slurm_log.resolve()
    custody_path = run_dir / "custody_start.json"
    exit_path = run_dir / "training_exit.json"
    custody = load_object(custody_path, "custody start")
    exit_receipt = load_object(exit_path, "training exit")
    if custody.get("slurm_job_id") != job_id:
        raise RuntimeError("custody Slurm identity drifted")
    if exit_receipt.get("slurm_job_id") != job_id:
        raise RuntimeError("training-exit Slurm identity drifted")
    if exit_receipt.get("returncode") != 1:
        raise RuntimeError("training return code is not one")
    if (run_dir / "in_job_gate.json").exists():
        raise RuntimeError("failed retry unexpectedly produced an in-job pass gate")
    training_root = run_dir / "training"
    if training_root.exists() and list(training_root.rglob("checkpoint-*")):
        raise RuntimeError("failed retry unexpectedly produced a checkpoint")

    log_text = slurm_log.read_text(encoding="utf-8", errors="replace")
    missing = [signature for signature in SIGNATURES if signature not in log_text]
    if missing:
        raise RuntimeError(f"Slurm log lacks registered signatures: {missing}")

    return {
        "schema_version": 2,
        "artifact_type": "opd_positive_control_one_step_terminal_failure",
        "campaign_id": "opd_identifiability_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "failed_before_optimization",
        "decision": "RUNTIME_COMPILE_CACHE_HOME_QUOTA_EXCEEDED",
        "failure_phase": "vllm_torchinductor_profile_compile_before_optimizer_creation",
        "optimizer_steps": 0,
        "checkpoint_created": False,
        "opd_result_created": False,
        "auditor_commit": auditor_commit,
        "slurm": accounting,
        "custody_start": str(custody_path),
        "custody_start_sha256": sha256(custody_path),
        "training_exit": str(exit_path),
        "training_exit_sha256": sha256(exit_path),
        "slurm_log": str(slurm_log),
        "slurm_log_sha256": sha256(slurm_log),
        "failure_signatures": list(SIGNATURES),
        "repair_boundary": (
            "Preserve the audited data, model revision, upstream objective, ordered "
            "rows, hardware, and one-step recipe. A newly preregistered retry may "
            "only redirect XDG, vLLM, TorchInductor, Triton, CUDA, Torch, and temporary "
            "runtime caches into a per-job EIT scratch namespace and must record and "
            "validate every resolved cache path before trainer launch."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--slurm-log", type=Path, required=True)
    parser.add_argument("--auditor-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    receipt = build_receipt(
        job_id=args.job_id,
        run_dir=args.run_dir,
        slurm_log=args.slurm_log,
        auditor_commit=args.auditor_commit,
        accounting=exact_job_accounting(args.job_id),
    )
    write_exclusive(args.output.resolve(), receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
