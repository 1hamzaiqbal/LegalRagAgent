#!/usr/bin/env python3
"""Seal the terminal pre-backward full-vocabulary KL OOM of a one-step job."""

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
    "vLLM generation done",
    "torch.OutOfMemoryError: CUDA out of memory",
    "generalized_jsd_loss",
    "F.kl_div(student_log_probs, teacher_log_probs",
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
    if custody.get("slurm_job_id") != job_id or exit_receipt.get("slurm_job_id") != job_id:
        raise RuntimeError("Slurm identity drifted")
    if exit_receipt.get("returncode") != 1:
        raise RuntimeError("training return code is not one")
    cache = custody.get("runtime_cache", {})
    if cache.get("decision") != "PER_JOB_EIT_RUNTIME_CACHE_PATHS_VALIDATED":
        raise RuntimeError("the predecessor did not pass runtime-cache custody")
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
        "status": "failed_before_backward_or_optimization",
        "decision": "A6000_FULL_VOCAB_MICROBATCH4_OOM",
        "failure_phase": "full_vocabulary_teacher_to_student_kl_before_backward",
        "student_generation_completed": True,
        "backward_completed": False,
        "optimizer_steps": 0,
        "checkpoint_created": False,
        "opd_result_created": False,
        "auditor_commit": auditor_commit,
        "slurm": accounting,
        "custody_start": str(custody_path),
        "custody_start_sha256": sha256(custody_path),
        "runtime_cache_custody": cache,
        "training_exit": str(exit_path),
        "training_exit_sha256": sha256(exit_path),
        "slurm_log": str(slurm_log),
        "slurm_log_sha256": sha256(slurm_log),
        "failure_signatures": list(SIGNATURES),
        "repair_boundary": (
            "Preserve the 1,024-token completion cap, effective batch size 32, "
            "four A6000 GPUs, model, data, objective, decoding, LoRA, seed, and "
            "per-job EIT cache policy. A newly preregistered memory retry may only "
            "change per-device microbatch from 4 to 2 and gradient accumulation "
            "from 2 to 4, retaining the same examples per optimizer update."
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
