#!/usr/bin/env python3
"""Seal the terminal pre-optimization TRL text-field failure of a retry job."""

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
    "Adding EOS to train dataset",
    "29434/29434",
    "KeyError: 'text'",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--slurm-log", type=Path, required=True)
    parser.add_argument("--auditor-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    accounting = exact_job_accounting(args.job_id)
    if accounting != {
        "job_id": args.job_id,
        "state": "FAILED",
        "exit_code": "1:0",
    }:
        raise RuntimeError(f"unexpected terminal accounting: {accounting}")
    run_dir = args.run_dir.resolve()
    custody_path = run_dir / "custody_start.json"
    exit_path = run_dir / "training_exit.json"
    custody = load_object(custody_path, "custody start")
    exit_receipt = load_object(exit_path, "training exit")
    if custody.get("slurm_job_id") != args.job_id:
        raise RuntimeError("custody Slurm identity drifted")
    if exit_receipt.get("returncode") != 1:
        raise RuntimeError("training return code is not one")
    if (run_dir / "in_job_gate.json").exists():
        raise RuntimeError("failed retry unexpectedly produced an in-job pass gate")
    if list((run_dir / "training").rglob("checkpoint-*")):
        raise RuntimeError("failed retry unexpectedly produced a checkpoint")
    log_text = args.slurm_log.read_text(encoding="utf-8", errors="replace")
    missing = [signature for signature in SIGNATURES if signature not in log_text]
    if missing:
        raise RuntimeError(f"Slurm log lacks registered signatures: {missing}")

    receipt = {
        "schema_version": 2,
        "artifact_type": "opd_positive_control_one_step_terminal_failure",
        "campaign_id": "opd_identifiability_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "failed_before_optimization",
        "decision": "TRAINER_CHATML_SOURCE_FIELD_MISSING",
        "failure_phase": "trl_sft_trainer_preprocessing_before_optimizer_creation",
        "rows_loaded": 29_434,
        "optimizer_steps": 0,
        "checkpoint_created": False,
        "opd_result_created": False,
        "auditor_commit": args.auditor_commit,
        "slurm": accounting,
        "custody_start": str(custody_path),
        "custody_start_sha256": sha256(custody_path),
        "training_exit": str(exit_path),
        "training_exit_sha256": sha256(exit_path),
        "slurm_log": str(args.slurm_log.resolve()),
        "slurm_log_sha256": sha256(args.slurm_log.resolve()),
        "failure_signatures": list(SIGNATURES),
        "repair_boundary": (
            "Preserve raw bytes and the previous projection; add the exact source "
            "conversations field required by pinned TRL 0.26 ChatML conversion; "
            "audit full conversion, tokenization, and the pinned custom collator on "
            "CPU before any newly preregistered one-step retry."
        ),
    }
    write_exclusive(args.output.resolve(), receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
