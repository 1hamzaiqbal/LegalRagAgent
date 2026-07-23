#!/usr/bin/env python3
"""Independently seal a terminal one-step OPSD diagnostic after Slurm exits."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

try:
    from scripts.opd.positive_control_one_step import (
        RUN_CONFIG,
        hash_tree,
        load_object,
        sha256,
        write_exclusive,
    )
except ModuleNotFoundError:  # Direct execution by absolute path on EIT.
    from positive_control_one_step import (  # type: ignore[no-redef]
        RUN_CONFIG,
        hash_tree,
        load_object,
        sha256,
        write_exclusive,
    )


def exact_job_accounting(job_id: str) -> dict[str, str]:
    output = subprocess.check_output(
        [
            "sacct",
            "-j",
            job_id,
            "--format=JobIDRaw,State,ExitCode",
            "--parsable2",
            "--noheader",
        ],
        text=True,
    )
    rows = []
    for line in output.splitlines():
        fields = line.strip().split("|")
        if len(fields) >= 3 and fields[0] == job_id:
            rows.append(
                {"job_id": fields[0], "state": fields[1], "exit_code": fields[2]}
            )
    if len(rows) != 1:
        raise RuntimeError(f"expected one exact sacct row for {job_id}, got {rows}")
    return rows[0]


def verify_recorded_tree(root: Path, recorded: list[dict]) -> None:
    observed = hash_tree(root)
    if observed != recorded:
        raise RuntimeError("training artifact tree changed after the in-job gate")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--slurm-log", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repository-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_object(config_path, "one-step preregistration")
    gate = config["pass_gate"]
    accounting = exact_job_accounting(args.job_id)
    if accounting["state"] != gate["terminal_slurm_state"]:
        raise RuntimeError(f"terminal Slurm state is {accounting['state']}")
    if accounting["exit_code"] != gate["terminal_exit_code"]:
        raise RuntimeError(f"terminal Slurm exit code is {accounting['exit_code']}")

    run_dir = args.run_dir.resolve()
    in_job_path = run_dir / "in_job_gate.json"
    custody_path = run_dir / "custody_start.json"
    exit_path = run_dir / "training_exit.json"
    in_job = load_object(in_job_path, "in-job gate")
    if in_job.get("status") != "passed":
        raise RuntimeError("in-job diagnostic gate is not passing")
    if in_job.get("decision") != "ONE_STEP_UPDATE_DIAGNOSTIC_PASSED_IN_JOB":
        raise RuntimeError("unexpected in-job diagnostic decision")
    if in_job.get("repository_commit") != args.repository_commit:
        raise RuntimeError("in-job gate repository commit drifted")
    if in_job.get("slurm_job_id") != args.job_id:
        raise RuntimeError("in-job gate Slurm identity drifted")
    if in_job.get("scientific_claim") != "none_plumbing_only":
        raise RuntimeError("in-job gate improperly claims a scientific result")
    if in_job.get("custody_start_sha256") != sha256(custody_path):
        raise RuntimeError("custody-start receipt changed after training")
    if in_job.get("training_exit_sha256") != sha256(exit_path):
        raise RuntimeError("training-exit receipt changed after training")

    recorded = in_job.get("audit", {}).get("training_artifact_files")
    if not isinstance(recorded, list):
        raise RuntimeError("in-job gate lacks the training artifact manifest")
    training_root = run_dir / "training" / RUN_CONFIG
    verify_recorded_tree(training_root, recorded)
    if not args.slurm_log.is_file():
        raise RuntimeError("Slurm stdout log is missing")

    receipt = {
        "schema_version": 1,
        "artifact_type": "opd_positive_control_one_step_terminal_audit",
        "campaign_id": config["campaign_id"],
        "stage_id": config["stage_id"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit": args.repository_commit,
        "slurm": accounting,
        "status": "passed",
        "decision": "ONE_STEP_FULL_CUSTODY_PASSED",
        "scientific_claim": "none_plumbing_only",
        "preregistration": str(config_path),
        "preregistration_sha256": sha256(config_path),
        "in_job_gate": str(in_job_path),
        "in_job_gate_sha256": sha256(in_job_path),
        "custody_start_sha256": sha256(custody_path),
        "training_exit_sha256": sha256(exit_path),
        "training_artifact_file_count": len(recorded),
        "slurm_log": str(args.slurm_log.resolve()),
        "slurm_log_sha256": sha256(args.slurm_log.resolve()),
        "release_state": "eligible_to_preregister_100_step_not_authorized_or_queued",
    }
    write_exclusive(args.output.resolve(), receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
