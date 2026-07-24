#!/usr/bin/env python3
"""Independently seal a terminal 4,096-token one-step OPSD diagnostic."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

try:
    from scripts.opd.positive_control_long_step import RUN_CONFIG
    from scripts.opd.positive_control_one_step import hash_tree, load_object, sha256, write_exclusive
    from scripts.opd.positive_control_one_step_terminal_audit import exact_job_accounting
except ModuleNotFoundError:  # Direct execution by absolute path on EIT.
    from positive_control_long_step import RUN_CONFIG  # type: ignore[no-redef]
    from positive_control_one_step import hash_tree, load_object, sha256, write_exclusive
    from positive_control_one_step_terminal_audit import exact_job_accounting


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
    config = load_object(config_path, "long-context one-step preregistration")
    accounting = exact_job_accounting(args.job_id)
    gate = config["pass_gate"]
    if accounting != {
        "job_id": args.job_id,
        "state": gate["terminal_slurm_state"],
        "exit_code": gate["terminal_exit_code"],
    }:
        raise RuntimeError(f"long-context terminal accounting failed: {accounting}")

    run_dir = args.run_dir.resolve()
    in_job_path = run_dir / "in_job_gate.json"
    custody_path = run_dir / "custody_start.json"
    exit_path = run_dir / "training_exit.json"
    in_job = load_object(in_job_path, "long-context in-job gate")
    if in_job.get("status") != "passed" or in_job.get("decision") != (
        "LONG4096_ONE_STEP_UPDATE_AND_TRAJECTORY_GATE_PASSED_IN_JOB"
    ):
        raise RuntimeError("long-context in-job gate is not passing")
    if in_job.get("repository_commit") != args.repository_commit:
        raise RuntimeError("long-context producer commit drifted")
    if in_job.get("slurm_job_id") != args.job_id:
        raise RuntimeError("long-context Slurm identity drifted")
    if in_job.get("scientific_claim") != "none_plumbing_and_length_custody_only":
        raise RuntimeError("long-context in-job gate improperly claims a result")
    if in_job.get("custody_start_sha256") != sha256(custody_path):
        raise RuntimeError("long-context custody receipt changed after training")
    if in_job.get("training_exit_sha256") != sha256(exit_path):
        raise RuntimeError("long-context exit receipt changed after training")

    audit = in_job.get("audit", {})
    trajectory = audit.get("trajectory_audit", {})
    if trajectory.get("status") != "passed" or trajectory.get(
        "trajectory_count"
    ) != gate["expected_trajectory_count"]:
        raise RuntimeError("long-context trajectory audit is not passing")
    if trajectory.get("at_cap_trajectory_count", 10**9) > gate[
        "maximum_at_cap_trajectory_count"
    ]:
        raise RuntimeError("long-context cap gate drifted")
    recorded = audit.get("training_artifact_files")
    if not isinstance(recorded, list):
        raise RuntimeError("long-context in-job gate lacks a training-tree manifest")
    training_root = run_dir / "training" / RUN_CONFIG
    if hash_tree(training_root) != recorded:
        raise RuntimeError("long-context training tree changed after the in-job gate")
    slurm_log = args.slurm_log.resolve()
    if not slurm_log.is_file():
        raise RuntimeError("long-context Slurm log is missing")

    receipt = {
        "schema_version": 1,
        "artifact_type": "opd_positive_control_long_step_terminal_audit",
        "campaign_id": config["campaign_id"],
        "stage_id": config["stage_id"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit": args.repository_commit,
        "slurm": accounting,
        "status": "passed",
        "decision": "LONG4096_ONE_STEP_FULL_CUSTODY_PASSED",
        "scientific_claim": "none_plumbing_and_length_custody_only",
        "preregistration": str(config_path),
        "preregistration_sha256": sha256(config_path),
        "in_job_gate": str(in_job_path),
        "in_job_gate_sha256": sha256(in_job_path),
        "custody_start_sha256": sha256(custody_path),
        "training_exit_sha256": sha256(exit_path),
        "training_artifact_file_count": len(recorded),
        "trajectory_audit": trajectory,
        "slurm_log": str(slurm_log),
        "slurm_log_sha256": sha256(slurm_log),
        "release_state": "eligible_to_preregister_long4096_train100_not_queued",
    }
    write_exclusive(args.output.resolve(), receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
