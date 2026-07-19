#!/usr/bin/env python3
"""Plan the complete O teacher evaluation from one exact timing shard.

The planner is deliberately narrower than a generic throughput calculator.  It
accepts only a base-model, one-shard, schema-v2 exact-environment evaluation of
the O ``teacher_gap_dev`` prefix.  The timing artifact, its post-promotion
custody companion, the complete task file, Slurm accounting, and stdout are all
reopened and hashed before a plan is created.

The Slurm input is the raw, headerless, pipe-delimited output of::

    sacct -X -n -P -j JOB_ID \
      --format=JobIDRaw,JobName,State,ExitCode,ElapsedRaw,AllocTRES,StdOut

Exactly one non-array ``opd_math_eval`` row is accepted.  Its numeric job ID
must occur in the canonical stdout filename, its ``StdOut`` field must equal
the supplied stdout path, and ``AllocTRES`` must record exactly one GPU.  The
raw capture and stdout are both hashed into the plan.

For candidate shard count ``S`` the conservative projection is exactly

``ceil(ceil(total_records / S) / timing_records) * ElapsedRaw * safety_factor``.

The primary plan is deliberately non-configurable: 4,585 total O records, a
32-record prefix, 1.25 safety factor, an 18-hour shard cap, and four-way array
concurrency.  Generic geometry remains available only as an explicitly marked
diagnostic API for tests and sensitivity analysis.  The output is created with
exclusive-create semantics and is never overwritten.  It authorizes only the
launch geometry consumed by the production wrapper, never a scientific result.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping

try:
    from .data_contract import iter_jsonl
    from .evaluate_math import (
        EVALUATION_CONTRACT,
        evaluation_environment_contract_unchanged,
        git_identity,
        post_promotion_custody_path,
        sha256_file,
    )
    from .merge_evaluations import validate_shard_artifact
    from .quality_gates import write_text_exclusive_fsync
    from .tokenizer_contract import canonical_sha256
except ImportError:
    from data_contract import iter_jsonl  # type: ignore
    from evaluate_math import (  # type: ignore
        EVALUATION_CONTRACT,
        evaluation_environment_contract_unchanged,
        git_identity,
        post_promotion_custody_path,
        sha256_file,
    )
    from merge_evaluations import validate_shard_artifact  # type: ignore
    from quality_gates import write_text_exclusive_fsync  # type: ignore
    from tokenizer_contract import canonical_sha256  # type: ignore


PLAN_KIND = "opd_math_o_evaluation_shard_plan_v1"
DIAGNOSTIC_PLAN_KIND = "opd_math_o_evaluation_shard_plan_diagnostic_v1"
DEFAULT_TOTAL_RECORDS = 4585
PRIMARY_TIMING_RECORDS = 32
DEFAULT_MAX_SHARD_SECONDS = 64_800
DEFAULT_MAX_CONCURRENT = 4
DEFAULT_SAFETY_FACTOR = Decimal("1.25")
EXPECTED_MODEL = "Qwen/Qwen3-8B"
EXPECTED_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
EXPECTED_SOURCE = "O"
EXPECTED_ROLE = "teacher_gap_dev"
EXPECTED_SAMPLES_PER_PROBLEM = 4
EXPECTED_DECODING = {
    "thinking": False,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "max_new_tokens": 1024,
    "seed": 0,
}
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
JOB_ID = re.compile(r"[1-9][0-9]*")
SACCT_JOB_NAME = "opd_math_eval"
SACCT_FIELDS = (
    "JobIDRaw",
    "JobName",
    "State",
    "ExitCode",
    "ElapsedRaw",
    "AllocTRES",
    "StdOut",
)
PLANNER_PATH = Path(__file__).resolve()


def _regular_file(path: Path, label: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink() or raw.parent.is_symlink() or not raw.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file: {raw}")
    return raw.resolve()


def _json_object(path: Path, label: str) -> dict[str, Any]:
    path = _regular_file(path, label)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not a UTF-8 JSON object: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain one JSON object: {path}")
    return payload


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _decimal(value: Any, label: str) -> Decimal:
    try:
        numeric = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{label} must be a finite decimal") from exc
    if not numeric.is_finite():
        raise ValueError(f"{label} must be a finite decimal")
    return numeric


def _json_number(value: Decimal) -> int | float:
    integral = value.to_integral_value()
    return int(integral) if value == integral else float(value)


def _allocated_gpu_count(raw: str) -> int:
    matches = []
    for field in raw.split(","):
        match = re.fullmatch(r"gres/gpu(?::[^=,]+)?=([0-9]+)", field)
        if match is not None:
            matches.append(match.group(1))
    counts = {int(value) for value in matches}
    if not matches or len(counts) != 1:
        raise ValueError("raw Slurm accounting lacks one consistent GPU allocation")
    return counts.pop()


def _validate_sacct_raw(sacct_path: Path, stdout_path: Path) -> tuple[dict[str, Any], Path]:
    sacct_path = _regular_file(sacct_path, "raw Slurm timing accounting")
    try:
        lines = [line for line in sacct_path.read_text(encoding="utf-8").splitlines() if line]
    except UnicodeDecodeError as exc:
        raise ValueError("raw Slurm timing accounting is not UTF-8") from exc
    if len(lines) != 1:
        raise ValueError("raw Slurm timing accounting must contain exactly one data row")
    values = lines[0].split("|")
    if len(values) != len(SACCT_FIELDS):
        raise ValueError(
            "raw Slurm timing accounting must use the exact headerless field order: "
            + ",".join(SACCT_FIELDS)
        )
    row = dict(zip(SACCT_FIELDS, values))
    job_id = row["JobIDRaw"]
    if JOB_ID.fullmatch(job_id) is None:
        raise ValueError("raw Slurm timing accounting requires one non-array numeric job ID")
    if row["JobName"] != SACCT_JOB_NAME:
        raise ValueError("raw Slurm timing accounting has the wrong job name")
    if row["State"] != "COMPLETED" or row["ExitCode"] != "0:0":
        raise ValueError("timing job must have Slurm state COMPLETED and exit code 0:0")
    try:
        elapsed = int(row["ElapsedRaw"])
    except ValueError as exc:
        raise ValueError("raw Slurm ElapsedRaw must be an integer") from exc
    _positive_int(elapsed, "Slurm ElapsedRaw")
    allocated_gpus = _allocated_gpu_count(row["AllocTRES"])
    if allocated_gpus != 1:
        raise ValueError("timing evaluation must account for exactly one allocated GPU")
    stdout_path = _regular_file(stdout_path, "timing stdout")
    stdout_template = row["StdOut"]
    if "%" in stdout_template:
        if stdout_template.count("%j") != 1 or "%" in stdout_template.replace("%j", ""):
            raise ValueError("raw Slurm stdout uses an unsupported template token")
        stdout_template = stdout_template.replace("%j", job_id)
    recorded_stdout = Path(stdout_template).expanduser()
    if not recorded_stdout.is_absolute() or recorded_stdout.resolve() != stdout_path:
        raise ValueError("timing stdout path differs from raw Slurm accounting")
    if stdout_path.name != f"opd_math_eval_{job_id}.out":
        raise ValueError("timing stdout filename does not bind the Slurm job ID")
    return {
        "job_id": job_id,
        "job_name": row["JobName"],
        "state": row["State"],
        "exit_code": row["ExitCode"],
        "elapsed_raw_seconds": elapsed,
        "allocated_gpus": allocated_gpus,
        "alloc_tres": row["AllocTRES"],
        "stdout_path": str(stdout_path),
        "raw_capture_path": str(sacct_path),
        "raw_capture_sha256": sha256_file(sacct_path),
    }, stdout_path


def _validate_stdout(
    stdout_path: Path, summary: Mapping[str, Any], output_dir: Path
) -> None:
    try:
        lines = stdout_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("timing stdout is not valid UTF-8") from exc
    expected_pass = f"PASS evaluation artifact only; no gate inferred: {output_dir.resolve()}"
    if lines.count(expected_pass) != 1:
        raise ValueError("timing stdout lacks one exact successful evaluation PASS line")
    matching_summaries = 0
    for line in lines:
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        matching_summaries += int(payload == dict(summary))
    if matching_summaries != 1:
        raise ValueError("timing stdout does not contain exactly one exact summary payload")


def choose_shard_geometry(
    *,
    total_records: int,
    timing_records: int,
    elapsed_raw_seconds: int,
    safety_factor: Decimal,
    max_shard_seconds: int,
    max_concurrent: int,
) -> dict[str, Any]:
    """Return the smallest conservative shard geometry satisfying the limit."""

    total_records = _positive_int(total_records, "total records")
    timing_records = _positive_int(timing_records, "timing records")
    elapsed_raw_seconds = _positive_int(elapsed_raw_seconds, "Slurm ElapsedRaw")
    max_shard_seconds = _positive_int(max_shard_seconds, "maximum shard seconds")
    max_concurrent = _positive_int(max_concurrent, "maximum concurrency")
    safety_factor = _decimal(safety_factor, "safety factor")
    if safety_factor < 1:
        raise ValueError("safety factor must be at least 1.0")
    if total_records <= timing_records:
        raise ValueError("complete evaluation must contain more records than its timing prefix")

    selected: tuple[int, int, int, Decimal] | None = None
    for shard_count in range(1, total_records + 1):
        records_per_shard = (total_records + shard_count - 1) // shard_count
        timing_blocks = (records_per_shard + timing_records - 1) // timing_records
        projected = Decimal(timing_blocks * elapsed_raw_seconds) * safety_factor
        if projected <= Decimal(max_shard_seconds):
            selected = (shard_count, records_per_shard, timing_blocks, projected)
            break
    if selected is None:
        minimum = Decimal(elapsed_raw_seconds) * safety_factor
        raise ValueError(
            "no shard count can satisfy the runtime limit: even one timing block "
            f"projects to {minimum} seconds"
        )

    shard_count, records_per_shard, timing_blocks, projected = selected
    throttle = min(max_concurrent, shard_count)
    concurrency_waves = math.ceil(shard_count / throttle)
    projected_gpu_seconds = projected * shard_count
    projected_wall_seconds = projected * concurrency_waves
    array_spec = f"0-{shard_count - 1}%{throttle}"
    previous_candidate = None
    if shard_count > 1:
        previous_count = shard_count - 1
        previous_records = (total_records + previous_count - 1) // previous_count
        previous_blocks = (previous_records + timing_records - 1) // timing_records
        previous_projected = (
            Decimal(previous_blocks * elapsed_raw_seconds) * safety_factor
        )
        previous_candidate = {
            "shard_count": previous_count,
            "records_per_shard_ceiling": previous_records,
            "timing_blocks_per_shard": previous_blocks,
            "projected_shard_seconds": _json_number(previous_projected),
            "exceeds_limit": previous_projected > Decimal(max_shard_seconds),
        }
    common_spec = {
        "array_spec": array_spec,
        "slurm_array_argument": f"--array={array_spec}",
        "shard_count": shard_count,
        "shard_index_start": 0,
        "shard_index_stop": shard_count - 1,
        "max_concurrent_tasks": throttle,
        "requested_max_concurrent_tasks": max_concurrent,
        "concurrency_waves": concurrency_waves,
        "gpus_per_task": 1,
        "records_per_shard_ceiling": records_per_shard,
        "timing_blocks_per_shard": timing_blocks,
        "projected_shard_seconds": _json_number(projected),
        "projected_gpu_seconds": _json_number(projected_gpu_seconds),
        "projected_wall_seconds_if_waves_serialize": _json_number(
            projected_wall_seconds
        ),
    }
    return {
        "selected_shard_count": shard_count,
        "selected_throttle": throttle,
        "smallest_passing_shard_count": True,
        "immediately_previous_candidate": previous_candidate,
        "base": dict(common_spec),
        "trained": dict(common_spec),
        "base_trained_specs_identical": True,
        "common_spec_sha256": canonical_sha256(common_spec),
        "projected_gpu_seconds_two_arms": _json_number(projected_gpu_seconds * 2),
        "projected_wall_seconds_two_arms_if_sequential": _json_number(
            projected_wall_seconds * 2
        ),
    }


def _require_primary_constraints(
    *,
    total_records: int,
    timing_records: int,
    safety_factor: Decimal,
    max_shard_seconds: int,
    max_concurrent: int,
) -> None:
    expected = {
        "total_records": DEFAULT_TOTAL_RECORDS,
        "timing_records": PRIMARY_TIMING_RECORDS,
        "safety_factor": DEFAULT_SAFETY_FACTOR,
        "max_shard_seconds": DEFAULT_MAX_SHARD_SECONDS,
        "max_concurrent": DEFAULT_MAX_CONCURRENT,
    }
    actual = {
        "total_records": total_records,
        "timing_records": timing_records,
        "safety_factor": _decimal(safety_factor, "safety factor"),
        "max_shard_seconds": max_shard_seconds,
        "max_concurrent": max_concurrent,
    }
    if actual != expected:
        raise ValueError(
            "primary O timing plan requires the canonical fixed constraints: "
            f"expected={expected}, actual={actual}"
        )


def build_plan(
    *,
    timing_summary: Path,
    timing_companion: Path,
    task_file: Path,
    sacct_raw: Path,
    stdout: Path,
    total_records: int = DEFAULT_TOTAL_RECORDS,
    safety_factor: Decimal = DEFAULT_SAFETY_FACTOR,
    max_shard_seconds: int = DEFAULT_MAX_SHARD_SECONDS,
    max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    primary_contract: bool = True,
) -> dict[str, Any]:
    """Validate all timing evidence and return one immutable plan payload."""

    timing_summary = _regular_file(timing_summary, "timing shard summary")
    timing_companion = _regular_file(timing_companion, "timing custody companion")
    task_file = _regular_file(task_file, "complete O evaluation task file")
    sacct_raw = _regular_file(sacct_raw, "raw Slurm timing accounting")
    accounting, stdout = _validate_sacct_raw(sacct_raw, stdout)

    validated = validate_shard_artifact(timing_summary, task_file=task_file)
    summary = validated["summary"]
    contract = validated["contract"]
    companion = validated["post_promotion_companion"]
    if contract.get("contract") != EVALUATION_CONTRACT or companion is None:
        raise ValueError("timing shard must use the schema-v2 exact-environment contract")
    environment_contract = contract.get("code", {}).get("environment_contract")
    if not isinstance(environment_contract, dict) or not evaluation_environment_contract_unchanged(
        environment_contract
    ):
        raise ValueError("timing planner requires the unchanged exact train environment")
    expected_companion = post_promotion_custody_path(timing_summary.parent).resolve()
    if timing_companion != expected_companion or companion["path"].resolve() != timing_companion:
        raise ValueError("supplied timing companion is not the shard's exact custody companion")
    if sha256_file(timing_companion) != companion["sha256"]:
        raise ValueError("timing custody companion changed after shard validation")

    planner_git = git_identity()
    commit = contract.get("code", {}).get("git_commit")
    if (
        planner_git.get("worktree_clean") is not True
        or not isinstance(commit, str)
        or HEX40.fullmatch(commit) is None
        or planner_git.get("commit") != commit
    ):
        raise ValueError(
            "planner must run from the same clean immutable commit as the timing shard"
        )
    if summary.get("model") != EXPECTED_MODEL or summary.get("model_revision") != EXPECTED_REVISION:
        raise ValueError("timing shard is not the pinned Qwen3-8B teacher base model")
    if summary.get("adapter") is not None or summary.get("adapter_tree_sha256") is not None:
        raise ValueError("timing shard must be an unadapted base-model evaluation")
    if summary.get("samples_per_problem") != EXPECTED_SAMPLES_PER_PROBLEM:
        raise ValueError("timing shard has the wrong samples-per-problem contract")
    if summary.get("decoding") != EXPECTED_DECODING:
        raise ValueError("timing shard has the wrong scientific decoding contract")
    shard = summary.get("shard")
    timing_records = summary.get("records")
    if (
        not isinstance(shard, dict)
        or shard.get("shard_count") != 1
        or shard.get("shard_index") != 0
        or shard.get("record_start") != 0
        or shard.get("record_stop") != timing_records
        or summary.get("eligible_records") != timing_records
    ):
        raise ValueError("timing artifact must be one complete shard of its exact prefix")
    timing_records = _positive_int(timing_records, "timing prefix records")

    task_rows = list(iter_jsonl(task_file))
    total_records = _positive_int(total_records, "total records")
    if len(task_rows) != total_records:
        raise ValueError(
            "complete task-file cardinality differs from the registered total: "
            f"expected={total_records}, actual={len(task_rows)}"
        )
    if any(row.get("source") != EXPECTED_SOURCE for row in task_rows):
        raise ValueError("complete timing task must contain only source O")
    if any(row.get("role") != EXPECTED_ROLE for row in task_rows):
        raise ValueError("complete timing task must contain only teacher_gap_dev rows")
    if total_records <= timing_records:
        raise ValueError("timing artifact must be a strict prefix of the complete task")
    if primary_contract:
        _require_primary_constraints(
            total_records=total_records,
            timing_records=timing_records,
            safety_factor=safety_factor,
            max_shard_seconds=max_shard_seconds,
            max_concurrent=max_concurrent,
        )

    _validate_stdout(stdout, summary, timing_summary.parent)
    geometry = choose_shard_geometry(
        total_records=total_records,
        timing_records=timing_records,
        elapsed_raw_seconds=accounting["elapsed_raw_seconds"],
        safety_factor=safety_factor,
        max_shard_seconds=max_shard_seconds,
        max_concurrent=max_concurrent,
    )
    generation_latency = summary.get("total_generation_latency_seconds")
    if type(generation_latency) not in (int, float) or not math.isfinite(
        float(generation_latency)
    ) or float(generation_latency) <= 0:
        raise ValueError("timing shard has invalid generation-latency accounting")
    if float(generation_latency) > accounting["elapsed_raw_seconds"]:
        raise ValueError("generation latency cannot exceed Slurm ElapsedRaw")

    payload: dict[str, Any] = {
        "schema_version": 1,
        "plan_kind": PLAN_KIND if primary_contract else DIAGNOSTIC_PLAN_KIND,
        "scientific_authorization": False,
        "authorizes_primary_o_launch_geometry": primary_contract,
        "claim_boundary": (
            "This plan authorizes only the O base/trained array geometry when the production "
            "wrapper revalidates it. It is not evaluation evidence and cannot authorize a "
            "teacher, checkpoint merge, or student training."
            if primary_contract
            else "This diagnostic geometry is non-authorizing operator guidance only."
        ),
        "evaluation": {
            "source": EXPECTED_SOURCE,
            "role": EXPECTED_ROLE,
            "model": EXPECTED_MODEL,
            "model_revision": EXPECTED_REVISION,
            "adapter": None,
            "samples_per_problem": EXPECTED_SAMPLES_PER_PROBLEM,
            "decoding": dict(EXPECTED_DECODING),
            "evaluation_contract": EVALUATION_CONTRACT,
        },
        "formula": (
            "ceil(ceil(total_records / shard_count) / timing_records) * "
            "elapsed_raw_timing_seconds * safety_factor"
        ),
        "inputs": {
            "timing_summary": {
                "path": str(timing_summary),
                "sha256": validated["summary_sha256"],
                "evaluation_contract_sha256": summary[
                    "evaluation_contract_sha256"
                ],
                "output_tree_sha256": companion["tree_sha256"],
            },
            "timing_samples": {
                "path": str(validated["samples_path"]),
                "sha256": validated["samples_sha256"],
            },
            "timing_companion": {
                "path": str(timing_companion),
                "sha256": companion["sha256"],
            },
            "task_file": {
                "path": str(task_file),
                "sha256": sha256_file(task_file),
                "records": total_records,
            },
            "sacct_raw": {
                "path": str(sacct_raw),
                "sha256": sha256_file(sacct_raw),
                "field_order": list(SACCT_FIELDS),
            },
            "stdout": {
                "path": str(stdout),
                "sha256": sha256_file(stdout),
            },
        },
        "timing": {
            "job_id": accounting["job_id"],
            "job_name": accounting["job_name"],
            "state": accounting["state"],
            "exit_code": accounting["exit_code"],
            "allocated_gpus": accounting["allocated_gpus"],
            "timing_records": timing_records,
            "elapsed_raw_timing_seconds": accounting["elapsed_raw_seconds"],
            "total_generation_latency_seconds": generation_latency,
        },
        "constraints": {
            "total_records": total_records,
            "safety_factor": _json_number(
                _decimal(safety_factor, "safety factor")
            ),
            "max_shard_seconds": max_shard_seconds,
            "requested_max_concurrent_tasks": max_concurrent,
            "base_trained_geometry_must_match": True,
            "primary_constraints_fixed": primary_contract,
        },
        "array_plan": geometry,
        "code": {
            "git": planner_git,
            "planner_file": str(PLANNER_PATH),
            "planner_file_sha256": sha256_file(PLANNER_PATH),
            "timing_evaluator_file_sha256": contract["code"][
                "evaluator_file_sha256"
            ],
            "exact_environment_contract_sha256": canonical_sha256(
                contract["code"]["environment_contract"]
            ),
            "exact_environment_contract": contract["code"][
                "environment_contract"
            ],
        },
    }
    result = dict(payload)
    result["plan_payload_sha256"] = canonical_sha256(payload)
    return result


def write_plan_exclusive(output: Path, plan: Mapping[str, Any]) -> Path:
    """Write sorted JSON once; refuse every existing file or symlink."""

    return write_text_exclusive_fsync(
        output,
        json.dumps(dict(plan), indent=2, sort_keys=True) + "\n",
        label="shard plan",
    )


def load_primary_plan(path: Path) -> tuple[Path, dict[str, Any]]:
    """Reopen and self-hash one canonical primary geometry plan."""

    path = _regular_file(path, "primary O evaluation shard plan")
    plan = _json_object(path, "primary O evaluation shard plan")
    recorded_hash = plan.get("plan_payload_sha256")
    payload = dict(plan)
    payload.pop("plan_payload_sha256", None)
    if not isinstance(recorded_hash, str) or HEX64.fullmatch(recorded_hash) is None:
        raise ValueError("primary O evaluation plan lacks its payload SHA-256")
    if canonical_sha256(payload) != recorded_hash:
        raise ValueError("primary O evaluation plan payload hash does not verify")
    if (
        plan.get("schema_version") != 1
        or plan.get("plan_kind") != PLAN_KIND
        or plan.get("authorizes_primary_o_launch_geometry") is not True
        or plan.get("scientific_authorization") is not False
    ):
        raise ValueError("evaluation launch requires the canonical primary O plan")
    constraints = plan.get("constraints")
    if not isinstance(constraints, dict):
        raise ValueError("primary O evaluation plan lacks fixed constraints")
    _require_primary_constraints(
        total_records=constraints.get("total_records"),
        timing_records=plan.get("timing", {}).get("timing_records"),
        safety_factor=constraints.get("safety_factor"),
        max_shard_seconds=constraints.get("max_shard_seconds"),
        max_concurrent=constraints.get("requested_max_concurrent_tasks"),
    )
    array_plan = plan.get("array_plan")
    if (
        not isinstance(array_plan, dict)
        or array_plan.get("base") != array_plan.get("trained")
        or array_plan.get("base_trained_specs_identical") is not True
        or canonical_sha256(array_plan.get("base"))
        != array_plan.get("common_spec_sha256")
    ):
        raise ValueError("primary O evaluation plan does not bind one shared array geometry")
    return path, plan


def validate_launch_against_plan(
    *,
    plan_path: Path,
    arm: str,
    phase: str,
    source: str,
    role: str,
    model: str,
    model_revision: str,
    task_file: Path,
    max_records: int,
    shard_count: int,
    git_commit: str,
    train_freeze: Path,
    adapter: Path | None,
    array_task_count: int | None = None,
    array_task_min: int | None = None,
    array_task_max: int | None = None,
) -> dict[str, Any]:
    """Bind one production shard/merge launch to the immutable primary plan."""

    plan_path, plan = load_primary_plan(plan_path)
    if arm not in {"base", "trained"} or phase not in {"shard", "merge"}:
        raise ValueError("plan arm/phase must be base|trained and shard|merge")
    if source != EXPECTED_SOURCE or role != EXPECTED_ROLE:
        raise ValueError("primary O plan may only launch O teacher_skill_dev evaluation")
    evaluation = plan["evaluation"]
    if model != evaluation["model"] or model_revision != evaluation["model_revision"]:
        raise ValueError("evaluation model identity differs from the primary O plan")
    if max_records != 0:
        raise ValueError("primary O planned evaluation must use the complete role file")
    task_file = _regular_file(task_file, "planned O evaluation task file")
    task_binding = plan["inputs"]["task_file"]
    if (
        str(task_file) != task_binding.get("path")
        or sha256_file(task_file) != task_binding.get("sha256")
    ):
        raise ValueError("evaluation task identity differs from the primary O plan")
    if git_commit != plan["code"]["git"]["commit"]:
        raise ValueError("evaluation Git commit differs from the primary O plan")
    train_freeze = _regular_file(train_freeze, "planned evaluation train freeze")
    planned_freeze = plan["code"]["exact_environment_contract"]["train_freeze"]
    if (
        str(train_freeze) != planned_freeze.get("path")
        or sha256_file(train_freeze) != planned_freeze.get("sha256")
    ):
        raise ValueError("evaluation train freeze differs from the primary O plan")
    spec = plan["array_plan"][arm]
    if shard_count != spec["shard_count"]:
        raise ValueError("evaluation shard count differs from the primary O plan")
    if arm == "base":
        if adapter is not None:
            raise ValueError("base planned evaluation must not use an adapter")
        adapter_path = None
    else:
        if adapter is None:
            raise ValueError("trained planned evaluation requires the teacher adapter")
        adapter_path = Path(adapter).expanduser()
        if adapter_path.is_symlink() or not adapter_path.is_dir():
            raise ValueError("trained planned evaluation adapter must be a real directory")
        adapter_path = adapter_path.resolve()
        if not (adapter_path / "adapter_config.json").is_file():
            raise ValueError("trained planned evaluation adapter lacks adapter_config.json")
    if phase == "shard":
        expected = (spec["shard_count"], 0, spec["shard_index_stop"])
        actual = (array_task_count, array_task_min, array_task_max)
        if actual != expected:
            raise ValueError(
                "Slurm array geometry differs from the primary O plan: "
                f"expected={expected}, actual={actual}"
            )
    elif any(value is not None for value in (array_task_count, array_task_min, array_task_max)):
        raise ValueError("merge-phase plan validation must not claim shard-array geometry")
    return {
        "plan": str(plan_path),
        "plan_file_sha256": sha256_file(plan_path),
        "plan_payload_sha256": plan["plan_payload_sha256"],
        "arm": arm,
        "phase": phase,
        "git_commit": git_commit,
        "task_file_sha256": task_binding["sha256"],
        "train_freeze_sha256": planned_freeze["sha256"],
        "array_spec": spec["array_spec"],
        "shard_count": shard_count,
        "adapter": None if adapter_path is None else str(adapter_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", help="create the fixed primary O plan")
    create.add_argument("--timing-summary", type=Path, required=True)
    create.add_argument("--timing-companion", type=Path, required=True)
    create.add_argument("--task-file", type=Path, required=True)
    create.add_argument("--sacct-raw", type=Path, required=True)
    create.add_argument("--stdout", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)

    validate = commands.add_parser(
        "validate-launch", help="bind a production shard or merge launch to the plan"
    )
    validate.add_argument("--plan", type=Path, required=True)
    validate.add_argument("--arm", choices=("base", "trained"), required=True)
    validate.add_argument("--phase", choices=("shard", "merge"), required=True)
    validate.add_argument("--source", required=True)
    validate.add_argument("--role", required=True)
    validate.add_argument("--model", required=True)
    validate.add_argument("--model-revision", required=True)
    validate.add_argument("--task-file", type=Path, required=True)
    validate.add_argument("--max-records", type=int, required=True)
    validate.add_argument("--shard-count", type=int, required=True)
    validate.add_argument("--git-commit", required=True)
    validate.add_argument("--train-freeze", type=Path, required=True)
    validate.add_argument("--adapter", type=Path)
    validate.add_argument("--array-task-count", type=int)
    validate.add_argument("--array-task-min", type=int)
    validate.add_argument("--array-task-max", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "create":
        plan = build_plan(
            timing_summary=args.timing_summary,
            timing_companion=args.timing_companion,
            task_file=args.task_file,
            sacct_raw=args.sacct_raw,
            stdout=args.stdout,
        )
        output = write_plan_exclusive(args.output, plan)
        result = {"output": str(output), "sha256": sha256_file(output)}
    else:
        result = validate_launch_against_plan(
            plan_path=args.plan,
            arm=args.arm,
            phase=args.phase,
            source=args.source,
            role=args.role,
            model=args.model,
            model_revision=args.model_revision,
            task_file=args.task_file,
            max_records=args.max_records,
            shard_count=args.shard_count,
            git_commit=args.git_commit,
            train_freeze=args.train_freeze,
            adapter=args.adapter,
            array_task_count=args.array_task_count,
            array_task_min=args.array_task_min,
            array_task_max=args.array_task_max,
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
