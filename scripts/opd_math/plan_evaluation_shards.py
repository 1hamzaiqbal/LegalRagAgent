#!/usr/bin/env python3
"""Plan the complete O teacher evaluation from paired exact timing shards.

The planner is deliberately narrower than a generic throughput calculator.  It
accepts one base-model and one trained-model, one-shard, schema-v2
exact-environment evaluation of the same O ``teacher_gap_dev`` prefix.  Both
timing artifacts, their post-promotion custody companions, the complete task
file, and separate Slurm accounting/stdout evidence are reopened and hashed
before a primary plan is created.  The larger of the two ``ElapsedRaw`` values
drives the shared full-evaluation geometry.

The Slurm input is the raw, headerless, pipe-delimited output of::

    sacct -X -n -P -j JOB_ID \
      --format=JobIDRaw,JobName,State,ExitCode,ElapsedRaw,AllocTRES,StdOut

Exactly one non-array ``opd_math_eval`` row is accepted.  Its numeric job ID
must occur in the canonical stdout filename, its ``StdOut`` field must equal
the supplied stdout path, and ``AllocTRES`` must record exactly one GPU.  The
raw capture and stdout are both hashed into the plan.

For candidate shard count ``S`` the conservative projection is exactly

``ceil(ceil(total_records / S) / timing_records) * ElapsedRaw * safety_factor``.

The primary successor plan is deliberately non-configurable: 4,585 total O
records, two 32-record prefixes, 1.25 safety factor, an 18-hour shard cap,
four-way array concurrency, and at least five shards.  The five-shard floor is
registered from predecessor long-run evidence (80,000 seconds for 48 timing
blocks): four shards project to 75,000 seconds after the safety factor, while
five project to about 60,416.7 seconds.  Generic geometry remains available as
an explicitly marked diagnostic API with a default minimum of one.  The output
is created with exclusive-create semantics and is never overwritten.  It
authorizes only the launch geometry consumed by the production wrapper, never
a scientific result.
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
    from .quality_gates import sha256_tree, write_text_exclusive_fsync
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
    from quality_gates import sha256_tree, write_text_exclusive_fsync  # type: ignore
    from tokenizer_contract import canonical_sha256  # type: ignore


PLAN_SCHEMA_VERSION = 2
PLAN_KIND = "opd_math_o_dual_timing_evaluation_shard_plan_successor_v2"
DIAGNOSTIC_PLAN_KIND = "opd_math_o_evaluation_shard_plan_diagnostic_v2"
DEFAULT_TOTAL_RECORDS = 4585
PRIMARY_TIMING_RECORDS = 32
PRIMARY_MIN_SHARDS = 5
DEFAULT_MAX_SHARD_SECONDS = 64_800
DEFAULT_MAX_CONCURRENT = 4
DEFAULT_SAFETY_FACTOR = Decimal("1.25")
PREDECESSOR_LONG_RUN_ELAPSED_SECONDS = 80_000
PREDECESSOR_LONG_RUN_TIMING_BLOCKS = 48
PREDECESSOR_LONG_RUN_JOB_ID = "107462"
PREDECESSOR_LONG_RUN_COMMIT = "ae90bc744ed43e3ca57580ca9b008935dec92d9b"
PREDECESSOR_LONG_RUN_SHARD_RECORDS = [1528, 1528, 1529]
PREDECESSOR_LONG_RUN_SHARD_ELAPSED_SECONDS = [80_000, 73_600, 75_987]
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
    minimum_shard_count: int = 1,
) -> dict[str, Any]:
    """Return the smallest authorized geometry satisfying the runtime limit.

    The default remains a generic diagnostic calculation.  A registered floor
    is applied only when the primary successor contract calls this function.
    """

    total_records = _positive_int(total_records, "total records")
    timing_records = _positive_int(timing_records, "timing records")
    elapsed_raw_seconds = _positive_int(elapsed_raw_seconds, "Slurm ElapsedRaw")
    max_shard_seconds = _positive_int(max_shard_seconds, "maximum shard seconds")
    max_concurrent = _positive_int(max_concurrent, "maximum concurrency")
    minimum_shard_count = _positive_int(
        minimum_shard_count, "minimum shard count"
    )
    safety_factor = _decimal(safety_factor, "safety factor")
    if safety_factor < 1:
        raise ValueError("safety factor must be at least 1.0")
    if total_records <= timing_records:
        raise ValueError("complete evaluation must contain more records than its timing prefix")
    if minimum_shard_count > total_records:
        raise ValueError("minimum shard count cannot exceed the total record count")

    runtime_smallest: int | None = None
    for shard_count in range(1, total_records + 1):
        records_per_shard = (total_records + shard_count - 1) // shard_count
        timing_blocks = (records_per_shard + timing_records - 1) // timing_records
        projected = Decimal(timing_blocks * elapsed_raw_seconds) * safety_factor
        if projected <= Decimal(max_shard_seconds):
            runtime_smallest = shard_count
            break
    selected: tuple[int, int, int, Decimal] | None = None
    for shard_count in range(minimum_shard_count, total_records + 1):
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
        "registered_minimum_shard_count": minimum_shard_count,
        "smallest_runtime_passing_shard_count": runtime_smallest,
        "smallest_passing_shard_count": shard_count == runtime_smallest,
        "smallest_authorized_shard_count": True,
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
    minimum_shard_count: int = PRIMARY_MIN_SHARDS,
) -> None:
    expected = {
        "total_records": DEFAULT_TOTAL_RECORDS,
        "timing_records": PRIMARY_TIMING_RECORDS,
        "safety_factor": DEFAULT_SAFETY_FACTOR,
        "max_shard_seconds": DEFAULT_MAX_SHARD_SECONDS,
        "max_concurrent": DEFAULT_MAX_CONCURRENT,
        "minimum_shard_count": PRIMARY_MIN_SHARDS,
    }
    actual = {
        "total_records": total_records,
        "timing_records": timing_records,
        "safety_factor": _decimal(safety_factor, "safety factor"),
        "max_shard_seconds": max_shard_seconds,
        "max_concurrent": max_concurrent,
        "minimum_shard_count": minimum_shard_count,
    }
    if actual != expected:
        raise ValueError(
            "primary O timing plan requires the canonical fixed constraints: "
            f"expected={expected}, actual={actual}"
        )


def _validate_timing_arm(
    *,
    arm: str,
    timing_summary: Path,
    timing_companion: Path,
    task_file: Path,
    sacct_raw: Path,
    stdout: Path,
) -> dict[str, Any]:
    """Reopen one arm's exact-v2 artifact and independent scheduler evidence."""

    timing_summary = _regular_file(timing_summary, f"{arm} timing shard summary")
    timing_companion = _regular_file(
        timing_companion, f"{arm} timing custody companion"
    )
    sacct_raw = _regular_file(sacct_raw, f"{arm} raw Slurm timing accounting")
    accounting, stdout = _validate_sacct_raw(sacct_raw, stdout)
    validated = validate_shard_artifact(timing_summary, task_file=task_file)
    summary = validated["summary"]
    contract = validated["contract"]
    companion = validated["post_promotion_companion"]
    if contract.get("contract") != EVALUATION_CONTRACT or companion is None:
        raise ValueError(
            f"{arm} timing shard must use the schema-v2 exact-environment contract"
        )
    environment_contract = contract.get("code", {}).get("environment_contract")
    if not isinstance(environment_contract, dict) or not evaluation_environment_contract_unchanged(
        environment_contract
    ):
        raise ValueError(
            f"{arm} timing planner input requires the unchanged exact train environment"
        )
    expected_companion = post_promotion_custody_path(timing_summary.parent).resolve()
    if timing_companion != expected_companion or companion["path"].resolve() != timing_companion:
        raise ValueError(
            f"supplied {arm} timing companion is not the shard's exact custody companion"
        )
    if sha256_file(timing_companion) != companion["sha256"]:
        raise ValueError(f"{arm} timing custody companion changed after validation")
    if summary.get("model") != EXPECTED_MODEL or summary.get("model_revision") != EXPECTED_REVISION:
        raise ValueError(f"{arm} timing shard is not the pinned Qwen3-8B teacher model")
    adapter = summary.get("adapter")
    adapter_hash = summary.get("adapter_tree_sha256")
    adapter_path: Path | None
    if arm == "base":
        if adapter is not None or adapter_hash is not None:
            raise ValueError("base timing shard must be an unadapted model evaluation")
        adapter_path = None
    elif arm == "trained":
        if not isinstance(adapter, str) or not isinstance(adapter_hash, str):
            raise ValueError("trained timing shard must bind its exact teacher adapter")
        raw_adapter = Path(adapter).expanduser()
        if raw_adapter.is_symlink() or not raw_adapter.is_absolute() or not raw_adapter.is_dir():
            raise ValueError("trained timing adapter must be an absolute non-symlink directory")
        adapter_path = raw_adapter.resolve()
        if str(adapter_path) != adapter:
            raise ValueError("trained timing adapter path must already be canonical")
        if not (adapter_path / "adapter_config.json").is_file():
            raise ValueError("trained timing adapter lacks adapter_config.json")
        if sha256_tree(adapter_path) != adapter_hash:
            raise ValueError("trained timing adapter tree changed after evaluation")
    else:
        raise ValueError("timing arm must be base or trained")
    if summary.get("samples_per_problem") != EXPECTED_SAMPLES_PER_PROBLEM:
        raise ValueError(f"{arm} timing shard has the wrong samples-per-problem contract")
    if summary.get("decoding") != EXPECTED_DECODING:
        raise ValueError(f"{arm} timing shard has the wrong scientific decoding contract")
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
        raise ValueError(
            f"{arm} timing artifact must be one complete shard of its exact prefix"
        )
    timing_records = _positive_int(timing_records, f"{arm} timing prefix records")
    _validate_stdout(stdout, summary, timing_summary.parent)
    generation_latency = summary.get("total_generation_latency_seconds")
    if type(generation_latency) not in (int, float) or not math.isfinite(
        float(generation_latency)
    ) or float(generation_latency) <= 0:
        raise ValueError(f"{arm} timing shard has invalid generation-latency accounting")
    if float(generation_latency) > accounting["elapsed_raw_seconds"]:
        raise ValueError(f"{arm} generation latency cannot exceed Slurm ElapsedRaw")
    return {
        "summary_path": timing_summary,
        "companion_path": timing_companion,
        "sacct_raw_path": sacct_raw,
        "stdout_path": stdout,
        "validated": validated,
        "summary": summary,
        "contract": contract,
        "companion": companion,
        "accounting": accounting,
        "timing_records": timing_records,
        "generation_latency": generation_latency,
        "adapter_path": adapter_path,
        "adapter_tree_sha256": adapter_hash,
    }


def _timing_identity(item: Mapping[str, Any]) -> dict[str, Any]:
    """Identity fields that must be equal across base and trained timings."""

    contract = item["contract"]
    return {
        "evaluation_contract": contract["contract"],
        "model": contract["model"],
        "model_revision": contract["model_revision"],
        "task_file": contract["task_file"],
        "task_file_sha256": contract["task_file_sha256"],
        "eligible_records": contract["eligible_records"],
        "eligible_record_ids_sha256": contract["eligible_record_ids_sha256"],
        "task_sources": contract["task_sources"],
        "task_roles": contract["task_roles"],
        "samples_per_problem": contract["samples_per_problem"],
        "decoding": contract["decoding"],
        "record_seed_contract": contract["record_seed_contract"],
        "reward_verifier": contract["reward_verifier"],
        "shard": contract["shard"],
        "tokenizer_contract_sha256": contract["tokenizer_contract_sha256"],
        "code": contract["code"],
    }


def _timing_input_payload(item: Mapping[str, Any]) -> dict[str, Any]:
    validated = item["validated"]
    companion = item["companion"]
    summary = item["summary"]
    return {
        "summary": {
            "path": str(item["summary_path"]),
            "sha256": validated["summary_sha256"],
            "evaluation_contract_sha256": summary["evaluation_contract_sha256"],
            "output_tree_sha256": companion["tree_sha256"],
        },
        "samples": {
            "path": str(validated["samples_path"]),
            "sha256": validated["samples_sha256"],
        },
        "companion": {
            "path": str(item["companion_path"]),
            "sha256": companion["sha256"],
        },
        "sacct_raw": {
            "path": str(item["sacct_raw_path"]),
            "sha256": sha256_file(item["sacct_raw_path"]),
            "field_order": list(SACCT_FIELDS),
        },
        "stdout": {
            "path": str(item["stdout_path"]),
            "sha256": sha256_file(item["stdout_path"]),
        },
    }


def _timing_metrics_payload(item: Mapping[str, Any]) -> dict[str, Any]:
    accounting = item["accounting"]
    return {
        "job_id": accounting["job_id"],
        "job_name": accounting["job_name"],
        "state": accounting["state"],
        "exit_code": accounting["exit_code"],
        "allocated_gpus": accounting["allocated_gpus"],
        "timing_records": item["timing_records"],
        "elapsed_raw_seconds": accounting["elapsed_raw_seconds"],
        "total_generation_latency_seconds": item["generation_latency"],
    }


def _predecessor_minimum_basis() -> dict[str, Any]:
    def candidate(shard_count: int) -> dict[str, Any]:
        records = math.ceil(DEFAULT_TOTAL_RECORDS / shard_count)
        blocks = math.ceil(records / PRIMARY_TIMING_RECORDS)
        projected = (
            Decimal(PREDECESSOR_LONG_RUN_ELAPSED_SECONDS * blocks)
            / Decimal(PREDECESSOR_LONG_RUN_TIMING_BLOCKS)
            * DEFAULT_SAFETY_FACTOR
        )
        return {
            "shard_count": shard_count,
            "records_per_shard_ceiling": records,
            "timing_blocks_per_shard": blocks,
            "projected_shard_seconds": _json_number(projected),
            "exceeds_primary_limit": projected > Decimal(DEFAULT_MAX_SHARD_SECONDS),
        }

    return {
        "source_job_id": PREDECESSOR_LONG_RUN_JOB_ID,
        "source_git_commit": PREDECESSOR_LONG_RUN_COMMIT,
        "source_shard_records": PREDECESSOR_LONG_RUN_SHARD_RECORDS,
        "source_shard_elapsed_raw_seconds": (
            PREDECESSOR_LONG_RUN_SHARD_ELAPSED_SECONDS
        ),
        "evidence_elapsed_raw_seconds": PREDECESSOR_LONG_RUN_ELAPSED_SECONDS,
        "evidence_timing_blocks": PREDECESSOR_LONG_RUN_TIMING_BLOCKS,
        "registered_safety_factor": _json_number(DEFAULT_SAFETY_FACTOR),
        "four_shards": candidate(4),
        "five_shards": candidate(5),
    }


def build_plan(
    *,
    base_timing_summary: Path,
    base_timing_companion: Path,
    task_file: Path,
    base_sacct_raw: Path,
    base_stdout: Path,
    trained_timing_summary: Path | None = None,
    trained_timing_companion: Path | None = None,
    trained_sacct_raw: Path | None = None,
    trained_stdout: Path | None = None,
    total_records: int = DEFAULT_TOTAL_RECORDS,
    safety_factor: Decimal = DEFAULT_SAFETY_FACTOR,
    max_shard_seconds: int = DEFAULT_MAX_SHARD_SECONDS,
    max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    primary_contract: bool = True,
) -> dict[str, Any]:
    """Validate timing evidence and return one immutable successor plan payload."""

    task_file = _regular_file(task_file, "complete O evaluation task file")
    trained_values = (
        trained_timing_summary,
        trained_timing_companion,
        trained_sacct_raw,
        trained_stdout,
    )
    has_trained = all(value is not None for value in trained_values)
    if any(value is not None for value in trained_values) and not has_trained:
        raise ValueError("trained timing evidence must supply summary, companion, sacct, and stdout")
    if primary_contract and not has_trained:
        raise ValueError(
            "primary successor plan requires both base and trained exact-v2 timing evidence"
        )

    base = _validate_timing_arm(
        arm="base",
        timing_summary=base_timing_summary,
        timing_companion=base_timing_companion,
        task_file=task_file,
        sacct_raw=base_sacct_raw,
        stdout=base_stdout,
    )
    trained = None
    if has_trained:
        trained = _validate_timing_arm(
            arm="trained",
            timing_summary=trained_timing_summary,  # type: ignore[arg-type]
            timing_companion=trained_timing_companion,  # type: ignore[arg-type]
            task_file=task_file,
            sacct_raw=trained_sacct_raw,  # type: ignore[arg-type]
            stdout=trained_stdout,  # type: ignore[arg-type]
        )
        if base["summary_path"] == trained["summary_path"] or base["companion_path"] == trained["companion_path"]:
            raise ValueError("base and trained timing artifacts must be separate")
        if base["sacct_raw_path"] == trained["sacct_raw_path"] or base["stdout_path"] == trained["stdout_path"]:
            raise ValueError("base and trained timing scheduler/stdout evidence must be separate")
        if base["accounting"]["job_id"] == trained["accounting"]["job_id"]:
            raise ValueError("base and trained timing evidence must come from distinct Slurm jobs")
        if _timing_identity(base) != _timing_identity(trained):
            raise ValueError(
                "base and trained timings must share one exact commit, environment, task, "
                "model, decoding, tokenizer, and prefix contract"
            )

    planner_git = git_identity()
    commit = base["contract"].get("code", {}).get("git_commit")
    if (
        planner_git.get("worktree_clean") is not True
        or not isinstance(commit, str)
        or HEX40.fullmatch(commit) is None
        or planner_git.get("commit") != commit
    ):
        raise ValueError(
            "planner must run from the same clean immutable commit as both timing shards"
        )

    timing_records = base["timing_records"]
    if trained is not None and trained["timing_records"] != timing_records:
        raise ValueError("base and trained timing prefixes must contain identical record counts")
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
        raise ValueError("timing artifacts must be a strict prefix of the complete task")
    minimum_shards = PRIMARY_MIN_SHARDS if primary_contract else 1
    if primary_contract:
        _require_primary_constraints(
            total_records=total_records,
            timing_records=timing_records,
            safety_factor=safety_factor,
            max_shard_seconds=max_shard_seconds,
            max_concurrent=max_concurrent,
            minimum_shard_count=minimum_shards,
        )

    elapsed_values = [base["accounting"]["elapsed_raw_seconds"]]
    if trained is not None:
        elapsed_values.append(trained["accounting"]["elapsed_raw_seconds"])
    planning_elapsed = max(elapsed_values)
    geometry = choose_shard_geometry(
        total_records=total_records,
        timing_records=timing_records,
        elapsed_raw_seconds=planning_elapsed,
        safety_factor=safety_factor,
        max_shard_seconds=max_shard_seconds,
        max_concurrent=max_concurrent,
        minimum_shard_count=minimum_shards,
    )
    trained_adapter = (
        None
        if trained is None
        else {
            "path": str(trained["adapter_path"]),
            "tree_sha256": trained["adapter_tree_sha256"],
        }
    )
    payload: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "plan_kind": PLAN_KIND if primary_contract else DIAGNOSTIC_PLAN_KIND,
        "plan_lineage": "O full-gap dual-timing successor after predecessor long-run audit",
        "scientific_authorization": False,
        "authorizes_primary_o_launch_geometry": primary_contract,
        "claim_boundary": (
            "This successor plan authorizes only one shared O base/trained array geometry "
            "when the production wrapper revalidates it and the exact trained adapter. It "
            "is not evaluation evidence and cannot authorize a teacher, checkpoint merge, "
            "or student training."
            if primary_contract
            else "This diagnostic geometry is non-authorizing operator guidance only."
        ),
        "evaluation": {
            "source": EXPECTED_SOURCE,
            "role": EXPECTED_ROLE,
            "model": EXPECTED_MODEL,
            "model_revision": EXPECTED_REVISION,
            "base_adapter": None,
            "trained_adapter": trained_adapter,
            "samples_per_problem": EXPECTED_SAMPLES_PER_PROBLEM,
            "decoding": dict(EXPECTED_DECODING),
            "evaluation_contract": EVALUATION_CONTRACT,
        },
        "formula": (
            "ceil(ceil(total_records / shard_count) / timing_records) * "
            "max(base_elapsed_raw_seconds, trained_elapsed_raw_seconds) * safety_factor"
            if trained is not None
            else "ceil(ceil(total_records / shard_count) / timing_records) * "
            "base_elapsed_raw_seconds * safety_factor"
        ),
        "inputs": {
            "base_timing": _timing_input_payload(base),
            "trained_timing": (
                None if trained is None else _timing_input_payload(trained)
            ),
            "task_file": {
                "path": str(task_file),
                "sha256": sha256_file(task_file),
                "records": total_records,
            },
        },
        "timing": {
            "selection_rule": (
                "max(base_elapsed_raw_seconds, trained_elapsed_raw_seconds)"
                if trained is not None
                else "base_elapsed_raw_seconds_diagnostic_only"
            ),
            "planning_elapsed_raw_seconds": planning_elapsed,
            "timing_records": timing_records,
            "base": _timing_metrics_payload(base),
            "trained": None if trained is None else _timing_metrics_payload(trained),
        },
        "constraints": {
            "total_records": total_records,
            "safety_factor": _json_number(_decimal(safety_factor, "safety factor")),
            "max_shard_seconds": max_shard_seconds,
            "requested_max_concurrent_tasks": max_concurrent,
            "minimum_shard_count": minimum_shards,
            "minimum_shard_count_basis": (
                _predecessor_minimum_basis() if primary_contract else None
            ),
            "base_trained_geometry_must_match": True,
            "primary_constraints_fixed": primary_contract,
        },
        "array_plan": geometry,
        "code": {
            "git": planner_git,
            "planner_file": str(PLANNER_PATH),
            "planner_file_sha256": sha256_file(PLANNER_PATH),
            "timing_evaluator_file_sha256": base["contract"]["code"][
                "evaluator_file_sha256"
            ],
            "exact_environment_contract_sha256": canonical_sha256(
                base["contract"]["code"]["environment_contract"]
            ),
            "exact_environment_contract": base["contract"]["code"][
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
        plan.get("schema_version") != PLAN_SCHEMA_VERSION
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
        minimum_shard_count=constraints.get("minimum_shard_count"),
    )
    if constraints.get("minimum_shard_count_basis") != _predecessor_minimum_basis():
        raise ValueError("primary O plan lacks the registered predecessor timing basis")
    timing = plan.get("timing")
    if not isinstance(timing, dict):
        raise ValueError("primary O evaluation plan lacks dual timing custody")
    base_timing = timing.get("base")
    trained_timing = timing.get("trained")
    if not isinstance(base_timing, dict) or not isinstance(trained_timing, dict):
        raise ValueError("primary O evaluation plan requires both base and trained timings")
    base_elapsed = _positive_int(
        base_timing.get("elapsed_raw_seconds"), "planned base ElapsedRaw"
    )
    trained_elapsed = _positive_int(
        trained_timing.get("elapsed_raw_seconds"), "planned trained ElapsedRaw"
    )
    if (
        timing.get("selection_rule")
        != "max(base_elapsed_raw_seconds, trained_elapsed_raw_seconds)"
        or timing.get("planning_elapsed_raw_seconds") != max(base_elapsed, trained_elapsed)
    ):
        raise ValueError("primary O plan does not use max(base, trained) ElapsedRaw")
    inputs = plan.get("inputs")
    if (
        not isinstance(inputs, dict)
        or not isinstance(inputs.get("base_timing"), dict)
        or not isinstance(inputs.get("trained_timing"), dict)
    ):
        raise ValueError("primary O plan lacks paired timing-artifact custody")
    for arm in ("base", "trained"):
        arm_inputs = inputs[f"{arm}_timing"]
        for field in ("summary", "samples", "companion", "sacct_raw", "stdout"):
            binding = arm_inputs.get(field)
            if not isinstance(binding, dict):
                raise ValueError(
                    f"primary O plan lacks {arm} timing {field} custody"
                )
            bound_path = binding.get("path")
            bound_sha256 = binding.get("sha256")
            if not isinstance(bound_path, str) or not isinstance(bound_sha256, str):
                raise ValueError(
                    f"primary O plan has invalid {arm} timing {field} custody"
                )
            current = _regular_file(
                Path(bound_path), f"planned {arm} timing {field}"
            )
            if str(current) != bound_path or sha256_file(current) != bound_sha256:
                raise ValueError(
                    f"planned {arm} timing {field} identity has drifted"
                )
    task_binding = inputs.get("task_file")
    if not isinstance(task_binding, dict):
        raise ValueError("primary O plan lacks complete task-file custody")
    task_path = _regular_file(
        Path(str(task_binding.get("path", ""))), "planned O task file"
    )
    if (
        str(task_path) != task_binding.get("path")
        or sha256_file(task_path) != task_binding.get("sha256")
    ):
        raise ValueError("planned O task-file identity has drifted")
    evaluation = plan.get("evaluation")
    trained_adapter = (
        evaluation.get("trained_adapter") if isinstance(evaluation, dict) else None
    )
    if not isinstance(trained_adapter, dict):
        raise ValueError("primary O plan lacks its trained adapter binding")
    adapter_path = Path(str(trained_adapter.get("path", ""))).expanduser()
    adapter_hash = trained_adapter.get("tree_sha256")
    if (
        not adapter_path.is_absolute()
        or adapter_path.is_symlink()
        or not adapter_path.is_dir()
        or not isinstance(adapter_hash, str)
        or HEX64.fullmatch(adapter_hash) is None
        or sha256_tree(adapter_path.resolve()) != adapter_hash
    ):
        raise ValueError("primary O plan trained adapter path/tree binding does not verify")
    array_plan = plan.get("array_plan")
    if (
        not isinstance(array_plan, dict)
        or array_plan.get("base") != array_plan.get("trained")
        or array_plan.get("base_trained_specs_identical") is not True
        or canonical_sha256(array_plan.get("base"))
        != array_plan.get("common_spec_sha256")
    ):
        raise ValueError("primary O evaluation plan does not bind one shared array geometry")
    expected_geometry = choose_shard_geometry(
        total_records=constraints["total_records"],
        timing_records=timing["timing_records"],
        elapsed_raw_seconds=timing["planning_elapsed_raw_seconds"],
        safety_factor=_decimal(constraints["safety_factor"], "safety factor"),
        max_shard_seconds=constraints["max_shard_seconds"],
        max_concurrent=constraints["requested_max_concurrent_tasks"],
        minimum_shard_count=constraints["minimum_shard_count"],
    )
    if array_plan != expected_geometry:
        raise ValueError("primary O evaluation plan geometry does not recompute exactly")
    # A self-hash is not an external trust anchor.  Re-run the complete timing,
    # Slurm, stdout, adapter, task, environment, and distinct-arm validations
    # from the bound primary inputs, then demand byte-equivalent semantics.
    rebuilt = build_plan(
        base_timing_summary=Path(inputs["base_timing"]["summary"]["path"]),
        base_timing_companion=Path(inputs["base_timing"]["companion"]["path"]),
        base_sacct_raw=Path(inputs["base_timing"]["sacct_raw"]["path"]),
        base_stdout=Path(inputs["base_timing"]["stdout"]["path"]),
        trained_timing_summary=Path(inputs["trained_timing"]["summary"]["path"]),
        trained_timing_companion=Path(
            inputs["trained_timing"]["companion"]["path"]
        ),
        trained_sacct_raw=Path(inputs["trained_timing"]["sacct_raw"]["path"]),
        trained_stdout=Path(inputs["trained_timing"]["stdout"]["path"]),
        task_file=task_path,
    )
    if rebuilt != plan:
        raise ValueError(
            "primary O evaluation plan does not exactly rederive from its bound evidence"
        )
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
    array_spec: str,
    samples_per_problem: int,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    seed: int,
    array_task_count: int | None = None,
    array_task_min: int | None = None,
    array_task_max: int | None = None,
) -> dict[str, Any]:
    """Bind one production shard/merge launch to the immutable primary plan."""

    plan_path, plan = load_primary_plan(plan_path)
    if arm not in {"base", "trained"} or phase not in {"shard", "merge"}:
        raise ValueError("plan arm/phase must be base|trained and shard|merge")
    if source != EXPECTED_SOURCE or role != EXPECTED_ROLE:
        raise ValueError("primary O plan may only launch O teacher_gap_dev evaluation")
    evaluation = plan["evaluation"]
    if model != evaluation["model"] or model_revision != evaluation["model_revision"]:
        raise ValueError("evaluation model identity differs from the primary O plan")
    if type(max_records) is not int or max_records != 0:
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
    if type(shard_count) is not int or shard_count != spec["shard_count"]:
        raise ValueError("evaluation shard count differs from the primary O plan")
    if not isinstance(array_spec, str) or array_spec != spec["array_spec"]:
        raise ValueError(
            "literal Slurm array specification differs from the primary O plan: "
            f"expected={spec['array_spec']!r}, actual={array_spec!r}"
        )
    actual_decoding = {
        "thinking": False,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_new_tokens": max_new_tokens,
        "seed": seed,
    }
    if type(samples_per_problem) is not int or samples_per_problem != EXPECTED_SAMPLES_PER_PROBLEM:
        raise ValueError("evaluation sample count differs from the primary O plan")
    if (
        type(temperature) is not float
        or type(top_p) is not float
        or type(top_k) is not int
        or type(max_new_tokens) is not int
        or type(seed) is not int
    ):
        raise ValueError("evaluation decoding values have noncanonical numeric types")
    if actual_decoding != EXPECTED_DECODING:
        raise ValueError(
            "evaluation decoding differs from the primary O plan: "
            f"expected={EXPECTED_DECODING}, actual={actual_decoding}"
        )
    if arm == "base":
        if adapter is not None:
            raise ValueError("base planned evaluation must not use an adapter")
        adapter_path = None
    else:
        if adapter is None:
            raise ValueError("trained planned evaluation requires the teacher adapter")
        adapter_path = Path(adapter).expanduser()
        if adapter_path.is_symlink() or not adapter_path.is_absolute() or not adapter_path.is_dir():
            raise ValueError("trained planned evaluation adapter must be a real directory")
        adapter_path = adapter_path.resolve()
        if not (adapter_path / "adapter_config.json").is_file():
            raise ValueError("trained planned evaluation adapter lacks adapter_config.json")
        planned_adapter = evaluation.get("trained_adapter")
        if not isinstance(planned_adapter, dict):
            raise ValueError("primary O plan lacks its trained adapter binding")
        if (
            str(adapter_path) != planned_adapter.get("path")
            or sha256_tree(adapter_path) != planned_adapter.get("tree_sha256")
        ):
            raise ValueError(
                "trained evaluation adapter path/tree differs from the primary O plan"
            )
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
    plan_binding = {
        "schema_version": 1,
        "binding_kind": "opd_math_o_primary_evaluation_plan_binding_v1",
        "plan": str(plan_path),
        "plan_file_sha256": sha256_file(plan_path),
        "plan_payload_sha256": plan["plan_payload_sha256"],
        "plan_schema_version": plan["schema_version"],
        "plan_kind": plan["plan_kind"],
        "arm": arm,
        "source": EXPECTED_SOURCE,
        "role": EXPECTED_ROLE,
        "model": model,
        "model_revision": model_revision,
        "task_file": str(task_file),
        "task_file_sha256": task_binding["sha256"],
        "max_records": max_records,
        "git_commit": git_commit,
        "train_freeze": str(train_freeze),
        "train_freeze_sha256": planned_freeze["sha256"],
        "array_spec": spec["array_spec"],
        "slurm_array_argument": spec["slurm_array_argument"],
        "array_geometry_sha256": plan["array_plan"]["common_spec_sha256"],
        "shard_count": shard_count,
        "samples_per_problem": samples_per_problem,
        "decoding": actual_decoding,
        "adapter": None if adapter_path is None else str(adapter_path),
        "adapter_tree_sha256": (
            None
            if adapter_path is None
            else plan["evaluation"]["trained_adapter"]["tree_sha256"]
        ),
    }
    return {
        "plan_binding": plan_binding,
        "launch_validation": {
            "schema_version": 1,
            "validation_kind": "opd_math_o_primary_plan_launch_validation_v1",
            "phase": phase,
            "array_spec_source": "predeclared_OPD_MATH_EVAL_ARRAY_SPEC_v1",
            "declared_array_spec": array_spec,
            "declared_slurm_array_argument": spec["slurm_array_argument"],
            "array_task_count": array_task_count,
            "array_task_min": array_task_min,
            "array_task_max": array_task_max,
            "validated": True,
        },
    }


def revalidate_plan_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    """Independently reopen every plan/input binding recorded in an artifact."""

    if not isinstance(binding, Mapping):
        raise ValueError("evaluation plan binding must be an object")
    try:
        adapter_raw = binding.get("adapter")
        validated = validate_launch_against_plan(
            plan_path=Path(str(binding["plan"])),
            arm=str(binding["arm"]),
            phase="merge",
            source=str(binding["source"]),
            role=str(binding["role"]),
            model=str(binding["model"]),
            model_revision=str(binding["model_revision"]),
            task_file=Path(str(binding["task_file"])),
            max_records=binding["max_records"],
            shard_count=binding["shard_count"],
            git_commit=str(binding["git_commit"]),
            train_freeze=Path(str(binding["train_freeze"])),
            adapter=None if adapter_raw is None else Path(str(adapter_raw)),
            array_spec=str(binding["array_spec"]),
            samples_per_problem=binding["samples_per_problem"],
            temperature=binding["decoding"]["temperature"],
            top_p=binding["decoding"]["top_p"],
            top_k=binding["decoding"]["top_k"],
            max_new_tokens=binding["decoding"]["max_new_tokens"],
            seed=binding["decoding"]["seed"],
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("evaluation plan binding is incomplete") from exc
    current = validated["plan_binding"]
    if dict(binding) != current:
        raise ValueError("recorded evaluation plan binding does not revalidate exactly")
    return dict(current)


def revalidate_plan_binding_for_contract(
    binding: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, Any]:
    """Revalidate a plan and prove it is the plan for this exact eval contract."""

    current = revalidate_plan_binding(binding)
    code = contract.get("code")
    shard = contract.get("shard")
    if not isinstance(code, Mapping) or not isinstance(shard, Mapping):
        raise ValueError("evaluation contract lacks code/shard custody for its plan")
    environment = code.get("environment_contract")
    freeze = (
        environment.get("train_freeze")
        if isinstance(environment, Mapping)
        else None
    )
    if not isinstance(freeze, Mapping):
        raise ValueError("planned evaluation contract lacks exact train-freeze custody")
    expected = {
        "model": contract.get("model"),
        "model_revision": contract.get("model_revision"),
        "task_file": contract.get("task_file"),
        "task_file_sha256": contract.get("task_file_sha256"),
        "git_commit": code.get("git_commit"),
        "train_freeze": freeze.get("path"),
        "train_freeze_sha256": freeze.get("sha256"),
        "shard_count": shard.get("shard_count"),
        "samples_per_problem": contract.get("samples_per_problem"),
        "decoding": contract.get("decoding"),
        "adapter": contract.get("adapter"),
        "adapter_tree_sha256": contract.get("adapter_tree_sha256"),
    }
    mismatched = [
        field for field, value in expected.items() if current.get(field) != value
    ]
    if current.get("max_records") != 0:
        mismatched.append("max_records")
    if contract.get("task_sources") != [current.get("source")]:
        mismatched.append("task_sources")
    if contract.get("task_roles") != [current.get("role")]:
        mismatched.append("task_roles")
    if mismatched:
        raise ValueError(
            "evaluation plan binding differs from its exact artifact contract: "
            f"{sorted(set(mismatched))}"
        )
    return current


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser(
        "create", help="create the fixed dual-timing primary O successor plan"
    )
    create.add_argument("--base-timing-summary", type=Path, required=True)
    create.add_argument("--base-timing-companion", type=Path, required=True)
    create.add_argument("--base-sacct-raw", type=Path, required=True)
    create.add_argument("--base-stdout", type=Path, required=True)
    create.add_argument("--trained-timing-summary", type=Path, required=True)
    create.add_argument("--trained-timing-companion", type=Path, required=True)
    create.add_argument("--trained-sacct-raw", type=Path, required=True)
    create.add_argument("--trained-stdout", type=Path, required=True)
    create.add_argument("--task-file", type=Path, required=True)
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
    validate.add_argument("--array-spec", required=True)
    validate.add_argument("--samples-per-problem", type=int, required=True)
    validate.add_argument("--temperature", type=float, required=True)
    validate.add_argument("--top-p", type=float, required=True)
    validate.add_argument("--top-k", type=int, required=True)
    validate.add_argument("--max-new-tokens", type=int, required=True)
    validate.add_argument("--seed", type=int, required=True)
    validate.add_argument("--array-task-count", type=int)
    validate.add_argument("--array-task-min", type=int)
    validate.add_argument("--array-task-max", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "create":
        plan = build_plan(
            base_timing_summary=args.base_timing_summary,
            base_timing_companion=args.base_timing_companion,
            base_sacct_raw=args.base_sacct_raw,
            base_stdout=args.base_stdout,
            trained_timing_summary=args.trained_timing_summary,
            trained_timing_companion=args.trained_timing_companion,
            trained_sacct_raw=args.trained_sacct_raw,
            trained_stdout=args.trained_stdout,
            task_file=args.task_file,
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
            array_spec=args.array_spec,
            samples_per_problem=args.samples_per_problem,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
            array_task_count=args.array_task_count,
            array_task_min=args.array_task_min,
            array_task_max=args.array_task_max,
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
