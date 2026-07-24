#!/usr/bin/env python3
"""Reconstruct preregistered OPD data-value diagnostics from explicit records."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


LN2 = math.log(2.0)
UNIT_FIELDS = (
    "source",
    "student_checkpoint",
    "teacher_checkpoint",
    "objective",
    "seed",
    "budget",
    "target",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_object(path: Path, label: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return payload


def finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise RuntimeError(f"{label} must be finite")
    return converted


def positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError(f"{label} must be a positive integer")
    return value


def validate_contract(contract: dict) -> None:
    if contract.get("campaign_id") != "opd_data_value_v1":
        raise RuntimeError("unexpected data-value campaign")
    if contract.get("status") != (
        "design_and_analysis_plumbing_only_blocked_on_positive_control_task_gain"
    ):
        raise RuntimeError("data-value campaign is not analysis-only")
    if tuple(contract.get("unit", ())) != UNIT_FIELDS:
        raise RuntimeError("experimental unit drifted")
    boundaries = contract.get("immutable_boundaries", {})
    if not boundaries or not all(boundaries.values()):
        raise RuntimeError("an immutable boundary is not enforced")
    if contract.get("P3_design", {}).get("current_allowed_training_sources") != ["O"]:
        raise RuntimeError("a closed teacher source was added")


def prequential_proxy(records: list[dict]) -> dict:
    if len(records) < 2:
        raise RuntimeError("prequential proxy requires at least two checkpoints")
    parsed = []
    for index, record in enumerate(records):
        step = record.get("checkpoint_step")
        tokens = record.get("cumulative_training_tokens")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise RuntimeError(f"prequential record {index} has invalid step")
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
            raise RuntimeError(f"prequential record {index} has invalid token count")
        nll = finite_number(
            record.get("source_nll_nats_per_token"),
            f"prequential record {index} NLL",
        )
        if nll < 0:
            raise RuntimeError("source NLL cannot be negative")
        parsed.append((step, tokens, nll))
    if parsed != sorted(parsed):
        raise RuntimeError("prequential records must be ordered by step and tokens")
    if len({row[0] for row in parsed}) != len(parsed):
        raise RuntimeError("prequential checkpoint steps must be unique")
    if any(right[1] <= left[1] for left, right in zip(parsed, parsed[1:])):
        raise RuntimeError("cumulative training tokens must strictly increase")

    terminal_nll = parsed[-1][2]
    signed_bits = 0.0
    positive_bits = 0.0
    intervals = []
    for left, right in zip(parsed, parsed[1:]):
        delta_tokens = right[1] - left[1]
        mean_excess_nats = ((left[2] + right[2]) / 2.0) - terminal_nll
        contribution_bits = delta_tokens * mean_excess_nats / LN2
        signed_bits += contribution_bits
        positive_bits += max(0.0, contribution_bits)
        intervals.append(
            {
                "left_step": left[0],
                "right_step": right[0],
                "delta_training_tokens": delta_tokens,
                "signed_excess_bits": contribution_bits,
            }
        )
    return {
        "label": "prequential_structural_information_proxy_bits",
        "method": "trapezoid_area_above_terminal_fixed_source_nll",
        "terminal_nll_nats_per_token": terminal_nll,
        "signed_bits": signed_bits,
        "positive_part_bits": positive_bits,
        "intervals": intervals,
        "formal_epiplexity_claim": False,
    }


def requential_code(records: list[dict]) -> dict:
    if not records:
        raise RuntimeError("requential code requires records")
    total_nats = 0.0
    total_tokens = 0
    for index, record in enumerate(records):
        if record.get("sampling_origin") != "teacher_generated_paths":
            raise RuntimeError(f"requential record {index} was not teacher sampled")
        if record.get("kl_direction") != "teacher||student":
            raise RuntimeError(f"requential record {index} has the wrong KL direction")
        if record.get("full_vocabulary") is not True:
            raise RuntimeError(f"requential record {index} is not full vocabulary")
        if record.get("unclipped") is not True:
            raise RuntimeError(f"requential record {index} is clipped")
        tokens = positive_integer(
            record.get("response_token_count"),
            f"requential record {index} response tokens",
        )
        kl = finite_number(
            record.get("teacher_student_kl_nats_per_token"),
            f"requential record {index} KL",
        )
        if kl < 0:
            raise RuntimeError("teacher-student KL cannot be negative")
        total_tokens += tokens
        total_nats += tokens * kl
    return {
        "label": "requential_teacher_student_code_bits",
        "response_tokens": total_tokens,
        "total_nats": total_nats,
        "total_bits": total_nats / LN2,
        "sampling_origin": "teacher_generated_paths",
        "kl_direction": "teacher||student",
        "full_vocabulary": True,
        "unclipped": True,
    }


def opd_state_proxy(records: list[dict]) -> dict:
    if not records:
        raise RuntimeError("OPD-state proxy requires records")
    total_tokens = 0
    unclipped_nats = 0.0
    executed_nats = 0.0
    for index, record in enumerate(records):
        if record.get("sampling_origin") != "student_generated_paths":
            raise RuntimeError(f"OPD-state record {index} was not student sampled")
        tokens = positive_integer(
            record.get("response_token_count"),
            f"OPD-state record {index} response tokens",
        )
        raw = finite_number(
            record.get("unclipped_divergence_nats_per_token"),
            f"OPD-state record {index} unclipped divergence",
        )
        executed = finite_number(
            record.get("executed_divergence_nats_per_token"),
            f"OPD-state record {index} executed divergence",
        )
        if raw < 0 or executed < 0:
            raise RuntimeError("OPD-state divergences cannot be negative")
        total_tokens += tokens
        unclipped_nats += tokens * raw
        executed_nats += tokens * executed
    return {
        "label": "student_path_teacher_student_divergence_proxy",
        "response_tokens": total_tokens,
        "unclipped_nats": unclipped_nats,
        "executed_nats": executed_nats,
        "unclipped_bits": unclipped_nats / LN2,
        "executed_bits": executed_nats / LN2,
        "sampling_origin": "student_generated_paths",
        "requential_or_epiplexity_claim": False,
    }


def value_outcomes(records: list[dict]) -> list[dict]:
    outputs = []
    seen = set()
    for index, record in enumerate(records):
        unit = tuple(record.get(field) for field in UNIT_FIELDS)
        if any(value is None for value in unit):
            raise RuntimeError(f"outcome record {index} has an incomplete unit")
        if unit in seen:
            raise RuntimeError(f"outcome record {index} duplicates an experimental unit")
        seen.add(unit)
        values = {
            name: finite_number(record.get(name), f"outcome record {index} {name}")
            for name in ("opd_pre", "opd_post", "matched_control_pre", "matched_control_post")
        }
        value = (values["opd_post"] - values["opd_pre"]) - (
            values["matched_control_post"] - values["matched_control_pre"]
        )
        outputs.append(
            {
                **{field: record[field] for field in UNIT_FIELDS},
                **values,
                "paired_difference_in_differences": value,
            }
        )
    return outputs


def analyze(contract: dict, payload: dict) -> dict:
    validate_contract(contract)
    return {
        "schema_version": 1,
        "artifact_type": "opd_data_value_metric_reconstruction",
        "campaign_id": contract["campaign_id"],
        "status": "analysis_only_no_training_authority",
        "prequential": prequential_proxy(payload.get("prequential_records", [])),
        "requential": requential_code(payload.get("requential_records", [])),
        "opd_state_proxy": opd_state_proxy(payload.get("opd_state_records", [])),
        "outcomes": value_outcomes(payload.get("outcome_records", [])),
    }


def write_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = load_object(args.contract.resolve(), "data-value contract")
    source = load_object(args.input.resolve(), "data-value records")
    result = analyze(contract, source)
    result["contract"] = str(args.contract.resolve())
    result["contract_sha256"] = sha256(args.contract.resolve())
    result["input"] = str(args.input.resolve())
    result["input_sha256"] = sha256(args.input.resolve())
    write_exclusive(args.output.resolve(), result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
