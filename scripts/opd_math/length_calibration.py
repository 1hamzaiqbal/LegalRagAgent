#!/usr/bin/env python3
"""Select a rollout cap from preregistered setup-only evaluation artifacts."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from .qualification_audit import analyze_evaluation_samples, source_receipt
except ImportError:
    from qualification_audit import analyze_evaluation_samples, source_receipt  # type: ignore


def parse_surface(value: str) -> tuple[str, str, int, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "expected FAMILY:ARM:TOKENS=/absolute/merged/directory"
        )
    identity, raw_path = value.split("=", 1)
    parts = identity.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "expected FAMILY:ARM:TOKENS=/absolute/merged/directory"
        )
    family, arm, raw_tokens = parts
    try:
        tokens = int(raw_tokens)
    except ValueError as error:
        raise argparse.ArgumentTypeError("TOKENS must be an integer") from error
    if family not in {"student", "teacher"} or not arm or tokens <= 0:
        raise argparse.ArgumentTypeError("invalid surface identity")
    return family, arm, tokens, Path(raw_path)


def sample_identities(path: Path) -> set[tuple[str, int]]:
    identities: set[tuple[str, int]] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            record_id = row.get("record_id")
            sample_idx = row.get("sample_idx")
            if not isinstance(record_id, str) or not isinstance(sample_idx, int):
                raise RuntimeError(f"invalid sample identity at {path}:{line_number}")
            identity = (record_id, sample_idx)
            if identity in identities:
                raise RuntimeError(f"duplicate sample identity at {path}:{line_number}")
            identities.add(identity)
    return identities


def analyze_surface(
    directory: Path,
    *,
    expected_tokens: int,
    expected_records: int,
    expected_samples_per_record: int,
) -> dict[str, Any]:
    summary_path = directory / "summary.json"
    samples_path = directory / "samples.jsonl"
    custody_path = directory.with_suffix(".custody.json")
    for path in (summary_path, samples_path, custody_path):
        if not path.is_file():
            raise RuntimeError(f"missing merged evaluation artifact: {path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("decoding", {}).get("max_new_tokens") != expected_tokens:
        raise RuntimeError(f"decoding cap mismatch in {summary_path}")
    if summary.get("records") != expected_records:
        raise RuntimeError(f"record count mismatch in {summary_path}")
    expected_samples = expected_records * expected_samples_per_record
    if summary.get("samples") != expected_samples:
        raise RuntimeError(f"sample count mismatch in {summary_path}")
    if summary.get("samples_per_problem") != expected_samples_per_record:
        raise RuntimeError(f"samples-per-record mismatch in {summary_path}")
    metrics = analyze_evaluation_samples(samples_path, max_tokens=expected_tokens)
    metrics["summary"] = source_receipt(summary_path)
    metrics["custody"] = source_receipt(custody_path)
    metrics["model"] = summary.get("model")
    metrics["model_revision"] = summary.get("model_revision")
    metrics["adapter"] = summary.get("adapter")
    metrics["task_file"] = summary.get("task_file")
    metrics["decoding"] = summary.get("decoding")
    metrics["sample_identities"] = sample_identities(samples_path)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--surface", action="append", type=parse_surface, default=[])
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()
    if not args.surface:
        parser.error("at least one --surface is required")
    if args.output_json.exists() or args.output_markdown.exists():
        parser.error("outputs must be fresh paths")

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    calibration = plan.get("length_calibration")
    if not isinstance(calibration, dict):
        raise RuntimeError("qualification plan lacks length_calibration")
    expected_records = calibration.get("records")
    expected_samples_per_record = calibration.get("samples_per_record")
    candidates = calibration.get("candidate_max_completion_tokens")
    rule = calibration.get("selection_rule")
    if (
        not isinstance(expected_records, int)
        or not isinstance(expected_samples_per_record, int)
        or not isinstance(candidates, list)
        or not all(isinstance(value, int) for value in candidates)
        or not isinstance(rule, dict)
    ):
        raise RuntimeError("invalid length-calibration plan")
    max_cap_fraction = float(rule["maximum_at_cap_fraction"])
    max_parse_below = float(rule["maximum_parse_failure_fraction_below_cap"])
    max_verifier = float(rule["maximum_verifier_error_fraction"])

    surfaces: dict[str, dict[str, dict[int, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    common_identities: set[tuple[str, int]] | None = None
    task_files: set[str] = set()
    for family, arm, tokens, directory in args.surface:
        if tokens not in candidates:
            raise RuntimeError(f"unregistered candidate cap: {tokens}")
        if tokens in surfaces[family][arm]:
            raise RuntimeError(f"duplicate surface: {family}:{arm}:{tokens}")
        metrics = analyze_surface(
            directory,
            expected_tokens=tokens,
            expected_records=expected_records,
            expected_samples_per_record=expected_samples_per_record,
        )
        identities = metrics.pop("sample_identities")
        if common_identities is None:
            common_identities = identities
        elif identities != common_identities:
            raise RuntimeError("calibration surfaces do not share exact sample identities")
        task_files.add(str(metrics["task_file"]))
        verifier_errors = metrics["status_counts"].get("verifier_error_zeroed", 0)
        metrics["verifier_error_fraction"] = verifier_errors / metrics["samples"]
        metrics["passes"] = (
            metrics["at_cap_fraction"] <= max_cap_fraction
            and metrics["parse_failure_below_cap_fraction"] <= max_parse_below
            and metrics["verifier_error_fraction"] <= max_verifier
        )
        surfaces[family][arm][tokens] = metrics
    if len(task_files) != 1:
        raise RuntimeError("calibration surfaces use different task files")

    decisions: dict[str, dict[str, Any]] = {}
    for family, arms in sorted(surfaces.items()):
        tested_caps = sorted(set.intersection(*(set(values) for values in arms.values())))
        selected = next(
            (
                cap
                for cap in candidates
                if cap in tested_caps and all(values[cap]["passes"] for values in arms.values())
            ),
            None,
        )
        all_candidates_tested = all(cap in tested_caps for cap in candidates)
        decisions[family] = {
            "arms": sorted(arms),
            "tested_common_caps": tested_caps,
            "selected_max_completion_tokens": selected,
            "status": (
                "QUALIFIED"
                if selected is not None
                else "FAILED_ALL_CANDIDATES"
                if all_candidates_tested
                else "NEEDS_NEXT_CANDIDATE"
            ),
        }

    serializable_surfaces = {
        family: {
            arm: {str(cap): metrics for cap, metrics in sorted(values.items())}
            for arm, values in sorted(arms.items())
        }
        for family, arms in sorted(surfaces.items())
    }
    payload = {
        "schema_version": 1,
        "artifact_type": "opd_math_setup_only_length_calibration",
        "scientific_training_authorized": False,
        "claim_boundary": (
            "Setup-only cap selection on registered student_opd records; no final "
            "evaluation role was opened and no model-training claim is authorized."
        ),
        "plan": source_receipt(args.plan),
        "task_file": next(iter(task_files)),
        "common_sample_identities": len(common_identities or set()),
        "criteria": {
            "maximum_at_cap_fraction": max_cap_fraction,
            "maximum_parse_failure_fraction_below_cap": max_parse_below,
            "maximum_verifier_error_fraction": max_verifier,
        },
        "surfaces": serializable_surfaces,
        "decisions": decisions,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# OPD math setup-only length calibration",
        "",
        "> Diagnostic only. No scientific training authorization.",
        "",
        "| Family | Status | Selected cap | Tested common caps |",
        "|---|---|---:|---|",
    ]
    for family, decision in decisions.items():
        selected = decision["selected_max_completion_tokens"]
        lines.append(
            f"| {family} | {decision['status']} | {selected or '—'} | "
            f"{', '.join(str(value) for value in decision['tested_common_caps'])} |"
        )
    lines.append("")
    args.output_markdown.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
