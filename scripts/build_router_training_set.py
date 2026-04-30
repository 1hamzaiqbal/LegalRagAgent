#!/usr/bin/env python3
"""Build an offline training table for bottleneck-aware method routing.

Each input arm is a completed detail log. The script joins arms by question ID,
extracts cheap task features from the question record, appends per-arm
correctness/cost/retrieval fields, and labels the row with an oracle arm under
a configurable cost-aware reward.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import compute_mcnemar  # type: ignore  # noqa: E402


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    if not rows:
        raise SystemExit(f"{path}: no records loaded")
    return rows


def resolve_arm(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise SystemExit(f"Invalid --arm {raw!r}; expected label=path")
    label, pattern = raw.split("=", 1)
    label = label.strip()
    if not label:
        raise SystemExit(f"Invalid --arm {raw!r}; empty label")
    matches = sorted(glob.glob(pattern))
    if not matches and Path(pattern).exists():
        matches = [pattern]
    if len(matches) != 1:
        raise SystemExit(f"{raw!r}: expected exactly one path, matched {len(matches)}")
    return label, Path(matches[0])


def choose_key(rows_by_arm: dict[str, list[dict[str, Any]]], requested: str | None) -> str:
    labels = list(rows_by_arm)
    if requested:
        for label, rows in rows_by_arm.items():
            values = [row.get(requested) for row in rows]
            if any(value is None for value in values) or len(set(values)) != len(values):
                raise SystemExit(f"{label}: key {requested!r} is missing or non-unique")
        return requested
    key = compute_mcnemar.choose_key_field(rows_by_arm[labels[0]], rows_by_arm[labels[1]], None)
    for label in labels[2:]:
        compute_mcnemar.choose_key_field(rows_by_arm[labels[0]], rows_by_arm[label], key)
    return key


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def choices_from_row(row: dict[str, Any]) -> list[str]:
    raw = row.get("choices")
    if isinstance(raw, list):
        return [str(item) for item in raw]
    if isinstance(raw, dict):
        return [str(raw[key]) for key in sorted(raw)]
    choices = []
    for letter in ("a", "b", "c", "d", "e"):
        value = row.get(f"choice_{letter}")
        if value:
            choices.append(str(value))
    return choices


def answer_format(row: dict[str, Any], choices: list[str]) -> str:
    dataset = str(row.get("dataset", ""))
    if choices:
        return f"mc{len(choices)}"
    if dataset == "housing":
        return "yes_no"
    if dataset == "musique":
        return "short_span"
    if dataset in {"legal_rag", "australian"}:
        return "long_form"
    return "unknown"


def static_features(row: dict[str, Any]) -> dict[str, Any]:
    question = str(row.get("formatted_question") or row.get("question") or "")
    intermediate = str(row.get("intermediate_question") or "")
    choices = choices_from_row(row)
    tokens = re.findall(r"[A-Za-z0-9_]+", question)
    named_entities = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", question)
    dates_numbers = re.findall(r"\b(?:19|20)\d{2}\b|\b\d+(?:\.\d+)?\b", question)
    legal_terms = re.findall(
        r"\b(?:statute|holding|case|court|contract|tort|evidence|jurisdiction|"
        r"liability|damages|tenant|landlord|plaintiff|defendant|precedent)\b",
        question,
        flags=re.IGNORECASE,
    )
    hop_terms = re.findall(
        r"\b(?:whose|after|before|where|which|father|mother|spouse|born|founded|"
        r"located|member|director|author|composer|played)\b",
        question,
        flags=re.IGNORECASE,
    )
    choice_lengths = [len(choice) for choice in choices]
    return {
        "dataset": row.get("dataset", ""),
        "provider": row.get("provider", ""),
        "subject": row.get("subject", ""),
        "answer_format": answer_format(row, choices),
        "question_chars": len(question),
        "question_tokens": len(tokens),
        "intermediate_chars": len(intermediate),
        "choice_count": len(choices),
        "avg_choice_chars": round(mean([float(v) for v in choice_lengths]), 3),
        "max_choice_chars": max(choice_lengths) if choice_lengths else 0,
        "named_entity_count": len(named_entities),
        "date_number_count": len(dates_numbers),
        "legal_term_count": len(legal_terms),
        "multi_hop_cue_count": len(hop_terms),
    }


def evidence_count(row: dict[str, Any]) -> int:
    evidence = row.get("evidence_store")
    if isinstance(evidence, list):
        return len(evidence)
    retrieved = row.get("retrieved_ids")
    if isinstance(retrieved, list):
        return len(retrieved)
    return 0


def max_ce_score(row: dict[str, Any]) -> float:
    scores = []
    for item in row.get("evidence_store") or []:
        if isinstance(item, dict) and item.get("cross_encoder_score") is not None:
            try:
                scores.append(float(item["cross_encoder_score"]))
            except (TypeError, ValueError):
                pass
    return max(scores) if scores else 0.0


def arm_metrics(row: dict[str, Any]) -> dict[str, Any]:
    correct = compute_mcnemar.correct_flag(row)
    calls = float(row.get("llm_calls_actual", row.get("llm_calls", 0)) or 0)
    latency = float(row.get("elapsed_sec", 0) or 0)
    return {
        "correct": int(correct),
        "calls": calls,
        "latency_sec": latency,
        "input_tokens": int(row.get("input_tokens", 0) or 0),
        "output_tokens": int(row.get("output_tokens", 0) or 0),
        "gold_retrieved": int(bool(row.get("gold_retrieved"))),
        "evidence_count": evidence_count(row),
        "max_ce_score": round(max_ce_score(row), 6),
        "parse_ok": int(all(bool(row.get(key)) for key in row if key.endswith("_parse_ok"))) if any(key.endswith("_parse_ok") for key in row) else "",
    }


def reward(metrics: dict[str, Any], call_penalty: float, latency_penalty: float) -> float:
    return float(metrics["correct"]) - call_penalty * float(metrics["calls"]) - latency_penalty * float(metrics["latency_sec"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True, help="Arm as label=detail_log.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="CSV output path")
    parser.add_argument("--key", help="Override join key")
    parser.add_argument("--call-penalty", type=float, default=0.02)
    parser.add_argument("--latency-penalty", type=float, default=0.0)
    args = parser.parse_args()

    rows_by_arm: dict[str, list[dict[str, Any]]] = {}
    for raw in args.arm:
        label, path = resolve_arm(raw)
        if label in rows_by_arm:
            raise SystemExit(f"Duplicate arm label {label!r}")
        rows_by_arm[label] = load_jsonl(path)

    labels = list(rows_by_arm)
    key = choose_key(rows_by_arm, args.key)
    by_key = {label: {row[key]: row for row in rows} for label, rows in rows_by_arm.items()}
    common_keys = sorted(set.intersection(*(set(rows) for rows in by_key.values())))
    if not common_keys:
        raise SystemExit(f"No common rows on key {key!r}")

    rows_out: list[dict[str, Any]] = []
    for row_key in common_keys:
        base_row = by_key[labels[0]][row_key]
        out = {"join_key": row_key, **static_features(base_row)}
        per_arm = {}
        for label in labels:
            metrics = arm_metrics(by_key[label][row_key])
            per_arm[label] = metrics
            for metric_name, value in metrics.items():
                out[f"{label}_{metric_name}"] = value

        correct_arms = [label for label in labels if per_arm[label]["correct"]]
        if correct_arms:
            out["oracle_accuracy_arm"] = min(
                correct_arms,
                key=lambda label: (per_arm[label]["calls"], per_arm[label]["latency_sec"], label),
            )
        else:
            out["oracle_accuracy_arm"] = min(
                labels,
                key=lambda label: (per_arm[label]["calls"], per_arm[label]["latency_sec"], label),
            )
        out["oracle_reward_arm"] = max(
            labels,
            key=lambda label: (
                reward(per_arm[label], args.call_penalty, args.latency_penalty),
                -per_arm[label]["calls"],
                -per_arm[label]["latency_sec"],
                label,
            ),
        )
        out["oracle_any_correct"] = int(bool(correct_arms))
        out["oracle_correct_count"] = len(correct_arms)
        rows_out.append(out)

    fieldnames = list(rows_out[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Wrote {len(rows_out)} rows x {len(fieldnames)} columns to {args.output}")


if __name__ == "__main__":
    main()
