#!/usr/bin/env python3
"""Estimate per-question routing headroom across completed eval detail logs.

This does not train a router. It answers the prior feasibility question:
if a controller could pick among several already-run methods per question, how
much accuracy and cost headroom is available?
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from collections import Counter
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
        raise SystemExit(f"Invalid --arm {raw!r}; label cannot be empty")
    matches = sorted(glob.glob(pattern))
    if not matches and Path(pattern).exists():
        matches = [pattern]
    if len(matches) != 1:
        raise SystemExit(f"{raw!r}: expected exactly one log path, matched {len(matches)}")
    return label, Path(matches[0])


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def choose_key(rows_by_arm: dict[str, list[dict[str, Any]]], requested: str | None) -> str:
    labels = list(rows_by_arm)
    key = requested
    if key:
        for label in labels:
            values = [row.get(key) for row in rows_by_arm[label]]
            if any(value is None for value in values) or len(set(values)) != len(values):
                raise SystemExit(f"{label}: requested key {key!r} is missing or non-unique")
        return key

    first = rows_by_arm[labels[0]]
    for other_label in labels[1:]:
        key = compute_mcnemar.choose_key_field(first, rows_by_arm[other_label], None)
        if key:
            return key
    raise SystemExit("Could not infer a shared key")


def reward(row: dict[str, Any], correct: bool, call_penalty: float, latency_penalty: float) -> float:
    calls = float(row.get("llm_calls_actual", row.get("llm_calls", 0)) or 0)
    latency = float(row.get("elapsed_sec", 0) or 0)
    return float(correct) - call_penalty * calls - latency_penalty * latency


def summarize_arm(label: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct = [compute_mcnemar.correct_flag(row) for row in rows]
    calls = [float(row.get("llm_calls_actual", row.get("llm_calls", 0)) or 0) for row in rows]
    latency = [float(row.get("elapsed_sec", 0) or 0) for row in rows]
    return {
        "label": label,
        "n": len(rows),
        "accuracy": sum(correct) / len(correct),
        "avg_calls": mean(calls),
        "avg_latency": mean(latency),
    }


def markdown(
    title: str,
    arm_summaries: list[dict[str, Any]],
    path_by_arm: dict[str, Path],
    common_n: int,
    key: str,
    accuracy_first: dict[str, Any],
    reward_first: dict[str, Any],
    call_penalty: float,
    latency_penalty: float,
) -> str:
    lines = [f"# {title}", ""]
    lines.append(
        "This is an offline upper-bound analysis. It does not prove an online router can identify the best arm; "
        "it estimates whether enough per-question variation exists to justify building one."
    )
    lines.append("")
    lines.extend([
        "## Arm Summary",
        "",
        "| Arm | N | Accuracy | Calls/q | Sec/q |",
        "|---|---:|---:|---:|---:|",
    ])
    for item in arm_summaries:
        lines.append(
            f"| {item['label']} | {item['n']} | {fmt_pct(item['accuracy'])} | "
            f"{item['avg_calls']:.2f} | {item['avg_latency']:.1f} |"
        )

    lines.extend([
        "",
        "## Oracle Routing Upper Bounds",
        "",
        f"Common rows: `{common_n}` on key `{key}`.",
        "",
        "| Policy | Accuracy | Calls/q | Sec/q | Notes |",
        "|---|---:|---:|---:|---|",
        (
            f"| Accuracy-first oracle | {fmt_pct(accuracy_first['accuracy'])} | "
            f"{accuracy_first['avg_calls']:.2f} | {accuracy_first['avg_latency']:.1f} | "
            "Choose the cheapest correct arm when any arm is correct; otherwise choose the cheapest arm. |"
        ),
        (
            f"| Reward oracle | {fmt_pct(reward_first['accuracy'])} | "
            f"{reward_first['avg_calls']:.2f} | {reward_first['avg_latency']:.1f} | "
            f"Maximize `correct - {call_penalty:g}*calls - {latency_penalty:g}*sec`. |"
        ),
        "",
        "## Chosen Arm Distribution",
        "",
        "| Policy | Arm | Count |",
        "|---|---|---:|",
    ])
    for policy_name, item in (("accuracy_first", accuracy_first), ("reward", reward_first)):
        for arm, count in sorted(item["chosen_counts"].items()):
            lines.append(f"| {policy_name} | {arm} | {count} |")

    lines.extend(["", "## Source Logs", ""])
    for label, path in path_by_arm.items():
        lines.append(f"- `{label}`: `{path}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True, help="Arm detail log as label=path")
    parser.add_argument("--key", help="Override join key")
    parser.add_argument("--title", default="Routing Oracle")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--call-penalty", type=float, default=0.0)
    parser.add_argument("--latency-penalty", type=float, default=0.0)
    args = parser.parse_args()

    rows_by_arm: dict[str, list[dict[str, Any]]] = {}
    path_by_arm: dict[str, Path] = {}
    for raw in args.arm:
        label, path = resolve_arm(raw)
        if label in rows_by_arm:
            raise SystemExit(f"Duplicate arm label {label!r}")
        rows_by_arm[label] = load_jsonl(path)
        path_by_arm[label] = path

    key = choose_key(rows_by_arm, args.key)
    by_key = {
        label: {row[key]: row for row in rows}
        for label, rows in rows_by_arm.items()
    }
    common_keys = sorted(set.intersection(*(set(rows) for rows in by_key.values())))
    if not common_keys:
        raise SystemExit(f"No common rows across arms on key {key!r}")

    arm_summaries = [summarize_arm(label, [by_key[label][k] for k in common_keys]) for label in rows_by_arm]
    acc_choices: list[tuple[str, dict[str, Any], bool]] = []
    reward_choices: list[tuple[str, dict[str, Any], bool]] = []
    for row_key in common_keys:
        candidates = []
        for label in rows_by_arm:
            row = by_key[label][row_key]
            correct = compute_mcnemar.correct_flag(row)
            calls = float(row.get("llm_calls_actual", row.get("llm_calls", 0)) or 0)
            latency = float(row.get("elapsed_sec", 0) or 0)
            candidates.append((label, row, correct, calls, latency))

        correct_candidates = [item for item in candidates if item[2]]
        if correct_candidates:
            acc_pick = min(correct_candidates, key=lambda item: (item[3], item[4], item[0]))
        else:
            acc_pick = min(candidates, key=lambda item: (item[3], item[4], item[0]))
        acc_choices.append((acc_pick[0], acc_pick[1], acc_pick[2]))

        reward_pick = max(
            candidates,
            key=lambda item: (
                reward(item[1], item[2], args.call_penalty, args.latency_penalty),
                -item[3],
                -item[4],
                item[0],
            ),
        )
        reward_choices.append((reward_pick[0], reward_pick[1], reward_pick[2]))

    def choice_summary(choices: list[tuple[str, dict[str, Any], bool]]) -> dict[str, Any]:
        return {
            "accuracy": sum(1 for _label, _row, correct in choices if correct) / len(choices),
            "avg_calls": mean([float(row.get("llm_calls_actual", row.get("llm_calls", 0)) or 0) for _label, row, _correct in choices]),
            "avg_latency": mean([float(row.get("elapsed_sec", 0) or 0) for _label, row, _correct in choices]),
            "chosen_counts": Counter(label for label, _row, _correct in choices),
        }

    output = markdown(
        args.title,
        arm_summaries,
        path_by_arm,
        len(common_keys),
        key,
        choice_summary(acc_choices),
        choice_summary(reward_choices),
        args.call_penalty,
        args.latency_penalty,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
