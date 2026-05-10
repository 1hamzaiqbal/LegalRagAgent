#!/usr/bin/env python3
"""Analyze row-level adaptive HyRE portfolios from existing detail logs.

This joins multiple method detail JSONLs by label and reports single-method
accuracy, oracle headroom, simple deterministic portfolios, and agreement
buckets. It is intentionally log-only: no Chroma, embeddings, or LLM calls.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            label = row.get("label")
            if not label:
                raise SystemExit(f"{path}:{line_no}: missing label")
            rows[str(label)] = row
    if not rows:
        raise SystemExit(f"{path}: no rows loaded")
    return rows


def pred(row: dict[str, Any] | None) -> str:
    value = None if row is None else row.get("predicted_answer")
    return "" if value is None else str(value)


def correct(row: dict[str, Any] | None) -> bool:
    return bool(row and row.get("is_correct"))


def gold(row: dict[str, Any]) -> str:
    return str(row.get("correct_answer") or row.get("gold") or "")


def calls(row: dict[str, Any] | None) -> float:
    if not row:
        return 0.0
    try:
        return float(row.get("llm_calls") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def pct(num: float, den: float) -> str:
    return "n/a" if den == 0 else f"{100 * num / den:.1f}%"


def vote(values: list[str], priority: list[str]) -> str:
    values = [v for v in values if v]
    if not values:
        return ""
    counts = Counter(values)
    best = max(counts.values())
    winners = {value for value, count in counts.items() if count == best}
    for value in priority:
        if value in winners:
            return value
    return values[0]


def load_logs(log_args: list[list[str]]) -> dict[str, dict[str, dict[str, Any]]]:
    datasets: dict[str, dict[str, dict[str, Any]]] = {}
    for dataset, method, path in log_args:
        datasets.setdefault(dataset, {})[method] = load_jsonl(Path(path))
    return datasets


def policy_answer(policy: str, rows: dict[str, dict[str, Any]], methods: list[str]) -> tuple[str, float]:
    if policy == "first":
        method = methods[0]
        return pred(rows.get(method)), calls(rows.get(method))
    if policy == "majority":
        answers = [pred(rows.get(method)) for method in methods]
        return vote(answers, answers), sum(calls(rows.get(method)) for method in methods)
    if policy.startswith("prefer:"):
        order = [item.strip() for item in policy.split(":", 1)[1].split(",") if item.strip()]
        for method in order:
            answer = pred(rows.get(method))
            if answer:
                return answer, calls(rows.get(method))
        return "", 0.0
    raise ValueError(f"unknown policy: {policy}")


def policy_applies(policy: str, methods: list[str]) -> bool:
    if policy in {"first", "majority"}:
        return True
    if policy.startswith("prefer:"):
        order = [item.strip() for item in policy.split(":", 1)[1].split(",") if item.strip()]
        return any(method in methods for method in order)
    return True


def analyze_dataset(dataset: str, logs: dict[str, dict[str, Any]], policies: list[str]) -> str:
    methods = list(logs)
    labels = sorted(set.intersection(*(set(rows) for rows in logs.values())))
    if not labels:
        raise SystemExit(f"{dataset}: no overlapping labels across {methods}")

    joined = {label: {method: logs[method][label] for method in methods} for label in labels}
    lines = [f"# Adaptive Portfolio Analysis: {dataset}", ""]
    lines.append(f"Rows joined: {len(labels)}")
    lines.append("")
    lines.append("## Single Methods")
    lines.append("")
    lines.append("| Method | Correct | Accuracy | Avg calls | Gold retrieved | Route / policy |")
    lines.append("|---|---:|---:|---:|---:|---|")
    method_correct: dict[str, int] = {}
    for method in methods:
        rows = [joined[label][method] for label in labels]
        good = sum(correct(row) for row in rows)
        method_correct[method] = good
        avg_calls = sum(calls(row) for row in rows) / len(rows)
        retrieved = [row.get("gold_retrieved") for row in rows if "gold_retrieved" in row]
        gold_cell = "n/a" if not retrieved else f"{sum(bool(x) for x in retrieved)}/{len(retrieved)}"
        route_values = Counter(str(row.get("hyre_route") or row.get("adaptive_policy") or "") for row in rows)
        route = ", ".join(f"{name}:{count}" for name, count in route_values.most_common(2) if name)
        lines.append(f"| `{method}` | {good}/{len(rows)} | {pct(good, len(rows))} | {avg_calls:.2f} | {gold_cell} | {route} |")

    oracle = sum(any(correct(rows[method]) for method in methods) for rows in joined.values())
    all_correct = sum(all(correct(rows[method]) for method in methods) for rows in joined.values())
    none_correct = len(labels) - oracle
    best_method, best_correct = max(method_correct.items(), key=lambda item: item[1])
    lines.extend(
        [
            "",
            "## Headroom",
            "",
            f"- Best single method: `{best_method}` at {best_correct}/{len(labels)} = {pct(best_correct, len(labels))}.",
            f"- Any-method oracle: {oracle}/{len(labels)} = {pct(oracle, len(labels))}.",
            f"- All methods correct: {all_correct}/{len(labels)} = {pct(all_correct, len(labels))}.",
            f"- No method correct: {none_correct}/{len(labels)} = {pct(none_correct, len(labels))}.",
            f"- Recoverable above best single method: {oracle - best_correct} rows.",
            "",
            "## Deterministic Portfolio Policies",
            "",
            "| Policy | Correct | Accuracy | Avg calls counted | Delta vs best |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for policy in policies:
        if not policy_applies(policy, methods):
            continue
        good = 0
        total_calls = 0.0
        for rows in joined.values():
            answer, spent = policy_answer(policy, rows, methods)
            good += int(answer == gold(next(iter(rows.values()))))
            total_calls += spent
        lines.append(
            f"| `{policy}` | {good}/{len(labels)} | {pct(good, len(labels))} | "
            f"{total_calls / len(labels):.2f} | {good - best_correct:+d} |"
        )

    lines.extend(["", "## Agreement Buckets", ""])
    if len(methods) >= 2:
        primary, secondary = methods[:2]
        lines.append(f"Primary pair: `{primary}` vs `{secondary}`")
        lines.append("")
        lines.append("| Bucket | Rows | Primary acc | Secondary acc | Oracle acc |")
        lines.append("|---|---:|---:|---:|---:|")
        buckets = {"agree": [], "disagree": []}
        for label, rows in joined.items():
            bucket = "agree" if pred(rows[primary]) == pred(rows[secondary]) else "disagree"
            buckets[bucket].append(rows)
        for bucket, bucket_rows in buckets.items():
            if not bucket_rows:
                continue
            primary_good = sum(correct(rows[primary]) for rows in bucket_rows)
            secondary_good = sum(correct(rows[secondary]) for rows in bucket_rows)
            oracle_good = sum(any(correct(rows[method]) for method in methods) for rows in bucket_rows)
            n = len(bucket_rows)
            lines.append(
                f"| {bucket} | {n} | {primary_good}/{n} = {pct(primary_good, n)} | "
                f"{secondary_good}/{n} = {pct(secondary_good, n)} | {oracle_good}/{n} = {pct(oracle_good, n)} |"
            )

    lines.extend(["", "## Disagreement Examples", ""])
    shown = 0
    for label, rows in joined.items():
        answers = {method: pred(rows[method]) for method in methods}
        if len(set(answer for answer in answers.values() if answer)) <= 1:
            continue
        flags = " ".join(f"{method}={answers[method]}{'*' if correct(rows[method]) else ''}" for method in methods)
        lines.append(f"- `{label}` gold={gold(next(iter(rows.values())))} | {flags}")
        shown += 1
        if shown >= 10:
            break
    if shown == 0:
        lines.append("- No prediction disagreements among joined methods.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", action="append", nargs=3, metavar=("DATASET", "METHOD", "PATH"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--policy",
        action="append",
        default=[],
        help="Portfolio policy: majority, first, or prefer:m1,m2,m3. May be repeated.",
    )
    args = parser.parse_args()

    policies = args.policy or ["majority", "first"]
    datasets = load_logs(args.log)
    sections = [analyze_dataset(dataset, logs, policies) for dataset, logs in datasets.items()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n\n".join(sections), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
