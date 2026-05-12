#!/usr/bin/env python3
"""Evaluate a diagnostic controller route plan against diagnostic summaries.

This is an evidence summarizer, not a fresh benchmark runner. It consumes the
route plan emitted by `diagnostic_controller.py` and the diagnostic table emitted
by `build_rag_diagnostic_table.py`, then compares the selected route for each
dataset against simple/control rows that are already present in the diagnostics.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


BASELINE_METHODS = {
    "rag_simple",
    "rag_state_filter",
}


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def delta_pp(a: float | None, b: float | None) -> str:
    if a is None or b is None:
        return "n/a"
    return f"{100 * (a - b):+.1f}"


def key(dataset: str, method: str) -> tuple[str, str]:
    return (dataset, method)


def best(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(rows, key=lambda row: row.get("accuracy") or 0.0)


def evaluate(diagnostics: list[dict[str, Any]], routes: list[dict[str, Any]]) -> dict[str, Any]:
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in diagnostics:
        dataset = str(row["dataset"])
        method = str(row["method"])
        by_dataset[dataset].append(row)
        by_key[key(dataset, method)] = row

    records: list[dict[str, Any]] = []
    for route in routes:
        dataset = str(route["dataset"])
        route_method = str(route["route"])
        selected = by_key.get(key(dataset, route_method))
        if selected is None:
            records.append({
                "dataset": dataset,
                "route": route_method,
                "status": "MISSING_ROUTE_ROW",
            })
            continue
        dataset_rows = by_dataset[dataset]
        baseline = best([row for row in dataset_rows if row["method"] in BASELINE_METHODS])
        best_available = best(dataset_rows)
        same_n_rows = [row for row in dataset_rows if row.get("n") == selected.get("n")]
        best_same_n = best(same_n_rows)
        mixed_n = bool(baseline and baseline.get("n") != selected.get("n")) or bool(best_available and best_available.get("n") != selected.get("n"))
        records.append({
            "dataset": dataset,
            "bottleneck": route.get("bottleneck"),
            "route": route_method,
            "n": selected.get("n"),
            "accuracy": selected.get("accuracy"),
            "avg_calls": selected.get("avg_calls"),
            "baseline_method": baseline.get("method") if baseline else None,
            "baseline_n": baseline.get("n") if baseline else None,
            "baseline_accuracy": baseline.get("accuracy") if baseline else None,
            "delta_vs_baseline_pp": (selected.get("accuracy") or 0.0) - (baseline.get("accuracy") or 0.0) if baseline else None,
            "best_available_method": best_available.get("method") if best_available else None,
            "best_available_n": best_available.get("n") if best_available else None,
            "best_available_accuracy": best_available.get("accuracy") if best_available else None,
            "delta_vs_best_available_pp": (selected.get("accuracy") or 0.0) - (best_available.get("accuracy") or 0.0) if best_available else None,
            "best_same_n_method": best_same_n.get("method") if best_same_n else None,
            "best_same_n_accuracy": best_same_n.get("accuracy") if best_same_n else None,
            "mixed_n_warning": mixed_n,
            "secondary_flags": route.get("secondary_flags", []),
            "status": "PASS_WITH_MIXED_N_CAVEAT" if mixed_n else "PASS",
        })

    valid = [record for record in records if record.get("status") != "MISSING_ROUTE_ROW"]
    macro_acc = sum(record["accuracy"] for record in valid) / len(valid) if valid else None
    macro_calls = sum(record["avg_calls"] for record in valid) / len(valid) if valid else None
    return {
        "records": records,
        "macro_accuracy": macro_acc,
        "macro_avg_calls": macro_calls,
        "mixed_n_datasets": [record["dataset"] for record in valid if record["mixed_n_warning"]],
    }


def to_markdown(result: dict[str, Any], diagnostics_path: Path, routes_path: Path) -> str:
    lines = [
        "# Diagnostic Controller Evaluation",
        "",
        f"Diagnostics: `{diagnostics_path}`",
        f"Routes: `{routes_path}`",
        "",
        "This report evaluates the route plan against existing diagnostic summaries. It is not a fresh held-out benchmark.",
        "",
        f"- Macro accuracy across selected routes: {pct(result['macro_accuracy'])}",
        f"- Macro average LLM calls: {result['macro_avg_calls']:.2f}" if result["macro_avg_calls"] is not None else "- Macro average LLM calls: n/a",
        f"- Mixed-N caveat datasets: {', '.join(result['mixed_n_datasets']) if result['mixed_n_datasets'] else 'none'}",
        "",
        "| Dataset | Bottleneck | Selected route | N | Acc | Calls | Baseline | Delta vs baseline pp | Best available | Delta vs best pp | Best same-N | Status |",
        "|---|---|---|---:|---:|---:|---|---:|---|---:|---|---|",
    ]
    for record in result["records"]:
        if record.get("status") == "MISSING_ROUTE_ROW":
            lines.append(
                f"| {record['dataset']} | n/a | `{record['route']}` | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | MISSING_ROUTE_ROW |"
            )
            continue
        baseline = f"`{record['baseline_method']}` ({record['baseline_n']})" if record["baseline_method"] else "n/a"
        best_available = f"`{record['best_available_method']}` ({record['best_available_n']})" if record["best_available_method"] else "n/a"
        best_same_n = f"`{record['best_same_n_method']}`" if record["best_same_n_method"] else "n/a"
        lines.append(
            f"| {record['dataset']} | `{record['bottleneck']}` | `{record['route']}` | {record['n']} | "
            f"{pct(record['accuracy'])} | {record['avg_calls']:.2f} | {baseline} | "
            f"{delta_pp(record['accuracy'], record['baseline_accuracy'])} | {best_available} | "
            f"{delta_pp(record['accuracy'], record['best_available_accuracy'])} | {best_same_n} ({pct(record['best_same_n_accuracy'])}) | "
            f"{record['status']} |"
        )
    lines.extend([
        "",
        "## Reading",
        "",
        "- `PASS_WITH_MIXED_N_CAVEAT` means the selected route is present and scored, but at least one comparison row uses a different N. Treat the route as a policy hypothesis, not a paired claim.",
        "- `Delta vs best pp` should be zero for a controller that selects the best available route in the diagnostic table. Negative values indicate that the controller intentionally chose a non-best route, usually for cost or calibration reasons.",
        "- A paper-grade controller result still needs a same-slice or held-out evaluation where all candidate routes are available on the same questions.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    diagnostics = json.loads(args.diagnostics.read_text())
    routes = json.loads(args.routes.read_text())
    result = evaluate(diagnostics, routes)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(to_markdown(result, args.diagnostics, args.routes), encoding="utf-8")
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {args.output_md}")
    if args.output_json:
        print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
