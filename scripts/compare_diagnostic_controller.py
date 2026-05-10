#!/usr/bin/env python3
"""Compare the diagnostic controller with fixed legal RAG portfolios.

This is an evidence-table generator, not a benchmark runner. It reads the
current diagnostic summaries and route plan, then reports portfolio-level
accuracy/cost against fixed alternatives where same-slice rows exist.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


BASELINE = {
    "barexam": "rag_simple",
    "casehold": "rag_simple",
    "housing": "rag_state_filter",
    "legalbench_scalr": "rag_simple",
}

HYRE_ONLY = {
    "barexam": "adaptive_snap_hyre_v2",
    "casehold": "adaptive_snap_hyre_diverse",
    "housing": "adaptive_snap_hyre_diverse",
    "legalbench_scalr": "rag_snap_hyde_2call",
}

NON_ADAPTIVE_CANDIDATES = {
    "rag_simple",
    "rag_state_filter",
    "rag_rewrite",
    "rag_snap_hyde_2call",
}


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def row_key(dataset: str, method: str) -> tuple[str, str]:
    return dataset, method


def method_row(
    by_key: dict[tuple[str, str], dict[str, Any]], dataset: str, method: str
) -> dict[str, Any] | None:
    return by_key.get(row_key(dataset, method))


def portfolio_record(name: str, rows: list[dict[str, Any] | None]) -> dict[str, Any]:
    present = [row for row in rows if row is not None]
    return {
        "portfolio": name,
        "datasets_covered": len(present),
        "macro_accuracy": mean([float(row["accuracy"]) for row in present]),
        "macro_avg_calls": mean([float(row["avg_calls"]) for row in present]),
        "rows": present,
        "missing": len(rows) - len(present),
    }


def build(diagnostics: list[dict[str, Any]], routes: list[dict[str, Any]]) -> dict[str, Any]:
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in diagnostics:
        dataset = str(row["dataset"])
        method = str(row["method"])
        by_dataset[dataset].append(row)
        by_key[row_key(dataset, method)] = row

    datasets = sorted(by_dataset)
    route_by_dataset = {str(route["dataset"]): str(route["route"]) for route in routes}

    controller_rows = [method_row(by_key, dataset, route_by_dataset[dataset]) for dataset in datasets]
    baseline_rows = [method_row(by_key, dataset, BASELINE[dataset]) for dataset in datasets]
    hyre_rows = [method_row(by_key, dataset, HYRE_ONLY[dataset]) for dataset in datasets]
    rewrite_rows = [method_row(by_key, dataset, "rag_rewrite") for dataset in datasets]

    best_non_adaptive = []
    for dataset in datasets:
        rows = [
            row
            for row in by_dataset[dataset]
            if row["method"] in NON_ADAPTIVE_CANDIDATES and row.get("n") == 200
        ]
        best_non_adaptive.append(max(rows, key=lambda row: row.get("accuracy") or 0.0) if rows else None)

    portfolios = [
        portfolio_record("diagnostic_controller", controller_rows),
        portfolio_record("baseline_retrieval", baseline_rows),
        portfolio_record("fixed_hyre_only", hyre_rows),
        portfolio_record("best_non_adaptive_same_n", best_non_adaptive),
        portfolio_record("query_rewrite_available", rewrite_rows),
    ]

    return {
        "datasets": datasets,
        "portfolios": portfolios,
    }


def dataset_table(result: dict[str, Any]) -> list[str]:
    datasets = result["datasets"]
    rows_by_portfolio = {
        portfolio["portfolio"]: {str(row["dataset"]): row for row in portfolio["rows"]}
        for portfolio in result["portfolios"]
    }
    lines = [
        "| Dataset | Controller | Baseline retrieval | Fixed HyRE-only | Best non-adaptive same-N | Query rewrite |",
        "|---|---|---|---|---|---|",
    ]
    for dataset in datasets:
        cells = [dataset]
        for portfolio in [
            "diagnostic_controller",
            "baseline_retrieval",
            "fixed_hyre_only",
            "best_non_adaptive_same_n",
            "query_rewrite_available",
        ]:
            row = rows_by_portfolio[portfolio].get(dataset)
            if row is None:
                cells.append("n/a")
            else:
                cells.append(
                    f"`{row['method']}` ({int(row['n'])}, {pct(float(row['accuracy']))}, {float(row['avg_calls']):.2f} calls)"
                )
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def to_markdown(result: dict[str, Any], diagnostics_path: Path, routes_path: Path) -> str:
    lines = [
        "# Diagnostic Controller Portfolio Comparison",
        "",
        f"Diagnostics: `{diagnostics_path}`",
        f"Routes: `{routes_path}`",
        "",
        "This is a source-summary comparison over available calibration evidence, not a fresh held-out benchmark.",
        "",
        "| Portfolio | Datasets covered | Macro acc | Macro calls | Missing rows |",
        "|---|---:|---:|---:|---:|",
    ]
    for portfolio in result["portfolios"]:
        lines.append(
            f"| `{portfolio['portfolio']}` | {portfolio['datasets_covered']} | "
            f"{pct(portfolio['macro_accuracy'])} | "
            f"{portfolio['macro_avg_calls']:.2f} | {portfolio['missing']} |"
        )
    lines.extend(["", "## Dataset Rows", ""])
    lines.extend(dataset_table(result))
    lines.extend(
        [
            "",
            "## Reading",
            "",
            "- `diagnostic_controller` is the current bottleneck-aware route plan.",
            "- `baseline_retrieval` uses the simple retrieval baseline for each dataset, with HousingQA using the state-filter baseline because that is the current legal metadata baseline.",
            "- `fixed_hyre_only` removes the targeted verifier/disagreement routes and asks how far a HyRE-style route gets without bottleneck-specific adaptation.",
            "- `best_non_adaptive_same_n` only uses N=200 rows from non-adaptive methods in the diagnostic table.",
            "- `query_rewrite_available` includes N=50 rows where N=200 rewrite rows are not yet available, so do not compare its macro score as a same-slice result.",
            "",
        ]
    )
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
    result = build(diagnostics, routes)

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
