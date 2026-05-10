#!/usr/bin/env python3
"""Route legal RAG tasks from bottleneck diagnostics.

This is the first rule-based controller draft. It consumes the JSON emitted by
build_rag_diagnostic_table.py and writes an auditable route plan.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def best_by(dataset_rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    return max(dataset_rows, key=lambda item: item.get(key) or 0.0)


def find_method(dataset_rows: list[dict[str, Any]], *names: str) -> dict[str, Any] | None:
    wanted = set(names)
    for row in dataset_rows:
        if row.get("method") in wanted:
            return row
    return None


def route_dataset(dataset: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    best = best_by(rows, "accuracy")
    plain = find_method(rows, "rag_simple", "rag_state_filter")
    rewrite = find_method(rows, "rag_rewrite")
    snap = find_method(rows, "rag_snap_hyde_2call", "adaptive_snap_hyre_v2", "adaptive_snap_hyre_diverse", "adaptive_snap_hyre_frontier")
    verifier = find_method(rows, "adaptive_snap_hyre_housing_verifier")
    disagreement = find_method(rows, "adaptive_snap_hyre_disagreement_replay", "adaptive_snap_hyre_disagreement_majority_prior")

    rationale: list[str] = []
    secondary: list[str] = []
    bottleneck = "stable_easy_case"
    route = str(best["method"])

    if dataset == "housing" and verifier:
        bottleneck = "statutory_entailment_gap"
        route = str(verifier["method"])
        rationale.append("Housing-style yes/no statutory QA benefits from conservative entailment verification.")
        if plain:
            rationale.append(f"Verifier accuracy {pct(verifier['accuracy'])} vs base {pct(plain['accuracy'])}.")
    elif dataset == "casehold":
        bottleneck = "answer_conversion_gap"
        route = str(best["method"])
        rationale.append("Gold retrieval improves without meaningful accuracy lift, indicating answer-option conversion rather than retrieval alone.")
        secondary.append("reject_or_escalate")
    elif disagreement and disagreement.get("accuracy", 0.0) >= best.get("accuracy", 0.0):
        bottleneck = "method_disagreement_gap"
        route = str(disagreement["method"])
        rationale.append("Cached disagreement arbitration matches or beats the strongest source route at low marginal call cost.")
    elif rewrite and plain and rewrite.get("accuracy", 0.0) > plain.get("accuracy", 0.0) and rewrite.get("accuracy", 0.0) >= best.get("accuracy", 0.0) - 0.01:
        bottleneck = "query_retrieval_gap"
        route = str(rewrite["method"])
        rationale.append("Legal query rewriting closes the gap to the strongest route, so use query formulation before generated-reasoning routes.")
    elif plain and snap and (snap.get("gold_retrieved", 0) > plain.get("gold_retrieved", 0)):
        bottleneck = "query_retrieval_gap"
        route = str(best["method"])
        rationale.append("Snap/HyRE route increases gold exposure over plain retrieval.")
        if rewrite:
            rationale.append(f"Query rewrite calibration accuracy is {pct(rewrite.get('accuracy'))}; compare before promoting HyRE-specific claims.")
    else:
        rationale.append("No diagnostic route beats the strongest current method; keep cheapest reliable route.")

    gold_wrong = best.get("gold_retrieved_wrong") or 0
    if gold_wrong:
        secondary.append("answer_conversion_gap")
        rationale.append(f"Best route still has {gold_wrong} gold-retrieved-but-wrong rows.")

    return {
        "dataset": dataset,
        "bottleneck": bottleneck,
        "route": route,
        "secondary_flags": sorted(set(secondary)),
        "best_method": best["method"],
        "best_accuracy": best["accuracy"],
        "avg_calls": best["avg_calls"],
        "rationale": rationale,
    }


def to_markdown(routes: list[dict[str, Any]], source: Path) -> str:
    lines = [
        "# Diagnostic Controller Route Plan",
        "",
        f"Source diagnostics: `{source}`",
        "",
        "| Dataset | Bottleneck | Route | Best acc | Calls | Secondary flags |",
        "|---|---|---|---:|---:|---|",
    ]
    for route in routes:
        flags = ", ".join(route["secondary_flags"]) if route["secondary_flags"] else "-"
        lines.append(
            f"| {route['dataset']} | `{route['bottleneck']}` | `{route['route']}` | "
            f"{pct(route['best_accuracy'])} | {route['avg_calls']:.2f} | {flags} |"
        )
    lines.extend(["", "## Rationale", ""])
    for route in routes:
        lines.append(f"### {route['dataset']}")
        for item in route["rationale"]:
            lines.append(f"- {item}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    items = json.loads(args.diagnostics.read_text())
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[str(item["dataset"])].append(item)
    routes = [route_dataset(dataset, rows) for dataset, rows in sorted(grouped.items())]

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(to_markdown(routes, args.diagnostics), encoding="utf-8")
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(routes, indent=2), encoding="utf-8")
    print(f"wrote {args.output_md}")
    if args.output_json:
        print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
