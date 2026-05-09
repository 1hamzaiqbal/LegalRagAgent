#!/usr/bin/env python3
"""Postprocess legal adaptive HyRE sweep logs into audit and paired-test tables."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import audit_adaptive_hyre_logs as hyre_audit  # type: ignore
import compute_mcnemar  # type: ignore


LEGAL_DATASETS = ("barexam", "housing", "casehold", "legalbench_scalr")
ADAPTIVE_MODES = {"adaptive_snap_hyre", "snap_hyre_option", "snap_hyre_state"}
DEFAULT_MODES = {
    "rag_simple",
    "rag_state_filter",
    "rag_snap_hyde_2call",
    "snap_hyre_option",
    "snap_hyre_state",
    "adaptive_snap_hyre",
}
DEFAULT_COMPARISONS = {
    "barexam": [("rag_simple", "adaptive_snap_hyre"), ("rag_snap_hyde_2call", "adaptive_snap_hyre"), ("snap_hyre_option", "adaptive_snap_hyre")],
    "casehold": [("rag_simple", "adaptive_snap_hyre"), ("rag_snap_hyde_2call", "adaptive_snap_hyre"), ("snap_hyre_option", "adaptive_snap_hyre")],
    "legalbench_scalr": [("rag_simple", "adaptive_snap_hyre"), ("rag_snap_hyde_2call", "adaptive_snap_hyre"), ("snap_hyre_option", "adaptive_snap_hyre")],
    "housing": [("rag_state_filter", "adaptive_snap_hyre"), ("snap_hyre_state", "adaptive_snap_hyre")],
}
EXPECTED_ADAPTIVE_MODES = {
    "barexam": ("snap_hyre_option", "adaptive_snap_hyre"),
    "casehold": ("snap_hyre_option", "adaptive_snap_hyre"),
    "legalbench_scalr": ("snap_hyre_option", "adaptive_snap_hyre"),
    "housing": ("snap_hyre_state", "adaptive_snap_hyre"),
}


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def discover_logs(log_dir: Path, pattern: str) -> list[Path]:
    return sorted(log_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)


def summarize_log(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    correct = sum(1 for r in rows if r.get("is_correct"))
    return {
        "path": path,
        "dataset": rows[0].get("dataset", "") if rows else "",
        "mode": rows[0].get("mode", "") if rows else "",
        "provider": rows[0].get("provider", "") if rows else "",
        "tag": rows[0].get("tag", "") if rows else "",
        "n": n,
        "correct": correct,
        "accuracy": correct / n if n else 0.0,
        "gold_retrieved": sum(1 for r in rows if r.get("gold_retrieved")),
        "empty_retrieval": sum(1 for r in rows if not r.get("retrieved_ids")),
        "avg_llm_calls": sum(float(r.get("llm_calls", 0) or 0) for r in rows) / n if n else 0.0,
        "mtime": path.stat().st_mtime,
    }


def select_latest(
    paths: list[Path],
    include_all_modes: bool,
    min_n: int,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    selected: dict[tuple[str, str, str], dict[str, Any]] = {}
    for path in paths:
        try:
            rows = load_rows(path)
        except (OSError, json.JSONDecodeError):
            continue
        if not rows:
            continue
        summary = summarize_log(path, rows)
        if summary["dataset"] not in LEGAL_DATASETS:
            continue
        if summary["n"] < min_n:
            continue
        if not include_all_modes and summary["mode"] not in DEFAULT_MODES:
            continue
        key = (summary["dataset"], summary["mode"], summary["provider"])
        if key not in selected or summary["mtime"] > selected[key]["mtime"]:
            summary["rows"] = rows
            selected[key] = summary
    return selected


def format_pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def audit_status(summary: dict[str, Any]) -> str:
    if summary["mode"] not in ADAPTIVE_MODES:
        return "-"
    report, ok = hyre_audit.audit(Path(summary["path"]), legal_only=True)
    if ok:
        return "PASS"
    last = [line for line in report.splitlines() if line.startswith("FAIL ")]
    return last[0] if last else "FAIL"


def pairwise_result(base: dict[str, Any], treat: dict[str, Any]) -> str:
    key_field = compute_mcnemar.choose_key_field(base["rows"], treat["rows"], None)
    result = compute_mcnemar.compute(
        base["rows"],
        treat["rows"],
        key_field,
        bootstrap_samples=2000,
        seed=42,
    )
    return (
        f"{result['n_paired']} | {format_pct(result['acc_baseline'])} | "
        f"{format_pct(result['acc_treatment'])} | {result['delta_pp']:.1f} | "
        f"{result['b']} / {result['c']} | {result['mcnemar_p']:.4g} | "
        f"[{result['ci_low']:.1f}, {result['ci_high']:.1f}]"
    )


def build_report(selected: dict[tuple[str, str, str], dict[str, Any]]) -> str:
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (dataset, _mode, _provider), summary in selected.items():
        by_dataset[dataset].append(summary)

    lines = [
        "# Adaptive HyRE Sweep Postprocess",
        "",
        "## Latest Logs",
        "",
        "| Dataset | Mode | Provider | N | Acc | Gold hit | Empty | Calls | Audit | Detail log |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for dataset in LEGAL_DATASETS:
        for summary in sorted(by_dataset.get(dataset, []), key=lambda s: (s["provider"], s["mode"])):
            n = summary["n"]
            lines.append(
                f"| {dataset} | {summary['mode']} | {summary['provider']} | {n} | "
                f"{format_pct(summary['accuracy'])} | {summary['gold_retrieved']}/{n} | "
                f"{summary['empty_retrieval']} | {summary['avg_llm_calls']:.2f} | "
                f"{audit_status(summary)} | `{summary['path']}` |"
            )

    lines.extend([
        "",
        "## Adaptive Coverage",
        "",
        "| Dataset | Provider | Present adaptive modes | Missing adaptive modes | Status |",
        "|---|---|---|---|---|",
    ])
    for dataset in LEGAL_DATASETS:
        providers = sorted({provider for (d, _m, provider) in selected if d == dataset})
        if not providers:
            expected = ", ".join(EXPECTED_ADAPTIVE_MODES[dataset])
            lines.append(f"| {dataset} | - | - | {expected} | MISSING |")
            continue
        for provider in providers:
            present = sorted(
                mode for (d, mode, p) in selected
                if d == dataset and p == provider and mode in EXPECTED_ADAPTIVE_MODES[dataset]
            )
            missing = [mode for mode in EXPECTED_ADAPTIVE_MODES[dataset] if mode not in present]
            lines.append(
                f"| {dataset} | {provider} | {', '.join(present) or '-'} | "
                f"{', '.join(missing) or '-'} | {'READY' if not missing else 'MISSING'} |"
            )

    lines.extend([
        "",
        "## Paired Comparisons",
        "",
        "| Dataset | Provider | Baseline -> Treatment | N | Baseline Acc | Treatment Acc | Delta pp | b / c | p | 95% CI pp |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for dataset, comparisons in DEFAULT_COMPARISONS.items():
        providers = sorted({provider for (d, _m, provider) in selected if d == dataset})
        for provider in providers:
            for base_mode, treat_mode in comparisons:
                base = selected.get((dataset, base_mode, provider))
                treat = selected.get((dataset, treat_mode, provider))
                if not base or not treat:
                    continue
                try:
                    result = pairwise_result(base, treat)
                except SystemExit as exc:
                    result = f"n/a | n/a | n/a | n/a | n/a | n/a | {exc}"
                lines.append(f"| {dataset} | {provider} | {base_mode} -> {treat_mode} | {result} |")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, default=REPO_ROOT / "logs")
    parser.add_argument("--pattern", default="eval_*_detail.jsonl")
    parser.add_argument("--output", type=Path, help="Optional markdown output path")
    parser.add_argument("--all-modes", action="store_true", help="Include all legal modes instead of only the adaptive sweep surface")
    parser.add_argument("--min-n", type=int, default=20, help="Minimum row count to include; use 1 for smoke logs")
    args = parser.parse_args()

    selected = select_latest(
        discover_logs(args.log_dir, args.pattern),
        include_all_modes=args.all_modes,
        min_n=args.min_n,
    )
    report = build_report(selected)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
