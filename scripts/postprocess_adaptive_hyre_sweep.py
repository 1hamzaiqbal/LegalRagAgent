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
ADAPTIVE_MODES = {"adaptive_snap_hyre", "adaptive_snap_hyre_anchor", "snap_hyre_option", "snap_hyre_state"}
POLICY_MODES = {"adaptive_snap_hyre", "adaptive_snap_hyre_anchor"}
DEFAULT_MODES = {
    "rag_simple",
    "rag_state_filter",
    "rag_snap_hyde_2call",
    "snap_hyre_option",
    "snap_hyre_state",
    "adaptive_snap_hyre",
    "adaptive_snap_hyre_anchor",
}
DEFAULT_COMPARISONS = {
    "barexam": [("rag_simple", "adaptive_snap_hyre"), ("rag_snap_hyde_2call", "adaptive_snap_hyre"), ("snap_hyre_option", "adaptive_snap_hyre"), ("adaptive_snap_hyre", "adaptive_snap_hyre_anchor")],
    "casehold": [("rag_simple", "adaptive_snap_hyre"), ("rag_snap_hyde_2call", "adaptive_snap_hyre"), ("snap_hyre_option", "adaptive_snap_hyre"), ("adaptive_snap_hyre", "adaptive_snap_hyre_anchor")],
    "legalbench_scalr": [("rag_simple", "adaptive_snap_hyre"), ("rag_snap_hyde_2call", "adaptive_snap_hyre"), ("snap_hyre_option", "adaptive_snap_hyre"), ("adaptive_snap_hyre", "adaptive_snap_hyre_anchor")],
    "housing": [("rag_state_filter", "adaptive_snap_hyre"), ("snap_hyre_state", "adaptive_snap_hyre"), ("adaptive_snap_hyre", "adaptive_snap_hyre_anchor")],
}
EXPECTED_ADAPTIVE_MODES = {
    "barexam": ("snap_hyre_option", "adaptive_snap_hyre", "adaptive_snap_hyre_anchor"),
    "casehold": ("snap_hyre_option", "adaptive_snap_hyre", "adaptive_snap_hyre_anchor"),
    "legalbench_scalr": ("snap_hyre_option", "adaptive_snap_hyre", "adaptive_snap_hyre_anchor"),
    "housing": ("snap_hyre_state", "adaptive_snap_hyre", "adaptive_snap_hyre_anchor"),
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


def delta_pp(treatment: float, baseline: float) -> float:
    return 100 * (treatment - baseline)


def audit_status(summary: dict[str, Any]) -> str:
    if summary["mode"] not in ADAPTIVE_MODES:
        return "-"
    report, ok = hyre_audit.audit(Path(summary["path"]), legal_only=True)
    if ok:
        return "PASS"
    last = [line for line in report.splitlines() if line.startswith("FAIL ")]
    return last[0] if last else "FAIL"


def summary_record(summary: dict[str, Any]) -> dict[str, Any]:
    """Serializable run summary without full per-row payloads."""
    n = summary["n"]
    return {
        "dataset": summary["dataset"],
        "mode": summary["mode"],
        "provider": summary["provider"],
        "tag": summary["tag"],
        "n": n,
        "correct": summary["correct"],
        "accuracy": summary["accuracy"],
        "gold_retrieved": summary["gold_retrieved"],
        "empty_retrieval": summary["empty_retrieval"],
        "avg_llm_calls": summary["avg_llm_calls"],
        "audit": audit_status(summary),
        "path": str(summary["path"]),
    }


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

    append_parity_frontier(lines, selected)

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


def _best_summary(summaries: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not summaries:
        return None
    return max(summaries, key=lambda s: (s["accuracy"], -s["avg_llm_calls"], s["n"]))


def _summary_cell(summary: dict[str, Any] | None, field: str) -> str:
    if not summary:
        return "-"
    if field == "accuracy":
        return format_pct(float(summary[field]))
    if field == "avg_llm_calls":
        return f"{float(summary[field]):.2f}"
    return str(summary[field])


def coverage_records(selected: dict[tuple[str, str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for dataset in LEGAL_DATASETS:
        providers = sorted({provider for (d, _m, provider) in selected if d == dataset})
        if not providers:
            records.append({
                "dataset": dataset,
                "provider": "",
                "present_adaptive_modes": [],
                "missing_adaptive_modes": list(EXPECTED_ADAPTIVE_MODES[dataset]),
                "status": "MISSING",
            })
            continue
        for provider in providers:
            present = sorted(
                mode for (d, mode, p) in selected
                if d == dataset and p == provider and mode in EXPECTED_ADAPTIVE_MODES[dataset]
            )
            missing = [mode for mode in EXPECTED_ADAPTIVE_MODES[dataset] if mode not in present]
            records.append({
                "dataset": dataset,
                "provider": provider,
                "present_adaptive_modes": present,
                "missing_adaptive_modes": missing,
                "status": "READY" if not missing else "MISSING",
            })
    return records


def parity_records(selected: dict[tuple[str, str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for dataset in LEGAL_DATASETS:
        providers = sorted({provider for (d, _m, provider) in selected if d == dataset})
        if not providers:
            records.append({
                "dataset": dataset,
                "provider": "",
                "best_control": None,
                "best_adaptive_policy": None,
                "delta_pp": None,
                "status": "MISSING",
            })
            continue
        for provider in providers:
            controls = [
                summary for (d, mode, p), summary in selected.items()
                if d == dataset and p == provider and mode not in POLICY_MODES
            ]
            policies = [
                summary for (d, mode, p), summary in selected.items()
                if d == dataset and p == provider and mode in POLICY_MODES
            ]
            best_control = _best_summary(controls)
            best_policy = _best_summary(policies)
            if not best_control or not best_policy:
                records.append({
                    "dataset": dataset,
                    "provider": provider,
                    "best_control": summary_record(best_control) if best_control else None,
                    "best_adaptive_policy": summary_record(best_policy) if best_policy else None,
                    "delta_pp": None,
                    "status": "MISSING",
                })
                continue
            gap = delta_pp(best_policy["accuracy"], best_control["accuracy"])
            status = "PARITY" if gap >= -1.0 else "GAP"
            if best_policy["accuracy"] > best_control["accuracy"]:
                status = "LEADS"
            records.append({
                "dataset": dataset,
                "provider": provider,
                "best_control": summary_record(best_control),
                "best_adaptive_policy": summary_record(best_policy),
                "delta_pp": gap,
                "status": status,
            })
    return records


def append_parity_frontier(
    lines: list[str],
    selected: dict[tuple[str, str, str], dict[str, Any]],
) -> None:
    lines.extend([
        "",
        "## Adaptive Parity Frontier",
        "",
        "| Dataset | Provider | Best control | Acc | Calls | Best adaptive policy | Acc | Calls | Delta pp | Status |",
        "|---|---|---|---:|---:|---|---:|---:|---:|---|",
    ])
    for record in parity_records(selected):
        best_control = record["best_control"]
        best_policy = record["best_adaptive_policy"]
        gap = record["delta_pp"]
        gap_cell = f"{gap:.1f}" if gap is not None else "-"
        lines.append(
            f"| {record['dataset']} | {record['provider'] or '-'} | "
            f"{_summary_cell(best_control, 'mode')} | "
            f"{_summary_cell(best_control, 'accuracy')} | "
            f"{_summary_cell(best_control, 'avg_llm_calls')} | "
            f"{_summary_cell(best_policy, 'mode')} | "
            f"{_summary_cell(best_policy, 'accuracy')} | "
            f"{_summary_cell(best_policy, 'avg_llm_calls')} | "
            f"{gap_cell} | {record['status']} |"
        )


def build_json_summary(selected: dict[tuple[str, str, str], dict[str, Any]]) -> dict[str, Any]:
    return {
        "legal_datasets": list(LEGAL_DATASETS),
        "expected_adaptive_modes": {
            dataset: list(modes) for dataset, modes in EXPECTED_ADAPTIVE_MODES.items()
        },
        "latest_logs": [
            summary_record(summary)
            for summary in sorted(
                selected.values(),
                key=lambda s: (s["dataset"], s["provider"], s["mode"]),
            )
        ],
        "adaptive_coverage": coverage_records(selected),
        "adaptive_parity_frontier": parity_records(selected),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, default=REPO_ROOT / "logs")
    parser.add_argument("--pattern", default="eval_*_detail.jsonl")
    parser.add_argument("--output", type=Path, help="Optional markdown output path")
    parser.add_argument("--json-output", type=Path, help="Optional machine-readable JSON summary path")
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
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(build_json_summary(selected), indent=2) + "\n")
    print(report)


if __name__ == "__main__":
    main()
