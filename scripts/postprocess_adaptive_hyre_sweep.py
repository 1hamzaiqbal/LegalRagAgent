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
ADAPTIVE_MODES = {"adaptive_snap_hyre", "adaptive_snap_hyre_anchor", "adaptive_snap_hyre_diverse", "adaptive_snap_hyre_v2", "adaptive_snap_hyre_frontier", "adaptive_snap_hyre_stability", "adaptive_snap_hyre_housing_verifier", "snap_hyre_option", "snap_hyre_state"}
POLICY_MODES = {"adaptive_snap_hyre", "adaptive_snap_hyre_anchor", "adaptive_snap_hyre_diverse", "adaptive_snap_hyre_v2", "adaptive_snap_hyre_frontier", "adaptive_snap_hyre_stability", "adaptive_snap_hyre_housing_verifier"}
DEFAULT_MODES = {
    "rag_simple",
    "rag_state_filter",
    "rag_snap_hyde_2call",
    "snap_hyre_option",
    "snap_hyre_state",
    "adaptive_snap_hyre",
    "adaptive_snap_hyre_anchor",
    "adaptive_snap_hyre_diverse",
    "adaptive_snap_hyre_v2",
    "adaptive_snap_hyre_frontier",
    "adaptive_snap_hyre_stability",
    "adaptive_snap_hyre_housing_verifier",
}
DEFAULT_COMPARISONS = {
    "barexam": [("rag_simple", "adaptive_snap_hyre"), ("rag_simple", "adaptive_snap_hyre_v2"), ("rag_simple", "adaptive_snap_hyre_frontier"), ("rag_simple", "adaptive_snap_hyre_stability"), ("rag_snap_hyde_2call", "adaptive_snap_hyre"), ("rag_snap_hyde_2call", "adaptive_snap_hyre_frontier"), ("snap_hyre_option", "adaptive_snap_hyre"), ("snap_hyre_option", "adaptive_snap_hyre_frontier"), ("adaptive_snap_hyre_frontier", "adaptive_snap_hyre_stability"), ("adaptive_snap_hyre", "adaptive_snap_hyre_anchor"), ("adaptive_snap_hyre_anchor", "adaptive_snap_hyre_diverse")],
    "casehold": [("rag_simple", "adaptive_snap_hyre"), ("rag_simple", "adaptive_snap_hyre_v2"), ("rag_simple", "adaptive_snap_hyre_frontier"), ("rag_simple", "adaptive_snap_hyre_stability"), ("rag_snap_hyde_2call", "adaptive_snap_hyre"), ("rag_snap_hyde_2call", "adaptive_snap_hyre_frontier"), ("snap_hyre_option", "adaptive_snap_hyre"), ("snap_hyre_option", "adaptive_snap_hyre_frontier"), ("adaptive_snap_hyre_frontier", "adaptive_snap_hyre_stability"), ("adaptive_snap_hyre", "adaptive_snap_hyre_anchor"), ("adaptive_snap_hyre_anchor", "adaptive_snap_hyre_diverse")],
    "legalbench_scalr": [("rag_simple", "adaptive_snap_hyre"), ("rag_simple", "adaptive_snap_hyre_v2"), ("rag_simple", "adaptive_snap_hyre_frontier"), ("rag_simple", "adaptive_snap_hyre_stability"), ("rag_snap_hyde_2call", "adaptive_snap_hyre"), ("rag_snap_hyde_2call", "adaptive_snap_hyre_v2"), ("rag_snap_hyde_2call", "adaptive_snap_hyre_frontier"), ("rag_snap_hyde_2call", "adaptive_snap_hyre_stability"), ("snap_hyre_option", "adaptive_snap_hyre"), ("snap_hyre_option", "adaptive_snap_hyre_frontier"), ("adaptive_snap_hyre_frontier", "adaptive_snap_hyre_stability"), ("adaptive_snap_hyre", "adaptive_snap_hyre_anchor"), ("adaptive_snap_hyre_anchor", "adaptive_snap_hyre_diverse")],
    "housing": [("rag_state_filter", "adaptive_snap_hyre"), ("rag_state_filter", "adaptive_snap_hyre_v2"), ("rag_state_filter", "adaptive_snap_hyre_frontier"), ("rag_state_filter", "adaptive_snap_hyre_stability"), ("rag_state_filter", "adaptive_snap_hyre_housing_verifier"), ("snap_hyre_state", "adaptive_snap_hyre"), ("snap_hyre_state", "adaptive_snap_hyre_v2"), ("snap_hyre_state", "adaptive_snap_hyre_frontier"), ("snap_hyre_state", "adaptive_snap_hyre_stability"), ("snap_hyre_state", "adaptive_snap_hyre_housing_verifier"), ("adaptive_snap_hyre_frontier", "adaptive_snap_hyre_stability"), ("adaptive_snap_hyre_frontier", "adaptive_snap_hyre_housing_verifier"), ("adaptive_snap_hyre", "adaptive_snap_hyre_anchor"), ("adaptive_snap_hyre_anchor", "adaptive_snap_hyre_diverse")],
}
EXPECTED_ADAPTIVE_MODES = {
    "barexam": ("snap_hyre_option", "adaptive_snap_hyre", "adaptive_snap_hyre_anchor", "adaptive_snap_hyre_diverse"),
    "casehold": ("snap_hyre_option", "adaptive_snap_hyre", "adaptive_snap_hyre_anchor", "adaptive_snap_hyre_diverse"),
    "legalbench_scalr": ("snap_hyre_option", "adaptive_snap_hyre", "adaptive_snap_hyre_anchor", "adaptive_snap_hyre_diverse"),
    "housing": ("snap_hyre_state", "adaptive_snap_hyre", "adaptive_snap_hyre_anchor", "adaptive_snap_hyre_diverse"),
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


def load_experiment_tags(log_dir: Path) -> dict[str, str]:
    """Map detail-log paths to run tags from experiments.jsonl."""
    tags: dict[str, str] = {}
    experiments_path = log_dir / "experiments.jsonl"
    if not experiments_path.exists():
        return tags
    try:
        lines = experiments_path.read_text().splitlines()
    except OSError:
        return tags
    repo_root = log_dir.parent.resolve()
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        detail = str(row.get("detail_log", "")).strip()
        tag = str(row.get("tag", "")).strip()
        if not detail or not tag:
            continue
        tags[detail] = tag
        tags[str((repo_root / detail).resolve())] = tag
        tags[str((log_dir / Path(detail).name).resolve())] = tag
    return tags


def select_latest(
    paths: list[Path],
    include_all_modes: bool,
    min_n: int,
    providers: set[str] | None = None,
    datasets: set[str] | None = None,
    tag_contains: str | None = None,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    selected: dict[tuple[str, str, str], dict[str, Any]] = {}
    experiment_tags = load_experiment_tags(paths[0].parent) if paths else {}
    for path in paths:
        try:
            rows = load_rows(path)
        except (OSError, json.JSONDecodeError):
            continue
        if not rows:
            continue
        summary = summarize_log(path, rows)
        if not summary["tag"]:
            summary["tag"] = experiment_tags.get(str(path.resolve())) or experiment_tags.get(str(path)) or ""
        if summary["dataset"] not in LEGAL_DATASETS:
            continue
        if datasets is not None and summary["dataset"] not in datasets:
            continue
        if providers is not None and summary["provider"] not in providers:
            continue
        if tag_contains is not None and tag_contains not in str(summary["tag"]):
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


def _providers_for_dataset(
    selected: dict[tuple[str, str, str], dict[str, Any]],
    dataset: str,
    expected_providers: set[str] | None,
) -> list[str]:
    providers = sorted({provider for (d, _m, provider) in selected if d == dataset})
    if expected_providers is not None:
        providers = sorted(set(providers) | expected_providers)
    return providers


def _target_datasets(datasets: set[str] | None = None) -> tuple[str, ...]:
    if datasets is None:
        return LEGAL_DATASETS
    return tuple(dataset for dataset in LEGAL_DATASETS if dataset in datasets)


def build_report(
    selected: dict[tuple[str, str, str], dict[str, Any]],
    expected_providers: set[str] | None = None,
    expected_datasets: set[str] | None = None,
    expected_runs: dict[str, set[str]] | None = None,
) -> str:
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
    for dataset in _target_datasets(expected_datasets):
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
    for dataset in _target_datasets(expected_datasets):
        providers = _providers_for_dataset(selected, dataset, expected_providers)
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
            audit_failures = sorted(
                mode for mode in present
                if audit_status(selected[(dataset, mode, provider)]) != "PASS"
            )
            status = "READY" if not missing and not audit_failures else "MISSING"
            lines.append(
                f"| {dataset} | {provider} | {', '.join(present) or '-'} | "
                f"{', '.join(missing) or '-'} | {status} |"
            )

    append_targeted_run_coverage(lines, selected, expected_providers=expected_providers, expected_runs=expected_runs)

    append_parity_frontier(lines, selected, expected_providers=expected_providers, expected_datasets=expected_datasets)

    lines.extend([
        "",
        "## Paired Comparisons",
        "",
        "| Dataset | Provider | Baseline -> Treatment | N | Baseline Acc | Treatment Acc | Delta pp | b / c | p | 95% CI pp |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for dataset, comparisons in DEFAULT_COMPARISONS.items():
        if expected_datasets is not None and dataset not in expected_datasets:
            continue
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


def targeted_run_records(
    selected: dict[tuple[str, str, str], dict[str, Any]],
    expected_providers: set[str] | None = None,
    expected_runs: dict[str, set[str]] | None = None,
) -> list[dict[str, Any]]:
    if not expected_runs:
        return []
    records: list[dict[str, Any]] = []
    for dataset in _target_datasets(set(expected_runs)):
        providers = _providers_for_dataset(selected, dataset, expected_providers)
        if not providers:
            providers = sorted(expected_providers or {""})
        for provider in providers:
            present = sorted(
                mode for (d, mode, p) in selected
                if d == dataset and p == provider and mode in expected_runs[dataset]
            )
            missing = sorted(expected_runs[dataset] - set(present))
            audit_failures = sorted(
                mode for mode in present
                if mode in ADAPTIVE_MODES and audit_status(selected[(dataset, mode, provider)]) != "PASS"
            )
            records.append({
                "dataset": dataset,
                "provider": provider,
                "present_modes": present,
                "missing_modes": missing,
                "audit_failed_modes": audit_failures,
                "status": "READY" if not missing and not audit_failures else "MISSING",
            })
    return records


def append_targeted_run_coverage(
    lines: list[str],
    selected: dict[tuple[str, str, str], dict[str, Any]],
    expected_providers: set[str] | None = None,
    expected_runs: dict[str, set[str]] | None = None,
) -> None:
    records = targeted_run_records(selected, expected_providers=expected_providers, expected_runs=expected_runs)
    if not records:
        return
    lines.extend([
        "",
        "## Targeted Run Coverage",
        "",
        "| Dataset | Provider | Present modes | Missing modes | Status |",
        "|---|---|---|---|---|",
    ])
    for record in records:
        lines.append(
            f"| {record['dataset']} | {record['provider'] or '-'} | "
            f"{', '.join(record['present_modes']) or '-'} | "
            f"{', '.join(record['missing_modes']) or '-'} | {record['status']} |"
        )


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


def coverage_records(
    selected: dict[tuple[str, str, str], dict[str, Any]],
    expected_providers: set[str] | None = None,
    expected_datasets: set[str] | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for dataset in _target_datasets(expected_datasets):
        providers = _providers_for_dataset(selected, dataset, expected_providers)
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
            audit_failures = sorted(
                mode for mode in present
                if audit_status(selected[(dataset, mode, provider)]) != "PASS"
            )
            records.append({
                "dataset": dataset,
                "provider": provider,
                "present_adaptive_modes": present,
                "missing_adaptive_modes": missing,
                "audit_failed_modes": audit_failures,
                "status": "READY" if not missing and not audit_failures else "MISSING",
            })
    return records


def parity_records(
    selected: dict[tuple[str, str, str], dict[str, Any]],
    expected_providers: set[str] | None = None,
    expected_datasets: set[str] | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for dataset in _target_datasets(expected_datasets):
        providers = _providers_for_dataset(selected, dataset, expected_providers)
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
                and audit_status(summary) == "PASS"
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
    expected_providers: set[str] | None = None,
    expected_datasets: set[str] | None = None,
) -> None:
    lines.extend([
        "",
        "## Adaptive Parity Frontier",
        "",
        "| Dataset | Provider | Best control | Acc | Calls | Best adaptive policy | Acc | Calls | Delta pp | Status |",
        "|---|---|---|---:|---:|---|---:|---:|---:|---|",
    ])
    for record in parity_records(selected, expected_providers=expected_providers, expected_datasets=expected_datasets):
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


def build_json_summary(
    selected: dict[tuple[str, str, str], dict[str, Any]],
    expected_providers: set[str] | None = None,
    expected_datasets: set[str] | None = None,
    expected_runs: dict[str, set[str]] | None = None,
) -> dict[str, Any]:
    return {
        "legal_datasets": list(_target_datasets(expected_datasets)),
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
        "adaptive_coverage": coverage_records(selected, expected_providers=expected_providers, expected_datasets=expected_datasets),
        "targeted_run_coverage": targeted_run_records(selected, expected_providers=expected_providers, expected_runs=expected_runs),
        "adaptive_parity_frontier": parity_records(selected, expected_providers=expected_providers, expected_datasets=expected_datasets),
    }


def readiness_failures(
    selected: dict[tuple[str, str, str], dict[str, Any]],
    expected_providers: set[str] | None = None,
    expected_datasets: set[str] | None = None,
    expected_runs: dict[str, set[str]] | None = None,
) -> list[str]:
    failures: list[str] = []
    if expected_runs:
        for record in targeted_run_records(selected, expected_providers=expected_providers, expected_runs=expected_runs):
            if record["status"] != "READY":
                failures.append(
                    "targeted "
                    f"{record['dataset']} {record['provider'] or '-'} "
                    f"missing={','.join(record['missing_modes'])}"
                )
            if record.get("audit_failed_modes"):
                failures.append(
                    "audit "
                    f"{record['dataset']} {record['provider'] or '-'} "
                    f"failed={','.join(record['audit_failed_modes'])}"
                )
        return failures
    for record in coverage_records(selected, expected_providers=expected_providers, expected_datasets=expected_datasets):
        if record["status"] != "READY":
            failures.append(
                "coverage "
                f"{record['dataset']} {record['provider'] or '-'} "
                f"missing={','.join(record['missing_adaptive_modes'])}"
            )
        if record.get("audit_failed_modes"):
            failures.append(
                "audit "
                f"{record['dataset']} {record['provider'] or '-'} "
                f"failed={','.join(record['audit_failed_modes'])}"
            )
    return failures


def parse_expected_runs(values: list[str] | None) -> dict[str, set[str]] | None:
    if not values:
        return None
    expected: dict[str, set[str]] = defaultdict(set)
    for value in values:
        if ":" not in value:
            raise SystemExit(f"--expected-run must be DATASET:MODE, got {value!r}")
        dataset, mode = value.split(":", 1)
        dataset = dataset.strip()
        mode = mode.strip()
        if dataset not in LEGAL_DATASETS:
            raise SystemExit(f"unknown dataset in --expected-run: {dataset!r}")
        if not mode:
            raise SystemExit(f"empty mode in --expected-run: {value!r}")
        expected[dataset].add(mode)
    return dict(expected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, default=REPO_ROOT / "logs")
    parser.add_argument("--pattern", default="eval_*_detail.jsonl")
    parser.add_argument("--output", type=Path, help="Optional markdown output path")
    parser.add_argument("--json-output", type=Path, help="Optional machine-readable JSON summary path")
    parser.add_argument("--all-modes", action="store_true", help="Include all legal modes instead of only the adaptive sweep surface")
    parser.add_argument("--min-n", type=int, default=20, help="Minimum row count to include; use 1 for smoke logs")
    parser.add_argument(
        "--provider",
        action="append",
        help="Restrict summary/readiness to one provider. May be repeated.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=LEGAL_DATASETS,
        help="Restrict summary/readiness to one legal dataset. May be repeated.",
    )
    parser.add_argument(
        "--tag-contains",
        help="Restrict summary/readiness to detail logs whose run tag contains this substring.",
    )
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Exit nonzero unless all expected adaptive modes are present for every discovered legal dataset/provider.",
    )
    parser.add_argument(
        "--expected-run",
        action="append",
        metavar="DATASET:MODE",
        help="Require a targeted dataset/mode pair instead of the full adaptive matrix. May be repeated.",
    )
    args = parser.parse_args()

    selected = select_latest(
        discover_logs(args.log_dir, args.pattern),
        include_all_modes=args.all_modes,
        min_n=args.min_n,
        providers=set(args.provider) if args.provider else None,
        datasets=set(args.dataset) if args.dataset else None,
        tag_contains=args.tag_contains,
    )
    expected_providers = set(args.provider) if args.provider else None
    expected_datasets = set(args.dataset) if args.dataset else None
    expected_runs = parse_expected_runs(args.expected_run)
    report = build_report(
        selected,
        expected_providers=expected_providers,
        expected_datasets=expected_datasets,
        expected_runs=expected_runs,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n")
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(build_json_summary(
            selected,
            expected_providers=expected_providers,
            expected_datasets=expected_datasets,
            expected_runs=expected_runs,
        ), indent=2) + "\n")
    print(report)
    if args.require_ready:
        failures = readiness_failures(
            selected,
            expected_providers=expected_providers,
            expected_datasets=expected_datasets,
            expected_runs=expected_runs,
        )
        if failures:
            print("\nREADINESS FAILURES", file=sys.stderr)
            for failure in failures:
                print(f"- {failure}", file=sys.stderr)
            raise SystemExit(1)


if __name__ == "__main__":
    main()
