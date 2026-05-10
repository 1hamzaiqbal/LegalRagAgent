#!/usr/bin/env python3
"""Audit adaptive HyRE detail logs before treating them as usable evidence."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean


LEGAL_DATASETS = {"barexam", "housing", "casehold", "legalbench_scalr"}
ADAPTIVE_MODES = {
    "adaptive_snap_hyre",
    "adaptive_snap_hyre_anchor",
    "adaptive_snap_hyre_diverse",
    "adaptive_snap_hyre_v2",
    "adaptive_snap_hyre_frontier",
    "adaptive_snap_hyre_stability",
    "adaptive_snap_hyre_housing_verifier",
    "snap_hyre_option",
    "snap_hyre_state",
}
EXPECTED_LLM_CALLS = {
    "adaptive_snap_hyre": 2.0,
    "adaptive_snap_hyre_anchor": 2.0,
    "adaptive_snap_hyre_diverse": 2.0,
    "adaptive_snap_hyre_v2": 2.0,
    "adaptive_snap_hyre_frontier": 2.0,
    "adaptive_snap_hyre_housing_verifier": 2.0,
    "snap_hyre_option": 2.0,
    "snap_hyre_state": 2.0,
}
EXPECTED_ROUTES = {
    "barexam": {"option_grounding"},
    "casehold": {"option_grounding"},
    "legalbench_scalr": {"option_grounding"},
    "housing": {"state_filter"},
}
EXPECTED_ROUTES_BY_MODE = {
    "adaptive_snap_hyre_v2": {
        "barexam": {"barexam_option_grounding"},
        "casehold": {"casehold_option_diverse"},
        "legalbench_scalr": {"scalr_plain_snap_hyde"},
        "housing": {"state_filter_diverse"},
    },
    "adaptive_snap_hyre_frontier": {
        "barexam": {"frontier_barexam_v2"},
        "casehold": {"frontier_casehold_diverse"},
        "legalbench_scalr": {"frontier_scalr_plain_snap_hyde"},
        "housing": {"frontier_housing_diverse"},
    },
    "adaptive_snap_hyre_stability": {
        "barexam": {"stability_barexam"},
        "casehold": {"stability_casehold"},
        "legalbench_scalr": {"stability_legalbench_scalr"},
        "housing": {"stability_housing"},
    },
    "adaptive_snap_hyre_housing_verifier": {
        "housing": {"housing_yes_no_verifier"},
    },
}
SUSPICIOUS_ANSWER_SNIPPETS = (
    "not mentioned",
    "not provided",
    "insufficient",
    "cannot determine",
    "unknown",
)


def _load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def _pct(num: int, den: int) -> str:
    return "n/a" if den == 0 else f"{100 * num / den:.1f}%"


def _mode_label(path: Path, rows: list[dict]) -> str:
    mode = rows[0].get("mode", path.stem) if rows else path.stem
    provider = rows[0].get("provider", "?") if rows else "?"
    dataset = rows[0].get("dataset", "?") if rows else "?"
    return f"{mode} | {dataset} | {provider}"


def audit(path: Path, legal_only: bool) -> tuple[str, bool]:
    rows = _load_rows(path)
    if not rows:
        return f"{path}\nFAIL empty detail log", False

    n = len(rows)
    dataset = str(rows[0].get("dataset", ""))
    mode = str(rows[0].get("mode", ""))
    routes = Counter(str(r.get("hyre_route", "")) for r in rows)
    errors = [r for r in rows if r.get("error")]
    empty_retrieval = [r for r in rows if not r.get("retrieved_ids")]
    missing_gold_field = [r for r in rows if "gold_retrieved" not in r]
    missing_prediction = [r for r in rows if not r.get("predicted_answer")]
    gold_hits = sum(1 for r in rows if r.get("gold_retrieved"))
    parse_fail = [r for r in rows if r.get("snap_hyre_parse_ok") is False]
    correct = sum(1 for r in rows if r.get("is_correct"))
    call_counts = [float(r.get("llm_calls", 0) or 0) for r in rows]
    suspicious = [
        r for r in rows
        if any(s in str(r.get("predicted_answer", "")).lower() for s in SUSPICIOUS_ANSWER_SNIPPETS)
    ]

    failures: list[str] = []
    warnings: list[str] = []

    if legal_only and dataset not in LEGAL_DATASETS:
        failures.append(f"non_legal_dataset={dataset}")
    if mode in ADAPTIVE_MODES and not routes:
        failures.append("missing_hyre_route")
    expected_routes = EXPECTED_ROUTES_BY_MODE.get(mode, {}).get(dataset) or EXPECTED_ROUTES.get(dataset)
    if mode in {"adaptive_snap_hyre", "adaptive_snap_hyre_anchor", "adaptive_snap_hyre_diverse", "adaptive_snap_hyre_v2", "adaptive_snap_hyre_frontier", "adaptive_snap_hyre_stability"} and expected_routes and not set(routes).issubset(expected_routes):
        failures.append(f"unexpected_routes={dict(routes)} expected_subset={sorted(expected_routes)}")
    if errors:
        failures.append(f"errors={len(errors)}")
    if empty_retrieval:
        failures.append(f"empty_retrieval={len(empty_retrieval)}")
    if missing_gold_field:
        failures.append(f"missing_gold_retrieved_field={len(missing_gold_field)}")
    if missing_prediction:
        failures.append(f"missing_prediction={len(missing_prediction)}")
    if parse_fail:
        failures.append(f"parse_fail={len(parse_fail)}")
    expected_calls = EXPECTED_LLM_CALLS.get(mode)
    if expected_calls is not None:
        def expected_call_count(row: dict) -> float:
            if row.get("hyre_cache_hit"):
                return max(expected_calls - 1.0, 0.0)
            return expected_calls

        bad_call_rows = [
            r for r in rows
            if float(r.get("llm_calls", 0) or 0) != expected_call_count(r)
        ]
        if bad_call_rows:
            counts = Counter(float(r.get("llm_calls", 0) or 0) for r in bad_call_rows)
            failures.append(f"unexpected_llm_calls={dict(counts)} expected={expected_calls:.0f}/cached={max(expected_calls - 1.0, 0.0):.0f}")
    if suspicious:
        warnings.append(f"suspicious_answers={len(suspicious)}")
    if n < 20:
        warnings.append(f"small_n={n}")

    avg_calls = mean(call_counts) if call_counts else 0.0
    lines = [
        str(path),
        f"label={_mode_label(path, rows)}",
        f"rows={n} correct={correct}/{n} accuracy={_pct(correct, n)}",
        f"gold_retrieved={gold_hits}/{n} ({_pct(gold_hits, n)}) empty_retrieval={len(empty_retrieval)}",
        f"routes={dict(routes)} avg_llm_calls={avg_calls:.2f}",
        f"errors={len(errors)} parse_fail={len(parse_fail)} missing_prediction={len(missing_prediction)} missing_gold_field={len(missing_gold_field)}",
    ]
    if warnings:
        lines.append("WARN " + "; ".join(warnings))
    if failures:
        lines.append("FAIL " + "; ".join(failures))
    else:
        lines.append("PASS log_health_checks")
    return "\n".join(lines), not failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Detail JSONL logs to audit")
    parser.add_argument(
        "--allow-nonlegal",
        action="store_true",
        help="Do not fail logs from non-legal datasets.",
    )
    args = parser.parse_args()

    all_ok = True
    for raw in args.paths:
        report, ok = audit(Path(raw), legal_only=not args.allow_nonlegal)
        print(report)
        print()
        all_ok = all_ok and ok
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
