#!/usr/bin/env python3
"""Evaluate cheap calibrated CaseHOLD selectors over existing detail logs.

This script is intentionally offline: it consumes already-generated CaseHOLD
detail JSONLs for multiple selector variants and evaluates deterministic
selection policies over their row-level predictions and score traces. It is for
testing whether observable calibration signals can improve answer-option
conversion without spending fresh retrieval or LLM calls.
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
        raise SystemExit(f"{path}: no records loaded")
    return rows


def pred(row: dict[str, Any] | None) -> str:
    value = (row or {}).get("predicted_answer")
    return "" if value is None else str(value).upper()


def gold(row: dict[str, Any]) -> str:
    return str(row.get("correct_answer", "")).upper()


def snap(row: dict[str, Any] | None) -> str:
    return str((row or {}).get("snap_letter") or "").upper()


def score_top_and_margin(row: dict[str, Any] | None) -> tuple[str, float]:
    scores = (row or {}).get("candidate_scores") or {}
    if not isinstance(scores, dict) or not scores:
        return "", float("-inf")
    ranked = sorted(((float(score), str(letter).upper()) for letter, score in scores.items()), reverse=True)
    if not ranked:
        return "", float("-inf")
    margin = ranked[0][0] - ranked[1][0] if len(ranked) > 1 else float("inf")
    return ranked[0][1], margin


def majority(values: list[str]) -> str:
    nonempty = [value for value in values if value]
    if not nonempty:
        return ""
    counts = Counter(nonempty)
    best = max(counts.values())
    for value in nonempty:
        if counts[value] == best:
            return value
    return nonempty[0]


def pct(num: int, den: int) -> str:
    return "n/a" if den == 0 else f"{100 * num / den:.1f}%"


def evaluate_policy(
    labels: list[str],
    logs: dict[str, dict[str, dict[str, Any]]],
    name: str,
    select_fn,
) -> dict[str, Any]:
    correct = 0
    used_counter: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for label in labels:
        row_bundle = {method: rows_by_label[label] for method, rows_by_label in logs.items()}
        base = next(iter(row_bundle.values()))
        selected, route = select_fn(row_bundle)
        ok = selected == gold(base)
        correct += int(ok)
        used_counter[route] += 1
        rows.append({"label": label, "gold": gold(base), "prediction": selected, "route": route, "is_correct": ok})
    return {
        "policy": name,
        "correct": correct,
        "total": len(labels),
        "accuracy": correct / len(labels) if labels else 0.0,
        "accuracy_pct": pct(correct, len(labels)),
        "route_counts": dict(sorted(used_counter.items())),
        "rows": rows,
    }


def render_markdown(results: list[dict[str, Any]], source_logs: dict[str, str]) -> str:
    lines = [
        "# CaseHOLD Offline Calibrated Selector Evaluation",
        "",
        "## Source Logs",
        "",
    ]
    for name, path in source_logs.items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend([
        "",
        "## Policy Results",
        "",
        "| Policy | Accuracy | Route counts |",
        "|---|---:|---|",
    ])
    for result in sorted(results, key=lambda item: item["accuracy"], reverse=True):
        route_counts = ", ".join(f"{k}={v}" for k, v in result["route_counts"].items())
        lines.append(
            f"| `{result['policy']}` | {result['correct']}/{result['total']} = "
            f"{result['accuracy_pct']} | {route_counts} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- These policies spend no new retrieval or LLM calls; they only route among existing CaseHOLD selector outputs.",
        "- Any improvement here is a calibration signal, not a new end-to-end method claim until validated on held-out rows.",
        "- A high score-margin override is useful only if it beats the best individual selector and survives held-out validation.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", action="append", nargs=2, metavar=("NAME", "PATH"), required=True)
    parser.add_argument(
        "--source-uri",
        action="append",
        nargs=2,
        metavar=("NAME", "URI"),
        default=[],
        help="Optional provenance URI/path to report for a log alias instead of the local input path.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--score-margin", type=float, default=2.0)
    args = parser.parse_args()

    logs = {name: load_jsonl(Path(path)) for name, path in args.log}
    labels = sorted(set.intersection(*(set(rows) for rows in logs.values())))
    if not labels:
        raise SystemExit("no overlapping labels across logs")

    required = {"candidate", "reranker", "score"}
    missing = sorted(required - set(logs))
    if missing:
        raise SystemExit("missing required log aliases: " + ", ".join(missing))

    policies: list[tuple[str, Any]] = []
    for method in logs:
        policies.append((f"always_{method}", lambda rows, method=method: (pred(rows[method]), method)))

    policies.extend(
        [
            (
                "candidate_reranker_agree_else_candidate",
                lambda rows: (
                    (pred(rows["candidate"]), "candidate_reranker_agree")
                    if pred(rows["candidate"]) == pred(rows["reranker"])
                    else (pred(rows["candidate"]), "candidate")
                ),
            ),
            (
                "candidate_reranker_snap_agree_else_candidate",
                lambda rows: (
                    (pred(rows["candidate"]), "candidate_reranker_snap_agree")
                    if pred(rows["candidate"]) == pred(rows["reranker"]) == snap(rows["candidate"])
                    else (pred(rows["candidate"]), "candidate")
                ),
            ),
            (
                "majority_candidate_reranker_replay_else_candidate",
                lambda rows: (
                    (
                        majority([pred(rows.get("candidate")), pred(rows.get("reranker")), pred(rows.get("replay"))]),
                        "majority_candidate_reranker_replay",
                    )
                    if rows.get("replay")
                    else (pred(rows["candidate"]), "candidate")
                ),
            ),
            (
                f"score_margin_{args.score_margin:g}_else_candidate",
                lambda rows: (
                    (score_top_and_margin(rows["score"])[0], "score_high_margin")
                    if score_top_and_margin(rows["score"])[1] >= args.score_margin
                    else (pred(rows["candidate"]), "candidate")
                ),
            ),
            (
                f"score_margin_{args.score_margin:g}_else_reranker",
                lambda rows: (
                    (score_top_and_margin(rows["score"])[0], "score_high_margin")
                    if score_top_and_margin(rows["score"])[1] >= args.score_margin
                    else (pred(rows["reranker"]), "reranker")
                ),
            ),
        ]
    )

    results = [evaluate_policy(labels, logs, name, select_fn) for name, select_fn in policies]
    source_overrides = dict(args.source_uri or [])
    payload = {
        "n": len(labels),
        "score_margin": args.score_margin,
        "source_logs": {name: source_overrides.get(name, str(path)) for name, path in args.log},
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    args.output_md.write_text(render_markdown(results, payload["source_logs"]))


if __name__ == "__main__":
    main()
