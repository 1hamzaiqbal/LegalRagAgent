#!/usr/bin/env python3
"""Analyze CaseHOLD row-level disagreements across selector variants."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
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
    if not row:
        return ""
    value = row.get("predicted_answer")
    return "" if value is None else str(value)


def correct(row: dict[str, Any] | None) -> bool:
    return bool(row and row.get("is_correct"))


def gold(row: dict[str, Any]) -> str:
    return str(row.get("correct_answer", ""))


def score_selected(row: dict[str, Any] | None) -> str:
    if not row:
        return ""
    return str(row.get("selected_candidate") or pred(row))


def vote(candidates: list[str]) -> str:
    nonempty = [c for c in candidates if c]
    if not nonempty:
        return ""
    counts = Counter(nonempty)
    best_count = max(counts.values())
    winners = [c for c in nonempty if counts[c] == best_count]
    return winners[0]


def snap_letter(row: dict[str, Any] | None) -> str:
    if not row:
        return ""
    return str(row.get("snap_letter") or "")


def pct(num: int, den: int) -> str:
    return "n/a" if den == 0 else f"{100 * num / den:.1f}%"


def bucket_stats(items: list[tuple[str, bool]]) -> list[tuple[str, int, int]]:
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for name, ok in items:
        counts[name][0] += 1
        counts[name][1] += int(ok)
    return [(name, total, good) for name, (total, good) in sorted(counts.items())]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", action="append", nargs=2, metavar=("NAME", "PATH"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    logs = {name: load_jsonl(Path(path)) for name, path in args.log}
    labels = sorted(set.intersection(*(set(rows) for rows in logs.values())))
    if not labels:
        raise SystemExit("no overlapping labels")

    names = list(logs)
    rows_by_label = {label: {name: logs[name][label] for name in names} for label in labels}

    method_correct = {name: sum(correct(rows[name]) for rows in rows_by_label.values()) for name in names}
    oracle_any = sum(any(correct(rows[name]) for name in names) for rows in rows_by_label.values())
    all_correct = sum(all(correct(rows[name]) for name in names) for rows in rows_by_label.values())
    none_correct = sum(not any(correct(rows[name]) for name in names) for rows in rows_by_label.values())

    # Simple deterministic ensemble probes over already-produced answers.
    rules: dict[str, int] = defaultdict(int)
    for rows in rows_by_label.values():
        g = gold(next(iter(rows.values())))
        candidate = vote([pred(rows.get("candidate")), pred(rows.get("reranker")), pred(rows.get("replay"))])
        rules["majority_candidate_reranker_replay"] += int(candidate == g)

        candidate = pred(rows.get("reranker")) or pred(rows.get("candidate")) or pred(rows.get("frontier"))
        rules["prefer_reranker"] += int(candidate == g)

        candidate = pred(rows.get("candidate")) or pred(rows.get("reranker")) or pred(rows.get("frontier"))
        rules["prefer_candidate"] += int(candidate == g)

        # Score-only was poor, but agreement with an LLM selector may be useful.
        score = score_selected(rows.get("score"))
        if score and score == pred(rows.get("reranker")):
            candidate = score
        else:
            candidate = pred(rows.get("reranker"))
        rules["score_agree_else_reranker"] += int(candidate == g)

        if score and score == pred(rows.get("candidate")):
            candidate = score
        else:
            candidate = pred(rows.get("candidate"))
        rules["score_agree_else_candidate"] += int(candidate == g)

    pair_patterns: Counter[str] = Counter()
    feature_items: dict[str, list[tuple[str, bool]]] = defaultdict(list)
    examples: list[str] = []
    for label, rows in rows_by_label.items():
        flags = {name: correct(rows[name]) for name in names}
        pattern = ",".join(name for name in names if flags[name]) or "none"
        pair_patterns[pattern] += 1
        g = gold(next(iter(rows.values())))
        method_preds = {name: pred(rows[name]) for name in names}
        llm_preds = [method_preds.get("candidate", ""), method_preds.get("reranker", ""), method_preds.get("replay", "")]
        unique_llm = len(set(p for p in llm_preds if p))
        score_pred = score_selected(rows.get("score"))
        snap = snap_letter(rows.get("reranker")) or snap_letter(rows.get("candidate")) or snap_letter(rows.get("frontier"))

        feature_items["candidate_accuracy_by_reranker_agreement"].append(
            ("agree" if method_preds.get("candidate") == method_preds.get("reranker") else "disagree", flags.get("candidate", False))
        )
        feature_items["reranker_accuracy_by_candidate_agreement"].append(
            ("agree" if method_preds.get("candidate") == method_preds.get("reranker") else "disagree", flags.get("reranker", False))
        )
        feature_items["candidate_accuracy_by_snap_agreement"].append(
            ("agree" if snap and snap == method_preds.get("candidate") else "disagree", flags.get("candidate", False))
        )
        feature_items["reranker_accuracy_by_snap_agreement"].append(
            ("agree" if snap and snap == method_preds.get("reranker") else "disagree", flags.get("reranker", False))
        )
        feature_items["reranker_accuracy_by_score_agreement"].append(
            ("agree" if score_pred and score_pred == method_preds.get("reranker") else "disagree", flags.get("reranker", False))
        )
        feature_items["candidate_accuracy_by_score_agreement"].append(
            ("agree" if score_pred and score_pred == method_preds.get("candidate") else "disagree", flags.get("candidate", False))
        )
        feature_items["oracle_by_llm_entropy"].append((f"{unique_llm}_unique_llm_answers", any(flags.values())))
        feature_items["reranker_accuracy_by_llm_entropy"].append((f"{unique_llm}_unique_llm_answers", flags.get("reranker", False)))
        feature_items["candidate_accuracy_by_llm_entropy"].append((f"{unique_llm}_unique_llm_answers", flags.get("candidate", False)))
        if len(examples) < 12 and len(set(pred(rows[name]) for name in names if pred(rows[name]))) > 1:
            parts = [f"{name}={pred(rows[name])}{'*' if flags[name] else ''}" for name in names]
            examples.append(f"- `{label}` gold={gold(next(iter(rows.values())))} | " + " ".join(parts))

    lines = [
        "# CaseHOLD Selector Disagreement Analysis",
        "",
        f"Rows: {len(labels)}",
        "",
        "## Method Accuracy",
        "",
        "| Method | Correct | Accuracy |",
        "|---|---:|---:|",
    ]
    for name in names:
        lines.append(f"| `{name}` | {method_correct[name]}/{len(labels)} | {pct(method_correct[name], len(labels))} |")

    lines.extend([
        "",
        "## Headroom",
        "",
        f"- any-method oracle: {oracle_any}/{len(labels)} = {pct(oracle_any, len(labels))}",
        f"- all methods correct: {all_correct}/{len(labels)} = {pct(all_correct, len(labels))}",
        f"- no method correct: {none_correct}/{len(labels)} = {pct(none_correct, len(labels))}",
        "",
        "## Simple Ensemble Probes",
        "",
        "| Rule | Correct | Accuracy |",
        "|---|---:|---:|",
    ])
    for name, count in sorted(rules.items()):
        lines.append(f"| `{name}` | {count}/{len(labels)} | {pct(count, len(labels))} |")

    lines.extend([
        "",
        "## Correctness Patterns",
        "",
        "| Correct methods | Rows |",
        "|---|---:|",
    ])
    for pattern, count in pair_patterns.most_common():
        lines.append(f"| `{pattern}` | {count} |")

    lines.extend([
        "",
        "## Feature Buckets",
        "",
    ])
    for feature_name, items in sorted(feature_items.items()):
        lines.extend([
            f"### `{feature_name}`",
            "",
            "| Bucket | Rows | Correct | Accuracy |",
            "|---|---:|---:|---:|",
        ])
        for bucket, total, good in bucket_stats(items):
            lines.append(f"| `{bucket}` | {total} | {good} | {pct(good, total)} |")
        lines.append("")

    lines.extend([
        "",
        "## Disagreement Examples",
        "",
        *examples,
        "",
    ])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
