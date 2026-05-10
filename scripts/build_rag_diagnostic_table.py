#!/usr/bin/env python3
"""Build bottleneck diagnostics from legal RAG detail logs.

The output is meant to support diagnostic adaptation: it separates answer
accuracy from retrieval exposure and call cost so a controller can decide
whether to spend budget on query formulation, metadata filtering, option
grounding, or answer verification.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    if not rows:
        raise SystemExit(f"{path}: no rows loaded")
    return rows


def as_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return bool(value)


def gold_rank(row: dict[str, Any]) -> int | None:
    retrieved = row.get("retrieved_ids") or []
    if not isinstance(retrieved, list) or not retrieved:
        return None
    candidates = []
    for key in ("gold_idx", "gold_id", "gold_passage_id"):
        value = row.get(key)
        if value is not None:
            candidates.append(str(value))
    gold_passage = row.get("gold_passage")
    if isinstance(gold_passage, dict):
        for key in ("idx", "id", "holding_id"):
            value = gold_passage.get(key)
            if value is not None:
                candidates.append(str(value))
    if not candidates:
        return None
    retrieved_str = [str(value) for value in retrieved]
    for candidate in candidates:
        if candidate in retrieved_str:
            return retrieved_str.index(candidate) + 1
    return None


def pct(num: int | float, den: int | float) -> str:
    return "n/a" if den == 0 else f"{100 * num / den:.1f}%"


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(dataset: str, method: str, path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    correct = sum(bool(row.get("is_correct")) for row in rows)
    gold_flags = [as_bool(row.get("gold_retrieved")) for row in rows if row.get("gold_retrieved") is not None]
    gold_true = sum(flag is True for flag in gold_flags)
    gold_false = sum(flag is False for flag in gold_flags)
    gold_true_correct = sum(bool(row.get("is_correct")) for row in rows if as_bool(row.get("gold_retrieved")) is True)
    gold_false_correct = sum(bool(row.get("is_correct")) for row in rows if as_bool(row.get("gold_retrieved")) is False)
    gold_retrieved_wrong = sum((not bool(row.get("is_correct"))) for row in rows if as_bool(row.get("gold_retrieved")) is True)
    gold_missing_correct = sum(bool(row.get("is_correct")) for row in rows if as_bool(row.get("gold_retrieved")) is False)
    ranks = [rank for row in rows if (rank := gold_rank(row)) is not None]
    calls = []
    for row in rows:
        try:
            calls.append(float(row.get("llm_calls") or 0.0))
        except (TypeError, ValueError):
            pass
    parse_fail = sum(1 for row in rows if row.get("predicted_answer") in {None, ""})
    errors = sum(1 for row in rows if row.get("error"))
    empty_retrieval = sum(1 for row in rows if not row.get("retrieved_ids"))
    return {
        "dataset": dataset,
        "method": method,
        "path": str(path),
        "n": n,
        "correct": correct,
        "accuracy": correct / n if n else 0.0,
        "gold_retrieved": gold_true,
        "gold_retrieved_n": len(gold_flags),
        "gold_missing": gold_false,
        "gold_retrieved_wrong": gold_retrieved_wrong,
        "gold_missing_correct": gold_missing_correct,
        "acc_given_gold_retrieved": gold_true_correct / gold_true if gold_true else None,
        "acc_given_gold_missing": gold_false_correct / gold_false if gold_false else None,
        "rank_observed": len(ranks),
        "recall_at_1": sum(rank <= 1 for rank in ranks) / len(ranks) if ranks else None,
        "recall_at_5": sum(rank <= 5 for rank in ranks) / len(ranks) if ranks else None,
        "recall_at_10": sum(rank <= 10 for rank in ranks) / len(ranks) if ranks else None,
        "mrr": mean([1.0 / rank for rank in ranks]) if ranks else None,
        "avg_calls": mean(calls),
        "parse_fail": parse_fail,
        "errors": errors,
        "empty_retrieval": empty_retrieval,
    }


def fmt_float(value: float | None, percent: bool = False) -> str:
    if value is None:
        return "n/a"
    if percent:
        return f"{100 * value:.1f}%"
    return f"{value:.3f}"


def to_markdown(summaries: list[dict[str, Any]]) -> str:
    lines = [
        "# Legal RAG Diagnostic Table",
        "",
        "This table is generated from detail logs. It separates retrieval exposure, answer conversion, and call budget for bottleneck-aware routing.",
        "",
        "| Dataset | Method | N | Acc | Gold retrieved | Gold retrieved but wrong | Gold missing but correct | Acc if gold retrieved | Acc if gold missing | R@1 | R@5 | R@10 | MRR | Calls | Health |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in summaries:
        health_bits = []
        if item["parse_fail"]:
            health_bits.append(f"parse_fail={item['parse_fail']}")
        if item["errors"]:
            health_bits.append(f"errors={item['errors']}")
        if item["empty_retrieval"]:
            health_bits.append(f"empty={item['empty_retrieval']}")
        health = "; ".join(health_bits) if health_bits else "PASS"
        lines.append(
            f"| {item['dataset']} | `{item['method']}` | {item['n']} | {fmt_float(item['accuracy'], True)} | "
            f"{item['gold_retrieved']}/{item['gold_retrieved_n']} | {item['gold_retrieved_wrong']} | "
            f"{item['gold_missing_correct']} | {fmt_float(item['acc_given_gold_retrieved'], True)} | "
            f"{fmt_float(item['acc_given_gold_missing'], True)} | {fmt_float(item['recall_at_1'], True)} | "
            f"{fmt_float(item['recall_at_5'], True)} | {fmt_float(item['recall_at_10'], True)} | "
            f"{fmt_float(item['mrr'])} | {item['avg_calls']:.2f} | {health} |"
        )
    lines.extend(["", "## Source Logs", ""])
    for item in summaries:
        lines.append(f"- {item['dataset']} / `{item['method']}`: `{item['path']}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", action="append", nargs=3, metavar=("DATASET", "METHOD", "PATH"), required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    summaries = []
    for dataset, method, path_str in args.log:
        path = Path(path_str)
        summaries.append(summarize(dataset, method, path, load_jsonl(path)))
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(to_markdown(summaries), encoding="utf-8")
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"wrote {args.output_md}")
    if args.output_json:
        print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
