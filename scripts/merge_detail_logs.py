#!/usr/bin/env python3
"""Merge chunked eval detail JSONL files with duplicate-key checks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    if not rows:
        raise SystemExit(f"{path}: no rows")
    return rows


def first_present_key(rows: list[dict[str, Any]], requested: str | None) -> str:
    if requested:
        if requested not in rows[0]:
            raise SystemExit(f"requested key {requested!r} is absent from first row")
        return requested
    for key in ("idx", "label", "question"):
        if key in rows[0]:
            return key
    raise SystemExit("could not infer merge key; pass --key")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--key", help="Unique record key, default: idx then label")
    parser.add_argument(
        "--on-duplicate",
        choices=("error", "first", "last"),
        default="error",
        help=(
            "How to handle duplicate merge keys. Default preserves historical "
            "strict behavior; 'last' is useful for merging a failed prefix with "
            "a repair tail."
        ),
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()

    merged: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    key_name: str | None = None

    for path in args.inputs:
        rows = load_jsonl(path)
        if key_name is None:
            key_name = first_present_key(rows, args.key)
        for row in rows:
            if key_name not in row:
                raise SystemExit(f"{path}: row missing merge key {key_name!r}")
            key = str(row[key_name])
            if key in seen:
                if args.on_duplicate == "error":
                    raise SystemExit(f"duplicate {key_name}={key!r} while merging {path}")
                if args.on_duplicate == "first":
                    continue
                merged[seen[key]] = row
                continue
            seen[key] = len(merged)
            merged.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for row in merged:
            handle.write(json.dumps(row) + "\n")

    correct = sum(1 for row in merged if row.get("is_correct"))
    empty = sum(1 for row in merged if row.get("retrieved_ids") == [])
    gold = sum(1 for row in merged if row.get("gold_retrieved"))
    calls = sum(float(row.get("llm_calls") or 0) for row in merged)
    tokens = sum(int(row.get("input_tokens") or 0) + int(row.get("output_tokens") or 0) for row in merged)

    print(f"output={args.output}")
    print(f"n={len(merged)}")
    print(f"correct={correct}")
    print(f"accuracy={correct / len(merged):.6f}")
    print(f"empty_retrieval={empty}")
    print(f"gold_retrieved={gold}")
    print(f"avg_llm_calls={calls / len(merged):.3f}")
    print(f"avg_tokens={tokens / len(merged):.1f}")


if __name__ == "__main__":
    main()
