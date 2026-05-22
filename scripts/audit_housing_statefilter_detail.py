#!/usr/bin/env python3
"""Audit HousingQA state-filter answer detail logs before signoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _row_key(row: dict[str, Any]) -> str:
    return str(row.get("label") or row.get("idx") or "")


def _truthy_fallback(row: dict[str, Any]) -> bool:
    falsey_strings = {"", "0", "false", "no", "none", "null", "[]", "{}"}
    for key, value in row.items():
        if "fallback" not in str(key).lower():
            continue
        if isinstance(value, bool):
            if value:
                return True
            continue
        if value is None:
            continue
        if isinstance(value, str):
            if value.strip().lower() not in falsey_strings:
                return True
            continue
        if isinstance(value, (list, tuple, set, dict)):
            if value:
                return True
            continue
        if value:
            return True
    return False


def _has_exact_final_line(row: dict[str, Any]) -> bool:
    lines = [
        line.strip()
        for line in str(row.get("final_answer", "")).splitlines()
        if line.strip()
    ]
    predicted = str(row.get("predicted_answer") or row.get("prediction") or "").strip().lower()
    if predicted == "yes":
        target = "Answer: Yes"
    elif predicted == "no":
        target = "Answer: No"
    else:
        return False
    return bool(lines) and lines[-1] == target


def _has_think_tag(row: dict[str, Any]) -> bool:
    text = "\n".join(str(row.get(key, "")) for key in ("final_answer", "hyde_passage", "snap_answer"))
    lowered = text.lower()
    return "<think" in lowered or "</think" in lowered


def _load_rows(paths: list[Path], merge_key: str) -> list[dict[str, Any]]:
    rows_by_key: dict[str, dict[str, Any]] = {}
    ordered_keys: list[str] = []
    for path in paths:
        with path.open(errors="ignore") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = str(row.get(merge_key) or _row_key(row))
                if not key:
                    raise SystemExit(f"row without merge key in {path}")
                if key not in rows_by_key:
                    ordered_keys.append(key)
                rows_by_key[key] = row
    return [rows_by_key[key] for key in ordered_keys]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--expected-rows", type=int, default=6853)
    parser.add_argument("--retrieval-k", type=int, default=5)
    parser.add_argument("--merge-key", default="label")
    parser.add_argument("--require-hyre-cache", action="store_true")
    args = parser.parse_args()

    rows = _load_rows(args.paths, args.merge_key)
    n = len(rows)
    correct = sum(row.get("is_correct") is True for row in rows)
    gold = sum(row.get("gold_retrieved") is True for row in rows)

    failures: dict[str, list[str]] = {
        "wrong_provider": [],
        "wrong_mode": [],
        "wrong_dataset": [],
        "missing_prediction": [],
        "error": [],
        "missing_state_filter": [],
        "retrieval_cache_miss": [],
        "doc_cache_miss": [],
        "hyre_cache_miss": [],
        "bad_evidence_len": [],
        "missing_exact_final": [],
        "fallback": [],
        "think_tag": [],
    }

    for row in rows:
        key = _row_key(row)
        if row.get("provider") != args.provider:
            failures["wrong_provider"].append(key)
        if row.get("mode") != args.mode:
            failures["wrong_mode"].append(key)
        if row.get("dataset") != "housing":
            failures["wrong_dataset"].append(key)
        if not (row.get("predicted_answer") or row.get("prediction")):
            failures["missing_prediction"].append(key)
        if row.get("error"):
            failures["error"].append(key)
        if row.get("housing_state_filter") is not True or not row.get("retrieval_where"):
            failures["missing_state_filter"].append(key)
        if row.get("retrieval_cache_hit") is not True:
            failures["retrieval_cache_miss"].append(key)
        if row.get("retrieval_doc_cache_hit") is not True:
            failures["doc_cache_miss"].append(key)
        if args.require_hyre_cache and row.get("hyre_cache_hit") is not True:
            failures["hyre_cache_miss"].append(key)
        if len(row.get("evidence_store") or []) != args.retrieval_k:
            failures["bad_evidence_len"].append(key)
        if not _has_exact_final_line(row):
            failures["missing_exact_final"].append(key)
        if _truthy_fallback(row):
            failures["fallback"].append(key)
        if _has_think_tag(row):
            failures["think_tag"].append(key)

    print(f"rows={n}")
    print(f"accuracy={correct}/{n} = {correct / n if n else 0:.6f}")
    print(f"gold_hit@{args.retrieval_k}={gold}/{n} = {gold / n if n else 0:.6f}")
    print(f"provider={args.provider} mode={args.mode} expected_rows={args.expected_rows}")
    for name, keys in failures.items():
        print(f"{name}={len(keys)}")

    bad = {name: keys for name, keys in failures.items() if keys}
    if n != args.expected_rows:
        raise SystemExit(f"expected {args.expected_rows} rows, found {n}")
    if bad:
        detail = "; ".join(f"{name}: {','.join(keys[:10])}" for name, keys in bad.items())
        raise SystemExit(detail)


if __name__ == "__main__":
    main()
