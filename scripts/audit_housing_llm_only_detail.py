#!/usr/bin/env python3
"""Audit HousingQA LLM-only detail logs before signoff."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


FALSEY_STRINGS = {"", "0", "false", "no", "none", "null", "[]", "{}"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(errors="ignore") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def row_key(row: dict[str, Any]) -> str:
    return str(row.get("label") or row.get("idx") or "")


def truthy_fallback(row: dict[str, Any]) -> bool:
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
            if value.strip().lower() not in FALSEY_STRINGS:
                return True
            continue
        if isinstance(value, (list, tuple, set, dict)):
            if value:
                return True
            continue
        if value:
            return True
    return False


def has_exact_final_line(row: dict[str, Any]) -> bool:
    lines = [line.strip() for line in str(row.get("final_answer") or "").splitlines() if line.strip()]
    predicted = str(row.get("predicted_answer") or row.get("prediction") or "").strip()
    if predicted not in {"Yes", "No"}:
        return False
    return bool(lines) and lines[-1] == f"Answer: {predicted}"


def has_think_tag(row: dict[str, Any]) -> bool:
    values = [str(row.get("final_answer") or "")]
    for item in row.get("call_trace") or []:
        if isinstance(item, dict):
            values.extend(str(item.get(key) or "") for key in ("response", "error"))
    text = "\n".join(values).lower()
    return "<think" in text or "</think" in text


def provider_route_label(row: dict[str, Any]) -> str:
    route = row.get("provider_route")
    if isinstance(route, dict):
        return str(route.get("openrouter_provider_only") or route)
    return str(route or "")


def has_evidence_payload(row: dict[str, Any]) -> bool:
    return any(bool(row.get(key)) for key in ("retrieved_ids", "evidence_store", "retrieved_passages", "retrieved_contexts"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--canonical-cache", type=Path, required=True)
    parser.add_argument("--canonical-start", type=int, default=0)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--provider", default="or-gemma4-26b")
    parser.add_argument("--expected-rows", type=int, default=6853)
    args = parser.parse_args()

    rows_by_key: dict[str, dict[str, Any]] = {}
    ordered_keys: list[str] = []
    for path in args.paths:
        for row in load_jsonl(path):
            key = row_key(row)
            if not key:
                raise SystemExit(f"row without label/idx in {path}")
            if key in rows_by_key:
                raise SystemExit(f"duplicate label while auditing: {key}")
            rows_by_key[key] = row
            ordered_keys.append(key)
    rows = [rows_by_key[key] for key in ordered_keys]

    canonical = [str(row.get("label") or row.get("idx") or "") for row in load_jsonl(args.canonical_cache)]
    failures: dict[str, list[str]] = {
        "wrong_provider": [],
        "wrong_mode": [],
        "wrong_dataset": [],
        "missing_prediction": [],
        "error": [],
        "missing_exact_final": [],
        "fallback": [],
        "think_tag": [],
        "evidence_payload": [],
    }

    for row in rows:
        key = row_key(row)
        if row.get("provider") != args.provider:
            failures["wrong_provider"].append(key)
        if row.get("mode") != "llm_only":
            failures["wrong_mode"].append(key)
        if row.get("dataset") != "housing":
            failures["wrong_dataset"].append(key)
        if str(row.get("predicted_answer") or row.get("prediction") or "") not in {"Yes", "No"}:
            failures["missing_prediction"].append(key)
        if row.get("error"):
            failures["error"].append(key)
        if not has_exact_final_line(row):
            failures["missing_exact_final"].append(key)
        if truthy_fallback(row):
            failures["fallback"].append(key)
        if has_think_tag(row):
            failures["think_tag"].append(key)
        if has_evidence_payload(row):
            failures["evidence_payload"].append(key)

    n = len(rows)
    correct = sum(row.get("is_correct") is True for row in rows)
    route_counts = Counter(provider_route_label(row) for row in rows)
    retry_rows = sum(max(int(row.get("llm_calls") or 0) - 1, 0) for row in rows)
    output_tokens = [int(row.get("output_tokens") or 0) for row in rows]
    max_output = max(output_tokens) if output_tokens else 0
    near_cap = sum(1 for value in output_tokens if value >= 1900)

    print(f"rows={n}")
    print(f"accuracy={correct}/{n} = {correct / n if n else 0:.6f}")
    print(f"provider={args.provider} mode=llm_only expected_rows={args.expected_rows}")
    print("provider_route_counts=" + json.dumps(dict(sorted(route_counts.items())), sort_keys=True))
    print(f"answer_format_retries={retry_rows}")
    print(f"max_output_tokens={max_output}")
    print(f"near_cap_rows={near_cap}")
    canonical_slice = canonical[args.canonical_start:args.canonical_start + n]
    canonical_order_match = ordered_keys == canonical_slice
    print(f"canonical_start={args.canonical_start}")
    print(f"canonical_order_match={canonical_order_match}")
    for name, keys in failures.items():
        print(f"{name}={len(keys)}")

    if n != args.expected_rows:
        raise SystemExit(f"expected {args.expected_rows} rows, found {n}")
    if not args.allow_partial and n != len(canonical):
        raise SystemExit(f"full audit expected {len(canonical)} canonical rows, found {n}")
    if not canonical_order_match:
        raise SystemExit("row labels do not match canonical HousingQA cache order")
    bad = {name: keys for name, keys in failures.items() if keys}
    if bad:
        detail = "; ".join(f"{name}: {','.join(keys[:10])}" for name, keys in bad.items())
        raise SystemExit(detail)


if __name__ == "__main__":
    main()
