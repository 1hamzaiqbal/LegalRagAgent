#!/usr/bin/env python3
"""Audit retrieval-id cache health and retrieval exposure metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _coerce_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("[") or stripped.startswith("{"):
            try:
                return _coerce_ids(json.loads(stripped))
            except json.JSONDecodeError:
                pass
        return [part.strip() for part in stripped.split(",") if part.strip()]
    if isinstance(value, dict):
        ids: list[str] = []
        for item in value.values():
            ids.extend(_coerce_ids(item))
        return ids
    if isinstance(value, (list, tuple, set)):
        ids: list[str] = []
        for item in value:
            ids.extend(_coerce_ids(item))
        return ids
    return [str(value).strip()] if str(value).strip() else []


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise SystemExit(f"{path}:{line_no}: expected object row")
            rows.append(value)
    if not rows:
        raise SystemExit(f"{path}: no rows loaded")
    return rows


def _mrr(retrieved: list[str], gold: set[str], k: int) -> float:
    for rank, idx in enumerate(retrieved[:k], 1):
        if idx in gold:
            return 1.0 / rank
    return 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _parse_ks(raw: str) -> list[int]:
    ks = sorted({int(part.strip()) for part in raw.split(",") if part.strip()})
    if not ks or any(k <= 0 for k in ks):
        raise argparse.ArgumentTypeError("--ks must contain positive integers")
    return ks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--dataset")
    parser.add_argument("--query-type")
    parser.add_argument("--min-k", type=int, default=10)
    parser.add_argument("--ks", type=_parse_ks, default=_parse_ks("1,3,5,10"))
    args = parser.parse_args()

    rows = _load_jsonl(args.cache)
    if args.dataset:
        rows = [row for row in rows if row.get("dataset") == args.dataset]
    if args.query_type:
        rows = [row for row in rows if row.get("query_type") == args.query_type]
    if not rows:
        raise SystemExit("No rows left after filters")

    duplicate_keys = 0
    seen: set[tuple[str, str, str, str, str]] = set()
    empty = 0
    short = 0
    missing_idx = 0
    no_gold = 0
    hit_by_k: dict[int, list[float]] = {k: [] for k in args.ks}
    recall_by_k: dict[int, list[float]] = {k: [] for k in args.ks}
    mrr_by_k: dict[int, list[float]] = {k: [] for k in args.ks}

    for row in rows:
        key = (
            str(row.get("idx", "")),
            str(row.get("label_prefix", "")),
            str(row.get("collection", "")),
            str(row.get("embedding_model", "")),
            json.dumps(row.get("where") or {}, sort_keys=True, separators=(",", ":")),
        )
        if key in seen:
            duplicate_keys += 1
        seen.add(key)
        idx_value = row.get("idx")
        if idx_value is None or idx_value == "":
            missing_idx += 1
        retrieved = _coerce_ids(row.get("retrieved_ids"))
        gold = set(_coerce_ids(row.get("gold_ids")))
        if not retrieved:
            empty += 1
        if len(retrieved) < args.min_k:
            short += 1
        if not gold:
            no_gold += 1
            continue
        for k in args.ks:
            hits = len(set(retrieved[:k]) & gold)
            hit_by_k[k].append(1.0 if hits else 0.0)
            recall_by_k[k].append(hits / len(gold))
            mrr_by_k[k].append(_mrr(retrieved, gold, k))

    print(f"cache={args.cache}")
    print(f"rows={len(rows)} duplicate_keys={duplicate_keys} missing_idx={missing_idx}")
    print(f"empty_retrieval={empty} rows_shorter_than_min_k={short} rows_without_gold={no_gold}")
    print("")
    print("| k | Hit@k | Recall@k | MRR@k |")
    print("|---:|---:|---:|---:|")
    for k in args.ks:
        print(
            f"| {k} | {_mean(hit_by_k[k]):.4f} | "
            f"{_mean(recall_by_k[k]):.4f} | {_mean(mrr_by_k[k]):.4f} |"
        )

    if duplicate_keys or missing_idx or empty or short:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
