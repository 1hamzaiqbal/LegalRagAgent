#!/usr/bin/env python3
"""Compile retrieval-cache audits into one top-k selection table.

This reads the JSONL caches produced by ``scripts/build_retrieval_cache.py`` and
emits per-cache metrics plus macro averages over all supplied caches. It is a
reporting helper only; health failures remain enforced by
``scripts/audit_retrieval_cache.py``.
"""

from __future__ import annotations

import argparse
import csv
import glob
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
    text = str(value).strip()
    return [text] if text else []


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise SystemExit(f"{path}:{line_no}: expected object row")
            rows.append(row)
    if not rows:
        raise SystemExit(f"{path}: no rows loaded")
    return rows


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _mrr(retrieved: list[str], gold: set[str], k: int) -> float:
    for rank, idx in enumerate(retrieved[:k], 1):
        if idx in gold:
            return 1.0 / rank
    return 0.0


def _parse_ks(raw: str) -> list[int]:
    try:
        ks = sorted({int(part.strip()) for part in raw.split(",") if part.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--ks must be comma-separated integers") from exc
    if not ks or any(k <= 0 for k in ks):
        raise argparse.ArgumentTypeError("--ks must contain positive integers")
    return ks


def _expand_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = [Path(p) for p in glob.glob(pattern)]
        if matches:
            paths.extend(matches)
        else:
            paths.append(Path(pattern))
    unique = sorted(dict.fromkeys(path.resolve() for path in paths))
    missing = [path for path in unique if not path.exists()]
    if missing:
        raise SystemExit("Missing cache path(s): " + ", ".join(str(path) for path in missing))
    return unique


def _method_name(row: dict[str, Any], path: Path) -> str:
    query_type = str(row.get("query_type") or "")
    label_prefix = str(row.get("label_prefix") or "")
    if query_type == "raw_question":
        return "rag_simple"
    if query_type == "golden_neighbors":
        return "golden_plus_neighbors"
    if query_type == "hyre_cache":
        return "snap_hyre"
    return label_prefix or query_type or path.stem


def _model_hint(path: Path) -> str:
    stem = path.stem.lower()
    for token in (
        "gemma4-e4b",
        "gemma-4-e4b",
        "gemma4-26b",
        "gemma-4-26b",
        "llama70b",
        "llama-70b",
        "llama3.3-70b",
        "groq-llama70b",
        "or-gemma4-26b",
    ):
        if token in stem:
            return token.replace("gemma-4", "gemma4")
    return "model_invariant"


def _health(rows: list[dict[str, Any]], min_k: int) -> dict[str, int]:
    seen: set[tuple[str, str, str, str, str]] = set()
    duplicate_keys = 0
    missing_idx = 0
    empty_retrieval = 0
    short_rows = 0
    rows_without_gold = 0
    for row in rows:
        key = (
            str(row.get("idx", "")),
            str(row.get("label_prefix", "")),
            str(row.get("collection", "")),
            str(row.get("embedding_model", "")),
            json.dumps(row.get("where") or {}, sort_keys=True, separators=(",", ":")),
        )
        duplicate_keys += int(key in seen)
        seen.add(key)
        if row.get("idx") in (None, ""):
            missing_idx += 1
        retrieved = _coerce_ids(row.get("retrieved_ids"))
        gold = _coerce_ids(row.get("gold_ids"))
        if not retrieved:
            empty_retrieval += 1
        if len(retrieved) < min_k:
            short_rows += 1
        if not gold:
            rows_without_gold += 1
    return {
        "duplicate_keys": duplicate_keys,
        "missing_idx": missing_idx,
        "empty_retrieval": empty_retrieval,
        "short_rows": short_rows,
        "rows_without_gold": rows_without_gold,
    }


def _cache_metrics(path: Path, rows: list[dict[str, Any]], ks: list[int], min_k: int) -> list[dict[str, Any]]:
    first = rows[0]
    dataset = str(first.get("dataset") or "unknown")
    method = _method_name(first, path)
    model = _model_hint(path)
    health = _health(rows, min_k=min_k)
    scored_rows = [row for row in rows if _coerce_ids(row.get("gold_ids"))]
    records: list[dict[str, Any]] = []
    for k in ks:
        hits: list[float] = []
        recalls: list[float] = []
        mrrs: list[float] = []
        for row in scored_rows:
            retrieved = _coerce_ids(row.get("retrieved_ids"))
            metric_retrieved = _coerce_ids(row.get("effective_retrieved_ids")) or retrieved
            gold = set(_coerce_ids(row.get("gold_ids")))
            match_count = len(set(metric_retrieved[:k]) & gold)
            hits.append(1.0 if match_count else 0.0)
            recalls.append(match_count / len(gold) if gold else 0.0)
            mrrs.append(_mrr(metric_retrieved, gold, k))
        records.append({
            "scope": "cache",
            "path": str(path),
            "dataset": dataset,
            "model": model,
            "method": method,
            "k": k,
            "rows": len(rows),
            "scored_rows": len(scored_rows),
            "hit": _mean(hits),
            "recall": _mean(recalls),
            "mrr": _mean(mrrs),
            **health,
        })
    return records


def _macro_records(records: list[dict[str, Any]], ks: list[int]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for k in ks:
        subset = [row for row in records if row["scope"] == "cache" and row["k"] == k]
        if not subset:
            continue
        out.append({
            "scope": "macro",
            "path": "",
            "dataset": "macro",
            "model": "macro",
            "method": "all_supplied_caches",
            "k": k,
            "rows": sum(int(row["rows"]) for row in subset),
            "scored_rows": sum(int(row["scored_rows"]) for row in subset),
            "hit": _mean([float(row["hit"]) for row in subset]),
            "recall": _mean([float(row["recall"]) for row in subset]),
            "mrr": _mean([float(row["mrr"]) for row in subset]),
            "duplicate_keys": sum(int(row["duplicate_keys"]) for row in subset),
            "missing_idx": sum(int(row["missing_idx"]) for row in subset),
            "empty_retrieval": sum(int(row["empty_retrieval"]) for row in subset),
            "short_rows": sum(int(row["short_rows"]) for row in subset),
            "rows_without_gold": sum(int(row["rows_without_gold"]) for row in subset),
        })
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scope", "dataset", "model", "method", "k", "rows", "scored_rows",
        "hit", "recall", "mrr", "duplicate_keys", "missing_idx",
        "empty_retrieval", "short_rows", "rows_without_gold", "path",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("# Retrieval Cache Matrix\n\n")
        f.write("| scope | dataset | model | method | k | rows | Hit@k | Recall@k | MRR@k | health |\n")
        f.write("|---|---|---|---|---:|---:|---:|---:|---:|---|\n")
        for row in rows:
            health = (
                f"dup={row['duplicate_keys']}, missing_idx={row['missing_idx']}, "
                f"empty={row['empty_retrieval']}, short={row['short_rows']}, "
                f"no_gold={row['rows_without_gold']}"
            )
            f.write(
                f"| {row['scope']} | {row['dataset']} | {row['model']} | "
                f"{row['method']} | {row['k']} | {row['rows']} | "
                f"{float(row['hit']):.4f} | {float(row['recall']):.4f} | "
                f"{float(row['mrr']):.4f} | {health} |\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", action="append", required=True, help="Cache path or glob. May be repeated.")
    parser.add_argument("--ks", type=_parse_ks, default=_parse_ks("1,3,5,10"))
    parser.add_argument("--min-k", type=int, default=10)
    parser.add_argument("--out-csv", type=Path)
    parser.add_argument("--out-md", type=Path)
    args = parser.parse_args()

    records: list[dict[str, Any]] = []
    for path in _expand_paths(args.cache):
        rows = _load_jsonl(path)
        records.extend(_cache_metrics(path, rows, args.ks, min_k=args.min_k))
    records.extend(_macro_records(records, args.ks))

    if args.out_csv:
        _write_csv(args.out_csv, records)
    if args.out_md:
        _write_md(args.out_md, records)
    if not args.out_csv and not args.out_md:
        _write_md(Path("/dev/stdout"), records)


if __name__ == "__main__":
    main()
