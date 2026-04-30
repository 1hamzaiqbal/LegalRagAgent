#!/usr/bin/env python3
"""Score retrieval IDs in eval detail logs against qrels or logged gold IDs.

This is intentionally answer-free: it scores whether the retrieval layer found
the target document/span IDs, independent of whether the LLM answered correctly.
It is the right shape for MLEB-SCALR / LegalBench-RAG style calibration and also
works for current detail logs that already contain ``gold_idx``.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if isinstance(value, dict):
                rows.append(value)
    if not rows:
        raise SystemExit(f"{path}: no rows loaded")
    return rows


def coerce_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() == "nan":
            return []
        if stripped.startswith("[") or stripped.startswith("{"):
            try:
                return coerce_ids(json.loads(stripped))
            except json.JSONDecodeError:
                pass
        return [part.strip() for part in stripped.split(",") if part.strip()]
    if isinstance(value, dict):
        ids: list[str] = []
        for item in value.values():
            ids.extend(coerce_ids(item))
        return ids
    if isinstance(value, (list, tuple, set)):
        ids = []
        for item in value:
            ids.extend(coerce_ids(item))
        return ids
    return [str(value).strip()] if str(value).strip() else []


def load_qrels(path: Path, query_col: str, doc_col: str, rel_col: str | None) -> dict[str, dict[str, float]]:
    if path.suffix.lower() == ".jsonl":
        rows = load_jsonl(path)
    else:
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f, delimiter="\t" if path.suffix.lower() == ".tsv" else ","))

    qrels: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        qid = str(row.get(query_col, "")).strip()
        doc_id = str(row.get(doc_col, "")).strip()
        if not qid or not doc_id:
            continue
        rel = 1.0
        if rel_col:
            try:
                rel = float(row.get(rel_col, 1) or 0)
            except ValueError:
                rel = 0.0
        if rel > 0:
            qrels[qid][doc_id] = rel
    if not qrels:
        raise SystemExit(f"{path}: no positive qrels loaded")
    return dict(qrels)


def qrels_from_detail(rows: list[dict[str, Any]], query_field: str, gold_field: str) -> dict[str, dict[str, float]]:
    qrels: dict[str, dict[str, float]] = {}
    for row in rows:
        qid = str(row.get(query_field, "")).strip()
        if not qid:
            continue
        ids = coerce_ids(row.get(gold_field, ""))
        if ids:
            qrels[qid] = {doc_id: 1.0 for doc_id in ids}
    if not qrels:
        raise SystemExit(f"No gold IDs found in detail rows using field {gold_field!r}")
    return qrels


def dcg(relevances: list[float]) -> float:
    return sum(rel / math.log2(rank + 2) for rank, rel in enumerate(relevances))


def score_query(retrieved: list[str], relevant: dict[str, float], k: int) -> dict[str, float]:
    top = retrieved[:k]
    rel_values = [relevant.get(doc_id, 0.0) for doc_id in top]
    hits = sum(1 for rel in rel_values if rel > 0)
    recall = hits / len(relevant) if relevant else 0.0
    precision = hits / k if k else 0.0
    hit = 1.0 if hits else 0.0
    mrr = 0.0
    for rank, rel in enumerate(rel_values, start=1):
        if rel > 0:
            mrr = 1.0 / rank
            break
    ideal = sorted(relevant.values(), reverse=True)[:k]
    ideal_dcg = dcg(ideal)
    ndcg = dcg(rel_values) / ideal_dcg if ideal_dcg > 0 else 0.0
    return {
        "recall": recall,
        "precision": precision,
        "hit": hit,
        "mrr": mrr,
        "ndcg": ndcg,
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def score_rows(
    rows: list[dict[str, Any]],
    qrels: dict[str, dict[str, float]],
    query_field: str,
    retrieved_field: str,
    ks: list[int],
) -> dict[str, Any]:
    per_k = {k: {"recall": [], "precision": [], "hit": [], "mrr": [], "ndcg": []} for k in ks}
    evaluated = 0
    missing_qrels = 0
    empty_retrieval = 0
    for row in rows:
        qid = str(row.get(query_field, "")).strip()
        if qid not in qrels:
            missing_qrels += 1
            continue
        retrieved = coerce_ids(row.get(retrieved_field, []))
        if not retrieved:
            empty_retrieval += 1
        evaluated += 1
        for k in ks:
            metrics = score_query(retrieved, qrels[qid], k)
            for name, value in metrics.items():
                per_k[k][name].append(value)

    return {
        "rows": len(rows),
        "evaluated": evaluated,
        "missing_qrels": missing_qrels,
        "empty_retrieval": empty_retrieval,
        "metrics": {
            str(k): {name: mean(values) for name, values in values_by_name.items()}
            for k, values_by_name in per_k.items()
        },
    }


def markdown(title: str, summary: dict[str, Any], detail_log: Path) -> str:
    lines = [
        f"# {title}",
        "",
        f"Detail log: `{detail_log}`",
        "",
        f"Rows: {summary['rows']}",
        f"Evaluated rows: {summary['evaluated']}",
        f"Rows missing qrels: {summary['missing_qrels']}",
        f"Rows with empty retrieval: {summary['empty_retrieval']}",
        "",
        "| k | Recall@k | Precision@k | Hit@k | MRR@k | nDCG@k |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for k, values in sorted(summary["metrics"].items(), key=lambda item: int(item[0])):
        lines.append(
            "| {k} | {recall:.4f} | {precision:.4f} | {hit:.4f} | {mrr:.4f} | {ndcg:.4f} |".format(
                k=k,
                recall=values["recall"],
                precision=values["precision"],
                hit=values["hit"],
                mrr=values["mrr"],
                ndcg=values["ndcg"],
            )
        )
    return "\n".join(lines) + "\n"


def parse_ks(raw: str) -> list[int]:
    ks = sorted({int(part.strip()) for part in raw.split(",") if part.strip()})
    if not ks or any(k <= 0 for k in ks):
        raise argparse.ArgumentTypeError("--ks must contain positive integers")
    return ks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detail-log", required=True, type=Path)
    parser.add_argument("--qrels", type=Path, help="CSV/TSV/JSONL qrels file. If omitted, use --gold-field from detail rows.")
    parser.add_argument("--query-field", default="idx")
    parser.add_argument("--retrieved-field", default="retrieved_ids")
    parser.add_argument("--gold-field", default="gold_idx")
    parser.add_argument("--qrels-query-col", default="query_id")
    parser.add_argument("--qrels-doc-col", default="doc_id")
    parser.add_argument("--qrels-rel-col", default="score")
    parser.add_argument("--ks", type=parse_ks, default=parse_ks("1,5,10"))
    parser.add_argument("--out", type=Path, help="Optional Markdown output path")
    parser.add_argument("--title", default="Retrieval Qrels Score")
    args = parser.parse_args()

    rows = load_jsonl(args.detail_log)
    if args.qrels:
        qrels = load_qrels(args.qrels, args.qrels_query_col, args.qrels_doc_col, args.qrels_rel_col)
    else:
        qrels = qrels_from_detail(rows, args.query_field, args.gold_field)
    summary = score_rows(rows, qrels, args.query_field, args.retrieved_field, args.ks)
    output = markdown(args.title, summary, args.detail_log)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output)
        print(f"Wrote {args.out}")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
