#!/usr/bin/env python3
"""Run retrieval-only evaluation and write detail JSONL rows.

This is for BEIR/MTEB-style legal retrieval datasets such as MLEB-SCALR where
there is no answer-generation step. Pair the output with
``scripts/score_retrieval_qrels.py``.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_qrels(path: Path | None) -> dict[str, list[str]]:
    if path is None:
        return {}
    result: dict[str, list[str]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t" if path.suffix.lower() == ".tsv" else ",")
        for row in reader:
            qid = str(row.get("query_id", "")).strip()
            doc_id = str(row.get("doc_id", "")).strip()
            score_raw = row.get("score", "1")
            try:
                score = float(score_raw or 0)
            except ValueError:
                score = 0.0
            if qid and doc_id and score > 0:
                result.setdefault(qid, []).append(doc_id)
    return result


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    from rag_utils import get_vectorstore, retrieve_documents_multi_query

    rows = read_csv(args.queries)
    qrels = load_qrels(args.qrels)
    max_k = max(args.ks)
    embedding_model = os.getenv("EVAL_EMBEDDING_MODEL", "").strip() or None
    vectorstore = get_vectorstore(args.collection, embedding_model=embedding_model)

    results: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        qid = str(row.get(args.query_id_col, "")).strip()
        query = str(row.get(args.query_col, "")).strip()
        if not qid or not query:
            continue
        start = time.time()
        docs = retrieve_documents_multi_query(
            queries=[query],
            k=max_k,
            vectorstore=vectorstore,
            use_bm25=args.use_bm25,
            rerank_query=query,
        )
        evidence_store = [
            {
                "idx": str(doc.metadata.get("idx", "")),
                "text": doc.page_content,
                "source": doc.metadata.get("source", "retrieval_only"),
                "cross_encoder_score": float(doc.metadata.get("cross_encoder_score", 0.0)),
            }
            for doc in docs
        ]
        retrieved_ids = [item["idx"] for item in evidence_store]
        gold_ids = qrels.get(qid, [])
        elapsed = time.time() - start
        record = {
            "label": f"{args.dataset}_{qid}",
            "idx": qid,
            "question": query,
            "mode": "retrieval_only",
            "provider": "none",
            "dataset": args.dataset,
            "embedding_model": embedding_model or "",
            "retrieval_k": max_k,
            "retrieved_ids": retrieved_ids,
            "evidence_store": evidence_store,
            "gold_idx": ",".join(gold_ids),
            "gold_retrieved": bool(set(gold_ids) & set(retrieved_ids)) if gold_ids else False,
            "elapsed_sec": round(elapsed, 3),
            "llm_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "is_correct": False,
            "error": None,
        }
        results.append(record)
        if idx % args.print_every == 0 or idx == len(rows):
            hits = sum(1 for item in results if item.get("gold_retrieved"))
            print(f"[{idx}/{len(rows)}] hit@{max_k}={hits}/{len(results)} elapsed={elapsed:.2f}s", flush=True)
    return results


def parse_ks(raw: str) -> list[int]:
    values = sorted({int(part.strip()) for part in raw.split(",") if part.strip()})
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("--ks must contain positive integers")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=REPO_ROOT / "datasets/mleb_scalr/queries.csv")
    parser.add_argument("--qrels", type=Path, default=REPO_ROOT / "datasets/mleb_scalr/qrels.csv")
    parser.add_argument("--collection", default="mleb_scalr_holdings")
    parser.add_argument("--dataset", default="mleb_scalr")
    parser.add_argument("--query-id-col", default="idx")
    parser.add_argument("--query-col", default="query")
    parser.add_argument("--ks", type=parse_ks, default=parse_ks("1,5,10"))
    parser.add_argument("--use-bm25", action="store_true")
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
