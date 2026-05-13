#!/usr/bin/env python3
"""Build deterministic retrieval-id caches for top-k replay.

The output can be passed to `eval/eval_harness.py --retrieval-cache-path`.
Each row stores the ordered corpus ids returned by retrieval at a large max-k;
answer-generation runs can then slice top-k without re-running embedding search.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from eval_config import EvalConfig, load_questions  # noqa: E402
from eval_harness import (  # noqa: E402
    _collection_for_config,
    _fmt_intermediate,
    _gold_ids,
    _is_gold_retrieved,
    _retrieval_question,
    _row_label,
    _where_from_config,
)
from rag_utils import get_vectorstore, retrieve_documents_multi_query  # noqa: E402


QUERY_TYPE_TO_LABEL_PREFIX = {
    "raw_question": "simple",
    "hyre_cache": "snap_hyde_2call",
    "golden_neighbors": "golden_plus_neighbors",
}


def _load_hyre_cache(path: Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            label = row.get("label")
            if not label:
                raise SystemExit(f"{path}:{line_no}: missing label")
            cache[str(label)] = row
    if not cache:
        raise SystemExit(f"{path}: no HyRE cache rows loaded")
    return cache


def _hash_texts(values: list[str]) -> str:
    h = hashlib.sha256()
    for value in values:
        h.update(value.encode("utf-8", errors="ignore"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def _score_from_doc(doc) -> float:
    try:
        return float(doc.metadata.get("cross_encoder_score", 0.0) or 0.0)
    except Exception:
        return 0.0


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value) != ""))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=[
        "barexam", "housing", "legal_rag", "australian", "casehold",
        "musique", "legalbench_scalr",
    ])
    parser.add_argument("--questions", default="full", help="'full' or integer N")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--sample-end", type=int)
    parser.add_argument("--query-type", required=True, choices=sorted(QUERY_TYPE_TO_LABEL_PREFIX))
    parser.add_argument("--label-prefix", help="Override cache label_prefix")
    parser.add_argument("--hyre-cache-path", type=Path, help="Required for --query-type hyre_cache")
    parser.add_argument("--max-k", type=int, default=10)
    parser.add_argument("--source-filter", default="")
    parser.add_argument("--collection", help="Override dataset collection")
    parser.add_argument("--embedding-model", default="", help="Override EVAL_EMBEDDING_MODEL")
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_k <= 0:
        raise SystemExit("--max-k must be positive")
    if args.embedding_model:
        os.environ["EVAL_EMBEDDING_MODEL"] = args.embedding_model
    if args.query_type == "hyre_cache" and not args.hyre_cache_path:
        raise SystemExit("--hyre-cache-path is required for --query-type hyre_cache")

    config = EvalConfig(
        mode="rag_simple",
        dataset=args.dataset,
        questions=args.questions,
        seed=args.seed,
        source_filter=args.source_filter,
        sample_start=args.sample_start,
        sample_end=args.sample_end,
        retrieval_k=args.max_k,
    )
    questions = load_questions(config)
    if args.sample_start or args.sample_end is not None:
        start = max(0, int(args.sample_start or 0))
        end = None if args.sample_end is None else max(start, int(args.sample_end))
        questions = questions.iloc[start:end].reset_index(drop=True)

    collection = args.collection or _collection_for_config(config)
    where = _where_from_config(config)
    embedding_model = os.getenv("EVAL_EMBEDDING_MODEL", "").strip() or ""
    label_prefix = args.label_prefix or QUERY_TYPE_TO_LABEL_PREFIX[args.query_type]
    hyre_cache = _load_hyre_cache(args.hyre_cache_path) if args.hyre_cache_path else {}
    vectorstore = get_vectorstore(collection, embedding_model=embedding_model or None)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    wrote = 0
    skipped = 0
    with args.out.open("w") as f:
        for fallback_i, row in questions.iterrows():
            label = _row_label(row, config, fallback_i=fallback_i)
            row_idx = str(row.get("idx", fallback_i))
            if args.query_type == "raw_question":
                queries = [_retrieval_question(row)]
            elif args.query_type == "golden_neighbors":
                gold = str(row.get("gold_passage", ""))
                if not gold or gold == "nan":
                    skipped += 1
                    continue
                queries = [gold]
            else:
                cache_entry = hyre_cache.get(label)
                if not cache_entry or not cache_entry.get("hyde_passage"):
                    skipped += 1
                    continue
                queries = [str(cache_entry["hyde_passage"])]

            docs = retrieve_documents_multi_query(
                queries=queries,
                k=args.max_k,
                vectorstore=vectorstore,
                where=where,
                rerank_query=None,
            )
            doc_ids = [str(doc.metadata.get("idx", "")) for doc in docs]
            scores = [_score_from_doc(doc) for doc in docs]
            if args.query_type == "golden_neighbors":
                gold_ids = _gold_ids(row)
                neighbor_ids = [idx for idx in doc_ids if idx not in set(gold_ids)]
                retrieved_ids = _dedupe(gold_ids + neighbor_ids)[:args.max_k]
                score_by_id = {idx: score for idx, score in zip(doc_ids, scores)}
                scores = [0.0 if idx in set(gold_ids) else score_by_id.get(idx, 0.0) for idx in retrieved_ids]
            else:
                retrieved_ids = _dedupe(doc_ids)[:args.max_k]
                scores = scores[:len(retrieved_ids)]

            record = {
                "label": label,
                "idx": row_idx,
                "dataset": args.dataset,
                "query_type": args.query_type,
                "label_prefix": label_prefix,
                "collection": collection,
                "embedding_model": embedding_model,
                "where": where or {},
                "max_k": args.max_k,
                "retrieved_ids": retrieved_ids,
                "scores": scores,
                "gold_ids": _gold_ids(row),
                "gold_retrieved": _is_gold_retrieved(row, retrieved_ids),
                "query_hash": _hash_texts(queries),
                "question_hash": _hash_texts([_fmt_intermediate(row, config)]),
            }
            f.write(json.dumps(record, sort_keys=True) + "\n")
            wrote += 1

    print(f"wrote {wrote} retrieval-cache rows to {args.out}")
    if skipped:
        print(f"skipped {skipped} rows without required query material")


if __name__ == "__main__":
    main()
