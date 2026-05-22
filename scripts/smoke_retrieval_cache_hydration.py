#!/usr/bin/env python3
"""Verify that a retrieval-id cache hydrates through the eval harness.

This catches the failure mode where a cache file exists and audits cleanly, but
the answer harness rejects it because the strict replay key does not match the
current dataset collection, embedding model, filter, or label prefix.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from eval_config import EvalConfig, load_questions  # noqa: E402
from eval_harness import (  # noqa: E402
    _collection_for_config,
    _documents_from_retrieval_cache,
    _retrieval_where_for_row,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--label-prefix", required=True)
    parser.add_argument("--questions", default="full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--sample-end", type=int)
    parser.add_argument("--collection", default="")
    parser.add_argument("--retrieval-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=5,
                        help="Number of question rows to hydrate; 0 means all")
    parser.add_argument("--source-filter", default="")
    parser.add_argument("--housing-state-filter", action="store_true",
                        help="Validate row-dependent HousingQA state-filter cache keys")
    parser.add_argument("--require-doc-cache", action="store_true",
                        help="Fail unless hydrated documents come from RETRIEVAL_DOC_CACHE_PATH")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.retrieval_k <= 0:
        raise SystemExit("--retrieval-k must be positive")
    os.environ["RETRIEVAL_CACHE_PATH"] = args.cache
    if args.collection:
        os.environ["EVAL_COLLECTION_OVERRIDE"] = args.collection

    config = EvalConfig(
        dataset=args.dataset,
        questions=args.questions,
        seed=args.seed,
        sample_start=args.sample_start,
        sample_end=args.sample_end,
        retrieval_k=args.retrieval_k,
        source_filter=args.source_filter,
        housing_state_filter=args.housing_state_filter,
    )
    questions = load_questions(config)
    if args.sample_start or args.sample_end is not None:
        start = max(0, int(args.sample_start or 0))
        end = None if args.sample_end is None else max(start, int(args.sample_end))
        questions = questions.iloc[start:end].reset_index(drop=True)
    if args.limit > 0:
        questions = questions.head(args.limit)

    collection = _collection_for_config(config)
    embedding_model = os.getenv("EVAL_EMBEDDING_MODEL", "").strip() or ""

    checked = 0
    cache_hits = 0
    doc_cache_hits = 0
    first_where = None
    for _, row in questions.iterrows():
        where = _retrieval_where_for_row(row, config)
        if first_where is None:
            first_where = where
        docs, entry = _documents_from_retrieval_cache(
            row,
            args.label_prefix,
            collection,
            where,
            embedding_model,
            args.retrieval_k,
        )
        checked += 1
        cache_hits += 1 if entry else 0
        doc_cache_hits += 1 if entry and entry.get("_doc_cache_hit") is True else 0
        if len(docs) < args.retrieval_k:
            raise SystemExit(f"hydrated only {len(docs)} docs, need {args.retrieval_k}")

    print(f"cache={args.cache}")
    print(f"dataset={args.dataset} label_prefix={args.label_prefix}")
    print(f"collection={collection} embedding_model={embedding_model!r} first_where={first_where or {}}")
    print(f"checked={checked} cache_hits={cache_hits} doc_cache_hits={doc_cache_hits} retrieval_k={args.retrieval_k}")
    if checked == 0 or cache_hits != checked:
        raise SystemExit(1)
    if args.require_doc_cache and doc_cache_hits != checked:
        raise SystemExit("document cache was not used for every checked row")


if __name__ == "__main__":
    main()
