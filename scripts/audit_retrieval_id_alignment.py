#!/usr/bin/env python3
"""Check whether dataset gold ids exist in the configured Chroma collection.

This is a source gate for retrieval-first claims. If a dataset's gold ids are
question ids or labels rather than corpus document ids, Hit@k/MRR against those
ids is not a valid retrieval-exposure metric for that collection.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from eval_config import EvalConfig, load_questions  # noqa: E402
from eval_harness import _collection_for_config, _gold_ids  # noqa: E402
from rag_utils import CHROMA_DB_DIR  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=[
        "barexam", "housing", "legal_rag", "australian", "casehold",
        "musique", "legalbench_scalr",
    ])
    parser.add_argument("--questions", default="full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--collection", default="")
    parser.add_argument("--min-exists", type=float, default=0.95,
                        help="Required fraction of unique gold ids found in Chroma")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--metadata-fallback", action="store_true",
                        help="Also try one batched metadata idx lookup for ids not found as doc_{idx}")
    parser.add_argument("--sample-missing", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = EvalConfig(dataset=args.dataset, questions=args.questions, seed=args.seed)
    collection = args.collection or _collection_for_config(config)
    questions = load_questions(config)

    gold_ids: list[str] = []
    rows_with_gold = 0
    for _, row in questions.iterrows():
        ids = _gold_ids(row)
        if ids:
            rows_with_gold += 1
            gold_ids.extend(ids)
    unique_gold = list(dict.fromkeys(gold_ids))
    if not unique_gold:
        raise SystemExit(f"{args.dataset}: no gold ids found")

    import chromadb

    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    chroma_collection = client.get_collection(collection)

    found_ids: set[str] = set()

    def _store_batch(batch: dict) -> None:
        chroma_ids = batch.get("ids") or []
        metadatas = batch.get("metadatas") or []
        for chroma_id, metadata in zip(chroma_ids, metadatas):
            metadata = dict(metadata or {})
            idx = str(metadata.get("idx") or str(chroma_id).removeprefix("doc_"))
            if idx:
                found_ids.add(idx)

    for offset in range(0, len(unique_gold), args.batch_size):
        batch = unique_gold[offset:offset + args.batch_size]
        _store_batch(chroma_collection.get(
            ids=[f"doc_{idx}" for idx in batch],
            include=["metadatas"],
        ))

    if args.metadata_fallback:
        missing_after_id = [idx for idx in unique_gold if idx not in found_ids]
        for offset in range(0, len(missing_after_id), args.batch_size):
            batch = missing_after_id[offset:offset + args.batch_size]
            _store_batch(chroma_collection.get(
                where={"idx": {"$in": batch}},
                include=["metadatas"],
            ))

    missing = [idx for idx in unique_gold if idx not in found_ids]
    exists_fraction = (len(unique_gold) - len(missing)) / len(unique_gold)
    print(f"dataset={args.dataset}")
    print(f"collection={collection}")
    print(f"rows={len(questions)} rows_with_gold={rows_with_gold}")
    print(f"unique_gold_ids={len(unique_gold)} found={len(unique_gold)-len(missing)} missing={len(missing)}")
    print(f"exists_fraction={exists_fraction:.4f} min_exists={args.min_exists:.4f}")
    print(f"metadata_fallback={args.metadata_fallback}")
    if missing:
        print("missing_examples=" + ",".join(missing[:args.sample_missing]))
    if exists_fraction < args.min_exists:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
