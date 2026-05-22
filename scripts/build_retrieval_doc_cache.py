#!/usr/bin/env python3
"""Build a document-text cache for strict retrieval-id replay.

Retrieval caches store deterministic passage IDs. This cache stores the
corresponding Chroma document text and metadata so answer replays can hydrate
evidence without opening a large Chroma collection on every run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from eval_config import EvalConfig, load_questions  # noqa: E402
from eval_harness import _collection_for_config, _gold_ids  # noqa: E402
from rag_utils import get_documents_by_idx  # noqa: E402


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value) != ""))


def _load_existing(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    seen: set[tuple[str, str]] = set()
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            collection = str(row.get("collection") or "")
            idx = str(row.get("idx") or "")
            if collection and idx:
                seen.add((collection, idx))
    return seen


def _ids_from_cache(path: Path, include_effective: bool) -> dict[str, set[str]]:
    needed: dict[str, set[str]] = {}
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            collection = str(row.get("collection") or "")
            if not collection:
                raise SystemExit(f"{path}:{line_no}: missing collection")
            ids = _dedupe([str(v) for v in row.get("retrieved_ids") or []])
            if include_effective:
                ids.extend(_dedupe([str(v) for v in row.get("effective_retrieved_ids") or []]))
            needed.setdefault(collection, set()).update(_dedupe(ids))
    return needed


def _ids_from_gold(dataset: str, questions: str, seed: int, collection: str | None) -> dict[str, set[str]]:
    config = EvalConfig(dataset=dataset, questions=questions, seed=seed)
    rows = load_questions(config)
    coll = collection or _collection_for_config(config)
    ids: set[str] = set()
    for _, row in rows.iterrows():
        ids.update(_gold_ids(row))
    return {coll: ids}


def _merge_needed(target: dict[str, set[str]], source: dict[str, set[str]]) -> None:
    for collection, ids in source.items():
        target.setdefault(collection, set()).update(ids)


def _serialize_doc(collection: str, doc) -> dict[str, Any]:
    metadata = dict(doc.metadata or {})
    idx = str(metadata.get("idx") or "")
    if not idx:
        raise ValueError("document missing metadata.idx")
    metadata.setdefault("idx", idx)
    return {
        "collection": collection,
        "idx": idx,
        "text": doc.page_content or "",
        "metadata": metadata,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-cache", action="append", type=Path, default=[],
                        help="Retrieval-id cache JSONL to hydrate. May be repeated.")
    parser.add_argument("--include-effective", action="store_true",
                        help="Also hydrate effective_retrieved_ids from cache rows.")
    parser.add_argument("--include-gold-dataset", action="append", default=[],
                        help="Also hydrate gold_idx docs for a dataset, e.g. housing. May be repeated.")
    parser.add_argument("--questions", default="full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--collection", default="",
                        help="Collection override for --include-gold-dataset when one dataset is provided.")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--strict", action="store_true",
                        help="Fail if any requested document cannot be hydrated.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if not args.retrieval_cache and not args.include_gold_dataset:
        raise SystemExit("provide --retrieval-cache and/or --include-gold-dataset")

    needed: dict[str, set[str]] = {}
    for cache_path in args.retrieval_cache:
        _merge_needed(needed, _ids_from_cache(cache_path, include_effective=args.include_effective))
    for dataset in args.include_gold_dataset:
        collection = args.collection if len(args.include_gold_dataset) == 1 else None
        _merge_needed(needed, _ids_from_gold(dataset, args.questions, args.seed, collection or None))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_existing(args.out) if args.resume else set()
    mode = "a" if args.resume else "w"
    total_needed = sum(len(ids) for ids in needed.values())
    total_remaining = sum(
        1 for collection, ids in needed.items() for idx in ids if (collection, idx) not in existing
    )
    print(f"doc_cache_out={args.out}")
    print(f"collections={len(needed)} total_needed={total_needed} remaining={total_remaining}")

    wrote = 0
    missing: list[tuple[str, str]] = []
    with args.out.open(mode) as out:
        for collection, ids_set in sorted(needed.items()):
            ids = [idx for idx in sorted(ids_set) if (collection, idx) not in existing]
            print(f"[collection] {collection} ids={len(ids)}", flush=True)
            for start in range(0, len(ids), args.batch_size):
                batch_ids = ids[start:start + args.batch_size]
                docs = get_documents_by_idx(collection, batch_ids, embedding_model=os.getenv("EVAL_EMBEDDING_MODEL", "").strip() or None)
                found = {str(doc.metadata.get("idx") or ""): doc for doc in docs}
                for idx in batch_ids:
                    doc = found.get(idx)
                    if doc is None:
                        missing.append((collection, idx))
                        continue
                    out.write(json.dumps(_serialize_doc(collection, doc), sort_keys=True) + "\n")
                    wrote += 1
                out.flush()
                print(
                    f"[progress] {collection} {min(start + args.batch_size, len(ids))}/{len(ids)} "
                    f"wrote={wrote} missing={len(missing)}",
                    flush=True,
                )

    if missing:
        preview = ", ".join(f"{collection}:{idx}" for collection, idx in missing[:10])
        message = f"missing {len(missing)} requested documents; first: {preview}"
        if args.strict:
            raise SystemExit(message)
        print(f"WARNING: {message}")
    print(f"wrote={wrote} existing={len(existing)} missing={len(missing)}")


if __name__ == "__main__":
    main()
