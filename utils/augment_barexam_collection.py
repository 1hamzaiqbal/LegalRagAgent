#!/usr/bin/env python3
"""Append BarExam validation/test passages to an existing Chroma collection.

The original `legal_passages` collection was built from the BarExam train
passages. Full BarExam QA rows can point at validation/test passage ids too, so
this utility patches the existing collection to the qrel-complete corpus without
re-embedding the already-populated train split.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CHROMA_DB_DIR = os.environ.get("CHROMA_DB_DIR", "./chroma_db")
DEFAULT_EMBEDDING_MODEL = "Alibaba-NLP/gte-large-en-v1.5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--passages", default="datasets/barexam_qa/passages/passages.tsv")
    parser.add_argument("--collection", default="legal_passages")
    parser.add_argument("--model", default=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--idx-col", default="idx")
    parser.add_argument("--embed-chunk", type=int, default=10000)
    parser.add_argument("--insert-batch", type=int, default=5000)
    parser.add_argument("--gpu-batch", type=int, default=128)
    parser.add_argument("--check-batch", type=int, default=5000)
    parser.add_argument("--limit", type=int, default=0,
                        help="Append at most this many missing passages; 0 means all")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _metadata(row: pd.Series, idx_col: str) -> dict[str, str]:
    metadata = {"idx": str(row[idx_col])}
    for col in ["source", "state", "citation"]:
        if col in row and pd.notna(row[col]):
            metadata[col] = str(row[col])
    return metadata


def main() -> None:
    args = parse_args()
    passages_path = Path(args.passages)
    if not passages_path.is_file():
        raise SystemExit(f"passages file not found: {passages_path}")

    import chromadb

    print(f"Reading {passages_path}...")
    started = time.time()
    df = pd.read_csv(passages_path, sep="\t")
    df = df.dropna(subset=[args.text_col, args.idx_col]).reset_index(drop=True)
    df[args.idx_col] = df[args.idx_col].astype(str)
    print(f"  loaded {len(df):,} valid passages in {time.time() - started:.1f}s")

    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    collection = client.get_collection(args.collection)
    before_count = collection.count()
    print(f"collection={args.collection} before_count={before_count:,}")

    idxs = df[args.idx_col].tolist()
    missing_mask = [False] * len(df)
    found = 0
    for offset in range(0, len(idxs), args.check_batch):
        batch_idxs = idxs[offset:offset + args.check_batch]
        batch_ids = [f"doc_{idx}" for idx in batch_idxs]
        result = collection.get(ids=batch_ids, include=["metadatas"])
        present = {
            str(metadata.get("idx") or chroma_id.removeprefix("doc_"))
            for chroma_id, metadata in zip(result.get("ids") or [], result.get("metadatas") or [])
        }
        found += len(present)
        for local_i, idx in enumerate(batch_idxs):
            if idx not in present:
                missing_mask[offset + local_i] = True

    missing_df = df.loc[missing_mask].reset_index(drop=True)
    if args.limit > 0:
        missing_df = missing_df.head(args.limit).reset_index(drop=True)
    print(f"target_rows={len(df):,} existing_ids_seen={found:,} missing_to_add={len(missing_df):,}")

    if args.dry_run or missing_df.empty:
        print("dry_run_or_nothing_to_add=true")
        return

    from sentence_transformers import SentenceTransformer
    import torch

    print(f"Loading embedding model: {args.model} (fp16)")
    model = SentenceTransformer(args.model, trust_remote_code=True,
                                model_kwargs={"dtype": torch.float16})
    model.max_seq_length = 512
    print(f"  dimension={model.get_sentence_embedding_dimension()} max_seq_length={model.max_seq_length}")

    total_embed_time = 0.0
    total_insert_time = 0.0
    start_time = time.time()
    total = len(missing_df)

    for chunk_start in range(0, total, args.embed_chunk):
        chunk_end = min(chunk_start + args.embed_chunk, total)
        chunk_df = missing_df.iloc[chunk_start:chunk_end]
        texts = chunk_df[args.text_col].astype(str).tolist()
        idx_values = chunk_df[args.idx_col].astype(str).tolist()
        ids = [f"doc_{idx}" for idx in idx_values]
        metadatas = [_metadata(row, args.idx_col) for _, row in chunk_df.iterrows()]

        t0 = time.time()
        embeddings = model.encode(
            texts,
            batch_size=args.gpu_batch,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        embed_dt = time.time() - t0
        total_embed_time += embed_dt

        t0 = time.time()
        embeddings_list = embeddings.tolist()
        for i in range(0, len(texts), args.insert_batch):
            end = min(i + args.insert_batch, len(texts))
            collection.add(
                ids=ids[i:end],
                embeddings=embeddings_list[i:end],
                documents=texts[i:end],
                metadatas=metadatas[i:end],
            )
        insert_dt = time.time() - t0
        total_insert_time += insert_dt

        del embeddings, embeddings_list

        elapsed = time.time() - start_time
        rate = chunk_end / elapsed if elapsed > 0 else 0.0
        eta = (total - chunk_end) / rate if rate > 0 else 0.0
        print(
            f"  {chunk_end:>8,}/{total:,} ({chunk_end/total*100:5.1f}%) | "
            f"embed={embed_dt:.0f}s insert={insert_dt:.0f}s | "
            f"{rate:.0f} docs/sec | ETA {eta/60:.0f}min"
        )

    print("Done.")
    print(f"  Embedding: {total_embed_time:.0f}s ({total_embed_time/60:.1f}min)")
    print(f"  Insertion: {total_insert_time:.0f}s ({total_insert_time/60:.1f}min)")
    print(f"  Collection: {collection.count():,} documents")


if __name__ == "__main__":
    main()
