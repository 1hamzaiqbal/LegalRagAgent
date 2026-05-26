#!/usr/bin/env python3
"""Embed normalized BEIR Phase 1 corpora into Chroma collections."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "datasets" / "beir"
REPORT = REPO_ROOT / "docs" / "generated" / "beir_phase1_phase1_embeddings_2026-05-26.md"
CHROMA_DB_DIR = Path(os.environ.get("CHROMA_DB_DIR", str(REPO_ROOT / "chroma_db")))
EMBEDDING_MODEL = "Alibaba-NLP/gte-large-en-v1.5"
SUBSETS = ("scifact", "nfcorpus", "fiqa", "trec-covid", "scidocs")
EMBED_TEXT_MAX_CHARS = int(os.environ.get("EMBED_TEXT_MAX_CHARS", "4096") or "0")


def dataset_key(subset: str) -> str:
    return "beir_" + subset.replace("-", "_")


def collection_name(subset: str) -> str:
    return dataset_key(subset)


def clean_meta_value(value: Any) -> str:
    text = str(value or "").strip()
    return text if text and text.lower() != "nan" else ""


def load_model():
    import torch
    from sentence_transformers import SentenceTransformer

    device = os.environ.get("EMBEDDING_DEVICE", "").strip() or ("cuda" if torch.cuda.is_available() else "cpu")
    model_kwargs: dict[str, Any] = {"trust_remote_code": True, "device": device}
    if device != "cpu":
        model_kwargs["model_kwargs"] = {"dtype": torch.float16}
    print(f"[embed] loading {EMBEDDING_MODEL} device={device}", flush=True)
    model = SentenceTransformer(EMBEDDING_MODEL, **model_kwargs)
    model.max_seq_length = int(os.environ.get("EMBEDDING_MAX_SEQ_LENGTH", "512") or "512")
    try:
        transformer = model[0]
        auto_model = getattr(transformer, "auto_model", None)
        emb_module = getattr(auto_model, "embeddings", None)
        if emb_module is not None and hasattr(emb_module, "position_ids"):
            max_positions = int(getattr(auto_model.config, "max_position_embeddings", 0)) or int(emb_module.position_ids.numel())
            emb_module.register_buffer(
                "position_ids",
                torch.arange(max_positions, device=emb_module.word_embeddings.weight.device, dtype=torch.long),
                persistent=False,
            )
    except Exception as exc:
        print(f"[embed] position-id repair skipped: {exc}", flush=True)
    return model


def embed_subset(
    *,
    subset: str,
    model,
    reset: bool,
    chunk_size: int,
    batch_size: int,
    add_batch_size: int,
) -> dict[str, Any]:
    import chromadb

    path = DATA_ROOT / subset / "corpus.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing normalized BEIR corpus: {path}")
    raw_df = pd.read_csv(path, keep_default_na=False)
    raw_count = len(raw_df)
    df = raw_df[raw_df["idx"].astype(str).str.strip().ne("")].copy().reset_index(drop=True)
    df["text"] = df["text"].astype(str)
    empty_text_count = int(df["text"].str.strip().eq("").sum())
    coll_name = collection_name(subset)
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    if reset:
        try:
            client.delete_collection(coll_name)
            print(f"[embed] cleared {coll_name}", flush=True)
        except Exception:
            pass
    collection = client.get_or_create_collection(
        name=coll_name,
        metadata={"hnsw:space": "cosine", "embedding_model": EMBEDDING_MODEL, "source": f"BeIR/{subset}"},
    )
    before = collection.count()
    existing_ids: set[str] = set()
    if not reset:
        existing_ids = existing_chroma_ids(collection, before)
    missing_df = df
    if existing_ids:
        missing_mask = [f"doc_{idx}" not in existing_ids for idx in df["idx"].astype(str)]
        missing_df = df.loc[missing_mask].reset_index(drop=True)

    if not reset and missing_df.empty and before >= len(df):
        missing_gold = count_missing_gold_docs(subset, collection)
        return {
            "subset": subset,
            "dataset_key": dataset_key(subset),
            "collection": coll_name,
            "raw_count": raw_count,
            "expected": len(df),
            "empty_text_docs": empty_text_count,
            "before": before,
            "after": before,
            "inserted": 0,
            "missing_gold_docs": missing_gold,
            "elapsed_sec": 0.0,
            "status": "already_complete",
        }

    start = time.time()
    inserted = 0
    total_to_insert = len(missing_df)
    for chunk_start in range(0, total_to_insert, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_to_insert)
        chunk = missing_df.iloc[chunk_start:chunk_end]
        if chunk.empty:
            continue
        ids = [f"doc_{str(idx)}" for idx in chunk["idx"].astype(str).tolist()]
        texts = [
            text if str(text).strip() else f"[empty BEIR document {idx}]"
            for idx, text in zip(chunk["idx"].astype(str), chunk["text"].astype(str))
        ]
        metadatas = []
        for _, row in chunk.iterrows():
            meta = {
                "idx": str(row["idx"]),
                "source": f"BeIR/{subset}",
                "dataset": dataset_key(subset),
            }
            title = clean_meta_value(row.get("title", ""))
            if title:
                meta["title"] = title
            metadatas.append(meta)
        embed_texts = [
            text[:EMBED_TEXT_MAX_CHARS] if EMBED_TEXT_MAX_CHARS and len(text) > EMBED_TEXT_MAX_CHARS else text
            for text in texts
        ]
        embeddings = model.encode(
            embed_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        vectors = embeddings.tolist()
        for start_i in range(0, len(texts), add_batch_size):
            end_i = min(start_i + add_batch_size, len(texts))
            collection.add(
                ids=ids[start_i:end_i],
                embeddings=vectors[start_i:end_i],
                documents=texts[start_i:end_i],
                metadatas=metadatas[start_i:end_i],
            )
        inserted = chunk_end
        elapsed = max(time.time() - start, 1e-6)
        rate = inserted / elapsed
        remaining = max(total_to_insert - inserted, 0)
        eta_min = remaining / rate / 60 if rate else 0.0
        print(
            f"[embed] {coll_name} {inserted}/{total_to_insert} missing "
            f"({rate:.1f} docs/s eta={eta_min:.1f}m)",
            flush=True,
        )
    after = collection.count()
    missing_gold = count_missing_gold_docs(subset, collection)
    return {
        "subset": subset,
        "dataset_key": dataset_key(subset),
        "collection": coll_name,
        "raw_count": raw_count,
        "expected": len(df),
        "empty_text_docs": empty_text_count,
        "before": before,
        "after": after,
        "inserted": inserted,
        "missing_gold_docs": missing_gold,
        "elapsed_sec": round(time.time() - start, 1),
        "status": "ok" if after == len(df) and missing_gold == 0 else "count_or_gold_mismatch",
    }


def existing_chroma_ids(collection, count: int, batch_size: int = 10000) -> set[str]:
    ids: set[str] = set()
    for offset in range(0, count, batch_size):
        try:
            batch = collection.get(offset=offset, limit=min(batch_size, count - offset), include=[])
        except Exception:
            batch = collection.get(offset=offset, limit=min(batch_size, count - offset), include=["metadatas"])
        ids.update(str(item) for item in batch.get("ids") or [])
    return ids


def count_missing_gold_docs(subset: str, collection, batch_size: int = 5000) -> int:
    questions_path = DATA_ROOT / subset / "questions.csv"
    if not questions_path.exists():
        return -1
    questions = pd.read_csv(questions_path, keep_default_na=False)
    gold_ids = set()
    for value in questions.get("gold_idx", []):
        try:
            gold_ids.update(str(item) for item in json.loads(str(value or "[]")))
        except json.JSONDecodeError:
            pass
    requested = sorted(gold_ids)
    found = set()
    for start in range(0, len(requested), batch_size):
        chunk = requested[start:start + batch_size]
        batch = collection.get(ids=[f"doc_{idx}" for idx in chunk], include=["metadatas"])
        for chroma_id, meta in zip(batch.get("ids") or [], batch.get("metadatas") or []):
            found.add(str((meta or {}).get("idx") or str(chroma_id).removeprefix("doc_")))
    return len(gold_ids - found)


def write_report(rows: list[dict[str, Any]], output: Path) -> None:
    lines = [
        "# BEIR Phase 1 Embeddings - 2026-05-26",
        "",
        f"Phase 1 embedded normalized BEIR corpora into Chroma at `{CHROMA_DB_DIR}` with `{EMBEDDING_MODEL}`. No files under `paper/` were edited.",
        f"Embedding inputs were capped at `{EMBED_TEXT_MAX_CHARS or 'uncapped'}` characters before tokenization while full document text was stored in Chroma; the model itself is configured with `max_seq_length=512`.",
        "",
        "| Dataset | Eval key | Collection | Raw corpus docs | Embedded docs expected | Empty-text docs | Before | After | Missing gold docs | Inserted this run | Status | Elapsed |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['subset']} | `{row['dataset_key']}` | `{row['collection']}` | {row['raw_count']} | "
            f"{row['expected']} | {row['empty_text_docs']} | {row['before']} | {row['after']} | "
            f"{row['missing_gold_docs']} | {row['inserted']} | {row['status']} | {row['elapsed_sec']}s |"
        )
    lines.extend([
        "",
        "## Reproduction",
        "",
        "```bash",
        "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python scripts/embed_beir_phase1.py",
        "```",
        "",
    ])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(SUBSETS), choices=SUBSETS)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--reset", action="store_true", help="Delete and rebuild each requested collection.")
    parser.add_argument("--chunk-size", type=int, default=int(os.environ.get("EMBED_CHUNK", "5000")))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("EMBED_GPU_BATCH", "64")))
    parser.add_argument("--add-batch-size", type=int, default=5000)
    args = parser.parse_args()

    model = load_model()
    rows = []
    for subset in args.datasets:
        rows.append(
            embed_subset(
                subset=subset,
                model=model,
                reset=args.reset,
                chunk_size=args.chunk_size,
                batch_size=args.batch_size,
                add_batch_size=args.add_batch_size,
            )
        )
    write_report(rows, args.report)
    print(args.report)


if __name__ == "__main__":
    main()
