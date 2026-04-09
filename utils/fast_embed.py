"""Fast bulk embedding: bypasses LangChain, uses sentence-transformers directly.

Embeds passages in GPU batches and inserts into ChromaDB in chunks to
avoid OOM on large corpora. ~10x faster than load_corpus.py.

Usage:
  uv run python utils/fast_embed.py barexam           # Full barexam corpus (686K)
  uv run python utils/fast_embed.py barexam 50000     # First 50K barexam passages
  uv run python utils/fast_embed.py housing            # Full housing statutes (1.84M)
  uv run python utils/fast_embed.py housing 200000     # First 200K housing statutes
  uv run python utils/fast_embed.py housing --resume   # Resume interrupted embedding
  uv run python utils/fast_embed.py status             # Check collection sizes

  # Embedding model A/B testing:
  uv run python utils/fast_embed.py barexam --model BAAI/bge-m3
  uv run python utils/fast_embed.py barexam --model BAAI/bge-large-en-v1.5
  uv run python utils/fast_embed.py barexam --model sentence-transformers/all-MiniLM-L6-v2

  Collections auto-suffixed: legal_passages → legal_passages__bge_m3
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import time
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHROMA_DB_DIR = os.environ.get("CHROMA_DB_DIR", "./chroma_db")

# Default embedding model (current baseline)
DEFAULT_EMBEDDING_MODEL = "Alibaba-NLP/gte-large-en-v1.5"

# Embedding models to A/B test.
# Key: short name (used in collection suffix and EMBEDDING_MODEL env var)
# Value: HuggingFace model ID
EMBEDDING_MODELS = {
    # --- Baseline ---
    "gte-large":       "Alibaba-NLP/gte-large-en-v1.5",    # Current baseline, 1024d, 8192 tok, ~1.7GB
    # --- Top MTEB retrieval candidates (RTX 3070 compatible) ---
    "stella-400m":     "dunzhang/stella_en_400M_v5",        # 400M, 1024d, 131k tok, ~0.8GB, MTEB ret ~58.5
    "stella-1.5b":     "dunzhang/stella_en_1.5B_v5",        # 1.5B, 1024d, 131k tok, ~3.0GB, MTEB ret ~59.8
    "bge-m3":          "BAAI/bge-m3",                       # 568M, 1024d, 8192 tok, ~1.1GB, hybrid retrieval
    "jina-v3":         "jinaai/jina-embeddings-v3",         # 570M, 1024d, 8192 tok, ~1.1GB, task-specific LoRA
    "arctic-l-v2":     "Snowflake/snowflake-arctic-embed-l-v2.0",  # 568M, 1024d, 8192 tok, ~1.1GB, no trust_remote
    "nomic-v2-moe":    "nomic-ai/nomic-embed-text-v2-moe",  # 475M MoE, 768d, 8192 tok, ~0.9GB
    # --- Legal domain ---
    "legal-bert":      "nlpaueb/legal-bert-base-uncased",   # 110M, 768d, 512 tok, ~0.4GB (domain-specific)
    # --- Previous candidates ---
    "bge-large":       "BAAI/bge-large-en-v1.5",            # 335M, 1024d, 512 tok
    "gte-qwen2-1.5b":  "Alibaba-NLP/gte-Qwen2-1.5B-instruct",  # 1.5B, 1536d, 32k tok
    "all-minilm":      "sentence-transformers/all-MiniLM-L6-v2", # 22M, 384d, 256 tok (speed baseline)
    "nomic-v1.5":      "nomic-ai/nomic-embed-text-v1.5",    # 137M, 768d, 8192 tok, ~0.5GB
}


def model_name_to_suffix(model_name: str) -> str:
    """Convert a HuggingFace model name to a collection suffix.

    'Alibaba-NLP/gte-large-en-v1.5' → 'gte_large_en_v1_5'
    'BAAI/bge-m3' → 'bge_m3'
    """
    # Take the part after / if present
    short = model_name.split("/")[-1]
    # Replace non-alphanumeric with underscore
    import re
    return re.sub(r"[^a-z0-9]+", "_", short.lower()).strip("_")


def resolve_collection_name(base_collection: str, model_name: str) -> str:
    """Get collection name for a specific embedding model.

    If it's the default model (gte-large-en-v1.5), returns the base name
    (backward compatible). Otherwise, appends a suffix.
    """
    if model_name == DEFAULT_EMBEDDING_MODEL:
        return base_collection
    suffix = model_name_to_suffix(model_name)
    return f"{base_collection}__{suffix}"


CORPORA = {
    "barexam": {
        "csv": "datasets/barexam_qa/barexam_qa_train.csv",
        "collection": "legal_passages",
        "text_col": "text",
        "idx_col": "idx",
    },
    "housing": {
        "csv": "datasets/housing_qa/statutes.csv",
        "collection": "housing_statutes",
        "text_col": "text",
        "idx_col": "idx",
    },
    "legal_rag": {
        "csv": "datasets/legal_rag_qa/passages.csv",
        "collection": "legal_rag_passages",
        "text_col": "text",
        "idx_col": "idx",
    },
    "australian": {
        "csv": "datasets/australian_legal_qa/passages.csv",
        "collection": "australian_legal",
        "text_col": "text",
        "idx_col": "idx",
    },
    "casehold": {
        "csv": "datasets/casehold/holdings_corpus.csv",
        "collection": "casehold_holdings",
        "text_col": "text",
        "idx_col": "idx",
    },
}

# Process in chunks of this size to avoid OOM
EMBED_CHUNK = 10000


def embed_corpus(corpus_name: str, max_passages: int = 0, resume: bool = False,
                 model_override: str = None):
    """Embed a corpus in memory-safe chunks: embed chunk → insert → free → repeat.

    Args:
        corpus_name: Key from CORPORA dict (e.g., 'barexam')
        max_passages: Limit number of passages (0 = all)
        resume: Resume from last inserted position
        model_override: HuggingFace model ID or short name from EMBEDDING_MODELS
    """
    from sentence_transformers import SentenceTransformer
    import torch
    import chromadb

    if corpus_name not in CORPORA:
        print(f"Unknown corpus: {corpus_name}. Options: {list(CORPORA.keys())}")
        return

    config = CORPORA[corpus_name]
    csv_path = config["csv"]

    if not os.path.isfile(csv_path):
        print(f"CSV not found: {csv_path}")
        return

    # Load CSV
    print(f"Reading {csv_path}...")
    t0 = time.time()
    df = pd.read_csv(csv_path)
    if max_passages > 0:
        df = df.head(max_passages)
    print(f"  Loaded {len(df):,} rows in {time.time()-t0:.1f}s")

    # Filter valid rows
    text_col = config["text_col"]
    idx_col = config["idx_col"]
    df = df.dropna(subset=[text_col, idx_col]).reset_index(drop=True)
    total = len(df)
    print(f"  {total:,} valid passages to embed")

    # Resolve embedding model
    if model_override:
        # Check if it's a short name from EMBEDDING_MODELS
        if model_override in EMBEDDING_MODELS:
            model_name = EMBEDDING_MODELS[model_override]
        else:
            model_name = model_override
    else:
        model_name = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)

    # Resolve collection name (auto-suffixed for non-default models)
    collection_name = resolve_collection_name(config["collection"], model_name)

    print(f"Loading embedding model: {model_name} (fp16)...")
    print(f"  Collection: {collection_name}")
    model = SentenceTransformer(model_name, trust_remote_code=True,
                                model_kwargs={"dtype": torch.float16})
    model.max_seq_length = 512
    dim = model.get_sentence_embedding_dimension()
    print(f"  Model loaded, dimension: {dim}, max_seq_length: {model.max_seq_length}")

    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    if resume:
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        existing = collection.count()
        print(f"  Resuming: {existing:,} docs already in collection")
    else:
        try:
            client.delete_collection(collection_name)
            print(f"  Cleared existing collection '{collection_name}'")
        except Exception:
            pass
        collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine", "embedding_model": model_name},
        )

    # Process in chunks: embed → insert → free
    gpu_batch = 128
    total_embed_time = 0
    total_insert_time = 0
    start_time = time.time()

    for chunk_start in range(0, total, EMBED_CHUNK):
        chunk_end = min(chunk_start + EMBED_CHUNK, total)
        chunk_df = df.iloc[chunk_start:chunk_end]

        texts = chunk_df[text_col].astype(str).tolist()
        idxs = chunk_df[idx_col].astype(str).tolist()
        batch_ids = [f"doc_{idx}" for idx in idxs]

        # Skip already-inserted chunks when resuming
        if resume and chunk_end <= existing:
            continue

        # Build metadata
        metadatas = []
        for _, row in chunk_df.iterrows():
            meta = {"idx": str(row[idx_col])}
            for col in ["source", "state", "citation"]:
                if col in row and pd.notna(row[col]):
                    meta[col] = str(row[col])
            metadatas.append(meta)

        # Embed this chunk
        t0 = time.time()
        embeddings = model.encode(
            texts, batch_size=gpu_batch,
            show_progress_bar=False, normalize_embeddings=True,
        )
        embed_dt = time.time() - t0
        total_embed_time += embed_dt

        # Insert into ChromaDB
        t0 = time.time()
        embeddings_list = embeddings.tolist()

        # ChromaDB add in sub-batches of 5000
        for i in range(0, len(texts), 5000):
            end = min(i + 5000, len(texts))
            collection.add(
                ids=batch_ids[i:end],
                embeddings=embeddings_list[i:end],
                documents=texts[i:end],
                metadatas=metadatas[i:end],
            )
        insert_dt = time.time() - t0
        total_insert_time += insert_dt

        # Free memory
        del embeddings, embeddings_list

        elapsed = time.time() - start_time
        rate = chunk_end / elapsed
        eta = (total - chunk_end) / rate if rate > 0 else 0
        print(
            f"  {chunk_end:>8,}/{total:,} ({chunk_end/total*100:5.1f}%) | "
            f"embed={embed_dt:.0f}s insert={insert_dt:.0f}s | "
            f"{rate:.0f} docs/sec | ETA {eta/60:.0f}min"
        )

    total_time = time.time() - start_time
    print(f"\nDone!")
    print(f"  Embedding:  {total_embed_time:.0f}s ({total_embed_time/60:.1f}min)")
    print(f"  Insertion:  {total_insert_time:.0f}s ({total_insert_time/60:.1f}min)")
    print(f"  Total:      {total_time:.0f}s ({total_time/60:.1f}min, {total_time/3600:.1f}hr)")
    print(f"  Collection: {collection.count():,} documents")


def show_status():
    """Show collection sizes."""
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    print("Collection sizes:")
    for c in client.list_collections():
        print(f"  {c.name}: {c.count():,} documents")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fast bulk embedding for ChromaDB")
    parser.add_argument("corpus", nargs="?", default="status",
                        help="Corpus to embed (barexam, housing, etc.) or 'status'")
    parser.add_argument("max_passages", nargs="?", type=int, default=0,
                        help="Max passages to embed (0 = all)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume interrupted embedding")
    parser.add_argument("--model", type=str, default=None,
                        help="Embedding model: short name (bge-m3, bge-large, legal-bert, "
                             "gte-qwen2-1.5b, all-minilm) or full HuggingFace model ID")
    parser.add_argument("--list-models", action="store_true",
                        help="List available embedding models for A/B testing")

    args = parser.parse_args()

    if args.list_models:
        print("\nAvailable embedding models for A/B testing:")
        print(f"{'Short Name':<20} {'HuggingFace ID':<50}")
        print("-" * 70)
        for short, full in EMBEDDING_MODELS.items():
            is_default = " (current default)" if full == DEFAULT_EMBEDDING_MODEL else ""
            print(f"{short:<20} {full:<50}{is_default}")
        print()
        return

    if args.corpus == "status":
        show_status()
        return

    embed_corpus(args.corpus, args.max_passages, resume=args.resume,
                 model_override=args.model)


if __name__ == "__main__":
    main()
