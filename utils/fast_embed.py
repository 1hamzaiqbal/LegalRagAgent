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
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import time
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHROMA_DB_DIR = "./chroma_db"

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


def embed_corpus(corpus_name: str, max_passages: int = 0, resume: bool = False):
    """Embed a corpus in memory-safe chunks: embed chunk → insert → free → repeat."""
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

    # Load embedding model
    model_name = os.getenv("EMBEDDING_MODEL", "Alibaba-NLP/gte-large-en-v1.5")
    print(f"Loading embedding model: {model_name} (fp16)...")
    model = SentenceTransformer(model_name, trust_remote_code=True,
                                model_kwargs={"dtype": torch.float16})
    model.max_seq_length = 512
    dim = model.get_sentence_embedding_dimension()
    print(f"  Model loaded, dimension: {dim}, max_seq_length: {model.max_seq_length}")

    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    if resume:
        collection = client.get_or_create_collection(
            name=config["collection"],
            metadata={"hnsw:space": "cosine"},
        )
        existing = collection.count()
        print(f"  Resuming: {existing:,} docs already in collection")
    else:
        try:
            client.delete_collection(config["collection"])
            print(f"  Cleared existing collection '{config['collection']}'")
        except Exception:
            pass
        collection = client.create_collection(
            name=config["collection"],
            metadata={"hnsw:space": "cosine"},
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
    args = sys.argv[1:]
    if not args or args[0] == "status":
        show_status()
        return

    corpus_name = args[0]
    resume = "--resume" in args
    remaining = [a for a in args[1:] if a != "--resume"]
    max_passages = int(remaining[0]) if remaining else 0
    embed_corpus(corpus_name, max_passages, resume=resume)


if __name__ == "__main__":
    main()
