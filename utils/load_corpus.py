"""Load passage corpora into ChromaDB.

Supports barexam_qa (bar exam passages) and housing_qa (housing statutes).

Usage:
  uv run python utils/load_corpus.py                  # Load all barexam passages
  uv run python utils/load_corpus.py 10000             # Load first 10K barexam passages
  uv run python utils/load_corpus.py status            # Check collection sizes
  uv run python utils/load_corpus.py curated           # Gold passages + 500 padding
  uv run python utils/load_corpus.py curated 2000      # Gold passages + 2000 padding
  uv run python utils/load_corpus.py housing            # Load all housing statutes
  uv run python utils/load_corpus.py housing 200000     # Load first 200K housing statutes
"""

# Windows: prevent OpenMP segfault when PyTorch and sentence-transformers
# each load their own OpenMP runtime (libiomp5md.dll conflict).
# Must be set before any torch/transformers import.
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from rag_utils import load_passages_to_chroma, get_vectorstore

# Barexam QA corpus
BAREXAM_PASSAGES_CSV = "datasets/barexam_qa/barexam_qa_train.csv"
BAREXAM_QA_CSV = "datasets/barexam_qa/qa/qa.csv"
BAREXAM_COLLECTION = "legal_passages"

# Housing QA corpus
HOUSING_STATUTES_CSV = "datasets/housing_qa/statutes.csv"
HOUSING_COLLECTION = "housing_statutes"


def load_curated(padding: int = 500):
    """Load gold passages from the barexam QA dataset + padding non-gold passages."""
    print(f"Loading curated barexam corpus (gold passages + {padding} padding)...")

    qa = pd.read_csv(BAREXAM_QA_CSV)
    gold_idxs = set(qa["gold_idx"].dropna().unique())
    print(f"Gold passages from QA dataset: {len(gold_idxs)}")

    passages = pd.read_csv(BAREXAM_PASSAGES_CSV)
    total_passages = len(passages)
    print(f"Total passages in CSV: {total_passages}")

    gold_mask = passages["idx"].isin(gold_idxs)
    gold_df = passages[gold_mask]
    non_gold_df = passages[~gold_mask]

    print(f"Gold passages found in CSV: {len(gold_df)}")

    padding_df = non_gold_df.head(padding)
    print(f"Padding passages: {len(padding_df)}")

    curated_df = pd.concat([gold_df, padding_df]).sort_index()
    print(f"Curated total: {len(curated_df)} passages")

    curated_csv = "datasets/barexam_qa/barexam_qa_curated.csv"
    curated_df.to_csv(curated_csv, index=False)

    start = time.time()
    vs = load_passages_to_chroma(curated_csv, max_passages=0,
                                  collection_name=BAREXAM_COLLECTION)
    elapsed = time.time() - start

    count = vs._collection.count()
    print(f"\nDone. Collection now has {count} documents.")
    print(f"Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Gold coverage: {len(gold_df)}/{len(gold_idxs)} "
          f"({len(gold_df)/len(gold_idxs)*100:.0f}%)")


def load_housing(max_passages: int = 0):
    """Load housing statutes into a separate ChromaDB collection."""
    if not os.path.isfile(HOUSING_STATUTES_CSV):
        print(f"Housing statutes CSV not found at {HOUSING_STATUTES_CSV}")
        print("Run: uv run python utils/download_housingqa.py")
        return

    label = f"first {max_passages}" if max_passages > 0 else "all"
    print(f"Loading {label} housing statutes from {HOUSING_STATUTES_CSV}...")

    start = time.time()
    vs = load_passages_to_chroma(HOUSING_STATUTES_CSV, max_passages=max_passages,
                                  collection_name=HOUSING_COLLECTION)
    elapsed = time.time() - start

    count = vs._collection.count()
    print(f"\nDone. Housing collection now has {count} documents.")
    print(f"Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")


def show_status():
    """Show collection sizes for all corpora."""
    print("Collection sizes:")
    for name in [BAREXAM_COLLECTION, HOUSING_COLLECTION]:
        try:
            vs = get_vectorstore(collection_name=name)
            count = vs._collection.count()
            print(f"  {name}: {count:,} documents")
        except Exception:
            print(f"  {name}: not loaded")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        show_status()
        return

    if len(sys.argv) > 1 and sys.argv[1] == "curated":
        padding = int(sys.argv[2]) if len(sys.argv) > 2 else 500
        load_curated(padding)
        return

    if len(sys.argv) > 1 and sys.argv[1] == "housing":
        max_p = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        load_housing(max_p)
        return

    # Default: load barexam passages
    max_passages = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    label = f"first {max_passages}" if max_passages > 0 else "all"
    print(f"Loading {label} barexam passages from {BAREXAM_PASSAGES_CSV}...")

    start = time.time()
    vs = load_passages_to_chroma(BAREXAM_PASSAGES_CSV, max_passages=max_passages,
                                  collection_name=BAREXAM_COLLECTION)
    elapsed = time.time() - start

    count = vs._collection.count()
    print(f"\nDone. Collection now has {count} documents.")
    print(f"Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
