"""Load the full bar exam passage corpus into ChromaDB.

Usage:
  uv run python load_corpus.py              # Load all ~220K passages
  uv run python load_corpus.py 10000        # Load first 10K passages (faster test)
  uv run python load_corpus.py status       # Check current collection size
  uv run python load_corpus.py curated      # Gold passages + 500 padding (~1.5K)
  uv run python load_corpus.py curated 2000 # Gold passages + 2000 padding (~3K)
"""

import sys
import time
import pandas as pd
from rag_utils import load_passages_to_chroma, get_vectorstore

PASSAGES_CSV = "datasets/barexam_qa/barexam_qa_train.csv"
QA_CSV = "datasets/barexam_qa/qa/qa.csv"


def load_curated(padding: int = 500):
    """Load gold passages from the QA dataset + padding non-gold passages as noise.

    Gives 100% gold coverage for eval in ~1.5K passages (~3 min embed time vs
    30 min for 20K).
    """
    print(f"Loading curated corpus (gold passages + {padding} padding)...")

    qa = pd.read_csv(QA_CSV)
    gold_idxs = set(qa["gold_idx"].dropna().unique())
    print(f"Gold passages from QA dataset: {len(gold_idxs)}")

    passages = pd.read_csv(PASSAGES_CSV)
    total_passages = len(passages)
    print(f"Total passages in CSV: {total_passages}")

    # Split into gold and non-gold
    gold_mask = passages["idx"].isin(gold_idxs)
    gold_df = passages[gold_mask]
    non_gold_df = passages[~gold_mask]

    print(f"Gold passages found in CSV: {len(gold_df)}")

    # Take first N non-gold passages as padding/noise
    padding_df = non_gold_df.head(padding)
    print(f"Padding passages: {len(padding_df)}")

    # Combine and sort by original order (preserves idx ordering)
    curated_df = pd.concat([gold_df, padding_df]).sort_index()
    print(f"Curated total: {len(curated_df)} passages")

    # Write to a temporary CSV and load via the standard pipeline
    curated_csv = "datasets/barexam_qa/barexam_qa_curated.csv"
    curated_df.to_csv(curated_csv, index=False)

    start = time.time()
    vs = load_passages_to_chroma(curated_csv, max_passages=0)
    elapsed = time.time() - start

    count = vs._collection.count()
    print(f"\nDone. Collection now has {count} documents.")
    print(f"Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Gold coverage: {len(gold_df)}/{len(gold_idxs)} "
          f"({len(gold_df)/len(gold_idxs)*100:.0f}%)")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        vs = get_vectorstore()
        print(f"Current collection size: {vs._collection.count()}")
        return

    if len(sys.argv) > 1 and sys.argv[1] == "curated":
        padding = int(sys.argv[2]) if len(sys.argv) > 2 else 500
        load_curated(padding)
        return

    max_passages = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    label = f"first {max_passages}" if max_passages > 0 else "all"
    print(f"Loading {label} passages from {PASSAGES_CSV}...")

    start = time.time()
    vs = load_passages_to_chroma(PASSAGES_CSV, max_passages=max_passages)
    elapsed = time.time() - start

    count = vs._collection.count()
    print(f"\nDone. Collection now has {count} documents.")
    print(f"Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
