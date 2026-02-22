"""Load the full bar exam passage corpus into ChromaDB.

Usage:
  uv run python load_corpus.py          # Load all ~220K passages
  uv run python load_corpus.py 10000    # Load first 10K passages (faster test)
  uv run python load_corpus.py status   # Check current collection size
"""

import sys
import time
from rag_utils import load_passages_to_chroma, get_vectorstore

PASSAGES_CSV = "datasets/barexam_qa/barexam_qa_train.csv"


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        vs = get_vectorstore()
        print(f"Current collection size: {vs._collection.count()}")
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
