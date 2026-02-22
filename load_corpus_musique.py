"""Load MuSiQue paragraphs into a separate ChromaDB collection.

Downloads the bdsaglam/musique dataset from HuggingFace, extracts unique
paragraphs from the validation split, and loads them into the
'musique_passages' ChromaDB collection (same ./chroma_db/ persist dir).

Usage:
  uv run python load_corpus_musique.py              # Load paragraphs
  uv run python load_corpus_musique.py status        # Show collection count
"""

import sys

from datasets import load_dataset
from langchain_core.documents import Document
from tqdm import tqdm

from rag_utils import get_vectorstore

MUSIQUE_COLLECTION = "musique_passages"


def load_musique_passages():
    """Download MuSiQue validation set and load unique paragraphs into ChromaDB."""
    print("Downloading MuSiQue dataset (answerable, validation split)...")
    ds = load_dataset("bdsaglam/musique", "answerable", split="validation")
    print(f"Dataset size: {len(ds)} questions")

    # Extract unique paragraphs across all questions
    # Each example has 'paragraphs' with title, paragraph_text, is_supporting
    seen_keys = set()
    documents = []

    for example in tqdm(ds, desc="Extracting paragraphs"):
        paragraphs = example.get("paragraphs", [])
        for para in paragraphs:
            title = para.get("title", "")
            text = para.get("paragraph_text", "")
            if not text:
                continue

            # Deduplicate by (title, first 100 chars)
            key = (title, text[:100])
            if key in seen_keys:
                continue
            seen_keys.add(key)

            idx = f"{title}_{len(documents)}"
            is_supporting = para.get("is_supporting", False)

            doc = Document(
                page_content=text,
                metadata={
                    "idx": idx,
                    "title": title,
                    "source": "musique",
                    "is_supporting": str(is_supporting),
                },
            )
            documents.append(doc)

    print(f"Extracted {len(documents)} unique paragraphs")

    # Load into ChromaDB
    vectorstore = get_vectorstore(MUSIQUE_COLLECTION)
    existing = vectorstore._collection.count()
    if existing >= len(documents):
        print(f"Collection already has {existing} documents (>= {len(documents)}). Skipping.")
        return vectorstore

    if existing > 0:
        print(f"Clearing existing {existing} documents...")
        ids = vectorstore._collection.get()["ids"]
        for i in range(0, len(ids), 5000):
            vectorstore._collection.delete(ids=ids[i:i + 5000])

    # Add in batches
    batch_size = 500
    total = len(documents)
    for i in range(0, total, batch_size):
        batch = documents[i:i + batch_size]
        vectorstore.add_documents(batch)
        done = min(i + batch_size, total)
        if done % 2000 == 0 or done == total:
            print(f"  Progress: {done}/{total} ({done / total * 100:.1f}%)")

    print(f"Loaded {total} paragraphs into '{MUSIQUE_COLLECTION}' collection.")
    return vectorstore


def show_status():
    """Show the current state of the MuSiQue collection."""
    vectorstore = get_vectorstore(MUSIQUE_COLLECTION)
    count = vectorstore._collection.count()
    print(f"MuSiQue collection '{MUSIQUE_COLLECTION}': {count} paragraphs")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        show_status()
    else:
        load_musique_passages()
