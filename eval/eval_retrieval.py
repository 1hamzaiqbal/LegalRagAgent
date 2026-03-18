"""Compare retrieval methods: bi-encoder only, BM25 only, and hybrid.

Runs the same queries through each method and reports:
- Cross-encoder scores (after reranking)
- Passage overlap between methods
- Timing
- Which method surfaces unique high-quality passages

Usage:
  uv run python eval/eval_retrieval.py
"""

import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from rag_utils import (
    retrieve_documents,
    _retrieve_bm25,
    _retrieve_dense,
    rerank_with_cross_encoder,
    get_bm25_index,
)


TEST_QUERIES = [
    # Conceptual — embeddings should do well
    "What are the elements of a negligence claim?",
    # Specific doctrine — BM25 should catch exact terms
    "What is the parol evidence rule and when does it apply?",
    # Case name + doctrine — BM25 advantage
    "Terry v Ohio reasonable suspicion standard for investigatory stops",
    # Latin legal term — pure keyword match
    "res judicata claim preclusion collateral estoppel",
    # Multi-concept — tests breadth
    "What exceptions limit the exclusionary rule such as good faith or inevitable discovery?",
    # Statute-heavy
    "UCC Article 2 statute of frauds requirements for sale of goods",
    # Constitutional
    "When does the Fourth Amendment automobile exception allow warrantless vehicle search?",
    # Simple direct
    "What is consideration in contract law?",
]


def _passage_summary(doc, max_len=120):
    """Short passage preview."""
    text = doc.page_content[:max_len].replace("\n", " ")
    if len(doc.page_content) > max_len:
        text += "..."
    return text


def run_comparison():
    k = 5
    fetch_k = k * 4

    # Pre-build BM25 index (one-time cost)
    print("=" * 80)
    t0 = time.time()
    get_bm25_index()
    bm25_build_time = time.time() - t0
    print(f"BM25 index build time: {bm25_build_time:.1f}s")
    print("=" * 80)

    for i, query in enumerate(TEST_QUERIES):
        print(f"\n{'='*80}")
        print(f"QUERY {i+1}: {query}")
        print(f"{'='*80}")

        # --- Dense only ---
        t0 = time.time()
        dense_raw = _retrieve_dense(query, k=fetch_k)
        dense_results = rerank_with_cross_encoder(query, dense_raw, top_k=k)
        dense_time = time.time() - t0

        # --- BM25 only (raw candidates, then cross-encoder rerank) ---
        t0 = time.time()
        bm25_raw = _retrieve_bm25(query, k=fetch_k)
        bm25_results = rerank_with_cross_encoder(query, bm25_raw, top_k=k)
        bm25_time = time.time() - t0

        # --- Hybrid (what retrieve_documents now does) ---
        t0 = time.time()
        hybrid_results = retrieve_documents(query, k=k)
        hybrid_time = time.time() - t0

        # --- Compare ---
        dense_ids = {d.metadata.get("idx", "") for d in dense_results}
        bm25_ids = {d.metadata.get("idx", "") for d in bm25_results}
        hybrid_ids = {d.metadata.get("idx", "") for d in hybrid_results}

        dense_scores = [d.metadata.get("cross_encoder_score", 0) for d in dense_results]
        bm25_scores = [d.metadata.get("cross_encoder_score", 0) for d in bm25_results]
        hybrid_scores = [d.metadata.get("cross_encoder_score", 0) for d in hybrid_results]

        overlap_db = len(dense_ids & bm25_ids)
        overlap_dh = len(dense_ids & hybrid_ids)
        hybrid_unique = hybrid_ids - dense_ids

        print(f"\n  {'Method':<12} {'Top Score':>10} {'Mean Score':>11} {'Time':>7}")
        print(f"  {'-'*42}")
        print(f"  {'Dense':<12} {max(dense_scores) if dense_scores else 0:>10.3f} {sum(dense_scores)/len(dense_scores) if dense_scores else 0:>11.3f} {dense_time:>6.2f}s")
        print(f"  {'BM25':<12} {max(bm25_scores) if bm25_scores else 0:>10.3f} {sum(bm25_scores)/len(bm25_scores) if bm25_scores else 0:>11.3f} {bm25_time:>6.2f}s")
        print(f"  {'Hybrid':<12} {max(hybrid_scores) if hybrid_scores else 0:>10.3f} {sum(hybrid_scores)/len(hybrid_scores) if hybrid_scores else 0:>11.3f} {hybrid_time:>6.2f}s")

        print(f"\n  Overlap: dense∩bm25={overlap_db}/{k}, dense∩hybrid={overlap_dh}/{k}")
        if hybrid_unique:
            print(f"  Hybrid found {len(hybrid_unique)} passage(s) that dense missed:")
            for doc in hybrid_results:
                idx = doc.metadata.get("idx", "")
                if idx in hybrid_unique:
                    score = doc.metadata.get("cross_encoder_score", 0)
                    print(f"    [{idx}] score={score:.3f} | {_passage_summary(doc)}")

        # Show top passage from each method
        print(f"\n  Top passage (dense):  [{dense_results[0].metadata.get('idx', '')}] {_passage_summary(dense_results[0])}")
        print(f"  Top passage (bm25):   [{bm25_results[0].metadata.get('idx', '')}] {_passage_summary(bm25_results[0])}")
        print(f"  Top passage (hybrid): [{hybrid_results[0].metadata.get('idx', '')}] {_passage_summary(hybrid_results[0])}")

    print(f"\n{'='*80}")
    print("DONE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    run_comparison()
