"""Quick retrieval eval comparing bi-encoder-only vs cross-encoder reranked retrieval.

Runs on a stratified sample of 100 QA pairs (evenly across subjects) to keep
runtime under 30 minutes while still being statistically meaningful.

Usage:
  uv run python eval_reranker.py          # 100-query sample
  uv run python eval_reranker.py 50       # 50-query sample (faster)
"""

import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

from rag_utils import (
    get_vectorstore,
    get_cross_encoder,
    rerank_with_cross_encoder,
    compute_confidence,
)


def retrieve_biencoder_only(vectorstore, query, k=5, fetch_k=20):
    """Hybrid bi-encoder retrieval WITHOUT cross-encoder rerank (old method)."""
    try:
        study_docs = vectorstore.similarity_search_with_relevance_scores(
            query, k=fetch_k, filter={"source": {"$in": ["mbe", "wex"]}}
        )
    except Exception:
        study_docs = []
    try:
        case_docs = vectorstore.similarity_search_with_relevance_scores(
            query, k=fetch_k, filter={"source": "caselaw"}
        )
    except Exception:
        case_docs = []

    # Old interleave logic: 3 study + 2 caselaw
    study_k = (k + 1) // 2
    result, seen = [], set()
    for doc, score in sorted(study_docs, key=lambda x: x[1], reverse=True)[:study_k]:
        idx = doc.metadata.get("idx", "")
        if idx not in seen:
            seen.add(idx)
            result.append(doc)
    for doc, score in sorted(case_docs, key=lambda x: x[1], reverse=True):
        if len(result) >= k:
            break
        idx = doc.metadata.get("idx", "")
        if idx not in seen:
            seen.add(idx)
            result.append(doc)
    if len(result) < k:
        for doc, score in sorted(study_docs + case_docs, key=lambda x: x[1], reverse=True):
            if len(result) >= k:
                break
            idx = doc.metadata.get("idx", "")
            if idx not in seen:
                seen.add(idx)
                result.append(doc)
    return result


def retrieve_with_rerank(vectorstore, query, k=5, fetch_k=20):
    """Source-aware cross-encoder rerank: rerank within each pool, then interleave."""
    try:
        study_results = vectorstore.similarity_search_with_relevance_scores(
            query, k=fetch_k, filter={"source": {"$in": ["mbe", "wex"]}}
        )
    except Exception:
        study_results = []
    try:
        case_results = vectorstore.similarity_search_with_relevance_scores(
            query, k=fetch_k, filter={"source": "caselaw"}
        )
    except Exception:
        case_results = []

    # Deduplicate within each pool
    study_cands, study_seen = [], set()
    for doc, _ in study_results:
        idx = doc.metadata.get("idx", "")
        if idx not in study_seen:
            study_seen.add(idx)
            study_cands.append(doc)

    case_cands, case_seen = [], set()
    for doc, _ in case_results:
        idx = doc.metadata.get("idx", "")
        if idx not in case_seen:
            case_seen.add(idx)
            case_cands.append(doc)

    # Source-aware rerank: within each pool separately
    study_k = (k + 1) // 2  # 3 out of 5
    case_k = k - study_k     # 2 out of 5

    reranked_study = rerank_with_cross_encoder(query, study_cands, top_k=study_k)
    reranked_case = rerank_with_cross_encoder(query, case_cands, top_k=case_k)

    # Interleave
    result = list(reranked_study)
    seen = {doc.metadata.get("idx", "") for doc in result}
    for doc in reranked_case:
        if len(result) >= k:
            break
        idx = doc.metadata.get("idx", "")
        if idx not in seen:
            seen.add(idx)
            result.append(doc)

    return result


def main():
    n_queries = int(sys.argv[1]) if len(sys.argv) > 1 else 100

    # Load QA pairs
    qa = pd.read_csv("datasets/barexam_qa/qa/qa.csv")
    passages = pd.read_csv("datasets/barexam_qa/barexam_qa_train.csv", nrows=1000)
    passage_ids = set(passages["idx"].tolist())
    qa_in_store = qa[qa["gold_idx"].isin(passage_ids)].copy()

    # Stratified sample: equal queries per subject
    subjects = qa_in_store["subject"].dropna().unique()
    per_subject = max(1, n_queries // len(subjects))
    sampled = pd.concat([
        qa_in_store[qa_in_store["subject"] == s].sample(n=min(per_subject, len(qa_in_store[qa_in_store["subject"] == s])), random_state=42)
        for s in subjects
    ]).head(n_queries)

    def full_question(row):
        prompt = str(row["prompt"]) if pd.notna(row["prompt"]) else ""
        return (prompt + " " + str(row["question"])).strip()

    sampled["full_q"] = sampled.apply(full_question, axis=1)

    print(f"\n{'='*70}")
    print(f"  CROSS-ENCODER RERANKER A/B COMPARISON")
    print(f"{'='*70}")
    print(f"Queries: {len(sampled)} (stratified across {len(subjects)} subjects)")
    print(f"Corpus: {get_vectorstore()._collection.count()} documents")
    print(f"Subject distribution: {dict(Counter(sampled['subject']))}")

    # Warm up cross-encoder model
    print("\nWarming up cross-encoder model...")
    _ = get_cross_encoder()

    vs = get_vectorstore()

    # Run both methods
    results_bi = {"hits": 0, "mrr": 0.0, "conf": 0.0}
    results_re = {"hits": 0, "mrr": 0.0, "conf": 0.0}
    subject_results = {}

    print(f"\nRunning {len(sampled)} queries (bi-encoder + cross-encoder)...\n")
    start = time.time()

    for _, row in tqdm(sampled.iterrows(), total=len(sampled)):
        query = row["full_q"]
        gold_idx = row["gold_idx"]
        subject = row["subject"]

        if subject not in subject_results:
            subject_results[subject] = {
                "bi": {"hits": 0, "total": 0, "mrr": 0.0},
                "re": {"hits": 0, "total": 0, "mrr": 0.0},
            }
        subject_results[subject]["bi"]["total"] += 1
        subject_results[subject]["re"]["total"] += 1

        # --- Bi-encoder only ---
        bi_docs = retrieve_biencoder_only(vs, query, k=5, fetch_k=5)
        bi_ids = [d.metadata.get("idx", "") for d in bi_docs]
        bi_conf = compute_confidence(query, bi_docs)
        results_bi["conf"] += bi_conf

        if gold_idx in bi_ids:
            results_bi["hits"] += 1
            subject_results[subject]["bi"]["hits"] += 1
            rank = bi_ids.index(gold_idx) + 1
            results_bi["mrr"] += 1.0 / rank
            subject_results[subject]["bi"]["mrr"] += 1.0 / rank

        # --- Cross-encoder reranked ---
        re_docs = retrieve_with_rerank(vs, query, k=5, fetch_k=20)
        re_ids = [d.metadata.get("idx", "") for d in re_docs]
        re_conf = compute_confidence(query, re_docs)
        results_re["conf"] += re_conf

        if gold_idx in re_ids:
            results_re["hits"] += 1
            subject_results[subject]["re"]["hits"] += 1
            rank = re_ids.index(gold_idx) + 1
            results_re["mrr"] += 1.0 / rank
            subject_results[subject]["re"]["mrr"] += 1.0 / rank

    elapsed = time.time() - start
    n = len(sampled)

    print(f"\n{'='*70}")
    print(f"  RESULTS ({elapsed:.0f}s total, {elapsed/n:.1f}s/query)")
    print(f"{'='*70}\n")

    print(f"{'Metric':<25} {'Bi-encoder':>15} {'+ Cross-encoder':>15} {'Delta':>10}")
    print("-" * 70)

    bi_recall = results_bi["hits"] / n
    re_recall = results_re["hits"] / n
    print(f"{'Recall@5':<25} {bi_recall:>15.4f} {re_recall:>15.4f} {re_recall - bi_recall:>+10.4f}")

    bi_mrr = results_bi["mrr"] / n
    re_mrr = results_re["mrr"] / n
    print(f"{'MRR':<25} {bi_mrr:>15.4f} {re_mrr:>15.4f} {re_mrr - bi_mrr:>+10.4f}")

    bi_avgconf = results_bi["conf"] / n
    re_avgconf = results_re["conf"] / n
    print(f"{'Avg confidence':<25} {bi_avgconf:>15.4f} {re_avgconf:>15.4f} {re_avgconf - bi_avgconf:>+10.4f}")

    print(f"{'Hits':<25} {results_bi['hits']:>15d} {results_re['hits']:>15d} {results_re['hits'] - results_bi['hits']:>+10d}")

    print(f"\n--- Per-subject Recall@5 ---")
    print(f"{'Subject':<20} {'Bi-encoder':>12} {'+ Reranker':>12} {'Delta':>10}")
    print("-" * 58)
    for subj in sorted(subject_results.keys()):
        sr = subject_results[subj]
        bi_r = sr["bi"]["hits"] / sr["bi"]["total"] if sr["bi"]["total"] else 0
        re_r = sr["re"]["hits"] / sr["re"]["total"] if sr["re"]["total"] else 0
        total = sr["bi"]["total"]
        print(f"{subj:<20} {bi_r:>12.4f} {re_r:>12.4f} {re_r - bi_r:>+10.4f}  (n={total})")


if __name__ == "__main__":
    main()
