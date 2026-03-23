"""Retrieval Recall Evaluation — does our retrieval find the gold passage?

Tests whether the gold passage (the passage used to write the answer in
the dataset) appears in the top-k retrieved passages. Measures Recall@k
for our retriever across the barexam QA dataset.

Important scope note:
- This evaluates direct question-to-passage retrieval using the full user
  question as the query.
- It does NOT evaluate the full agent path with planning, query rewriting,
  step decomposition, or judge-driven retries.

No LLM calls — pure retrieval evaluation.

Usage:
  uv run python eval/eval_retrieval_recall.py 50        # Test 50 QA pairs
  uv run python eval/eval_retrieval_recall.py 50 --k 10 # Recall@10
"""

import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_utils import retrieve_documents
from eval.eval_utils import select_qa_queries


def main():
    args = sys.argv[1:]

    k = 5
    if "--k" in args:
        idx = args.index("--k")
        k = int(args[idx + 1])
        args = args[:idx] + args[idx + 2:]

    n = int(args[0]) if args else 25

    os.makedirs("logs", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H")
    log_file = f"logs/eval_retrieval_recall_{timestamp}.txt"

    print(f"\n{'='*80}")
    print(f"RETRIEVAL RECALL EVALUATION (Recall@{k}, {n} queries)")
    print(f"{'='*80}\n")

    queries = select_qa_queries(n)

    hits = 0
    total = 0
    results = []

    for i, q in enumerate(queries):
        gold_idx = str(q.get("gold_idx", ""))
        if not gold_idx or gold_idx == "nan":
            continue

        question = q["question"]
        choices = q.get("choices", {})
        if choices and any(choices.values()):
            choice_text = "\n".join(f"  ({k_}) {v}" for k_, v in sorted(choices.items()) if v)
            question = f"{question}\n\nAnswer choices:\n{choice_text}"

        # Direct retrieval benchmark: raw question -> retrieve_documents()
        t0 = time.time()
        docs = retrieve_documents(question, k=k)
        elapsed = time.time() - t0

        retrieved_ids = [str(doc.metadata.get("idx", "")) for doc in docs]
        found = gold_idx in retrieved_ids
        rank = retrieved_ids.index(gold_idx) + 1 if found else None

        total += 1
        if found:
            hits += 1

        tag = f"HIT@{rank}" if found else "MISS"
        print(f"[{i+1}/{n}] {q['label']:<25} {tag:<8} gold={gold_idx:<20} ({elapsed:.1f}s)")

        results.append({
            "label": q["label"],
            "gold_idx": gold_idx,
            "found": found,
            "rank": rank,
            "retrieved_ids": retrieved_ids,
            "elapsed": elapsed,
        })

    recall = hits / total * 100 if total else 0
    mrr = sum(1.0 / r["rank"] for r in results if r["rank"]) / total if total else 0

    print(f"\n{'='*80}")
    print(f"RETRIEVAL RECALL RESULTS")
    print(f"{'='*80}")
    print(f"Recall@{k}:         {hits}/{total} ({recall:.1f}%)")
    print(f"MRR@{k}:            {mrr:.3f}")
    print(f"Total queries:     {total}")

    # Per-subject breakdown
    from collections import defaultdict
    by_subject = defaultdict(lambda: {"total": 0, "hits": 0})
    for q, r in zip(queries, results):
        subj = q.get("subject", "unknown")
        if subj and str(subj) != "nan":
            by_subject[subj]["total"] += 1
            if r["found"]:
                by_subject[subj]["hits"] += 1

    if by_subject:
        print(f"\nPer-subject Recall@{k}:")
        for subj in sorted(by_subject.keys()):
            s = by_subject[subj]
            rate = s["hits"] / s["total"] * 100 if s["total"] else 0
            print(f"  {subj:<25} {s['hits']}/{s['total']} ({rate:.0f}%)")

    print(f"{'='*80}\n")

    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"RETRIEVAL RECALL | Recall@{k} | {n} queries\n")
        f.write(f"Recall@{k}: {hits}/{total} ({recall:.1f}%)\n")
        f.write(f"MRR@{k}: {mrr:.3f}\n\n")
        for r in results:
            tag = f"HIT@{r['rank']}" if r["found"] else "MISS"
            f.write(f"{r['label']:<30} {tag:<8} gold={r['gold_idx']}\n")
    print(f"Log saved to {log_file}")


if __name__ == "__main__":
    main()
