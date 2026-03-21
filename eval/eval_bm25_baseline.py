"""BM25-only RAG Baseline — simple retrieve-and-answer without the agent.

BM25 retrieves top-5 passages, appends them to the question, LLM answers
directly. No planning, no judge, no replanner — just retrieve + answer.
Shows what simple keyword RAG gets you vs the full agentic pipeline.

Usage:
  uv run python eval/eval_bm25_baseline.py 25
"""

import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import _llm_call, _get_metrics
from rag_utils import retrieve_documents
from llm_config import get_provider_info
from eval.eval_utils import (
    select_qa_queries, check_mc_correctness, capture_balance, compute_cost,
)


SYSTEM_PROMPT = """You are an expert legal assistant. You are given a legal question and several retrieved passages from a legal corpus.

Using ONLY the information in the provided passages, answer the multiple-choice question.

Format your response EXACTLY as:
**Answer: (X)**
Reasoning: [Brief explanation grounded in the passages]"""


def run_bm25_query(q: dict):
    """Retrieve passages via hybrid retrieval and answer directly."""
    objective = q["question"]
    choices = q.get("choices", {})
    if choices and any(choices.values()):
        choice_text = "\n".join(f"  ({k}) {v}" for k, v in sorted(choices.items()) if v)
        objective = f"{objective}\n\nAnswer choices:\n{choice_text}"

    start = time.time()
    error = None
    answer = ""

    try:
        # Retrieve using hybrid (BM25 + dense + cross-encoder) — same as pipeline
        docs = retrieve_documents(objective, k=5)
        passages_text = "\n\n".join(
            f"[Passage {i+1}] ({doc.metadata.get('source', 'unknown')})\n{doc.page_content}"
            for i, doc in enumerate(docs)
        )

        user_prompt = f"RETRIEVED PASSAGES:\n{passages_text}\n\nQUESTION:\n{objective}"
        answer = _llm_call(SYSTEM_PROMPT, user_prompt, label="bm25_baseline").strip()
    except Exception as e:
        error = str(e)

    elapsed = time.time() - start
    is_correct = check_mc_correctness(answer, q.get("correct_answer", ""))
    gold_idx = str(q.get("gold_idx", ""))

    # Check if gold passage was in retrieved results
    retrieved_ids = [str(doc.metadata.get("idx", "")) for doc in docs] if not error else []
    gold_retrieved = gold_idx in retrieved_ids

    return {
        "label": q["label"],
        "subject": q["subject"],
        "elapsed_sec": round(elapsed, 1),
        "error": error,
        "is_correct": is_correct,
        "gold_retrieved": gold_retrieved,
        "retrieved_ids": retrieved_ids,
    }


def main():
    args = sys.argv[1:]
    n = int(args[0]) if args else 10

    provider_name = os.getenv("LLM_PROVIDER", "default").strip().lower()
    os.makedirs("logs", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H")
    log_file = f"logs/eval_bm25_baseline_{provider_name}_{timestamp}.txt"

    print(f"\n{'='*80}")
    print(f"BM25 RAG BASELINE ({n} QUERIES)")
    print(f"{'='*80}\n")

    _, initial_totals = capture_balance()
    pinfo = get_provider_info()
    print(f"Provider: {pinfo['provider']} | Model: {pinfo['model']}")

    queries = select_qa_queries(n)

    results = []
    for i, q in enumerate(queries):
        print(f"[{i+1}/{n}] {q['label']}...", end=" ")
        res = run_bm25_query(q)
        tag = "CORRECT" if res["is_correct"] else ("ERROR" if res["error"] else "WRONG")
        gold = "gold=Y" if res.get("gold_retrieved") else "gold=N"
        print(f"{tag} {gold} ({res['elapsed_sec']}s)")
        results.append(res)

    correct = sum(1 for r in results if r["is_correct"])
    errors = sum(1 for r in results if r["error"])
    gold_hit = sum(1 for r in results if r.get("gold_retrieved"))
    accuracy = correct / len(queries) * 100 if queries else 0
    gold_rate = gold_hit / len(queries) * 100 if queries else 0
    cost_strs = compute_cost(initial_totals)

    print(f"\n{'='*80}")
    print("BM25 RAG BASELINE RESULTS")
    print(f"{'='*80}")
    print(f"Accuracy:          {correct}/{len(queries)} ({accuracy:.1f}%)")
    print(f"Gold Recall@5:     {gold_hit}/{len(queries)} ({gold_rate:.1f}%)")
    print(f"Errors:            {errors}")
    if cost_strs:
        print(f"Cost:              {', '.join(cost_strs)}")
    print(f"\nThis is SIMPLE RAG — 1 retrieval + 1 LLM call per query, no agent.")
    print(f"{'='*80}\n")

    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"BM25 RAG BASELINE | {n} queries | {provider_name}\n")
        f.write(f"Accuracy: {correct}/{len(queries)} ({accuracy:.1f}%)\n")
        f.write(f"Gold Recall@5: {gold_hit}/{len(queries)} ({gold_rate:.1f}%)\n\n")
        for r in sorted(results, key=lambda x: x["label"]):
            status = "PASS" if r["is_correct"] else ("ERR" if r["error"] else "FAIL")
            gold = "gold=Y" if r.get("gold_retrieved") else "gold=N"
            f.write(f"{r['label']:<30} {status:<6} {gold}\n")
    print(f"Log saved to {log_file}")


if __name__ == "__main__":
    main()
