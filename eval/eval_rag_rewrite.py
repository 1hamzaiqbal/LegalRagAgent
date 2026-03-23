"""RAG with Query Rewriting Baseline — isolates the rewriter's contribution.

Same as simple RAG but adds the query rewriter step before retrieval:
  question → LLM rewrite to doctrine-level queries → multi-query retrieve → answer

This tests: "does query rewriting + retrieval beat LLM-only (85%)?"
2 LLM calls per query (rewrite + answer). No planning, no judge, no replanner.

Usage:
  uv run python eval/eval_rag_rewrite.py 100
"""

import os
import sys
import time
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import _llm_call, _parse_json, load_skill
from rag_utils import retrieve_documents_multi_query
from llm_config import get_provider_info
from eval.eval_utils import (
    select_qa_queries, check_mc_correctness, extract_mc_letter,
    capture_balance, compute_cost,
)


ANSWER_PROMPT = """Answer the following legal question using the retrieved passages as your primary source. You may also apply established legal doctrine where the passages are insufficient.

Reason through it step by step, then give your final answer as **Answer: (X)**"""


def rewrite_query(question: str) -> list[str]:
    """Use the query rewriter skill to transform a raw question into retrieval queries."""
    rewrite_prompt = (
        f"Original legal research question: {question}\n\n"
        f"Sub-question: {question}\n"
        f"Authority target: legal doctrine\n"
        f"Retrieval hints: none"
    )
    raw = _llm_call(load_skill("query_rewriter"), rewrite_prompt, label="rewrite")
    parsed = _parse_json(raw)

    if parsed and "primary" in parsed:
        return [parsed["primary"]] + parsed.get("alternatives", [])
    return [question]


def run_rewrite_query(q: dict):
    """Rewrite → multi-query retrieve → answer."""
    objective = q["question"]
    choices = q.get("choices", {})
    if choices and any(choices.values()):
        choice_text = "\n".join(f"  ({k}) {v}" for k, v in sorted(choices.items()) if v)
        objective = f"{objective}\n\nAnswer choices:\n{choice_text}"

    start = time.time()
    error = None
    answer = ""
    queries = []
    docs = []

    try:
        # Step 1: Rewrite to doctrine-level queries
        queries = rewrite_query(q["question"])

        # Step 2: Multi-query retrieval
        docs = retrieve_documents_multi_query(queries, k=5)
        passages_text = "\n\n".join(
            f"[Passage {i+1}] ({doc.metadata.get('source', 'unknown')})\n{doc.page_content}"
            for i, doc in enumerate(docs)
        )

        # Step 3: Answer with passages + question
        user_prompt = f"RETRIEVED PASSAGES:\n{passages_text}\n\nQUESTION:\n{objective}"
        answer = _llm_call(ANSWER_PROMPT, user_prompt, label="rag_rewrite").strip()
    except Exception as e:
        error = str(e)

    elapsed = time.time() - start
    is_correct = check_mc_correctness(answer, q.get("correct_answer", ""))
    gold_idx = str(q.get("gold_idx", ""))

    retrieved_ids = [str(doc.metadata.get("idx", "")) for doc in docs] if docs else []
    gold_retrieved = gold_idx in retrieved_ids

    return {
        "label": q["label"],
        "subject": q["subject"],
        "elapsed_sec": round(elapsed, 1),
        "error": error,
        "is_correct": is_correct,
        "correct_answer": q.get("correct_answer", ""),
        "chosen_letter": extract_mc_letter(answer) or "?",
        "gold_retrieved": gold_retrieved,
        "retrieved_ids": retrieved_ids,
        "rewrite_queries": queries,
        "llm_response": answer,
        "question": q["question"][:300],
    }


def main():
    args = sys.argv[1:]
    n = int(args[0]) if args else 10

    provider_name = os.getenv("LLM_PROVIDER", "default").strip().lower()
    os.makedirs("logs", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H")
    log_file = f"logs/eval_rag_rewrite_{provider_name}_{timestamp}.txt"

    print(f"\n{'='*80}")
    print(f"RAG WITH QUERY REWRITE BASELINE ({n} QUERIES)")
    print(f"{'='*80}\n")

    _, initial_totals = capture_balance()
    pinfo = get_provider_info()
    print(f"Provider: {pinfo['provider']} | Model: {pinfo['model']}")

    queries = select_qa_queries(n)

    results = []
    for i, q in enumerate(queries):
        print(f"[{i+1}/{n}] {q['label']}...", end=" ", flush=True)
        res = run_rewrite_query(q)
        tag = "CORRECT" if res["is_correct"] else ("ERROR" if res["error"] else "WRONG")
        gold = "gold=Y" if res.get("gold_retrieved") else "gold=N"
        print(f"{tag} {gold} ({res['elapsed_sec']}s)")
        if res.get("rewrite_queries"):
            print(f"    Primary: {res['rewrite_queries'][0][:80]}")
        results.append(res)

    correct = sum(1 for r in results if r["is_correct"])
    errors = sum(1 for r in results if r["error"])
    gold_hit = sum(1 for r in results if r.get("gold_retrieved"))
    accuracy = correct / len(queries) * 100 if queries else 0
    gold_rate = gold_hit / len(queries) * 100 if queries else 0
    cost_strs = compute_cost(initial_totals)

    avg_time = sum(r["elapsed_sec"] for r in results) / len(results) if results else 0

    print(f"\n{'='*80}")
    print("RAG WITH QUERY REWRITE RESULTS")
    print(f"{'='*80}")
    print(f"Accuracy:          {correct}/{len(queries)} ({accuracy:.1f}%)")
    print(f"Gold Recall@5:     {gold_hit}/{len(queries)} ({gold_rate:.1f}%)")
    print(f"Errors:            {errors}")
    print(f"Avg time/query:    {avg_time:.1f}s")
    if cost_strs:
        print(f"Cost:              {', '.join(cost_strs)}")
    print(f"\nThis is RAG + QUERY REWRITE — 2 LLM calls per query (rewrite + answer), no agent.")
    print(f"{'='*80}\n")

    # Save summary
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"RAG WITH QUERY REWRITE | {n} queries | {provider_name}\n")
        f.write(f"Accuracy: {correct}/{len(queries)} ({accuracy:.1f}%)\n")
        f.write(f"Gold Recall@5: {gold_hit}/{len(queries)} ({gold_rate:.1f}%)\n")
        f.write(f"Avg time/query: {avg_time:.1f}s\n\n")
        for r in sorted(results, key=lambda x: x["label"]):
            status = "PASS" if r["is_correct"] else ("ERR" if r["error"] else "FAIL")
            gold = "gold=Y" if r.get("gold_retrieved") else "gold=N"
            f.write(f"{r['label']:<30} {status:<6} {gold}  primary: {r['rewrite_queries'][0][:60] if r.get('rewrite_queries') else 'n/a'}\n")
    print(f"Log saved to {log_file}")

    # Save detailed JSONL
    detail_file = log_file.replace(".txt", "_detail.jsonl")
    with open(detail_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Detail log saved to {detail_file}")


if __name__ == "__main__":
    main()
