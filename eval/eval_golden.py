"""Golden Passage Baseline — upper bound for pipeline accuracy.

Gives the LLM the gold passage directly (perfect retrieval) and asks it
to answer the MC question. This measures: "if retrieval were perfect,
how often does the LLM get the answer right?"

Usage:
  uv run python eval/eval_golden.py 25           # Run on 25 questions
  uv run python eval/eval_golden.py 25 --continue # Resume from log
"""

import os
import sys
import time
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import _llm_call, _get_metrics, _get_deepseek_balance
from llm_config import get_provider_info
from eval.eval_utils import (
    select_qa_queries, check_mc_correctness, capture_balance, compute_cost,
)
import pandas as pd


def _load_gold_passages() -> dict:
    """Load gold passage text keyed by idx."""
    qa = pd.read_csv("datasets/barexam_qa/qa/qa.csv")
    return dict(zip(qa["gold_idx"].astype(str), qa["gold_passage"].astype(str)))


SYSTEM_PROMPT = """You are an expert legal assistant. You are given a legal question and a relevant passage from a legal textbook.

Using ONLY the information in the provided passage, answer the multiple-choice question.

Format your response EXACTLY as:
**Answer: (X)**
Reasoning: [Brief explanation grounded in the passage]"""


def run_golden_query(q: dict, gold_passages: dict):
    """Give the LLM the gold passage directly and ask it to answer."""
    gold_text = gold_passages.get(str(q["gold_idx"]), "")
    if not gold_text or gold_text == "nan":
        return {
            "label": q["label"],
            "subject": q["subject"],
            "elapsed_sec": 0,
            "error": "no gold passage",
            "is_correct": False,
        }

    objective = q["question"]
    choices = q.get("choices", {})
    if choices and any(choices.values()):
        choice_text = "\n".join(f"  ({k}) {v}" for k, v in sorted(choices.items()) if v)
        objective = f"{objective}\n\nAnswer choices:\n{choice_text}"

    user_prompt = f"PASSAGE:\n{gold_text}\n\nQUESTION:\n{objective}"

    start = time.time()
    error = None
    answer = ""

    try:
        answer = _llm_call(SYSTEM_PROMPT, user_prompt, label="golden").strip()
    except Exception as e:
        error = str(e)

    elapsed = time.time() - start
    is_correct = check_mc_correctness(answer, q.get("correct_answer", ""))

    return {
        "label": q["label"],
        "subject": q["subject"],
        "elapsed_sec": round(elapsed, 1),
        "error": error,
        "is_correct": is_correct,
        "question": q["question"],
        "correct_answer": q.get("correct_answer", ""),
        "llm_response": answer,
        "gold_passage": gold_text[:500],
    }


def main():
    args = sys.argv[1:]

    continue_eval = "--continue" in args
    if continue_eval:
        args.remove("--continue")
    n = int(args[0]) if args else 10

    provider_name = os.getenv("LLM_PROVIDER", "default").strip().lower()
    os.makedirs("logs", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H")
    log_file = f"logs/eval_golden_{provider_name}_{timestamp}.txt"

    print(f"\n{'='*80}")
    print(f"GOLDEN PASSAGE BASELINE ({n} QUERIES)")
    print(f"{'='*80}\n")

    _, initial_totals = capture_balance()
    pinfo = get_provider_info()
    print(f"Provider: {pinfo['provider']} | Model: {pinfo['model']}")

    queries = select_qa_queries(n)
    gold_passages = _load_gold_passages()
    print(f"Gold passages loaded: {len(gold_passages)}")

    results = []
    for i, q in enumerate(queries):
        print(f"[{i+1}/{n}] {q['label']}...", end=" ")
        res = run_golden_query(q, gold_passages)
        tag = "CORRECT" if res["is_correct"] else ("ERROR" if res["error"] else "WRONG")
        print(f"{tag} ({res['elapsed_sec']}s)")
        results.append(res)

    correct = sum(1 for r in results if r["is_correct"])
    errors = sum(1 for r in results if r["error"])
    accuracy = correct / len(queries) * 100 if queries else 0
    cost_strs = compute_cost(initial_totals)

    print(f"\n{'='*80}")
    print("GOLDEN PASSAGE BASELINE RESULTS")
    print(f"{'='*80}")
    print(f"Accuracy:          {correct}/{len(queries)} ({accuracy:.1f}%)")
    print(f"Errors:            {errors}")
    if cost_strs:
        print(f"Cost:              {', '.join(cost_strs)}")
    print(f"\nThis is the UPPER BOUND — perfect retrieval, 1 LLM call per query.")
    print(f"{'='*80}\n")

    # Save summary log
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"GOLDEN PASSAGE BASELINE | {n} queries | {provider_name}\n")
        f.write(f"Accuracy: {correct}/{len(queries)} ({accuracy:.1f}%)\n\n")
        for r in sorted(results, key=lambda x: x["label"]):
            status = "PASS" if r["is_correct"] else ("ERR" if r["error"] else "FAIL")
            f.write(f"{r['label']:<30} {status}\n")
    print(f"Log saved to {log_file}")

    # Save detailed JSONL with full reasoning
    import json
    detail_file = log_file.replace(".txt", "_detail.jsonl")
    with open(detail_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Detail log saved to {detail_file}")


if __name__ == "__main__":
    main()
