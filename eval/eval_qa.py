"""Simple eval harness for the Legal RAG pipeline.

Usage:
    uv run python eval/eval_qa.py 10          # Run 10 questions
    uv run python eval/eval_qa.py 100         # Run 100 questions
"""
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from main import run, _get_metrics, _reset_llm_call_counter, _get_deepseek_balance


def extract_answer(text: str) -> str | None:
    """Extract MC answer letter from LLM response."""
    patterns = [
        r'\*\*(?:Answer|ANSWER)[:\s]*\(?([A-D])\)?',
        r'(?:Answer|ANSWER)[:\s]*\(?([A-D])\)?',
        r'\b([A-D])\b\s*(?:is correct|is the (?:best|correct|strongest))',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1]  # last match (conclusion)
    return None


def load_qa(n: int = 10) -> pd.DataFrame:
    """Load N bar exam QA pairs."""
    qa = pd.read_csv("datasets/barexam_qa/qa/qa.csv")
    return qa.sample(n=min(n, len(qa)), random_state=42).reset_index(drop=True)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    qa = load_qa(n)

    results = []
    correct = 0
    start = time.time()

    print(f"Running {n} questions through full pipeline...\n")

    for i, row in qa.iterrows():
        _reset_llm_call_counter()
        q_start = time.time()

        label = f"qa_{row.get('subject', 'nan')}_{row.get('idx', i)}"
        question = str(row["question"])
        gold = str(row["correct_answer"]).strip().upper()

        try:
            result = run(question, print_output=False)
            answer_text = result.get("final_answer", "")
            predicted = extract_answer(answer_text)
            is_correct = predicted == gold
        except Exception as e:
            answer_text = f"ERROR: {e}"
            predicted = None
            is_correct = False

        elapsed = time.time() - q_start
        metrics = _get_metrics()

        if is_correct:
            correct += 1

        status = "PASS" if is_correct else "FAIL"
        print(f"[{i+1}/{n}] {label:<30} {status:<6} gold={gold} pred={predicted} ({elapsed:.1f}s, {metrics['count']} LLM calls)")

        results.append({
            "label": label,
            "is_correct": is_correct,
            "correct_answer": gold,
            "predicted_answer": predicted,
            "llm_response": answer_text[:2000],
            "elapsed_sec": round(elapsed, 1),
            "llm_calls": metrics["count"],
        })

    total_time = time.time() - start
    accuracy = correct / n * 100

    print(f"\n{'='*60}")
    print(f"RESULTS: {correct}/{n} ({accuracy:.1f}%)")
    print(f"Total time: {total_time:.0f}s ({total_time/n:.1f}s/query)")
    print(f"{'='*60}")

    # Save detail log
    ts = time.strftime("%Y%m%d_%H%M")
    log_path = f"logs/eval_qa_{ts}_detail.jsonl"
    os.makedirs("logs", exist_ok=True)
    with open(log_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Detail log: {log_path}")


if __name__ == "__main__":
    main()
