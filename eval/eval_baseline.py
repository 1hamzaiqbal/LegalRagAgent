"""Simple baseline QA evaluation script.

Tests exactly N randomly sampled questions from the BarExam QA dataset
using a fixed random seed for consistent benchmarks across runs.
This script bypasses LangGraph entirely and asks the LLM the question directly,
acting as a baseline to measure the RAG pipeline against.

Usage:
  uv run python eval_baseline.py 20                 # Evaluate 20 bar-exam questions
  uv run python eval_baseline.py 10 --suite web     # Evaluate fixed web-search benchmark
  uv run python eval_baseline.py 100 --continue     # Resume from log
"""

import os
import sys
import time
import re

# Add parent directory to sys.path to allow absolute imports from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import _llm_call, _get_metrics
from llm_config import get_provider_info
from eval.eval_utils import (
    select_qa_queries, check_mc_correctness, extract_mc_letter,
    capture_balance, compute_cost,
)
from eval.web_search_suite import select_web_search_queries


def run_baseline_query(q: dict):
    """Run one query directly against the LLM (no LangGraph) and extract correctness."""
    objective = q["question"]
    choices = q.get("choices", {})
    if choices and any(choices.values()):
        choice_text = "\n".join(f"  ({k}) {v}" for k, v in sorted(choices.items()) if v)
        objective = f"{objective}\n\nAnswer choices:\n{choice_text}"

    system_prompt = (
        "Answer the following legal question. Reason through it step by step, "
        "then give your final answer as **Answer: (X)**"
    )

    start = time.time()
    error = None
    answer = ""
    
    try:
        # Pass directly to LLM bypassing all RAG Graph harnesses
        answer = _llm_call(system_prompt, objective, label="eval_baseline").strip()
    except Exception as e:
        error = str(e)

    elapsed = time.time() - start
    
    chosen_letter = extract_mc_letter(answer) or "?"
    is_correct = check_mc_correctness(answer, q.get("correct_answer", ""))

    return {
        "label": q["label"],
        "subject": q["subject"],
        "elapsed_sec": round(elapsed, 1),
        "error": error,
        "is_correct": is_correct,
        "chosen_letter": chosen_letter,
        "correct_letter": q.get("correct_answer", ""),
        "llm_response": answer,
        "question": q["question"][:200],
    }


def main():
    # Parse arguments
    args = sys.argv[1:]

    suite = "bar"
    if "--suite" in args:
        idx = args.index("--suite")
        suite = args[idx + 1].strip().lower()
        args = args[:idx] + args[idx + 2:]
    if suite not in {"bar", "web"}:
        raise ValueError(f"Unsupported suite: {suite}. Expected 'bar' or 'web'.")
    
    continue_eval = False
    if "--continue" in args:
        continue_eval = True
        args.remove("--continue")

    n = int(args[0]) if len(args) > 0 else 10
    
    # Setup buffered DualLogger
    provider_name = os.getenv("LLM_PROVIDER", "default").strip().lower()
    os.makedirs("logs", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H")
    suite_suffix = "" if suite == "bar" else f"_{suite}"
    run_log_file = f"logs/eval_baseline{suite_suffix}_{provider_name}_{timestamp}.txt"
    
    completed_queries = {}
    if continue_eval and os.path.exists(run_log_file):
        with open(run_log_file, "r", encoding="utf-8") as f:
            log_content = f.read()
            pattern = re.compile(
                r'\[\d+/\d+\] Evaluating ([\w_]+)\.\.\..*?'
                r'-> Result: (CORRECT|WRONG|ERROR)\s+\|\s+([\d\.]+)s\s+\|\s+TOTAL:\s+\d+\s+calls\s+\((\d+)\s+this\s+q\)\s+\|\s+IN:\s+\d+\s+tokens\s+\((\d+)\s+this\s+q\)\s+\|\s+OUT:\s+\d+\s+tokens\s+\((\d+)\s+this\s+q\)',
                re.DOTALL
            )
            for m in pattern.finditer(log_content):
                label = m.group(1)
                res_str = m.group(2)
                elapsed = float(m.group(3))
                
                q_calls = int(m.group(4))
                q_in = int(m.group(5))
                q_out = int(m.group(6))
                
                completed_queries[label] = {
                    "label": label,
                    "subject": "unknown",
                    "elapsed_sec": elapsed,
                    "error": "error" if res_str == "ERROR" else None,
                    "llm_calls": q_calls,
                    "input_chars": q_in,
                    "output_chars": q_out,
                    "is_correct": res_str == "CORRECT",
                    "chosen_letter": "?",  # We don't save this in the historic regex yet
                    "correct_letter": "?"
                }

    try:
        file_mode = "a" if continue_eval else "w"
        with open(run_log_file, file_mode, encoding="utf-8") as f:
            if not continue_eval:
                f.write(f"COMMAND RUN: uv run python {' '.join(sys.argv)}\n")
                f.write("=" * 60 + "\n\n")
    except Exception as e:
        print(f"Failed to setup file logger: {e}")

    def log_and_print(msg: str):
        print('='*50)
        print(msg)
        try:
            with open(run_log_file, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
                f.flush()
        except Exception:
            pass

    log_and_print(f"\n{'='*80}")
    log_and_print(f"BASELINE LLM EVALUATION ({n} QUERIES, SUITE={suite})")
    log_and_print(f"{'='*80}\n")
    
    # Capture initial balance
    _, initial_totals = capture_balance()

    # Log provider info
    pinfo = get_provider_info()
    log_and_print(f"Provider: {pinfo['provider']} | Model: {pinfo['model']}")

    queries = select_qa_queries(n) if suite == "bar" else select_web_search_queries(n)
    queries_to_run = [q for q in queries if q["label"] not in completed_queries]
    
    if continue_eval:
        log_and_print(f"\nFound --continue flag. Recovered {len(completed_queries)} previously completed queries from {run_log_file}.")
        
    if len(queries_to_run) == 0:
        log_and_print("\nAll queries have already been completed!")
    else:
        log_and_print(f"\nEvaluating {len(queries_to_run)} questions (Sequential Baseline)...\n")

    results = list(completed_queries.values())
    eval_start_time = time.time()
    
    def worker_func(i, q):
        start_counts = _get_metrics()
        start_calls = start_counts["count"]
        start_in = start_counts["input_tokens"]
        start_out = start_counts["output_tokens"]

        log_and_print(f"[{i+1}/{n}] Evaluating {q['label']}...")
        
        res = run_baseline_query(q)

        mc_tag = "CORRECT" if res["is_correct"] else ("ERROR" if res["error"] else "WRONG")
        
        end_counts = _get_metrics()
        end_calls = end_counts["count"]
        end_in = end_counts["input_tokens"]
        end_out = end_counts["output_tokens"]

        q_calls = end_calls - start_calls
        q_in = end_in - start_in
        q_out = end_out - start_out
        
        res["llm_calls"] = q_calls
        res["input_chars"] = q_in
        res["output_chars"] = q_out

        log_and_print(f"  -> Result: {mc_tag} (Ans: {res.get('correct_letter')}, Chose: {res.get('chosen_letter')}) | {res['elapsed_sec']}s | TOTAL: {end_calls} calls ({q_calls} this q) | IN: {end_in} tokens ({q_in} this q) | OUT: {end_out} tokens ({q_out} this q)")
        if res["error"]:
            log_and_print(f"  -> Error: {res['error']}")
            
        return res

    # Run sequentially
    for i, q in enumerate(queries_to_run):
        results.append(worker_func(i, q))

    if continue_eval:
        eval_total_time = sum(r["elapsed_sec"] for r in results)
    else:
        eval_total_time = time.time() - eval_start_time

    # Evaluate Accuracy
    correct = sum(1 for r in results if r["is_correct"])
    errors = sum(1 for r in results if r["error"])
    accuracy = correct / len(queries) * 100 if queries else 0

    # Evaluate Cost
    cost_strs = compute_cost(initial_totals)

    # Evaluate Stats
    total_api_calls = sum(r["llm_calls"] for r in results)
    avg_in_tokens = int(sum(r.get("input_chars", 0) for r in results) / len(queries)) if queries else 0
    avg_out_tokens = int(sum(r.get("output_chars", 0) for r in results) / len(queries)) if queries else 0

    log_and_print(f"\n\n{'='*80}")
    log_and_print("FINAL BASELINE BENCHMARK REPORT")
    log_and_print(f"{'='*80}")
    log_and_print("--- EXPERIMENT SETTINGS ---")
    log_and_print(f"LLM Provider/Model:   {os.getenv('LLM_PROVIDER', 'default')} / {pinfo['model']}")
    log_and_print("\n--- STATISTICS ---")
    log_and_print(f"Accuracy:             {correct}/{len(queries)} ({accuracy:.1f}%)")
    log_and_print(f"Failed to execute:    {errors}")
    log_and_print(f"Total time elapsed:   {eval_total_time:.1f}s")
    log_and_print(f"Total LLM API Calls:  {total_api_calls}")
    log_and_print(f"Avg Input Tokens:     {avg_in_tokens} per query")
    log_and_print(f"Avg Output Tokens:    {avg_out_tokens} per query")
    if cost_strs:
        log_and_print(f"Total API cost:       {', '.join(cost_strs)}")
    
    # Write a quick breakdown at the end
    log_and_print("-" * 80)
    log_and_print(f"{'Label':<30} {'Result':<10} {'Ans':<5} {'Chose':<5} {'Time':>6} {'LLM':>4}")
    log_and_print("-" * 80)
    for r in sorted(results, key=lambda x: x["label"]):
        status = "PASS" if r["is_correct"] else ("ERR" if r["error"] else "FAIL")
        ans = r.get("correct_letter", "?")
        chose = r.get("chosen_letter", "?")
        log_and_print(f"{r['label']:<30} {status:<10} {ans:<5} {chose:<5} {r['elapsed_sec']:>5.1f}s {r['llm_calls']:>4}")
    log_and_print(f"{'='*80}\n")

    # Save detailed JSONL
    import json
    detail_file = run_log_file.replace(".txt", "_detail.jsonl")
    with open(detail_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log_and_print(f"Detail log saved to {detail_file}")


if __name__ == "__main__":
    main()
