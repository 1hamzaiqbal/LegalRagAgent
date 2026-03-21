"""Simple parallel QA evaluation script for the Legal RAG pipeline.

Tests exactly N randomly sampled questions from the BarExam QA dataset
using a fixed random seed for consistent benchmarks across runs.

Usage:
  uv run python eval_qa.py 20                  # Evaluate 20 bar-exam questions
  uv run python eval_qa.py 50 --parallel 5     # Evaluate 50 questions with 5 workers
  uv run python eval_qa.py 10 --suite web      # Evaluate fixed web-search benchmark
"""

import os
import sys
import time
import re

# Add parent directory to sys.path to allow absolute imports from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import build_graph, _get_metrics
from llm_config import get_provider_info
from eval.eval_utils import (
    select_qa_queries, check_mc_correctness, capture_balance, compute_cost,
)
from eval.web_search_suite import select_web_search_queries


def run_single_query(app, q: dict):
    """Run one query through the LangGraph and extract correctness."""
    objective = q["question"]
    choices = q.get("choices", {})
    if choices and any(choices.values()):
        choice_text = "\n".join(f"  ({k}) {v}" for k, v in sorted(choices.items()) if v)
        objective = f"{objective}\n\nAnswer choices:\n{choice_text}"

    initial_state = {
        "agent_metadata": {
            "provider": os.getenv("LLM_PROVIDER", "default"),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "inputs": {"question": objective},
        "run_config": {"max_steps": 7},
        "collections": [],
        "planning_table": [],
        "evidence_store": [],
        "final_answer": "",
        "audit_log": [],
    }

    start = time.time()
    final_state = None
    error = None

    try:
        for output in app.stream(initial_state):
            for node_name, node_state in output.items():
                final_state = node_state
    except Exception as e:
        error = str(e)

    elapsed = time.time() - start
    fs = final_state or {}
    answer = fs.get("final_answer", "")
    is_correct = check_mc_correctness(answer, q.get("correct_answer", ""))

    return {
        "label": q["label"],
        "subject": q["subject"],
        "elapsed_sec": round(elapsed, 1),
        "error": error,
        "llm_calls": 0,
        "input_chars": 0,
        "output_chars": 0,
        "is_correct": is_correct
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
    
    parallel_workers = 1
    if "--parallel" in args:
        idx = args.index("--parallel")
        parallel_workers = int(args[idx + 1])
        args = args[:idx] + args[idx + 2:]

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
    run_log_file = f"logs/eval_qa{suite_suffix}_{provider_name}_{timestamp}.txt"
    
    import re
    completed_queries = {}
    if continue_eval and os.path.exists(run_log_file):
        with open(run_log_file, "r", encoding="utf-8") as f:
            log_content = f.read()
            pattern = re.compile(
                r'\[\d+/\d+\] Evaluating ([\w_]+)\.\.\..*?'
                r'-> Result: (CORRECT|WRONG|ERROR)\s+\|\s+([\d\.]+)s\s+\|\s+(?:TOTAL:\s+\d+\s+calls\s+\((\d+)\s+this\s+q\)\s+\|\s+IN:\s+\d+\s+tokens\s+\((\d+)\s+this\s+q\)\s+\|\s+OUT:\s+\d+\s+tokens\s+\((\d+)\s+this\s+q\)|(\d+)\s+LLM calls)',
                re.DOTALL
            )
            for m in pattern.finditer(log_content):
                label = m.group(1)
                res_str = m.group(2)
                elapsed = float(m.group(3))
                
                if m.group(7): # Old format
                    q_calls = int(m.group(7))
                    q_in, q_out = 0, 0
                else: # New format
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
                    "is_correct": res_str == "CORRECT"
                }

    try:
        file_mode = "a" if continue_eval else "w"
        with open(run_log_file, file_mode, encoding="utf-8") as f:
            if not continue_eval:
                f.write(f"COMMAND RUN: uv run python {' '.join(sys.argv)}\n")
                f.write("=" * 60 + "\n\n")

        class DualLogger:
            def __init__(self, filepath):
                self.terminal = sys.stdout
                self.log = open(filepath, "a", encoding="utf-8")

            def write(self, message):
                self.terminal.write(message)
                self.log.write(message)

            def flush(self):
                self.terminal.flush()
                self.log.flush()

        sys.stdout = DualLogger(run_log_file)
        sys.stderr = sys.stdout
    except Exception as e:
        print(f"Failed to setup file logger: {e}")

    print(f"\n{'='*80}")
    print(f"QA EVALUATION ({n} QUERIES, SUITE={suite})")
    print(f"{'='*80}\n")
    
    # Capture initial balance
    _, initial_totals = capture_balance()

    # Log provider info
    pinfo = get_provider_info()
    print(f"Provider: {pinfo['provider']} | Model: {pinfo['model']}")

    queries = select_qa_queries(n) if suite == "bar" else select_web_search_queries(n)
    app = build_graph()
    
    queries_to_run = [q for q in queries if q["label"] not in completed_queries]
    
    if continue_eval:
        print(f"\nFound --continue flag. Recovered {len(completed_queries)} previously completed queries from {run_log_file}.")
        
    if len(queries_to_run) == 0:
        print("\nAll queries have already been completed!")
    else:
        print(f"\nEvaluating {len(queries_to_run)} questions {'in parallel (' + str(parallel_workers) + ' threads) ' if parallel_workers > 1 else ''}...\n")

    results = list(completed_queries.values())
    eval_start_time = time.time()
    
    def worker_func(i, q):
        try:
            from main import _get_metrics
            start_counts = _get_metrics()
            start_calls = start_counts["count"]
            start_in = start_counts["input_tokens"]
            start_out = start_counts["output_tokens"]

            print(f"[{i+1}/{n}] Evaluating {q['label']}...")
            
            res = run_single_query(app, q)

            mc_tag = "CORRECT" if res["is_correct"] else ("ERROR" if res["error"] else "WRONG")
            
            end_counts = _get_metrics()
            end_calls = end_counts["count"]
            end_in = end_counts["input_tokens"]
            end_out = end_counts["output_tokens"]

            q_calls = end_calls - start_calls
            q_in = end_in - start_in
            q_out = end_out - start_out

            print(f"  -> Result: {mc_tag} | {res['elapsed_sec']}s | TOTAL: {end_calls} calls ({q_calls} this q) | IN: {end_in} tokens ({q_in} this q) | OUT: {end_out} tokens ({q_out} this q)")
            if res["error"]:
                print(f"  -> Error: {res['error']}")
                
            # Override with per-query local metrics
            res["global_llm_calls"] = res["llm_calls"]
            res["global_input_chars"] = res["input_chars"]
            res["global_output_chars"] = res["output_chars"]
            res["llm_calls"] = q_calls
            res["input_chars"] = q_in
            res["output_chars"] = q_out
            
            return res
        finally:
            if hasattr(sys.stdout, 'flush_thread_buffer'):
                sys.stdout.flush_thread_buffer()

    # Run sequentially (parallelization disabled)
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

    print(f"\n\n{'='*80}")
    print("FINAL QA BENCHMARK REPORT")
    print(f"{'='*80}")
    print("--- EXPERIMENT SETTINGS ---")
    print(f"Embedding Model:      {os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')}")
    print(f"LLM Provider/Model:   {os.getenv('LLM_PROVIDER', 'default')} / {pinfo['model']}")
    print(f"Confidence Threshold: {os.getenv('EVAL_CONFIDENCE_THRESHOLD', '0.0')} (cross-encoder logits)")
    print("\n--- STATISTICS ---")
    print(f"Accuracy:             {correct}/{len(queries)} ({accuracy:.1f}%)")
    print(f"Failed to execute:    {errors}")
    print(f"Total time elapsed:   {eval_total_time:.1f}s")
    print(f"Total LLM API Calls:  {total_api_calls}")
    print(f"Avg Input Tokens:     {avg_in_tokens} per query")
    print(f"Avg Output Tokens:    {avg_out_tokens} per query")
    if cost_strs:
        print(f"Total API cost:       {', '.join(cost_strs)}")
    
    # Write a quick breakdown at the end
    print("-" * 80)
    print(f"{'Label':<30} {'Result':<10} {'Time':>6} {'LLM':>4}")
    print("-" * 80)
    for r in sorted(results, key=lambda x: x["label"]):
        status = "PASS" if r["is_correct"] else ("ERR" if r["error"] else "FAIL")
        print(f"{r['label']:<30} {status:<10} {r['elapsed_sec']:>5.1f}s {r['llm_calls']:>4}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
