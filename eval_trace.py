"""Traced experiment: run a few questions through the pipeline with detailed logging.

Shows the full journey of each question:
  1. Classification decision
  2. Plan generated
  3. Query rewrite (primary + alternatives)
  4. Retrieved passages (with source, idx, and preview)
  5. Gold passage check (if applicable)
  6. Synthesized answer
  7. Verification result
  8. MC correctness check (if applicable)

Usage:
  uv run python eval_trace.py                  # Run all trace queries
  uv run python eval_trace.py 3                # Run first N queries only
  uv run python eval_trace.py --query "..."    # Run a custom query
"""

import json
import os
import re
import sys
import time
import pandas as pd

os.environ.setdefault("SKIP_INJECTION_CHECK", "1")

from main import (
    build_graph, _reset_llm_call_counter, _llm_call_counter,
    skill_query_rewrite,
)
from llm_config import get_provider_info
from rag_utils import (
    retrieve_documents, retrieve_documents_multi_query,
    compute_confidence, get_vectorstore,
)
from eval_comprehensive import _check_mc_correctness


def _load_qa_with_gold():
    """Load QA pairs that have gold passages in the current store."""
    qa = pd.read_csv("datasets/barexam_qa/qa/qa.csv")
    vs = get_vectorstore()
    count = vs._collection.count()
    passages = pd.read_csv("datasets/barexam_qa/barexam_qa_train.csv", nrows=count)
    passage_ids = set(passages["idx"].tolist())
    qa_in = qa[qa["gold_idx"].isin(passage_ids)].copy()

    def full_question(row):
        prompt = str(row["prompt"]) if pd.notna(row["prompt"]) else ""
        q = str(row["question"])
        return (prompt + " " + q).strip()

    qa_in["full_q"] = qa_in.apply(full_question, axis=1)
    return qa_in


def _get_gold_passage(gold_idx: str) -> str:
    """Retrieve the gold passage text from the CSV."""
    vs = get_vectorstore()
    count = vs._collection.count()
    passages = pd.read_csv("datasets/barexam_qa/barexam_qa_train.csv", nrows=count)
    match = passages[passages["idx"] == gold_idx]
    if len(match) > 0:
        return str(match.iloc[0]["text"])
    return ""


def trace_retrieval_only(question: str, gold_idx: str = ""):
    """Trace just the retrieval pipeline for a single question."""
    print(f"\n{'─'*80}")
    print(f"RETRIEVAL TRACE")
    print(f"{'─'*80}")
    print(f"Question: {question[:120]}...")

    # 1. Raw retrieval (no rewrite)
    print(f"\n  ── Stage 1: Raw bi-encoder + cross-encoder (no rewrite) ──")
    t0 = time.time()
    raw_docs = retrieve_documents(question, k=5)
    raw_time = time.time() - t0
    raw_ids = [doc.metadata.get("idx", "") for doc in raw_docs]
    raw_conf = compute_confidence(question, raw_docs)
    gold_in_raw = gold_idx in raw_ids if gold_idx else None

    for i, doc in enumerate(raw_docs):
        idx = doc.metadata.get("idx", "")
        src = doc.metadata.get("source", "")
        gold_marker = " ★ GOLD" if idx == gold_idx else ""
        print(f"    [{i+1}] {idx} ({src}) — {doc.page_content[:100]}...{gold_marker}")
    print(f"    Confidence: {raw_conf:.3f} | Gold found: {gold_in_raw} | Time: {raw_time:.1f}s")

    # 2. Query rewrite
    print(f"\n  ── Stage 2: Query rewrite ──")
    rewrite = skill_query_rewrite(question)
    print(f"    Primary: {rewrite['primary'][:100]}...")
    for j, alt in enumerate(rewrite.get("alternatives", [])):
        print(f"    Alt {j+1}:    {alt[:100]}...")

    # 3. Multi-query retrieval
    print(f"\n  ── Stage 3: Multi-query retrieval ──")
    all_queries = [rewrite["primary"]] + rewrite.get("alternatives", [])
    t0 = time.time()
    mq_docs = retrieve_documents_multi_query(all_queries, k=5)
    mq_time = time.time() - t0
    mq_ids = [doc.metadata.get("idx", "") for doc in mq_docs]
    mq_conf = compute_confidence(rewrite["primary"], mq_docs)
    gold_in_mq = gold_idx in mq_ids if gold_idx else None

    for i, doc in enumerate(mq_docs):
        idx = doc.metadata.get("idx", "")
        src = doc.metadata.get("source", "")
        gold_marker = " ★ GOLD" if idx == gold_idx else ""
        print(f"    [{i+1}] {idx} ({src}) — {doc.page_content[:100]}...{gold_marker}")
    print(f"    Confidence: {mq_conf:.3f} | Gold found: {gold_in_mq} | Time: {mq_time:.1f}s")

    # 4. Gold passage comparison
    if gold_idx:
        gold_text = _get_gold_passage(gold_idx)
        if gold_text:
            print(f"\n  ── Gold passage ({gold_idx}) ──")
            print(f"    {gold_text[:200]}...")

    return {
        "raw_recall": gold_in_raw,
        "mq_recall": gold_in_mq,
        "raw_conf": raw_conf,
        "mq_conf": mq_conf,
        "rewrite": rewrite,
    }


def trace_full_pipeline(question: str, gold_idx: str = "", correct_answer: str = "",
                         choices: dict = None):
    """Run the full pipeline on a single question with detailed tracing."""
    print(f"\n{'='*80}")
    print(f"FULL PIPELINE TRACE")
    print(f"{'='*80}")
    print(f"Question: {question[:200]}...")
    if correct_answer:
        correct_text = choices.get(correct_answer, "") if choices else ""
        print(f"Correct answer: {correct_answer}" + (f" — {correct_text[:80]}" if correct_text else ""))

    # Run retrieval trace first (uses raw question, no choices — tests pure retrieval)
    ret_trace = trace_retrieval_only(question, gold_idx)

    # Build the pipeline objective: append MC choices so the LLM can select among them
    objective = question
    if choices and any(choices.values()):
        choice_text = "\n".join(f"  ({k}) {v}" for k, v in sorted(choices.items()) if v)
        objective = f"{question}\n\nAnswer choices:\n{choice_text}"

    # Run full pipeline
    print(f"\n{'─'*80}")
    print(f"PIPELINE EXECUTION")
    print(f"{'─'*80}")

    _reset_llm_call_counter()
    app = build_graph()
    initial_state = {
        "global_objective": objective,
        "planning_table": [],
        "query_type": "",
        "final_cited_answer": "",
        "accumulated_context": [],
        "iteration_count": 0,
        "injection_check": {},
        "verification_result": {},
        "verification_retries": 0,
        "memory_hit": {},
        "run_metrics": {},
    }

    t0 = time.time()
    final_state = None
    try:
        for output in app.stream(initial_state):
            for node_name, node_state in output.items():
                final_state = node_state
    except Exception as e:
        print(f"\nERROR: {e}")

    elapsed = time.time() - t0
    answer = final_state.get("final_cited_answer", "") if final_state else ""

    print(f"\n{'─'*80}")
    print(f"FINAL ANSWER ({len(answer)} chars, {elapsed:.1f}s)")
    print(f"{'─'*80}")
    print(answer[:500] if answer else "(no answer)")

    # MC correctness check
    mc_result = None
    if correct_answer and choices:
        mc_result = _check_mc_correctness(answer, correct_answer, choices)
        status = "CORRECT ✓" if mc_result["correct"] else "WRONG ✗"
        print(f"\n  MC Check: {status} (method: {mc_result['method']})")
        print(f"  Details: {mc_result['details']}")

    # Summary
    metrics = final_state.get("run_metrics", {}) if final_state else {}
    print(f"\n{'─'*80}")
    print(f"TRACE SUMMARY")
    print(f"{'─'*80}")
    print(f"  Time: {elapsed:.1f}s | LLM calls: {metrics.get('total_llm_calls', '?')}")
    print(f"  Query type: {final_state.get('query_type', '?') if final_state else '?'}")
    print(f"  Steps: {metrics.get('steps_completed', 0)} completed, {metrics.get('steps_failed', 0)} failed")
    print(f"  Verified: {final_state.get('verification_result', {}).get('is_verified', '?') if final_state else '?'}")
    print(f"  Raw retrieval found gold: {ret_trace['raw_recall']}")
    print(f"  Multi-query found gold: {ret_trace['mq_recall']}")
    print(f"  MC correct: {mc_result['correct'] if mc_result else 'n/a'}")

    return {
        **ret_trace,
        "answer": answer,
        "elapsed": elapsed,
        "mc_result": mc_result,
        "metrics": metrics,
    }


def select_trace_queries(n: int = 8):
    """Select a diverse set of questions for tracing."""
    qa = _load_qa_with_gold()
    queries = []

    # Pick 1 per subject (shortest = easiest)
    for subj in ["TORTS", "CONTRACTS", "CRIM. LAW", "EVIDENCE", "CONST. LAW", "REAL PROP."]:
        subj_qs = qa[qa["subject"] == subj].sort_values("full_q", key=lambda x: x.str.len())
        if len(subj_qs) > 0:
            row = subj_qs.iloc[0]
            queries.append({
                "label": f"trace_{subj.lower().replace(' ', '').replace('.', '')}",
                "question": row["full_q"],
                "gold_idx": row["gold_idx"],
                "correct_answer": row["answer"],
                "choices": {
                    "A": str(row["choice_a"]) if pd.notna(row["choice_a"]) else "",
                    "B": str(row["choice_b"]) if pd.notna(row["choice_b"]) else "",
                    "C": str(row["choice_c"]) if pd.notna(row["choice_c"]) else "",
                    "D": str(row["choice_d"]) if pd.notna(row["choice_d"]) else "",
                },
                "subject": subj,
            })

    # Add 1 multi-hop
    queries.append({
        "label": "trace_multihop",
        "question": (
            "A consumer is injured by a defective product. Under what theories can the "
            "manufacturer be held liable, and what defenses are available?"
        ),
        "gold_idx": "",
        "correct_answer": "",
        "choices": {},
        "subject": "MULTI_HOP",
    })

    # Add 1 out-of-corpus
    queries.append({
        "label": "trace_oof",
        "question": "What are the requirements for obtaining asylum in the United States?",
        "gold_idx": "",
        "correct_answer": "",
        "choices": {},
        "subject": "OUT_OF_CORPUS",
    })

    return queries[:n]


def main():
    pinfo = get_provider_info()
    print(f"Provider: {pinfo['provider']} | Model: {pinfo['model']}")
    vs = get_vectorstore()
    print(f"Corpus: {vs._collection.count()} passages")
    print(f"Embedding: {os.getenv('EMBEDDING_MODEL', 'Alibaba-NLP/gte-large-en-v1.5')}")

    # Parse args
    if len(sys.argv) > 1 and sys.argv[1] == "--query":
        query = " ".join(sys.argv[2:])
        trace_full_pipeline(query)
        return

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    queries = select_trace_queries(n)

    print(f"\nTracing {len(queries)} queries...\n")

    results = []
    for i, q in enumerate(queries):
        print(f"\n{'#'*80}")
        print(f"# [{i+1}/{len(queries)}] {q['label']} ({q['subject']})")
        print(f"{'#'*80}")

        result = trace_full_pipeline(
            question=q["question"],
            gold_idx=q["gold_idx"],
            correct_answer=q["correct_answer"],
            choices=q["choices"],
        )
        result["label"] = q["label"]
        result["subject"] = q["subject"]
        results.append(result)

    # Final summary table
    print(f"\n\n{'='*80}")
    print(f"EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    print(f"{'Label':<25} {'Subj':<12} {'RawGold':>8} {'MQGold':>8} {'MC':>8} {'Time':>6} {'LLM':>4}")
    print("-" * 80)
    for r in results:
        raw_g = "Y" if r["raw_recall"] else ("N" if r["raw_recall"] is not None else ".")
        mq_g = "Y" if r["mq_recall"] else ("N" if r["mq_recall"] is not None else ".")
        mc = "Y" if r.get("mc_result", {}) and r["mc_result"]["correct"] else (
            "N" if r.get("mc_result") else "."
        )
        t = f"{r['elapsed']:.0f}s"
        llm = str(r.get("metrics", {}).get("total_llm_calls", "?"))
        print(f"{r['label']:<25} {r['subject']:<12} {raw_g:>8} {mq_g:>8} {mc:>8} {t:>6} {llm:>4}")


if __name__ == "__main__":
    main()
