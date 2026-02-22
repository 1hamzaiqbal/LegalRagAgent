"""MuSiQue multi-hop QA evaluation for the Legal RAG agent.

Two-phase evaluation using the bdsaglam/musique dataset:
  Phase 1 — Retrieval quality: Recall@5, MRR against gold supporting paragraphs
  Phase 2 — Full pipeline: Run agent on sampled questions, measure answer correctness

Usage:
  uv run python eval_musique.py                    # Both phases
  uv run python eval_musique.py retrieval           # Phase 1 only
  uv run python eval_musique.py pipeline            # Phase 2 only
  uv run python eval_musique.py pipeline 5          # Phase 2, first N queries only
"""

import json
import os
import re
import sys
import time
import traceback
from collections import Counter, defaultdict

from datasets import load_dataset
from tqdm import tqdm

from load_corpus_musique import MUSIQUE_COLLECTION

# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Normalize text for comparison: lowercase, strip, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _f1_token_overlap(prediction: str, gold: str) -> float:
    """Compute token-level F1 between prediction and gold answer."""
    pred_tokens = _normalize(prediction).split()
    gold_tokens = _normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _exact_match(prediction: str, gold: str, aliases: list = None) -> bool:
    """Check if prediction matches gold answer or any alias (normalized)."""
    pred_norm = _normalize(prediction)
    targets = [gold] + (aliases or [])
    return any(_normalize(t) in pred_norm for t in targets)


def _get_hop_count(example: dict) -> int:
    """Determine hop count from question decomposition."""
    decomp = example.get("question_decomposition", [])
    return len(decomp) if decomp else 2


# ────────────────────────────────────────────────────────────────────────────
# PHASE 1: Retrieval Quality
# ────────────────────────────────────────────────────────────────────────────

def phase1_retrieval(k: int = 5, max_queries: int = None):
    """Evaluate retrieval quality on MuSiQue validation questions."""
    from rag_utils import retrieve_documents, compute_confidence, get_vectorstore

    print(f"\n{'='*80}")
    print(f"PHASE 1: MUSIQUE RETRIEVAL QUALITY (Recall@{k}, MRR)")
    print(f"{'='*80}\n")

    # Check collection exists
    vs = get_vectorstore(MUSIQUE_COLLECTION)
    doc_count = vs._collection.count()
    if doc_count == 0:
        print("ERROR: MuSiQue collection is empty. Run: uv run python load_corpus_musique.py")
        return None

    print(f"MuSiQue collection: {doc_count} paragraphs")

    # Load dataset
    ds = load_dataset("bdsaglam/musique", "answerable", split="validation")
    if max_queries:
        ds = ds.select(range(min(max_queries, len(ds))))

    print(f"Evaluating {len(ds)} questions...\n")

    hits_at_k = 0
    mrr_sum = 0.0
    conf_sum = 0.0
    hop_stats = defaultdict(lambda: {"hits": 0, "total": 0, "mrr_sum": 0.0})
    sub_q_hits = 0
    sub_q_total = 0

    for example in tqdm(ds, desc="Retrieval eval"):
        question = example["question"]
        hops = _get_hop_count(example)

        # Get gold supporting paragraph titles
        gold_titles = set()
        for para in example.get("paragraphs", []):
            if para.get("is_supporting", False):
                gold_titles.add(para["title"])

        # Retrieve
        docs = retrieve_documents(question, k=k, vectorstore=vs)
        retrieved_titles = {doc.metadata.get("title", "") for doc in docs}
        conf = compute_confidence(question, docs)
        conf_sum += conf

        hop_stats[hops]["total"] += 1

        # Recall@K: any gold paragraph retrieved?
        if gold_titles & retrieved_titles:
            hits_at_k += 1
            hop_stats[hops]["hits"] += 1

        # MRR: rank of first gold paragraph
        for rank, doc in enumerate(docs, 1):
            if doc.metadata.get("title", "") in gold_titles:
                mrr_sum += 1.0 / rank
                hop_stats[hops]["mrr_sum"] += 1.0 / rank
                break

        # Sub-question retrieval (oracle baseline)
        decomp = example.get("question_decomposition", [])
        for step in decomp:
            sub_q = step.get("question", "")
            gold_para_id = step.get("paragraph_support_idx")
            if not sub_q or gold_para_id is None:
                continue

            # Find the gold paragraph's title
            paragraphs = example.get("paragraphs", [])
            if gold_para_id < len(paragraphs):
                gold_title = paragraphs[gold_para_id]["title"]
                sub_docs = retrieve_documents(sub_q, k=k, vectorstore=vs)
                sub_titles = {d.metadata.get("title", "") for d in sub_docs}
                sub_q_total += 1
                if gold_title in sub_titles:
                    sub_q_hits += 1

    total = len(ds)
    recall = hits_at_k / total if total > 0 else 0
    mrr = mrr_sum / total if total > 0 else 0
    avg_conf = conf_sum / total if total > 0 else 0

    # Print results
    print(f"\n{'─'*60}")
    print(f"OVERALL RETRIEVAL RESULTS")
    print(f"{'─'*60}")
    print(f"  Total queries:     {total}")
    print(f"  Recall@{k}:         {recall:.4f} ({hits_at_k}/{total})")
    print(f"  MRR:               {mrr:.4f}")
    print(f"  Avg confidence:    {avg_conf:.4f}")

    if sub_q_total > 0:
        print(f"\n  Sub-question retrieval (oracle):")
        print(f"    Recall@{k}:       {sub_q_hits/sub_q_total:.4f} ({sub_q_hits}/{sub_q_total})")

    print(f"\n{'─'*60}")
    print(f"BY HOP COUNT")
    print(f"{'─'*60}")
    print(f"  {'Hops':<8} {'Recall@5':>10} {'MRR':>10} {'Count':>8}")
    print(f"  {'-'*38}")
    for hops in sorted(hop_stats.keys()):
        s = hop_stats[hops]
        r = s["hits"] / s["total"] if s["total"] > 0 else 0
        m = s["mrr_sum"] / s["total"] if s["total"] > 0 else 0
        print(f"  {hops:<8} {r:>10.4f} {m:>10.4f} {s['total']:>8}")

    return {"recall_at_k": recall, "mrr": mrr, "avg_confidence": avg_conf, "total": total}


# ────────────────────────────────────────────────────────────────────────────
# PHASE 2: Full Pipeline Evaluation
# ────────────────────────────────────────────────────────────────────────────

def _select_pipeline_queries(max_queries: int = None):
    """Select stratified MuSiQue questions by hop count."""
    ds = load_dataset("bdsaglam/musique", "answerable", split="validation")

    # Group by hop count
    by_hops = defaultdict(list)
    for i, example in enumerate(ds):
        hops = _get_hop_count(example)
        by_hops[hops].append((i, example))

    # Stratified sample: up to N/3 per hop count (2, 3, 4-hop)
    queries = []
    per_hop = max(1, (max_queries or 15) // max(len(by_hops), 1))

    for hops in sorted(by_hops.keys()):
        examples = by_hops[hops][:per_hop]
        for idx, example in examples:
            queries.append({
                "ds_idx": idx,
                "question": example["question"],
                "gold_answer": example["answer"],
                "answer_aliases": example.get("answer_aliases", []),
                "hops": hops,
                "question_decomposition": example.get("question_decomposition", []),
                "paragraphs": example.get("paragraphs", []),
            })

    if max_queries:
        queries = queries[:max_queries]

    return queries


def _grade_musique(result: dict) -> str:
    """Assign a letter grade based on answer correctness and quality signals."""
    if result["error"]:
        return "F"
    if result["answer_len"] < 20:
        return "F"

    score = 0

    # Answer correctness (most important)
    if result["exact_match"]:
        score += 4
    elif result["f1_score"] >= 0.5:
        score += 3
    elif result["f1_score"] >= 0.2:
        score += 1

    # Quality signals
    if result["answer_len"] >= 100:
        score += 1
    if result["is_verified"] is True:
        score += 2
    if result["llm_calls"] <= 12:
        score += 1
    if result["parse_failures"] == 0:
        score += 1

    if score >= 7:
        return "A"
    elif score >= 5:
        return "B"
    elif score >= 3:
        return "C"
    elif score >= 1:
        return "D"
    return "F"


def phase2_pipeline(max_queries: int = None):
    """Run the full pipeline on MuSiQue questions and evaluate end-to-end."""
    from main import build_graph, _reset_llm_call_counter, _llm_call_counter, _parse_failure_count

    # Set environment to skip injection check for eval efficiency
    os.environ["SKIP_INJECTION_CHECK"] = "1"

    print(f"\n{'='*80}")
    print(f"PHASE 2: MUSIQUE FULL PIPELINE EVALUATION")
    print(f"{'='*80}\n")

    queries = _select_pipeline_queries(max_queries)
    print(f"Total queries: {len(queries)}")
    hop_counts = Counter(q["hops"] for q in queries)
    for hops, count in sorted(hop_counts.items()):
        print(f"  {hops}-hop: {count}")

    app = build_graph()
    results = []
    total_start = time.time()

    for i, query in enumerate(queries):
        print(f"\n{'#'*80}")
        print(f"# [{query['hops']}-HOP] QUERY {i+1}/{len(queries)}")
        print(f"# {query['question'][:120]}")
        print(f"# Gold: {query['gold_answer']}")
        print(f"{'#'*80}")

        _reset_llm_call_counter()
        initial_state = {
            "global_objective": query["question"],
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

        start = time.time()
        final_state = None
        error = None
        try:
            for output in app.stream(initial_state):
                for node_name, node_state in output.items():
                    final_state = node_state
        except Exception as e:
            error = str(e)
            traceback.print_exc()

        elapsed = time.time() - start
        fs = final_state or {}
        metrics = fs.get("run_metrics", {})
        verification = fs.get("verification_result", {})
        answer = fs.get("final_cited_answer", "")

        # Compute answer correctness
        em = _exact_match(answer, query["gold_answer"], query["answer_aliases"])
        f1 = _f1_token_overlap(answer, query["gold_answer"])

        # Check decomposition similarity
        agent_steps = [s.question for s in fs.get("planning_table", [])]
        gold_decomp = [s.get("question", "") for s in query["question_decomposition"]]

        result = {
            "ds_idx": query["ds_idx"],
            "hops": query["hops"],
            "question": query["question"],
            "gold_answer": query["gold_answer"],
            "elapsed_sec": round(elapsed, 1),
            "error": error,
            "llm_calls": metrics.get("total_llm_calls", 0),
            "parse_failures": metrics.get("parse_failures", 0),
            "iterations": fs.get("iteration_count", 0),
            "query_type": fs.get("query_type", ""),
            "answer_len": len(answer),
            "answer_preview": (answer[:300] + "...") if len(answer) > 300 else answer,
            "exact_match": em,
            "f1_score": round(f1, 4),
            "is_verified": verification.get("is_verified", None),
            "verification_retries": fs.get("verification_retries", 0),
            "memory_hit": fs.get("memory_hit", {}).get("found", False),
            "steps_completed": metrics.get("steps_completed", 0),
            "steps_failed": metrics.get("steps_failed", 0),
            "agent_steps": agent_steps,
            "gold_decomp": gold_decomp,
        }
        result["grade"] = _grade_musique(result)
        results.append(result)

        print(f"\n>>> {result['grade']} | {result['elapsed_sec']}s | "
              f"EM: {result['exact_match']} | F1: {result['f1_score']:.3f} | "
              f"LLM: {result['llm_calls']} | Ans: {result['answer_len']}ch")

    total_elapsed = time.time() - total_start
    _print_results(results, total_elapsed)
    return results


def _print_results(results, total_elapsed):
    """Print detailed pipeline evaluation results."""
    print(f"\n\n{'='*100}")
    print("DETAILED RESULTS")
    print(f"{'='*100}")
    hdr = (f"{'Idx':<6} {'Hops':<6} {'Grd':>3} {'EM':>4} {'F1':>6} "
           f"{'Time':>6} {'LLM':>4} {'Ans':>5} {'Vrfy':>4} {'Err':>3}")
    print(hdr)
    print("-" * 100)
    for r in results:
        v = "Y" if r["is_verified"] is True else ("N" if r["is_verified"] is False else "-")
        err = "Y" if r["error"] else "."
        em = "Y" if r["exact_match"] else "N"
        print(f"{r['ds_idx']:<6} {r['hops']:<6} {r['grade']:>3} {em:>4} "
              f"{r['f1_score']:>6.3f} {r['elapsed_sec']:>5.1f}s {r['llm_calls']:>4} "
              f"{r['answer_len']:>5} {v:>4} {err:>3}")

    # By hop count
    print(f"\n{'='*80}")
    print("BY HOP COUNT")
    print(f"{'='*80}")
    by_hops = defaultdict(list)
    for r in results:
        by_hops[r["hops"]].append(r)

    for hops in sorted(by_hops.keys()):
        hr = by_hops[hops]
        grades = [r["grade"] for r in hr]
        em_rate = sum(1 for r in hr if r["exact_match"]) / len(hr)
        avg_f1 = sum(r["f1_score"] for r in hr) / len(hr)
        print(f"\n  {hops}-HOP ({len(hr)} queries)")
        print(f"    Grades:    {' '.join(grades)}")
        print(f"    EM rate:   {em_rate:.0%}")
        print(f"    Avg F1:    {avg_f1:.3f}")

    # Overall
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")
    all_grades = [r["grade"] for r in results]
    grade_dist = {g: all_grades.count(g) for g in ["A", "B", "C", "D", "F"]}
    em_total = sum(1 for r in results if r["exact_match"])
    avg_f1 = sum(r["f1_score"] for r in results) / len(results) if results else 0

    print(f"  Total queries:      {len(results)}")
    print(f"  Total time:         {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
    print(f"  Exact match:        {em_total}/{len(results)} ({em_total/len(results)*100:.0f}%)")
    print(f"  Avg F1:             {avg_f1:.3f}")
    print(f"  Grade distribution: A={grade_dist['A']}  B={grade_dist['B']}  "
          f"C={grade_dist['C']}  D={grade_dist['D']}  F={grade_dist['F']}")
    ab = grade_dist["A"] + grade_dist["B"]
    print(f"  A+B rate:           {ab/len(results)*100:.0f}%")
    print(f"  Total LLM calls:    {sum(r['llm_calls'] for r in results)}")
    print(f"  Errors:             {sum(1 for r in results if r['error'])}")
    print(f"{'='*80}")

    # JSON dump
    print(f"\n{'='*80}")
    print("FULL RESULTS JSON")
    print(f"{'='*80}")
    json_results = []
    for r in results:
        jr = dict(r)
        jr.pop("answer_preview", None)
        jr.pop("agent_steps", None)
        jr.pop("gold_decomp", None)
        json_results.append(jr)
    print(json.dumps(json_results, indent=2, default=str))


# ────────────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────────────

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    max_queries = int(sys.argv[2]) if len(sys.argv) > 2 else None

    if mode in ("retrieval", "both"):
        phase1_retrieval(k=5, max_queries=max_queries)

    if mode in ("pipeline", "both"):
        phase2_pipeline(max_queries=max_queries)


if __name__ == "__main__":
    main()
