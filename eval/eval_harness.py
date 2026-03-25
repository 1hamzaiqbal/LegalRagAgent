"""Multi-model evaluation harness for the Legal RAG pipeline.

Supports multiple eval modes, providers, question sets, and skill overrides.
Produces rich per-question JSONL logs and appends run summaries to experiments.jsonl.

Usage:
    uv run python eval/eval_harness.py --mode llm_only --provider deepseek --questions 10
    uv run python eval/eval_harness.py --mode full_pipeline --provider gemma --questions curated
    uv run python eval/eval_harness.py --mode rag_rewrite --provider deepseek --questions 30 --tag "aspect-queries"
    uv run python eval/eval_harness.py --mode golden_passage --provider openai --questions 100
    uv run python eval/eval_harness.py --mode full_pipeline --skill-dir skills_v2 --questions curated
"""
import argparse
import json
import os
import subprocess
import sys
from typing import List
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from eval_config import EvalConfig, EVAL_MODES, load_questions, extract_answer_mc, format_question_prompt
from main import (
    run as run_pipeline,
    _llm_call,
    _get_metrics,
    _reset_llm_call_counter,
    _parse_json,
    load_skill,
)
from llm_config import get_provider_info, _get_llm_cached
from rag_utils import retrieve_documents_multi_query, get_vectorstore


# ---------------------------------------------------------------------------
# Mode Runner Functions
# ---------------------------------------------------------------------------

def run_full_pipeline(row: pd.Series, config: EvalConfig) -> dict:
    """Run the full agentic pipeline and capture complete state."""
    question = format_question_prompt(row)
    result = run_pipeline(question, print_output=False)

    # Serialize PlanningStep objects
    planning_table = []
    for s in result.get("planning_table", []):
        if hasattr(s, "model_dump"):
            planning_table.append(s.model_dump())
        elif hasattr(s, "__dict__"):
            planning_table.append(vars(s))
        else:
            planning_table.append(s)

    # Check if gold passage was retrieved
    gold_idx = str(row.get("gold_idx", ""))
    retrieved_ids = [ev.get("idx", "") for ev in result.get("evidence_store", [])]
    gold_retrieved = gold_idx in retrieved_ids if gold_idx else False

    return {
        "final_answer": result.get("final_answer", ""),
        "collections": result.get("collections", []),
        "planning_table": planning_table,
        "evidence_store": result.get("evidence_store", []),
        "audit_log": result.get("audit_log", []),
        "completeness_verdict": result.get("completeness_verdict", {}),
        "parallel_rounds": result.get("parallel_round", 1) - 1,
        "retrieved_ids": retrieved_ids,
        "gold_retrieved": gold_retrieved,
    }


def run_llm_only(row: pd.Series, config: EvalConfig) -> dict:
    """Direct LLM answer with no retrieval."""
    question = format_question_prompt(row)
    system = (
        "You are a legal expert. Answer the multiple-choice question below. "
        "Reason step by step, then give your final answer as: Answer: (X)"
    )
    answer = _llm_call(system, question, label="llm_only")
    return {"final_answer": answer}


def run_golden_passage(row: pd.Series, config: EvalConfig) -> dict:
    """LLM answer with the gold passage injected as context."""
    question = format_question_prompt(row)
    gold = str(row.get("gold_passage", ""))
    if not gold or gold == "nan":
        # Fall back to llm_only if no gold passage
        return run_llm_only(row, config)

    system = (
        "You are a legal expert. Reason through the multiple-choice question "
        "step by step. A reference passage is provided — use it to verify or "
        "refine your reasoning, but think through the problem independently first. "
        "Give your final answer as: Answer: (X)"
    )
    user = f"## Reference Passage\n{gold}\n\n## Question\n{question}"
    answer = _llm_call(system, user, label="golden_passage")
    return {"final_answer": answer}


def _golden_arb_common(row: pd.Series, config: EvalConfig, arb_system: str, label_prefix: str) -> dict:
    """Shared logic for golden arbitration variants."""
    question = format_question_prompt(row)
    gold = str(row.get("gold_passage", ""))
    if not gold or gold == "nan":
        return run_llm_only(row, config)

    # Step 1: Naive LLM answer (the "snap")
    snap_system = (
        "You are a legal expert. Answer the multiple-choice question below. "
        "Reason step by step, then give your final answer as: Answer: (X)"
    )
    snap_answer = _llm_call(snap_system, question, label=f"{label_prefix}/snap")
    snap_letter = extract_answer_mc(snap_answer)

    # Step 2: Show evidence and ask to confirm or revise
    arb_user = (
        f"## Your Previous Answer\n{snap_answer}\n\n"
        f"## Reference Passage\n{gold}\n\n"
        f"## Question\n{question}"
    )
    final_answer = _llm_call(arb_system, arb_user, label=f"{label_prefix}/arbitrate")
    final_letter = extract_answer_mc(final_answer)

    return {
        "final_answer": final_answer,
        "snap_answer": snap_answer,
        "snap_letter": snap_letter,
        "final_letter": final_letter,
        "changed": snap_letter != final_letter,
    }


def run_golden_arbitration(row: pd.Series, config: EvalConfig) -> dict:
    """LLM answers naively, then sees golden passage — neutral framing (no bias toward keeping/changing)."""
    arb_system = (
        "You are a legal expert. You previously answered a question based on your knowledge. "
        "Now you are given a reference passage that may contain relevant legal authority. "
        "Review the passage carefully against your previous reasoning. "
        "Reason step by step, then give your final answer as: Answer: (X)"
    )
    return _golden_arb_common(row, config, arb_system, "golden_arb")


def run_golden_arb_conservative(row: pd.Series, config: EvalConfig) -> dict:
    """LLM answers naively, then sees golden passage — conservative framing (biased toward keeping original)."""
    arb_system = (
        "You are a legal expert. You previously answered a question based on your knowledge. "
        "Now you are given a reference passage that may contain relevant legal authority. "
        "Review the passage carefully. If the evidence supports your original answer, keep it. "
        "If the evidence clearly points to a different answer, change it. "
        "Do not change your answer unless the evidence gives you a strong reason to. "
        "Reason step by step, then give your final answer as: Answer: (X)"
    )
    return _golden_arb_common(row, config, arb_system, "golden_arb_cons")


def _retrieve_and_format(row: pd.Series, queries: List[str], k: int = 5,
                         label_prefix: str = "rag", where: dict = None) -> dict:
    """Shared retrieval + evidence formatting. Returns dict with passages, evidence_store, metadata."""
    vs = get_vectorstore("legal_passages")
    docs = retrieve_documents_multi_query(queries=queries, k=k, vectorstore=vs, where=where)

    passages = []
    evidence_store = []
    for i, doc in enumerate(docs, 1):
        text = doc.page_content
        idx = str(doc.metadata.get("idx", f"{label_prefix}_{i}"))
        ce_score = doc.metadata.get("cross_encoder_score", 0.0)
        passages.append(f"[Source {i}]\n{text}")
        evidence_store.append({
            "idx": idx,
            "text": text,
            "source": doc.metadata.get("source", "unknown"),
            "cross_encoder_score": ce_score,
        })

    gold_idx = str(row.get("gold_idx", ""))
    retrieved_ids = [ev["idx"] for ev in evidence_store]

    return {
        "passages": passages,
        "evidence_store": evidence_store,
        "retrieved_ids": retrieved_ids,
        "gold_retrieved": gold_idx in retrieved_ids if gold_idx else False,
        "max_ce_score": max((ev["cross_encoder_score"] for ev in evidence_store), default=0.0),
    }


def _rewrite_query(question: str, label: str = "rag_rewrite/rewrite") -> List[str]:
    """LLM query rewrite → list of queries (primary + alternatives)."""
    rewrite_prompt = (
        f"Original legal research question: {question}\n\n"
        f"Sub-question: {question}\n"
        f"Authority target: \n"
        f"Retrieval hints: none"
    )
    raw_rewrite = _llm_call(load_skill("query_rewriter"), rewrite_prompt, label=label)
    parsed = _parse_json(raw_rewrite)

    if parsed and "primary" in parsed:
        return [parsed["primary"]] + parsed.get("alternatives", [])
    return [question]


_RAG_SYSTEM = (
    "You are a legal expert. Reason through the multiple-choice question "
    "step by step. Retrieved passages are provided — use them to verify or "
    "refine your reasoning, but think through the problem independently first. "
    "Give your final answer as: Answer: (X)"
)


def _where_from_config(config: EvalConfig) -> dict | None:
    """Build ChromaDB where filter from config.source_filter."""
    if config.source_filter:
        return {"source": config.source_filter}
    return None


def run_rag_rewrite(row: pd.Series, config: EvalConfig) -> dict:
    """Query rewrite → retrieval → answer with evidence."""
    question = format_question_prompt(row)
    queries = _rewrite_query(question)

    retrieval = _retrieve_and_format(row, queries, k=5, label_prefix="rewrite",
                                     where=_where_from_config(config))
    passage_block = "\n\n".join(retrieval["passages"])

    user = f"## Retrieved Passages\n{passage_block}\n\n## Question\n{question}"
    answer = _llm_call(_RAG_SYSTEM, user, label="rag_rewrite/answer")

    return {
        "final_answer": answer,
        "evidence_store": retrieval["evidence_store"],
        "retrieved_ids": retrieval["retrieved_ids"],
        "gold_retrieved": retrieval["gold_retrieved"],
        "rewrite_queries": queries,
    }


def run_rag_simple(row: pd.Series, config: EvalConfig) -> dict:
    """Raw question → retrieval → answer with evidence (no rewrite)."""
    question = format_question_prompt(row)
    raw_question = str(row["question"])

    retrieval = _retrieve_and_format(row, [raw_question], k=5, label_prefix="simple",
                                     where=_where_from_config(config))
    passage_block = "\n\n".join(retrieval["passages"])

    user = f"## Retrieved Passages\n{passage_block}\n\n## Question\n{question}"
    answer = _llm_call(_RAG_SYSTEM, user, label="rag_simple/answer")

    return {
        "final_answer": answer,
        "evidence_store": retrieval["evidence_store"],
        "retrieved_ids": retrieval["retrieved_ids"],
        "gold_retrieved": retrieval["gold_retrieved"],
    }


def run_rag_arbitration(row: pd.Series, config: EvalConfig) -> dict:
    """LLM answers naively, then reviews retrieved passages (conservative framing)."""
    question = format_question_prompt(row)
    queries = _rewrite_query(question, label="rag_arb/rewrite")

    # Step 1: Snap answer (no evidence)
    snap_system = (
        "You are a legal expert. Answer the multiple-choice question below. "
        "Reason step by step, then give your final answer as: Answer: (X)"
    )
    snap_answer = _llm_call(snap_system, question, label="rag_arb/snap")
    snap_letter = extract_answer_mc(snap_answer)

    # Step 2: Retrieve
    retrieval = _retrieve_and_format(row, queries, k=5, label_prefix="rag_arb",
                                     where=_where_from_config(config))
    passage_block = "\n\n".join(retrieval["passages"])

    # Step 3: Arbitrate with conservative framing
    arb_system = (
        "You are a legal expert. You previously answered a question based on your knowledge. "
        "Now you are given retrieved legal passages that may be relevant. "
        "Review the passages carefully. If the evidence supports your original answer, keep it. "
        "If the evidence clearly points to a different answer, change it. "
        "Do not change your answer unless the evidence gives you a strong reason to. "
        "Reason step by step, then give your final answer as: Answer: (X)"
    )
    arb_user = (
        f"## Your Previous Answer\n{snap_answer}\n\n"
        f"## Retrieved Passages\n{passage_block}\n\n"
        f"## Question\n{question}"
    )
    final_answer = _llm_call(arb_system, arb_user, label="rag_arb/arbitrate")
    final_letter = extract_answer_mc(final_answer)

    return {
        "final_answer": final_answer,
        "snap_answer": snap_answer,
        "snap_letter": snap_letter,
        "final_letter": final_letter,
        "changed": snap_letter != final_letter,
        "evidence_store": retrieval["evidence_store"],
        "retrieved_ids": retrieval["retrieved_ids"],
        "gold_retrieved": retrieval["gold_retrieved"],
        "rewrite_queries": queries,
        "max_ce_score": retrieval["max_ce_score"],
    }


MODE_RUNNERS = {
    "full_pipeline": run_full_pipeline,
    "llm_only": run_llm_only,
    "rag_rewrite": run_rag_rewrite,
    "rag_simple": run_rag_simple,
    "golden_passage": run_golden_passage,
    "golden_arbitration": run_golden_arbitration,
    "golden_arb_conservative": run_golden_arb_conservative,
    "rag_arbitration": run_rag_arbitration,
}


# ---------------------------------------------------------------------------
# Harness Core
# ---------------------------------------------------------------------------

def _setup_provider(config: EvalConfig):
    """Set env vars and clear caches for provider/skill switching."""
    os.environ["LLM_PROVIDER"] = config.provider
    _get_llm_cached.cache_clear()

    if config.skill_dir != "skills":
        os.environ["SKILL_DIR"] = config.skill_dir
        load_skill.cache_clear()


def _git_commit_short() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def _serialize_result(result: dict) -> dict:
    """Ensure all values are JSON-serializable."""
    out = {}
    for k, v in result.items():
        if isinstance(v, float) and (v != v):  # NaN check
            out[k] = None
        else:
            try:
                json.dumps(v)
                out[k] = v
            except (TypeError, ValueError):
                out[k] = str(v)
    return out


def run_eval(config: EvalConfig):
    """Run evaluation with the given config."""
    if config.mode not in MODE_RUNNERS:
        print(f"Unknown mode '{config.mode}'. Available: {', '.join(MODE_RUNNERS)}")
        sys.exit(1)

    _setup_provider(config)
    runner = MODE_RUNNERS[config.mode]
    provider_info = get_provider_info()

    qa = load_questions(config)
    n = len(qa)

    print(f"\n{'=' * 70}")
    filter_str = f" | filter={config.source_filter}" if config.source_filter else ""
    print(f"EVAL: {config.mode} | {provider_info['provider']} ({provider_info['model']}) | {n} questions{filter_str}")
    if config.skill_dir != "skills":
        print(f"Skills: {config.skill_dir}")
    if config.tag:
        print(f"Tag: {config.tag}")
    print(f"{'=' * 70}\n")

    results = []
    correct = 0
    total_start = time.time()

    for i, row in qa.iterrows():
        _reset_llm_call_counter()
        q_start = time.time()

        subject = str(row.get("subject", "unknown"))
        idx = str(row.get("idx", i))
        label = f"qa_{subject}_{idx}"
        gold = str(row["answer"]).strip().upper()

        try:
            result = runner(row, config)
            answer_text = result.get("final_answer", "")
            predicted = extract_answer_mc(answer_text)
            is_correct = predicted == gold
            error = None
        except Exception as e:
            result = {}
            answer_text = ""
            predicted = None
            is_correct = False
            error = str(e)

        elapsed = time.time() - q_start
        metrics = _get_metrics()

        if is_correct:
            correct += 1

        status = "PASS" if is_correct else "FAIL"
        print(
            f"[{i+1}/{n}] {label:<35} {status:<6} "
            f"gold={gold} pred={predicted} "
            f"({elapsed:.1f}s, {metrics['count']} calls)",
            flush=True,
        )

        # Build per-question record
        record = {
            "label": label,
            "subject": subject,
            "idx": idx,
            "question": str(row["question"]),
            "choices": {
                letter: str(row.get(f"choice_{letter.lower()}", ""))
                for letter in "ABCD"
            },
            "correct_answer": gold,
            "predicted_answer": predicted,
            "is_correct": is_correct,
            "error": error,
            "elapsed_sec": round(elapsed, 1),
            "llm_calls": metrics["count"],
            "input_tokens": metrics["input_tokens"],
            "output_tokens": metrics["output_tokens"],
            "gold_passage": str(row.get("gold_passage", ""))[:500],
            "gold_idx": str(row.get("gold_idx", "")),
            "final_answer": answer_text,
            "mode": config.mode,
            "provider": config.provider,
        }
        # Merge mode-specific fields (evidence_store, audit_log, etc.)
        for k, v in result.items():
            if k != "final_answer" and k not in record:
                record[k] = v

        results.append(_serialize_result(record))

    total_time = time.time() - total_start
    accuracy = correct / n if n > 0 else 0

    # --- Print summary ---
    print(f"\n{'=' * 70}")
    print(f"RESULTS: {correct}/{n} ({accuracy*100:.1f}%)")
    print(f"Total time: {total_time:.0f}s ({total_time/n:.1f}s/query)")

    # By-subject breakdown
    by_subject = {}
    for r in results:
        subj = r.get("subject", "unknown")
        if subj not in by_subject:
            by_subject[subj] = [0, 0]
        by_subject[subj][1] += 1
        if r["is_correct"]:
            by_subject[subj][0] += 1

    if len(by_subject) > 1:
        print("\nBy subject:")
        for subj, (c, t) in sorted(by_subject.items()):
            print(f"  {subj:<15} {c}/{t} ({c/t*100:.0f}%)")

    print(f"{'=' * 70}")

    # --- Save detail log ---
    ts = time.strftime("%Y%m%d_%H%M")
    question_set = config.questions if config.questions in ("curated", "full") else f"n{config.questions}"
    detail_filename = f"eval_{config.mode}_{config.provider}_{ts}_detail.jsonl"
    detail_path = os.path.join("logs", detail_filename)
    os.makedirs("logs", exist_ok=True)

    with open(detail_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nDetail log: {detail_path}")

    # --- Append to experiments.jsonl ---
    run_id = f"{ts}_{config.mode}_{config.provider}"
    if config.tag:
        run_id += f"_{config.tag}"

    summary = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": config.mode,
        "provider": provider_info["provider"],
        "model": provider_info["model"],
        "question_set": question_set,
        "n_questions": n,
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": n,
        "by_subject": by_subject,
        "avg_latency_sec": round(total_time / n, 1) if n else 0,
        "avg_llm_calls": round(sum(r["llm_calls"] for r in results) / n, 1) if n else 0,
        "total_input_tokens": sum(r["input_tokens"] for r in results),
        "total_output_tokens": sum(r["output_tokens"] for r in results),
        "skill_dir": config.skill_dir,
        "tag": config.tag,
        "source_filter": config.source_filter,
        "detail_log": detail_path,
        "git_commit": _git_commit_short(),
    }

    experiments_path = os.path.join("logs", "experiments.jsonl")
    with open(experiments_path, "a") as f:
        f.write(json.dumps(summary) + "\n")
    print(f"Run summary appended to: {experiments_path}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multi-model evaluation harness for Legal RAG pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(f"  {k:<20} {v}" for k, v in EVAL_MODES.items()),
    )
    parser.add_argument("--mode", default="full_pipeline", choices=EVAL_MODES.keys(),
                        help="Evaluation mode (default: full_pipeline)")
    parser.add_argument("--provider", default="deepseek",
                        help="LLM provider key from llm_config.py (default: deepseek)")
    parser.add_argument("--questions", default="30",
                        help="'curated', 'full', or integer N (default: 30)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for question sampling (default: 42)")
    parser.add_argument("--skill-dir", default="skills",
                        help="Directory containing skill prompts (default: skills)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("--tag", default="",
                        help="Optional label for this run")
    parser.add_argument("--source-filter", default="",
                        help="Metadata source filter for retrieval, e.g. 'mbe' (default: none)")

    args = parser.parse_args()

    if args.verbose:
        os.environ["VERBOSE"] = "1"

    config = EvalConfig(
        mode=args.mode,
        provider=args.provider,
        questions=args.questions,
        seed=args.seed,
        skill_dir=args.skill_dir,
        verbose=args.verbose,
        tag=args.tag,
        source_filter=args.source_filter,
    )

    run_eval(config)


if __name__ == "__main__":
    main()
