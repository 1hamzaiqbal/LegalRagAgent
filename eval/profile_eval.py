"""Shared evaluation runner for profile-driven experiments."""

from __future__ import annotations

import glob
import json
import os
import time
from typing import Dict, List

from legal_rag import _get_metrics, get_profile, run_experiment
from eval.eval_utils import (
    capture_balance,
    check_mc_correctness,
    compute_cost,
    extract_mc_letter,
    load_eval_labels,
    select_qa_queries,
    select_qa_queries_by_labels,
)
from eval.web_search_suite import select_web_search_queries


def _select_queries(suite: str, n: int, labels_path: str | None = None) -> List[Dict]:
    if labels_path:
        labels = load_eval_labels(labels_path)
        if suite != "bar":
            raise ValueError("Explicit label replay is currently supported only for the bar suite.")
        return select_qa_queries_by_labels(labels[:n] if n else labels)
    if suite == "bar":
        return select_qa_queries(n)
    if suite == "web":
        return select_web_search_queries(n)
    raise ValueError(f"Unsupported suite: {suite}")


def _format_objective(question: str, choices: Dict[str, str]) -> str:
    if not choices or not any(choices.values()):
        return question
    choice_text = "\n".join(f"  ({key}) {value}" for key, value in sorted(choices.items()) if value)
    return f"{question}\n\nAnswer choices:\n{choice_text}"


def _load_completed(detail_file: str) -> Dict[str, Dict]:
    if not os.path.exists(detail_file):
        return {}
    completed = {}
    with open(detail_file, "r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            completed[record["label"]] = record
    return completed


def _resolve_output_paths(
    *,
    log_prefix: str,
    suite: str,
    provider_name: str,
    continue_eval: bool,
) -> tuple[str, str, str]:
    suite_suffix = "" if suite == "bar" else f"_{suite}"
    stem = f"logs/{log_prefix}{suite_suffix}_{provider_name}"
    if continue_eval:
        candidates = glob.glob(f"{stem}_*_detail.jsonl")
        if candidates:
            detail_file = max(candidates, key=os.path.getmtime)
            return (
                detail_file.replace("_detail.jsonl", ".txt"),
                detail_file,
                detail_file.replace("_detail.jsonl", "_labels.json"),
            )
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = f"{stem}_{timestamp}.txt"
    return log_file, log_file.replace(".txt", "_detail.jsonl"), log_file.replace(".txt", "_labels.json")


def _build_summary_lines(
    *,
    report_title: str,
    profile_name: str,
    suite: str,
    n: int,
    results: List[Dict],
    queries: List[Dict],
    initial_totals: Dict[str, float],
) -> List[str]:
    correct = sum(1 for item in results if item["is_correct"])
    errors = sum(1 for item in results if item["error"])
    accuracy = correct / len(queries) * 100 if queries else 0.0
    gold_hits = sum(1 for item in results if item.get("gold_retrieved"))
    gold_rate = gold_hits / len(queries) * 100 if suite == "bar" and queries else 0.0
    cost_strings = compute_cost(initial_totals)

    lines = [
        f"{report_title} | profile={profile_name} | suite={suite} | n={n}",
        f"Accuracy: {correct}/{len(queries)} ({accuracy:.1f}%)",
        f"Errors: {errors}",
    ]
    if suite == "bar":
        lines.append(f"Gold Recall@5: {gold_hits}/{len(queries)} ({gold_rate:.1f}%)")
    if cost_strings:
        lines.append(f"Cost: {', '.join(cost_strings)}")
    return lines


def _write_summary_log(log_file: str, lines: List[str], results: List[Dict]) -> None:
    with open(log_file, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
        handle.write("-" * 80 + "\n")
        for item in sorted(results, key=lambda row: row["label"]):
            status = "PASS" if item["is_correct"] else ("ERR" if item["error"] else "FAIL")
            handle.write(f"{item['label']:<30} {status:<6} {item['elapsed_sec']:>5.1f}s {item.get('llm_calls', 0):>4}\n")


def evaluate_single_query(profile_name: str, q: Dict, *, write_run_artifact: bool) -> Dict:
    objective = _format_objective(q["question"], q.get("choices", {}))
    error = None
    final_answer = ""
    retrieved_ids: List[str] = []
    rewrite_queries: List[str] = []
    artifact_path = None
    completeness_verdict = {}
    parallel_round = 0
    collections: List[str] = []
    run_timings = {}
    run_llm_metrics = {}
    started = time.time()

    try:
        result = run_experiment(
            objective,
            profile=profile_name,
            raw_question=q["question"],
            print_output=False,
            write_run_artifact=write_run_artifact,
        )
        final_answer = result.final_answer
        retrieved_ids = result.extra.get("retrieved_ids", []) or [str(item.get("idx", "")) for item in result.evidence_store]
        rewrite_queries = result.extra.get("rewrite_queries", [])
        artifact_path = result.artifact_path
        completeness_verdict = result.completeness_verdict
        parallel_round = max(0, result.parallel_round - 1)
        collections = result.collections
        run_timings = result.extra.get("timings", {})
        run_llm_metrics = result.extra.get("llm_metrics", {})
    except Exception as exc:
        error = str(exc)

    elapsed = time.time() - started
    gold_idx = str(q.get("gold_idx", ""))
    return {
        "label": q["label"],
        "subject": q.get("subject", "unknown"),
        "elapsed_sec": round(elapsed, 1),
        "error": error,
        "is_correct": check_mc_correctness(final_answer, q.get("correct_answer", "")),
        "chosen_letter": extract_mc_letter(final_answer) or "?",
        "correct_answer": q.get("correct_answer", ""),
        "gold_retrieved": bool(gold_idx and gold_idx in retrieved_ids),
        "retrieved_ids": retrieved_ids,
        "rewrite_queries": rewrite_queries,
        "artifact_path": artifact_path,
        "parallel_rounds": parallel_round,
        "collections": collections,
        "completeness_verdict": completeness_verdict,
        "run_timings": run_timings,
        "run_llm_metrics": run_llm_metrics,
        "llm_response": final_answer,
        "question": q["question"][:300],
    }


def run_profile_evaluation(
    *,
    profile_name: str,
    log_prefix: str,
    report_title: str,
    n: int,
    suite: str = "bar",
    continue_eval: bool = False,
    labels_path: str | None = None,
    write_run_artifact: bool | None = None,
) -> None:
    profile = get_profile(profile_name)
    if write_run_artifact is None:
        write_run_artifact = profile.kind == "full"
    provider_name = os.getenv("LLM_PROVIDER", "default").strip().lower()
    os.makedirs("logs", exist_ok=True)
    log_file, detail_file, labels_file = _resolve_output_paths(
        log_prefix=log_prefix,
        suite=suite,
        provider_name=provider_name,
        continue_eval=continue_eval,
    )

    completed = _load_completed(detail_file) if continue_eval else {}
    queries = _select_queries(suite, n, labels_path=labels_path)
    with open(labels_file, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "profile": profile.name,
                "suite": suite,
                "count": len(queries),
                "source_labels_path": labels_path,
                "labels": [query["label"] for query in queries],
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    todo = [query for query in queries if query["label"] not in completed]

    print(f"\n{'=' * 80}")
    print(f"{report_title} ({n} QUERIES, SUITE={suite}, PROFILE={profile.name})")
    print(f"{'=' * 80}\n")

    _, initial_totals = capture_balance()
    results = list(completed.values())
    for index, query in enumerate(todo, 1):
        start_metrics = _get_metrics()
        print(f"[{index}/{len(todo)}] {query['label']}...", end=" ", flush=True)
        result = evaluate_single_query(profile.name, query, write_run_artifact=write_run_artifact)
        end_metrics = _get_metrics()

        result["llm_calls"] = end_metrics["count"] - start_metrics["count"]
        result["input_tokens"] = end_metrics["input_tokens"] - start_metrics["input_tokens"]
        result["output_tokens"] = end_metrics["output_tokens"] - start_metrics["output_tokens"]

        tag = "CORRECT" if result["is_correct"] else ("ERROR" if result["error"] else "WRONG")
        gold = ""
        if suite == "bar":
            gold = " gold=Y" if result["gold_retrieved"] else " gold=N"
        print(f"{tag}{gold} ({result['elapsed_sec']}s)")
        results.append(result)
        with open(detail_file, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        lines = _build_summary_lines(
            report_title=report_title,
            profile_name=profile.name,
            suite=suite,
            n=n,
            results=results,
            queries=queries,
            initial_totals=initial_totals,
        )
        _write_summary_log(log_file, lines, results)

    lines = _build_summary_lines(
        report_title=report_title,
        profile_name=profile.name,
        suite=suite,
        n=n,
        results=results,
        queries=queries,
        initial_totals=initial_totals,
    )

    print(f"\n{'=' * 80}")
    for line in lines:
        print(line)
    print(f"{'=' * 80}\n")

    _write_summary_log(log_file, lines, results)

    with open(detail_file, "w", encoding="utf-8") as handle:
        for item in results:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Log saved to {log_file}")
    print(f"Detail log saved to {detail_file}")
    print(f"Label manifest saved to {labels_file}")
