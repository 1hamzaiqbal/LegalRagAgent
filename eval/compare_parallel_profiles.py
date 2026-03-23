"""Run side-by-side review batches for `full_parallel` vs `full_parallel_aspect`."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

try:
    from ._path_setup import ensure_project_root_on_path
except ImportError:
    from _path_setup import ensure_project_root_on_path

ensure_project_root_on_path()

from legal_rag import _get_metrics, run_experiment
from eval.eval_utils import check_mc_correctness, extract_mc_letter, select_qa_queries
from playtests.cases import PLAYTEST_CASES

PROFILES = ("full_parallel", "full_parallel_aspect")


def _format_objective(question: str, choices: Dict[str, str]) -> str:
    if not choices or not any(choices.values()):
        return question
    choice_text = "\n".join(f"  ({key}) {value}" for key, value in sorted(choices.items()) if value)
    return f"{question}\n\nAnswer choices:\n{choice_text}"


def _sample_cases(n_random: int) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    for case in PLAYTEST_CASES:
        cases.append(
            {
                "label": case["name"],
                "kind": "playtest",
                "question": case["question"],
                "choices": {},
                "correct_answer": "",
                "subject": "playtest",
                "goal": case["goal"],
            }
        )

    for query in select_qa_queries(n_random):
        cases.append(
            {
                "label": query["label"],
                "kind": "bar_sample",
                "question": query["question"],
                "choices": query.get("choices", {}),
                "correct_answer": query.get("correct_answer", ""),
                "subject": query.get("subject", "unknown"),
                "goal": "Random deterministic bar sample",
            }
        )
    return cases


def _summarise_steps(planning_table: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "step_id": step["step_id"],
            "sub_question": step["sub_question"],
            "action_type": step["action_type"],
            "status": step["status"],
            "confidence": step["confidence"],
            "judge_verdict": step.get("judge_verdict"),
        }
        for step in planning_table
    ]


def _run_case(profile_name: str, case: Dict[str, Any]) -> Dict[str, Any]:
    objective = _format_objective(case["question"], case.get("choices", {}))
    before = _get_metrics()
    result = run_experiment(
        objective,
        profile=profile_name,
        raw_question=case["question"],
        print_output=False,
        write_run_artifact=True,
    )
    after = _get_metrics()

    chosen = extract_mc_letter(result.final_answer) or "?"
    is_correct = bool(case.get("correct_answer")) and check_mc_correctness(result.final_answer, case["correct_answer"])
    return {
        "profile": profile_name,
        "artifact_path": result.artifact_path,
        "parallel_round": result.parallel_round - 1,
        "collections": result.collections,
        "completeness_verdict": result.completeness_verdict,
        "planning_table": _summarise_steps([step.model_dump() for step in result.planning_table]),
        "step_traces": result.run_artifact.get("step_traces", []),
        "replanning_brief": result.extra.get("replanning_brief", ""),
        "final_answer": result.final_answer,
        "chosen_letter": chosen,
        "is_correct": is_correct,
        "llm_calls": after["count"] - before["count"],
        "input_tokens": after["input_tokens"] - before["input_tokens"],
        "output_tokens": after["output_tokens"] - before["output_tokens"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare `full_parallel` and `full_parallel_aspect` on playtests plus a random bar sample.")
    parser.add_argument("--n-random", type=int, default=3, help="Number of deterministic random bar questions to include.")
    args = parser.parse_args()

    cases = _sample_cases(args.n_random)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    os.makedirs("logs/review", exist_ok=True)
    out_path = os.path.join("logs", "review", f"parallel_profile_review_{timestamp}.json")

    records = []
    print(f"Running {len(cases)} cases across {len(PROFILES)} profiles")
    for index, case in enumerate(cases, 1):
        print(f"\n[{index}/{len(cases)}] {case['label']} ({case['kind']})")
        case_record = {
            "label": case["label"],
            "kind": case["kind"],
            "subject": case["subject"],
            "goal": case["goal"],
            "question": case["question"],
            "correct_answer": case["correct_answer"],
            "profiles": {},
        }
        for profile_name in PROFILES:
            print(f"  -> {profile_name}")
            case_record["profiles"][profile_name] = _run_case(profile_name, case)
        records.append(case_record)

    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
    print(f"\nSaved review batch to {out_path}")


if __name__ == "__main__":
    main()
