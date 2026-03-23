"""Run the curated playtest suite against profile-driven runtime variants."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

try:
    from ._path_setup import ensure_project_root_on_path
except ImportError:
    from _path_setup import ensure_project_root_on_path

ensure_project_root_on_path()

from legal_rag import run_experiment
from playtests.cases import list_playtest_case_names, resolve_playtest_cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the curated Legal RAG playtest suite.")
    parser.add_argument("--profile", help="Override the case profile for every playtest.")
    parser.add_argument("--case", help="Run only one named playtest case.")
    parser.add_argument("--list-cases", action="store_true", help="Print available playtest case names and exit.")
    args = parser.parse_args()

    if args.list_cases:
        for case_name in list_playtest_case_names():
            print(case_name)
        return

    os.makedirs("logs/playtests", exist_ok=True)
    summary = []
    for case in resolve_playtest_cases(args.case):
        profile = args.profile or case["profile"]
        print(f"\n{'=' * 80}")
        print(f"PLAYTEST: {case['name']} | profile={profile}")
        print(case["goal"])
        print(f"{'=' * 80}")
        result = run_experiment(case["question"], profile=profile, raw_question=case["question"])
        summary.append(
            {
                "name": case["name"],
                "profile": profile,
                "artifact_path": result.artifact_path,
                "completeness": result.completeness_verdict,
                "steps_completed": sum(1 for step in result.planning_table if step.status == "completed"),
            }
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = os.path.join("logs", "playtests", f"summary_{timestamp}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(f"\nPlaytest summary saved to {path}")


if __name__ == "__main__":
    main()
