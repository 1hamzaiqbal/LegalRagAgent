"""Legacy-named wrapper for the `simple_rag` profile evaluation."""

from __future__ import annotations

import sys

try:
    from ._path_setup import ensure_project_root_on_path
except ImportError:
    from _path_setup import ensure_project_root_on_path

ensure_project_root_on_path()

from eval.profile_eval import run_profile_evaluation


def main() -> None:
    args = sys.argv[1:]
    n = int(args[0]) if args else 10
    run_profile_evaluation(
        profile_name="simple_rag",
        log_prefix="eval_bm25_baseline",
        report_title="SIMPLE RAG BASELINE (legacy script name: eval_bm25_baseline)",
        n=n,
        suite="bar",
        continue_eval=False,
    )


if __name__ == "__main__":
    main()
