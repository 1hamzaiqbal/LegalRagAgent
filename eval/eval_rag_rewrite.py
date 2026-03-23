"""RAG-with-query-rewrite profile evaluation wrapper."""

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
        profile_name="rewrite_rag",
        log_prefix="eval_rag_rewrite",
        report_title="RAG WITH QUERY REWRITE BASELINE",
        n=n,
        suite="bar",
        continue_eval=False,
    )


if __name__ == "__main__":
    main()
