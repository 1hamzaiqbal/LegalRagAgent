"""Direct-LLM profile evaluation wrapper."""

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
    suite = "bar"
    if "--suite" in args:
        index = args.index("--suite")
        suite = args[index + 1].strip().lower()
        args = args[:index] + args[index + 2 :]

    continue_eval = "--continue" in args
    if continue_eval:
        args.remove("--continue")

    n = int(args[0]) if args else 10
    run_profile_evaluation(
        profile_name="llm_only",
        log_prefix="eval_baseline",
        report_title="BASELINE LLM EVALUATION",
        n=n,
        suite=suite,
        continue_eval=continue_eval,
    )


if __name__ == "__main__":
    main()
