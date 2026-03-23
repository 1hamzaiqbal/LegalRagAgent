"""Full profile evaluation wrapper for LangGraph-backed profiles."""

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
    profile = "full_parallel"
    labels_path = None
    write_run_artifact = True

    if "--suite" in args:
        index = args.index("--suite")
        suite = args[index + 1].strip().lower()
        args = args[:index] + args[index + 2 :]

    if "--profile" in args:
        index = args.index("--profile")
        profile = args[index + 1].strip()
        args = args[:index] + args[index + 2 :]

    if "--labels-file" in args:
        index = args.index("--labels-file")
        labels_path = args[index + 1].strip()
        args = args[:index] + args[index + 2 :]

    if "--no-artifacts" in args:
        write_run_artifact = False
        args.remove("--no-artifacts")

    if "--parallel" in args:
        raise SystemExit("`--parallel` is no longer supported. Use `--profile full_parallel` or `--profile full_seq`.")

    continue_eval = "--continue" in args
    if continue_eval:
        args.remove("--continue")

    n = int(args[0]) if args else 10
    run_profile_evaluation(
        profile_name=profile,
        log_prefix="eval_qa",
        report_title="PROFILE QA EVALUATION",
        n=n,
        suite=suite,
        continue_eval=continue_eval,
        labels_path=labels_path,
        write_run_artifact=write_run_artifact,
    )


if __name__ == "__main__":
    main()
