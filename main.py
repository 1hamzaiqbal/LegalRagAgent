"""Thin CLI and compatibility wrapper for the Legal RAG runtime.

If you are trying to understand the pipeline itself, read
`legal_rag/runtime.py` next. This file intentionally stays small so the CLI
does not become the new source of architectural complexity.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

from legal_rag import (
    _get_deepseek_balance,
    _get_metrics,
    _llm_call,
    _parse_json,
    _reset_llm_call_counter,
    DEMO_QUERIES,
    build_graph,
    get_profile,
    list_profiles,
    load_skill,
    run_experiment,
)
from legal_rag import core as core_module


def run(
    question: str,
    max_steps: int = 7,
    *,
    profile: str = "full_parallel",
    raw_question: str | None = None,
    print_output: bool = True,
) -> Any:
    """Compatibility wrapper for the current runtime."""
    return run_experiment(
        question,
        profile=profile,
        raw_question=raw_question,
        max_steps=max_steps,
        print_output=print_output,
    )


def _resolve_question(query: str | None, explicit_question: str | None) -> tuple[str, str]:
    if explicit_question:
        return explicit_question, explicit_question
    if not query:
        demo = DEMO_QUERIES["simple"]
        return demo, demo
    if query in DEMO_QUERIES:
        demo = DEMO_QUERIES[query]
        return demo, demo
    return query, query


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Legal RAG agent or a profile baseline.")
    parser.add_argument(
        "query",
        nargs="?",
        help="Demo key (`simple`, `medium`, `multi_hop`) or a literal question.",
    )
    parser.add_argument(
        "--question",
        help="Explicit literal question. Overrides the positional query.",
    )
    parser.add_argument(
        "--profile",
        default="full_parallel",
        help="Experiment profile to run (default: full_parallel).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=7,
        help="Maximum planned steps for full profiles (default: 7).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose LLM logging.",
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="Print available experiment profiles and exit.",
    )
    return parser


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    if args.verbose:
        os.environ["VERBOSE"] = "1"
        core_module.VERBOSE = True

    if args.list_profiles:
        for profile in list_profiles():
            print(f"{profile.name:<22} {profile.description}")
        raise SystemExit(0)

    get_profile(args.profile)
    question, raw_question = _resolve_question(args.query, args.question)
    run(question, max_steps=args.max_steps, profile=args.profile, raw_question=raw_question)
