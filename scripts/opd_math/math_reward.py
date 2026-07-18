#!/usr/bin/env python3
"""Pinned, structured math-verifier rewards for student rollouts."""
from __future__ import annotations

from typing import Any


def _verifier_imports():
    try:
        from latex2sympy2_extended import NormalizationConfig
        from math_verify import LatexExtractionConfig, parse, verify
    except ImportError as exc:
        raise RuntimeError(
            "math reward requires math-verify[antlr4_13_2]==0.9.0 from requirements/opd-math.txt"
        ) from exc
    return NormalizationConfig, LatexExtractionConfig, parse, verify


def verify_completion(completion: str, gold_solution: str) -> dict[str, Any]:
    """Return a binary reward plus a non-lossy verifier status.

    Prediction parse failures are task failures and receive zero. Gold parse
    failures and verifier exceptions are infrastructure/data errors; callers
    must skip or abort rather than silently treating them as wrong answers.
    """
    NormalizationConfig, LatexExtractionConfig, parse, verify = _verifier_imports()
    gold = parse(str(gold_solution), extraction_mode="first_match")
    if not gold:
        return {"reward": None, "status": "gold_parse_failed"}

    prediction = parse(
        str(completion),
        extraction_config=[
            LatexExtractionConfig(
                normalization_config=NormalizationConfig(
                    nits=False,
                    malformed_operators=False,
                    basic_latex=True,
                    equations=True,
                    boxed="all",
                    units=True,
                ),
                boxed_match_priority=0,
                try_extract_without_anchor=False,
            )
        ],
        extraction_mode="first_match",
    )
    if not prediction:
        return {"reward": 0.0, "status": "prediction_parse_failed"}
    try:
        correct = bool(verify(gold, prediction))
    except Exception as exc:  # verifier failures must remain distinguishable
        return {
            "reward": None,
            "status": "verifier_error",
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
        }
    return {"reward": float(correct), "status": "correct" if correct else "incorrect"}


def verify_trl_accuracy_completion(completion: str, gold_solution: str) -> dict[str, Any]:
    """Recompute TRL 1.8 ``accuracy_reward`` for one traced teacher sample.

    The teacher deliberately uses TRL's reward contract, whose prediction
    normalization differs from the student reward above.  Keep the two paths
    separate so a scientific teacher gate can verify the exact reward that was
    optimized rather than silently substituting the student verifier.
    """

    NormalizationConfig, LatexExtractionConfig, parse, verify = _verifier_imports()
    gold = parse(str(gold_solution), parsing_timeout=10)
    if not gold:
        return {"reward": None, "status": "gold_parse_failed"}

    prediction = parse(
        str(completion),
        extraction_config=[
            LatexExtractionConfig(
                normalization_config=NormalizationConfig(units=True),
                boxed_match_priority=0,
                try_extract_without_anchor=False,
            )
        ],
        extraction_mode="first_match",
        parsing_timeout=10,
    )
    try:
        correct = bool(verify(gold, prediction, timeout_seconds=5))
    except Exception as exc:  # verifier failures are infrastructure failures
        return {
            "reward": None,
            "status": "verifier_error",
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
        }
    if not prediction:
        return {"reward": 0.0, "status": "prediction_parse_failed"}
    return {"reward": float(correct), "status": "correct" if correct else "incorrect"}


def rewards_for_samples(samples: list[dict]) -> tuple[list[float], list[str]]:
    rewards: list[float] = []
    statuses: list[str] = []
    for sample in samples:
        result = verify_completion(sample["completion_text"], sample["solution"])
        status = str(result["status"])
        if status in ("gold_parse_failed", "verifier_error"):
            raise RuntimeError(
                f"math verifier infrastructure/data failure for {sample.get('record_id')}: {result}"
            )
        rewards.append(float(result["reward"]))
        statuses.append(status)
    return rewards, statuses
