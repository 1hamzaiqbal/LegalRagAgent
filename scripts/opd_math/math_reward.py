#!/usr/bin/env python3
"""Pinned, structured math-verifier rewards for student rollouts."""
from __future__ import annotations

import threading
from typing import Any


EVALUATION_VERIFIER_ERROR_POLICY = "conservative_zero_with_sensitivity_v1"
MAX_EVALUATION_VERIFIER_ERROR_FRACTION = 0.001
EVALUATION_VERIFIER_MAX_ATTEMPTS = 3


def _verifier_imports():
    try:
        from latex2sympy2_extended import NormalizationConfig
        from math_verify import LatexExtractionConfig, parse, verify
        from math_verify.errors import TimeoutException
    except ImportError as exc:
        raise RuntimeError(
            "math reward requires math-verify[antlr4_13_2]==0.9.0 from requirements/opd-math.txt"
        ) from exc
    return NormalizationConfig, LatexExtractionConfig, parse, verify, TimeoutException


def verify_completion(completion: str, gold_solution: str) -> dict[str, Any]:
    """Return a binary reward plus a non-lossy verifier status.

    Prediction parse failures are task failures and receive zero. Gold parse
    failures and verifier exceptions are infrastructure/data errors; callers
    must skip or abort rather than silently treating them as wrong answers.
    """
    NormalizationConfig, LatexExtractionConfig, parse, verify, TimeoutException = (
        _verifier_imports()
    )
    try:
        gold = parse(
            str(gold_solution),
            extraction_mode="first_match",
            raise_on_error=True,
        )
    except (Exception, TimeoutException) as exc:
        return {
            "reward": None,
            "status": "gold_parser_error",
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
        }
    if not gold:
        return {"reward": None, "status": "gold_parse_failed"}

    try:
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
            raise_on_error=True,
        )
    except (Exception, TimeoutException) as exc:
        return {
            "reward": None,
            "status": "prediction_parser_error",
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
        }
    if not prediction:
        return {"reward": 0.0, "status": "prediction_parse_failed"}
    try:
        correct = bool(verify(gold, prediction, raise_on_error=True))
    except (Exception, TimeoutException) as exc:  # TimeoutException derives from BaseException
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

    NormalizationConfig, LatexExtractionConfig, parse, verify, TimeoutException = (
        _verifier_imports()
    )
    is_main_thread = threading.current_thread() is threading.main_thread()
    if not is_main_thread:
        return {
            "reward": None,
            "status": "verifier_error",
            "error_type": "UnsupportedThreadedVerifier",
            "error": (
                "strict TRL math verification requires the main interpreter thread "
                "so parser/verifier timeouts remain bounded"
            ),
        }
    parsing_timeout = 10
    verify_timeout = 5

    try:
        gold = parse(
            str(gold_solution),
            parsing_timeout=parsing_timeout,
            raise_on_error=True,
        )
    except (Exception, TimeoutException) as exc:
        return {
            "reward": None,
            "status": "gold_parser_error",
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
        }
    if not gold:
        return {"reward": None, "status": "gold_parse_failed"}

    try:
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
            parsing_timeout=parsing_timeout,
            raise_on_error=True,
        )
    except (Exception, TimeoutException) as exc:
        return {
            "reward": None,
            "status": "prediction_parser_error",
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
        }
    try:
        correct = bool(
            verify(
                gold,
                prediction,
                timeout_seconds=verify_timeout,
                raise_on_error=True,
            )
        )
    except (Exception, TimeoutException) as exc:  # TimeoutException derives from BaseException
        return {
            "reward": None,
            "status": "verifier_error",
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
        }
    if not prediction:
        return {"reward": 0.0, "status": "prediction_parse_failed"}
    return {"reward": float(correct), "status": "correct" if correct else "incorrect"}


def apply_evaluation_verifier_error_policy(verdict: dict[str, Any]) -> dict[str, Any]:
    """Make candidate-side verifier failures visible and conservatively wrong.

    Training remains fail-closed. Large repeated evaluations instead retain a
    bounded error surface so one pathological model expression cannot erase an
    otherwise complete immutable evaluation. Gold parse failures still abort.
    """

    if verdict.get("status") not in {"verifier_error", "prediction_parser_error"}:
        return dict(verdict)
    return {
        "reward": 0.0,
        "status": "verifier_error_zeroed",
        "verifier_error_type": verdict.get("error_type"),
        "verifier_error": verdict.get("error"),
        "verifier_stage": (
            "prediction_parse"
            if verdict.get("status") == "prediction_parser_error"
            else "symbolic_verify"
        ),
        "policy": EVALUATION_VERIFIER_ERROR_POLICY,
    }


def verify_evaluation_completion(completion: str, gold_solution: str) -> dict[str, Any]:
    """Retry candidate-side infrastructure failures before retaining uncertainty.

    Math-Verify's signal-bounded symbolic checks can time out nondeterministically
    near the timeout boundary. Evaluation retries candidate-side parser/verifier
    failures a fixed number of times. A determinate verdict wins; only an
    all-error sequence is retained as a bounded unknown reward. Gold-side
    failures are never retried into a candidate reward and remain fatal.
    """

    error_history: list[dict[str, str]] = []
    for attempt in range(1, EVALUATION_VERIFIER_MAX_ATTEMPTS + 1):
        verdict = verify_completion(completion, gold_solution)
        if verdict.get("reward") is not None:
            result = dict(verdict)
            result["evaluation_verifier_attempts"] = attempt
            result["transient_candidate_error_count"] = len(error_history)
            return result
        if verdict.get("status") not in {"verifier_error", "prediction_parser_error"}:
            return dict(verdict)
        error_history.append(
            {
                "status": str(verdict.get("status")),
                "error_type": str(verdict.get("error_type")),
                "error": str(verdict.get("error")),
            }
        )

    result = apply_evaluation_verifier_error_policy(verdict)
    result["verifier_attempts"] = EVALUATION_VERIFIER_MAX_ATTEMPTS
    result["verifier_error_history"] = error_history
    return result


def strict_trl_accuracy_rewards(
    completions: list[list[dict[str, str]]],
    gold_solutions: list[str],
    *,
    record_ids: list[str] | None = None,
) -> list[float]:
    """Apply TRL's math reward semantics while failing closed on verifier errors."""

    if len(completions) != len(gold_solutions):
        raise ValueError("teacher reward completions and solutions have inconsistent lengths")
    if record_ids is not None and len(record_ids) != len(completions):
        raise ValueError("teacher reward record IDs have an inconsistent length")

    rewards: list[float] = []
    for index, (completion, gold_solution) in enumerate(
        zip(completions, gold_solutions, strict=True)
    ):
        content = (
            completion[0].get("content")
            if isinstance(completion, list) and completion
            else None
        )
        if not isinstance(content, str):
            raise ValueError(f"teacher reward completion {index} is not conversational text")
        result = verify_trl_accuracy_completion(content, gold_solution)
        if result.get("reward") is None:
            identity = record_ids[index] if record_ids is not None else f"index {index}"
            raise RuntimeError(f"teacher math verifier failure for {identity}: {result}")
        rewards.append(float(result["reward"]))
    return rewards


def rewards_for_samples(samples: list[dict]) -> tuple[list[float], list[str]]:
    rewards: list[float] = []
    statuses: list[str] = []
    for sample in samples:
        result = verify_completion(sample["completion_text"], sample["solution"])
        status = str(result["status"])
        if result.get("reward") is None:
            raise RuntimeError(
                f"math verifier infrastructure/data failure for {sample.get('record_id')}: {result}"
            )
        rewards.append(float(result["reward"]))
        statuses.append(status)
    return rewards, statuses
