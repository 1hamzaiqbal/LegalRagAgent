from __future__ import annotations

import pytest

from scripts.opd_math import math_reward


class _Config:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class _VerifierTimeout(BaseException):
    pass


def _imports_with(verify):
    def parse(text, **kwargs):
        return [f"parsed:{text}"]

    return _Config, _Config, parse, verify, _VerifierTimeout


@pytest.mark.parametrize(
    ("reward_fn", "expected_timeout"),
    [
        (math_reward.verify_completion, None),
        (math_reward.verify_trl_accuracy_completion, 5),
    ],
)
def test_reward_verifier_requests_strict_error_propagation(
    monkeypatch, reward_fn, expected_timeout
):
    calls = []

    def verify(gold, prediction, **kwargs):
        calls.append({"gold": gold, "prediction": prediction, **kwargs})
        if not kwargs.get("raise_on_error"):
            return False
        raise _VerifierTimeout("strict verifier failure")

    monkeypatch.setattr(math_reward, "_verifier_imports", lambda: _imports_with(verify))

    result = reward_fn("completion", "gold")

    assert result == {
        "reward": None,
        "status": "verifier_error",
        "error_type": "_VerifierTimeout",
        "error": "strict verifier failure",
    }
    assert len(calls) == 1
    assert calls[0]["raise_on_error"] is True
    if expected_timeout is None:
        assert "timeout_seconds" not in calls[0]
    else:
        assert calls[0]["timeout_seconds"] == expected_timeout


@pytest.mark.parametrize(
    "reward_fn",
    [math_reward.verify_completion, math_reward.verify_trl_accuracy_completion],
)
def test_reward_verifier_preserves_success_with_strict_mode(monkeypatch, reward_fn):
    def verify(_gold, _prediction, **kwargs):
        assert kwargs["raise_on_error"] is True
        return True

    monkeypatch.setattr(math_reward, "_verifier_imports", lambda: _imports_with(verify))

    assert reward_fn("completion", "gold") == {
        "reward": 1.0,
        "status": "correct",
    }


def test_strict_trl_batch_aborts_on_verifier_failure(monkeypatch):
    monkeypatch.setattr(
        math_reward,
        "verify_trl_accuracy_completion",
        lambda _completion, _gold: {
            "reward": None,
            "status": "verifier_error",
            "error_type": "TimeoutException",
            "error": "timed out",
        },
    )

    with pytest.raises(RuntimeError, match="teacher math verifier failure for record-1"):
        math_reward.strict_trl_accuracy_rewards(
            [[{"role": "assistant", "content": "answer"}]],
            ["gold"],
            record_ids=["record-1"],
        )


def test_strict_trl_batch_returns_binary_rewards(monkeypatch):
    verdicts = iter(
        [
            {"reward": 1.0, "status": "correct"},
            {"reward": 0.0, "status": "prediction_parse_failed"},
        ]
    )
    monkeypatch.setattr(
        math_reward,
        "verify_trl_accuracy_completion",
        lambda _completion, _gold: next(verdicts),
    )

    assert math_reward.strict_trl_accuracy_rewards(
        [
            [{"role": "assistant", "content": "answer 1"}],
            [{"role": "assistant", "content": "answer 2"}],
        ],
        ["gold 1", "gold 2"],
    ) == [1.0, 0.0]


@pytest.mark.parametrize(
    "reward_fn",
    [math_reward.verify_completion, math_reward.verify_trl_accuracy_completion],
)
def test_candidate_parser_timeout_is_explicit_and_strict(monkeypatch, reward_fn):
    parse_calls = []

    def parse(text, **kwargs):
        parse_calls.append({"text": text, **kwargs})
        assert kwargs["raise_on_error"] is True
        if text == "completion":
            raise _VerifierTimeout("parser timed out")
        return [f"parsed:{text}"]

    def verify(_gold, _prediction, **_kwargs):
        raise AssertionError("verification must not run after a parser timeout")

    monkeypatch.setattr(
        math_reward,
        "_verifier_imports",
        lambda: (_Config, _Config, parse, verify, _VerifierTimeout),
    )

    assert reward_fn("completion", "gold") == {
        "reward": None,
        "status": "prediction_parser_error",
        "error_type": "_VerifierTimeout",
        "error": "parser timed out",
    }
    assert len(parse_calls) == 2


def test_evaluation_retries_transient_candidate_errors(monkeypatch):
    verdicts = iter(
        [
            {
                "reward": None,
                "status": "verifier_error",
                "error_type": "TimeoutException",
                "error": "timeout",
            },
            {"reward": 1.0, "status": "correct"},
        ]
    )
    monkeypatch.setattr(
        math_reward,
        "verify_completion",
        lambda _completion, _gold: next(verdicts),
    )

    assert math_reward.verify_evaluation_completion("completion", "gold") == {
        "reward": 1.0,
        "status": "correct",
        "evaluation_verifier_attempts": 2,
        "transient_candidate_error_count": 1,
    }


def test_evaluation_retains_all_error_retry_history(monkeypatch):
    monkeypatch.setattr(
        math_reward,
        "verify_completion",
        lambda _completion, _gold: {
            "reward": None,
            "status": "verifier_error",
            "error_type": "TimeoutException",
            "error": "timeout",
        },
    )

    result = math_reward.verify_evaluation_completion("completion", "gold")

    assert result["reward"] == 0.0
    assert result["status"] == "verifier_error_zeroed"
    assert result["verifier_attempts"] == math_reward.EVALUATION_VERIFIER_MAX_ATTEMPTS
    assert len(result["verifier_error_history"]) == result["verifier_attempts"]


def test_trl_reward_fails_closed_outside_main_thread(monkeypatch):
    monkeypatch.setattr(
        math_reward,
        "_verifier_imports",
        lambda: _imports_with(lambda *_args, **_kwargs: True),
    )
    monkeypatch.setattr(math_reward.threading, "current_thread", lambda: object())
    monkeypatch.setattr(math_reward.threading, "main_thread", lambda: object())

    result = math_reward.verify_trl_accuracy_completion("completion", "gold")

    assert result["reward"] is None
    assert result["status"] == "verifier_error"
    assert result["error_type"] == "UnsupportedThreadedVerifier"


@pytest.mark.parametrize(
    ("completion", "gold", "error_type"),
    [
        (r"\boxed{17:00}", r"\boxed{05\!:\!00}", "ValueError"),
        (r"\boxed{m \ge 3}", r"\boxed{m \neq 4}", "KeyError"),
    ],
)
def test_observed_symbolic_verifier_failures_are_not_silent(completion, gold, error_type):
    verdict = math_reward.verify_completion(completion, gold)

    assert verdict["reward"] is None
    assert verdict["status"] == "verifier_error"
    assert verdict["error_type"] == error_type

    evaluation_verdict = math_reward.apply_evaluation_verifier_error_policy(verdict)
    assert evaluation_verdict["reward"] == 0.0
    assert evaluation_verdict["status"] == "verifier_error_zeroed"
    assert evaluation_verdict["verifier_error_type"] == error_type
    assert evaluation_verdict["policy"] == math_reward.EVALUATION_VERIFIER_ERROR_POLICY
