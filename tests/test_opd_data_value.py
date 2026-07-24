from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from scripts.opd import data_value_metrics as metrics


def contract() -> dict:
    root = Path(__file__).resolve().parents[1]
    return json.loads((root / "configs/opd_math/data_value_v1.json").read_text())


def test_contract_keeps_closed_sources_and_training_blocked() -> None:
    payload = contract()
    metrics.validate_contract(payload)
    assert payload["P3_design"]["current_allowed_training_sources"] == ["O"]
    assert payload["immutable_boundaries"]["failed_M_teacher_is_never_retrained_merged_or_used"]
    assert payload["immutable_boundaries"]["DeepMath_negative_qualification_is_not_reopened"]
    assert payload["immutable_boundaries"]["no_new_teacher_or_training_launch_authorized"]


def test_prequential_proxy_uses_token_weighted_trapezoid_area() -> None:
    result = metrics.prequential_proxy(
        [
            {"checkpoint_step": 0, "cumulative_training_tokens": 0, "source_nll_nats_per_token": 3.0},
            {"checkpoint_step": 1, "cumulative_training_tokens": 100, "source_nll_nats_per_token": 2.0},
            {"checkpoint_step": 2, "cumulative_training_tokens": 300, "source_nll_nats_per_token": 1.0},
        ]
    )
    expected_nats = 100 * 1.5 + 200 * 0.5
    assert result["signed_bits"] == pytest.approx(expected_nats / math.log(2))
    assert result["positive_part_bits"] == pytest.approx(result["signed_bits"])
    assert result["formal_epiplexity_claim"] is False


def test_requential_code_requires_teacher_paths_full_vocab_and_no_clip() -> None:
    row = {
        "optimizer_step": 1,
        "sampling_origin": "teacher_generated_paths",
        "kl_direction": "teacher||student",
        "full_vocabulary": True,
        "unclipped": True,
        "response_token_count": 10,
        "teacher_student_kl_nats_per_token": 0.2,
    }
    result = metrics.requential_code([row])
    assert result["total_nats"] == pytest.approx(2.0)
    assert result["total_bits"] == pytest.approx(2.0 / math.log(2))

    for key, invalid in (
        ("sampling_origin", "student_generated_paths"),
        ("kl_direction", "student||teacher"),
        ("full_vocabulary", False),
        ("unclipped", False),
    ):
        bad = dict(row)
        bad[key] = invalid
        with pytest.raises(RuntimeError):
            metrics.requential_code([bad])


def test_opd_state_proxy_is_explicitly_not_requential() -> None:
    result = metrics.opd_state_proxy(
        [
            {
                "sampling_origin": "student_generated_paths",
                "response_token_count": 20,
                "unclipped_divergence_nats_per_token": 0.3,
                "executed_divergence_nats_per_token": 0.05,
            }
        ]
    )
    assert result["unclipped_nats"] == pytest.approx(6.0)
    assert result["executed_nats"] == pytest.approx(1.0)
    assert result["requential_or_epiplexity_claim"] is False


def test_value_outcome_is_paired_difference_in_differences() -> None:
    unit = {
        "source": "O_block_00",
        "student_checkpoint": "qwen3_1.7b",
        "teacher_checkpoint": "privileged_qwen3_1.7b",
        "objective": "opsd",
        "seed": 0,
        "budget": "25_steps",
        "target": "MATH_dev",
    }
    result = metrics.value_outcomes(
        [
            {
                **unit,
                "opd_pre": 0.40,
                "opd_post": 0.50,
                "matched_control_pre": 0.40,
                "matched_control_post": 0.43,
            }
        ]
    )
    assert result[0]["paired_difference_in_differences"] == pytest.approx(0.07)
