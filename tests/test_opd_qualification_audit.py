from __future__ import annotations

import json
from pathlib import Path

from scripts.opd_math.qualification_audit import (
    analyze_evaluation_samples,
    analyze_paired_evaluations,
)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def sample(record_id: str, sample_idx: int, reward: float, status: str, tokens: int):
    return {
        "record_id": record_id,
        "sample_idx": sample_idx,
        "reward": reward,
        "reward_status": status,
        "completion_tokens": tokens,
    }


def test_evaluation_audit_separates_cap_and_below_cap_parse_failures(tmp_path: Path):
    path = tmp_path / "samples.jsonl"
    write_jsonl(
        path,
        [
            sample("a", 0, 0.0, "prediction_parse_failed", 512),
            sample("a", 1, 0.0, "prediction_parse_failed", 50),
            sample("b", 0, 1.0, "correct", 512),
            sample("b", 1, 0.0, "incorrect", 40),
        ],
    )
    result = analyze_evaluation_samples(path, max_tokens=512)
    assert result["records"] == 2
    assert result["samples"] == 4
    assert result["accuracy"] == 0.25
    assert result["at_cap_samples"] == 2
    assert result["parse_failures_at_cap"] == 1
    assert result["parse_failures_below_cap"] == 1


def test_paired_audit_reports_directional_reward_and_cap_transitions(tmp_path: Path):
    base = tmp_path / "base.jsonl"
    trained = tmp_path / "trained.jsonl"
    write_jsonl(
        base,
        [
            sample("a", 0, 0.0, "prediction_parse_failed", 512),
            sample("b", 0, 1.0, "correct", 100),
        ],
    )
    write_jsonl(
        trained,
        [
            sample("a", 0, 1.0, "correct", 200),
            sample("b", 0, 0.0, "incorrect", 100),
        ],
    )
    result = analyze_paired_evaluations(base, trained, max_tokens=512)
    assert result["reward_transitions"] == {"0_to_1": 1, "1_to_0": 1}
    assert result["reward_net_correct"] == 0
    assert result["cap_transitions"] == {"0_to_0": 1, "1_to_0": 1}
