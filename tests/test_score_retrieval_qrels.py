#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import score_retrieval_qrels as scorer  # type: ignore  # noqa: E402


def test_coerce_ids_handles_json_and_commas() -> None:
    assert scorer.coerce_ids('["a", "b"]') == ["a", "b"]
    assert scorer.coerce_ids("a,b") == ["a", "b"]
    assert scorer.coerce_ids([{"id": "x"}, "y"]) == ["x", "y"]


def test_score_rows_from_detail_gold() -> None:
    rows = [
        {"idx": "q1", "gold_idx": "d2", "retrieved_ids": ["d1", "d2", "d3"]},
        {"idx": "q2", "gold_idx": "d4,d5", "retrieved_ids": ["d0", "d4"]},
    ]
    qrels = scorer.qrels_from_detail(rows, "idx", "gold_idx")
    summary = scorer.score_rows(rows, qrels, "idx", "retrieved_ids", [1, 2])
    assert summary["evaluated"] == 2
    assert summary["metrics"]["1"]["hit"] == 0.0
    assert summary["metrics"]["2"]["hit"] == 1.0
    assert summary["metrics"]["2"]["recall"] == 0.75


if __name__ == "__main__":
    test_coerce_ids_handles_json_and_commas()
    test_score_rows_from_detail_gold()
