#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "eval"))

import eval_metrics  # type: ignore  # noqa: E402


def test_answer_contains_gold_uses_aliases_for_free_form() -> None:
    row = {
        "dataset": "musique",
        "correct_answer": "New York City",
        "aliases_used": ["NYC"],
        "final_answer": "The answer is NYC after the second hop.",
    }
    assert eval_metrics.answer_contains_gold(row) is True


def test_answer_contains_gold_skips_closed_set_tasks() -> None:
    row = {
        "dataset": "barexam",
        "correct_answer": "A",
        "final_answer": "The answer is A.",
    }
    assert eval_metrics.answer_contains_gold(row) is None


def test_evidence_and_generated_context_token_counts() -> None:
    row = {
        "evidence_store": [
            {"text": "alpha beta gamma"},
            {"text": "delta epsilon"},
        ],
        "hyde_passages": ["one two", "three four five"],
    }
    assert eval_metrics.evidence_doc_count(row) == 2
    assert eval_metrics.approx_token_count(eval_metrics.evidence_text(row)) == 5
    assert eval_metrics.approx_token_count(eval_metrics.generated_context_text(row)) == 5


def test_summarize_records_reports_speculative_style_fields() -> None:
    rows = [
        {
            "dataset": "musique",
            "mode": "rag_simple",
            "provider": "test",
            "is_correct": True,
            "em": True,
            "f1": 0.5,
            "correct_answer": "Paris",
            "final_answer": "Paris",
            "gold_retrieved": True,
            "evidence_store": [{"text": "Paris is in France"}],
            "hyde_passage": "France capital Paris",
            "elapsed_sec": 1.0,
            "llm_calls": 2,
            "input_tokens": 10,
            "output_tokens": 4,
        },
        {
            "dataset": "musique",
            "mode": "rag_simple",
            "provider": "test",
            "is_correct": False,
            "em": False,
            "f1": 0.0,
            "correct_answer": "Rome",
            "final_answer": "Milan",
            "gold_retrieved": False,
            "evidence_store": [],
            "elapsed_sec": 3.0,
            "llm_calls": 1,
            "input_tokens": 6,
            "output_tokens": 2,
        },
    ]
    summary = eval_metrics.summarize_records(rows, label="toy")
    assert summary["accuracy"] == 0.5
    assert summary["contains_gold_rate"] == 0.5
    assert summary["avg_f1"] == 0.25
    assert summary["gold_hit_rate"] == 0.5
    assert summary["avg_llm_calls"] == 1.5
    assert summary["avg_latency_sec"] == 2.0
    assert summary["avg_evidence_docs"] == 0.5
    assert summary["spec_score_row_rate"] == 0.0


if __name__ == "__main__":
    test_answer_contains_gold_uses_aliases_for_free_form()
    test_answer_contains_gold_skips_closed_set_tasks()
    test_evidence_and_generated_context_token_counts()
    test_summarize_records_reports_speculative_style_fields()
