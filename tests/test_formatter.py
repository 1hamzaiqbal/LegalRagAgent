"""Regression tests for dataset-to-prompt formatters.

Locks in the 2026-04-22 prompt-column fix (commit f95f316). BarExam
has a 'prompt' column containing the shared fact pattern for MBE
items with the same prompt_id; the pre-fix harness dropped this
column entirely, leaving 445/1195 questions (37%) without their
fact pattern.

Run:
    uv run python tests/test_formatter.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from eval_config import format_question_prompt  # noqa: E402


def _row(**kw):
    """Build a pandas Series mimicking a dataset row."""
    return pd.Series(kw)


def test_barexam_includes_prompt_when_nonempty():
    """mbe_1014 regression — prompt fact pattern must be injected."""
    row = _row(
        idx="mbe_1014",
        prompt=(
            "In a single writing, Painter contracted with Farmer to paint "
            "three identical barns on her rural estate for $2,000 each. "
            "The contract provided for Farmer's payment of $6,000 upon "
            "Painter's completion of the work on all three barns."
        ),
        question="Is Farmer obligated to make the $4,000 payment?",
        choice_a="No, because Farmer has no duty under the contract to pay anything to Painter until all three barns have been painted.",
        choice_b="No, because Painter waived her right.",
        choice_c="Yes, because the contract is divisible.",
        choice_d="Yes, because Painter has substantially performed.",
        answer="A",
    )
    prompt = format_question_prompt(row, dataset="barexam")

    # Fact pattern must be present
    assert "Painter contracted with Farmer" in prompt
    assert "three identical barns" in prompt
    # Question stem also present
    assert "Is Farmer obligated to make the $4,000 payment?" in prompt
    # All 4 choices present
    for letter in "ABCD":
        assert f"({letter})" in prompt
    # Output instruction present
    assert "Answer: (X)" in prompt


def test_barexam_omits_prompt_when_empty():
    """When prompt is empty/missing, formatter must still work fine."""
    row = _row(
        idx="mbe_0",
        prompt="",
        question="Vic's testimony is admissible because the trial judge should rule that",
        choice_a="it was relevant",
        choice_b="it was hearsay",
        choice_c="both A and B",
        choice_d="none of the above",
        answer="A",
    )
    prompt = format_question_prompt(row, dataset="barexam")
    # Question should be present
    assert "Vic's testimony" in prompt
    # No leading blank / phantom prompt section
    assert prompt.strip().startswith("Vic's testimony")


def test_barexam_handles_nan_prompt():
    """A NaN value in the prompt column must not crash."""
    import numpy as np
    row = _row(
        idx="mbe_test",
        prompt=np.nan,
        question="Test question?",
        choice_a="a", choice_b="b", choice_c="c", choice_d="d",
        answer="A",
    )
    prompt = format_question_prompt(row, dataset="barexam")
    assert "Test question?" in prompt
    assert prompt.strip().startswith("Test question?")


def test_barexam_prompt_before_question():
    """If prompt is present, it must come BEFORE the question stem."""
    row = _row(
        idx="mbe_test",
        prompt="FACTPATTERN_MARKER: fact pattern here",
        question="QUESTION_MARKER: call of the question",
        choice_a="a", choice_b="b", choice_c="c", choice_d="d",
        answer="A",
    )
    prompt = format_question_prompt(row, dataset="barexam")
    fp_pos = prompt.index("FACTPATTERN_MARKER")
    q_pos = prompt.index("QUESTION_MARKER")
    assert fp_pos < q_pos, "Prompt fact pattern must precede the question stem"


def test_barexam_end_to_end_on_real_csv():
    """Full integration test on the real dataset."""
    csv_path = ROOT / "datasets/barexam_qa/qa/qa.csv"
    if not csv_path.exists():
        print("SKIP: dataset CSV not available locally")
        return
    qa = pd.read_csv(csv_path)

    # Count non-empty prompt rows + verify formatter includes the content
    nonempty_prompt = qa["prompt"].fillna("").astype(str).str.strip() != ""
    assert nonempty_prompt.sum() > 400, "Expected ~445 rows with non-empty prompt"

    # Spot-check 3 idx values
    for idx in ["mbe_1014", "mbe_366", "mbe_97"]:
        row = qa[qa["idx"] == idx].iloc[0]
        prompt = format_question_prompt(row, dataset="barexam")
        # The first 80 chars of the prompt column must appear in the formatted output
        p_head = str(row["prompt"])[:80]
        assert p_head in prompt, f"Formatted prompt for {idx} missing prompt-column content"


def test_retrieval_question_includes_prompt():
    """_retrieval_question must include the BarExam prompt column for retrieval/rerank queries."""
    import importlib.util
    # Import eval_harness's _retrieval_question without triggering the full import chain
    # (which pulls vllm/langchain etc.)
    import re as _re
    src = (ROOT / "eval" / "eval_harness.py").read_text()
    match = _re.search(
        r"(def _retrieval_question\(row: pd\.Series\) -> str:.*?)\n\ndef ",
        src,
        _re.DOTALL,
    )
    assert match, "_retrieval_question helper not found in eval_harness.py"
    ns = {"pd": pd}
    exec(match.group(1), ns)
    _retrieval_question = ns["_retrieval_question"]

    # With a prompt column — should include it
    row_with = _row(
        prompt="FACT_PATTERN_MARKER",
        question="CALL_MARKER",
    )
    out = _retrieval_question(row_with)
    assert "FACT_PATTERN_MARKER" in out
    assert "CALL_MARKER" in out
    assert out.index("FACT_PATTERN_MARKER") < out.index("CALL_MARKER")

    # Without a prompt (empty string) — should be just the question
    row_without = _row(prompt="", question="CALL_ONLY")
    assert _retrieval_question(row_without) == "CALL_ONLY"

    # With NaN prompt — should be just the question
    import numpy as np
    row_nan = _row(prompt=np.nan, question="CALL_ONLY")
    assert _retrieval_question(row_nan) == "CALL_ONLY"


def test_non_barexam_datasets_untouched():
    """Housing/casehold/legal_rag/australian should not be affected by the BarExam fix."""
    # Housing: Yes/No question format
    row = _row(
        state="California",
        question="Does the landlord have to provide 30 days notice?",
    )
    p = format_question_prompt(row, dataset="housing")
    assert "California" in p
    assert "30 days notice" in p


if __name__ == "__main__":
    count_pass = count_fail = 0
    test_funcs = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for fn in test_funcs:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            count_pass += 1
        except AssertionError as e:
            print(f"FAIL  {fn.__name__}: {e}")
            count_fail += 1
    print(f"\n{count_pass} passed, {count_fail} failed ({len(test_funcs)} total)")
    sys.exit(0 if count_fail == 0 else 1)
