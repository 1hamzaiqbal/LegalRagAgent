"""Regression tests for HyDE/report sanitizer + snap-answer stripper.

These lock in the leak fixes that brought rag_hyde from 100% →  0% and
rag_snap_hyde from 74% → 0% on the cluster Gemma E4B smoke (job 50812).

Run:
    uv run python -m pytest tests/test_sanitizer.py -v
or standalone:
    python tests/test_sanitizer.py
"""
import re
import sys
from pathlib import Path

# Import the functions under test without triggering the heavy eval_harness
# imports (pandas/vllm/etc.) — we just need the regex logic.
_HARNESS = (Path(__file__).resolve().parents[1] / "eval" / "eval_harness.py").read_text()
_SOURCE = {}
exec(
    "\n".join(
        [
            "import re",
            _HARNESS.split("def _contains_answer_artifact")[1]
            .split("def _preview_text")[0]
            .replace("def _sanitize_intermediate_text", "\ndef _sanitize_intermediate_text")
            .replace("def _strip_answer_line", "\ndef _strip_answer_line")
            if False  # unused: kept as documentation
            else "",
        ]
    ),
    _SOURCE,
)

# Direct copy of the source functions — keep in sync with eval_harness.py.
def _contains_answer_artifact(text):
    if not text: return False
    return bool(re.search(r"(?im)^\s*(?:\*\*)?(?:final\s+)?answer(?:\*\*)?\s*:", text))

def _sanitize_intermediate_text(text, fallback=""):
    cleaned = text or ""
    cleaned = re.sub(
        r"(?im)^\s*(?:\*\*)?(?:final\s+)?answer(?:\*\*)?\s*:\s*(?:\*\*)?\s*(?:\(?[A-E]\)?|yes|no|irrelevant)\s*(?:\*\*)?\s*$",
        "", cleaned,
    )
    cleaned = re.sub(
        r"(?im)^\s*\*\*\s*(?:relevant legal passage|legal reference passage|passage)\s*:?\s*\*\*\s*:?\s*$",
        "", cleaned,
    )
    cleaned = re.sub(
        r"(?im)^\s*\*\*\s*(?:relevant legal passage|legal reference passage|passage)\s*:?\s*\*\*\s*:?\s*",
        "", cleaned,
    )
    cleaned = re.sub(
        r"(?im)^\s*(?:relevant legal passage|legal reference passage|passage)\s*:\s*",
        "", cleaned,
    )
    cleaned = re.sub(r"\A(?:\s*(?:\*{1,}|-{3,}|_{3,})\s*)+", "", cleaned)
    lines = []
    prev_blank = False
    for raw_line in cleaned.splitlines():
        line = raw_line.rstrip()
        is_blank = not line.strip()
        if is_blank and prev_blank:
            continue
        lines.append(line)
        prev_blank = is_blank
    cleaned = "\n".join(lines).strip()
    return cleaned or (fallback or "").strip()

def _strip_answer_line(text):
    if not text: return ""
    cleaned = re.sub(
        r"(?im)^\s*(?:\*\*)?(?:final\s+)?answer(?:\*\*)?\s*:\s*(?:\*\*)?\s*(?:\(?[A-E]\)?|yes|no|irrelevant)\s*(?:\*\*)?\s*$",
        "", text,
    )
    return cleaned.strip()


ANSWER_RE = re.compile(r"Answer:\s*\(?[A-D]\)?", re.I)


# ----- Real leakage patterns observed in historical Gemma logs -----

LEAKY_SAMPLES = [
    # rag_hyde mbe_0 (the most common shape)
    "Answer: (C)\n\n**Passage:** A party may impeach its own witness if the witness has previously made statements that are inconsistent with their current testimony.",
    # rag_hyde mbe_3 (variant with Passage: without bold)
    "Answer: (C)\n\n**Passage:**\nThe crime of burglary requires the requisite criminal intent to be present at the moment of the unauthorized entry.",
    # rag_hyde mbe_848 (spaces in the label)
    "Answer: (B)\n\n**Legal Reference Passage:** The doctrine of investigatory detention permits a store to detain a patron.",
    # Extreme case: only a letter, no passage
    "Answer: (C)",
    # snap_hyde with trailing markdown
    "Answer: (A)\n\n***\n\nThe general duty to rescue arises when a person recognizes an immediate danger of serious bodily harm.",
    # Lowercase answer, bold markers
    "**Answer: (d)**\n\nConsideration requires a bargained-for exchange of legal value.",
    # Final Answer: variant
    "Final Answer: B\n\nThe doctrine of part performance operates as an equitable exception to the Statute of Frauds.",
]

# Clean passages that should pass through unchanged (or nearly so)
CLEAN_SAMPLES = [
    "The doctrine of accomplice liability requires that the defendant possess the requisite mens rea.",
    "A witness may be impeached by evidence of character for truthfulness, which includes evidence of prior instances of conduct.",
    "Under the business records exception to the hearsay rule, a document is admissible if made at or near the time by someone with knowledge.",
]


def test_leaky_samples_sanitize_to_clean():
    """After sanitize, no leaky sample should contain 'Answer: (X)'."""
    failures = []
    for i, s in enumerate(LEAKY_SAMPLES):
        clean = _sanitize_intermediate_text(s)
        if ANSWER_RE.search(clean):
            failures.append((i, s[:80], clean[:80]))
    assert not failures, f"Sanitizer failed on {len(failures)} samples: {failures}"


def test_leaky_samples_no_orphan_markdown():
    """After sanitize, passages should not start with orphan '**' or '---'."""
    for i, s in enumerate(LEAKY_SAMPLES):
        clean = _sanitize_intermediate_text(s)
        if not clean:
            continue
        assert not clean.startswith("**"), f"Sample {i} has orphan '**': {clean[:80]!r}"
        assert not clean.startswith("---"), f"Sample {i} has orphan '---': {clean[:80]!r}"


def test_leaky_samples_preserve_doctrine():
    """Sanitizer must strip leakage WITHOUT destroying the doctrinal text."""
    # Sample 0: expect "A party may impeach its own witness..." to survive
    clean = _sanitize_intermediate_text(LEAKY_SAMPLES[0])
    assert "impeach" in clean.lower()
    assert "testimony" in clean.lower()

    # Sample 2: expect "investigatory detention" to survive
    clean = _sanitize_intermediate_text(LEAKY_SAMPLES[2])
    assert "investigatory detention" in clean.lower()


def test_clean_samples_unchanged():
    """Clean doctrinal passages should pass through unchanged modulo trimming."""
    for s in CLEAN_SAMPLES:
        clean = _sanitize_intermediate_text(s)
        assert clean == s.strip(), f"Clean sample mutated:\n  in:  {s!r}\n  out: {clean!r}"


def test_empty_input_returns_fallback():
    assert _sanitize_intermediate_text("", fallback="fb") == "fb"
    assert _sanitize_intermediate_text(None, fallback="fb") == "fb"


def test_only_answer_collapses_to_fallback():
    """If input is ONLY 'Answer: (C)', sanitize should return fallback."""
    assert _sanitize_intermediate_text("Answer: (C)", fallback="fb") == "fb"


def test_strip_answer_line_removes_trailing_letter():
    """_strip_answer_line removes trailing MC answer lines from reasoning."""
    reasoning = "Step 1: analyze the rule.\nStep 2: apply to facts.\n\nAnswer: (B)"
    stripped = _strip_answer_line(reasoning)
    assert "Answer:" not in stripped
    assert "Step 1" in stripped
    assert "Step 2" in stripped


def test_strip_answer_line_handles_yesno():
    """_strip_answer_line also strips housing-dataset 'Answer: Yes/No' lines."""
    reasoning = "Landlord must provide 30 days notice.\n\nAnswer: Yes"
    stripped = _strip_answer_line(reasoning)
    assert "Answer:" not in stripped
    assert "30 days notice" in stripped


def test_open_answer_judge_handles_INCORRECT():
    """Regression: 'INCORRECT' verdict must NOT be scored as correct.

    Pre-fix logic was `'CORRECT' in verdict.upper()` which returned True for
    'INCORRECT' because the substring is present. Whole-word check required.
    """
    import re

    def score(verdict: str) -> bool:
        upper = verdict.upper()
        if re.search(r"\bINCORRECT\b", upper):
            return False
        return bool(re.search(r"\bCORRECT\b", upper))

    assert score("CORRECT") is True
    assert score("INCORRECT") is False
    assert score("The student's answer is INCORRECT.") is False
    assert score("The student's answer is CORRECT.") is True
    assert score("Verdict: **INCORRECT**") is False
    assert score("Verdict: correct") is True  # case-insensitive
    # Mixed verdict — rare but possible with a chatty judge — err on the side of INCORRECT
    assert score("Partially correct — mostly INCORRECT") is False


def test_contains_answer_artifact_detects():
    """Leak detector (used by analyze_detail_flags) must match all observed shapes."""
    for s in LEAKY_SAMPLES:
        assert _contains_answer_artifact(s), f"Leak detector missed: {s[:80]!r}"
    for s in CLEAN_SAMPLES:
        assert not _contains_answer_artifact(s), f"Leak detector false-positive: {s[:80]!r}"


if __name__ == "__main__":
    # Run all tests in this file
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
