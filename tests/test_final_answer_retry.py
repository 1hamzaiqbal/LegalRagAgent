from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from eval_config import EvalConfig  # noqa: E402
import eval_harness  # noqa: E402


def _scalr_row() -> pd.Series:
    return pd.Series(
        {
            "idx": "scalr_test",
            "question": "Which holding is being referenced?",
            "choice_a": "holding alpha",
            "choice_b": "holding beta",
            "choice_c": "holding gamma",
            "choice_d": "holding delta",
            "choice_e": "holding epsilon",
            "answer": "A",
        }
    )


def test_llm_only_near_cap_response_gets_logged_format_retry(monkeypatch) -> None:
    config = EvalConfig(mode="llm_only", dataset="legalbench_scalr")
    result: dict = {}
    metrics = {"count": 1, "input_tokens": 100, "output_tokens": 2040}
    calls: list[tuple[str, str, str]] = []

    def fake_metrics() -> dict:
        return dict(metrics)

    def fake_llm_call(system: str, user: str, label: str = "") -> str:
        calls.append((system, user, label))
        metrics["count"] = 2
        metrics["output_tokens"] = 2050
        return "Answer: (A)"

    monkeypatch.setenv("EVAL_FINAL_FORMAT_RETRY", "1")
    monkeypatch.setenv("LLM_MAX_COMPLETION_TOKENS", "2048")
    monkeypatch.setenv("EVAL_OUTPUT_TOKEN_MARGIN", "16")
    monkeypatch.setattr(eval_harness, "_get_metrics", fake_metrics)
    monkeypatch.setattr(eval_harness, "_llm_call", fake_llm_call)

    answer, predicted = eval_harness._maybe_retry_final_answer_format(
        _scalr_row(),
        config,
        result,
        "Initial response was long but ended.\nAnswer: (A)",
        "A",
    )

    assert answer == "Answer: (A)"
    assert predicted == "A"
    assert result["answer_format_retry"] is True
    assert result["answer_format_retry_reason"] == "near_completion_cap"
    assert result["answer_format_retry_output_tokens"] == 10
    assert result["answer_format_retry_mode"] == "format_existing_prediction"
    assert result["answer_format_retry_target_line"] == "Answer: (A)"
    assert result["answer_format_retry_valid"] is True
    assert calls and calls[0][2] == "llm_only/answer_format_retry"
    assert "Return exactly this line" in calls[0][1]
    assert "## Question" not in calls[0][1]


def test_llm_only_missing_marker_response_gets_format_retry(monkeypatch) -> None:
    config = EvalConfig(mode="llm_only", dataset="legalbench_scalr")
    result: dict = {}
    metrics = {"count": 1, "input_tokens": 100, "output_tokens": 100}

    def fake_metrics() -> dict:
        return dict(metrics)

    def fake_llm_call(system: str, user: str, label: str = "") -> str:
        metrics["count"] = 2
        metrics["output_tokens"] = 112
        return "Answer: (A)"

    monkeypatch.setenv("EVAL_FINAL_FORMAT_RETRY", "1")
    monkeypatch.setenv("LLM_MAX_COMPLETION_TOKENS", "2048")
    monkeypatch.setattr(eval_harness, "_get_metrics", fake_metrics)
    monkeypatch.setattr(eval_harness, "_llm_call", fake_llm_call)

    answer, predicted = eval_harness._maybe_retry_final_answer_format(
        _scalr_row(),
        config,
        result,
        "The answer is A, but the explicit marker is missing.",
        "A",
    )

    assert answer.endswith("Answer: (A)")
    assert predicted == "A"
    assert result["answer_format_retry_reason"] == "missing_marker"
    assert result["answer_format_retry_mode"] == "format_existing_prediction"
    assert result["answer_format_retry_output_tokens"] == 12
    assert result["answer_format_retry_valid"] is True


def test_non_exact_final_answer_line_gets_format_retry(monkeypatch) -> None:
    config = EvalConfig(mode="llm_only", dataset="legalbench_scalr")
    result: dict = {}
    metrics = {"count": 1, "input_tokens": 100, "output_tokens": 100}

    def fake_metrics() -> dict:
        return dict(metrics)

    def fake_llm_call(system: str, user: str, label: str = "") -> str:
        metrics["count"] = 2
        metrics["output_tokens"] = 105
        return "Answer: (B)"

    monkeypatch.setenv("EVAL_FINAL_FORMAT_RETRY", "1")
    monkeypatch.setenv("LLM_MAX_COMPLETION_TOKENS", "2048")
    monkeypatch.setattr(eval_harness, "_get_metrics", fake_metrics)
    monkeypatch.setattr(eval_harness, "_llm_call", fake_llm_call)

    answer, predicted = eval_harness._maybe_retry_final_answer_format(
        _scalr_row(),
        config,
        result,
        "Reasoning text.\n\n**Final Answer**: The closest choice is **(B)**, but",
        "B",
    )

    assert answer == "Answer: (B)"
    assert predicted == "B"
    assert result["answer_format_retry_reason"] == "missing_marker"
    assert result["answer_format_retry_mode"] == "format_existing_prediction"
    assert result["answer_format_retry_valid"] is True


def test_exact_final_answer_line_overrides_earlier_answer_mentions(monkeypatch) -> None:
    config = EvalConfig(mode="rag_simple", dataset="legalbench_scalr")
    result: dict = {}
    metrics = {"count": 1, "input_tokens": 100, "output_tokens": 100}
    calls: list[tuple[str, str, str]] = []

    def fake_metrics() -> dict:
        return dict(metrics)

    def fake_llm_call(system: str, user: str, label: str = "") -> str:
        calls.append((system, user, label))
        return "Answer: (E)"

    answer_text = "An earlier parser path selected A.\n\nFinal choice.\nAnswer: (E)"

    monkeypatch.setenv("EVAL_FINAL_FORMAT_RETRY", "1")
    monkeypatch.setenv("LLM_MAX_COMPLETION_TOKENS", "2048")
    monkeypatch.setattr(eval_harness, "_get_metrics", fake_metrics)
    monkeypatch.setattr(eval_harness, "_llm_call", fake_llm_call)

    answer, predicted = eval_harness._maybe_retry_final_answer_format(
        _scalr_row(),
        config,
        result,
        answer_text,
        "A",
    )

    assert answer == answer_text
    assert predicted == "E"
    assert "answer_format_retry" not in result
    assert calls == []


def test_no_silent_guard_blocks_near_cap_retry(monkeypatch) -> None:
    config = EvalConfig(mode="llm_only", dataset="legalbench_scalr")
    monkeypatch.setenv("LLM_MAX_COMPLETION_TOKENS", "2048")
    monkeypatch.setenv("EVAL_OUTPUT_TOKEN_MARGIN", "16")

    violations = eval_harness._fallback_guard_violations(
        {
            "final_answer": "Answer: (A)",
            "llm_calls": 2,
            "output_tokens": 4090,
            "answer_format_retry_output_tokens": 2040,
        },
        config,
    )

    assert any("answer_format_retry_output_tokens=2040" in item for item in violations)


def test_snap_hyre_generation_missing_passage_gets_logged_retry(monkeypatch) -> None:
    config = EvalConfig(mode="snap_hyre", dataset="legalbench_scalr")
    calls: list[tuple[str, str, str]] = []

    def fake_llm_call(system: str, user: str, label: str = "") -> str:
        calls.append((system, user, label))
        if len(calls) == 1:
            return "Reasoning that commits to beta.\nAnswer: (B)"
        return (
            "Reasoning that commits to beta.\nAnswer: (B)\n\n"
            "## Passage\n"
            "Courts compare the cited context to the controlling holding. "
            "The reference passage states the legal rule without option labels."
        )

    monkeypatch.setenv("EVAL_GENERATION_FORMAT_RETRY", "1")
    monkeypatch.setattr(eval_harness, "_llm_call", fake_llm_call)

    raw, snap, passage, parse_ok, meta = eval_harness._generate_snap_hyre_blocks(
        config,
        question="Which holding is being referenced?",
        fallback_passage="fallback question text",
        label="snap_hyre/snap_and_hyre",
    )

    assert parse_ok is True
    assert raw.startswith("Reasoning that commits")
    assert snap.endswith("Answer: (B)")
    assert "reference passage states the legal rule" in passage
    assert meta["snap_hyre_format_retry"] is True
    assert meta["snap_hyre_format_retry_reasons"] == ["missing_passage_block"]
    assert meta["snap_hyre_format_retry_target_line"] == "Answer: (B)"
    assert calls[1][2] == "snap_hyre/snap_and_hyre/format_retry"
    assert "preserve it exactly" in calls[1][1]


def test_snap_hyre_generation_missing_snap_answer_gets_logged_retry(monkeypatch) -> None:
    config = EvalConfig(mode="snap_hyre", dataset="legalbench_scalr")
    calls: list[tuple[str, str, str]] = []

    def fake_llm_call(system: str, user: str, label: str = "") -> str:
        calls.append((system, user, label))
        if len(calls) == 1:
            return (
                "Reasoning that refuses the candidate labels.\n"
                "Answer: (None of the above)\n\n"
                "## Passage\n"
                "Courts compare the cited context to the controlling holding. "
                "The passage states the rule without option labels."
            )
        return (
            "Reasoning that commits to beta.\n"
            "Answer: (B)\n\n"
            "## Passage\n"
            "Courts compare the cited context to the controlling holding. "
            "The retry passage states the legal rule without option labels."
        )

    monkeypatch.setenv("EVAL_GENERATION_FORMAT_RETRY", "1")
    monkeypatch.setattr(eval_harness, "_llm_call", fake_llm_call)

    raw, snap, passage, parse_ok, meta = eval_harness._generate_snap_hyre_blocks(
        config,
        question="Which holding is being referenced?",
        fallback_passage="fallback question text",
        label="snap_hyre/snap_and_hyre",
    )

    assert parse_ok is True
    assert raw.startswith("Reasoning that commits")
    assert snap.endswith("Answer: (B)")
    assert "retry passage states the legal rule" in passage
    assert meta["snap_hyre_format_retry"] is True
    assert meta["snap_hyre_format_retry_reasons"] == ["missing_snap_answer_line"]
    assert calls[1][2] == "snap_hyre/snap_and_hyre/format_retry"


def test_no_silent_guard_blocks_snap_hyre_missing_snap_final_line() -> None:
    config = EvalConfig(mode="snap_hyre", dataset="legalbench_scalr")

    violations = eval_harness._fallback_guard_violations(
        {
            "final_answer": "Answer: (A)",
            "snap_answer": "Reasoning.\nAnswer: (None of the above)",
            "snap_hyre_parse_ok": True,
            "retrieval_cache_hit": True,
            "hyre_cache_hit": True,
        },
        config,
    )

    assert "snap_answer_missing_required_final_line" in violations


def test_rag_rewrite_parse_failure_gets_logged_format_retry(monkeypatch) -> None:
    calls: list[tuple[str, str, str]] = []

    def fake_llm_call(system: str, user: str, label: str = "") -> str:
        calls.append((system, user, label))
        if len(calls) == 1:
            return "Here are some useful search terms: due process notice hearing"
        return (
            '{"primary":"procedural due process notice hearing",'
            '"alternatives":["due process elements deprivation","notice opportunity to be heard"]}'
        )

    monkeypatch.setenv("EVAL_GENERATION_FORMAT_RETRY", "1")
    monkeypatch.setenv("NO_SILENT_FALLBACK", "1")
    monkeypatch.setattr(eval_harness, "_llm_call", fake_llm_call)

    queries, meta = eval_harness._rewrite_query_with_meta(
        "What process is due before deprivation?",
        label="rag_rewrite/rewrite",
    )

    assert queries == [
        "procedural due process notice hearing",
        "due process elements deprivation",
        "notice opportunity to be heard",
    ]
    assert meta["rewrite_parse_ok"] is True
    assert meta["rewrite_format_retry"] is True
    assert meta["rewrite_format_retry_reasons"] == ["invalid_json"]
    assert meta["rewrite_format_retry_valid"] is True
    assert meta["rewrite_used_fallback"] is False
    assert calls[1][2] == "rag_rewrite/rewrite/format_retry"
    assert "Previous malformed output" in calls[1][1]


def test_rag_rewrite_partial_json_repair_uses_model_queries(monkeypatch) -> None:
    calls: list[tuple[str, str, str]] = []

    def fake_llm_call(system: str, user: str, label: str = "") -> str:
        calls.append((system, user, label))
        return (
            '```json\n{"primary":"340B statutory silence private right",'
            '"alternatives":["third party beneficiary federal common law",'
            '"contract enforcement congressional intent",'
            '"statutory silence statutory silence statutory silence'
        )

    monkeypatch.setenv("EVAL_GENERATION_FORMAT_RETRY", "1")
    monkeypatch.setenv("NO_SILENT_FALLBACK", "1")
    monkeypatch.setattr(eval_harness, "_llm_call", fake_llm_call)

    queries, meta = eval_harness._rewrite_query_with_meta(
        "Can federal common law create a 340B private right?",
        label="rag_rewrite/rewrite",
    )

    assert queries == [
        "340B statutory silence private right",
        "third party beneficiary federal common law",
        "contract enforcement congressional intent",
    ]
    assert meta["rewrite_parse_ok"] is True
    assert meta["rewrite_parse_kind"] == "partial_json"
    assert meta["rewrite_partial_json_repair"] is True
    assert meta["rewrite_format_retry"] is False
    assert meta["rewrite_used_fallback"] is False
    assert [call[2] for call in calls] == ["rag_rewrite/rewrite"]


def test_rag_rewrite_partial_json_repair_handles_multiline_strings(monkeypatch) -> None:
    calls: list[tuple[str, str, str]] = []

    def fake_llm_call(system: str, user: str, label: str = "") -> str:
        calls.append((system, user, label))
        return (
            "```json\n"
            "{\n"
            '  "primary": "(\n'
            "    police arrest warrant AND deadly force justification\n"
            "  )\",\n"
            '  "alternatives": [\n'
            '    "(\n'
            "      felony arrest resistance AND necessity defense\n"
            "    )\",\n"
            '    "Fourth Amendment excessive force standards"\n'
            "  ]\n"
            "}\n"
            "```"
        )

    monkeypatch.setenv("EVAL_GENERATION_FORMAT_RETRY", "1")
    monkeypatch.setenv("NO_SILENT_FALLBACK", "1")
    monkeypatch.setattr(eval_harness, "_llm_call", fake_llm_call)

    queries, meta = eval_harness._rewrite_query_with_meta(
        "Was deadly force justified during an arrest?",
        label="rag_rewrite/rewrite",
    )

    assert queries == [
        "(\n    police arrest warrant AND deadly force justification\n  )",
        "(\n      felony arrest resistance AND necessity defense\n    )",
        "Fourth Amendment excessive force standards",
    ]
    assert meta["rewrite_parse_ok"] is True
    assert meta["rewrite_parse_kind"] == "partial_json"
    assert meta["rewrite_partial_json_repair"] is True
    assert meta["rewrite_format_retry"] is False
    assert meta["rewrite_used_fallback"] is False
    assert [call[2] for call in calls] == ["rag_rewrite/rewrite"]


def test_rag_rewrite_partial_json_repair_handles_invalid_apostrophe_escapes(monkeypatch) -> None:
    calls: list[tuple[str, str, str]] = []

    def fake_llm_call(system: str, user: str, label: str = "") -> str:
        calls.append((system, user, label))
        return (
            "```json\n"
            "{\n"
            "  \"primary\": \"('Dunn\\'s negligence' OR 'intervening cause') AND 'proximate cause'\",\n"
            "  \"alternatives\": [\n"
            "    \"('manufacturer\\'s duty' OR 'product defect') AND 'foreseeability'\",\n"
            "    \"Restatement Second Torts section 443 superseding cause\"\n"
            "  ]\n"
            "}\n"
            "```"
        )

    monkeypatch.setenv("EVAL_GENERATION_FORMAT_RETRY", "1")
    monkeypatch.setenv("NO_SILENT_FALLBACK", "1")
    monkeypatch.setattr(eval_harness, "_llm_call", fake_llm_call)

    queries, meta = eval_harness._rewrite_query_with_meta(
        "Did Dunn's negligence supersede manufacturer liability?",
        label="rag_rewrite/rewrite",
    )

    assert queries == [
        "('Dunn's negligence' OR 'intervening cause') AND 'proximate cause'",
        "('manufacturer's duty' OR 'product defect') AND 'foreseeability'",
        "Restatement Second Torts section 443 superseding cause",
    ]
    assert meta["rewrite_parse_ok"] is True
    assert meta["rewrite_parse_kind"] == "partial_json"
    assert meta["rewrite_partial_json_repair"] is True
    assert meta["rewrite_format_retry"] is False
    assert meta["rewrite_used_fallback"] is False
    assert [call[2] for call in calls] == ["rag_rewrite/rewrite"]


def test_rag_rewrite_retry_partial_json_repair_is_logged(monkeypatch) -> None:
    calls: list[tuple[str, str, str]] = []

    def fake_llm_call(system: str, user: str, label: str = "") -> str:
        calls.append((system, user, label))
        if len(calls) == 1:
            return "not json"
        return (
            '{"primary":"Hague Convention ne exeat custody right",'
            '"alternatives":["wrongful removal custody rights",'
            '"parental consent child abduction",'
            '"jurisdiction jurisdiction jurisdiction'
        )

    monkeypatch.setenv("EVAL_GENERATION_FORMAT_RETRY", "1")
    monkeypatch.setenv("NO_SILENT_FALLBACK", "1")
    monkeypatch.setattr(eval_harness, "_llm_call", fake_llm_call)

    queries, meta = eval_harness._rewrite_query_with_meta(
        "Does a ne exeat clause confer Hague custody rights?",
        label="rag_rewrite/rewrite",
    )

    assert queries == [
        "Hague Convention ne exeat custody right",
        "wrongful removal custody rights",
        "parental consent child abduction",
    ]
    assert meta["rewrite_parse_ok"] is True
    assert meta["rewrite_format_retry"] is True
    assert meta["rewrite_format_retry_valid"] is True
    assert meta["rewrite_format_retry_parse_kind"] == "partial_json"
    assert meta["rewrite_partial_json_repair"] is True
    assert meta["rewrite_used_fallback"] is False
    assert [call[2] for call in calls] == [
        "rag_rewrite/rewrite",
        "rag_rewrite/rewrite/format_retry",
    ]


def test_rag_rewrite_strict_parse_failure_does_not_fallback(monkeypatch) -> None:
    calls: list[tuple[str, str, str]] = []

    def fake_llm_call(system: str, user: str, label: str = "") -> str:
        calls.append((system, user, label))
        return "not json"

    monkeypatch.setenv("EVAL_GENERATION_FORMAT_RETRY", "1")
    monkeypatch.setenv("NO_SILENT_FALLBACK", "1")
    monkeypatch.setattr(eval_harness, "_llm_call", fake_llm_call)

    with pytest.raises(RuntimeError, match="NO_SILENT_FALLBACK blocked rag_rewrite"):
        eval_harness._rewrite_query_with_meta("What process is due?", label="rag_rewrite/rewrite")

    assert [call[2] for call in calls] == [
        "rag_rewrite/rewrite",
        "rag_rewrite/rewrite/format_retry",
    ]


def test_no_silent_guard_blocks_rewrite_parse_flag() -> None:
    config = EvalConfig(mode="rag_rewrite", dataset="legalbench_scalr")

    violations = eval_harness._fallback_guard_violations(
        {
            "final_answer": "Answer: (A)",
            "rewrite_parse_ok": False,
        },
        config,
    )

    assert "rewrite_parse_ok=False" in violations


def test_no_silent_guard_blocks_non_exact_final_line() -> None:
    config = EvalConfig(mode="rag_simple", dataset="legalbench_scalr")

    violations = eval_harness._fallback_guard_violations(
        {
            "final_answer": "**Final Answer**: The closest choice is **(B)**, but",
            "predicted_answer": "B",
        },
        config,
    )

    assert "missing_required_final_answer_line" in violations
