#!/usr/bin/env python3
"""Summarize artifact and leakage flags from eval detail logs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _clip(text: str | None, n: int = 120) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text[:n] + ("..." if len(text) > n else "")


def _load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _contains_answer_artifact(text: str | None) -> bool:
    text = text or ""
    artifact_patterns = [
        r"(?im)^\s*(?:\*\*)?(?:final\s+)?answer(?:\*\*)?\s*:",
        r"(?i)\b(?:answer|option|choice)\s+(?:is|must be|would be)\s*\(?[A-E]\)?\b",
        # Avoid flagging ordinary prose like "it is a fair representation" as
        # option-letter leakage.
        r"(?i)\b(?:it'?s|it is)\s*(?:\([A-E]\)|[A-E](?=\s*(?:$|[.,;:!?])))",
        r"(?i)\bself-correction\b",
        r"(?im)^\s*wait[,:\s]",
    ]
    return any(re.search(pattern, text) for pattern in artifact_patterns)


def _first_text(row: dict, *keys: str) -> str:
    for key in keys:
        if key in row and row.get(key):
            return str(row.get(key, ""))
    return ""


def _row_flag(row: dict, key: str, *text_keys: str) -> bool:
    if key in row:
        return bool(row.get(key))
    return _contains_answer_artifact(_first_text(row, *text_keys))


def _collect_flagged_examples(rows: list[dict], key: str, text_keys: tuple[str, ...], nested: bool = False) -> list[tuple[str, str]]:
    examples: list[tuple[str, str]] = []
    for row in rows:
        idx = row.get("idx", "?")
        if not nested:
            if _row_flag(row, key, *text_keys):
                examples.append((idx, _clip(_first_text(row, *text_keys))))
            continue

        for item in row.get("gap_results", []) or []:
            if item and _row_flag(item, key, *text_keys):
                examples.append((idx, _clip(_first_text(item, *text_keys))))
                break
    return examples


def _has_retrieval_payload(row: dict) -> bool:
    """Return whether a row carries any retrieved evidence payload."""
    for key in ("retrieved_ids", "evidence_store", "retrieved_passages", "retrieved_contexts", "contexts", "retrieval_results"):
        value = row.get(key)
        if value:
            return True
    return False


def _has_retrieval_key(row: dict) -> bool:
    return any(
        key in row
        for key in ("retrieved_ids", "evidence_store", "retrieved_passages", "retrieved_contexts", "contexts", "retrieval_results")
    )


def _numeric(row: dict, *keys: str) -> float | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def summarize(path: Path) -> str:
    rows = _load_rows(path)
    if not rows:
        return f"{path}: empty log"

    n = len(rows)
    correct = sum(1 for r in rows if r.get("is_correct"))
    mode = rows[0].get("mode", path.stem)

    top_hyde = sum(1 for r in rows if _row_flag(r, "hyde_contains_answer_artifact", "hyde_passage_raw", "hyde_passage"))
    top_report = sum(1 for r in rows if _row_flag(r, "report_contains_answer_artifact", "report_raw", "report"))
    top_knowledge = sum(1 for r in rows if _row_flag(r, "knowledge_contains_answer_artifact", "knowledge_note_raw", "knowledge_note"))
    nested_hyde = sum(
        1 for r in rows
        if any(item and _row_flag(item, "hyde_contains_answer_artifact", "hyde_passage_raw", "hyde_passage") for item in (r.get("gap_results") or []))
    )
    nested_report = sum(
        1 for r in rows
        if any(item and _row_flag(item, "report_contains_answer_artifact", "report_raw", "report") for item in (r.get("gap_results") or []))
    )
    rows_with_formatted_question = sum(1 for r in rows if "formatted_question" in r)
    rows_with_intermediate_question = sum(1 for r in rows if "intermediate_question" in r)
    rows_with_retrieval_queries = sum(1 for r in rows if "retrieval_queries" in r)
    rows_with_final_prompt_preview = sum(1 for r in rows if "final_prompt_preview" in r)
    rows_with_call_trace = sum(1 for r in rows if "call_trace" in r)
    rows_with_trace_events = sum(1 for r in rows if "trace_events" in r)
    errors = sum(1 for r in rows if r.get("error"))
    missing_predictions = sum(1 for r in rows if not r.get("predicted_answer"))
    parse_failures = sum(1 for r in rows if r.get("parse_failed") or r.get("parse_failure"))
    retrieval_rows = [r for r in rows if _has_retrieval_key(r)]
    empty_retrieval = sum(1 for r in retrieval_rows if not _has_retrieval_payload(r))
    call_values = [_numeric(r, "llm_calls", "num_llm_calls", "n_llm_calls", "call_count") for r in rows]
    call_values = [v for v in call_values if v is not None]
    output_tokens = [_numeric(r, "output_tokens", "total_output_tokens") for r in rows]
    output_tokens = [v for v in output_tokens if v is not None]
    answer_lengths = [(r.get("idx", "?"), len(str(r.get("final_answer") or ""))) for r in rows]
    max_answer_idx, max_answer_chars = max(answer_lengths, key=lambda item: item[1])
    long_answer_examples = [(idx, chars) for idx, chars in answer_lengths if chars > 20000]

    lines = [
        f"{path}",
        f"mode={mode} rows={n} accuracy={correct}/{n}",
        f"errors={errors}",
        f"missing_predicted_answer={missing_predictions}",
        f"parse_failures={parse_failures}",
        f"retrieval_key_rows={len(retrieval_rows)}",
        f"empty_retrieval_rows={empty_retrieval}",
        f"avg_llm_calls={(sum(call_values) / len(call_values)):.2f}" if call_values else "avg_llm_calls=n/a",
        f"max_output_tokens={int(max(output_tokens))}" if output_tokens else "max_output_tokens=n/a",
        f"max_final_answer_chars={max_answer_chars} idx={max_answer_idx}",
        f"long_final_answer_rows={len(long_answer_examples)}",
        f"top_level_hyde_artifacts={top_hyde}",
        f"top_level_report_artifacts={top_report}",
        f"top_level_knowledge_artifacts={top_knowledge}",
        f"nested_gap_hyde_artifacts={nested_hyde}",
        f"nested_gap_report_artifacts={nested_report}",
        f"rows_with_formatted_question={rows_with_formatted_question}",
        f"rows_with_intermediate_question={rows_with_intermediate_question}",
        f"rows_with_retrieval_queries={rows_with_retrieval_queries}",
        f"rows_with_final_prompt_preview={rows_with_final_prompt_preview}",
        f"rows_with_call_trace={rows_with_call_trace}",
        f"rows_with_trace_events={rows_with_trace_events}",
    ]

    example_specs = [
        ("top_hyde_examples", _collect_flagged_examples(rows, "hyde_contains_answer_artifact", ("hyde_passage_raw", "hyde_passage"))),
        ("top_report_examples", _collect_flagged_examples(rows, "report_contains_answer_artifact", ("report_raw", "report"))),
        ("nested_hyde_examples", _collect_flagged_examples(rows, "hyde_contains_answer_artifact", ("hyde_passage_raw", "hyde_passage"), nested=True)),
        ("nested_report_examples", _collect_flagged_examples(rows, "report_contains_answer_artifact", ("report_raw", "report"), nested=True)),
    ]
    for label, examples in example_specs:
        if not examples:
            continue
        joined = " | ".join(f"{idx}: {snippet}" for idx, snippet in examples[:3])
        lines.append(f"{label}={joined}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize answer-artifact leakage flags in detail logs.")
    parser.add_argument("paths", nargs="+", help="Detail log paths")
    args = parser.parse_args()

    for raw in args.paths:
        print(summarize(Path(raw)))
        print()


if __name__ == "__main__":
    main()
