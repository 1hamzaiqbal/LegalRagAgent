"""Offline eval metrics shared by research-report scripts.

The focus here is post-hoc analysis of detail JSONL logs. These functions avoid
provider calls and only compute metrics that are present or derivable from the
records already written by ``eval_harness.py``.
"""
from __future__ import annotations

import json
import math
import re
import statistics
import string
from collections import Counter
from pathlib import Path
from typing import Any


CLOSED_SET_DATASETS = {"barexam", "housing", "casehold", "legalbench_scalr"}
GENERATED_CONTEXT_FIELDS = (
    "hyde_passage",
    "hyde_passage_raw",
    "hyde_passages",
    "hyde_passages_raw",
    "snap_and_hyde_raw",
    "snap_answer",
    "planning_table",
    "subagent_reports",
    "reports",
    "drafts",
    "answer_drafts",
    "draft_candidates",
    "spec_drafts",
)
DRAFT_FIELDS = ("drafts", "answer_drafts", "draft_candidates", "spec_drafts")
SPEC_SCORE_FIELDS = (
    "draft_score",
    "rho_draft",
    "rhoDraft",
    "self_containment_score",
    "rho_self_contain",
    "rhoSelfContain",
    "self_reflection_score",
    "rho_self_reflect",
    "rhoSelfReflect",
)
TEXT_KEYS = (
    "text",
    "content",
    "passage",
    "rationale",
    "finding",
    "report",
    "answer",
    "response",
    "summary",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if isinstance(value, dict):
                rows.append(value)
    if not rows:
        raise ValueError(f"{path}: no records loaded")
    return rows


def normalize_text(value: Any) -> str:
    """SQuAD-style normalization for answer matching."""
    text = str(value or "").lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [token for token in text.split() if token not in {"a", "an", "the"}]
    return " ".join(tokens)


def approx_token_count(value: Any) -> int:
    return len(re.findall(r"\S+", str(value or "")))


def _flatten_texts(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, dict):
        texts: list[str] = []
        for key in TEXT_KEYS:
            item = value.get(key)
            if item is not None:
                texts.extend(_flatten_texts(item))
        if texts:
            return texts
        for item in value.values():
            texts.extend(_flatten_texts(item))
        return texts
    if isinstance(value, (list, tuple, set)):
        texts = []
        for item in value:
            texts.extend(_flatten_texts(item))
        return texts
    return [str(value)] if value else []


def parse_aliases(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("[") or stripped.startswith("{"):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                return [stripped]
            return parse_aliases(parsed)
        return [part.strip() for part in re.split(r"\s*\|\s*|\s*;\s*", stripped) if part.strip()]
    if isinstance(value, dict):
        aliases: list[str] = []
        for item in value.values():
            aliases.extend(parse_aliases(item))
        return aliases
    if isinstance(value, (list, tuple, set)):
        aliases = []
        for item in value:
            aliases.extend(parse_aliases(item))
        return aliases
    return [str(value)]


def token_f1(prediction: Any, gold: Any) -> float:
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def correct_flag(record: dict[str, Any]) -> bool:
    if "is_correct" in record:
        return bool(record.get("is_correct"))
    if "em" in record:
        return bool(record.get("em"))
    if "judge_score" in record:
        return bool(record.get("judge_score"))
    predicted = record.get("predicted_answer")
    gold = record.get("correct_answer")
    return normalize_text(predicted) == normalize_text(gold) if predicted is not None and gold is not None else False


def answer_contains_gold(record: dict[str, Any]) -> bool | None:
    """Speculative-RAG-style free-form answer containment.

    Closed-set tasks use accuracy instead, so this returns ``None`` there.
    """
    dataset = str(record.get("dataset") or "").lower()
    if dataset in CLOSED_SET_DATASETS:
        return None

    gold_values = parse_aliases(record.get("correct_answer") or record.get("answer") or record.get("gold_answer"))
    for field in ("aliases_used", "answer_aliases", "aliases"):
        gold_values.extend(parse_aliases(record.get(field)))
    gold_values = [value for value in dict.fromkeys(gold_values) if normalize_text(value)]
    if not gold_values:
        return None

    prediction_text = "\n".join(
        text
        for field in ("final_answer", "predicted_answer", "answer")
        for text in _flatten_texts(record.get(field))
    )
    normalized_prediction = normalize_text(prediction_text)
    if not normalized_prediction:
        return False
    return any(normalize_text(gold) in normalized_prediction for gold in gold_values)


def evidence_doc_count(record: dict[str, Any]) -> int:
    evidence = record.get("evidence_store")
    if isinstance(evidence, list):
        return len(evidence)
    if isinstance(evidence, dict):
        return len(evidence)
    retrieved_ids = record.get("retrieved_ids")
    if isinstance(retrieved_ids, (list, tuple, set)):
        return len(retrieved_ids)
    if isinstance(retrieved_ids, str) and retrieved_ids.strip():
        return 1
    return 0


def evidence_text(record: dict[str, Any]) -> str:
    return "\n".join(_flatten_texts(record.get("evidence_store")))


def generated_context_text(record: dict[str, Any]) -> str:
    texts: list[str] = []
    for field in GENERATED_CONTEXT_FIELDS:
        texts.extend(_flatten_texts(record.get(field)))
    return "\n".join(texts)


def draft_count(record: dict[str, Any]) -> int:
    for field in DRAFT_FIELDS:
        value = record.get(field)
        if isinstance(value, (list, tuple, set)):
            return len(value)
        if isinstance(value, dict):
            return len(value)
    return 0


def has_speculative_score(record: dict[str, Any]) -> bool:
    return any(field in record and record.get(field) is not None for field in SPEC_SCORE_FIELDS)


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize_records(rows: list[dict[str, Any]], label: str | None = None, path: Path | None = None) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize zero records")
    first = rows[0]
    n = len(rows)
    correct = sum(1 for row in rows if correct_flag(row))
    contains_values = [value for row in rows if (value := answer_contains_gold(row)) is not None]
    f1_values = [float(row["f1"]) for row in rows if isinstance(row.get("f1"), (int, float))]
    em_values = [float(row["em"]) for row in rows if isinstance(row.get("em"), (int, float, bool))]
    judge_values = [float(row["judge_score"]) for row in rows if isinstance(row.get("judge_score"), (int, float, bool))]
    latencies = [float(row["elapsed_sec"]) for row in rows if isinstance(row.get("elapsed_sec"), (int, float))]
    calls = [float(row.get("llm_calls", 0) or 0) for row in rows]
    input_tokens = [float(row.get("input_tokens", 0) or 0) for row in rows]
    output_tokens = [float(row.get("output_tokens", 0) or 0) for row in rows]
    evidence_docs = [float(evidence_doc_count(row)) for row in rows]
    evidence_tokens = [float(approx_token_count(evidence_text(row))) for row in rows]
    generated_tokens = [float(approx_token_count(generated_context_text(row))) for row in rows]
    retrieved_rows = [row for row in rows if evidence_doc_count(row) > 0]
    empty_retrieval_rows = [
        row
        for row in rows
        if ("evidence_store" in row or "retrieved_ids" in row) and evidence_doc_count(row) == 0
    ]
    gold_values = [bool(row.get("gold_retrieved")) for row in rows if "gold_retrieved" in row]
    avg_evidence_tokens = _mean(evidence_tokens)
    avg_generated_tokens = _mean(generated_tokens)

    return {
        "label": label or str(first.get("label") or ""),
        "path": str(path) if path else "",
        "dataset": first.get("dataset", "-"),
        "provider": first.get("provider", "-"),
        "mode": first.get("mode", "-"),
        "n": n,
        "correct": correct,
        "accuracy": correct / n,
        "contains_gold_rate": (sum(1 for value in contains_values if value) / len(contains_values)) if contains_values else None,
        "contains_gold_n": len(contains_values),
        "avg_f1": _mean(f1_values),
        "em_rate": _mean(em_values),
        "judge_score_rate": _mean(judge_values),
        "gold_hit_rate": (sum(1 for value in gold_values if value) / len(gold_values)) if gold_values else None,
        "gold_hit_n": len(gold_values),
        "retrieval_row_rate": len(retrieved_rows) / n,
        "empty_retrieval_rate": len(empty_retrieval_rows) / n,
        "avg_evidence_docs": _mean(evidence_docs),
        "avg_evidence_tokens": avg_evidence_tokens,
        "avg_generated_context_tokens": avg_generated_tokens,
        "generated_to_evidence_token_ratio": (
            avg_generated_tokens / avg_evidence_tokens
            if avg_generated_tokens is not None and avg_evidence_tokens not in (None, 0)
            else None
        ),
        "avg_llm_calls": _mean(calls),
        "avg_latency_sec": _mean(latencies),
        "p50_latency_sec": _percentile(latencies, 0.5),
        "p95_latency_sec": _percentile(latencies, 0.95),
        "avg_input_tokens": _mean(input_tokens),
        "avg_output_tokens": _mean(output_tokens),
        "avg_total_tokens": _mean([i + o for i, o in zip(input_tokens, output_tokens, strict=False)]),
        "avg_draft_count": _mean([float(draft_count(row)) for row in rows]),
        "spec_score_row_rate": sum(1 for row in rows if has_speculative_score(row)) / n,
    }
