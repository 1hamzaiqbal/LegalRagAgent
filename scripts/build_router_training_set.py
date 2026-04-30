#!/usr/bin/env python3
"""Build an offline training table for bottleneck-aware method routing.

Each input arm is a completed detail log. The script joins arms by question ID,
extracts cheap task features from the question record, appends per-arm
correctness/cost/retrieval fields, and labels the row with an oracle arm under
a configurable cost-aware reward.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import compute_mcnemar  # type: ignore  # noqa: E402

_field_limit = sys.maxsize
while True:
    try:
        csv.field_size_limit(_field_limit)
        break
    except OverflowError:
        _field_limit //= 10


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    if not rows:
        raise SystemExit(f"{path}: no records loaded")
    return rows


def resolve_arm(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise SystemExit(f"Invalid --arm {raw!r}; expected label=path")
    label, pattern = raw.split("=", 1)
    label = label.strip()
    if not label:
        raise SystemExit(f"Invalid --arm {raw!r}; empty label")
    matches = sorted(glob.glob(pattern))
    if not matches and Path(pattern).exists():
        matches = [pattern]
    if len(matches) != 1:
        raise SystemExit(f"{raw!r}: expected exactly one path, matched {len(matches)}")
    return label, Path(matches[0])


def choose_key(rows_by_arm: dict[str, list[dict[str, Any]]], requested: str | None) -> str:
    labels = list(rows_by_arm)
    if requested:
        for label, rows in rows_by_arm.items():
            values = [row.get(requested) for row in rows]
            if any(value is None for value in values) or len(set(values)) != len(values):
                raise SystemExit(f"{label}: key {requested!r} is missing or non-unique")
        return requested
    key = compute_mcnemar.choose_key_field(rows_by_arm[labels[0]], rows_by_arm[labels[1]], None)
    for label in labels[2:]:
        compute_mcnemar.choose_key_field(rows_by_arm[labels[0]], rows_by_arm[label], key)
    return key


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def choices_from_row(row: dict[str, Any]) -> list[str]:
    raw = row.get("choices")
    if isinstance(raw, list):
        return [str(item) for item in raw]
    if isinstance(raw, dict):
        return [str(raw[key]) for key in sorted(raw)]
    choices = []
    for letter in ("a", "b", "c", "d", "e"):
        value = row.get(f"choice_{letter}")
        if value:
            choices.append(str(value))
    return choices


def token_set(text: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[A-Za-z0-9_]+", text) if len(token) > 1}


def overlap_ratio(left: str, right: str) -> float:
    left_tokens = token_set(left)
    if not left_tokens:
        return 0.0
    right_tokens = token_set(right)
    return len(left_tokens & right_tokens) / len(left_tokens)


def answer_format(row: dict[str, Any], choices: list[str]) -> str:
    dataset = str(row.get("dataset", ""))
    if choices:
        return f"mc{len(choices)}"
    if dataset == "housing":
        return "yes_no"
    if dataset == "musique":
        return "short_span"
    if dataset in {"legal_rag", "australian"}:
        return "long_form"
    return "unknown"


def static_features(row: dict[str, Any]) -> dict[str, Any]:
    question = str(row.get("formatted_question") or row.get("question") or "")
    intermediate = str(row.get("intermediate_question") or "")
    choices = choices_from_row(row)
    tokens = re.findall(r"[A-Za-z0-9_]+", question)
    named_entities = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", question)
    dates_numbers = re.findall(r"\b(?:19|20)\d{2}\b|\b\d+(?:\.\d+)?\b", question)
    legal_terms = re.findall(
        r"\b(?:statute|holding|case|court|contract|tort|evidence|jurisdiction|"
        r"liability|damages|tenant|landlord|plaintiff|defendant|precedent)\b",
        question,
        flags=re.IGNORECASE,
    )
    hop_terms = re.findall(
        r"\b(?:whose|after|before|where|which|father|mother|spouse|born|founded|"
        r"located|member|director|author|composer|played)\b",
        question,
        flags=re.IGNORECASE,
    )
    choice_lengths = [len(choice) for choice in choices]
    choice_sets = [token_set(choice) for choice in choices]
    choice_similarities = []
    for i, left in enumerate(choice_sets):
        for right in choice_sets[i + 1 :]:
            union = left | right
            choice_similarities.append(len(left & right) / len(union) if union else 0.0)
    return {
        "dataset": row.get("dataset", ""),
        "provider": row.get("provider", ""),
        "subject": row.get("subject", ""),
        "answer_format": answer_format(row, choices),
        "question_chars": len(question),
        "question_tokens": len(tokens),
        "intermediate_chars": len(intermediate),
        "choice_count": len(choices),
        "avg_choice_chars": round(mean([float(v) for v in choice_lengths]), 3),
        "max_choice_chars": max(choice_lengths) if choice_lengths else 0,
        "max_choice_jaccard": round(max(choice_similarities), 6) if choice_similarities else 0.0,
        "avg_choice_question_overlap": round(mean([overlap_ratio(choice, question) for choice in choices]), 6),
        "named_entity_count": len(named_entities),
        "date_number_count": len(dates_numbers),
        "legal_term_count": len(legal_terms),
        "multi_hop_cue_count": len(hop_terms),
    }


def evidence_items(row: dict[str, Any]) -> list[dict[str, Any]]:
    evidence = row.get("evidence_store")
    if isinstance(evidence, list):
        return [item for item in evidence if isinstance(item, dict)]
    return []


def evidence_count(row: dict[str, Any]) -> int:
    evidence = evidence_items(row)
    if evidence:
        return len(evidence)
    retrieved = row.get("retrieved_ids")
    if isinstance(retrieved, list):
        return len(retrieved)
    return 0


def ce_scores(row: dict[str, Any]) -> list[float]:
    scores = []
    for item in evidence_items(row):
        if item.get("cross_encoder_score") is not None:
            try:
                scores.append(float(item["cross_encoder_score"]))
            except (TypeError, ValueError):
                pass
    return scores


def max_ce_score(row: dict[str, Any]) -> float:
    scores = ce_scores(row)
    return max(scores) if scores else 0.0


def normalized_score_entropy(scores: list[float]) -> float:
    if len(scores) <= 1:
        return 0.0
    max_score = max(scores)
    weights = [pow(2.718281828459045, score - max_score) for score in scores]
    total = sum(weights)
    if not total:
        return 0.0
    probs = [weight / total for weight in weights]
    entropy = -sum(prob * math.log2(prob) for prob in probs if prob > 0)
    return entropy / math.log2(len(scores))


def source_key(item: dict[str, Any]) -> str:
    source = item.get("source")
    if source:
        return str(source)
    idx = str(item.get("idx", ""))
    return idx.rsplit("_", 1)[0] if "_" in idx else idx


def evidence_text(row: dict[str, Any]) -> str:
    return "\n".join(str(item.get("text", "")) for item in evidence_items(row))


def norm_state(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def load_statute_states(path: Path) -> dict[str, str]:
    states: dict[str, str] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = str(row.get("idx", "")).strip()
            if idx:
                states[idx] = str(row.get("state", "")).strip()
    return states


def split_gold_ids(row: dict[str, Any]) -> set[str]:
    raw = row.get("gold_idx")
    if raw is None:
        return set()
    if isinstance(raw, list):
        return {str(item).strip() for item in raw if str(item).strip()}
    return {part.strip() for part in str(raw).split(",") if part.strip()}


def housing_state_metrics(row: dict[str, Any], statute_states: dict[str, str] | None) -> dict[str, Any]:
    if str(row.get("dataset", "")).lower() != "housing" or not statute_states:
        return {
            "top1_state_match": 0,
            "any_state_match": 0,
            "all_state_match": 0,
            "state_match_frac": 0.0,
            "unique_retrieved_states": 0,
            "state_diversity": 0.0,
        }

    q_state = norm_state(row.get("state"))
    retrieved = [str(value) for value in row.get("retrieved_ids") or []]
    states = [statute_states.get(idx, "") for idx in retrieved]
    known_states = [state for state in states if state]
    matches = [norm_state(state) == q_state for state in known_states]
    unique_states = {norm_state(state) for state in known_states if norm_state(state)}
    return {
        "top1_state_match": int(bool(matches[0]) if matches else False),
        "any_state_match": int(any(matches)),
        "all_state_match": int(bool(matches) and all(matches)),
        "state_match_frac": round(sum(matches) / len(matches), 6) if matches else 0.0,
        "unique_retrieved_states": len(unique_states),
        "state_diversity": round(len(unique_states) / len(known_states), 6) if known_states else 0.0,
    }


def evidence_probe_metrics(row: dict[str, Any], statute_states: dict[str, str] | None = None) -> dict[str, Any]:
    evidence = evidence_items(row)
    scores = ce_scores(row)
    padded = scores + [0.0] * max(0, 5 - len(scores))
    top1, top2, _top3, _top4, top5 = padded[:5]
    source_keys = [source_key(item) for item in evidence if source_key(item)]
    unique_sources = len(set(source_keys))
    text = evidence_text(row)
    choices = choices_from_row(row)
    choice_overlaps = sorted((overlap_ratio(choice, text) for choice in choices), reverse=True)
    choice_overlap_max = choice_overlaps[0] if choice_overlaps else 0.0
    choice_overlap_second = choice_overlaps[1] if len(choice_overlaps) > 1 else 0.0
    question = str(row.get("formatted_question") or row.get("question") or "")
    gold_ids = split_gold_ids(row)
    retrieved_ids = [str(value) for value in row.get("retrieved_ids") or []]
    gold_ranks = [retrieved_ids.index(gold_id) + 1 for gold_id in gold_ids if gold_id in retrieved_ids]
    metrics = {
        "ce_top1": round(top1, 6),
        "ce_top2": round(top2, 6),
        "ce_top5": round(top5, 6),
        "ce_top1_top2_margin": round(top1 - top2, 6),
        "ce_top1_top5_margin": round(top1 - top5, 6),
        "ce_score_entropy": round(normalized_score_entropy(scores), 6),
        "unique_source_count": unique_sources,
        "source_diversity": round(unique_sources / len(evidence), 6) if evidence else 0.0,
        "retrieved_chars": len(text),
        "question_evidence_overlap": round(overlap_ratio(question, text), 6),
        "choice_evidence_overlap_max": round(choice_overlap_max, 6),
        "choice_evidence_overlap_margin": round(choice_overlap_max - choice_overlap_second, 6),
        "gold_rank": min(gold_ranks) if gold_ranks else 0,
    }
    metrics.update(housing_state_metrics(row, statute_states))
    return metrics


def arm_metrics(row: dict[str, Any], statute_states: dict[str, str] | None = None) -> dict[str, Any]:
    correct = compute_mcnemar.correct_flag(row)
    calls = float(row.get("llm_calls_actual", row.get("llm_calls", 0)) or 0)
    latency = float(row.get("elapsed_sec", 0) or 0)
    metrics = {
        "correct": int(correct),
        "calls": calls,
        "latency_sec": latency,
        "input_tokens": int(row.get("input_tokens", 0) or 0),
        "output_tokens": int(row.get("output_tokens", 0) or 0),
        "gold_retrieved": int(bool(row.get("gold_retrieved"))),
        "evidence_count": evidence_count(row),
        "max_ce_score": round(max_ce_score(row), 6),
        "parse_ok": int(all(bool(row.get(key)) for key in row if key.endswith("_parse_ok"))) if any(key.endswith("_parse_ok") for key in row) else "",
    }
    metrics.update(evidence_probe_metrics(row, statute_states))
    return metrics


def reward(metrics: dict[str, Any], call_penalty: float, latency_penalty: float) -> float:
    return float(metrics["correct"]) - call_penalty * float(metrics["calls"]) - latency_penalty * float(metrics["latency_sec"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True, help="Arm as label=detail_log.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="CSV output path")
    parser.add_argument("--key", help="Override join key")
    parser.add_argument("--call-penalty", type=float, default=0.02)
    parser.add_argument("--latency-penalty", type=float, default=0.0)
    parser.add_argument(
        "--housing-statutes",
        type=Path,
        default=REPO_ROOT / "datasets/housing_qa/statutes.csv",
        help="CSV with HousingQA statute idx/state columns for metadata features",
    )
    args = parser.parse_args()

    rows_by_arm: dict[str, list[dict[str, Any]]] = {}
    for raw in args.arm:
        label, path = resolve_arm(raw)
        if label in rows_by_arm:
            raise SystemExit(f"Duplicate arm label {label!r}")
        rows_by_arm[label] = load_jsonl(path)

    has_housing = any(
        str(row.get("dataset", "")).lower() == "housing"
        for rows in rows_by_arm.values()
        for row in rows
    )
    statute_states = load_statute_states(args.housing_statutes) if has_housing and args.housing_statutes.exists() else None

    labels = list(rows_by_arm)
    key = choose_key(rows_by_arm, args.key)
    by_key = {label: {row[key]: row for row in rows} for label, rows in rows_by_arm.items()}
    common_keys = sorted(set.intersection(*(set(rows) for rows in by_key.values())))
    if not common_keys:
        raise SystemExit(f"No common rows on key {key!r}")

    rows_out: list[dict[str, Any]] = []
    for row_key in common_keys:
        base_row = by_key[labels[0]][row_key]
        out = {"join_key": row_key, **static_features(base_row)}
        per_arm = {}
        for label in labels:
            metrics = arm_metrics(by_key[label][row_key], statute_states)
            per_arm[label] = metrics
            for metric_name, value in metrics.items():
                out[f"{label}_{metric_name}"] = value

        correct_arms = [label for label in labels if per_arm[label]["correct"]]
        if correct_arms:
            out["oracle_accuracy_arm"] = min(
                correct_arms,
                key=lambda label: (per_arm[label]["calls"], per_arm[label]["latency_sec"], label),
            )
        else:
            out["oracle_accuracy_arm"] = min(
                labels,
                key=lambda label: (per_arm[label]["calls"], per_arm[label]["latency_sec"], label),
            )
        out["oracle_reward_arm"] = max(
            labels,
            key=lambda label: (
                reward(per_arm[label], args.call_penalty, args.latency_penalty),
                -per_arm[label]["calls"],
                -per_arm[label]["latency_sec"],
                label,
            ),
        )
        out["oracle_any_correct"] = int(bool(correct_arms))
        out["oracle_correct_count"] = len(correct_arms)
        rows_out.append(out)

    fieldnames = list(rows_out[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Wrote {len(rows_out)} rows x {len(fieldnames)} columns to {args.output}")


if __name__ == "__main__":
    main()
