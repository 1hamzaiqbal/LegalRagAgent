#!/usr/bin/env python3
"""Emit a signoff-log row for a completed HousingQA state-filter detail log."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROWS = 6853
HYRE_MODES = {"rag_hyde", "snap_hyre", "snap_hyre_exemplar"}


def truthy_fallback(row: dict) -> bool:
    falsey_strings = {"", "0", "false", "no", "none", "null", "[]", "{}"}
    for key, value in row.items():
        if "fallback" not in str(key).lower():
            continue
        if isinstance(value, bool):
            if value:
                return True
            continue
        if value is None:
            continue
        if isinstance(value, str):
            if value.strip().lower() not in falsey_strings:
                return True
            continue
        if isinstance(value, (list, tuple, set, dict)):
            if value:
                return True
            continue
        if value:
            return True
    return False


def retry_row(row: dict) -> bool:
    for event in row.get("trace_events") or []:
        if isinstance(event, dict) and event.get("type") == "llm_call":
            if "retry" in str(event.get("label", "")).lower():
                return True
    return bool(row.get("answer_format_retry_output_tokens"))


def expected_final_line(row: dict) -> str | None:
    predicted = str(row.get("predicted_answer") or row.get("prediction") or "").strip().lower()
    if predicted == "yes":
        return "Answer: Yes"
    if predicted == "no":
        return "Answer: No"
    return None


def provider_route_matches(row: dict, openrouter_provider_only: str) -> bool:
    if not openrouter_provider_only:
        return True
    route = row.get("provider_route") or {}
    return (
        isinstance(route, dict)
        and str(route.get("openrouter_provider_only") or "") == openrouter_provider_only
    )


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(errors="ignore") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def coerce_ids(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("[") or stripped.startswith("{"):
            try:
                return coerce_ids(json.loads(stripped))
            except json.JSONDecodeError:
                pass
        return [part.strip() for part in stripped.split(",") if part.strip()]
    if isinstance(value, dict):
        ids: list[str] = []
        for item in value.values():
            ids.extend(coerce_ids(item))
        return ids
    if isinstance(value, (list, tuple, set)):
        ids: list[str] = []
        for item in value:
            ids.extend(coerce_ids(item))
        return ids
    text = str(value).strip()
    return [text] if text else []


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("detail_log", type=Path)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--expected-rows", type=int, default=EXPECTED_ROWS)
    parser.add_argument("--status", default="")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--require-openrouter-provider-only", default="")
    args = parser.parse_args()

    path = args.detail_log
    rows = load_rows(path)
    correct = sum(1 for row in rows if row.get("is_correct") is True)
    gold_hits = 0
    reciprocal_sum = 0.0
    evidence_bad = 0
    failures: list[str] = []
    counters = {
        "wrong_provider": 0,
        "wrong_mode": 0,
        "wrong_dataset": 0,
        "errors": 0,
        "missing_prediction": 0,
        "missing_state_filter": 0,
        "retrieval_cache_miss": 0,
        "doc_cache_miss": 0,
        "hyre_cache_miss": 0,
        "missing_final": 0,
        "fallback": 0,
        "think": 0,
        "retries": 0,
        "near_cap": 0,
        "provider_route_miss": 0,
    }
    max_output = 0
    max_final_chars = 0
    max_final_label = ""

    for row in rows:
        counters["wrong_provider"] += int(row.get("provider") != args.provider)
        counters["wrong_mode"] += int(row.get("mode") != args.mode)
        counters["wrong_dataset"] += int(row.get("dataset") != "housing")
        counters["provider_route_miss"] += int(
            not provider_route_matches(row, args.require_openrouter_provider_only)
        )
        counters["errors"] += int(bool(row.get("error")))
        counters["missing_prediction"] += int(not str(row.get("predicted_answer") or "").strip())
        where = row.get("retrieval_where") or row.get("where") or {}
        state_filtered = row.get("housing_state_filter") is True or (
            isinstance(where, dict) and bool(str(where.get("state", "")).strip())
        )
        counters["missing_state_filter"] += int(not state_filtered)
        counters["retrieval_cache_miss"] += int(row.get("retrieval_cache_hit") is not True)
        counters["doc_cache_miss"] += int(row.get("retrieval_doc_cache_hit") is not True)
        if args.mode in HYRE_MODES:
            counters["hyre_cache_miss"] += int(row.get("hyre_cache_hit") is not True)
        evidence = row.get("evidence_store") or []
        evidence_bad += int(len(evidence) != 5)
        final_lines = [
            line.strip()
            for line in str(row.get("final_answer") or "").splitlines()
            if line.strip()
        ]
        counters["missing_final"] += int(not final_lines or final_lines[-1] != expected_final_line(row))
        counters["fallback"] += int(truthy_fallback(row))
        text = "\n".join(str(row.get(key, "")) for key in ("final_answer", "hyde_passage", "snap_answer")).lower()
        counters["think"] += int("<think" in text or "</think" in text)
        counters["retries"] += int(retry_row(row))
        output_tokens = int(row.get("output_tokens") or 0)
        retry_tokens = int(row.get("answer_format_retry_output_tokens") or 0)
        max_output = max(max_output, output_tokens, retry_tokens)
        counters["near_cap"] += int(output_tokens >= 2032 or retry_tokens >= 2032)
        final_chars = len(str(row.get("final_answer") or ""))
        if final_chars > max_final_chars:
            max_final_chars = final_chars
            max_final_label = str(row.get("label") or row.get("question_id") or "")

        retrieved_ids = [str(item) for item in coerce_ids(row.get("retrieved_ids"))[:5]]
        gold_ids = set(coerce_ids(row.get("gold_idx") or row.get("gold_id") or row.get("gold_ids")))
        matched_ranks = [index + 1 for index, item in enumerate(retrieved_ids) if item in gold_ids]
        if matched_ranks:
            rank = min(matched_ranks)
            gold_hits += 1
            reciprocal_sum += 1.0 / rank
        elif row.get("gold_retrieved") is True:
            gold_hits += 1

    if len(rows) != args.expected_rows:
        failures.append(f"expected {args.expected_rows} rows, found {len(rows)}")
    for key, value in counters.items():
        if key in {"retries", "near_cap"}:
            continue
        if value:
            failures.append(f"{key}={value}")
    if evidence_bad:
        failures.append(f"bad_evidence_len={evidence_bad}")

    if failures and not args.allow_incomplete:
        print("not signoff-ready: " + "; ".join(failures), file=sys.stderr)
        return 1

    rows_n = len(rows)
    accuracy = correct / rows_n if rows_n else 0.0
    hit5 = gold_hits / rows_n if rows_n else 0.0
    mrr5 = reciprocal_sum / rows_n if rows_n else 0.0
    if not args.status:
        status = "✅ COMPREHENSIVE-CLEAN-STATEFILTER"
        if counters["retries"] or counters["near_cap"]:
            status = "⚠️ COMPREHENSIVE-CITE-STATEFILTER/RETRY-CAVEAT"
    else:
        status = args.status

    evidence = (
        f"jurisdiction-filtered `{args.mode}` row with `housing_state_filter=true` "
        f"and strict cache replay; rows {rows_n}/{args.expected_rows}, "
        f"retrieved/evidence length 5 on {rows_n - evidence_bad}/{rows_n} rows, "
        f"{gold_hits}/{rows_n} gold retrieved, Hit@5 {hit5:.4f} / MRR@5 {mrr5:.4f}; "
        f"health counters: errors {counters['errors']}, missing predictions {counters['missing_prediction']}, "
        f"state-filter misses {counters['missing_state_filter']}, retrieval-cache misses {counters['retrieval_cache_miss']}, "
        f"doc-cache misses {counters['doc_cache_miss']}, HyRE-cache misses {counters['hyre_cache_miss']}, "
        f"missing exact final answers {counters['missing_final']}, fallback rows {counters['fallback']}, "
        f"think tags {counters['think']}, answer-format retries {counters['retries']}, "
        f"provider-route misses {counters['provider_route_miss']}, "
        f"near-cap rows {counters['near_cap']}, max output {max_output} tokens, "
        f"max final chars {max_final_chars} at `{max_final_label}`"
    )
    print(
        f"| HousingQA state-filtered | `{args.provider}` | `{args.mode}` | "
        f"`{rel(path)}` | {correct}/{rows_n} = {accuracy:.1%} | {evidence} | {status} |"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
