#!/usr/bin/env python3
"""Compile token, latency, and call-efficiency metrics from detail JSONL logs.

This is an offline report builder: it never calls an LLM.  The intent is to
separate observed answer-replay cost from conceptual method cost.  Cached
HyDE/Snap-HyRE answer rows usually spend one live answer call during replay, but
their detail rows also record ``logical_llm_calls`` and
``cached_generation_calls`` so the report can show the full method footprint.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(errors="ignore") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if isinstance(value, dict):
                rows.append(value)
    if not rows:
        raise SystemExit(f"{path}: no rows")
    return rows


def parse_log_arg(raw: str) -> tuple[str, Path, str]:
    parts = raw.split("=", 2)
    if len(parts) == 1:
        path = resolve_path(raw)
        return path.stem, path, "unspecified"
    if len(parts) == 2:
        label, path_raw = parts
        return label.strip(), resolve_path(path_raw), "unspecified"
    label, status, path_raw = parts
    return label.strip(), resolve_path(path_raw), status.strip()


def pct(value: float | None) -> str:
    return "--" if value is None else f"{value * 100:.1f}%"


def num(value: float | None, digits: int = 1) -> str:
    return "--" if value is None or math.isnan(value) else f"{value:.{digits}f}"


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def percentile(values: list[float], frac: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * frac
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def row_key(row: dict[str, Any]) -> str:
    return str(row.get("label") or row.get("idx") or "")


def final_answer_ok(row: dict[str, Any]) -> bool:
    dataset = str(row.get("dataset") or "").lower()
    if dataset == "housing":
        return bool(re.search(r"(?im)^\s*Answer:\s*(Yes|No)\s*$", str(row.get("final_answer") or "")))
    if dataset == "barexam":
        return bool(re.search(r"(?im)^\s*Answer:\s*\(?[A-E]\)?\s*$", str(row.get("final_answer") or "")))
    return bool(row.get("predicted_answer") or row.get("prediction"))


def truthy_fallback(row: dict[str, Any]) -> bool:
    return any("fallback" in key and value is True for key, value in row.items())


def summarize(label: str, status: str, path: Path, near_cap_threshold: int) -> dict[str, Any]:
    rows = load_jsonl(path)
    n = len(rows)
    first = rows[0]
    correct = sum(1 for row in rows if row.get("is_correct") is True)
    latencies = [float(row["elapsed_sec"]) for row in rows if isinstance(row.get("elapsed_sec"), (int, float))]
    input_tokens = [float(row.get("input_tokens", 0) or 0) for row in rows]
    output_tokens = [float(row.get("output_tokens", 0) or 0) for row in rows]
    total_tokens = [i + o for i, o in zip(input_tokens, output_tokens, strict=False)]
    actual_calls = [float(row.get("llm_calls", 0) or 0) for row in rows]
    logical_calls = [
        float(row.get("logical_llm_calls", row.get("llm_calls", 0)) or 0)
        for row in rows
    ]
    cached_generation_calls = [float(row.get("cached_generation_calls", 0) or 0) for row in rows]
    gold_flags = [bool(row.get("gold_retrieved")) for row in rows if "gold_retrieved" in row]
    evidence_lens = [len(row.get("evidence_store") or []) for row in rows if "evidence_store" in row]
    final_ok = sum(1 for row in rows if final_answer_ok(row))
    fallback_rows = sum(1 for row in rows if truthy_fallback(row))
    state_filter_den = [row for row in rows if "housing_state_filter" in row]
    state_filter_rows = sum(1 for row in state_filter_den if row.get("housing_state_filter") is True)
    retrieval_cache_den = [row for row in rows if "retrieval_cache_hit" in row]
    retrieval_cache_rows = sum(1 for row in retrieval_cache_den if row.get("retrieval_cache_hit") is True)
    doc_cache_den = [row for row in rows if "retrieval_doc_cache_hit" in row]
    doc_cache_rows = sum(1 for row in doc_cache_den if row.get("retrieval_doc_cache_hit") is True)
    hyre_cache_den = [row for row in rows if "hyre_cache_hit" in row]
    hyre_cache_rows = sum(1 for row in hyre_cache_den if row.get("hyre_cache_hit") is True)
    near_cap_rows = 0
    for row in rows:
        row_output_tokens = int(row.get("output_tokens") or 0)
        retry_output_tokens = int(row.get("answer_format_retry_output_tokens") or 0)
        if int(row.get("llm_calls") or 0) <= 1 and row_output_tokens >= near_cap_threshold:
            near_cap_rows += 1
        elif retry_output_tokens >= near_cap_threshold:
            near_cap_rows += 1
    total_token_sum = sum(total_tokens)
    output_token_sum = sum(output_tokens)

    return {
        "label": label,
        "status": status,
        "path": path,
        "dataset": first.get("dataset", ""),
        "provider": first.get("provider", ""),
        "mode": first.get("mode", ""),
        "rows": n,
        "correct": correct,
        "accuracy": correct / n if n else None,
        "gold_hit_rate": (sum(gold_flags) / len(gold_flags)) if gold_flags else None,
        "avg_evidence_docs": mean([float(v) for v in evidence_lens]),
        "avg_latency_sec": mean(latencies),
        "p50_latency_sec": percentile(latencies, 0.50),
        "p95_latency_sec": percentile(latencies, 0.95),
        "avg_input_tokens": mean(input_tokens),
        "avg_output_tokens": mean(output_tokens),
        "avg_total_tokens": mean(total_tokens),
        "avg_actual_calls": mean(actual_calls),
        "avg_logical_calls": mean(logical_calls),
        "avg_cached_generation_calls": mean(cached_generation_calls),
        "correct_per_1m_tokens": (correct / total_token_sum * 1_000_000) if total_token_sum else None,
        "correct_per_1m_output_tokens": (correct / output_token_sum * 1_000_000) if output_token_sum else None,
        "tokens_per_correct": (total_token_sum / correct) if correct else None,
        "final_answer_ok_rate": final_ok / n if n else None,
        "fallback_rows": fallback_rows,
        "state_filter_rate": state_filter_rows / len(state_filter_den) if state_filter_den else None,
        "retrieval_cache_hit_rate": retrieval_cache_rows / len(retrieval_cache_den) if retrieval_cache_den else None,
        "doc_cache_hit_rate": doc_cache_rows / len(doc_cache_den) if doc_cache_den else None,
        "hyre_cache_hit_rate": hyre_cache_rows / len(hyre_cache_den) if hyre_cache_den else None,
        "near_cap_rows": near_cap_rows,
    }


def write_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "label",
        "status",
        "dataset",
        "provider",
        "mode",
        "rows",
        "correct",
        "accuracy",
        "gold_hit_rate",
        "avg_evidence_docs",
        "avg_latency_sec",
        "p50_latency_sec",
        "p95_latency_sec",
        "avg_input_tokens",
        "avg_output_tokens",
        "avg_total_tokens",
        "avg_actual_calls",
        "avg_logical_calls",
        "avg_cached_generation_calls",
        "correct_per_1m_tokens",
        "correct_per_1m_output_tokens",
        "tokens_per_correct",
        "final_answer_ok_rate",
        "fallback_rows",
        "state_filter_rate",
        "retrieval_cache_hit_rate",
        "doc_cache_hit_rate",
        "hyre_cache_hit_rate",
        "near_cap_rows",
        "path",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in summaries:
            row = dict(item)
            row["path"] = str(item["path"].relative_to(REPO_ROOT) if item["path"].is_relative_to(REPO_ROOT) else item["path"])
            writer.writerow({field: row.get(field, "") for field in fields})


def write_markdown(path: Path, title: str, summaries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {title}",
        "",
        "Generated offline from eval detail JSONL logs. Active/partial rows are operational diagnostics only; paper-facing result claims still need `docs/signoff_log.md` signoff.",
        "",
        "Interpretation notes:",
        "",
        "- `Actual calls/q` is the live answer-pass calls recorded in the detail log.",
        "- `Logical calls/q` is the conceptual method footprint recorded by the harness; cached HyDE/Snap-HyRE answer replays usually show one actual answer call but two logical calls.",
        "- Latency mixes provider latency and local harness overhead, so compare only like-for-like runs.",
        "- `Correct / 1M tok.` uses answer-pass input plus output tokens. Generation-cache token cost is not included unless the generation run itself is passed to this script.",
        "- Health `near-cap` counts rows whose logged answer-pass output tokens meet the configured near-cap threshold, not merely verbose rows.",
        "",
        "| Label | Status | Rows | Acc | Hit@5 | Tok/q | In/q | Out/q | Actual calls/q | Logical calls/q | Lat avg/p95 | Correct / 1M tok. | Tokens / correct | Health |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in summaries:
        health = []
        health.append(f"final={pct(item['final_answer_ok_rate'])}")
        if item["state_filter_rate"] is not None and str(item["dataset"]).lower() == "housing" and item["mode"] not in {"llm_only", "golden_passage"}:
            health.append(f"state={pct(item['state_filter_rate'])}")
        if item["retrieval_cache_hit_rate"] is not None:
            health.append(f"ret={pct(item['retrieval_cache_hit_rate'])}")
        if item["doc_cache_hit_rate"] is not None:
            health.append(f"doc={pct(item['doc_cache_hit_rate'])}")
        if item["hyre_cache_hit_rate"] is not None and item["mode"] in {"rag_hyde", "snap_hyre", "snap_hyre_exemplar"}:
            health.append(f"hyre={pct(item['hyre_cache_hit_rate'])}")
        if item["fallback_rows"]:
            health.append(f"fallback={item['fallback_rows']}")
        if item["near_cap_rows"]:
            health.append(f"near-cap={item['near_cap_rows']}")
        lines.append(
            "| {label} | {status} | {rows} | {acc} | {hit} | {tok} | {inp} | {out} | {actual} | {logical} | {lat}/{p95} | {density} | {tpc} | {health} |".format(
                label=item["label"],
                status=item["status"],
                rows=item["rows"],
                acc=pct(item["accuracy"]),
                hit=pct(item["gold_hit_rate"]),
                tok=num(item["avg_total_tokens"], 0),
                inp=num(item["avg_input_tokens"], 0),
                out=num(item["avg_output_tokens"], 0),
                actual=num(item["avg_actual_calls"], 2),
                logical=num(item["avg_logical_calls"], 2),
                lat=num(item["avg_latency_sec"], 2),
                p95=num(item["p95_latency_sec"], 2),
                density=num(item["correct_per_1m_tokens"], 1),
                tpc=num(item["tokens_per_correct"], 0),
                health=", ".join(health),
            )
        )
    lines.extend(["", "## Provenance", "", "| Label | Detail log |", "|---|---|"])
    for item in summaries:
        rel = item["path"].relative_to(REPO_ROOT) if item["path"].is_relative_to(REPO_ROOT) else item["path"]
        lines.append(f"| {item['label']} | `{rel}` |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log",
        action="append",
        required=True,
        help="Log spec: label=path or label=status=path. Status can be signed, active_partial, probe, etc.",
    )
    parser.add_argument("--out-md", required=True, help="Markdown output path")
    parser.add_argument("--out-csv", required=True, help="CSV output path")
    parser.add_argument("--title", default="Efficiency Metrics")
    parser.add_argument(
        "--near-cap-threshold",
        type=int,
        default=2032,
        help="Output-token threshold used for near-cap health counts. Default matches 2048 minus a 16-token margin.",
    )
    args = parser.parse_args()

    summaries = []
    for raw in args.log:
        label, path, status = parse_log_arg(raw)
        if not label:
            raise SystemExit(f"Missing label in --log {raw!r}")
        summaries.append(summarize(label, status, path, args.near_cap_threshold))
    write_markdown(resolve_path(args.out_md), args.title, summaries)
    write_csv(resolve_path(args.out_csv), summaries)
    print(f"Wrote {args.out_md}")
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
