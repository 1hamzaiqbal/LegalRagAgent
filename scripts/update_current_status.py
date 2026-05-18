#!/usr/bin/env python3
"""Regenerate the current comprehensive Snap-HyRE status dashboard.

The dashboard is intentionally operational, not a result signoff. It combines
completed rows from ``logs/experiments.jsonl``, live partial detail logs, and
cache-level retrieval metrics so ``current_status.md`` can be checked quickly
while long answer cells run.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "current_status.md"
EXPERIMENTS = REPO_ROOT / "logs" / "experiments.jsonl"
SIGNOFF = REPO_ROOT / "docs" / "signoff_log.md"
CACHE_MATRIX_PATHS = [
    REPO_ROOT / "docs" / "generated" / "snap_hyre_package" / "retrieval_topk_status.csv",
    REPO_ROOT / "docs" / "generated" / "retrieval_cache_matrix.csv",
    REPO_ROOT / "docs" / "generated" / "retrieval_cache_matrix_groq-llama70b_generated.csv",
    REPO_ROOT / "docs" / "generated" / "retrieval_cache_matrix_or-gemma4-26b_generated.csv",
    REPO_ROOT / "docs" / "generated" / "retrieval_cache_matrix_gemma4-26b_generated.csv",
    REPO_ROOT / "docs" / "generated" / "retrieval_cache_matrix_or-ministral-8b_generated.csv",
]

MODELS = ["or-ministral-8b", "or-gemma4-26b", "groq-llama70b"]
MODEL_SHORT = {
    "or-ministral-8b": "Ministral 8B",
    "or-gemma4-26b": "Gemma 26B",
    "groq-llama70b": "Llama 70B",
}
BENCHMARKS = [
    ("LegalBench-SCALR", "legalbench_scalr", 571),
    ("BarExamQA", "barexam", 1195),
    ("CaseHOLD", "casehold", 3600),
    ("HousingQA", "housing", 6853),
]
BENCHMARK_TOTALS = {dataset: total for _name, dataset, total in BENCHMARKS}
MODES = [
    "llm_only",
    "rag_simple",
    "golden_passage",
    "golden_plus_neighbors",
    "rag_hyde",
    "snap_hyre",
    "rag_rewrite",
]
ACTIVE_MAX_AGE_SEC = 30 * 60


@dataclass
class DetailStats:
    rows: int = 0
    correct: int = 0
    f_acc: float | None = None
    r_acc: float | None = None
    recall: float | None = None
    mrr: float | None = None
    gold_retrieved: int = 0
    evaluated_retrieval: int = 0
    errors: int = 0
    missing_predictions: int = 0
    empty_retrieval: int = 0
    format_retry_rows: int = 0
    fallback_key_rows: int = 0
    think_rows: int = 0
    near_cap_rows: int = 0
    max_output_tokens: int = 0
    retrieved_lens: dict[int, int] | None = None


@dataclass
class Cell:
    benchmark: str
    dataset: str
    total: int
    model: str
    mode: str
    rows: int = 0
    correct: int = 0
    f_acc: float | None = None
    r_acc: float | None = None
    mrr: float | None = None
    status: str = "not started"
    detail_log: str = ""
    run_id: str = ""
    timestamp: str = ""
    signed: bool = False
    source: str = ""
    cache_metric: bool = False
    cache_rows: int = 0
    updated_recently: bool = False
    health: DetailStats | None = None


@dataclass
class CacheProgress:
    benchmark: str
    dataset: str
    total: int
    model: str
    mode: str
    rows: int
    path: Path
    errors: int = 0
    missing_passages: int = 0
    parse_failures: int = 0
    missing_snap_letters: int = 0
    fallback_rows: int = 0


def load_jsonl(path: Path, tolerate_live_tail: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    lines = path.read_text(errors="replace").splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            if tolerate_live_tail and index == len(lines) - 1:
                continue
            raise
        if isinstance(value, dict):
            rows.append(value)
    return rows


def coerce_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() == "nan":
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


def retry_row(row: dict[str, Any]) -> bool:
    for event in row.get("trace_events") or []:
        if isinstance(event, dict) and event.get("type") == "llm_call":
            if "retry" in str(event.get("label", "")).lower():
                return True
    return False


def fallback_flag_row(row: dict[str, Any]) -> bool:
    """Return true only for truthy fallback metadata, not false guard fields."""
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
            if len(value) > 0:
                return True
            continue
        if value:
            return True
    return False


def detail_stats_for_rows(rows: list[dict[str, Any]]) -> DetailStats:
    stats = DetailStats(rows=len(rows), retrieved_lens={})
    if not rows:
        return stats

    stats.correct = sum(1 for row in rows if row.get("is_correct") is True)
    stats.f_acc = stats.correct / len(rows)
    reciprocal_ranks: list[float] = []
    recalls: list[float] = []
    hits = 0
    evaluated = 0

    for row in rows:
        if row.get("error"):
            stats.errors += 1
        if row.get("predicted_answer") is None:
            stats.missing_predictions += 1
        if row.get("retrieved_ids") == []:
            stats.empty_retrieval += 1
        if retry_row(row):
            stats.format_retry_rows += 1
        if fallback_flag_row(row):
            stats.fallback_key_rows += 1
        if "<think" in str(row.get("final_answer") or "").lower():
            stats.think_rows += 1
        output_tokens = row.get("output_tokens")
        if isinstance(output_tokens, int):
            stats.max_output_tokens = max(stats.max_output_tokens, output_tokens)
            if output_tokens >= 1900:
                stats.near_cap_rows += 1
        retrieved = coerce_ids(row.get("retrieved_ids"))
        if retrieved:
            stats.retrieved_lens[len(retrieved)] = stats.retrieved_lens.get(len(retrieved), 0) + 1
        gold_ids = set(coerce_ids(row.get("gold_idx")))
        if not gold_ids:
            continue
        evaluated += 1
        top = retrieved[:5]
        hit_count = sum(1 for doc_id in set(top) if doc_id in gold_ids)
        if hit_count:
            hits += 1
            stats.gold_retrieved += 1
        recalls.append(hit_count / len(gold_ids))
        rr = 0.0
        for rank, doc_id in enumerate(top, start=1):
            if doc_id in gold_ids:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

    stats.evaluated_retrieval = evaluated
    if evaluated:
        stats.r_acc = hits / evaluated
        stats.recall = sum(recalls) / evaluated
        stats.mrr = sum(reciprocal_ranks) / evaluated
    return stats


def detail_stats(path: Path) -> DetailStats:
    return detail_stats_for_rows(load_jsonl(path, tolerate_live_tail=True))


def combined_detail_rows(paths: list[Path]) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for path in sorted(paths, key=lambda p: p.stat().st_mtime if p.exists() else 0):
        for offset, row in enumerate(load_jsonl(path, tolerate_live_tail=True)):
            key = str(row.get("idx") or row.get("label") or f"{path.name}:{offset}")
            by_key[key] = row
    return list(by_key.values())


def rel_path(path: Path | str) -> str:
    p = Path(path)
    if not p.is_absolute():
        return str(p)
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def signed_lookup() -> str:
    return SIGNOFF.read_text(errors="replace") if SIGNOFF.exists() else ""


def is_signed(detail_log: str, signoff_text: str) -> bool:
    if not detail_log:
        return False
    rel = rel_path(detail_log)
    return rel in signoff_text or Path(rel).name in signoff_text


def model_from_cache_row(row: dict[str, str]) -> list[str]:
    method = row.get("method", "")
    model = row.get("model", "")
    path_text = " ".join(str(row.get(key, "")) for key in ("model", "path", "source_csv", "qrel_report"))
    if model == "model_invariant":
        if "or-ministral-8b" in path_text:
            return ["or-ministral-8b"]
        if "or-gemma4-26b" in path_text or "gemma4-26b" in path_text:
            return ["or-gemma4-26b"]
        if "groq-llama70b" in path_text or "llama70b" in path_text:
            return ["groq-llama70b"]
        if method in {"rag_simple", "golden_plus_neighbors"}:
            return MODELS[:]
        return []
    if model in {"llama70b", "groq-llama70b"}:
        return ["groq-llama70b"]
    if model in {"gemma4-26b", "or-gemma4-26b"}:
        return ["or-gemma4-26b"]
    if model == "or-ministral-8b":
        return ["or-ministral-8b"]
    return []


def load_cache_metrics() -> dict[tuple[str, str, str], tuple[float, float, int]]:
    metrics: dict[tuple[str, str, str], tuple[float, float, int]] = {}
    for path in CACHE_MATRIX_PATHS:
        if not path.exists():
            continue
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                if row.get("scope") != "cache":
                    continue
                dataset = row.get("dataset", "")
                method = row.get("method", "")
                if dataset not in BENCHMARK_TOTALS or method not in MODES:
                    continue
                try:
                    k = int(row.get("k", "0"))
                    rows = int(float(row.get("rows", "0")))
                    hit = float(row.get("hit", "nan"))
                    mrr = float(row.get("mrr", "nan"))
                except ValueError:
                    continue
                if k != 5:
                    continue
                total = BENCHMARK_TOTALS[dataset]
                if rows != total:
                    continue
                if row.get("qrel_status") not in {"aligned", ""}:
                    continue
                for model in model_from_cache_row(row):
                    key = (dataset, model, method)
                    existing = metrics.get(key)
                    if existing is None or rows >= existing[2]:
                        metrics[key] = (hit, mrr, rows)
    return metrics


def fresh_enough(path: Path) -> bool:
    try:
        return (time.time() - path.stat().st_mtime) <= ACTIVE_MAX_AGE_SEC
    except FileNotFoundError:
        return False


@lru_cache(maxsize=1)
def active_process_text() -> str:
    try:
        result = subprocess.run(
            ["ps", "-eo", "cmd"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return ""
    return result.stdout


def process_mentions(dataset: str, model: str, mode: str) -> bool:
    text = active_process_text()
    return (
        f"--dataset {dataset}" in text
        and f"--provider {model}" in text
        and f"--mode {mode}" in text
    ) or (
        f"{dataset}_{'qfull_seed42'}_{model}_{mode}.jsonl" in text
        and "--mode " + mode in text
    )


def make_empty_grid() -> dict[tuple[str, str, str], Cell]:
    grid: dict[tuple[str, str, str], Cell] = {}
    for benchmark, dataset, total in BENCHMARKS:
        for model in MODELS:
            for mode in MODES:
                grid[(dataset, model, mode)] = Cell(
                    benchmark=benchmark,
                    dataset=dataset,
                    total=total,
                    model=model,
                    mode=mode,
                )
    return grid


def apply_stats_to_cell(
    cell: Cell,
    stats: DetailStats,
    *,
    detail_log: str,
    status: str,
    source: str,
    run_id: str = "",
    timestamp: str = "",
    signed: bool = False,
    updated_recently: bool = False,
) -> None:
    cell.rows = stats.rows
    cell.correct = stats.correct
    cell.f_acc = stats.f_acc
    if cell.mode != "llm_only":
        cell.r_acc = stats.r_acc
        cell.mrr = stats.mrr
    cell.health = stats
    cell.detail_log = detail_log
    cell.status = status
    cell.source = source
    cell.run_id = run_id
    cell.timestamp = timestamp
    cell.signed = signed
    cell.updated_recently = updated_recently


def apply_detail_to_cell(
    cell: Cell,
    path: Path,
    status: str,
    source: str,
    run_id: str = "",
    timestamp: str = "",
    signed: bool = False,
) -> None:
    stats = detail_stats(path)
    apply_stats_to_cell(
        cell,
        stats,
        detail_log=rel_path(path),
        status=status,
        source=source,
        run_id=run_id,
        timestamp=timestamp,
        signed=signed,
        updated_recently=fresh_enough(path),
    )


def load_experiment_grid(signoff_text: str) -> dict[tuple[str, str, str], Cell]:
    grid = make_empty_grid()
    if not EXPERIMENTS.exists():
        return grid
    for row in load_jsonl(EXPERIMENTS):
        dataset = str(row.get("dataset", ""))
        provider = str(row.get("provider", ""))
        mode = str(row.get("mode", ""))
        key = (dataset, provider, mode)
        if key not in grid:
            continue
        total = BENCHMARK_TOTALS[dataset]
        n_questions = int(row.get("n_questions") or row.get("total") or 0)
        if row.get("question_set") != "full" and n_questions < total:
            continue
        if n_questions < total:
            continue
        detail_log = str(row.get("detail_log") or "")
        detail_path = REPO_ROOT / detail_log if detail_log else None
        if not detail_path or not detail_path.exists():
            continue
        cell = grid[key]
        row_ts = str(row.get("timestamp", ""))
        if cell.rows >= total and row_ts <= cell.timestamp:
            continue
        apply_detail_to_cell(
            cell,
            detail_path,
            status="signed" if is_signed(detail_log, signoff_text) else "complete pending signoff",
            source="experiments",
            run_id=str(row.get("run_id", "")),
            timestamp=row_ts,
            signed=is_signed(detail_log, signoff_text),
        )
        # Prefer exact summary counts when present; stored accuracy is rounded.
        if row.get("correct") is not None and row.get("total") is not None:
            cell.correct = int(row["correct"])
            total_rows = int(row["total"])
            if total_rows:
                cell.f_acc = cell.correct / total_rows
        elif row.get("accuracy") is not None:
            cell.f_acc = float(row["accuracy"])
    return grid


def candidate_detail_paths(dataset: str, model: str, mode: str) -> list[Path]:
    tag = f"local-snap-hyre-{model}-{dataset}-{mode}-nfull-k5"
    paths = list((REPO_ROOT / "logs").glob(f"*{tag}*_detail.jsonl"))
    return sorted(set(paths), key=lambda p: p.stat().st_mtime if p.exists() else 0)


def apply_live_details(grid: dict[tuple[str, str, str], Cell], signoff_text: str) -> None:
    for benchmark, dataset, total in BENCHMARKS:
        for model in MODELS:
            for mode in MODES:
                key = (dataset, model, mode)
                cell = grid[key]
                if cell.rows >= total:
                    continue
                candidates = candidate_detail_paths(dataset, model, mode)
                if candidates:
                    combined_rows = combined_detail_rows(candidates)
                    combined_stats = detail_stats_for_rows(combined_rows)
                    if combined_stats.rows > cell.rows:
                        recent_paths = [path for path in candidates if fresh_enough(path)]
                        is_live = bool(recent_paths) and process_mentions(dataset, model, mode)
                        status = "active" if is_live and combined_stats.rows < total else "partial stale"
                        if combined_stats.rows >= total:
                            status = "complete pending signoff"
                        detail_label = ", ".join(rel_path(path) for path in candidates[-3:])
                        if len(candidates) > 3:
                            detail_label = f"{len(candidates)} detail logs; latest: {detail_label}"
                        apply_stats_to_cell(
                            cell,
                            combined_stats,
                            detail_log=detail_label,
                            status=status,
                            source="combined detail logs",
                            signed=False,
                            updated_recently=is_live,
                        )
                        continue
                best_path: Path | None = None
                best_rows = cell.rows
                for path in candidates:
                    stats = detail_stats(path)
                    if stats.rows <= best_rows:
                        continue
                    best_path = path
                    best_rows = stats.rows
                if best_path is None:
                    continue
                is_live = fresh_enough(best_path) and process_mentions(dataset, model, mode)
                status = "active" if is_live and best_rows < total else "partial stale"
                if best_rows >= total:
                    status = "signed" if is_signed(rel_path(best_path), signoff_text) else "complete pending signoff"
                apply_detail_to_cell(
                    cell,
                    best_path,
                    status=status,
                    source="detail log",
                    signed=is_signed(rel_path(best_path), signoff_text),
                )


def cache_fallback_flag(row: dict[str, Any]) -> bool:
    for key, value in row.items():
        if "fallback" not in str(key).lower():
            continue
        if value not in (False, None, "", 0, [], {}):
            return True
    return False


def active_generation_caches() -> list[CacheProgress]:
    caches: list[CacheProgress] = []
    cache_root = REPO_ROOT / "caches" / "hyre" / "full"
    for benchmark, dataset, total in BENCHMARKS:
        prefix = f"{dataset}_qfull_seed42_"
        for path in cache_root.glob(f"{prefix}*.jsonl"):
            if not fresh_enough(path):
                continue
            name = path.name
            for mode in ("rag_hyde", "snap_hyre"):
                suffix = f"_{mode}.jsonl"
                if not name.endswith(suffix):
                    continue
                model = name[len(prefix):-len(suffix)]
                if model not in MODELS:
                    continue
                if not process_mentions(dataset, model, mode):
                    continue
                rows = load_jsonl(path, tolerate_live_tail=True)
                if not rows or len(rows) >= total:
                    continue
                caches.append(
                    CacheProgress(
                        benchmark=benchmark,
                        dataset=dataset,
                        total=total,
                        model=model,
                        mode=mode,
                        rows=len(rows),
                        path=path,
                        errors=sum(1 for row in rows if row.get("error")),
                        missing_passages=sum(1 for row in rows if not row.get("hyde_passage")),
                        parse_failures=sum(
                            1
                            for row in rows
                            if row.get("hyde_parse_ok") is False
                            or (mode == "snap_hyre" and row.get("snap_hyre_parse_ok") is False)
                        ),
                        missing_snap_letters=sum(
                            1 for row in rows if mode == "snap_hyre" and not row.get("snap_letter")
                        ),
                        fallback_rows=sum(1 for row in rows if cache_fallback_flag(row)),
                    )
                )
                break
    return sorted(caches, key=lambda item: (item.dataset, item.model, item.mode))


def apply_cache_metrics(grid: dict[tuple[str, str, str], Cell]) -> None:
    for (dataset, model, mode), (hit, mrr, rows) in load_cache_metrics().items():
        key = (dataset, model, mode)
        cell = grid.get(key)
        if not cell:
            continue
        if cell.r_acc is None:
            cell.r_acc = hit
            cell.mrr = mrr
            cell.cache_metric = True
            cell.cache_rows = rows
    for dataset in {"barexam", "casehold", "legalbench_scalr"}:
        for model in MODELS:
            cell = grid[(dataset, model, "golden_passage")]
            if cell.r_acc is None:
                cell.r_acc = 1.0
                cell.mrr = 1.0
                cell.cache_metric = True
                cell.cache_rows = BENCHMARK_TOTALS[dataset]


def format_pct(cell: Cell) -> str:
    pct = min(100.0, (cell.rows / cell.total * 100.0) if cell.total else 0.0)
    if cell.rows >= cell.total:
        suffix = "signed" if cell.signed else "pending"
        return f"100% {suffix}"
    if cell.rows:
        suffix = "active" if cell.status == "active" else "partial"
        return f"{pct:.1f}% {suffix}"
    return "0%"


def format_float(value: float | None, partial: bool = False) -> str:
    if value is None:
        return "--"
    suffix = "*" if partial else ""
    return f"{value:.4f}{suffix}"


def format_acc(value: float | None, partial: bool = False) -> str:
    if value is None:
        return "--"
    suffix = "*" if partial else ""
    return f"{value * 100:.1f}%{suffix}"


def metric_value(cell: Cell, metric: str) -> str:
    partial = 0 < cell.rows < cell.total
    if metric == "f_acc":
        return format_acc(cell.f_acc, partial)
    if cell.mode == "llm_only":
        return "n/a"
    if metric == "r_acc":
        return format_float(cell.r_acc, partial and not cell.cache_metric)
    if metric == "mrr":
        return format_float(cell.mrr, partial and not cell.cache_metric)
    return "--"


def health_summary(cell: Cell) -> str:
    stats = cell.health
    if not stats:
        return ""
    parts = [
        f"{cell.rows}/{cell.total}",
        f"correct {cell.correct}",
        f"errors {stats.errors}",
        f"missing {stats.missing_predictions}",
    ]
    if cell.mode != "llm_only":
        parts.extend([
            f"empty retrieval {stats.empty_retrieval}",
            f"gold {stats.gold_retrieved}/{stats.evaluated_retrieval}",
        ])
    parts.extend([
        f"answer retries {stats.format_retry_rows}",
        f"fallback keys {stats.fallback_key_rows}",
        f"think tags {stats.think_rows}",
        f"near-cap {stats.near_cap_rows}",
    ])
    return "; ".join(parts)


def build_markdown(grid: dict[tuple[str, str, str], Cell], interval: int | None) -> str:
    now = datetime.now(ZoneInfo("America/Chicago"))
    cells = list(grid.values())
    signed = sum(1 for cell in cells if cell.signed)
    complete = sum(1 for cell in cells if cell.rows >= cell.total)
    active = sum(1 for cell in cells if 0 < cell.rows < cell.total and cell.status == "active")
    partial = sum(1 for cell in cells if 0 < cell.rows < cell.total and cell.status != "active")
    not_started = len(cells) - complete - active - partial

    lines = [
        "# Snap-HyRE Comprehensive Current Status",
        "",
        f"Snapshot: {now:%Y-%m-%d %H:%M:%S %Z}.",
        "",
        "Generated by `scripts/update_current_status.py`. This is an operational dashboard; use `docs/signoff_log.md` as the citation gate for paper-facing claims.",
        "",
        "Scope:",
        "",
        "- Models: `or-ministral-8b`, `or-gemma4-26b`, `groq-llama70b`",
        "- Benchmarks: LegalBench-SCALR, BarExamQA, CaseHOLD, HousingQA",
        "- Canonical modes: `llm_only`, `rag_simple`, `golden_passage`, `golden_plus_neighbors`, `rag_hyde`, `snap_hyre`, `rag_rewrite`",
        "- Main answer depth: `RETRIEVAL_K=5`",
        "- Completion is row-count completion from `logs/experiments.jsonl` and live detail logs; `signed` means the detail log appears in `docs/signoff_log.md`.",
        "- `r_acc@5` is retrieval Hit@5 over visible retrieved IDs or a full aligned retrieval cache when the answer row has not run yet; `mrr@5` follows the same source. `f_acc` is final answer accuracy.",
        "",
        f"Overall answer-cell status: {signed}/84 signed, {complete}/84 full-row complete, {active}/84 active, {partial}/84 partial stale, {not_started}/84 not started.",
    ]
    if interval:
        lines.append(f"Recurring monitor interval: {interval} seconds.")
    lines.append("")

    lines.append("## Completion Matrix")
    lines.append("")
    for benchmark, dataset, total in BENCHMARKS:
        lines.extend([
            f"### {benchmark}, N={total}",
            "",
            "| Method | `or-ministral-8b` | `or-gemma4-26b` | `groq-llama70b` |",
            "|---|---:|---:|---:|",
        ])
        for mode in MODES:
            row = [f"`{mode}`"]
            for model in MODELS:
                row.append(format_pct(grid[(dataset, model, mode)]))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    lines.append("## Metric Ledger")
    lines.append("")
    lines.append("A `*` marks an active or partial answer row. Cache-only retrieval metrics can appear before `f_acc` exists.")
    lines.append("")
    for benchmark, dataset, total in BENCHMARKS:
        lines.extend([
            f"### {benchmark} Metrics",
            "",
            "| Method | Ministral r_acc@5 | Ministral mrr@5 | Ministral f_acc | Gemma r_acc@5 | Gemma mrr@5 | Gemma f_acc | Llama r_acc@5 | Llama mrr@5 | Llama f_acc |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for mode in MODES:
            row = [f"`{mode}`"]
            for model in MODELS:
                cell = grid[(dataset, model, mode)]
                row.extend([
                    metric_value(cell, "r_acc"),
                    metric_value(cell, "mrr"),
                    metric_value(cell, "f_acc"),
                ])
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    active_rows = [cell for cell in cells if cell.rows and cell.rows < cell.total]
    if active_rows:
        lines.extend([
            "## Active / Partial Rows",
            "",
            "| Benchmark | Model | Mode | Progress | Health | Detail log |",
            "|---|---|---|---:|---|---|",
        ])
        for cell in sorted(active_rows, key=lambda c: (c.dataset, c.model, c.mode)):
            lines.append(
                "| {benchmark} | `{model}` | `{mode}` | {progress} | {health} | `{detail}` |".format(
                    benchmark=cell.benchmark,
                    model=cell.model,
                    mode=cell.mode,
                    progress=format_pct(cell),
                    health=health_summary(cell),
                    detail=cell.detail_log,
                )
            )
        lines.append("")

    active_caches = active_generation_caches()
    if active_caches:
        lines.extend([
            "## Active Generation Caches",
            "",
            "| Benchmark | Model | Mode | Progress | Health | Cache path |",
            "|---|---|---|---:|---|---|",
        ])
        for cache in active_caches:
            health = (
                f"{cache.rows}/{cache.total}; errors {cache.errors}; "
                f"missing passages {cache.missing_passages}; parse failures {cache.parse_failures}; "
                f"missing snap letters {cache.missing_snap_letters}; fallback rows {cache.fallback_rows}"
            )
            lines.append(
                f"| {cache.benchmark} | `{cache.model}` | `{cache.mode}` | "
                f"{100 * cache.rows / cache.total:.1f}% active | {health} | `{rel_path(cache.path)}` |"
            )
        lines.append("")

    complete_unsigned = [cell for cell in cells if cell.rows >= cell.total and not cell.signed]
    if complete_unsigned:
        lines.extend([
            "## Full Rows Pending Signoff",
            "",
            "| Benchmark | Model | Mode | f_acc | Detail log |",
            "|---|---|---|---:|---|",
        ])
        for cell in sorted(complete_unsigned, key=lambda c: (c.dataset, c.model, c.mode)):
            lines.append(
                f"| {cell.benchmark} | `{cell.model}` | `{cell.mode}` | {format_acc(cell.f_acc)} | `{cell.detail_log}` |"
            )
        lines.append("")

    lines.extend([
        "## Regeneration",
        "",
        "One-shot update:",
        "",
        "```bash",
        "python3 scripts/update_current_status.py",
        "```",
        "",
        "Recurring local monitor:",
        "",
        "```bash",
        "scripts/local/status_monitor.sh start",
        "scripts/local/status_monitor.sh status",
        "```",
    ])
    return "\n".join(lines) + "\n"


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def update_once(out: Path, interval: int | None = None) -> None:
    signoff_text = signed_lookup()
    grid = load_experiment_grid(signoff_text)
    apply_live_details(grid, signoff_text)
    apply_cache_metrics(grid)
    write_atomic(out, build_markdown(grid, interval))
    print(f"updated {rel_path(out)} at {datetime.now(ZoneInfo('America/Chicago')):%Y-%m-%d %H:%M:%S %Z}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--watch", action="store_true", help="keep regenerating until interrupted")
    parser.add_argument("--interval", type=int, default=300, help="watch interval in seconds")
    args = parser.parse_args()

    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    if not args.watch:
        update_once(out, interval=args.interval)
        return
    while True:
        try:
            update_once(out, interval=args.interval)
        except Exception as exc:  # dashboard failure should not kill eval jobs
            print(f"[status-monitor] update failed: {exc}", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
