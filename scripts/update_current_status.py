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
    REPO_ROOT / "docs" / "generated" / "retrieval_cache_matrix_groq-llama8b_generated.csv",
]
RETRIEVAL_CACHE_DIR = REPO_ROOT / "caches" / "retrieval" / "full"

MODELS = ["groq-llama8b", "or-gemma4-26b", "groq-llama70b"]
MODEL_SHORT = {
    "groq-llama8b": "Llama 8B",
    "or-gemma4-26b": "Gemma 26B",
    "groq-llama70b": "Llama 70B",
}
BENCHMARKS = [
    ("BarExamQA", "barexam", 1195),
    ("HousingQA", "housing", 6853),
    ("Legal-Link-EU", "legal_link_eu", 1127),
    ("MASLegalBench", "mas_legal_bench", 303),
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
HOUSING_STATE_FILTER_RETRIEVAL_MODES = {
    "rag_simple",
    "golden_plus_neighbors",
    "rag_hyde",
    "snap_hyre",
    "rag_rewrite",
}
NOT_APPLICABLE_MODES = {
    ("mas_legal_bench", "golden_passage"),
    ("mas_legal_bench", "golden_plus_neighbors"),
}
ACTIVE_MAX_AGE_SEC = 30 * 60
MIN_EXPERIMENT_TIMESTAMP = "2026-05-12T00:00:00Z"


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
    retrieval_cache_checked: int = 0
    retrieval_cache_miss: int = 0
    retrieval_doc_cache_checked: int = 0
    retrieval_doc_cache_miss: int = 0
    hyre_cache_checked: int = 0
    hyre_cache_miss: int = 0
    housing_state_filter_missing: int = 0
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


@dataclass
class RetrievalCacheProgress:
    benchmark: str
    dataset: str
    total: int
    models: list[str]
    mode: str
    rows: int
    path: Path
    active: bool
    housing_state_filter: bool = False
    empty_rows: int = 0
    rows_without_gold: int = 0


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


def housing_state_filter_required(dataset: str, mode: str) -> bool:
    return dataset == "housing" and mode in HOUSING_STATE_FILTER_RETRIEVAL_MODES


def rows_have_housing_state_filter(rows: list[dict[str, Any]]) -> bool:
    if not rows:
        return False
    checked = 0
    for row in rows:
        if row.get("error"):
            continue
        checked += 1
        where = row.get("retrieval_where") or row.get("where") or {}
        if row.get("housing_state_filter") is True:
            continue
        if isinstance(where, dict) and str(where.get("state", "")).strip():
            continue
        return False
    return checked > 0


def detail_log_has_housing_state_filter(path: Path) -> bool:
    try:
        rows = load_jsonl(path, tolerate_live_tail=True)
    except Exception:
        return False
    return rows_have_housing_state_filter(rows)


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
        if "retrieval_cache_hit" in row:
            stats.retrieval_cache_checked += 1
            if row.get("retrieval_cache_hit") is not True:
                stats.retrieval_cache_miss += 1
        if "retrieval_doc_cache_hit" in row:
            stats.retrieval_doc_cache_checked += 1
            if row.get("retrieval_doc_cache_hit") is not True:
                stats.retrieval_doc_cache_miss += 1
        if "hyre_cache_hit" in row:
            stats.hyre_cache_checked += 1
            if row.get("hyre_cache_hit") is not True:
                stats.hyre_cache_miss += 1
        row_mode = str(row.get("mode") or "")
        if (
            row.get("dataset") == "housing"
            and row_mode in HOUSING_STATE_FILTER_RETRIEVAL_MODES
            and row.get("housing_state_filter") is not True
        ):
            stats.housing_state_filter_missing += 1
        if retry_row(row):
            stats.format_retry_rows += 1
        if fallback_flag_row(row):
            stats.fallback_key_rows += 1
        if "<think" in str(row.get("final_answer") or "").lower():
            stats.think_rows += 1
        output_tokens = row.get("output_tokens")
        if isinstance(output_tokens, int):
            stats.max_output_tokens = max(stats.max_output_tokens, output_tokens)
            # The row-level output token count is cumulative across all LLM
            # calls. Treat it as a truncation-risk signal only for single-call
            # rows, matching scripts/local/run_answer_cell.sh.
            if int(row.get("llm_calls") or 0) <= 1 and output_tokens >= 1900:
                stats.near_cap_rows += 1
        retry_output_tokens = row.get("answer_format_retry_output_tokens")
        if isinstance(retry_output_tokens, int) and retry_output_tokens >= 1900:
            stats.near_cap_rows += 1
        retrieved = coerce_ids(row.get("retrieved_ids"))
        if retrieved:
            stats.retrieved_lens[len(retrieved)] = stats.retrieved_lens.get(len(retrieved), 0) + 1
        gold_ids = set(coerce_ids(row.get("gold_idx")))
        if not gold_ids and row.get("dataset") == "mas_legal_bench" and retrieved:
            # MASLegalBench lacks official qrels. Use same-source retrieved ids
            # as an operational source-document proxy for dashboard visibility.
            evaluated += 1
            same_source_ids = set(coerce_ids(row.get("same_source_retrieved_ids")))
            top = retrieved[:5]
            hit_count = sum(1 for doc_id in set(top) if doc_id in same_source_ids)
            if hit_count:
                hits += 1
                stats.gold_retrieved += 1
            recalls.append(1.0 if hit_count else 0.0)
            rr = 0.0
            for rank, doc_id in enumerate(top, start=1):
                if doc_id in same_source_ids:
                    rr = 1.0 / rank
                    break
            reciprocal_ranks.append(rr)
            continue
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


def blocking_detail_failure(stats: DetailStats) -> bool:
    """Return true when a partial detail log has already failed signoff gates."""
    return any(
        [
            stats.errors,
            stats.missing_predictions,
            stats.empty_retrieval,
            stats.fallback_key_rows,
            stats.retrieval_cache_miss,
            stats.retrieval_doc_cache_miss,
            stats.hyre_cache_miss,
            stats.housing_state_filter_missing,
        ]
    )


def combined_detail_rows(paths: list[Path]) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for path in sorted(paths, key=lambda p: p.stat().st_mtime if p.exists() else 0):
        for offset, row in enumerate(load_jsonl(path, tolerate_live_tail=True)):
            key = str(row.get("idx") or row.get("label") or f"{path.name}:{offset}")
            by_key[key] = row
    return list(by_key.values())


def combined_detail_rows_with_sources(paths: list[Path]) -> tuple[list[dict[str, Any]], list[Path]]:
    by_key: dict[str, tuple[dict[str, Any], Path]] = {}
    for path in sorted(paths, key=lambda p: p.stat().st_mtime if p.exists() else 0):
        for offset, row in enumerate(load_jsonl(path, tolerate_live_tail=True)):
            key = str(row.get("idx") or row.get("label") or f"{path.name}:{offset}")
            by_key[key] = (row, path)
    rows = [row for row, _path in by_key.values()]
    source_paths = sorted({path for _row, path in by_key.values()}, key=lambda p: p.stat().st_mtime if p.exists() else 0)
    return rows, source_paths


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
        if "groq-llama8b" in path_text or "llama8b" in path_text or "llama-3.1-8b" in path_text:
            return ["groq-llama8b"]
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
    if model in {"llama8b", "groq-llama8b"}:
        return ["groq-llama8b"]
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
                if housing_state_filter_required(dataset, method) and "statefilter" not in str(row.get("path", "")):
                    continue
                if row.get("qrel_status") not in {"aligned", ""}:
                    continue
                for model in model_from_cache_row(row):
                    key = (dataset, model, method)
                    existing = metrics.get(key)
                    if existing is None or rows >= existing[2]:
                        metrics[key] = (hit, mrr, rows)
    metrics.update(load_direct_retrieval_cache_metrics())
    metrics.update(load_mas_same_source_cache_metrics())
    return metrics


def retrieval_cache_model_mode(path: Path) -> tuple[list[str], str, int] | None:
    """Parse active full-cache filenames into dashboard model/mode keys.

    Some newer caches encode cross-encoder limits in the filename, for example
    Legal-Link-EU uses ``raw_question_ce22000``. The generated CSV matrix can lag
    behind those files, so read the JSONL caches directly as a fallback.
    """
    name = path.name
    suffix = "_k10.jsonl"
    if not name.endswith(suffix):
        return None
    for dataset in BENCHMARK_TOTALS:
        if dataset == "mas_legal_bench":
            continue
        prefix = f"{dataset}_qfull_seed42_"
        if not name.startswith(prefix):
            continue
        middle = name[len(prefix):-len(suffix)]
        priority = 0
        if middle.startswith("statefilter_"):
            middle = middle[len("statefilter_"):]
            priority = max(priority, 1)
        if middle.startswith("raw_question"):
            if "ce22000" in middle:
                priority = 22000
            elif "ce12000" in middle:
                priority = 12000
            return MODELS[:], "rag_simple", priority
        if middle == "golden_neighbors":
            return MODELS[:], "golden_plus_neighbors", priority
        for mode in ("rag_hyde", "snap_hyre"):
            mode_suffix = f"_{mode}"
            if middle.endswith(mode_suffix):
                model = middle[:-len(mode_suffix)]
                if model in MODELS:
                    return [model], mode, priority
    return None


def retrieval_metrics_from_rows(rows: list[dict[str, Any]], k: int = 5) -> tuple[float, float] | None:
    if not rows:
        return None
    hits = 0
    reciprocal_rank_sum = 0.0
    evaluated = 0
    for row in rows:
        gold = set(coerce_ids(row.get("gold_ids")))
        if not gold:
            continue
        retrieved = coerce_ids(row.get("effective_retrieved_ids")) or coerce_ids(row.get("retrieved_ids"))
        evaluated += 1
        rr = 0.0
        for rank, doc_id in enumerate(retrieved[:k], 1):
            if doc_id in gold:
                hits += 1
                rr = 1.0 / rank
                break
        reciprocal_rank_sum += rr
    if not evaluated:
        return None
    return hits / evaluated, reciprocal_rank_sum / evaluated


def load_direct_retrieval_cache_metrics() -> dict[tuple[str, str, str], tuple[float, float, int]]:
    candidates: dict[tuple[str, str, str], tuple[float, float, int, int]] = {}
    for path in RETRIEVAL_CACHE_DIR.glob("*_qfull_seed42_*_k10.jsonl"):
        parsed = retrieval_cache_model_mode(path)
        if parsed is None:
            continue
        models, mode, priority = parsed
        dataset = next((d for d in BENCHMARK_TOTALS if path.name.startswith(f"{d}_qfull_seed42_")), "")
        total = BENCHMARK_TOTALS.get(dataset)
        if not total:
            continue
        rows = load_jsonl(path)
        if len(rows) != total:
            continue
        if housing_state_filter_required(dataset, mode) and not rows_have_housing_state_filter(rows):
            continue
        metrics = retrieval_metrics_from_rows(rows, k=5)
        if metrics is None:
            continue
        hit, mrr = metrics
        for model in models:
            key = (dataset, model, mode)
            existing = candidates.get(key)
            if existing is None or (len(rows), priority) >= (existing[2], existing[3]):
                candidates[key] = (hit, mrr, len(rows), priority)
    return {key: (hit, mrr, rows) for key, (hit, mrr, rows, _priority) in candidates.items()}


def mas_cache_model_mode(path: Path) -> tuple[list[str], str] | None:
    prefix = "mas_legal_bench_qfull_seed42_"
    suffix = "_k10.jsonl"
    name = path.name
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    middle = name[len(prefix):-len(suffix)]
    if middle == "raw_question":
        return MODELS[:], "rag_simple"
    for mode in ("rag_hyde", "snap_hyre"):
        mode_suffix = f"_{mode}"
        if middle.endswith(mode_suffix):
            model = middle[:-len(mode_suffix)]
            if model in MODELS:
                return [model], mode
    return None


def load_mas_same_source_cache_metrics() -> dict[tuple[str, str, str], tuple[float, float, int]]:
    """Return MASLegalBench source-document proxy retrieval metrics.

    MASLegalBench has no official qrels/gold passage ids. Retrieval caches still
    record whether a retrieved passage comes from the same source penalty notice
    as the question; use that as an operational proxy in current_status.md.
    """
    metrics: dict[tuple[str, str, str], tuple[float, float, int]] = {}
    total = BENCHMARK_TOTALS["mas_legal_bench"]
    for path in RETRIEVAL_CACHE_DIR.glob("mas_legal_bench_qfull_seed42_*_k10.jsonl"):
        parsed = mas_cache_model_mode(path)
        if parsed is None:
            continue
        models, mode = parsed
        rows = load_jsonl(path)
        if len(rows) != total:
            continue
        hits = 0
        reciprocal_rank_sum = 0.0
        for row in rows:
            retrieved = coerce_ids(row.get("effective_retrieved_ids")) or coerce_ids(row.get("retrieved_ids"))
            same_source = set(coerce_ids(row.get("same_source_retrieved_ids")))
            rr = 0.0
            for rank, doc_id in enumerate(retrieved[:5], 1):
                if doc_id in same_source:
                    hits += 1
                    rr = 1.0 / rank
                    break
            reciprocal_rank_sum += rr
        hit = hits / total
        mrr = reciprocal_rank_sum / total
        for model in models:
            metrics[("mas_legal_bench", model, mode)] = (hit, mrr, total)
    return metrics


def fresh_enough(path: Path) -> bool:
    try:
        return (time.time() - path.stat().st_mtime) <= ACTIVE_MAX_AGE_SEC
    except FileNotFoundError:
        return False


def active_process_text() -> str:
    try:
        result = subprocess.run(
            ["ps", "-eww", "-o", "cmd="],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return ""
    return result.stdout


HOUSING_LOCK_DIRS = [
    ("after-reset launcher", Path("/tmp/housing_gemma_after_key_reset.lock")),
    ("after-reset launcher", REPO_ROOT / "logs/monitors/locks/housing_gemma_after_key_reset.lock"),
    ("rag_simple resume", Path("/tmp/housing_gemma_rag_simple_resume.lock")),
    ("rag_simple resume", REPO_ROOT / "logs/monitors/locks/housing_gemma_rag_simple_resume.lock"),
    ("Gemma core queue", Path("/tmp/housing_gemma_core_queue.lock")),
    ("Gemma core queue", REPO_ROOT / "logs/monitors/locks/housing_gemma_core_queue.lock"),
    ("full exemplar", Path("/tmp/housing_gemma_exemplar_full.lock")),
    ("full exemplar", REPO_ROOT / "logs/monitors/locks/housing_gemma_exemplar_full.lock"),
    ("budget watcher", Path("/tmp/housing_gemma_budget_watch.lock")),
    ("budget watcher", REPO_ROOT / "logs/monitors/locks/housing_gemma_budget_watch.lock"),
]


def lock_metadata(lock_dir: Path) -> dict[str, str]:
    meta_path = lock_dir / "metadata"
    if not meta_path.exists():
        return {}
    metadata: dict[str, str] = {}
    try:
        for line in meta_path.read_text(errors="replace").splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            metadata[key.strip()] = value.strip()
    except OSError:
        return {}
    return metadata


def pid_liveness(pid_text: str) -> str:
    pid_text = pid_text.strip()
    if not pid_text:
        return ""
    try:
        pid = int(pid_text)
    except ValueError:
        return f"pid `{pid_text}` invalid"
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return f"pid `{pid}` not live"
    except PermissionError:
        return f"pid `{pid}` present but not inspectable"
    return f"pid `{pid}` live"


def housing_lock_notes() -> list[str]:
    notes: list[str] = []
    for label, lock_dir in HOUSING_LOCK_DIRS:
        if not lock_dir.exists():
            continue
        metadata = lock_metadata(lock_dir)
        pid_status = pid_liveness(metadata.get("pid", ""))
        if label == "budget watcher":
            launch = metadata.get("launch_on_ready", "?")
            interval = metadata.get("interval_seconds", "?")
            log_file = metadata.get("log_file", "")
            created = metadata.get("created_utc", "")
            pieces = [
                f"HousingQA Gemma budget watcher lock is present (`{lock_dir}`)",
                f"`LAUNCH_ON_READY={launch}`",
                f"poll interval `{interval}s`",
            ]
            if pid_status:
                pieces.append(pid_status)
            if created:
                pieces.append(f"created `{created}`")
            if log_file:
                try:
                    pieces.append(f"log `{rel_path(Path(log_file))}`")
                except Exception:
                    pieces.append(f"log `{log_file}`")
            notes.append("; ".join(pieces) + ".")
        else:
            created = metadata.get("created_utc", "")
            suffix_parts = []
            if pid_status:
                suffix_parts.append(pid_status)
            if created:
                suffix_parts.append(f"created `{created}`")
            suffix = f" ({'; '.join(suffix_parts)})" if suffix_parts else ""
            notes.append(f"HousingQA {label} lock is present (`{lock_dir}`){suffix}; inspect before launching another job.")
    return notes


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
                status = "not applicable" if (dataset, mode) in NOT_APPLICABLE_MODES else "not started"
                grid[(dataset, model, mode)] = Cell(
                    benchmark=benchmark,
                    dataset=dataset,
                    total=total,
                    model=model,
                    mode=mode,
                    status=status,
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
        if str(row.get("timestamp", "")) < MIN_EXPERIMENT_TIMESTAMP:
            continue
        if (dataset, mode) in NOT_APPLICABLE_MODES:
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
        if housing_state_filter_required(dataset, mode) and not detail_log_has_housing_state_filter(detail_path):
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
    logs_dir = REPO_ROOT / "logs"
    paths = list(logs_dir.glob(f"*{tag}*_detail.jsonl"))
    # One-row/tail repair logs intentionally carry explicit repair tags so the
    # failed-closed prefix remains traceable. They still belong to the same
    # dataset/model/mode cell and should supersede the bad row by idx/label.
    paths.extend(logs_dir.glob(f"eval_{mode}_{model}_*_{dataset}_*repair*_detail.jsonl"))
    merged_dir = REPO_ROOT / "logs" / "merged"
    if merged_dir.exists():
        paths.extend(merged_dir.glob(f"eval_{mode}_{model}_*{dataset}*merged_detail.jsonl"))
        paths.extend(merged_dir.glob(f"*{model}*{dataset}*{mode}*merged_detail.jsonl"))
        paths.extend(merged_dir.glob(f"*{dataset}*{model}*{mode}*detail.jsonl"))
        paths.extend(merged_dir.glob(f"*{model}*{dataset}*{mode}*detail.jsonl"))
    return sorted(set(paths), key=lambda p: p.stat().st_mtime if p.exists() else 0)


def apply_live_details(grid: dict[tuple[str, str, str], Cell], signoff_text: str) -> None:
    for benchmark, dataset, total in BENCHMARKS:
        for model in MODELS:
            for mode in MODES:
                if (dataset, mode) in NOT_APPLICABLE_MODES:
                    continue
                key = (dataset, model, mode)
                cell = grid[key]
                if cell.rows >= total:
                    continue
                candidates = candidate_detail_paths(dataset, model, mode)
                if housing_state_filter_required(dataset, mode):
                    candidates = [
                        path for path in candidates
                        if detail_log_has_housing_state_filter(path)
                    ]
                if candidates:
                    combined_rows, contributing_paths = combined_detail_rows_with_sources(candidates)
                    combined_stats = detail_stats_for_rows(combined_rows)
                    if combined_stats.rows > cell.rows:
                        recent_paths = [path for path in candidates if fresh_enough(path)]
                        # Some runs are launched from a separate tmux/session
                        # context that is not visible to this status process.
                        # A freshly appended detail log is still authoritative
                        # evidence that the row is moving; process visibility is
                        # only a supplemental signal.
                        is_live = (
                            bool(recent_paths) or process_mentions(dataset, model, mode)
                        ) and not blocking_detail_failure(combined_stats)
                        signed_paths = [path for path in contributing_paths if is_signed(rel_path(path), signoff_text)]
                        signed_candidates = [path for path in candidates if is_signed(rel_path(path), signoff_text)]
                        combined_signed = combined_stats.rows >= total and bool(signed_paths or signed_candidates)
                        status = "active" if is_live and combined_stats.rows < total else "partial stale"
                        if combined_stats.rows >= total:
                            status = "signed" if combined_signed else "complete pending signoff"
                        if combined_stats.rows >= total and (signed_paths or signed_candidates):
                            detail_label = rel_path((signed_paths or signed_candidates)[-1])
                        else:
                            label_paths = contributing_paths or candidates
                            detail_label = ", ".join(rel_path(path) for path in label_paths[-3:])
                            if len(label_paths) > 3:
                                detail_label = f"{len(label_paths)} contributing detail logs; latest: {detail_label}"
                        apply_stats_to_cell(
                            cell,
                            combined_stats,
                            detail_log=detail_label,
                            status=status,
                            source="combined detail logs",
                            signed=combined_signed,
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
                stats = detail_stats(best_path)
                is_live = (
                    fresh_enough(best_path) or process_mentions(dataset, model, mode)
                ) and not blocking_detail_failure(stats)
                status = "active" if is_live and best_rows < total else "partial stale"
                if best_rows >= total:
                    status = "signed" if is_signed(rel_path(best_path), signoff_text) else "complete pending signoff"
                apply_stats_to_cell(
                    cell,
                    stats,
                    detail_log=rel_path(best_path),
                    status=status,
                    source="detail log",
                    signed=is_signed(rel_path(best_path), signoff_text),
                    updated_recently=is_live,
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
    for benchmark, dataset, total in BENCHMARKS:
        prefix = f"{dataset}_qfull_seed42_"
        cache_paths: list[Path] = []
        for cache_root in (
            REPO_ROOT / "caches" / "hyre" / "full",
            REPO_ROOT / "caches" / "generation" / "full",
        ):
            cache_paths.extend(cache_root.glob(f"{prefix}*.jsonl"))
        for path in cache_paths:
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
                # Generation queues are often launched from a host tmux/session
                # that is not visible inside the current sandbox. A freshly
                # appended partial cache is the authoritative active signal.
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


def active_retrieval_caches() -> list[RetrievalCacheProgress]:
    caches: list[RetrievalCacheProgress] = []
    process_text = active_process_text()
    for benchmark, dataset, total in BENCHMARKS:
        prefix = f"{dataset}_qfull_seed42_"
        for path in RETRIEVAL_CACHE_DIR.glob(f"{prefix}*_k10.jsonl"):
            parsed = retrieval_cache_model_mode(path) if dataset != "mas_legal_bench" else mas_cache_model_mode(path)
            if parsed is None:
                continue
            if dataset == "mas_legal_bench":
                models, mode = parsed
            else:
                models, mode, _priority = parsed
            if mode not in MODES:
                continue
            rows = load_jsonl(path, tolerate_live_tail=True)
            if not rows or len(rows) >= total:
                continue
            if housing_state_filter_required(dataset, mode) and not rows_have_housing_state_filter(rows):
                continue
            running = path.name in process_text or str(path) in process_text
            recent = fresh_enough(path)
            if not running and not recent:
                continue
            caches.append(
                RetrievalCacheProgress(
                    benchmark=benchmark,
                    dataset=dataset,
                    total=total,
                    models=models,
                    mode=mode,
                    rows=len(rows),
                    path=path,
                    active=running,
                    housing_state_filter=rows_have_housing_state_filter(rows),
                    empty_rows=sum(1 for row in rows if not coerce_ids(row.get("retrieved_ids"))),
                    rows_without_gold=sum(1 for row in rows if not coerce_ids(row.get("gold_ids"))),
                )
            )
    return sorted(caches, key=lambda item: (item.dataset, item.mode, rel_path(item.path)))


def apply_cache_metrics(grid: dict[tuple[str, str, str], Cell]) -> None:
    for (dataset, model, mode), (hit, mrr, rows) in load_cache_metrics().items():
        key = (dataset, model, mode)
        cell = grid.get(key)
        if not cell:
            continue
        if (dataset, mode) in NOT_APPLICABLE_MODES:
            continue
        total = BENCHMARK_TOTALS.get(dataset, 0)
        # Full retrieval caches are the citation-grade retrieval source even
        # while a matching answer row is still partial. Do not let early
        # answer-slice retrieval exposure overwrite the full-cache metric.
        if cell.r_acc is None or (total and rows >= total):
            cell.r_acc = hit
            cell.mrr = mrr
            cell.cache_metric = True
            cell.cache_rows = rows
    for dataset in {"barexam", "housing", "legal_link_eu"}:
        for model in MODELS:
            cell = grid[(dataset, model, "golden_passage")]
            if cell.r_acc is None:
                cell.r_acc = 1.0
                cell.mrr = 1.0
                cell.cache_metric = True
                cell.cache_rows = BENCHMARK_TOTALS[dataset]


def format_pct(cell: Cell) -> str:
    if cell.status == "not applicable":
        return "n/a"
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
    if cell.status == "not applicable":
        return "n/a"
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
    if stats.retrieval_cache_checked:
        parts.append(f"ret cache miss {stats.retrieval_cache_miss}/{stats.retrieval_cache_checked}")
    if stats.retrieval_doc_cache_checked:
        parts.append(f"doc cache miss {stats.retrieval_doc_cache_miss}/{stats.retrieval_doc_cache_checked}")
    if stats.hyre_cache_checked:
        parts.append(f"hyre cache miss {stats.hyre_cache_miss}/{stats.hyre_cache_checked}")
    if cell.dataset == "housing" and cell.mode in HOUSING_STATE_FILTER_RETRIEVAL_MODES:
        parts.append(f"state filter missing {stats.housing_state_filter_missing}")
    parts.extend([
        f"answer retries {stats.format_retry_rows}",
        f"fallback keys {stats.fallback_key_rows}",
        f"think tags {stats.think_rows}",
        f"near-cap {stats.near_cap_rows}",
    ])
    return "; ".join(parts)


def build_markdown(grid: dict[tuple[str, str, str], Cell], interval: int | None) -> str:
    now = datetime.now(ZoneInfo("America/Chicago"))
    all_cells = list(grid.values())
    cells = [cell for cell in all_cells if cell.status != "not applicable"]
    total_cells = len(cells)
    signed = sum(1 for cell in cells if cell.signed)
    complete = sum(1 for cell in cells if cell.rows >= cell.total)
    active = sum(1 for cell in cells if 0 < cell.rows < cell.total and cell.status == "active")
    partial = sum(1 for cell in cells if 0 < cell.rows < cell.total and cell.status != "active")
    not_started = total_cells - complete - active - partial
    benchmark_names = ", ".join(benchmark for benchmark, _dataset, _total in BENCHMARKS)

    lines = [
        "# Snap-HyRE Comprehensive Current Status",
        "",
        f"Snapshot: {now:%Y-%m-%d %H:%M:%S %Z}.",
        "",
        "Generated by `scripts/update_current_status.py`. This is an operational dashboard; use `docs/signoff_log.md` as the citation gate for paper-facing claims.",
        "",
        "Scope:",
        "",
        "- Models: `groq-llama8b`, `or-gemma4-26b`, `groq-llama70b`",
        f"- Benchmarks: {benchmark_names}",
        "- Canonical modes: `llm_only`, `rag_simple`, `golden_passage`, `golden_plus_neighbors`, `rag_hyde`, `snap_hyre`, `rag_rewrite`",
        "- Main answer depth: `RETRIEVAL_K=5`",
        "- Completion is row-count completion from `logs/experiments.jsonl` and live detail logs; `signed` means the detail log appears in `docs/signoff_log.md`.",
        "- `r_acc@5` is retrieval Hit@5 over visible retrieved IDs or a full aligned retrieval cache when the answer row has not run yet; `mrr@5` follows the same source. `f_acc` is final answer accuracy.",
        "- Legal-Link-EU full retrieval caches should use `CROSS_ENCODER_MAX_CHARS=22000`; detail logs should show role/title/citation headers and source/target document hit fields.",
        "- HousingQA retrieval rows now require the jurisdiction state filter for the main matrix. Unfiltered Housing retrieval logs are provenance/ablation rows only and are not counted here unless a detail log or cache records `housing_state_filter=true` / `retrieval_where={\"state\": ...}`.",
        "- MASLegalBench has no official per-question gold evidence IDs, so `golden_passage` and `golden_plus_neighbors` are marked not applicable; its `r_acc@5`/`mrr@5` cells are same-source-document retrieval proxies from the retrieval caches.",
        "- Legal RAG Bench is embedded and tracked in `docs/candidate_benchmark_feasibility_2026-05-18.md`, but is not in this exact-scored matrix because downstream scoring is open-ended.",
        "",
        f"Overall answer-cell status: {signed}/{total_cells} signed, {complete}/{total_cells} full-row complete, {active}/{total_cells} active, {partial}/{total_cells} partial stale, {not_started}/{total_cells} not started.",
    ]
    if interval:
        lines.append(f"Recurring monitor interval: {interval} seconds.")
    lines.append("")

    operational_notes: list[str] = []
    if active == 0 and partial > 0:
        operational_notes.append(
            "No answer cells are currently marked active by the dashboard; partial rows are stale/not-running unless a launcher has been restarted."
        )
    housing_gemma_rag_simple = grid.get(("housing", "or-gemma4-26b", "rag_simple"))
    if (
        housing_gemma_rag_simple
        and housing_gemma_rag_simple.rows
        and housing_gemma_rag_simple.rows < housing_gemma_rag_simple.total
        and housing_gemma_rag_simple.health
        and housing_gemma_rag_simple.health.errors > 0
    ):
        operational_notes.append(
            "HousingQA `or-gemma4-26b` `rag_simple` is a blocked partial with failed-closed OpenRouter key-limit rows; do not sign it until those labels are superseded by same-model reruns, merged, and audited."
        )
        operational_notes.append(
            "Use `scripts/local/check_housing_gemma_readiness.sh` for a read-only local gate, or `CHECK_NETWORK=1 scripts/local/check_housing_gemma_readiness.sh` for a non-launching OpenRouter preflight. After the key/account is reset and the network gate passes, run `scripts/local/run_housing_gemma_after_key_reset.sh`."
        )
    operational_notes.extend(housing_lock_notes())
    if operational_notes:
        lines.extend(["## Operational Notes", ""])
        lines.extend(f"- {note}" for note in operational_notes)
        lines.append("")

    lines.append("## Completion Matrix")
    lines.append("")
    for benchmark, dataset, total in BENCHMARKS:
        lines.extend([
            f"### {benchmark}, N={total}",
            "",
            "| Method | `groq-llama8b` | `or-gemma4-26b` | `groq-llama70b` |",
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
            "| Method | Llama 8B r_acc@5 | Llama 8B mrr@5 | Llama 8B f_acc | Gemma r_acc@5 | Gemma mrr@5 | Gemma f_acc | Llama 70B r_acc@5 | Llama 70B mrr@5 | Llama 70B f_acc |",
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

    active_retrieval = active_retrieval_caches()
    if active_retrieval:
        lines.extend([
            "## Active / Partial Retrieval Caches",
            "",
            "| Benchmark | Model scope | Mode | Progress | Health | Cache path |",
            "|---|---|---|---:|---|---|",
        ])
        for cache in active_retrieval:
            model_scope = ", ".join(f"`{model}`" for model in cache.models)
            if len(cache.models) == len(MODELS):
                model_scope = "all canonical models"
            health_parts = [
                f"{cache.rows}/{cache.total}",
                f"empty rows {cache.empty_rows}",
                f"rows without gold {cache.rows_without_gold}",
            ]
            if cache.dataset == "housing":
                health_parts.append(f"state filter {'on' if cache.housing_state_filter else 'off'}")
            progress_status = "active" if cache.active else "partial/no process"
            lines.append(
                f"| {cache.benchmark} | {model_scope} | `{cache.mode}` | "
                f"{100 * cache.rows / cache.total:.1f}% {progress_status} | {'; '.join(health_parts)} | `{rel_path(cache.path)}` |"
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
    monitor_active = args.watch or os.environ.get("CURRENT_STATUS_MONITOR") == "1"
    if not args.watch:
        update_once(out, interval=args.interval if monitor_active else None)
        return
    while True:
        try:
            update_once(out, interval=args.interval)
        except Exception as exc:  # dashboard failure should not kill eval jobs
            print(f"[status-monitor] update failed: {exc}", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
