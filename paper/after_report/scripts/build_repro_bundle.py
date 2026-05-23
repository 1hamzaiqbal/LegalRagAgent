#!/usr/bin/env python3
"""Build compact reproducibility manifests for the paper audit.

The raw detail logs and retrieval caches referenced by the paper are multiple
GiB, so this script does not copy them. It records exact paths, checksums, row
counts, and compact answer/retrieval summaries that are small enough to commit.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
AFTER_REPORT = ROOT / "paper" / "after_report"
TABLE_DIR = AFTER_REPORT / "tables"
OUT_DIR = AFTER_REPORT / "repro_bundle"

KS = [1, 3, 5, 10]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def line_count(path: Path) -> int:
    with path.open("rb") as f:
        return sum(1 for _ in f)


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def add_source(sources: set[Path], value: str) -> None:
    value = value.strip()
    if not value or value.startswith("http"):
        return
    if value.startswith("../../"):
        path = (AFTER_REPORT / value).resolve()
    else:
        path = (ROOT / value).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError:
        return
    if path.exists() and path.is_file():
        sources.add(path)


def collect_sources() -> list[Path]:
    sources: set[Path] = set()

    lineage = AFTER_REPORT / "number_lineage.md"
    text = lineage.read_text()
    for match in re.finditer(r"\]\(([^)]+)\)", text):
        add_source(sources, match.group(1))

    for csv_path in TABLE_DIR.glob("*.csv"):
        sources.add(csv_path.resolve())
        with csv_path.open(newline="") as f:
            for row in csv.DictReader(f):
                for field in ("source_path", "detail_log", "source"):
                    if field in row:
                        add_source(sources, row[field])

    for tex_path in TABLE_DIR.glob("*.tex"):
        sources.add(tex_path.resolve())

    for extra in [
        ROOT / "docs" / "signoff_log.md",
        ROOT / "docs" / "compiled_results.md",
        ROOT / "logs" / "experiments.jsonl",
        ROOT / "paper" / "archive" / "reported_data_lineage.md",
        ROOT / "paper" / "archive" / "icml_submission_damage_report.md",
        AFTER_REPORT / "damage_report.md",
        AFTER_REPORT / "data_generation_and_reproducibility.md",
        AFTER_REPORT / "internal_discrepancies_and_recommendations.md",
        AFTER_REPORT / "number_lineage.md",
        AFTER_REPORT / "scripts" / "regenerate_figure3_from_final_csv.py",
    ]:
        if extra.exists():
            sources.add(extra.resolve())

    return sorted(sources, key=rel)


def file_kind(path: Path) -> str:
    r = rel(path)
    if r.startswith("logs/") and r.endswith(".jsonl"):
        return "answer_detail_log"
    if r.startswith("caches/retrieval/") and r.endswith(".jsonl"):
        return "retrieval_cache"
    if r.startswith("paper/after_report/tables/"):
        return "paper_table_or_csv"
    if r.startswith("docs/"):
        return "source_doc"
    if r.startswith("paper/after_report/") or r.startswith("paper/"):
        return "audit_doc_or_script"
    return "other"


def source_manifest(sources: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in sources:
        suffix = path.suffix.lower()
        rows.append(
            {
                "path": rel(path),
                "kind": file_kind(path),
                "bytes": path.stat().st_size,
                "line_count": line_count(path) if suffix in {".jsonl", ".csv", ".tex", ".md", ".py"} else "",
                "sha256": sha256_file(path),
            }
        )
    return rows


def safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize_answer_log(path: Path) -> dict[str, Any]:
    rows = 0
    correct = 0
    errors = 0
    missing_prediction = 0
    input_tokens = 0.0
    output_tokens = 0.0
    llm_calls = 0.0
    elapsed = 0.0
    input_n = output_n = calls_n = elapsed_n = 0
    first: dict[str, Any] = {}

    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if not first:
                first = obj
            rows += 1
            if obj.get("is_correct") is True:
                correct += 1
            if obj.get("error"):
                errors += 1
            if not obj.get("predicted_answer"):
                missing_prediction += 1
            for key, total_name in [
                ("input_tokens", "input"),
                ("output_tokens", "output"),
                ("llm_calls", "calls"),
                ("elapsed_sec", "elapsed"),
            ]:
                value = safe_float(obj.get(key))
                if value is None:
                    continue
                if total_name == "input":
                    input_tokens += value
                    input_n += 1
                elif total_name == "output":
                    output_tokens += value
                    output_n += 1
                elif total_name == "calls":
                    llm_calls += value
                    calls_n += 1
                else:
                    elapsed += value
                    elapsed_n += 1

    return {
        "path": rel(path),
        "rows": rows,
        "correct": correct,
        "accuracy": f"{(correct / rows):.6f}" if rows else "",
        "errors": errors,
        "missing_prediction": missing_prediction,
        "avg_input_tokens": f"{(input_tokens / input_n):.4f}" if input_n else "",
        "avg_output_tokens": f"{(output_tokens / output_n):.4f}" if output_n else "",
        "avg_llm_calls": f"{(llm_calls / calls_n):.4f}" if calls_n else "",
        "avg_elapsed_sec": f"{(elapsed / elapsed_n):.4f}" if elapsed_n else "",
        "first_dataset": first.get("dataset", ""),
        "first_idx": first.get("idx", ""),
    }


def get_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        return [value]
    return []


def summarize_retrieval_cache(path: Path) -> dict[str, Any]:
    rows = 0
    hits = {k: 0 for k in KS}
    rr = {k: 0.0 for k in KS}
    rows_with_gold = 0
    first: dict[str, Any] = {}

    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if not first:
                first = obj
            retrieved = get_ids(obj.get("effective_retrieved_ids")) or get_ids(obj.get("retrieved_ids"))
            gold = set(get_ids(obj.get("gold_ids")) or get_ids(obj.get("gold_idx")))
            rows += 1
            if gold:
                rows_with_gold += 1
            for k in KS:
                rank = None
                for i, doc_id in enumerate(retrieved[:k], start=1):
                    if doc_id in gold:
                        rank = i
                        break
                if rank is not None:
                    hits[k] += 1
                    rr[k] += 1.0 / rank

    out: dict[str, Any] = {
        "path": rel(path),
        "rows": rows,
        "rows_with_gold": rows_with_gold,
        "dataset": first.get("dataset", ""),
        "query_type": first.get("query_type", ""),
        "label_prefix": first.get("label_prefix", ""),
    }
    for k in KS:
        denom = rows_with_gold or rows
        out[f"hit@{k}"] = f"{(hits[k] / denom):.6f}" if denom else ""
        out[f"mrr@{k}"] = f"{(rr[k] / denom):.6f}" if denom else ""
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sources = collect_sources()
    manifest = source_manifest(sources)
    write_csv(
        OUT_DIR / "source_file_manifest.csv",
        manifest,
        ["path", "kind", "bytes", "line_count", "sha256"],
    )
    (OUT_DIR / "source_file_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    answer_rows = [
        summarize_answer_log(path)
        for path in sources
        if file_kind(path) == "answer_detail_log"
    ]
    write_csv(
        OUT_DIR / "answer_log_summaries.csv",
        answer_rows,
        [
            "path",
            "rows",
            "correct",
            "accuracy",
            "errors",
            "missing_prediction",
            "avg_input_tokens",
            "avg_output_tokens",
            "avg_llm_calls",
            "avg_elapsed_sec",
            "first_dataset",
            "first_idx",
        ],
    )

    retrieval_rows = [
        summarize_retrieval_cache(path)
        for path in sources
        if file_kind(path) == "retrieval_cache"
    ]
    write_csv(
        OUT_DIR / "retrieval_cache_summaries.csv",
        retrieval_rows,
        [
            "path",
            "rows",
            "rows_with_gold",
            "dataset",
            "query_type",
            "label_prefix",
            "hit@1",
            "mrr@1",
            "hit@3",
            "mrr@3",
            "hit@5",
            "mrr@5",
            "hit@10",
            "mrr@10",
        ],
    )

    total_bytes = sum(int(row["bytes"]) for row in manifest)
    readme = f"""# Reproducibility Bundle

Generated by `paper/after_report/scripts/build_repro_bundle.py`.

This directory records the source data behind the paper without duplicating
the full raw logs. The raw JSONL detail logs and retrieval caches referenced by
the paper total {total_bytes / 1024 / 1024:.1f} MiB across {len(sources)} source
files, so they are tracked by canonical path and SHA-256 instead of being copied
under `paper/`.

Files:

- `source_file_manifest.csv` and `.json`: every source path, kind, byte size,
  line count, and SHA-256.
- `answer_log_summaries.csv`: compact accuracy/token/health summaries from
  answer detail logs.
- `retrieval_cache_summaries.csv`: compact Hit@k/MRR@k summaries from retrieval
  caches.

The paper-facing plot/table CSVs remain in `paper/after_report/tables/`. Figure
3 can be regenerated with:

```bash
python3 paper/after_report/scripts/regenerate_figure3_from_final_csv.py
```

To rebuild these summaries from raw logs/caches:

```bash
python3 paper/after_report/scripts/build_repro_bundle.py
```
"""
    (OUT_DIR / "README.md").write_text(readme)
    print(OUT_DIR)


if __name__ == "__main__":
    main()
