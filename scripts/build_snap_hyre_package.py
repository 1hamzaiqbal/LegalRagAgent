#!/usr/bin/env python3
"""Build a source-gated Snap-HyRE package status report from local artifacts.

The script is intentionally conservative. It reports which full-corpus rows,
retrieval-cache rows, and plots are backed by files that actually exist. Missing
cells remain missing instead of being filled from older narrative docs.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


DATASETS = ["barexam", "housing", "casehold", "legalbench_scalr"]
PROVIDERS = ["groq-llama8b", "or-gemma4-26b", "groq-llama70b"]
MODES = [
    "llm_only",
    "rag_simple",
    "rag_rewrite",
    "rag_hyde",
    "snap_hyre",
    "golden_passage",
    "golden_plus_neighbors",
]
MODE_ALIASES = {
    "rag_snap_hyde_2call": "snap_hyre",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _read_csvs(patterns: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    paths: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            paths.extend(Path(p) for p in matches)
        else:
            path = Path(pattern)
            if path.exists():
                paths.append(path)
    for path in sorted(dict.fromkeys(paths)):
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row = dict(row)
                row["source_csv"] = str(path)
                rows.append(row)
    return rows


def _empty_health(status: str) -> dict[str, Any]:
    return {
        "detail_status": status,
        "detail_rows": 0,
        "missing_pred": 0,
        "error_rows": 0,
        "long_rows": 0,
        "fallback_rows": 0,
        "missing_oracle_rows": 0,
    }


def _detail_health(path: str | None, expected_total: int | None, *, mode: str, failed_tag: bool) -> dict[str, Any]:
    if not path:
        return _empty_health("missing_detail_ref")
    detail_path = Path(path)
    if not detail_path.exists():
        return _empty_health("missing_detail_file")

    rows = _read_jsonl(detail_path)
    missing_pred = 0
    error_rows = 0
    long_rows = 0
    fallback_rows = 0
    missing_oracle_rows = 0
    for row in rows:
        pred = row.get("predicted_answer")
        if pred is None or str(pred).strip() == "":
            missing_pred += 1
        if row.get("error"):
            error_rows += 1
        answer = str(row.get("final_answer") or row.get("answer") or "")
        if len(answer) > 20000:
            long_rows += 1
        routed_to = str(row.get("routed_to") or "")
        if "fallback" in routed_to.lower():
            fallback_rows += 1
        elif any(key.endswith("_fallback") and value for key, value in row.items()):
            fallback_rows += 1
        elif row.get("snap_hyre_parse_ok") is False or row.get("snap_hyde_2call_parse_ok") is False:
            fallback_rows += 1
        if mode in {"golden_passage", "golden_plus_neighbors"}:
            if not row.get("gold_retrieved") or not row.get("evidence_store"):
                missing_oracle_rows += 1

    status = "clean"
    if failed_tag:
        status = "failed_tag"
    if expected_total is not None and len(rows) != expected_total:
        status = "row_count_mismatch"
    if missing_pred or error_rows or long_rows or fallback_rows or missing_oracle_rows:
        status = "caveated"
    if failed_tag:
        status = "failed_tag"
    return {
        "detail_status": status,
        "detail_rows": len(rows),
        "missing_pred": missing_pred,
        "error_rows": error_rows,
        "long_rows": long_rows,
        "fallback_rows": fallback_rows,
        "missing_oracle_rows": missing_oracle_rows,
    }


def _load_answer_rows(experiments_path: Path, tag_prefix: str, min_questions: int) -> list[dict[str, Any]]:
    rows = []
    for exp in _read_jsonl(experiments_path):
        dataset = str(exp.get("dataset") or "barexam")
        provider = str(exp.get("provider") or "")
        mode = MODE_ALIASES.get(str(exp.get("mode") or ""), str(exp.get("mode") or ""))
        tag = str(exp.get("tag") or "")
        if dataset not in DATASETS or provider not in PROVIDERS or mode not in MODES:
            continue
        if tag_prefix and not tag.startswith(tag_prefix):
            continue
        total = exp.get("total") or exp.get("n_questions")
        try:
            expected_total = int(total) if total is not None else None
        except (TypeError, ValueError):
            expected_total = None
        if expected_total is not None and expected_total < min_questions:
            continue
        failed_tag = "_FAILED" in tag or "do-not-use" in tag
        health = _detail_health(
            exp.get("detail_log"),
            expected_total,
            mode=mode,
            failed_tag=failed_tag,
        )
        rows.append({
            "dataset": dataset,
            "provider": provider,
            "mode": mode,
            "run_id": exp.get("run_id", ""),
            "tag": tag,
            "n_questions": exp.get("n_questions", ""),
            "accuracy": exp.get("accuracy", ""),
            "correct": exp.get("correct", ""),
            "total": exp.get("total", ""),
            "avg_llm_calls": exp.get("avg_llm_calls", ""),
            "total_input_tokens": exp.get("total_input_tokens", ""),
            "total_output_tokens": exp.get("total_output_tokens", ""),
            "detail_log": exp.get("detail_log", ""),
            **health,
        })
    return rows


def _load_answer_rows_from_details(patterns: list[str], min_questions: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    paths: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            paths.extend(Path(p) for p in matches)
    for path in sorted(dict.fromkeys(paths)):
        detail_rows = _read_jsonl(path)
        if len(detail_rows) < min_questions:
            continue
        first = detail_rows[0]
        dataset = str(first.get("dataset") or "")
        provider = str(first.get("provider") or "")
        mode = MODE_ALIASES.get(str(first.get("mode") or ""), str(first.get("mode") or ""))
        if dataset not in DATASETS or provider not in PROVIDERS or mode not in MODES:
            continue
        correct = sum(1 for row in detail_rows if row.get("is_correct"))
        total = len(detail_rows)
        health = _detail_health(str(path), total, mode=mode, failed_tag=False)
        rows.append({
            "dataset": dataset,
            "provider": provider,
            "mode": mode,
            "run_id": path.stem,
            "tag": "detail-log-scan",
            "n_questions": total,
            "accuracy": correct / total if total else "",
            "correct": correct,
            "total": total,
            "avg_llm_calls": "",
            "total_input_tokens": "",
            "total_output_tokens": "",
            "detail_log": str(path),
            **health,
        })
    return rows


def _latest_answer_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}

    def sort_key(row: dict[str, Any]) -> tuple[int, str]:
        raw_total = row.get("total") or row.get("n_questions") or row.get("detail_rows") or 0
        try:
            total = int(raw_total)
        except (TypeError, ValueError):
            total = 0
        return total, str(row.get("run_id", ""))

    for row in rows:
        key = (row["dataset"], row["provider"], row["mode"])
        if key not in by_key or sort_key(row) > sort_key(by_key[key]):
            by_key[key] = row
    return [by_key[key] for key in sorted(by_key)]


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt_pct(value: Any) -> str:
    try:
        return f"{100 * float(value):.1f}%"
    except (TypeError, ValueError):
        return ""


def _answer_pivot(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, dict[str, Any]]]:
    pivot: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        pivot[(row["provider"], row["dataset"])][row["mode"]] = row
    return pivot


def _retrieval_summary(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    clean = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for row in rows:
        if row.get("scope") != "cache":
            continue
        if row.get("method") not in {"rag_simple", "rag_hyde", "snap_hyre", "golden_plus_neighbors"}:
            continue
        key = (
            row.get("dataset", ""),
            row.get("model", ""),
            row.get("method", ""),
            str(row.get("k", "")),
            row.get("path", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        clean.append(row)
    return clean


def _write_markdown(
    path: Path,
    answer_rows: list[dict[str, Any]],
    retrieval_rows: list[dict[str, str]],
    tag_prefix: str,
    min_answer_questions: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pivot = _answer_pivot(answer_rows)
    missing_answer = []
    for provider in PROVIDERS:
        for dataset in DATASETS:
            modes = pivot.get((provider, dataset), {})
            for mode in MODES:
                if mode not in modes:
                    missing_answer.append((provider, dataset, mode))

    retrieval_seen = {
        (row.get("dataset", ""), row.get("model", ""), row.get("method", ""), str(row.get("k", "")))
        for row in retrieval_rows
    }

    with path.open("w") as f:
        f.write("# Snap-HyRE Package Status\n\n")
        f.write("This file is generated from local artifacts only. Missing cells are not inferred from older docs.\n\n")
        f.write(f"- Experiments tag prefix: `{tag_prefix or '(none)'}`\n")
        f.write(f"- Minimum answer-row questions: `{min_answer_questions}`\n")
        f.write(f"- Latest answer rows found: {len(answer_rows)} / {len(DATASETS) * len(PROVIDERS) * len(MODES)} expected cells\n")
        f.write(f"- Retrieval matrix rows found: {len(retrieval_rows)}\n\n")

        f.write("## Answer Ladder\n\n")
        f.write("| provider | dataset | " + " | ".join(MODES) + " |\n")
        f.write("|---|---|" + "|".join(["---:"] * len(MODES)) + "|\n")
        for provider in PROVIDERS:
            for dataset in DATASETS:
                modes = pivot.get((provider, dataset), {})
                cells = []
                for mode in MODES:
                    row = modes.get(mode)
                    if not row:
                        cells.append("missing")
                        continue
                    status = row.get("detail_status", "")
                    suffix = "" if status == "clean" else f" ({status})"
                    cells.append(f"{_fmt_pct(row.get('accuracy'))}{suffix}")
                f.write(f"| {provider} | {dataset} | " + " | ".join(cells) + " |\n")

        f.write("\n## Retrieval Top-k Rows\n\n")
        if retrieval_rows:
            f.write("| dataset | model | method | k | rows | Hit@k | MRR@k | qrels | health |\n")
            f.write("|---|---|---|---:|---:|---:|---:|---|---|\n")
            for row in retrieval_rows:
                health = (
                    f"empty={row.get('empty_retrieval', '')}, "
                    f"short={row.get('short_rows', '')}, "
                    f"no_gold={row.get('rows_without_gold', '')}"
                )
                f.write(
                    f"| {row.get('dataset', '')} | {row.get('model', '')} | {row.get('method', '')} | "
                    f"{row.get('k', '')} | {row.get('rows', '')} | {_fmt_pct(row.get('hit'))} | "
                    f"{float(row.get('mrr') or 0):.3f} | {row.get('qrel_status', '')} | {health} |\n"
                )
        else:
            f.write("No retrieval matrix rows found yet.\n")

        f.write("\n## Missing Answer Cells\n\n")
        if missing_answer:
            f.write("| provider | dataset | mode |\n|---|---|---|\n")
            for provider, dataset, mode in missing_answer:
                f.write(f"| {provider} | {dataset} | {mode} |\n")
        else:
            f.write("No missing answer cells for the expected grid.\n")

        f.write("\n## Retrieval Coverage Notes\n\n")
        expected_methods = {"rag_simple", "rag_hyde", "snap_hyre", "golden_plus_neighbors"}
        coverage_notes = []
        for dataset in DATASETS:
            methods = {row[2] for row in retrieval_seen if row[0] == dataset}
            missing = sorted(expected_methods - methods)
            if missing:
                coverage_notes.append(f"- `{dataset}` missing retrieval rows for: {', '.join(missing)}")
        if coverage_notes:
            f.write("\n".join(coverage_notes) + "\n")
        else:
            f.write("All expected retrieval method families have at least one cache row.\n")


def _maybe_write_plots(out_dir: Path, answer_rows: list[dict[str, Any]], retrieval_rows: list[dict[str, str]]) -> list[str]:
    written: list[str] = []
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return written

    if retrieval_rows:
        grouped: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
        for row in retrieval_rows:
            try:
                k = int(row["k"])
                hit = float(row["hit"])
            except (KeyError, TypeError, ValueError):
                continue
            label = (row.get("dataset", ""), row.get("method", ""))
            grouped[label].append((k, hit))
        if grouped:
            fig, ax = plt.subplots(figsize=(10, 5.5))
            for (dataset, method), points in sorted(grouped.items()):
                xs, ys = zip(*sorted(points))
                ax.plot(xs, [100 * y for y in ys], marker="o", label=f"{dataset}/{method}")
            ax.set_xlabel("retrieval k")
            ax.set_ylabel("Hit@k (%)")
            ax.set_title("Retrieval Exposure By k")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=7, ncol=2)
            fig.tight_layout()
            out = out_dir / "retrieval_hit_at_k.png"
            fig.savefig(out, dpi=200)
            plt.close(fig)
            written.append(str(out))

    clean_answer_rows = [row for row in answer_rows if row.get("detail_status") == "clean"]
    if clean_answer_rows:
        by_mode: dict[str, list[float]] = defaultdict(list)
        for row in clean_answer_rows:
            try:
                by_mode[row["mode"]].append(float(row["accuracy"]))
            except (TypeError, ValueError):
                pass
        if by_mode:
            modes = [mode for mode in MODES if mode in by_mode]
            vals = [100 * sum(by_mode[mode]) / len(by_mode[mode]) for mode in modes]
            fig, ax = plt.subplots(figsize=(9, 4.8))
            ax.bar(modes, vals, color="#4c78a8")
            ax.set_ylabel("Mean clean-cell accuracy (%)")
            ax.set_title("Current Clean Answer Rows")
            ax.tick_params(axis="x", rotation=25)
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            out = out_dir / "answer_accuracy_by_method.png"
            fig.savefig(out, dpi=200)
            plt.close(fig)
            written.append(str(out))

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", type=Path, default=Path("logs/experiments.jsonl"))
    parser.add_argument("--detail-log", action="append", default=["logs/merged/*_detail.jsonl"])
    parser.add_argument("--retrieval-csv", action="append", default=["docs/generated/retrieval_cache_matrix*.csv"])
    parser.add_argument("--out-dir", type=Path, default=Path("docs/generated/snap_hyre_package"))
    parser.add_argument("--tag-prefix", default="local-snap-hyre")
    parser.add_argument("--min-answer-questions", type=int, default=50)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    answer_rows = _latest_answer_rows(
        _load_answer_rows(args.experiments, args.tag_prefix, args.min_answer_questions)
        + _load_answer_rows_from_details(args.detail_log, args.min_answer_questions)
    )
    retrieval_rows = _retrieval_summary(_read_csvs(args.retrieval_csv))

    answer_fields = [
        "provider", "dataset", "mode", "run_id", "tag", "n_questions", "accuracy",
        "correct", "total", "avg_llm_calls", "total_input_tokens", "total_output_tokens",
        "detail_status", "detail_rows", "missing_pred", "error_rows", "long_rows",
        "fallback_rows", "missing_oracle_rows", "detail_log",
    ]
    retrieval_fields = [
        "scope", "dataset", "model", "method", "k", "rows", "scored_rows", "hit", "recall", "mrr",
        "qrel_status", "qrel_exists_fraction", "duplicate_keys", "missing_idx", "empty_retrieval",
        "short_rows", "rows_without_gold", "path", "qrel_report", "source_csv",
    ]

    _write_csv(args.out_dir / "answer_ladder_status.csv", answer_rows, answer_fields)
    _write_csv(args.out_dir / "retrieval_topk_status.csv", retrieval_rows, retrieval_fields)
    _write_markdown(
        args.out_dir / "package_status.md",
        answer_rows,
        retrieval_rows,
        args.tag_prefix,
        args.min_answer_questions,
    )
    written_plots = _maybe_write_plots(args.out_dir, answer_rows, retrieval_rows)

    print(f"wrote {args.out_dir / 'package_status.md'}")
    print(f"wrote {args.out_dir / 'answer_ladder_status.csv'}")
    print(f"wrote {args.out_dir / 'retrieval_topk_status.csv'}")
    if written_plots:
        for path in written_plots:
            print(f"wrote {path}")
    else:
        print("plots not written: matplotlib unavailable or no plottable rows")


if __name__ == "__main__":
    main()
