#!/usr/bin/env python3
"""Compare explanatory axes for SCOPE-vs-raw benefit.

This reuses the dataset/source definitions and corpus-perplexity utilities from
``analyze_perplexity_axis.py``. It adds question length and raw-retrieval
difficulty axes while keeping the same signed retrieval caches and answer logs.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from analyze_perplexity_axis import (  # noqa: E402
    DATASETS,
    MODELS,
    MODEL_LABELS,
    build_or_load_lm,
    correlation_row,
    fmt_float,
    hit_at_5,
    load_by_label,
    mean,
    median,
    pct,
    question_scores,
    source_paths_for,
    spearman,
    pearson,
)


AXES = {
    "question_tokens": {
        "label": "Question tokens",
        "direction": "higher = longer / more specific",
    },
    "log_perplexity": {
        "label": "Log perplexity",
        "direction": "higher = less corpus-like",
    },
    "raw_hit_at5": {
        "label": "Raw Hit@5",
        "direction": "higher = raw already retrieved gold",
    },
    "raw_gold_rank_at10": {
        "label": "Raw gold rank@10",
        "direction": "higher = harder; 11 means not in raw top-10",
    },
}


def first_gold_rank(retrieved_ids: list[Any], gold_ids: list[str], *, cap: int = 10) -> int:
    gold = {str(x) for x in gold_ids if str(x)}
    if not gold:
        return cap + 1
    for rank, idx in enumerate([str(x) for x in retrieved_ids[:cap]], 1):
        if idx in gold:
            return rank
    return cap + 1


def build_axis_points(dataset: str, lm_cache_dir: Path, batch_size: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    spec = DATASETS[dataset]
    lms = build_or_load_lm(spec, lm_cache_dir, batch_size)
    q_scores = question_scores(spec, lms)
    raw_cache = load_by_label(spec.raw_cache)
    points: list[dict[str, Any]] = []

    for model in MODELS:
        scope_cache = load_by_label(spec.scope_cache_by_model[model])
        raw_log = load_by_label(spec.raw_log_by_model[model])
        scope_log = load_by_label(spec.scope_log_by_model[model])
        missing = [
            label
            for label in q_scores
            if label not in raw_cache or label not in scope_cache or label not in raw_log or label not in scope_log
        ]
        if missing:
            raise RuntimeError(f"{dataset}/{model}: missing labels {missing[:5]} n={len(missing)}")

        for label, score in q_scores.items():
            gold = score["gold_ids"]
            raw_ids = raw_cache[label].get("retrieved_ids") or []
            scope_ids = scope_cache[label].get("retrieved_ids") or []
            raw_hit = hit_at_5(raw_ids, gold)
            scope_hit = hit_at_5(scope_ids, gold)
            raw_correct = int(bool(raw_log[label].get("is_correct")))
            scope_correct = int(bool(scope_log[label].get("is_correct")))
            rank = first_gold_rank(raw_ids, gold, cap=10)
            points.append({
                "dataset": dataset,
                "dataset_display": spec.display,
                "model": model,
                "label": label,
                "question_tokens": float(score["token_count"]),
                "log_perplexity": float(score["log_perplexity"]),
                "raw_hit_at5": float(raw_hit),
                "raw_gold_rank_at10": float(rank),
                "raw_missed_at10": float(rank == 11),
                "scope_retrieval_win": int(scope_hit == 1 and raw_hit == 0),
                "raw_retrieval_win": int(raw_hit == 1 and scope_hit == 0),
                "retrieval_delta": int(scope_hit) - int(raw_hit),
                "scope_answer_win": int(scope_correct == 1 and raw_correct == 0),
                "raw_answer_win": int(raw_correct == 1 and scope_correct == 0),
                "answer_delta": int(scope_correct) - int(raw_correct),
                "raw_correct": raw_correct,
                "scope_correct": scope_correct,
            })
    question_summary = {
        "n_questions": len(q_scores),
        "median_tokens": median([float(v["token_count"]) for v in q_scores.values()]),
        "median_ppl": median([float(v["perplexity"]) for v in q_scores.values()]),
        "mean_log_ppl": mean([float(v["log_perplexity"]) for v in q_scores.values()]),
    }
    return points, question_summary


def summarize_outcomes(points: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "n": len(points),
        "scope_retrieval_win": mean([p["scope_retrieval_win"] for p in points]),
        "raw_retrieval_win": mean([p["raw_retrieval_win"] for p in points]),
        "retrieval_delta": mean([p["retrieval_delta"] for p in points]),
        "scope_answer_win": mean([p["scope_answer_win"] for p in points]),
        "raw_answer_win": mean([p["raw_answer_win"] for p in points]),
        "answer_delta": mean([p["answer_delta"] for p in points]),
        "raw_accuracy": mean([p["raw_correct"] for p in points]),
        "scope_accuracy": mean([p["scope_correct"] for p in points]),
    }


def axis_correlation(points: list[dict[str, Any]], axis: str) -> dict[str, Any]:
    x = [float(p[axis]) for p in points]
    return {
        "axis": axis,
        "n": len(points),
        "pearson_retrieval": pearson(x, [p["retrieval_delta"] for p in points]),
        "spearman_retrieval": spearman(x, [p["retrieval_delta"] for p in points]),
        "pearson_answer": pearson(x, [p["answer_delta"] for p in points]),
        "spearman_answer": spearman(x, [p["answer_delta"] for p in points]),
    }


def binned_curve(points: list[dict[str, Any]], axis: str, bins: int = 5) -> list[dict[str, Any]]:
    ordered = sorted(points, key=lambda p: (float(p[axis]), p["label"], p["model"]))
    n = len(ordered)
    out: list[dict[str, Any]] = []
    for b in range(bins):
        lo = round(b * n / bins)
        hi = round((b + 1) * n / bins)
        chunk = ordered[lo:hi]
        if not chunk:
            continue
        vals = [float(p[axis]) for p in chunk]
        s = summarize_outcomes(chunk)
        out.append({
            "bin": b + 1,
            "axis_min": min(vals),
            "axis_median": median(vals),
            "axis_max": max(vals),
            **s,
        })
    return out


def best_axis_for(points: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    rows = {axis: axis_correlation(points, axis) for axis in AXES}
    # Retrieval is the cleaner first-stage signal. Use absolute Spearman for
    # rank robustness; answer deltas remain secondary in the report.
    best = max(rows, key=lambda axis: abs(rows[axis]["spearman_retrieval"]))
    return best, rows[best]


def binned_axis_for(best_axis: str) -> str:
    # raw_hit_at5 is the strongest signal, but it is binary. The corresponding
    # ranked difficulty axis gives a more informative quintile curve.
    if best_axis == "raw_hit_at5":
        return "raw_gold_rank_at10"
    return best_axis


def axis_table(dataset_results: dict[str, Any]) -> list[str]:
    lines = []
    lines.append("| Dataset | Model | Axis | N | Pearson retrieval | Spearman retrieval | Pearson answer | Spearman answer |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    for dataset in ("barexam", "housing"):
        result = dataset_results[dataset]
        for model in (*MODELS, "pooled"):
            points = result["points"] if model == "pooled" else [p for p in result["points"] if p["model"] == model]
            for axis in AXES:
                row = axis_correlation(points, axis)
                lines.append(
                    f"| {result['display']} | {MODEL_LABELS.get(model, 'Pooled')} | {AXES[axis]['label']} | {row['n']} | "
                    f"{fmt_float(row['pearson_retrieval'])} | {fmt_float(row['spearman_retrieval'])} | "
                    f"{fmt_float(row['pearson_answer'])} | {fmt_float(row['spearman_answer'])} |"
                )
    return lines


def make_report(output: Path, dataset_results: dict[str, Any], lm_cache_dir: Path) -> None:
    lines: list[str] = []
    lines.append("# SCOPE Benefit Axes - 2026-05-25")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("This results-lane analysis reuses `scripts/analyze_perplexity_axis.py` and the same signed raw/SCOPE retrieval caches and detail logs. It tests whether SCOPE benefit is better explained by question length/specificity or raw-retrieval difficulty than by unigram corpus perplexity. No files under `paper/` were edited.")
    lines.append("")
    lines.append("Axes tested:")
    lines.append("")
    for axis, meta in AXES.items():
        lines.append(f"- `{axis}`: {meta['label']} ({meta['direction']}).")
    lines.append("")

    lines.append("## Dataset-Level Axis Values")
    lines.append("")
    lines.append("| Dataset | Questions | Median question tokens | Median perplexity | Mean log perplexity | Raw Hit@5 | Raw gold in top-10 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for dataset in ("barexam", "housing"):
        result = dataset_results[dataset]
        raw_points = {}
        for p in result["points"]:
            raw_points.setdefault(p["label"], p)
        raw_hit = mean([p["raw_hit_at5"] for p in raw_points.values()])
        raw_top10 = mean([float(p["raw_gold_rank_at10"] <= 10) for p in raw_points.values()])
        q = result["question_summary"]
        lines.append(
            f"| {result['display']} | {q['n_questions']} | {q['median_tokens']:.0f} | "
            f"{q['median_ppl']:.1f} | {q['mean_log_ppl']:.3f} | {pct(raw_hit)} | {pct(raw_top10)} |"
        )
    lines.append("")

    lines.append("## Correlations")
    lines.append("")
    lines.append("Outcome signs are `SCOPE - raw`: retrieval delta is Hit@5 movement, answer delta is exact-answer correctness movement. For `raw_hit_at5`, a negative correlation means SCOPE helps more when raw retrieval misses.")
    lines.append("")
    lines.extend(axis_table(dataset_results))
    lines.append("")

    lines.append("## Strongest Axis")
    lines.append("")
    lines.append("| Dataset | Strongest pooled axis | Spearman retrieval | Pearson retrieval | Spearman answer | Binned curve axis | Mean retrieval delta | Mean answer delta |")
    lines.append("|---|---|---:|---:|---:|---|---:|---:|")
    for dataset in ("barexam", "housing"):
        result = dataset_results[dataset]
        axis, corr = result["best_axis"]
        baxis = binned_axis_for(axis)
        s = summarize_outcomes(result["points"])
        lines.append(
            f"| {result['display']} | {AXES[axis]['label']} | {fmt_float(corr['spearman_retrieval'])} | "
            f"{fmt_float(corr['pearson_retrieval'])} | {fmt_float(corr['spearman_answer'])} | {AXES[baxis]['label']} | "
            f"{pct(s['retrieval_delta'])} | {pct(s['answer_delta'])} |"
        )
    lines.append("")

    for dataset in ("barexam", "housing"):
        result = dataset_results[dataset]
        axis, _corr = result["best_axis"]
        baxis = binned_axis_for(axis)
        lines.append(f"## {result['display']} Binned Curve")
        lines.append("")
        if baxis != axis:
            lines.append(
                f"The strongest axis is `{axis}`, but it is binary; these quintiles use `{baxis}`, "
                f"the rank-form of the same raw-difficulty signal. Direction: {AXES[baxis]['direction']}."
            )
        else:
            lines.append(f"Quintiles are sorted by `{axis}`: {AXES[axis]['direction']}.")
        lines.append("")
        lines.append("| Bin | N | Axis median | Axis range | SCOPE retrieval win | Raw retrieval win | Net retrieval delta | SCOPE answer win | Raw answer win | Net answer delta |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in binned_curve(result["points"], baxis):
            lines.append(
                f"| {row['bin']} | {row['n']} | {row['axis_median']:.2f} | "
                f"{row['axis_min']:.2f}-{row['axis_max']:.2f} | "
                f"{pct(row['scope_retrieval_win'])} | {pct(row['raw_retrieval_win'])} | {pct(row['retrieval_delta'])} | "
                f"{pct(row['scope_answer_win'])} | {pct(row['raw_answer_win'])} | {pct(row['answer_delta'])} |"
            )
        lines.append("")

    lines.append("## Reading")
    lines.append("")
    be = dataset_results["barexam"]
    hq = dataset_results["housing"]
    be_len = axis_correlation(be["points"], "question_tokens")
    hq_len = axis_correlation(hq["points"], "question_tokens")
    be_raw = axis_correlation(be["points"], "raw_hit_at5")
    hq_raw = axis_correlation(hq["points"], "raw_hit_at5")
    be_ppl = axis_correlation(be["points"], "log_perplexity")
    hq_ppl = axis_correlation(hq["points"], "log_perplexity")
    lines.append(
        f"- Question length strongly separates the datasets at the median level ({be['question_summary']['median_tokens']:.0f} BarExam tokens vs {hq['question_summary']['median_tokens']:.0f} Housing tokens), "
        f"but within-dataset length has weak pooled Spearman correlation with retrieval delta: {fmt_float(be_len['spearman_retrieval'])} on BarExamQA and {fmt_float(hq_len['spearman_retrieval'])} on HousingQA."
    )
    lines.append(
        f"- Raw-retrieval difficulty is the strongest axis in both datasets. `raw_hit_at5` has pooled Spearman correlations of {fmt_float(be_raw['spearman_retrieval'])} on BarExamQA and {fmt_float(hq_raw['spearman_retrieval'])} on HousingQA. "
        "This is partly mechanical because a positive SCOPE-minus-raw retrieval delta requires raw to miss, but it is still the clearest practical gating signal."
    )
    lines.append(
        f"- Log-perplexity remains weaker: pooled Spearman retrieval correlations are {fmt_float(be_ppl['spearman_retrieval'])} on BarExamQA and {fmt_float(hq_ppl['spearman_retrieval'])} on HousingQA. "
        "The previous null result holds."
    )
    lines.append(
        "- Answer-delta correlations are consistently smaller than retrieval-delta correlations. The best explanatory axis for downstream answer movement is therefore still indirect: first identify raw retrieval failures, then test whether SCOPE repairs them without introducing answer-context dilution."
    )
    lines.append("")

    lines.append("## Sources")
    lines.append("")
    seen: list[str] = []
    for dataset in ("barexam", "housing"):
        for path in source_paths_for(DATASETS[dataset]):
            if path not in seen:
                seen.append(path)
    for path in seen:
        lines.append(f"- `{path}`")
    lines.append("")
    lines.append(f"Unigram LM cache directory reused for log-perplexity: `{lm_cache_dir}`")
    lines.append("")
    lines.append("## Reproduction")
    lines.append("")
    lines.append("```bash")
    lines.append("HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python scripts/analyze_scope_benefit_axes.py \\")
    lines.append("  --output docs/generated/scope_benefit_axes_2026-05-25.md")
    lines.append("```")
    lines.append("")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "docs/generated/scope_benefit_axes_2026-05-25.md")
    parser.add_argument("--lm-cache-dir", type=Path, default=Path("/tmp/perplexity_axis_lm_cache_2026-05-25"))
    parser.add_argument("--batch-size", type=int, default=20000)
    parser.add_argument("--datasets", nargs="+", default=["barexam", "housing"], choices=sorted(DATASETS))
    args = parser.parse_args()

    dataset_results: dict[str, Any] = {}
    for dataset in args.datasets:
        points, question_summary = build_axis_points(dataset, args.lm_cache_dir, args.batch_size)
        spec = DATASETS[dataset]
        dataset_results[dataset] = {
            "display": spec.display,
            "points": points,
            "question_summary": question_summary,
        }
        dataset_results[dataset]["best_axis"] = best_axis_for(points)

    make_report(args.output, dataset_results, args.lm_cache_dir)
    print(args.output)


if __name__ == "__main__":
    main()
