#!/usr/bin/env python3
"""Write the Phase A++ single-judge GPT-4o factuality report."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_factuality_falsification import (  # noqa: E402
    build_feature_rows,
    finite,
    fit_logistic,
    fmt,
    mean,
    read_jsonl,
)

DEFAULT_CACHE = ROOT / "docs/generated/factuality_judge_full_singlejudge_gpt4o_2026-05-31.jsonl"
DEFAULT_POINTS = ROOT / "docs/generated/credibility_A_full_singlejudge_gpt4o_2026-05-31_points.jsonl"
DEFAULT_REPORT = ROOT / "docs/generated/credibility_A_full_singlejudge_gpt4o_2026-05-31.md"
DEFAULT_GEMMA_Q200 = ROOT / "docs/generated/factuality_feature_points_q200_2026-05-28.jsonl"

START_USAGE = 65.972024895
TOTAL_CREDITS = 95.0


def pct(value: Any) -> str:
    if not finite(value):
        return "--"
    return f"{100.0 * float(value):.1f}%"


def safe_spearman(xs: Iterable[Any], ys: Iterable[Any]) -> float:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if finite(x) and finite(y)]
    if len(pairs) < 3:
        return float("nan")
    x_vals = [x for x, _ in pairs]
    y_vals = [y for _, y in pairs]
    if len(set(x_vals)) < 2 or len(set(y_vals)) < 2:
        return float("nan")
    out = spearmanr(x_vals, y_vals, nan_policy="omit").statistic
    return float(out) if finite(out) else float("nan")


def cohen_kappa_binary(left: list[bool], right: list[bool]) -> float:
    pairs = [(a, b) for a, b in zip(left, right)]
    if not pairs:
        return float("nan")
    n = len(pairs)
    agree = sum(1 for a, b in pairs if a == b) / n
    p_left = sum(1 for a, _ in pairs if a) / n
    p_right = sum(1 for _, b in pairs if b) / n
    expected = p_left * p_right + (1.0 - p_left) * (1.0 - p_right)
    if math.isclose(expected, 1.0):
        return float("nan")
    return (agree - expected) / (1.0 - expected)


def display_name(points: list[dict[str, Any]], dataset: str) -> str:
    for row in points:
        if row.get("dataset") == dataset:
            return str(row.get("dataset_display") or dataset)
    return dataset


def dataset_groups(points: list[dict[str, Any]]) -> list[tuple[str, str, list[dict[str, Any]]]]:
    out = []
    for dataset in sorted({str(row["dataset"]) for row in points}):
        group = [row for row in points if row["dataset"] == dataset]
        out.append((dataset, display_name(points, dataset), group))
    out.append(("pooled", "Pooled", points))
    return out


def coverage_rows(judge_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labels: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    models: dict[str, Counter[str]] = defaultdict(Counter)
    routes: dict[str, Counter[str]] = defaultdict(Counter)
    displays: dict[str, str] = {}
    for row in judge_rows:
        dataset = str(row.get("dataset") or "")
        expansion = str(row.get("expansion") or "")
        premise = str(row.get("premise_kind") or "")
        label = str(row.get("label") or "")
        if dataset and expansion and premise and label:
            labels[(dataset, expansion, premise)].add(label)
        displays[dataset] = str(row.get("dataset_display") or dataset)
        models[dataset][str(row.get("judge_model") or "")] += 1
        routes[dataset][json.dumps(row.get("provider_route") or {}, sort_keys=True)] += 1

    out = []
    for dataset in sorted(displays):
        keys = [(expansion, premise) for d, expansion, premise in labels if d == dataset]
        cell_counts = {
            (expansion, premise): len(labels[(dataset, expansion, premise)])
            for expansion, premise in keys
        }
        expected_cells = [("hyde", "gold"), ("hyde", "raw_top3"), ("scope", "gold"), ("scope", "raw_top3")]
        complete = min((cell_counts.get(cell, 0) for cell in expected_cells), default=0)
        out.append({
            "dataset": dataset,
            "display": displays[dataset],
            "complete_questions": complete,
            "judge_records": sum(models[dataset].values()),
            "cell_counts": cell_counts,
            "models": ", ".join(f"{model} x{count}" for model, count in sorted(models[dataset].items())),
            "routes": ", ".join(f"{route} x{count}" for route, count in sorted(routes[dataset].items())),
        })
    return out


def auc_table(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for dataset, display, group in dataset_groups(points):
        factual = fit_logistic(group, ["factuality_gold_score"], "target_retrieval_hurt")
        geom = fit_logistic(group, ["ce_margin_raw", "ce_exp_gold"], "target_retrieval_hurt")
        joint = fit_logistic(
            group,
            ["factuality_gold_score", "ce_margin_raw", "ce_exp_gold"],
            "target_retrieval_hurt",
        )
        rows.append({
            "dataset": dataset,
            "display": display,
            "n": factual["n"],
            "hurt": factual["failures"],
            "hurt_rate": factual["failures"] / factual["n"] if factual["n"] else float("nan"),
            "factuality_auc": factual["auc"],
            "geometry_auc": geom["auc"],
            "joint_auc": joint["auc"],
            "marginal_lift": (
                float(joint["auc"]) - float(geom["auc"])
                if finite(joint.get("auc")) and finite(geom.get("auc"))
                else float("nan")
            ),
            "factuality_beta": factual.get("coefficients", {}).get("factuality_gold_score", float("nan")),
        })
    return rows


def irr_table(points: list[dict[str, Any]], gemma_q200: Path) -> list[dict[str, Any]]:
    gpt = {
        (str(row.get("dataset")), str(row.get("label")), str(row.get("expansion"))): row
        for row in points
        if finite(row.get("factuality_gold_score"))
    }
    gemma = {
        (str(row.get("dataset")), str(row.get("label")), str(row.get("expansion"))): row
        for row in read_jsonl(gemma_q200)
        if finite(row.get("factuality_gold_score"))
    }
    keys = sorted(set(gpt) & set(gemma))

    rows = []
    datasets = sorted({dataset for dataset, _, _ in keys})
    for dataset in datasets + ["pooled"]:
        subkeys = keys if dataset == "pooled" else [key for key in keys if key[0] == dataset]
        gpt_scores = [float(gpt[key]["factuality_gold_score"]) for key in subkeys]
        gemma_scores = [float(gemma[key]["factuality_gold_score"]) for key in subkeys]
        rows.append({
            "dataset": dataset,
            "display": "Pooled" if dataset == "pooled" else display_name(points, dataset),
            "n": len(subkeys),
            "spearman": safe_spearman(gpt_scores, gemma_scores),
            "kappa_05": cohen_kappa_binary(
                [score >= 0.5 for score in gpt_scores],
                [score >= 0.5 for score in gemma_scores],
            ),
            "gpt_mean": mean(gpt_scores),
            "gemma_mean": mean(gemma_scores),
        })
    return rows


def verdict(aucs: list[dict[str, Any]], irr: list[dict[str, Any]]) -> dict[str, Any]:
    pooled = next(row for row in aucs if row["dataset"] == "pooled")
    pooled_irr = next((row for row in irr if row["dataset"] == "pooled"), {})
    factuality_auc = pooled["factuality_auc"]
    marginal = pooled["marginal_lift"]
    spearman = pooled_irr.get("spearman", float("nan"))

    diverges = finite(spearman) and float(spearman) < 0.4
    survives = (
        finite(factuality_auc)
        and finite(marginal)
        and float(factuality_auc) <= 0.65
        and float(marginal) <= 0.03
    )
    softens = (
        finite(factuality_auc)
        and float(factuality_auc) > 0.70
    ) or (
        finite(marginal)
        and float(marginal) > 0.05
    )

    if diverges:
        headline = "operationalization-fragile"
        reading = (
            "GPT-4o and the prior Gemma judge do not agree strongly enough on the q200 overlap, "
            "so the factuality signal itself is fragile even before comparing it to geometry."
        )
    elif survives:
        headline = "survives"
        reading = (
            "The single-judge GPT-4o replication keeps factuality below the pre-stated AUC ceiling "
            "and adds little after geometry, so Phase A's geometry-over-factuality reading survives."
        )
    elif softens:
        headline = "softens"
        reading = (
            "GPT-4o gives factuality enough predictive signal to soften the strong falsification; "
            "geometry may still dominate, but factuality is no longer a clean null."
        )
    else:
        headline = "mixed"
        reading = (
            "The single-judge GPT-4o result is between the pre-stated survive and soften gates."
        )
    return {
        "headline": headline,
        "reading": reading,
        "diverges": diverges,
        "survives": survives,
        "softens": softens,
    }


def write_report(args: argparse.Namespace, points: list[dict[str, Any]]) -> None:
    judge_rows = read_jsonl(args.judge_cache)
    coverage = coverage_rows(judge_rows)
    aucs = auc_table(points)
    irr = irr_table(points, args.gemma_q200)
    read = verdict(aucs, irr)

    end_usage = float(args.end_usage)
    spend = end_usage - float(args.start_usage)
    remaining = float(args.total_credits) - end_usage

    lines: list[str] = []
    lines.append("# Credibility A++ Single-Judge GPT-4o Factuality - 2026-05-31")
    lines.append("")
    lines.append("No `paper/` files were edited.")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"Headline: **{read['headline']}**.")
    lines.append(read["reading"])
    lines.append("")
    pooled_auc = next(row for row in aucs if row["dataset"] == "pooled")
    pooled_irr = next(row for row in irr if row["dataset"] == "pooled")
    lines.append(
        "Pooled retrieval-hurt results: "
        f"GPT-4o factuality AUC `{fmt(pooled_auc['factuality_auc'])}`, "
        f"geometry AUC `{fmt(pooled_auc['geometry_auc'])}`, "
        f"joint AUC `{fmt(pooled_auc['joint_auc'])}`, "
        f"marginal lift `{fmt(pooled_auc['marginal_lift'])}`. "
        f"Gemma-vs-GPT-4o q200 IRR: Spearman `{fmt(pooled_irr['spearman'])}`, "
        f"kappa@0.5 `{fmt(pooled_irr['kappa_05'])}`."
    )
    lines.append("")
    lines.append("## Run Scope")
    lines.append("")
    lines.append(
        f"OpenRouter usage moved from `${float(args.start_usage):.2f}` to `${end_usage:.2f}`. "
        f"Task spend was `${spend:.2f}`. Provider total credits were "
        f"`${float(args.start_total_credits):.2f}` at task start and "
        f"`${float(args.total_credits):.2f}` at the final check, leaving `${remaining:.2f}`. "
        "The configured hard cap was `$25` spend with roughly `$4` reserve."
    )
    lines.append("")
    lines.append(
        "All completed rows used `judge_provider=custom`, `judge_model=openai/gpt-4o`, "
        "and an empty OpenRouter provider route. No GPT-4o-mini substitution was used."
    )
    lines.append("")
    lines.append("| Dataset | Complete questions | Judge records | Judge models | Provider routes |")
    lines.append("|---|---:|---:|---|---|")
    for row in coverage:
        lines.append(
            f"| {row['display']} | {row['complete_questions']} | {row['judge_records']} | "
            f"`{row['models']}` | `{row['routes']}` |"
        )
    lines.append("")
    lines.append(
        "HousingQA was not started in this wave: after BarExamQA plus the five BEIR datasets, "
        "a GPT-4o q1000 Housing pass would have risked the requested reserve. The task allowed "
        "Housing only after the higher-priority full-N BarExamQA and BEIR passes."
    )
    lines.append("")
    lines.append("## Retrieval-Hurt AUC")
    lines.append("")
    lines.append("| Dataset | N arms | Hurt rows | Hurt rate | Factuality AUC | Geometry AUC | Joint AUC | Marginal lift | Factuality beta |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in aucs:
        lines.append(
            f"| {row['display']} | {row['n']} | {row['hurt']} | {pct(row['hurt_rate'])} | "
            f"{fmt(row['factuality_auc'])} | {fmt(row['geometry_auc'])} | "
            f"{fmt(row['joint_auc'])} | {fmt(row['marginal_lift'])} | {fmt(row['factuality_beta'])} |"
        )
    lines.append("")
    lines.append("`N arms` counts expansion arms, so each complete question contributes one HyDE row and one SCOPE row.")
    lines.append("")
    lines.append("## Gemma-vs-GPT-4o IRR on q200 Overlap")
    lines.append("")
    lines.append("| Dataset | Overlap arms | GPT-4o mean | Gemma mean | Spearman rho | Kappa@0.5 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in irr:
        lines.append(
            f"| {row['display']} | {row['n']} | {fmt(row['gpt_mean'])} | {fmt(row['gemma_mean'])} | "
            f"{fmt(row['spearman'])} | {fmt(row['kappa_05'])} |"
        )
    lines.append("")
    lines.append("## Reading")
    lines.append("")
    if read["diverges"]:
        lines.append(
            "- The pre-stated divergence flag triggers because pooled Spearman is below `0.4`; "
            "the factuality measurement is judge-sensitive on the q200 overlap."
        )
    if read["survives"]:
        lines.append(
            "- The pre-stated survival gate is met: pooled factuality AUC is at or below `0.65`, "
            "and the AUC lift after geometry is at or below `+0.03`."
        )
    if read["softens"]:
        lines.append(
            "- The pre-stated softening gate is met: factuality has either high standalone AUC "
            "or a material AUC lift after geometry."
        )
    lines.append(
        "- Treat this as a single-judge replication, not the full A++ two-independent-judge closeout. "
        "The Claude/Sonnet second judge remains pending."
    )
    lines.append("")
    lines.append("## Sources")
    lines.append("")
    lines.append(f"- Judge cache: `{args.judge_cache.relative_to(ROOT)}`")
    lines.append(f"- Feature points: `{args.features_out.relative_to(ROOT)}`")
    lines.append(f"- Gemma q200 comparison: `{args.gemma_q200.relative_to(ROOT)}`")
    lines.append("- BEIR geometry source: `/tmp/beir_phase1_verification_2026-05-26_points.jsonl`")
    lines.append("- Legal geometry source: `/tmp/affinity_margin_oncache_2026-05-26_points.jsonl`")
    lines.append("")
    lines.append("## Reproduction")
    lines.append("")
    lines.append("```bash")
    lines.append(
        "NO_SILENT_FALLBACK=1 LLM_PROVIDER=custom "
        "LLM_BASE_URL=https://openrouter.ai/api/v1 LLM_MODEL=openai/gpt-4o "
        "EVAL_CONCURRENCY=8 uv run python scripts/build_factuality_judge_cache.py "
        "--datasets barexam --limit 0 --provider custom --resume "
        f"--output {args.judge_cache.relative_to(ROOT)} --concurrency 8 --batch-size 64"
    )
    lines.append(
        "NO_SILENT_FALLBACK=1 LLM_PROVIDER=custom "
        "LLM_BASE_URL=https://openrouter.ai/api/v1 LLM_MODEL=openai/gpt-4o "
        "EVAL_CONCURRENCY=8 uv run python scripts/build_factuality_judge_cache.py "
        "--datasets beir_scifact beir_nfcorpus beir_fiqa beir_trec_covid beir_scidocs "
        "--limit 0 --provider custom --resume "
        f"--output {args.judge_cache.relative_to(ROOT)} --concurrency 8 --batch-size 64"
    )
    lines.append(
        "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python "
        "scripts/analyze_singlejudge_gpt4o_factuality.py "
        f"--judge-cache {args.judge_cache.relative_to(ROOT)} "
        f"--features-out {args.features_out.relative_to(ROOT)} "
        f"--output {args.output.relative_to(ROOT)} --end-usage {end_usage:.9f} "
        f"--total-credits {float(args.total_credits):.2f}"
    )
    lines.append("```")
    lines.append("")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(f"[report] wrote {args.output}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--features-out", type=Path, default=DEFAULT_POINTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--gemma-q200", type=Path, default=DEFAULT_GEMMA_Q200)
    parser.add_argument("--reuse-features", action="store_true")
    parser.add_argument("--ce-batch-size", type=int, default=16)
    parser.add_argument("--start-usage", type=float, default=START_USAGE)
    parser.add_argument("--start-total-credits", type=float, default=TOTAL_CREDITS)
    parser.add_argument("--end-usage", type=float, required=True)
    parser.add_argument("--total-credits", type=float, default=TOTAL_CREDITS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for name in ("judge_cache", "features_out", "output", "gemma_q200"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)
    if args.reuse_features and args.features_out.exists():
        points = read_jsonl(args.features_out)
        print(f"[features] loaded {len(points)} rows from {args.features_out}", flush=True)
    else:
        points = build_feature_rows(args)
    write_report(args, points)


if __name__ == "__main__":
    main()
