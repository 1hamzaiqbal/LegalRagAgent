#!/usr/bin/env python3
"""MedQA-USMLE corpus perplexity pre-screen.

Builds an add-1 unigram LM over the MedRAG textbook Chroma collection and
compares MedQA question perplexity to the existing BarExamQA and HousingQA
perplexity-axis baselines. This phase has no answer-model calls.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "eval"))

from analyze_perplexity_axis import (  # noqa: E402
    DATASETS,
    DatasetSpec,
    build_or_load_lm,
    cohen_d,
    fmt_float,
    mean,
    median,
    pct,
    percentile,
    question_scores,
    rank_auc_greater,
)


MEDQA_SPEC = DatasetSpec(
    key="medqa",
    display="MedQA-USMLE",
    collection="medqa_textbooks",
    raw_cache="",
    scope_cache_by_model={},
    raw_log_by_model={},
    scope_log_by_model={},
)


def _summary(scores: dict[str, dict[str, Any]], *, lm_scope: str) -> dict[str, Any]:
    rows = list(scores.values())
    ppls = [float(row["perplexity"]) for row in rows]
    logs = [float(row["log_perplexity"]) for row in rows]
    toks = [float(row["token_count"]) for row in rows]
    oovs = [float(row["oov_rate"]) for row in rows]
    return {
        "n": len(rows),
        "lm_scope": lm_scope,
        "median_ppl": median(ppls),
        "p25_ppl": percentile(ppls, 0.25),
        "p75_ppl": percentile(ppls, 0.75),
        "mean_log_ppl": mean(logs),
        "median_log_ppl": median(logs),
        "mean_oov_rate": mean(oovs),
        "median_tokens": median(toks),
        "logs": logs,
    }


def _rows_for_report(summaries: dict[str, dict[str, Any]]) -> list[str]:
    display = {
        "barexam": "BarExamQA",
        "housing": "HousingQA state-filtered",
        "medqa": "MedQA-USMLE",
    }
    lines = [
        "| Dataset | Questions | LM scope | Median PPL | IQR PPL | Mean log PPL | Mean OOV rate | Median tokens |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for key in ("barexam", "housing", "medqa"):
        row = summaries[key]
        lines.append(
            f"| {display[key]} | {row['n']} | {row['lm_scope']} | "
            f"{row['median_ppl']:.1f} | {row['p25_ppl']:.1f}-{row['p75_ppl']:.1f} | "
            f"{row['mean_log_ppl']:.3f} | {pct(row['mean_oov_rate'])} | "
            f"{row['median_tokens']:.0f} |"
        )
    return lines


def update_report(report_path: Path, section: str) -> None:
    if not report_path.exists():
        raise FileNotFoundError(report_path)
    text = report_path.read_text()
    marker = "## Phase 3 - Perplexity Pre-Screen"
    next_marker = "## Current Status"
    if marker in text:
        before, rest = text.split(marker, 1)
        if next_marker not in rest:
            raise RuntimeError(f"Could not find {next_marker!r} after existing Phase 3 section")
        _, after = rest.split(next_marker, 1)
        text = before.rstrip() + "\n\n" + section.rstrip() + "\n\n" + next_marker + after
    else:
        if next_marker not in text:
            raise RuntimeError(f"Could not find insertion marker {next_marker!r}")
        before, after = text.split(next_marker, 1)
        text = before.rstrip() + "\n\n" + section.rstrip() + "\n\n" + next_marker + after
    report_path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/perplexity_axis_lm_cache_2026-05-25"))
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--report", type=Path, default=REPO_ROOT / "docs/generated/medqa_usmle_widening_2026-05-26.md")
    args = parser.parse_args()

    specs = {
        "barexam": DATASETS["barexam"],
        "housing": DATASETS["housing"],
        "medqa": MEDQA_SPEC,
    }

    scores: dict[str, dict[str, dict[str, Any]]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for key, spec in specs.items():
        lms = build_or_load_lm(spec, args.cache_dir, args.batch_size)
        scores[key] = question_scores(spec, lms)
        summaries[key] = _summary(
            scores[key],
            lm_scope="per state" if key == "housing" else "corpus-wide",
        )

    medqa_vs_housing_auc = rank_auc_greater(summaries["medqa"]["logs"], summaries["housing"]["logs"])
    medqa_vs_barexam_auc = rank_auc_greater(summaries["medqa"]["logs"], summaries["barexam"]["logs"])
    medqa_minus_housing_d = cohen_d(summaries["medqa"]["logs"], summaries["housing"]["logs"])
    medqa_minus_barexam_d = cohen_d(summaries["medqa"]["logs"], summaries["barexam"]["logs"])
    medqa_housing_gap = summaries["medqa"]["mean_log_ppl"] - summaries["housing"]["mean_log_ppl"]
    medqa_barexam_gap = summaries["medqa"]["mean_log_ppl"] - summaries["barexam"]["mean_log_ppl"]

    go_answer_budget = (
        medqa_housing_gap > 0.10
        and medqa_vs_housing_auc >= 0.60
        and summaries["medqa"]["median_tokens"] > summaries["housing"]["median_tokens"]
    )

    section_lines = [
        "## Phase 3 - Perplexity Pre-Screen",
        "",
        "Add-1 unigram LMs were built over the retrieval corpora and scored against intermediate-generation question text. MedQA uses `medqa_textbooks`; BarExamQA and HousingQA reuse the existing perplexity-axis LM cache.",
        "",
        *_rows_for_report(summaries),
        "",
        "Separation checks on log-perplexity:",
        "",
        "| Comparison | AUC first > second | Cohen's d | Mean log-PPL gap |",
        "|---|---:|---:|---:|",
        f"| MedQA > HousingQA | {medqa_vs_housing_auc:.3f} | {medqa_minus_housing_d:.2f} | {medqa_housing_gap:.3f} |",
        f"| MedQA > BarExamQA | {medqa_vs_barexam_auc:.3f} | {medqa_minus_barexam_d:.2f} | {medqa_barexam_gap:.3f} |",
        "",
        "Reading:",
        "",
        f"- MedQA is {'materially higher' if go_answer_budget else 'not materially higher'} than HousingQA on the corpus-surprise pre-screen by the configured gate: mean log-PPL gap {medqa_housing_gap:.3f}, AUC {medqa_vs_housing_auc:.3f}.",
        f"- MedQA median token count is {summaries['medqa']['median_tokens']:.0f}, compared with HousingQA {summaries['housing']['median_tokens']:.0f} and BarExamQA {summaries['barexam']['median_tokens']:.0f}.",
        f"- Gate decision: {'continue to q200 downstream answer probe' if go_answer_budget else 'stop before answer budget; the prescreen did not show the intended weak-query gap'}.",
        "",
        "Reproduction:",
        "",
        "```bash",
        "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \\",
        "  uv run python scripts/analyze_medqa_prescreen.py",
        "```",
    ]
    section = "\n".join(section_lines)
    update_report(args.report, section)
    print(section)
    if not go_answer_budget:
        raise SystemExit(10)


if __name__ == "__main__":
    main()
