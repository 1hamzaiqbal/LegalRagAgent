#!/usr/bin/env python3
"""Build paper-local figures for the fixed-method Snap-HyRE draft.

The script reads the generated result package as an evidence source and writes
only into this paper directory. A few retrieval values are pinned here because
the current package does not expose every full-corpus qrel row in one CSV; the
source paths are recorded in ``figure_metrics.csv``.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
PAPER = Path(__file__).resolve().parent
FIGURES = PAPER / "figures"
TABLES = PAPER / "tables"
ANSWER_CSV = ROOT / "docs/generated/snap_hyre_package/answer_ladder_status.csv"
RETRIEVAL_CSV = ROOT / "docs/generated/snap_hyre_package/retrieval_topk_status.csv"
Q100_TOPK_CSV = ROOT / "docs/generated/retrieval_cache_matrix_or-gemma4-26b_q100_k1_to_k10.csv"

PROVIDERS = ["or-ministral-8b", "or-gemma4-26b", "groq-llama70b"]
DATASETS = ["barexam", "housing", "casehold", "legalbench_scalr"]
MODES = [
    "llm_only",
    "rag_simple",
    "rag_hyde",
    "snap_hyre",
    "rag_rewrite",
    "golden_passage",
    "golden_plus_neighbors",
]
GENERATED_MODES = ["rag_hyde", "snap_hyre", "rag_rewrite"]

DATASET_N = {
    "barexam": 1195,
    "housing": 6853,
    "casehold": 3600,
    "legalbench_scalr": 571,
}

DATASET_LABEL = {
    "barexam": "BarExamQA",
    "casehold": "CaseHOLD",
    "housing": "HousingQA",
    "legalbench_scalr": "SCALR",
}

MODEL_LABEL = {
    "or-ministral-8b": "Ministral 8B",
    "or-gemma4-26b": "Gemma 4 26B",
    "groq-llama70b": "Llama 3.3 70B",
}

METHOD_LABEL = {
    "llm_only": "LLM-only",
    "rag_simple": "Raw RAG",
    "rag_hyde": "HyDE",
    "snap_hyre": "Snap-HyRE",
    "rag_rewrite": "Rewrite",
    "golden_passage": "Gold",
    "golden_plus_neighbors": "Gold+Nbrs",
}

METHOD_COLORS = {
    "llm_only": "#4c566a",
    "rag_simple": "#5e81ac",
    "rag_hyde": "#88c0d0",
    "snap_hyre": "#a3be8c",
    "rag_rewrite": "#d08770",
    "golden_passage": "#b48ead",
    "golden_plus_neighbors": "#ebcb8b",
}


def pct(value: float | None) -> str:
    if value is None:
        return r"\textsc{tbd}"
    return f"{value:.1f}"


def fmt_half_up(value: float, digits: int = 1, signed: bool = False) -> str:
    factor = 10**digits
    magnitude = int(abs(value) * factor + 0.5) / factor
    sign = "-" if value < 0 else "+" if signed else ""
    return f"{sign}{magnitude:.{digits}f}"


def tex_escape(value: str) -> str:
    return value.replace("_", r"\_")


def clean_answer_rows(answers: dict[tuple[str, str, str], dict[str, str]]) -> dict[tuple[str, str, str], float]:
    out: dict[tuple[str, str, str], float] = {}
    for key, row in answers.items():
        if row.get("detail_status") == "clean":
            out[key] = 100.0 * float(row["accuracy"])
    return out


def read_answer_rows() -> dict[tuple[str, str, str], dict[str, str]]:
    with ANSWER_CSV.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return {(r["provider"], r["dataset"], r["mode"]): r for r in rows}


def read_retrieval_rows() -> list[dict[str, str]]:
    with RETRIEVAL_CSV.open(newline="") as f:
        return list(csv.DictReader(f))


def answer_pct(answers: dict[tuple[str, str, str], dict[str, str]], provider: str, dataset: str, mode: str) -> float | None:
    row = answers.get((provider, dataset, mode))
    if not row:
        return None
    return 100.0 * float(row["accuracy"])


def figure_answer_ladder(answers: dict[tuple[str, str, str], dict[str, str]]) -> list[dict[str, str]]:
    cells = [
        ("or-gemma4-26b", "barexam"),
        ("groq-llama70b", "barexam"),
        ("or-ministral-8b", "legalbench_scalr"),
        ("or-gemma4-26b", "legalbench_scalr"),
        ("groq-llama70b", "legalbench_scalr"),
        ("groq-llama70b", "casehold"),
        ("groq-llama70b", "housing"),
    ]
    methods = [
        "llm_only",
        "rag_simple",
        "rag_hyde",
        "snap_hyre",
        "rag_rewrite",
        "golden_passage",
        "golden_plus_neighbors",
    ]
    fig, axes = plt.subplots(2, 4, figsize=(13.6, 6.6), sharey=False)
    axes = axes.ravel()
    metrics: list[dict[str, str]] = []
    for ax, (provider, dataset) in zip(axes, cells):
        vals = []
        labels = []
        colors = []
        for mode in methods:
            val = answer_pct(answers, provider, dataset, mode)
            if val is None:
                continue
            vals.append(val)
            labels.append(METHOD_LABEL[mode])
            colors.append(METHOD_COLORS[mode])
            metrics.append(
                {
                    "figure": "01_answer_ladder_by_dataset.png",
                    "provider": provider,
                    "dataset": dataset,
                    "method": mode,
                    "metric": "accuracy_pct",
                    "value": f"{val:.2f}",
                    "source": "docs/generated/snap_hyre_package/answer_ladder_status.csv",
                }
            )
        ax.bar(range(len(vals)), vals, color=colors, width=0.72)
        ax.set_title(f"{DATASET_LABEL[dataset]}\n{MODEL_LABEL[provider]}", fontsize=10)
        ax.set_xticks(range(len(vals)), labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylim(max(0, min(vals) - 8), min(100, max(vals) + 6))
        ax.grid(axis="y", alpha=0.22)
        for i, val in enumerate(vals):
            ax.text(i, val + 0.5, f"{val:.1f}", ha="center", va="bottom", fontsize=7)
    axes[-1].axis("off")
    fig.suptitle("Full-corpus answer accuracy at k=5", fontsize=14, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = FIGURES / "01_answer_ladder_by_dataset.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return metrics


# Retrieval and paired-answer facts used in the conversion plot. Values are
# signed in docs/signoff_log.md or generated qrel reports named in source.
CONVERSION_ROWS = [
    {
        "dataset": "barexam",
        "provider": "or-gemma4-26b",
        "method": "rag_hyde",
        "raw_hit": 1.42,
        "method_hit": 11.38,
        "raw_acc": 77.99,
        "method_acc": 80.25,
        "source": "docs/signoff_log.md; docs/generated/retrieval_qrels_barexam_or-gemma4-26b_rag_simple.md",
    },
    {
        "dataset": "barexam",
        "provider": "or-gemma4-26b",
        "method": "snap_hyre",
        "raw_hit": 1.42,
        "method_hit": 12.05,
        "raw_acc": 77.99,
        "method_acc": 82.01,
        "source": "docs/signoff_log.md; docs/generated/retrieval_qrels_barexam_or-gemma4-26b_rag_simple.md",
    },
    {
        "dataset": "barexam",
        "provider": "or-gemma4-26b",
        "method": "rag_rewrite",
        "raw_hit": 1.42,
        "method_hit": 12.22,
        "raw_acc": 77.99,
        "method_acc": 80.67,
        "source": "docs/signoff_log.md; docs/generated/retrieval_qrels_barexam_or-gemma4-26b_rag_rewrite.md",
    },
    {
        "dataset": "barexam",
        "provider": "groq-llama70b",
        "method": "rag_hyde",
        "raw_hit": 1.42,
        "method_hit": 10.46,
        "raw_acc": 74.56,
        "method_acc": 80.17,
        "source": "docs/signoff_log.md; docs/generated/retrieval_qrels_barexam_groq-llama70b_rag_simple.md",
    },
    {
        "dataset": "barexam",
        "provider": "groq-llama70b",
        "method": "snap_hyre",
        "raw_hit": 1.42,
        "method_hit": 11.05,
        "raw_acc": 74.56,
        "method_acc": 79.75,
        "source": "docs/signoff_log.md; docs/generated/retrieval_qrels_barexam_groq-llama70b_rag_simple.md",
    },
    {
        "dataset": "barexam",
        "provider": "groq-llama70b",
        "method": "rag_rewrite",
        "raw_hit": 1.42,
        "method_hit": 12.22,
        "raw_acc": 74.56,
        "method_acc": 77.24,
        "source": "docs/signoff_log.md; docs/generated/retrieval_qrels_barexam_groq-llama70b_rag_rewrite.md",
    },
    {
        "dataset": "legalbench_scalr",
        "provider": "or-ministral-8b",
        "method": "rag_hyde",
        "raw_hit": 49.56,
        "method_hit": 60.25,
        "raw_acc": 67.95,
        "method_acc": 71.10,
        "source": "docs/signoff_log.md; docs/generated/retrieval_qrels_scalr_or-ministral-8b_rag_simple.md",
    },
    {
        "dataset": "legalbench_scalr",
        "provider": "or-ministral-8b",
        "method": "snap_hyre",
        "raw_hit": 49.56,
        "method_hit": 62.00,
        "raw_acc": 67.95,
        "method_acc": 69.88,
        "source": "docs/signoff_log.md; docs/generated/retrieval_qrels_scalr_or-ministral-8b_rag_simple.md",
    },
    {
        "dataset": "legalbench_scalr",
        "provider": "or-ministral-8b",
        "method": "rag_rewrite",
        "raw_hit": 49.56,
        "method_hit": 64.97,
        "raw_acc": 67.95,
        "method_acc": 69.88,
        "source": "docs/signoff_log.md; docs/generated/retrieval_qrels_scalr_or-ministral-8b_rag_rewrite.md",
    },
    {
        "dataset": "legalbench_scalr",
        "provider": "or-gemma4-26b",
        "method": "rag_hyde",
        "raw_hit": 49.56,
        "method_hit": 70.75,
        "raw_acc": 73.38,
        "method_acc": 72.15,
        "source": "docs/signoff_log.md",
    },
    {
        "dataset": "legalbench_scalr",
        "provider": "or-gemma4-26b",
        "method": "snap_hyre",
        "raw_hit": 49.56,
        "method_hit": 72.68,
        "raw_acc": 73.38,
        "method_acc": 73.91,
        "source": "docs/signoff_log.md",
    },
    {
        "dataset": "legalbench_scalr",
        "provider": "or-gemma4-26b",
        "method": "rag_rewrite",
        "raw_hit": 49.56,
        "method_hit": 67.43,
        "raw_acc": 73.38,
        "method_acc": 73.91,
        "source": "docs/signoff_log.md; docs/generated/retrieval_qrels_scalr_or-gemma4-26b_rag_rewrite.md",
    },
    {
        "dataset": "legalbench_scalr",
        "provider": "groq-llama70b",
        "method": "rag_hyde",
        "raw_hit": 49.56,
        "method_hit": 61.47,
        "raw_acc": 72.85,
        "method_acc": 70.40,
        "source": "docs/signoff_log.md; docs/generated/retrieval_qrels_scalr_groq-llama70b_rag_simple.md",
    },
    {
        "dataset": "legalbench_scalr",
        "provider": "groq-llama70b",
        "method": "snap_hyre",
        "raw_hit": 49.56,
        "method_hit": 55.17,
        "raw_acc": 72.85,
        "method_acc": 71.28,
        "source": "docs/signoff_log.md; docs/generated/retrieval_qrels_scalr_groq-llama70b_rag_simple.md",
    },
    {
        "dataset": "legalbench_scalr",
        "provider": "groq-llama70b",
        "method": "rag_rewrite",
        "raw_hit": 49.56,
        "method_hit": 57.62,
        "raw_acc": 72.85,
        "method_acc": 71.63,
        "source": "docs/signoff_log.md; docs/generated/retrieval_qrels_scalr_groq-llama70b_rag_rewrite.md",
    },
    {
        "dataset": "casehold",
        "provider": "groq-llama70b",
        "method": "rag_hyde",
        "raw_hit": 17.94,
        "method_hit": 51.22,
        "raw_acc": 70.75,
        "method_acc": 70.33,
        "source": "docs/signoff_log.md; docs/generated/retrieval_qrels_casehold_groq-llama70b_rag_simple.md",
    },
    {
        "dataset": "casehold",
        "provider": "groq-llama70b",
        "method": "snap_hyre",
        "raw_hit": 17.94,
        "method_hit": 44.97,
        "raw_acc": 70.75,
        "method_acc": 70.50,
        "source": "docs/signoff_log.md; docs/generated/retrieval_qrels_casehold_groq-llama70b_snap_hyre_mixed.md",
    },
    {
        "dataset": "casehold",
        "provider": "groq-llama70b",
        "method": "rag_rewrite",
        "raw_hit": 17.94,
        "method_hit": 45.08,
        "raw_acc": 70.75,
        "method_acc": 70.61,
        "source": "docs/signoff_log.md; docs/generated/retrieval_qrels_casehold_groq-llama70b_rag_rewrite.md",
    },
]


def figure_retrieval_conversion() -> list[dict[str, str]]:
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    markers = {"rag_hyde": "o", "snap_hyre": "s", "rag_rewrite": "^"}
    colors = {
        "barexam": "#5e81ac",
        "legalbench_scalr": "#a3be8c",
        "casehold": "#d08770",
    }
    metrics: list[dict[str, str]] = []
    for row in CONVERSION_ROWS:
        dx = row["method_hit"] - row["raw_hit"]
        dy = row["method_acc"] - row["raw_acc"]
        ax.scatter(
            dx,
            dy,
            s=90,
            marker=markers[row["method"]],
            color=colors[row["dataset"]],
            edgecolor="#2e3440",
            linewidth=0.7,
            alpha=0.92,
        )
        short_model = MODEL_LABEL[row["provider"]].replace(" 3.3", "").replace(" 26B", "")
        ax.text(dx + 0.35, dy + 0.08, f"{DATASET_LABEL[row['dataset']]} {short_model}", fontsize=7)
        for metric, value in {
            "delta_hit5_pp": dx,
            "delta_accuracy_pp": dy,
        }.items():
            metrics.append(
                {
                    "figure": "02_retrieval_answer_conversion.png",
                    "provider": row["provider"],
                    "dataset": row["dataset"],
                    "method": row["method"],
                    "metric": metric,
                    "value": f"{value:.2f}",
                    "source": row["source"],
                }
            )
    ax.axhline(0, color="#2e3440", lw=1.0, alpha=0.65)
    ax.axvline(0, color="#2e3440", lw=1.0, alpha=0.65)
    ax.set_xlabel("Hit@5 change vs raw RAG (percentage points)")
    ax.set_ylabel("Accuracy change vs raw RAG (percentage points)")
    ax.set_title("Retrieval exposure can rise without an answer-accuracy lift")
    ax.grid(alpha=0.24)
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label="HyDE", markerfacecolor="#888", markeredgecolor="#2e3440", markersize=8),
        plt.Line2D([0], [0], marker="s", color="w", label="Snap-HyRE", markerfacecolor="#888", markeredgecolor="#2e3440", markersize=8),
        plt.Line2D([0], [0], marker="^", color="w", label="Rewrite", markerfacecolor="#888", markeredgecolor="#2e3440", markersize=8),
        plt.Line2D([0], [0], marker="o", color="w", label="BarExamQA", markerfacecolor=colors["barexam"], markersize=8),
        plt.Line2D([0], [0], marker="o", color="w", label="SCALR", markerfacecolor=colors["legalbench_scalr"], markersize=8),
        plt.Line2D([0], [0], marker="o", color="w", label="CaseHOLD", markerfacecolor=colors["casehold"], markersize=8),
    ]
    ax.legend(handles=handles, fontsize=8, ncol=2, frameon=True)
    fig.tight_layout()
    out = FIGURES / "02_retrieval_answer_conversion.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return metrics


def figure_accuracy_vs_hit5() -> list[dict[str, str]]:
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    colors = {
        "barexam": "#5e81ac",
        "legalbench_scalr": "#a3be8c",
        "casehold": "#d08770",
    }
    markers = {"rag_simple": "D", "rag_hyde": "o", "snap_hyre": "s", "rag_rewrite": "^"}
    rows: list[dict[str, object]] = []
    for row in CONVERSION_ROWS:
        rows.append(
            {
                "dataset": row["dataset"],
                "provider": row["provider"],
                "method": row["method"],
                "hit": row["method_hit"],
                "acc": row["method_acc"],
                "source": row["source"],
            }
        )
    seen_raw = set()
    for row in CONVERSION_ROWS:
        key = (row["dataset"], row["provider"])
        if key in seen_raw:
            continue
        seen_raw.add(key)
        rows.append(
            {
                "dataset": row["dataset"],
                "provider": row["provider"],
                "method": "rag_simple",
                "hit": row["raw_hit"],
                "acc": row["raw_acc"],
                "source": row["source"],
            }
        )

    metrics: list[dict[str, str]] = []
    for row in rows:
        dataset = str(row["dataset"])
        provider = str(row["provider"])
        method = str(row["method"])
        hit = float(row["hit"])
        acc = float(row["acc"])
        ax.scatter(
            hit,
            acc,
            s=86,
            marker=markers[method],
            color=colors[dataset],
            edgecolor="#2e3440",
            linewidth=0.65,
            alpha=0.9,
        )
        if method in {"rag_simple", "snap_hyre"}:
            label = f"{DATASET_LABEL[dataset]} {MODEL_LABEL[provider].replace(' 3.3', '').replace(' 26B', '')} {METHOD_LABEL[method]}"
            if hit > 60:
                ax.text(hit - 0.55, acc + 0.1, label, fontsize=6.5, ha="right")
            else:
                ax.text(hit + 0.4, acc + 0.1, label, fontsize=6.5)
        metrics.extend(
            [
                {
                    "figure": "02_accuracy_vs_hit5.png",
                    "provider": provider,
                    "dataset": dataset,
                    "method": method,
                    "metric": "hit5_pct",
                    "value": f"{hit:.2f}",
                    "source": str(row["source"]),
                },
                {
                    "figure": "02_accuracy_vs_hit5.png",
                    "provider": provider,
                    "dataset": dataset,
                    "method": method,
                    "metric": "accuracy_pct",
                    "value": f"{acc:.2f}",
                    "source": str(row["source"]),
                },
            ]
        )
    ax.set_xlabel("Exact-qrel Hit@5 (%)")
    ax.set_ylabel("Final answer accuracy (%)")
    ax.set_title("Final accuracy versus exact gold retrieval exposure")
    ax.set_xlim(-2.2, 77.0)
    ax.set_ylim(67.2, 82.7)
    ax.grid(alpha=0.24)
    handles = [
        plt.Line2D([0], [0], marker="D", color="w", label="Raw RAG", markerfacecolor="#888", markeredgecolor="#2e3440", markersize=8),
        plt.Line2D([0], [0], marker="o", color="w", label="HyDE", markerfacecolor="#888", markeredgecolor="#2e3440", markersize=8),
        plt.Line2D([0], [0], marker="s", color="w", label="Snap-HyRE", markerfacecolor="#888", markeredgecolor="#2e3440", markersize=8),
        plt.Line2D([0], [0], marker="^", color="w", label="Rewrite", markerfacecolor="#888", markeredgecolor="#2e3440", markersize=8),
        plt.Line2D([0], [0], marker="o", color="w", label="BarExamQA", markerfacecolor=colors["barexam"], markersize=8),
        plt.Line2D([0], [0], marker="o", color="w", label="SCALR", markerfacecolor=colors["legalbench_scalr"], markersize=8),
        plt.Line2D([0], [0], marker="o", color="w", label="CaseHOLD", markerfacecolor=colors["casehold"], markersize=8),
    ]
    ax.legend(handles=handles, fontsize=8, ncol=2, frameon=True)
    fig.tight_layout()
    out = FIGURES / "02_accuracy_vs_hit5.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return metrics


def figure_oracle_gap(answers: dict[tuple[str, str, str], dict[str, str]]) -> list[dict[str, str]]:
    cells = [
        ("or-gemma4-26b", "barexam"),
        ("groq-llama70b", "barexam"),
        ("groq-llama70b", "legalbench_scalr"),
        ("groq-llama70b", "casehold"),
        ("groq-llama70b", "housing"),
    ]
    methods = ["rag_simple", "snap_hyre", "golden_passage", "golden_plus_neighbors"]
    labels = [f"{DATASET_LABEL[d]}\n{MODEL_LABEL[p]}" for p, d in cells]
    x = list(range(len(cells)))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10.4, 5.4))
    metrics: list[dict[str, str]] = []
    for j, method in enumerate(methods):
        vals = [answer_pct(answers, p, d, method) for p, d in cells]
        offsets = [i + (j - 1.5) * width for i in x]
        for idx, (off, val) in enumerate(zip(offsets, vals)):
            ax.bar(
                off,
                val if val is not None else 0,
                width=width,
                color=METHOD_COLORS[method],
                label=METHOD_LABEL[method] if idx == 0 else None,
                alpha=1.0 if val is not None else 0.18,
            )
        for off, val, (provider, dataset) in zip(offsets, vals, cells):
            if val is None:
                ax.text(off, 3, "n/a", ha="center", va="bottom", rotation=90, fontsize=7, color="#4c566a")
                continue
            ax.text(off, val + 0.8, f"{val:.1f}", ha="center", va="bottom", rotation=90, fontsize=7)
            metrics.append(
                {
                    "figure": "03_oracle_gap.png",
                    "provider": provider,
                    "dataset": dataset,
                    "method": method,
                    "metric": "accuracy_pct",
                    "value": f"{val:.2f}",
                    "source": "docs/generated/snap_hyre_package/answer_ladder_status.csv",
                }
            )
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Gold evidence exposes large remaining context-use headroom")
    ax.set_ylim(0, 104)
    ax.grid(axis="y", alpha=0.24)
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    out = FIGURES / "03_oracle_gap.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return metrics


def retrieval_points(
    rows: list[dict[str, str]],
    dataset: str,
    model: str,
    method: str,
    metric: str,
    min_rows: int = 500,
) -> list[tuple[int, float]]:
    out: list[tuple[int, float]] = []
    for row in rows:
        if row["dataset"] != dataset or row["model"] != model or row["method"] != method:
            continue
        if int(row["rows"]) < min_rows:
            continue
        out.append((int(row["k"]), 100.0 * float(row[metric])))
    return sorted(out)


def figure_retrieval_curves(retrieval_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Legacy Hit@k figure retained for continuity with earlier drafts."""
    panels = [
        ("legalbench_scalr", "llama70b", "SCALR / Llama 70B"),
        ("casehold", "llama70b", "CaseHOLD / Llama 70B"),
        ("barexam", "llama70b", "BarExamQA / Llama 70B"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2), sharey=False)
    metrics: list[dict[str, str]] = []
    for ax, (dataset, model, title) in zip(axes, panels):
        for method in ["rag_hyde", "snap_hyre"]:
            pts = retrieval_points(retrieval_rows, dataset, model, method, "hit")
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker="o", lw=2, color=METHOD_COLORS[method], label=METHOD_LABEL[method])
            for k, val in pts:
                metrics.append(
                    {
                        "figure": "04_retrieval_hit_curves.png",
                        "provider": "groq-llama70b",
                        "dataset": dataset,
                        "method": method,
                        "metric": f"hit_at_{k}_pct",
                        "value": f"{val:.2f}",
                        "source": "docs/generated/snap_hyre_package/retrieval_topk_status.csv",
                    }
                )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("retrieval k")
        ax.set_ylabel("Hit@k (%)")
        ax.set_xticks([1, 3, 5, 10])
        ax.grid(alpha=0.24)
    axes[0].legend(fontsize=8)
    fig.suptitle("Generated-query retrieval curves on full Llama 70B caches", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = FIGURES / "04_retrieval_hit_curves.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return metrics


def figure_retrieval_hit_mrr_curves_full(retrieval_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    panels = [
        ("barexam", "llama70b", "BarExamQA / Llama 70B"),
        ("casehold", "llama70b", "CaseHOLD / Llama 70B"),
        ("legalbench_scalr", "llama70b", "SCALR / Llama 70B"),
    ]
    methods = ["rag_simple", "rag_hyde", "snap_hyre"]
    method_model = {"rag_simple": "model_invariant", "rag_hyde": None, "snap_hyre": None}
    line_styles = {"rag_simple": "--", "rag_hyde": "-", "snap_hyre": "-"}
    markers = {"rag_simple": "D", "rag_hyde": "o", "snap_hyre": "s"}

    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.0), sharex=True)
    metrics: list[dict[str, str]] = []
    for col, (dataset, model, title) in enumerate(panels):
        for row_idx, metric in enumerate(["hit", "mrr"]):
            ax = axes[row_idx][col]
            for method in methods:
                lookup_model = method_model[method] or model
                pts = retrieval_points(retrieval_rows, dataset, lookup_model, method, metric)
                if not pts:
                    continue
                xs, ys = zip(*pts)
                ax.plot(
                    xs,
                    ys,
                    marker=markers[method],
                    lw=2,
                    linestyle=line_styles[method],
                    color=METHOD_COLORS[method],
                    label=METHOD_LABEL[method],
                )
                for k, val in pts:
                    metrics.append(
                        {
                            "figure": "04_retrieval_hit_mrr_curves_full.png",
                            "provider": "groq-llama70b" if method != "rag_simple" else "model_invariant",
                            "dataset": dataset,
                            "method": method,
                            "metric": f"{metric}_at_{k}_pct",
                            "value": f"{val:.2f}",
                            "source": "docs/generated/snap_hyre_package/retrieval_topk_status.csv",
                        }
                    )
            if row_idx == 0:
                ax.set_title(title, fontsize=10)
            ax.set_xticks([1, 3, 5, 10])
            ax.set_ylabel("Hit@k (%)" if metric == "hit" else "MRR@k (%)")
            ax.grid(alpha=0.24)
            if not ax.lines:
                ax.text(0.5, 0.5, "full rows unavailable", ha="center", va="center", transform=ax.transAxes)
        axes[1][col].set_xlabel("retrieval k")
    axes[0][0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Full-row retrieval curves: Hit@k and MRR@k", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = FIGURES / "04_retrieval_hit_mrr_curves_full.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return metrics


def read_q100_rows() -> list[dict[str, str]]:
    with Q100_TOPK_CSV.open(newline="") as f:
        return list(csv.DictReader(f))


def q100_macro_points(rows: list[dict[str, str]], metric: str) -> list[tuple[int, float]]:
    return [
        (int(r["k"]), 100.0 * float(r[metric]))
        for r in rows
        if r["scope"] == "macro" and r["method"] == "all_supplied_caches"
    ]


def figure_q100_macro_topk_probe(q100_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    metrics: list[dict[str, str]] = []
    for metric, label, color, marker in [
        ("hit", "Macro Hit@k", "#5e81ac", "o"),
        ("mrr", "Macro MRR@k", "#d08770", "s"),
    ]:
        pts = q100_macro_points(q100_rows, metric)
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker=marker, lw=2.2, color=color, label=label)
        for k, val in pts:
            metrics.append(
                {
                    "figure": "05_q100_macro_topk_probe.png",
                    "provider": "or-gemma4-26b",
                    "dataset": "macro_q100",
                    "method": "all_supplied_caches",
                    "metric": f"{metric}_at_{k}_pct",
                    "value": f"{val:.2f}",
                    "source": "docs/generated/retrieval_cache_matrix_or-gemma4-26b_q100_k1_to_k10.csv",
                }
            )
    ax.axvline(5, color="#2e3440", lw=1.0, alpha=0.6, linestyle="--")
    ax.text(5.1, 20.5, "main answer k=5", fontsize=8, color="#2e3440")
    ax.set_xlabel("retrieval k")
    ax.set_ylabel("Macro retrieval metric (%)")
    ax.set_xticks(range(1, 11))
    ax.set_title("q100 prelaunch probe: recall rises after k=5, MRR barely moves")
    ax.grid(alpha=0.24)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = FIGURES / "05_q100_macro_topk_probe.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return metrics


def write_text(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n")


def table_cell(value: float | None) -> str:
    return pct(value)


def write_answer_matrix_table(answers: dict[tuple[str, str, str], dict[str, str]]) -> None:
    lines = [
        r"\begin{table*}[!htbp]",
        r"\centering",
        r"\caption{Current signed full-corpus answer accuracy at $k=5$ (\%). \textsc{tbd} cells are absent from the current fixed-method package rather than inferred from older runs.}",
        r"\label{tab:answer_matrix}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.8pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"Dataset / model & $N$ & LLM & Raw RAG & HyDE & Snap-HyRE & Rewrite & Gold & Gold+Nbrs \\",
        r"\midrule",
    ]
    clean = clean_answer_rows(answers)
    for dataset in DATASETS:
        for provider in PROVIDERS:
            values = [clean.get((provider, dataset, mode)) for mode in MODES]
            rendered = [table_cell(v) for v in values]
            lines.append(
                f"{DATASET_LABEL[dataset]} / {MODEL_LABEL[provider]} & {DATASET_N[dataset]} & "
                + " & ".join(rendered)
                + r" \\"
            )
        if dataset != DATASETS[-1]:
            lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule", r"\end{tabular}}", r"\end{table*}"])
    write_text(TABLES / "answer_matrix.tex", lines)


def write_completion_matrix_table(answers: dict[tuple[str, str, str], dict[str, str]]) -> None:
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Coverage of the expected 84-cell answer ladder. Generated-method cells are HyDE, Snap-HyRE, and rewrite.}",
        r"\label{tab:coverage_matrix}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{llcl}",
        r"\toprule",
        r"Dataset & Model & Signed cells & Missing generated cells \\",
        r"\midrule",
    ]
    for dataset in DATASETS:
        for provider in PROVIDERS:
            present = [mode for mode in MODES if (provider, dataset, mode) in answers]
            missing_generated = [mode for mode in GENERATED_MODES if (provider, dataset, mode) not in answers]
            if not missing_generated:
                missing = r"\textsc{none}"
            else:
                missing = ", ".join(METHOD_LABEL[m] for m in missing_generated)
            lines.append(
                f"{DATASET_LABEL[dataset]} & {MODEL_LABEL[provider]} & {len(present)}/7 & {missing} " + r"\\"
            )
        if dataset != DATASETS[-1]:
            lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule", r"\end{tabular}}", r"\end{table}"])
    write_text(TABLES / "coverage_matrix.tex", lines)


def write_completed_cell_means_table(answers: dict[tuple[str, str, str], dict[str, str]]) -> None:
    clean = clean_answer_rows(answers)
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Unbalanced averages across completed clean answer cells. These means summarize the current package only; they are not benchmark-balanced leaderboard scores.}",
        r"\label{tab:completed_means}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Mode & Cells & Mean acc. & Mean $\Delta$ vs raw \\",
        r"\midrule",
    ]
    for mode in MODES:
        vals = [value for (provider, dataset, m), value in clean.items() if m == mode]
        mean = sum(vals) / len(vals) if vals else None
        deltas: list[float] = []
        for (provider, dataset, m), value in clean.items():
            if m != mode:
                continue
            raw = clean.get((provider, dataset, "rag_simple"))
            if raw is not None:
                deltas.append(value - raw)
        delta = sum(deltas) / len(deltas) if deltas else None
        lines.append(
            f"{METHOD_LABEL[mode]} & {len(vals)} & {table_cell(mean)} & {table_cell(delta)} " + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    write_text(TABLES / "completed_cell_means.tex", lines)


def write_method_conversion_table(answers: dict[tuple[str, str, str], dict[str, str]]) -> None:
    by_cell: dict[tuple[str, str], dict[str, dict[str, object]]] = defaultdict(dict)
    for row in CONVERSION_ROWS:
        by_cell[(row["dataset"], row["provider"])][row["method"]] = row
    clean = clean_answer_rows(answers)
    cell_order = [
        ("barexam", "or-gemma4-26b"),
        ("barexam", "groq-llama70b"),
        ("legalbench_scalr", "or-ministral-8b"),
        ("legalbench_scalr", "or-gemma4-26b"),
        ("legalbench_scalr", "groq-llama70b"),
        ("casehold", "groq-llama70b"),
    ]
    lines = [
        r"\begin{table*}[!htbp]",
        r"\centering",
        r"\caption{Retrieval exposure and answer conversion at $k=5$ for RAG-simple, HyDE, Snap-HyRE, and RAG-rewrite rows with signed full answer and retrieval results. Deltas are method answer accuracy minus the RAG-simple answer accuracy in the same dataset/model cell.}",
        r"\label{tab:method_conversion}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrrrrr}",
        r"\toprule",
        r"Dataset & Model & \multicolumn{2}{c}{\method{rag_simple}} & \multicolumn{2}{c}{\method{rag_hyde}} & \multicolumn{2}{c}{\method{snap_hyre}} & \multicolumn{2}{c}{\method{rag_rewrite}} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}",
        r" & & Hit@5 & Acc. & Hit@5 & $\Delta$ Acc. & Hit@5 & $\Delta$ Acc. & Hit@5 & $\Delta$ Acc. \\",
        r"\midrule",
    ]
    for dataset, provider in cell_order:
        methods = by_cell[(dataset, provider)]
        first = next(iter(methods.values()))
        raw_hit = float(first["raw_hit"])
        raw_acc = clean.get((provider, dataset, "rag_simple"), float(first["raw_acc"]))
        row_cells = [DATASET_LABEL[dataset], MODEL_LABEL[provider], fmt_half_up(raw_hit), fmt_half_up(raw_acc)]
        for method in ["rag_hyde", "snap_hyre", "rag_rewrite"]:
            method_row = methods.get(method)
            if method_row is None:
                row_cells.extend([r"\textsc{tbd}", r"\textsc{tbd}"])
                continue
            hit = float(method_row["method_hit"])
            method_acc = clean.get((provider, dataset, method), float(method_row["method_acc"]))
            delta = method_acc - raw_acc
            row_cells.extend([fmt_half_up(hit), fmt_half_up(delta, digits=2, signed=True)])
        lines.append(" & ".join(row_cells) + r" \\")
    lines.append(
        r"HousingQA & Llama 3.3 70B & \textsc{tbd} & 47.3 & \textsc{tbd} & \textsc{tbd} & \textsc{tbd} & \textsc{tbd} & \textsc{tbd} & \textsc{tbd} \\"
    )
    lines.extend([r"\bottomrule", r"\end{tabular}}", r"\end{table*}"])
    write_text(TABLES / "method_conversion.tex", lines)


def write_current_snap_hyde_decomposition_table(answers: dict[tuple[str, str, str], dict[str, str]]) -> None:
    clean = clean_answer_rows(answers)
    cell_order = [
        ("barexam", "or-gemma4-26b"),
        ("barexam", "groq-llama70b"),
        ("legalbench_scalr", "or-ministral-8b"),
        ("legalbench_scalr", "or-gemma4-26b"),
        ("legalbench_scalr", "groq-llama70b"),
        ("casehold", "groq-llama70b"),
    ]
    lines = [
        r"\begin{table*}[!htbp]",
        r"\centering",
        r"\caption{Current-package Snap-HyRE versus HyDE answer comparison. The current fixed-method package does not include signed \method{snap_only_in_final} rows, so snap-only is left as \textsc{tbd}. ``Parity'' means within 0.25 percentage points of HyDE.}",
        r"\label{tab:current_snap_hyde_decomposition}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabularx}{\textwidth}{llrrrrY}",
        r"\toprule",
        r"Dataset & Model & \method{snap_only_in_final} & \method{rag_hyde} & \method{snap_hyre} & Snap-HyRE $-$ HyDE & Reading \\",
        r"\midrule",
    ]
    for dataset, provider in cell_order:
        hyde = clean.get((provider, dataset, "rag_hyde"))
        snap_hyre = clean.get((provider, dataset, "snap_hyre"))
        if hyde is None or snap_hyre is None:
            continue
        delta = snap_hyre - hyde
        if delta > 0.25:
            reading = "beats HyDE"
        elif delta < -0.25:
            reading = "below HyDE"
        else:
            reading = "near parity"
        lines.append(
            " & ".join(
                [
                    DATASET_LABEL[dataset],
                    MODEL_LABEL[provider],
                    r"\textsc{tbd}",
                    fmt_half_up(hyde),
                    fmt_half_up(snap_hyre),
                    fmt_half_up(delta, digits=2, signed=True),
                    reading,
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table*}"])
    write_text(TABLES / "current_snap_hyde_decomposition.tex", lines)


def write_legacy_snap_only_decomposition_table() -> None:
    rows = [
        {
            "model": "Gemma 4 26B-A4B",
            "snap_only": 80.59,
            "hyde": 78.91,
            "combo": 81.17,
            "source": r"\method{docs/compiled_results.md}",
        },
        {
            "model": "Gemma 4 E4B",
            "snap_only": 57.82,
            "hyde": 60.59,
            "combo": 62.18,
            "source": r"\method{docs/compiled_results.md}",
        },
    ]
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Legacy BarExamQA snap-only decomposition from the older audited \method{rag_snap_hyde} ladder. This is a mechanism probe, not a current-package \method{snap_hyre} result.}",
        r"\label{tab:legacy_snap_only_decomposition}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lrrrrl}",
        r"\toprule",
        r"Model & \method{snap_only_in_final} & \method{rag_hyde} & \method{rag_snap_hyde} & Combo beats & Source \\",
        r"\midrule",
    ]
    for row in rows:
        combo = float(row["combo"])
        snap_only = float(row["snap_only"])
        hyde = float(row["hyde"])
        beats = f"snap {fmt_half_up(combo - snap_only, digits=2, signed=True)}, HyDE {fmt_half_up(combo - hyde, digits=2, signed=True)}"
        lines.append(
            " & ".join(
                [
                    str(row["model"]),
                    fmt_half_up(snap_only, digits=2),
                    fmt_half_up(hyde, digits=2),
                    fmt_half_up(combo, digits=2),
                    beats,
                    str(row["source"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}}", r"\end{table}"])
    write_text(TABLES / "legacy_snap_only_decomposition.tex", lines)


def write_significance_table() -> None:
    rows = [
        ("BarExamQA", "Gemma 4 26B", "+4.02", "0.000699", "+1.17", "0.348", "+1.76", "0.0987"),
        ("BarExamQA", "Llama 70B", "+5.19", "2.70e-05", "+1.09", "0.388", "-0.42", "0.754"),
        ("SCALR", "Ministral 8B", "+1.93", "0.260", "+2.63", "0.110", "-1.23", "0.457"),
        ("SCALR", "Gemma 4 26B", "+0.53", "0.780", "+0.88", "0.560", "+1.75", "0.220"),
        ("SCALR", "Llama 70B", "-1.58", "0.281", "-3.15", "0.0222", "+0.88", "0.542"),
        ("CaseHOLD", "Llama 70B", "-0.25", "0.722", "-1.31", "0.0295", "+0.17", "0.812"),
    ]
    lines = [
        r"\begin{table*}[!htbp]",
        r"\centering",
        r"\caption{Paired McNemar comparisons for Snap-HyRE against selected controls. Deltas are percentage points in answer accuracy.}",
        r"\label{tab:snap_significance}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Dataset & Model & $\Delta$ raw & $p$ raw & $\Delta$ LLM & $p$ LLM & $\Delta$ HyDE & $p$ HyDE \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}}", r"\end{table*}"])
    write_text(TABLES / "snap_hyre_significance.tex", lines)


def write_caveat_ledger_table() -> None:
    rows = [
        ("BarExamQA", "Gemma 4 26B", "Snap-HyRE", "OpenRouter upstream retries recovered in place; 3 answer-format retries; 4 near-cap rows."),
        ("BarExamQA", "Llama 70B", "Snap-HyRE", "Clean row with one final-answer repair."),
        ("SCALR", "Ministral 8B", "Snap-HyRE", "Retry caveat; 9 final-answer repairs; no fallback keys."),
        ("SCALR", "Gemma 4 26B", "Snap-HyRE", "10 answer-format retries; 5 near-cap pre-repair rows."),
        ("SCALR", "Llama 70B", "Snap-HyRE", "Clean signed row."),
        ("CaseHOLD", "Llama 70B", "Snap-HyRE", "Mixed same-model provider recovery and repaired cache row; 16 answer-format retries."),
        ("HousingQA", "All", "Generated methods", "Full generated-method answer rows are TBD in the current package."),
    ]
    lines = [
        r"\begin{table*}[!htbp]",
        r"\centering",
        r"\caption{Caveats that should travel with exact row claims. Detailed provenance remains in the signoff log.}",
        r"\label{tab:caveat_ledger}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabularx}{\textwidth}{lllY}",
        r"\toprule",
        r"Dataset & Model & Row & Caveat \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table*}"])
    write_text(TABLES / "caveat_ledger.tex", lines)


def write_q100_probe_table(q100_rows: list[dict[str, str]]) -> None:
    macro = {
        int(r["k"]): (100.0 * float(r["hit"]), float(r["mrr"]))
        for r in q100_rows
        if r["scope"] == "macro" and r["method"] == "all_supplied_caches"
    }
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{q100 prelaunch top-$k$ probe. Retrieval exposure rises after $k=5$, but MRR changes little and BarExamQA downstream accuracy did not favor $k=10$.}",
        r"\label{tab:q100_topk_probe}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{rrrr}",
        r"\toprule",
        r"$k$ & Macro Hit@k & Macro MRR@k & BarExam q100 acc. \\",
        r"\midrule",
    ]
    downstream = {5: "raw 83, HyDE 87", 10: "raw 81, HyDE 84"}
    for k in [1, 5, 10]:
        hit, mrr = macro[k]
        downstream_cell = downstream.get(k, r"\textsc{n/a}")
        lines.append(f"{k} & {hit:.1f} & {mrr:.3f} & {downstream_cell} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    write_text(TABLES / "q100_topk_probe.tex", lines)


def write_all_tables(answers: dict[tuple[str, str, str], dict[str, str]], q100_rows: list[dict[str, str]]) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    write_answer_matrix_table(answers)
    write_completion_matrix_table(answers)
    write_completed_cell_means_table(answers)
    write_method_conversion_table(answers)
    write_current_snap_hyde_decomposition_table(answers)
    write_legacy_snap_only_decomposition_table()
    write_significance_table()
    write_caveat_ledger_table()
    write_q100_probe_table(q100_rows)


def write_metrics(rows: list[dict[str, str]]) -> None:
    fields = ["figure", "provider", "dataset", "method", "metric", "value", "source"]
    with (FIGURES / "figure_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    answers = read_answer_rows()
    retrieval_rows = read_retrieval_rows()
    q100_rows = read_q100_rows()
    metrics: list[dict[str, str]] = []
    metrics.extend(figure_answer_ladder(answers))
    metrics.extend(figure_retrieval_conversion())
    metrics.extend(figure_accuracy_vs_hit5())
    metrics.extend(figure_oracle_gap(answers))
    metrics.extend(figure_retrieval_curves(retrieval_rows))
    metrics.extend(figure_retrieval_hit_mrr_curves_full(retrieval_rows))
    metrics.extend(figure_q100_macro_topk_probe(q100_rows))
    write_all_tables(answers, q100_rows)
    write_metrics(metrics)
    print(f"wrote {len(metrics)} figure metrics to {FIGURES / 'figure_metrics.csv'}")
    print(f"wrote paper tables to {TABLES}")


if __name__ == "__main__":
    main()
