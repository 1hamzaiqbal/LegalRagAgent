"""Focused 5-tier leaderboard for presentation slides.

Plots exactly the headline tiers (LLM-only / Vanilla RAG / Vanilla HyDE /
SNAP-HyDE / Golden Passage) at N=1195 on Gemma 4 E4B (cluster-vllm).

Two variants:
  - OLD: rag_hyde = 54.3% (broken Gemma HyDE prompt; pre-fix)
  - NEW: rag_hyde = 57.9% (fixed HyDE prompt)

Run: uv run python docs/charts/generate_focused_tiers.py
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

MPLCONFIGDIR = Path(tempfile.gettempdir()) / "legalragagent_mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

OUT_DIR = Path(__file__).resolve().parent


TIERS = [
    ("LLM only",             "llm_only",            0.521),
    ("Vanilla RAG",          "rag_simple",          0.542),
    ("Vanilla HyDE",         "rag_hyde",            None),   # filled per variant
    ("SNAP-HyDE",            "rag_snap_hyde",       0.579),
    ("LLM + Golden Passage", "golden_passage",      0.622),
]

HYDE_OLD = 0.543
HYDE_NEW = 0.579


def plot_variant(variant: str, hyde_acc: float) -> None:
    tiers = [(label, mode, hyde_acc if mode == "rag_hyde" else acc)
             for label, mode, acc in TIERS]

    labels = [t[0] for t in tiers]
    values = [t[2] for t in tiers]

    colors = []
    for _, mode, _ in tiers:
        if mode == "golden_passage":
            colors.append("#D4A017")
        elif mode == "llm_only":
            colors.append("#7F7F7F")
        elif mode == "rag_simple":
            colors.append("#4E79A7")
        elif mode == "rag_hyde":
            colors.append("#E15759" if variant == "old" else "#59A14F")
        elif mode == "rag_snap_hyde":
            colors.append("#76B7B2")
        else:
            colors.append("#4682B4")

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        np.arange(len(labels)),
        values,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.bar_label(bars, labels=[f"{v:.1%}" for v in values], padding=4, fontsize=11)

    # title_tag = "OLD (broken HyDE prompt)" if variant == "old" else "NEW (fixed HyDE prompt)"
    ax.set_title(f"Core Retrieval Tiers\nGemma 4 E4B, N=1195, BarExam")
    ax.set_xlabel("Method")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=10, ha="right")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_ylim(0, max(values) + 0.1)

    fig.tight_layout()
    out = OUT_DIR / f"focused_tiers_{variant}.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved {out}")


GOLD_TIERS = [
    ("Vanilla RAG",  "rag_simple",    0.011),
    ("Vanilla HyDE", "rag_hyde_new",  0.077),
    ("SNAP-HyDE",    "rag_snap_hyde", 0.062),
]


def plot_gold_retrieval() -> None:
    labels = [t[0] for t in GOLD_TIERS]
    values = [t[2] for t in GOLD_TIERS]

    colors = []
    for _, mode, _ in GOLD_TIERS:
        if mode == "rag_simple":
            colors.append("#4E79A7")
        elif mode == "rag_hyde_old":
            colors.append("#E15759")
        elif mode == "rag_hyde_new":
            colors.append("#59A14F")
        elif mode == "rag_snap_hyde":
            colors.append("#76B7B2")
        else:
            colors.append("#7F7F7F")

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        np.arange(len(labels)),
        values,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.bar_label(bars, labels=[f"{v:.1%}" for v in values], padding=4, fontsize=11)

    ax.set_title("Gold-Passage Retrieval Rate\nGemma 4 E4B, N=1195, BarExam")
    ax.set_xlabel("Method")
    ax.set_ylabel("Gold retrieved")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_ylim(0, max(values) + 0.025)

    fig.tight_layout()
    out = OUT_DIR / "gold_retrieval_rate.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })

    plot_variant("old", HYDE_OLD)
    plot_variant("new", HYDE_NEW)
    plot_gold_retrieval()


if __name__ == "__main__":
    main()
