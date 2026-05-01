#!/usr/bin/env python3
"""Build figures for the final class report from checked detail logs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "final_class_report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def read_detail(relpath: str) -> list[dict]:
    path = ROOT / relpath
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"empty detail log: {relpath}")
    return rows


def summarize(relpath: str) -> dict:
    rows = read_detail(relpath)
    n = len(rows)
    correct = sum(1 for row in rows if row.get("is_correct"))
    calls = sum(float(row.get("llm_calls") or 0) for row in rows) / n
    input_tokens = sum(int(row.get("input_tokens") or 0) for row in rows)
    output_tokens = sum(int(row.get("output_tokens") or 0) for row in rows)
    empty = sum(1 for row in rows if len(row.get("retrieved_ids") or []) == 0)
    gold_field = [row for row in rows if "gold_retrieved" in row]
    gold_hits = sum(1 for row in gold_field if row.get("gold_retrieved"))
    return {
        "log": relpath,
        "n": n,
        "correct": correct,
        "accuracy": correct / n,
        "calls_per_q": calls,
        "tokens_per_q": (input_tokens + output_tokens) / n,
        "input_tokens_per_q": input_tokens / n,
        "output_tokens_per_q": output_tokens / n,
        "empty_retrieval": empty,
        "gold_retrieval": gold_hits / n if gold_field else None,
    }


RUNS = {
    # Paired BarExam N=200 efficiency rows.
    "barexam_top1": summarize("logs/eval_rag_simple_or-gemma4-26b_20260428_0138_detail.jsonl"),
    "barexam_top5": summarize("logs/eval_rag_simple_or-gemma4-26b_20260428_0231_detail.jsonl"),
    "barexam_snap": summarize("logs/eval_rag_snap_hyde_or-gemma4-26b_20260428_0257_detail.jsonl"),
    "barexam_2call": summarize("logs/eval_rag_snap_hyde_2call_or-gemma4-26b_20260428_1435_detail.jsonl"),
    # Housing depth and two-call rows.
    "housing_top1": summarize("logs/eval_rag_simple_or-gemma4-26b_20260430_0415_detail.jsonl"),
    "housing_top5": summarize("logs/eval_rag_simple_or-gemma4-26b_20260430_0502_detail.jsonl"),
    "housing_top10": summarize("logs/eval_rag_simple_or-gemma4-26b_20260430_0542_detail.jsonl"),
    "housing_2call": summarize("logs/eval_rag_snap_hyde_2call_or-gemma4-26b_20260430_0644_detail.jsonl"),
    "housing_state5": summarize("logs/eval_rag_state_filter_or-gemma4-26b_20260501_1406_detail.jsonl"),
    "housing_state10": summarize("logs/eval_rag_state_filter_or-gemma4-26b_20260501_k10_merged_detail.jsonl"),
    # SCALR depth and HyRE rows.
    "scalr_top1": summarize("logs/eval_rag_simple_groq-llama70b_20260429_2159_detail.jsonl"),
    "scalr_top5": summarize("logs/eval_rag_simple_groq-llama70b_20260428_1508_detail.jsonl"),
    "scalr_top10": summarize("logs/eval_rag_simple_groq-llama70b_20260430_0054_detail.jsonl"),
    "scalr_2call": summarize("logs/eval_rag_snap_hyde_2call_groq-llama70b_20260428_1520_detail.jsonl"),
    # Repaired CaseHOLD rows.
    "casehold_top1": summarize("logs/eval_rag_simple_groq-llama70b_20260501_1432_detail.jsonl"),
    "casehold_top5": summarize("logs/eval_rag_simple_groq-llama70b_20260430_1738_detail.jsonl"),
    "casehold_top10": summarize("logs/eval_rag_simple_groq-llama70b_20260501_1440_detail.jsonl"),
    "casehold_hyde": summarize("logs/eval_rag_hyde_groq-llama70b_20260501_1449_detail.jsonl"),
    "casehold_2call": summarize("logs/eval_rag_snap_hyde_2call_groq-llama70b_20260430_1751_detail.jsonl"),
}


def pct(x: float) -> float:
    return 100.0 * x


def write_metrics() -> None:
    with (OUT / "figure_metrics.csv").open("w", newline="") as f:
        fieldnames = [
            "name",
            "log",
            "n",
            "correct",
            "accuracy",
            "calls_per_q",
            "tokens_per_q",
            "input_tokens_per_q",
            "output_tokens_per_q",
            "empty_retrieval",
            "gold_retrieval",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for name, row in RUNS.items():
            writer.writerow({"name": name, **row})


def plot_barexam_cross_size() -> None:
    fig, ax = plt.subplots(figsize=(7.3, 2.35), dpi=260)
    rows = [
        ("Gemma 4 26B-A4B", "N=1195 full corpus", 78.08, 81.17, "+3.09 pp"),
        ("Gemma 4 E4B", "N=1195 full corpus", 58.49, 62.18, "+3.68 pp"),
    ]
    y_positions = [1, 0]
    for y, (model, scope, simple, snap, delta) in zip(y_positions, rows):
        ax.plot([simple, snap], [y, y], color="#AEB7B7", linewidth=2.3, solid_capstyle="round", zorder=1)
        ax.scatter(simple, y, s=115, color="#788384", edgecolor="white", linewidth=1.1, zorder=3)
        ax.scatter(snap, y, s=135, color="#2EAD63", edgecolor="white", linewidth=1.1, zorder=4)
        ax.text(simple, y - 0.15, f"{simple:.2f}%", ha="center", va="top", fontsize=8.2, color="#3F4646")
        ax.text(snap, y - 0.15, f"{snap:.2f}%", ha="center", va="top", fontsize=8.2, color="#145A32", weight="bold")
        ax.text(
            (simple + snap) / 2,
            y + 0.22,
            f"HyRE {delta}",
            ha="center",
            va="bottom",
            fontsize=8.8,
            color="#106B35",
            weight="bold",
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"{model}\n{scope}" for model, scope, *_ in rows], fontsize=8.4)
    ax.set_xlabel("Full-corpus accuracy (%)", fontsize=9)
    ax.set_xlim(55.5, 84.5)
    ax.set_ylim(-0.50, 1.45)
    ax.set_xticks([60, 65, 70, 75, 80])
    ax.grid(axis="x", alpha=0.22, linewidth=0.7)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=8)
    ax.scatter([], [], s=90, color="#788384", label="simple RAG")
    ax.scatter([], [], s=90, color="#2EAD63", label="HyRE")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.45, 1.09), fontsize=8.0, ncol=2, handletextpad=0.4)
    fig.subplots_adjust(left=0.24, right=0.98, top=0.80, bottom=0.26)
    fig.savefig(OUT / "02_barexam_cross_size.png", bbox_inches="tight")


def plot_depth_and_conversion() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1), dpi=220)
    ax = axes[0]
    groups = [
        ("BarExam", ["top-1", "top-5"], ["barexam_top1", "barexam_top5"]),
        ("Housing", ["top-1", "top-5", "top-10", "state-5", "state-10"], ["housing_top1", "housing_top5", "housing_top10", "housing_state5", "housing_state10"]),
        ("SCALR", ["top-1", "top-5", "top-10"], ["scalr_top1", "scalr_top5", "scalr_top10"]),
        ("CaseHOLD", ["top-1", "top-5", "top-10"], ["casehold_top1", "casehold_top5", "casehold_top10"]),
    ]
    colors = {"top-1": "#4C78A8", "top-5": "#59A14F", "top-10": "#F28E2B", "state-5": "#B07AA1", "state-10": "#9C755F"}
    x = 0.0
    xticks = []
    xticklabels = []
    for dataset, labels, keys in groups:
        start = x
        for label, key in zip(labels, keys):
            val = pct(RUNS[key]["accuracy"])
            ax.bar(x, val, width=0.55, color=colors[label], label=label)
            ax.text(x, val + 1.0, f"{val:.1f}", ha="center", va="bottom", fontsize=7)
            x += 0.62
        xticks.append((start + x - 0.62) / 2)
        xticklabels.append(dataset)
        x += 0.55
    ax.set_title("Top-k sensitivity differs by legal task", fontsize=10)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(45, 90)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), frameon=False, fontsize=8, ncol=3, loc="upper left")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)

    ax = axes[1]
    labels = ["top-1", "top-5", "top-10", "HyDE", "2-call"]
    keys = ["casehold_top1", "casehold_top5", "casehold_top10", "casehold_hyde", "casehold_2call"]
    answer = [pct(RUNS[key]["accuracy"]) for key in keys]
    gold = [pct(RUNS[key]["gold_retrieval"] or 0.0) for key in keys]
    xs = range(len(keys))
    ax.plot(xs, answer, marker="o", linewidth=2, color="#4C78A8", label="answer accuracy")
    ax.plot(xs, gold, marker="o", linewidth=2, color="#E15759", label="gold retrieved")
    for x_i, a, g in zip(xs, answer, gold):
        ax.text(x_i, a + 2.0, f"{a:.1f}", ha="center", fontsize=7, color="#1f3f63")
        ax.text(x_i, g + 2.0, f"{g:.1f}", ha="center", fontsize=7, color="#8a2f31")
    ax.set_title("CaseHOLD: retrieval recall is not enough", fontsize=10)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 82)
    ax.set_ylabel("Percent")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)

    fig.tight_layout()
    fig.savefig(OUT / "01_depth_and_conversion.png", bbox_inches="tight")


def plot_cost_accuracy() -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.6), dpi=220)
    points = [
        ("BarExam", "simple", "barexam_top5"),
        ("BarExam", "HyRE", "barexam_snap"),
        ("BarExam", "2-call HyRE", "barexam_2call"),
        ("Housing", "top-10", "housing_top10"),
        ("Housing", "state-10", "housing_state10"),
        ("Housing", "2-call HyRE", "housing_2call"),
        ("SCALR", "simple", "scalr_top5"),
        ("SCALR", "2-call HyRE", "scalr_2call"),
        ("CaseHOLD", "top-5", "casehold_top5"),
        ("CaseHOLD", "HyDE", "casehold_hyde"),
        ("CaseHOLD", "2-call HyRE", "casehold_2call"),
    ]
    offsets = {
        "scalr_2call": (8, 10),
        "housing_top10": (8, -20),
        "housing_2call": (8, 10),
        "housing_state10": (8, 4),
        "barexam_snap": (8, 4),
        "casehold_hyde": (8, -12),
        "casehold_2call": (8, -34),
    }
    annotation_labels = {
        "casehold_hyde": "CaseHOLD HyDE/2-call",
        "casehold_2call": "",
    }
    colors = {
        "BarExam": "#4C78A8",
        "Housing": "#F28E2B",
        "SCALR": "#59A14F",
        "CaseHOLD": "#E15759",
    }
    line_groups = {
        "BarExam": ["barexam_top5", "barexam_2call", "barexam_snap"],
        "Housing": ["housing_top10", "housing_state10", "housing_2call"],
        "SCALR": ["scalr_top5", "scalr_2call"],
        "CaseHOLD": ["casehold_top5", "casehold_hyde", "casehold_2call"],
    }
    for dataset, keys in line_groups.items():
        xs = [RUNS[key]["calls_per_q"] for key in keys]
        ys = [pct(RUNS[key]["accuracy"]) for key in keys]
        ax.plot(xs, ys, color=colors[dataset], alpha=0.22, linewidth=1.3, zorder=1)

    for dataset, label, key in points:
        row = RUNS[key]
        size = max(60, row["tokens_per_q"] / 16.0)
        ax.scatter(
            row["calls_per_q"],
            pct(row["accuracy"]),
            s=size,
            color=colors[dataset],
            alpha=0.78,
            edgecolor="white",
            linewidth=0.8,
            label=dataset,
            zorder=3,
        )
        annotation = annotation_labels.get(key, f"{dataset} {label}")
        if annotation:
            ax.annotate(
                annotation,
                (row["calls_per_q"], pct(row["accuracy"])),
                textcoords="offset points",
                xytext=offsets.get(key, (6, 4)),
                fontsize=6.8,
            )
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    dataset_legend = ax.legend(unique.values(), unique.keys(), frameon=False, fontsize=8, loc="lower right")
    ax.add_artist(dataset_legend)
    size_handles = [
        ax.scatter([], [], s=max(60, tokens / 16.0), color="#9AA0A6", alpha=0.38, edgecolor="white", linewidth=0.8)
        for tokens in (1000, 3000, 5000)
    ]
    ax.legend(
        size_handles,
        ["1k", "3k", "5k"],
        title="tokens/q",
        frameon=False,
        fontsize=7.2,
        title_fontsize=7.5,
        loc="upper left",
        borderpad=0.2,
        labelspacing=0.7,
        handletextpad=1.4,
    )
    ax.set_xlabel("LLM calls per question")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs LLM budget on paired legal subsets", fontsize=10)
    ax.set_xlim(0.75, 3.35)
    ax.set_ylim(48, 90)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["1 call", "2 calls", "3 calls"])
    ax.grid(alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(OUT / "02_cost_accuracy.png", bbox_inches="tight")


def plot_adaptive_hyre_routes() -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.6), dpi=260)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.93, "Adaptive HyRE routes by failure mode", ha="center", va="center", fontsize=16, weight="bold")
    ax.text(
        0.5,
        0.86,
        "Snap reasoning is baked into the hypothetical snippet, then routed to the active bottleneck.",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#444444",
    )

    cards = [
        (0.07, 0.55, "#2D6CDF", "Answer framing", "Preserve the legal issue and rule\nwhen evidence is noisy."),
        (0.57, 0.55, "#21884A", "Candidate set", "Retrieve enough plausible options\nbefore reranking."),
        (0.07, 0.18, "#C98200", "Metadata filter", "Constrain by jurisdiction when it\nis part of the task."),
        (0.57, 0.18, "#C5332F", "Option mapping", "Convert retrieved holdings back\nto answer choices."),
    ]

    def card(x: float, y: float, color: str, title: str, body: str) -> None:
        ax.add_patch(
            patches.FancyBboxPatch(
                (x, y),
                0.36,
                0.23,
                boxstyle="round,pad=0.018,rounding_size=0.018",
                facecolor="white",
                edgecolor=color,
                linewidth=1.4,
            )
        )
        ax.add_patch(
            patches.FancyBboxPatch(
                (x + 0.02, y + 0.05),
                0.055,
                0.13,
                boxstyle="round,pad=0.01,rounding_size=0.02",
                facecolor=color,
                edgecolor=color,
                linewidth=0,
                alpha=0.16,
            )
        )
        ax.text(x + 0.095, y + 0.155, title, ha="left", va="center", fontsize=11.2, weight="bold", color=color)
        ax.text(x + 0.095, y + 0.085, body, ha="left", va="center", fontsize=8.6, color="#222222")

    for item in cards:
        card(*item)

    ax.add_patch(
        patches.FancyBboxPatch(
            (0.39, 0.425),
            0.22,
            0.14,
            boxstyle="round,pad=0.018,rounding_size=0.02",
            facecolor="#F7F8F9",
            edgecolor="#778080",
            linewidth=1.1,
        )
    )
    ax.text(0.5, 0.505, "HyRE router", ha="center", va="center", fontsize=11, weight="bold", color="#1F2933")
    ax.text(0.5, 0.465, "snap -> reasoning snippet", ha="center", va="center", fontsize=8.4, color="#4A5560")

    arrow_targets = [(0.39, 0.62), (0.57, 0.62), (0.39, 0.29), (0.57, 0.29)]
    arrow_starts = [(0.45, 0.565), (0.55, 0.565), (0.45, 0.425), (0.55, 0.425)]
    for start, end in zip(arrow_starts, arrow_targets):
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.1, "color": "#6B7280"})

    fig.tight_layout()
    fig.savefig(OUT / "03_adaptive_snap_hyde_controller.png", bbox_inches="tight")


def plot_route_map() -> None:
    """Summarize the current evidence as routes for an adaptive controller."""

    rows = ["BarExamQA", "SCALR", "HousingQA", "CaseHOLD"]
    cols = ["Answer\nframe", "Candidate\nexpansion", "Metadata\nfilter", "Option\nconverter"]
    # 0 = not active, 1 = plausible intervention, 2 = observed support.
    values = [
        [2, 0, 0, 0],
        [0, 2, 0, 1],
        [0, 1, 2, 0],
        [0, 1, 0, 1],
    ]
    labels = [
        ["observed:\nHyRE +3.09pp", "", "", ""],
        ["", "observed:\ntop-5 +17.5pp", "", "next:\noption rerank"],
        ["", "possible:\nk10 +7.5pp", "observed:\nstate 62.5%", ""],
        ["", "weak signal:\nk1 -> k5 +5.0pp", "", "next:\noption mapper"],
    ]
    cmap = ListedColormap(["#F7F7F7", "#F4D35E", "#5AA469"])
    fig, ax = plt.subplots(figsize=(9.8, 4.5), dpi=220)
    ax.imshow(values, cmap=cmap, vmin=0, vmax=2)

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=9)
    ax.set_title("Evidence-backed routes for adaptive HyRE", fontsize=11, pad=12)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for y, row in enumerate(labels):
        for x, text in enumerate(row):
            ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, edgecolor="#D0D0D0", linewidth=0.8))
            if text:
                color = "#0F2F1A" if values[y][x] == 2 else "#4D3B00"
                ax.text(x, y, text, ha="center", va="center", fontsize=7.2, color=color)

    handles = [
        patches.Patch(facecolor="#5AA469", edgecolor="#A0A0A0", label="observed signal"),
        patches.Patch(facecolor="#F4D35E", edgecolor="#A0A0A0", label="plausible next route"),
        patches.Patch(facecolor="#F7F7F7", edgecolor="#A0A0A0", label="not active"),
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=3,
        frameon=False,
        fontsize=8,
        columnspacing=1.8,
        handlelength=1.2,
        handletextpad=0.4,
    )

    fig.tight_layout()
    fig.savefig(OUT / "04_bottleneck_route_map.png", bbox_inches="tight")


def main() -> None:
    write_metrics()
    plot_barexam_cross_size()
    plot_depth_and_conversion()
    plot_cost_accuracy()
    plot_adaptive_hyre_routes()
    plot_route_map()


if __name__ == "__main__":
    main()
