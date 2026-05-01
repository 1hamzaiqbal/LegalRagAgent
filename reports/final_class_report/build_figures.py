#!/usr/bin/env python3
"""Build figures for the final class report from checked detail logs."""

from __future__ import annotations

import csv
import json
import textwrap
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
    # SCALR depth and snap-HyDE rows.
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
        ("BarExam", "snap", "barexam_snap"),
        ("BarExam", "2-call", "barexam_2call"),
        ("Housing", "top-10", "housing_top10"),
        ("Housing", "state-10", "housing_state10"),
        ("Housing", "2-call", "housing_2call"),
        ("SCALR", "simple", "scalr_top5"),
        ("SCALR", "2-call", "scalr_2call"),
        ("CaseHOLD", "top-5", "casehold_top5"),
        ("CaseHOLD", "HyDE", "casehold_hyde"),
        ("CaseHOLD", "2-call", "casehold_2call"),
    ]
    offsets = {
        "scalr_2call": (8, 10),
        "housing_top10": (8, -20),
        "housing_2call": (8, 10),
        "housing_state10": (8, 4),
        "barexam_snap": (8, 4),
        "casehold_hyde": (8, -12),
        "casehold_2call": (8, -38),
    }
    colors = {
        "BarExam": "#4C78A8",
        "Housing": "#F28E2B",
        "SCALR": "#59A14F",
        "CaseHOLD": "#E15759",
    }
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
        )
        ax.annotate(
            f"{dataset} {label}\n{row['tokens_per_q'] / 1000:.1f}k tok/q",
            (row["calls_per_q"], pct(row["accuracy"])),
            textcoords="offset points",
            xytext=offsets.get(key, (6, 4)),
            fontsize=7,
        )
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), frameon=False, fontsize=8, loc="lower right")
    ax.set_xlabel("LLM calls per question")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy/cost tradeoff on paired N=200 legal slices", fontsize=10)
    ax.set_xlim(0.75, 3.25)
    ax.set_ylim(48, 90)
    ax.grid(alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(OUT / "02_cost_accuracy.png", bbox_inches="tight")


def wrapped(text: str, width: int = 28) -> str:
    return "\n".join(textwrap.wrap(text, width=width))


def add_box(ax, xy: tuple[float, float], width: float, height: float, text: str, face: str, edge: str, fontsize: float = 8.0) -> None:
    box = patches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.025,rounding_size=0.025",
        linewidth=1.1,
        facecolor=face,
        edgecolor=edge,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#1B1B1B",
    )


def add_arrow(ax, start: tuple[float, float], end: tuple[float, float], color: str = "#555555") -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={"arrowstyle": "->", "lw": 1.25, "color": color, "shrinkA": 2, "shrinkB": 2},
    )


def plot_adaptive_controller() -> None:
    """Draw a grounded controller schematic tied to the observed result rows."""

    fig, ax = plt.subplots(figsize=(9.8, 5.3), dpi=220)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_box(
        ax,
        (0.04, 0.80),
        0.22,
        0.11,
        "Legal question + snap",
        "#EEF4FB",
        "#4C78A8",
        9.0,
    )
    add_box(
        ax,
        (0.38, 0.80),
        0.22,
        0.11,
        "Cheap diagnostics",
        "#F8F8F8",
        "#777777",
        9.0,
    )
    add_box(
        ax,
        (0.74, 0.80),
        0.22,
        0.11,
        "Route-specific HyDE use",
        "#F8F8F8",
        "#777777",
        9.0,
    )
    add_arrow(ax, (0.26, 0.855), (0.38, 0.855))
    add_arrow(ax, (0.60, 0.855), (0.74, 0.855))

    headers = [("Observed signal", 0.05, 0.30), ("Route", 0.37, 0.20), ("Concrete next fix", 0.62, 0.31)]
    for label, x, width in headers:
        ax.text(x + width / 2, 0.72, label, ha="center", va="center", fontsize=9.2, weight="bold", color="#333333")

    rows = [
        {
            "y": 0.60,
            "color": "#4C78A8",
            "route": "Answer-frame\nHyDE",
            "signal": "BarExam: full-corpus snap-HyDE +3.09pp; top-k flat",
            "fix": "Use snap to frame evidence, not just retrieve more text",
        },
        {
            "y": 0.43,
            "color": "#59A14F",
            "route": "Candidate\nexpansion",
            "signal": "SCALR: top-1 -> top-5 +17.5pp; top-10 flat",
            "fix": "Default to k=5, then rerank or ground options",
        },
        {
            "y": 0.26,
            "color": "#F28E2B",
            "route": "Metadata\nfiltering",
            "signal": "Housing: state-filter k10 62.5%; 0 empty rows",
            "fix": "Normalize jurisdiction metadata before retrieval",
        },
        {
            "y": 0.09,
            "color": "#E15759",
            "route": "Option\nconversion",
            "signal": "CaseHOLD: gold-hit 16% -> 47%; answer lift not reliable",
            "fix": "Map retrieved holdings back to answer choices",
        },
    ]

    for row in rows:
        y = row["y"]
        add_box(ax, (0.05, y), 0.30, 0.115, wrapped(row["signal"], 31), "#FBFBFB", row["color"], 8.0)
        add_box(ax, (0.39, y), 0.16, 0.115, row["route"], "#FFFFFF", row["color"], 8.4)
        add_box(ax, (0.61, y), 0.34, 0.115, wrapped(row["fix"], 38), "#FBFBFB", row["color"], 8.0)
        add_arrow(ax, (0.35, y + 0.057), (0.39, y + 0.057), row["color"])
        add_arrow(ax, (0.55, y + 0.057), (0.61, y + 0.057), row["color"])

    fig.tight_layout()
    fig.savefig(OUT / "03_adaptive_snap_hyde_controller.png", bbox_inches="tight")


def plot_route_map() -> None:
    """Summarize the current evidence as routes for an adaptive controller."""

    rows = ["BarExamQA", "SCALR", "HousingQA", "CaseHOLD"]
    cols = ["Answer\nframe", "Candidate\nexpansion", "Metadata\nfilter", "Option\nconverter"]
    # 0 = not active from current evidence, 1 = promising/next, 2 = supported by landed rows.
    values = [
        [2, 0, 0, 1],
        [0, 2, 0, 2],
        [0, 1, 2, 0],
        [0, 1, 0, 2],
    ]
    labels = [
        ["full-corpus\n+3.09pp", "", "", "keep frame;\naudit evidence"],
        ["", "top-1 -> top-5\n+17.5pp", "", "next:\nrerank options"],
        ["", "generic top-10\n+7.5pp", "state k10\n62.5%", ""],
        ["", "k1 -> k5\n+5.0pp", "", "gold-hit\n+31pp"],
    ]
    cmap = ListedColormap(["#F7F7F7", "#F4D35E", "#5AA469"])
    fig, ax = plt.subplots(figsize=(9.8, 4.5), dpi=220)
    ax.imshow(values, cmap=cmap, vmin=0, vmax=2)

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=9)
    ax.set_title("Evidence-backed routes for adaptive snap-HyDE", fontsize=11, pad=12)
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
        patches.Patch(facecolor="#5AA469", edgecolor="#A0A0A0", label="supported"),
        patches.Patch(facecolor="#F4D35E", edgecolor="#A0A0A0", label="promising / next"),
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
    plot_depth_and_conversion()
    plot_cost_accuracy()
    plot_adaptive_controller()
    plot_route_map()


if __name__ == "__main__":
    main()
