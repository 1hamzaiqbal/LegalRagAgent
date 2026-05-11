#!/usr/bin/env python3
"""Build meeting-ready diagnostic-adaptation figures from source-gated summaries."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as patches


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "presentation" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

PORTFOLIO_JSON = ROOT / "docs" / "diagnostic_controller_portfolio_comparison_2026-05-10.json"
HELDOUT_CONTROLLER_JSON = ROOT / "docs" / "heldout_controller_eval_2026-05-10.json"
HELDOUT_REWRITE_JSON = ROOT / "docs" / "heldout_query_rewrite_2026-05-10.json"
SNAP_ONLY_JSON = ROOT / "docs" / "snap_only_controls_2026-05-11.json"

DATASETS = ["barexam", "housing", "casehold", "legalbench_scalr"]
DATASET_LABELS = {
    "barexam": "BarExam",
    "housing": "HousingQA",
    "casehold": "CaseHOLD",
    "legalbench_scalr": "SCALR",
}
COLORS = {
    "ink": "#1f2933",
    "muted": "#52616b",
    "line": "#c9d1d9",
    "paper": "#ffffff",
    "soft": "#f5f7fa",
    "green": "#1f9d55",
    "blue": "#2f80ed",
    "orange": "#f2994a",
    "red": "#d64545",
    "purple": "#7c3aed",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}"


def portfolio_rows() -> dict[str, dict[str, Any]]:
    data = load_json(PORTFOLIO_JSON)
    return {row["portfolio"]: row for row in data["portfolios"]}


def row_by_dataset(portfolio: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["dataset"]: row for row in portfolio["rows"]}


def make_table_figure(
    filename: str,
    title: str,
    subtitle: str,
    rows: list[tuple[str, list[str], str]],
    source_note: str,
) -> None:
    fig, ax = plt.subplots(figsize=(12.4, 5.1), dpi=220)
    ax.axis("off")
    ax.text(0.0, 1.03, title, transform=ax.transAxes, fontsize=16, weight="bold", color=COLORS["ink"])
    ax.text(0.0, 0.965, subtitle, transform=ax.transAxes, fontsize=9.6, color=COLORS["muted"])

    columns = ["Model & Method", "BarExam", "HousingQA", "CaseHOLD", "SCALR", "Avg.", "Calls", "Gate"]
    cell_text = [[label, *values, gate] for label, values, gate in rows]
    table = ax.table(
        cellText=cell_text,
        colLabels=columns,
        loc="upper left",
        cellLoc="center",
        colLoc="center",
        bbox=[0.0, 0.06, 1.0, 0.83],
        colWidths=[0.31, 0.10, 0.11, 0.11, 0.09, 0.08, 0.08, 0.12],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.4)
    table.scale(1.0, 1.28)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(COLORS["line"])
        cell.set_linewidth(0.7)
        if r == 0:
            cell.set_facecolor("#e9eef3")
            cell.get_text().set_weight("bold")
            cell.get_text().set_color(COLORS["ink"])
        elif r == len(rows):
            cell.set_facecolor("#eaf7ef")
            cell.get_text().set_weight("bold")
        elif r % 2 == 0:
            cell.set_facecolor(COLORS["soft"])
        else:
            cell.set_facecolor(COLORS["paper"])
        if c == 0:
            cell.get_text().set_ha("left")

    ax.text(
        0.0,
        0.0,
        source_note,
        transform=ax.transAxes,
        fontsize=7.6,
        color=COLORS["muted"],
    )
    fig.savefig(OUT / filename, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def calibration_ablation() -> None:
    portfolios = portfolio_rows()
    snap = load_json(SNAP_ONLY_JSON)
    snap_rows = {row["dataset"]: row for row in snap["rows"]}
    base = row_by_dataset(portfolios["baseline_retrieval"])
    rewrite = row_by_dataset(portfolios["query_rewrite_available"])
    hyre = row_by_dataset(portfolios["fixed_hyre_only"])
    controller = row_by_dataset(portfolios["diagnostic_controller"])

    def vals(rows: dict[str, dict[str, Any]], portfolio: str) -> list[str]:
        return [
            pct(rows[dataset]["accuracy"]) for dataset in DATASETS
        ] + [
            pct(portfolios[portfolio]["macro_accuracy"]),
            f"{portfolios[portfolio]['macro_avg_calls']:.2f}",
        ]

    table_rows = [
        ("Gemma 4 26B + baseline retrieval", vals(base, "baseline_retrieval"), "N=200"),
        (
            "+ snap-only reasoning",
            [pct(snap_rows[dataset]["accuracy"]) for dataset in DATASETS]
            + [pct(snap["macro_accuracy"]), f"{snap['macro_avg_calls']:.2f}"],
            "N=200",
        ),
        ("+ legal query rewrite control", vals(rewrite, "query_rewrite_available"), "mixed N"),
        ("+ fixed HyRE family", vals(hyre, "fixed_hyre_only"), "N=200"),
        ("+ diagnostic controller routes", vals(controller, "diagnostic_controller"), "N=200 / replay"),
    ]
    make_table_figure(
        "12_diagnostic_adaptation_calibration_ablation.png",
        "Inherited Ablation: Calibration Portfolio",
        "Controller routes improve macro accuracy while using fewer calls than fixed HyRE.",
        table_rows,
        "Sources: docs/diagnostic_controller_portfolio_comparison_2026-05-10.json and "
        "docs/snap_only_controls_2026-05-11.json.",
    )


def heldout_ablation() -> None:
    controller = load_json(HELDOUT_CONTROLLER_JSON)
    rewrite = load_json(HELDOUT_REWRITE_JSON)
    by_dataset = {row["dataset"]: row for row in controller["records"]}
    rewrite_by_dataset = {row["dataset"]: row for row in rewrite["rows"]}

    baseline_values = [
        pct(by_dataset[dataset]["baseline_accuracy"]) for dataset in DATASETS
    ] + [
        pct(controller["macro_baseline_accuracy"]),
        f"{controller['macro_baseline_calls']:.2f}",
    ]
    rewrite_values = [
        pct(rewrite_by_dataset[dataset]["rewrite_accuracy"]) for dataset in DATASETS
    ] + [
        pct(rewrite["macro_rewrite_accuracy"]),
        f"{rewrite['macro_rewrite_calls']:.2f}",
    ]
    selected_values = [
        pct(by_dataset[dataset]["selected_accuracy"]) for dataset in DATASETS
    ] + [
        pct(controller["macro_selected_accuracy"]),
        f"{controller['macro_selected_calls']:.2f}",
    ]
    table_rows = [
        ("Gemma 4 26B + held-out baseline", baseline_values, "rows 200-249"),
        ("+ legal query rewrite", rewrite_values, "same rows"),
        ("+ selected diagnostic routes", selected_values, "same rows"),
    ]
    make_table_figure(
        "13_diagnostic_adaptation_heldout_ablation.png",
        "Inherited Ablation: Held-Out Slice",
        "Held-out rows validate the routing story, with HousingQA and CaseHOLD carrying the clearest gains.",
        table_rows,
        "Sources: docs/heldout_controller_eval_2026-05-10.json and "
        "docs/heldout_query_rewrite_2026-05-10.json.",
    )


def controller_lift_bar() -> None:
    portfolios = portfolio_rows()
    heldout_controller = load_json(HELDOUT_CONTROLLER_JSON)

    labels = ["Calibration\nmacro", "Held-out\nmacro"]
    baseline = [
        100 * portfolios["baseline_retrieval"]["macro_accuracy"],
        100 * heldout_controller["macro_baseline_accuracy"],
    ]
    controller = [
        100 * portfolios["diagnostic_controller"]["macro_accuracy"],
        100 * heldout_controller["macro_selected_accuracy"],
    ]
    calls_base = [
        portfolios["baseline_retrieval"]["macro_avg_calls"],
        heldout_controller["macro_baseline_calls"],
    ]
    calls_controller = [
        portfolios["diagnostic_controller"]["macro_avg_calls"],
        heldout_controller["macro_selected_calls"],
    ]

    fig, ax = plt.subplots(figsize=(7.8, 4.4), dpi=220)
    xs = [0, 1]
    width = 0.28
    ax.bar([x - width / 2 for x in xs], baseline, width=width, color="#8292a2", label="baseline")
    ax.bar([x + width / 2 for x in xs], controller, width=width, color=COLORS["green"], label="controller")
    for i, x in enumerate(xs):
        ax.text(x - width / 2, baseline[i] + 0.8, f"{baseline[i]:.1f}\n{calls_base[i]:.2f} calls", ha="center", fontsize=8)
        ax.text(x + width / 2, controller[i] + 0.8, f"{controller[i]:.1f}\n{calls_controller[i]:.2f} calls", ha="center", fontsize=8, weight="bold")
        ax.annotate(
            f"+{controller[i] - baseline[i]:.1f} pp",
            xy=(x, max(controller[i], baseline[i]) + 5.0),
            ha="center",
            fontsize=10,
            color=COLORS["green"],
            weight="bold",
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Macro accuracy (%)")
    ax.set_ylim(60, 88)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Diagnostic Routing Adds Accuracy Without Fixed 2-Call Cost", fontsize=13, weight="bold")
    ax.text(
        0.0,
        -0.18,
        "Labels show accuracy and average LLM calls per question.",
        transform=ax.transAxes,
        fontsize=8,
        color=COLORS["muted"],
    )
    fig.savefig(OUT / "14_diagnostic_controller_macro_lift.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def route_map() -> None:
    cards = [
        {
            "dataset": "BarExam",
            "bottleneck": "Query / legal-reasoning formulation",
            "signal": "Snap-HyRE v2 86.0 vs baseline 80.0\nHyRE-only 82.0; held-out rewrite wins",
            "route": "rewrite-vs-HyRE selector",
            "color": COLORS["blue"],
        },
        {
            "dataset": "HousingQA",
            "bottleneck": "Statutory entailment / false-positive Yes",
            "signal": "Verifier 74.5 vs state baseline 60.5\nHyRE-only 50.0; held-out 76.0 vs 62.0",
            "route": "state filter + verifier",
            "color": COLORS["orange"],
        },
        {
            "dataset": "CaseHOLD",
            "bottleneck": "Answer-option conversion",
            "signal": "Diverse HyRE 78.0 vs baseline 68.0\nHyRE-only 71.5; option table 70.0",
            "route": "diverse HyRE + reject/escalate",
            "color": COLORS["red"],
        },
        {
            "dataset": "SCALR",
            "bottleneck": "Method disagreement /\ncandidate exposure",
            "signal": "Calibration controller 77.5 vs 74.0\nHeld-out frontier 84.0; route ties at 80.0",
            "route": "disagreement arbitration",
            "color": COLORS["green"],
        },
    ]

    fig, ax = plt.subplots(figsize=(12.8, 6.2), dpi=220)
    ax.axis("off")
    ax.text(0.02, 0.95, "Bottleneck-Aware Diagnostic Adaptation", fontsize=18, weight="bold", color=COLORS["ink"], transform=ax.transAxes)
    ax.text(0.02, 0.90, "Calibration traces type each legal benchmark, then route to the cheapest useful intervention.", fontsize=10, color=COLORS["muted"], transform=ax.transAxes)

    col_x = [0.05, 0.37, 0.69]
    headers = ["Evidence signal", "Diagnosed bottleneck", "Policy route"]
    for x, header in zip(col_x, headers):
        ax.text(x, 0.82, header, fontsize=11.5, weight="bold", color=COLORS["ink"], transform=ax.transAxes)

    y = 0.70
    h = 0.145
    for card in cards:
        for x in col_x:
            rect = patches.FancyBboxPatch(
                (x, y),
                0.25,
                h,
                boxstyle="round,pad=0.012,rounding_size=0.012",
                transform=ax.transAxes,
                facecolor="white",
                edgecolor=card["color"],
                linewidth=1.25,
            )
            ax.add_patch(rect)
        ax.text(col_x[0] + 0.012, y + h - 0.026, card["dataset"], fontsize=10.2, weight="bold", color=card["color"], va="top", transform=ax.transAxes)
        ax.text(col_x[0] + 0.012, y + h / 2 - 0.016, card["signal"], fontsize=8.0, color=COLORS["ink"], va="center", transform=ax.transAxes)
        ax.text(col_x[1] + 0.012, y + h / 2, card["bottleneck"], fontsize=8.8, color=COLORS["ink"], va="center", transform=ax.transAxes)
        ax.text(col_x[2] + 0.012, y + h / 2, card["route"], fontsize=9.2, weight="bold", color=COLORS["ink"], va="center", transform=ax.transAxes)
        ax.annotate("", xy=(col_x[1] - 0.018, y + h / 2), xytext=(col_x[0] + 0.258, y + h / 2), xycoords=ax.transAxes, arrowprops=dict(arrowstyle="->", color=COLORS["line"], lw=1.5))
        ax.annotate("", xy=(col_x[2] - 0.018, y + h / 2), xytext=(col_x[1] + 0.258, y + h / 2), xycoords=ax.transAxes, arrowprops=dict(arrowstyle="->", color=COLORS["line"], lw=1.5))
        y -= 0.17

    ax.text(
        0.02,
        0.02,
        "Source: docs/meeting_prep_2026-05-11_diagnostic_adaptation.md and linked source-gated result docs.",
        fontsize=8,
        color=COLORS["muted"],
        transform=ax.transAxes,
    )
    fig.savefig(OUT / "15_bottleneck_diagnostic_route_map.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def method_ladder_flowchart() -> None:
    """Show the inherited method family as a compact process diagram."""
    methods = [
        {
            "name": "Baseline RAG",
            "steps": ["Question", "Corpus retrieval", "Evidence-grounded answer"],
            "note": "1 LLM call after retrieval",
            "color": "#8292a2",
        },
        {
            "name": "Snap Only",
            "steps": ["Question", "Quick legal reasoning", "Final answer, no corpus"],
            "note": "Tests parametric reasoning and anchoring",
            "color": COLORS["purple"],
        },
        {
            "name": "HyRE / HyDE",
            "steps": ["Question", "Hypothetical legal passage", "Retrieve by generated reasoning", "Answer with evidence"],
            "note": "Generated reasoning is the retrieval query",
            "color": COLORS["blue"],
        },
        {
            "name": "Snap-HyRE",
            "steps": ["Question", "Snap answer + HyRE passage", "Retrieve with HyRE", "Answer with evidence"],
            "note": "Reasoning is snap-conditioned before retrieval",
            "color": COLORS["green"],
        },
        {
            "name": "Diagnostic Controller",
            "steps": ["Calibration traces", "Bottleneck label", "Route: rewrite, HyRE, filter, verifier, option, reject"],
            "note": "Adapts the intervention to the failure mode",
            "color": COLORS["orange"],
        },
    ]

    fig, ax = plt.subplots(figsize=(13.2, 7.1), dpi=220)
    ax.axis("off")
    ax.text(0.02, 0.965, "Inherited Method Ladder", fontsize=18, weight="bold", color=COLORS["ink"], transform=ax.transAxes)
    ax.text(
        0.02,
        0.925,
        "The ablation is not a bag of prompts: each row inherits the previous retrieval/answer surface and adds one reasoning or routing mechanism.",
        fontsize=10,
        color=COLORS["muted"],
        transform=ax.transAxes,
    )

    y0 = 0.79
    row_h = 0.125
    step_w = 0.17
    x_name = 0.035
    x_steps = [0.24, 0.43, 0.62, 0.81]
    for ridx, method in enumerate(methods):
        y = y0 - ridx * 0.15
        ax.text(x_name, y + row_h * 0.62, method["name"], fontsize=11.3, weight="bold", color=method["color"], va="center", transform=ax.transAxes)
        ax.text(x_name, y + row_h * 0.25, textwrap.fill(method["note"], 28), fontsize=8.2, color=COLORS["muted"], va="center", transform=ax.transAxes)

        for sidx, step in enumerate(method["steps"]):
            x = x_steps[sidx]
            rect = patches.FancyBboxPatch(
                (x, y),
                step_w,
                row_h,
                boxstyle="round,pad=0.010,rounding_size=0.012",
                transform=ax.transAxes,
                facecolor="white",
                edgecolor=method["color"],
                linewidth=1.2,
            )
            ax.add_patch(rect)
            ax.text(
                x + step_w / 2,
                y + row_h / 2,
                textwrap.fill(step, 22),
                ha="center",
                va="center",
                fontsize=8.5,
                color=COLORS["ink"],
                transform=ax.transAxes,
            )
            if sidx < len(method["steps"]) - 1:
                ax.annotate(
                    "",
                    xy=(x_steps[sidx + 1] - 0.004, y + row_h / 2),
                    xytext=(x + step_w + 0.004, y + row_h / 2),
                    xycoords=ax.transAxes,
                    arrowprops=dict(arrowstyle="->", color=COLORS["line"], lw=1.35),
                )

    ax.text(
        0.02,
        0.025,
        "Use this with the ablation table: baseline -> snap-only -> HyRE-only -> fixed Snap-HyRE -> bottleneck-aware routing.",
        fontsize=8.2,
        color=COLORS["muted"],
        transform=ax.transAxes,
    )
    fig.savefig(OUT / "16_method_ladder_flowchart.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    calibration_ablation()
    heldout_ablation()
    controller_lift_bar()
    route_map()
    method_ladder_flowchart()
    for path in sorted(OUT.glob("1[2-6]_*.png")):
        print(path)


if __name__ == "__main__":
    main()
