#!/usr/bin/env python3
"""Regenerate Figure 3 from the final package top-k retrieval CSV.

This script intentionally reads the final paper-facing CSV instead of the
older paper workspace generator. The CSV already points each row back to its
retrieval-cache JSONL source via `source_path`.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TOPK_KS = [1, 3, 5, 10]
COLORS = {
    "rag_simple": "#2f6f9f",
    "rag_hyde": "#48a9a6",
    "snap_hyre": "#d4942f",
}
MODE_LABEL = {
    "rag_simple": "Raw question RAG",
    "rag_hyde": "HyDE",
    "snap_hyre": "Snap-HyRE (ours)",
}


def nice_upper(values: list[float], floor: float = 0.0, ceiling: float = 100.0) -> float:
    if not values:
        return floor
    top = max(values) * 1.12
    step = 5.0 if top <= 60 else 10.0
    return min(ceiling, max(floor, math.ceil(top / step) * step))


def load_rows(csv_path: Path) -> list[dict[str, object]]:
    rows_by_key: dict[tuple[object, ...], dict[str, object]] = {}
    with csv_path.open(newline="") as f:
        for record in csv.DictReader(f):
            key = (
                record["dataset"],
                record["provider"],
                record["model"],
                record["mode"],
                int(record["n"]),
                record["source_path"],
            )
            row = rows_by_key.setdefault(
                key,
                {
                    "dataset": record["dataset"],
                    "provider": record["provider"],
                    "model": record["model"],
                    "mode": record["mode"],
                    "n": int(record["n"]),
                    "path": record["source_path"],
                    "hit": {},
                    "mrr": {},
                },
            )
            k = int(record["k"])
            row["hit"][k] = float(record["hit"])  # type: ignore[index]
            row["mrr"][k] = float(record["mrr"])  # type: ignore[index]
    return list(rows_by_key.values())


def aggregate(rows: list[dict[str, object]], dataset: str, mode: str) -> dict[str, object] | None:
    group = [
        row
        for row in rows
        if row["dataset"] == dataset and row["mode"] == mode and row["provider"] != "shared"
    ]
    if not group:
        return None
    return {
        "dataset": dataset,
        "provider": "mean",
        "model": f"Mean over {3 if dataset == 'BarExamQA' else 2} models",
        "mode": mode,
        "n": group[0]["n"],
        "hit": {
            k: sum(row["hit"][k] for row in group) / len(group)  # type: ignore[index]
            for k in TOPK_KS
        },
        "mrr": {
            k: sum(row["mrr"][k] for row in group) / len(group)  # type: ignore[index]
            for k in TOPK_KS
        },
    }


def plot(rows: list[dict[str, object]], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 5.0), sharex=True, constrained_layout=True)
    handles_for_legend = []
    labels_for_legend = []

    for col, dataset in enumerate(["BarExamQA", "HousingQA"]):
        for mode in ["rag_simple", "rag_hyde", "snap_hyre"]:
            if mode in {"rag_hyde", "snap_hyre"}:
                row = aggregate(rows, dataset, mode)
            else:
                candidates = [
                    row for row in rows if row["dataset"] == dataset and row["mode"] == mode
                ]
                row = candidates[0] if candidates else None
            if row is None:
                continue
            hit_values = [100 * row["hit"][k] for k in TOPK_KS]  # type: ignore[index]
            mrr_values = [100 * row["mrr"][k] for k in TOPK_KS]  # type: ignore[index]
            label = MODE_LABEL[mode] + (" mean" if row["provider"] == "mean" else "")
            line0, = axes[0, col].plot(
                TOPK_KS,
                hit_values,
                marker="o",
                markersize=5.0,
                linewidth=2.0,
                color=COLORS[mode],
                label=label,
            )
            axes[1, col].plot(
                TOPK_KS,
                mrr_values,
                marker="o",
                markersize=5.0,
                linewidth=2.0,
                color=COLORS[mode],
                label=label,
            )
            if col == 0 and label not in labels_for_legend:
                handles_for_legend.append(line0)
                labels_for_legend.append(label)

        axes[0, col].set_title(
            "HousingQA (state-filtered)" if dataset == "HousingQA" else "BarExamQA",
            fontsize=12,
            fontweight="bold",
        )
        axes[1, col].set_xlabel("k", fontsize=11)
        axes[0, col].set_ylabel("Hit@k (%)", fontsize=11)
        axes[1, col].set_ylabel("MRR@k (%)", fontsize=11)
        for row_ax in axes[:, col]:
            row_ax.set_xticks(TOPK_KS)
            row_ax.tick_params(axis="both", labelsize=10)
            row_ax.set_ylim(bottom=0)

    axes[0, 0].set_ylim(0, 18)
    axes[1, 0].set_ylim(0, 8)

    housing_hit_vals = []
    housing_mrr_vals = []
    for row in rows:
        if row["dataset"] == "HousingQA":
            housing_hit_vals.extend([100 * row["hit"][k] for k in TOPK_KS])  # type: ignore[index]
            housing_mrr_vals.extend([100 * row["mrr"][k] for k in TOPK_KS])  # type: ignore[index]
    axes[0, 1].set_ylim(0, nice_upper(housing_hit_vals, floor=40.0, ceiling=55.0))
    axes[1, 1].set_ylim(0, nice_upper(housing_mrr_vals, floor=25.0, ceiling=100.0))

    fig.legend(
        handles_for_legend,
        labels_for_legend,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=len(labels_for_legend),
        frameon=False,
        fontsize=10,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="paper/after_report/tables/topk_retrieval_metrics.csv",
        type=Path,
    )
    parser.add_argument(
        "--out",
        default="paper/after_report/plots/figure3_topk_retrieval_curves_regenerated.png",
        type=Path,
    )
    args = parser.parse_args()
    plot(load_rows(args.csv), args.out)
    print(args.out)


if __name__ == "__main__":
    main()
