from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

MPLCONFIGDIR = Path(tempfile.gettempdir()) / "legalragagent_mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MPLCONFIGDIR))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "logs"
DEFAULT_OUT_DIR = ROOT.parent / "figures"


@dataclass(frozen=True)
class RunSpec:
    key: str
    label: str
    filename: str
    color: str

    @property
    def path(self) -> Path:
        return LOG_DIR / self.filename


RUNS: dict[str, RunSpec] = {
    "llm_only": RunSpec(
        key="llm_only",
        label="LLM only",
        filename="eval_llm_only_cluster-vllm_20260408_1811_detail.jsonl",
        color="#7F7F7F",
    ),
    "rag_simple": RunSpec(
        key="rag_simple",
        label="Vanilla RAG",
        filename="eval_rag_simple_cluster-vllm_20260408_1813_detail.jsonl",
        color="#4E79A7",
    ),
    "rag_hyde_old": RunSpec(
        key="rag_hyde_old",
        label="Vanilla HyDE",
        filename="eval_rag_hyde_cluster-vllm_20260415_1346_detail.jsonl",
        color="#E15759",
    ),
    "rag_snap_hyde": RunSpec(
        key="rag_snap_hyde",
        label="SNAP-HyDE",
        filename="eval_rag_snap_hyde_cluster-vllm_20260409_0819_detail.jsonl",
        color="#76B7B2",
    ),
    "golden_passage": RunSpec(
        key="golden_passage",
        label="Golden passage",
        filename="eval_golden_passage_cluster-vllm_20260408_1615_detail.jsonl",
        color="#D4A017",
    ),
}

CONDITIONAL_KEYS = ["rag_simple", "rag_hyde_old", "rag_snap_hyde"]
EFFICIENCY_KEYS = ["llm_only", "rag_simple", "rag_hyde_old", "rag_snap_hyde", "golden_passage"]
HARD_KEYS = ["llm_only", "rag_simple", "rag_hyde_old", "rag_snap_hyde"]

EFFICIENCY_ACCURACY_SETS = {
    "old": {
        "llm_only": 0.521,
        "rag_simple": 0.542,
        "rag_hyde_old": 0.543,
        "rag_snap_hyde": 0.579,
        "golden_passage": 0.622,
    },
    "recent": {
        "llm_only": 0.521,
        "rag_simple": 0.542,
        "rag_hyde_old": 0.579,
        "rag_snap_hyde": 0.579,
        "golden_passage": 0.622,
    },
}


def normalize_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            record["is_correct"] = normalize_bool(record.get("is_correct"))
            record["gold_retrieved"] = normalize_bool(record.get("gold_retrieved"))
            rows.append(record)
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    return rows


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def accuracy(rows: list[dict]) -> float:
    return mean(1.0 if row["is_correct"] else 0.0 for row in rows)


def average(rows: list[dict], field: str) -> float:
    return mean(float(row.get(field, 0.0)) for row in rows)


def build_conditional_metrics(rows_by_key: dict[str, list[dict]]) -> dict[str, dict[str, float | int]]:
    metrics: dict[str, dict[str, float | int]] = {}
    for key in CONDITIONAL_KEYS:
        rows = rows_by_key[key]
        retrieved = [row for row in rows if row["gold_retrieved"]]
        missed = [row for row in rows if not row["gold_retrieved"]]
        if not retrieved or not missed:
            raise RuntimeError(f"Expected both gold-retrieved and missed rows for {key}")
        retrieved_acc = accuracy(retrieved)
        missed_acc = accuracy(missed)
        metrics[key] = {
            "retrieved_n": len(retrieved),
            "missed_n": len(missed),
            "retrieved_acc": retrieved_acc,
            "missed_acc": missed_acc,
            "lift": retrieved_acc - missed_acc,
        }
    return metrics


def build_efficiency_metrics(
    rows_by_key: dict[str, list[dict]],
    accuracy_overrides: dict[str, float],
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for key in EFFICIENCY_KEYS:
        rows = rows_by_key[key]
        metrics[key] = {
            "accuracy": accuracy_overrides.get(key, accuracy(rows)),
            "llm_calls": average(rows, "llm_calls"),
            "input_tokens": average(rows, "input_tokens"),
            "output_tokens": average(rows, "output_tokens"),
            "total_tokens": average(rows, "input_tokens") + average(rows, "output_tokens"),
        }
    return metrics


def build_hard_metrics(
    rows_by_key: dict[str, list[dict]],
    golden_key: str = "golden_passage",
) -> tuple[dict[str, dict[str, float | int]], int, int]:
    golden_by_idx = {row["idx"]: row for row in rows_by_key[golden_key]}
    metrics: dict[str, dict[str, float | int]] = {}
    hard_n = 0
    easy_n = 0
    for key in HARD_KEYS:
        hard_rows: list[dict] = []
        easy_rows: list[dict] = []
        for row in rows_by_key[key]:
            golden_row = golden_by_idx.get(row["idx"])
            if golden_row is None:
                raise RuntimeError(f"Missing golden row for idx={row['idx']}")
            if golden_row["is_correct"]:
                easy_rows.append(row)
            else:
                hard_rows.append(row)
        hard_n = len(hard_rows)
        easy_n = len(easy_rows)
        metrics[key] = {
            "hard_acc": accuracy(hard_rows),
            "easy_acc": accuracy(easy_rows),
            "hard_n": hard_n,
            "easy_n": easy_n,
        }
    return metrics, hard_n, easy_n


def set_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "font.family": "DejaVu Sans",
        }
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_conditional_accuracy(
    outdir: Path,
    conditional_metrics: dict[str, dict[str, float | int]],
) -> None:
    labels = [RUNS[key].label for key in CONDITIONAL_KEYS]
    retrieved_values = [float(conditional_metrics[key]["retrieved_acc"]) for key in CONDITIONAL_KEYS]
    missed_values = [float(conditional_metrics[key]["missed_acc"]) for key in CONDITIONAL_KEYS]
    retrieved_counts = [int(conditional_metrics[key]["retrieved_n"]) for key in CONDITIONAL_KEYS]
    missed_counts = [int(conditional_metrics[key]["missed_n"]) for key in CONDITIONAL_KEYS]
    colors = [RUNS[key].color for key in CONDITIONAL_KEYS]

    x = np.arange(len(CONDITIONAL_KEYS))
    width = 0.34

    fig, ax = plt.subplots(figsize=(11, 6.5))
    retrieved_bars = ax.bar(
        x - width / 2,
        retrieved_values,
        width=width,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        alpha=0.95,
        label="When gold passage is retrieved",
    )
    missed_bars = ax.bar(
        x + width / 2,
        missed_values,
        width=width,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        alpha=0.35,
        label="When gold passage is not retrieved",
    )

    for bars, values, counts in (
        (retrieved_bars, retrieved_values, retrieved_counts),
        (missed_bars, missed_values, missed_counts),
    ):
        for bar, value, count in zip(bars, values, counts, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.015,
                f"{pct(value)}\n(n={count})",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    ax.set_title("Accuracy Conditional on Gold Retrieval")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_ylim(0, 1.08)
    legend_handles = [
        Patch(facecolor="#374151", edgecolor="black", alpha=0.95, label="When gold passage is retrieved"),
        Patch(facecolor="#374151", edgecolor="black", alpha=0.35, label="When gold passage is not retrieved"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=True)
    fig.text(
        0.5,
        0.01,
        "Old HyDE pinned to eval_rag_hyde_cluster-vllm_20260415_1346_detail.jsonl.",
        ha="center",
        fontsize=10,
    )
    save_figure(fig, outdir / "old_hyde_conditional_accuracy.png")


def plot_efficiency_table(
    outdir: Path,
    efficiency_metrics: dict[str, dict[str, float]],
    accuracy_set: str,
) -> None:
    column_labels = [
        "Mode",
        "LLM calls",
        "Input tok/q",
        "Output tok/q",
        "Total tok/q",
        "Acc.",
    ]
    row_keys = EFFICIENCY_KEYS
    row_labels = [RUNS[key].label for key in row_keys]
    cell_text = [
        [
            row_labels[idx],
            f"{efficiency_metrics[key]['llm_calls']:.2f}",
            f"{efficiency_metrics[key]['input_tokens']:.0f}",
            f"{efficiency_metrics[key]['output_tokens']:.0f}",
            f"{efficiency_metrics[key]['total_tokens']:.0f}",
            pct(efficiency_metrics[key]["accuracy"]),
        ]
        for idx, key in enumerate(row_keys)
    ]

    fig, ax = plt.subplots(figsize=(12.5, 4.8))
    ax.axis("off")
    ax.set_title("Efficiency Per Question", pad=18)

    table = ax.table(
        cellText=cell_text,
        colLabels=column_labels,
        loc="center",
        cellLoc="center",
        colLoc="center",
        bbox=[0.0, 0.02, 1.0, 0.82],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1, 1.5)

    for col in range(len(column_labels)):
        table[(0, col)].set_facecolor("#1F2937")
        table[(0, col)].set_text_props(color="white", weight="bold")
        table[(0, col)].set_edgecolor("#1F2937")

    for row_idx, key in enumerate(row_keys, start=1):
        base_color = RUNS[key].color
        for col_idx in range(len(column_labels)):
            cell = table[(row_idx, col_idx)]
            cell.set_edgecolor("#D1D5DB")
            if col_idx == 0:
                cell.set_facecolor(base_color)
                cell.set_text_props(color="white", weight="bold")
            else:
                cell.set_facecolor("#F8FAFC" if row_idx % 2 else "#EEF2F7")

    fig.text(
        0.5,
        0.02,
        (
            "Token/call columns use local detail-log averages; the accuracy column uses the "
            f"{accuracy_set} slide-tier values from the core retrieval chart."
        ),
        ha="center",
        fontsize=10,
    )
    save_figure(fig, outdir / "old_hyde_efficiency_table.png")


def plot_hard_questions(
    outdir: Path,
    hard_metrics: dict[str, dict[str, float | int]],
    hard_n: int,
    easy_n: int,
) -> None:
    groups = [f"Hard questions\n(golden failed, n={hard_n})", f"Easy questions\n(golden passed, n={easy_n})"]
    x = np.arange(len(groups))
    width = 0.18

    fig, ax = plt.subplots(figsize=(12, 6.8))
    for offset, key in enumerate(HARD_KEYS):
        values = [
            float(hard_metrics[key]["hard_acc"]),
            float(hard_metrics[key]["easy_acc"]),
        ]
        bars = ax.bar(
            x + (offset - 1.5) * width,
            values,
            width=width,
            color=RUNS[key].color,
            edgecolor="black",
            linewidth=0.5,
            label=RUNS[key].label,
        )
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.012,
                pct(value),
                ha="center",
                va="bottom",
                fontsize=10,
            )

    ax.set_title("Accuracy on Hard vs Easy Questions")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_ylim(0, 0.82)
    ax.legend(ncol=2, frameon=True, loc="upper left")
    fig.text(
        0.5,
        0.01,
        "Partition is defined by the local golden_passage detail log: eval_golden_passage_cluster-vllm_20260408_1615_detail.jsonl.",
        ha="center",
        fontsize=10,
    )
    save_figure(fig, outdir / "old_hyde_hard_questions.png")


def write_summary(
    outdir: Path,
    conditional_metrics: dict[str, dict[str, float | int]],
    efficiency_metrics: dict[str, dict[str, float]],
    hard_metrics: dict[str, dict[str, float | int]],
    hard_n: int,
    easy_n: int,
) -> None:
    source_lines = [
        f"- `{spec.key}`: `{spec.filename}`"
        for spec in (RUNS[key] for key in EFFICIENCY_KEYS)
    ]

    conditional_rows = [
        (
            RUNS[key].label,
            pct(float(conditional_metrics[key]["retrieved_acc"])),
            int(conditional_metrics[key]["retrieved_n"]),
            pct(float(conditional_metrics[key]["missed_acc"])),
            int(conditional_metrics[key]["missed_n"]),
            f"{float(conditional_metrics[key]['lift']) * 100:+.1f}pp",
        )
        for key in CONDITIONAL_KEYS
    ]

    efficiency_rows = [
        (
            RUNS[key].label,
            f"{efficiency_metrics[key]['llm_calls']:.2f}",
            f"{efficiency_metrics[key]['input_tokens']:.0f}",
            f"{efficiency_metrics[key]['output_tokens']:.0f}",
            f"{efficiency_metrics[key]['total_tokens']:.0f}",
            pct(efficiency_metrics[key]["accuracy"]),
        )
        for key in EFFICIENCY_KEYS
    ]

    hard_rows = [
        (
            RUNS[key].label,
            pct(float(hard_metrics[key]["hard_acc"])),
            pct(float(hard_metrics[key]["easy_acc"])),
        )
        for key in HARD_KEYS
    ]

    lines = [
        "# Old HyDE Figure Inputs",
        "",
        "These are the exact values used to render the slide figures.",
        "",
        "## Source runs",
        "",
        *source_lines,
        "",
        "## Accuracy conditional on gold retrieval",
        "",
        "| Mode | When gold retrieved | n | When gold not retrieved | n | Lift |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        *[
            f"| {mode} | {retrieved} | {retrieved_n} | {missed} | {missed_n} | {lift} |"
            for mode, retrieved, retrieved_n, missed, missed_n, lift in conditional_rows
        ],
        "",
        "## Efficiency per question",
        "",
        "| Mode | LLM calls | Input tok/q | Output tok/q | Total tok/q | Accuracy |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        *[
            f"| {mode} | {llm_calls} | {input_tokens} | {output_tokens} | {total_tokens} | {acc} |"
            for mode, llm_calls, input_tokens, output_tokens, total_tokens, acc in efficiency_rows
        ],
        "",
        f"## Hard vs easy split (golden failed n={hard_n}, golden passed n={easy_n})",
        "",
        "| Mode | Hard questions | Easy questions |",
        "| --- | ---: | ---: |",
        *[
            f"| {mode} | {hard_acc} | {easy_acc} |"
            for mode, hard_acc, easy_acc in hard_rows
        ],
        "",
    ]

    output_path = outdir / "old_hyde_figure_inputs.md"
    output_path.write_text("\n".join(lines))
    print(f"Saved {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate old-HyDE slide figures from local detail logs.")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory where the figures should be written.",
    )
    parser.add_argument(
        "--efficiency-accuracy-set",
        choices=sorted(EFFICIENCY_ACCURACY_SETS),
        default="old",
        help="Which presentation accuracy values to use for the efficiency table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    set_style()

    rows_by_key = {key: load_rows(spec.path) for key, spec in RUNS.items()}
    conditional_metrics = build_conditional_metrics(rows_by_key)
    efficiency_metrics = build_efficiency_metrics(
        rows_by_key,
        EFFICIENCY_ACCURACY_SETS[args.efficiency_accuracy_set],
    )
    hard_metrics, hard_n, easy_n = build_hard_metrics(rows_by_key)

    plot_conditional_accuracy(outdir, conditional_metrics)
    plot_efficiency_table(outdir, efficiency_metrics, args.efficiency_accuracy_set)
    plot_hard_questions(outdir, hard_metrics, hard_n, easy_n)
    write_summary(outdir, conditional_metrics, efficiency_metrics, hard_metrics, hard_n, easy_n)


if __name__ == "__main__":
    main()
