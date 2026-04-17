from __future__ import annotations

import json
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

MPLCONFIGDIR = Path(tempfile.gettempdir()) / "legalragagent_mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MPLCONFIGDIR))

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "logs"
OUT_DIR = ROOT / "docs" / "charts"


@dataclass(frozen=True)
class DetailRun:
    key: str
    filename: str
    mode: str
    label: str


DETAIL_RUNS: list[DetailRun] = [
    DetailRun(
        key="llm_only_20260408_1811",
        filename="eval_llm_only_cluster-vllm_20260408_1811_detail.jsonl",
        mode="llm_only",
        label="LLM only",
    ),
    DetailRun(
        key="rag_snap_hyde_20260413_1102",
        filename="eval_rag_snap_hyde_cluster-vllm_20260413_1102_detail.jsonl",
        mode="rag_snap_hyde",
        label="RAG + SNAP + HyDE",
    ),
    DetailRun(
        key="rag_simple_20260408_1813",
        filename="eval_rag_simple_cluster-vllm_20260408_1813_detail.jsonl",
        mode="rag_simple",
        label="Simple RAG",
    ),
    DetailRun(
        key="golden_passage_20260408_1615",
        filename="eval_golden_passage_cluster-vllm_20260408_1615_detail.jsonl",
        mode="golden_passage",
        label="Golden passage",
    ),
    DetailRun(
        key="subagent_rag_20260414_1115",
        filename="eval_subagent_rag_cluster-vllm_20260414_1115_detail.jsonl",
        mode="subagent_rag",
        label="Subagent RAG",
    ),
    DetailRun(
        key="entity_search_20260415_0454",
        filename="eval_entity_search_cluster-vllm_20260415_0454_detail.jsonl",
        mode="entity_search",
        label="Entity search",
    ),
    DetailRun(
        key="ce_threshold_20260415_2022",
        filename="eval_ce_threshold_cluster-vllm_20260415_2022_detail.jsonl",
        mode="ce_threshold",
        label="CE threshold",
    ),
    DetailRun(
        key="rag_hyde_20260415_1346",
        filename="eval_rag_hyde_cluster-vllm_20260415_1346_detail.jsonl",
        mode="rag_hyde",
        label="RAG + HyDE",
    ),
    DetailRun(
        key="gap_rag_nosnap_20260416_0544",
        filename="eval_gap_rag_nosnap_cluster-vllm_20260416_0544_detail.jsonl",
        mode="gap_rag_nosnap",
        label="Gap RAG (no SNAP)",
    ),
    DetailRun(
        key="subagent_rag_20260416_1720",
        filename="eval_subagent_rag_cluster-vllm_20260416_1720_detail.jsonl",
        mode="subagent_rag",
        label="Subagent RAG (1-gap)",
    ),
]

KEY_MODE_ORDER = [
    "llm_only_20260408_1811",
    "rag_simple_20260408_1813",
    "rag_snap_hyde_20260413_1102",
    "subagent_rag_20260414_1115",
    "golden_passage_20260408_1615",
]

SWITCH_PAIRS = [
    ("llm_only_20260408_1811", "rag_snap_hyde_20260413_1102"),
    ("llm_only_20260408_1811", "subagent_rag_20260414_1115"),
    ("rag_snap_hyde_20260413_1102", "subagent_rag_20260414_1115"),
    ("rag_simple_20260408_1813", "rag_snap_hyde_20260413_1102"),
]

MODE_LABELS = {
    "llm_only": "LLM only",
    "rag_simple": "Simple RAG",
    "rag_snap_hyde": "RAG + SNAP + HyDE",
    "golden_passage": "Golden passage",
    "subagent_rag": "Subagent RAG",
    "entity_search": "Entity search",
    "ce_threshold": "CE threshold",
    "rag_hyde": "RAG + HyDE",
    "gap_rag_nosnap": "Gap RAG (no SNAP)",
}


def normalize_subject(subject: Any) -> str:
    if subject is None:
        return "UNLABELED"
    text = str(subject).strip()
    if not text or text.lower() == "nan":
        return "UNLABELED"
    return text


def normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            record["subject"] = normalize_subject(record.get("subject"))
            record["is_correct"] = normalize_bool(record.get("is_correct"))
            rows.append(record)
    return rows


def compute_subject_accuracy(
    rows_by_run: dict[str, list[dict[str, Any]]]
) -> tuple[list[str], dict[str, dict[str, float]], dict[str, float]]:
    all_subjects = sorted(
        {
            record["subject"]
            for rows in rows_by_run.values()
            for record in rows
        }
    )
    accuracy_by_run: dict[str, dict[str, float]] = {}
    overall_accuracy: dict[str, float] = {}
    for run_key, rows in rows_by_run.items():
        grouped: dict[str, list[bool]] = defaultdict(list)
        for record in rows:
            grouped[record["subject"]].append(record["is_correct"])
        accuracy_by_run[run_key] = {
            subject: float(np.mean(grouped[subject]))
            for subject in all_subjects
        }
        overall_accuracy[run_key] = float(
            np.mean([record["is_correct"] for record in rows])
        )
    return all_subjects, accuracy_by_run, overall_accuracy


def save_figure(fig: plt.Figure, output_name: str) -> None:
    fig.tight_layout()
    output_path = OUT_DIR / output_name
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_subject_accuracy_bars(
    runs: dict[str, DetailRun],
    accuracy_by_run: dict[str, dict[str, float]],
) -> None:
    subjects = sorted(
        accuracy_by_run["llm_only_20260408_1811"].keys(),
        key=lambda subject: accuracy_by_run["llm_only_20260408_1811"][subject],
        reverse=True,
    )
    bar_width = 0.16
    x = np.arange(len(subjects))
    colors = {
        "llm_only_20260408_1811": "#7F7F7F",
        "rag_simple_20260408_1813": "#4E79A7",
        "rag_snap_hyde_20260413_1102": "#76B7B2",
        "subagent_rag_20260414_1115": "#59A14F",
        "golden_passage_20260408_1615": "#D4A017",
    }

    fig, ax = plt.subplots(figsize=(14, 7))
    for offset, run_key in enumerate(KEY_MODE_ORDER):
        values = [accuracy_by_run[run_key][subject] for subject in subjects]
        ax.bar(
            x + (offset - 2) * bar_width,
            values,
            width=bar_width,
            label=runs[run_key].label,
            color=colors[run_key],
            edgecolor="black",
            linewidth=0.4,
        )

    ax.set_title("Accuracy by Subject Across Key Modes (N=1195)")
    ax.set_xlabel("Subject")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=25, ha="right")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_ylim(0, 1)
    ax.legend(ncol=3, frameon=True)
    save_figure(fig, "subject_accuracy_bars.png")


def plot_subject_accuracy_heatmap(
    runs: dict[str, DetailRun],
    accuracy_by_run: dict[str, dict[str, float]],
) -> None:
    subjects = sorted(
        accuracy_by_run["llm_only_20260408_1811"].keys(),
        key=lambda subject: accuracy_by_run["llm_only_20260408_1811"][subject],
        reverse=True,
    )
    run_keys = [detail_run.key for detail_run in DETAIL_RUNS]
    data = np.array(
        [
            [accuracy_by_run[run_key][subject] for subject in subjects]
            for run_key in run_keys
        ]
    )

    fig, ax = plt.subplots(figsize=(16, 7))
    norm = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
    image = ax.imshow(data, cmap="RdYlGn", norm=norm, aspect="auto")
    colorbar = fig.colorbar(image, ax=ax, shrink=0.9)
    colorbar.set_label("Accuracy")

    ax.set_title("Subject Accuracy Heatmap Across 10 Modes (N=1195)")
    ax.set_xticks(np.arange(len(subjects)))
    ax.set_xticklabels(subjects, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(run_keys)))
    ax.set_yticklabels([runs[run_key].label for run_key in run_keys])

    for row_index, run_key in enumerate(run_keys):
        for col_index, subject in enumerate(subjects):
            value = accuracy_by_run[run_key][subject]
            text_color = "white" if abs(value - 0.5) > 0.18 else "black"
            ax.text(
                col_index,
                row_index,
                f"{value:.0%}",
                ha="center",
                va="center",
                fontsize=9,
                color=text_color,
            )

    save_figure(fig, "subject_accuracy_heatmap.png")


def classify_switches(
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    left_by_idx = {record["idx"]: record for record in left_rows}
    right_by_idx = {record["idx"]: record for record in right_rows}
    common_ids = sorted(set(left_by_idx) & set(right_by_idx))
    stats = {
        "n_common": len(common_ids),
        "fixed": 0,
        "broke": 0,
        "both_right": 0,
        "both_wrong": 0,
        "subject_fixed": defaultdict(int),
        "subject_broke": defaultdict(int),
    }

    for idx in common_ids:
        left = left_by_idx[idx]
        right = right_by_idx[idx]
        left_correct = left["is_correct"]
        right_correct = right["is_correct"]

        if (not left_correct) and right_correct:
            stats["fixed"] += 1
            stats["subject_fixed"][left["subject"]] += 1
        elif left_correct and (not right_correct):
            stats["broke"] += 1
            stats["subject_broke"][left["subject"]] += 1
        elif left_correct and right_correct:
            stats["both_right"] += 1
        else:
            stats["both_wrong"] += 1

    stats["net"] = stats["fixed"] - stats["broke"]
    stats["pct_changed"] = (
        (stats["fixed"] + stats["broke"]) / stats["n_common"]
        if stats["n_common"]
        else 0.0
    )
    return stats


def print_switch_summary(
    runs: dict[str, DetailRun],
    switch_results: list[tuple[str, dict[str, Any]]],
) -> None:
    print("\nSwitch analysis summary")
    header = f"{'pair':40} {'#fixed':>8} {'#broke':>8} {'net':>8} {'%changed':>10}"
    print(header)
    print("-" * len(header))
    for pair_label, stats in switch_results:
        print(
            f"{pair_label:40} "
            f"{stats['fixed']:8d} "
            f"{stats['broke']:8d} "
            f"{stats['net']:8d} "
            f"{stats['pct_changed'] * 100:9.1f}%"
        )


def plot_switch_analysis_bars(
    switch_results: list[tuple[str, dict[str, Any]]]
) -> None:
    labels = [pair_label for pair_label, _ in switch_results]
    fixed_counts = [stats["fixed"] for _, stats in switch_results]
    broke_counts = [stats["broke"] for _, stats in switch_results]
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(13, 6.5))
    fixed_bars = ax.bar(
        x - width / 2,
        fixed_counts,
        width=width,
        color="#2E8B57",
        label="Fixed",
        edgecolor="black",
        linewidth=0.4,
    )
    broke_bars = ax.bar(
        x + width / 2,
        broke_counts,
        width=width,
        color="#C0392B",
        label="Broke",
        edgecolor="black",
        linewidth=0.4,
    )
    ax.bar_label(fixed_bars, padding=3)
    ax.bar_label(broke_bars, padding=3)

    ax.set_title("Switch Analysis Across Mode Pairs")
    ax.set_xlabel("Mode pair")
    ax.set_ylabel("Question count")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend(frameon=True)
    save_figure(fig, "switch_analysis_bars.png")


def plot_switch_by_subject(
    fixed_by_subject: dict[str, int],
    broke_by_subject: dict[str, int],
) -> None:
    subjects = sorted(
        set(fixed_by_subject) | set(broke_by_subject),
        key=lambda subject: fixed_by_subject.get(subject, 0) + broke_by_subject.get(subject, 0),
        reverse=True,
    )
    y = np.arange(len(subjects))
    height = 0.36
    fixed_counts = [fixed_by_subject.get(subject, 0) for subject in subjects]
    broke_counts = [broke_by_subject.get(subject, 0) for subject in subjects]

    fig, ax = plt.subplots(figsize=(12, 6.5))
    fixed_bars = ax.barh(
        y - height / 2,
        fixed_counts,
        height=height,
        color="#2E8B57",
        label="Fixed",
        edgecolor="black",
        linewidth=0.4,
    )
    broke_bars = ax.barh(
        y + height / 2,
        broke_counts,
        height=height,
        color="#C0392B",
        label="Broke",
        edgecolor="black",
        linewidth=0.4,
    )
    ax.bar_label(fixed_bars, padding=3)
    ax.bar_label(broke_bars, padding=3)

    ax.set_title("LLM only -> RAG + SNAP + HyDE: Fixes and Breaks by Subject")
    ax.set_xlabel("Question count")
    ax.set_ylabel("Subject")
    ax.set_yticks(y)
    ax.set_yticklabels(subjects)
    ax.invert_yaxis()
    ax.legend(frameon=True)
    save_figure(fig, "switch_by_subject.png")


def plot_leaderboard(
    runs: dict[str, DetailRun],
    overall_accuracy: dict[str, float],
) -> None:
    sorted_runs = sorted(
        overall_accuracy.items(),
        key=lambda item: item[1],
        reverse=False,
    )
    labels = [runs[run_key].label for run_key, _ in sorted_runs]
    values = [value for _, value in sorted_runs]

    colors = []
    for run_key, _ in sorted_runs:
        mode = runs[run_key].mode
        if mode == "golden_passage":
            colors.append("#D4A017")
        elif mode == "llm_only":
            colors.append("#7F7F7F")
        else:
            colors.append("#4682B4")

    fig, ax = plt.subplots(figsize=(11, 7))
    bars = ax.barh(
        np.arange(len(labels)),
        values,
        color=colors,
        edgecolor="black",
        linewidth=0.4,
    )
    ax.bar_label(bars, labels=[f"{value:.1%}" for value in values], padding=3)
    ax.set_title("Leaderboard Across 10 N=1195 Modes")
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Mode")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_xlim(0, max(values) + 0.08)
    save_figure(fig, "leaderboard_n1195.png")


def load_experiments() -> list[dict[str, Any]]:
    path = LOG_DIR / "experiments.jsonl"
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def select_scale_rows(experiments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    strict_rows = [
        row for row in experiments
        if row.get("provider") == "cluster-vllm"
        and row.get("dataset", "barexam") == "barexam"
    ]
    if strict_rows:
        print(
            "\nScale comparison filter: using strict provider == 'cluster-vllm' "
            f"({len(strict_rows)} rows)."
        )
        return strict_rows

    fallback_rows = [
        row
        for row in experiments
        if row.get("provider") == "custom"
        and row.get("model") == "google/gemma-4-E4B-it"
        and "cluster-vllm" in str(row.get("detail_log", ""))
        and row.get("dataset", "barexam") == "barexam"
    ]
    print(
        "\nScale comparison filter: provider == 'cluster-vllm' returned 0 rows. "
        "Falling back to provider == 'custom', model == 'google/gemma-4-E4B-it', "
        "and detail_log containing 'cluster-vllm' because that is how the cluster "
        "Gemma summary rows are recorded in logs/experiments.jsonl."
    )
    print(f"Fallback row count: {len(fallback_rows)}")
    return fallback_rows


def latest_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(rows, key=lambda row: row.get("timestamp", ""))


def plot_scale_comparison(experiments: list[dict[str, Any]]) -> None:
    filtered_rows = select_scale_rows(experiments)
    by_mode: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: {"n200": [], "n1195": []}
    )
    for row in filtered_rows:
        n_questions = row.get("n_questions", 0)
        if 195 <= n_questions <= 205:
            by_mode[row["mode"]]["n200"].append(row)
        elif n_questions >= 1000:
            by_mode[row["mode"]]["n1195"].append(row)

    selected_modes: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for mode, buckets in by_mode.items():
        if buckets["n200"] and buckets["n1195"]:
            selected_modes.append(
                (mode, latest_row(buckets["n200"]), latest_row(buckets["n1195"]))
            )

    selected_modes.sort(key=lambda item: item[1]["accuracy"], reverse=True)
    print(f"Scale comparison modes with both N=200 and N=1195 runs: {len(selected_modes)}")

    labels = [MODE_LABELS.get(mode, mode.replace("_", " ")) for mode, _, _ in selected_modes]
    values_200 = [n200["accuracy"] for _, n200, _ in selected_modes]
    values_1195 = [n1195["accuracy"] for _, _, n1195 in selected_modes]
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(13, 7))
    bars_200 = ax.bar(
        x - width / 2,
        values_200,
        width=width,
        color="#4E79A7",
        label="N=200",
        edgecolor="black",
        linewidth=0.4,
    )
    bars_1195 = ax.bar(
        x + width / 2,
        values_1195,
        width=width,
        color="#9C755F",
        label="N=1195",
        edgecolor="black",
        linewidth=0.4,
    )
    ax.bar_label(bars_200, labels=[f"{value:.1%}" for value in values_200], padding=3)
    ax.bar_label(bars_1195, labels=[f"{value:.1%}" for value in values_1195], padding=3)
    ax.set_title("Scale Comparison: N=200 vs N=1195")
    ax.set_xlabel("Mode")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_ylim(0, max(values_200 + values_1195) + 0.1)
    ax.legend(frameon=True)
    save_figure(fig, "scale_comparison.png")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )

    runs = {detail_run.key: detail_run for detail_run in DETAIL_RUNS}
    rows_by_run = {
        detail_run.key: load_jsonl(LOG_DIR / detail_run.filename)
        for detail_run in DETAIL_RUNS
    }

    for detail_run in DETAIL_RUNS:
        n_rows = len(rows_by_run[detail_run.key])
        if n_rows != 1195:
            raise ValueError(
                f"Expected 1195 rows for {detail_run.filename}, found {n_rows}."
            )

    print("Normalized subject label 'nan' -> 'UNLABELED' for chart readability.")
    subjects, accuracy_by_run, overall_accuracy = compute_subject_accuracy(rows_by_run)
    print(f"Loaded {len(DETAIL_RUNS)} detail logs across {len(subjects)} subjects.")

    plot_subject_accuracy_bars(runs, accuracy_by_run)
    plot_subject_accuracy_heatmap(runs, accuracy_by_run)

    switch_results: list[tuple[str, dict[str, Any]]] = []
    first_pair_stats: dict[str, Any] | None = None
    for left_key, right_key in SWITCH_PAIRS:
        stats = classify_switches(rows_by_run[left_key], rows_by_run[right_key])
        pair_label = f"{runs[left_key].label} -> {runs[right_key].label}"
        switch_results.append((pair_label, stats))
        if left_key == "llm_only_20260408_1811" and right_key == "rag_snap_hyde_20260413_1102":
            first_pair_stats = stats

    print_switch_summary(runs, switch_results)
    plot_switch_analysis_bars(switch_results)
    if first_pair_stats is None:
        raise RuntimeError("Expected llm_only -> rag_snap_hyde switch pair was not computed.")
    plot_switch_by_subject(
        fixed_by_subject=dict(first_pair_stats["subject_fixed"]),
        broke_by_subject=dict(first_pair_stats["subject_broke"]),
    )

    plot_leaderboard(runs, overall_accuracy)
    plot_scale_comparison(load_experiments())


if __name__ == "__main__":
    main()
