#!/usr/bin/env python3
"""Train simple offline baselines for bottleneck-aware routing tables.

The goal is deliberately modest: test whether cheap task features can predict
the oracle arm better than a fixed static policy. This is a research diagnostic,
not a production router.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LABEL_COLUMNS = {"oracle_accuracy_arm", "oracle_reward_arm"}
META_COLUMNS = {"join_key"}
STATIC_EXCLUDE_SUFFIXES = (
    "_correct",
    "_calls",
    "_latency_sec",
    "_input_tokens",
    "_output_tokens",
    "_gold_retrieved",
    "_evidence_count",
    "_max_ce_score",
    "_parse_ok",
)
PROBE_SUFFIXES = ("_evidence_count", "_max_ce_score")


def load_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                row = dict(row)
                row["_source_csv"] = str(path)
                rows.append(row)
    if not rows:
        raise SystemExit("No rows loaded")
    return rows


def infer_arms(rows: list[dict[str, str]]) -> list[str]:
    arms = set()
    for key in rows[0]:
        if key.endswith("_correct") and not key.startswith("oracle_"):
            arms.add(key[: -len("_correct")])
    if not arms:
        raise SystemExit("Could not infer arm columns ending in _correct")
    return sorted(arms)


def feature_columns(
    rows: list[dict[str, str]],
    include_dataset: bool,
    include_provider: bool,
    include_subject: bool,
    include_probe_features: bool,
) -> list[str]:
    cols = []
    for key in rows[0]:
        if key in META_COLUMNS or key in LABEL_COLUMNS or key.startswith("_") or key.startswith("oracle_"):
            continue
        if not include_dataset and key == "dataset":
            continue
        if not include_provider and key == "provider":
            continue
        if not include_subject and key == "subject":
            continue
        if include_probe_features and any(key.endswith(suffix) for suffix in PROBE_SUFFIXES):
            cols.append(key)
            continue
        if any(key.endswith(suffix) for suffix in STATIC_EXCLUDE_SUFFIXES):
            continue
        cols.append(key)
    return cols


def to_float(value: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except ValueError:
        return None
    if math.isnan(result):
        return None
    return result


def majority(labels: list[str], fallback: str) -> str:
    if not labels:
        return fallback
    counts = Counter(labels)
    return sorted(counts, key=lambda label: (-counts[label], label))[0]


def accuracy_for_arm(rows: list[dict[str, str]], arm: str) -> float:
    return sum(int(row.get(f"{arm}_correct", "0") or 0) for row in rows) / len(rows)


def avg_calls_for_arm(rows: list[dict[str, str]], arm: str) -> float:
    return sum(float(row.get(f"{arm}_calls", "0") or 0) for row in rows) / len(rows)


def evaluate_predictions(rows: list[dict[str, str]], predictions: list[str], label_column: str) -> dict[str, Any]:
    correct = 0
    label_match = 0
    calls = 0.0
    invalid = 0
    for row, arm in zip(rows, predictions):
        if f"{arm}_correct" not in row:
            invalid += 1
            continue
        correct += int(row.get(f"{arm}_correct", "0") or 0)
        label_match += int(arm == row[label_column])
        calls += float(row.get(f"{arm}_calls", "0") or 0)
    n = len(rows)
    return {
        "n": n,
        "accuracy": correct / n if n else 0.0,
        "label_match": label_match / n if n else 0.0,
        "avg_calls": calls / n if n else 0.0,
        "invalid": invalid,
        "chosen": Counter(predictions),
    }


class DecisionStump:
    def __init__(
        self,
        label_column: str,
        features: list[str],
        arms: list[str],
        call_penalty: float,
        latency_penalty: float,
    ):
        self.label_column = label_column
        self.features = features
        self.arms = arms
        self.call_penalty = call_penalty
        self.latency_penalty = latency_penalty
        self.global_label = ""
        self.rule: dict[str, Any] = {}

    def row_reward(self, row: dict[str, str], arm: str) -> float:
        correct = float(row.get(f"{arm}_correct", "0") or 0)
        calls = float(row.get(f"{arm}_calls", "0") or 0)
        latency = float(row.get(f"{arm}_latency_sec", "0") or 0)
        return correct - self.call_penalty * calls - self.latency_penalty * latency

    def best_arm(self, rows: list[dict[str, str]]) -> str:
        if not rows:
            return self.global_label
        return max(
            self.arms,
            key=lambda arm: (
                sum(self.row_reward(row, arm) for row in rows) / len(rows),
                -avg_calls_for_arm(rows, arm),
                arm,
            ),
        )

    def policy_score(self, rows: list[dict[str, str]], predictions: list[str]) -> float:
        return sum(self.row_reward(row, arm) for row, arm in zip(rows, predictions)) / len(rows)

    def fit(self, rows: list[dict[str, str]]) -> None:
        labels = [row[self.label_column] for row in rows]
        self.global_label = self.best_arm(rows) if rows else majority(labels, fallback=labels[0])
        best_score = float("-inf")
        best_rule: dict[str, Any] = {"type": "constant", "label": self.global_label}

        for feature in self.features:
            numeric_values = [(to_float(row.get(feature, "")), row[self.label_column]) for row in rows]
            numeric_present = [(value, label) for value, label in numeric_values if value is not None]
            if len(numeric_present) >= max(5, len(rows) // 10):
                values = sorted(set(value for value, _label in numeric_present))
                if len(values) > 1:
                    thresholds = [
                        values[int((len(values) - 1) * pct)]
                        for pct in (0.1, 0.25, 0.5, 0.75, 0.9)
                    ]
                    for threshold in sorted(set(thresholds)):
                        left_rows = [row for row in rows if (to_float(row.get(feature, "")) or 0.0) <= threshold]
                        right_rows = [row for row in rows if (to_float(row.get(feature, "")) or 0.0) > threshold]
                        left_label = self.best_arm(left_rows)
                        right_label = self.best_arm(right_rows)
                        predictions = [
                            left_label if (to_float(row.get(feature, "")) or 0.0) <= threshold else right_label
                            for row in rows
                        ]
                        score = self.policy_score(rows, predictions)
                        if score > best_score:
                            best_score = score
                            best_rule = {
                                "type": "numeric",
                                "feature": feature,
                                "threshold": threshold,
                                "left_label": left_label,
                                "right_label": right_label,
                                "train_reward": score,
                            }

            values = sorted(set(row.get(feature, "") for row in rows))
            if 1 < len(values) <= 50:
                for value in values:
                    yes_rows = [row for row in rows if row.get(feature, "") == value]
                    no_rows = [row for row in rows if row.get(feature, "") != value]
                    yes_label = self.best_arm(yes_rows)
                    no_label = self.best_arm(no_rows)
                    predictions = [
                        yes_label if row.get(feature, "") == value else no_label
                        for row in rows
                    ]
                    score = self.policy_score(rows, predictions)
                    if score > best_score:
                        best_score = score
                        best_rule = {
                            "type": "categorical",
                            "feature": feature,
                            "value": value,
                            "yes_label": yes_label,
                            "no_label": no_label,
                            "train_reward": score,
                        }
        self.rule = best_rule

    def predict_one(self, row: dict[str, str]) -> str:
        if self.rule.get("type") == "numeric":
            value = to_float(row.get(self.rule["feature"], "")) or 0.0
            return self.rule["left_label"] if value <= self.rule["threshold"] else self.rule["right_label"]
        if self.rule.get("type") == "categorical":
            return self.rule["yes_label"] if row.get(self.rule["feature"], "") == self.rule["value"] else self.rule["no_label"]
        return self.global_label

    def predict(self, rows: list[dict[str, str]]) -> list[str]:
        return [self.predict_one(row) for row in rows]

    def describe(self) -> str:
        if self.rule.get("type") == "numeric":
            return (
                f"{self.rule['feature']} <= {self.rule['threshold']:.3g} -> "
                f"{self.rule['left_label']}; else {self.rule['right_label']}"
            )
        if self.rule.get("type") == "categorical":
            return (
                f"{self.rule['feature']} == {self.rule['value']!r} -> "
                f"{self.rule['yes_label']}; else {self.rule['no_label']}"
            )
        return f"constant -> {self.global_label}"


def best_static_arm(train_rows: list[dict[str, str]], arms: list[str]) -> str:
    return max(arms, key=lambda arm: (accuracy_for_arm(train_rows, arm), -avg_calls_for_arm(train_rows, arm), arm))


def split_random(rows: list[dict[str, str]], test_fraction: float, seed: int) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    n_test = max(1, int(round(len(rows) * test_fraction)))
    return shuffled[n_test:], shuffled[:n_test]


def format_result(name: str, stats: dict[str, Any]) -> str:
    chosen = ", ".join(f"{arm}={count}" for arm, count in sorted(stats["chosen"].items()))
    return (
        f"| {name} | {stats['n']} | {stats['accuracy']*100:.1f}% | "
        f"{stats['label_match']*100:.1f}% | {stats['avg_calls']:.2f} | {chosen} |"
    )


def evaluate_split(
    name: str,
    train_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    arms: list[str],
    label_column: str,
    features: list[str],
    call_penalty: float,
    latency_penalty: float,
) -> tuple[list[str], str]:
    static_arm = best_static_arm(train_rows, arms)
    global_label = majority([row[label_column] for row in train_rows], fallback=static_arm)
    stump = DecisionStump(label_column, features, arms, call_penalty, latency_penalty)
    stump.fit(train_rows)

    lines = [f"### {name}", ""]
    lines.extend([
        "| Policy | N | Accuracy | Oracle-label match | Calls/q | Chosen arms |",
        "|---|---:|---:|---:|---:|---|",
        format_result("static_best_train", evaluate_predictions(test_rows, [static_arm] * len(test_rows), label_column)),
        format_result("majority_oracle_label", evaluate_predictions(test_rows, [global_label] * len(test_rows), label_column)),
        format_result("decision_stump", evaluate_predictions(test_rows, stump.predict(test_rows), label_column)),
        format_result("oracle_label", evaluate_predictions(test_rows, [row[label_column] for row in test_rows], label_column)),
    ])
    lines.extend([
        "",
        f"- static arm from train: `{static_arm}`",
        f"- majority oracle label from train: `{global_label}`",
        f"- decision stump: `{stump.describe()}`",
        "",
    ])
    return lines, stump.describe()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csvs", nargs="+", type=Path)
    parser.add_argument("--label-column", default="oracle_reward_arm", choices=sorted(LABEL_COLUMNS))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--include-dataset-feature", action="store_true")
    parser.add_argument("--include-provider-feature", action="store_true")
    parser.add_argument("--include-subject-feature", action="store_true")
    parser.add_argument("--include-probe-features", action="store_true")
    parser.add_argument("--call-penalty", type=float, default=0.02)
    parser.add_argument("--latency-penalty", type=float, default=0.0)
    args = parser.parse_args()

    rows = load_rows(args.csvs)
    arms = infer_arms(rows)
    features = feature_columns(
        rows,
        include_dataset=args.include_dataset_feature,
        include_provider=args.include_provider_feature,
        include_subject=args.include_subject_feature,
        include_probe_features=args.include_probe_features,
    )

    lines = [
        "# Router Baseline Report",
        "",
        "This report tests whether cheap static task features can predict the oracle method arm.",
        "It is intentionally lightweight: a fixed static arm, a majority-oracle label, and a one-rule decision stump.",
        "",
        f"Rows: `{len(rows)}`",
        f"Arms: `{', '.join(arms)}`",
        f"Label column: `{args.label_column}`",
        f"Include dataset feature: `{args.include_dataset_feature}`",
        f"Include provider feature: `{args.include_provider_feature}`",
        f"Include subject feature: `{args.include_subject_feature}`",
        f"Include retrieval-probe features: `{args.include_probe_features}`",
        f"Policy reward: `correct - {args.call_penalty:g}*calls - {args.latency_penalty:g}*sec`",
        f"Features: `{', '.join(features)}`",
        "",
    ]

    train_rows, test_rows = split_random(rows, args.test_fraction, args.seed)
    split_lines, _ = evaluate_split(
        "Random Split", train_rows, test_rows, arms, args.label_column, features,
        args.call_penalty, args.latency_penalty,
    )
    lines.extend(split_lines)

    datasets = sorted(set(row.get("dataset", "") for row in rows))
    if len(datasets) > 1:
        lines.extend(["## Leave-One-Dataset-Out", ""])
        for dataset in datasets:
            train = [row for row in rows if row.get("dataset", "") != dataset]
            test = [row for row in rows if row.get("dataset", "") == dataset]
            split_lines, _ = evaluate_split(
                f"Hold Out `{dataset}`", train, test, arms, args.label_column, features,
                args.call_penalty, args.latency_penalty,
            )
            lines.extend(split_lines)

    output = "\n".join(lines).rstrip() + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
