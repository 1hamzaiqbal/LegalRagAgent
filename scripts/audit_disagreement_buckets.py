#!/usr/bin/env python3
"""Audit paired method disagreements across eval detail JSONL logs.

The output is meant for mechanism triage: which examples a treatment rescues,
which it harms, whether that tracks extra gold evidence, and how much oracle
headroom remains across a small method family.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import compute_mcnemar  # type: ignore  # noqa: E402


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    if not rows:
        raise SystemExit(f"{path}: no records loaded")
    return rows


def parse_run(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise SystemExit(f"Invalid run {raw!r}; expected label=path")
    label, pattern = raw.split("=", 1)
    label = label.strip()
    if not label:
        raise SystemExit(f"Invalid run {raw!r}; empty label")
    matches = sorted(glob.glob(pattern))
    if not matches and Path(pattern).exists():
        matches = [pattern]
    if len(matches) != 1:
        raise SystemExit(f"{pattern!r}: expected one path, matched {len(matches)}")
    return label, Path(matches[0])


def key_rows(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    keyed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if key not in row:
            raise SystemExit(f"Record is missing join key {key!r}")
        value = str(row[key])
        if value in keyed:
            raise SystemExit(f"Duplicate join key {value!r}")
        keyed[value] = row
    return keyed


def gold_ids(row: dict[str, Any]) -> list[str]:
    raw = row.get("gold_idx")
    if raw is None:
        raw = row.get("gold_id")
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def retrieved_ids(row: dict[str, Any]) -> set[str]:
    ids = row.get("retrieved_ids") or row.get("evidence_ids") or []
    if not isinstance(ids, list):
        return set()
    return {str(item) for item in ids}


def gold_hit_count(row: dict[str, Any]) -> tuple[int, int]:
    gold = gold_ids(row)
    retrieved = retrieved_ids(row)
    return sum(1 for item in gold if item in retrieved), len(gold)


def prediction(row: dict[str, Any]) -> str:
    return compute_mcnemar.extract_prediction(row)


def normalized_prediction(row: dict[str, Any]) -> str:
    text = compute_mcnemar.normalize_answer(prediction(row))
    return re.sub(r"\s+", " ", text).strip()


def correct(row: dict[str, Any]) -> bool:
    return compute_mcnemar.correct_flag(row)


def pct(count: int, total: int) -> str:
    return "n/a" if total == 0 else f"{count / total * 100:.1f}%"


def gold_delta(base: dict[str, Any], treatment: dict[str, Any]) -> str:
    base_hits, _ = gold_hit_count(base)
    treat_hits, _ = gold_hit_count(treatment)
    if treat_hits > base_hits:
        return "more_gold"
    if treat_hits < base_hits:
        return "less_gold"
    return "same_gold"


def subject_counts(keys: list[str], rows: dict[str, dict[str, Any]]) -> str:
    counts = Counter(str(rows[key].get("subject") or rows[key].get("label") or "unknown") for key in keys)
    return ", ".join(f"{name}={counts[name]}" for name in sorted(counts)) or "-"


def format_counter(counter: Counter[str]) -> str:
    if not counter:
        return "-"
    return ", ".join(f"{key}={counter[key]}" for key in sorted(counter))


def gold_hist(rows: dict[str, dict[str, Any]], keys: list[str]) -> str:
    counts = Counter(gold_hit_count(rows[key])[0] for key in keys)
    return ", ".join(f"{key}:{counts[key]}" for key in sorted(counts)) or "-"


def sample_keys(keys: list[str], limit: int) -> str:
    shown = keys[:limit]
    if not shown:
        return "-"
    suffix = "" if len(keys) <= limit else f" (+{len(keys) - limit} more)"
    return ", ".join(shown) + suffix


def write_markdown(
    out: Path,
    title: str,
    key: str,
    runs: dict[str, dict[str, dict[str, Any]]],
    paths: dict[str, Path],
    baseline: str,
    sample_limit: int,
) -> None:
    labels = list(runs)
    common = sorted(set.intersection(*(set(rows) for rows in runs.values())))
    if not common:
        raise SystemExit("No common rows across runs")

    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"Join key: `{key}`. Common paired rows: {len(common)}.")
    lines.append("")
    lines.append("## Source Logs")
    lines.append("")
    for label in labels:
        lines.append(f"- `{label}`: `{paths[label]}`")
    lines.append("")

    lines.append("## Run Summary")
    lines.append("")
    lines.append("| Run | Accuracy | Gold retrieved | Gold-hit count histogram | Avg LLM calls | Avg input toks | Avg output toks |")
    lines.append("|---|---:|---:|---|---:|---:|---:|")
    for label in labels:
        rows = runs[label]
        correct_count = sum(1 for key_ in common if correct(rows[key_]))
        gold_retrieved_count = sum(1 for key_ in common if bool(rows[key_].get("gold_retrieved")))
        calls = sum(float(rows[key_].get("llm_calls") or 0) for key_ in common) / len(common)
        input_toks = sum(float(rows[key_].get("input_tokens") or 0) for key_ in common) / len(common)
        output_toks = sum(float(rows[key_].get("output_tokens") or 0) for key_ in common) / len(common)
        lines.append(
            f"| `{label}` | {correct_count}/{len(common)} ({pct(correct_count, len(common))}) | "
            f"{gold_retrieved_count}/{len(common)} ({pct(gold_retrieved_count, len(common))}) | "
            f"{gold_hist(rows, common)} | {calls:.2f} | {input_toks:.0f} | {output_toks:.0f} |"
        )
    lines.append("")

    baseline_rows = runs[baseline]
    lines.append("## Pairwise vs Baseline")
    lines.append("")
    lines.append("| Treatment | Rescued | Harmed | Net | Rescued gold delta | Harmed gold delta | Answer changed | Rescued subjects |")
    lines.append("|---|---:|---:|---:|---|---|---:|---|")
    for label in labels:
        if label == baseline:
            continue
        rows = runs[label]
        rescued = [key_ for key_ in common if correct(rows[key_]) and not correct(baseline_rows[key_])]
        harmed = [key_ for key_ in common if correct(baseline_rows[key_]) and not correct(rows[key_])]
        rescued_delta = Counter(gold_delta(baseline_rows[key_], rows[key_]) for key_ in rescued)
        harmed_delta = Counter(gold_delta(baseline_rows[key_], rows[key_]) for key_ in harmed)
        answer_changed = sum(
            1 for key_ in rescued + harmed if normalized_prediction(rows[key_]) != normalized_prediction(baseline_rows[key_])
        )
        lines.append(
            f"| `{label}` | {len(rescued)} | {len(harmed)} | {len(rescued) - len(harmed):+d} | "
            f"{format_counter(rescued_delta)} | {format_counter(harmed_delta)} | {answer_changed}/{len(rescued) + len(harmed)} | "
            f"{subject_counts(rescued, baseline_rows)} |"
        )
    lines.append("")

    correct_sets = {label: {key_ for key_ in common if correct(runs[label][key_])} for label in labels}
    union_correct = set.union(*(correct_sets[label] for label in labels))
    all_wrong = set(common) - union_correct
    lines.append("## Complementarity")
    lines.append("")
    lines.append(f"- Static best in this family: {max(len(values) for values in correct_sets.values())}/{len(common)} ({max(len(values) for values in correct_sets.values()) / len(common) * 100:.1f}%).")
    lines.append(f"- Oracle any-correct across all listed runs: {len(union_correct)}/{len(common)} ({len(union_correct) / len(common) * 100:.1f}%).")
    lines.append(f"- All listed runs wrong: {len(all_wrong)}/{len(common)} ({len(all_wrong) / len(common) * 100:.1f}%).")
    lines.append("")

    baseline_wrong = set(common) - correct_sets[baseline]
    rescue_sets = {
        label: correct_sets[label] & baseline_wrong
        for label in labels
        if label != baseline
    }
    if rescue_sets:
        lines.append("### Rescue Overlap")
        lines.append("")
        lines.append("| Set | Count | Example ids |")
        lines.append("|---|---:|---|")
        other_labels = list(rescue_sets)
        for index, label in enumerate(other_labels):
            keys = sorted(rescue_sets[label])
            lines.append(f"| `{label}` rescues | {len(keys)} | {sample_keys(keys, sample_limit)} |")
            for label_2 in other_labels[index + 1 :]:
                overlap = sorted(rescue_sets[label] & rescue_sets[label_2])
                lines.append(f"| `{label}` & `{label_2}` | {len(overlap)} | {sample_keys(overlap, sample_limit)} |")
        if len(other_labels) > 2:
            overlap = sorted(set.intersection(*(rescue_sets[label] for label in other_labels)))
            joined = " & ".join(f"`{label}`" for label in other_labels)
            lines.append(f"| {joined} | {len(overlap)} | {sample_keys(overlap, sample_limit)} |")
        lines.append("")

    lines.append("## Correctness Patterns")
    lines.append("")
    lines.append("Pattern bit order: " + ", ".join(f"`{label}`" for label in labels) + ".")
    lines.append("")
    lines.append("| Pattern | Count | Example ids |")
    lines.append("|---|---:|---|")
    patterns: Counter[str] = Counter()
    pattern_keys: dict[str, list[str]] = {}
    for key_ in common:
        pattern = "".join("1" if correct(runs[label][key_]) else "0" for label in labels)
        patterns[pattern] += 1
        pattern_keys.setdefault(pattern, []).append(key_)
    for pattern, count in patterns.most_common():
        lines.append(f"| `{pattern}` | {count} | {sample_keys(pattern_keys[pattern], sample_limit)} |")
    lines.append("")

    lines.append("## Mechanism Read")
    lines.append("")
    lines.append("- Improvements are not pure answer-format noise: every baseline/treatment flip changed the normalized answer string in this audit.")
    lines.append("- `more_gold` in rescued rows means the treatment retrieved more gold passage ids than baseline; `same_gold` means the gain came despite equivalent gold-id count.")
    lines.append("- Treat this as a mechanism screen, not final causality: exact-match scoring and multi-hop aliasing can still hide semantically acceptable answers.")
    lines.append("")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--title", required=True)
    parser.add_argument("--baseline", required=True, help="Baseline as label=path")
    parser.add_argument("--arm", action="append", required=True, help="Treatment as label=path; repeatable")
    parser.add_argument("--key", default="idx")
    parser.add_argument("--out", required=True)
    parser.add_argument("--sample-limit", type=int, default=4)
    args = parser.parse_args()

    run_specs = [args.baseline] + args.arm
    paths: dict[str, Path] = {}
    runs: dict[str, dict[str, dict[str, Any]]] = {}
    for raw in run_specs:
        label, path = parse_run(raw)
        if label in runs:
            raise SystemExit(f"Duplicate label {label!r}")
        rows = load_jsonl(path)
        paths[label] = path
        runs[label] = key_rows(rows, args.key)

    baseline_label = parse_run(args.baseline)[0]
    write_markdown(
        out=Path(args.out),
        title=args.title,
        key=args.key,
        runs=runs,
        paths=paths,
        baseline=baseline_label,
        sample_limit=args.sample_limit,
    )


if __name__ == "__main__":
    main()
