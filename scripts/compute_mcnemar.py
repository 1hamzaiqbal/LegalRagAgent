#!/usr/bin/env python3
"""Paired McNemar test for two evaluation detail JSONL logs."""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import string
import sys
from pathlib import Path
from typing import Any

try:
    from scipy.stats import binomtest
except ImportError as exc:
    raise SystemExit(
        "scipy is required for the exact McNemar binomial test. "
        "Install it with `uv add scipy` and rerun."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "eval"))

try:
    import eval_config  # type: ignore
except ImportError:
    eval_config = None  # type: ignore

_EXTRACT_ANSWER = getattr(eval_config, "_extract_answer", None)
_NORMALIZE_ANSWER = getattr(eval_config, "_normalize_answer", None)
_EXTRACT_ANSWER_MUSIQUE = getattr(eval_config, "extract_answer_musique", None)
_MUSIQUE_EM_F1 = getattr(eval_config, "musique_em_f1", None)

KEY_CANDIDATES = ("record_id", "question_id", "idx", "label", "question")


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


def choose_key_field(
    baseline_rows: list[dict[str, Any]],
    treatment_rows: list[dict[str, Any]],
    requested: str | None,
) -> str:
    candidates = (requested,) if requested else KEY_CANDIDATES
    for key in candidates:
        if not key:
            continue
        if key not in baseline_rows[0] or key not in treatment_rows[0]:
            continue
        base_values = [row.get(key) for row in baseline_rows]
        treat_values = [row.get(key) for row in treatment_rows]
        if any(value is None for value in base_values + treat_values):
            continue
        if len(set(base_values)) != len(base_values):
            continue
        if len(set(treat_values)) != len(treat_values):
            continue
        if set(base_values) & set(treat_values):
            return key
    tried = ", ".join(candidates)
    raise SystemExit(f"Could not find a unique overlapping join key. Tried: {tried}")


def load_experiment_metadata() -> dict[Path, list[dict[str, Any]]]:
    path = REPO_ROOT / "logs" / "experiments.jsonl"
    metadata: dict[Path, list[dict[str, Any]]] = {}
    if not path.exists():
        return metadata
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            detail_log = row.get("detail_log")
            if not detail_log:
                continue
            detail_path = (REPO_ROOT / detail_log).resolve()
            metadata.setdefault(detail_path, []).append(row)
    return metadata


def validate_expected_tag(
    path: Path,
    rows: list[dict[str, Any]],
    expected: str | None,
    metadata: dict[Path, list[dict[str, Any]]],
) -> None:
    if not expected:
        return

    row_tags = [str(row.get("tag", "")) for row in rows if row.get("tag")]
    if row_tags and any(expected in tag for tag in row_tags):
        return

    meta_rows = metadata.get(path.resolve(), [])
    meta_tags = [str(row.get("tag", "")) for row in meta_rows if row.get("tag")]
    if meta_tags and any(expected in tag for tag in meta_tags):
        return

    seen = sorted(set(row_tags + meta_tags))
    detail = ", ".join(seen) if seen else "no tag found in detail rows or experiments metadata"
    raise SystemExit(f"{path}: expected tag containing {expected!r}; saw {detail}")


def parse_aliases(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(item) for item in raw]
    if isinstance(raw, str):
        if not raw.strip():
            return []
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return [raw]
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
        return [str(parsed)]
    return [str(raw)]


def fallback_normalize(value: Any) -> str:
    text = str(value or "").lower().strip()
    table = str.maketrans("", "", string.punctuation)
    text = text.translate(table)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_answer(value: Any) -> str:
    if _NORMALIZE_ANSWER is not None:
        return str(_NORMALIZE_ANSWER(str(value or "")))
    return fallback_normalize(value)


def extract_prediction(record: dict[str, Any]) -> str:
    predicted = record.get("predicted_answer")
    if predicted is not None:
        return str(predicted)

    final_answer = record.get("final_answer", "")
    if _EXTRACT_ANSWER is not None:
        try:
            extracted = _EXTRACT_ANSWER(str(final_answer))
            return "" if extracted is None else str(extracted)
        except TypeError:
            pass

    if record.get("dataset") == "musique" and _EXTRACT_ANSWER_MUSIQUE is not None:
        return str(_EXTRACT_ANSWER_MUSIQUE(str(final_answer)))
    return str(final_answer or "")


def correct_flag(record: dict[str, Any]) -> bool:
    gold = record.get("correct_answer")
    if gold is None:
        gold = record.get("answer")
    if gold is None:
        raise SystemExit("Record is missing correct_answer/answer; cannot score")

    predicted = extract_prediction(record)
    if record.get("dataset") == "musique" and _MUSIQUE_EM_F1 is not None:
        aliases = parse_aliases(record.get("aliases_used", record.get("answer_aliases")))
        em, _f1 = _MUSIQUE_EM_F1(predicted, str(gold), aliases)
        return bool(em)

    return normalize_answer(predicted) == normalize_answer(gold)


def warn_stored_mismatches(path: Path, rows: list[dict[str, Any]]) -> None:
    mismatches = 0
    for row in rows:
        if "is_correct" in row and bool(row["is_correct"]) != correct_flag(row):
            mismatches += 1
    if mismatches:
        print(
            f"[warn] {path}: recomputed correctness differs from stored is_correct "
            f"on {mismatches}/{len(rows)} rows",
            file=sys.stderr,
        )


def percentile(values: list[float], pct: float) -> float:
    if not values:
        raise ValueError("cannot take percentile of empty list")
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    position = (len(ordered) - 1) * pct / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_ci(
    paired_diffs: list[int],
    samples: int,
    seed: int,
) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(paired_diffs)
    draws: list[float] = []
    for _ in range(samples):
        total = 0
        for _ in range(n):
            total += paired_diffs[rng.randrange(n)]
        draws.append(total / n * 100.0)
    return percentile(draws, 2.5), percentile(draws, 97.5)


def compute(
    baseline_rows: list[dict[str, Any]],
    treatment_rows: list[dict[str, Any]],
    key_field: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, float | int]:
    baseline_by_key = {row[key_field]: row for row in baseline_rows}
    treatment_by_key = {row[key_field]: row for row in treatment_rows}
    common_keys = sorted(set(baseline_by_key) & set(treatment_by_key))
    if not common_keys:
        raise SystemExit(f"No paired rows found on key {key_field!r}")

    baseline_correct: list[bool] = []
    treatment_correct: list[bool] = []
    for key in common_keys:
        baseline_correct.append(correct_flag(baseline_by_key[key]))
        treatment_correct.append(correct_flag(treatment_by_key[key]))

    b = sum(1 for base, treat in zip(baseline_correct, treatment_correct) if treat and not base)
    c = sum(1 for base, treat in zip(baseline_correct, treatment_correct) if base and not treat)
    discordant = b + c
    p_value = 1.0 if discordant == 0 else binomtest(
        min(b, c),
        n=discordant,
        p=0.5,
        alternative="two-sided",
    ).pvalue

    paired_diffs = [int(treat) - int(base) for base, treat in zip(baseline_correct, treatment_correct)]
    ci_low, ci_high = bootstrap_ci(paired_diffs, bootstrap_samples, seed)
    n_paired = len(common_keys)
    acc_baseline = sum(baseline_correct) / n_paired
    acc_treatment = sum(treatment_correct) / n_paired

    return {
        "n_paired": n_paired,
        "acc_baseline": acc_baseline,
        "acc_treatment": acc_treatment,
        "delta_pp": (acc_treatment - acc_baseline) * 100.0,
        "b": b,
        "c": c,
        "mcnemar_p": p_value,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="Baseline detail JSONL path")
    parser.add_argument("treatment", type=Path, help="Treatment detail JSONL path")
    parser.add_argument("--key", help="Override join key field")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expect-tag", help="Require both logs to have a tag containing this string")
    parser.add_argument("--expect-baseline-tag", help="Require baseline log tag containing this string")
    parser.add_argument("--expect-treatment-tag", help="Require treatment log tag containing this string")
    args = parser.parse_args()

    baseline_rows = load_jsonl(args.baseline)
    treatment_rows = load_jsonl(args.treatment)
    metadata = load_experiment_metadata()
    validate_expected_tag(
        args.baseline,
        baseline_rows,
        args.expect_baseline_tag or args.expect_tag,
        metadata,
    )
    validate_expected_tag(
        args.treatment,
        treatment_rows,
        args.expect_treatment_tag or args.expect_tag,
        metadata,
    )

    warn_stored_mismatches(args.baseline, baseline_rows)
    warn_stored_mismatches(args.treatment, treatment_rows)

    key_field = choose_key_field(baseline_rows, treatment_rows, args.key)
    results = compute(
        baseline_rows,
        treatment_rows,
        key_field,
        args.bootstrap_samples,
        args.seed,
    )
    for name in (
        "n_paired",
        "acc_baseline",
        "acc_treatment",
        "delta_pp",
        "b",
        "c",
        "mcnemar_p",
        "ci_low",
        "ci_high",
    ):
        value = results[name]
        if isinstance(value, int):
            print(f"{name}={value}")
        else:
            print(f"{name}={value:.10g}")


if __name__ == "__main__":
    main()
