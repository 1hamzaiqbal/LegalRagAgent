#!/usr/bin/env python3
"""Audit HousingQA retrieval depth against statute state metadata."""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

_field_limit = sys.maxsize
while True:
    try:
        csv.field_size_limit(_field_limit)
        break
    except OverflowError:
        _field_limit //= 10


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def parse_labeled_path(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        path = resolve_path(raw)
        return path.stem, path
    label, path_raw = raw.split("=", 1)
    label = label.strip()
    if not label:
        raise SystemExit(f"Invalid empty label in {raw!r}")
    return label, resolve_path(path_raw)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if isinstance(value, dict):
                rows.append(value)
    if not rows:
        raise SystemExit(f"{path}: no rows loaded")
    return rows


def norm_state(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def load_statute_states(path: Path) -> dict[str, str]:
    states: dict[str, str] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = str(row.get("idx", "")).strip()
            if idx:
                states[idx] = str(row.get("state", "")).strip()
    if not states:
        raise SystemExit(f"{path}: no idx/state rows loaded")
    return states


def correct(row: dict[str, Any]) -> bool:
    if "is_correct" in row:
        return bool(row.get("is_correct"))
    pred = str(row.get("predicted_answer", "")).strip().lower()
    gold = str(row.get("correct_answer", row.get("answer", ""))).strip().lower()
    return bool(pred and gold and pred == gold)


def retrieved_ids(row: dict[str, Any]) -> list[str]:
    raw = row.get("retrieved_ids")
    if isinstance(raw, list):
        return [str(item) for item in raw if str(item).strip()]
    if isinstance(raw, str) and raw.strip():
        return [raw.strip()]
    evidence = row.get("evidence_store")
    if isinstance(evidence, list):
        return [str(item.get("idx", "")) for item in evidence if isinstance(item, dict) and item.get("idx")]
    return []


def gold_ids(row: dict[str, Any]) -> set[str]:
    raw = row.get("gold_idx")
    if raw is None:
        return set()
    if isinstance(raw, list):
        return {str(item).strip() for item in raw if str(item).strip()}
    return {part.strip() for part in str(raw).split(",") if part.strip()}


def annotate(row: dict[str, Any], statute_states: dict[str, str]) -> dict[str, Any]:
    ids = retrieved_ids(row)
    query_state = norm_state(row.get("state"))
    states = [statute_states.get(idx, "") for idx in ids]
    state_matches = [norm_state(state) == query_state for state in states if state]
    top1_match = bool(state_matches[0]) if state_matches else False
    any_state_match = any(state_matches)
    all_state_match = bool(state_matches) and all(state_matches)
    state_match_frac = sum(state_matches) / len(state_matches) if state_matches else 0.0
    gold = gold_ids(row)
    return {
        "idx": str(row.get("idx")),
        "state": row.get("state", ""),
        "correct": correct(row),
        "retrieved_ids": ids,
        "retrieved_states": states,
        "n_retrieved": len(ids),
        "top1_state_match": top1_match,
        "any_state_match": any_state_match,
        "all_state_match": all_state_match,
        "state_match_frac": state_match_frac,
        "gold_hit": bool(gold & set(ids)) if gold else False,
    }


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(label: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    return {
        "label": label,
        "n": n,
        "accuracy": sum(1 for row in rows if row["correct"]) / n,
        "gold_hit": sum(1 for row in rows if row["gold_hit"]) / n,
        "avg_docs": avg([float(row["n_retrieved"]) for row in rows]),
        "top1_state_match": sum(1 for row in rows if row["top1_state_match"]) / n,
        "any_state_match": sum(1 for row in rows if row["any_state_match"]) / n,
        "all_state_match": sum(1 for row in rows if row["all_state_match"]) / n,
        "avg_state_match_frac": avg([float(row["state_match_frac"]) for row in rows]),
    }


def compare(
    base_label: str,
    treat_label: str,
    base_rows: list[dict[str, Any]],
    treat_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    base = {row["idx"]: row for row in base_rows}
    treat = {row["idx"]: row for row in treat_rows}
    keys = sorted(set(base) & set(treat))
    if not keys:
        raise SystemExit(f"No overlapping idx rows for {base_label} vs {treat_label}")
    rescued = [k for k in keys if not base[k]["correct"] and treat[k]["correct"]]
    harmed = [k for k in keys if base[k]["correct"] and not treat[k]["correct"]]

    def delta(metric: str, selected: list[str]) -> float:
        return avg([float(treat[k][metric]) - float(base[k][metric]) for k in selected])

    def bool_flips(metric: str, selected: list[str]) -> tuple[int, int]:
        false_to_true = sum(1 for k in selected if not base[k][metric] and treat[k][metric])
        true_to_false = sum(1 for k in selected if base[k][metric] and not treat[k][metric])
        return false_to_true, true_to_false

    state_all = delta("state_match_frac", keys)
    state_rescued = delta("state_match_frac", rescued)
    state_harmed = delta("state_match_frac", harmed)
    any_ft, any_tf = bool_flips("any_state_match", keys)
    gold_ft, gold_tf = bool_flips("gold_hit", keys)
    return {
        "base": base_label,
        "treatment": treat_label,
        "n": len(keys),
        "base_acc": sum(1 for k in keys if base[k]["correct"]) / len(keys),
        "treat_acc": sum(1 for k in keys if treat[k]["correct"]) / len(keys),
        "rescued": len(rescued),
        "harmed": len(harmed),
        "state_delta_all": state_all,
        "state_delta_rescued": state_rescued,
        "state_delta_harmed": state_harmed,
        "any_state_false_to_true": any_ft,
        "any_state_true_to_false": any_tf,
        "gold_false_to_true": gold_ft,
        "gold_true_to_false": gold_tf,
    }


def markdown(
    title: str,
    summaries: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    logs: list[tuple[str, Path]],
    statute_path: Path,
) -> str:
    lines = [
        f"# {title}",
        "",
        "Generated offline from HousingQA detail logs and `datasets/housing_qa/statutes.csv`.",
        "",
        "## Headline Read",
        "",
    ]
    by_label = {item["label"]: item for item in summaries}
    if "top1" in by_label and "top10" in by_label:
        acc_delta = (by_label["top10"]["accuracy"] - by_label["top1"]["accuracy"]) * 100
        top1_state = by_label["top1"]["avg_state_match_frac"] * 100
        top10_state = by_label["top10"]["avg_state_match_frac"] * 100
        lines.extend(
            [
                f"- `rag_simple` top-10 improves answer accuracy over top-1 by {acc_delta:+.1f}pp, "
                f"but average same-state retrieval fraction stays tiny: {top1_state:.1f}% at top-1 "
                f"and {top10_state:.1f}% at top-10. That means the top-10 lift is not explained "
                "by a simple jurisdiction repair story.",
            ]
        )
    else:
        lines.append("- Compare accuracy deltas against same-state retrieval deltas before interpreting HousingQA depth effects.")

    if "two_call" in by_label and "top10" in by_label:
        two_call_any = by_label["two_call"]["any_state_match"] * 100
        two_call_frac = by_label["two_call"]["avg_state_match_frac"] * 100
        two_call_acc = by_label["two_call"]["accuracy"] * 100
        top10_acc = by_label["top10"]["accuracy"] * 100
        lines.extend(
            [
                f"- `rag_snap_hyde_2call` retrieves same-state statutes much more often "
                f"(any-state {two_call_any:.1f}%, average state-match fraction {two_call_frac:.1f}%), "
                f"but it scores {two_call_acc:.1f}% versus top-10 `rag_simple` at {top10_acc:.1f}%. "
                "State targeting helps retrieval diagnostics, but it is not sufficient for answer correctness.",
            ]
        )

    lines.extend(
        [
            "- The next method should be explicit state-filtered retrieval or state-aware reranking, not SpecRAG-lite yet.",
            "",
        "## Run-Level Metadata",
        "",
        "| Run | N | Accuracy | Gold hit | Docs/q | Top-1 state match | Any state match | All state match | Avg state-match fraction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in summaries:
        lines.append(
            "| {label} | {n} | {acc} | {gold} | {docs:.1f} | {top1} | {any_state} | {all_state} | {frac} |".format(
                label=item["label"],
                n=item["n"],
                acc=pct(item["accuracy"]),
                gold=pct(item["gold_hit"]),
                docs=item["avg_docs"],
                top1=pct(item["top1_state_match"]),
                any_state=pct(item["any_state_match"]),
                all_state=pct(item["all_state_match"]),
                frac=pct(item["avg_state_match_frac"]),
            )
        )

    lines.extend(
        [
            "",
            "## Paired Metadata Deltas",
            "",
            "| Baseline | Treatment | N | Acc delta | Rescued | Harmed | Avg state-frac delta | Rescued state-frac delta | Harmed state-frac delta | Any-state F->T / T->F | Gold-hit F->T / T->F |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in comparisons:
        lines.append(
            "| {base} | {treatment} | {n} | {delta:+.1f}pp | {rescued} | {harmed} | {state_all:+.1f}pp | {state_rescued:+.1f}pp | {state_harmed:+.1f}pp | {any_ft}/{any_tf} | {gold_ft}/{gold_tf} |".format(
                base=item["base"],
                treatment=item["treatment"],
                n=item["n"],
                delta=(item["treat_acc"] - item["base_acc"]) * 100,
                rescued=item["rescued"],
                harmed=item["harmed"],
                state_all=item["state_delta_all"] * 100,
                state_rescued=item["state_delta_rescued"] * 100,
                state_harmed=item["state_delta_harmed"] * 100,
                any_ft=item["any_state_false_to_true"],
                any_tf=item["any_state_true_to_false"],
                gold_ft=item["gold_false_to_true"],
                gold_tf=item["gold_true_to_false"],
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This audit measures whether retrieved statutes come from the same state as the HousingQA question.",
            "- It does not prove the retrieved statute is legally controlling; same-state retrieval is a metadata sanity signal, not a relevance label.",
            "- If accuracy improves while same-state fraction does not, deeper retrieval is probably adding topical/legal context rather than fixing jurisdiction.",
            "- If rescued rows show large same-state gains, metadata filtering should be tested before heavier draft/verifier methods.",
            "",
            "## Provenance",
            "",
            f"- Statute metadata: `{statute_path.relative_to(REPO_ROOT) if statute_path.is_relative_to(REPO_ROOT) else statute_path}`",
        ]
    )
    for label, path in logs:
        rel = path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path
        lines.append(f"- `{label}`: `{rel}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statutes", default="datasets/housing_qa/statutes.csv")
    parser.add_argument("--log", action="append", required=True, help="label=detail.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default="HousingQA Metadata/Depth Audit - 2026-04-30")
    args = parser.parse_args()

    statute_path = resolve_path(args.statutes)
    statute_states = load_statute_states(statute_path)
    logs = [parse_labeled_path(raw) for raw in args.log]
    annotated: dict[str, list[dict[str, Any]]] = {}
    for label, path in logs:
        annotated[label] = [annotate(row, statute_states) for row in load_jsonl(path)]

    summaries = [summarize(label, annotated[label]) for label, _path in logs]
    desired_pairs = [
        ("top1", "top5"),
        ("top1", "top10"),
        ("top5", "two_call"),
        ("top10", "two_call"),
    ]
    comparisons = [
        compare(a, b, annotated[a], annotated[b])
        for a, b in desired_pairs
        if a in annotated and b in annotated
    ]

    out_path = resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown(args.title, summaries, comparisons, logs, statute_path))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
