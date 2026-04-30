#!/usr/bin/env python3
"""Build a compact evidence matrix from eval detail logs.

This is intended for paper-triage passes where the question is not just
"which run won?", but whether the evidence supports a dataset-level bottleneck
claim: accuracy, retrieval/gold-hit behavior, parse/route health, cost, and
paired deltas.
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import compute_mcnemar  # type: ignore  # noqa: E402


PARSE_FIELD_SUFFIX = "_parse_ok"
KNOWN_PARSE_FIELDS = {
    "adaptive_parse_ok",
    "passage_parse_ok",
    "route_parse_ok",
    "snap_hyde_1call_parse_ok",
    "snap_hyde_2call_parse_ok",
}
ROUTE_FIELDS = ("route_decision", "routed_to")


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


def resolve_log(raw: str) -> tuple[str | None, Path]:
    label: str | None = None
    pattern = raw
    if "=" in raw:
        label, pattern = raw.split("=", 1)
        label = label.strip()
        if not label:
            raise SystemExit(f"Invalid --log label in {raw!r}")
    matches = sorted(glob.glob(pattern))
    if not matches:
        path = Path(pattern)
        if path.exists():
            matches = [str(path)]
    if len(matches) != 1:
        raise SystemExit(f"{raw!r}: expected exactly one log path, matched {len(matches)}")
    return label, Path(matches[0])


def infer_label(path: Path, rows: list[dict[str, Any]]) -> str:
    first = rows[0]
    mode = first.get("mode") or path.stem
    provider = first.get("provider") or "unknown-provider"
    dataset = first.get("dataset") or "unknown-dataset"
    return f"{dataset}:{provider}:{mode}"


def correct_count(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if compute_mcnemar.correct_flag(row))


def pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.1f}%"


def num(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return bool(value)


def sequence_len(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return 1 if value.strip() else 0
    if isinstance(value, (list, tuple, set, dict)):
        return len(value)
    return 1 if value else 0


def evidence_chars(value: Any) -> int:
    if isinstance(value, list):
        total = 0
        for item in value:
            if isinstance(item, dict):
                total += len(str(item.get("text", "")))
            else:
                total += len(str(item))
        return total
    if isinstance(value, dict):
        return sum(len(str(item.get("text", item))) if isinstance(item, dict) else len(str(item)) for item in value.values())
    return len(str(value or ""))


def summarize(label: str, path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    first = rows[0]
    correct = correct_count(rows)
    retrieved_rows = [row for row in rows if nonempty(row.get("evidence_store")) or nonempty(row.get("retrieved_ids"))]
    empty_retrieval_rows = [row for row in rows if "evidence_store" in row and "retrieved_ids" in row and not nonempty(row.get("evidence_store")) and not nonempty(row.get("retrieved_ids"))]
    gold_values = [row.get("gold_retrieved") for row in rows if "gold_retrieved" in row]
    gold_count = sum(1 for value in gold_values if bool(value))
    parse_fields = sorted(
        {
            key
            for row in rows
            for key in row
            if key in KNOWN_PARSE_FIELDS or key.endswith(PARSE_FIELD_SUFFIX)
        }
    )
    parse_summary = []
    for field in parse_fields:
        present = [row for row in rows if field in row]
        ok = sum(1 for row in present if bool(row.get(field)))
        parse_summary.append(f"{field}={ok}/{len(present)}")

    route_summary = []
    for field in ROUTE_FIELDS:
        values = [str(row.get(field)) for row in rows if row.get(field)]
        if values:
            counts = Counter(values)
            route_summary.append(f"{field}: " + ", ".join(f"{key}={counts[key]}" for key in sorted(counts)))

    calls = [float(row.get("llm_calls", 0) or 0) for row in rows]
    latency = [float(row.get("elapsed_sec", 0) or 0) for row in rows if row.get("elapsed_sec") is not None]
    input_tokens = [float(row.get("input_tokens", 0) or 0) for row in rows]
    output_tokens = [float(row.get("output_tokens", 0) or 0) for row in rows]
    evidence_doc_counts = [sequence_len(row.get("evidence_store") or row.get("retrieved_ids")) for row in rows if nonempty(row.get("evidence_store")) or nonempty(row.get("retrieved_ids"))]
    evidence_char_counts = [evidence_chars(row.get("evidence_store")) for row in rows if nonempty(row.get("evidence_store"))]

    return {
        "label": label,
        "path": path,
        "mode": first.get("mode", "-"),
        "provider": first.get("provider", "-"),
        "dataset": first.get("dataset", "-"),
        "n": n,
        "correct": correct,
        "accuracy": correct / n if n else None,
        "gold_rate": gold_count / len(gold_values) if gold_values else None,
        "retrieval_rate": len(retrieved_rows) / n if n else None,
        "empty_retrieval_rate": len(empty_retrieval_rows) / n if n else None,
        "avg_evidence_docs": mean([float(value) for value in evidence_doc_counts]),
        "avg_evidence_chars": mean([float(value) for value in evidence_char_counts]),
        "avg_calls": mean(calls),
        "avg_latency": mean(latency),
        "avg_input_tokens": mean(input_tokens),
        "avg_output_tokens": mean(output_tokens),
        "parse_summary": "; ".join(parse_summary) if parse_summary else "-",
        "route_summary": "; ".join(route_summary) if route_summary else "-",
    }


def parse_pair(raw: str) -> tuple[str, str, str]:
    parts = raw.split(":")
    if len(parts) != 3:
        raise SystemExit(f"Invalid --pair {raw!r}; expected name:baseline_label:treatment_label")
    name, baseline, treatment = (part.strip() for part in parts)
    if not name or not baseline or not treatment:
        raise SystemExit(f"Invalid --pair {raw!r}; labels cannot be empty")
    return name, baseline, treatment


def paired_summary(
    name: str,
    baseline_label: str,
    treatment_label: str,
    rows_by_label: dict[str, list[dict[str, Any]]],
    key: str | None,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    if baseline_label not in rows_by_label:
        raise SystemExit(f"Pair {name!r}: unknown baseline label {baseline_label!r}")
    if treatment_label not in rows_by_label:
        raise SystemExit(f"Pair {name!r}: unknown treatment label {treatment_label!r}")
    baseline_rows = rows_by_label[baseline_label]
    treatment_rows = rows_by_label[treatment_label]
    key_field = compute_mcnemar.choose_key_field(baseline_rows, treatment_rows, key)
    stats = compute_mcnemar.compute(baseline_rows, treatment_rows, key_field, bootstrap_samples, seed)
    return {
        "name": name,
        "baseline": baseline_label,
        "treatment": treatment_label,
        "key": key_field,
        **stats,
    }


def markdown(
    title: str,
    summaries: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    notes: list[str],
) -> str:
    lines: list[str] = [f"# {title}", ""]
    if notes:
        lines.extend(notes)
        lines.append("")

    lines.extend(
        [
            "## Run Matrix",
            "",
            "| Label | Dataset | Provider | Mode | N | Acc | Gold hit | Retrieval rows | Empty retrieval | Calls/q | Sec/q | Evidence docs/q | In tok/q | Out tok/q |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in summaries:
        lines.append(
            "| {label} | {dataset} | {provider} | {mode} | {n} | {correct}/{n} ({acc}) | {gold} | {retrieval} | {empty} | {calls} | {sec} | {docs} | {input_tok} | {output_tok} |".format(
                label=item["label"],
                dataset=item["dataset"],
                provider=item["provider"],
                mode=item["mode"],
                n=item["n"],
                correct=item["correct"],
                acc=pct(item["accuracy"]),
                gold=pct(item["gold_rate"]),
                retrieval=pct(item["retrieval_rate"]),
                empty=pct(item["empty_retrieval_rate"]),
                calls=num(item["avg_calls"], 2),
                sec=num(item["avg_latency"], 1),
                docs=num(item["avg_evidence_docs"], 1),
                input_tok=num(item["avg_input_tokens"], 0),
                output_tok=num(item["avg_output_tokens"], 0),
            )
        )

    if pairs:
        lines.extend(
            [
                "",
                "## Paired Deltas",
                "",
                "| Pair | Baseline | Treatment | Key | N | Baseline acc | Treatment acc | Delta | b/c | McNemar p | 95% bootstrap CI |",
                "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for item in pairs:
            lines.append(
                "| {name} | {baseline} | {treatment} | {key} | {n} | {base} | {treat} | {delta:+.1f}pp | {b}/{c} | {p:.4g} | [{lo:+.1f}, {hi:+.1f}] pp |".format(
                    name=item["name"],
                    baseline=item["baseline"],
                    treatment=item["treatment"],
                    key=item["key"],
                    n=item["n_paired"],
                    base=pct(float(item["acc_baseline"])),
                    treat=pct(float(item["acc_treatment"])),
                    delta=float(item["delta_pp"]),
                    b=int(item["b"]),
                    c=int(item["c"]),
                    p=float(item["mcnemar_p"]),
                    lo=float(item["ci_low"]),
                    hi=float(item["ci_high"]),
                )
            )

    lines.extend(["", "## Parse And Route Health", ""])
    for item in summaries:
        lines.append(f"- `{item['label']}`: parse `{item['parse_summary']}`; route `{item['route_summary']}`")

    lines.extend(["", "## Source Logs", ""])
    for item in summaries:
        lines.append(f"- `{item['label']}`: `{item['path']}`")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log",
        action="append",
        required=True,
        help="Detail log path/glob, optionally label=path. Labels are used by --pair.",
    )
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        help="Paired comparison in the form name:baseline_label:treatment_label.",
    )
    parser.add_argument("--key", help="Override paired join key field")
    parser.add_argument("--title", default="Evidence Matrix")
    parser.add_argument("--note", action="append", default=[], help="Markdown note paragraph to include near the top")
    parser.add_argument("--output", type=Path, help="Write markdown to this path instead of stdout")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows_by_label: dict[str, list[dict[str, Any]]] = {}
    summaries: list[dict[str, Any]] = []
    for raw in args.log:
        requested_label, path = resolve_log(raw)
        rows = load_jsonl(path)
        label = requested_label or infer_label(path, rows)
        if label in rows_by_label:
            raise SystemExit(f"Duplicate log label {label!r}")
        rows_by_label[label] = rows
        summaries.append(summarize(label, path, rows))

    pair_rows = [
        paired_summary(name, baseline, treatment, rows_by_label, args.key, args.bootstrap_samples, args.seed)
        for name, baseline, treatment in (parse_pair(raw) for raw in args.pair)
    ]
    output = markdown(args.title, summaries, pair_rows, args.note)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
