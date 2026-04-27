#!/usr/bin/env python3
"""Analyze whether friend/foe attribution changes answer outcomes."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "docs" / "friend_foe_bias_analysis_2026-04-27.md"

LOGS = [
    {
        "model": "Gemma 3 27B",
        "provider": "or-gemma27b",
        "reported_accuracy": "10.0%",
        "path": ROOT
        / "logs"
        / "eval_friend_foe_attribution_or-gemma27b_20260427_0249_detail.jsonl",
    },
    {
        "model": "Llama 70B",
        "provider": "groq-llama70b",
        "reported_accuracy": "13.3%",
        "path": ROOT
        / "logs"
        / "eval_friend_foe_attribution_groq-llama70b_20260427_0305_detail.jsonl",
    },
]

ANSWER_RE = re.compile(r"Answer:\s*(.+?)(?:\n|$)", re.IGNORECASE)


def normalize_answer(answer: str | None) -> str:
    """Lowercase, remove punctuation, and normalize whitespace for comparisons."""
    if not answer:
        return ""
    lowered = answer.lower().strip()
    no_punct = "".join(
        ch for ch in lowered if not unicodedata.category(ch).startswith("P")
    )
    return re.sub(r"\s+", " ", no_punct).strip()


def extract_answer(value: Any, *, fallback_raw: bool = False) -> dict[str, str]:
    """Extract the last Answer: line from a model response."""
    if value is None:
        return {"raw": "", "norm": "", "line": ""}

    text = str(value)
    matches = list(ANSWER_RE.finditer(text))
    if matches:
        match = matches[-1]
        raw = match.group(1).strip()
        line = match.group(0).rstrip("\n")
    elif fallback_raw:
        raw = text.strip()
        line = f"Answer: {raw}" if raw else ""
    else:
        raw = ""
        line = ""

    return {"raw": raw, "norm": normalize_answer(raw), "line": line}


def kept_snap(snap_norm: str, review_norm: str) -> bool | None:
    if not snap_norm or not review_norm:
        return None
    return snap_norm == review_norm


def pct(num: int, den: int) -> float:
    return (100.0 * num / den) if den else 0.0


def fmt_pct(num: int, den: int) -> str:
    return f"{num}/{den} ({pct(num, den):.1f}%)"


def md_cell(value: Any) -> str:
    text = str(value)
    return text.replace("\n", "<br>").replace("|", "\\|")


def short_answer_line(line: str, max_len: int = 180) -> str:
    if len(line) <= max_len:
        return line
    return line[: max_len - 3].rstrip() + "..."


def load_log(config: dict[str, Any]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    with config["path"].open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            record_id = raw.get("label") or raw.get("idx") or f"line_{line_no}"

            snap = extract_answer(raw.get("snap_answer"))
            self_review = extract_answer(raw.get("self_review_answer"))
            foe_review = extract_answer(raw.get("foe_review_answer"))
            control_review = extract_answer(raw.get("control_review_answer"))
            predicted = extract_answer(raw.get("predicted_answer"), fallback_raw=True)

            outcomes = [
                self_review["norm"],
                foe_review["norm"],
                control_review["norm"],
            ]
            attribution_changed = (
                all(outcomes) and len(set(outcomes)) > 1
            )

            records.append(
                {
                    "record_id": record_id,
                    "line_no": line_no,
                    "question": raw.get("question", ""),
                    "correct_answer": raw.get("correct_answer", ""),
                    "provider": raw.get("provider", config["provider"]),
                    "foe_label": raw.get("foe_label", ""),
                    "snap": snap,
                    "self": self_review,
                    "foe": foe_review,
                    "control": control_review,
                    "predicted": predicted,
                    "self_kept_snap": kept_snap(snap["norm"], self_review["norm"]),
                    "foe_kept_snap": kept_snap(snap["norm"], foe_review["norm"]),
                    "control_kept_snap": kept_snap(
                        snap["norm"], control_review["norm"]
                    ),
                    "attribution_changed_outcome": attribution_changed,
                    "is_correct": raw.get("is_correct"),
                }
            )

    summary: dict[str, Any] = {
        "model": config["model"],
        "provider": config["provider"],
        "reported_accuracy": config["reported_accuracy"],
        "path": config["path"],
        "records": records,
        "n_records": len(records),
    }
    for key in ("self_kept_snap", "foe_kept_snap", "control_kept_snap"):
        values = [record[key] for record in records]
        summary[f"{key}_true"] = sum(value is True for value in values)
        summary[f"{key}_false"] = sum(value is False for value in values)
        summary[f"{key}_missing"] = sum(value is None for value in values)
        summary[f"{key}_den"] = sum(value is not None for value in values)

    changed_records = [
        record for record in records if record["attribution_changed_outcome"]
    ]
    summary["changed_records"] = changed_records
    summary["n_attribution_changed_outcome"] = len(changed_records)
    return summary


def build_log_table(summary: dict[str, Any]) -> str:
    rows = [
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Records | {summary['n_records']} |",
        f"| Reported run accuracy | {summary['reported_accuracy']} |",
        "| Self kept snap | "
        f"{fmt_pct(summary['self_kept_snap_true'], summary['self_kept_snap_den'])} |",
        "| Foe kept snap | "
        f"{fmt_pct(summary['foe_kept_snap_true'], summary['foe_kept_snap_den'])} |",
        "| Control kept snap | "
        f"{fmt_pct(summary['control_kept_snap_true'], summary['control_kept_snap_den'])} |",
        "| Attribution changed outcome | "
        f"{fmt_pct(summary['n_attribution_changed_outcome'], summary['n_records'])} |",
    ]
    return "\n".join(rows)


def build_changed_table(summary: dict[str, Any]) -> str:
    if not summary["changed_records"]:
        return "_No records had different self/foe/control answer strings._"

    rows = [
        "| Record ID | Self answer | Foe answer | Control answer | Predicted final |",
        "| --- | --- | --- | --- | --- |",
    ]
    for record in summary["changed_records"]:
        rows.append(
            "| "
            + " | ".join(
                [
                    md_cell(record["record_id"]),
                    md_cell(record["self"]["raw"]),
                    md_cell(record["foe"]["raw"]),
                    md_cell(record["control"]["raw"]),
                    md_cell(record["predicted"]["raw"]),
                ]
            )
            + " |"
        )
    return "\n".join(rows)


def comparison_text(summaries: list[dict[str, Any]]) -> str:
    left, right = summaries
    left_rate = pct(left["n_attribution_changed_outcome"], left["n_records"])
    right_rate = pct(right["n_attribution_changed_outcome"], right["n_records"])
    diff = right_rate - left_rate

    if diff > 0:
        direction = (
            f"{right['model']} shows more substantive attribution sensitivity "
            f"than {left['model']} by {diff:.1f} percentage points."
        )
    elif diff < 0:
        direction = (
            f"{left['model']} shows more substantive attribution sensitivity "
            f"than {right['model']} by {abs(diff):.1f} percentage points."
        )
    else:
        direction = (
            f"{left['model']} and {right['model']} show the same substantive "
            "attribution-change rate."
        )

    keep_rows = [
        "| Model | Self kept snap | Foe kept snap | Control kept snap | Changed outcomes |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        keep_rows.append(
            "| "
            + " | ".join(
                [
                    summary["model"],
                    fmt_pct(
                        summary["self_kept_snap_true"],
                        summary["self_kept_snap_den"],
                    ),
                    fmt_pct(
                        summary["foe_kept_snap_true"],
                        summary["foe_kept_snap_den"],
                    ),
                    fmt_pct(
                        summary["control_kept_snap_true"],
                        summary["control_kept_snap_den"],
                    ),
                    fmt_pct(
                        summary["n_attribution_changed_outcome"],
                        summary["n_records"],
                    ),
                ]
            )
            + " |"
        )

    return "\n".join(keep_rows) + "\n\n" + direction


def example_blocks(summaries: list[dict[str, Any]], limit: int = 5) -> str:
    changed = []
    for summary in summaries:
        for record in summary["changed_records"]:
            changed.append((summary, record))

    if not changed:
        return "_No concrete changed-answer examples exist in these two logs._"

    blocks: list[str] = []
    for summary, record in changed[:limit]:
        lines = [
            f"### {md_cell(record['record_id'])} ({summary['model']})",
            "",
            f"- Question: {md_cell(record['question'])}",
            f"- Correct answer: {md_cell(record['correct_answer'])}",
            "",
            "```text",
            f"snap_answer: {short_answer_line(record['snap']['line'])}",
            f"self_review_answer: {short_answer_line(record['self']['line'])}",
            f"foe_review_answer: {short_answer_line(record['foe']['line'])}",
            f"control_review_answer: {short_answer_line(record['control']['line'])}",
            f"predicted_answer: {short_answer_line(record['predicted']['line'])}",
            "```",
        ]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def verdict_text(summaries: list[dict[str, Any]]) -> str:
    total_changed = sum(summary["n_attribution_changed_outcome"] for summary in summaries)
    total_records = sum(summary["n_records"] for summary in summaries)
    if total_changed:
        return (
            f"**Verdict: REAL, but limited in frequency.** Attribution changed "
            f"the final answer string on {total_changed}/{total_records} records "
            f"({pct(total_changed, total_records):.1f}%). Because these are answer-string "
            "changes across self/foe/control review passes, the observed effect is "
            "not merely tonal in these logs."
        )
    return (
        f"**Verdict: SHALLOW.** Attribution changed the wording/tone but did not "
        f"change any final answer string across {total_records} records."
    )


def build_markdown(summaries: list[dict[str, Any]]) -> str:
    lines: list[str] = [
        "# Friend/Foe Attribution Bias Analysis",
        "",
        "Inputs:",
    ]
    for summary in summaries:
        lines.append(f"- `{summary['path'].relative_to(ROOT)}` ({summary['model']})")

    lines.extend(["", "## Per-log summary tables", ""])
    for summary in summaries:
        lines.extend(
            [
                f"### {summary['model']} ({summary['provider']})",
                "",
                build_log_table(summary),
                "",
                "Changed-outcome records:",
                "",
                build_changed_table(summary),
                "",
            ]
        )

    lines.extend(
        [
            "## Cross-model comparison",
            "",
            comparison_text(summaries),
            "",
            "## Concrete examples",
            "",
            example_blocks(summaries),
            "",
            "## Verdict: REAL vs SHALLOW",
            "",
            verdict_text(summaries),
            "",
        ]
    )
    return "\n".join(lines)


def print_stdout_summary(summaries: list[dict[str, Any]]) -> None:
    total_changed = sum(summary["n_attribution_changed_outcome"] for summary in summaries)
    total_records = sum(summary["n_records"] for summary in summaries)
    verdict = "REAL" if total_changed else "SHALLOW"

    print(f"Wrote {OUT_PATH.relative_to(ROOT)}")
    for summary in summaries:
        print(
            f"{summary['model']}: self {pct(summary['self_kept_snap_true'], summary['self_kept_snap_den']):.1f}%, "
            f"foe {pct(summary['foe_kept_snap_true'], summary['foe_kept_snap_den']):.1f}%, "
            f"control {pct(summary['control_kept_snap_true'], summary['control_kept_snap_den']):.1f}%; "
            f"changed outcomes {summary['n_attribution_changed_outcome']}/{summary['n_records']}."
        )

    left, right = summaries
    diff = pct(right["n_attribution_changed_outcome"], right["n_records"]) - pct(
        left["n_attribution_changed_outcome"], left["n_records"]
    )
    print(
        f"Model difference: {right['model']} minus {left['model']} changed-outcome rate = {diff:.1f} pp."
    )
    print(f"Verdict: {verdict} with {total_changed}/{total_records} substantive answer-string changes.")

    examples = [
        (summary, record)
        for summary in summaries
        for record in summary["changed_records"]
    ][:2]
    for summary, record in examples:
        print(
            f"Example {record['record_id']} ({summary['model']}): "
            f"self '{record['self']['line']}', foe '{record['foe']['line']}', "
            f"control '{record['control']['line']}'."
        )


def main() -> int:
    summaries = [load_log(config) for config in LOGS]
    markdown = build_markdown(summaries)
    OUT_PATH.write_text(markdown, encoding="utf-8")
    print_stdout_summary(summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
