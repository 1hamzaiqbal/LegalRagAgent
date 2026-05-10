#!/usr/bin/env python3
"""Replay a CaseHOLD final selector from an existing detail log.

This is for testing answer-option conversion without doing any fresh Chroma
retrieval or embedding. It consumes a detail JSONL that already contains the
question, choices, snap/HyRE text, and candidate evidence prompt, then asks one
new final-selector LLM call per row.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "eval"))

from main import _get_metrics, _llm_call, _reset_llm_call_counter  # type: ignore


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
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


def extract_answer(text: str) -> str | None:
    cleaned = (text or "").replace("*", "")
    patterns = [
        r"(?:Answer|ANSWER)[:\s]*\(?([A-E])\)?",
        r"(?:final answer|correct answer|best answer) (?:is|:)\s*\(?([A-E])\)?",
        r"\(([A-E])\)\s*$",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, cleaned, re.IGNORECASE | re.MULTILINE)
        if matches:
            return matches[-1].upper()
    return None


def source_user_prompt(row: dict[str, Any]) -> str:
    trace = row.get("call_trace") or []
    if trace and isinstance(trace, list):
        user = trace[0].get("user")
        if user:
            return str(user)
    preview = row.get("final_prompt_preview")
    if preview:
        return str(preview)
    raise ValueError(f"{row.get('label', '?')}: no replayable selector prompt")


def selector_system(variant: str) -> str:
    base = (
        "You are a careful legal holding selector for CaseHOLD. You must choose "
        "one displayed candidate holding A-E. Use the citing context and the "
        "candidate evidence bundles, but do not treat retrieval score, bundle "
        "length, or lexical overlap as decisive by themselves. Compare the rule, "
        "legal relationship, procedural posture, and fact pattern. End with "
        "exactly one final line in the form: Answer: (X)."
    )
    if variant == "strict":
        return base
    if variant == "snap_guard":
        return (
            base
            + "\n\nSNAP-GUARD RULES:\n"
            "- The snap answer is an initial hypothesis, not an oracle.\n"
            "- If changing away from the snap answer, identify the specific candidate evidence that justifies the change.\n"
            "- Prefer the displayed holding that is shortest while preserving the cited rule unless the context requires extra specificity."
        )
    if variant == "minimal_rule":
        return (
            base
            + "\n\nMINIMAL-RULE RULES:\n"
            "- CaseHOLD distractors often differ by unnecessary specificity.\n"
            "- Prefer the candidate that states the cited holding at the right level of generality.\n"
            "- Penalize candidates that add parties, agencies, facts, or procedural details not required by the citing context."
        )
    raise ValueError(f"unknown variant: {variant}")


def replay_user_prompt(row: dict[str, Any], variant: str) -> str:
    original = source_user_prompt(row)
    parts = [
        "## Original Candidate Evidence Prompt",
        original,
        "## Snap Answer Hypothesis",
        str(row.get("snap_answer", "")),
        "## HyRE Passage",
        str(row.get("hyde_passage", "")),
    ]
    if variant == "minimal_rule":
        parts.extend([
            "## Selection Instruction",
            "Choose the candidate holding that best states the rule actually cited, at the right level of generality.",
        ])
    else:
        parts.extend([
            "## Selection Instruction",
            "Choose the candidate holding best supported by the citing context and candidate evidence bundles.",
        ])
    return "\n\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-log", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--provider", default=os.getenv("LLM_PROVIDER", "or-gemma4-26b"))
    parser.add_argument("--variant", choices=["strict", "snap_guard", "minimal_rule"], default="minimal_rule")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    os.environ["LLM_PROVIDER"] = args.provider
    rows = load_jsonl(args.source_log)
    if args.limit:
        rows = rows[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    system = selector_system(args.variant)

    correct = 0
    with args.output.open("w") as out:
        for i, row in enumerate(rows, 1):
            _reset_llm_call_counter()
            start = time.time()
            error = None
            answer = ""
            predicted = None
            try:
                user = replay_user_prompt(row, args.variant)
                answer = _llm_call(system, user, label=f"casehold_selector_replay/{args.variant}")
                predicted = extract_answer(answer)
            except Exception as exc:
                error = str(exc)
            gold = str(row.get("correct_answer", "")).upper()
            is_correct = bool(predicted and predicted == gold)
            if is_correct:
                correct += 1
            metrics = _get_metrics()
            record = {
                **row,
                "source_mode": row.get("mode"),
                "source_final_answer": row.get("final_answer", ""),
                "source_predicted_answer": row.get("predicted_answer"),
                "source_is_correct": row.get("is_correct"),
                "mode": f"adaptive_snap_hyre_option_replay_{args.variant}",
                "provider": args.provider,
                "final_answer": answer,
                "predicted_answer": predicted,
                "is_correct": is_correct,
                "error": error,
                "llm_calls": metrics["count"],
                "input_tokens": metrics["input_tokens"],
                "output_tokens": metrics["output_tokens"],
                "elapsed_sec": round(time.time() - start, 1),
                "replay_variant": args.variant,
                "replay_source_log": str(args.source_log),
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            status = "PASS" if is_correct else "FAIL"
            print(f"[{i}/{len(rows)}] {row.get('label')} {status} gold={gold} pred={predicted} error={error}", flush=True)

    print(f"RESULT {correct}/{len(rows)} = {correct / len(rows):.3f}")
    print(f"detail_log={args.output}")


if __name__ == "__main__":
    main()
