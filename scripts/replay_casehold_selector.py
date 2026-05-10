#!/usr/bin/env python3
"""Replay a CaseHOLD final selector from an existing detail log.

This is for testing answer-option conversion without doing any fresh Chroma
retrieval or embedding. It consumes a detail JSONL that already contains the
question, choices, snap/HyRE text, and candidate evidence prompt, then asks one
new final-selector LLM call per row. It can also consume escalation rows
exported by analyze_casehold_disagreements.py, which lets us test targeted
answer-conversion prompts without re-running retrieval.
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

from dotenv import load_dotenv
import requests

load_dotenv(REPO_ROOT / ".env")


PROVIDERS = {
    "or-gemma4-26b": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "google/gemma-4-26b-a4b-it"),
    "or-gemma4-26b-free": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "google/gemma-4-26b-a4b-it:free"),
    "or-llama70b": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "meta-llama/llama-3.3-70b-instruct:free"),
    "cerebras": ("https://api.cerebras.ai/v1", "CEREBRAS_API_KEY", "llama-3.3-70b"),
}


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


def is_escalation_row(row: dict[str, Any]) -> bool:
    return "evidence_snippets" in row and "method_predictions" in row and "choices" in row


def format_choices(choices: Any) -> str:
    if isinstance(choices, dict):
        items = sorted((str(k).upper(), str(v)) for k, v in choices.items())
    elif isinstance(choices, list):
        letters = ["A", "B", "C", "D", "E"]
        items = [(letters[i], str(value)) for i, value in enumerate(choices[:5])]
    else:
        return str(choices or "")
    return "\n".join(f"({letter}) {text}" for letter, text in items)


def format_method_predictions(row: dict[str, Any]) -> str:
    predictions = row.get("method_predictions") or {}
    correctness = row.get("method_correct") or {}
    if not isinstance(predictions, dict):
        return str(predictions)
    lines = []
    for name in sorted(predictions):
        marker = ""
        if isinstance(correctness, dict) and name in correctness:
            marker = " [correct]" if correctness[name] else " [wrong]"
        lines.append(f"- {name}: {predictions[name]}{marker}")
    return "\n".join(lines) if lines else "(none)"


def format_evidence(row: dict[str, Any], limit: int = 6) -> str:
    snippets = row.get("evidence_snippets") or []
    if not isinstance(snippets, list):
        return str(snippets)
    lines = []
    for i, snippet in enumerate(snippets[:limit], 1):
        if isinstance(snippet, dict):
            letter = str(snippet.get("choice", snippet.get("letter", "?"))).upper()
            text = snippet.get("text") or snippet.get("snippet") or snippet.get("content") or snippet
            lines.append(f"[{i}] Candidate {letter}: {text}")
        else:
            lines.append(f"[{i}] {snippet}")
    return "\n\n".join(lines) if lines else "(no evidence snippets)"


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
    if variant == "rule_frame":
        return (
            base
            + "\n\nRULE-FRAME RULES:\n"
            "- First normalize every answer option into a compact rule frame: legal rule, parties/object, procedural posture, required facts, and extra specificity.\n"
            "- Match the citing context to the normalized rule frames before relying on retrieved snippets.\n"
            "- Treat previous method predictions as weak signals only; disagreement means you should explain the discriminating feature.\n"
            "- Prefer the least over-specified candidate whose rule frame fully covers the citing context."
        )
    raise ValueError(f"unknown variant: {variant}")


def resolve_provider(provider: str) -> tuple[str, str, str]:
    provider = provider.strip().lower()
    if provider in PROVIDERS:
        base_url, key_env, model = PROVIDERS[provider]
        api_key = os.getenv(key_env or "", "") if key_env else "ollama"
    else:
        base_url = os.getenv("LLM_BASE_URL", "https://api.cerebras.ai/v1")
        api_key = os.getenv("LLM_API_KEY", "no-key-set")
        model = os.getenv("LLM_MODEL", "llama-3.3-70b")
    if not api_key or api_key == "no-key-set":
        raise SystemExit(f"missing API key for provider={provider}")
    return base_url.rstrip("/"), api_key, model


def llm_call(base_url: str, api_key: str, model: str, system: str, user: str) -> str:
    if "gemma" in model.lower():
        messages = [{"role": "user", "content": f"[Instructions]\n{system}\n\n[Query]\n{user}"}]
    else:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    response = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "messages": messages, "temperature": 0.0},
        timeout=75,
    )
    response.raise_for_status()
    payload = response.json()
    return str(payload["choices"][0]["message"]["content"] or "")


def token_estimate(text: str) -> int:
    return max(1, len(text or "") // 4)


def replay_user_prompt(row: dict[str, Any], variant: str) -> str:
    if variant == "rule_frame" and is_escalation_row(row):
        question = row.get("question") or row.get("query") or row.get("citing_context") or ""
        parts = [
            "## Citing Context / Question",
            str(question),
            "## Candidate Holdings",
            format_choices(row.get("choices")),
            "## Previous Method Predictions",
            format_method_predictions(row),
            "## Snap Answer Hypothesis",
            f"letter={row.get('snap_letter')} text={row.get('snap_answer', '')}",
            "## HyRE Passage",
            str(row.get("hyde_passage", "")),
            "## Retrieved Evidence Snippets",
            format_evidence(row),
            "## Task",
            (
                "Write one short rule frame for each candidate A-E, then choose the candidate whose "
                "rule frame best matches the citing context. End with exactly one final line: Answer: (X)."
            ),
        ]
        return "\n\n".join(parts)

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
    parser.add_argument(
        "--variant",
        choices=["strict", "snap_guard", "minimal_rule", "rule_frame"],
        default="minimal_rule",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-user-chars", type=int, default=12000)
    args = parser.parse_args()

    rows = load_jsonl(args.source_log)
    if args.limit:
        rows = rows[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    system = selector_system(args.variant)
    base_url, api_key, model = resolve_provider(args.provider)
    print(f"provider={args.provider} model={model} rows={len(rows)} variant={args.variant}", flush=True)

    correct = 0
    with args.output.open("w") as out:
        for i, row in enumerate(rows, 1):
            start = time.time()
            error = None
            answer = ""
            predicted = None
            user = ""
            try:
                user = replay_user_prompt(row, args.variant)
                if args.max_user_chars > 0 and len(user) > args.max_user_chars:
                    user = user[: args.max_user_chars] + "\n\n[Replay prompt truncated to configured character budget.]"
                print(f"[{i}/{len(rows)}] calling {row.get('label')} user_chars={len(user)}", flush=True)
                answer = llm_call(base_url, api_key, model, system, user)
                predicted = extract_answer(answer)
            except Exception as exc:
                error = str(exc)
            gold = str(row.get("correct_answer") or row.get("gold") or "").upper()
            is_correct = bool(predicted and predicted == gold)
            if is_correct:
                correct += 1
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
                "llm_calls": 0 if error else 1,
                "input_tokens": token_estimate(system + user),
                "output_tokens": token_estimate(answer),
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
