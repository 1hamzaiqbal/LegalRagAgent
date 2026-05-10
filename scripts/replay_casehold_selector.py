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

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from llm_config import PROVIDERS  # type: ignore

load_dotenv(REPO_ROOT / ".env")


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


def make_llm(provider: str) -> tuple[ChatOpenAI, str]:
    provider = provider.strip().lower()
    if provider in PROVIDERS:
        base_url, key_env, model, _, _ = PROVIDERS[provider]
        api_key = os.getenv(key_env or "", "") if key_env else "ollama"
    else:
        base_url = os.getenv("LLM_BASE_URL", "https://api.cerebras.ai/v1")
        api_key = os.getenv("LLM_API_KEY", "no-key-set")
        model = os.getenv("LLM_MODEL", "llama-3.3-70b")
    if not api_key or api_key == "no-key-set":
        raise SystemExit(f"missing API key for provider={provider}")
    return (
        ChatOpenAI(
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=0.0,
            timeout=60,
            max_retries=1,
        ),
        model,
    )


def llm_call(llm: ChatOpenAI, model: str, system: str, user: str) -> str:
    if "gemma" in model.lower():
        messages = [HumanMessage(content=f"[Instructions]\n{system}\n\n[Query]\n{user}")]
    else:
        messages = [SystemMessage(content=system), HumanMessage(content=user)]
    response = llm.invoke(messages)
    return str(response.content or "")


def token_estimate(text: str) -> int:
    return max(1, len(text or "") // 4)


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
    parser.add_argument("--max-user-chars", type=int, default=12000)
    args = parser.parse_args()

    rows = load_jsonl(args.source_log)
    if args.limit:
        rows = rows[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    system = selector_system(args.variant)
    llm, model = make_llm(args.provider)
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
                answer = llm_call(llm, model, system, user)
                predicted = extract_answer(answer)
            except Exception as exc:
                error = str(exc)
            gold = str(row.get("correct_answer", "")).upper()
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
