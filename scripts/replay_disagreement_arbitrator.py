#!/usr/bin/env python3
"""Replay a disagreement-only arbitrator from completed detail logs.

The script joins method logs by label. If all methods predict the same answer,
it keeps the primary method's answer. If predictions disagree, it spends one
LLM call to choose among the completed method answers using the original
question, displayed options, and each method's reasoning.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

PROVIDERS = {
    "or-gemma4-26b": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "google/gemma-4-26b-a4b-it"),
    "or-gemma4-26b-free": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "google/gemma-4-26b-a4b-it:free"),
    "or-llama70b": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "meta-llama/llama-3.3-70b-instruct:free"),
    "cerebras": ("https://api.cerebras.ai/v1", "CEREBRAS_API_KEY", "llama-3.3-70b"),
}


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            label = row.get("label")
            if not label:
                raise SystemExit(f"{path}:{line_no}: missing label")
            rows[str(label)] = row
    if not rows:
        raise SystemExit(f"{path}: no rows loaded")
    return rows


def extract_answer(text: str, dataset: str) -> str | None:
    cleaned = (text or "").replace("*", "")
    if dataset == "housing":
        matches = re.findall(r"(?:Answer|ANSWER)[:\s]*(Yes|No)\b", cleaned, re.IGNORECASE)
        if matches:
            return matches[-1].capitalize()
        return None
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


def predicted(row: dict[str, Any] | None, dataset: str) -> str:
    if not row:
        return ""
    value = row.get("predicted_answer")
    if value:
        return str(value)
    return extract_answer(str(row.get("final_answer", "")), dataset) or ""


def gold(row: dict[str, Any]) -> str:
    return str(row.get("correct_answer") or row.get("gold") or "")


def token_estimate(text: str) -> int:
    return max(1, len(text or "") // 4)


def resolve_provider(provider: str) -> tuple[str, str, str]:
    provider = provider.strip().lower()
    if provider in PROVIDERS:
        base_url, key_env, model = PROVIDERS[provider]
        api_key = os.getenv(key_env, "")
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
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    response = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "messages": messages, "temperature": 0.0},
        timeout=90,
    )
    response.raise_for_status()
    payload = response.json()
    return str(payload["choices"][0]["message"]["content"] or "")


def fmt_choices(row: dict[str, Any]) -> str:
    choices = row.get("choices")
    if isinstance(choices, dict):
        return "\n".join(f"({str(k).upper()}) {v}" for k, v in sorted(choices.items()))
    if isinstance(choices, list):
        letters = ["A", "B", "C", "D", "E"]
        return "\n".join(f"({letters[i]}) {value}" for i, value in enumerate(choices[:5]))
    return str(choices or "")


def short(text: Any, limit: int) -> str:
    value = str(text or "")
    return value if len(value) <= limit else value[:limit] + "\n[truncated]"


def build_prompt(
    dataset: str,
    methods: list[str],
    rows: dict[str, dict[str, Any]],
    answer_chars: int,
    variant: str,
) -> tuple[str, str]:
    first = next(iter(rows.values()))
    if dataset == "housing":
        answer_format = "End with exactly one final line: Answer: Yes or Answer: No"
    else:
        answer_format = "End with exactly one final line in the form: Answer: (X)"
    if variant == "majority_prior":
        system = (
            "You are a careful legal answer arbitrator. Several completed legal RAG "
            "methods answered the same item but disagree. Start from the answer "
            "chosen by the majority of methods as a prior, but override it when "
            "the dissenting answer more specifically matches the cited legal "
            "holding, procedural posture, and answer option text. Do not count "
            f"votes mechanically. {answer_format}."
        )
    else:
        system = (
            "You are a careful legal answer arbitrator. Several completed legal RAG "
            "methods answered the same item but disagree. Choose the answer best "
            "supported by the question, answer options, and the methods' cited "
            f"reasoning. Do not average votes. {answer_format}."
        )
    parts = [
        "## Question",
        str(first.get("formatted_question") or first.get("question") or ""),
    ]
    choices = fmt_choices(first)
    if choices:
        parts.extend(["## Answer Options", choices])
    parts.append("## Method Answers")
    for method in methods:
        row = rows[method]
        parts.append(
            f"### {method}\n"
            f"Predicted: {predicted(row, dataset)}\n"
            f"Gold retrieved: {row.get('gold_retrieved', 'n/a')}\n"
            f"Route: {row.get('hyre_route') or row.get('adaptive_policy') or 'n/a'}\n"
            f"Reasoning:\n{short(row.get('final_answer', ''), answer_chars)}"
        )
    parts.append("## Task\nPick the single best answer. Explain briefly, then provide the required final answer line.")
    return system, "\n\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", action="append", nargs=2, metavar=("NAME", "PATH"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--provider", default=os.getenv("LLM_PROVIDER", "or-gemma4-26b"))
    parser.add_argument("--variant", choices=["neutral", "majority_prior"], default="neutral")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--answer-chars", type=int, default=1800)
    args = parser.parse_args()

    methods = [name for name, _ in args.method]
    logs = {name: load_jsonl(Path(path)) for name, path in args.method}
    labels = sorted(set.intersection(*(set(rows) for rows in logs.values())))
    if args.limit:
        labels = labels[: args.limit]
    if not labels:
        raise SystemExit("no overlapping labels")

    base_url, api_key, model = resolve_provider(args.provider)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    correct = 0
    arbitrated = 0
    print(f"provider={args.provider} model={model} dataset={args.dataset} rows={len(labels)}", flush=True)

    with args.output.open("w") as out:
        for i, label in enumerate(labels, 1):
            start = time.time()
            rows = {method: logs[method][label] for method in methods}
            preds = {method: predicted(rows[method], args.dataset) for method in methods}
            unique = {value for value in preds.values() if value}
            answer = ""
            error = None
            if len(unique) <= 1:
                final_prediction = preds[methods[0]]
                selected_source = "agreement"
            else:
                selected_source = "arbitrated"
                arbitrated += 1
                try:
                    system, user = build_prompt(args.dataset, methods, rows, args.answer_chars, args.variant)
                    print(f"[{i}/{len(labels)}] arbitrate {label} preds={preds}", flush=True)
                    answer = llm_call(base_url, api_key, model, system, user)
                    final_prediction = extract_answer(answer, args.dataset) or ""
                except Exception as exc:
                    error = str(exc)
                    final_prediction = preds[methods[0]]
                    system, user = "", ""
            g = gold(next(iter(rows.values())))
            ok = bool(final_prediction and final_prediction == g)
            correct += int(ok)
            record = {
                "label": label,
                "dataset": args.dataset,
                "mode": "adaptive_snap_hyre_disagreement_replay",
                "replay_variant": args.variant,
                "provider": args.provider,
                "methods": methods,
                "method_predictions": preds,
                "correct_answer": g,
                "predicted_answer": final_prediction,
                "is_correct": ok,
                "selected_source": selected_source,
                "arbitration_used": selected_source == "arbitrated",
                "final_answer": answer,
                "error": error,
                "llm_calls": 1 if selected_source == "arbitrated" and not error else 0,
                "input_tokens": token_estimate((system + user) if selected_source == "arbitrated" and not error else ""),
                "output_tokens": token_estimate(answer),
                "elapsed_sec": round(time.time() - start, 1),
                "source_rows": {
                    method: {
                        "mode": rows[method].get("mode"),
                        "hyre_route": rows[method].get("hyre_route"),
                        "predicted_answer": preds[method],
                        "is_correct": rows[method].get("is_correct"),
                    }
                    for method in methods
                },
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[{i}/{len(labels)}] {label} {'PASS' if ok else 'FAIL'} gold={g} pred={final_prediction} source={selected_source} error={error}", flush=True)

    print(f"RESULT {correct}/{len(labels)} = {correct / len(labels):.3f} arbitrated={arbitrated}")
    print(f"detail_log={args.output}")


if __name__ == "__main__":
    main()
