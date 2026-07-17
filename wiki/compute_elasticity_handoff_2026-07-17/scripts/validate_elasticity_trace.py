#!/usr/bin/env python3
"""Validate framework-neutral elasticity trajectories and rescore prices."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable


REQUIRED = {
    "episode_id",
    "task_id",
    "task_family",
    "difficulty",
    "split",
    "model_id",
    "teacher_or_student",
    "skill_condition",
    "price_condition",
    "hard_limits",
    "messages",
    "actions",
    "task_reward_native",
    "task_success_exact",
    "usage",
    "termination",
    "code_version",
    "environment_version",
}


def validate_episode(row: dict[str, Any]) -> list[str]:
    errors = []
    missing = sorted(REQUIRED - row.keys())
    if missing:
        errors.append(f"missing fields: {missing}")

    exact = row.get("task_success_exact")
    if exact not in (0, 1):
        errors.append("task_success_exact must be 0 or 1")

    native = row.get("task_reward_native")
    if not isinstance(native, (int, float)) or not math.isfinite(native):
        errors.append("task_reward_native must be finite numeric")

    price = row.get("price_condition", {}).get("tool")
    calls = row.get("usage", {}).get("tool_calls")
    if not isinstance(price, (int, float)) or price < 0 or not math.isfinite(price):
        errors.append("price_condition.tool must be finite and nonnegative")
    if not isinstance(calls, int) or calls < 0:
        errors.append("usage.tool_calls must be a nonnegative integer")

    max_calls = row.get("hard_limits", {}).get("max_tool_calls")
    if isinstance(calls, int) and isinstance(max_calls, int) and calls > max_calls:
        errors.append("usage.tool_calls exceeds hard limit")

    if not isinstance(row.get("messages"), list) or not isinstance(row.get("actions"), list):
        errors.append("messages and actions must be lists")
    return errors


def rescore(row: dict[str, Any], tool_price: float, token_price: float = 0.0) -> float:
    return (
        float(row["task_success_exact"])
        - tool_price * int(row["usage"]["tool_calls"])
        - token_price * int(row["usage"].get("output_tokens", 0))
    )


def load_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                yield line_number, json.loads(line)


def synthetic_episode(episode_id: str, exact: int, calls: int, termination: str) -> dict[str, Any]:
    return {
        "episode_id": episode_id,
        "task_id": "smoke-task",
        "task_family": "synthetic",
        "difficulty": "smoke",
        "split": "development",
        "model_id": "scripted-policy",
        "teacher_or_student": "baseline",
        "skill_condition": "none",
        "price_condition": {"tool": 1.0, "token": 0.0},
        "hard_limits": {"max_tokens": 32, "max_tool_calls": 2},
        "messages": [],
        "actions": [],
        "task_reward_native": float(exact),
        "task_success_exact": exact,
        "usage": {"input_tokens": 0, "output_tokens": 4, "tool_calls": calls},
        "termination": termination,
        "code_version": "self-test",
        "environment_version": "self-test",
    }


def self_test() -> None:
    episodes = [
        synthetic_episode("correct-internal", 1, 0, "final_answer"),
        synthetic_episode("correct-tool", 1, 1, "final_answer"),
        synthetic_episode("wrong-tool", 0, 1, "final_answer"),
        synthetic_episode("timeout", 0, 0, "timeout"),
    ]
    assert all(not validate_episode(row) for row in episodes)
    assert rescore(episodes[0], tool_price=4.0) > rescore(episodes[1], tool_price=4.0)
    assert rescore(episodes[0], tool_price=0.0) == rescore(episodes[1], tool_price=0.0)
    assert rescore(episodes[2], tool_price=1.0) < rescore(episodes[3], tool_price=1.0)
    print("self-test passed: 4 golden trajectories; component rescoring is consistent")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path, nargs="?")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    if args.jsonl:
        failures = 0
        count = 0
        for line_number, row in load_jsonl(args.jsonl):
            count += 1
            errors = validate_episode(row)
            if errors:
                failures += 1
                print(json.dumps({"line": line_number, "errors": errors}, sort_keys=True))
        print(json.dumps({"rows": count, "invalid_rows": failures}, sort_keys=True))
        if failures:
            raise SystemExit(1)
    if not args.self_test and not args.jsonl:
        parser.error("provide --self-test or a JSONL path")


if __name__ == "__main__":
    main()
