#!/usr/bin/env python3
"""Fail closed unless an LLM provider resolves to the intended model id."""
from __future__ import annotations

import argparse
import ast
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_ENV_PROVIDERS = {"custom", "cluster-vllm"}
TRUTHY = {"1", "true", "yes", "on"}


def load_env() -> None:
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        if not name or name in os.environ:
            continue
        os.environ[name] = value.strip().strip('"').strip("'")


def configured_model(provider: str) -> str:
    if provider in LEGACY_ENV_PROVIDERS:
        return os.getenv("LLM_MODEL", "").strip()
    module = ast.parse((REPO_ROOT / "llm_config.py").read_text(errors="ignore"))
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "PROVIDERS" for target in node.targets):
            continue
        providers = ast.literal_eval(node.value)
        if provider not in providers:
            known = ", ".join(sorted([*providers, *LEGACY_ENV_PROVIDERS]))
            raise ValueError(f"Unknown provider {provider!r}. Known providers: {known}")
        return str(providers[provider][2]).strip()
    raise RuntimeError("Could not find PROVIDERS in llm_config.py")


def env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in TRUTHY


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--expected-model", required=True)
    parser.add_argument(
        "--expected-label",
        default="",
        help="Optional MODEL_LABEL value required by the launcher.",
    )
    parser.add_argument(
        "--allow-openrouter-free-suffix",
        action="store_true",
        help=(
            "Explicitly accept an OpenRouter ':free' model id when its base "
            "model exactly matches --expected-model. Disabled by default so "
            "canonical rows still fail closed on route/model drift."
        ),
    )
    args = parser.parse_args()

    load_env()
    try:
        actual_model = configured_model(args.provider)
    except Exception as exc:
        print(f"provider/model check failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    allow_free_suffix = args.allow_openrouter_free_suffix or env_truthy(
        "OPENROUTER_ALLOW_FREE_SUFFIX"
    )
    accepted_free_suffix = (
        allow_free_suffix
        and actual_model == f"{args.expected_model}:free"
    )
    if actual_model != args.expected_model and not accepted_free_suffix:
        print(
            "provider/model check failed: "
            f"provider={args.provider!r} resolved model={actual_model!r}, "
            f"expected {args.expected_model!r}",
            file=sys.stderr,
        )
        return 3

    if args.expected_label:
        actual_label = os.getenv("MODEL_LABEL", "").strip()
        if actual_label != args.expected_label:
            print(
                "provider/model check failed: "
                f"MODEL_LABEL={actual_label!r}, expected {args.expected_label!r}",
                file=sys.stderr,
            )
            return 4

    suffix_note = " allow_openrouter_free_suffix=1" if accepted_free_suffix else ""
    print(f"provider/model check ok: provider={args.provider} model={actual_model}{suffix_note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
