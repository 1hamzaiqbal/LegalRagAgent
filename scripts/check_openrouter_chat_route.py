#!/usr/bin/env python3
"""Tiny OpenRouter chat-completion route smoke before long eval jobs."""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - dotenv is optional for shell-sourced envs.
    load_dotenv = None


REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_ENV_PROVIDERS = {"custom", "cluster-vllm"}
TRUTHY = {"1", "true", "yes", "on"}


def load_env() -> None:
    if load_dotenv is not None:
        load_dotenv()
        return
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
        if not any(
            isinstance(target, ast.Name) and target.id == "PROVIDERS"
            for target in node.targets
        ):
            continue
        providers = ast.literal_eval(node.value)
        if provider not in providers:
            known = ", ".join(sorted([*providers, *LEGACY_ENV_PROVIDERS]))
            raise ValueError(f"Unknown provider {provider!r}. Known providers: {known}")
        return str(providers[provider][2]).strip()
    raise RuntimeError("Could not find PROVIDERS in llm_config.py")


def env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in TRUTHY


def csv_values(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def provider_payload(
    provider_only: str,
    provider_order: str,
    provider_ignore: str,
    allow_fallbacks: bool,
) -> dict:
    payload: dict[str, object] = {
        "allow_fallbacks": allow_fallbacks,
        "require_parameters": True,
    }
    route_controls = (
        ("only", provider_only),
        ("order", provider_order),
        ("ignore", provider_ignore),
    )
    for field, value in route_controls:
        values = csv_values(value)
        if values:
            payload[field] = values
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--expected-model", required=True)
    parser.add_argument(
        "--allow-openrouter-free-suffix",
        action="store_true",
        help="Explicitly accept and smoke an OpenRouter ':free' suffix.",
    )
    parser.add_argument(
        "--provider-only",
        default="",
        help="Optional OpenRouter provider.only route, e.g. Cloudflare.",
    )
    parser.add_argument(
        "--provider-order",
        default="",
        help="Optional comma-separated OpenRouter provider.order route.",
    )
    parser.add_argument(
        "--provider-ignore",
        default="",
        help="Optional comma-separated OpenRouter provider.ignore route.",
    )
    parser.add_argument(
        "--allow-fallbacks",
        action="store_true",
        help="Allow OpenRouter provider fallbacks. Defaults to disabled.",
    )
    parser.add_argument(
        "--disable-reasoning",
        action="store_true",
        default=env_truthy("OPENROUTER_DISABLE_REASONING"),
        help="Pass OpenRouter reasoning.enabled=false for reactive instruct checks.",
    )
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--expected-content", default="OK")
    args = parser.parse_args()

    if args.max_tokens < 1:
        print("--max-tokens must be >= 1", file=sys.stderr)
        return 2

    load_env()
    provider_only = args.provider_only.strip() or os.getenv(
        "OPENROUTER_PROVIDER_ONLY", ""
    ).strip()
    provider_order = args.provider_order.strip() or os.getenv(
        "OPENROUTER_PROVIDER_ORDER", ""
    ).strip()
    provider_ignore = args.provider_ignore.strip() or os.getenv(
        "OPENROUTER_PROVIDER_IGNORE", ""
    ).strip()
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not key:
        print("OPENROUTER_API_KEY is missing", file=sys.stderr)
        return 2

    try:
        model = configured_model(args.provider)
    except Exception as exc:
        print(f"OpenRouter route smoke failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    allow_free_suffix = args.allow_openrouter_free_suffix or env_truthy(
        "OPENROUTER_ALLOW_FREE_SUFFIX"
    )
    if model != args.expected_model:
        if not (allow_free_suffix and model == f"{args.expected_model}:free"):
            print(
                "OpenRouter route smoke failed: "
                f"provider={args.provider!r} resolved model={model!r}, "
                f"expected {args.expected_model!r}",
                file=sys.stderr,
            )
            return 3

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": f"Answer with exactly: {args.expected_content}"},
            {"role": "user", "content": f"Return {args.expected_content}."},
        ],
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "provider": provider_payload(
            provider_only,
            provider_order,
            provider_ignore,
            args.allow_fallbacks,
        ),
    }
    if args.disable_reasoning:
        body["reasoning"] = {"enabled": False}
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")[:1000]
        print(
            f"OpenRouter route smoke failed: HTTP {exc.code}: {text}",
            file=sys.stderr,
        )
        return 4
    except Exception as exc:
        print(
            f"OpenRouter route smoke failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 4

    choice = (payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = str(message.get("content") or "").strip()
    returned_model = payload.get("model")
    usage = payload.get("usage") or {}
    provider_name = payload.get("provider") or payload.get("provider_name") or ""
    print(
        "OpenRouter route smoke ok: "
        f"provider={args.provider} model={model} returned_model={returned_model!r} "
        f"provider_name={provider_name!r} content={content!r} usage={usage}"
    )
    if content != args.expected_content:
        print(
            "OpenRouter route smoke failed: "
            f"expected content {args.expected_content!r}, got {content!r}",
            file=sys.stderr,
        )
        return 5
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
