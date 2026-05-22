#!/usr/bin/env python3
"""Check OpenRouter key budget before launching long eval jobs.

This script intentionally prints only non-secret account metadata returned by
OpenRouter's key endpoint. It fails closed when the key is missing, the endpoint
cannot be queried, or the remaining monthly budget is below the configured
minimum. Some no-limit keys return null limit fields; those can be accepted only
with the explicit --allow-missing-limit opt-in.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from decimal import Decimal, InvalidOperation
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - dotenv is optional for shell-sourced envs.
    load_dotenv = None


def money(value: object) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def load_env() -> None:
    """Load .env without requiring python-dotenv in the system Python."""
    if load_dotenv is not None:
        load_dotenv()
        return
    env_path = Path(".env")
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
        value = value.strip().strip('"').strip("'")
        os.environ[name] = value


def truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def fetch_key_status(key: str, timeout: float) -> dict:
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/key",
        headers={"Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise ValueError("OpenRouter key endpoint returned non-object JSON")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min-limit-remaining",
        default="0.01",
        help="Minimum acceptable OpenRouter limit_remaining value.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=int(os.getenv("OPENROUTER_KEY_CHECK_RETRIES", "3")),
        help="Number of attempts for transient network failures.",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=float(os.getenv("OPENROUTER_KEY_CHECK_RETRY_DELAY", "5.0")),
        help="Seconds to sleep between retry attempts.",
    )
    parser.add_argument(
        "--allow-missing-limit",
        action="store_true",
        default=truthy_env("OPENROUTER_ALLOW_MISSING_LIMIT"),
        help=(
            "Accept a key status response with missing/non-numeric limit_remaining. "
            "Use only for explicitly approved no-limit keys; route/model smoke "
            "checks should still run before long jobs."
        ),
    )
    args = parser.parse_args()

    if args.retries < 1:
        print("--retries must be >= 1", file=sys.stderr)
        return 2
    if args.retry_delay < 0:
        print("--retry-delay must be >= 0", file=sys.stderr)
        return 2

    load_env()

    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not key:
        print("OPENROUTER_API_KEY is missing", file=sys.stderr)
        return 2

    payload = None
    for attempt in range(1, args.retries + 1):
        try:
            payload = fetch_key_status(key, timeout=args.timeout)
            break
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:500]
            transient = exc.code == 429 or exc.code >= 500
            print(
                f"OpenRouter key check failed: HTTP {exc.code}: {body}",
                file=sys.stderr,
            )
            if not transient or attempt == args.retries:
                return 3
        except Exception as exc:
            print(
                f"OpenRouter key check failed: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            if attempt == args.retries:
                return 3
        if attempt < args.retries and args.retry_delay:
            print(
                f"Retrying OpenRouter key check in {args.retry_delay:g}s "
                f"(attempt {attempt + 1}/{args.retries})",
                file=sys.stderr,
            )
            time.sleep(args.retry_delay)

    if payload is None:
        print("OpenRouter key check failed: no response payload", file=sys.stderr)
        return 3

    data = payload.get("data", payload)
    label = str(data.get("label", "unknown"))
    usage = data.get("usage")
    limit = data.get("limit")
    remaining = data.get("limit_remaining")
    print(
        "OpenRouter key status: "
        f"label={label} usage={usage} limit={limit} limit_remaining={remaining}",
        flush=True,
    )

    remaining_value = money(remaining)
    minimum = money(args.min_limit_remaining)
    if minimum is None:
        print(f"Invalid --min-limit-remaining={args.min_limit_remaining!r}", file=sys.stderr)
        return 2
    if remaining_value is None:
        if args.allow_missing_limit and limit is None:
            print(
                "OpenRouter key status accepted with missing limit_remaining "
                "because --allow-missing-limit/OPENROUTER_ALLOW_MISSING_LIMIT=1 "
                "was set.",
                flush=True,
            )
            return 0
        print("OpenRouter key check failed: missing/non-numeric limit_remaining", file=sys.stderr)
        return 3
    if remaining_value < minimum:
        print(
            "OpenRouter key check failed closed: "
            f"limit_remaining={remaining_value} < required {minimum}",
            file=sys.stderr,
        )
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
