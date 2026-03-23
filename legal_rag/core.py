"""Shared prompt, parsing, metrics, and LLM helpers."""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import math
import os
import re
import threading
import time
from typing import Any, Dict

import requests
import tiktoken
from langchain_core.messages import HumanMessage, SystemMessage

from llm_config import get_llm

logger = logging.getLogger(__name__)
_TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")
VERBOSE = os.getenv("VERBOSE", "0") == "1"


@functools.lru_cache(maxsize=16)
def load_skill(name: str) -> str:
    """Load a skill prompt from skills/<name>.md."""
    path = os.path.join("skills", f"{name}.md")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()
    except FileNotFoundError:
        return f"[WARNING: Skill file '{path}' not found]"


def get_prompt_version(name: str, content: str | None = None) -> str:
    """Return a short stable content hash for prompt/artifact logging."""
    body = content if content is not None else load_skill(name)
    return hashlib.sha1(body.encode("utf-8")).hexdigest()[:10]


def _parse_json(text: str) -> Any:
    """Forgiving JSON parser: handles fences, trailing commas, and JS comments."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    cleaned = re.sub(r"```(?:json)?\s*", "", str(text)).strip().rstrip("`")
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        pass

    for pattern in [r"(\{[\s\S]*\})", r"(\[[\s\S]*\])"]:
        match = re.search(pattern, cleaned)
        if not match:
            continue
        candidate = match.group(1)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
            fixed = re.sub(r"//.*?$", "", fixed, flags=re.MULTILINE)
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                continue
    return None


class MetricsState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.llm_call_counter = {"count": 0, "input_tokens": 0, "output_tokens": 0}


_metrics_state = MetricsState()


def _get_metrics() -> Dict[str, int]:
    return dict(_metrics_state.llm_call_counter)


def _reset_llm_call_counter() -> None:
    with _metrics_state.lock:
        _metrics_state.llm_call_counter = {"count": 0, "input_tokens": 0, "output_tokens": 0}


def _record_tokens(system_prompt: str, user_prompt: str, content: str) -> None:
    try:
        input_tokens = len(_TIKTOKEN_ENC.encode(system_prompt + user_prompt))
        output_tokens = len(_TIKTOKEN_ENC.encode(content))
    except Exception:
        input_tokens = len(system_prompt) + len(user_prompt)
        output_tokens = len(content)

    with _metrics_state.lock:
        _metrics_state.llm_call_counter["count"] += 1
        _metrics_state.llm_call_counter["input_tokens"] += input_tokens
        _metrics_state.llm_call_counter["output_tokens"] += output_tokens


def _get_deepseek_balance() -> Dict[str, Any]:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()

    if "deepseek" not in provider or not api_key:
        return {"is_available": False}

    try:
        response = requests.get(
            "https://api.deepseek.com/user/balance",
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        data["is_available"] = True
        return data
    except Exception as exc:
        logger.warning("Failed to fetch DeepSeek balance: %s", exc)
        return {"is_available": False}


def _llm_call(system_prompt: str, user_prompt: str, label: str = "") -> str:
    """Invoke the configured LLM with retry handling."""
    llm = get_llm()
    transient_tokens = ("429", "connection", "timeout", "rate", "overloaded", "unavailable")

    for attempt in range(3):
        try:
            model_name = getattr(llm, "model_name", "") or ""
            if "gemma" in model_name.lower():
                combined = f"[Instructions]\n{system_prompt}\n\n[Query]\n{user_prompt}"
                messages = [HumanMessage(content=combined)]
            else:
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            response = llm.invoke(messages)
            content = response.content
            _record_tokens(system_prompt, user_prompt, content)

            if label:
                if VERBOSE:
                    in_tokens = len(_TIKTOKEN_ENC.encode(system_prompt + user_prompt))
                    out_tokens = len(_TIKTOKEN_ENC.encode(content))
                    print(f"    [{label}] {len(content)} chars ({in_tokens} in / {out_tokens} out tokens)")
                else:
                    print(f"    [{label}] {len(content)} chars")
            return content
        except Exception as exc:
            err = str(exc).lower()
            is_transient = any(token in err for token in transient_tokens)
            if is_transient and attempt < 2:
                wait = 5 * (attempt + 1)
                retry_after = re.search(r"retry.after['\"]:\s*(\d+)", err)
                if retry_after:
                    wait = int(retry_after.group(1))
                print(f"    [{label}] Transient error (attempt {attempt + 1}/3), retry in {wait}s: {exc}")
                time.sleep(wait)
                continue
            print(f"    [{label}] LLM error: {exc}")
            raise
    return ""


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _normalise_confidence(raw_logit: float) -> float:
    return round(_sigmoid(raw_logit), 4)
