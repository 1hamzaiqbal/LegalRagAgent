#!/usr/bin/env python
"""HTTP client for vLLM OpenAI-compatible teacher scoring.

OPD needs teacher logprobs for the exact completion tokens sampled by the
student. That alignment is only valid when teacher and student share a tokenizer
family, for example Qwen3-to-Qwen3 or Llama-3.x-to-Llama-3.x. The caller passes
the student tokenizer so this client can count prompt tokens and slice vLLM's
prompt logprobs down to completion-token logprobs.

vLLM exposes `prompt_logprobs` as an OpenAI extra body field. With the raw HTTP
API used here, that extra field is merged into the JSON body as
`"prompt_logprobs": 1`, equivalent to OpenAI SDK
`extra_body={"prompt_logprobs": 1}`.
"""
import time
from typing import Any


def _requests():
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("teacher_client requires the requests package") from exc
    return requests


def _clean_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def _encode(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def _logprob_value(value: Any) -> float:
    if isinstance(value, (float, int)):
        return float(value)
    if isinstance(value, dict) and "logprob" in value:
        return float(value["logprob"])
    raise ValueError(f"cannot extract logprob from value: {value!r}")


def _entry_logprob(entry: Any, token_id: int, pos: int) -> float:
    if entry is None:
        raise ValueError(f"missing logprob at token position {pos}")
    if isinstance(entry, (float, int)):
        raise ValueError(
            f"unkeyed scalar logprob at token position {pos} cannot prove "
            f"identity for expected token_id={token_id}"
        )
    if not isinstance(entry, dict):
        raise ValueError(f"unexpected logprob entry at position {pos}: {entry!r}")

    for key in (token_id, str(token_id)):
        if key in entry:
            return _logprob_value(entry[key])

    if "token_id" in entry:
        observed = int(entry["token_id"])
        if observed != token_id:
            raise ValueError(
                f"teacher returned token_id={observed} for expected token_id={token_id} "
                f"at position {pos}"
            )
        if "logprob" in entry:
            return _logprob_value(entry)

    keys = ", ".join(str(k) for k in list(entry)[:8])
    raise ValueError(
        f"teacher did not return logprob for token_id={token_id} at position {pos}; "
        f"available keys: {keys}"
    )


def _sleep_backoff(attempt: int, base_sleep: float) -> None:
    if base_sleep > 0:
        time.sleep(base_sleep * (2 ** attempt))


def _request_json(
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: tuple[float, float] = (10.0, 120.0),
    retries: int = 3,
    backoff: float = 0.5,
) -> dict[str, Any]:
    requests = _requests()
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            if method == "GET":
                resp = requests.get(url, timeout=timeout)
            elif method == "POST":
                resp = requests.post(url, json=payload, timeout=timeout)
            else:
                raise ValueError(f"unsupported HTTP method: {method}")
        except requests.RequestException as exc:
            last_error = exc
            if attempt + 1 == retries:
                raise RuntimeError(f"{method} {url} failed after {retries} attempts: {exc}") from exc
            _sleep_backoff(attempt, backoff)
            continue

        if 200 <= resp.status_code < 300:
            try:
                return resp.json()
            except ValueError as exc:
                body = resp.text[:1000]
                raise RuntimeError(f"{method} {url} returned non-JSON body: {body}") from exc

        body = resp.text[:1000]
        last_error = RuntimeError(f"{method} {url} returned HTTP {resp.status_code}: {body}")
        if resp.status_code < 500 or attempt + 1 == retries:
            raise last_error
        _sleep_backoff(attempt, backoff)

    raise RuntimeError(f"{method} {url} failed: {last_error}")


def healthcheck(base_url: str, timeout: float = 5.0, retries: int = 3, raise_on_error: bool = False) -> bool:
    """Return whether the vLLM server responds on `/health` or `/v1/models`."""
    requests = _requests()
    base = _clean_base_url(base_url)
    last_error: Exception | None = None
    for path in ("/health", "/v1/models"):
        url = base + path
        for attempt in range(retries):
            try:
                resp = requests.get(url, timeout=(min(2.0, timeout), timeout))
                if 200 <= resp.status_code < 300:
                    return True
                last_error = RuntimeError(f"GET {url} returned HTTP {resp.status_code}: {resp.text[:500]}")
            except requests.RequestException as exc:
                last_error = exc
            if attempt + 1 < retries:
                _sleep_backoff(attempt, 0.5)
    if raise_on_error:
        raise RuntimeError(f"vLLM healthcheck failed for {base}: {last_error}")
    return False


def score_completion_token_logprobs(
    base_url: str,
    model: str,
    prompt_token_ids: list[int],
    completion_token_ids: list[int],
    *,
    timeout: tuple[float, float] = (10.0, 120.0),
    retries: int = 3,
) -> list[float]:
    """Return teacher logprobs for exact student-generated token IDs.

    The request sends one tokenized prompt to `/v1/completions` with
    `echo=True`, `logprobs=1`, and vLLM `prompt_logprobs=1`. The returned prompt
    logprobs cover the exact concatenated token sequence. This avoids the BPE
    boundary bug caused by decoding and re-tokenizing prompt+completion text.
    """
    prompt_ids = [int(x) for x in prompt_token_ids]
    completion_ids = [int(x) for x in completion_token_ids]
    if not prompt_ids:
        raise ValueError("prompt_token_ids must not be empty")
    if not completion_ids:
        return []
    prompt_count = len(prompt_ids)

    payload = {
        "model": model,
        "prompt": prompt_ids + completion_ids,
        "max_tokens": 1,
        "temperature": 0.0,
        "echo": True,
        "logprobs": 1,
        "prompt_logprobs": 1,
    }
    data = _request_json(
        "POST",
        _clean_base_url(base_url) + "/v1/completions",
        payload=payload,
        timeout=timeout,
        retries=retries,
    )
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"teacher response had no choices: {data}")
    choice = choices[0]

    prompt_logprobs = choice.get("prompt_logprobs")
    if prompt_logprobs is None:
        prompt_logprobs = data.get("prompt_logprobs")
    if prompt_logprobs is not None:
        out = []
        start = prompt_count
        end = start + len(completion_ids)
        if len(prompt_logprobs) < end:
            raise RuntimeError(
                f"teacher returned {len(prompt_logprobs)} prompt logprob entries, "
                f"need at least {end}"
            )
        for i, token_id in enumerate(completion_ids):
            pos = start + i
            out.append(_entry_logprob(prompt_logprobs[pos], token_id, pos))
        return out

    raise RuntimeError(
        "teacher response lacked token-ID-keyed prompt_logprobs; plain token_logprobs "
        "cannot prove exact student/teacher token alignment"
    )


def score_completion_logprobs(
    base_url: str,
    model: str,
    prompt_text: str,
    completion_text: str,
    tokenizer: Any,
    *,
    timeout: tuple[float, float] = (10.0, 120.0),
    retries: int = 3,
) -> list[float]:
    """Compatibility wrapper that tokenizes each segment independently.

    New OPD code should preserve rollout token IDs and call
    :func:`score_completion_token_logprobs` directly.
    """
    return score_completion_token_logprobs(
        base_url,
        model,
        _encode(tokenizer, prompt_text),
        _encode(tokenizer, completion_text),
        timeout=timeout,
        retries=retries,
    )


def sample_from_server(
    base_url: str,
    model: str,
    prompt_text: str,
    *,
    max_tokens: int = 128,
    temperature: float = 1.0,
    top_p: float = 1.0,
    n: int = 1,
    stop: list[str] | None = None,
    timeout: tuple[float, float] = (10.0, 120.0),
    retries: int = 3,
) -> list[str]:
    """Sample one or more completions from an OpenAI-compatible vLLM server."""
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt_text,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "n": n,
    }
    if stop:
        payload["stop"] = stop
    data = _request_json(
        "POST",
        _clean_base_url(base_url) + "/v1/completions",
        payload=payload,
        timeout=timeout,
        retries=retries,
    )
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"sample response had no choices: {data}")
    return [str(choice.get("text", "")) for choice in choices]
