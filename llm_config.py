"""LLM configuration — single entry point for all LLM calls.

Supports multiple providers via LLM_PROVIDER env var:
  gemma, gemini-flash, gemini-flash-lite,
  groq-llama70b, groq-llama8b, groq-maverick, groq-scout, groq-gpt120b, groq-kimi, groq-qwen,
  or-llama70b, or-gpt20b, or-ministral-8b, or-gemma3n-e4b, or-gemma3-4b, or-gemma27b,
  or-gemma4-26b, or-gemma4-31b,
  or-qwen3-coder, or-nemotron, or-mistral, or-hermes,
  ollama, cerebras

Falls back to raw LLM_BASE_URL/LLM_API_KEY/LLM_MODEL only when LLM_PROVIDER is
unset, or when LLM_PROVIDER is explicitly custom/cluster-vllm.
"""

import functools
import os
import threading
from dotenv import load_dotenv

load_dotenv()

# Provider registry: name -> (base_url, api_key_env, model, rpd, tpd)
PROVIDERS = {
    # --- OpenAI ---
    "gpt-4.1-nano":       ("https://api.openai.com/v1", "OPENAI_API_KEY", "gpt-4.1-nano",              None, None),
    "gpt-4.1-mini":       ("https://api.openai.com/v1", "OPENAI_API_KEY", "gpt-4.1-mini",              None, None),
    "gpt-5.4-nano":       ("https://api.openai.com/v1", "OPENAI_API_KEY", "gpt-5.4-nano",              None, None),
    "gpt-5.4-mini":       ("https://api.openai.com/v1", "OPENAI_API_KEY", "gpt-5.4-mini",              None, None),
    # --- DeepSeek ---
    "deepseek":           ("https://api.deepseek.com", "DEEPSEEK_API_KEY", "deepseek-chat",            None, None),
    "deepseek-reasoner":  ("https://api.deepseek.com", "DEEPSEEK_API_KEY", "deepseek-reasoner",        None, None),
    # --- Google AI Studio ---
    "gemma":              ("https://generativelanguage.googleapis.com/v1beta/openai/", "GOOGLE_API_KEY", "gemma-3-27b-it",       14_400, None),
    "gemma-4b":           ("https://generativelanguage.googleapis.com/v1beta/openai/", "GOOGLE_API_KEY", "gemma-3-4b-it",        14_400, None),
    "gemini-flash":       ("https://generativelanguage.googleapis.com/v1beta/openai/", "GOOGLE_API_KEY", "gemini-2.5-flash",     20,     None),
    "gemini-flash-lite":  ("https://generativelanguage.googleapis.com/v1beta/openai/", "GOOGLE_API_KEY", "gemini-2.5-flash-lite", 20,    None),
    # --- Groq ---
    "groq-llama70b":      ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "llama-3.3-70b-versatile",                    1_000, 100_000),
    "groq-llama8b":       ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "llama-3.1-8b-instant",                      14_400, 500_000),
    "groq-maverick":      ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "meta-llama/llama-4-maverick-17b-128e-instruct", 1_000, 500_000),
    "groq-scout":         ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "meta-llama/llama-4-scout-17b-16e-instruct", 1_000, 500_000),
    "groq-gpt120b":       ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "openai/gpt-oss-120b",                       1_000, 200_000),
    "groq-kimi":          ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "moonshotai/kimi-k2-instruct",               1_000, 300_000),
    "groq-qwen":          ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "qwen/qwen3-32b",                            1_000, 500_000),
    # groq-qwen8b removed — Groq doesn't carry qwen3-8b
    # --- Qwen3 small + reasoning (OpenRouter; not a main-grid default because it can emit think traces) ---
    "or-qwen3-8b":        ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "qwen/qwen3-8b",                          None, None),
    "or-qwen3-14b":       ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "qwen/qwen3-14b",                         None, None),
    "or-qwen3-32b":       ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "qwen/qwen3-32b",                         None, None),
    "or-qwen3-30b-moe":   ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "qwen/qwen3-30b-a3b",                     None, None),
    "or-qwen35-9b":       ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "qwen/qwen3.5-9b",                        None, None),
    # --- Small API model replacement for historical Gemma 4 E4B ---
    "or-ministral-8b":     ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "mistralai/ministral-8b-2512",              None, None),
    # --- OpenRouter (paid) ---
    "or-phi4":            ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "microsoft/phi-4",                       None, None),
    "or-mistral-nemo":    ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "mistralai/mistral-nemo",                None, None),
    # --- OpenRouter (free tier — weekly token limits, no RPD cap) ---
    "or-llama70b":        ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "meta-llama/llama-3.3-70b-instruct:free",     None, None),
    "or-llama70b-paid":   ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "meta-llama/llama-3.3-70b-instruct",          None, None),
    "or-gpt20b":          ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "openai/gpt-oss-20b:free",                    None, None),
    "or-gemma3n-e4b":     ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "google/gemma-3n-e4b-it",                 None, None),
    "or-gemma3-4b":       ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "google/gemma-3-4b-it",                  None, None),
    # Legacy alias retained for old scripts. This is Gemma 3 4B, not Gemma 4 E4B.
    "or-gemma4b":         ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "google/gemma-3-4b-it",                  None, None),
    "or-gemma27b":        ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "google/gemma-3-27b-it",                 None, None),
    # --- Gemma 4 via OpenRouter (matches our cluster vLLM Gemma 4 models) ---
    "or-gemma4-26b":      ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "google/gemma-4-26b-a4b-it",             None, None),
    "or-gemma4-26b-free": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "google/gemma-4-26b-a4b-it:free",        None, None),
    "or-gemma4-31b":      ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "google/gemma-4-31b-it",                 None, None),
    "or-gemma4-31b-free": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "google/gemma-4-31b-it:free",            None, None),
    "or-qwen3-coder":     ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "qwen/qwen3-coder-480b-a35b:free",           None, None),
    # Reasoning-trace model; keep available for explicit experiments, not main-grid defaults.
    "or-nemotron":        ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "nvidia/nemotron-nano-9b-v2:free",            None, None),
    "or-mistral":         ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "mistralai/mistral-small-3.1-24b-instruct:free", None, None),
    "or-hermes":          ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "nousresearch/hermes-3-llama-3.1-405b:free",  None, None),
    # --- Cerebras ---
    "cerebras":           ("https://api.cerebras.ai/v1", "CEREBRAS_API_KEY", "llama-3.3-70b",                              14_000, 1_000_000),
    # --- Ollama (local) ---
    "ollama":             ("http://localhost:11434/v1", None, "llama3",                                                     None,  None),
}

LEGACY_ENV_PROVIDERS = {"custom", "cluster-vllm"}


def _legacy_env_config(provider: str):
    """Resolve explicit raw OpenAI-compatible endpoint settings."""
    base_url = os.getenv("LLM_BASE_URL", "").strip()
    api_key = os.getenv("LLM_API_KEY", "").strip()
    model = os.getenv("LLM_MODEL", "").strip()
    missing = [
        name
        for name, value in (
            ("LLM_BASE_URL", base_url),
            ("LLM_API_KEY", api_key),
            ("LLM_MODEL", model),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            f"LLM_PROVIDER={provider!r} requires explicit "
            + ", ".join(missing)
        )
    return base_url, api_key, model


def _resolve_provider():
    """Resolve (base_url, api_key, model) from LLM_PROVIDER or legacy env vars."""
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()

    if provider and provider in PROVIDERS:
        base_url, key_env, model, _, _ = PROVIDERS[provider]
        api_key = os.getenv(key_env, "") if key_env else "ollama"
        if key_env and not api_key.strip():
            raise RuntimeError(f"LLM_PROVIDER={provider!r} requires {key_env}")
        return base_url, api_key, model
    if provider and provider in LEGACY_ENV_PROVIDERS:
        return _legacy_env_config(provider)
    if provider:
        known = ", ".join(sorted([*PROVIDERS, *LEGACY_ENV_PROVIDERS]))
        raise ValueError(f"Unknown LLM_PROVIDER={provider!r}. Known providers: {known}")
    if os.getenv("NO_SILENT_FALLBACK", "").strip().lower() in {"1", "true", "yes", "on"}:
        raise RuntimeError("NO_SILENT_FALLBACK requires explicit LLM_PROVIDER")

    # Legacy fallback for non-eval/demo entrypoints that predate LLM_PROVIDER.
    return (
        os.getenv("LLM_BASE_URL", "https://api.cerebras.ai/v1"),
        os.getenv("LLM_API_KEY", "no-key-set"),
        os.getenv("LLM_MODEL", "llama-3.3-70b"),
    )


def _max_completion_tokens() -> int | None:
    raw = os.getenv("LLM_MAX_COMPLETION_TOKENS", "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"LLM_MAX_COMPLETION_TOKENS must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError("LLM_MAX_COMPLETION_TOKENS must be positive")
    return value


def _uses_openrouter(base_url: str) -> bool:
    return "openrouter.ai" in base_url.lower()


def _csv_env(name: str) -> list[str]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _openrouter_extra_body(max_completion_tokens: int | None = None) -> dict:
    """OpenRouter run controls that prevent implicit backend changes."""
    provider = {
        "allow_fallbacks": False,
        "require_parameters": True,
    }
    for env_name, field in (
        ("OPENROUTER_PROVIDER_ORDER", "order"),
        ("OPENROUTER_PROVIDER_ONLY", "only"),
        ("OPENROUTER_PROVIDER_IGNORE", "ignore"),
    ):
        values = _csv_env(env_name)
        if values:
            provider[field] = values
    extra_body = {
        "provider": provider,
    }
    if max_completion_tokens is not None:
        # ChatOpenAI rewrites max_tokens to max_completion_tokens, but
        # OpenRouter enforces the legacy max_tokens field for these models.
        extra_body["max_tokens"] = max_completion_tokens
    return extra_body


def get_provider_info() -> dict:
    """Return current provider name, model, and rate limits (for eval logging)."""
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if provider and provider in PROVIDERS:
        base_url, key_env, model, rpd, tpd = PROVIDERS[provider]
        return {"provider": provider, "model": model, "rpd": rpd, "tpd": tpd}
    if provider and provider in LEGACY_ENV_PROVIDERS:
        return {
            "provider": provider,
            "model": os.getenv("LLM_MODEL", ""),
            "rpd": None,
            "tpd": None,
        }
    if provider:
        known = ", ".join(sorted([*PROVIDERS, *LEGACY_ENV_PROVIDERS]))
        raise ValueError(f"Unknown LLM_PROVIDER={provider!r}. Known providers: {known}")
    return {
        "provider": "custom",
        "model": os.getenv("LLM_MODEL", "llama-3.3-70b"),
        "rpd": None,
        "tpd": None,
    }


@functools.lru_cache(maxsize=64)
def get_llm(temperature: float = 0.0, _provider: str = ""):
    """Returns a cached ChatOpenAI instance configured from environment variables.

    The _provider param is resolved automatically from LLM_PROVIDER and included
    in the cache key so that switching providers mid-process returns a fresh client.
    Callers should not pass _provider directly — use the wrapper below.
    """
    from langchain_openai import ChatOpenAI

    base_url, api_key, model = _resolve_provider()
    max_completion_tokens = _max_completion_tokens()
    kwargs = {
        "base_url": base_url,
        "api_key": api_key,
        "model": model,
        "temperature": temperature,
        "timeout": 90,
        "max_retries": 1,
    }
    if max_completion_tokens is not None:
        if _uses_openrouter(base_url):
            kwargs["extra_body"] = _openrouter_extra_body(max_completion_tokens)
        else:
            kwargs["max_completion_tokens"] = max_completion_tokens
    elif _uses_openrouter(base_url):
        kwargs["extra_body"] = _openrouter_extra_body()
    return ChatOpenAI(**kwargs)


# Re-wrap so callers don't need to pass _provider manually
_get_llm_cached = get_llm

def get_llm(temperature: float = 0.0):
    """Returns a cached ChatOpenAI instance, keyed on (temperature, provider)."""
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    cache_provider = provider
    if provider.startswith("or-"):
        try:
            concurrency = int(os.getenv("EVAL_CONCURRENCY", "0") or "0")
        except ValueError:
            concurrency = 0
        if concurrency > 1:
            cache_provider = f"{provider}:thread:{threading.get_ident()}"
    return _get_llm_cached(temperature=temperature, _provider=cache_provider)


def list_providers():
    """Print all available providers with their rate limits."""
    print(f"\n{'Provider':<22} {'Model':<52} {'RPD':>8} {'TPD':>10}")
    print("-" * 95)
    for name, (_, _, model, rpd, tpd) in sorted(PROVIDERS.items()):
        rpd_str = f"{rpd:,}" if rpd else "local"
        tpd_str = f"{tpd:,}" if tpd else "local"
        print(f"{name:<22} {model:<52} {rpd_str:>8} {tpd_str:>10}")
    print()
    info = get_provider_info()
    print(f"Active: {info['provider']} ({info['model']})")


if __name__ == "__main__":
    list_providers()
