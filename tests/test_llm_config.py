from __future__ import annotations

import pytest

import llm_config


def test_unknown_provider_does_not_fall_back_to_legacy_env(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "groq-llama70bb")
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:9999/v1")
    monkeypatch.setenv("LLM_API_KEY", "dummy")
    monkeypatch.setenv("LLM_MODEL", "wrong-model")

    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        llm_config._resolve_provider()
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        llm_config.get_provider_info()


def test_cluster_vllm_requires_explicit_legacy_env(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "cluster-vllm")
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    with pytest.raises(RuntimeError, match="requires explicit"):
        llm_config._resolve_provider()


def test_cluster_vllm_uses_explicit_legacy_env(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "cluster-vllm")
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setenv("LLM_API_KEY", "dummy")
    monkeypatch.setenv("LLM_MODEL", "google/gemma-4-E4B-it")

    assert llm_config._resolve_provider() == (
        "http://127.0.0.1:8000/v1",
        "dummy",
        "google/gemma-4-E4B-it",
    )
    assert llm_config.get_provider_info()["provider"] == "cluster-vllm"
    assert llm_config.get_provider_info()["model"] == "google/gemma-4-E4B-it"


def test_registered_provider_requires_expected_key(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "groq-llama70b")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="requires GROQ_API_KEY"):
        llm_config._resolve_provider()


def test_no_silent_fallback_requires_explicit_provider(monkeypatch) -> None:
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("NO_SILENT_FALLBACK", "1")
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:9999/v1")
    monkeypatch.setenv("LLM_API_KEY", "dummy")
    monkeypatch.setenv("LLM_MODEL", "wrong-model")

    with pytest.raises(RuntimeError, match="requires explicit LLM_PROVIDER"):
        llm_config._resolve_provider()


def test_openrouter_extra_body_disables_provider_fallbacks() -> None:
    assert llm_config._openrouter_extra_body(1024) == {
        "provider": {
            "allow_fallbacks": False,
            "require_parameters": True,
        },
        "max_tokens": 1024,
    }
