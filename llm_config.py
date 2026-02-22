"""LLM configuration — single entry point for all LLM calls.

Works with any OpenAI-compatible API: Cerebras (default), Google AI Studio, Groq, Together, Ollama, OpenAI.
Configure via env vars: LLM_BASE_URL, LLM_API_KEY, LLM_MODEL
"""

import functools
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# Defaults: Cerebras (14K requests/day, 1M tokens/day free tier)
DEFAULT_BASE_URL = "https://api.cerebras.ai/v1"
DEFAULT_MODEL = "llama-3.3-70b"


@functools.lru_cache(maxsize=4)
def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    """Returns a cached ChatOpenAI instance configured from environment variables."""
    return ChatOpenAI(
        base_url=os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL),
        api_key=os.getenv("LLM_API_KEY", "no-key-set"),
        model=os.getenv("LLM_MODEL", DEFAULT_MODEL),
        temperature=temperature,
    )
