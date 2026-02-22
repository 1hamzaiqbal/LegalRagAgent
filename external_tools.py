"""Placeholder external tools for teammate's Playwright-based web lookup API.

These stubs are decorated with @tool so they can be bound to an LLM via
`llm.bind_tools()` as soon as the real API is connected. To activate,
set EXTERNAL_TOOLS_BASE_URL and EXTERNAL_TOOLS_API_KEY in your .env file,
then swap the stub implementations below for real HTTP calls.
"""

import os
from typing import Any, Dict, List

from langchain_core.tools import tool

EXTERNAL_TOOLS_BASE_URL = os.getenv("EXTERNAL_TOOLS_BASE_URL", "")
EXTERNAL_TOOLS_API_KEY = os.getenv("EXTERNAL_TOOLS_API_KEY", "")


@tool
def web_search(query: str) -> str:
    """Search the web for legal information using an external Playwright-based service.

    Args:
        query: The search query string.

    Returns:
        Search results as formatted text.
    """
    return (
        f"[PLACEHOLDER] Web search not yet connected. "
        f"Query: {query!r}. "
        f"Configure EXTERNAL_TOOLS_BASE_URL to enable."
    )


@tool
def web_scrape(url: str) -> str:
    """Scrape a web page for content using an external Playwright-based service.

    Args:
        url: The URL to scrape.

    Returns:
        Page content as text.
    """
    return (
        f"[PLACEHOLDER] Web scrape not yet connected. "
        f"URL: {url!r}. "
        f"Configure EXTERNAL_TOOLS_BASE_URL to enable."
    )


@tool
def external_api_call(endpoint: str, payload: str) -> str:
    """Make a generic API call to the teammate's external tool wrapper.

    Args:
        endpoint: The API endpoint path (e.g., '/search', '/summarize').
        payload: JSON-encoded payload string for the request.

    Returns:
        API response as text.
    """
    return (
        f"[PLACEHOLDER] External API call not yet connected. "
        f"Endpoint: {endpoint!r}, Payload: {payload!r}. "
        f"Configure EXTERNAL_TOOLS_BASE_URL to enable."
    )


def get_external_tools() -> List:
    """Return list of all available external tools for binding to an LLM."""
    return [web_search, web_scrape, external_api_call]
