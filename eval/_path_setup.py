"""Helpers for running eval scripts directly from the `eval/` directory."""

from __future__ import annotations

import os
import sys


def ensure_project_root_on_path() -> None:
    """Insert the project root so `uv run python eval/...` works reliably."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
