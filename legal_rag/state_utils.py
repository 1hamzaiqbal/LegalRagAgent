"""Shared state helpers used across the runtime modules."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

from .models import ExperimentProfile, LegalAgentState, PlanningStep


def profile_from_state(state: LegalAgentState) -> ExperimentProfile:
    """Deserialize the active experiment profile from graph state."""
    return ExperimentProfile(**state["profile"])


def research_question_from_state(state: LegalAgentState) -> str:
    """Return the question nodes should use for planning and retrieval."""
    return state["inputs"].get("research_question") or state["inputs"]["question"]


def append_audit_log(state: LegalAgentState, entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Append a timestamped entry to the state's audit log."""
    entry.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    return state.get("audit_log", []) + [entry]


def serialise_step(step: PlanningStep) -> Dict[str, Any]:
    """Serialize a planning step for logs and artifacts."""
    return step.model_dump()
