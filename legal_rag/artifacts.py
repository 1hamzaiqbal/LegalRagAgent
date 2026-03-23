"""Structured artifact helpers for runtime executions."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

from .models import ExecutionResult, PlanningStep


def _serialise_step(step: PlanningStep) -> Dict[str, Any]:
    return step.model_dump()


def build_run_artifact(
    result: ExecutionResult,
    question: str,
    raw_question: str | None = None,
) -> Dict[str, Any]:
    """Build a structured artifact from an execution result."""
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "profile_name": result.profile.name,
        "profile_kind": result.profile.kind,
        "graph_mode": result.profile.step_execution_mode,
        "question": question,
        "research_question": raw_question or question,
        "provider": result.agent_metadata.get("provider", "unknown"),
        "model": result.agent_metadata.get("model", "unknown"),
        "collections": result.collections,
        "planning_table": [_serialise_step(step) for step in result.planning_table],
        "evidence_store": result.evidence_store,
        "completeness_verdict": result.completeness_verdict,
        "audit_log": result.audit_log,
        "replanning_brief": result.extra.get("replanning_brief", ""),
        "step_traces": result.run_artifact.get("step_traces", []),
        "prompt_versions": result.agent_metadata.get("prompt_versions", {}),
        "timings": result.extra.get("timings", {}),
        "llm_metrics": result.extra.get("llm_metrics", {}),
        "final_answer": result.final_answer,
        "extra": result.extra,
    }


def write_run_artifact(artifact: Dict[str, Any]) -> str:
    """Write an execution artifact to logs/run_artifacts/."""
    os.makedirs("logs/run_artifacts", exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_profile = artifact.get("profile_name", "run").replace("/", "_")
    path = os.path.join("logs", "run_artifacts", f"{timestamp}_{safe_profile}.json")
    payload = dict(artifact)
    payload["artifact_path"] = path
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return path
