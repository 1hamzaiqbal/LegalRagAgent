"""Runtime entrypoints for profile-driven execution.

Read order for the runtime:
1. `main.py` for CLI entrypoints and compatibility exports
2. this file for graph construction and top-level orchestration
3. `legal_rag/models.py` and `legal_rag/profiles.py` for schema/config
4. `legal_rag/execution.py` for node behavior and step execution
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any, Dict

from langgraph.graph import END, START, StateGraph

from llm_config import get_provider_info

from .artifacts import build_run_artifact, write_run_artifact
from .core import _get_metrics
from .execution import (
    execute_round_node,
    planner_node,
    replanner_node,
    route_after_execution_round,
    route_after_synthesizer,
    router_node,
    synthesizer_node,
)
from .models import ExecutionResult, ExperimentProfile, LegalAgentState, PROFILE_KIND_FULL
from .prompts import inline_prompt_versions
from .profiles import get_profile


def build_graph(profile: str | ExperimentProfile = "full_parallel") -> Any:
    """Build the shared graph used by full profiles."""
    resolved = get_profile(profile)
    if resolved.kind != PROFILE_KIND_FULL:
        raise ValueError(f"Profile '{resolved.name}' does not use the LangGraph runtime")

    workflow = StateGraph(LegalAgentState)
    workflow.add_node("router_node", router_node)
    workflow.add_node("planner_node", planner_node)
    workflow.add_node("execute_round_node", execute_round_node)
    workflow.add_node("synthesizer_node", synthesizer_node)
    workflow.add_node("replanner_node", replanner_node)

    workflow.add_edge(START, "router_node")
    workflow.add_edge("router_node", "planner_node")
    workflow.add_edge("planner_node", "execute_round_node")
    workflow.add_conditional_edges(
        "execute_round_node",
        route_after_execution_round,
        ["execute_round_node", "synthesizer_node"],
    )
    workflow.add_conditional_edges(
        "synthesizer_node",
        route_after_synthesizer,
        ["replanner_node", "__end__"],
    )
    workflow.add_edge("replanner_node", "planner_node")
    return workflow.compile()


def _initial_state(question: str, raw_question: str, profile: ExperimentProfile, max_steps: int) -> LegalAgentState:
    provider = get_provider_info()
    return {
        "agent_metadata": {
            "provider": provider.get("provider", "unknown"),
            "model": provider.get("model", "unknown"),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "profile_name": profile.name,
            "prompt_versions": inline_prompt_versions(profile),
        },
        "inputs": {
            "question": question,
            "research_question": raw_question,
        },
        "run_config": {
            "max_steps": max_steps,
            "max_parallel_rounds": 3,
        },
        "profile": profile.to_state_dict(),
        "collections": [],
        "planning_table": [],
        "evidence_store": [],
        "final_answer": "",
        "audit_log": [],
        "completeness_verdict": {},
        "parallel_round": 1,
        "replanning_brief": "",
        "step_traces": [],
        "run_artifact": {},
    }


def _build_result_from_state(
    state: LegalAgentState,
    profile: ExperimentProfile,
    question: str,
    raw_question: str,
    artifact_path: str | None,
) -> ExecutionResult:
    result = ExecutionResult(
        profile=profile,
        final_answer=state.get("final_answer", ""),
        planning_table=state.get("planning_table", []),
        evidence_store=state.get("evidence_store", []),
        audit_log=state.get("audit_log", []),
        completeness_verdict=state.get("completeness_verdict", {}),
        collections=state.get("collections", []),
        parallel_round=state.get("parallel_round", 1),
        agent_metadata=state.get("agent_metadata", {}),
        artifact_path=artifact_path,
        extra={"replanning_brief": state.get("replanning_brief", "")},
    )
    result.run_artifact = build_run_artifact(result, question=question, raw_question=raw_question)
    result.run_artifact["step_traces"] = state.get("step_traces", [])
    return result


def _attach_run_metadata(
    result: ExecutionResult,
    *,
    started_at: datetime,
    elapsed_sec: float,
    start_metrics: Dict[str, int],
) -> None:
    finished_at = datetime.now(timezone.utc)
    end_metrics = _get_metrics()
    llm_metrics = {
        "calls": end_metrics["count"] - start_metrics["count"],
        "input_tokens": end_metrics["input_tokens"] - start_metrics["input_tokens"],
        "output_tokens": end_metrics["output_tokens"] - start_metrics["output_tokens"],
    }
    timings = {
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "elapsed_sec": round(elapsed_sec, 3),
    }
    result.agent_metadata["finished_at"] = finished_at.isoformat()
    result.extra["timings"] = timings
    result.extra["llm_metrics"] = llm_metrics
    result.run_artifact["timings"] = timings
    result.run_artifact["llm_metrics"] = llm_metrics
    result.run_artifact["extra"] = result.extra


def _print_result(result: ExecutionResult) -> None:
    print(f"\n{'=' * 80}")
    print("FINAL ANSWER:")
    print(f"{'=' * 80}")
    print(result.final_answer or "(no answer generated)")
    print(f"\n{'=' * 80}")
    print("RUN SUMMARY")
    print(f"{'=' * 80}")
    print(f"Profile         : {result.profile.name}")
    print(f"Mode            : {result.profile.step_execution_mode}")
    print(f"Steps completed : {sum(1 for step in result.planning_table if step.status == 'completed')}")
    print(f"Evidence pieces : {len(result.evidence_store)}")
    print(f"Rounds          : {result.parallel_round - 1}")
    completeness = result.completeness_verdict or {}
    print(f"Answered        : {'yes' if completeness.get('complete', True) else 'no'}")
    if completeness.get("terminal_reason"):
        print(f"Terminal reason : {completeness.get('terminal_reason')}")
    if result.artifact_path:
        print(f"Artifact        : {result.artifact_path}")
    print("\nStep breakdown:")
    evidence_map = {item["idx"]: item for item in result.evidence_store}
    for step in result.planning_table:
        verdict = step.judge_verdict or {}
        sufficient = verdict.get("sufficient", "full")
        judge_label = "full" if sufficient == "full" else ("partial" if sufficient == "partial" else "insufficient")
        support_level = getattr(step, "support_level", "primary")
        origin = getattr(step, "result_origin", step.action_type)
        sources = {}
        for evidence_id in step.evidence_ids:
            evidence = evidence_map.get(evidence_id)
            if not evidence:
                continue
            src = evidence.get("source", "unknown")
            sources[src] = sources.get(src, 0) + 1
        summary = ", ".join(f"{count} {src}" for src, count in sources.items()) if sources else "none"
        print(
            f"  Step {step.step_id:>4} [{step.action_type:<14}] "
            f"conf={step.confidence:.3f} judge={judge_label} support={support_level} "
            f"origin={origin} evidence=[{summary}]"
        )
    print(f"{'=' * 80}\n")


def run_experiment(
    question: str,
    profile: str | ExperimentProfile = "full_parallel",
    *,
    raw_question: str | None = None,
    max_steps: int = 7,
    write_run_artifact: bool | None = None,
    print_output: bool = True,
) -> ExecutionResult:
    """Run either a baseline profile or a full pipeline profile."""
    resolved = get_profile(profile)
    raw_research_question = raw_question or question
    started_at = datetime.now(timezone.utc)
    started_perf = time.perf_counter()
    start_metrics = _get_metrics()

    if resolved.kind != PROFILE_KIND_FULL:
        from .baselines import run_llm_only, run_rag_baseline

        if resolved.kind == "baseline_llm":
            result = run_llm_only(question, resolved, raw_question=raw_research_question)
        else:
            result = run_rag_baseline(question, resolved, raw_question=raw_research_question)
        artifact = build_run_artifact(result, question=question, raw_question=raw_research_question)
        result.run_artifact = artifact
        _attach_run_metadata(
            result,
            started_at=started_at,
            elapsed_sec=time.perf_counter() - started_perf,
            start_metrics=start_metrics,
        )
        should_write = resolved.write_run_artifact if write_run_artifact is None else write_run_artifact
        if should_write:
            result.artifact_path = write_run_artifact_fn(artifact)
            result.run_artifact["artifact_path"] = result.artifact_path
        if print_output:
            _print_result(result)
        return result

    graph = build_graph(resolved)
    state = _initial_state(question, raw_research_question, resolved, max_steps=max_steps)

    if print_output:
        print(f"\n{'=' * 80}")
        print(f"QUESTION: {question}")
        print(f"PROFILE: {resolved.name}")
        print(f"{'=' * 80}")

    final_state = graph.invoke(state)
    result = _build_result_from_state(final_state, resolved, question, raw_research_question, artifact_path=None)
    _attach_run_metadata(
        result,
        started_at=started_at,
        elapsed_sec=time.perf_counter() - started_perf,
        start_metrics=start_metrics,
    )

    should_write = resolved.write_run_artifact if write_run_artifact is None else write_run_artifact
    if should_write:
        result.artifact_path = write_run_artifact_fn(result.run_artifact)
        result.run_artifact["artifact_path"] = result.artifact_path
    if print_output:
        _print_result(result)
    return result


def write_run_artifact_fn(artifact: Dict[str, Any]) -> str:
    """Small wrapper kept separate for test patching."""
    return write_run_artifact(artifact)


def run(
    question: str,
    max_steps: int = 7,
    *,
    profile: str | ExperimentProfile = "full_parallel",
    raw_question: str | None = None,
    print_output: bool = True,
) -> ExecutionResult:
    """Compatibility wrapper mirroring the old top-level run helper."""
    return run_experiment(
        question,
        profile=profile,
        raw_question=raw_question,
        max_steps=max_steps,
        print_output=print_output,
    )
