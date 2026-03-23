"""Graph nodes for routing, planning, synthesis, and replanning."""

from __future__ import annotations

from typing import Any, Dict, List, Literal

from .core import _llm_call, _parse_json, load_skill
from .models import (
    ACTION_DIRECT,
    ACTION_RAG,
    ACTION_WEB,
    ExperimentProfile,
    LegalAgentState,
    PlanningStep,
    STATUS_COMPLETED,
)
from .prompts import COLLECTIONS_REGISTRY, COMPLETENESS_CHECK_PROMPT, ROUTER_PROMPT
from .state_utils import append_audit_log, profile_from_state, research_question_from_state, serialise_step


def _sanitize_action_type(action_type: str, profile: ExperimentProfile) -> str:
    if action_type == ACTION_WEB and not profile.allow_web_search:
        return ACTION_RAG
    if action_type == ACTION_DIRECT and not profile.allow_direct_answer_fallback:
        return ACTION_RAG
    return action_type


def _truncate_for_brief(text: str, max_chars: int = 260) -> str:
    collapsed = " ".join((text or "").split())
    if len(collapsed) <= max_chars:
        return collapsed
    return collapsed[: max_chars - 3].rstrip() + "..."


def _format_attempt_path(trace: Dict[str, Any] | None) -> str:
    if not trace:
        return "not recorded"
    labels = []
    for attempt in trace.get("attempts", []):
        label = str(attempt.get("action_type", "unknown"))
        rewrite_attempt = int(attempt.get("rewrite_attempt", 0) or 0)
        if rewrite_attempt:
            label = f"{label}/rewrite{rewrite_attempt}"
        if not labels or labels[-1] != label:
            labels.append(label)
    return " -> ".join(labels) if labels else "not recorded"


def _fallback_follow_up_steps(
    missing_topics: List[Any],
    profile: ExperimentProfile,
    starting_step_id: int,
) -> List[PlanningStep]:
    fallback_steps = []
    for index, topic in enumerate(missing_topics, 1):
        fallback_steps.append(
            PlanningStep(
                step_id=starting_step_id + index,
                sub_question=topic if isinstance(topic, str) else str(topic),
                authority_target="follow-up research gap",
                retrieval_hints=[],
                action_type=_sanitize_action_type(ACTION_RAG, profile),
                max_retries=1,
            )
        )
    return fallback_steps


def _build_replanning_brief(state: LegalAgentState, final_answer: str) -> str:
    question = state["inputs"]["question"]
    table = state.get("planning_table", [])
    completeness = state.get("completeness_verdict", {})
    missing_topics = completeness.get("missing_topics", [])
    trace_by_step = {trace.get("step_id"): trace for trace in state.get("step_traces", [])}

    strong_takeaways = []
    weak_takeaways = []
    for step in table:
        if step.status != STATUS_COMPLETED:
            continue
        verdict = (step.judge_verdict or {}).get("sufficient", "full")
        base = f"Step {step.step_id}: {step.sub_question}"
        if verdict == "full":
            strong_takeaways.append(f"- {base}. Takeaway: {_truncate_for_brief(step.result)}")
            continue

        reason = (step.judge_verdict or {}).get("missing") or (step.judge_verdict or {}).get("reason") or "coverage gap"
        weak_takeaways.append(
            f"- {base}. Outcome: {verdict}. Gap: {_truncate_for_brief(str(reason), 180)}. "
            f"Tried: {_format_attempt_path(trace_by_step.get(step.step_id))}."
        )

    sections = [
        "This is a follow-up research round. Generate only new steps for unresolved gaps and avoid repeating fully covered issues.",
        f"Original question: {question}",
        f"Answer so far: {_truncate_for_brief(final_answer, 500) or '(no answer yet)'}",
    ]
    if missing_topics:
        sections.append("Open gaps:\n" + "\n".join(f"- {topic}" for topic in missing_topics))
    if strong_takeaways:
        sections.append("Covered so far:\n" + "\n".join(strong_takeaways))
    if weak_takeaways:
        sections.append("Weak or incomplete areas:\n" + "\n".join(weak_takeaways))
    return "\n\n".join(section for section in sections if section)


def router_node(state: LegalAgentState) -> Dict[str, Any]:
    from . import execution as execution_module

    profile = profile_from_state(state)
    print("\n--- ROUTER ---")
    if not profile.use_router:
        collections = ["legal_passages"]
        print(f"  Router disabled by profile -> {collections}")
        return {
            "collections": collections,
            "audit_log": append_audit_log(state, {"node": "router", "collections": collections, "mode": "disabled"}),
        }

    question = research_question_from_state(state)
    raw = execution_module._llm_call(ROUTER_PROMPT, question, label="router")
    parsed = _parse_json(raw)
    if parsed and "collections" in parsed:
        collections = [item for item in parsed["collections"] if item in COLLECTIONS_REGISTRY]
    else:
        collections = []
    if not collections:
        collections = ["legal_passages"]

    print(f"  Collections: {collections}")
    return {
        "collections": collections,
        "audit_log": append_audit_log(state, {"node": "router", "collections": collections}),
    }


def planner_node(state: LegalAgentState) -> Dict[str, Any]:
    from . import execution as execution_module

    profile = profile_from_state(state)
    print("\n--- PLANNER ---")
    question = research_question_from_state(state)
    existing_table = list(state.get("planning_table", []))
    replanning_brief = state.get("replanning_brief", "").strip()
    missing_topics = state.get("completeness_verdict", {}).get("missing_topics", [])
    is_follow_up = bool(replanning_brief and existing_table)
    starting_step_id = max((step.step_id for step in existing_table), default=0)

    if not profile.use_planner:
        fallback_steps = (
            _fallback_follow_up_steps(missing_topics, profile, starting_step_id)
            if is_follow_up and missing_topics
            else [PlanningStep(step_id=1, sub_question=question, action_type=ACTION_RAG, max_retries=1)]
        )
        first_step = fallback_steps[0]
        print(f"  Planner disabled by profile -> Step {first_step.step_id} [{first_step.action_type}]")
        return {
            "planning_table": existing_table + fallback_steps if is_follow_up else fallback_steps,
            "replanning_brief": "",
            "audit_log": append_audit_log(
                state,
                {
                    "node": "planner",
                    "mode": "disabled_follow_up" if is_follow_up else "disabled",
                    "steps_generated": len(fallback_steps),
                },
            ),
        }

    planner_input = f"Legal research question:\n{question}"
    if is_follow_up:
        planner_input += (
            "\n\nThis is a follow-up research round. Create only NEW self-contained research steps for the unresolved gaps below. "
            "Do not repeat issues already covered unless a prior step was partial and needs a narrower follow-up.\n\n"
            f"FOLLOW-UP RESEARCH BRIEF:\n{replanning_brief}"
        )

    raw = execution_module._llm_call(load_skill("planner"), planner_input, label="planner")
    parsed = _parse_json(raw)

    complexity = "moderate"
    steps: List[PlanningStep] = []
    if parsed and "steps" in parsed:
        complexity = parsed.get("complexity", "moderate")
        for index, item in enumerate(parsed["steps"], 1):
            step_id = starting_step_id + index if is_follow_up else index
            if isinstance(item, dict):
                steps.append(
                    PlanningStep(
                        step_id=step_id,
                        sub_question=item.get("sub_question", question),
                        authority_target=item.get("authority_target", ""),
                        retrieval_hints=item.get("retrieval_hints", []),
                        action_type=_sanitize_action_type(item.get("action_type", ACTION_RAG), profile),
                        max_retries=min(int(item.get("max_retries", 2)), 3),
                    )
                )
            else:
                steps.append(PlanningStep(step_id=step_id, sub_question=str(item)))
    else:
        print("  [planner] Parse failed -> fallback")
        if is_follow_up and missing_topics:
            steps = _fallback_follow_up_steps(missing_topics, profile, starting_step_id)
        else:
            steps = [PlanningStep(step_id=1, sub_question=question, action_type=ACTION_RAG)]

    max_steps = 3 if is_follow_up else 5
    if len(steps) > max_steps:
        print(f"  [planner] Capping {len(steps)} steps to {max_steps}")
        steps = steps[:max_steps]

    if is_follow_up:
        fully_covered = {
            step.sub_question.strip().casefold()
            for step in existing_table
            if (step.judge_verdict or {}).get("sufficient", "full") == "full"
        }
        filtered_steps = []
        for step in steps:
            if step.sub_question.strip().casefold() in fully_covered:
                continue
            filtered_steps.append(step)
        steps = filtered_steps or _fallback_follow_up_steps(missing_topics, profile, starting_step_id)

    print(f"  Complexity: {complexity} | Plan ({len(steps)} step{'s' if len(steps) != 1 else ''}):")
    for step in steps:
        print(f"    Step {step.step_id} [{step.action_type}] (max_retries={step.max_retries}): {step.sub_question}")

    return {
        "planning_table": existing_table + steps if is_follow_up else steps,
        "replanning_brief": "",
        "audit_log": append_audit_log(
            state,
            {
                "node": "planner",
                "mode": "follow_up" if is_follow_up else "initial",
                "complexity": complexity,
                "steps_generated": len(steps),
                "steps": [serialise_step(step) for step in steps],
            },
        ),
    }


def synthesizer_node(state: LegalAgentState) -> Dict[str, Any]:
    from . import execution as execution_module

    profile = profile_from_state(state)
    print("\n--- SYNTHESIZER ---")
    table = state["planning_table"]
    evidence_store = state.get("evidence_store", [])
    question = state["inputs"]["question"]
    completed = [step for step in table if step.status == STATUS_COMPLETED]
    parallel_round = state.get("parallel_round", 1)

    step_summaries = "\n\n".join(
        f"### Research Step {step.step_id} ({step.action_type}): {step.sub_question}\n{step.result}"
        for step in completed
    )
    evidence_index = "\n".join(
        f"[Evidence {index + 1}] (sources={','.join(item.get('retrieval_aspects', [])) or item.get('source', 'unknown')}): "
        f"{item['text']}"
        for index, item in enumerate(evidence_store)
    )
    step_verdicts = "\n".join(
        f"- Step {step.step_id} [{step.action_type}] {step.sub_question}: "
        f"judge={(step.judge_verdict or {}).get('sufficient', 'unknown')}; "
        f"gap={((step.judge_verdict or {}).get('missing') or (step.judge_verdict or {}).get('reason') or 'none')}"
        for step in completed
    )
    user_prompt = (
        f"ORIGINAL QUESTION:\n{question}\n\n"
        f"RESEARCH FINDINGS:\n{step_summaries}\n\n"
        f"STEP VERDICTS:\n{step_verdicts or '(none)'}\n\n"
        f"EVIDENCE INDEX (cite as [Evidence N]):\n{evidence_index}"
    )
    final_answer = execution_module._llm_call(load_skill("synthesizer"), user_prompt, label="synthesizer")
    print(f"  Final answer: {len(final_answer)} chars")

    if not profile.use_completeness_loop:
        completeness = {"complete": True, "reasoning": "completeness loop disabled by profile", "missing_topics": []}
    else:
        max_rounds = state.get("run_config", {}).get("max_parallel_rounds", 3)
        if parallel_round >= max_rounds:
            print(f"  Max research rounds ({max_rounds}) reached -> marking complete")
            completeness = {"complete": True, "reasoning": "max rounds reached", "missing_topics": []}
        else:
            prompt = (
                f"ORIGINAL QUESTION:\n{question}\n\n"
                f"RESEARCH FINDINGS:\n{step_summaries}\n\n"
                f"STEP VERDICTS:\n{step_verdicts or '(none)'}\n\n"
                f"SYNTHESIZED ANSWER:\n{final_answer}\n\n"
                f"Research round: {parallel_round} of {max_rounds} max"
            )
            raw = execution_module._llm_call(COMPLETENESS_CHECK_PROMPT, prompt, label="completeness")
            parsed = _parse_json(raw)
            if parsed and "complete" in parsed:
                completeness = {
                    "complete": parsed.get("complete", True),
                    "reasoning": parsed.get("reasoning", ""),
                    "missing_topics": parsed.get("missing_topics", []),
                }
            else:
                completeness = {
                    "complete": True,
                    "reasoning": "parse failure — defaulting to complete",
                    "missing_topics": [],
                }

    verdict = "COMPLETE" if completeness.get("complete", True) else "INCOMPLETE"
    print(f"  Completeness: {verdict} — {completeness.get('reasoning', '')}")
    if completeness.get("missing_topics"):
        print(f"  Missing topics: {completeness['missing_topics']}")

    return {
        "final_answer": final_answer,
        "completeness_verdict": completeness,
        "parallel_round": parallel_round + 1,
        "audit_log": append_audit_log(
            state,
            {
                "node": "synthesizer",
                "completed_steps": len(completed),
                "parallel_round": parallel_round,
                "completeness": completeness,
            },
        ),
    }


def replanner_node(state: LegalAgentState) -> Dict[str, Any]:
    print("\n--- REPLANNER ---")
    completeness = state.get("completeness_verdict", {})
    missing_topics = completeness.get("missing_topics", [])
    if not missing_topics:
        print("  No missing topics -> nothing to add")
        return {}

    brief = _build_replanning_brief(state, state.get("final_answer", ""))
    print("  Follow-up brief prepared for planner")
    for topic in missing_topics:
        print(f"    Gap: {topic}")

    return {
        "replanning_brief": brief,
        "audit_log": append_audit_log(
            state,
            {
                "node": "replanner",
                "missing_topics": missing_topics,
                "brief_preview": _truncate_for_brief(brief, 240),
            },
        ),
    }


def route_after_synthesizer(state: LegalAgentState) -> Literal["replanner_node", "__end__"]:
    completeness = state.get("completeness_verdict", {})
    if completeness.get("complete", True):
        print("  -> END")
        return "__end__"
    print("  -> replanner")
    return "replanner_node"


__all__ = [
    "planner_node",
    "replanner_node",
    "route_after_synthesizer",
    "router_node",
    "synthesizer_node",
]
