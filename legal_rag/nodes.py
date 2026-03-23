"""Graph nodes for routing, planning, synthesis, and replanning."""

from __future__ import annotations

import re
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
    SUPPORT_PRIMARY,
    TERMINAL_ANSWERED,
    TERMINAL_LOOP_DISABLED,
    TERMINAL_MAX_ROUNDS,
    TERMINAL_PARSE_FAILURE,
    TERMINAL_STALLED,
)
from .prompts import (
    ARBITRATION_PROMPT,
    COLLECTIONS_REGISTRY,
    COMPLETENESS_CHECK_PROMPT,
    LLM_SNAP_PROMPT,
    MC_ANSWER_PROMPT,
    ROUTER_PROMPT,
)
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


_SIMILARITY_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "for",
    "in",
    "on",
    "under",
    "when",
    "does",
    "is",
    "are",
    "be",
    "what",
    "which",
    "whether",
    "how",
    "all",
    "with",
    "that",
    "this",
}


def _similarity_tokens(text: str) -> set[str]:
    tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(token) > 2 and token not in _SIMILARITY_STOPWORDS
    }
    return tokens


def _question_similarity(left: str, right: str) -> float:
    left_tokens = _similarity_tokens(left)
    right_tokens = _similarity_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _best_similarity(target: str, candidates: List[str]) -> float:
    if not candidates:
        return 0.0
    return max(_question_similarity(target, candidate) for candidate in candidates)


def _last_replanner_topics(state: LegalAgentState) -> List[str]:
    for entry in reversed(state.get("audit_log", [])):
        if entry.get("node") == "replanner":
            return [str(topic) for topic in entry.get("missing_topics", [])]
    return []


def _detect_round_stall(state: LegalAgentState, completeness: Dict[str, Any]) -> str | None:
    missing_topics = [str(topic) for topic in completeness.get("missing_topics", [])]
    if completeness.get("complete", True) or not missing_topics:
        return None

    parallel_round = int(state.get("parallel_round", 1) or 1)
    if parallel_round <= 1:
        return None

    traces = state.get("step_traces", [])
    current_round = [trace for trace in traces if int(trace.get("round", 0) or 0) == parallel_round]
    previous_round = [trace for trace in traces if int(trace.get("round", 0) or 0) == parallel_round - 1]
    if not current_round or not previous_round:
        return None

    prior_missing_topics = _last_replanner_topics(state)
    current_questions = [str(trace.get("sub_question", "")) for trace in current_round]
    previous_questions = [str(trace.get("sub_question", "")) for trace in previous_round]
    duplicate_score = (
        sum(_best_similarity(question, previous_questions) for question in current_questions) / len(current_questions)
        if current_questions
        else 0.0
    )
    missing_overlap = (
        sum(_best_similarity(topic, prior_missing_topics) for topic in missing_topics) / len(missing_topics)
        if prior_missing_topics and missing_topics
        else 0.0
    )

    strong_new_steps = [
        trace
        for trace in current_round
        if (trace.get("final_verdict") or {}).get("sufficient") == "full"
        and trace.get("support_level", SUPPORT_PRIMARY) == SUPPORT_PRIMARY
    ]

    evidence_counts = [
        int(entry.get("canonical_evidence_entries", 0) or 0)
        for entry in state.get("audit_log", [])
        if entry.get("node") == "executor_round"
    ]
    evidence_delta = evidence_counts[-1] - evidence_counts[-2] if len(evidence_counts) >= 2 else None

    if strong_new_steps:
        return None
    if duplicate_score >= 0.7 and missing_overlap >= 0.7 and evidence_delta is not None and evidence_delta <= 6:
        return "follow-up rounds are repeating near-duplicate questions without resolving the missing doctrinal gap"
    if missing_overlap >= 0.85 and evidence_delta is not None and evidence_delta <= 2:
        return "missing topics have not meaningfully changed and the latest round produced little new evidence"
    return None


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
        support_level = getattr(step, "support_level", SUPPORT_PRIMARY)
        base = f"Step {step.step_id}: {step.sub_question}"
        if verdict == "full" and support_level == SUPPORT_PRIMARY:
            strong_takeaways.append(f"- {base}. Takeaway: {_truncate_for_brief(step.result)}")
            continue

        reason = (step.judge_verdict or {}).get("missing") or (step.judge_verdict or {}).get("reason") or "coverage gap"
        weak_takeaways.append(
            f"- {base}. Outcome: {verdict}. Support level: {support_level}. Gap: {_truncate_for_brief(str(reason), 180)}. "
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


def _extract_choice_letters(question: str) -> List[str]:
    return re.findall(r"^\s*\(([A-Z])\)\s+", question, flags=re.MULTILINE)


def _normalise_mc_answer(final_answer: str, answer_letter: str) -> str:
    answer_block = f"**Answer: ({answer_letter})**"
    if re.search(r"\*\*Answer:\s*\(", final_answer):
        return re.sub(r"\*\*Answer:\s*\([^)]*\)\*\*", answer_block, final_answer, count=1)
    return f"{final_answer.rstrip()}\n\n{answer_block}"


def _adjudicate_multiple_choice(
    *,
    question: str,
    step_summaries: str,
    step_verdicts: str,
    final_answer: str,
) -> Dict[str, Any] | None:
    valid_letters = _extract_choice_letters(question)
    if not valid_letters:
        return None

    prompt = (
        f"QUESTION WITH ANSWER CHOICES:\n{question}\n\n"
        f"RESEARCH FINDINGS:\n{step_summaries}\n\n"
        f"STEP VERDICTS:\n{step_verdicts or '(none)'}\n\n"
        f"SYNTHESIZED ANALYSIS:\n{final_answer}\n\n"
        f"Valid answer letters: {', '.join(valid_letters)}"
    )
    raw = _llm_call(MC_ANSWER_PROMPT, prompt, label="mc/answer")
    parsed = _parse_json(raw)
    if not parsed:
        return None
    answer = str(parsed.get("answer", "")).strip().upper()
    if answer not in valid_letters:
        return None
    return {
        "answer": answer,
        "reasoning": str(parsed.get("reasoning", "")).strip(),
    }


def _extract_answer_letter(text: str) -> str:
    """Extract a letter answer from LLM output like **Answer: (B)**."""
    match = re.search(r"\*\*Answer:\s*\(([A-Z])\)\*\*", text)
    if match:
        return match.group(1)
    match = re.search(r"Answer:\s*\(?([A-Z])\)?", text)
    if match:
        return match.group(1)
    return ""


def llm_snap_node(state: LegalAgentState) -> Dict[str, Any]:
    """Quick LLM-only answer before any retrieval — provides a baseline for arbitration."""
    from . import execution as execution_module

    print("\n--- LLM SNAP ---")
    question = state["inputs"]["question"]
    answer = execution_module._llm_call(LLM_SNAP_PROMPT, question, label="llm_snap").strip()
    letter = _extract_answer_letter(answer)
    print(f"  LLM snap answer: {letter or '(no letter extracted)'}")
    return {
        "llm_snap_answer": answer,
        "llm_snap_letter": letter,
        "audit_log": append_audit_log(state, {"node": "llm_snap", "letter": letter}),
    }


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
        f"support={getattr(step, 'support_level', SUPPORT_PRIMARY)}; "
        f"origin={getattr(step, 'result_origin', step.action_type or 'unknown')}; "
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
    mc_decision = _adjudicate_multiple_choice(
        question=question,
        step_summaries=step_summaries,
        step_verdicts=step_verdicts,
        final_answer=final_answer,
    )
    if mc_decision:
        final_answer = _normalise_mc_answer(final_answer, mc_decision["answer"])

    # --- Arbitration: compare pipeline answer with LLM snap ---
    pipeline_letter = mc_decision["answer"] if mc_decision else _extract_answer_letter(final_answer)
    snap_letter = state.get("llm_snap_letter", "")
    snap_answer = state.get("llm_snap_answer", "")
    arbitration_decision = None

    if snap_letter and pipeline_letter and snap_letter != pipeline_letter:
        print(f"  DISAGREEMENT: LLM snap={snap_letter}, pipeline={pipeline_letter} -> arbitrating")
        arb_prompt = (
            f"QUESTION:\n{question}\n\n"
            f"ANALYSIS A (doctrinal reasoning, no retrieval):\n{snap_answer}\n\n"
            f"ANALYSIS B (research pipeline with retrieved evidence):\n{final_answer}\n\n"
            f"Analysis A chose: ({snap_letter})\n"
            f"Analysis B chose: ({pipeline_letter})\n"
        )
        arb_raw = execution_module._llm_call(ARBITRATION_PROMPT, arb_prompt, label="arbitration")
        arb_parsed = _parse_json(arb_raw)
        if arb_parsed and arb_parsed.get("chosen") in ("A", "B"):
            arbitration_decision = {
                "chosen": arb_parsed["chosen"],
                "reasoning": str(arb_parsed.get("reasoning", "")),
                "snap_letter": snap_letter,
                "pipeline_letter": pipeline_letter,
            }
            if arb_parsed["chosen"] == "A":
                print(f"  Arbitrator chose LLM snap ({snap_letter}) over pipeline ({pipeline_letter}): {arb_parsed.get('reasoning', '')}")
                final_answer = _normalise_mc_answer(final_answer, snap_letter)
            else:
                print(f"  Arbitrator kept pipeline ({pipeline_letter}): {arb_parsed.get('reasoning', '')}")
        else:
            print(f"  Arbitration parse failed -> keeping pipeline answer ({pipeline_letter})")
    elif snap_letter and pipeline_letter:
        print(f"  LLM snap and pipeline agree: ({pipeline_letter})")

    print(f"  Final answer: {len(final_answer)} chars")

    max_rounds = state.get("run_config", {}).get("max_parallel_rounds", 3)
    if not profile.use_completeness_loop:
        completeness = {
            "complete": True,
            "terminal": True,
            "terminal_reason": TERMINAL_LOOP_DISABLED,
            "reasoning": "completeness loop disabled by profile",
            "missing_topics": [],
        }
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
                "complete": bool(parsed.get("complete", True)),
                "reasoning": str(parsed.get("reasoning", "")),
                "missing_topics": parsed.get("missing_topics", []),
            }
            terminal_reason = None
            if completeness["complete"]:
                terminal_reason = TERMINAL_ANSWERED
            else:
                stall_reason = _detect_round_stall(state, completeness)
                if stall_reason:
                    reason = completeness.get("reasoning", "").strip()
                    completeness["reasoning"] = f"{reason} Stopping because research stalled: {stall_reason}".strip()
                    terminal_reason = TERMINAL_STALLED
                elif parallel_round >= max_rounds:
                    completeness["reasoning"] = (
                        f"{completeness.get('reasoning', '').strip()} Stopping because the maximum research rounds were reached."
                    ).strip()
                    terminal_reason = TERMINAL_MAX_ROUNDS
            completeness["terminal"] = terminal_reason is not None
            completeness["terminal_reason"] = terminal_reason
        else:
            completeness = {
                "complete": True,
                "terminal": True,
                "terminal_reason": TERMINAL_PARSE_FAILURE,
                "reasoning": "parse failure — defaulting to complete",
                "missing_topics": [],
            }

    verdict = "COMPLETE" if completeness.get("complete", True) else "INCOMPLETE"
    terminal_suffix = ""
    if completeness.get("terminal_reason"):
        terminal_suffix = f" | terminal={completeness['terminal_reason']}"
    print(f"  Completeness: {verdict}{terminal_suffix} — {completeness.get('reasoning', '')}")
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
                "mc_decision": mc_decision,
                "arbitration": arbitration_decision,
                "terminal_reason": completeness.get("terminal_reason"),
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
    if completeness.get("terminal", completeness.get("complete", True)):
        print("  -> END")
        return "__end__"
    print("  -> replanner")
    return "replanner_node"


__all__ = [
    "llm_snap_node",
    "planner_node",
    "replanner_node",
    "route_after_synthesizer",
    "router_node",
    "synthesizer_node",
]
