"""Round execution and per-step escalation helpers."""

from __future__ import annotations

import concurrent.futures
import copy
import hashlib
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Tuple

from .core import _llm_call, _normalise_confidence, _parse_json, load_skill
from .models import (
    ACTION_DIRECT,
    ACTION_RAG,
    ACTION_WEB,
    ExperimentProfile,
    LegalAgentState,
    PlanningStep,
    STATUS_COMPLETED,
    STATUS_PENDING,
    STEP_MODE_PARALLEL,
)
from .retrieval import _execute_direct_answer, _execute_rag_search, _execute_web_search
from .state_utils import append_audit_log, profile_from_state, research_question_from_state


@dataclass
class StepExecutionResult:
    """Normalized output of a single step execution."""

    step: PlanningStep
    evidence: List[Dict[str, Any]]
    trace: Dict[str, Any]


def _make_evidence_key(ev: Dict[str, Any]) -> str:
    idx = str(ev.get("idx", "")).strip()
    if idx:
        return idx
    digest = hashlib.sha1(
        f"{ev.get('source', '')}:{ev.get('text', '')}".encode("utf-8")
    ).hexdigest()[:12]
    return f"anon_{digest}"


def _summarise_attempt(
    action_type: str,
    sub_question: str,
    rewrite_attempt: int,
    elapsed_sec: float,
    confidence: float,
    evidence: List[Dict[str, Any]],
    verdict: Dict[str, Any],
    queries: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    return {
        "action_type": action_type,
        "sub_question": sub_question,
        "rewrite_attempt": rewrite_attempt,
        "elapsed_sec": round(elapsed_sec, 3),
        "confidence": confidence,
        "evidence_count": len(evidence),
        "judge_verdict": verdict,
        "queries": queries or {},
    }


def _build_trace(
    original_step: PlanningStep,
    final_step: PlanningStep,
    profile: ExperimentProfile,
    attempts: List[Dict[str, Any]],
    verdict: Dict[str, Any],
    round_number: int,
) -> Dict[str, Any]:
    return {
        "step_id": original_step.step_id,
        "round": round_number,
        "profile": profile.name,
        "initial_action_type": original_step.action_type,
        "final_action_type": final_step.action_type,
        "sub_question": final_step.sub_question,
        "initial_sub_question": original_step.sub_question,
        "authority_target": final_step.authority_target,
        "retrieval_hints": list(final_step.retrieval_hints),
        "attempt_count": len(attempts),
        "elapsed_sec_total": round(sum(item.get("elapsed_sec", 0.0) for item in attempts), 3),
        "attempts": attempts,
        "final_status": final_step.status,
        "final_verdict": verdict,
        "evidence_ids": list(final_step.evidence_ids),
    }


def _call_judge(
    step: PlanningStep,
    result: str,
    passages: List[str],
    question: str,
    profile: ExperimentProfile,
) -> Dict[str, Any]:
    from . import execution as execution_module

    if not profile.use_judge:
        return {
            "sufficient": "full",
            "reason": "judge disabled by profile",
            "missing": None,
            "suggested_rewrite": None,
        }

    skill_name = "verifier" if step.action_type == ACTION_DIRECT else "judge"
    passages_block = (
        "\n\n".join(f"[Passage {index + 1}] {text}" for index, text in enumerate(passages))
        if passages
        else "[No retrieved passages — evaluated against established legal doctrine]"
    )
    user_prompt = (
        f"ORIGINAL QUESTION:\n{question}\n\n"
        f"SUB-QUESTION:\n{step.sub_question}\n\n"
        f"ACTION TYPE: {step.action_type}\n"
        f"REWRITE ATTEMPT: {step.rewrite_attempt}\n\n"
        f"RETRIEVED PASSAGES:\n{passages_block}\n\n"
        f"ANSWER DRAFT:\n{result}"
    )
    raw = execution_module._llm_call(load_skill(skill_name), user_prompt, label=f"judge/{skill_name}")
    parsed = _parse_json(raw)
    if not parsed or "sufficient" not in parsed:
        return {
            "sufficient": "full",
            "reason": f"parse failure ({skill_name}) — defaulting to full",
            "missing": None,
            "suggested_rewrite": None,
        }

    raw_sufficient = parsed.get("sufficient", "full")
    if raw_sufficient == "partial":
        sufficient = "partial"
    elif raw_sufficient in (True, "full", "true"):
        sufficient = "full"
    else:
        sufficient = False

    return {
        "sufficient": sufficient,
        "reason": str(parsed.get("reason", "")),
        "missing": parsed.get("missing") or None,
        "suggested_rewrite": parsed.get("suggested_rewrite") or None,
    }


def _execute_step_with_escalation(
    step: PlanningStep,
    state: LegalAgentState,
    table_snapshot: List[PlanningStep],
    evidence_snapshot: List[Dict[str, Any]],
    profile: ExperimentProfile,
) -> StepExecutionResult:
    from . import execution as execution_module

    question = research_question_from_state(state)
    round_number = int(state.get("parallel_round", 1) or 1)
    current = step.model_copy()
    all_new_evidence: List[Dict[str, Any]] = []
    attempts: List[Dict[str, Any]] = []
    max_attempts = min(1 + current.max_retries, 4)

    for attempt_index in range(max_attempts):
        print(
            f"\n  [Step {current.step_id}] Attempt {attempt_index + 1} [{current.action_type}] "
            f"(rewrite_attempt={current.rewrite_attempt}): {current.sub_question}"
        )

        started = time.perf_counter()
        if current.action_type == ACTION_DIRECT:
            result, new_evidence, raw_logit, query_trace = execution_module._execute_direct_answer(current, state)
        elif current.action_type == ACTION_WEB:
            result, new_evidence, raw_logit, query_trace = execution_module._execute_web_search(current, state)
        else:
            result, new_evidence, raw_logit, query_trace = execution_module._execute_rag_search(
                current,
                state,
                table_snapshot,
                evidence_snapshot + all_new_evidence,
                profile,
            )
        elapsed = time.perf_counter() - started

        confidence = _normalise_confidence(raw_logit) if raw_logit != 0.0 else 0.0
        passage_texts = [item["text"] for item in new_evidence]
        verdict = execution_module._call_judge(current, result, passage_texts, question, profile)

        print(f"  [Step {current.step_id}] Confidence: {confidence:.3f}")
        print(f"  [Step {current.step_id}] Judge: sufficient={verdict['sufficient']} | {verdict.get('reason', '')}")

        attempts.append(
            _summarise_attempt(
                action_type=current.action_type,
                sub_question=current.sub_question,
                rewrite_attempt=current.rewrite_attempt,
                elapsed_sec=elapsed,
                confidence=confidence,
                evidence=new_evidence,
                verdict=verdict,
                queries=query_trace,
            )
        )
        all_new_evidence.extend(new_evidence)

        if verdict["sufficient"] in ("full", "partial", True):
            updated = current.model_copy(
                update={
                    "status": STATUS_COMPLETED,
                    "result": result,
                    "confidence": confidence,
                    "judge_verdict": verdict,
                }
            )
            print(f"  [Step {current.step_id}] Completed (judge={verdict['sufficient']})")
            return StepExecutionResult(
                step=updated,
                evidence=all_new_evidence,
                trace=_build_trace(step, updated, profile, attempts, verdict, round_number),
            )

        if current.action_type == ACTION_RAG and current.rewrite_attempt == 0:
            revised = verdict.get("suggested_rewrite") or current.sub_question
            print(f"  [Step {current.step_id}] Escalating: rag_search -> rewrite")
            current = current.model_copy(update={"sub_question": revised, "rewrite_attempt": 1})
            continue

        if current.action_type == ACTION_RAG and current.rewrite_attempt >= 1:
            if profile.allow_direct_answer_fallback:
                print(f"  [Step {current.step_id}] Escalating: rag_search -> direct_answer")
                current = current.model_copy(update={"action_type": ACTION_DIRECT, "rewrite_attempt": 0})
                continue
            if profile.allow_web_search:
                print(f"  [Step {current.step_id}] Escalating: rag_search -> web_search")
                current = current.model_copy(update={"action_type": ACTION_WEB, "rewrite_attempt": 0})
                continue
            print(f"  [Step {current.step_id}] No fallback allowed after rag_search exhaustion")
            break

        if current.action_type == ACTION_WEB:
            if profile.allow_direct_answer_fallback:
                print(f"  [Step {current.step_id}] Escalating: web_search -> direct_answer")
                current = current.model_copy(update={"action_type": ACTION_DIRECT, "rewrite_attempt": 0})
                continue
            print(f"  [Step {current.step_id}] No direct-answer fallback allowed after web_search")
            break

        if current.action_type == ACTION_DIRECT:
            print(f"  [Step {current.step_id}] All escalations exhausted, keeping best result")
            break

    updated = current.model_copy(
        update={
            "status": STATUS_COMPLETED,
            "result": result,
            "confidence": confidence,
            "judge_verdict": verdict,
        }
    )
    return StepExecutionResult(
        step=updated,
        evidence=all_new_evidence,
        trace=_build_trace(step, updated, profile, attempts, verdict, round_number),
    )


def _merge_round_results(
    table: List[PlanningStep],
    evidence_store: List[Dict[str, Any]],
    results: List[StepExecutionResult],
) -> Tuple[List[PlanningStep], List[Dict[str, Any]]]:
    canonical_store = [copy.deepcopy(item) for item in evidence_store]
    evidence_index = {_make_evidence_key(item): index for index, item in enumerate(canonical_store)}

    result_by_step = {item.step.step_id: item for item in results}
    ordered_step_ids = [step.step_id for step in table if step.step_id in result_by_step]

    for step_id in ordered_step_ids:
        result = result_by_step[step_id]
        canonical_ids = []
        for evidence in result.evidence:
            key = _make_evidence_key(evidence)
            canonical_ids.append(key)
            if key in evidence_index:
                target = canonical_store[evidence_index[key]]
                step_ids = set(target.get("step_ids", []))
                step_ids.update(evidence.get("step_ids", [result.step.step_id]))
                target["step_ids"] = sorted(step_ids)
                if evidence.get("retrieval_aspect"):
                    aspects = set(target.get("retrieval_aspects", []))
                    aspects.add(evidence["retrieval_aspect"])
                    target["retrieval_aspects"] = sorted(aspects)
                target["cross_encoder_score"] = max(
                    float(target.get("cross_encoder_score", 0.0)),
                    float(evidence.get("cross_encoder_score", 0.0)),
                )
                continue

            stored = copy.deepcopy(evidence)
            stored["idx"] = key
            if stored.get("retrieval_aspect"):
                stored["retrieval_aspects"] = [stored["retrieval_aspect"]]
            evidence_index[key] = len(canonical_store)
            canonical_store.append(stored)

        result.step = result.step.model_copy(update={"evidence_ids": canonical_ids})

    new_table = []
    for step in table:
        if step.step_id in result_by_step:
            new_table.append(result_by_step[step.step_id].step)
        else:
            new_table.append(step)

    return new_table, canonical_store


def execute_round_node(state: LegalAgentState) -> Dict[str, Any]:
    from . import execution as execution_module

    profile = profile_from_state(state)
    heading = "PARALLEL EXECUTOR" if profile.step_execution_mode == STEP_MODE_PARALLEL else "SEQUENTIAL EXECUTOR"
    print(f"\n--- {heading} ---")

    table = state["planning_table"]
    evidence_store = list(state.get("evidence_store", []))
    pending = [step for step in table if step.status == STATUS_PENDING]
    if not pending:
        print("  No pending steps.")
        return {}

    selected = pending if profile.step_execution_mode == STEP_MODE_PARALLEL else pending[:1]
    print(f"  Executing {len(selected)} step(s) from round snapshot")

    table_snapshot = [step.model_copy() for step in table]
    evidence_snapshot = [copy.deepcopy(item) for item in evidence_store]
    ordered_ids = [step.step_id for step in selected]
    results: List[StepExecutionResult] = []

    if profile.step_execution_mode == STEP_MODE_PARALLEL and len(selected) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(selected)) as pool:
            future_map = {
                pool.submit(
                    execution_module._execute_step_with_escalation,
                    step,
                    state,
                    table_snapshot,
                    evidence_snapshot,
                    profile,
                ): step.step_id
                for step in selected
            }
            for future in concurrent.futures.as_completed(future_map):
                results.append(future.result())
    else:
        for step in selected:
                results.append(
                execution_module._execute_step_with_escalation(
                    step, state, table_snapshot, evidence_snapshot, profile
                )
            )

    results.sort(key=lambda item: ordered_ids.index(item.step.step_id))
    new_table, new_evidence_store = _merge_round_results(table, evidence_store, results)
    for item in results:
        item.trace["sub_question"] = item.step.sub_question
        item.trace["final_action_type"] = item.step.action_type
        item.trace["evidence_ids"] = list(item.step.evidence_ids)

    completed_count = sum(1 for step in new_table if step.status == STATUS_COMPLETED)
    print(
        f"\n  Round complete: {len(results)} step(s) executed, "
        f"{completed_count} total completed, {len(new_evidence_store)} canonical evidence entries"
    )
    return {
        "planning_table": new_table,
        "evidence_store": new_evidence_store,
        "step_traces": state.get("step_traces", []) + [item.trace for item in results],
        "audit_log": append_audit_log(
            state,
            {
                "node": "executor_round",
                "mode": profile.step_execution_mode,
                "steps_executed": [item.step.step_id for item in results],
                "canonical_evidence_entries": len(new_evidence_store),
            },
        ),
    }


def route_after_execution_round(state: LegalAgentState) -> Literal["execute_round_node", "synthesizer_node"]:
    profile = profile_from_state(state)
    has_pending = any(step.status == STATUS_PENDING for step in state.get("planning_table", []))
    if profile.step_execution_mode == STEP_MODE_PARALLEL:
        print("  -> synthesizer")
        return "synthesizer_node"
    if has_pending:
        print("  -> execute_round")
        return "execute_round_node"
    print("  -> synthesizer")
    return "synthesizer_node"


__all__ = [
    "StepExecutionResult",
    "_call_judge",
    "_execute_step_with_escalation",
    "execute_round_node",
    "route_after_execution_round",
]
