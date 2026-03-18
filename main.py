"""Plan-and-Execute Legal RAG Agent.

Architecture (follows LangGraph plan-and-execute pattern):
  START → planner → executor → replanner → executor  (retry / next step)
                                          → synthesizer → END

Four top-level nodes:
  planner    — decomposes the question into an ordered planning table
  executor   — executes one step via direct_answer, rag_search, or web_search,
               then runs LLM judge to evaluate sufficiency
  replanner  — applies deterministic escalation (rag→rewrite→web→direct),
               then LLM for next/complete when judge says sufficient
  synthesizer — aggregates all completed steps into a final IRAC answer

All system prompts are loaded from skills/*.md at runtime.
"""

# Windows: prevent OpenMP segfault when PyTorch and sentence-transformers
# each load their own OpenMP runtime (libiomp5md.dll conflict).
# Must be set before any torch/transformers import.
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import functools
import json
import logging
import math
import re
import requests
import sys
import time
import tiktoken
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from llm_config import get_llm, get_provider_info
from rag_utils import compute_confidence, retrieve_documents_multi_query

logger = logging.getLogger(__name__)
_TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")


# ---------------------------------------------------------------------------
# 1. Constants & Data Models
# ---------------------------------------------------------------------------

# Action types — used by planner, executor, replanner, and escalation logic
ACTION_RAG = "rag_search"
ACTION_WEB = "web_search"
ACTION_DIRECT = "direct_answer"

# Step statuses
STATUS_PENDING = "pending"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"


class PlanningStep(BaseModel):
    """A single research step in the planning table."""

    step_id: int
    sub_question: str
    authority_target: str = ""
    retrieval_hints: List[str] = Field(default_factory=list)

    # Action routing — assigned by planner, updated by replanner on escalation
    action_type: Literal["direct_answer", "rag_search", "web_search"] = ACTION_RAG
    # Tracks how many times this step's query has been rewritten (not a new action type)
    rewrite_attempt: int = 0

    status: Literal["pending", "completed", "failed"] = STATUS_PENDING
    result: str = ""
    confidence: float = 0.0  # for logging only — not used for control flow
    evidence_ids: List[str] = Field(default_factory=list)
    retry_of: Optional[int] = None   # step_id this is a retry of
    judge_verdict: Optional[Dict[str, Any]] = None  # stored after judge evaluation


class LegalAgentState(TypedDict):
    """Shared state propagated across all graph nodes."""

    agent_metadata: Dict[str, Any]      # provider, model, call counts, timestamps
    inputs: Dict[str, Any]              # original question and any runtime inputs
    run_config: Dict[str, Any]          # max_steps, flags
    planning_table: List[PlanningStep]
    evidence_store: List[Dict[str, Any]]  # all retrieved passages (accumulated)
    final_answer: str
    audit_log: List[Dict[str, Any]]     # per-node trace entries


# ---------------------------------------------------------------------------
# 2. Core Helpers
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=8)
def load_skill(name: str) -> str:
    """Load a skill prompt from skills/<name>.md (cached after first read)."""
    path = os.path.join("skills", f"{name}.md")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"[WARNING: Skill file '{path}' not found]"


def _parse_json(text: str) -> Any:
    """Forgiving JSON parser: handles markdown fences, trailing commas, JS comments."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`")
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        pass

    for pattern in [r"(\[[\s\S]*\])", r"(\{[\s\S]*\})"]:
        match = re.search(pattern, cleaned)
        if match:
            candidate = match.group(1)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
                fixed = re.sub(r"//.*?$", "", fixed, flags=re.MULTILINE)
                try:
                    return json.loads(fixed)
                except json.JSONDecodeError:
                    pass
    return None


# ---------------------------------------------------------------------------
# Metrics Tracking
# ---------------------------------------------------------------------------

class MetricsState:
    def __init__(self):
        self.llm_call_counter = {"count": 0, "input_tokens": 0, "output_tokens": 0}


_metrics_state = MetricsState()


def _get_metrics():
    return _metrics_state.llm_call_counter


def _reset_llm_call_counter():
    _metrics_state.llm_call_counter = {"count": 0, "input_tokens": 0, "output_tokens": 0}


def _get_deepseek_balance() -> Dict[str, Any]:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()

    if "deepseek" not in provider or not api_key:
        return {"is_available": False}

    try:
        url = "https://api.deepseek.com/user/balance"
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()
        data = res.json()
        data["is_available"] = True
        return data
    except Exception as e:
        logger.warning(f"Failed to fetch DeepSeek balance: {e}")
        return {"is_available": False}


def _llm_call(system_prompt: str, user_prompt: str, label: str = "") -> str:
    """Invoke the LLM with retry on transient errors (429, connection, timeout).

    Gemma models do not support system messages via the OpenAI-compatible API;
    they are automatically merged into a single HumanMessage.
    """
    llm = get_llm()
    TRANSIENT = ("429", "connection", "timeout", "rate", "overloaded", "unavailable")

    for attempt in range(3):
        try:
            model_name = getattr(llm, "model_name", "") or ""
            if "gemma" in model_name.lower():
                combined = f"[Instructions]\n{system_prompt}\n\n[Query]\n{user_prompt}"
                messages = [HumanMessage(content=combined)]
            else:
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            response = llm.invoke(messages)
            content = response.content

            # Track metrics
            _metrics_state.llm_call_counter["count"] += 1
            try:
                _metrics_state.llm_call_counter["input_tokens"] += len(_TIKTOKEN_ENC.encode(system_prompt + user_prompt))
                _metrics_state.llm_call_counter["output_tokens"] += len(_TIKTOKEN_ENC.encode(content))
            except Exception:
                # Fallback to rough char-based estimate if tiktoken fails
                _metrics_state.llm_call_counter["input_tokens"] += len(system_prompt) + len(user_prompt)
                _metrics_state.llm_call_counter["output_tokens"] += len(content)

            if label:
                print(f"    [{label}] {len(content)} chars")
            return content

        except Exception as exc:
            err = str(exc).lower()
            is_transient = any(t in err for t in TRANSIENT)
            if is_transient and attempt < 2:
                wait = 5 * (attempt + 1)
                m = re.search(r"retry.after['\"]:\s*(\d+)", err)
                if m:
                    wait = int(m.group(1))
                print(f"    [{label}] Transient error (attempt {attempt+1}/3), retry in {wait}s: {exc}")
                time.sleep(wait)
            else:
                print(f"    [{label}] LLM error: {exc}")
                raise

    return ""


def _sigmoid(x: float) -> float:
    """Map a raw cross-encoder logit to [0.0, 1.0] via sigmoid."""
    return 1.0 / (1.0 + math.exp(-x))


def _normalise_confidence(raw_logit: float) -> float:
    """Clamp raw ms-marco cross-encoder logit to [0.0, 1.0]."""
    return round(_sigmoid(raw_logit), 4)


def _log(state: LegalAgentState, entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return updated audit_log list with a new timestamped entry appended."""
    entry.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    return state.get("audit_log", []) + [entry]


def _insert_retry(table: List[PlanningStep], retry_step: PlanningStep) -> None:
    """Insert retry_step before the first pending step (mutates table in place)."""
    first_pending_idx = next(
        (i for i, s in enumerate(table) if s.status == STATUS_PENDING),
        len(table),
    )
    table.insert(first_pending_idx, retry_step)


# ---------------------------------------------------------------------------
# 3. Web Search Wrapper
# ---------------------------------------------------------------------------

def web_search(query: str, k: int = 5) -> List[str]:
    """DDGS text search — only place the web-search client is called.

    Returns a list of result body strings (empty list on error).
    """
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            # Backward-compatible fallback for older environments.
            from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=k))

        snippets = []
        for r in results:
            title = (r.get("title") or "").strip()
            body = (r.get("body") or "").strip()
            href = (r.get("href") or "").strip()
            parts = [part for part in [title, body, href] if part]
            if parts:
                snippets.append("\n".join(parts))
        return snippets
    except Exception as exc:
        print(f"    [web_search] Error: {exc}")
        return []


# ---------------------------------------------------------------------------
# 4. Judge Helper
# ---------------------------------------------------------------------------

def _call_judge(
    step: PlanningStep,
    result: str,
    passages: List[str],
    question: str,
) -> Dict[str, Any]:
    """Call the LLM judge to evaluate whether this step's result is sufficient.

    Uses skills/judge.md for rag_search / web_search steps.
    Uses skills/verifier.md for direct_answer steps (no retrieved passages).

    Returns a dict with keys: sufficient (bool), reason (str), suggested_rewrite (str|None).
    """
    skill_name = "verifier" if step.action_type == ACTION_DIRECT else "judge"

    if passages:
        passages_block = "\n\n".join(
            f"[Passage {i + 1}] {p}" for i, p in enumerate(passages)
        )
    else:
        passages_block = "[No retrieved passages — evaluated against established legal doctrine]"

    user_prompt = (
        f"ORIGINAL QUESTION:\n{question}\n\n"
        f"SUB-QUESTION:\n{step.sub_question}\n\n"
        f"ACTION TYPE: {step.action_type}\n"
        f"REWRITE ATTEMPT: {step.rewrite_attempt}\n\n"
        f"RETRIEVED PASSAGES:\n{passages_block}\n\n"
        f"ANSWER DRAFT:\n{result}"
    )

    raw = _llm_call(load_skill(skill_name), user_prompt, label=f"judge/{skill_name}")
    parsed = _parse_json(raw)

    if not parsed or "sufficient" not in parsed:
        # Fail-open: default to sufficient to avoid infinite loops on parse errors
        return {
            "sufficient": True,
            "reason": f"parse failure ({skill_name}) — defaulting to sufficient",
            "suggested_rewrite": None,
        }

    return {
        "sufficient": bool(parsed.get("sufficient", True)),
        "reason": str(parsed.get("reason", "")),
        "suggested_rewrite": parsed.get("suggested_rewrite") or None,
    }


# ---------------------------------------------------------------------------
# 5. Execution Strategies
# ---------------------------------------------------------------------------

def _execute_direct_answer(
    step: PlanningStep,
    state: LegalAgentState,
) -> tuple[str, List[Dict[str, Any]], float]:
    """Answer using LLM reasoning from established legal doctrine (no retrieval).

    Returns (result_text, new_evidence, raw_logit).
    """
    question = state["inputs"]["question"]
    prompt = (
        f"Original legal research question: {question}\n\n"
        f"Sub-question to answer from established legal doctrine:\n{step.sub_question}\n\n"
        f"Authority target: {step.authority_target}\n\n"
        f"Provide a clear, well-grounded answer based on established legal doctrine. "
        f"Flag any uncertainty or contested areas explicitly."
    )
    result = _llm_call(load_skill("synthesize_and_cite"), prompt, label="executor/direct")
    return result, [], 0.0  # no passages retrieved, no raw cross-encoder logit


def _execute_rag_search(
    step: PlanningStep,
    state: LegalAgentState,
    table: List[PlanningStep],
    evidence_store: List[Dict[str, Any]],
) -> tuple[str, List[Dict[str, Any]], float]:
    """Retrieve passages from ChromaDB and synthesize a cited sub-answer.

    Returns (result_text, new_evidence, raw_logit).
    """
    # --- Query rewrite ---
    question = state["inputs"]["question"]
    rewrite_prompt = (
        f"Original legal research question: {question}\n\n"
        f"Sub-question: {step.sub_question}\n"
        f"Authority target: {step.authority_target}\n"
        f"Retrieval hints: {', '.join(step.retrieval_hints) if step.retrieval_hints else 'none'}"
    )
    raw_rewrite = _llm_call(load_skill("query_rewriter"), rewrite_prompt, label="executor/rewrite")
    parsed_rewrite = _parse_json(raw_rewrite)

    if parsed_rewrite and "primary" in parsed_rewrite:
        queries = [parsed_rewrite["primary"]] + parsed_rewrite.get("alternatives", [])
    else:
        queries = [step.sub_question]

    print(f"    Primary query: {queries[0]}")

    # --- Retrieval with cross-step deduplication ---
    prior_ids: set = set()
    for s in table:
        prior_ids.update(s.evidence_ids)
    for ev in evidence_store:
        prior_ids.add(ev.get("idx", ""))

    docs = retrieve_documents_multi_query(
        queries=queries,
        k=5,
        exclude_ids=prior_ids if prior_ids else None,
    )
    print(f"    Retrieved {len(docs)} passage(s)")

    # --- Build numbered evidence passages ---
    passages: List[str] = []
    new_evidence: List[Dict[str, Any]] = []
    for i, doc in enumerate(docs, 1):
        text = doc.page_content
        source = doc.metadata.get("source", "unknown")
        idx = str(doc.metadata.get("idx", f"step{step.step_id}_{i}"))
        passages.append(f"[Source {i}] ({source})\n{text}")
        new_evidence.append({
            "idx": idx,
            "text": text,
            "source": source,
            "step_id": step.step_id,
            "cross_encoder_score": doc.metadata.get("cross_encoder_score", 0.0),
        })

    # --- Sub-answer synthesis ---
    evidence_block = "\n\n".join(passages) if passages else "[No passages retrieved]"
    synth_prompt = (
        f"Original legal research question: {question}\n\n"
        f"Research sub-question: {step.sub_question}\n\n"
        f"Evidence passages:\n{evidence_block}"
    )
    result = _llm_call(load_skill("synthesize_and_cite"), synth_prompt, label="executor/synth")

    # --- Raw confidence logit for logging ---
    raw_logit = compute_confidence(queries[0], docs) if docs else 0.0

    return result, new_evidence, raw_logit


def _execute_web_search(
    step: PlanningStep,
    state: LegalAgentState,
) -> tuple[str, List[Dict[str, Any]], float]:
    """Search the web via DuckDuckGo and synthesize a cited sub-answer.

    Returns (result_text, new_evidence, raw_logit).
    """
    snippets = web_search(step.sub_question, k=5)
    print(f"    Web results: {len(snippets)} snippet(s)")

    if not snippets:
        return "[No web results found]", [], 0.0

    question = state["inputs"]["question"]
    passages = [f"[WebResult {i + 1}] {s}" for i, s in enumerate(snippets)]
    evidence_block = "\n\n".join(passages)
    synth_prompt = (
        f"Original legal research question: {question}\n\n"
        f"Research sub-question: {step.sub_question}\n\n"
        f"Web search results:\n{evidence_block}"
    )
    result = _llm_call(load_skill("synthesize_and_cite"), synth_prompt, label="executor/web")

    new_evidence = [
        {
            "idx": f"web_{step.step_id}_{i}",
            "text": s,
            "source": "web",
            "step_id": step.step_id,
            "cross_encoder_score": 0.0,  # no cross-encoder for web results
        }
        for i, s in enumerate(snippets)
    ]

    return result, new_evidence, 0.0  # no cross-encoder logit for web


# ---------------------------------------------------------------------------
# 6. Node: Planner
# ---------------------------------------------------------------------------

def planner_node(state: LegalAgentState) -> dict:
    """Decompose the original question into an ordered planning table.

    Loads: skills/planner.md
    Parses: sub_question, authority_target, retrieval_hints, action_type per step.
    """
    print("\n--- PLANNER ---")
    question = state["inputs"]["question"]

    raw = _llm_call(load_skill("planner"), f"Legal research question:\n{question}", label="planner")
    parsed = _parse_json(raw)

    steps: List[PlanningStep] = []
    if parsed and "steps" in parsed:
        for i, s in enumerate(parsed["steps"]):
            if isinstance(s, dict):
                steps.append(PlanningStep(
                    step_id=i + 1,
                    sub_question=s.get("sub_question", question),
                    authority_target=s.get("authority_target", ""),
                    retrieval_hints=s.get("retrieval_hints", []),
                    action_type=s.get("action_type", ACTION_RAG),
                ))
            else:
                steps.append(PlanningStep(step_id=i + 1, sub_question=str(s)))
    else:
        print("  [planner] Parse failed — single-step fallback")
        steps = [PlanningStep(step_id=1, sub_question=question)]

    print(f"  Plan ({len(steps)} step{'s' if len(steps) != 1 else ''}):")
    for s in steps:
        print(f"    Step {s.step_id} [{s.action_type}]: {s.sub_question}")

    return {
        "planning_table": steps,
        "audit_log": _log(state, {
            "node": "planner",
            "steps_generated": len(steps),
            "steps": [
                {"step_id": s.step_id, "action_type": s.action_type, "sub_question": s.sub_question}
                for s in steps
            ],
        }),
    }


# ---------------------------------------------------------------------------
# 7. Node: Executor
# ---------------------------------------------------------------------------

def executor_node(state: LegalAgentState) -> dict:
    """Execute the first pending planning step.

    Routes based on step.action_type:
      direct_answer — LLM reasoning from established doctrine (no retrieval)
      rag_search    — ChromaDB multi-query retrieval + cross-encoder reranking
      web_search    — DuckDuckGo search (isolated via web_search() wrapper)

    After execution, runs the LLM judge (skills/judge.md or skills/verifier.md)
    to evaluate retrieval sufficiency. The verdict drives replanner escalation.

    Confidence is computed and stored for logging only — not used for control flow.
    """
    print("\n--- EXECUTOR ---")
    table = state["planning_table"]
    evidence_store = list(state.get("evidence_store", []))
    question = state["inputs"]["question"]

    current = next((s for s in table if s.status == STATUS_PENDING), None)
    if current is None:
        print("  No pending steps.")
        return {}

    print(f"  Step {current.step_id} [{current.action_type}] (rewrite_attempt={current.rewrite_attempt}): {current.sub_question}")

    # --- Route to appropriate execution strategy ---
    if current.action_type == ACTION_DIRECT:
        result, new_evidence, raw_logit = _execute_direct_answer(current, state)
    elif current.action_type == ACTION_WEB:
        result, new_evidence, raw_logit = _execute_web_search(current, state)
    else:  # rag_search (default)
        result, new_evidence, raw_logit = _execute_rag_search(
            current, state, table, evidence_store
        )

    # Confidence — for logging only — not used for control flow
    confidence = _normalise_confidence(raw_logit) if raw_logit != 0.0 else 0.0
    print(f"  Confidence: {confidence:.3f} (raw logit: {raw_logit:.3f}) [for analysis only]")

    # --- Judge evaluation ---
    passage_texts = [ev["text"] for ev in new_evidence]
    verdict = _call_judge(current, result, passage_texts, question)
    print(f"  Judge: sufficient={verdict['sufficient']} | {verdict.get('reason', '')}")
    if verdict.get("suggested_rewrite"):
        print(f"  Judge rewrite suggestion: {verdict['suggested_rewrite']}")

    # --- Update the planning table ---
    new_table = [
        s.model_copy(update={
            "status": STATUS_COMPLETED,
            "result": result,
            "confidence": confidence,  # for logging only — not used for control flow
            "evidence_ids": [ev["idx"] for ev in new_evidence],
            "judge_verdict": verdict,
        }) if s.step_id == current.step_id else s
        for s in table
    ]

    return {
        "planning_table": new_table,
        "evidence_store": evidence_store + new_evidence,
        "audit_log": _log(state, {
            "node": "executor",
            "step_id": current.step_id,
            "action_type": current.action_type,
            "rewrite_attempt": current.rewrite_attempt,
            "docs_retrieved": len(new_evidence),
            "confidence": confidence,  # for analysis only
            "raw_logit": raw_logit,
            "judge_sufficient": verdict["sufficient"],
            "judge_reason": verdict.get("reason", ""),
            "judge_suggested_rewrite": verdict.get("suggested_rewrite"),
        }),
    }


# ---------------------------------------------------------------------------
# 8. Node: Replanner
# ---------------------------------------------------------------------------

def _replanner_llm_decide(
    state: LegalAgentState,
    last: PlanningStep,
    completed: List[PlanningStep],
    pending: List[PlanningStep],
    max_steps: int,
) -> tuple[str, str, str]:
    """Call replanner LLM (skills/replanner.md) when judge says sufficient.

    Returns (action, reasoning, revised_question).
    """
    question = state["inputs"]["question"]

    evidence_summary = "\n\n".join(
        f"Step {s.step_id} [{s.authority_target or 'research'}] ({s.action_type}):\n"
        f"Q: {s.sub_question}\nA: {s.result}"
        for s in completed
    )
    pending_summary = (
        "\n".join(f"  Step {s.step_id}: {s.sub_question}" for s in pending)
        if pending else "(none)"
    )

    user_prompt = (
        f"ORIGINAL QUESTION:\n{question}\n\n"
        f"RESEARCH COMPLETED SO FAR:\n{evidence_summary}\n\n"
        f"LAST STEP (Step {last.step_id}):\n"
        f"Sub-question: {last.sub_question}\n"
        f"Action type: {last.action_type}\n"
        f"Result: {last.result}\n\n"
        f"PENDING STEPS REMAINING:\n{pending_summary}\n\n"
        f"Completed: {len(completed)} / {max_steps} max"
    )

    raw = _llm_call(load_skill("replanner"), user_prompt, label="replanner")
    parsed = _parse_json(raw)

    if not parsed:
        fallback = "next" if pending else "complete"
        return fallback, "parse failure — fallback", ""

    action = parsed.get("action", "next" if pending else "complete")
    reasoning = parsed.get("reasoning", "")
    revised_q = parsed.get("revised_question", "")
    print(f"  LLM replanner: {action} — {reasoning}")
    return action, reasoning, revised_q


def replanner_node(state: LegalAgentState) -> dict:
    """Evaluate the last completed step and decide next action.

    Escalation chain (deterministic — no LLM call):
      1. rag_search + sufficient=False + rewrite_attempt==0
           → rewrite query, stay rag_search, rewrite_attempt=1
      2. rag_search + sufficient=False + rewrite_attempt>=1
           → escalate action_type to web_search, rewrite_attempt=0
      3. web_search + sufficient=False
           → escalate action_type to direct_answer, rewrite_attempt=0
      4. direct_answer + sufficient=False
           → log uncertainty, continue to LLM decision (next/complete)

    When judge says sufficient (or fallback):
      → LLM (skills/replanner.md) decides next / complete.
      → LLM retry is still honoured (once) as a fallback.
    """
    print("\n--- REPLANNER ---")
    table = state["planning_table"]
    question = state["inputs"]["question"]

    completed = [s for s in table if s.status == STATUS_COMPLETED]
    pending = [s for s in table if s.status == STATUS_PENDING]
    last = next((s for s in reversed(table) if s.status == STATUS_COMPLETED), None)

    if last is None:
        print("  No completed steps — completing immediately.")
        return {"audit_log": _log(state, {
            "node": "replanner", "action": "complete", "reason": "no completed steps",
        })}

    max_steps = state.get("run_config", {}).get("max_steps", 5)
    if len(completed) >= max_steps:
        print(f"  Max steps ({max_steps}) reached → complete")
        return {"audit_log": _log(state, {
            "node": "replanner", "action": "complete", "reason": "max_steps",
        })}

    verdict = last.judge_verdict or {"sufficient": True, "reason": "no verdict stored"}
    sufficient = verdict.get("sufficient", True)
    new_table = list(table)
    action: Optional[str] = None
    reasoning: str = ""

    # --- Deterministic escalation when judge says insufficient ---
    if not sufficient and last.action_type != ACTION_DIRECT:

        if last.action_type == ACTION_RAG and last.rewrite_attempt == 0:
            # First RAG failure → rewrite query, stay rag_search
            revised_q = verdict.get("suggested_rewrite") or last.sub_question
            retry_step = PlanningStep(
                step_id=last.step_id * 10 + 5,
                sub_question=revised_q,
                authority_target=last.authority_target,
                retrieval_hints=last.retrieval_hints,
                action_type=ACTION_RAG,
                rewrite_attempt=1,
                retry_of=last.step_id,
            )
            _insert_retry(new_table, retry_step)
            action = "retry"
            reasoning = f"rag_search rewrite (attempt 1): {revised_q}"
            print(f"  Escalation: rag_search → rewrite (step {retry_step.step_id})")
            print(f"  Judge reason: {verdict.get('reason', '')}")

        elif last.action_type == ACTION_RAG and last.rewrite_attempt >= 1:
            # Second RAG failure → escalate to web_search
            retry_step = PlanningStep(
                step_id=last.step_id * 10 + 5,
                sub_question=last.sub_question,
                authority_target=last.authority_target,
                retrieval_hints=last.retrieval_hints,
                action_type=ACTION_WEB,
                rewrite_attempt=0,
                retry_of=last.step_id,
            )
            _insert_retry(new_table, retry_step)
            action = "retry"
            reasoning = "rag_search exhausted → escalating to web_search"
            print(f"  Escalation: rag_search → web_search (step {retry_step.step_id})")
            print(f"  Judge reason: {verdict.get('reason', '')}")

        elif last.action_type == ACTION_WEB:
            # Web search failure → fall back to direct_answer with uncertainty flag
            retry_step = PlanningStep(
                step_id=last.step_id * 10 + 5,
                sub_question=last.sub_question,
                authority_target=last.authority_target,
                retrieval_hints=last.retrieval_hints,
                action_type=ACTION_DIRECT,
                rewrite_attempt=0,
                retry_of=last.step_id,
            )
            _insert_retry(new_table, retry_step)
            action = "retry"
            reasoning = "web_search failed → falling back to direct_answer with uncertainty flagging"
            print(f"  Escalation: web_search → direct_answer (step {retry_step.step_id})")
            print(f"  Judge reason: {verdict.get('reason', '')}")

    # --- LLM decision when judge is satisfied or escalation handled ---
    if action is None:
        if not sufficient and last.action_type == ACTION_DIRECT:
            print(f"  direct_answer insufficient (no further escalation) — deferring to LLM")

        llm_action, reasoning, revised_q = _replanner_llm_decide(
            state, last, completed, pending, max_steps
        )
        action = llm_action

        if action == "retry":
            # Honour LLM retry once as a fallback (e.g. LLM disagrees with judge)
            retry_count = sum(1 for s in table if s.retry_of == last.step_id)
            if retry_count >= 1:
                print(f"  Step {last.step_id} already retried — advancing instead.")
                action = "next" if pending else "complete"
            else:
                retry_step = PlanningStep(
                    step_id=last.step_id * 10 + 5,
                    sub_question=revised_q or last.sub_question,
                    authority_target=last.authority_target,
                    retrieval_hints=last.retrieval_hints,
                    action_type=last.action_type,
                    rewrite_attempt=last.rewrite_attempt,
                    retry_of=last.step_id,
                )
                _insert_retry(new_table, retry_step)
                print(f"  LLM retry step {retry_step.step_id}: {retry_step.sub_question}")

    return {
        "planning_table": new_table,
        "audit_log": _log(state, {
            "node": "replanner",
            "last_step_id": last.step_id,
            "last_action_type": last.action_type,
            "last_rewrite_attempt": last.rewrite_attempt,
            "judge_sufficient": sufficient,
            "judge_reason": verdict.get("reason", ""),
            "action": action,
            "reasoning": reasoning,
        }),
    }


# ---------------------------------------------------------------------------
# 9. Node: Synthesizer
# ---------------------------------------------------------------------------

def synthesizer_node(state: LegalAgentState) -> dict:
    """Aggregate all completed research steps into a final IRAC-style answer.

    Loads: skills/synthesizer.md
    Maps citations to evidence IDs from the evidence store.
    """
    print("\n--- SYNTHESIZER ---")
    table = state["planning_table"]
    evidence_store = state.get("evidence_store", [])
    question = state["inputs"]["question"]

    completed = [s for s in table if s.status == STATUS_COMPLETED]

    step_summaries = "\n\n".join(
        f"### Research Step {s.step_id} ({s.action_type}): {s.sub_question}\n{s.result}"
        for s in completed
    )

    # Build an evidence index the synthesizer can cite by number
    evidence_index = "\n".join(
        f"[Evidence {i + 1}] (step={ev['step_id']}, source={ev['source']}): "
        f"{ev['text']}"
        for i, ev in enumerate(evidence_store)
    )

    user_prompt = (
        f"ORIGINAL QUESTION:\n{question}\n\n"
        f"RESEARCH FINDINGS:\n{step_summaries}\n\n"
        f"EVIDENCE INDEX (cite as [Evidence N]):\n{evidence_index}"
    )

    final_answer = _llm_call(load_skill("synthesizer"), user_prompt, label="synthesizer")
    print(f"  Final answer: {len(final_answer)} chars")

    return {
        "final_answer": final_answer,
        "audit_log": _log(state, {
            "node": "synthesizer",
            "completed_steps": len(completed),
            "evidence_entries": len(evidence_store),
        }),
    }


# ---------------------------------------------------------------------------
# 10. Routing
# ---------------------------------------------------------------------------

def route_after_replanner(
    state: LegalAgentState,
) -> Literal["executor_node", "synthesizer_node"]:
    """Route after replanner based on the action recorded in the last audit entry."""
    audit_log = state.get("audit_log", [])
    last_replan = next(
        (a for a in reversed(audit_log) if a.get("node") == "replanner"), None
    )
    action = last_replan.get("action", "next") if last_replan else "next"

    has_pending = any(s.status == STATUS_PENDING for s in state.get("planning_table", []))

    if action == "complete" or not has_pending:
        print("  → synthesizer")
        return "synthesizer_node"
    print("  → executor")
    return "executor_node"


# ---------------------------------------------------------------------------
# 11. Graph
# ---------------------------------------------------------------------------

def build_graph() -> Any:
    """Compile the plan-and-execute LangGraph state machine."""
    workflow = StateGraph(LegalAgentState)

    workflow.add_node("planner_node", planner_node)
    workflow.add_node("executor_node", executor_node)
    workflow.add_node("replanner_node", replanner_node)
    workflow.add_node("synthesizer_node", synthesizer_node)

    # Fixed edges
    workflow.add_edge(START, "planner_node")
    workflow.add_edge("planner_node", "executor_node")
    workflow.add_edge("executor_node", "replanner_node")
    workflow.add_edge("synthesizer_node", END)

    # Conditional: replanner → executor (retry/next) | synthesizer (complete)
    workflow.add_conditional_edges(
        "replanner_node",
        route_after_replanner,
        ["executor_node", "synthesizer_node"],
    )

    return workflow.compile()


# ---------------------------------------------------------------------------
# 12. Runner
# ---------------------------------------------------------------------------

DEMO_QUERIES = {
    "simple": "What are the elements of a negligence claim?",
    "multi_hop": (
        "A police officer stopped a vehicle without reasonable suspicion, found drugs "
        "during a warrantless search, and the defendant seeks suppression. "
        "What are the defendant's strongest Fourth Amendment arguments and how would "
        "a court analyze application of the exclusionary rule?"
    ),
    "medium": "What is the standard for granting a preliminary injunction?",
}


def run(question: str, max_steps: int = 5) -> LegalAgentState:
    """Run the plan-and-execute agent and return the final state."""
    graph = build_graph()
    provider = get_provider_info()

    initial_state: LegalAgentState = {
        "agent_metadata": {
            "provider": provider.get("provider", "unknown"),
            "model": provider.get("model", "unknown"),
            "started_at": datetime.now(timezone.utc).isoformat(),
        },
        "inputs": {"question": question},
        "run_config": {"max_steps": max_steps},
        "planning_table": [],
        "evidence_store": [],
        "final_answer": "",
        "audit_log": [],
    }

    print(f"\n{'=' * 80}")
    print(f"QUESTION: {question}")
    print(f"{'=' * 80}")

    result = graph.invoke(initial_state)

    completed_steps = [s for s in result.get("planning_table", []) if s.status == STATUS_COMPLETED]
    evidence_count = len(result.get("evidence_store", []))

    print(f"\n{'=' * 80}")
    print("FINAL ANSWER:")
    print(f"{'=' * 80}")
    print(result.get("final_answer", "(no answer generated)"))

    print(f"\n{'=' * 80}")
    print("RUN SUMMARY")
    print(f"{'=' * 80}")
    print(f"Steps completed : {len(completed_steps)}")
    print(f"Evidence pieces : {evidence_count}")
    print(f"Audit entries   : {len(result.get('audit_log', []))}")

    print("\nStep breakdown:")
    for s in result.get("planning_table", []):
        verdict = s.judge_verdict or {}
        sufficient_str = "ok" if verdict.get("sufficient", True) else "insufficient"
        print(
            f"  Step {s.step_id:>4} [{s.action_type:<14}] retry={s.rewrite_attempt} "
            f"status={s.status:<10} conf={s.confidence:.3f} [analysis only] "
            f"judge={sufficient_str}"
        )
    print(f"{'=' * 80}\n")

    return result


if __name__ == "__main__":
    query_key = sys.argv[1] if len(sys.argv) > 1 else "simple"
    question = DEMO_QUERIES.get(query_key, DEMO_QUERIES["simple"])
    run(question)
