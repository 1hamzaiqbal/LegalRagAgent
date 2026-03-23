"""Plan-and-Execute Legal RAG Agent.

Two execution modes (selected via PARALLEL_MODE=1 env var or --parallel CLI flag):

Sequential mode (default):
  START → router → planner → executor → replanner → executor  (retry / next step)
                                                   → synthesizer → END

Parallel mode (PARALLEL_MODE=1):
  START → router → planner → parallel_executor → parallel_synthesizer
                       ↑                               |
                       └── parallel_replanner ←────────┘ (if incomplete)
                                                       → END (if complete)

All system prompts are loaded from skills/*.md at runtime.
"""

# Windows: prevent OpenMP segfault when PyTorch and sentence-transformers
# each load their own OpenMP runtime (libiomp5md.dll conflict).
# Must be set before any torch/transformers import.
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import concurrent.futures
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

# Logging verbosity — set via --verbose CLI flag or VERBOSE=1 env var
VERBOSE = os.getenv("VERBOSE", "0") == "1"



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
    # Max retries before giving up on this step (set by planner based on importance)
    max_retries: int = 2
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
    collections: List[str]              # which ChromaDB collections to search
    planning_table: List[PlanningStep]
    evidence_store: List[Dict[str, Any]]  # all retrieved passages (accumulated)
    final_answer: str
    audit_log: List[Dict[str, Any]]     # per-node trace entries
    completeness_verdict: Dict[str, Any]  # parallel mode: synthesizer completeness check
    parallel_round: int                   # parallel mode: current planner→executor round
    # Parallel mode fields
    completeness_verdict: Dict[str, Any]  # synthesizer completeness check result
    parallel_round: int                   # tracks planner→executor→synthesizer iterations


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
                if VERBOSE:
                    in_tok = len(_TIKTOKEN_ENC.encode(system_prompt + user_prompt))
                    out_tok = len(_TIKTOKEN_ENC.encode(content))
                    print(f"    [{label}] {len(content)} chars ({in_tok} in / {out_tok} out tokens)")
                else:
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
# 3. Web Search + Scrape
# ---------------------------------------------------------------------------

def web_search(query: str, k: int = 5) -> List[Dict[str, str]]:
    """DuckDuckGo text search. Returns list of {title, body, href} dicts."""
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=k))

        results = []
        for r in raw:
            title = (r.get("title") or "").strip()
            body = (r.get("body") or "").strip()
            href = (r.get("href") or "").strip()
            if body or title:
                results.append({"title": title, "body": body, "href": href})
        return results
    except Exception as exc:
        print(f"    [web_search] Error: {exc}")
        return []


def _enrich_with_scraper(search_results: List[Dict[str, str]],
                         max_scrape: int = 2) -> List[Dict[str, str]]:
    """Scrape full page content from top search result URLs.

    Returns enriched results: original snippet results + scraped full-text
    results (marked with source='scraped'). Scraping failures are silently
    skipped — the original snippets are always available as fallback.
    """
    from web_scraper import scrape_urls

    urls = [r["href"] for r in search_results if r.get("href")]
    if not urls:
        return search_results

    scraped = scrape_urls(urls, max_results=max_scrape, max_chars=6000)
    print(f"    Scraped {len(scraped)}/{len(urls)} URLs successfully")

    enriched = list(search_results)
    for s in scraped:
        enriched.append({
            "title": s["title"],
            "body": s["text"],
            "href": s["url"],
            "source": "scraped",
        })
    return enriched


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

    Returns a dict with keys:
      sufficient: "full" | "partial" | False
      reason: str
      missing: str | None (what's missing, for partial verdicts)
      suggested_rewrite: str | None
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
        return {
            "sufficient": "full",
            "reason": f"parse failure ({skill_name}) — defaulting to full",
            "missing": None,
            "suggested_rewrite": None,
        }

    # Normalize the sufficient field: "full", "partial", or False
    raw_suf = parsed.get("sufficient", "full")
    if raw_suf == "partial":
        sufficient = "partial"
    elif raw_suf in (True, "full", "true"):
        sufficient = "full"
    else:
        sufficient = False

    return {
        "sufficient": sufficient,
        "reason": str(parsed.get("reason", "")),
        "missing": parsed.get("missing") or None,
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
    if len(queries) > 1:
        for j, alt in enumerate(queries[1:], 1):
            print(f"    Alt query {j}: {alt}")

    # --- Retrieval with cross-step deduplication ---
    prior_ids: set = set()
    for s in table:
        prior_ids.update(s.evidence_ids)
    for ev in evidence_store:
        prior_ids.add(ev.get("idx", ""))

    # Retrieve from all routed collections
    collections = state.get("collections", ["legal_passages"])
    all_docs = []
    for coll_name in collections:
        from rag_utils import get_vectorstore as _get_vs
        vs = _get_vs(collection_name=coll_name)
        coll_docs = retrieve_documents_multi_query(
            queries=queries,
            k=5,
            exclude_ids=prior_ids if prior_ids else None,
            vectorstore=vs,
        )
        all_docs.extend(coll_docs)

    # If multiple collections, re-rank the combined pool
    if len(collections) > 1 and len(all_docs) > 5:
        from rag_utils import rerank_with_cross_encoder
        all_docs = rerank_with_cross_encoder(queries[0], all_docs, top_k=5)

    docs = all_docs
    print(f"    Retrieved {len(docs)} passage(s) from {collections}")

    # --- Build numbered evidence passages ---
    passages: List[str] = []
    new_evidence: List[Dict[str, Any]] = []
    for i, doc in enumerate(docs, 1):
        text = doc.page_content
        source = doc.metadata.get("source", "unknown")
        idx = str(doc.metadata.get("idx", f"step{step.step_id}_{i}"))
        ce_score = doc.metadata.get("cross_encoder_score", 0.0)
        passages.append(f"[Source {i}] ({source})\n{text}")
        new_evidence.append({
            "idx": idx,
            "text": text,
            "source": source,
            "step_id": step.step_id,
            "cross_encoder_score": ce_score,
        })
        if VERBOSE:
            preview = text[:300].replace('\n', ' ')
            print(f"      [{i}] idx={idx} source={source} ce_score={ce_score:.3f}")
            print(f"          {preview}{'...' if len(text) > 300 else ''}")

    # --- Sub-answer synthesis ---
    evidence_block = "\n\n".join(passages) if passages else "[No passages retrieved]"
    synth_prompt = (
        f"Original legal research question: {question}\n\n"
        f"Research sub-question: {step.sub_question}\n\n"
        f"Evidence passages:\n{evidence_block}"
    )
    result = _llm_call(load_skill("synthesize_and_cite"), synth_prompt, label="executor/synth")
    if VERBOSE:
        print(f"    --- Sub-answer ---")
        print(f"    {result[:500]}{'...' if len(result) > 500 else ''}")
        print(f"    --- End sub-answer ---")

    # --- Raw confidence logit for logging ---
    raw_logit = compute_confidence(queries[0], docs) if docs else 0.0

    return result, new_evidence, raw_logit


def _execute_web_search(
    step: PlanningStep,
    state: LegalAgentState,
) -> tuple[str, List[Dict[str, Any]], float]:
    """Search the web via DuckDuckGo, scrape top URLs, and synthesize.

    Two-stage: DuckDuckGo finds results (snippets + URLs), then the scraper
    fetches full page content from the top 2 URLs. Both snippets and scraped
    text are passed to the synthesizer as evidence.

    Returns (result_text, new_evidence, raw_logit).
    """
    search_results = web_search(step.sub_question, k=5)
    print(f"    Web results: {len(search_results)} result(s)")
    for j, r in enumerate(search_results, 1):
        print(f"      [{j}] {r.get('title', '(no title)')}")
        if r.get("href"):
            print(f"          {r['href']}")

    if not search_results:
        return "[No web results found]", [], 0.0

    # Enrich with full page content from top URLs
    enriched = _enrich_with_scraper(search_results, max_scrape=2)

    question = state["inputs"]["question"]
    passages = []
    for i, r in enumerate(enriched, 1):
        source_tag = "Scraped" if r.get("source") == "scraped" else "Snippet"
        title = r.get("title", "")
        body = r.get("body", "")
        href = r.get("href", "")
        header = f"[WebResult {i}] ({source_tag}) {title}"
        if href:
            header += f"\nURL: {href}"
        passages.append(f"{header}\n{body}")
        if VERBOSE:
            char_count = len(body)
            preview = body[:400].replace('\n', ' ')
            print(f"      [{i}] ({source_tag}) {title} [{char_count} chars]")
            print(f"          {preview}{'...' if char_count > 400 else ''}")

    evidence_block = "\n\n".join(passages)
    synth_prompt = (
        f"Original legal research question: {question}\n\n"
        f"Research sub-question: {step.sub_question}\n\n"
        f"Web search results:\n{evidence_block}"
    )
    result = _llm_call(load_skill("synthesize_and_cite"), synth_prompt, label="executor/web")
    if VERBOSE:
        print(f"    --- Sub-answer ---")
        print(f"    {result[:500]}{'...' if len(result) > 500 else ''}")
        print(f"    --- End sub-answer ---")

    new_evidence = [
        {
            "idx": f"web_{step.step_id}_{i}",
            "text": r.get("body", ""),
            "source": "web_scraped" if r.get("source") == "scraped" else "web_snippet",
            "step_id": step.step_id,
            "cross_encoder_score": 0.0,
        }
        for i, r in enumerate(enriched)
    ]

    return result, new_evidence, 0.0


# ---------------------------------------------------------------------------
# 6. Node: Collection Router
# ---------------------------------------------------------------------------

# Available collections and their descriptions for the router
COLLECTIONS_REGISTRY = {
    "legal_passages": "Bar exam study materials, case law, and legal doctrine (torts, contracts, property, constitutional law, criminal law, evidence, etc.)",
    "housing_statutes": "US housing statutes across all 50 states — landlord-tenant law, eviction, security deposits, habitability, lease termination, rent control",
}

_ROUTER_PROMPT = f"""You are a legal research router. Given a legal question, decide which document collection(s) to search.

Available collections:
{chr(10).join(f'- "{name}": {desc}' for name, desc in COLLECTIONS_REGISTRY.items())}

Return ONLY valid JSON — no prose, no markdown fences:
{{"collections": ["collection_name_1"]}}

Rules:
- Choose the collection(s) most likely to contain relevant passages for this question.
- Most questions need only ONE collection. Use multiple only if the question clearly spans both domains.
- If uncertain, default to "legal_passages" (it has the broadest coverage).
"""


def router_node(state: LegalAgentState) -> dict:
    """Lightweight LLM call to decide which ChromaDB collection(s) to search."""
    print("\n--- ROUTER ---")
    question = state["inputs"]["question"]

    raw = _llm_call(_ROUTER_PROMPT, question, label="router")
    parsed = _parse_json(raw)

    if parsed and "collections" in parsed:
        collections = [c for c in parsed["collections"] if c in COLLECTIONS_REGISTRY]
    else:
        collections = []

    if not collections:
        collections = ["legal_passages"]

    print(f"  Collections: {collections}")
    return {
        "collections": collections,
        "audit_log": _log(state, {
            "node": "router",
            "collections": collections,
        }),
    }


# ---------------------------------------------------------------------------
# 7. Node: Planner
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

    complexity = "moderate"  # default
    steps: List[PlanningStep] = []
    if parsed and "steps" in parsed:
        complexity = parsed.get("complexity", "moderate")
        for i, s in enumerate(parsed["steps"]):
            if isinstance(s, dict):
                steps.append(PlanningStep(
                    step_id=i + 1,
                    sub_question=s.get("sub_question", question),
                    authority_target=s.get("authority_target", ""),
                    retrieval_hints=s.get("retrieval_hints", []),
                    action_type=s.get("action_type", ACTION_RAG),
                    max_retries=min(s.get("max_retries", 2), 3),  # cap at 3
                ))
            else:
                steps.append(PlanningStep(step_id=i + 1, sub_question=str(s)))
    else:
        print("  [planner] Parse failed — single-step fallback")
        steps = [PlanningStep(step_id=1, sub_question=question)]

    # Hard cap: max 5 steps regardless of what the LLM returns
    if len(steps) > 5:
        print(f"  [planner] Capping {len(steps)} steps to 5")
        steps = steps[:5]

    print(f"  Complexity: {complexity} | Plan ({len(steps)} step{'s' if len(steps) != 1 else ''}):")
    for s in steps:
        print(f"    Step {s.step_id} [{s.action_type}] (max_retries={s.max_retries}): {s.sub_question}")

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
# 7. Parallel Executor (with per-step escalation)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 9b. Parallel Mode: Parallel Executor Node
# ---------------------------------------------------------------------------

def _execute_step_with_escalation(
    step: PlanningStep,
    state: LegalAgentState,
    table: List[PlanningStep],
    evidence_store: List[Dict[str, Any]],
) -> tuple[PlanningStep, List[Dict[str, Any]]]:
    """Execute a single step with the full escalation chain (rag → rewrite → web → direct).

    This mirrors the sequential executor+replanner loop but runs it all within
    a single function call, so multiple steps can be executed independently.

    Returns (updated_step, new_evidence_for_this_step).
    """
    question = state["inputs"]["question"]
    current = step.model_copy()
    all_new_evidence: List[Dict[str, Any]] = []
    # Max attempts = 1 (initial) + max_retries (escalations), hard cap at 4
    max_attempts = min(1 + current.max_retries, 4)

    for escalation in range(max_attempts):
        print(f"\n  [Step {current.step_id}] Attempt {escalation + 1} [{current.action_type}] "
              f"(rewrite_attempt={current.rewrite_attempt}): {current.sub_question}")

        # --- Execute based on action type ---
        if current.action_type == ACTION_DIRECT:
            result, new_evidence, raw_logit = _execute_direct_answer(current, state)
        elif current.action_type == ACTION_WEB:
            result, new_evidence, raw_logit = _execute_web_search(current, state)
        else:  # rag_search
            result, new_evidence, raw_logit = _execute_rag_search(
                current, state, table, evidence_store + all_new_evidence
            )

        confidence = _normalise_confidence(raw_logit) if raw_logit != 0.0 else 0.0
        print(f"  [Step {current.step_id}] Confidence: {confidence:.3f}")

        # --- Judge evaluation ---
        passage_texts = [ev["text"] for ev in new_evidence]
        verdict = _call_judge(current, result, passage_texts, question)
        print(f"  [Step {current.step_id}] Judge: sufficient={verdict['sufficient']} | "
              f"{verdict.get('reason', '')}")

        all_new_evidence.extend(new_evidence)

        # If judge says sufficient (full or partial), we're done with this step
        if verdict["sufficient"] in ("full", "partial", True):
            updated = current.model_copy(update={
                "status": STATUS_COMPLETED,
                "result": result,
                "confidence": confidence,
                "evidence_ids": [ev["idx"] for ev in all_new_evidence],
                "judge_verdict": verdict,
            })
            print(f"  [Step {current.step_id}] Completed (judge={verdict['sufficient']})")
            return updated, all_new_evidence

        # --- Escalation logic ---
        # RAG steps: rag → rewrite → direct_answer (skip web for doctrinal queries)
        # Web steps: web → direct_answer
        if current.action_type == ACTION_RAG and current.rewrite_attempt == 0:
            # First RAG failure → rewrite query, stay rag_search
            revised_q = verdict.get("suggested_rewrite") or current.sub_question
            print(f"  [Step {current.step_id}] Escalating: rag_search → rewrite")
            current = current.model_copy(update={
                "sub_question": revised_q,
                "rewrite_attempt": 1,
            })

        elif current.action_type == ACTION_RAG and current.rewrite_attempt >= 1:
            # Second RAG failure → fall back to direct_answer (LLM knows doctrine)
            print(f"  [Step {current.step_id}] Escalating: rag_search → direct_answer")
            current = current.model_copy(update={
                "action_type": ACTION_DIRECT,
                "rewrite_attempt": 0,
            })

        elif current.action_type == ACTION_WEB:
            # Web search failure → fall back to direct_answer
            print(f"  [Step {current.step_id}] Escalating: web_search → direct_answer")
            current = current.model_copy(update={
                "action_type": ACTION_DIRECT,
                "rewrite_attempt": 0,
            })

        elif current.action_type == ACTION_DIRECT:
            # direct_answer also failed — keep whatever we have
            print(f"  [Step {current.step_id}] All escalations exhausted, keeping best result")
            break

    # All escalations exhausted — keep the last result
    updated = current.model_copy(update={
        "status": STATUS_COMPLETED,
        "result": result,
        "confidence": confidence,
        "evidence_ids": [ev["idx"] for ev in all_new_evidence],
        "judge_verdict": verdict,
    })
    return updated, all_new_evidence


def parallel_executor_node(state: LegalAgentState) -> dict:
    """Execute ALL pending planning steps, each with its own escalation chain.

    In parallel mode, this replaces the sequential executor_node + replanner_node
    loop. Each step independently runs: query rewrite → retrieve → synthesize → judge,
    and if the judge says insufficient, escalates (rag → rewrite → web → direct)
    within its own loop before returning.

    Currently uses sequential execution (for deterministic debugging). True thread/
    process parallelism can be enabled by switching to concurrent.futures below.
    """
    print("\n--- PARALLEL EXECUTOR ---")
    table = state["planning_table"]
    evidence_store = list(state.get("evidence_store", []))
    question = state["inputs"]["question"]

    pending = [s for s in table if s.status == STATUS_PENDING]
    if not pending:
        print("  No pending steps.")
        return {}

    print(f"  Executing {len(pending)} step(s) in parallel mode")

    # --- Execute all pending steps ---
    # Sequential execution for now. To enable true parallelism, uncomment the
    # ThreadPoolExecutor block below and comment out the sequential block.
    results: List[tuple[PlanningStep, List[Dict[str, Any]]]] = []

    # Option A: Sequential execution (deterministic, easier to debug)
    for step in pending:
        updated_step, new_evidence = _execute_step_with_escalation(
            step, state, table, evidence_store
        )
        results.append((updated_step, new_evidence))
        # Accumulate evidence so later steps can deduplicate
        evidence_store.extend(new_evidence)

    # Option B: True parallel execution (uncomment to enable)
    # with concurrent.futures.ThreadPoolExecutor(max_workers=len(pending)) as executor:
    #     futures = {
    #         executor.submit(
    #             _execute_step_with_escalation,
    #             step, state, table, evidence_store
    #         ): step.step_id
    #         for step in pending
    #     }
    #     for future in concurrent.futures.as_completed(futures):
    #         updated_step, new_evidence = future.result()
    #         results.append((updated_step, new_evidence))

    # --- Build updated planning table ---
    completed_ids = {r[0].step_id for r in results}
    new_table = []
    for s in table:
        if s.step_id in completed_ids:
            updated = next(r[0] for r in results if r[0].step_id == s.step_id)
            new_table.append(updated)
        else:
            new_table.append(s)

    # Collect all new evidence
    all_new_evidence = []
    for _, new_ev in results:
        all_new_evidence.extend(new_ev)

    completed_count = sum(1 for s in new_table if s.status == STATUS_COMPLETED)
    print(f"\n  Parallel execution complete: {completed_count} step(s) completed, "
          f"{len(all_new_evidence)} evidence pieces collected")

    return {
        "planning_table": new_table,
        "evidence_store": state.get("evidence_store", []) + all_new_evidence,
        "audit_log": _log(state, {
            "node": "parallel_executor",
            "steps_executed": len(pending),
            "steps_completed": completed_count,
            "evidence_collected": len(all_new_evidence),
        }),
    }


# ---------------------------------------------------------------------------
# 9c. Parallel Mode: Synthesizer with Completeness Check
# ---------------------------------------------------------------------------

_COMPLETENESS_CHECK_PROMPT = """You are evaluating whether the accumulated research evidence is sufficient to fully answer the original legal question.

Given the original question, the research findings, and the synthesized answer, assess completeness.

Return ONLY valid JSON — no prose, no markdown fences:

{
  "complete": true,
  "reasoning": "One sentence explaining why the evidence is or is not sufficient.",
  "missing_topics": []
}

Fields:
- `complete`: true if the synthesized answer adequately addresses the original question; false if there are significant gaps that additional research could fill.
- `reasoning`: Brief explanation of your assessment.
- `missing_topics`: If complete is false, list 1-3 specific legal topics/doctrines that need additional research. Each should be a focused sub-question suitable for a new research step. Empty list if complete is true.

Guidelines:
- An answer does not need to be perfect to be complete. If the key legal rules, elements, and application are covered, it is complete.
- Only mark incomplete if there is a clearly identifiable gap that would materially change the answer.
- If the question is simple (single doctrine), the bar for completeness is low — one well-supported step is usually enough.
- Do not mark incomplete just because more depth is theoretically possible. Mark incomplete only when the current answer would be misleading or materially deficient without the missing information.
- Maximum 3 rounds of research are allowed, so be conservative about requesting more.
"""


def parallel_synthesizer_node(state: LegalAgentState) -> dict:
    """Aggregate research and check completeness. In parallel mode, this replaces
    synthesizer_node and adds a completeness verdict that can route back to planner.

    Loads: skills/synthesizer.md for synthesis, then runs a separate completeness
    check to decide whether to loop back for more research.
    """
    print("\n--- PARALLEL SYNTHESIZER ---")
    table = state["planning_table"]
    evidence_store = state.get("evidence_store", [])
    question = state["inputs"]["question"]
    parallel_round = state.get("parallel_round", 1)

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

    # --- Completeness check ---
    # Skip completeness check if we've hit the max rounds (3)
    max_rounds = state.get("run_config", {}).get("max_parallel_rounds", 3)
    if parallel_round >= max_rounds:
        print(f"  Max parallel rounds ({max_rounds}) reached — marking complete")
        completeness = {"complete": True, "reasoning": "max rounds reached", "missing_topics": []}
    else:
        completeness_prompt = (
            f"ORIGINAL QUESTION:\n{question}\n\n"
            f"RESEARCH FINDINGS:\n{step_summaries}\n\n"
            f"SYNTHESIZED ANSWER:\n{final_answer}\n\n"
            f"Research round: {parallel_round} of {max_rounds} max"
        )
        raw = _llm_call(_COMPLETENESS_CHECK_PROMPT, completeness_prompt, label="completeness_check")
        parsed = _parse_json(raw)

        if parsed and "complete" in parsed:
            completeness = {
                "complete": parsed.get("complete", True),
                "reasoning": parsed.get("reasoning", ""),
                "missing_topics": parsed.get("missing_topics", []),
            }
        else:
            # Parse failure — default to complete to avoid infinite loops
            completeness = {"complete": True, "reasoning": "parse failure — defaulting to complete", "missing_topics": []}

    is_complete = completeness.get("complete", True)
    print(f"  Completeness: {'COMPLETE' if is_complete else 'INCOMPLETE'} — {completeness.get('reasoning', '')}")
    if not is_complete and completeness.get("missing_topics"):
        print(f"  Missing topics: {completeness['missing_topics']}")

    return {
        "final_answer": final_answer,
        "completeness_verdict": completeness,
        "parallel_round": parallel_round + 1,
        "audit_log": _log(state, {
            "node": "parallel_synthesizer",
            "completed_steps": len(completed),
            "evidence_entries": len(evidence_store),
            "parallel_round": parallel_round,
            "completeness": completeness,
        }),
    }


# ---------------------------------------------------------------------------
# 9d. Parallel Mode: Replanner (generates additional steps from missing topics)
# ---------------------------------------------------------------------------

def parallel_replanner_node(state: LegalAgentState) -> dict:
    """Generate new planning steps from the synthesizer's missing topics.

    Called when the parallel synthesizer determines the evidence is incomplete.
    Takes the missing_topics from the completeness verdict and creates new
    PlanningStep entries with status=pending for the next executor round.
    """
    print("\n--- PARALLEL REPLANNER ---")
    table = state["planning_table"]
    completeness = state.get("completeness_verdict", {})
    missing_topics = completeness.get("missing_topics", [])

    if not missing_topics:
        print("  No missing topics — nothing to replan")
        return {}

    # Find the max step_id so we can assign new unique IDs
    max_id = max((s.step_id for s in table), default=0)

    new_steps: List[PlanningStep] = []
    for i, topic in enumerate(missing_topics, 1):
        new_step = PlanningStep(
            step_id=max_id + i,
            sub_question=topic if isinstance(topic, str) else str(topic),
            authority_target="",
            retrieval_hints=[],
            action_type=ACTION_RAG,
        )
        new_steps.append(new_step)
        print(f"  New step {new_step.step_id}: {new_step.sub_question}")

    print(f"  Added {len(new_steps)} new step(s) for round {state.get('parallel_round', 2)}")

    return {
        "planning_table": list(table) + new_steps,
        "audit_log": _log(state, {
            "node": "parallel_replanner",
            "new_steps": len(new_steps),
            "missing_topics": missing_topics,
        }),
    }


# ---------------------------------------------------------------------------
# 10. Routing
# ---------------------------------------------------------------------------

def route_after_parallel_synthesizer(
    state: LegalAgentState,
) -> Literal["parallel_replanner_node", "__end__"]:
    """Route after parallel synthesizer: loop back if incomplete, else END."""
    completeness = state.get("completeness_verdict", {})
    if completeness.get("complete", True):
        print("  → END (research complete)")
        return "__end__"
    print("  → parallel_replanner (evidence incomplete)")
    return "parallel_replanner_node"


# ---------------------------------------------------------------------------
# 9. Graph
# ---------------------------------------------------------------------------

def build_graph() -> Any:
    """Compile the parallel plan-and-execute LangGraph state machine.

    Architecture:
      START → router → planner → parallel_executor → parallel_synthesizer
                          ↑                               |
                          └── parallel_replanner ←────────┘ (if incomplete)
                                                          └→ END (if complete)

    Key differences from sequential mode:
    - parallel_executor runs ALL pending steps at once (with internal escalation)
    - parallel_synthesizer checks completeness and can loop back for more research
    - No sequential replanner_node — escalation happens inside the parallel executor
    """
    workflow = StateGraph(LegalAgentState)

    workflow.add_node("router_node", router_node)
    workflow.add_node("planner_node", planner_node)
    workflow.add_node("parallel_executor_node", parallel_executor_node)
    workflow.add_node("parallel_synthesizer_node", parallel_synthesizer_node)
    workflow.add_node("parallel_replanner_node", parallel_replanner_node)

    # Fixed edges
    workflow.add_edge(START, "router_node")
    workflow.add_edge("router_node", "planner_node")
    workflow.add_edge("planner_node", "parallel_executor_node")
    workflow.add_edge("parallel_executor_node", "parallel_synthesizer_node")
    # Replanner feeds back to executor for the new steps
    workflow.add_edge("parallel_replanner_node", "parallel_executor_node")

    # Conditional: parallel_synthesizer → END (complete) | parallel_replanner (incomplete)
    workflow.add_conditional_edges(
        "parallel_synthesizer_node",
        route_after_parallel_synthesizer,
        ["parallel_replanner_node", "__end__"],
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


def run(question: str, max_steps: int = 7, **kwargs) -> LegalAgentState:
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
        "run_config": {"max_steps": max_steps, "max_parallel_rounds": 3},
        "collections": [],  # populated by router_node
        "planning_table": [],
        "evidence_store": [],
        "final_answer": "",
        "audit_log": [],
        "completeness_verdict": {},
        "parallel_round": 1,
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

    # LLM call metrics
    llm_counts = _get_metrics()

    rounds = result.get("parallel_round", 1) - 1

    print(f"\n{'=' * 80}")
    print("RUN SUMMARY")
    print(f"{'=' * 80}")
    print(f"Steps completed : {len(completed_steps)}")
    print(f"Evidence pieces : {evidence_count}")
    print(f"LLM calls       : {llm_counts['count']}")
    print(f"Tokens (in/out) : {llm_counts['input_tokens']:,} / {llm_counts['output_tokens']:,}")
    print(f"Rounds          : {rounds}")

    print("\nStep breakdown:")
    for s in result.get("planning_table", []):
        if s.status == STATUS_PENDING:
            print(f"  Step {s.step_id:>4} [{s.action_type:<14}] status=pending (skipped)")
        else:
            verdict = s.judge_verdict or {}
            suf = verdict.get("sufficient", "full")
            judge_str = "full" if suf == "full" else ("partial" if suf == "partial" else "insufficient")
            src_counts = {}
            for ev in result.get("evidence_store", []):
                if ev.get("step_id") == s.step_id:
                    src = ev.get("source", "unknown")
                    src_counts[src] = src_counts.get(src, 0) + 1
            src_summary = ", ".join(f"{v} {k}" for k, v in src_counts.items()) if src_counts else "none"
            print(
                f"  Step {s.step_id:>4} [{s.action_type:<14}] "
                f"conf={s.confidence:.3f} judge={judge_str} "
                f"evidence=[{src_summary}]"
            )
    print(f"{'=' * 80}\n")

    return result


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--verbose" in args:
        VERBOSE = True
        args.remove("--verbose")
    query_key = args[0] if args else "simple"
    question = DEMO_QUERIES.get(query_key, DEMO_QUERIES["simple"])
    run(question)
