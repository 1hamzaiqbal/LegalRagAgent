"""Multi-model evaluation harness for the Legal RAG pipeline.

Supports multiple eval modes, providers, question sets, and skill overrides.
Produces rich per-question JSONL logs and appends run summaries to experiments.jsonl.

Usage:
    uv run python eval/eval_harness.py --mode llm_only --provider deepseek --questions 10
    uv run python eval/eval_harness.py --mode full_pipeline --provider gemma --questions curated
    uv run python eval/eval_harness.py --mode rag_rewrite --provider deepseek --questions 30 --tag "aspect-queries"
    uv run python eval/eval_harness.py --mode golden_passage --provider openai --questions 100
    uv run python eval/eval_harness.py --mode full_pipeline --skill-dir skills_v2 --questions curated
"""
import argparse
import json
import os
import re
import subprocess
import sys
from typing import List
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from eval_config import EvalConfig, EVAL_MODES, load_questions, extract_answer_mc, extract_answer_mc5, extract_answer_yn, format_question_prompt, extract_answer_musique, musique_em_f1
from main import (
    run as run_pipeline,
    _llm_call as _base_llm_call,
    _get_metrics,
    _reset_llm_call_counter,
    _parse_json,
    load_skill,
)
from llm_config import get_provider_info, _get_llm_cached
from rag_utils import retrieve_documents_multi_query, get_vectorstore


_CALL_TRACE: list[dict] = []
_TRACE_EVENTS: list[dict] = []


def _trace_calls_enabled() -> bool:
    return os.getenv("EVAL_TRACE_CALLS", "").strip().lower() not in ("", "0", "false", "no")


def _trace_events_enabled() -> bool:
    raw = os.getenv("EVAL_TRACE_EVENTS", "").strip().lower()
    if raw == "":
        return _trace_calls_enabled()
    return raw not in ("0", "false", "no")


def _trace_text(text: str) -> str:
    limit_raw = os.getenv("EVAL_TRACE_MAX_CHARS", "0").strip()
    try:
        limit = int(limit_raw)
    except ValueError:
        limit = 0
    if limit <= 0:
        return text
    return text[:limit] + ("..." if len(text) > limit else "")


def _reset_call_trace() -> None:
    _CALL_TRACE.clear()


def _get_call_trace() -> list[dict]:
    return list(_CALL_TRACE)


def _trace_value(value):
    if isinstance(value, str):
        return _trace_text(value)
    if isinstance(value, list):
        return [_trace_value(v) for v in value]
    if isinstance(value, tuple):
        return [_trace_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _trace_value(v) for k, v in value.items()}
    return value


def _record_trace_event(event_type: str, **payload) -> None:
    if not _trace_events_enabled():
        return
    _TRACE_EVENTS.append({
        "type": event_type,
        **_trace_value(payload),
    })


def _reset_trace_events() -> None:
    _TRACE_EVENTS.clear()


def _get_trace_events() -> list[dict]:
    return list(_TRACE_EVENTS)


def _llm_call(system: str, user: str, label: str = "") -> str:
    """Wrapper around main._llm_call that optionally records exact call I/O.

    Retries once on transient JSON-parse / connection errors (common with
    OpenRouter routing to flaky downstream providers). The underlying
    client also has max_retries=1 for HTTP-level retries; this layer
    catches body-parse failures the OpenAI client doesn't retry.
    """
    try:
        response = _base_llm_call(system, user, label=label)
    except (json.JSONDecodeError, ConnectionError, TimeoutError) as exc:
        # One retry — these are transient OpenRouter routing failures
        time.sleep(2)
        try:
            response = _base_llm_call(system, user, label=label)
        except Exception as exc2:
            if _trace_calls_enabled():
                _CALL_TRACE.append({
                    "label": label, "system": _trace_text(system), "user": _trace_text(user),
                    "response": "", "error": f"retry_failed: {exc2}",
                    "system_chars": len(system or ""), "user_chars": len(user or ""), "response_chars": 0,
                })
            raise exc2 from exc
    except Exception as exc:
        if _trace_calls_enabled():
            _CALL_TRACE.append({
                "label": label,
                "system": _trace_text(system),
                "user": _trace_text(user),
                "response": "",
                "error": str(exc),
                "system_chars": len(system or ""),
                "user_chars": len(user or ""),
                "response_chars": 0,
            })
            _record_trace_event(
                "llm_call",
                label=label,
                system=system,
                user=user,
                response="",
                error=str(exc),
                system_chars=len(system or ""),
                user_chars=len(user or ""),
                response_chars=0,
            )
        raise

    if _trace_calls_enabled():
        _CALL_TRACE.append({
            "label": label,
            "system": _trace_text(system),
            "user": _trace_text(user),
            "response": _trace_text(response),
            "error": "",
            "system_chars": len(system or ""),
            "user_chars": len(user or ""),
            "response_chars": len(response or ""),
        })
    _record_trace_event(
        "llm_call",
        label=label,
        system=system,
        user=user,
        response=response,
        error="",
        system_chars=len(system or ""),
        user_chars=len(user or ""),
        response_chars=len(response or ""),
    )
    return response


# ---------------------------------------------------------------------------
# Mode Runner Functions
# ---------------------------------------------------------------------------

def _fmt(row: pd.Series, config: EvalConfig) -> str:
    """Format question prompt based on dataset."""
    return format_question_prompt(row, dataset=config.dataset)


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> reasoning blocks from model output.

    Qwen3 models (and some Llama variants) emit chain-of-thought wrapped in
    `<think>...</think>` tags by default. If the response is truncated mid-think,
    the close tag never appears and the answer is never reached. Stripping
    closed think-blocks at extraction time means at least the visible-reasoning
    suffix gets parsed. Open (unclosed) think-blocks indicate a truncated
    response — we leave those alone so the caller can detect the failure.
    """
    if not text:
        return ""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL | re.IGNORECASE)


def _extract_answer(text: str, config: EvalConfig) -> str | None:
    """Extract answer using the right extractor for the dataset."""
    # Strip closed <think>...</think> blocks before extraction (Qwen3, etc.)
    text = _strip_think_tags(text)
    if config.dataset == "housing":
        return extract_answer_yn(text)
    if config.dataset == "casehold":
        return extract_answer_mc5(text)
    if config.dataset == "musique":
        # Short-answer span — extract the post-Answer span, EM/F1 scored downstream
        return extract_answer_musique(text)
    if config.dataset in ("legal_rag", "australian"):
        return text  # open-ended: return full text, scored by LLM judge
    return extract_answer_mc(text)


def _retrieval_question(row: pd.Series) -> str:
    """Full query text for retrieval/rerank — includes BarExam prompt column
    (shared fact pattern) when present. Many call sites used to just do
    `str(row["question"])` which silently dropped 37% of the fact-pattern
    context. Prefer this helper for any path that feeds text into
    retrieval, reranking, entity search, or keyword generation.
    """
    prompt = row.get("prompt", "")
    stem = str(row.get("question", "") or "")
    if pd.notna(prompt) and str(prompt).strip():
        return f"{str(prompt).strip()}\n\n{stem}"
    return stem


def _fmt_intermediate(row: pd.Series, config: EvalConfig) -> str:
    """Format a question for retrieval-side generation steps.

    Unlike `_fmt()`, this strips answer-format instructions and, for MC tasks,
    removes answer letters from the choices so intermediate generators are not
    pushed toward emitting `Answer: (X)` artifacts.
    """
    if config.dataset in ("legal_rag", "australian", "musique"):
        return str(row["question"])

    if config.dataset == "housing":
        state = str(row.get("state", ""))
        return f"Regarding {state} housing law:\n\n{row['question']}"

    if config.dataset == "casehold":
        context = str(row["question"])
        holdings = []
        for letter in ["A", "B", "C", "D", "E"]:
            col = f"choice_{letter.lower()}"
            if col in row and pd.notna(row[col]):
                holdings.append(f"- {row[col]}")

        parts = [
            "The following excerpt from a court opinion cites a legal holding.",
            f"## Citing Context\n{context}",
        ]
        if holdings:
            parts.append("## Candidate Holdings\n" + "\n".join(holdings))
        return "\n\n".join(parts)

    # Pull the shared fact pattern from 'prompt' column when present; same fix
    # as format_question_prompt (37% of BarExam rows need this context).
    prompt_ctx = row.get("prompt", "")
    prompt_prefix = ""
    if pd.notna(prompt_ctx) and str(prompt_ctx).strip():
        prompt_prefix = str(prompt_ctx).strip() + "\n\n"

    stem = prompt_prefix + str(row["question"])
    choices = []
    for letter in ["A", "B", "C", "D"]:
        col = f"choice_{letter.lower()}"
        if col in row and pd.notna(row[col]):
            choices.append(f"- {row[col]}")

    if not choices:
        return stem

    return "\n\n".join([
        stem,
        "## Candidate Answer Framing\n" + "\n".join(choices),
    ])


def _contains_answer_artifact(text: str) -> bool:
    """Detect explicit answer labels leaking into intermediate artifacts."""
    if not text:
        return False
    return bool(re.search(r"(?im)^\s*(?:\*\*)?(?:final\s+)?answer(?:\*\*)?\s*:", text))


def _sanitize_intermediate_text(text: str, fallback: str = "") -> str:
    """Strip answer-label artifacts and prompt-ish headers from intermediate text."""
    cleaned = text or ""

    # Remove standalone answer lines like `Answer: (B)`, `**Answer: (d)**`,
    # `Final Answer: Yes`, or `**Answer:** (B)`. The optional `**` may wrap the
    # whole line (bold-wrapped) OR sit between label and letter.
    cleaned = re.sub(
        r"(?im)^\s*(?:\*\*)?(?:final\s+)?answer(?:\*\*)?\s*:\s*(?:\*\*)?\s*(?:\(?[A-E]\)?|yes|no|irrelevant)\s*(?:\*\*)?\s*$",
        "",
        cleaned,
    )

    # Remove heading-only lines like `**Passage:**` or `**Legal Reference Passage:**`
    # (with optional closing bold markers after the colon).
    cleaned = re.sub(
        r"(?im)^\s*\*\*\s*(?:relevant legal passage|legal reference passage|passage)\s*:?\s*\*\*\s*:?\s*$",
        "",
        cleaned,
    )
    # Remove inline label prefixes like `**Passage:** <text>` or `Passage: <text>`.
    cleaned = re.sub(
        r"(?im)^\s*\*\*\s*(?:relevant legal passage|legal reference passage|passage)\s*:?\s*\*\*\s*:?\s*",
        "",
        cleaned,
    )
    cleaned = re.sub(
        r"(?im)^\s*(?:relevant legal passage|legal reference passage|passage)\s*:\s*",
        "",
        cleaned,
    )

    # Strip any leftover orphaned bold markers or decorative separators at the start
    # (e.g., stray `**` left after the label was removed, or `***` dividers).
    cleaned = re.sub(r"\A(?:\s*(?:\*{1,}|-{3,}|_{3,})\s*)+", "", cleaned)

    lines = []
    prev_blank = False
    for raw_line in cleaned.splitlines():
        line = raw_line.rstrip()
        is_blank = not line.strip()
        if is_blank and prev_blank:
            continue
        lines.append(line)
        prev_blank = is_blank

    cleaned = "\n".join(lines).strip()
    fallback = (fallback or "").strip()
    return cleaned or fallback


def _preview_text(text: str, limit: int = 1200) -> str:
    """Store a bounded prompt preview in logs without blowing up row size."""
    text = text or ""
    return text[:limit] + ("..." if len(text) > limit else "")


def _question_only_hyde_user(question_text: str) -> str:
    """Structured user payload for question-only HyDE generation."""
    return (
        "## Task\n"
        "Write a passage that would appear in a legal treatise or casebook — the kind of "
        "passage a researcher would find when looking up the doctrine behind the scenario "
        "below. This is NOT a multiple-choice task; do not pick an option.\n\n"
        "## Scenario (for context only)\n"
        f"{question_text}\n\n"
        "## Passage Requirements\n"
        "- 2-3 sentences, legal reference style\n"
        "- State the controlling rule, doctrine, holding, exception, or principle directly\n"
        "- Focus on the legal issue most likely controlling the scenario\n"
        "- Start with the doctrinal text itself — no 'Answer:', no letter labels, "
        "no '**Passage:**' header, no bold or markdown\n"
    )


def _strip_answer_line(text: str) -> str:
    """Remove trailing 'Answer: (X)' lines from reasoning before it becomes HyDE context."""
    if not text:
        return ""
    cleaned = re.sub(
        r"(?im)^\s*(?:\*\*)?(?:final\s+)?answer(?:\*\*)?\s*:\s*(?:\*\*)?\s*(?:\(?[A-E]\)?|yes|no|irrelevant)\s*(?:\*\*)?\s*$",
        "",
        text,
    )
    return cleaned.strip()


def _snap_hyde_user(question_text: str, snap_answer: str, gap_focus: str = "") -> str:
    """Shared user payload for snap-informed HyDE generation."""
    reasoning = _strip_answer_line(snap_answer)
    if gap_focus:
        reasoning = (
            f"{reasoning}\n\n"
            f"Focus on verifying or correcting this specific issue:\n{gap_focus}"
        ).strip()
    return (
        "## Task\n"
        "Write a passage that would appear in a legal treatise or casebook — one that a "
        "researcher would use to verify or correct the reasoning below. This is NOT a "
        "multiple-choice task; do not pick an option.\n\n"
        f"## Student's Reasoning (for context only)\n{reasoning}\n\n"
        f"## Scenario (for context only)\n{question_text}\n\n"
        "## Passage Requirements\n"
        "- 2-3 sentences, legal reference style\n"
        "- State the controlling rule, doctrine, holding, exception, or principle directly\n"
        "- Start with the doctrinal text itself — no 'Answer:', no letter labels, "
        "no '**Passage:**' header, no bold or markdown\n"
    )


def _generate_hyde(config: EvalConfig, role: str, user: str, label: str, fallback: str) -> dict:
    """Generate and sanitize a HyDE-style intermediate passage."""
    raw = _llm_call(_system_prompt(config, role), user, label=label)
    return {
        "text": _sanitize_intermediate_text(raw, fallback=fallback),
        "raw": raw,
        "contains_answer": _contains_answer_artifact(raw),
    }


def _report_prompt(max_words: int = 100, include_model_knowledge: bool = False) -> str:
    """Shared prompt for report-writing intermediate steps."""
    strict = (
        "\n\nSTRICT OUTPUT RULES:\n"
        "- This is NOT a multiple-choice task; do not pick an option.\n"
        "- Do NOT begin with 'Answer:', 'Answer (X)', or any multiple-choice letter.\n"
        "- Do NOT say which option is correct or reference choices by letter.\n"
        "- Do NOT use '**Passage:**' or similar headers before the report body.\n"
        "- Do NOT use markdown bolding, bullet points, or section dividers.\n"
        "- Output only the report body, nothing else."
    )
    if include_model_knowledge:
        return (
            "You are a legal research assistant. You have retrieved passages AND your own legal knowledge. "
            "Write a brief, focused report answering the sub-question by combining both sources. "
            "State what the law says directly. Flag if retrieved passages conflict with known law. "
            f"Keep under {max_words} words." + strict
        )
    return (
        "You are a legal research assistant. Read the retrieved passages and write a brief, focused "
        "report that states the relevant legal rules, doctrines, holdings, and uncertainties directly. "
        "If the passages are irrelevant or unhelpful, say so clearly. "
        f"Keep under {max_words} words." + strict
    )


def _generate_report(system: str, user: str, label: str, fallback: str) -> dict:
    """Generate and sanitize a report-style intermediate artifact."""
    raw = _llm_call(system, user, label=label)
    return {
        "text": _sanitize_intermediate_text(raw, fallback=fallback),
        "raw": raw,
        "contains_answer": _contains_answer_artifact(raw),
    }


def _system_prompt(config: EvalConfig, role: str = "answer") -> str:
    """Get dataset-appropriate system prompt."""
    if config.dataset == "housing":
        prompts = {
            "answer": (
                "You are a legal expert specializing in housing law. Answer the Yes/No question below. "
                "Reason step by step, then give your final answer as: Answer: Yes or Answer: No"
            ),
            "rag": (
                "You are a legal expert specializing in housing law. Reason through the question "
                "step by step. Retrieved passages are provided — use them to verify or "
                "refine your reasoning, but think through the problem independently first. "
                "Give your final answer as: Answer: Yes or Answer: No"
            ),
            "research": (
                "You are a legal expert specializing in housing law. Reason through the question "
                "step by step. Research findings are provided — use them to verify or "
                "refine your reasoning, but think through the problem independently first. "
                "Give your final answer as: Answer: Yes or Answer: No"
            ),
            "hyde": (
                "You are a legal textbook author specializing in housing law. Given a legal question, "
                "write a short passage (2-3 sentences) that would appear in a reference guide as the answer. "
                "Write in the style of a legal reference — state the statute, rule, or "
                "regulation directly. Do not discuss the question itself or say 'the answer is'."
            ),
            "snap_hyde": (
                "You are a legal textbook author specializing in housing law. A student has answered a legal question "
                "and provided their reasoning. Write a short passage (2-3 sentences) from a legal reference that "
                "would be most relevant to verifying or correcting this answer. Focus on the specific "
                "statute, regulation, or rule at the heart of the question. Write in reference style — "
                "state the law directly."
            ),
        }
        prompts["devil_hyde"] = (
            "You are a legal textbook author specializing in housing law. A student has answered a legal question. "
            "Your job is to play DEVIL'S ADVOCATE: write a short passage (2-3 sentences) from a legal reference "
            "that would CHALLENGE or CONTRADICT the student's answer. Focus on the rule, exception, or statute "
            "that supports the OPPOSITE conclusion. Write in reference style — state the law directly."
        )
        prompts["top2_snap"] = (
            "You are a legal expert specializing in housing law. Answer the Yes/No question below. "
            "Reason step by step. Identify what your FIRST choice answer is, and also what the ALTERNATIVE "
            "answer would be and why someone might argue for it. "
            "Give your final answer as: Answer: Yes or Answer: No"
        )
        prompts["top2_hyde"] = (
            "You are a legal textbook author specializing in housing law. A student has answered a legal question. "
            "Write a short passage (2-3 sentences) from a legal reference that would support the ALTERNATIVE "
            "or SECOND-CHOICE answer — the answer the student considered but rejected. Focus on the specific "
            "statute, regulation, or rule that would support that alternative. Write in reference style."
        )
        return prompts.get(role, prompts["answer"])
    if config.dataset == "casehold":
        prompts = {
            "answer": (
                "You are a legal expert specializing in case law. Read the citing context from a court opinion "
                "and determine which holding is most likely being referenced. "
                "Reason step by step, then give your final answer as: Answer: (X)"
            ),
            "rag": (
                "You are a legal expert specializing in case law. Reason through the question "
                "step by step. Retrieved holdings are provided — use them to verify or "
                "refine your reasoning, but think through the problem independently first. "
                "Give your final answer as: Answer: (X)"
            ),
            "research": (
                "You are a legal expert specializing in case law. Reason through the question "
                "step by step. Research findings are provided — use them to verify or "
                "refine your reasoning, but think through the problem independently first. "
                "Give your final answer as: Answer: (X)"
            ),
            "hyde": (
                "You are a legal textbook author. Given a court opinion excerpt that cites a holding, "
                "write a short passage (2-3 sentences) stating the likely holding being referenced. "
                "Write in the style of a case holding — state the rule directly."
            ),
            "snap_hyde": (
                "You are a legal textbook author. A student has identified what they think is the correct "
                "holding for a citation. Write a short passage (2-3 sentences) from a legal reference "
                "that would be most relevant to verifying this holding. Write in reference style."
            ),
        }
        return prompts.get(role, prompts["answer"])
    if config.dataset in ("legal_rag", "australian"):
        domain = "criminal law" if config.dataset == "legal_rag" else "Australian law"
        prompts = {
            "answer": (
                f"You are a legal expert specializing in {domain}. Answer the question below "
                f"thoroughly and accurately. Provide a detailed answer."
            ),
            "rag": (
                f"You are a legal expert specializing in {domain}. Reason through the question "
                f"step by step. Retrieved passages are provided — use them to verify or "
                f"refine your reasoning, but think through the problem independently first. "
                f"Provide a detailed answer."
            ),
            "research": (
                f"You are a legal expert specializing in {domain}. Reason through the question "
                f"step by step. Research findings are provided — use them to verify or "
                f"refine your reasoning, but think through the problem independently first. "
                f"Provide a detailed answer."
            ),
            "hyde": (
                f"You are a legal textbook author specializing in {domain}. Given a legal question, "
                f"write a short passage (2-3 sentences) that would appear in a reference guide as the answer. "
                f"Write in the style of a legal reference — state the rule directly."
            ),
            "snap_hyde": (
                f"You are a legal textbook author specializing in {domain}. A student has answered a legal question "
                f"and provided their reasoning. Write a short passage (2-3 sentences) from a legal reference that "
                f"would be most relevant to verifying or correcting this answer. Write in reference style."
            ),
        }
        return prompts.get(role, prompts["answer"])
    # BarExam defaults
    prompts = {
        "answer": (
            "You are a legal expert. Answer the multiple-choice question below. "
            "Reason step by step, then give your final answer as: Answer: (X)"
        ),
        "rag": _RAG_SYSTEM,
        "hyde": (
            "You are a legal textbook author. Write a short passage (2-3 sentences) from "
            "a legal reference that states the doctrine, rule, or exception most relevant "
            "to the question below. Write in reference style — state the law directly.\n\n"
            "STRICT OUTPUT RULES:\n"
            "- Begin your response with the doctrinal text itself.\n"
            "- Do NOT begin with 'Answer:', 'Answer (X)', or any multiple-choice letter.\n"
            "- Do NOT include headers like '**Passage:**', '**Legal Reference Passage:**', "
            "'Relevant Legal Passage:', or any label before the passage.\n"
            "- Do NOT mention the question, the choices, or which option is correct.\n"
            "- Do NOT use markdown bolding, bullet points, or section dividers.\n"
            "- Output only the passage body, nothing else."
        ),
        "snap_hyde": (
            "You are a legal textbook author. A student has answered a legal question and "
            "provided their reasoning. Write a short passage (2-3 sentences) from a legal "
            "reference stating the doctrine, rule, or exception most relevant to verifying "
            "or correcting that reasoning. Write in reference style — state the law directly.\n\n"
            "STRICT OUTPUT RULES:\n"
            "- Begin your response with the doctrinal text itself.\n"
            "- Do NOT begin with 'Answer:', 'Answer (X)', or any multiple-choice letter.\n"
            "- Do NOT include headers like '**Passage:**', '**Legal Reference Passage:**', "
            "'Relevant Legal Passage:', or any label before the passage.\n"
            "- Do NOT mention the student's reasoning, the question, the choices, or which "
            "option is correct.\n"
            "- Do NOT use markdown bolding, bullet points, or section dividers.\n"
            "- Output only the passage body, nothing else."
        ),
        "devil_hyde": (
            "You are a legal textbook author. A student has answered a legal question. "
            "Your job is to play DEVIL'S ADVOCATE: write a short passage (2-3 sentences) from "
            "a legal reference that would CHALLENGE or CONTRADICT the student's conclusion. "
            "Focus on the doctrine, rule, or exception that supports the opposite outcome. "
            "Write in reference style — state the law directly.\n\n"
            "STRICT OUTPUT RULES:\n"
            "- Begin your response with the doctrinal text itself.\n"
            "- Do NOT begin with 'Answer:', 'Answer (X)', or any multiple-choice letter.\n"
            "- Do NOT include headers like '**Passage:**' or any label before the passage.\n"
            "- Do NOT reference choice letters (A/B/C/D) or 'option X'.\n"
            "- Do NOT use markdown bolding, bullet points, or section dividers.\n"
            "- Output only the passage body."
        ),
        "top2_snap": (
            "You are a legal expert. Answer the multiple-choice question below. "
            "Reason step by step. Identify what your FIRST choice answer is, and also what your SECOND choice "
            "would be and why it's a plausible alternative. "
            "Give your final answer as: Answer: (X)"
        ),
        "top2_hyde": (
            "You are a legal textbook author. A student has answered a legal question and kept a "
            "second-choice alternative in mind. Write a short passage (2-3 sentences) from a legal "
            "reference that would support the SECOND-CHOICE conclusion — the one the student considered "
            "but rejected. Focus on the doctrine, rule, or exception that makes the alternative plausible. "
            "Write in reference style — state the law directly.\n\n"
            "STRICT OUTPUT RULES:\n"
            "- Begin your response with the doctrinal text itself.\n"
            "- Do NOT begin with 'Answer:', 'Answer (X)', or any multiple-choice letter.\n"
            "- Do NOT include headers like '**Passage:**' or any label before the passage.\n"
            "- Do NOT reference choice letters (A/B/C/D) or 'option X'.\n"
            "- Do NOT use markdown bolding, bullet points, or section dividers.\n"
            "- Output only the passage body."
        ),
    }
    return prompts.get(role, prompts["answer"])


DATASET_COLLECTIONS = {
    "barexam": "legal_passages",
    "housing": "housing_statutes",
    "legal_rag": "legal_rag_passages",
    "australian": "australian_legal",
    "casehold": "casehold_holdings",
    "musique": "musique_passages",
}


def _judge_open_answer(question: str, gold: str, predicted: str, config: EvalConfig) -> bool:
    """Use LLM to judge whether an open-ended answer is correct.

    Returns True if the predicted answer captures the key facts from the gold answer.
    Uses a simple binary correct/incorrect judgment to keep scoring consistent.
    """
    judge_system = (
        "You are a legal exam grader. Compare the student's answer to the reference answer. "
        "Judge whether the student's answer captures the key legal facts, rules, and conclusions "
        "from the reference answer. Minor differences in wording or additional context are acceptable. "
        "The student's answer must get the core legal point RIGHT to be correct.\n\n"
        "Respond with exactly one word: CORRECT or INCORRECT"
    )
    judge_user = (
        f"## Question\n{question}\n\n"
        f"## Reference Answer\n{gold}\n\n"
        f"## Student's Answer\n{predicted}"
    )
    verdict = _llm_call(judge_system, judge_user, label="judge")
    # Bug fix: "INCORRECT" contains "CORRECT", so a substring check silently
    # scored every negative verdict as correct. Extract a whole-word token.
    # Priority: if INCORRECT appears as a standalone word, it wins.
    upper = verdict.upper()
    if re.search(r"\bINCORRECT\b", upper):
        return False
    return bool(re.search(r"\bCORRECT\b", upper))


def _collection_for_config(config: EvalConfig) -> str:
    """Return the ChromaDB collection name for the dataset.

    Supports EVAL_COLLECTION_OVERRIDE env var for embedding A/B testing.
    When set, uses the override collection name instead of the default.
    """
    override = os.getenv("EVAL_COLLECTION_OVERRIDE", "").strip()
    if override:
        return override
    return DATASET_COLLECTIONS.get(config.dataset, "legal_passages")


def run_full_pipeline(row: pd.Series, config: EvalConfig) -> dict:
    """Run the full agentic pipeline and capture complete state."""
    question = format_question_prompt(row, dataset=config.dataset)
    result = run_pipeline(question, print_output=False)

    # Serialize PlanningStep objects
    planning_table = []
    for s in result.get("planning_table", []):
        if hasattr(s, "model_dump"):
            planning_table.append(s.model_dump())
        elif hasattr(s, "__dict__"):
            planning_table.append(vars(s))
        else:
            planning_table.append(s)

    # Check if gold passage was retrieved
    gold_idx = str(row.get("gold_idx", ""))
    retrieved_ids = [ev.get("idx", "") for ev in result.get("evidence_store", [])]
    gold_retrieved = gold_idx in retrieved_ids if gold_idx else False

    return {
        "final_answer": result.get("final_answer", ""),
        "collections": result.get("collections", []),
        "planning_table": planning_table,
        "evidence_store": result.get("evidence_store", []),
        "audit_log": result.get("audit_log", []),
        "completeness_verdict": result.get("completeness_verdict", {}),
        "parallel_rounds": result.get("parallel_round", 1) - 1,
        "retrieved_ids": retrieved_ids,
        "gold_retrieved": gold_retrieved,
    }


def run_llm_only(row: pd.Series, config: EvalConfig) -> dict:
    """Direct LLM answer with no retrieval."""
    question = _fmt(row, config)
    answer = _llm_call(_system_prompt(config, "answer"), question, label="llm_only")
    return {"final_answer": answer}


def run_golden_passage(row: pd.Series, config: EvalConfig) -> dict:
    """LLM answer with the gold passage injected as context."""
    question = _fmt(row, config)
    gold = str(row.get("gold_passage", ""))
    if not gold or gold == "nan":
        return run_llm_only(row, config)

    system = _system_prompt(config, "rag")
    user = f"## Reference Passage\n{gold}\n\n## Question\n{question}"
    answer = _llm_call(system, user, label="golden_passage")
    # The gold passage was injected directly — mark gold_retrieved=True so
    # downstream analyzers don't report this mode as "no retrieval". Keep the
    # retrieved_ids/evidence_store shape consistent with retrieval modes.
    gold_idx = str(row.get("gold_idx", ""))
    return {
        "final_answer": answer,
        "gold_retrieved": bool(gold_idx),
        "retrieved_ids": [gold_idx] if gold_idx else [],
        "evidence_store": [{"idx": gold_idx, "text": gold, "cross_encoder_score": 0.0}] if gold_idx else [],
    }


def _golden_arb_common(row: pd.Series, config: EvalConfig, arb_system: str, label_prefix: str) -> dict:
    """Shared logic for golden arbitration variants."""
    question = _fmt(row, config)
    gold = str(row.get("gold_passage", ""))
    if not gold or gold == "nan":
        return run_llm_only(row, config)

    # Step 1: Naive LLM answer (the "snap")
    snap_answer = _llm_call(_system_prompt(config, "answer"), question, label=f"{label_prefix}/snap")
    snap_letter = _extract_answer(snap_answer, config)

    # Step 2: Show reasoning (not the letter) + evidence, ask for a fresh verdict.
    arb_user = (
        f"## Your Previous Reasoning\n{_strip_answer_line(snap_answer)}\n\n"
        f"## Reference Passage\n{gold}\n\n"
        f"## Question\n{question}"
    )
    final_answer = _llm_call(arb_system, arb_user, label=f"{label_prefix}/arbitrate")
    final_letter = _extract_answer(final_answer, config)

    return {
        "final_answer": final_answer,
        "snap_answer": snap_answer,
        "snap_letter": snap_letter,
        "final_letter": final_letter,
        "changed": snap_letter != final_letter,
    }


def run_golden_arbitration(row: pd.Series, config: EvalConfig) -> dict:
    """LLM answers naively, then sees golden passage — neutral framing (no bias toward keeping/changing)."""
    arb_system = (
        "You are a legal expert. You previously answered a question based on your knowledge. "
        "Now you are given a reference passage that may contain relevant legal authority. "
        "Review the passage carefully against your previous reasoning. "
        "Reason step by step, then give your final answer as: Answer: (X)"
    )
    return _golden_arb_common(row, config, arb_system, "golden_arb")


def run_golden_arb_conservative(row: pd.Series, config: EvalConfig) -> dict:
    """LLM answers naively, then sees golden passage — conservative framing (biased toward keeping original)."""
    arb_system = (
        "You are a legal expert. You previously answered a question based on your knowledge. "
        "Now you are given a reference passage that may contain relevant legal authority. "
        "Review the passage carefully. If the evidence supports your original answer, keep it. "
        "If the evidence clearly points to a different answer, change it. "
        "Do not change your answer unless the evidence gives you a strong reason to. "
        "Reason step by step, then give your final answer as: Answer: (X)"
    )
    return _golden_arb_common(row, config, arb_system, "golden_arb_cons")


_musique_paragraphs_cache: dict | None = None


def _load_musique_paragraphs() -> dict:
    """Load MuSiQue passages.csv into a {q_id: [{idx,title,text}, ...]} cache."""
    global _musique_paragraphs_cache
    if _musique_paragraphs_cache is not None:
        return _musique_paragraphs_cache
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(os.path.join(base, "datasets/musique/passages.csv"))
    cache: dict = {}
    for _, r in df.iterrows():
        cache.setdefault(str(r["q_id"]), []).append({
            "idx": str(r["idx"]),
            "title": str(r.get("title", "")),
            "text": str(r["text"]),
            "is_supporting": bool(r.get("is_supporting", False)),
        })
    _musique_paragraphs_cache = cache
    return cache


def _retrieve_musique_in_row(row: pd.Series, queries: List[str], k: int = 5,
                             label_prefix: str = "rag") -> dict:
    """In-row BM25 retrieval over the question's ~20 paragraphs.

    Bypasses ChromaDB for MuSiQue — each question already carries its own
    paragraph pool (gold + distractors). Cheap, deterministic, no embedding
    model needed. Multi-query: max-score pooling across queries.
    """
    from rank_bm25 import BM25Okapi

    cache = _load_musique_paragraphs()
    q_id = str(row.get("idx", ""))
    paragraphs = cache.get(q_id, [])
    if not paragraphs:
        return {
            "passages": [], "evidence_store": [], "retrieved_ids": [],
            "gold_retrieved": False, "max_ce_score": 0.0,
        }

    docs_tokens = [p["text"].lower().split() for p in paragraphs]
    bm25 = BM25Okapi(docs_tokens)

    pooled = [-1e9] * len(paragraphs)
    for q in queries:
        if not q:
            continue
        q_tokens = str(q).lower().split()
        if not q_tokens:
            continue
        scores = bm25.get_scores(q_tokens)
        for i, s in enumerate(scores):
            if s > pooled[i]:
                pooled[i] = float(s)

    top_idxs = sorted(range(len(paragraphs)), key=lambda i: -pooled[i])[:k]
    top = [paragraphs[i] for i in top_idxs]

    passages = [f"[Source {i+1}]\n{p['title']}: {p['text']}" for i, p in enumerate(top)]
    evidence_store = [{
        "idx": p["idx"], "text": p["text"], "source": "musique_in_row",
        "cross_encoder_score": pooled[top_idxs[i]],
    } for i, p in enumerate(top)]
    retrieved_ids = [p["idx"] for p in top]

    gold_idx_str = str(row.get("gold_idx", ""))
    gold_idxs = {s.strip() for s in gold_idx_str.split(",") if s.strip()}
    gold_retrieved = bool(gold_idxs & set(retrieved_ids))
    max_ce_score = max(pooled) if pooled else 0.0

    _record_trace_event(
        "retrieval",
        label=label_prefix,
        queries=queries,
        rerank_query="",
        k=k,
        where={},
        collection="musique_in_row",
        embedding_model="bm25",
        results=evidence_store,
        retrieved_ids=retrieved_ids,
        gold_idx=gold_idx_str,
        gold_retrieved=gold_retrieved,
        max_ce_score=max_ce_score,
    )

    return {
        "passages": passages,
        "evidence_store": evidence_store,
        "retrieved_ids": retrieved_ids,
        "gold_retrieved": gold_retrieved,
        "max_ce_score": max_ce_score,
    }


def _retrieve_and_format(row: pd.Series, queries: List[str], k: int = 5,
                         label_prefix: str = "rag", where: dict = None,
                         collection: str = "legal_passages",
                         rerank_query: str = None) -> dict:
    """Shared retrieval + evidence formatting. Returns dict with passages, evidence_store, metadata.

    Args:
        rerank_query: If provided, cross-encoder reranks against this text instead of
            the retrieval queries. Decouples dense retrieval from reranking (e.g., HyDE
            for embedding but raw question for cross-encoder).
    """
    # MuSiQue uses in-row BM25 (no ChromaDB) — each question carries its own paragraph pool
    if collection == "musique_passages":
        return _retrieve_musique_in_row(row, queries, k=k, label_prefix=label_prefix)

    embedding_model = os.getenv("EVAL_EMBEDDING_MODEL", "").strip() or None
    vs = get_vectorstore(collection, embedding_model=embedding_model)
    docs = retrieve_documents_multi_query(queries=queries, k=k, vectorstore=vs, where=where,
                                          rerank_query=rerank_query)

    passages = []
    evidence_store = []
    for i, doc in enumerate(docs, 1):
        text = doc.page_content
        idx = str(doc.metadata.get("idx", f"{label_prefix}_{i}"))
        ce_score = doc.metadata.get("cross_encoder_score", 0.0)
        passages.append(f"[Source {i}]\n{text}")
        evidence_store.append({
            "idx": idx,
            "text": text,
            "source": doc.metadata.get("source", "unknown"),
            "cross_encoder_score": ce_score,
        })

    gold_idx = str(row.get("gold_idx", ""))
    retrieved_ids = [ev["idx"] for ev in evidence_store]
    gold_retrieved = gold_idx in retrieved_ids if gold_idx else False
    max_ce_score = max((ev["cross_encoder_score"] for ev in evidence_store), default=0.0)

    _record_trace_event(
        "retrieval",
        label=label_prefix,
        queries=queries,
        rerank_query=rerank_query or "",
        k=k,
        where=where or {},
        collection=collection,
        embedding_model=embedding_model or "",
        results=evidence_store,
        retrieved_ids=retrieved_ids,
        gold_idx=gold_idx,
        gold_retrieved=gold_retrieved,
        max_ce_score=max_ce_score,
    )

    return {
        "passages": passages,
        "evidence_store": evidence_store,
        "retrieved_ids": retrieved_ids,
        "gold_retrieved": gold_retrieved,
        "max_ce_score": max_ce_score,
    }


def _rewrite_query(question: str, label: str = "rag_rewrite/rewrite") -> List[str]:
    """LLM query rewrite → list of queries (primary + alternatives)."""
    rewrite_prompt = (
        f"Original legal research question: {question}\n\n"
        f"Sub-question: {question}\n"
        f"Authority target: \n"
        f"Retrieval hints: none"
    )
    raw_rewrite = _llm_call(load_skill("query_rewriter"), rewrite_prompt, label=label)
    parsed = _parse_json(raw_rewrite)

    if parsed and "primary" in parsed:
        return [parsed["primary"]] + parsed.get("alternatives", [])
    return [question]


def run_rag_hyde(row: pd.Series, config: EvalConfig) -> dict:
    """HyDE: generate hypothetical answer passage, embed it, retrieve similar real passages."""
    question = _fmt(row, config)
    question_intermediate = _fmt_intermediate(row, config)

    # Step 1: Generate hypothetical passage
    hyde = _generate_hyde(
        config,
        "hyde",
        _question_only_hyde_user(question_intermediate),
        label="hyde/generate",
        fallback=question_intermediate,
    )

    # Step 2: Retrieve using the hypothetical passage as query
    retrieval = _retrieve_and_format(row, [hyde["text"]], k=config.retrieval_k, label_prefix="hyde",
                                     where=_where_from_config(config),
                                     collection=_collection_for_config(config))
    passage_block = "\n\n".join(retrieval["passages"])

    # Step 3: Answer with evidence
    user = f"## Retrieved Passages\n{passage_block}\n\n## Question\n{question}"
    answer = _llm_call(_system_prompt(config, "rag"), user, label="hyde/answer")

    return {
        "final_answer": answer,
        "formatted_question": question,
        "intermediate_question": question_intermediate,
        "hyde_passage": hyde["text"],
        "hyde_passage_raw": hyde["raw"],
        "hyde_contains_answer_artifact": hyde["contains_answer"],
        "retrieval_queries": [hyde["text"]],
        "rerank_query": "",
        "final_context_fields": ["retrieved_passages", "question"],
        "final_prompt_preview": _preview_text(user),
        "evidence_store": retrieval["evidence_store"],
        "retrieved_ids": retrieval["retrieved_ids"],
        "gold_retrieved": retrieval["gold_retrieved"],
    }


def run_rag_multi_hyde(row: pd.Series, config: EvalConfig) -> dict:
    """Multi-HyDE: generate 3 hypothetical passages (rule/exception/application), pool retrievals."""
    question = _fmt(row, config)
    question_intermediate = _fmt_intermediate(row, config)

    system = (
        "You are a legal textbook author. Given a legal question, write THREE short passages "
        "(2-3 sentences each) that would appear in a study guide, targeting different dimensions:\n"
        "1. RULE: The governing legal rule or doctrine\n"
        "2. EXCEPTION: Key exceptions, defenses, or limitations\n"
        "3. APPLICATION: How the rule applies to specific facts\n\n"
        "Write each passage in the style of a legal reference. Separate with blank lines. "
        "Do not label them or discuss the question itself."
    )
    raw = _llm_call(system, question_intermediate, label="multi_hyde/generate")

    # Split into separate passages for retrieval, filter out empty
    raw_hyde_passages = [p.strip() for p in raw.split("\n\n") if p.strip() and len(p.strip()) > 30]
    hyde_passages = [
        _sanitize_intermediate_text(p, fallback=question_intermediate)
        for p in raw_hyde_passages
    ]
    if not hyde_passages:
        raw_hyde_passages = [raw]
        hyde_passages = [_sanitize_intermediate_text(raw, fallback=question_intermediate)]

    # Retrieve with each passage, pool results
    retrieval = _retrieve_and_format(row, hyde_passages, k=config.retrieval_k, label_prefix="multi_hyde",
                                     where=_where_from_config(config),
                                     collection=_collection_for_config(config))
    passage_block = "\n\n".join(retrieval["passages"])

    user = f"## Retrieved Passages\n{passage_block}\n\n## Question\n{question}"
    answer = _llm_call(_system_prompt(config, "rag"), user, label="multi_hyde/answer")

    return {
        "final_answer": answer,
        "hyde_passages": hyde_passages,
        "hyde_passages_raw": raw_hyde_passages,
        "hyde_contains_answer_artifact": any(_contains_answer_artifact(p) for p in raw_hyde_passages),
        "evidence_store": retrieval["evidence_store"],
        "retrieved_ids": retrieval["retrieved_ids"],
        "gold_retrieved": retrieval["gold_retrieved"],
    }


def run_multi_hyde_diverse(row: pd.Series, config: EvalConfig) -> dict:
    """Multi-HyDE designed to fight single-hop commitment bias on multi-hop QA.

    Failure mode being attacked: on MuSiQue, single HyDE commits to ONE
    wrong-hop entity (e.g. "Norah Jones" for "spouse of the Green performer")
    and BM25 retrieves only that entity's paragraphs, missing the actual gold.

    Mitigation: generate THREE diverse candidate answer-passages (different
    entities/angles), pool BM25/dense retrieval across all + the raw question
    as an anchor. Diversity is encouraged in the prompt but not enforced by
    de-duplication beyond the natural BM25 max-pooling.
    """
    question = _fmt(row, config)
    question_intermediate = _fmt_intermediate(row, config)

    system = (
        "You are a research assistant. Given a question, write THREE different "
        "short hypothetical answer-passages (2-3 sentences each).\n\n"
        "Diversity rules:\n"
        "- Each passage must give a DIFFERENT plausible answer or focus on a "
        "DIFFERENT entity/aspect/sub-question\n"
        "- Do NOT label them with numbers, headers, or bullets\n"
        "- Do NOT pick a multiple-choice letter (A/B/C/D) or commit to a single answer\n"
        "- Write each passage in factual encyclopedia/textbook style\n"
        "- Separate the three passages with one blank line"
    )
    raw = _llm_call(system, question_intermediate, label="multi_hyde_diverse/generate")

    raw_hyde_passages = [p.strip() for p in raw.split("\n\n") if p.strip() and len(p.strip()) > 30]
    hyde_passages = [
        _sanitize_intermediate_text(p, fallback=question_intermediate)
        for p in raw_hyde_passages
    ]
    routed_to = None
    if not hyde_passages:
        if not (raw or "").strip():
            raise RuntimeError("multi_hyde_diverse generate returned empty response")
        raw_hyde_passages = [raw or question_intermediate]
        hyde_passages = [_sanitize_intermediate_text(raw or question_intermediate, fallback=question_intermediate)]
        routed_to = "single_hyde_fallback_empty_gen"
    elif len(hyde_passages) < 3:
        routed_to = f"single_hyde_fallback_only_{len(hyde_passages)}_passages"

    queries = hyde_passages + [question_intermediate]

    retrieval = _retrieve_and_format(
        row, queries, k=config.retrieval_k, label_prefix="multi_hyde_diverse",
        where=_where_from_config(config),
        collection=_collection_for_config(config),
    )
    passage_block = "\n\n".join(retrieval["passages"])

    user = f"## Retrieved Passages\n{passage_block}\n\n## Question\n{question}"
    answer = _llm_call(_system_prompt(config, "rag"), user, label="multi_hyde_diverse/answer")

    out = {
        "final_answer": answer,
        "formatted_question": question,
        "retrieval_queries": hyde_passages,
        "rerank_query": "",
        "hyde_passages": hyde_passages,
        "hyde_passages_raw": raw_hyde_passages,
        "n_hyde_passages": len(hyde_passages),
        "hyde_contains_answer_artifact": any(_contains_answer_artifact(p) for p in raw_hyde_passages),
        "evidence_store": retrieval["evidence_store"],
        "retrieved_ids": retrieval["retrieved_ids"],
        "gold_retrieved": retrieval["gold_retrieved"],
    }
    if routed_to:
        out["routed_to"] = routed_to
    return out


def run_rag_snap_hyde(row: pd.Series, config: EvalConfig) -> dict:
    """Snap-informed HyDE: LLM answers first, then generates targeted HyDE passage based on its reasoning."""
    question = _fmt(row, config)
    question_intermediate = _fmt_intermediate(row, config)

    # Step 1: Snap answer
    snap_answer = _llm_call(_system_prompt(config, "answer"), question, label="snap_hyde/snap")
    snap_letter = _extract_answer(snap_answer, config)

    # Step 2: Generate HyDE passage informed by the snap reasoning
    hyde = _generate_hyde(
        config,
        "snap_hyde",
        _snap_hyde_user(question_intermediate, snap_answer),
        label="snap_hyde/generate",
        fallback=question_intermediate,
    )

    # Step 3: Retrieve
    retrieval = _retrieve_and_format(row, [hyde["text"]], k=config.retrieval_k, label_prefix="snap_hyde",
                                     where=_where_from_config(config),
                                     collection=_collection_for_config(config))
    passage_block = "\n\n".join(retrieval["passages"])

    # Step 4: Answer with evidence (direct, not arbitration — 70B does better without conservative bias)
    user = f"## Retrieved Passages\n{passage_block}\n\n## Question\n{question}"
    answer = _llm_call(_system_prompt(config, "rag"), user, label="snap_hyde/answer")

    return {
        "final_answer": answer,
        "formatted_question": question,
        "intermediate_question": question_intermediate,
        "snap_answer": snap_answer,
        "snap_letter": snap_letter,
        "hyde_passage": hyde["text"],
        "hyde_passage_raw": hyde["raw"],
        "hyde_contains_answer_artifact": hyde["contains_answer"],
        "retrieval_queries": [hyde["text"]],
        "rerank_query": "",
        "final_context_fields": ["retrieved_passages", "question"],
        "final_prompt_preview": _preview_text(user),
        "evidence_store": retrieval["evidence_store"],
        "retrieved_ids": retrieval["retrieved_ids"],
        "gold_retrieved": retrieval["gold_retrieved"],
    }


def _snap_hyde_2call_system(config: EvalConfig) -> str:
    """Compose a dataset-aware 2call system prompt: dataset's normal 'answer'
    system prompt + an additional requirement to emit a '## Passage' block
    after the answer. Keeps dataset-appropriate answer formatting (MC letter,
    Yes/No, open-ended) while adding the passage block for retrieval."""
    base_answer = _system_prompt(config, "answer")
    passage_instruction = (
        "\n\nADDITIONAL OUTPUT REQUIREMENT (REQUIRED, do not skip):\n"
        "After your final 'Answer:' line, append a blank line, then a header line that reads exactly:\n"
        "## Passage\n"
        "Followed by a 2-3 sentence reference passage that states the controlling rule, doctrine, "
        "fact, or principle most relevant to this question. The passage will be used to retrieve "
        "supporting context from a corpus. Constraints for the passage block:\n"
        "- Do NOT mention any answer choice (no '(A)', '(B)', 'Yes', 'No', etc.) in the passage.\n"
        "- Do NOT use 'Answer:', 'Passage:' headers inside the block, no bold, no bullets.\n"
        "- Write in plain reference / encyclopedia / treatise style — state the relevant fact "
        "or rule directly so it could appear verbatim in a knowledge source.\n"
        "Both the answer block AND the '## Passage' block are required."
    )
    return base_answer + passage_instruction


def _split_snap_and_hyde(raw: str, fallback_passage: str) -> tuple[str, str, bool]:
    """Parse a 2-call combined response into (snap_block, hyde_passage, parse_ok).

    Tries the canonical '## Passage' header first, then a few common variants.
    On parse failure, returns (raw, fallback_passage, False) so the caller can
    log a routed_to marker.
    """
    text = raw or ""
    for marker in ("## Passage", "##Passage", "## passage", "**Passage:**", "Passage:"):
        if marker in text:
            head, _, tail = text.partition(marker)
            snap_block = head.strip()
            hyde_passage = _sanitize_intermediate_text(tail.strip(), fallback=fallback_passage)
            if hyde_passage:
                return snap_block, hyde_passage, True
    return text.strip(), fallback_passage, False


def run_rag_snap_hyde_2call(row: pd.Series, config: EvalConfig) -> dict:
    """2-call efficiency variant of rag_snap_hyde: snap + HyDE in one LLM call,
    then retrieve, then final synthesis. Same final-context as rag_snap_hyde
    (retrieved passages only — snap letter NOT shown to final agent).

    Goal: preserve most of the rag_snap_hyde lift with 33% fewer LLM calls.
    """
    question = _fmt(row, config)
    question_intermediate = _fmt_intermediate(row, config)

    # Step 1: Single LLM call producing snap reasoning + HyDE passage.
    combined_raw = _llm_call(_snap_hyde_2call_system(config), question, label="snap_hyde_2call/snap_and_hyde")
    snap_block, hyde_passage, parse_ok = _split_snap_and_hyde(combined_raw, fallback_passage=question_intermediate)
    snap_letter = _extract_answer(snap_block, config)
    hyde_contains_answer = _contains_answer_artifact(hyde_passage)

    # Step 2: Retrieve using parsed HyDE passage (same retrieval as rag_snap_hyde).
    retrieval = _retrieve_and_format(row, [hyde_passage], k=config.retrieval_k, label_prefix="snap_hyde_2call",
                                     where=_where_from_config(config),
                                     collection=_collection_for_config(config))
    passage_block = "\n\n".join(retrieval["passages"])

    # Step 3: Final synthesis (call #2). Snap letter NOT shown — same final-context contract as rag_snap_hyde.
    user = f"## Retrieved Passages\n{passage_block}\n\n## Question\n{question}"
    answer = _llm_call(_system_prompt(config, "rag"), user, label="snap_hyde_2call/answer")

    out = {
        "final_answer": answer,
        "formatted_question": question,
        "intermediate_question": question_intermediate,
        "snap_answer": snap_block,
        "snap_letter": snap_letter,
        "snap_and_hyde_raw": combined_raw,
        "snap_hyde_2call_parse_ok": parse_ok,
        "hyde_passage": hyde_passage,
        "hyde_passage_raw": combined_raw,
        "hyde_contains_answer_artifact": hyde_contains_answer,
        "retrieval_queries": [hyde_passage],
        "rerank_query": "",
        "final_context_fields": ["retrieved_passages", "question"],
        "final_prompt_preview": _preview_text(user),
        "evidence_store": retrieval["evidence_store"],
        "retrieved_ids": retrieval["retrieved_ids"],
        "gold_retrieved": retrieval["gold_retrieved"],
    }
    if not parse_ok:
        out["routed_to"] = "snap_hyde_2call_parse_failed_fallback_to_question"
    return out


def run_snap_hyde_aligned(row: pd.Series, config: EvalConfig) -> dict:
    """Snap-HyDE with question-aligned reranking.

    Dense retrieval uses the HyDE passage (testing embedding model's passage→passage ability),
    but cross-encoder reranks against the raw question (same as rag_simple).
    This isolates the embedding model's contribution from the reranking step.
    """
    question = _fmt(row, config)
    raw_question = _retrieval_question(row)
    question_intermediate = _fmt_intermediate(row, config)

    # Step 1: Snap answer
    snap_answer = _llm_call(_system_prompt(config, "answer"), question, label="snap_hyde_aligned/snap")
    snap_letter = _extract_answer(snap_answer, config)

    # Step 2: Generate HyDE passage informed by the snap reasoning
    hyde = _generate_hyde(
        config,
        "snap_hyde",
        _snap_hyde_user(question_intermediate, snap_answer),
        label="snap_hyde_aligned/generate",
        fallback=question_intermediate,
    )

    # Step 3: Retrieve using HyDE for dense embedding, but rerank against raw question
    retrieval = _retrieve_and_format(row, [hyde["text"]], k=config.retrieval_k, label_prefix="snap_hyde_aligned",
                                     where=_where_from_config(config),
                                     collection=_collection_for_config(config),
                                     rerank_query=raw_question)
    passage_block = "\n\n".join(retrieval["passages"])

    # Step 4: Answer with evidence
    user = f"## Retrieved Passages\n{passage_block}\n\n## Question\n{question}"
    answer = _llm_call(_system_prompt(config, "rag"), user, label="snap_hyde_aligned/answer")

    return {
        "final_answer": answer,
        "formatted_question": question,
        "intermediate_question": question_intermediate,
        "snap_answer": snap_answer,
        "snap_letter": snap_letter,
        "hyde_passage": hyde["text"],
        "hyde_passage_raw": hyde["raw"],
        "hyde_contains_answer_artifact": hyde["contains_answer"],
        "retrieval_queries": [hyde["text"]],
        "rerank_query": raw_question,
        "final_context_fields": ["retrieved_passages", "question"],
        "final_prompt_preview": _preview_text(user),
        "evidence_store": retrieval["evidence_store"],
        "retrieved_ids": retrieval["retrieved_ids"],
        "gold_retrieved": retrieval["gold_retrieved"],
    }


def _gap_analysis(snap_answer: str, question: str) -> list[dict]:
    """Analyze gaps in the snap answer. Returns 0-3 structured gaps.

    Uses loose text format instead of JSON — more robust with small models.
    Each gap has: description (what's uncertain), sub_question (focused query).
    Returns empty list if model finds no gaps (high confidence).
    """
    system = (
        "You are a legal reasoning analyst. A student answered a legal question. "
        "Identify the single most important evidence gap that could prove the student's current answer wrong.\n\n"
        "Use this format:\n"
        "- gap: <what specific rule, fact, or exception is uncertain> | ask: <focused sub-question>\n\n"
        "Rules:\n"
        "- Stress-test the student's answer instead of helping it.\n"
        "- Prefer the unresolved rule, fact, or exception that, if answered the other way, would most likely reverse the student's conclusion.\n"
        "- Do not ask a confirmatory question that simply assumes the student's legal premise is already correct.\n"
        "- Give exactly 1 gap — the one most likely to change the answer.\n"
        "- Only give 2 gaps if there are truly two independent uncertainties.\n"
        "- If the reasoning is solid and you are confident in the answer, reply exactly: NONE\n"
        "- Do not emit 'Answer: (X)' or any multiple-choice letter in your gap output."
    )
    # Strip trailing 'Answer: (X)' from the snap so the analyst focuses on the
    # reasoning rather than the answer letter itself.
    user = (
        f"## Student's Answer and Reasoning\n{_strip_answer_line(snap_answer)}\n\n"
        f"## Original Question\n{question}"
    )
    raw = _llm_call(system, user, label="gap/analyze")

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        return []
    if len(lines) == 1 and lines[0].upper().rstrip(".") == "NONE":
        return []

    gaps = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or not line.startswith("-"):
            continue
        body = line.lstrip("- ").strip()
        # Case-insensitive parsing for model output variation
        body_lower = body.lower()
        if "| ask:" in body_lower:
            split_pos = body_lower.index("| ask:")
            desc = body[:split_pos].strip()
            subq = body[split_pos + 6:].strip()
        else:
            desc = body
            subq = ""
        # Strip "gap:" prefix if present (case-insensitive)
        if desc.lower().startswith("gap:"):
            desc = desc[4:].strip()
        subq = subq or desc
        if desc:
            gaps.append({"description": desc, "sub_question": subq})
    return gaps[:3]


GAP_MIN_CE = -100.0  # disabled — pass all evidence through (CE=1.0 was filtering 90%+ of passages)


def _gap_retrieve(gap: dict, question: str, row: pd.Series,
                  config: EvalConfig, gap_idx: int,
                  method: str = "hyde",
                  snap_answer: str = "") -> dict | None:
    """Run one gap investigation and return the gathered context.

    Supported methods:
      - 'hyde': generate a gap-focused hypothetical passage, then retrieve real passages
      - 'rag': retrieve directly from the gap sub-question
      - 'vectorless': generate a parametric legal note only (no corpus retrieval)
      - 'subagent_rag': retrieve passages, then summarize them into a short report
      - 'subagent_hybrid': retrieve passages, then synthesize a report with model knowledge
    """
    raw_question = _retrieval_question(row)
    desc = gap.get("description", "")
    subq = gap.get("sub_question", desc)

    if not desc and not subq:
        return None  # malformed gap, skip

    if method == "vectorless":
        # LLM generates knowledge per gap — no vector store
        gen_user = (
            f"## Evidence Gap\n{desc}\n\n"
            f"## Sub-question\n{subq}\n\n"
            f"## Original Question\n{question}"
        )
        knowledge_raw = _llm_call(_VECTORLESS_DIRECT, gen_user, label=f"gap/vless_{gap_idx}")
        knowledge = _sanitize_intermediate_text(knowledge_raw, fallback=knowledge_raw)
        return {
            "gap": gap,
            "passages": [f"[Generated Note]\n{knowledge}"],
            "evidence_store": [{"idx": f"vless_{gap_idx}", "text": knowledge, "cross_encoder_score": 0}],
            "max_ce_score": 0,
            "retrieval_query": "",
            "report": knowledge,
            "report_raw": knowledge_raw,
            "report_contains_answer_artifact": _contains_answer_artifact(knowledge_raw),
        }

    if method == "subagent_rag":
        # Subagent: RAG retrieves → LLM reads and summarizes findings
        query = subq or desc
        retrieval = _retrieve_and_format(
            row, [query], k=config.retrieval_k, label_prefix=f"sub_rag_{gap_idx}",
            where=_where_from_config(config),
            collection=_collection_for_config(config),
            rerank_query=raw_question,
        )
        passage_text = "\n\n".join(retrieval["passages"])
        # Subagent reads passages and writes a focused report
        report_user = (
            f"## Sub-question\n{subq}\n\n"
            f"## Retrieved Passages\n{passage_text}\n\n"
            f"## Original Question\n{question}"
        )
        report = _generate_report(
            _report_prompt(100),
            report_user,
            label=f"gap/sub_rag_{gap_idx}",
            fallback="Retrieved passages were not helpful.",
        )
        return {
            "gap": gap,
            "passages": retrieval["passages"],
            "evidence_store": retrieval["evidence_store"],
            "max_ce_score": retrieval["max_ce_score"],
            "retrieval_query": query,
            "report": report["text"],
            "report_raw": report["raw"],
            "report_contains_answer_artifact": report["contains_answer"],
        }

    if method == "subagent_hyde":
        # Subagent: HyDE retrieval (snap-informed) + LLM summarization
        gap_focus = []
        if desc:
            gap_focus.append(f"Evidence gap to verify: {desc}")
        if subq and subq != desc:
            gap_focus.append(f"Focused sub-question: {subq}")
        hyde = _generate_hyde(
            config,
            "snap_hyde",
            _snap_hyde_user(question, snap_answer, "\n".join(gap_focus)),
            label=f"gap/sub_hyde_{gap_idx}",
            fallback=subq or desc or question,
        )

        retrieval = _retrieve_and_format(
            row, [hyde["text"]], k=config.retrieval_k, label_prefix=f"sub_hyde_{gap_idx}",
            where=_where_from_config(config),
            collection=_collection_for_config(config),
            rerank_query=raw_question,
        )
        passage_text = "\n\n".join(retrieval["passages"])
        report_user = (
            f"## Sub-question\n{subq}\n\n"
            f"## Retrieved Passages\n{passage_text}\n\n"
            f"## Original Question\n{question}"
        )
        report = _generate_report(
            _report_prompt(100),
            report_user,
            label=f"gap/sub_hyde_rpt_{gap_idx}",
            fallback="Retrieved passages were not helpful.",
        )
        return {
            "gap": gap,
            "passages": retrieval["passages"],
            "evidence_store": retrieval["evidence_store"],
            "max_ce_score": retrieval["max_ce_score"],
            "retrieval_query": hyde["text"],
            "report": report["text"],
            "report_raw": report["raw"],
            "report_contains_answer_artifact": report["contains_answer"],
            "hyde_passage": hyde["text"],
            "hyde_passage_raw": hyde["raw"],
            "hyde_contains_answer_artifact": hyde["contains_answer"],
        }

    if method == "subagent_hybrid":
        # Subagent: RAG retrieves + LLM generates knowledge → combined report
        query = subq or desc
        retrieval = _retrieve_and_format(
            row, [query], k=3, label_prefix=f"sub_hyb_{gap_idx}",
            where=_where_from_config(config),
            collection=_collection_for_config(config),
            rerank_query=raw_question,
        )
        passage_text = "\n\n".join(retrieval["passages"])
        # Subagent synthesizes retrieved evidence + own knowledge
        report_user = (
            f"## Sub-question\n{subq}\n\n"
            f"## Retrieved Passages\n{passage_text}\n\n"
            f"## Original Question\n{question}"
        )
        report = _generate_report(
            _report_prompt(120, include_model_knowledge=True),
            report_user,
            label=f"gap/sub_hyb_{gap_idx}",
            fallback="Retrieved passages were not helpful.",
        )
        return {
            "gap": gap,
            "passages": retrieval["passages"],
            "evidence_store": retrieval["evidence_store"],
            "max_ce_score": retrieval["max_ce_score"],
            "retrieval_query": query,
            "report": report["text"],
            "report_raw": report["raw"],
            "report_contains_answer_artifact": report["contains_answer"],
        }

    if method == "hyde":
        gap_focus = []
        if desc:
            gap_focus.append(f"Evidence gap to verify: {desc}")
        if subq and subq != desc:
            gap_focus.append(f"Focused sub-question: {subq}")
        hyde = _generate_hyde(
            config,
            "snap_hyde",
            _snap_hyde_user(question, snap_answer, "\n".join(gap_focus)),
            label=f"gap/hyde_{gap_idx}",
            fallback=subq or desc or question,
        )
        query = hyde["text"]
    else:
        query = subq or desc

    retrieval = _retrieve_and_format(
        row, [query], k=config.retrieval_k, label_prefix=f"gap_{method}_{gap_idx}",
        where=_where_from_config(config),
        collection=_collection_for_config(config),
        rerank_query=raw_question,
    )

    if retrieval["max_ce_score"] < GAP_MIN_CE:
        return None

    result = {
        "gap": gap,
        "passages": retrieval["passages"],
        "evidence_store": retrieval["evidence_store"],
        "max_ce_score": retrieval["max_ce_score"],
        "retrieval_query": query,
    }
    if method == "hyde":
        result["hyde_passage"] = hyde["text"]
        result["hyde_passage_raw"] = hyde["raw"]
        result["hyde_contains_answer_artifact"] = hyde["contains_answer"]
    return result


def _build_gap_final_prompt(snap_answer: str, question: str, gaps: list[dict],
                            gap_results: list[dict | None], config: EvalConfig,
                            final_input: str = "full") -> tuple[str, str, list[str]]:
    """Assemble the final-answer prompt from the chosen gap artifacts.

    Supported final_input values:
      - 'full': snap answer + structured gap descriptions + evidence
      - 'evidence_only': flat evidence only
      - 'no_snap': structured gaps + evidence, but hide the snap answer
      - 'snap_and_evidence': snap answer + flat evidence, without gap structure
      - 'reports_nosnap': subagent/vectorless reports only
      - 'reports_and_evidence': reports plus the supporting raw passages
      - 'reports_and_snap': reports plus the snap answer (no raw passages)
      - 'reports_snap_evidence': reports + snap + raw passages (maximum info)
    """
    # Build evidence from gap results
    all_passages = []
    gap_sections = []
    for i, (gap, result) in enumerate(zip(gaps, gap_results), 1):
        desc = gap.get("description", f"Gap {i}")
        subq = gap.get("sub_question", "")
        if result is None:
            gap_sections.append(f"### Gap {i}: {desc}\nNo relevant evidence found.")
        else:
            passage_text = "\n\n".join(result["passages"])
            all_passages.extend(result["passages"])
            gap_sections.append(
                f"### Gap {i}: {desc}\n"
                f"Sub-question: {subq}\n"
                f"Retrieved evidence:\n{passage_text}"
            )

    # Build report sections (for subagent modes)
    report_sections = []
    for i, (gap, result) in enumerate(zip(gaps, gap_results), 1):
        desc = gap.get("description", f"Gap {i}")
        subq = gap.get("sub_question", "")
        report = result.get("report", "") if result else ""
        if result is None:
            report_sections.append(f"### Investigation {i}: {desc}\nNo findings.")
        elif report:
            report_sections.append(
                f"### Investigation {i}: {desc}\n"
                f"Sub-question: {subq}\n"
                f"Findings: {report}"
            )
        else:
            report_sections.append(
                f"### Investigation {i}: {desc}\n"
                f"Sub-question: {subq}\n"
                f"(No structured report available)"
            )

    gap_block = "\n\n".join(gap_sections) if gap_sections else "No evidence gaps identified."
    flat_passages = "\n\n".join(all_passages) if all_passages else "No evidence retrieved."
    report_block = "\n\n".join(report_sections) if report_sections else "No investigations completed."

    # Strip trailing 'Answer: (X)' from the snap reasoning when it is about to
    # be rendered back to the final agent — prevents the final call from simply
    # copying the snap letter (anchoring), while preserving the reasoning itself.
    snap_reasoning = _strip_answer_line(snap_answer)

    system = _system_prompt(config, "research" if final_input.startswith("reports") else "rag")
    context_fields: list[str]

    if final_input == "evidence_only":
        user = f"## Retrieved Passages\n{flat_passages}\n\n## Question\n{question}"
        context_fields = ["retrieved_passages", "question"]
    elif final_input == "no_snap":
        user = (
            f"## Evidence Gathered for Identified Gaps\n{gap_block}\n\n"
            f"## Question\n{question}"
        )
        context_fields = ["gap_evidence", "question"]
    elif final_input == "snap_and_evidence":
        user = (
            f"## Your Initial Reasoning\n{snap_reasoning}\n\n"
            f"## Retrieved Passages\n{flat_passages}\n\n"
            f"## Question\n{question}"
        )
        context_fields = ["snap_answer", "retrieved_passages", "question"]
    elif final_input == "reports_nosnap":
        user = (
            f"## Research Findings\n{report_block}\n\n"
            f"## Question\n{question}"
        )
        context_fields = ["research_findings", "question"]
    elif final_input == "reports_and_evidence":
        user = (
            f"## Research Findings\n{report_block}\n\n"
            f"## Supporting Passages\n{flat_passages}\n\n"
            f"## Question\n{question}"
        )
        context_fields = ["research_findings", "supporting_passages", "question"]
    elif final_input == "reports_and_snap":
        user = (
            f"## Your Initial Reasoning\n{snap_reasoning}\n\n"
            f"## Research Findings\n{report_block}\n\n"
            f"## Question\n{question}"
        )
        context_fields = ["snap_answer", "research_findings", "question"]
    elif final_input == "reports_snap_evidence":
        user = (
            f"## Your Initial Reasoning\n{snap_reasoning}\n\n"
            f"## Research Findings\n{report_block}\n\n"
            f"## Supporting Passages\n{flat_passages}\n\n"
            f"## Question\n{question}"
        )
        context_fields = ["snap_answer", "research_findings", "supporting_passages", "question"]
    else:  # full
        user = (
            f"## Your Initial Reasoning\n{snap_reasoning}\n\n"
            f"## Evidence Gathered for Identified Gaps\n{gap_block}\n\n"
            f"## Question\n{question}"
        )
        context_fields = ["snap_answer", "gap_evidence", "question"]

    return system, user, context_fields


def _gap_final_answer(snap_answer: str, question: str, gaps: list[dict],
                      gap_results: list[dict | None], config: EvalConfig,
                      final_input: str = "full") -> str:
    """Call the final answer stage for gap-driven methods."""
    system, user, _ = _build_gap_final_prompt(
        snap_answer, question, gaps, gap_results, config, final_input=final_input
    )
    return _llm_call(system, user, label="gap/final_answer")


def _run_gap(row: pd.Series, config: EvalConfig,
             method: str = "hyde", label: str = "gap_hyde",
             final_input: str = "full") -> dict:
    """Unified gap-informed retrieval: snap → gap analysis → per-gap retrieval → final answer.

    Args:
        method: one of 'hyde', 'rag', 'vectorless', 'subagent_rag', 'subagent_hybrid'
        label: prefix for LLM call labels
        final_input: one of 'full', 'evidence_only', 'no_snap', 'snap_and_evidence',
            'reports_nosnap', or 'reports_and_evidence'
    """
    question = _fmt(row, config)
    raw_question = _retrieval_question(row)
    question_intermediate = _fmt_intermediate(row, config)

    # Step 1: Snap
    snap_answer = _llm_call(_system_prompt(config, "answer"), question, label=f"{label}/snap")
    snap_letter = _extract_answer(snap_answer, config)

    # Step 2: Gap analysis
    gaps = _gap_analysis(snap_answer, question_intermediate)

    # Step 3: No gaps → use snap directly (unless reports_nosnap, then fresh answer)
    # Audit 2026-04-26 (silent_fallback_audit) caught that small-model runs hit
    # this fallback ~40% of the time, silently degrading to llm_only or snap-only
    # while reporting as the original mode. Add `routed_to` marker so post-hoc
    # analysis can stratify.
    if not gaps:
        if final_input in ("reports_nosnap", "reports_and_evidence", "no_snap", "evidence_only"):
            # Don't leak snap — make a fresh answer call
            fresh_answer = _llm_call(_system_prompt(config, "answer"), question, label=f"{label}/fresh")
            return {
                "final_answer": fresh_answer,
                "formatted_question": question,
                "intermediate_question": question_intermediate,
                "snap_answer": snap_answer,
                "snap_letter": snap_letter,
                "gaps": [],
                "gap_results": [],
                "retrieval_queries": [],
                "rerank_query": raw_question,
                "final_context_fields": ["question"],
                "final_prompt_preview": _preview_text(question),
                "evidence_store": [],
                "retrieved_ids": [],
                "gold_retrieved": False,
                "routed_to": "llm_only_fallback_no_gaps",  # silent-fallback marker
            }
        return {
            "final_answer": snap_answer,
            "formatted_question": question,
            "intermediate_question": question_intermediate,
            "snap_answer": snap_answer,
            "snap_letter": snap_letter,
            "gaps": [],
            "gap_results": [],
            "retrieval_queries": [],
            "rerank_query": raw_question,
            "final_context_fields": [],
            "final_prompt_preview": "",
            "evidence_store": [],
            "retrieved_ids": [],
            "gold_retrieved": False,
            "routed_to": "snap_only_fallback_no_gaps",  # silent-fallback marker
        }

    # Step 4: Per-gap retrieval
    gap_results = []
    all_evidence = []
    all_ids = []
    for i, gap in enumerate(gaps):
        result = _gap_retrieve(gap, question_intermediate, row, config, i, method=method, snap_answer=snap_answer)
        gap_results.append(result)
        if result:
            all_evidence.extend(result["evidence_store"])
            all_ids.extend([ev["idx"] for ev in result["evidence_store"]])

    # Step 5: Final answer
    gold_idx = str(row.get("gold_idx", ""))
    final_system, final_user, final_context_fields = _build_gap_final_prompt(
        snap_answer, question, gaps, gap_results, config, final_input=final_input
    )
    answer = _llm_call(final_system, final_user, label="gap/final_answer")

    return {
        "final_answer": answer,
        "formatted_question": question,
        "intermediate_question": question_intermediate,
        "snap_answer": snap_answer,
        "snap_letter": snap_letter,
        "gaps": gaps,
        "gap_results": [
            {
                "gap": r["gap"],
                "retrieval_query": r.get("retrieval_query", ""),
                "max_ce": r.get("max_ce_score", 0),
                "report": r.get("report", ""),
                "report_raw": r.get("report_raw", ""),
                "report_contains_answer_artifact": r.get("report_contains_answer_artifact", False),
                "hyde_passage": r.get("hyde_passage", ""),
                "hyde_passage_raw": r.get("hyde_passage_raw", ""),
                "hyde_contains_answer_artifact": r.get("hyde_contains_answer_artifact", False),
            } if r else None
            for r in gap_results
        ],
        "retrieval_queries": [r.get("retrieval_query", "") for r in gap_results if r and r.get("retrieval_query", "")],
        "rerank_query": raw_question,
        "final_context_fields": final_context_fields,
        "final_prompt_preview": _preview_text(final_user),
        "evidence_store": all_evidence,
        "retrieved_ids": all_ids,
        "gold_retrieved": gold_idx in all_ids if gold_idx else False,
    }


def run_gap_hyde(row: pd.Series, config: EvalConfig) -> dict:
    """Gap-informed HyDE: snap + gaps + evidence in final call (full context)."""
    return _run_gap(row, config, method="hyde", label="gap_hyde")


def run_gap_hyde_ev(row: pd.Series, config: EvalConfig) -> dict:
    """Gap-informed HyDE: evidence only in final call (no snap, no gap structure)."""
    return _run_gap(row, config, method="hyde", label="gap_hyde_ev", final_input="evidence_only")


def run_gap_hyde_nosnap(row: pd.Series, config: EvalConfig) -> dict:
    """Gap-informed HyDE: gaps + evidence but no snap answer in final call."""
    return _run_gap(row, config, method="hyde", label="gap_hyde_ns", final_input="no_snap")


def run_gap_hyde_flat(row: pd.Series, config: EvalConfig) -> dict:
    """Gap-informed HyDE: snap + flat evidence (no gap structure) in final call."""
    return _run_gap(row, config, method="hyde", label="gap_hyde_flat", final_input="snap_and_evidence")


def run_gap_rag_nosnap(row: pd.Series, config: EvalConfig) -> dict:
    """Gap RAG without snap in final — tests anchoring hypothesis.

    Same retrieval as gap_rag but hides snap answer from the final call.
    If this beats gap_rag (63.5%), anchoring is confirmed as the bottleneck.
    """
    return _run_gap(row, config, method="rag", label="gap_rag_ns", final_input="no_snap")


def run_gap_vectorless(row: pd.Series, config: EvalConfig) -> dict:
    """Gap + vectorless: gap analysis → per-gap LLM knowledge → reports only (no snap).

    Combines gap targeting with vectorless knowledge generation.
    Final call sees subagent reports + question only (no snap, no retrieval).
    """
    return _run_gap(row, config, method="vectorless", label="gap_vless", final_input="reports_nosnap")


def run_subagent_hyde(row: pd.Series, config: EvalConfig) -> dict:
    """Subagent HyDE: gap analysis → per-gap HyDE retrieval + LLM summarization → reports (no snap).

    Each subagent generates a HyDE passage for the gap, retrieves with it,
    reads the passages, and writes a focused report.
    Main agent sees subagent reports + question. No snap, no raw passages.
    """
    return _run_gap(row, config, method="subagent_hyde", label="sub_hyde", final_input="reports_nosnap")


def run_subagent_rag(row: pd.Series, config: EvalConfig) -> dict:
    """Subagent RAG: gap analysis → per-gap RAG + LLM summarization → reports only (no snap).

    Each subagent retrieves passages, reads them, and writes a focused report.
    Main agent sees subagent reports + question. No snap, no raw passages.
    """
    return _run_gap(row, config, method="subagent_rag", label="sub_rag", final_input="reports_nosnap")


def run_subagent_hybrid(row: pd.Series, config: EvalConfig) -> dict:
    """Subagent hybrid: gap analysis → per-gap RAG + LLM knowledge → synthesized reports (no snap).

    Each subagent retrieves passages AND generates own knowledge, then writes a combined report.
    Main agent sees subagent reports + question. No snap, no raw passages.
    """
    return _run_gap(row, config, method="subagent_hybrid", label="sub_hyb", final_input="reports_nosnap")


def run_subagent_rag_evidence(row: pd.Series, config: EvalConfig) -> dict:
    """Subagent RAG with evidence: reports + raw passages (no snap).

    Same as subagent_rag but main agent also sees the raw passages alongside reports.
    Tests whether raw evidence adds value on top of subagent summaries.
    """
    return _run_gap(row, config, method="subagent_rag", label="sub_rag_ev", final_input="reports_and_evidence")


def run_subagent_rag_snap(row: pd.Series, config: EvalConfig) -> dict:
    """Subagent RAG + snap visible: reports + snap answer in final call.

    Same retrieval/summarization as subagent_rag, but the final agent also sees
    the snap answer alongside the reports. Tests whether snap anchoring helps or
    hurts when mediated by independently-generated reports.
    """
    return _run_gap(row, config, method="subagent_rag", label="sub_rag_snap", final_input="reports_and_snap")


def run_subagent_rag_full(row: pd.Series, config: EvalConfig) -> dict:
    """Subagent RAG maximum info: reports + snap + raw passages.

    Final agent sees everything: snap answer, subagent reports, AND raw passages.
    Tests whether maximum information helps or creates noise/anchoring.
    """
    return _run_gap(row, config, method="subagent_rag", label="sub_rag_full", final_input="reports_snap_evidence")


def run_snap_hyde_report(row: pd.Series, config: EvalConfig) -> dict:
    """Snap-HyDE + summarization: snap_hyde retrieval pipeline with report denoising.

    Combines snap_hyde's proven retrieval (58.6% at N=1195) with subagent-style
    summarization. Unlike subagent modes, this skips gap analysis — retrieves with
    the full snap HyDE passage, summarizes ALL retrieved passages into a single
    report, then final agent sees report only (no snap, no raw passages).

    Tests whether noise filtering via summarization improves snap_hyde.
    """
    question = _fmt(row, config)
    question_intermediate = _fmt_intermediate(row, config)

    # Step 1: Snap answer
    snap_answer = _llm_call(_system_prompt(config, "answer"), question, label="shr/snap")
    snap_letter = _extract_answer(snap_answer, config)

    # Step 2: Generate HyDE passage from snap (same as snap_hyde)
    hyde = _generate_hyde(
        config,
        "snap_hyde",
        _snap_hyde_user(question_intermediate, snap_answer),
        label="shr/hyde",
        fallback=question_intermediate,
    )

    # Step 3: Retrieve using HyDE passage
    retrieval = _retrieve_and_format(row, [hyde["text"]], k=config.retrieval_k, label_prefix="shr",
                                     where=_where_from_config(config),
                                     collection=_collection_for_config(config))
    passage_block = "\n\n".join(retrieval["passages"])

    # Step 4: Summarize retrieved passages into a focused report
    report_user = (
        f"## Retrieved Passages\n{passage_block}\n\n"
        f"## Original Question\n{question_intermediate}"
    )
    report = _generate_report(
        _report_prompt(150),
        report_user,
        label="shr/report",
        fallback="Retrieved passages were not helpful.",
    )

    # Step 5: Final answer with report only (no snap, no raw passages)
    user = f"## Research Findings\n{report['text']}\n\n## Question\n{question}"
    answer = _llm_call(_system_prompt(config, "research"), user, label="shr/answer")

    return {
        "final_answer": answer,
        "formatted_question": question,
        "intermediate_question": question_intermediate,
        "snap_answer": snap_answer,
        "snap_letter": snap_letter,
        "hyde_passage": hyde["text"],
        "hyde_passage_raw": hyde["raw"],
        "hyde_contains_answer_artifact": hyde["contains_answer"],
        "retrieval_queries": [hyde["text"]],
        "rerank_query": "",
        "report": report["text"],
        "report_raw": report["raw"],
        "report_contains_answer_artifact": report["contains_answer"],
        "final_context_fields": ["research_findings", "question"],
        "final_prompt_preview": _preview_text(user),
        "evidence_store": retrieval["evidence_store"],
        "retrieved_ids": retrieval["retrieved_ids"],
        "gold_retrieved": retrieval["gold_retrieved"],
    }


def run_snap_hyde_report_snap(row: pd.Series, config: EvalConfig) -> dict:
    """Snap-HyDE + summarization + snap visible: report + snap in final call.

    Same as snap_hyde_report but the final agent also sees the snap answer.
    Tests snap anchoring when combined with a summarized report.
    """
    question = _fmt(row, config)
    question_intermediate = _fmt_intermediate(row, config)

    # Step 1: Snap
    snap_answer = _llm_call(_system_prompt(config, "answer"), question, label="shrs/snap")
    snap_letter = _extract_answer(snap_answer, config)

    # Step 2: HyDE from snap
    hyde = _generate_hyde(
        config,
        "snap_hyde",
        _snap_hyde_user(question_intermediate, snap_answer),
        label="shrs/hyde",
        fallback=question_intermediate,
    )

    # Step 3: Retrieve
    retrieval = _retrieve_and_format(row, [hyde["text"]], k=config.retrieval_k, label_prefix="shrs",
                                     where=_where_from_config(config),
                                     collection=_collection_for_config(config))
    passage_block = "\n\n".join(retrieval["passages"])

    # Step 4: Summarize
    report_user = (
        f"## Retrieved Passages\n{passage_block}\n\n"
        f"## Original Question\n{question_intermediate}"
    )
    report = _generate_report(
        _report_prompt(150),
        report_user,
        label="shrs/report",
        fallback="Retrieved passages were not helpful.",
    )

    # Step 5: Final answer with report + snap reasoning (letter stripped to avoid anchoring)
    user = (
        f"## Your Initial Reasoning\n{_strip_answer_line(snap_answer)}\n\n"
        f"## Research Findings\n{report['text']}\n\n"
        f"## Question\n{question}"
    )
    answer = _llm_call(_system_prompt(config, "research"), user, label="shrs/answer")

    return {
        "final_answer": answer,
        "formatted_question": question,
        "intermediate_question": question_intermediate,
        "snap_answer": snap_answer,
        "snap_letter": snap_letter,
        "hyde_passage": hyde["text"],
        "hyde_passage_raw": hyde["raw"],
        "hyde_contains_answer_artifact": hyde["contains_answer"],
        "retrieval_queries": [hyde["text"]],
        "rerank_query": "",
        "report": report["text"],
        "report_raw": report["raw"],
        "report_contains_answer_artifact": report["contains_answer"],
        "final_context_fields": ["snap_answer", "research_findings", "question"],
        "final_prompt_preview": _preview_text(user),
        "evidence_store": retrieval["evidence_store"],
        "retrieved_ids": retrieval["retrieved_ids"],
        "gold_retrieved": retrieval["gold_retrieved"],
    }


def run_gap_rag(row: pd.Series, config: EvalConfig) -> dict:
    """Gap-informed RAG: snap + gaps + evidence in final call (full context)."""
    return _run_gap(row, config, method="rag", label="gap_rag")


def run_snap_rag(row: pd.Series, config: EvalConfig) -> dict:
    """Snap + simple RAG: answer first, then retrieve with raw question, re-answer with snap + evidence.

    Tests whether snap context improves a simple RAG answer without any gap analysis or HyDE.
    2 LLM calls: snap + final answer with evidence.
    """
    question = _fmt(row, config)
    raw_question = _retrieval_question(row)

    # Step 1: Snap
    snap_answer = _llm_call(_system_prompt(config, "answer"), question, label="snap_rag/snap")
    snap_letter = _extract_answer(snap_answer, config)

    # Step 2: Retrieve with raw question, rerank against raw question (same as rag_simple)
    retrieval = _retrieve_and_format(row, [raw_question], k=config.retrieval_k, label_prefix="snap_rag",
                                     where=_where_from_config(config),
                                     collection=_collection_for_config(config))
    passage_block = "\n\n".join(retrieval["passages"])

    # Step 3: Final answer with snap context + evidence
    # Strip Answer: (X) line from snap before showing as context to avoid the
    # final agent simply echoing the snap letter (audit 2026-04-26 caught this).
    system = _system_prompt(config, "rag")
    user = (
        f"## Your Initial Reasoning\n{_strip_answer_line(snap_answer)}\n\n"
        f"## Retrieved Passages\n{passage_block}\n\n"
        f"## Question\n{question}"
    )
    answer = _llm_call(system, user, label="snap_rag/answer")

    return {
        "final_answer": answer,
        "snap_answer": snap_answer,
        "snap_letter": snap_letter,
        "evidence_store": retrieval["evidence_store"],
        "retrieved_ids": retrieval["retrieved_ids"],
        "gold_retrieved": retrieval["gold_retrieved"],
    }


def run_snap_rag_nosnap(row: pd.Series, config: EvalConfig) -> dict:
    """Snap + simple RAG but final call only sees evidence (no snap). Controls for whether snap helps final answer."""
    question = _fmt(row, config)
    raw_question = _retrieval_question(row)

    snap_answer = _llm_call(_system_prompt(config, "answer"), question, label="snap_rag_ns/snap")
    snap_letter = _extract_answer(snap_answer, config)

    retrieval = _retrieve_and_format(row, [raw_question], k=config.retrieval_k, label_prefix="snap_rag_ns",
                                     where=_where_from_config(config),
                                     collection=_collection_for_config(config))
    passage_block = "\n\n".join(retrieval["passages"])

    # Final answer WITHOUT snap — just evidence + question (same as rag_simple but 2 calls)
    user = f"## Retrieved Passages\n{passage_block}\n\n## Question\n{question}"
    answer = _llm_call(_system_prompt(config, "rag"), user, label="snap_rag_ns/answer")

    return {
        "final_answer": answer,
        "snap_answer": snap_answer,
        "snap_letter": snap_letter,
        "evidence_store": retrieval["evidence_store"],
        "retrieved_ids": retrieval["retrieved_ids"],
        "gold_retrieved": retrieval["gold_retrieved"],
    }


# ---------------------------------------------------------------------------
# Historical "vectorless" family: LLM generates parametric knowledge instead of
# searching the corpus. Same 3-call skeleton as rag_snap_hyde.
# ---------------------------------------------------------------------------

_VECTORLESS_FINAL = (
    "You are a legal expert. Reason through the multiple-choice question independently first. "
    "Generated legal reference notes are provided — use them to verify, refine, or challenge "
    "your reasoning, but do not treat them as automatically correct. "
    "If a note is generic, circular, or contradicted by stronger reasoning, ignore it. "
    "Give your final answer as: Answer: (X)"
)

_VECTORLESS_STRICT = (
    "\n\nSTRICT OUTPUT RULES:\n"
    "- Output ONLY the requested bullets — no preamble, no trailing summary.\n"
    "- Do NOT begin with 'Answer:', 'Answer (X)', or any multiple-choice letter.\n"
    "- Do NOT say 'the correct answer is' or reference options by letter.\n"
    "- Do NOT include markdown headers like '**Note:**' or '**Passage:**'."
)

_VECTORLESS_DIRECT = (
    "You are a legal reference guide. A student answered a legal question. "
    "Write a short doctrinal note to help verify or correct their answer.\n\n"
    "Return ONLY these 4 bullets:\n"
    "- Governing rule:\n"
    "- Key exception or limitation:\n"
    "- Dispositive fact trigger:\n"
    "- What would make a different answer plausible:\n\n"
    "State black-letter law directly. Keep under 120 words." + _VECTORLESS_STRICT
)

_VECTORLESS_ROLES = {
    "textbook": (
        "You are a legal textbook author. A student answered a legal question.\n"
        "Return ONLY 3 bullets:\n- Rule:\n- Exception/limitation:\n- Fact that controls:\n\n"
        "State the law directly. Keep under 90 words." + _VECTORLESS_STRICT
    ),
    "casebook": (
        "You are a casebook editor. A student answered a legal question.\n"
        "Return ONLY 3 bullets:\n- Holding-style rule:\n- Fact pattern that triggers it:\n"
        "- Common overread to avoid:\n\n"
        "Keep under 90 words." + _VECTORLESS_STRICT
    ),
    "barprep": (
        "You are a bar-prep tutor. A student answered a legal question.\n"
        "Return ONLY 3 bullets:\n- Rule:\n- Trap:\n- Decisive fact:\n\n"
        "Keep under 90 words." + _VECTORLESS_STRICT
    ),
}

_VECTORLESS_ELEMENTS = (
    "You are a legal issue spotter. A student answered a legal question.\n"
    "Identify the 2-4 dispositive legal elements and assess each.\n\n"
    "For each element, use this format:\n"
    "- [element name]: [rule] | fact=[fact signal] | pressure=[leans_correct/leans_wrong/ambiguous]\n\n"
    "Keep each element to one line." + _VECTORLESS_STRICT
)

_VECTORLESS_CHOICE_MAP = (
    "You are a bar exam differentiator. A student answered a legal question.\n"
    "Return ONLY 3 bullets:\n"
    "- Governing rule:\n"
    "- Strongest distractor pattern (the most plausible wrong answer and why):\n"
    "- Fact that flips the result:\n\n"
    "Focus on distinguishing the closest wrong answer. Keep under 90 words." + _VECTORLESS_STRICT
)


def _run_vectorless(row: pd.Series, config: EvalConfig,
                    gen_system: str, label: str = "vdirect",
                    include_snap: bool = False) -> dict:
    """Unified historical 'vectorless' flow: snap → generate parametric knowledge → final answer.

    Args:
        gen_system: system prompt for the knowledge generation step
        label: prefix for LLM call labels
        include_snap: if True, show snap answer in the final call alongside generated note
    """
    question = _fmt(row, config)
    question_intermediate = _fmt_intermediate(row, config)

    # Step 1: Snap
    snap_answer = _llm_call(_system_prompt(config, "answer"), question, label=f"{label}/snap")
    snap_letter = _extract_answer(snap_answer, config)

    # Step 2: Generate knowledge from parametric memory
    # Strip the snap's trailing 'Answer: (X)' so generation focuses on reasoning,
    # not the letter. Knowledge artifact is already sanitized below.
    snap_reasoning = _strip_answer_line(snap_answer)
    gen_user = f"## Student's Initial Analysis\n{snap_reasoning}\n\n## Original Question\n{question_intermediate}"
    knowledge_raw = _llm_call(gen_system, gen_user, label=f"{label}/generate")
    knowledge = _sanitize_intermediate_text(knowledge_raw, fallback=knowledge_raw)

    # Step 3: Final answer with generated knowledge
    if include_snap:
        final_user = (
            f"## Your Initial Reasoning\n{snap_reasoning}\n\n"
            f"## Generated Legal Reference Note\n{knowledge}\n\n"
            f"## Question\n{question}"
        )
    else:
        final_user = (
            f"## Generated Legal Reference Note\n{knowledge}\n\n"
            f"## Question\n{question}"
        )
    answer = _llm_call(_VECTORLESS_FINAL, final_user, label=f"{label}/answer")

    return {
        "final_answer": answer,
        "snap_answer": snap_answer,
        "snap_letter": snap_letter,
        "knowledge_note": knowledge,
        "knowledge_note_raw": knowledge_raw,
        "knowledge_contains_answer_artifact": _contains_answer_artifact(knowledge_raw),
        "evidence_store": [],
        "retrieved_ids": [],
        "gold_retrieved": False,
    }


def run_vectorless_direct(row: pd.Series, config: EvalConfig) -> dict:
    """Historical 'vectorless' parametric reasoning: snap → doctrinal note → answer."""
    return _run_vectorless(row, config, _VECTORLESS_DIRECT, label="vdirect")


def run_vectorless_role(row: pd.Series, config: EvalConfig) -> dict:
    """Historical 'vectorless' reasoning with role-conditioned generation. Use --tag textbook|casebook|barprep."""
    role = (config.tag.split("-")[-1] if config.tag else "barprep").strip().lower()
    system = _VECTORLESS_ROLES.get(role, _VECTORLESS_ROLES["barprep"])
    return _run_vectorless(row, config, system, label=f"vrole/{role}")


def run_vectorless_elements(row: pd.Series, config: EvalConfig) -> dict:
    """Historical 'vectorless' reasoning: snap → identify dispositive legal elements → answer."""
    return _run_vectorless(row, config, _VECTORLESS_ELEMENTS, label="velem")


def run_vectorless_choice_map(row: pd.Series, config: EvalConfig) -> dict:
    """Historical 'vectorless' reasoning: snap → map rule + distractor + decisive fact → answer."""
    return _run_vectorless(row, config, _VECTORLESS_CHOICE_MAP, label="vchoice")


def run_vectorless_nosnap(row: pd.Series, config: EvalConfig) -> dict:
    """Historical 'vectorless' reasoning without snap: question → generate knowledge → answer.

    Control for snap ablation. Compares with vectorless_direct (3 calls, with snap)
    to measure the snap contribution to vectorless knowledge generation.
    """
    question = _fmt(row, config)
    question_intermediate = _fmt_intermediate(row, config)

    # No snap — generate knowledge directly from the question
    gen_user = f"## Legal Question\n{question_intermediate}"
    knowledge_raw = _llm_call(_VECTORLESS_DIRECT, gen_user, label="vnosnap/generate")
    knowledge = _sanitize_intermediate_text(knowledge_raw, fallback=knowledge_raw)

    # Answer with generated knowledge
    final_user = (
        f"## Generated Legal Reference Note\n{knowledge}\n\n"
        f"## Question\n{question}"
    )
    answer = _llm_call(_VECTORLESS_FINAL, final_user, label="vnosnap/answer")

    return {
        "final_answer": answer,
        "snap_answer": "",
        "snap_letter": None,
        "knowledge_note": knowledge,
        "knowledge_note_raw": knowledge_raw,
        "knowledge_contains_answer_artifact": _contains_answer_artifact(knowledge_raw),
        "evidence_store": [],
        "retrieved_ids": [],
        "gold_retrieved": False,
    }


def run_vectorless_hybrid(row: pd.Series, config: EvalConfig) -> dict:
    """Hybrid: generated parametric knowledge + vector RAG evidence pooled together.

    Tests whether LLM-generated knowledge + retrieved passages > either alone.
    4 LLM calls: snap + generate knowledge + retrieve + answer with both.
    """
    question = _fmt(row, config)
    raw_question = _retrieval_question(row)
    question_intermediate = _fmt_intermediate(row, config)

    # Step 1: Snap
    snap_answer = _llm_call(_system_prompt(config, "answer"), question, label="vhybrid/snap")
    snap_letter = _extract_answer(snap_answer, config)

    # Step 2: Generate knowledge (vectorless). Strip the snap's Answer: (X)
    # line before feeding into the knowledge generator so we don't bias the
    # generated doctrine toward the snap's letter (audit 2026-04-26).
    gen_user = (
        f"## Student's Initial Analysis\n{_strip_answer_line(snap_answer)}\n\n"
        f"## Original Question\n{question_intermediate}"
    )
    knowledge_raw = _llm_call(_VECTORLESS_DIRECT, gen_user, label="vhybrid/generate")
    knowledge = _sanitize_intermediate_text(knowledge_raw, fallback=knowledge_raw)

    # Step 3: Also retrieve via snap_hyde path (vector RAG)
    hyde = _generate_hyde(
        config,
        "snap_hyde",
        _snap_hyde_user(question_intermediate, snap_answer),
        label="vhybrid/hyde",
        fallback=question_intermediate,
    )

    retrieval = _retrieve_and_format(row, [hyde["text"]], k=3, label_prefix="vhybrid",
                                     where=_where_from_config(config),
                                     collection=_collection_for_config(config))
    passage_block = "\n\n".join(retrieval["passages"])

    # Step 4: Answer with both generated knowledge and retrieved evidence
    final_user = (
        f"## Generated Legal Reference Note\n{knowledge}\n\n"
        f"## Retrieved Passages\n{passage_block}\n\n"
        f"## Question\n{question}"
    )
    answer = _llm_call(_VECTORLESS_FINAL, final_user, label="vhybrid/answer")

    return {
        "final_answer": answer,
        "snap_answer": snap_answer,
        "snap_letter": snap_letter,
        "knowledge_note": knowledge,
        "knowledge_note_raw": knowledge_raw,
        "knowledge_contains_answer_artifact": _contains_answer_artifact(knowledge_raw),
        "hyde_passage": hyde["text"],
        "hyde_passage_raw": hyde["raw"],
        "hyde_contains_answer_artifact": hyde["contains_answer"],
        "evidence_store": retrieval["evidence_store"],
        "retrieved_ids": retrieval["retrieved_ids"],
        "gold_retrieved": retrieval["gold_retrieved"],
    }


def run_vectorless_keyword(row: pd.Series, config: EvalConfig) -> dict:
    """Historical 'vectorless' keyword baseline: snap → generate search terms → retrieve → answer.

    This variant still searches the corpus. It asks the LLM for targeted keyword-style
    queries, retrieves with those queries, and reranks against the raw question.
    """
    from rag_utils import rerank_with_cross_encoder

    question = _fmt(row, config)
    question_intermediate = _fmt_intermediate(row, config)
    raw_question = _retrieval_question(row)

    # Step 1: Snap
    snap_answer = _llm_call(_system_prompt(config, "answer"), question, label="vkeyword/snap")
    snap_letter = _extract_answer(snap_answer, config)

    # Step 2: Generate targeted search terms (strip trailing snap letter so
    # generated keywords don't regurgitate 'Answer: (X)' as a query line).
    keyword_system = (
        "You are a legal research assistant. Based on a student's reasoning about a legal "
        "question, generate 3-5 specific search keywords or phrases to find relevant legal "
        "authorities.\n\n"
        "Focus on: legal doctrine names, rule names, statute sections, case law concepts, "
        "and specific legal terms that would appear in a legal reference.\n\n"
        "Return one search phrase per line, nothing else. No 'Answer:', no letter labels, "
        "no preamble or explanation — keywords only."
    )
    keyword_user = (
        f"## Student's Reasoning\n{_strip_answer_line(snap_answer)}\n\n"
        f"## Question\n{question_intermediate}"
    )
    keywords_raw = _llm_call(keyword_system, keyword_user, label="vkeyword/terms")

    # Parse keywords into search queries
    keywords = [
        _sanitize_intermediate_text(k.strip().lstrip("- •*0123456789."), fallback="")
        for k in keywords_raw.splitlines()
        if k.strip()
    ][:5]
    keywords = [k for k in keywords if k]

    # Step 3: Retrieve using each generated keyword, then rerank against the raw question.
    all_docs = []
    vs = get_vectorstore(_collection_for_config(config))
    for kw in keywords:
        if not kw:
            continue
        retriever = vs.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(kw)
        all_docs.extend(docs)

    # Dedup and rerank against raw question
    seen = set()
    unique_docs = []
    for doc in all_docs:
        idx = doc.metadata.get("idx", "")
        if idx not in seen:
            seen.add(idx)
            unique_docs.append(doc)

    reranked = rerank_with_cross_encoder(raw_question, unique_docs, top_k=config.retrieval_k)

    passages = [f"[Source {i+1}]\n{doc.page_content}" for i, doc in enumerate(reranked)]
    evidence_store = [{"idx": doc.metadata.get("idx", ""), "text": doc.page_content,
                       "cross_encoder_score": doc.metadata.get("cross_encoder_score", 0)}
                      for doc in reranked]

    passage_block = "\n\n".join(passages)
    gold_idx = str(row.get("gold_idx", ""))
    retrieved_ids = [e["idx"] for e in evidence_store]

    # Step 4: Answer with evidence
    final_user = (
        f"## Retrieved Legal Authorities\n{passage_block}\n\n"
        f"## Question\n{question}"
    )
    answer = _llm_call(_system_prompt(config, "rag"), final_user, label="vkeyword/answer")

    return {
        "final_answer": answer,
        "snap_answer": snap_answer,
        "snap_letter": snap_letter,
        "keywords": keywords,
        "evidence_store": evidence_store,
        "retrieved_ids": retrieved_ids,
        "gold_retrieved": gold_idx in retrieved_ids if gold_idx else False,
    }


# ---------------------------------------------------------------------------
# Entity graph search: real corpus search without embeddings.
# Uses pre-built NLP entity graph + inverted index from build_entity_graph.py.
# Zero LLM calls for preprocessing. 1-2 LLM calls at query time.
# ---------------------------------------------------------------------------

_ENTITY_GRAPH = None  # lazy-loaded singleton
_CORPUS_DF = None  # lazy-loaded corpus for entity search


def _load_entity_graph():
    """Load the entity graph and inverted index. Cached after first load."""
    global _ENTITY_GRAPH
    if _ENTITY_GRAPH is not None:
        return _ENTITY_GRAPH

    import json
    graph_path = os.path.join("datasets", "barexam_qa", "entity_graph", "entity_graph.json")
    if not os.path.exists(graph_path):
        print(f"[entity_graph] WARNING: {graph_path} not found. Run utils/build_entity_graph.py first.")
        return None

    print(f"[entity_graph] Loading graph from {graph_path}...")
    with open(graph_path) as f:
        _ENTITY_GRAPH = json.load(f)
    print(f"[entity_graph] Loaded: {_ENTITY_GRAPH['n_entities']:,} entities, "
          f"{_ENTITY_GRAPH['n_edges']:,} edges, {_ENTITY_GRAPH.get('n_communities', 0)} communities")
    return _ENTITY_GRAPH


def _entity_search(question: str, graph: dict, corpus_df=None, top_k: int = 15) -> list:
    """Search corpus via entity graph inverted index.

    1. Extract entities from question using same regex/spaCy as build time
    2. Look up each entity in inverted index → candidate passage IDs
    3. Score candidates by number of matching entities
    4. Return top_k passage IDs sorted by match count
    """
    import re

    inverted_index = graph['inverted_index']

    # Extract entities from question (simplified — matches build_entity_graph regex patterns)
    q_lower = question.lower()
    q_entities = set()

    # Legal Latin, doctrine names, and common legal concepts
    LEGAL_TERMS = [
        'res ipsa loquitur', 'habeas corpus', 'mens rea', 'actus reus',
        'prima facie', 'bona fide', 'due process', 'equal protection',
        'strict liability', 'negligence per se', 'proximate cause',
        'consideration', 'promissory estoppel', 'specific performance',
        'hearsay', 'adverse possession', 'easement', 'eminent domain',
        'felony murder', 'manslaughter', 'larceny', 'robbery', 'burglary', 'arson',
        'negligence', 'duty of care', 'breach of duty', 'causation', 'damages',
        'offer', 'acceptance', 'mutual assent', 'statute of frauds',
        'parol evidence', 'third party beneficiary', 'assignment',
        'search and seizure', 'miranda', 'exclusionary rule', 'probable cause',
        'reasonable suspicion', 'warrant', 'plain view',
        'double jeopardy', 'self incrimination', 'right to counsel',
        'free speech', 'establishment clause', 'free exercise', 'commerce clause',
        'substantive due process', 'procedural due process', 'rational basis',
        'strict scrutiny', 'intermediate scrutiny',
        'best evidence rule', 'business records', 'dying declaration',
        'excited utterance', 'present sense impression', 'prior inconsistent',
        'character evidence', 'impeachment', 'expert testimony',
        'fee simple', 'life estate', 'future interest', 'remainder',
        'covenant', 'servitude', 'recording act', 'bona fide purchaser',
        'joint tenancy', 'tenancy in common', 'landlord tenant',
        'intentional tort', 'battery', 'assault', 'false imprisonment',
        'trespass', 'conversion', 'defamation', 'libel', 'slander',
        'comparative negligence', 'contributory negligence', 'assumption of risk',
        'vicarious liability', 'respondeat superior', 'joint and several',
        'products liability', 'warranty', 'merchantability',
    ]
    for term in LEGAL_TERMS:
        if term in q_lower:
            q_entities.add(term)

    # Statute references
    for m in re.finditer(r'\b((?:section|rule|article|amendment|clause)\s+\w+)\b', q_lower):
        q_entities.add(m.group(1))

    # Quoted terms
    for m in re.finditer(r'"([^"]{3,50})"', question):
        phrase = m.group(1).lower().strip()
        if phrase in inverted_index:
            q_entities.add(phrase)

    # Multi-word capitalized phrases
    for m in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', question):
        phrase = m.group(1).lower()
        if phrase in inverted_index:
            q_entities.add(phrase)

    # Try 2-word and 3-word sliding window phrases against the index
    words = q_lower.split()
    for n in (2, 3):
        for i in range(len(words) - n + 1):
            phrase = " ".join(words[i:i+n])
            if phrase in inverted_index:
                q_entities.add(phrase)

    # Individual words (6+ chars) against the index
    for word in words:
        if len(word) > 5 and word in inverted_index:
            q_entities.add(word)

    if not q_entities:
        return []

    # Look up candidates
    from collections import Counter
    candidate_scores = Counter()
    for entity in q_entities:
        if entity in inverted_index:
            for pid in inverted_index[entity]:
                candidate_scores[pid] += 1

    # Return top_k by match count
    return [pid for pid, score in candidate_scores.most_common(top_k)]


def run_entity_search(row: pd.Series, config: EvalConfig) -> dict:
    """Entity graph search: real corpus search without embeddings.

    Uses pre-built NLP inverted index to find passages containing
    entities from the question. Cross-encoder reranks. 1 LLM call.
    Zero LLM preprocessing. Zero embeddings.
    """
    if config.dataset != "barexam":
        result = run_llm_only(row, config)
        result["entity_fallback"] = "dataset_not_supported"
        return result

    from rag_utils import rerank_with_cross_encoder
    import pandas as _pd

    question = _fmt(row, config)
    raw_question = _retrieval_question(row)

    graph = _load_entity_graph()
    if graph is None:
        # Fallback to llm_only
        answer = _llm_call(_system_prompt(config, "answer"), question, label="entity/answer")
        return {"final_answer": answer, "snap_answer": "", "snap_letter": None,
                "evidence_store": [], "retrieved_ids": [], "gold_retrieved": False}

    # Search via entity inverted index
    candidate_ids = _entity_search(raw_question, graph, top_k=30)

    if not candidate_ids:
        answer = _llm_call(_system_prompt(config, "answer"), question, label="entity/answer")
        return {"final_answer": answer, "snap_answer": "", "snap_letter": None,
                "evidence_store": [], "retrieved_ids": [], "gold_retrieved": False}

    # Load candidate passages from cached corpus
    global _CORPUS_DF
    if _CORPUS_DF is None:
        print("[entity_graph] Loading corpus CSV (one-time)...")
        _CORPUS_DF = _pd.read_csv("datasets/barexam_qa/barexam_qa_train.csv", usecols=['idx', 'text'])
        _CORPUS_DF['idx'] = _CORPUS_DF['idx'].astype(str)
    candidates = _CORPUS_DF[_CORPUS_DF['idx'].isin(set(candidate_ids))]

    if candidates.empty:
        answer = _llm_call(_system_prompt(config, "answer"), question, label="entity/answer")
        return {"final_answer": answer, "snap_answer": "", "snap_letter": None,
                "evidence_store": [], "retrieved_ids": [], "gold_retrieved": False}

    # Build Document objects for cross-encoder reranking
    from langchain_core.documents import Document
    docs = []
    for _, r in candidates.iterrows():
        doc = Document(page_content=str(r['text']), metadata={"idx": str(r['idx'])})
        docs.append(doc)

    reranked = rerank_with_cross_encoder(raw_question, docs, top_k=config.retrieval_k)

    passages = [f"[Source {i+1}]\n{doc.page_content}" for i, doc in enumerate(reranked)]
    evidence_store = [{"idx": doc.metadata.get("idx", ""), "text": doc.page_content,
                       "cross_encoder_score": doc.metadata.get("cross_encoder_score", 0)}
                      for doc in reranked]
    passage_block = "\n\n".join(passages)
    gold_idx = str(row.get("gold_idx", ""))
    retrieved_ids = [e["idx"] for e in evidence_store]

    # Single LLM call — answer with retrieved passages
    user = f"## Retrieved Passages\n{passage_block}\n\n## Question\n{question}"
    answer = _llm_call(_system_prompt(config, "rag"), user, label="entity/answer")

    return {
        "final_answer": answer,
        "snap_answer": "",
        "snap_letter": None,
        "evidence_store": evidence_store,
        "retrieved_ids": retrieved_ids,
        "gold_retrieved": gold_idx in retrieved_ids if gold_idx else False,
    }


def run_snap_entity_search(row: pd.Series, config: EvalConfig) -> dict:
    """Snap + entity graph search: snap first, then entity search, answer fresh.

    2 LLM calls. Snap steers nothing (entity search uses the raw question),
    but the snap reasoning may still help the final answer indirectly.
    Tests snap contribution to entity-based retrieval.
    """
    if config.dataset != "barexam":
        result = run_llm_only(row, config)
        result["entity_fallback"] = "dataset_not_supported"
        return result

    from rag_utils import rerank_with_cross_encoder

    question = _fmt(row, config)
    raw_question = _retrieval_question(row)

    # Step 1: Snap
    snap_answer = _llm_call(_system_prompt(config, "answer"), question, label="snap_entity/snap")
    snap_letter = _extract_answer(snap_answer, config)

    # Step 2: Entity search (same as run_entity_search but we have snap)
    graph = _load_entity_graph()
    if graph is None:
        # Fallback: answer fresh without snap (don't leak snap)
        answer = _llm_call(_system_prompt(config, "answer"), question, label="snap_entity/fresh")
        return {"final_answer": answer, "snap_answer": snap_answer,
                "snap_letter": snap_letter, "evidence_store": [],
                "retrieved_ids": [], "gold_retrieved": False}

    candidate_ids = _entity_search(raw_question, graph, top_k=30)

    if not candidate_ids:
        answer = _llm_call(_system_prompt(config, "answer"), question, label="snap_entity/fresh")
        return {"final_answer": answer, "snap_answer": snap_answer,
                "snap_letter": snap_letter, "evidence_store": [],
                "retrieved_ids": [], "gold_retrieved": False}

    import pandas as _pd
    from langchain_core.documents import Document
    global _CORPUS_DF
    if _CORPUS_DF is None:
        print("[entity_graph] Loading corpus CSV (one-time)...")
        _CORPUS_DF = _pd.read_csv("datasets/barexam_qa/barexam_qa_train.csv", usecols=['idx', 'text'])
        _CORPUS_DF['idx'] = _CORPUS_DF['idx'].astype(str)
    candidates = _CORPUS_DF[_CORPUS_DF['idx'].isin(set(candidate_ids))]

    if candidates.empty:
        answer = _llm_call(_system_prompt(config, "answer"), question, label="snap_entity/fresh")
        return {"final_answer": answer, "snap_answer": snap_answer,
                "snap_letter": snap_letter, "evidence_store": [],
                "retrieved_ids": [], "gold_retrieved": False}

    docs = [Document(page_content=str(r['text']), metadata={"idx": str(r['idx'])})
            for _, r in candidates.iterrows()]
    reranked = rerank_with_cross_encoder(raw_question, docs, top_k=config.retrieval_k)

    passages = [f"[Source {i+1}]\n{doc.page_content}" for i, doc in enumerate(reranked)]
    evidence_store = [{"idx": doc.metadata.get("idx", ""), "text": doc.page_content,
                       "cross_encoder_score": doc.metadata.get("cross_encoder_score", 0)}
                      for doc in reranked]
    passage_block = "\n\n".join(passages)
    gold_idx = str(row.get("gold_idx", ""))
    retrieved_ids = [e["idx"] for e in evidence_store]

    # Answer fresh — no snap shown (avoids anchoring)
    user = f"## Retrieved Passages\n{passage_block}\n\n## Question\n{question}"
    answer = _llm_call(_system_prompt(config, "rag"), user, label="snap_entity/answer")

    return {
        "final_answer": answer,
        "snap_answer": snap_answer,
        "snap_letter": snap_letter,
        "evidence_store": evidence_store,
        "retrieved_ids": retrieved_ids,
        "gold_retrieved": gold_idx in retrieved_ids if gold_idx else False,
    }


def run_snap_entity_informed(row: pd.Series, config: EvalConfig) -> dict:
    """Snap-informed entity search: extract entities from snap reasoning + question.

    The snap surfaces legal terms not in the question (e.g., question says
    "garage that encroached" but snap reasons about "adverse possession").
    Entity search equivalent of HyDE — snap reasoning generates better search terms.
    2 LLM calls. Snap hidden from final answer.
    """
    if config.dataset != "barexam":
        result = run_llm_only(row, config)
        result["entity_fallback"] = "dataset_not_supported"
        return result

    from rag_utils import rerank_with_cross_encoder
    import pandas as _pd
    from langchain_core.documents import Document

    question = _fmt(row, config)
    raw_question = _retrieval_question(row)

    # Step 1: Snap
    snap_answer = _llm_call(_system_prompt(config, "answer"), question, label="snap_ent_inf/snap")
    snap_letter = _extract_answer(snap_answer, config)

    # Step 2: Entity search using BOTH question AND snap reasoning
    graph = _load_entity_graph()
    if graph is None:
        answer = _llm_call(_system_prompt(config, "answer"), question, label="snap_ent_inf/fresh")
        return {"final_answer": answer, "snap_answer": snap_answer,
                "snap_letter": snap_letter, "evidence_store": [],
                "retrieved_ids": [], "gold_retrieved": False}

    # Extract entities from combined text — snap reasoning surfaces legal terms.
    # Strip the trailing 'Answer: (X)' from snap so the entity extractor doesn't
    # see letter tokens or 'option A' phrasings as entities.
    combined_text = f"{raw_question}\n\n{_strip_answer_line(snap_answer)}"
    candidate_ids = _entity_search(combined_text, graph, top_k=30)

    if not candidate_ids:
        answer = _llm_call(_system_prompt(config, "answer"), question, label="snap_ent_inf/fresh")
        return {"final_answer": answer, "snap_answer": snap_answer,
                "snap_letter": snap_letter, "evidence_store": [],
                "retrieved_ids": [], "gold_retrieved": False}

    # Load passages
    global _CORPUS_DF
    if _CORPUS_DF is None:
        print("[entity_graph] Loading corpus CSV (one-time)...")
        _CORPUS_DF = _pd.read_csv("datasets/barexam_qa/barexam_qa_train.csv", usecols=['idx', 'text'])
        _CORPUS_DF['idx'] = _CORPUS_DF['idx'].astype(str)
    candidates = _CORPUS_DF[_CORPUS_DF['idx'].isin(set(candidate_ids))]

    if candidates.empty:
        answer = _llm_call(_system_prompt(config, "answer"), question, label="snap_ent_inf/fresh")
        return {"final_answer": answer, "snap_answer": snap_answer,
                "snap_letter": snap_letter, "evidence_store": [],
                "retrieved_ids": [], "gold_retrieved": False}

    docs = [Document(page_content=str(r['text']), metadata={"idx": str(r['idx'])})
            for _, r in candidates.iterrows()]
    reranked = rerank_with_cross_encoder(raw_question, docs, top_k=config.retrieval_k)

    passages = [f"[Source {i+1}]\n{doc.page_content}" for i, doc in enumerate(reranked)]
    evidence_store = [{"idx": doc.metadata.get("idx", ""), "text": doc.page_content,
                       "cross_encoder_score": doc.metadata.get("cross_encoder_score", 0)}
                      for doc in reranked]
    passage_block = "\n\n".join(passages)
    gold_idx = str(row.get("gold_idx", ""))
    retrieved_ids = [e["idx"] for e in evidence_store]

    # Answer fresh — no snap shown
    user = f"## Retrieved Passages\n{passage_block}\n\n## Question\n{question}"
    answer = _llm_call(_system_prompt(config, "rag"), user, label="snap_ent_inf/answer")

    return {
        "final_answer": answer,
        "snap_answer": snap_answer,
        "snap_letter": snap_letter,
        "evidence_store": evidence_store,
        "retrieved_ids": retrieved_ids,
        "gold_retrieved": gold_idx in retrieved_ids if gold_idx else False,
    }


def run_ce_threshold(row: pd.Series, config: EvalConfig) -> dict:
    """Score-thresholded Snap-HyDE: if best CE score < threshold, discard evidence and use snap answer."""
    CE_THRESHOLD = 4.0  # calibrated from N=200 BarExam analysis: snap=78% below, RAG=78% above
    question = _fmt(row, config)
    question_intermediate = _fmt_intermediate(row, config)

    # Step 1: Snap answer
    snap_answer = _llm_call(_system_prompt(config, "answer"), question, label="ce_thresh/snap")
    snap_letter = _extract_answer(snap_answer, config)

    # Step 2: Generate HyDE passage
    hyde = _generate_hyde(
        config,
        "snap_hyde",
        _snap_hyde_user(question_intermediate, snap_answer),
        label="ce_thresh/generate",
        fallback=question_intermediate,
    )

    # Step 3: Retrieve
    retrieval = _retrieve_and_format(row, [hyde["text"]], k=config.retrieval_k, label_prefix="ce_thresh",
                                     where=_where_from_config(config),
                                     collection=_collection_for_config(config))

    # Step 4: Check CE threshold — if best passage is below threshold, use snap answer directly
    max_ce = retrieval["max_ce_score"]
    if max_ce < CE_THRESHOLD:
        return {
            "final_answer": snap_answer,
            "snap_answer": snap_answer,
            "snap_letter": snap_letter,
            "hyde_passage": hyde["text"],
            "hyde_passage_raw": hyde["raw"],
            "hyde_contains_answer_artifact": hyde["contains_answer"],
            "evidence_store": retrieval["evidence_store"],
            "retrieved_ids": retrieval["retrieved_ids"],
            "gold_retrieved": retrieval["gold_retrieved"],
            "routed_to": "snap_only",
            "max_ce_score": max_ce,
            "ce_threshold": CE_THRESHOLD,
        }

    # Step 5: Answer with evidence (above threshold — evidence is likely relevant)
    passage_block = "\n\n".join(retrieval["passages"])
    user = f"## Retrieved Passages\n{passage_block}\n\n## Question\n{question}"
    answer = _llm_call(_system_prompt(config, "rag"), user, label="ce_thresh/answer")

    return {
        "final_answer": answer,
        "snap_answer": snap_answer,
        "snap_letter": snap_letter,
        "hyde_passage": hyde["text"],
        "hyde_passage_raw": hyde["raw"],
        "hyde_contains_answer_artifact": hyde["contains_answer"],
        "evidence_store": retrieval["evidence_store"],
        "retrieved_ids": retrieval["retrieved_ids"],
        "gold_retrieved": retrieval["gold_retrieved"],
        "routed_to": "rag",
        "max_ce_score": max_ce,
        "ce_threshold": CE_THRESHOLD,
    }


def run_snap_hyde_aspect(row: pd.Series, config: EvalConfig) -> dict:
    """Snap-HyDE + aspect queries: HyDE passage + rule/exception aspect queries for broader retrieval."""
    question = _fmt(row, config)
    question_intermediate = _fmt_intermediate(row, config)

    # Step 1: Snap answer
    snap_answer = _llm_call(_system_prompt(config, "answer"), question, label="aspect/snap")
    snap_letter = _extract_answer(snap_answer, config)

    # Step 2: Generate HyDE passage
    hyde = _generate_hyde(
        config,
        "snap_hyde",
        _snap_hyde_user(question_intermediate, snap_answer),
        label="aspect/hyde",
        fallback=question_intermediate,
    )

    # Step 3: Generate aspect queries (rule + exception) based on the snap reasoning.
    # Strip the trailing snap letter so the query generator doesn't emit 'Answer:'
    # as one of its JSON values.
    aspect_prompt = (
        f"Based on this legal question and reasoning, generate two short search queries "
        f"targeting different legal dimensions. Return ONLY a JSON object.\n\n"
        f"Question: {question_intermediate}\n\n"
        f"Reasoning: {_strip_answer_line(snap_answer)}\n\n"
        f'Return: {{"rule": "query targeting the governing rule, statute, or doctrine", '
        f'"exception": "query targeting exceptions, defenses, or limitations"}}\n\n'
        f'Rules: queries must not contain letter labels (A/B/C/D), "Answer:", or option references.'
    )
    aspect_raw = _llm_call("You are a legal search query generator. Return ONLY valid JSON.",
                            aspect_prompt, label="aspect/queries")
    aspect_parsed = _parse_json(aspect_raw)

    # Build query list: HyDE passage (primary for reranking) + aspect queries
    queries = [hyde["text"]]
    if aspect_parsed:
        if "rule" in aspect_parsed:
            rule_query = _sanitize_intermediate_text(str(aspect_parsed["rule"]), fallback="")
            if rule_query:
                queries.append(rule_query)
                aspect_parsed["rule"] = rule_query
        if "exception" in aspect_parsed:
            exception_query = _sanitize_intermediate_text(str(aspect_parsed["exception"]), fallback="")
            if exception_query:
                queries.append(exception_query)
                aspect_parsed["exception"] = exception_query

    # Step 4: Multi-query retrieval (pools candidates from all queries, reranks against primary)
    retrieval = _retrieve_and_format(row, queries, k=config.retrieval_k, label_prefix="aspect",
                                     where=_where_from_config(config),
                                     collection=_collection_for_config(config))
    passage_block = "\n\n".join(retrieval["passages"])

    # Step 5: Answer with evidence
    user = f"## Retrieved Passages\n{passage_block}\n\n## Question\n{question}"
    answer = _llm_call(_system_prompt(config, "rag"), user, label="aspect/answer")

    return {
        "final_answer": answer,
        "snap_answer": snap_answer,
        "snap_letter": snap_letter,
        "hyde_passage": hyde["text"],
        "hyde_passage_raw": hyde["raw"],
        "hyde_contains_answer_artifact": hyde["contains_answer"],
        "aspect_queries": aspect_parsed,
        "num_queries": len(queries),
        "evidence_store": retrieval["evidence_store"],
        "retrieved_ids": retrieval["retrieved_ids"],
        "gold_retrieved": retrieval["gold_retrieved"],
    }


def run_ce_threshold_k3(row: pd.Series, config: EvalConfig) -> dict:
    """CE-thresholded Snap-HyDE with k=3 instead of k=5. Tests whether fewer, higher-quality passages help."""
    CE_THRESHOLD = 4.0
    question = _fmt(row, config)
    question_intermediate = _fmt_intermediate(row, config)

    snap_answer = _llm_call(_system_prompt(config, "answer"), question, label="ce_k3/snap")
    snap_letter = _extract_answer(snap_answer, config)

    hyde = _generate_hyde(
        config,
        "snap_hyde",
        _snap_hyde_user(question_intermediate, snap_answer),
        label="ce_k3/generate",
        fallback=question_intermediate,
    )

    retrieval = _retrieve_and_format(row, [hyde["text"]], k=3, label_prefix="ce_k3",
                                     where=_where_from_config(config),
                                     collection=_collection_for_config(config))

    max_ce = retrieval["max_ce_score"]
    if max_ce < CE_THRESHOLD:
        return {
            "final_answer": snap_answer,
            "snap_answer": snap_answer,
            "snap_letter": snap_letter,
            "hyde_passage": hyde["text"],
            "hyde_passage_raw": hyde["raw"],
            "hyde_contains_answer_artifact": hyde["contains_answer"],
            "evidence_store": retrieval["evidence_store"],
            "retrieved_ids": retrieval["retrieved_ids"],
            "gold_retrieved": retrieval["gold_retrieved"],
            "routed_to": "snap_only",
            "max_ce_score": max_ce,
            "k": 3,
        }

    passage_block = "\n\n".join(retrieval["passages"])
    user = f"## Retrieved Passages\n{passage_block}\n\n## Question\n{question}"
    answer = _llm_call(_system_prompt(config, "rag"), user, label="ce_k3/answer")

    return {
        "final_answer": answer,
        "snap_answer": snap_answer,
        "snap_letter": snap_letter,
        "hyde_passage": hyde["text"],
        "hyde_passage_raw": hyde["raw"],
        "hyde_contains_answer_artifact": hyde["contains_answer"],
        "evidence_store": retrieval["evidence_store"],
        "retrieved_ids": retrieval["retrieved_ids"],
        "gold_retrieved": retrieval["gold_retrieved"],
        "routed_to": "rag",
        "max_ce_score": max_ce,
        "k": 3,
    }


def run_rag_devil_hyde(row: pd.Series, config: EvalConfig) -> dict:
    """Devil's advocate HyDE: retrieve for snap answer AND for the opposing answer, present both."""
    question = _fmt(row, config)
    question_intermediate = _fmt_intermediate(row, config)

    # Step 1: Snap answer
    snap_answer = _llm_call(_system_prompt(config, "answer"), question, label="devil_hyde/snap")
    snap_letter = _extract_answer(snap_answer, config)

    # Step 2: Generate supporting HyDE passage (same as snap_hyde)
    hyde_user = _snap_hyde_user(question_intermediate, snap_answer)
    support = _generate_hyde(
        config,
        "snap_hyde",
        hyde_user,
        label="devil_hyde/support",
        fallback=question_intermediate,
    )

    # Step 3: Generate devil's advocate HyDE passage (opposing the snap answer)
    devil = _generate_hyde(
        config,
        "devil_hyde",
        hyde_user,
        label="devil_hyde/oppose",
        fallback=question_intermediate,
    )

    # Step 4: Retrieve with BOTH passages pooled
    collection = _collection_for_config(config)
    retrieval = _retrieve_and_format(row, [support["text"], devil["text"]], k=config.retrieval_k,
                                     label_prefix="devil_hyde",
                                     where=_where_from_config(config),
                                     collection=collection)
    passage_block = "\n\n".join(retrieval["passages"])

    # Step 5: Answer with evidence (direct — let model weigh both sides)
    user = f"## Retrieved Passages\n{passage_block}\n\n## Question\n{question}"
    answer = _llm_call(_system_prompt(config, "rag"), user, label="devil_hyde/answer")

    return {
        "final_answer": answer,
        "snap_answer": snap_answer,
        "snap_letter": snap_letter,
        "support_passage": support["text"],
        "support_passage_raw": support["raw"],
        "support_contains_answer_artifact": support["contains_answer"],
        "devil_passage": devil["text"],
        "devil_passage_raw": devil["raw"],
        "devil_contains_answer_artifact": devil["contains_answer"],
        "evidence_store": retrieval["evidence_store"],
        "retrieved_ids": retrieval["retrieved_ids"],
        "gold_retrieved": retrieval["gold_retrieved"],
    }


def run_rag_top2_hyde(row: pd.Series, config: EvalConfig) -> dict:
    """Top-2 HyDE: snap answer identifies top 2 choices, generate HyDE for each, pool retrieval."""
    question = _fmt(row, config)
    question_intermediate = _fmt_intermediate(row, config)

    # Step 1: Snap answer — ask for top 2 choices with reasoning
    top2_system = _system_prompt(config, "top2_snap")
    snap_answer = _llm_call(top2_system, question, label="top2_hyde/snap")
    snap_letter = _extract_answer(snap_answer, config)

    # Step 2: Generate HyDE for primary answer
    hyde_user_1 = _snap_hyde_user(question_intermediate, snap_answer)
    hyde_1 = _generate_hyde(
        config,
        "snap_hyde",
        hyde_user_1,
        label="top2_hyde/primary",
        fallback=question_intermediate,
    )

    # Step 3: Generate HyDE for second-choice answer
    hyde_2 = _generate_hyde(
        config,
        "top2_hyde",
        hyde_user_1,
        label="top2_hyde/secondary",
        fallback=question_intermediate,
    )

    # Step 4: Retrieve with both HyDE passages
    collection = _collection_for_config(config)
    retrieval = _retrieve_and_format(row, [hyde_1["text"], hyde_2["text"]], k=config.retrieval_k,
                                     label_prefix="top2_hyde",
                                     where=_where_from_config(config),
                                     collection=collection)
    passage_block = "\n\n".join(retrieval["passages"])

    # Step 5: Answer with evidence
    user = f"## Retrieved Passages\n{passage_block}\n\n## Question\n{question}"
    answer = _llm_call(_system_prompt(config, "rag"), user, label="top2_hyde/answer")

    return {
        "final_answer": answer,
        "snap_answer": snap_answer,
        "snap_letter": snap_letter,
        "hyde_primary": hyde_1["text"],
        "hyde_primary_raw": hyde_1["raw"],
        "hyde_primary_contains_answer_artifact": hyde_1["contains_answer"],
        "hyde_secondary": hyde_2["text"],
        "hyde_secondary_raw": hyde_2["raw"],
        "hyde_secondary_contains_answer_artifact": hyde_2["contains_answer"],
        "evidence_store": retrieval["evidence_store"],
        "retrieved_ids": retrieval["retrieved_ids"],
        "gold_retrieved": retrieval["gold_retrieved"],
    }


def run_rag_hyde_arb(row: pd.Series, config: EvalConfig) -> dict:
    """HyDE retrieval + conservative arbitration: snap → HyDE retrieve → review."""
    question = _fmt(row, config)
    question_intermediate = _fmt_intermediate(row, config)

    # Step 1: Snap answer
    snap_answer = _llm_call(_system_prompt(config, "answer"), question, label="hyde_arb/snap")
    snap_letter = _extract_answer(snap_answer, config)

    # Step 2: HyDE retrieval
    hyde = _generate_hyde(
        config,
        "hyde",
        _question_only_hyde_user(question_intermediate),
        label="hyde_arb/generate",
        fallback=question_intermediate,
    )
    retrieval = _retrieve_and_format(row, [hyde["text"]], k=config.retrieval_k, label_prefix="hyde_arb",
                                     where=_where_from_config(config),
                                     collection=_collection_for_config(config))
    passage_block = "\n\n".join(retrieval["passages"])

    # Step 3: Arbitrate
    arb_system = (
        "You are a legal expert. You previously answered a question based on your knowledge. "
        "Now you are given retrieved legal passages that may be relevant. "
        "Review the passages carefully. If the evidence supports your original answer, keep it. "
        "If the evidence clearly points to a different answer, change it. "
        "Do not change your answer unless the evidence gives you a strong reason to. "
        "Reason step by step, then give your final answer as: Answer: (X)"
    )
    arb_user = (
        f"## Your Previous Reasoning\n{_strip_answer_line(snap_answer)}\n\n"
        f"## Retrieved Passages\n{passage_block}\n\n"
        f"## Question\n{question}"
    )
    final_answer = _llm_call(arb_system, arb_user, label="hyde_arb/arbitrate")
    final_letter = _extract_answer(final_answer, config)

    return {
        "final_answer": final_answer,
        "snap_answer": snap_answer,
        "snap_letter": snap_letter,
        "final_letter": final_letter,
        "changed": snap_letter != final_letter,
        "hyde_passage": hyde["text"],
        "hyde_passage_raw": hyde["raw"],
        "hyde_contains_answer_artifact": hyde["contains_answer"],
        "evidence_store": retrieval["evidence_store"],
        "retrieved_ids": retrieval["retrieved_ids"],
        "gold_retrieved": retrieval["gold_retrieved"],
    }


_RAG_SYSTEM = (
    "You are a legal expert. Reason through the multiple-choice question "
    "step by step. Retrieved passages are provided — use them to verify or "
    "refine your reasoning, but think through the problem independently first. "
    "Give your final answer as: Answer: (X)"
)


def _where_from_config(config: EvalConfig) -> dict | None:
    """Build ChromaDB where filter from config.source_filter."""
    if config.source_filter:
        return {"source": config.source_filter}
    return None


def run_rag_rewrite(row: pd.Series, config: EvalConfig) -> dict:
    """Query rewrite → retrieval → answer with evidence."""
    question = _fmt(row, config)
    question_intermediate = _fmt_intermediate(row, config)
    queries = _rewrite_query(question_intermediate)

    retrieval = _retrieve_and_format(row, queries, k=config.retrieval_k, label_prefix="rewrite",
                                     where=_where_from_config(config),
                                     collection=_collection_for_config(config))
    passage_block = "\n\n".join(retrieval["passages"])

    user = f"## Retrieved Passages\n{passage_block}\n\n## Question\n{question}"
    answer = _llm_call(_system_prompt(config, "rag"), user, label="rag_rewrite/answer")

    return {
        "final_answer": answer,
        "evidence_store": retrieval["evidence_store"],
        "retrieved_ids": retrieval["retrieved_ids"],
        "gold_retrieved": retrieval["gold_retrieved"],
        "rewrite_queries": queries,
    }


def run_rag_simple(row: pd.Series, config: EvalConfig) -> dict:
    """Raw question → retrieval → answer with evidence (no rewrite)."""
    question = _fmt(row, config)
    raw_question = _retrieval_question(row)

    retrieval = _retrieve_and_format(row, [raw_question], k=config.retrieval_k, label_prefix="simple",
                                     where=_where_from_config(config),
                                     collection=_collection_for_config(config))
    passage_block = "\n\n".join(retrieval["passages"])

    user = f"## Retrieved Passages\n{passage_block}\n\n## Question\n{question}"
    answer = _llm_call(_system_prompt(config, "rag"), user, label="rag_simple/answer")

    return {
        "final_answer": answer,
        "evidence_store": retrieval["evidence_store"],
        "retrieved_ids": retrieval["retrieved_ids"],
        "gold_retrieved": retrieval["gold_retrieved"],
    }


def run_rag_arbitration(row: pd.Series, config: EvalConfig) -> dict:
    """LLM answers naively, then reviews retrieved passages (conservative framing)."""
    question = _fmt(row, config)
    question_intermediate = _fmt_intermediate(row, config)
    queries = _rewrite_query(question_intermediate, label="rag_arb/rewrite")

    # Step 1: Snap answer (no evidence)
    snap_answer = _llm_call(_system_prompt(config, "answer"), question, label="rag_arb/snap")
    snap_letter = _extract_answer(snap_answer, config)

    # Step 2: Retrieve
    retrieval = _retrieve_and_format(row, queries, k=config.retrieval_k, label_prefix="rag_arb",
                                     where=_where_from_config(config),
                                     collection=_collection_for_config(config))
    passage_block = "\n\n".join(retrieval["passages"])

    # Step 3: Arbitrate with conservative framing
    arb_system = (
        "You are a legal expert. You previously answered a question based on your knowledge. "
        "Now you are given retrieved legal passages that may be relevant. "
        "Review the passages carefully. If the evidence supports your original answer, keep it. "
        "If the evidence clearly points to a different answer, change it. "
        "Do not change your answer unless the evidence gives you a strong reason to. "
        "Reason step by step, then give your final answer as: Answer: (X)"
    )
    arb_user = (
        f"## Your Previous Reasoning\n{_strip_answer_line(snap_answer)}\n\n"
        f"## Retrieved Passages\n{passage_block}\n\n"
        f"## Question\n{question}"
    )
    final_answer = _llm_call(arb_system, arb_user, label="rag_arb/arbitrate")
    final_letter = _extract_answer(final_answer, config)

    return {
        "final_answer": final_answer,
        "snap_answer": snap_answer,
        "snap_letter": snap_letter,
        "final_letter": final_letter,
        "changed": snap_letter != final_letter,
        "evidence_store": retrieval["evidence_store"],
        "retrieved_ids": retrieval["retrieved_ids"],
        "gold_retrieved": retrieval["gold_retrieved"],
        "rewrite_queries": queries,
        "max_ce_score": retrieval["max_ce_score"],
    }


def run_decompose(row: pd.Series, config: EvalConfig) -> dict:
    """Decompose-then-answer: break question into sub-questions, answer each, synthesize."""
    question = _fmt(row, config)
    question_intermediate = _fmt_intermediate(row, config)

    # Determine which decomposition variant to use (controlled by tag)
    variant = "structured" if "structured" in (config.tag or "") else "natural"

    if variant == "structured":
        # Variant A: IRAC-structured decomposition
        decompose_system = (
            "You are a legal analyst. Given a legal question, identify the 2-3 key sub-issues "
            "that must be resolved to answer it. Structure them as:\n"
            "1. RULE: What is the governing legal rule or doctrine?\n"
            "2. APPLICATION: How do the specific facts interact with the rule?\n"
            "3. EXCEPTION: Are there any exceptions, defenses, or limitations that apply?\n\n"
            "Output ONLY a JSON list of sub-questions, e.g.:\n"
            '[\"What is the rule for...\", \"How do the facts...\", \"Are there exceptions...\"]'
        )
    else:
        # Variant B: Natural decomposition — let model decide what matters
        decompose_system = (
            "You are a legal analyst. Given a legal question, identify the 2-3 key issues "
            "you need to resolve to answer it correctly. Think about what makes this question "
            "hard and what you'd need to figure out.\n\n"
            "Output ONLY a JSON list of sub-questions, e.g.:\n"
            '[\"Does X apply here?\", \"What is the standard for...\", \"Is there an exception when...\"]'
        )

    # Step 1: Decompose
    raw_decomp = _llm_call(decompose_system, question_intermediate, label="decompose/split")
    sub_questions = _parse_json(raw_decomp)
    if not isinstance(sub_questions, list) or not sub_questions:
        # Fallback: if decomposition fails, just answer directly
        sub_questions = [question_intermediate]

    # Cap at 3 sub-questions
    sub_questions = sub_questions[:3]

    # Step 2: Answer each sub-question independently
    sub_answers = []
    answer_system = _system_prompt(config, "answer")
    for i, sq in enumerate(sub_questions):
        # Give the sub-question in context of the original
        sub_prompt = f"In the context of this question:\n{question_intermediate}\n\nAddress this specific issue:\n{sq}"
        sub_ans = _llm_call(answer_system, sub_prompt, label=f"decompose/sub_{i}")
        sub_answers.append({"question": sq, "answer": sub_ans})

    # Step 3: Synthesize sub-answers into final answer
    # Strip trailing "Answer: (X)" from each sub-answer so the synthesizer reasons
    # from the sub-analysis rather than piggybacking the sub-agent's letter vote.
    synth_parts = []
    for sa in sub_answers:
        clean_analysis = _strip_answer_line(sa["answer"])
        synth_parts.append(f"Issue: {sa['question']}\nAnalysis: {clean_analysis}")
    synth_block = "\n\n".join(synth_parts)

    synth_system = _system_prompt(config, "answer")
    synth_user = (
        f"You previously analyzed a legal question by breaking it into sub-issues. "
        f"Now synthesize your analysis into a final answer.\n\n"
        f"## Sub-Issue Analyses\n{synth_block}\n\n"
        f"## Original Question\n{question}"
    )
    final_answer = _llm_call(synth_system, synth_user, label="decompose/synthesize")

    return {
        "final_answer": final_answer,
        "variant": variant,
        "sub_questions": sub_questions,
        "sub_answers": [sa["answer"] for sa in sub_answers],
        "num_sub_questions": len(sub_questions),
    }


def run_decompose_rag(row: pd.Series, config: EvalConfig) -> dict:
    """Decompose + Snap-HyDE RAG: break into sub-questions, RAG each, synthesize with evidence."""
    question = _fmt(row, config)
    question_intermediate = _fmt_intermediate(row, config)

    # Determine decomposition variant (controlled by tag)
    variant = "structured" if "structured" in (config.tag or "") else "natural"

    if variant == "structured":
        decompose_system = (
            "You are a legal analyst. Given a legal question, identify the 2-3 key sub-issues "
            "that must be resolved to answer it. Structure them as:\n"
            "1. RULE: What is the governing legal rule or doctrine?\n"
            "2. APPLICATION: How do the specific facts interact with the rule?\n"
            "3. EXCEPTION: Are there any exceptions, defenses, or limitations that apply?\n\n"
            "Output ONLY a JSON list of sub-questions, e.g.:\n"
            '[\"What is the rule for...\", \"How do the facts...\", \"Are there exceptions...\"]'
        )
    else:
        decompose_system = (
            "You are a legal analyst. Given a legal question, identify the 2-3 key issues "
            "you need to resolve to answer it correctly. Think about what makes this question "
            "hard and what you'd need to figure out.\n\n"
            "Output ONLY a JSON list of sub-questions, e.g.:\n"
            '[\"Does X apply here?\", \"What is the standard for...\", \"Is there an exception when...\"]'
        )

    # Step 1: Decompose
    raw_decomp = _llm_call(decompose_system, question_intermediate, label="decomp_rag/split")
    sub_questions = _parse_json(raw_decomp)
    if not isinstance(sub_questions, list) or not sub_questions:
        sub_questions = [question_intermediate]
    sub_questions = sub_questions[:3]

    # Step 2: For each sub-question — snap answer → HyDE → retrieve
    sub_results = []
    all_evidence = []
    all_retrieved_ids = []
    any_gold = False

    for i, sq in enumerate(sub_questions):
        sub_prompt = f"In the context of this question:\n{question_intermediate}\n\nAddress this specific issue:\n{sq}"

        # Snap answer this sub-question
        sub_snap = _llm_call(_system_prompt(config, "answer"), sub_prompt, label=f"decomp_rag/snap_{i}")

        # Generate HyDE passage from the sub-answer
        hyde = _generate_hyde(
            config,
            "snap_hyde",
            _snap_hyde_user(sq, sub_snap),
            label=f"decomp_rag/hyde_{i}",
            fallback=sq,
        )

        # Retrieve evidence for this sub-question
        retrieval = _retrieve_and_format(row, [hyde["text"]], k=3, label_prefix=f"decomp_rag_{i}",
                                         where=_where_from_config(config),
                                         collection=_collection_for_config(config))

        sub_results.append({
            "sub_question": sq,
            "snap_answer": sub_snap,
            "hyde_passage": hyde["text"],
            "hyde_passage_raw": hyde["raw"],
            "hyde_contains_answer_artifact": hyde["contains_answer"],
            "passages": retrieval["passages"],
        })
        all_evidence.extend(retrieval["evidence_store"])
        all_retrieved_ids.extend(retrieval["retrieved_ids"])
        if retrieval["gold_retrieved"]:
            any_gold = True

    # Step 3: Synthesize all sub-answers + evidence into final answer
    # Strip trailing "Answer: (X)" from each sub-snap so the final agent reasons
    # from the sub-analysis rather than copying sub-letter votes.
    synth_parts = []
    for sr in sub_results:
        evidence_block = "\n".join(sr["passages"]) if sr["passages"] else "(no evidence retrieved)"
        clean_snap = _strip_answer_line(sr["snap_answer"])
        synth_parts.append(
            f"Issue: {sr['sub_question']}\n"
            f"Analysis: {clean_snap}\n"
            f"Supporting Evidence:\n{evidence_block}"
        )
    synth_block = "\n\n---\n\n".join(synth_parts)

    synth_system = _system_prompt(config, "rag")
    synth_user = (
        f"You previously analyzed a legal question by breaking it into sub-issues. "
        f"Each sub-issue has been analyzed and supporting evidence has been retrieved. "
        f"Now synthesize everything into a final answer.\n\n"
        f"## Sub-Issue Analyses with Evidence\n{synth_block}\n\n"
        f"## Original Question\n{question}"
    )
    final_answer = _llm_call(synth_system, synth_user, label="decomp_rag/synthesize")

    return {
        "final_answer": final_answer,
        "variant": variant,
        "sub_questions": sub_questions,
        "sub_answers": [sr["snap_answer"] for sr in sub_results],
        "hyde_passages": [sr["hyde_passage"] for sr in sub_results],
        "hyde_passages_raw": [sr["hyde_passage_raw"] for sr in sub_results],
        "hyde_contains_answer_artifact": any(sr["hyde_contains_answer_artifact"] for sr in sub_results),
        "num_sub_questions": len(sub_questions),
        "evidence_store": all_evidence,
        "retrieved_ids": all_retrieved_ids,
        "gold_retrieved": any_gold,
    }


def run_conf_ce_threshold(row: pd.Series, config: EvalConfig) -> dict:
    """Combined: confidence gating (3 snap votes) + CE threshold on the RAG path."""
    CE_THRESHOLD = 4.0
    question = _fmt(row, config)
    question_intermediate = _fmt_intermediate(row, config)

    # Step 1: Take 3 snap answers
    snaps = []
    snap_letters = []
    for k in range(3):
        answer = _llm_call(_system_prompt(config, "answer"), question, label=f"conf_ce/snap_{k}")
        letter = _extract_answer(answer, config)
        snaps.append(answer)
        snap_letters.append(letter)

    # Step 2: Check consensus
    from collections import Counter
    vote_counts = Counter(snap_letters)
    majority_answer, majority_count = vote_counts.most_common(1)[0]
    unanimous = majority_count == 3
    majority_idx = snap_letters.index(majority_answer)

    if unanimous:
        return {
            "final_answer": snaps[majority_idx],
            "snap_answers": snaps,
            "snap_letters": snap_letters,
            "routed_to": "skip_rag",
            "consensus": "unanimous",
            "majority_answer": majority_answer,
        }

    # Step 3: Low confidence — Snap-HyDE with CE threshold
    hyde = _generate_hyde(
        config,
        "snap_hyde",
        _snap_hyde_user(question_intermediate, snaps[majority_idx]),
        label="conf_ce/hyde",
        fallback=question_intermediate,
    )

    retrieval = _retrieve_and_format(row, [hyde["text"]], k=config.retrieval_k, label_prefix="conf_ce",
                                     where=_where_from_config(config),
                                     collection=_collection_for_config(config))

    # Step 4: CE threshold — if evidence is low quality, use majority snap answer
    max_ce = retrieval["max_ce_score"]
    if max_ce < CE_THRESHOLD:
        return {
            "final_answer": snaps[majority_idx],
            "snap_answers": snaps,
            "snap_letters": snap_letters,
            "routed_to": "snap_ce_fallback",
            "consensus": f"{majority_count}/3",
            "majority_answer": majority_answer,
            "hyde_passage": hyde["text"],
            "hyde_passage_raw": hyde["raw"],
            "hyde_contains_answer_artifact": hyde["contains_answer"],
            "max_ce_score": max_ce,
            "ce_threshold": CE_THRESHOLD,
            "evidence_store": retrieval["evidence_store"],
            "retrieved_ids": retrieval["retrieved_ids"],
            "gold_retrieved": retrieval["gold_retrieved"],
        }

    # Step 5: Good evidence — answer with RAG
    passage_block = "\n\n".join(retrieval["passages"])
    user = f"## Retrieved Passages\n{passage_block}\n\n## Question\n{question}"
    answer = _llm_call(_system_prompt(config, "rag"), user, label="conf_ce/answer")

    return {
        "final_answer": answer,
        "snap_answers": snaps,
        "snap_letters": snap_letters,
        "routed_to": "rag",
        "consensus": f"{majority_count}/3",
        "majority_answer": majority_answer,
        "hyde_passage": hyde["text"],
        "hyde_passage_raw": hyde["raw"],
        "hyde_contains_answer_artifact": hyde["contains_answer"],
        "max_ce_score": max_ce,
        "ce_threshold": CE_THRESHOLD,
        "evidence_store": retrieval["evidence_store"],
        "retrieved_ids": retrieval["retrieved_ids"],
        "gold_retrieved": retrieval["gold_retrieved"],
    }


def run_confidence_gated(row: pd.Series, config: EvalConfig) -> dict:
    """Confidence-gated RAG: 3 snap answers vote; unanimous = skip RAG, disagreement = Snap-HyDE."""
    question = _fmt(row, config)
    question_intermediate = _fmt_intermediate(row, config)

    # Step 1: Take 3 snap answers
    snaps = []
    snap_letters = []
    for k in range(3):
        answer = _llm_call(_system_prompt(config, "answer"), question, label=f"conf_gate/snap_{k}")
        letter = _extract_answer(answer, config)
        snaps.append(answer)
        snap_letters.append(letter)

    # Step 2: Check consensus
    from collections import Counter
    vote_counts = Counter(snap_letters)
    majority_answer, majority_count = vote_counts.most_common(1)[0]
    unanimous = majority_count == 3
    majority_idx = snap_letters.index(majority_answer)  # use this snap's reasoning

    if unanimous:
        # High confidence — skip RAG, return majority snap answer
        return {
            "final_answer": snaps[majority_idx],
            "snap_answers": snaps,
            "snap_letters": snap_letters,
            "routed_to": "skip_rag",
            "consensus": "unanimous",
            "majority_answer": majority_answer,
        }

    # Step 3: Low confidence — apply Snap-HyDE using majority snap's reasoning
    hyde = _generate_hyde(
        config,
        "snap_hyde",
        _snap_hyde_user(question_intermediate, snaps[majority_idx]),
        label="conf_gate/hyde",
        fallback=question_intermediate,
    )

    retrieval = _retrieve_and_format(row, [hyde["text"]], k=config.retrieval_k, label_prefix="conf_gate",
                                     where=_where_from_config(config),
                                     collection=_collection_for_config(config))
    passage_block = "\n\n".join(retrieval["passages"])

    user = f"## Retrieved Passages\n{passage_block}\n\n## Question\n{question}"
    answer = _llm_call(_system_prompt(config, "rag"), user, label="conf_gate/answer")

    return {
        "final_answer": answer,
        "snap_answers": snaps,
        "snap_letters": snap_letters,
        "routed_to": "snap_hyde",
        "consensus": f"{majority_count}/3",
        "majority_answer": majority_answer,
        "hyde_passage": hyde["text"],
        "hyde_passage_raw": hyde["raw"],
        "hyde_contains_answer_artifact": hyde["contains_answer"],
        "evidence_store": retrieval["evidence_store"],
        "retrieved_ids": retrieval["retrieved_ids"],
        "gold_retrieved": retrieval["gold_retrieved"],
    }


# ---------------------------------------------------------------------------
# Atomic Blocks — testing individual reasoning strategies
# ---------------------------------------------------------------------------

def run_self_verify(row: pd.Series, config: EvalConfig) -> dict:
    """Self-verification: snap answer, then ask the model to review and correct it.
    Tests whether a second pass catches errors. 2 LLM calls, no retrieval."""
    question = _fmt(row, config)

    # Step 1: Snap answer
    snap_answer = _llm_call(_system_prompt(config, "answer"), question, label="verify/snap")
    snap_letter = _extract_answer(snap_answer, config)

    # Step 2: Self-review (strip prior letter so reviewer re-derives the answer)
    review_prompt = (
        f"You previously answered a legal question. Review your reasoning carefully for errors "
        f"in legal reasoning, missed elements, or incorrect conclusions. If you find an error, "
        f"provide the corrected answer. If your reasoning is sound, restate the conclusion.\n\n"
        f"## Your Previous Reasoning\n{_strip_answer_line(snap_answer)}\n\n"
        f"## Original Question\n{question}"
    )
    verified = _llm_call(
        "You are a careful legal reviewer. Check the answer for errors and correct if needed.",
        review_prompt, label="verify/review"
    )

    return {
        "final_answer": verified,
        "snap_answer": snap_answer,
        "snap_letter": snap_letter,
    }


def run_friend_foe_attribution(row: pd.Series, config: EvalConfig) -> dict:
    """Attribution-bias probe: snap answer, then review the same snap three ways.

    The self-attribution review is used as final_answer for scoring because it is
    the natural baseline; the foe and control reviews are logged to compare how
    attribution framing changes scrutiny of the identical snap content.
    4 LLM calls, no retrieval.
    """
    question = _fmt(row, config)

    # Step 1: Snap answer
    snap_text = _llm_call(_system_prompt(config, "answer"), question, label="friend_foe/snap")
    if not (snap_text or "").strip():
        raise RuntimeError("friend_foe_attribution snap pass returned empty response")

    try:
        idx_position = int(row.name)
    except (TypeError, ValueError):
        idx_position = sum((i + 1) * ord(ch) for i, ch in enumerate(question))

    foe_names = ("Grok-2", "GPT-4o", "an unknown source")
    foe_label = foe_names[idx_position % len(foe_names)]
    review_system = _system_prompt(config, "rag")

    self_user = (
        "Here is your earlier response to the question. Review it and produce a final answer.\n\n"
        f"## Earlier response\n{snap_text}\n\n"
        f"## Question\n{question}"
    )
    foe_user = (
        f"Here is a response from {foe_label} to the question. Review it and produce a final answer.\n\n"
        f"## Response from {foe_label}\n{snap_text}\n\n"
        f"## Question\n{question}"
    )
    control_user = (
        "Here is a candidate response to the question. Review it and produce a final answer.\n\n"
        f"## Candidate response\n{snap_text}\n\n"
        f"## Question\n{question}"
    )

    self_review_answer = _llm_call(review_system, self_user, label="friend_foe/self_review")
    if not (self_review_answer or "").strip():
        raise RuntimeError("friend_foe_attribution self review returned empty response")
    foe_review_answer = _llm_call(review_system, foe_user, label="friend_foe/foe_review")
    if not (foe_review_answer or "").strip():
        raise RuntimeError("friend_foe_attribution foe review returned empty response")
    control_review_answer = _llm_call(review_system, control_user, label="friend_foe/control_review")
    if not (control_review_answer or "").strip():
        raise RuntimeError("friend_foe_attribution control review returned empty response")

    snap_extracted = _extract_answer(snap_text, config)

    def kept_snap(review_text: str) -> bool:
        if snap_extracted in (None, ""):
            return False
        return bool(_extract_answer(review_text, config) == snap_extracted)

    return {
        "final_answer": self_review_answer,
        "formatted_question": question,
        "snap_answer": snap_text,
        "self_review_answer": self_review_answer,
        "foe_review_answer": foe_review_answer,
        "control_review_answer": control_review_answer,
        "foe_label": foe_label,
        "self_kept_snap": kept_snap(self_review_answer),
        "foe_kept_snap": kept_snap(foe_review_answer),
        "control_kept_snap": kept_snap(control_review_answer),
        "evidence_store": [],
        "retrieved_ids": [],
        "gold_retrieved": False,
    }


def run_double_snap(row: pd.Series, config: EvalConfig) -> dict:
    """Double-snap: two independent answers. If same → use it. If different → CE-threshold RAG.
    Tests the cheapest confidence signal (2 calls when confident). 2-4 LLM calls."""
    CE_THRESHOLD = 4.0
    question = _fmt(row, config)
    question_intermediate = _fmt_intermediate(row, config)

    # Step 1: Two independent snap answers
    snap1 = _llm_call(_system_prompt(config, "answer"), question, label="dsnap/snap1")
    letter1 = _extract_answer(snap1, config)
    snap2 = _llm_call(_system_prompt(config, "answer"), question, label="dsnap/snap2")
    letter2 = _extract_answer(snap2, config)

    if letter1 == letter2:
        # Agreement — high confidence, skip RAG
        return {
            "final_answer": snap1,
            "snap1": snap1, "snap2": snap2,
            "letter1": letter1, "letter2": letter2,
            "routed_to": "snap_agree",
        }

    # Step 2: Disagreement — CE-threshold RAG using snap1's reasoning
    hyde = _generate_hyde(
        config,
        "snap_hyde",
        _snap_hyde_user(question_intermediate, snap1),
        label="dsnap/hyde",
        fallback=question_intermediate,
    )

    retrieval = _retrieve_and_format(row, [hyde["text"]], k=config.retrieval_k, label_prefix="dsnap",
                                     where=_where_from_config(config),
                                     collection=_collection_for_config(config))

    max_ce = retrieval["max_ce_score"]
    if max_ce < CE_THRESHOLD:
        return {
            "final_answer": snap1,
            "snap1": snap1, "snap2": snap2,
            "letter1": letter1, "letter2": letter2,
            "routed_to": "snap_ce_fallback",
            "hyde_passage": hyde["text"],
            "hyde_passage_raw": hyde["raw"],
            "hyde_contains_answer_artifact": hyde["contains_answer"],
            "max_ce_score": max_ce,
            "evidence_store": retrieval["evidence_store"],
            "retrieved_ids": retrieval["retrieved_ids"],
            "gold_retrieved": retrieval["gold_retrieved"],
        }

    passage_block = "\n\n".join(retrieval["passages"])
    user = f"## Retrieved Passages\n{passage_block}\n\n## Question\n{question}"
    answer = _llm_call(_system_prompt(config, "rag"), user, label="dsnap/answer")

    return {
        "final_answer": answer,
        "snap1": snap1, "snap2": snap2,
        "letter1": letter1, "letter2": letter2,
        "routed_to": "rag",
        "hyde_passage": hyde["text"],
        "hyde_passage_raw": hyde["raw"],
        "hyde_contains_answer_artifact": hyde["contains_answer"],
        "max_ce_score": max_ce,
        "evidence_store": retrieval["evidence_store"],
        "retrieved_ids": retrieval["retrieved_ids"],
        "gold_retrieved": retrieval["gold_retrieved"],
    }


def run_snap_debate(row: pd.Series, config: EvalConfig) -> dict:
    """Snap-debate: snap answer, then a second call sees the first and critiques it.
    Tests whether adversarial self-review improves over simple self-verification. 2 LLM calls."""
    question = _fmt(row, config)

    # Step 1: Snap answer
    snap_answer = _llm_call(_system_prompt(config, "answer"), question, label="debate/snap")
    snap_letter = _extract_answer(snap_answer, config)

    # Step 2: Adversarial review (strip prior letter so critic re-derives, not echoes)
    debate_prompt = (
        f"A student answered a legal question. Your job is to find flaws in their reasoning. "
        f"Look for: incorrect legal rules, missing elements, wrong conclusions, or misapplied "
        f"standards. If you find errors, provide the correct answer with your reasoning. "
        f"If the reasoning is genuinely sound, confirm it and explain why.\n\n"
        f"## Student's Reasoning\n{_strip_answer_line(snap_answer)}\n\n"
        f"## Original Question\n{question}"
    )
    debated = _llm_call(
        "You are a law professor grading an exam. Be critical and precise.",
        debate_prompt, label="debate/critique"
    )

    return {
        "final_answer": debated,
        "snap_answer": snap_answer,
        "snap_letter": snap_letter,
    }


def run_snap_only_in_final(row: pd.Series, config: EvalConfig) -> dict:
    """Ablation cell: snap reasoning visible to final agent, NO retrieval.

    Isolates the snap-reasoning contribution from the retrieval contribution.
    Compared against:
      - llm_only          (snap only, no final pass)
      - rag_simple        (no snap, retrieval only)
      - rag_snap_hyde     (snap + retrieval, snap hidden from final)

    2 LLM calls, no retrieval.
    """
    question = _fmt(row, config)

    # Step 1: Snap
    snap_answer = _llm_call(_system_prompt(config, "answer"), question, label="snap_only/snap")
    snap_letter = _extract_answer(snap_answer, config)

    # Step 2: Final — snap reasoning visible (letter stripped to avoid pure letter-copy)
    user = (
        f"## Your Initial Reasoning\n{_strip_answer_line(snap_answer)}\n\n"
        f"## Question\n{question}"
    )
    answer = _llm_call(_system_prompt(config, "rag"), user, label="snap_only/answer")

    return {
        "final_answer": answer,
        "snap_answer": snap_answer,
        "snap_letter": snap_letter,
    }


def run_planning_table(row: pd.Series, config: EvalConfig) -> dict:
    """Planning-table-with-scratchpad: explicit planning + per-TODO retrieval + table state.

    Distinct from subagent_rag: the final agent sees the COMPLETE populated
    planning table (todo + finding pairs) as scratchpad context, not just
    aggregated reports. Designed for multi-hop where explicit per-hop
    state-tracking should help vs single-shot snap+HyDE.

    Steps (≈5-7 LLM calls):
      1. Snap → initial answer + reasoning
      2. Plan-gen: read snap, emit 2-3 fact-focused TODO sub-questions
      3. For each TODO: retrieve k=3 passages → write a 2-3 sentence finding
      4. Final answer with the full populated table as scratchpad + question
    """
    question = _fmt(row, config)
    question_intermediate = _fmt_intermediate(row, config)

    # Step 1: snap
    snap_answer = _llm_call(_system_prompt(config, "answer"), question, label="ptable/snap")
    snap_letter = _extract_answer(snap_answer, config)
    snap_clean = _strip_answer_line(snap_answer)

    # Step 2: plan generation
    plan_system = (
        "You are a research planner. Read a student's initial reasoning on a question "
        "and identify 2-3 specific sub-questions that, if answered, would let you verify "
        "or correct the conclusion.\n\n"
        "Output exactly 2-3 sub-questions, one per line, in this format:\n"
        "TODO: <focused sub-question>\n"
        "TODO: <focused sub-question>\n"
        "TODO: <focused sub-question>\n\n"
        "STRICT OUTPUT RULES:\n"
        "- Sub-questions should be FACT-FOCUSED (what is X? when did Y happen? who is Z?)\n"
        "- Avoid restating the original question; aim at decomposable sub-claims\n"
        "- Do NOT pick an answer letter; output only TODO lines\n"
        "- Each TODO on its own line, no markdown bullets"
    )
    plan_user = (
        f"## Original Question\n{question_intermediate}\n\n"
        f"## Initial Reasoning\n{snap_clean}"
    )
    plan_raw = _llm_call(plan_system, plan_user, label="ptable/plan")

    # Parse TODO lines
    todos: list[str] = []
    for line in plan_raw.splitlines():
        line = line.strip()
        if not line:
            continue
        # Accept TODO:, TODO -, **TODO:**, etc.
        m = re.match(r"^\s*(?:\*\*)?\s*TODO\s*[:\-]\s*(?:\*\*)?\s*(.+?)\s*(?:\*\*)?\s*$", line, re.IGNORECASE)
        if m:
            todo = m.group(1).strip().lstrip("-•").strip()
            if todo:
                todos.append(todo)
    # Deduplicate TODOs (case + whitespace insensitive). Near-synonymous TODOs
    # waste a retrieval+finding LLM call for zero new evidence (audit found
    # this happened on 1/5 smoke samples).
    seen_normalized: set[str] = set()
    deduped: list[str] = []
    for t in todos:
        norm = re.sub(r"\s+", " ", t.lower().strip().rstrip("?.!"))
        if norm and norm not in seen_normalized:
            seen_normalized.add(norm)
            deduped.append(t)
    todos = deduped[:3]
    if not todos:
        # fallback: use the original question as a single TODO
        todos = [question_intermediate]

    # Step 3: per-TODO retrieve + finding
    table_entries: list[dict] = []
    all_retrieved_ids: list[str] = []
    finding_system = (
        "You are a research assistant. Answer the sub-question concisely "
        "(2-3 sentences max) using ONLY the retrieved passages. If the passages don't "
        "contain the answer, say so explicitly. Do NOT pick a multiple-choice letter."
    )
    for i, todo in enumerate(todos):
        retrieval = _retrieve_and_format(
            row, [todo], k=3,
            label_prefix=f"ptable/todo_{i}",
            where=_where_from_config(config),
            collection=_collection_for_config(config),
        )
        passages_block = "\n\n".join(retrieval["passages"]) if retrieval["passages"] else "(no passages retrieved)"
        finding_user = (
            f"## Retrieved Passages\n{passages_block}\n\n"
            f"## Sub-Question\n{todo}"
        )
        finding = _generate_report(
            finding_system,
            finding_user,
            label=f"ptable/finding_{i}",
            fallback="No relevant information found in the retrieved passages.",
        )
        table_entries.append({
            "todo": todo,
            "finding": finding["text"],
            "evidence_ids": retrieval.get("retrieved_ids", []),
        })
        all_retrieved_ids.extend(retrieval.get("retrieved_ids", []))

    # Step 4: final answer with populated table
    table_text = "\n\n".join(
        f"### TODO {i+1}: {e['todo']}\n**Finding:** {e['finding']}"
        for i, e in enumerate(table_entries)
    )
    # Audit 2026-04-26 found 17/30 records where findings said "passages do
    # not contain X" but the final agent ignored them and asserted a parametric
    # guess anyway. Tighten the synthesizer instruction explicitly.
    final_user = (
        f"## Planning Table (your sub-investigations)\n{table_text}\n\n"
        f"## Question\n{question}\n\n"
        "## Synthesis instructions\n"
        "1. Use the planning table findings as your PRIMARY evidence. If a "
        "finding directly contradicts a possible answer, weight it heavily.\n"
        "2. If multiple findings need to compose the answer, walk through "
        "the chain explicitly before concluding.\n"
        "3. ALWAYS commit to a final answer using the format `Answer: ...`. "
        "Even if findings are incomplete, give your best single-span guess — "
        "do NOT abstain or say 'information not provided'."
    )
    final_answer = _llm_call(_system_prompt(config, "rag"), final_user, label="ptable/final")

    # Gold-retrieved tracking — accept if ANY gold idx appears across the per-TODO retrievals
    gold_idx = str(row.get("gold_idx", ""))
    gold_idxs = {s.strip() for s in gold_idx.split(",") if s.strip()}
    gold_retrieved = bool(gold_idxs & set(all_retrieved_ids)) if gold_idxs else False

    return {
        "final_answer": final_answer,
        "snap_answer": snap_answer,
        "snap_letter": snap_letter,
        "planning_table": table_entries,
        "todos_count": len(todos),
        "retrieved_ids": list(dict.fromkeys(all_retrieved_ids)),
        "gold_retrieved": gold_retrieved,
        "evidence_store": [
            {"idx": eid, "text": "", "source": "ptable", "cross_encoder_score": 0.0}
            for eid in dict.fromkeys(all_retrieved_ids)
        ],
    }


def run_advisor_planning_table(row: pd.Series, config: EvalConfig) -> dict:
    """Two-LLM advisor pattern: cheap LLM does plan + per-TODO findings,
    strong LLM does the final synthesis.

    Inspired by Anthropic deep-research / Claude Code advisor patterns.
    Cost story: most LLM calls are the plan-gen + per-TODO findings (4-6
    cheap calls); ONE expensive synthesis call sees the populated table.
    Tests whether allocating reasoning capacity to synthesis (vs spreading
    across cheap intermediate steps) helps multi-hop.

    Provider config:
      - "advisor" / cheap LLM: Llama 3.3 8b instant (Groq, 14.4K RPD/500K TPD)
      - "synthesizer" / strong LLM: whatever the eval is run with (config.provider)

    Note: cheap LLM is HARDCODED to groq-llama8b for this initial impl.
    Future: add EVAL_ADVISOR_PROVIDER env var.

    Steps (~5-7 cheap calls + 1 strong call):
      1. (cheap) gen 2-3 fact-focused TODOs from question alone (no snap)
      2. for each TODO: (cheap) retrieve + finding
      3. (strong) final answer with full populated table + v2 synthesis prompt
    """
    import os
    from llm_config import _resolve_provider, PROVIDERS
    from openai import OpenAI

    question = _fmt(row, config)
    question_intermediate = _fmt_intermediate(row, config)

    # Build a separate cheap-LLM client (advisor)
    advisor_provider = os.getenv("EVAL_ADVISOR_PROVIDER", "groq-llama8b")
    if advisor_provider not in PROVIDERS:
        raise RuntimeError(f"advisor provider '{advisor_provider}' not in llm_config.PROVIDERS")
    advisor_base, advisor_key_env, advisor_model, _, _ = PROVIDERS[advisor_provider]
    advisor_key = os.getenv(advisor_key_env, "")
    if not advisor_key:
        raise RuntimeError(f"advisor provider '{advisor_provider}' needs env var {advisor_key_env}")
    advisor_client = OpenAI(base_url=advisor_base, api_key=advisor_key, timeout=60, max_retries=1)

    def cheap_call(system: str, user: str, label: str = "advisor") -> str:
        """Direct call bypassing the global llm provider — uses cheap_advisor model."""
        try:
            r = advisor_client.chat.completions.create(
                model=advisor_model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.0,
                max_tokens=512,
            )
            return r.choices[0].message.content or ""
        except Exception as exc:
            return ""  # fallback to empty; caller handles

    # Step 1 (cheap): plan-gen
    plan_system = (
        "You are a research planner. Read a multi-hop question and emit 2-3 "
        "fact-focused sub-questions to investigate.\n\n"
        "Format:\n"
        "TODO: <focused sub-question>\n"
        "TODO: <focused sub-question>\n"
        "TODO: <focused sub-question>\n\n"
        "Rules:\n"
        "- Sub-questions decompose multi-hop into single-hop facts\n"
        "- Do NOT pick an answer; only emit TODO lines"
    )
    plan_raw = cheap_call(plan_system, f"## Question\n{question_intermediate}", label="advisor/plan")

    todos: list[str] = []
    for line in plan_raw.splitlines():
        m = re.match(r"^\s*(?:\*\*)?\s*TODO\s*[:\-]\s*(?:\*\*)?\s*(.+?)\s*(?:\*\*)?\s*$", line.strip(), re.IGNORECASE)
        if m:
            todos.append(m.group(1).strip().lstrip("-•").strip())
    seen: set[str] = set()
    deduped = []
    for t in todos:
        norm = re.sub(r"\s+", " ", t.lower().strip().rstrip("?.!"))
        if norm and norm not in seen:
            seen.add(norm)
            deduped.append(t)
    todos = deduped[:3]
    if not todos:
        todos = [question_intermediate]

    # Step 2 (cheap per-TODO): retrieve + finding using advisor LLM
    table_entries: list[dict] = []
    all_retrieved_ids: list[str] = []
    finding_system = (
        "You are a research assistant. Answer the sub-question concisely "
        "(2-3 sentences) using ONLY the retrieved passages. If the passages "
        "do not contain the answer, say so. Do NOT pick a multiple-choice letter."
    )
    for i, todo in enumerate(todos):
        retrieval = _retrieve_and_format(
            row, [todo], k=3,
            label_prefix=f"advisor/todo_{i}",
            where=_where_from_config(config),
            collection=_collection_for_config(config),
        )
        passages_block = "\n\n".join(retrieval["passages"]) if retrieval["passages"] else "(no passages retrieved)"
        finding_text = cheap_call(
            finding_system,
            f"## Retrieved Passages\n{passages_block}\n\n## Sub-Question\n{todo}",
            label=f"advisor/finding_{i}",
        )
        if not finding_text:
            finding_text = "No relevant information found in the retrieved passages."
        table_entries.append({
            "todo": todo,
            "finding": finding_text,
            "evidence_ids": retrieval.get("retrieved_ids", []),
        })
        all_retrieved_ids.extend(retrieval.get("retrieved_ids", []))

    # Step 3 (STRONG / config.provider): final synthesis
    table_text = "\n\n".join(
        f"### TODO {i+1}: {e['todo']}\n**Finding:** {e['finding']}"
        for i, e in enumerate(table_entries)
    )
    final_user = (
        f"## Planning Table (sub-investigations from research advisor)\n{table_text}\n\n"
        f"## Question\n{question}\n\n"
        "## Synthesis instructions\n"
        "1. Use the planning table findings as your PRIMARY evidence; weight a "
        "finding heavily if it directly addresses an option.\n"
        "2. Walk through the multi-hop chain explicitly, naming each intermediate "
        "entity from the findings, before concluding.\n"
        "3. ALWAYS commit to a final answer using the format `Answer: ...`. "
        "Even if the chain is incomplete, give your best single-span guess; "
        "do NOT abstain or say 'information not provided'."
    )
    final_answer = _llm_call(_system_prompt(config, "rag"), final_user, label="advisor/synth")

    gold_idx = str(row.get("gold_idx", ""))
    gold_idxs = {s.strip() for s in gold_idx.split(",") if s.strip()}
    gold_retrieved = bool(gold_idxs & set(all_retrieved_ids)) if gold_idxs else False

    return {
        "final_answer": final_answer,
        "planning_table": table_entries,
        "todos_count": len(todos),
        "advisor_provider": advisor_provider,
        "advisor_model": advisor_model,
        "retrieved_ids": list(dict.fromkeys(all_retrieved_ids)),
        "gold_retrieved": gold_retrieved,
        "evidence_store": [
            {"idx": eid, "text": "", "source": "advisor", "cross_encoder_score": 0.0}
            for eid in dict.fromkeys(all_retrieved_ids)
        ],
    }


def run_iterative_planning_table(row: pd.Series, config: EvalConfig) -> dict:
    """Multi-round (deep-research-style) planning_table.

    Inspired by Anthropic deep-research / advisor patterns. Where `planning_table`
    decomposes once upfront and runs all TODOs in parallel, this version:

      1. Generates ONE focused next-TODO conditioned on findings so far
      2. Retrieves + writes a finding for that TODO
      3. Asks the model: "ready to answer, or need another sub-question?"
      4. Loops up to MAX_ROUNDS (3) — early-exit if model says READY
      5. Final answer with the full populated trace

    The hypothesis: 1-shot ptable produces TODOs that are bad guesses about
    what's needed; iteratively letting the model see results and decide the
    next TODO should yield more useful retrievals on multi-hop.

    Steps: 1 (initial TODO) + per round: retrieve + finding + ready-check +
    next-TODO, capped at 3 rounds → ~7-10 LLM calls per question.
    """
    MAX_ROUNDS = 3
    question = _fmt(row, config)
    question_intermediate = _fmt_intermediate(row, config)

    # Step 1: generate initial focused TODO
    initial_todo_system = (
        "You are a research planner working on a multi-hop question. Given the "
        "question, produce ONE focused sub-question that you would investigate "
        "FIRST to start answering the multi-hop chain.\n\n"
        "Output exactly one line in this format:\n"
        "TODO: <focused first sub-question>\n\n"
        "Rules:\n"
        "- Pick the most upstream / foundational sub-question (what entity to identify first)\n"
        "- Do NOT pick an answer; output only the TODO line\n"
        "- Be specific and fact-focused"
    )
    initial_user = f"## Multi-hop Question\n{question_intermediate}"
    initial_raw = _llm_call(initial_todo_system, initial_user, label="iter_ptable/init_todo")
    initial_todo = ""
    for line in initial_raw.splitlines():
        m = re.match(r"^\s*(?:\*\*)?\s*TODO\s*[:\-]\s*(?:\*\*)?\s*(.+?)\s*(?:\*\*)?\s*$", line.strip(), re.IGNORECASE)
        if m:
            initial_todo = m.group(1).strip().lstrip("-•").strip()
            break
    if not initial_todo:
        initial_todo = question_intermediate  # fallback

    # Round loop
    table_entries: list[dict] = []
    all_retrieved_ids: list[str] = []
    current_todo = initial_todo
    finding_system = (
        "You are a research assistant. Answer the sub-question concisely "
        "(2-3 sentences max) using ONLY the retrieved passages. If the passages "
        "don't contain the answer, say so explicitly. Do NOT pick a multiple-choice letter."
    )
    next_todo_system_template = (
        "You are a research planner working iteratively on a multi-hop question. "
        "You have completed {n_rounds} sub-investigations so far (the planning "
        "table below). Your job: decide whether the planning table is now "
        "sufficient to answer the original question, or whether you need ONE "
        "more sub-investigation.\n\n"
        "Output EXACTLY ONE of these two formats:\n\n"
        "READY: I have enough information to answer the question.\n\n"
        "OR\n\n"
        "TODO: <next focused sub-question conditioned on what you've learned>\n\n"
        "Rules:\n"
        "- Only emit READY if you can confidently answer from the findings\n"
        "- If picking TODO, the next sub-question should USE the prior findings "
        "to narrow what to investigate (e.g., if Round 1 found 'X is in Country Y', "
        "Round 2 might ask about a fact specific to Country Y)\n"
        "- Do NOT restate the original question or commit to an answer"
    )
    rounds_completed = 0
    early_exit = False
    for round_idx in range(MAX_ROUNDS):
        # Retrieve + finding for current TODO
        retrieval = _retrieve_and_format(
            row, [current_todo], k=3,
            label_prefix=f"iter_ptable/r{round_idx}",
            where=_where_from_config(config),
            collection=_collection_for_config(config),
        )
        passages_block = "\n\n".join(retrieval["passages"]) if retrieval["passages"] else "(no passages retrieved)"
        finding_user = (
            f"## Retrieved Passages\n{passages_block}\n\n"
            f"## Sub-Question\n{current_todo}"
        )
        finding = _generate_report(
            finding_system,
            finding_user,
            label=f"iter_ptable/finding_r{round_idx}",
            fallback="No relevant information found in the retrieved passages.",
        )
        table_entries.append({
            "todo": current_todo,
            "finding": finding["text"],
            "evidence_ids": retrieval.get("retrieved_ids", []),
            "round": round_idx + 1,
        })
        all_retrieved_ids.extend(retrieval.get("retrieved_ids", []))
        rounds_completed += 1

        # Don't ask for more TODOs after the last allowed round — go to final answer
        if round_idx + 1 >= MAX_ROUNDS:
            break

        # Decide: READY or next TODO?
        table_text = "\n\n".join(
            f"### Round {e['round']} — TODO: {e['todo']}\n**Finding:** {e['finding']}"
            for e in table_entries
        )
        decide_user = (
            f"## Original Question\n{question_intermediate}\n\n"
            f"## Planning Table So Far\n{table_text}"
        )
        decide_system = next_todo_system_template.format(n_rounds=len(table_entries))
        decide_raw = _llm_call(decide_system, decide_user, label=f"iter_ptable/decide_r{round_idx}")

        # Parse: READY or TODO
        decide_clean = decide_raw.strip()
        if re.search(r"\bREADY\b", decide_clean[:200], re.IGNORECASE) and not re.search(r"\bTODO\s*:", decide_clean[:200], re.IGNORECASE):
            early_exit = True
            break
        next_todo = ""
        for line in decide_clean.splitlines():
            m = re.match(r"^\s*(?:\*\*)?\s*TODO\s*[:\-]\s*(?:\*\*)?\s*(.+?)\s*(?:\*\*)?\s*$", line.strip(), re.IGNORECASE)
            if m:
                next_todo = m.group(1).strip().lstrip("-•").strip()
                break
        if not next_todo:
            # No clean TODO emitted, treat as ready
            early_exit = True
            break
        current_todo = next_todo

    # Final answer with full table + v2-style synthesizer instructions
    table_text = "\n\n".join(
        f"### Round {e['round']} — TODO: {e['todo']}\n**Finding:** {e['finding']}"
        for e in table_entries
    )
    final_user = (
        f"## Iterative Planning Table\n{table_text}\n\n"
        f"## Question\n{question}\n\n"
        "## Synthesis instructions\n"
        "1. Use the planning table findings as your PRIMARY evidence; weight a "
        "finding heavily if it directly addresses an option.\n"
        "2. Walk through the multi-hop chain explicitly, naming each intermediate "
        "entity from the findings, before concluding.\n"
        "3. ALWAYS commit to a final answer using the format `Answer: ...`. "
        "Even if the chain is incomplete, give your best single-span guess; "
        "do NOT abstain or say 'information not provided'."
    )
    final_answer = _llm_call(_system_prompt(config, "rag"), final_user, label="iter_ptable/final")

    gold_idx = str(row.get("gold_idx", ""))
    gold_idxs = {s.strip() for s in gold_idx.split(",") if s.strip()}
    gold_retrieved = bool(gold_idxs & set(all_retrieved_ids)) if gold_idxs else False

    return {
        "final_answer": final_answer,
        "planning_table": table_entries,
        "rounds_completed": rounds_completed,
        "early_exit": early_exit,
        "retrieved_ids": list(dict.fromkeys(all_retrieved_ids)),
        "gold_retrieved": gold_retrieved,
        "evidence_store": [
            {"idx": eid, "text": "", "source": "iter_ptable", "cross_encoder_score": 0.0}
            for eid in dict.fromkeys(all_retrieved_ids)
        ],
    }


def run_iter_hyde(row: pd.Series, config: EvalConfig) -> dict:
    """Multi-round HyDE conditioned on prior-round findings. The deep-research
    answer to single-round `multi_hyde_diverse`.

    User feedback 2026-04-26: "multi hop is going to be bad unless it has
    multi step or multiple rounds." Single-round mhd fights the *single-hop
    commitment bias* of HyDE but does NOT fight the *composition over
    multiple passages* bottleneck.

    Per round:
      1. Generate ONE HyDE passage (encyclopedia/textbook style, 2-3 sentences)
         conditioned on what the prior rounds' findings established. Round 1
         HyDE is the next-hop-likely-entity; round 2 HyDE is conditioned on
         round 1's finding; etc.
      2. Use that HyDE passage as the retrieval query (embeds well in dense
         retrievers, BM25-tokenizes for in-row MuSiQue).
      3. Write a 2-3 sentence finding from the retrieved passages.
      4. Ask the synth-decider node: "READY to answer, or need another HyDE
         round?" — early-exit if READY (mirrors iter_planning_table).
      5. Max MAX_ROUNDS = 3.

    Final synthesis sees the full HyDE-finding chain, walks through the
    multi-hop reasoning explicitly, commits to a final answer.

    Steps: 1 (initial HyDE) + per round: retrieve + finding + ready-check +
    next-HyDE, capped at 3 rounds → ~7-10 LLM calls per question (same as
    iter_planning_table).
    """
    MAX_ROUNDS = 3
    question = _fmt(row, config)
    question_intermediate = _fmt_intermediate(row, config)

    initial_hyde_system = (
        "You are an encyclopedia author. Given a multi-hop question, write ONE "
        "short hypothetical answer-passage (2-3 sentences) about the FIRST "
        "intermediate fact you would need to investigate.\n\n"
        "Rules:\n"
        "- Focus on the FIRST hop (the upstream entity to identify), not the "
        "final answer\n"
        "- Write factual encyclopedia/textbook prose — not a question, not a "
        "multiple-choice letter\n"
        "- Do NOT use angle brackets, square brackets, or placeholders\n"
        "- Be specific and entity-rich (helps BM25 retrieval)"
    )
    initial_user = f"## Multi-hop Question\n{question_intermediate}"
    initial_raw = _llm_call(initial_hyde_system, initial_user, label="iter_hyde/init_hyde")
    routed_to = None
    current_hyde = _sanitize_intermediate_text(initial_raw, fallback=question_intermediate).strip()
    if not current_hyde:
        current_hyde = question_intermediate
        routed_to = "iter_hyde_initial_empty_fallback"

    table_entries: list[dict] = []
    all_retrieved_ids: list[str] = []
    finding_system = (
        "You are a research assistant. Read the retrieved passages and write a "
        "concise finding (2-3 sentences) about the sub-investigation focus. "
        "If the passages don't contain the answer, say so explicitly. Do NOT "
        "pick a multiple-choice letter."
    )
    next_hyde_system_template = (
        "You are an encyclopedia author working iteratively on a multi-hop "
        "question. You have completed {n_rounds} HyDE-and-find round(s) so far "
        "(the chain below). Your job: decide whether the chain is now "
        "sufficient to answer the original question, or whether you need ONE "
        "more HyDE round.\n\n"
        "Output EXACTLY ONE of these two formats:\n\n"
        "READY: I have enough information to answer the question.\n\n"
        "OR\n\n"
        "HYDE: <one short hypothetical-answer passage (2-3 sentences) about the "
        "next intermediate fact, conditioned on what's been learned>\n\n"
        "Rules:\n"
        "- Only emit READY if the chain unambiguously composes to an answer\n"
        "- If picking HYDE, USE the prior findings to focus the next passage "
        "(e.g., if Round 1 found 'Movie X stars Actor Y', Round 2 might HyDE "
        "facts about Actor Y's other roles)\n"
        "- Do NOT pick a multiple-choice letter or commit to a final answer"
    )
    rounds_completed = 0
    early_exit = False
    for round_idx in range(MAX_ROUNDS):
        retrieval = _retrieve_and_format(
            row, [current_hyde], k=3,
            label_prefix=f"iter_hyde/r{round_idx}",
            where=_where_from_config(config),
            collection=_collection_for_config(config),
        )
        passages_block = "\n\n".join(retrieval["passages"]) if retrieval["passages"] else "(no passages retrieved)"
        finding_user = (
            f"## Retrieved Passages\n{passages_block}\n\n"
            f"## HyDE Focus (round {round_idx + 1})\n{current_hyde}"
        )
        finding = _generate_report(
            finding_system,
            finding_user,
            label=f"iter_hyde/finding_r{round_idx}",
            fallback="No relevant information found in the retrieved passages.",
        )
        table_entries.append({
            "hyde": current_hyde,
            "finding": finding["text"],
            "evidence_ids": retrieval.get("retrieved_ids", []),
            "round": round_idx + 1,
        })
        all_retrieved_ids.extend(retrieval.get("retrieved_ids", []))
        rounds_completed += 1

        if round_idx + 1 >= MAX_ROUNDS:
            break

        chain_text = "\n\n".join(
            f"### Round {e['round']} — HyDE: {e['hyde']}\n**Finding:** {e['finding']}"
            for e in table_entries
        )
        decide_user = (
            f"## Original Question\n{question_intermediate}\n\n"
            f"## HyDE-Finding Chain So Far\n{chain_text}"
        )
        decide_system = next_hyde_system_template.format(n_rounds=len(table_entries))
        decide_raw = _llm_call(decide_system, decide_user, label=f"iter_hyde/decide_r{round_idx}")

        decide_clean = decide_raw.strip()
        if re.search(r"\bREADY\b", decide_clean[:200], re.IGNORECASE) and not re.search(r"\bHYDE\s*:", decide_clean[:200], re.IGNORECASE):
            early_exit = True
            break
        next_hyde = ""
        m = re.search(r"^\s*(?:\*\*)?\s*HYDE\s*[:\-]\s*(?:\*\*)?\s*(.+?)\s*(?:\*\*)?\s*$",
                      decide_clean, re.IGNORECASE | re.MULTILINE | re.DOTALL)
        if m:
            next_hyde = _sanitize_intermediate_text(m.group(1), fallback=question_intermediate).strip()
        if not next_hyde:
            early_exit = True
            routed_to = "iter_hyde_early_exit_empty_decider"
            break
        current_hyde = next_hyde

    chain_text = "\n\n".join(
        f"### Round {e['round']} — HyDE: {e['hyde']}\n**Finding:** {e['finding']}"
        for e in table_entries
    )
    final_user = (
        f"## Iterative HyDE Chain\n{chain_text}\n\n"
        f"## Question\n{question}\n\n"
        "## Synthesis instructions\n"
        "1. Use the HyDE-finding chain as your PRIMARY evidence. If a finding "
        "directly addresses an option, weight it heavily.\n"
        "2. Walk through the multi-hop chain explicitly, naming each "
        "intermediate entity from the findings, before concluding.\n"
        "3. ALWAYS commit to a final answer using the format `Answer: ...`. "
        "Even if the chain is incomplete, give your best single-span guess; "
        "do NOT abstain or say 'information not provided'."
    )
    final_answer = _llm_call(_system_prompt(config, "rag"), final_user, label="iter_hyde/final")
    if not (final_answer or "").strip():
        raise RuntimeError("iter_hyde final answer returned empty response")

    gold_idx = str(row.get("gold_idx", ""))
    gold_idxs = {s.strip() for s in gold_idx.split(",") if s.strip()}
    gold_retrieved = bool(gold_idxs & set(all_retrieved_ids)) if gold_idxs else False

    return {
        "final_answer": final_answer,
        "formatted_question": question,
        "hyde_chain": table_entries,
        "rounds_completed": rounds_completed,
        "early_exit": early_exit,
        "retrieved_ids": list(dict.fromkeys(all_retrieved_ids)),
        "gold_retrieved": gold_retrieved,
        "routed_to": routed_to,
        "evidence_store": [
            {"idx": eid, "text": "", "source": "iter_hyde", "cross_encoder_score": 0.0}
            for eid in dict.fromkeys(all_retrieved_ids)
        ],
    }


def run_rag_multi_query(row: pd.Series, config: EvalConfig) -> dict:
    """Multi-query rag_simple: 2-3 question rewrites → pool retrievals → answer once.

    NEW method designed to test whether RETRIEVAL DIVERSITY alone (without snap,
    HyDE, or per-hop decomposition) beats single-query rag_simple on multi-hop.

    Avoids both failure modes from earlier multi-hop runs:
      - Snap-bias: never generates a hypothesis answer that biases retrieval
      - Composition tax: still has ONE final reasoning step over all retrieved
        passages (like rag_simple), not per-hop synthesis

    Steps (~3 LLM calls):
      1. Generate 2 question rewrites that target different sub-aspects of the
         multi-hop question
      2. Retrieve k=3 passages for each rewrite + the original question
      3. Pool, dedupe, configured top-k by max BM25 score
      4. Answer once with the pooled passages + original question
    """
    question = _fmt(row, config)
    question_intermediate = _fmt_intermediate(row, config)

    # Step 1: rewrite generation
    rewrite_system = (
        "You are a research planner. Given a multi-hop question, write TWO "
        "different sub-question rewrites that target different aspects of the "
        "question. The goal is RETRIEVAL DIVERSITY — each rewrite should be "
        "phrased to find DIFFERENT passages than the original.\n\n"
        "Output exactly two rewrites, one per line, in this format:\n"
        "REWRITE: <question rephrased to target hop 1 / entity A>\n"
        "REWRITE: <question rephrased to target hop 2 / entity B>\n\n"
        "STRICT OUTPUT RULES:\n"
        "- Do NOT pick an answer\n"
        "- Each rewrite is a question, not a statement\n"
        "- Phrase each to surface DIFFERENT topics than the original"
    )
    rewrite_user = f"## Multi-hop Question\n{question_intermediate}"
    rewrite_raw = _llm_call(rewrite_system, rewrite_user, label="multi_query/rewrite")

    rewrites: list[str] = []
    for line in rewrite_raw.splitlines():
        line = line.strip()
        m = re.match(r"^\s*(?:\*\*)?\s*REWRITE\s*[:\-]\s*(?:\*\*)?\s*(.+?)\s*(?:\*\*)?\s*$", line, re.IGNORECASE)
        if m:
            rewrite = m.group(1).strip().lstrip("-•").strip()
            if rewrite:
                rewrites.append(rewrite)
    rewrites = rewrites[:2]

    # Always include the original question as a query — even if rewrites fail
    raw_question = _retrieval_question(row)
    queries = [raw_question] + rewrites

    # Step 2-3: retrieve once with the pooled query list (max-pool over queries)
    retrieval = _retrieve_and_format(
        row, queries, k=config.retrieval_k,
        label_prefix="multi_query",
        where=_where_from_config(config),
        collection=_collection_for_config(config),
    )
    passage_block = "\n\n".join(retrieval["passages"])

    # Step 4: single-shot answer with pooled passages
    user = (
        f"## Retrieved Passages (from {len(queries)} diverse queries)\n{passage_block}\n\n"
        f"## Question\n{question}"
    )
    answer = _llm_call(_system_prompt(config, "rag"), user, label="multi_query/answer")

    return {
        "final_answer": answer,
        "rewrites": rewrites,
        "n_queries": len(queries),
        "evidence_store": retrieval["evidence_store"],
        "retrieved_ids": retrieval["retrieved_ids"],
        "gold_retrieved": retrieval["gold_retrieved"],
    }


def run_planning_table_no_snap(row: pd.Series, config: EvalConfig) -> dict:
    """Ablation of planning_table — generates TODOs from the QUESTION ALONE, no snap.

    Tests the snap-bias hypothesis from MuSiQue: planning_table at 20.7% EM
    matches every other snap-driven mode (rag_hyde, rag_snap_hyde, subagent_rag)
    on multi-hop, while plain rag_simple gets 26.7%. If removing snap recovers
    the rag_simple baseline, the failure source is snap-bias not pipeline depth.

    Same structure as planning_table but skips Step 1 (snap). Plan-gen reads
    only the question.
    """
    question = _fmt(row, config)
    question_intermediate = _fmt_intermediate(row, config)

    # Step 1 (formerly): SKIPPED — no snap

    # Step 2: plan generation FROM QUESTION ONLY
    plan_system = (
        "You are a research planner. Read a question and identify 2-3 specific "
        "fact-focused sub-questions that, if answered, would let you answer the original.\n\n"
        "Output exactly 2-3 sub-questions, one per line, in this format:\n"
        "TODO: <focused sub-question>\n"
        "TODO: <focused sub-question>\n"
        "TODO: <focused sub-question>\n\n"
        "STRICT OUTPUT RULES:\n"
        "- Sub-questions should be FACT-FOCUSED (what is X? when did Y happen? who is Z?)\n"
        "- Decompose multi-hop into individual hops where possible\n"
        "- Do NOT pick an answer; output only TODO lines"
    )
    plan_user = f"## Question\n{question_intermediate}"
    plan_raw = _llm_call(plan_system, plan_user, label="ptable_ns/plan")

    todos: list[str] = []
    for line in plan_raw.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^\s*(?:\*\*)?\s*TODO\s*[:\-]\s*(?:\*\*)?\s*(.+?)\s*(?:\*\*)?\s*$", line, re.IGNORECASE)
        if m:
            todo = m.group(1).strip().lstrip("-•").strip()
            if todo:
                todos.append(todo)

    # Dedup
    seen_normalized: set[str] = set()
    deduped: list[str] = []
    for t in todos:
        norm = re.sub(r"\s+", " ", t.lower().strip().rstrip("?.!"))
        if norm and norm not in seen_normalized:
            seen_normalized.add(norm)
            deduped.append(t)
    todos = deduped[:3]
    if not todos:
        todos = [question_intermediate]

    # Step 3: per-TODO retrieve + finding (same as planning_table)
    table_entries: list[dict] = []
    all_retrieved_ids: list[str] = []
    finding_system = (
        "You are a research assistant. Answer the sub-question concisely "
        "(2-3 sentences max) using ONLY the retrieved passages. If the passages don't "
        "contain the answer, say so explicitly. Do NOT pick a multiple-choice letter."
    )
    for i, todo in enumerate(todos):
        retrieval = _retrieve_and_format(
            row, [todo], k=3,
            label_prefix=f"ptable_ns/todo_{i}",
            where=_where_from_config(config),
            collection=_collection_for_config(config),
        )
        passages_block = "\n\n".join(retrieval["passages"]) if retrieval["passages"] else "(no passages retrieved)"
        finding_user = (
            f"## Retrieved Passages\n{passages_block}\n\n"
            f"## Sub-Question\n{todo}"
        )
        finding = _generate_report(
            finding_system,
            finding_user,
            label=f"ptable_ns/finding_{i}",
            fallback="No relevant information found in the retrieved passages.",
        )
        table_entries.append({
            "todo": todo,
            "finding": finding["text"],
            "evidence_ids": retrieval.get("retrieved_ids", []),
        })
        all_retrieved_ids.extend(retrieval.get("retrieved_ids", []))

    # Step 4: final answer
    table_text = "\n\n".join(
        f"### TODO {i+1}: {e['todo']}\n**Finding:** {e['finding']}"
        for i, e in enumerate(table_entries)
    )
    # Audit 2026-04-26 found 17/30 records where findings said "passages do
    # not contain X" but the final agent ignored them and asserted a parametric
    # guess anyway. Tighten the synthesizer instruction explicitly.
    final_user = (
        f"## Planning Table (your sub-investigations)\n{table_text}\n\n"
        f"## Question\n{question}\n\n"
        "## Synthesis instructions\n"
        "1. Use the planning table findings as your PRIMARY evidence. If a "
        "finding directly contradicts a possible answer, weight it heavily.\n"
        "2. If multiple findings need to compose the answer, walk through "
        "the chain explicitly before concluding.\n"
        "3. ALWAYS commit to a final answer using the format `Answer: ...`. "
        "Even if findings are incomplete, give your best single-span guess — "
        "do NOT abstain or say 'information not provided'."
    )
    final_answer = _llm_call(_system_prompt(config, "rag"), final_user, label="ptable_ns/final")

    gold_idx = str(row.get("gold_idx", ""))
    gold_idxs = {s.strip() for s in gold_idx.split(",") if s.strip()}
    gold_retrieved = bool(gold_idxs & set(all_retrieved_ids)) if gold_idxs else False

    return {
        "final_answer": final_answer,
        "planning_table": table_entries,
        "todos_count": len(todos),
        "retrieved_ids": list(dict.fromkeys(all_retrieved_ids)),
        "gold_retrieved": gold_retrieved,
        "evidence_store": [
            {"idx": eid, "text": "", "source": "ptable_ns", "cross_encoder_score": 0.0}
            for eid in dict.fromkeys(all_retrieved_ids)
        ],
    }


# Modes that do NOT use ChromaDB retrieval. Used by pre-flight collection
# check and the empty-retrieval summary guard to skip non-RAG modes.
# Includes golden_passage variants (they INJECT row['golden_passage'] into the
# prompt rather than retrieving from a vector store) and the historical
# vectorless / pure-LLM modes.
_NO_CHROMA_MODES = {
    "llm_only", "snap_only_in_final", "decompose", "self_verify",
    "friend_foe_attribution", "double_snap", "snap_debate",
    "vectorless_direct", "vectorless_role", "vectorless_elements",
    "vectorless_choice_map", "vectorless_nosnap", "golden_passage",
    "golden_arbitration", "golden_arb_conservative",
}


MODE_RUNNERS = {
    "full_pipeline": run_full_pipeline,
    "llm_only": run_llm_only,
    "rag_rewrite": run_rag_rewrite,
    "rag_simple": run_rag_simple,
    "golden_passage": run_golden_passage,
    "golden_arbitration": run_golden_arbitration,
    "golden_arb_conservative": run_golden_arb_conservative,
    "rag_arbitration": run_rag_arbitration,
    "rag_hyde": run_rag_hyde,
    "rag_hyde_arb": run_rag_hyde_arb,
    "rag_multi_hyde": run_rag_multi_hyde,
    "rag_snap_hyde": run_rag_snap_hyde,
    "rag_snap_hyde_2call": run_rag_snap_hyde_2call,
    "snap_hyde_aligned": run_snap_hyde_aligned,
    "gap_hyde": run_gap_hyde,
    "gap_hyde_ev": run_gap_hyde_ev,
    "gap_hyde_nosnap": run_gap_hyde_nosnap,
    "gap_hyde_flat": run_gap_hyde_flat,
    "gap_rag": run_gap_rag,
    "gap_rag_nosnap": run_gap_rag_nosnap,
    "gap_vectorless": run_gap_vectorless,
    "subagent_hyde": run_subagent_hyde,
    "subagent_rag": run_subagent_rag,
    "subagent_hybrid": run_subagent_hybrid,
    "subagent_rag_evidence": run_subagent_rag_evidence,
    "subagent_rag_snap": run_subagent_rag_snap,
    "subagent_rag_full": run_subagent_rag_full,
    "snap_hyde_report": run_snap_hyde_report,
    "snap_hyde_report_snap": run_snap_hyde_report_snap,
    "snap_rag": run_snap_rag,
    "snap_rag_nosnap": run_snap_rag_nosnap,
    "vectorless_direct": run_vectorless_direct,
    "vectorless_role": run_vectorless_role,
    "vectorless_elements": run_vectorless_elements,
    "vectorless_choice_map": run_vectorless_choice_map,
    "vectorless_nosnap": run_vectorless_nosnap,
    "vectorless_hybrid": run_vectorless_hybrid,
    "vectorless_keyword": run_vectorless_keyword,
    "entity_search": run_entity_search,
    "snap_entity_search": run_snap_entity_search,
    "snap_entity_informed": run_snap_entity_informed,
    "rag_devil_hyde": run_rag_devil_hyde,
    "rag_top2_hyde": run_rag_top2_hyde,
    "confidence_gated": run_confidence_gated,
    "decompose": run_decompose,
    "decompose_rag": run_decompose_rag,
    "ce_threshold": run_ce_threshold,
    "conf_ce_threshold": run_conf_ce_threshold,
    "snap_hyde_aspect": run_snap_hyde_aspect,
    "ce_threshold_k3": run_ce_threshold_k3,
    "self_verify": run_self_verify,
    "double_snap": run_double_snap,
    "snap_debate": run_snap_debate,
    "snap_only_in_final": run_snap_only_in_final,
    "planning_table": run_planning_table,
    "planning_table_no_snap": run_planning_table_no_snap,
    "rag_multi_query": run_rag_multi_query,
    "iterative_planning_table": run_iterative_planning_table,
    "advisor_planning_table": run_advisor_planning_table,
    "multi_hyde_diverse": run_multi_hyde_diverse,
    "iter_hyde": run_iter_hyde,
    "friend_foe_attribution": run_friend_foe_attribution,
}


# ---------------------------------------------------------------------------
# Harness Core
# ---------------------------------------------------------------------------

def _setup_provider(config: EvalConfig):
    """Set env vars and clear caches for provider/skill switching."""
    os.environ["LLM_PROVIDER"] = config.provider
    _get_llm_cached.cache_clear()

    if config.skill_dir != "skills":
        os.environ["SKILL_DIR"] = config.skill_dir
        load_skill.cache_clear()


def _git_commit_short() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def _serialize_result(result: dict) -> dict:
    """Ensure all values are JSON-serializable."""
    out = {}
    for k, v in result.items():
        if isinstance(v, float) and (v != v):  # NaN check
            out[k] = None
        else:
            try:
                json.dumps(v)
                out[k] = v
            except (TypeError, ValueError):
                out[k] = str(v)
    return out


def run_eval(config: EvalConfig):
    """Run evaluation with the given config."""
    if config.mode not in MODE_RUNNERS:
        print(f"Unknown mode '{config.mode}'. Available: {', '.join(MODE_RUNNERS)}")
        sys.exit(1)

    _setup_provider(config)
    runner = MODE_RUNNERS[config.mode]
    provider_info = get_provider_info()
    embedding_model = os.getenv("EVAL_EMBEDDING_MODEL", "").strip() or None

    qa = load_questions(config)
    n = len(qa)

    print(f"\n{'=' * 70}")
    filter_str = f" | filter={config.source_filter}" if config.source_filter else ""
    dataset_str = f" | dataset={config.dataset}" if config.dataset != "barexam" else ""
    print(f"EVAL: {config.mode} | {provider_info['provider']} ({provider_info['model']}) | {n} questions{dataset_str}{filter_str}")
    if config.skill_dir != "skills":
        print(f"Skills: {config.skill_dir}")
    if config.tag:
        print(f"Tag: {config.tag}")
    print(f"{'=' * 70}\n")

    # Pre-flight smoke: fire ONE test call to catch auth/404 failures BEFORE
    # iterating questions. Audit 2026-04-26 caught 7 silent-auth-failure rows
    # where 100% of records errored and the run wrote a misleading 0% accuracy.
    # Skip for cluster-vllm (already running locally and visible).
    if config.provider not in ("custom", "cluster-vllm"):
        try:
            _smoke = _base_llm_call(
                "You are a test endpoint.",
                "Reply with exactly: OK",
                label="preflight_smoke",
            )
            if not _smoke or "OK" not in _smoke.upper():
                print(f"[preflight] WARNING: smoke returned unexpected: {_smoke!r}")
            else:
                print(f"[preflight] provider={config.provider} OK")
        except Exception as exc:
            print(f"[preflight] FAILED for provider={config.provider}: {exc}")
            print(f"[preflight] aborting before logging garbage. Verify API key/model availability.")
            raise SystemExit(2)

    # Pre-flight collection check: verify the configured ChromaDB collection
    # has docs BEFORE the run starts. Catches the "empty corpus → silent zero
    # retrieval → garbage findings" failure mode discovered 2026-04-26 on the
    # advisor BarExam run (50/50 rows had retrieved_ids=[], all findings said
    # "No answer available", but the strong LLM still produced 72% from
    # parametric knowledge — a misleading number we almost cited).
    # Skip for modes that don't use ChromaDB (musique uses in-row BM25;
    # llm_only and snap_only_in_final don't retrieve at all).
    if config.dataset != "musique" and config.mode not in _NO_CHROMA_MODES:
        try:
            from rag_utils import get_vectorstore
            _coll_name = _collection_for_config(config) if "_collection_for_config" in globals() else "legal_passages"
            _vs = get_vectorstore(_coll_name, embedding_model=os.getenv("EVAL_EMBEDDING_MODEL", "").strip() or None)
            _coll_count = _vs._collection.count() if hasattr(_vs, "_collection") else None
            if _coll_count is not None and _coll_count == 0:
                print(f"[preflight] FAILED: collection '{_coll_name}' is EMPTY (0 docs).")
                print(f"[preflight] mode={config.mode} requires retrieval but corpus is missing.")
                print(f"[preflight] Rebuild via: uv run python utils/fast_embed.py barexam (or housing)")
                raise SystemExit(4)
            elif _coll_count is not None:
                print(f"[preflight] collection={_coll_name} has {_coll_count:,} docs OK")
        except SystemExit:
            raise
        except Exception as exc:
            print(f"[preflight] WARNING: could not verify collection: {exc}")

    results = []
    correct = 0
    total_start = time.time()
    consecutive_errors = 0

    is_open_ended = config.dataset in ("legal_rag", "australian")
    is_short_span = config.dataset == "musique"

    for i, row in qa.iterrows():
        _reset_llm_call_counter()
        _reset_call_trace()
        _reset_trace_events()
        q_start = time.time()

        # Dataset-specific labeling
        if config.dataset == "housing":
            subject = str(row.get("state", "unknown"))
            label = f"hqa_{subject}_{row.get('idx', i)}"
        elif config.dataset == "casehold":
            subject = "casehold"
            label = f"ch_{row.get('idx', i)}"
        elif config.dataset == "legal_rag":
            subject = "crim_law"
            label = f"lrq_{row.get('idx', i)}"
        elif config.dataset == "australian":
            subject = str(row.get("jurisdiction", "unknown"))
            label = f"aus_{subject}_{row.get('idx', i)}"
        elif config.dataset == "musique":
            subject = f"{int(row.get('n_hops', 0))}-hop"
            label = f"mq_{row.get('idx', i)}"
        else:
            subject = str(row.get("subject", "unknown"))
            label = f"qa_{subject}_{row.get('idx', i)}"
        idx = str(row.get("idx", i))

        # Gold answer formatting
        gold = str(row["answer"]).strip()
        if config.dataset == "housing":
            gold = gold.capitalize()
        elif config.dataset in ("barexam", "casehold"):
            gold = gold.upper()
        # open-ended: gold stays as-is

        try:
            result = runner(row, config)
            answer_text = result.get("final_answer", "")
            predicted = _extract_answer(answer_text, config)

            if is_open_ended:
                is_correct = _judge_open_answer(row["question"], gold, answer_text, config)
                result["judge_score"] = is_correct  # store for analysis
            elif is_short_span:
                # MuSiQue uses EM/F1 with answer_aliases for tolerant scoring
                aliases_raw = row.get("answer_aliases", "")
                try:
                    aliases = json.loads(aliases_raw) if aliases_raw else []
                except Exception:
                    aliases = []
                em, f1 = musique_em_f1(predicted or "", gold, aliases)
                is_correct = em
                result["em"] = em
                result["f1"] = f1
                result["aliases_used"] = aliases
            else:
                is_correct = predicted == gold
            error = None
        except Exception as e:
            result = {}
            answer_text = ""
            predicted = None
            is_correct = False
            error = str(e)

        # Circuit breaker: if 5 consecutive questions error, the run is broken —
        # abort before logging more garbage. Common cause: auth, rate-limit,
        # model-not-found.
        if error:
            consecutive_errors += 1
            if consecutive_errors >= 5:
                print(
                    f"\n[circuit-breaker] {consecutive_errors} consecutive errors. "
                    f"Last error: {error[:200]}\n"
                    f"[circuit-breaker] Aborting at question {i+1} to avoid garbage results."
                )
                raise SystemExit(3)
        else:
            consecutive_errors = 0

        elapsed = time.time() - q_start
        metrics = _get_metrics()

        if is_correct:
            correct += 1

        status = "PASS" if is_correct else "FAIL"
        if is_open_ended:
            print(
                f"[{i+1}/{n}] {label:<35} {status:<6} "
                f"({elapsed:.1f}s, {metrics['count']} calls)",
                flush=True,
            )
        else:
            print(
                f"[{i+1}/{n}] {label:<35} {status:<6} "
                f"gold={gold} pred={predicted} "
                f"({elapsed:.1f}s, {metrics['count']} calls)",
                flush=True,
            )

        # Build per-question record
        record = {
            "label": label,
            "subject": subject,
            "idx": idx,
            "question": str(row["question"])[:500],
            "correct_answer": gold[:500] if is_open_ended else gold,
            "predicted_answer": str(predicted)[:500] if is_open_ended else predicted,
            "is_correct": is_correct,
            "error": error,
            "elapsed_sec": round(elapsed, 1),
            "llm_calls": metrics["count"],
            "input_tokens": metrics["input_tokens"],
            "output_tokens": metrics["output_tokens"],
            "gold_idx": str(row.get("gold_idx", "")),
            "final_answer": answer_text[:500] if is_open_ended else answer_text,
            "mode": config.mode,
            "provider": config.provider,
            "dataset": config.dataset,
            "embedding_model": embedding_model,
        }
        if _trace_calls_enabled():
            record["call_trace"] = _get_call_trace()
        if _trace_events_enabled():
            record["trace_events"] = _get_trace_events()
            record["trace_schema_version"] = 1
        if config.dataset == "housing":
            record["state"] = str(row.get("state", ""))
        elif config.dataset == "casehold":
            record["choices"] = {
                letter: str(row.get(f"choice_{letter.lower()}", ""))[:200]
                for letter in "ABCDE"
            }
        elif config.dataset == "australian":
            record["jurisdiction"] = str(row.get("jurisdiction", ""))
        elif config.dataset == "legal_rag":
            record["relevant_passages"] = str(row.get("relevant_passages", ""))
        else:
            record["choices"] = {
                letter: str(row.get(f"choice_{letter.lower()}", ""))
                for letter in "ABCD"
            }
            record["gold_passage"] = str(row.get("gold_passage", ""))[:500]
        # Merge mode-specific fields (evidence_store, audit_log, etc.)
        for k, v in result.items():
            if k != "final_answer" and k not in record:
                record[k] = v

        # Normalize retrieval-schema keys: every record must have gold_retrieved,
        # retrieved_ids, and evidence_store, even on no-retrieval modes. Downstream
        # analysis scripts use key-presence to detect retrieval, so the absent-vs-
        # explicit-False distinction matters.
        record.setdefault("gold_retrieved", False)
        record.setdefault("retrieved_ids", [])
        record.setdefault("evidence_store", [])
        if "snap1" in record:
            record.setdefault("snap_answer", record["snap1"])
        snap_list = record.get("snaps")
        if not isinstance(snap_list, list):
            snap_list = record.get("snap_answers")
        if snap_list:
            record.setdefault("snap_answer", snap_list[0])
        if "letter1" in record:
            record.setdefault("snap_letter", record["letter1"])

        results.append(_serialize_result(record))

    total_time = time.time() - total_start
    accuracy = correct / n if n > 0 else 0

    # --- Print summary ---
    print(f"\n{'=' * 70}")
    print(f"RESULTS: {correct}/{n} ({accuracy*100:.1f}%)")
    print(f"Total time: {total_time:.0f}s ({total_time/n:.1f}s/query)")

    # By-subject breakdown
    by_subject = {}
    for r in results:
        subj = r.get("subject", "unknown")
        if subj not in by_subject:
            by_subject[subj] = [0, 0]
        by_subject[subj][1] += 1
        if r["is_correct"]:
            by_subject[subj][0] += 1

    if len(by_subject) > 1:
        print("\nBy subject:")
        for subj, (c, t) in sorted(by_subject.items()):
            print(f"  {subj:<15} {c}/{t} ({c/t*100:.0f}%)")

    print(f"{'=' * 70}")

    # --- Save detail log ---
    ts = time.strftime("%Y%m%d_%H%M")
    question_set = config.questions if config.questions in ("curated", "full") else f"n{config.questions}"
    detail_filename = f"eval_{config.mode}_{config.provider}_{ts}_detail.jsonl"
    detail_path = os.path.join("logs", detail_filename)
    os.makedirs("logs", exist_ok=True)

    with open(detail_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nDetail log: {detail_path}")

    # --- Append to experiments.jsonl ---
    run_id = f"{ts}_{config.mode}_{config.provider}"
    if config.tag:
        run_id += f"_{config.tag}"

    summary = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": config.mode,
        "dataset": config.dataset,
        "provider": provider_info["provider"],
        "model": provider_info["model"],
        "embedding_model": embedding_model,
        "question_set": question_set,
        "n_questions": n,
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": n,
        "by_subject": by_subject,
        "avg_latency_sec": round(total_time / n, 1) if n else 0,
        "avg_llm_calls": round(sum(r["llm_calls"] for r in results) / n, 1) if n else 0,
        "total_input_tokens": sum(r["input_tokens"] for r in results),
        "total_output_tokens": sum(r["output_tokens"] for r in results),
        "skill_dir": config.skill_dir,
        "tag": config.tag,
        "source_filter": config.source_filter,
        "detail_log": detail_path,
        "git_commit": _git_commit_short(),
    }

    # Audit 2026-04-26 caught silent-failure rows polluting experiments.jsonl
    # (100% errors → accuracy=0.0 logged as legitimate baseline). Refuse to
    # append a normal-looking row when the run was mostly broken; tag it
    # explicitly instead.
    error_count = sum(1 for r in results if r.get("error"))
    error_rate = error_count / max(n, 1)
    if error_rate > 0.5:
        summary["tag"] = (config.tag + "_FAILED-do-not-use").lstrip("_")
        summary["error_count"] = error_count
        summary["error_rate"] = round(error_rate, 3)
        print(
            f"\n[summary-guard] {error_count}/{n} ({error_rate:.0%}) records errored. "
            f"Tagging summary as FAILED-do-not-use to avoid polluting analysis."
        )

    # Empty-retrieval guard: if a RAG mode produced 0 retrieved docs on >50%
    # of rows, the corpus or routing is broken and the accuracy number reflects
    # parametric knowledge of the LLM only — not the method. Tag as FAILED.
    # Discovered 2026-04-26 on advisor BarExam where legal_passages had 0 docs
    # on this Mac and 50/50 rows had retrieved_ids=[] but accuracy=72%.
    is_rag_mode = config.mode not in _NO_CHROMA_MODES and config.dataset != "musique"
    if is_rag_mode:
        empty_ret = sum(1 for r in results if r.get("retrieved_ids") == [])
        empty_ret_rate = empty_ret / max(n, 1)
        if empty_ret_rate > 0.5:
            summary["tag"] = (config.tag + "_FAILED-EMPTY-RETRIEVAL").lstrip("_")
            summary["empty_retrieval_rate"] = round(empty_ret_rate, 3)
            summary["empty_retrieval_count"] = empty_ret
            print(
                f"\n[summary-guard] {empty_ret}/{n} ({empty_ret_rate:.0%}) records had empty retrieval. "
                f"Tagging summary as FAILED-EMPTY-RETRIEVAL — accuracy is from parametric LLM knowledge, not the method."
            )

    experiments_path = os.path.join("logs", "experiments.jsonl")
    with open(experiments_path, "a") as f:
        f.write(json.dumps(summary) + "\n")
    print(f"Run summary appended to: {experiments_path}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multi-model evaluation harness for Legal RAG pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(f"  {k:<20} {v}" for k, v in EVAL_MODES.items()),
    )
    parser.add_argument("--mode", default="full_pipeline", choices=EVAL_MODES.keys(),
                        help="Evaluation mode (default: full_pipeline)")
    parser.add_argument("--provider", default="deepseek",
                        help="LLM provider key from llm_config.py (default: deepseek)")
    parser.add_argument("--questions", default="30",
                        help="'curated', 'full', or integer N (default: 30)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for question sampling (default: 42)")
    parser.add_argument("--skill-dir", default="skills",
                        help="Directory containing skill prompts (default: skills)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("--tag", default="",
                        help="Optional label for this run")
    parser.add_argument("--source-filter", default="",
                        help="Metadata source filter for retrieval, e.g. 'mbe' (default: none)")
    parser.add_argument("--dataset", default="barexam",
                        choices=["barexam", "housing", "legal_rag", "australian", "casehold", "musique"],
                        help="Dataset to evaluate on (default: barexam)")
    parser.add_argument("--retrieval-k", type=int, default=5,
                        help="Final top-k after rerank for retrieval modes (default 5; meeting ask: top-1 vs top-5 ablation)")

    args = parser.parse_args()

    if args.verbose:
        os.environ["VERBOSE"] = "1"

    config = EvalConfig(
        mode=args.mode,
        provider=args.provider,
        questions=args.questions,
        seed=args.seed,
        skill_dir=args.skill_dir,
        verbose=args.verbose,
        tag=args.tag,
        source_filter=args.source_filter,
        dataset=args.dataset,
        retrieval_k=args.retrieval_k,
    )

    run_eval(config)


if __name__ == "__main__":
    main()
