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
import concurrent.futures
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
from typing import Any, List
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from langchain_core.documents import Document
from eval_config import (
    BEIR_DATASETS,
    EvalConfig,
    EVAL_MODES,
    extract_answer_mc,
    extract_answer_mc5,
    extract_answer_musique,
    extract_answer_yn,
    format_question_prompt,
    is_beir_dataset,
    load_questions,
    musique_em_f1,
)
from main import (
    run as run_pipeline,
    _llm_call as _base_llm_call,
    _get_metrics,
    _reset_llm_call_counter,
    _parse_json,
    load_skill,
)
from llm_config import get_provider_info, _get_llm_cached
from rag_utils import retrieve_documents_multi_query, get_documents_by_idx, get_vectorstore


_TRACE_STATE = threading.local()
_RETRIEVAL_DOC_CACHE: dict[tuple[str, str], Document] | None = None
_RETRIEVAL_DOC_CACHE_PATH: str | None = None


def _call_trace_buffer() -> list[dict]:
    buf = getattr(_TRACE_STATE, "call_trace", None)
    if buf is None:
        buf = []
        _TRACE_STATE.call_trace = buf
    return buf


def _trace_events_buffer() -> list[dict]:
    buf = getattr(_TRACE_STATE, "trace_events", None)
    if buf is None:
        buf = []
        _TRACE_STATE.trace_events = buf
    return buf


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
    _TRACE_STATE.call_trace = []


def _get_call_trace() -> list[dict]:
    return list(_call_trace_buffer())


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
    _trace_events_buffer().append({
        "type": event_type,
        **_trace_value(payload),
    })


def _reset_trace_events() -> None:
    _TRACE_STATE.trace_events = []


def _get_trace_events() -> list[dict]:
    return list(_trace_events_buffer())


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
                _call_trace_buffer().append({
                    "label": label, "system": _trace_text(system), "user": _trace_text(user),
                    "response": "", "error": f"retry_failed: {exc2}",
                    "system_chars": len(system or ""), "user_chars": len(user or ""), "response_chars": 0,
                })
            raise exc2 from exc
    except Exception as exc:
        if _trace_calls_enabled():
            _call_trace_buffer().append({
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
        _call_trace_buffer().append({
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


STRICT_FINAL_LINE_DATASETS = {
    "barexam",
    "housing",
    "casehold",
    "legalbench_scalr",
    "mas_legal_bench",
    "legal_link_eu",
    "medqa",
}


def _requires_strict_answer_line(config: EvalConfig) -> bool:
    return config.dataset in STRICT_FINAL_LINE_DATASETS


def _extract_answer(text: str, config: EvalConfig) -> str | None:
    """Extract answer using the right extractor for the dataset."""
    # Strip closed <think>...</think> blocks before extraction (Qwen3, etc.)
    text = _strip_think_tags(text)
    if is_beir_dataset(config.dataset):
        return text
    if config.dataset == "housing":
        return extract_answer_yn(text)
    if config.dataset == "casehold":
        return extract_answer_mc5(text)
    if config.dataset == "legalbench_scalr":
        return extract_answer_mc5(text)
    if config.dataset == "musique":
        # Short-answer span — extract the post-Answer span, EM/F1 scored downstream
        return extract_answer_musique(text)
    if config.dataset in ("legal_rag", "legal_rag_bench", "australian"):
        return text  # open-ended: return full text, scored by LLM judge
    return extract_answer_mc(text)


def _mc_choice_letters(dataset: str) -> str:
    """Return displayed MC labels for dataset-specific detail logging."""
    return "ABCDE" if dataset in ("casehold", "legalbench_scalr") else "ABCD"


def _record_choices(row: pd.Series, dataset: str) -> dict[str, str]:
    return {
        letter: str(row.get(f"choice_{letter.lower()}", ""))[:200]
        for letter in _mc_choice_letters(dataset)
    }


def _choice_texts(row: pd.Series, config: EvalConfig) -> dict[str, str]:
    choices: dict[str, str] = {}
    for letter in _mc_choice_letters(config.dataset):
        col = f"choice_{letter.lower()}"
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            choices[letter] = str(row[col]).strip()
    return choices


def _gold_choice_text(row: pd.Series, gold: str) -> str:
    if not gold:
        return ""
    return str(row.get(f"choice_{gold.lower()}", ""))[:500]


def _gold_reference_text(row: pd.Series, config: EvalConfig) -> str:
    """Return the text to inject for oracle/reference-passage controls."""
    gold = str(row.get("gold_passage", "")).strip()
    if gold and gold.lower() != "nan":
        return gold
    if config.dataset in ("casehold", "legalbench_scalr"):
        gold = _gold_choice_text(row, str(row.get("answer", "")).strip()).strip()
        if gold:
            return gold
    gold_ids = _gold_ids(row)
    if gold_ids:
        try:
            docs = _get_documents_by_idx_for_replay(
                _collection_for_config(config),
                gold_ids,
                embedding_model=os.getenv("EVAL_EMBEDDING_MODEL", "").strip() or None,
            )
            texts = [str(doc.page_content).strip() for doc in docs if str(doc.page_content).strip()]
            if texts:
                return "\n\n".join(texts)
        except Exception:
            return ""
    return ""


def _stable_holding_id(text: str, prefix: str = "casehold") -> str:
    normalized = " ".join(str(text or "").split())
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _lexical_overlap_score(query: str, text: str) -> float:
    query_tokens = set(re.findall(r"[a-z0-9]+", str(query).lower()))
    text_tokens = set(re.findall(r"[a-z0-9]+", str(text).lower()))
    if not query_tokens or not text_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / max(1, len(query_tokens | text_tokens))


def _score_option_table_choices(
    query: str,
    choices: dict[str, str],
    *,
    source: str = "casehold_option",
) -> list[dict]:
    """Score displayed CaseHOLD options directly, without querying Chroma.

    CaseHOLD's displayed answer options are themselves the candidate holdings.
    Using them directly isolates answer-option conversion and avoids a brittle
    extra candidate-conditioned embedding query for every option.
    """
    letters = list(choices)
    texts = [choices[letter] for letter in letters]
    score_source = "lexical_overlap"
    scores = [_lexical_overlap_score(query, text) for text in texts]

    if texts and not _env_truthy("DISABLE_CROSS_ENCODER"):
        try:
            from rag_utils import get_cross_encoder

            predicted_scores = get_cross_encoder().predict([(query, text) for text in texts])
            scores = [float(score) for score in predicted_scores]
            score_source = "cross_encoder"
        except Exception as exc:
            _record_trace_event(
                "option_table_score_fallback",
                reason=str(exc),
                fallback=score_source,
            )

    rows: list[dict] = []
    for letter, text, score in zip(letters, texts, scores):
        rows.append({
            "candidate": letter,
            "holding": text,
            "score": float(score),
            "score_source": score_source,
            "idx": _stable_holding_id(text),
            "snippet": _preview_text(text, limit=360),
            "source": source,
        })
    return rows


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
    if is_beir_dataset(config.dataset):
        return str(row["question"])

    if config.dataset in ("legal_rag", "legal_rag_bench", "australian", "musique"):
        return str(row["question"])

    if config.dataset == "housing":
        state = str(row.get("state", ""))
        return f"Regarding {state} housing law:\n\n{row['question']}"

    if config.dataset in ("casehold", "legalbench_scalr"):
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

    if config.dataset == "mas_legal_bench":
        choices = []
        for letter in ["A", "B", "C", "D"]:
            col = f"choice_{letter.lower()}"
            if col in row and pd.notna(row[col]) and str(row[col]).strip():
                choices.append(f"- {row[col]}")
        parts = [
            "The following question asks about GDPR/data-protection enforcement and legal reasoning.",
            f"## Question\n{row['question']}",
        ]
        if choices:
            parts.append("## Candidate Answer Framing\n" + "\n".join(choices))
        return "\n\n".join(parts)

    if config.dataset == "legal_link_eu":
        choices = []
        for letter in ["A", "B", "C", "D"]:
            col = f"choice_{letter.lower()}"
            if col in row and pd.notna(row[col]) and str(row[col]).strip():
                choices.append(f"- {row[col]}")
        relation = str(row.get("relation_type", "") or "").replace("_", " ")
        parts = [
            "The following question asks about the legal relationship between EU legal acts.",
            f"Relation type: {relation}" if relation else "",
            f"## Question\n{row['question']}",
        ]
        if choices:
            parts.append("## Candidate Answer Framing\n" + "\n".join(choices))
        return "\n\n".join(part for part in parts if part)

    if config.dataset == "medqa":
        choices = []
        for letter in ["A", "B", "C", "D"]:
            col = f"choice_{letter.lower()}"
            if col in row and pd.notna(row[col]) and str(row[col]).strip():
                choices.append(f"- {row[col]}")
        parts = [
            "The following question asks about medical diagnosis, mechanism, management, or clinical reasoning.",
            f"## Question\n{row['question']}",
        ]
        if choices:
            parts.append("## Candidate Answer Framing\n" + "\n".join(choices))
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
    artifact_patterns = [
        r"(?im)^\s*(?:\*\*)?(?:final\s+)?answer(?:\*\*)?\s*:",
        r"(?i)\b(?:answer|option|choice)\s+(?:is|must be|would be)\s*\(?[A-E]\)?\b",
        # Keep this stricter than the answer/option patterns above so ordinary
        # legal prose such as "it is a fair representation" is not treated as
        # an option-letter leak.
        r"(?i)\b(?:it'?s|it is)\s*(?:\([A-E]\)|[A-E](?=\s*(?:$|[.,;:!?])))",
        r"(?i)\bself-correction\b",
        r"(?im)^\s*wait[,:\s]",
    ]
    return any(re.search(pattern, text) for pattern in artifact_patterns)


def _has_explicit_answer_marker(text: str) -> bool:
    """Whether a discrete-task final answer used the required Answer: marker."""
    if not text:
        return False
    return bool(re.search(r"(?im)^\s*(?:\*\*)?(?:final\s+)?answer(?:\*\*)?\s*:", text))


def _has_required_final_answer_line(text: str, predicted: str | None, config: EvalConfig) -> bool:
    """Whether the last non-empty line is exactly the required final answer."""
    final_line_prediction = _extract_required_final_line_prediction(text, config)
    if final_line_prediction is not None:
        return _required_answer_line_from_prediction(final_line_prediction, config) == (
            _required_answer_line_from_prediction(predicted, config)
        )
    target_line = _required_answer_line_from_prediction(predicted, config)
    if not target_line:
        return False
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    if not lines:
        return False
    return lines[-1] == target_line


def _extract_predicted_answer(text: str, config: EvalConfig) -> str | None:
    """Extract the answer from a leading PREDICTED line, if present."""
    if not text:
        return None
    match = re.search(r"(?im)^\s*PREDICTED\s*:\s*(.+?)\s*$", text)
    if not match:
        return _extract_answer(text, config)
    return _extract_answer(match.group(1), config)


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


_HYRE_CACHE: dict[str, dict] | None = None
_HYRE_CACHE_PATH: str | None = None
_RETRIEVAL_CACHE: dict[tuple[str, str, str, str, str], dict] | None = None
_RETRIEVAL_CACHE_PATH: str | None = None


def _row_label(row: pd.Series, config: EvalConfig, fallback_i: int | None = None) -> str:
    """Stable per-row label shared by logs and optional HyRE replay caches."""
    i = row.get("idx", fallback_i if fallback_i is not None else "")
    if is_beir_dataset(config.dataset):
        i_str = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(i)).strip("_")
        return f"{config.dataset}_{i_str or fallback_i or 'unknown'}"
    if config.dataset == "housing":
        return f"hqa_{row.get('state', 'unknown')}_{i}"
    if config.dataset == "casehold":
        return f"ch_{i}"
    if config.dataset == "legal_rag":
        return f"lrq_{i}"
    if config.dataset == "legal_rag_bench":
        return f"lrb_{i}"
    if config.dataset == "mas_legal_bench":
        i_str = str(i)
        return i_str if i_str.startswith("maslb_") else f"maslb_{i_str}"
    if config.dataset == "legal_link_eu":
        i_str = str(i)
        return i_str if i_str.startswith("complex_legallink_") else f"lle_{i_str}"
    if config.dataset == "australian":
        return f"aus_{row.get('jurisdiction', 'unknown')}_{i}"
    if config.dataset == "musique":
        return f"mq_{i}"
    if config.dataset == "medqa":
        i_str = str(i)
        return i_str if i_str.startswith("medqa_") else f"medqa_{i_str}"
    return f"qa_{row.get('subject', 'unknown')}_{i}"


def _load_hyre_cache(path: str) -> dict[str, dict]:
    """Load a JSONL cache keyed by row label for deterministic HyRE replay."""
    global _HYRE_CACHE, _HYRE_CACHE_PATH
    if not path:
        return {}
    if _HYRE_CACHE is not None and _HYRE_CACHE_PATH == path:
        return _HYRE_CACHE

    cache: dict[str, dict] = {}
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            label = row.get("label")
            if not label:
                raise ValueError(f"{path}:{line_no}: missing label")
            cache[str(label)] = row
    _HYRE_CACHE = cache
    _HYRE_CACHE_PATH = path
    return cache


def _hyre_cache_entry(row: pd.Series, config: EvalConfig) -> dict | None:
    path = (config.hyre_cache_path or os.getenv("HYRE_CACHE_PATH", "")).strip()
    if not path:
        return None
    entry = _load_hyre_cache(path).get(_row_label(row, config))
    if entry is not None:
        _validate_hyre_cache_entry(entry, config, path)
    return entry


def _expected_hyre_source_modes(config: EvalConfig) -> set[str]:
    """Generation-cache modes that may legally feed the current answer mode."""
    mode = config.mode
    if mode == "rag_hyde":
        return {"rag_hyde"}
    if mode in {"snap_hyre", "rag_snap_hyde_2call"}:
        # `rag_snap_hyde_2call` is the historical alias for the current
        # two-call Snap-HyRE structure.
        return {"snap_hyre", "rag_snap_hyde_2call"}
    if mode in {"rag_hyde_exemplar", "snap_hyre_exemplar"}:
        return {mode}
    return {mode}


def _allow_cross_provider_generation_cache() -> bool:
    return os.getenv("EVAL_ALLOW_CROSS_PROVIDER_GENERATION_CACHE", "").strip().lower() in {
        "1", "true", "yes", "on"
    }


def _validate_hyre_cache_entry(entry: dict, config: EvalConfig, path: str) -> None:
    """Fail closed when a generation cache belongs to a different method/model."""
    violations: list[str] = []
    source_mode = str(entry.get("source_mode") or entry.get("mode") or "").strip()
    expected_modes = _expected_hyre_source_modes(config)
    if not source_mode:
        violations.append("missing source_mode/mode")
    elif source_mode not in expected_modes:
        violations.append(f"source_mode={source_mode!r} not in {sorted(expected_modes)!r}")

    dataset = str(entry.get("dataset") or "").strip()
    if dataset and dataset != config.dataset:
        violations.append(f"dataset={dataset!r} != expected {config.dataset!r}")

    provider = str(entry.get("provider") or "").strip()
    if provider and provider != config.provider and not _allow_cross_provider_generation_cache():
        violations.append(
            f"provider={provider!r} != expected {config.provider!r}; "
            "set EVAL_ALLOW_CROSS_PROVIDER_GENERATION_CACHE=1 only for an intentional reuse"
        )

    if config.mode in {"rag_hyde_exemplar", "snap_hyre_exemplar"}:
        expected_variant = _passage_style_signal_variant(config)
        entry_variant = _entry_passage_style_variant(entry, source_mode)
        if entry_variant != expected_variant:
            violations.append(
                f"passage_style_signal_variant={entry_variant!r} != expected {expected_variant!r}"
            )

    if violations:
        label = str(entry.get("label") or "")
        raise RuntimeError(
            "generation cache provenance mismatch "
            f"path={path} label={label}: " + "; ".join(violations)
        )


def _json_key(value) -> str:
    """Stable JSON key for retrieval-cache metadata."""
    return json.dumps(value or {}, sort_keys=True, separators=(",", ":"))


def _hash_texts(values: list[str]) -> str:
    """Stable short hash used to bind retrieval caches to the exact query text."""
    h = hashlib.sha256()
    for value in values:
        h.update(str(value).encode("utf-8", errors="ignore"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def _row_idx_for_cache(row: pd.Series) -> str:
    value = row.get("idx", "")
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def _load_retrieval_cache(path: str) -> dict[tuple[str, str, str, str, str], dict]:
    """Load deterministic retrieval-id cache keyed by row idx and retrieval settings."""
    global _RETRIEVAL_CACHE, _RETRIEVAL_CACHE_PATH
    if not path:
        return {}
    if _RETRIEVAL_CACHE is not None and _RETRIEVAL_CACHE_PATH == path:
        return _RETRIEVAL_CACHE

    cache: dict[tuple[str, str, str, str, str], dict] = {}
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            entry = json.loads(line)
            raw_idx = entry.get("idx")
            if raw_idx is None or raw_idx == "":
                raw_idx = entry.get("row_idx", "")
            idx = str(raw_idx)
            label_prefix = str(entry.get("label_prefix") or "")
            collection = str(entry.get("collection") or "")
            embedding_model = str(entry.get("embedding_model") or "")
            where_key = _json_key(entry.get("where") or {})
            retrieved_ids = entry.get("retrieved_ids") or []
            if not idx:
                raise ValueError(f"{path}:{line_no}: missing idx")
            if not label_prefix:
                raise ValueError(f"{path}:{line_no}: missing label_prefix")
            if not collection:
                raise ValueError(f"{path}:{line_no}: missing collection")
            if not isinstance(retrieved_ids, list):
                raise ValueError(f"{path}:{line_no}: retrieved_ids must be a list")
            cache[(idx, label_prefix, collection, embedding_model, where_key)] = entry
    _RETRIEVAL_CACHE = cache
    _RETRIEVAL_CACHE_PATH = path
    return cache


def _retrieval_cache_entry(
    row: pd.Series,
    label_prefix: str,
    collection: str,
    where: dict | None,
    embedding_model: str | None,
) -> dict | None:
    path = os.getenv("RETRIEVAL_CACHE_PATH", "").strip()
    if not path:
        return None
    cache = _load_retrieval_cache(path)
    key = (
        _row_idx_for_cache(row),
        label_prefix,
        collection,
        embedding_model or "",
        _json_key(where or {}),
    )
    if key not in cache:
        raise KeyError(
            "retrieval cache miss for "
            f"idx={key[0]} label_prefix={label_prefix} collection={collection} "
            f"embedding_model={embedding_model or ''} where={where or {}} path={path}"
    )
    return cache[key]


def _load_retrieval_doc_cache(path: str) -> dict[tuple[str, str], Document]:
    """Load a collection/id -> Document cache for strict retrieval replay.

    Retrieval caches intentionally store ordered passage IDs. This optional
    cache stores the corresponding Chroma text/metadata snapshot so answer
    replay can hydrate cached IDs without opening a large local Chroma
    collection.
    """
    global _RETRIEVAL_DOC_CACHE, _RETRIEVAL_DOC_CACHE_PATH
    if not path:
        return {}
    if _RETRIEVAL_DOC_CACHE is not None and _RETRIEVAL_DOC_CACHE_PATH == path:
        return _RETRIEVAL_DOC_CACHE

    cache: dict[tuple[str, str], Document] = {}
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            entry = json.loads(line)
            collection = str(entry.get("collection") or "")
            idx = str(entry.get("idx") or "")
            text = str(entry.get("text") or "")
            metadata = dict(entry.get("metadata") or {})
            if not collection:
                raise ValueError(f"{path}:{line_no}: missing collection")
            if not idx:
                raise ValueError(f"{path}:{line_no}: missing idx")
            metadata.setdefault("idx", idx)
            cache[(collection, idx)] = Document(page_content=text, metadata=metadata)
    _RETRIEVAL_DOC_CACHE = cache
    _RETRIEVAL_DOC_CACHE_PATH = path
    return cache


def _documents_from_doc_cache(
    collection: str,
    retrieved_ids: list[str],
) -> list[Document] | None:
    path = os.getenv("RETRIEVAL_DOC_CACHE_PATH", "").strip()
    if not path:
        return None
    cache = _load_retrieval_doc_cache(path)
    missing = [idx for idx in retrieved_ids if (collection, idx) not in cache]
    if missing:
        if _no_silent_fallback_enabled() or os.getenv("RETRIEVAL_DOC_CACHE_STRICT", "").strip().lower() in {"1", "true", "yes", "on"}:
            raise ValueError(
                f"retrieval document cache miss in {path} for collection={collection}: {missing[:5]}"
            )
        return None
    docs = []
    for idx in retrieved_ids:
        doc = cache[(collection, idx)]
        docs.append(Document(page_content=doc.page_content, metadata=dict(doc.metadata or {})))
    return docs


def _get_documents_by_idx_for_replay(
    collection: str,
    idxs: list[str],
    embedding_model: str | None = None,
    return_cache_hit: bool = False,
) -> list[Document] | tuple[list[Document], bool]:
    cached = _documents_from_doc_cache(collection, [str(idx) for idx in idxs])
    if cached is not None:
        return (cached, True) if return_cache_hit else cached
    docs = get_documents_by_idx(collection, idxs, embedding_model=embedding_model)
    return (docs, False) if return_cache_hit else docs


def _documents_from_retrieval_cache(
    row: pd.Series,
    label_prefix: str,
    collection: str,
    where: dict | None,
    embedding_model: str | None,
    k: int,
    queries: List[str] | None = None,
):
    entry = _retrieval_cache_entry(row, label_prefix, collection, where, embedding_model)
    if entry is None:
        return None
    if queries is not None and entry.get("query_hash"):
        expected_hash = _hash_texts([str(q) for q in queries])
        if str(entry.get("query_hash")) != expected_hash:
            raise ValueError(
                f"retrieval cache query_hash mismatch for idx={_row_idx_for_cache(row)} "
                f"label_prefix={label_prefix}: cache={entry.get('query_hash')} "
                f"current={expected_hash}"
            )
    retrieved_ids = [str(idx) for idx in entry.get("retrieved_ids", [])]
    if len(retrieved_ids) < k:
        raise ValueError(
            f"retrieval cache row idx={_row_idx_for_cache(row)} label_prefix={label_prefix} "
            f"has {len(retrieved_ids)} ids, need k={k}"
        )
    requested_ids = retrieved_ids[:k]
    docs, doc_cache_hit = _get_documents_by_idx_for_replay(
        collection,
        requested_ids,
        embedding_model=embedding_model,
        return_cache_hit=True,
    )
    got_ids = {str(doc.metadata.get("idx", "")) for doc in docs}
    missing = [idx for idx in requested_ids if idx not in got_ids]
    if missing:
        raise ValueError(
            f"retrieval cache row idx={_row_idx_for_cache(row)} label_prefix={label_prefix} "
            f"references ids not found in {collection}: {missing[:5]}"
        )
    score_by_id = {
        str(idx): score
        for idx, score in zip(retrieved_ids, entry.get("scores") or [])
    }
    for doc in docs:
        idx = str(doc.metadata.get("idx", ""))
        if "cross_encoder_score" not in doc.metadata:
            doc.metadata["cross_encoder_score"] = float(score_by_id.get(idx, 0.0) or 0.0)
        doc.metadata["retrieval_doc_cache_hit"] = doc_cache_hit
    entry["_doc_cache_hit"] = doc_cache_hit
    return docs, entry


_PASSAGE_STYLE_VARIANT_ALIASES = {
    "": "single",
    "default": "single",
    "realpassage": "single",
    "single": "single",
    "one": "single",
    "multi": "multi3",
    "multi3": "multi3",
    "parallel": "parallel3",
    "parallel3": "parallel3",
    "three": "multi3",
    "3": "multi3",
}


def _passage_style_signal_variant(config: EvalConfig) -> str:
    raw = str(
        getattr(config, "passage_style_variant", "")
        or os.getenv("EVAL_PASSAGE_STYLE_VARIANT", "")
        or "single"
    ).strip().lower()
    variant = _PASSAGE_STYLE_VARIANT_ALIASES.get(raw)
    if not variant:
        raise ValueError(
            f"Unsupported passage-style exemplar variant {raw!r}; expected single, multi3, or parallel3"
        )
    return variant


def _passage_style_signal_ids(config: EvalConfig) -> list[str]:
    variant = _passage_style_signal_variant(config)
    if variant not in {"multi3", "parallel3"}:
        ids_by_dataset = {
            "barexam": ["mbe_4"],
            "housing": ["single_housing_realpassage"],
            "legal_link_eu": ["single_legal_link_eu_realpassage"],
            "mas_legal_bench": ["single_mas_legal_bench_realpassage"],
        }
        return ids_by_dataset.get(config.dataset, [])
    ids_by_dataset = {
        "barexam": ["mbe_4", "mbe_20", "mbe_308"],
        "housing": ["1508532", "1038490", "1727814"],
    }
    return ids_by_dataset.get(config.dataset, [])


def _passage_style_signal_metadata(config: EvalConfig) -> dict:
    return {
        "passage_style_signal_variant": _passage_style_signal_variant(config),
        "passage_style_signal_ids": _passage_style_signal_ids(config),
    }


def _entry_passage_style_variant(entry: dict, source_mode: str) -> str:
    variant = str(entry.get("passage_style_signal_variant") or "").strip().lower()
    if variant:
        return _PASSAGE_STYLE_VARIANT_ALIASES.get(variant, variant)
    if source_mode in {"rag_hyde_exemplar", "snap_hyre_exemplar"}:
        return "single"
    return ""


def _orthogonal_passage_style_signals(config: EvalConfig) -> list[dict[str, Any]]:
    """Three independent exemplar prompts for retrieval-only parallel probes."""
    banks: dict[str, list[dict[str, Any]]] = {
        "barexam": [
            {
                "key": "torts_res_ipsa",
                "ids": ["mbe_4"],
                "signal": (
                    "A useful BarExamQA retrieval passage names the doctrine first, "
                    "then states the operative elements in neutral black-letter form. "
                    "It does not restate the fact pattern or argue for an answer choice.\n\n"
                    "Corpus passage excerpt: The res ipsa loquitur doctrine enables "
                    "a jury presented only with circumstantial evidence to infer "
                    "negligence from the fact that an event happened. The criteria "
                    "include an event that ordinarily does not occur without "
                    "negligence, an agency or instrumentality within the defendant's "
                    "exclusive control, and no voluntary action or contribution by "
                    "the plaintiff."
                ),
            },
            {
                "key": "criminal_search_consent",
                "ids": ["mbe_20"],
                "signal": (
                    "A useful BarExamQA retrieval passage states the constitutional "
                    "rule, exception, and required elements in reference style. It "
                    "does not summarize the exam facts or select an answer.\n\n"
                    "Corpus passage excerpt: The Fourth Amendment bars unreasonable "
                    "searches and seizures, and a warrantless search is per se "
                    "unreasonable unless it falls within a specifically established "
                    "exception. One exception is valid consent, which must be knowing "
                    "and voluntary and given by a person with authority to consent."
                ),
            },
            {
                "key": "equitable_specific_performance",
                "ids": ["mbe_308"],
                "signal": (
                    "A useful BarExamQA retrieval passage identifies the remedy or "
                    "doctrinal category, then states the legal standard and its "
                    "usual application. It avoids answer labels and advocacy.\n\n"
                    "Corpus passage excerpt: Specific performance is an equitable "
                    "remedy ordered when the legal remedy, usually money damages, is "
                    "inadequate or impracticable. When land is the subject matter of "
                    "the agreement, the legal remedy is generally treated as "
                    "inadequate because each parcel of land is unique."
                ),
            },
        ],
        "housing": [
            {
                "key": "eviction_appeal_stay",
                "ids": ["1508532"],
                "signal": (
                    "A useful HousingQA retrieval passage preserves the state, the "
                    "procedural term, and the legal consequence in statutory style. "
                    "It does not guess a yes/no answer.\n\n"
                    "Corpus passage excerpt: In Idaho eviction proceedings, an "
                    "appeal taken by the defendant does not stay proceedings upon "
                    "the judgment unless the court so directs."
                ),
            },
            {
                "key": "essential_services_remedies",
                "ids": ["1038490"],
                "signal": (
                    "A useful HousingQA retrieval passage names the landlord duty, "
                    "tenant notice requirement, and available statutory remedies. "
                    "It preserves state-specific terms and avoids generic national "
                    "housing-law phrasing.\n\n"
                    "Corpus passage excerpt: In Montana, if the landlord "
                    "purposefully or negligently fails to supply heat, running "
                    "water, hot water, electricity, gas, or other essential "
                    "services, the tenant may give written notice specifying the "
                    "breach and may procure reasonable services and deduct their "
                    "actual and reasonable cost from rent, recover damages based on "
                    "diminished rental value, or procure reasonable substitute "
                    "housing during the noncompliance period."
                ),
            },
            {
                "key": "foreclosure_tenancy_notice",
                "ids": ["1727814"],
                "signal": (
                    "A useful HousingQA retrieval passage captures foreclosure, "
                    "tenancy type, notice period, and statutory exceptions in the "
                    "same form as a state code excerpt. It should preserve the "
                    "jurisdiction named in the current question.\n\n"
                    "Corpus passage excerpt: In California, a tenant or subtenant "
                    "in possession of a rental housing unit under a month-to-month "
                    "lease or periodic tenancy when the property is sold in "
                    "foreclosure must receive 90 days' written notice to quit "
                    "before removal. A tenant holding under a fixed-term "
                    "residential lease entered before the foreclosure sale may "
                    "remain until the end of the lease term unless a statutory "
                    "exception applies."
                ),
            },
        ],
        "legal_link_eu": [
            {
                "key": "extends_application",
                "ids": [],
                "signal": (
                    "A useful Legal-Link-EU retrieval passage preserves source and "
                    "target act identifiers, relation words, article numbers, and "
                    "institution names. It should read like EU legal context, not "
                    "like an answer explanation.\n\n"
                    "Corpus passage excerpt: A source act can extend the "
                    "application of a target Commission decision by identifying the "
                    "advisory committee, its membership or quorum rule, and the "
                    "legal instrument whose procedure applies to the new sector."
                ),
            },
            {
                "key": "rendered_obsolete",
                "ids": [],
                "signal": (
                    "A useful Legal-Link-EU retrieval passage states whether a "
                    "later notice, codification, repeal, or omission from the "
                    "active acquis changes the status of an earlier instrument. It "
                    "keeps CELEX-style identifiers and dates when present.\n\n"
                    "Corpus passage excerpt: A later EU notice may render an "
                    "earlier decision obsolete by removing it from the active "
                    "Community acquis and replacing the operative publication, "
                    "reporting, or notification framework."
                ),
            },
            {
                "key": "annex_correction_amendment",
                "ids": [],
                "signal": (
                    "A useful Legal-Link-EU retrieval passage names the amending or "
                    "correcting act, the affected annex or article, and the exact "
                    "legal relationship such as replaces, corrects, repeals, or "
                    "extends validity.\n\n"
                    "Corpus passage excerpt: An implementing regulation may amend "
                    "annexes to an earlier regulation by replacing tables, product "
                    "codes, quota periods, or eligibility criteria while leaving the "
                    "underlying source act in force."
                ),
            },
        ],
        "mas_legal_bench": [
            {
                "key": "gdpr_security_framework",
                "ids": [],
                "signal": (
                    "A useful MASLegalBench retrieval passage identifies the legal "
                    "basis, controller or processor duty, and statutory factors "
                    "used by the regulator. It should resemble an enforcement notice "
                    "or legal framework paragraph.\n\n"
                    "Corpus passage excerpt: Article 32 UK GDPR requires security "
                    "measures appropriate to the risk, taking account of the state "
                    "of the art, implementation costs, the nature, scope, context, "
                    "and purposes of processing, and the risk to rights and freedoms."
                ),
            },
            {
                "key": "breach_notification",
                "ids": [],
                "signal": (
                    "A useful MASLegalBench retrieval passage preserves the "
                    "regulated actor, the breach event, the reporting timeline, and "
                    "the authority's finding without choosing an option.\n\n"
                    "Corpus passage excerpt: In the case of a personal data breach, "
                    "the controller must notify the supervisory authority without "
                    "undue delay and, where feasible, within 72 hours after becoming "
                    "aware of the breach, unless the breach is unlikely to result in "
                    "risk to rights and freedoms."
                ),
            },
            {
                "key": "penalty_mitigation",
                "ids": [],
                "signal": (
                    "A useful MASLegalBench retrieval passage states the "
                    "contravention, aggravating or mitigating factors, and penalty "
                    "assessment considerations in enforcement-notice style.\n\n"
                    "Corpus passage excerpt: When determining an administrative "
                    "penalty, the Commissioner may consider the nature, gravity, and "
                    "duration of the infringement, categories of personal data, "
                    "cooperation with the investigation, prior compliance history, "
                    "and measures taken to reduce harm."
                ),
            },
        ],
    }
    return list(banks.get(config.dataset, []))


def _hyre_passage_style_signal(config: EvalConfig) -> str:
    """Dataset-specific style signal for probe-only HyRE exemplar modes.

    The examples are real corpus-style passages with the associated question,
    answer, and document id removed. They are meant to align passage shape to
    each corpus without providing row-specific evidence for the current
    question.
    """
    variant = _passage_style_signal_variant(config)
    if variant == "multi3":
        multi3_by_dataset = {
            "barexam": (
                "A useful BarExamQA retrieval passage names the doctrine first, "
                "then states the operative element, exception, or admissibility "
                "rule in neutral black-letter form. It does not restate the fact "
                "pattern or argue for an answer choice.\n\n"
                "Corpus passage excerpt 1: The res ipsa loquitur doctrine enables "
                "a jury presented only with circumstantial evidence to infer "
                "negligence from the fact that an event happened. The criteria "
                "include an event that ordinarily does not occur without "
                "negligence, an agency or instrumentality within the defendant's "
                "exclusive control, and no voluntary action or contribution by "
                "the plaintiff.\n\n"
                "Corpus passage excerpt 2: The Fourth Amendment bars unreasonable "
                "searches and seizures, and a warrantless search is per se "
                "unreasonable unless it falls within a specifically established "
                "exception. One exception is valid consent, which must be knowing "
                "and voluntary and given by a person with authority to consent.\n\n"
                "Corpus passage excerpt 3: Specific performance is an equitable "
                "remedy ordered when the legal remedy, usually money damages, is "
                "inadequate or impracticable. When land is the subject matter of "
                "the agreement, the legal remedy is generally treated as "
                "inadequate because each parcel of land is unique."
            ),
            "housing": (
                "A useful HousingQA retrieval passage sounds like a state statutory "
                "definition or landlord-tenant procedure section. It should preserve "
                "the state or territory named in the question, preserve legal terms "
                "from the question, name the actor and authority when relevant, and "
                "avoid guessing a yes/no answer.\n\n"
                "Corpus passage excerpt 1: In Idaho eviction proceedings, an appeal "
                "taken by the defendant does not stay proceedings upon the judgment "
                "unless the court so directs.\n\n"
                "Corpus passage excerpt 2: In Montana, if the landlord purposefully "
                "or negligently fails to supply heat, running water, hot water, "
                "electricity, gas, or other essential services, the tenant may give "
                "written notice specifying the breach and may procure reasonable "
                "services and deduct their actual and reasonable cost from rent, "
                "recover damages based on diminished rental value, or procure "
                "reasonable substitute housing during the noncompliance period.\n\n"
                "Corpus passage excerpt 3: In California, a tenant or subtenant in "
                "possession of a rental housing unit under a month-to-month lease "
                "or periodic tenancy when the property is sold in foreclosure must "
                "receive 90 days' written notice to quit before removal. A tenant "
                "holding under a fixed-term residential lease entered before the "
                "foreclosure sale may remain until the end of the lease term unless "
                "a statutory exception applies."
            ),
        }
        if config.dataset in multi3_by_dataset:
            return multi3_by_dataset[config.dataset]

    style_by_dataset = {
        "barexam": (
            "A useful BarExamQA retrieval passage names the doctrine first, then "
            "states the operative element, exception, or admissibility rule in "
            "neutral black-letter form. It does not restate the fact pattern or "
            "argue for an answer choice.\n\n"
            "Corpus passage excerpt: The res ipsa loquitur doctrine enables a "
            "jury presented only with circumstantial evidence to infer negligence "
            "simply from the fact that an event happened. The criteria for "
            "applying res ipsa loquitur include that the event must be of a kind "
            "which ordinarily does not occur in the absence of negligence, must "
            "be caused by an agency or instrumentality within the exclusive "
            "control of the defendant, and must not be due to any voluntary "
            "action or contribution by the plaintiff."
        ),
        "housing": (
            "A useful HousingQA retrieval passage sounds like a state statutory "
            "definition or landlord-tenant procedure section. It should preserve "
            "the state or territory named in the question, preserve "
            "legal terms from the question, name the actor and authority when "
            "relevant, and avoid guessing a yes/no answer.\n\n"
            "Corpus passage excerpt: In an eviction action, if the court finds "
            "that the plaintiff is entitled to possession, the court shall "
            "immediately enter an order for judgment for the restitution of the "
            "premises to the plaintiff. At the time of ordering judgment for the "
            "restitution of premises, the court shall immediately order that a "
            "writ of restitution be issued, and the writ may be delivered to the "
            "sheriff for execution."
        ),
        "legal_link_eu": (
            "A useful Legal-Link-EU retrieval passage resembles EU legal text: it "
            "names the source act, target act, article or annex, and legal "
            "relationship such as amends, repeals, corrects, extends application, "
            "or extends validity. It should include dates, entities, and thresholds "
            "when they are central.\n\n"
            "Corpus passage excerpt: Council Regulation (EU) No 833/2014 "
            "concerning restrictive measures in view of Russia's actions "
            "destabilising the situation in Ukraine applies restrictions on "
            "certain dual-use goods and technology, related services, and certain "
            "technologies for the oil industry. The measures are kept under "
            "review and may be suspended, withdrawn, or supplemented in light of "
            "developments on the ground."
        ),
        "mas_legal_bench": (
            "A useful MASLegalBench retrieval passage resembles a GDPR enforcement "
            "notice or legal basis summary. It should identify the processing "
            "actor, data category, legal basis, obligation, sanction factor, or "
            "authority finding without predicting the answer label.\n\n"
            "Corpus passage excerpt: Article 32 UK GDPR provides that, taking "
            "into account the state of the art, the costs of implementation, and "
            "the nature, scope, context and purposes of processing, as well as "
            "the risk of varying likelihood and severity for the rights and "
            "freedoms of natural persons, the controller and processor shall "
            "implement appropriate technical and organisational measures to "
            "ensure a level of security appropriate to the risk."
        ),
    }
    return style_by_dataset.get(
        config.dataset,
        (
            "A useful retrieval passage states the controlling legal rule in "
            "neutral reference style, names the key actor or legal relationship, "
            "and avoids answer labels or advocacy."
        ),
    )


def _style_signal_block(config: EvalConfig) -> str:
    return (
        "## Passage Style Signal (probe only; not evidence)\n"
        "Use this signal only to match the shape and specificity of a useful "
        "retrieval passage for this dataset. Do not copy it, do not treat it as "
        "evidence, and do not use it to answer the current question.\n\n"
        f"{_hyre_passage_style_signal(config)}\n\n"
    )


def _question_only_hyde_user(question_text: str, config: EvalConfig | None = None, use_style_signal: bool = False) -> str:
    """Structured user payload for question-only HyDE generation."""
    style_signal = _style_signal_block(config) if use_style_signal and config is not None else ""
    return (
        "## Task\n"
        "Write a passage that would appear in a legal treatise or casebook — the kind of "
        "passage a researcher would find when looking up the doctrine behind the scenario "
        "below. This is NOT a multiple-choice task; do not pick an option.\n\n"
        f"{style_signal}"
        "## Scenario (for context only)\n"
        f"{question_text}\n\n"
        "## Passage Requirements\n"
        "- 2-3 sentences, legal reference style\n"
        "- 120 words maximum; do not repeat phrases or sentences\n"
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
        "- 120 words maximum; do not repeat phrases or sentences\n"
        "- State the controlling rule, doctrine, holding, exception, or principle directly\n"
        "- Start with the doctrinal text itself — no 'Answer:', no letter labels, "
        "no '**Passage:**' header, no bold or markdown\n"
    )


def _generate_hyde(config: EvalConfig, role: str, user: str, label: str, fallback: str) -> dict:
    """Generate and sanitize a HyDE-style intermediate passage."""
    raw = _llm_call(_system_prompt(config, role), user, label=label)
    cleaned = _sanitize_intermediate_text(raw, fallback="")
    fallback_text = (fallback or "").strip()
    return {
        "text": cleaned or fallback_text,
        "raw": raw,
        "contains_answer": _contains_answer_artifact(raw),
        "used_fallback": not bool(cleaned),
    }


def _has_answer_options(row: pd.Series, config: EvalConfig) -> bool:
    """Whether the row exposes discrete answer candidates worth grounding."""
    return any(
        col in row and pd.notna(row[col]) and str(row[col]).strip()
        for col in (f"choice_{letter.lower()}" for letter in _mc_choice_letters(config.dataset))
    )


def _option_grounding_system(config: EvalConfig) -> str:
    """Dataset-aware final prompt for converting evidence into a displayed option.

    This is deliberately a final-synthesis change, not a retrieval change. The
    CaseHOLD/SCALR bottleneck we have seen is that better gold retrieval does
    not reliably convert into the correct option, so force the model to compare
    each displayed candidate against the retrieved holdings before emitting the
    answer.
    """
    base = _system_prompt(config, "rag")
    if config.dataset in ("casehold", "legalbench_scalr"):
        return (
            base
            + "\n\nOPTION-GROUNDING REQUIREMENTS:\n"
            "- Treat the five displayed holdings as candidates, not as retrieval queries.\n"
            "- Compare the citing context against each candidate holding using the retrieved holdings.\n"
            "- Prefer the candidate whose rule or fact pattern is entailed by the retrieved evidence and citation context.\n"
            "- If retrieved evidence is noisy, still decide from the displayed candidate text and cite-context fit.\n"
            "- End with exactly one final line in the form: Answer: (X)"
        )
    if config.dataset == "barexam":
        return (
            base
            + "\n\nOPTION-GROUNDING REQUIREMENTS:\n"
            "- Compare the retrieved rule against each displayed answer choice.\n"
            "- Reject distractors whose rule statement or application conflicts with the retrieved evidence.\n"
            "- End with exactly one final line in the form: Answer: (X)"
        )
    return base


def _housing_verifier_system(config: EvalConfig) -> str:
    """Conservative final prompt for HousingQA yes/no statutory entailment."""
    base = _system_prompt(config, "rag")
    if config.dataset != "housing":
        return base
    return (
        base
        + "\n\nHOUSING YES/NO VERIFICATION REQUIREMENTS:\n"
        "- Treat the retrieved statutes as the controlling evidence.\n"
        "- Answer Yes only if the retrieved statutes affirmatively authorize or require the proposition in the question.\n"
        "- Answer No if the retrieved statutes contradict it, omit a required condition, create an exception, or leave authorization uncertain.\n"
        "- Do not infer a landlord/tenant power from general housing-law background if the retrieved statutes do not support it.\n"
        "- End with exactly one final line: Answer: Yes or Answer: No"
    )


def _candidate_verifier_system(config: EvalConfig) -> str:
    """Candidate-first final prompt for holding-selection tasks."""
    base = _system_prompt(config, "rag")
    if config.dataset not in ("casehold", "legalbench_scalr"):
        return base
    return (
        base
        + "\n\nCANDIDATE VERIFICATION REQUIREMENTS:\n"
        "- The displayed holdings are the answer candidates; compare all five directly against the citing context.\n"
        "- Use retrieved passages as supporting evidence or tie-breakers, not as replacement candidates.\n"
        "- If retrieved passages are noisy or do not mention the correct candidate, choose the displayed holding whose rule and fact pattern best fit the citing context.\n"
        "- Prefer semantic fit between the citation signal and candidate holding over superficial word overlap.\n"
        "- End with exactly one final line in the form: Answer: (X)"
    )


def _option_reranker_system(config: EvalConfig) -> str:
    """Final prompt for per-candidate evidence bundles."""
    base = _system_prompt(config, "rag")
    if config.dataset != "casehold":
        return base
    return (
        base
        + "\n\nOPTION RERANKING REQUIREMENTS:\n"
        "- Each candidate holding has its own retrieved evidence bundle.\n"
        "- Compare the citing context to each displayed holding first, then use that candidate's evidence bundle to verify fit.\n"
        "- Do not choose a candidate merely because its bundle is longer or has more word overlap.\n"
        "- Prefer the candidate whose rule, legal relationship, and facts best explain the citing context.\n"
        "- End with exactly one final line in the form: Answer: (X)"
    )


def _option_table_selector_system(config: EvalConfig) -> str:
    """Bounded final prompt for compact CaseHOLD option evidence tables."""
    base = _system_prompt(config, "rag")
    if config.dataset != "casehold":
        return base
    return (
        base
        + "\n\nOPTION TABLE SELECTION REQUIREMENTS:\n"
        "- The table lists every displayed holding, its best candidate-conditioned retrieval score, and a short evidence snippet.\n"
        "- Cross-encoder scores are only retrieval features; do not pick the maximum score unless the holding also fits the cited context.\n"
        "- Compare the rule, legal relationship, procedural posture, and fact pattern of all five holdings.\n"
        "- Use the score and snippet to break ties, but reject a high-scoring option if it is only lexical overlap.\n"
        "- End with exactly one final line in the form: Answer: (X)"
    )


def _final_answer_contract(config: EvalConfig) -> str:
    """User-level output contract for strict final-answer postchecks."""
    if config.dataset == "housing":
        return (
            "## Required Output\n"
            "End your response with exactly one final line, either:\n"
            "Answer: Yes\n"
            "Answer: No\n"
            "Do not put any text after that final Answer line."
        )
    if config.dataset in {"barexam", "casehold", "legalbench_scalr", "mas_legal_bench", "legal_link_eu", "medqa"}:
        choices = ", ".join(f"Answer: ({letter})" for letter in _mc_choice_letters(config.dataset))
        return (
            "## Required Output\n"
            "End your response with exactly one final line in this form: Answer: (X)\n"
            f"Valid final lines are: {choices}\n"
            "Do not put any text after that final Answer line."
        )
    return ""


def _required_answer_line_from_prediction(predicted: str | None, config: EvalConfig) -> str:
    """Convert an already-extracted prediction into the required final line."""
    if not predicted:
        return ""
    value = str(predicted).strip()
    if not value:
        return ""
    if config.dataset == "housing":
        lowered = value.lower()
        if lowered == "yes":
            return "Answer: Yes"
        if lowered == "no":
            return "Answer: No"
        return ""
    letters = set(_mc_choice_letters(config.dataset))
    letter = value.upper()
    if letter in letters:
        return f"Answer: ({letter})"
    return ""


def _extract_required_final_line_prediction(text: str, config: EvalConfig) -> str | None:
    """Extract a prediction only when the last non-empty line exactly matches the contract."""
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    if not lines:
        return None
    last = lines[-1]
    if config.dataset == "housing":
        if last == "Answer: Yes":
            return "Yes"
        if last == "Answer: No":
            return "No"
        return None
    if config.dataset in {"barexam", "casehold", "legalbench_scalr", "mas_legal_bench", "legal_link_eu", "medqa"}:
        letters = re.escape(_mc_choice_letters(config.dataset))
        match = re.fullmatch(rf"Answer: \(([{letters}])\)", last)
        if match:
            return match.group(1)
    return None


def _retrieved_answer_user(config: EvalConfig, passage_block: str, question: str) -> str:
    parts = [
        f"## Retrieved Passages\n{passage_block}",
        f"## Question\n{question}",
    ]
    contract = _final_answer_contract(config)
    if contract:
        parts.append(contract)
    return "\n\n".join(parts)


def _evidence_passage_block(result: dict) -> str:
    passages: list[str] = []
    for item in result.get("evidence_store") or []:
        if not isinstance(item, dict):
            continue
        idx = str(item.get("idx") or "").strip()
        text = str(item.get("text") or item.get("snippet") or "").strip()
        if not text:
            continue
        prefix = f"[{idx}] " if idx else ""
        passages.append(prefix + text)
    return "\n\n".join(passages)


def _near_completion_cap(output_tokens: int) -> bool:
    max_completion_tokens = _env_int("LLM_MAX_COMPLETION_TOKENS", 0)
    output_token_margin = _env_int("EVAL_OUTPUT_TOKEN_MARGIN", 16)
    return (
        max_completion_tokens > 0
        and output_tokens >= max(1, max_completion_tokens - output_token_margin)
    )


def _maybe_retry_final_answer_format(
    row: pd.Series,
    config: EvalConfig,
    result: dict,
    answer_text: str,
    predicted: str | None,
) -> tuple[str, str | None]:
    """Retry only malformed final-answer formatting, with the same evidence."""
    if not _env_truthy("EVAL_FINAL_FORMAT_RETRY"):
        return answer_text, predicted
    if not _requires_strict_answer_line(config):
        return answer_text, predicted

    final_line_prediction = _extract_required_final_line_prediction(answer_text, config)
    if final_line_prediction is not None:
        predicted = final_line_prediction

    metrics_before_retry = _get_metrics()
    output_tokens_before_retry = int(metrics_before_retry.get("output_tokens") or 0)
    malformed = final_line_prediction is None
    near_cap = (
        int(metrics_before_retry.get("count") or 0) <= 1
        and _near_completion_cap(output_tokens_before_retry)
    )
    if not malformed and not near_cap:
        return answer_text, predicted

    reasons = []
    if malformed:
        reasons.append("missing_marker" if predicted else "missing_prediction")
    if near_cap:
        reasons.append("near_completion_cap")

    retry_system = (
        "You are a strict final-answer formatter. Do not reason, explain, or solve "
        "the task. Return exactly one final Answer line in the required format."
    )
    retry_mode = "select_final_line"
    target_line = _required_answer_line_from_prediction(predicted, config)
    if target_line:
        retry_mode = "format_existing_prediction"
        retry_user = "\n\n".join([
            "## Required Output",
            f"Return exactly this line and nothing else:\n{target_line}",
            "## Retry Instruction",
            "The previous response already contained a parseable prediction but did not satisfy "
            "the output contract or was too close to the token cap. Preserve that prediction; "
            "only repair the final-answer format.",
        ])
    else:
        retry_mode = "force_discrete_same_evidence"
        evidence_block = _evidence_passage_block(result)
        if not evidence_block:
            evidence_block = "(No retrieved evidence was available in the previous attempt.)"
        retry_user = (
            "\n\n".join([
                "## Original Question",
                _fmt(row, config),
                "## Evidence Used In Previous Attempt",
                evidence_block,
                "## Previous Response",
                answer_text,
                "## Required Output",
                _final_answer_contract(config),
                "## Retry Instruction",
                "Your previous response did not contain a parseable final answer line. "
                "This is a same-model, same-evidence repair for a required discrete-answer task. "
                "Using only the original question, the evidence above, and your previous response, "
                "choose the best-supported allowed answer label. Unknown, indeterminate, and "
                "insufficient-evidence outputs are invalid for this benchmark. Return exactly one "
                "required final Answer line and nothing else. Do not add reasoning, cite evidence, "
                "or write any text after the Answer line.",
            ])
        )
    retry_answer = _llm_call(
        retry_system,
        retry_user,
        label=f"{config.mode}/answer_format_retry",
    )
    retry_predicted = _extract_required_final_line_prediction(retry_answer, config)
    if retry_predicted is None:
        retry_predicted = _extract_answer(retry_answer, config)
    metrics_after_retry = _get_metrics()
    retry_output_tokens = max(
        0,
        int(metrics_after_retry.get("output_tokens") or 0) - output_tokens_before_retry,
    )

    result["answer_format_retry"] = True
    result["answer_format_retry_reason"] = ",".join(reasons)
    result["answer_format_retry_reasons"] = reasons
    result["answer_format_retry_mode"] = retry_mode
    result["answer_format_retry_input_prediction"] = predicted
    if target_line:
        result["answer_format_retry_target_line"] = target_line
    result["answer_format_retry_output_tokens"] = retry_output_tokens
    result["answer_format_retry_near_cap"] = _near_completion_cap(retry_output_tokens)
    result["final_answer_before_format_retry"] = answer_text
    result["final_answer"] = retry_answer
    if (
        not retry_predicted
        or not _has_required_final_answer_line(retry_answer, retry_predicted, config)
        or result["answer_format_retry_near_cap"]
    ):
        result["answer_format_retry_valid"] = False
        return retry_answer, retry_predicted
    result["answer_format_retry_valid"] = True
    return retry_answer, retry_predicted


def _adaptive_hyre_route(row: pd.Series, config: EvalConfig) -> str:
    """Task-shape route for the one-policy HyRE runner."""
    if config.dataset == "housing" and _housing_state_where(row, config):
        return "state_filter"
    if config.dataset in ("casehold", "legalbench_scalr") or _has_answer_options(row, config):
        return "option_grounding"
    return "aligned_hyre"


def _snap_hyre_retrieve_and_answer(
    row: pd.Series,
    config: EvalConfig,
    *,
    label_prefix: str,
    where: dict | None = None,
    rerank_query: str | None = None,
    final_system: str | None = None,
    include_raw_anchor: bool = False,
    include_snap_anchor: bool = False,
) -> dict:
    """Shared 2-call HyRE path: snap+HyDE, retrieve, then synthesize."""
    question = _fmt(row, config)
    raw_question = _retrieval_question(row)
    question_intermediate = _fmt_intermediate(row, config)

    cache_entry = _hyre_cache_entry(row, config)
    if cache_entry:
        combined_raw = str(cache_entry.get("snap_and_hyre_raw") or cache_entry.get("hyde_passage_raw") or "")
        snap_block = str(cache_entry.get("snap_answer") or "")
        hyre_passage = str(cache_entry.get("hyde_passage") or "")
        parse_ok = bool(cache_entry.get("snap_hyre_parse_ok", True))
        snap_hyre_generation_meta = {}
        if not hyre_passage:
            snap_block, hyre_passage, parse_ok = _split_snap_and_hyde(
                combined_raw,
                fallback_passage=question_intermediate,
            )
    else:
        (
            combined_raw,
            snap_block,
            hyre_passage,
            parse_ok,
            snap_hyre_generation_meta,
        ) = _generate_snap_hyre_blocks(
            config,
            label=f"{label_prefix}/snap_and_hyre",
            question=question,
            fallback_passage=question_intermediate,
        )
    snap_letter = _extract_answer(snap_block, config)
    hyre_contains_answer = _contains_answer_artifact(hyre_passage)
    queries = [hyre_passage]
    if include_raw_anchor:
        queries.append(question_intermediate)
    if include_snap_anchor:
        snap_anchor = _strip_answer_line(snap_block)
        if snap_anchor and snap_anchor not in queries:
            queries.append(snap_anchor)

    retrieval = _retrieve_and_format(
        row,
        queries,
        k=config.retrieval_k,
        label_prefix=label_prefix,
        where=where if where is not None else _where_from_config(config),
        collection=_collection_for_config(config),
        rerank_query=rerank_query,
    )
    passage_block = "\n\n".join(retrieval["passages"])

    user = _retrieved_answer_user(config, passage_block, question)
    answer = _llm_call(final_system or _system_prompt(config, "rag"), user, label=f"{label_prefix}/answer")

    out = {
        "final_answer": answer,
        "formatted_question": question,
        "intermediate_question": question_intermediate,
        "snap_answer": snap_block,
        "snap_letter": snap_letter,
        "snap_and_hyre_raw": combined_raw,
        "snap_hyre_parse_ok": parse_ok,
        "hyre_cache_hit": bool(cache_entry),
        "hyre_cache_label": _row_label(row, config) if cache_entry else "",
        "logical_llm_calls": 2,
        "cached_generation_calls": 1 if cache_entry else 0,
        "hyde_passage": hyre_passage,
        "hyde_passage_raw": combined_raw,
        "hyde_contains_answer_artifact": hyre_contains_answer,
        "retrieval_queries": queries,
        "rerank_query": rerank_query or "",
        "retrieval_where": where or _where_from_config(config) or {},
        "final_context_fields": ["retrieved_passages", "question"],
        "final_prompt_preview": _preview_text(user),
        "evidence_store": retrieval["evidence_store"],
        "retrieved_ids": retrieval["retrieved_ids"],
        "gold_retrieved": retrieval["gold_retrieved"],
        **_retrieval_cache_audit_fields(retrieval),
    }
    if not parse_ok:
        out["routed_to"] = f"{label_prefix}_parse_failed_fallback_to_question"
    if include_raw_anchor:
        out["raw_anchor_included"] = True
    if include_snap_anchor:
        out["snap_anchor_included"] = True
    return out


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
    cleaned = _sanitize_intermediate_text(raw, fallback="")
    fallback_text = (fallback or "").strip()
    return {
        "text": cleaned or fallback_text,
        "raw": raw,
        "contains_answer": _contains_answer_artifact(raw),
        "used_fallback": not bool(cleaned),
    }


def _system_prompt(config: EvalConfig, role: str = "answer") -> str:
    """Get dataset-appropriate system prompt."""
    if is_beir_dataset(config.dataset):
        subset = BEIR_DATASETS[config.dataset]
        domains = {
            "scifact": "scientific claim verification",
            "nfcorpus": "medical and nutrition literature",
            "fiqa": "financial question answering",
            "trec-covid": "COVID-19 biomedical literature",
            "scidocs": "scientific-paper citation and related-work search",
        }
        domain = domains.get(subset, "open-domain information retrieval")
        prompts = {
            "answer": (
                f"You are an information retrieval assistant for BEIR/{subset}, specializing in {domain}. "
                "Answer the query briefly and accurately from general knowledge when needed. "
                "No multiple-choice or final-answer marker is required."
            ),
            "rag": (
                f"You are an information retrieval assistant for BEIR/{subset}, specializing in {domain}. "
                "Retrieved corpus passages are provided. Use them to answer the query briefly and identify "
                "the relevant facts, entities, mechanisms, or claims. No multiple-choice or final-answer "
                "marker is required."
            ),
            "research": (
                f"You are an information retrieval assistant for BEIR/{subset}, specializing in {domain}. "
                "Research findings are provided. Synthesize them briefly and accurately."
            ),
            "hyde": (
                f"You are a corpus author for BEIR/{subset}, specializing in {domain}. Given a search query, "
                "write a short neutral passage (2-3 sentences) that could appear in a relevant source document. "
                "Use concrete terminology and entities from the query when possible. Do not mention the query, "
                "do not include answer labels, and do not output an `Answer:` line."
            ),
            "snap_hyde": (
                f"You are a corpus author for BEIR/{subset}, specializing in {domain}. A user has reasoned "
                "about an information need. Write a short neutral source-style passage (2-3 sentences) that "
                "would be most useful for retrieving relevant corpus documents. Use the reasoning only to target "
                "the relevant facts, entities, mechanisms, or claims. Do not output answer labels or an `Answer:` line."
            ),
        }
        return prompts.get(role, prompts["answer"])
    if config.dataset == "housing":
        prompts = {
            "answer": (
                "You are a legal expert specializing in housing law. Answer the Yes/No question below. "
                "Apply the state or territory named in the question; do not substitute law from another jurisdiction. "
                "Reason step by step, then end with exactly one final line: Answer: Yes or Answer: No"
            ),
            "rag": (
                "You are a legal expert specializing in housing law. Reason through the question "
                "step by step. Retrieved passages are provided — use them to verify or "
                "refine your reasoning, but think through the problem independently first. "
                "Apply the state or territory named in the question; do not treat another jurisdiction's passage as controlling. "
                "End with exactly one final line: Answer: Yes or Answer: No"
            ),
            "research": (
                "You are a legal expert specializing in housing law. Reason through the question "
                "step by step. Research findings are provided — use them to verify or "
                "refine your reasoning, but think through the problem independently first. "
                "Apply the state or territory named in the question; do not treat another jurisdiction's passage as controlling. "
                "End with exactly one final line: Answer: Yes or Answer: No"
            ),
            "hyde": (
                "You are a legal textbook author specializing in housing law. Given a legal question, "
                "write a short passage (2-3 sentences) that would appear in a reference guide as the answer. "
                "If the question names a state or territory, preserve that jurisdiction and write as though describing that jurisdiction's statute. "
                "Write in the style of a legal reference — state the statute, rule, or "
                "regulation directly. Do not discuss the question itself or say 'the answer is'."
            ),
            "snap_hyde": (
                "You are a legal textbook author specializing in housing law. A student has answered a legal question "
                "and provided their reasoning. Write a short passage (2-3 sentences) from a legal reference that "
                "would be most relevant to verifying or correcting this answer. Focus on the specific "
                "state or territory named in the question and the specific "
                "statute, regulation, or rule at the heart of the question. Write in reference style — "
                "state the law directly."
            ),
        }
        prompts["devil_hyde"] = (
            "You are a legal textbook author specializing in housing law. A student has answered a legal question. "
            "Your job is to play DEVIL'S ADVOCATE: write a short passage (2-3 sentences) from a legal reference "
            "that would CHALLENGE or CONTRADICT the student's answer. Focus on the state or territory named in the question and the rule, exception, or statute "
            "that supports the OPPOSITE conclusion. Write in reference style — state the law directly."
        )
        prompts["top2_snap"] = (
            "You are a legal expert specializing in housing law. Answer the Yes/No question below. "
            "Reason step by step. Identify what your FIRST choice answer is, and also what the ALTERNATIVE "
            "answer would be and why someone might argue for it. "
            "End with exactly one final line: Answer: Yes or Answer: No"
        )
        prompts["top2_hyde"] = (
            "You are a legal textbook author specializing in housing law. A student has answered a legal question. "
            "Write a short passage (2-3 sentences) from a legal reference that would support the ALTERNATIVE "
            "or SECOND-CHOICE answer — the answer the student considered but rejected. Focus on the state or territory named in the question and the specific "
            "statute, regulation, or rule that would support that alternative. Write in reference style."
        )
        return prompts.get(role, prompts["answer"])
    if config.dataset in ("casehold", "legalbench_scalr"):
        prompts = {
            "answer": (
                "You are a legal expert specializing in case law. Read the citing context from a court opinion "
                "and determine which holding is most likely being referenced. "
                "Reason step by step, then end with exactly one final line in the form: Answer: (X)"
            ),
            "rag": (
                "You are a legal expert specializing in case law. Reason through the question "
                "step by step. Retrieved holdings are provided — use them to verify or "
                "refine your reasoning, but think through the problem independently first. "
                "End with exactly one final line in the form: Answer: (X)"
            ),
            "research": (
                "You are a legal expert specializing in case law. Reason through the question "
                "step by step. Research findings are provided — use them to verify or "
                "refine your reasoning, but think through the problem independently first. "
                "End with exactly one final line in the form: Answer: (X)"
            ),
            "hyde": (
                "You are a legal textbook author. Given a court opinion excerpt that cites a holding, "
                "write a short passage (2-3 sentences) stating the likely holding being referenced. "
                "Write in the style of a case holding — state the rule directly. Do not choose an "
                "answer letter, do not mention candidate labels, and do not output a final answer."
            ),
            "snap_hyde": (
                "You are a legal textbook author. A student has identified what they think is the correct "
                "holding for a citation. Write a short passage (2-3 sentences) from a legal reference "
                "that would be most relevant to verifying or correcting the student's reasoning. Use "
                "the reasoning to target the discriminating legal issue, but write only a neutral case "
                "holding/reference passage. Do not choose an answer letter, do not mention candidate "
                "labels, and do not output a final answer."
            ),
        }
        return prompts.get(role, prompts["answer"])
    if config.dataset == "mas_legal_bench":
        prompts = {
            "answer": (
                "You are a legal expert specializing in GDPR and data-protection enforcement. "
                "Answer the multiple-choice question below. Reason from the legal facts and "
                "rules, then end with exactly one final line in the form: Answer: (X)"
            ),
            "rag": (
                "You are a legal expert specializing in GDPR and data-protection enforcement. "
                "Reason through the question step by step. Retrieved context passages are "
                "provided — use them to verify or refine your reasoning, but do not assume "
                "every passage is dispositive. End with exactly one final line in the form: "
                "Answer: (X)"
            ),
            "research": (
                "You are a legal expert specializing in GDPR and data-protection enforcement. "
                "Research findings are provided — use them to verify or refine your reasoning. "
                "End with exactly one final line in the form: Answer: (X)"
            ),
            "hyde": (
                "You are a legal reference author specializing in GDPR and data-protection "
                "enforcement. Given a legal question, write a short passage (2-3 sentences) "
                "that would help locate the controlling facts, legal framework, or enforcement "
                "principle. Do not choose an answer letter, do not mention candidate labels, "
                "and do not output a final answer."
            ),
            "snap_hyde": (
                "You are a legal reference author specializing in GDPR and data-protection "
                "enforcement. A student has reasoned through a legal question. Write a short "
                "neutral reference passage (2-3 sentences) that would be most relevant to "
                "verifying or correcting the student's reasoning. Use the reasoning to target "
                "the legal issue, but do not choose an answer letter, mention candidate labels, "
                "or output a final answer."
            ),
        }
        return prompts.get(role, prompts["answer"])
    if config.dataset == "legal_link_eu":
        prompts = {
            "answer": (
                "You are a legal expert specializing in European Union law and EUR-Lex legal "
                "authority relationships. Answer the multiple-choice question below. Track "
                "whether one act repeals, corrects, completes, extends, or renders another "
                "act obsolete. Reason from the legal materials and end with exactly one final "
                "line in the form: Answer: (X)"
            ),
            "rag": (
                "You are a legal expert specializing in European Union law and EUR-Lex legal "
                "authority relationships. Retrieved EUR-Lex passages are provided. Use them "
                "to determine the operative relationship between the legal acts and choose "
                "the best option. End with exactly one final line in the form: Answer: (X)"
            ),
            "research": (
                "You are a legal expert specializing in European Union law and EUR-Lex legal "
                "authority relationships. Research findings are provided. Use them to verify "
                "the legal relationship between the acts and end with exactly one final line "
                "in the form: Answer: (X)"
            ),
            "hyde": (
                "You are a legal reference author specializing in European Union law. Given "
                "a question about the relationship between legal acts, write a short neutral "
                "reference passage (2-3 sentences) that would help locate the controlling "
                "source, target act, amendment, repeal, correction, obsolescence, or validity "
                "extension. Do not choose an answer letter or mention candidate labels."
            ),
            "snap_hyde": (
                "You are a legal reference author specializing in European Union law. A "
                "student has reasoned through a question about legal authority relationships. "
                "Write a short neutral reference passage (2-3 sentences) that would be most "
                "relevant to verifying or correcting that reasoning. Do not choose an answer "
                "letter, mention candidate labels, or output a final answer."
            ),
        }
        return prompts.get(role, prompts["answer"])
    if config.dataset == "medqa":
        prompts = {
            "answer": (
                "You are a medical expert answering a USMLE-style multiple-choice question. "
                "Reason through the clinical facts and mechanisms, then end with exactly one "
                "final line in the form: Answer: (X)"
            ),
            "rag": (
                "You are a medical expert answering a USMLE-style multiple-choice question. "
                "Retrieved textbook passages are provided; use them to verify or refine your "
                "clinical reasoning, but choose the best answer from the options. End with "
                "exactly one final line in the form: Answer: (X)"
            ),
            "research": (
                "You are a medical expert answering a USMLE-style multiple-choice question. "
                "Research findings are provided; use them to verify or refine your reasoning. "
                "End with exactly one final line in the form: Answer: (X)"
            ),
            "hyde": (
                "You are a medical textbook author. Given a USMLE-style question, write a "
                "short neutral textbook passage (2-3 sentences) that states the disease, "
                "mechanism, diagnostic principle, treatment principle, or physiology most "
                "relevant to the question. Do not choose an answer letter, mention candidate "
                "labels, or output a final answer."
            ),
            "snap_hyde": (
                "You are a medical textbook author. A student has reasoned through a "
                "USMLE-style question. Write a short neutral textbook passage (2-3 sentences) "
                "that would be most relevant to verifying or correcting that reasoning. Do "
                "not choose an answer letter, mention candidate labels, or output a final answer."
            ),
        }
        return prompts.get(role, prompts["answer"])
    if config.dataset in ("legal_rag", "legal_rag_bench", "australian"):
        if config.dataset == "legal_rag_bench":
            domain = "Victorian criminal law and procedure"
        else:
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
            "Reason step by step, then end with exactly one final line in the form: Answer: (X)"
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
            "End with exactly one final line in the form: Answer: (X)"
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
    "legal_rag_bench": "legal_rag_bench_passages",
    "mas_legal_bench": "mas_legal_bench_passages",
    "legal_link_eu": "legal_link_eu_passages",
    "australian": "australian_legal",
    "casehold": "casehold_holdings",
    "musique": "musique_passages",
    "legalbench_scalr": "legalbench_scalr_holdings",
    "medqa": "medqa_textbooks",
    "beir_scifact": "beir_scifact",
    "beir_nfcorpus": "beir_nfcorpus",
    "beir_fiqa": "beir_fiqa",
    "beir_trec_covid": "beir_trec_covid",
    "beir_scidocs": "beir_scidocs",
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


def _coerce_gold_ids(value) -> list[str]:
    """Normalize scalar/list/comma/JSON gold-id fields to a list of strings."""
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() == "nan":
            return []
        if stripped.startswith("[") or stripped.startswith("{"):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            if parsed is not None:
                return _coerce_gold_ids(parsed)
        return [part.strip() for part in stripped.split(",") if part.strip()]
    if isinstance(value, dict):
        ids: list[str] = []
        for item in value.values():
            ids.extend(_coerce_gold_ids(item))
        return ids
    if isinstance(value, (list, tuple, set)):
        ids = []
        for item in value:
            ids.extend(_coerce_gold_ids(item))
        return ids
    return [str(value).strip()] if str(value).strip() else []


def _gold_ids(row: pd.Series) -> list[str]:
    """Return all acceptable retrieval target ids for a question row.

    Most datasets use ``gold_idx``. Legal-RAG-QA is packaged with a
    ``relevant_passages`` list instead, so accept that as the fallback target set.
    """
    ids = _coerce_gold_ids(row.get("gold_idx", ""))
    if ids:
        return list(dict.fromkeys(ids))
    return list(dict.fromkeys(_coerce_gold_ids(row.get("relevant_passages", ""))))


def _gold_idx_string(row: pd.Series) -> str:
    return ",".join(_gold_ids(row))


def _is_gold_retrieved(row: pd.Series, retrieved_ids: list[str]) -> bool:
    gold = set(_gold_ids(row))
    return bool(gold & {str(idx) for idx in retrieved_ids}) if gold else False


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
    retrieved_ids = [ev.get("idx", "") for ev in result.get("evidence_store", [])]
    gold_retrieved = _is_gold_retrieved(row, retrieved_ids)

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
    gold = _gold_reference_text(row, config)
    if not gold:
        raise RuntimeError(
            f"golden_passage missing gold reference for idx={_row_idx_for_cache(row)}"
        )

    system = _system_prompt(config, "rag")
    user = f"## Reference Passage\n{gold}\n\n## Question\n{question}"
    answer = _llm_call(system, user, label="golden_passage")
    # The gold passage was injected directly — mark gold_retrieved=True so
    # downstream analyzers don't report this mode as "no retrieval". Keep the
    # retrieved_ids/evidence_store shape consistent with retrieval modes.
    gold_ids = _gold_ids(row)
    gold_idx = ",".join(gold_ids)
    return {
        "final_answer": answer,
        "gold_retrieved": bool(gold_ids),
        "retrieved_ids": gold_ids,
        "evidence_store": [{"idx": gold_idx, "text": gold, "cross_encoder_score": 0.0}] if gold_idx else [],
    }


def run_golden_plus_neighbors(row: pd.Series, config: EvalConfig) -> dict:
    """Gold passage plus nearest corpus neighbors.

    This is a diagnostic control for the observed "gold passage can underperform
    LLM-only" issue. It keeps the gold passage first, then fills the remaining
    context budget with passages retrieved by embedding the gold passage.
    """
    question = _fmt(row, config)
    gold = _gold_reference_text(row, config)
    if not gold:
        raise RuntimeError(
            f"golden_plus_neighbors missing gold reference for idx={_row_idx_for_cache(row)}"
        )

    gold_ids = _gold_ids(row)
    gold_idx = ",".join(gold_ids)
    max_neighbors = max(config.retrieval_k - 1, 0)
    retrieval = _retrieve_and_format(
        row,
        [gold],
        k=max(config.retrieval_k, 1),
        label_prefix="golden_plus_neighbors",
        where=_retrieval_where_for_row(row, config),
        collection=_collection_for_config(config),
    )

    gold_id_set = set(gold_ids)
    neighbor_evidence = [
        ev for ev in retrieval["evidence_store"]
        if str(ev.get("idx", "")) not in gold_id_set
    ][:max_neighbors]

    evidence_store = []
    if gold_idx:
        evidence_store.append({
            "idx": gold_idx,
            "text": gold,
            "source": "golden_passage",
            "cross_encoder_score": 0.0,
        })
    evidence_store.extend(neighbor_evidence)

    passages = [
        f"[Source {i}]\n{ev.get('text', '')}"
        for i, ev in enumerate(evidence_store, 1)
    ]
    passage_block = "\n\n".join(passages)
    user = _retrieved_answer_user(config, passage_block, question)
    answer = _llm_call(_system_prompt(config, "rag"), user, label="golden_plus_neighbors")

    retrieved_ids = list(gold_ids)
    retrieved_ids.extend(str(ev.get("idx", "")) for ev in neighbor_evidence)
    return {
        "final_answer": answer,
        "gold_retrieved": bool(gold_ids),
        "retrieved_ids": retrieved_ids,
        "evidence_store": evidence_store,
        "neighbor_retrieved_ids": [str(ev.get("idx", "")) for ev in neighbor_evidence],
        **_retrieval_cache_audit_fields(retrieval),
        "final_context_fields": ["gold_passage", "retrieved_neighbors", "question"],
        "final_prompt_preview": _preview_text(user),
    }


def _golden_arb_common(row: pd.Series, config: EvalConfig, arb_system: str, label_prefix: str) -> dict:
    """Shared logic for golden arbitration variants."""
    question = _fmt(row, config)
    gold = _gold_reference_text(row, config)
    if not gold:
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
        "Reason step by step, then end with exactly one final line in the form: Answer: (X)"
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
        "Reason step by step, then end with exactly one final line in the form: Answer: (X)"
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
    cached = _documents_from_retrieval_cache(
        row=row,
        label_prefix=label_prefix,
        collection=collection,
        where=where,
        embedding_model=embedding_model,
        k=k,
        queries=queries,
    )
    retrieval_cache_hit = cached is not None
    retrieval_cache_entry = cached[1] if cached is not None else None
    retrieval_doc_cache_hit = bool((retrieval_cache_entry or {}).get("_doc_cache_hit"))
    cross_encoder_max_chars = (retrieval_cache_entry or {}).get(
        "cross_encoder_max_chars",
        os.getenv("CROSS_ENCODER_MAX_CHARS", ""),
    )
    if cached is not None:
        docs = cached[0]
    else:
        vs = get_vectorstore(collection, embedding_model=embedding_model)
        docs = retrieve_documents_multi_query(queries=queries, k=k, vectorstore=vs, where=where,
                                              rerank_query=rerank_query)

    passages = []
    evidence_store = []
    for i, doc in enumerate(docs, 1):
        text = doc.page_content
        metadata = dict(doc.metadata or {})
        idx = str(doc.metadata.get("idx", f"{label_prefix}_{i}"))
        ce_score = doc.metadata.get("cross_encoder_score", 0.0)
        header = _format_evidence_header(i, idx, metadata)
        passages.append(f"{header}\n{text}")
        evidence_store.append({
            "idx": idx,
            "text": text,
            "source": metadata.get("source", "unknown"),
            "citation": metadata.get("citation", ""),
            "role": metadata.get("role", ""),
            "context_title": metadata.get("context_title", ""),
            "cross_encoder_score": ce_score,
            "cross_encoder_query_truncated": bool(metadata.get("cross_encoder_query_truncated", False)),
            "cross_encoder_doc_truncated": bool(metadata.get("cross_encoder_doc_truncated", False)),
        })

    gold_idx = _gold_idx_string(row)
    retrieved_ids = [ev["idx"] for ev in evidence_store]
    gold_retrieved = _is_gold_retrieved(row, retrieved_ids)
    max_ce_score = max((ev["cross_encoder_score"] for ev in evidence_store), default=0.0)
    row_source = str(row.get("source", "") or "").strip()
    same_source_retrieved_ids = [
        ev["idx"] for ev in evidence_store
        if row_source and str(ev.get("source", "") or "").strip() == row_source
    ]
    same_source_retrieved = bool(same_source_retrieved_ids)
    cross_encoder_doc_truncated_count = sum(
        1 for ev in evidence_store if ev.get("cross_encoder_doc_truncated")
    )
    cross_encoder_query_truncated = any(
        bool(ev.get("cross_encoder_query_truncated")) for ev in evidence_store
    )
    if retrieval_cache_entry:
        cross_encoder_doc_truncated_count = int(
            retrieval_cache_entry.get("cross_encoder_doc_truncated_count") or cross_encoder_doc_truncated_count
        )
        cross_encoder_query_truncated = bool(
            retrieval_cache_entry.get("cross_encoder_query_truncated") or cross_encoder_query_truncated
        )
    source_doc = str(row.get("source_doc", "") or "").strip()
    target_doc = str(row.get("target_doc", "") or "").strip()
    source_doc_retrieved_ids: list[str] = []
    target_doc_retrieved_ids: list[str] = []
    if source_doc or target_doc:
        for ev in evidence_store:
            ev_source = str(ev.get("source", "") or "").strip()
            ev_citation = str(ev.get("citation", "") or "").strip()
            ev_text_id = ev_source or ev_citation
            if source_doc and (ev_text_id == source_doc or ev_source == source_doc or ev_citation == source_doc):
                source_doc_retrieved_ids.append(ev["idx"])
            if target_doc and (ev_text_id == target_doc or ev_source == target_doc or ev_citation == target_doc):
                target_doc_retrieved_ids.append(ev["idx"])

    _record_trace_event(
        "retrieval",
        label=label_prefix,
        queries=queries,
        rerank_query=rerank_query or "",
        k=k,
        where=where or {},
        collection=collection,
        embedding_model=embedding_model or "",
        retrieval_cache_hit=retrieval_cache_hit,
        retrieval_doc_cache_hit=retrieval_doc_cache_hit,
        retrieval_cache_label=(retrieval_cache_entry or {}).get("label", ""),
        retrieval_cache_query_hash=(retrieval_cache_entry or {}).get("query_hash", ""),
        retrieval_query_hash=_hash_texts([str(q) for q in queries]),
        results=evidence_store,
        retrieved_ids=retrieved_ids,
        gold_idx=gold_idx,
        gold_retrieved=gold_retrieved,
        row_source=row_source,
        same_source_retrieved=same_source_retrieved,
        same_source_retrieved_ids=same_source_retrieved_ids,
        cross_encoder_doc_truncated_count=cross_encoder_doc_truncated_count,
        cross_encoder_query_truncated=cross_encoder_query_truncated,
        cross_encoder_max_chars=cross_encoder_max_chars,
        source_doc=source_doc,
        target_doc=target_doc,
        source_doc_retrieved=bool(source_doc_retrieved_ids),
        source_doc_retrieved_ids=source_doc_retrieved_ids,
        target_doc_retrieved=bool(target_doc_retrieved_ids),
        target_doc_retrieved_ids=target_doc_retrieved_ids,
        max_ce_score=max_ce_score,
    )

    return {
        "passages": passages,
        "evidence_store": evidence_store,
        "retrieved_ids": retrieved_ids,
        "gold_retrieved": gold_retrieved,
        "same_source_retrieved": same_source_retrieved,
        "same_source_retrieved_ids": same_source_retrieved_ids,
        "source_doc_retrieved": bool(source_doc_retrieved_ids),
        "source_doc_retrieved_ids": source_doc_retrieved_ids,
        "target_doc_retrieved": bool(target_doc_retrieved_ids),
        "target_doc_retrieved_ids": target_doc_retrieved_ids,
        "cross_encoder_doc_truncated_count": cross_encoder_doc_truncated_count,
        "cross_encoder_query_truncated": cross_encoder_query_truncated,
        "cross_encoder_max_chars": cross_encoder_max_chars,
        "max_ce_score": max_ce_score,
        "retrieval_cache_hit": retrieval_cache_hit,
        "retrieval_doc_cache_hit": retrieval_doc_cache_hit,
        "retrieval_where": where or {},
        "retrieval_cache_query_hash": (retrieval_cache_entry or {}).get("query_hash", ""),
        "retrieval_query_hash": _hash_texts([str(q) for q in queries]),
    }


def _format_evidence_header(i: int, idx: str, metadata: dict) -> str:
    """Human-readable metadata header for retrieved evidence shown to the final LLM."""
    role = str(metadata.get("role", "") or "").strip()
    context_title = str(metadata.get("context_title", "") or "").strip()
    citation = str(metadata.get("citation", "") or "").strip()
    source = str(metadata.get("source", "") or "").strip()
    if role or context_title or citation:
        lines = [f"[Evidence {i}]", f"passage_id: {idx}"]
        if role:
            lines.append(f"legal_link_role: {role}")
        if context_title:
            lines.append(f"title: {context_title}")
        if citation:
            lines.append(f"citation: {citation}")
        elif source:
            lines.append(f"source: {source}")
        return "\n".join(lines)
    return f"[Source {i}]"


def _retrieval_cache_audit_fields(retrieval: dict) -> dict:
    """Top-level retrieval-cache fields for detail logs."""
    return {
        "retrieval_cache_hit": retrieval.get("retrieval_cache_hit", False),
        "retrieval_doc_cache_hit": retrieval.get("retrieval_doc_cache_hit", False),
        "retrieval_where": retrieval.get("retrieval_where", {}),
        "retrieval_cache_query_hash": retrieval.get("retrieval_cache_query_hash", ""),
        "retrieval_query_hash": retrieval.get("retrieval_query_hash", ""),
        "same_source_retrieved": retrieval.get("same_source_retrieved", False),
        "same_source_retrieved_ids": retrieval.get("same_source_retrieved_ids", []),
        "source_doc_retrieved": retrieval.get("source_doc_retrieved", False),
        "source_doc_retrieved_ids": retrieval.get("source_doc_retrieved_ids", []),
        "target_doc_retrieved": retrieval.get("target_doc_retrieved", False),
        "target_doc_retrieved_ids": retrieval.get("target_doc_retrieved_ids", []),
        "cross_encoder_doc_truncated_count": retrieval.get("cross_encoder_doc_truncated_count", 0),
        "cross_encoder_query_truncated": retrieval.get("cross_encoder_query_truncated", False),
        "cross_encoder_max_chars": retrieval.get("cross_encoder_max_chars", ""),
        "max_ce_score": retrieval.get("max_ce_score", 0.0),
    }


def _coerce_rewrite_queries(parsed: Any) -> List[str]:
    """Validate query-rewriter JSON and return non-empty unique queries."""
    if not isinstance(parsed, dict):
        return []
    primary = str(parsed.get("primary") or "").strip()
    if not primary:
        return []
    queries = [primary]
    alternatives = parsed.get("alternatives") or []
    if isinstance(alternatives, list):
        queries.extend(str(item).strip() for item in alternatives if str(item).strip())
    deduped: List[str] = []
    seen: set[str] = set()
    for query in queries:
        key = query.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(query)
    return deduped


def _json_string_literals(text: str) -> list[str]:
    """Extract JSON string literals from a partial JSON fragment."""
    values: list[str] = []
    for match in re.finditer(r'"(?:\\.|[^"\\])*"', text):
        value = _loads_json_string_literal(match.group(0))
        if value is None:
            continue
        values.append(value)
    return values


def _loads_json_string_literal(literal: str) -> str | None:
    """Load a JSON string, repairing narrow model-output string literal errors."""
    try:
        return str(json.loads(literal))
    except json.JSONDecodeError:
        pass
    if not (isinstance(literal, str) and len(literal) >= 2 and literal[0] == literal[-1] == '"'):
        return None
    fixed_chars: list[str] = []
    index = 0
    inner = literal[1:-1]
    while index < len(inner):
        char = inner[index]
        if char == "\n":
            fixed_chars.append("\\n")
        elif char == "\r":
            fixed_chars.append("\\r")
        elif char == "\t":
            fixed_chars.append("\\t")
        elif char == "\\" and index + 1 < len(inner) and inner[index + 1] == "'":
            # JSON permits apostrophes unescaped inside double-quoted strings;
            # some models emit Python/SQL-style \' escapes in generated queries.
            fixed_chars.append("'")
            index += 1
        else:
            fixed_chars.append(char)
        index += 1
    try:
        return str(json.loads('"' + "".join(fixed_chars) + '"'))
    except json.JSONDecodeError:
        return None


def _parse_rewrite_json(text: str) -> tuple[Any, str]:
    """Parse query-rewriter JSON, with explicit recovery for partial objects."""
    parsed = _parse_json(text)
    if parsed is not None:
        return parsed, "json"

    raw = str(text or "")
    primary_match = re.search(r'"primary"\s*:\s*("(?:\\.|[^"\\])*")', raw)
    if not primary_match:
        return None, "invalid_json"
    primary = _loads_json_string_literal(primary_match.group(1))
    if primary is None:
        return None, "invalid_json"
    primary = primary.strip()
    if not primary:
        return None, "invalid_json"

    alternatives: list[str] = []
    alt_match = re.search(r'"alternatives"\s*:\s*\[([\s\S]*)', raw)
    if alt_match:
        for value in _json_string_literals(alt_match.group(1)):
            value = str(value).strip()
            if value:
                alternatives.append(value)
            if len(alternatives) >= 2:
                break
    return {"primary": primary, "alternatives": alternatives}, "partial_json"


def _rewrite_parse_failure_reason(parsed: Any) -> str:
    if parsed is None:
        return "invalid_json"
    if not isinstance(parsed, dict):
        return f"json_{type(parsed).__name__}"
    if not str(parsed.get("primary") or "").strip():
        return "missing_primary"
    return "empty_queries"


def _rewrite_query_with_meta(
    question: str,
    label: str = "rag_rewrite/rewrite",
) -> tuple[List[str], dict]:
    """LLM query rewrite with explicit retry/fallback metadata."""
    rewrite_prompt = (
        f"Original legal research question: {question}\n\n"
        f"Sub-question: {question}\n"
        f"Authority target: \n"
        f"Retrieval hints: none"
    )
    raw_rewrite = _llm_call(load_skill("query_rewriter"), rewrite_prompt, label=label)
    parsed, parse_kind = _parse_rewrite_json(raw_rewrite)
    queries = _coerce_rewrite_queries(parsed)
    meta = {
        "rewrite_raw": raw_rewrite,
        "rewrite_parse_ok": bool(queries),
        "rewrite_parse_kind": parse_kind if queries else "",
        "rewrite_partial_json_repair": parse_kind == "partial_json" and bool(queries),
        "rewrite_format_retry": False,
        "rewrite_format_retry_reasons": [],
        "rewrite_used_fallback": False,
    }
    if queries:
        return queries, meta

    reasons = [_rewrite_parse_failure_reason(parsed)]
    if _env_truthy("EVAL_GENERATION_FORMAT_RETRY"):
        retry_system = (
            load_skill("query_rewriter")
            + "\n\nYou are repairing a malformed query rewrite. Return ONLY valid JSON "
            "with string field `primary` and list field `alternatives`. Do not include "
            "markdown fences, prose, comments, or any other keys."
        )
        retry_prompt = "\n\n".join([
            "## Original legal research question",
            question,
            "## Previous malformed output",
            raw_rewrite or "",
            "## Required Output",
            (
                "Return only JSON in this exact shape:\n"
                '{"primary":"main search query","alternatives":["alternate query 1","alternate query 2"]}'
            ),
        ])
        retry_raw = _llm_call(retry_system, retry_prompt, label=f"{label}/format_retry")
        retry_parsed, retry_parse_kind = _parse_rewrite_json(retry_raw)
        retry_queries = _coerce_rewrite_queries(retry_parsed)
        meta.update({
            "rewrite_raw_before_format_retry": raw_rewrite,
            "rewrite_raw": retry_raw,
            "rewrite_parse_ok": bool(retry_queries),
            "rewrite_parse_kind": retry_parse_kind if retry_queries else "",
            "rewrite_format_retry_parse_kind": retry_parse_kind if retry_queries else "",
            "rewrite_partial_json_repair": retry_parse_kind == "partial_json" and bool(retry_queries),
            "rewrite_format_retry": True,
            "rewrite_format_retry_reason": ",".join(reasons),
            "rewrite_format_retry_reasons": reasons,
            "rewrite_format_retry_valid": bool(retry_queries),
        })
        if retry_queries:
            return retry_queries, meta
        reasons.append(_rewrite_parse_failure_reason(retry_parsed))
        meta["rewrite_format_retry_reasons"] = reasons
        meta["rewrite_format_retry_reason"] = ",".join(reasons)

    meta["rewrite_used_fallback"] = True
    meta["rewrite_parse_ok"] = False
    if _no_silent_fallback_enabled():
        raise RuntimeError(
            "NO_SILENT_FALLBACK blocked rag_rewrite query rewrite parse failure: "
            + ",".join(reasons)
        )
    return [question], meta


def _rewrite_query(question: str, label: str = "rag_rewrite/rewrite") -> List[str]:
    """LLM query rewrite -> list of queries (primary + alternatives)."""
    queries, _ = _rewrite_query_with_meta(question, label=label)
    return queries


def run_rag_hyde(row: pd.Series, config: EvalConfig) -> dict:
    """HyDE: generate hypothetical answer passage, embed it, retrieve similar real passages."""
    question = _fmt(row, config)
    question_intermediate = _fmt_intermediate(row, config)

    # Step 1: Generate hypothetical passage
    cache_entry = _hyre_cache_entry(row, config)
    if cache_entry and cache_entry.get("hyde_passage"):
        hyde = {
            "text": str(cache_entry.get("hyde_passage") or ""),
            "raw": str(cache_entry.get("hyde_passage_raw") or cache_entry.get("hyde_passage") or ""),
            "contains_answer": bool(cache_entry.get("hyde_contains_answer_artifact", False)),
            "used_fallback": bool(cache_entry.get("hyde_used_fallback", False)),
        }
    else:
        hyde = _generate_hyde(
            config,
            "hyde",
            _question_only_hyde_user(question_intermediate),
            label="hyde/generate",
            fallback=question_intermediate,
        )

    # Step 2: Retrieve using the hypothetical passage as query
    retrieval = _retrieve_and_format(row, [hyde["text"]], k=config.retrieval_k, label_prefix="hyde",
                                     where=_retrieval_where_for_row(row, config),
                                     collection=_collection_for_config(config))
    passage_block = "\n\n".join(retrieval["passages"])

    # Step 3: Answer with evidence
    user = _retrieved_answer_user(config, passage_block, question)
    answer = _llm_call(_system_prompt(config, "rag"), user, label="hyde/answer")

    return {
        "final_answer": answer,
        "formatted_question": question,
        "intermediate_question": question_intermediate,
        "hyde_passage": hyde["text"],
        "hyde_passage_raw": hyde["raw"],
        "hyde_contains_answer_artifact": hyde["contains_answer"],
        "hyde_used_fallback": hyde.get("used_fallback", False),
        "hyde_cache_hit": bool(cache_entry),
        "hyre_cache_hit": bool(cache_entry),
        "hyre_cache_label": _row_label(row, config) if cache_entry else "",
        "logical_llm_calls": 2,
        "cached_generation_calls": 1 if cache_entry else 0,
        "retrieval_queries": [hyde["text"]],
        "rerank_query": "",
        "final_context_fields": ["retrieved_passages", "question"],
        "final_prompt_preview": _preview_text(user),
        "evidence_store": retrieval["evidence_store"],
        "retrieved_ids": retrieval["retrieved_ids"],
        "gold_retrieved": retrieval["gold_retrieved"],
        **_retrieval_cache_audit_fields(retrieval),
    }


def run_rag_hyde_exemplar(row: pd.Series, config: EvalConfig) -> dict:
    """Probe-only HyDE with dataset-specific passage-style guidance.

    This does not provide answer evidence. It only gives the generator a
    fixed real-passage style signal for the target dataset so we can isolate whether
    passage-shape alignment helps retrieval independently of snap reasoning.
    """
    question = _fmt(row, config)
    question_intermediate = _fmt_intermediate(row, config)

    cache_entry = _hyre_cache_entry(row, config)
    if cache_entry:
        raw = str(cache_entry.get("hyde_passage_raw") or cache_entry.get("raw") or "")
        text = str(cache_entry.get("hyde_passage") or cache_entry.get("text") or "")
        if not text:
            text = _sanitize_intermediate_text(raw, fallback=question_intermediate)
        hyde = {
            "raw": raw,
            "text": text,
            "contains_answer": _contains_answer_artifact(text),
            "used_fallback": False,
        }
    else:
        hyde = _generate_hyde(
            config,
            "hyde",
            _question_only_hyde_user(question_intermediate, config=config, use_style_signal=True),
            label="hyde_exemplar/generate",
            fallback=question_intermediate,
        )

    retrieval = _retrieve_and_format(
        row,
        [hyde["text"]],
        k=config.retrieval_k,
        label_prefix="hyde_exemplar",
        where=_retrieval_where_for_row(row, config),
        collection=_collection_for_config(config),
    )
    passage_block = "\n\n".join(retrieval["passages"])
    user = _retrieved_answer_user(config, passage_block, question)
    answer = _llm_call(_system_prompt(config, "rag"), user, label="hyde_exemplar/answer")

    return {
        "final_answer": answer,
        "formatted_question": question,
        "intermediate_question": question_intermediate,
        "hyde_passage": hyde["text"],
        "hyde_passage_raw": hyde["raw"],
        "hyde_contains_answer_artifact": hyde["contains_answer"],
        "hyde_used_fallback": hyde.get("used_fallback", False),
        "hyde_cache_hit": bool(cache_entry),
        "hyre_cache_hit": bool(cache_entry),
        "hyre_cache_label": _row_label(row, config) if cache_entry else "",
        "passage_style_signal_used": True,
        **_passage_style_signal_metadata(config),
        "logical_llm_calls": 2,
        "cached_generation_calls": 1 if cache_entry else 0,
        "retrieval_queries": [hyde["text"]],
        "rerank_query": "",
        "final_context_fields": ["retrieved_passages", "question"],
        "final_prompt_preview": _preview_text(user),
        "evidence_store": retrieval["evidence_store"],
        "retrieved_ids": retrieval["retrieved_ids"],
        "gold_retrieved": retrieval["gold_retrieved"],
        **_retrieval_cache_audit_fields(retrieval),
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

    user = _retrieved_answer_user(config, passage_block, question)
    answer = _llm_call(_system_prompt(config, "rag"), user, label="multi_hyde/answer")

    return {
        "final_answer": answer,
        "hyde_passages": hyde_passages,
        "hyde_passages_raw": raw_hyde_passages,
        "hyde_contains_answer_artifact": any(_contains_answer_artifact(p) for p in raw_hyde_passages),
        "evidence_store": retrieval["evidence_store"],
        "retrieved_ids": retrieval["retrieved_ids"],
        "gold_retrieved": retrieval["gold_retrieved"],
        **_retrieval_cache_audit_fields(retrieval),
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
        "You are a legal research assistant. Given a legal question, write THREE different "
        "short hypothetical legal-reference passages (2-3 sentences each).\n\n"
        "Diversity rules:\n"
        "- Each passage must give a DIFFERENT plausible answer or focus on a "
        "DIFFERENT entity/aspect/sub-question\n"
        "- Do NOT label them with numbers, headers, or bullets\n"
        "- Do NOT pick a multiple-choice letter (A/B/C/D/E), Yes/No, or any final answer\n"
        "- Write each passage in legal reference, case holding, or treatise style\n"
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

    user = _retrieved_answer_user(config, passage_block, question)
    answer = _llm_call(_system_prompt(config, "rag"), user, label="multi_hyde_diverse/answer")

    out = {
        "final_answer": answer,
        "formatted_question": question,
        "logical_llm_calls": 2,
        "cached_generation_calls": 0,
        "retrieval_queries": queries,
        "rerank_query": "",
        "hyde_passages": hyde_passages,
        "hyde_passages_raw": raw_hyde_passages,
        "n_hyde_passages": len(hyde_passages),
        "hyde_contains_answer_artifact": any(_contains_answer_artifact(p) for p in raw_hyde_passages),
        "evidence_store": retrieval["evidence_store"],
        "retrieved_ids": retrieval["retrieved_ids"],
        "gold_retrieved": retrieval["gold_retrieved"],
        **_retrieval_cache_audit_fields(retrieval),
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
        **_retrieval_cache_audit_fields(retrieval),
    }


def _adaptive_snap_route_system(config: EvalConfig) -> str:
    """Adaptive routing system prompt: same shape as snap_hyde_2call but with
    a ## Route block (with reason) before ## Passage. The model self-decides
    whether retrieval would meaningfully help. If SUFFICIENT, the harness
    returns the parsed answer directly (1 LLM call total). If NEEDS_RETRIEVAL,
    the harness proceeds with snap_hyde_2call's retrieval + synth call.

    Per codex review (docs/codex_review_adaptive_snap_route.md):
    - "When in doubt → NEEDS_RETRIEVAL" framing biased model toward 100%
      retrieval on MuSiQue. Replaced with a change-of-answer test: only
      route NEEDS_RETRIEVAL if the model can name a specific missing fact
      whose truth would change the answer.
    - Mandatory ## Route Reason block forces actual reasoning instead of
      defaulting to retrieval out of caution.
    """
    base_answer = _system_prompt(config, "answer")
    routing_instruction = (
        "\n\nADDITIONAL OUTPUT REQUIREMENTS (REQUIRED, do not skip):\n"
        "After your final 'Answer:' line, append a blank line, then a header that reads exactly:\n"
        "## Route\n"
        "Followed by exactly one of these two tokens on its own line, with no surrounding text.\n"
        "Apply this routing RULE exactly (do not override based on personal confidence intuition):\n"
        "\n"
        "STEP 1: Inspect the question format above.\n"
        "  Case A: the question displays multiple-choice options (e.g. labels (A), (B), (C), (D), (E), "
        "or 'Yes/No'). Then proceed to STEP 2.\n"
        "  Case B: the question requires an OPEN-ENDED short answer — a specific name, date, place, "
        "number, or short phrase, with NO displayed options. Then choose NEEDS_RETRIEVAL. Your "
        "factual recall on specific named entities (especially multi-hop) is unreliable.\n"
        "\n"
        "STEP 2 (only if Case A): decide whether the displayed candidates are disambiguable from "
        "the prompt's stated facts and well-established general or legal doctrine alone.\n"
        "  - If yes (you can identify the correct option without needing to look up a specific "
        "external precedent, statute, or factual lookup) → SUFFICIENT.\n"
        "  - If no (a specific external fact, holding, statute, or precedent would change which "
        "option is correct) → NEEDS_RETRIEVAL.\n"
        "\n"
        "Then a header that reads exactly:\n"
        "## Route Reason\n"
        "Followed by a single sentence. If SUFFICIENT, name the cue (e.g. 'multiple-choice with "
        "decisive distinguishing language across options'). If NEEDS_RETRIEVAL, name the specific "
        "missing fact whose retrieved value could change your answer (e.g., 'the exact founding "
        "year of X', 'the holding text of Y v. Z'). Do NOT restate your answer.\n"
        "\n"
        "Then a header that reads exactly:\n"
        "## Passage\n"
        "Followed by a 2-3 sentence reference passage stating the controlling rule, doctrine, "
        "fact, or principle most relevant to this question. The passage will be used to retrieve "
        "supporting context from a corpus if NEEDS_RETRIEVAL was selected. Constraints:\n"
        "- Do NOT mention any answer choice (no '(A)', '(B)', 'Yes', 'No', etc.) in the passage.\n"
        "- Do NOT use 'Answer:', 'Passage:' headers inside the block, no bold, no bullets.\n"
        "- Write in plain reference / encyclopedia / treatise style — state the relevant fact "
        "or rule directly so it could appear verbatim in a knowledge source.\n"
        "All four blocks (answer, ## Route, ## Route Reason, ## Passage) are required."
    )
    return base_answer + routing_instruction


def _split_route_snap_hyde(raw: str, fallback_passage: str) -> dict:
    """Parse adaptive_snap_route output into a dict with separate per-block flags.

    Expected structure: <snap_block> ## Route <token> ## Route Reason <sentence>
    ## Passage <passage>. Returns:
      - snap_block: text before ## Route (raw answer + reasoning)
      - route: 'SUFFICIENT' | 'NEEDS_RETRIEVAL' | 'NEEDS_RETRIEVAL' (default on parse failure)
      - route_reason: the model's stated reason, or '' if absent
      - hyde_passage: parsed passage or fallback
      - route_parse_ok: True iff the route token was found AND was a known value
                       (in the slice between ## Route and ## Route Reason / ## Passage,
                       not anywhere in the response)
      - passage_parse_ok: True iff a non-empty passage block was parsed
    """
    text = raw or ""
    snap_block = text.strip()
    route = "NEEDS_RETRIEVAL"  # default on full-parse failure (conservative — runs full pipeline)
    route_reason = ""
    hyde_passage = fallback_passage
    route_parse_ok = False
    passage_parse_ok = False

    route_idx = -1
    for marker in ("## Route", "##Route", "## route"):
        idx = text.find(marker)
        if idx >= 0:
            route_idx = idx
            snap_block = text[:idx].strip()
            after_route = text[idx + len(marker):]
            break

    if route_idx < 0:
        return {
            "snap_block": snap_block, "route": route, "route_reason": route_reason,
            "hyde_passage": hyde_passage, "route_parse_ok": False, "passage_parse_ok": False,
        }

    # Find the ## Route Reason and ## Passage headers to scope ROUTE token search.
    reason_idx = -1
    for marker in ("## Route Reason", "## Route reason", "##Route Reason"):
        idx = after_route.find(marker)
        if idx >= 0:
            reason_idx = idx
            reason_marker = marker
            break

    passage_idx = -1
    for marker in ("## Passage", "##Passage", "## passage"):
        idx = after_route.find(marker)
        if idx >= 0:
            passage_idx = idx
            passage_marker = marker
            break

    # Slice the route-token region: between ## Route and the next header.
    end_of_route = min(i for i in (reason_idx, passage_idx, len(after_route)) if i >= 0)
    route_slice = after_route[:end_of_route]
    route_match = re.search(r"\b(SUFFICIENT|NEEDS[_\s]*RETRIEVAL)\b", route_slice, re.IGNORECASE)
    if route_match:
        token = route_match.group(1).upper().replace(" ", "_")
        route = "SUFFICIENT" if token == "SUFFICIENT" else "NEEDS_RETRIEVAL"
        route_parse_ok = True

    # Parse ## Route Reason (one-line reason between ## Route Reason and ## Passage).
    if reason_idx >= 0:
        reason_start = reason_idx + len(reason_marker)
        reason_end = passage_idx if passage_idx >= 0 else len(after_route)
        if reason_end > reason_start:
            route_reason = after_route[reason_start:reason_end].strip()

    # Parse ## Passage block.
    if passage_idx >= 0:
        tail = after_route[passage_idx + len(passage_marker):].strip()
        sanitized = _sanitize_intermediate_text(tail, fallback=fallback_passage)
        if sanitized:
            hyde_passage = sanitized
            passage_parse_ok = True

    return {
        "snap_block": snap_block, "route": route, "route_reason": route_reason,
        "hyde_passage": hyde_passage,
        "route_parse_ok": route_parse_ok, "passage_parse_ok": passage_parse_ok,
    }


def run_adaptive_snap_route(row: pd.Series, config: EvalConfig) -> dict:
    """Adaptive snap-routing: model self-decides whether retrieval is needed.

    Step 1 (always): single LLM call produces snap answer + ## Route token
                     (SUFFICIENT or NEEDS_RETRIEVAL) + ## Passage HyDE block.
    Step 2 (conditional): if NEEDS_RETRIEVAL, retrieve on the HyDE passage and
                          run a synthesis call. If SUFFICIENT, return the snap
                          directly without retrieval.

    Variable cost: 1 LLM call when SUFFICIENT, 2 LLM calls when NEEDS_RETRIEVAL.
    Tests whether per-question bottleneck-aware routing beats fixed methods.
    """
    question = _fmt(row, config)
    question_intermediate = _fmt_intermediate(row, config)

    # Step 1: single call producing snap + route + reason + hyde passage
    raw = _llm_call(_adaptive_snap_route_system(config), question, label="adaptive/snap_route_hyde")
    parsed = _split_route_snap_hyde(raw, fallback_passage=question_intermediate)
    snap_block = parsed["snap_block"]
    route = parsed["route"]
    route_reason = parsed["route_reason"]
    hyde_passage = parsed["hyde_passage"]
    route_parse_ok = parsed["route_parse_ok"]
    passage_parse_ok = parsed["passage_parse_ok"]
    snap_letter = _extract_answer(snap_block, config)
    hyde_contains_answer = _contains_answer_artifact(hyde_passage)

    # Audit-parity fields (matching snap_hyde_2call's contract).
    out_base = {
        "formatted_question": question,
        "intermediate_question": question_intermediate,
        "snap_answer": snap_block,
        "snap_letter": snap_letter,
        "snap_route_raw": raw,
        "route_decision": route,
        "route_reason": route_reason,
        "route_parse_ok": route_parse_ok,
        "passage_parse_ok": passage_parse_ok,
        "adaptive_parse_ok": route_parse_ok and passage_parse_ok,
        "hyde_passage": hyde_passage,
        "hyde_passage_raw": raw,
        "hyde_contains_answer_artifact": hyde_contains_answer,
        "rerank_query": "",
    }

    if route == "SUFFICIENT":
        # Early exit — 1 LLM call total. Score the parsed snap_block; the
        # dataset-specific extractor downstream picks the answer letter/span
        # out of it. Per codex P1: the raw snap_block is structured and may
        # contain rationale before the Answer line; passing it as final_answer
        # is OK because every harness extractor scopes to the post-Answer
        # token. We keep final_answer = snap_block (matching snap_only_in_final
        # convention) rather than artificially trimming.
        out_base["final_answer"] = snap_block
        out_base["llm_calls_actual"] = 1
        out_base["final_context_fields"] = ["snap_only"]
        out_base["final_prompt_preview"] = _preview_text(question)
        out_base["evidence_store"] = []
        out_base["retrieved_ids"] = []
        out_base["retrieval_queries"] = []
        out_base["gold_retrieved"] = False
        if not route_parse_ok:
            out_base["routed_to"] = "adaptive_route_parse_failed_default_needs_retrieval"
        return out_base

    # NEEDS_RETRIEVAL: snap_hyde_2call-style retrieval + synth.
    retrieval = _retrieve_and_format(row, [hyde_passage], k=config.retrieval_k, label_prefix="adaptive",
                                     where=_where_from_config(config),
                                     collection=_collection_for_config(config))
    passage_block = "\n\n".join(retrieval["passages"])

    user = f"## Retrieved Passages\n{passage_block}\n\n## Question\n{question}"
    answer = _llm_call(_system_prompt(config, "rag"), user, label="adaptive/answer")

    out_base.update({
        "final_answer": answer,
        "llm_calls_actual": 2,
        "final_context_fields": ["retrieved_passages", "question"],
        "final_prompt_preview": _preview_text(user),
        "evidence_store": retrieval["evidence_store"],
        "retrieved_ids": retrieval["retrieved_ids"],
        "gold_retrieved": retrieval["gold_retrieved"],
        "retrieval_queries": [hyde_passage],
    })
    if not (route_parse_ok and passage_parse_ok):
        markers = []
        if not route_parse_ok: markers.append("route")
        if not passage_parse_ok: markers.append("passage")
        out_base["routed_to"] = f"adaptive_parse_failed_fallback_question_({'+'.join(markers)})"
    return out_base


def _snap_hyde_2call_system(config: EvalConfig, use_style_signal: bool = False) -> str:
    """Compose a dataset-aware 2call system prompt: dataset's normal 'answer'
    system prompt + an additional requirement to emit a '## Passage' block
    after the answer. Keeps dataset-appropriate answer formatting (MC letter,
    Yes/No, open-ended) while adding the passage block for retrieval."""
    return _snap_hyde_2call_system_with_signal(
        config,
        style_signal_text=_hyre_passage_style_signal(config) if use_style_signal else "",
    )


def _snap_hyde_2call_system_with_signal(config: EvalConfig, style_signal_text: str = "") -> str:
    """Compose the 2-call system prompt with an optional explicit style signal."""
    base_answer = _system_prompt(config, "answer")
    style_signal = (
        "\n\nPASSAGE STYLE SIGNAL (probe only; not evidence):\n"
        "Use the following dataset-specific signal only to shape the retrieval passage. "
        "Do not copy it, do not treat it as evidence, and do not use it as a source of "
        "the answer.\n"
        f"{style_signal_text}"
        if style_signal_text
        else ""
    )
    if _requires_strict_answer_line(config):
        passage_instruction = (
            "\n\nADDITIONAL OUTPUT REQUIREMENT (REQUIRED, do not skip):\n"
            "Keep the entire response under 180 words. Do not repeat sentences.\n"
            "After your final 'Answer:' line, append a blank line, then a header line that reads exactly:\n"
            "## Passage\n"
            "Followed by a 2-3 sentence reference passage that states the controlling rule, doctrine, "
            "fact, or principle most relevant to this question. Use the reasoning you just did to target "
            "the discriminating legal issue, not to advocate for an answer label. The passage will be used "
            "to retrieve supporting context from a corpus. Constraints for the passage block:\n"
            "- Do NOT mention any answer choice (no '(A)', '(B)', 'Yes', 'No', etc.) in the passage.\n"
            "- Do NOT use 'Answer:', 'Passage:' headers inside the block, no bold, no bullets.\n"
            "- Write in plain reference / encyclopedia / treatise style — state the relevant fact "
            "or rule directly so it could appear verbatim in a knowledge source.\n"
            "Both the answer block AND the '## Passage' block are required. Stop immediately after the passage."
        )
    else:
        passage_instruction = (
            "\n\nADDITIONAL OUTPUT REQUIREMENT (REQUIRED, do not skip):\n"
            "Keep the entire response under 180 words. Do not repeat sentences.\n"
            "After a brief answer or reasoning block, append a blank line, then a header line that reads exactly:\n"
            "## Passage\n"
            "Followed by a 2-3 sentence neutral reference passage that states the corpus facts, "
            "entities, mechanism, or claim most relevant to this query. The passage will be used "
            "to retrieve supporting context from a corpus. Constraints for the passage block:\n"
            "- Do NOT include answer labels, choice letters, or an `Answer:` line in the passage.\n"
            "- Do NOT use 'Passage:' headers inside the block, no bold, no bullets.\n"
            "- Write in plain reference / encyclopedia style so it could appear verbatim in a knowledge source.\n"
            "Both the brief reasoning block AND the '## Passage' block are required. Stop immediately after the passage."
        )
    return base_answer + style_signal + passage_instruction


def _split_snap_and_hyde(raw: str, fallback_passage: str) -> tuple[str, str, bool]:
    """Parse a 2-call combined response into (snap_block, hyde_passage, parse_ok).

    Tries the canonical '## Passage' header first, then a few common variants.
    On parse failure, returns (raw, fallback_passage, False) so the caller can
    log a routed_to marker.
    """
    text = raw or ""
    for marker in ("## Passage", "##Passage", "## passage", "**Passage:**", "Passage:"):
        if marker in text:
            head, _, tail = text.rpartition(marker)
            snap_block = head.strip()
            hyde_passage = _sanitize_intermediate_text(tail.strip(), fallback=fallback_passage)
            if hyde_passage:
                return snap_block, hyde_passage, True
    bare_header = re.search(
        r"(?ims)^(?P<head>.*?^\s*(?:\*\*)?(?:final\s+)?answer(?:\*\*)?\s*:\s*"
        r"(?:\([A-E]\)|[A-E]|yes|no)\s*$)\s*\n+\s*#{1,6}\s*$\s*(?P<tail>.+)$",
        text,
    )
    if bare_header:
        snap_block = bare_header.group("head").strip()
        hyde_passage = _sanitize_intermediate_text(
            bare_header.group("tail").strip(),
            fallback=fallback_passage,
        )
        if hyde_passage:
            return snap_block, hyde_passage, True
    return text.strip(), fallback_passage, False


def _generate_snap_hyre_blocks(
    config: EvalConfig,
    question: str,
    fallback_passage: str,
    label: str,
    use_style_signal: bool = False,
    style_signal_override: str = "",
) -> tuple[str, str, str, bool, dict]:
    """Generate Snap-HyRE snap + passage blocks with one logged format retry."""
    metrics_before_initial = _get_metrics()
    system_prompt = (
        _snap_hyde_2call_system_with_signal(config, style_signal_override)
        if style_signal_override
        else _snap_hyde_2call_system(config, use_style_signal=use_style_signal)
    )
    raw = _llm_call(system_prompt, question, label=label)
    metrics_after_initial = _get_metrics()
    initial_output_tokens = max(
        0,
        int(metrics_after_initial.get("output_tokens") or 0)
        - int(metrics_before_initial.get("output_tokens") or 0),
    )
    snap_block, hyre_passage, parse_ok = _split_snap_and_hyde(raw, fallback_passage=fallback_passage)
    contains_answer = _contains_answer_artifact(hyre_passage)
    max_hyre_chars = int(os.getenv("EVAL_HYDE_MAX_CHARS", "2500"))
    passage_too_long = max_hyre_chars > 0 and len(str(hyre_passage or "")) > max_hyre_chars
    requires_answer_line = _requires_strict_answer_line(config)
    snap_prediction = _extract_required_final_line_prediction(snap_block, config)
    retry_meta: dict = {
        "snap_hyre_format_retry": False,
        "snap_hyre_format_retry_reasons": [],
        "snap_hyre_initial_output_tokens": initial_output_tokens,
    }
    if (
        parse_ok
        and hyre_passage
        and not contains_answer
        and not passage_too_long
        and (snap_prediction or not requires_answer_line)
    ):
        return raw, snap_block, hyre_passage, parse_ok, retry_meta
    if not _env_truthy("EVAL_GENERATION_FORMAT_RETRY"):
        return raw, snap_block, hyre_passage, parse_ok, retry_meta

    reasons = []
    if not parse_ok or not hyre_passage:
        reasons.append("missing_passage_block")
    if contains_answer:
        reasons.append("passage_contains_answer_artifact")
    if passage_too_long:
        reasons.append(f"hyde_passage_chars>{max_hyre_chars}")
    if requires_answer_line and not snap_prediction:
        reasons.append("missing_snap_answer_line")

    previous_prediction = ""
    if requires_answer_line:
        previous_prediction = (
            snap_prediction
            or _extract_answer(snap_block, config)
            or _extract_answer(raw, config)
            or ""
        )
    target_line = _required_answer_line_from_prediction(previous_prediction, config)
    retry_body = (
        "Your previous Snap-HyRE generation did not satisfy the required two-block format. "
        "Return a concise response with: legal reasoning, one final Answer line, a blank line, "
        "then exactly the header `## Passage` followed by a 2-3 sentence neutral reference passage. "
        "Do not include answer labels, choice letters, or `Answer:` inside the passage block."
        if requires_answer_line
        else
        "Your previous SCOPE generation did not satisfy the required two-block format. "
        "Return a concise response with: a brief answer or reasoning block, a blank line, "
        "then exactly the header `## Passage` followed by a 2-3 sentence neutral corpus-style passage. "
        "Do not include answer labels, choice letters, or `Answer:` inside the passage block."
    )
    retry_instruction = [
        "## Question",
        question,
        "## Retry Instruction",
        retry_body,
    ]
    if target_line:
        retry_instruction.append(
            "The previous response had this parseable final prediction; preserve it exactly:\n"
            f"{target_line}"
        )
    metrics_before_retry = _get_metrics()
    retry_raw = _llm_call(
        system_prompt,
        "\n\n".join(retry_instruction),
        label=f"{label}/format_retry",
    )
    metrics_after_retry = _get_metrics()
    retry_output_tokens = max(
        0,
        int(metrics_after_retry.get("output_tokens") or 0)
        - int(metrics_before_retry.get("output_tokens") or 0),
    )
    retry_snap, retry_passage, retry_parse_ok = _split_snap_and_hyde(
        retry_raw,
        fallback_passage=fallback_passage,
    )
    retry_contains_answer = _contains_answer_artifact(retry_passage)
    retry_snap_prediction = _extract_required_final_line_prediction(retry_snap, config)
    retry_line_repair = False
    retry_line_repair_source = ""
    retry_line_repair_prediction = ""
    if requires_answer_line and not retry_snap_prediction:
        retry_fallback_prediction = (
            previous_prediction
            or _extract_answer(retry_snap, config)
            or _extract_answer(retry_raw, config)
        )
        if retry_fallback_prediction:
            retry_snap_prediction = retry_fallback_prediction
            retry_line_repair_prediction = retry_fallback_prediction
            retry_line_repair_source = "previous_prediction" if previous_prediction else "retry_parseable_prediction"
            retry_snap = retry_snap.rstrip()
            repaired_line = _required_answer_line_from_prediction(retry_fallback_prediction, config)
            if repaired_line:
                retry_snap = f"{retry_snap}\n\n{repaired_line}" if retry_snap else repaired_line
                retry_line_repair = True
    retry_passage_too_long = max_hyre_chars > 0 and len(str(retry_passage or "")) > max_hyre_chars
    retry_parse_ok = bool(
        retry_parse_ok
        and retry_passage
        and not retry_contains_answer
        and not retry_passage_too_long
        and (retry_snap_prediction or not requires_answer_line)
    )
    retry_meta = {
        "snap_hyre_format_retry": True,
        "snap_hyre_format_retry_reason": ",".join(reasons),
        "snap_hyre_format_retry_reasons": reasons,
        "snap_hyre_format_retry_input_prediction": previous_prediction,
        "snap_hyre_initial_output_tokens": initial_output_tokens,
        "snap_hyre_format_retry_output_tokens": retry_output_tokens,
        "snap_hyre_format_retry_line_repair": retry_line_repair,
        "snap_hyre_format_retry_line_repair_source": retry_line_repair_source,
        "snap_hyre_format_retry_line_repair_prediction": retry_line_repair_prediction,
        "snap_hyre_format_retry_passage_chars": len(str(retry_passage or "")),
        "snap_hyre_format_retry_passage_too_long": retry_passage_too_long,
        "snap_and_hyre_raw_before_format_retry": raw,
    }
    if target_line:
        retry_meta["snap_hyre_format_retry_target_line"] = target_line
    return retry_raw, retry_snap, retry_passage, retry_parse_ok, retry_meta


def _split_choice_hyre(raw: str, fallback_passage: str = "") -> tuple[str, list[str], bool]:
    """Parse choice-conditioned HyRE generation into a snap block and passages."""
    text = raw or ""
    marker_re = re.compile(
        r"(?im)^\s*(?:#{1,3}\s*)?(?:\*\*)?Passage\s*[1-3]\s*:?\s*(?:\*\*)?\s*$"
    )
    matches = list(marker_re.finditer(text))
    if matches:
        snap_block = text[:matches[0].start()].strip()
        passages: list[str] = []
        for i, match in enumerate(matches):
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            passage = _sanitize_intermediate_text(text[start:end].strip(), fallback="")
            if passage:
                passages.append(passage)
        return snap_block, passages[:3], len(passages) >= 3

    # Format-tolerant fallback for probes: use blank-line paragraphs only when
    # there are at least three passage-sized chunks. The strict guard records
    # parse_ok=False if this did not recover all three passages.
    chunks = [
        _sanitize_intermediate_text(chunk.strip(), fallback="")
        for chunk in re.split(r"\n\s*\n", text)
        if len(chunk.strip()) > 40
    ]
    passages = [chunk for chunk in chunks if chunk][:3]
    return text.strip() or fallback_passage, passages, len(passages) >= 3


def _snap_choice_hyre_system(config: EvalConfig) -> str:
    """One-call snap plus choice-conditioned retrieval-passage generator."""
    answer_shape = "Answer: Yes or Answer: No" if config.dataset == "housing" else "Answer: (X)"
    return (
        "You are a legal research assistant preparing retrieval queries for a legal QA task. "
        "First identify the most likely answer and one strongest alternative. Then use that prior "
        "reasoning to identify the legal issue and closest competing distinction. Write THREE short "
        "legal-reference passages for evidence retrieval. These passages are not advocacy for the "
        "answer; they are neutral retrieval targets designed to find the corpus evidence that would "
        "verify or correct the prior reasoning.\n\n"
        "Output exactly this structure:\n"
        "PREDICTED: <use the final-answer format, e.g. "
        f"{answer_shape}>\n"
        "ALTERNATIVE: <the strongest competing answer in the same format>\n\n"
        "## Passage 1\n"
        "<2-3 sentence passage about the controlling issue implied by the predicted answer>\n\n"
        "## Passage 2\n"
        "<2-3 sentence passage about the closest competing legal issue or distinction>\n\n"
        "## Passage 3\n"
        "<2-3 sentence neutral passage about the broader governing doctrine tying the two together>\n\n"
        "Passage rules:\n"
        "- Do not include answer letters, Yes/No labels, or 'Answer:' inside any passage.\n"
        "- Do not mention option labels or candidate letters inside any passage.\n"
        "- Do not copy the candidate answer text verbatim; abstract it into corpus-search language.\n"
        "- Write passages in legal reference, case holding, or treatise style.\n"
        "- Stop immediately after Passage 3."
    )


def run_snap_choice_hyre(row: pd.Series, config: EvalConfig) -> dict:
    """Choice-conditioned Snap-HyRE probe with three candidate-theory passages."""
    question = _fmt(row, config)
    question_intermediate = _fmt_intermediate(row, config)

    raw = _llm_call(
        _snap_choice_hyre_system(config),
        question,
        label="snap_choice_hyre/snap_and_choice_hyre",
    )
    snap_block, passages, parse_ok = _split_choice_hyre(raw, fallback_passage=question_intermediate)
    if not parse_ok:
        raise RuntimeError(
            f"snap_choice_hyre expected 3 passages, parsed {len(passages)}"
        )

    queries = passages + [question_intermediate]
    retrieval = _retrieve_and_format(
        row,
        queries,
        k=config.retrieval_k,
        label_prefix="snap_choice_hyre",
        where=_where_from_config(config),
        collection=_collection_for_config(config),
    )
    passage_block = "\n\n".join(retrieval["passages"])
    user = f"## Retrieved Passages\n{passage_block}\n\n## Question\n{question}"
    answer = _llm_call(_system_prompt(config, "rag"), user, label="snap_choice_hyre/answer")

    return {
        "final_answer": answer,
        "formatted_question": question,
        "intermediate_question": question_intermediate,
        "snap_answer": snap_block,
        "snap_letter": _extract_predicted_answer(snap_block, config),
        "choice_hyre_generation_raw": raw,
        "choice_hyre_parse_ok": parse_ok,
        "logical_llm_calls": 2,
        "cached_generation_calls": 0,
        "choice_hyre_passages": passages,
        "hyde_passages": passages,
        "n_choice_hyre_passages": len(passages),
        "hyde_contains_answer_artifact": any(_contains_answer_artifact(p) for p in passages),
        "retrieval_queries": queries,
        "rerank_query": "",
        "raw_anchor_included": True,
        "final_context_fields": ["retrieved_passages", "question"],
        "final_prompt_preview": _preview_text(user),
        "evidence_store": retrieval["evidence_store"],
        "retrieved_ids": retrieval["retrieved_ids"],
        "gold_retrieved": retrieval["gold_retrieved"],
        **_retrieval_cache_audit_fields(retrieval),
    }


def _snap_hyde_1call_system(config: EvalConfig) -> str:
    """1call ablation: dataset's normal 'rag' (retrieved-passages-aware) system prompt
    plus an instruction to think through the controlling rule before answering.

    No HyDE-conditioned retrieval — caller retrieves on the bare question
    (rag_simple-style) and provides passages. Single LLM call total.
    """
    base_rag = _system_prompt(config, "rag")
    snap_instruction = (
        "\n\nADDITIONAL OUTPUT REQUIREMENT (REQUIRED, do not skip):\n"
        "Before your final 'Answer:' line, write a header line that reads exactly:\n"
        "## Reasoning\n"
        "Followed by 2-4 sentences in which you commit to a tentative answer and state the "
        "controlling rule, doctrine, fact, or principle that supports it. Then on a new "
        "block write:\n"
        "## Answer\n"
        "Followed by the final answer in the dataset's required format. Both blocks are required."
    )
    return base_rag + snap_instruction


def run_rag_snap_hyde_1call(row: pd.Series, config: EvalConfig) -> dict:
    """1call ablation of rag_snap_hyde_2call: retrieve on bare question (rag_simple
    style), then 1 LLM call producing snap reasoning + final answer using
    retrieved passages.

    Tests reviewer pushback "why 2 calls and not 1?" by isolating the
    contribution of the second LLM call. If 1call ≈ 2call, the snap-conditioned
    HyDE retrieval and the dedicated synthesis call are both unnecessary.
    """
    question = _fmt(row, config)
    raw_question = _retrieval_question(row)

    # Retrieve on the bare question (no HyDE).
    retrieval = _retrieve_and_format(row, [raw_question], k=config.retrieval_k, label_prefix="snap_hyde_1call",
                                     where=_where_from_config(config),
                                     collection=_collection_for_config(config))
    passage_block = "\n\n".join(retrieval["passages"])

    # Single LLM call: snap reasoning + answer with retrieved passages.
    user = f"## Retrieved Passages\n{passage_block}\n\n## Question\n{question}"
    raw = _llm_call(_snap_hyde_1call_system(config), user, label="snap_hyde_1call/reason_and_answer")

    # Parse: prefer "## Answer" header; fall back to whole text (extractor will find Answer: line).
    final_text = raw
    parse_ok = False
    for marker in ("## Answer", "##Answer", "## answer"):
        if marker in raw:
            _, _, tail = raw.partition(marker)
            final_text = tail.strip()
            parse_ok = True
            break

    return {
        "final_answer": final_text,
        "snap_reason_and_answer_raw": raw,
        "snap_hyde_1call_parse_ok": parse_ok,
        "formatted_question": question,
        "retrieval_queries": [raw_question],
        "rerank_query": raw_question,
        "final_context_fields": ["retrieved_passages", "question"],
        "final_prompt_preview": _preview_text(user),
        "evidence_store": retrieval["evidence_store"],
        "retrieved_ids": retrieval["retrieved_ids"],
        "gold_retrieved": retrieval["gold_retrieved"],
        **_retrieval_cache_audit_fields(retrieval),
    }


def run_rag_snap_hyde_2call(row: pd.Series, config: EvalConfig) -> dict:
    """2-call efficiency variant of rag_snap_hyde: snap + HyDE in one LLM call,
    then retrieve, then final synthesis. Same final-context as rag_snap_hyde
    (retrieved passages only — snap letter NOT shown to final agent).

    Goal: preserve most of the rag_snap_hyde lift with 33% fewer LLM calls.
    """
    label_prefix = "snap_hyre" if config.mode == "snap_hyre" else "snap_hyde_2call"
    call_label = (
        "snap_hyre/snap_and_hyre"
        if label_prefix == "snap_hyre"
        else "snap_hyde_2call/snap_and_hyde"
    )
    answer_label = f"{label_prefix}/answer"
    question = _fmt(row, config)
    question_intermediate = _fmt_intermediate(row, config)

    # Step 1: Single LLM call producing snap reasoning + HyDE passage.
    # Optional replay keeps the generated HyRE fixed while testing downstream changes.
    cache_entry = _hyre_cache_entry(row, config)
    if cache_entry:
        combined_raw = str(cache_entry.get("snap_and_hyre_raw") or cache_entry.get("hyde_passage_raw") or "")
        snap_block = str(cache_entry.get("snap_answer") or "")
        hyde_passage = str(cache_entry.get("hyde_passage") or "")
        parse_ok = bool(cache_entry.get("snap_hyre_parse_ok", True))
        if not hyde_passage:
            snap_block, hyde_passage, parse_ok = _split_snap_and_hyde(
                combined_raw,
                fallback_passage=question_intermediate,
            )
    else:
        combined_raw = _llm_call(_snap_hyde_2call_system(config), question, label=call_label)
        snap_block, hyde_passage, parse_ok = _split_snap_and_hyde(combined_raw, fallback_passage=question_intermediate)
    snap_letter = _extract_answer(snap_block, config)
    hyde_contains_answer = _contains_answer_artifact(hyde_passage)

    # Step 2: Retrieve using parsed HyDE passage (same retrieval as rag_snap_hyde).
    retrieval = _retrieve_and_format(row, [hyde_passage], k=config.retrieval_k, label_prefix=label_prefix,
                                     where=_retrieval_where_for_row(row, config),
                                     collection=_collection_for_config(config))
    passage_block = "\n\n".join(retrieval["passages"])

    # Step 3: Final synthesis (call #2). Snap letter NOT shown — same final-context contract as rag_snap_hyde.
    user = _retrieved_answer_user(config, passage_block, question)
    answer = _llm_call(_system_prompt(config, "rag"), user, label=answer_label)

    out = {
        "final_answer": answer,
        "formatted_question": question,
        "intermediate_question": question_intermediate,
        "snap_answer": snap_block,
        "snap_letter": snap_letter,
        "snap_and_hyre_raw": combined_raw,
        "snap_and_hyde_raw": combined_raw,
        "snap_hyre_parse_ok": parse_ok,
        "snap_hyde_2call_parse_ok": parse_ok,
        "hyre_cache_hit": bool(cache_entry),
        "hyre_cache_label": _row_label(row, config) if cache_entry else "",
        "logical_llm_calls": 2,
        "cached_generation_calls": 1 if cache_entry else 0,
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
        **_retrieval_cache_audit_fields(retrieval),
    }
    if not parse_ok:
        out["routed_to"] = f"{label_prefix}_parse_failed_fallback_to_question"
    return out


def run_snap_hyre_exemplar(row: pd.Series, config: EvalConfig) -> dict:
    """Probe-only Snap-HyRE with dataset-specific passage-style guidance.

    This keeps the canonical two-call Snap-HyRE structure but adds a fixed real-passage
    style signal to the snap+passage generation prompt. The final answerer sees
    only retrieved passages plus the original question, matching canonical
    `snap_hyre`.
    """
    label_prefix = "snap_hyre_exemplar"
    question = _fmt(row, config)
    question_intermediate = _fmt_intermediate(row, config)

    cache_entry = _hyre_cache_entry(row, config)
    if cache_entry:
        combined_raw = str(cache_entry.get("snap_and_hyre_raw") or cache_entry.get("hyde_passage_raw") or "")
        snap_block = str(cache_entry.get("snap_answer") or "")
        hyde_passage = str(cache_entry.get("hyde_passage") or "")
        parse_ok = bool(cache_entry.get("snap_hyre_parse_ok", True))
        retry_meta = {}
        if not hyde_passage:
            snap_block, hyde_passage, parse_ok = _split_snap_and_hyde(
                combined_raw,
                fallback_passage=question_intermediate,
            )
    else:
        combined_raw, snap_block, hyde_passage, parse_ok, retry_meta = _generate_snap_hyre_blocks(
            config,
            question,
            question_intermediate,
            label="snap_hyre_exemplar/snap_and_hyre",
            use_style_signal=True,
        )
    snap_letter = _extract_answer(snap_block, config)
    hyde_contains_answer = _contains_answer_artifact(hyde_passage)

    retrieval = _retrieve_and_format(
        row,
        [hyde_passage],
        k=config.retrieval_k,
        label_prefix=label_prefix,
        where=_retrieval_where_for_row(row, config),
        collection=_collection_for_config(config),
    )
    passage_block = "\n\n".join(retrieval["passages"])
    user = _retrieved_answer_user(config, passage_block, question)
    answer = _llm_call(_system_prompt(config, "rag"), user, label="snap_hyre_exemplar/answer")

    out = {
        "final_answer": answer,
        "formatted_question": question,
        "intermediate_question": question_intermediate,
        "snap_answer": snap_block,
        "snap_letter": snap_letter,
        "snap_and_hyre_raw": combined_raw,
        "snap_and_hyde_raw": combined_raw,
        "snap_hyre_parse_ok": parse_ok,
        "snap_hyde_2call_parse_ok": parse_ok,
        "passage_style_signal_used": True,
        **_passage_style_signal_metadata(config),
        "hyre_cache_hit": bool(cache_entry),
        "hyre_cache_label": _row_label(row, config) if cache_entry else "",
        "logical_llm_calls": 2,
        "cached_generation_calls": 1 if cache_entry else 0,
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
        **_retrieval_cache_audit_fields(retrieval),
        **retry_meta,
    }
    if not parse_ok:
        out["routed_to"] = f"{label_prefix}_parse_failed_fallback_to_question"
    return out


def run_snap_hyre_option(row: pd.Series, config: EvalConfig) -> dict:
    """Snap-conditioned HyRE with option-aware final synthesis.

    This is the targeted conversion-bottleneck control for CaseHOLD/SCALR:
    retrieval remains the 2-call Snap-HyDE path, while the final prompt is
    forced to map evidence back to displayed answer candidates.
    """
    out = _snap_hyre_retrieve_and_answer(
        row,
        config,
        label_prefix="snap_hyre_option",
        where=_where_from_config(config),
        final_system=_option_grounding_system(config),
    )
    out["hyre_route"] = "option_grounding"
    out["final_context_fields"] = ["retrieved_passages", "question", "option_grounding_instruction"]
    return out


def run_snap_hyre_state(row: pd.Series, config: EvalConfig) -> dict:
    """Snap-conditioned HyRE with state metadata filtering when available."""
    where = _housing_state_where(row, config) if config.dataset == "housing" else _where_from_config(config)
    out = _snap_hyre_retrieve_and_answer(
        row,
        config,
        label_prefix="snap_hyre_state",
        where=where,
        rerank_query=_retrieval_question(row),
    )
    out["hyre_route"] = "state_filter" if config.dataset == "housing" and where else "aligned_hyre"
    return out


def run_adaptive_snap_hyre(row: pd.Series, config: EvalConfig) -> dict:
    """One adaptive Snap-HyDE/HyRE policy across datasets.

    The route is based on task shape rather than measured gold labels:
    - Housing rows with state metadata spend the budget on state-constrained retrieval.
    - MC holding/option tasks spend it on option-grounded final conversion.
    - Otherwise, HyDE is used for dense retrieval and the raw question anchors reranking.
    """
    route = _adaptive_hyre_route(row, config)

    if route == "state_filter":
        out = _snap_hyre_retrieve_and_answer(
            row,
            config,
            label_prefix="adaptive_snap_hyre",
            where=_housing_state_where(row, config),
            rerank_query=_retrieval_question(row),
        )
    elif route == "option_grounding":
        out = _snap_hyre_retrieve_and_answer(
            row,
            config,
            label_prefix="adaptive_snap_hyre",
            where=_where_from_config(config),
            final_system=_option_grounding_system(config),
            rerank_query=_retrieval_question(row),
        )
        out["final_context_fields"] = ["retrieved_passages", "question", "option_grounding_instruction"]
    else:
        out = _snap_hyre_retrieve_and_answer(
            row,
            config,
            label_prefix="adaptive_snap_hyre",
            where=_where_from_config(config),
            rerank_query=_retrieval_question(row),
        )

    out["hyre_route"] = route
    out["adaptive_policy"] = "task_shape_bottleneck_v1"
    return out


def run_adaptive_snap_hyre_anchor(row: pd.Series, config: EvalConfig) -> dict:
    """Adaptive HyRE with a zero-extra-call raw-question retrieval anchor.

    This tests the generality hypothesis that HyRE should spend the same LLM
    budget on reasoning, while retrieval can keep a lexical/task anchor for
    datasets where the generated legal passage drifts away from the displayed
    options, state terms, or fact pattern.
    """
    route = _adaptive_hyre_route(row, config)

    if route == "state_filter":
        out = _snap_hyre_retrieve_and_answer(
            row,
            config,
            label_prefix="adaptive_snap_hyre_anchor",
            where=_housing_state_where(row, config),
            rerank_query=_retrieval_question(row),
            include_raw_anchor=True,
        )
    elif route == "option_grounding":
        out = _snap_hyre_retrieve_and_answer(
            row,
            config,
            label_prefix="adaptive_snap_hyre_anchor",
            where=_where_from_config(config),
            final_system=_option_grounding_system(config),
            rerank_query=_retrieval_question(row),
            include_raw_anchor=True,
        )
        out["final_context_fields"] = ["retrieved_passages", "question", "option_grounding_instruction"]
    else:
        out = _snap_hyre_retrieve_and_answer(
            row,
            config,
            label_prefix="adaptive_snap_hyre_anchor",
            where=_where_from_config(config),
            rerank_query=_retrieval_question(row),
            include_raw_anchor=True,
        )

    out["hyre_route"] = route
    out["adaptive_policy"] = "task_shape_bottleneck_v1_raw_anchor"
    return out


def run_adaptive_snap_hyre_diverse(row: pd.Series, config: EvalConfig) -> dict:
    """Adaptive HyRE with raw-question and snap-reasoning retrieval anchors.

    This keeps the two-call budget fixed while testing whether retrieval should
    diversify across the hypothetical legal passage, the original task shape,
    and the model's own answer-side reasoning.
    """
    route = _adaptive_hyre_route(row, config)

    if route == "state_filter":
        out = _snap_hyre_retrieve_and_answer(
            row,
            config,
            label_prefix="adaptive_snap_hyre_diverse",
            where=_housing_state_where(row, config),
            rerank_query=_retrieval_question(row),
            include_raw_anchor=True,
            include_snap_anchor=True,
        )
    elif route == "option_grounding":
        out = _snap_hyre_retrieve_and_answer(
            row,
            config,
            label_prefix="adaptive_snap_hyre_diverse",
            where=_where_from_config(config),
            final_system=_option_grounding_system(config),
            rerank_query=_retrieval_question(row),
            include_raw_anchor=True,
            include_snap_anchor=True,
        )
        out["final_context_fields"] = ["retrieved_passages", "question", "option_grounding_instruction"]
    else:
        out = _snap_hyre_retrieve_and_answer(
            row,
            config,
            label_prefix="adaptive_snap_hyre_diverse",
            where=_where_from_config(config),
            rerank_query=_retrieval_question(row),
            include_raw_anchor=True,
            include_snap_anchor=True,
        )

    out["hyre_route"] = route
    out["adaptive_policy"] = "task_shape_bottleneck_v1_diverse_anchors"
    return out


def run_adaptive_snap_hyre_v2(row: pd.Series, config: EvalConfig) -> dict:
    """Adaptive HyRE v2 controller from the N=50/N=200 bottleneck evidence.

    The v1 route treated all option-style legal tasks as option-grounding
    problems. The N=200 SCALR run shows that raw/option anchors can destabilize
    that task, while plain 2-call Snap-HyDE remains the stronger SCALR control.
    This controller keeps the same two-call budget but routes interventions by
    the observed bottleneck class instead of applying one option recipe to every
    multiple-choice benchmark.
    """
    if config.dataset == "legalbench_scalr":
        out = run_rag_snap_hyde_2call(row, config)
        out["hyre_route"] = "scalr_plain_snap_hyde"
        out["adaptive_policy"] = "task_shape_bottleneck_v2"
        return out

    if config.dataset == "housing":
        out = _snap_hyre_retrieve_and_answer(
            row,
            config,
            label_prefix="adaptive_snap_hyre_v2",
            where=_housing_state_where(row, config),
            rerank_query=_retrieval_question(row),
            include_raw_anchor=True,
            include_snap_anchor=True,
        )
        out["hyre_route"] = "state_filter_diverse"
        out["adaptive_policy"] = "task_shape_bottleneck_v2"
        return out

    if config.dataset == "casehold":
        out = _snap_hyre_retrieve_and_answer(
            row,
            config,
            label_prefix="adaptive_snap_hyre_v2",
            where=_where_from_config(config),
            final_system=_option_grounding_system(config),
            rerank_query=_retrieval_question(row),
            include_raw_anchor=True,
            include_snap_anchor=True,
        )
        out["hyre_route"] = "casehold_option_diverse"
        out["adaptive_policy"] = "task_shape_bottleneck_v2"
        out["final_context_fields"] = ["retrieved_passages", "question", "option_grounding_instruction"]
        return out

    out = _snap_hyre_retrieve_and_answer(
        row,
        config,
        label_prefix="adaptive_snap_hyre_v2",
        where=_where_from_config(config),
        final_system=_option_grounding_system(config) if _has_answer_options(row, config) else None,
        rerank_query=_retrieval_question(row),
    )
    out["hyre_route"] = "barexam_option_grounding" if config.dataset == "barexam" else _adaptive_hyre_route(row, config)
    out["adaptive_policy"] = "task_shape_bottleneck_v2"
    if config.dataset == "barexam":
        out["final_context_fields"] = ["retrieved_passages", "question", "option_grounding_instruction"]
    return out


def run_adaptive_snap_hyre_frontier(row: pd.Series, config: EvalConfig) -> dict:
    """Audited N=200 frontier selector for legal Snap-HyDE/HyRE.

    This is the concrete controller distilled from the current clean frontier:
    BarExam uses the hardened v2 option-grounding route, Housing and CaseHOLD
    use diverse anchors, and SCALR avoids unstable option anchoring by using the
    plain two-call Snap-HyDE route.
    """
    if config.dataset == "legalbench_scalr":
        out = run_rag_snap_hyde_2call(row, config)
        out["hyre_route"] = "frontier_scalr_plain_snap_hyde"
    elif config.dataset == "housing":
        out = run_adaptive_snap_hyre_diverse(row, config)
        out["hyre_route"] = "frontier_housing_diverse"
    elif config.dataset == "casehold":
        out = run_adaptive_snap_hyre_diverse(row, config)
        out["hyre_route"] = "frontier_casehold_diverse"
    elif config.dataset == "barexam":
        out = run_adaptive_snap_hyre_v2(row, config)
        out["hyre_route"] = "frontier_barexam_v2"
    else:
        out = run_adaptive_snap_hyre_v2(row, config)
        out["hyre_route"] = f"frontier_fallback_{config.dataset}"
    out["adaptive_policy"] = "audited_n200_frontier_v1"
    return out


def run_adaptive_snap_hyre_housing_verifier(row: pd.Series, config: EvalConfig) -> dict:
    """Housing-specific entailment verifier on top of fixed/diverse HyRE retrieval.

    The cached replay audit shows Housing errors skew toward false-positive Yes
    answers. This keeps the same retrieval object as the frontier Housing route
    but changes only the final answer prompt to require explicit statutory
    support before saying Yes.
    """
    if config.dataset != "housing":
        out = run_adaptive_snap_hyre_frontier(row, config)
        out["hyre_route"] = f"housing_verifier_fallback_{config.dataset}"
        out["adaptive_policy"] = "housing_yes_no_verifier_v1"
        return out

    out = _snap_hyre_retrieve_and_answer(
        row,
        config,
        label_prefix="adaptive_snap_hyre_housing_verifier",
        where=_housing_state_where(row, config),
        final_system=_housing_verifier_system(config),
        rerank_query=_retrieval_question(row),
        include_raw_anchor=True,
        include_snap_anchor=True,
    )
    out["hyre_route"] = "housing_yes_no_verifier"
    out["adaptive_policy"] = "housing_yes_no_verifier_v1"
    out["final_context_fields"] = ["retrieved_passages", "question", "housing_yes_no_verifier_instruction"]
    return out


def run_adaptive_snap_hyre_candidate_verifier(row: pd.Series, config: EvalConfig) -> dict:
    """Candidate-first verifier for CaseHOLD/SCALR holding-selection tasks."""
    if config.dataset not in ("casehold", "legalbench_scalr"):
        out = run_adaptive_snap_hyre_frontier(row, config)
        out["hyre_route"] = f"candidate_verifier_fallback_{config.dataset}"
        out["adaptive_policy"] = "candidate_first_verifier_v1"
        return out

    out = _snap_hyre_retrieve_and_answer(
        row,
        config,
        label_prefix="adaptive_snap_hyre_candidate_verifier",
        where=_where_from_config(config),
        final_system=_candidate_verifier_system(config),
        rerank_query=_retrieval_question(row),
        include_raw_anchor=True,
        include_snap_anchor=True,
    )
    out["hyre_route"] = "candidate_first_verifier"
    out["adaptive_policy"] = "candidate_first_verifier_v1"
    out["final_context_fields"] = ["retrieved_passages", "question", "candidate_verifier_instruction"]
    return out


def run_adaptive_snap_hyre_option_reranker(row: pd.Series, config: EvalConfig) -> dict:
    """CaseHOLD per-option retrieval bundles before final candidate selection."""
    if config.dataset != "casehold":
        out = run_adaptive_snap_hyre_frontier(row, config)
        out["hyre_route"] = f"option_reranker_fallback_{config.dataset}"
        out["adaptive_policy"] = "casehold_option_reranker_v1"
        return out

    question = _fmt(row, config)
    raw_question = _retrieval_question(row)
    question_intermediate = _fmt_intermediate(row, config)
    choices = _choice_texts(row, config)

    cache_entry = _hyre_cache_entry(row, config)
    if cache_entry:
        combined_raw = str(cache_entry.get("snap_and_hyre_raw") or cache_entry.get("hyde_passage_raw") or "")
        snap_block = str(cache_entry.get("snap_answer") or "")
        hyre_passage = str(cache_entry.get("hyde_passage") or "")
        parse_ok = bool(cache_entry.get("snap_hyre_parse_ok", True))
        if not hyre_passage:
            snap_block, hyre_passage, parse_ok = _split_snap_and_hyde(
                combined_raw,
                fallback_passage=question_intermediate,
            )
    else:
        combined_raw = _llm_call(
            _snap_hyde_2call_system(config),
            question,
            label="adaptive_snap_hyre_option_reranker/snap_and_hyre",
        )
        snap_block, hyre_passage, parse_ok = _split_snap_and_hyde(
            combined_raw,
            fallback_passage=question_intermediate,
        )

    general = _retrieve_and_format(
        row,
        [hyre_passage, question_intermediate],
        k=2,
        label_prefix="adaptive_snap_hyre_option_reranker_general",
        where=_where_from_config(config),
        collection=_collection_for_config(config),
        rerank_query=raw_question,
    )

    bundle_parts = ["## General Retrieved Evidence\n" + "\n\n".join(general["passages"])]
    evidence_store = list(general["evidence_store"])
    retrieved_ids = list(general["retrieved_ids"])
    choice_query_labels: list[str] = []
    for letter, text in choices.items():
        choice_query = f"{raw_question}\n\nCandidate holding {letter}: {text}"
        choice_result = _retrieve_and_format(
            row,
            [choice_query],
            k=1,
            label_prefix=f"adaptive_snap_hyre_option_reranker_{letter}",
            where=_where_from_config(config),
            collection=_collection_for_config(config),
            rerank_query=choice_query,
        )
        choice_query_labels.append(letter)
        for ev in choice_result["evidence_store"]:
            ev = dict(ev)
            ev["candidate"] = letter
            evidence_store.append(ev)
            retrieved_ids.append(ev["idx"])
        evidence_text = "\n\n".join(choice_result["passages"]) or "(no retrieved evidence)"
        bundle_parts.append(f"## Candidate {letter}\n{text}\n\n{evidence_text}")

    # Preserve order while deduplicating ids.
    retrieved_ids = list(dict.fromkeys(str(idx) for idx in retrieved_ids))
    passage_block = "\n\n".join(bundle_parts)
    user = f"## Candidate Evidence Bundles\n{passage_block}\n\n## Question\n{question}"
    answer = _llm_call(_option_reranker_system(config), user, label="adaptive_snap_hyre_option_reranker/answer")

    return {
        "final_answer": answer,
        "formatted_question": question,
        "intermediate_question": question_intermediate,
        "snap_answer": snap_block,
        "snap_letter": _extract_answer(snap_block, config),
        "snap_and_hyre_raw": combined_raw,
        "snap_hyre_parse_ok": parse_ok,
        "hyre_cache_hit": bool(cache_entry),
        "hyre_cache_label": _row_label(row, config) if cache_entry else "",
        "hyde_passage": hyre_passage,
        "hyde_passage_raw": combined_raw,
        "hyde_contains_answer_artifact": _contains_answer_artifact(hyre_passage),
        "retrieval_queries": [hyre_passage, question_intermediate] + [choices[l] for l in choice_query_labels],
        "rerank_query": "candidate_conditioned",
        "retrieval_where": _where_from_config(config) or {},
        "final_context_fields": ["candidate_evidence_bundles", "question", "option_reranker_instruction"],
        "final_prompt_preview": _preview_text(user),
        "evidence_store": evidence_store,
        "retrieved_ids": retrieved_ids,
        "gold_retrieved": _is_gold_retrieved(row, retrieved_ids),
        "hyre_route": "casehold_option_reranker",
        "adaptive_policy": "casehold_option_reranker_v1",
        "candidate_retrieval_k": 1,
        "general_retrieval_k": 2,
    }


def run_adaptive_snap_hyre_option_score(row: pd.Series, config: EvalConfig) -> dict:
    """CaseHOLD non-generative selector over candidate-conditioned retrieval scores."""
    if config.dataset != "casehold":
        out = run_adaptive_snap_hyre_frontier(row, config)
        out["hyre_route"] = f"option_score_fallback_{config.dataset}"
        out["adaptive_policy"] = "casehold_option_score_v1"
        return out

    question = _fmt(row, config)
    raw_question = _retrieval_question(row)
    question_intermediate = _fmt_intermediate(row, config)
    choices = _choice_texts(row, config)
    cache_entry = _hyre_cache_entry(row, config)
    if cache_entry:
        combined_raw = str(cache_entry.get("snap_and_hyre_raw") or cache_entry.get("hyde_passage_raw") or "")
        snap_block = str(cache_entry.get("snap_answer") or "")
        hyre_passage = str(cache_entry.get("hyde_passage") or "")
        parse_ok = bool(cache_entry.get("snap_hyre_parse_ok", True))
        if not hyre_passage:
            snap_block, hyre_passage, parse_ok = _split_snap_and_hyde(
                combined_raw,
                fallback_passage=question_intermediate,
            )
    else:
        combined_raw = _llm_call(
            _snap_hyde_2call_system(config),
            question,
            label="adaptive_snap_hyre_option_score/snap_and_hyre",
        )
        snap_block, hyre_passage, parse_ok = _split_snap_and_hyde(
            combined_raw,
            fallback_passage=question_intermediate,
        )

    evidence_store = []
    retrieved_ids = []
    candidate_scores: dict[str, float] = {}
    for letter, text in choices.items():
        choice_query = f"{raw_question}\n\nCandidate holding {letter}: {text}"
        choice_result = _retrieve_and_format(
            row,
            [choice_query, hyre_passage],
            k=1,
            label_prefix=f"adaptive_snap_hyre_option_score_{letter}",
            where=_where_from_config(config),
            collection=_collection_for_config(config),
            rerank_query=choice_query,
        )
        score = max((float(ev.get("cross_encoder_score", 0.0) or 0.0) for ev in choice_result["evidence_store"]), default=float("-inf"))
        candidate_scores[letter] = score
        for ev in choice_result["evidence_store"]:
            ev = dict(ev)
            ev["candidate"] = letter
            evidence_store.append(ev)
            retrieved_ids.append(ev["idx"])

    selected = max(candidate_scores, key=candidate_scores.get) if candidate_scores else "A"
    final_answer = f"Answer: ({selected})"
    retrieved_ids = list(dict.fromkeys(str(idx) for idx in retrieved_ids))
    return {
        "final_answer": final_answer,
        "formatted_question": question,
        "intermediate_question": question_intermediate,
        "snap_answer": snap_block,
        "snap_letter": _extract_answer(snap_block, config),
        "snap_and_hyre_raw": combined_raw,
        "snap_hyre_parse_ok": parse_ok,
        "hyre_cache_hit": bool(cache_entry),
        "hyre_cache_label": _row_label(row, config) if cache_entry else "",
        "hyde_passage": hyre_passage,
        "hyde_passage_raw": combined_raw,
        "hyde_contains_answer_artifact": _contains_answer_artifact(hyre_passage),
        "retrieval_queries": [choices[l] for l in choices],
        "rerank_query": "candidate_score",
        "retrieval_where": _where_from_config(config) or {},
        "final_context_fields": ["candidate_scores"],
        "final_prompt_preview": "",
        "evidence_store": evidence_store,
        "retrieved_ids": retrieved_ids,
        "gold_retrieved": _is_gold_retrieved(row, retrieved_ids),
        "hyre_route": "casehold_option_score",
        "adaptive_policy": "casehold_option_score_v1",
        "candidate_scores": candidate_scores,
        "selected_candidate": selected,
    }


def run_adaptive_snap_hyre_option_table(row: pd.Series, config: EvalConfig) -> dict:
    """CaseHOLD compact LLM selector over candidate-conditioned score snippets."""
    if config.dataset != "casehold":
        out = run_adaptive_snap_hyre_frontier(row, config)
        out["hyre_route"] = f"option_table_fallback_{config.dataset}"
        out["adaptive_policy"] = "casehold_option_table_v1"
        return out

    question = _fmt(row, config)
    raw_question = _retrieval_question(row)
    question_intermediate = _fmt_intermediate(row, config)
    choices = _choice_texts(row, config)
    cache_entry = _hyre_cache_entry(row, config)
    if cache_entry:
        combined_raw = str(cache_entry.get("snap_and_hyre_raw") or cache_entry.get("hyde_passage_raw") or "")
        snap_block = str(cache_entry.get("snap_answer") or "")
        hyre_passage = str(cache_entry.get("hyde_passage") or "")
        parse_ok = bool(cache_entry.get("snap_hyre_parse_ok", True))
        if not hyre_passage:
            snap_block, hyre_passage, parse_ok = _split_snap_and_hyde(
                combined_raw,
                fallback_passage=question_intermediate,
            )
    else:
        combined_raw = _llm_call(
            _snap_hyde_2call_system(config),
            question,
            label="adaptive_snap_hyre_option_table/snap_and_hyre",
        )
        snap_block, hyre_passage, parse_ok = _split_snap_and_hyde(
            combined_raw,
            fallback_passage=question_intermediate,
        )

    evidence_store = []
    retrieved_ids = []
    table_rows: list[dict] = []
    option_query_chars = int(os.getenv("OPTION_TABLE_QUERY_CHARS", "420"))
    bounded_raw_question = _preview_text(raw_question, limit=option_query_chars)
    option_score_query = f"{bounded_raw_question}\n\nHyRE retrieval passage:\n{_preview_text(hyre_passage, limit=option_query_chars)}"
    table_rows = _score_option_table_choices(option_score_query, choices)
    for item in table_rows:
        ev = {
            "idx": item["idx"],
            "text": item["holding"],
            "source": item["source"],
            "cross_encoder_score": item["score"],
            "candidate": item["candidate"],
            "score_source": item["score_source"],
        }
        evidence_store.append(ev)
        retrieved_ids.append(ev["idx"])
    _record_trace_event(
        "option_table",
        label="adaptive_snap_hyre_option_table",
        score_query=option_score_query,
        score_source=table_rows[0]["score_source"] if table_rows else "",
        candidates=table_rows,
    )

    score_lines = []
    for item in table_rows:
        score = item["score"]
        score_text = "n/a" if score == float("-inf") else f"{score:.4f}"
        score_lines.append(
            f"Candidate {item['candidate']} | score={score_text}\n"
            f"Holding: {item['holding']}\n"
            f"Evidence: {item['snippet']}"
        )
    score_table = "\n\n".join(score_lines)
    user = (
        "## CaseHOLD Citing Context And Answer Options\n"
        f"{question}\n\n"
        "## Snap Reasoning Signal\n"
        f"{snap_block}\n\n"
        "## HyRE Retrieval Passage\n"
        f"{hyre_passage}\n\n"
        "## Per-Candidate Retrieval Score Table\n"
        f"{score_table}"
    )
    answer = _llm_call(_option_table_selector_system(config), user, label="adaptive_snap_hyre_option_table/answer")

    retrieved_ids = list(dict.fromkeys(str(idx) for idx in retrieved_ids))
    return {
        "final_answer": answer,
        "formatted_question": question,
        "intermediate_question": question_intermediate,
        "snap_answer": snap_block,
        "snap_letter": _extract_answer(snap_block, config),
        "snap_and_hyre_raw": combined_raw,
        "snap_hyre_parse_ok": parse_ok,
        "hyre_cache_hit": bool(cache_entry),
        "hyre_cache_label": _row_label(row, config) if cache_entry else "",
        "hyde_passage": hyre_passage,
        "hyde_passage_raw": combined_raw,
        "hyde_contains_answer_artifact": _contains_answer_artifact(hyre_passage),
        "retrieval_queries": [choices[l] for l in choices],
        "rerank_query": "direct_option_score_table",
        "retrieval_where": _where_from_config(config) or {},
        "final_context_fields": ["question", "snap_reasoning", "hyde_passage", "candidate_score_table"],
        "final_prompt_preview": _preview_text(user),
        "evidence_store": evidence_store,
        "retrieved_ids": retrieved_ids,
        "gold_retrieved": _is_gold_retrieved(row, retrieved_ids),
        "hyre_route": "casehold_option_table",
        "adaptive_policy": "casehold_option_table_v1",
        "candidate_score_table": table_rows,
    }


def _stability_control(row: pd.Series, config: EvalConfig) -> tuple[str, dict]:
    """Return the current strongest cheap comparison path for stability checks."""
    if config.dataset == "barexam":
        return "adaptive_snap_hyre_v2", run_adaptive_snap_hyre_v2(row, config)
    if config.dataset == "housing":
        return "snap_hyre_state", run_snap_hyre_state(row, config)
    if config.dataset == "casehold":
        return "adaptive_snap_hyre_diverse", run_adaptive_snap_hyre_diverse(row, config)
    if config.dataset == "legalbench_scalr":
        return "rag_snap_hyde_2call", run_rag_snap_hyde_2call(row, config)
    return "adaptive_snap_hyre_v2", run_adaptive_snap_hyre_v2(row, config)


def run_adaptive_snap_hyre_stability(row: pd.Series, config: EvalConfig) -> dict:
    """Adaptive HyRE with disagreement-triggered answer arbitration.

    The N=200 selector run showed that the controller is wired correctly but
    online HyRE generations vary enough to flip answers. This mode tests a
    targeted stability layer: run the frontier selector and a dataset-appropriate
    control, keep the shared answer when they agree, and spend one extra call
    only when they disagree or one side fails to emit a parseable answer.
    """
    question = _fmt(row, config)
    frontier = run_adaptive_snap_hyre_frontier(row, config)
    control_name, control = _stability_control(row, config)
    frontier_answer = _extract_answer(frontier.get("final_answer", ""), config)
    control_answer = _extract_answer(control.get("final_answer", ""), config)

    arbitration_used = frontier_answer != control_answer or not frontier_answer or not control_answer
    if arbitration_used:
        if config.dataset == "housing":
            answer_format = "End with exactly one final line: Answer: Yes or Answer: No"
        else:
            answer_format = "End with exactly one final line in the form: Answer: (X)"
        arb_system = (
            "You are a careful legal answer arbitrator. Two legal RAG strategies answered the "
            "same question. Compare their reasoning and choose the answer best supported by the "
            "question, displayed options, and cited legal evidence. Do not average the answers. "
            f"{answer_format}."
        )
        arb_user = (
            f"## Question\n{question}\n\n"
            f"## Frontier Selector Answer\n{frontier.get('final_answer', '')}\n\n"
            f"## Control Strategy ({control_name}) Answer\n{control.get('final_answer', '')}"
        )
        final_answer = _llm_call(arb_system, arb_user, label="adaptive_snap_hyre_stability/arbitrate")
        selected_source = "arbitrated"
    else:
        final_answer = frontier.get("final_answer", "")
        selected_source = "agreement"

    frontier_ids = list(frontier.get("retrieved_ids", []) or [])
    control_ids = list(control.get("retrieved_ids", []) or [])
    merged_ids = list(dict.fromkeys(frontier_ids + control_ids))
    merged_evidence = list(frontier.get("evidence_store", []) or []) + list(control.get("evidence_store", []) or [])

    out = {
        "final_answer": final_answer,
        "frontier_answer": frontier.get("final_answer", ""),
        "frontier_predicted_answer": frontier_answer,
        "control_mode": control_name,
        "control_answer": control.get("final_answer", ""),
        "control_predicted_answer": control_answer,
        "arbitration_used": arbitration_used,
        "selected_source": selected_source,
        "hyre_route": f"stability_{config.dataset}",
        "adaptive_policy": "frontier_stability_arbitration_v1",
        "retrieved_ids": merged_ids,
        "evidence_store": merged_evidence,
        "gold_retrieved": bool(frontier.get("gold_retrieved")) or bool(control.get("gold_retrieved")),
        "frontier_gold_retrieved": frontier.get("gold_retrieved"),
        "control_gold_retrieved": control.get("gold_retrieved"),
        "frontier_detail": {
            "hyre_route": frontier.get("hyre_route"),
            "snap_hyre_parse_ok": frontier.get("snap_hyre_parse_ok"),
            "retrieval_queries": frontier.get("retrieval_queries", []),
        },
        "control_detail": {
            "hyre_route": control.get("hyre_route"),
            "snap_hyre_parse_ok": control.get("snap_hyre_parse_ok"),
            "retrieval_queries": control.get("retrieval_queries", []),
        },
    }
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
        **snap_hyre_generation_meta,
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
        "gold_retrieved": _is_gold_retrieved(row, all_ids),
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
    "End with exactly one final line in the form: Answer: (X)"
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
        "gold_retrieved": _is_gold_retrieved(row, retrieved_ids),
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
        "gold_retrieved": _is_gold_retrieved(row, retrieved_ids),
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
        "gold_retrieved": _is_gold_retrieved(row, retrieved_ids),
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
        "gold_retrieved": _is_gold_retrieved(row, retrieved_ids),
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
        "Reason step by step, then end with exactly one final line in the form: Answer: (X)"
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
    "End with exactly one final line in the form: Answer: (X)"
)


def _where_from_config(config: EvalConfig) -> dict | None:
    """Build ChromaDB where filter from config.source_filter."""
    if config.source_filter:
        return {"source": config.source_filter}
    return None


def _housing_state_where(row: pd.Series, config: EvalConfig) -> dict | None:
    """Build a HousingQA state metadata filter for Chroma retrieval."""
    if config.dataset != "housing":
        return _where_from_config(config)
    # Housing statute metadata is embedded from `datasets/housing_qa/statutes.csv`,
    # where state names are lowercase. Question rows keep display-case names.
    state = str(row.get("state", "") or "").strip().lower()
    if not state:
        return _where_from_config(config)
    if config.source_filter:
        return {"$and": [{"source": config.source_filter}, {"state": state}]}
    return {"state": state}


def _housing_state_filter_enabled(config: EvalConfig) -> bool:
    value = os.getenv("EVAL_HOUSING_STATE_FILTER", "").strip().lower()
    return bool(getattr(config, "housing_state_filter", False)) or value in {"1", "true", "yes", "on"}


def _retrieval_where_for_row(row: pd.Series, config: EvalConfig) -> dict | None:
    """Canonical retrieval filter for rows whose metadata can safely constrain search."""
    if config.dataset == "housing" and _housing_state_filter_enabled(config):
        return _housing_state_where(row, config)
    return _where_from_config(config)


def run_rag_rewrite(row: pd.Series, config: EvalConfig) -> dict:
    """Query rewrite → retrieval → answer with evidence."""
    question = _fmt(row, config)
    question_intermediate = _fmt_intermediate(row, config)
    queries, rewrite_meta = _rewrite_query_with_meta(question_intermediate)

    retrieval = _retrieve_and_format(row, queries, k=config.retrieval_k, label_prefix="rewrite",
                                     where=_retrieval_where_for_row(row, config),
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
        **_retrieval_cache_audit_fields(retrieval),
        **rewrite_meta,
    }


def run_rag_simple(row: pd.Series, config: EvalConfig) -> dict:
    """Raw question → retrieval → answer with evidence (no rewrite)."""
    question = _fmt(row, config)
    raw_question = _retrieval_question(row)

    retrieval = _retrieve_and_format(row, [raw_question], k=config.retrieval_k, label_prefix="simple",
                                     where=_retrieval_where_for_row(row, config),
                                     collection=_collection_for_config(config))
    passage_block = "\n\n".join(retrieval["passages"])

    user = _retrieved_answer_user(config, passage_block, question)
    answer = _llm_call(_system_prompt(config, "rag"), user, label="rag_simple/answer")

    return {
        "final_answer": answer,
        "evidence_store": retrieval["evidence_store"],
        "retrieved_ids": retrieval["retrieved_ids"],
        "gold_retrieved": retrieval["gold_retrieved"],
        **_retrieval_cache_audit_fields(retrieval),
    }


def run_rag_state_filter(row: pd.Series, config: EvalConfig) -> dict:
    """HousingQA state-filtered RAG using Chroma metadata."""
    if config.dataset != "housing":
        result = run_rag_simple(row, config)
        result["state_filter_fallback"] = "dataset_not_supported"
        return result

    question = _fmt(row, config)
    raw_question = _retrieval_question(row)
    where = _housing_state_where(row, config)

    retrieval = _retrieve_and_format(
        row,
        [raw_question],
        k=config.retrieval_k,
        label_prefix="state_filter",
        where=where,
        collection=_collection_for_config(config),
    )
    passage_block = "\n\n".join(retrieval["passages"])

    user = f"## Retrieved Passages\n{passage_block}\n\n## Question\n{question}"
    answer = _llm_call(_system_prompt(config, "rag"), user, label="rag_state_filter/answer")

    return {
        "final_answer": answer,
        "retrieval_where": where or {},
        "evidence_store": retrieval["evidence_store"],
        "retrieved_ids": retrieval["retrieved_ids"],
        "gold_retrieved": retrieval["gold_retrieved"],
        **_retrieval_cache_audit_fields(retrieval),
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
        "Reason step by step, then end with exactly one final line in the form: Answer: (X)"
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

    # Gold-retrieved tracking — accept if ANY gold id appears across per-TODO retrievals.
    gold_retrieved = _is_gold_retrieved(row, all_retrieved_ids)

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

    gold_retrieved = _is_gold_retrieved(row, all_retrieved_ids)

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

    gold_retrieved = _is_gold_retrieved(row, all_retrieved_ids)

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

    gold_retrieved = _is_gold_retrieved(row, all_retrieved_ids)

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

    gold_retrieved = _is_gold_retrieved(row, all_retrieved_ids)

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
    "adaptive_snap_hyre_option_table",
}


def _housing_retrieval_mode(mode: str) -> bool:
    return mode not in _NO_CHROMA_MODES


def _allow_unfiltered_housing_retrieval() -> bool:
    return _env_truthy("EVAL_ALLOW_UNFILTERED_HOUSING_RETRIEVAL")


def _canonical_answer_mode(mode: str) -> bool:
    return mode in {
        "llm_only",
        "rag_simple",
        "golden_passage",
        "golden_plus_neighbors",
        "rag_hyde",
        "snap_hyre",
        "rag_rewrite",
    }


MODE_RUNNERS = {
    "full_pipeline": run_full_pipeline,
    "llm_only": run_llm_only,
    "rag_rewrite": run_rag_rewrite,
    "rag_simple": run_rag_simple,
    "rag_state_filter": run_rag_state_filter,
    "golden_passage": run_golden_passage,
    "golden_plus_neighbors": run_golden_plus_neighbors,
    "golden_arbitration": run_golden_arbitration,
    "golden_arb_conservative": run_golden_arb_conservative,
    "rag_arbitration": run_rag_arbitration,
    "rag_hyde": run_rag_hyde,
    "rag_hyde_exemplar": run_rag_hyde_exemplar,
    "rag_hyde_arb": run_rag_hyde_arb,
    "rag_multi_hyde": run_rag_multi_hyde,
    "rag_snap_hyde": run_rag_snap_hyde,
    "rag_snap_hyde_1call": run_rag_snap_hyde_1call,
    "snap_hyre": run_rag_snap_hyde_2call,
    "snap_hyre_exemplar": run_snap_hyre_exemplar,
    "snap_choice_hyre": run_snap_choice_hyre,
    "rag_snap_hyde_2call": run_rag_snap_hyde_2call,
    "adaptive_snap_route": run_adaptive_snap_route,
    "snap_hyde_aligned": run_snap_hyde_aligned,
    "snap_hyre_option": run_snap_hyre_option,
    "snap_hyre_state": run_snap_hyre_state,
    "adaptive_snap_hyre": run_adaptive_snap_hyre,
    "adaptive_snap_hyre_anchor": run_adaptive_snap_hyre_anchor,
    "adaptive_snap_hyre_diverse": run_adaptive_snap_hyre_diverse,
    "adaptive_snap_hyre_v2": run_adaptive_snap_hyre_v2,
    "adaptive_snap_hyre_frontier": run_adaptive_snap_hyre_frontier,
    "adaptive_snap_hyre_stability": run_adaptive_snap_hyre_stability,
    "adaptive_snap_hyre_housing_verifier": run_adaptive_snap_hyre_housing_verifier,
    "adaptive_snap_hyre_candidate_verifier": run_adaptive_snap_hyre_candidate_verifier,
    "adaptive_snap_hyre_option_reranker": run_adaptive_snap_hyre_option_reranker,
    "adaptive_snap_hyre_option_score": run_adaptive_snap_hyre_option_score,
    "adaptive_snap_hyre_option_table": run_adaptive_snap_hyre_option_table,
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
    allow_env_caches = os.getenv("EVAL_ALLOW_ENV_CACHE_PATHS", "").strip().lower() in {
        "1", "true", "yes", "on"
    }
    if config.hyre_cache_path:
        os.environ["HYRE_CACHE_PATH"] = config.hyre_cache_path
    elif not allow_env_caches:
        os.environ.pop("HYRE_CACHE_PATH", None)
    if config.retrieval_cache_path:
        os.environ["RETRIEVAL_CACHE_PATH"] = config.retrieval_cache_path
    elif not allow_env_caches:
        os.environ.pop("RETRIEVAL_CACHE_PATH", None)
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


def _detail_log_path(config: EvalConfig, ts: str) -> str:
    detail_suffix = f"{config.dataset}_{config.tag}" if config.tag else config.dataset
    detail_suffix = re.sub(r"[^A-Za-z0-9_.-]+", "-", detail_suffix).strip("-")
    return os.path.join("logs", f"eval_{config.mode}_{config.provider}_{ts}_{detail_suffix}_detail.jsonl")


def _append_detail_record(detail_path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(detail_path) or ".", exist_ok=True)
    with open(detail_path, "a") as f:
        f.write(json.dumps(_serialize_result(record)) + "\n")


def _no_silent_fallback_enabled() -> bool:
    return os.getenv("NO_SILENT_FALLBACK", "").strip().lower() in {"1", "true", "yes", "on"}


def _provider_route_metadata() -> dict:
    """Provider route controls that affect which backend serves a model."""
    route = {}
    for key in (
        "OPENROUTER_PROVIDER_ONLY",
        "OPENROUTER_PROVIDER_ORDER",
        "OPENROUTER_PROVIDER_IGNORE",
    ):
        value = os.getenv(key, "").strip()
        if value:
            route[key.lower()] = value
    return route


def _fallback_guard_violations(record: dict, config: EvalConfig) -> list[str]:
    """Return fallback/cache/oracle violations that must not pass as normal rows."""
    violations: list[str] = []

    if record.get("error"):
        violations.append(f"row_error={str(record.get('error'))[:160]}")

    routed_to = str(record.get("routed_to") or "")
    if "fallback" in routed_to.lower():
        violations.append(f"routed_to={routed_to}")

    hyre_route = str(record.get("hyre_route") or "")
    if "fallback" in hyre_route.lower():
        violations.append(f"hyre_route={hyre_route}")

    for key, value in record.items():
        if (key.endswith("_fallback") or key.endswith("_used_fallback")) and value:
            violations.append(f"{key}={value}")
        if key.endswith("_contains_answer_artifact") and value:
            violations.append(f"{key}=True")

    for key in (
        "hyde_parse_ok",
        "snap_hyre_parse_ok",
        "snap_hyde_2call_parse_ok",
        "snap_hyde_1call_parse_ok",
        "choice_hyre_parse_ok",
        "rewrite_parse_ok",
        "route_parse_ok",
        "passage_parse_ok",
        "adaptive_parse_ok",
    ):
        if record.get(key) is False:
            violations.append(f"{key}=False")

    if "<think>" in str(record.get("final_answer") or "").lower():
        violations.append("unclosed_think_or_reasoning_tag_in_final_answer")

    if _requires_strict_answer_line(config):
        final_answer = str(record.get("final_answer") or "")
        predicted = record.get("predicted_answer")
        if not _has_explicit_answer_marker(final_answer):
            violations.append("missing_explicit_answer_marker")
        elif not _has_required_final_answer_line(final_answer, predicted, config):
            violations.append("missing_required_final_answer_line")

    max_answer_chars = _env_int("EVAL_MAX_FINAL_ANSWER_CHARS", 20_000)
    final_answer_chars = len(str(record.get("final_answer") or ""))
    if max_answer_chars > 0 and final_answer_chars > max_answer_chars:
        violations.append(
            f"final_answer_chars={final_answer_chars}>EVAL_MAX_FINAL_ANSWER_CHARS={max_answer_chars}"
        )

    max_completion_tokens = _env_int("LLM_MAX_COMPLETION_TOKENS", 0)
    output_token_margin = _env_int("EVAL_OUTPUT_TOKEN_MARGIN", 16)
    output_tokens = int(record.get("output_tokens") or 0)
    llm_calls = int(record.get("llm_calls") or 0)
    if (
        max_completion_tokens > 0
        and llm_calls <= 1
        and _near_completion_cap(output_tokens)
    ):
        violations.append(
            f"output_tokens={output_tokens} near LLM_MAX_COMPLETION_TOKENS={max_completion_tokens}"
        )
    retry_output_tokens = int(record.get("answer_format_retry_output_tokens") or 0)
    if max_completion_tokens > 0 and retry_output_tokens and _near_completion_cap(retry_output_tokens):
        violations.append(
            "answer_format_retry_output_tokens="
            f"{retry_output_tokens} near LLM_MAX_COMPLETION_TOKENS={max_completion_tokens}"
        )

    if config.retrieval_cache_path and config.mode in {
        "rag_simple",
        "rag_hyde",
        "rag_hyde_exemplar",
        "snap_hyre",
        "snap_hyre_exemplar",
        "rag_snap_hyde_2call",
        "golden_plus_neighbors",
    }:
        if record.get("retrieval_cache_hit") is not True:
            violations.append("retrieval_cache_hit!=True")

    if config.hyre_cache_path and config.mode in {
        "rag_hyde",
        "rag_hyde_exemplar",
        "snap_hyre",
        "snap_hyre_exemplar",
        "rag_snap_hyde_2call",
    }:
        if record.get("hyre_cache_hit") is not True and record.get("hyde_cache_hit") is not True:
            violations.append("hyre_cache_hit!=True")

    if config.mode in {"snap_hyre", "snap_hyre_exemplar", "rag_snap_hyde_2call"} and _requires_strict_answer_line(config):
        snap_answer = str(record.get("snap_answer") or "")
        if not _extract_required_final_line_prediction(snap_answer, config):
            violations.append("snap_answer_missing_required_final_line")

    if config.mode in {"golden_passage", "golden_plus_neighbors"}:
        if record.get("gold_retrieved") is not True:
            violations.append("oracle_gold_not_injected")
        if not record.get("evidence_store"):
            violations.append("oracle_evidence_store_empty")

    return violations


class NoSilentFallbackViolation(RuntimeError):
    """Fail-closed row wrapper that preserves the blocked detail record."""

    def __init__(self, label: str, record: dict, violations: list[str]):
        self.label = label
        self.record = record
        self.violations = violations
        super().__init__(f"{label}: " + "; ".join(violations))


def _is_openrouter_provider(provider: str) -> bool:
    return str(provider or "").strip().lower().startswith("or-")


def _resolved_concurrency(config: EvalConfig, provider: str) -> int:
    configured = int(getattr(config, "concurrency", 0) or 0)
    if configured <= 0:
        raw = os.getenv("EVAL_CONCURRENCY", "").strip()
        if raw:
            try:
                configured = int(raw)
            except ValueError as exc:
                raise SystemExit(f"EVAL_CONCURRENCY must be an integer, got {raw!r}") from exc
    if configured <= 0:
        configured = 8 if _is_openrouter_provider(provider) else 1
    return max(1, configured)


def _preload_eval_caches(config: EvalConfig) -> None:
    """Load read-only caches before threaded workers can race on lazy globals."""
    hyre_path = (config.hyre_cache_path or os.getenv("HYRE_CACHE_PATH", "")).strip()
    retrieval_path = (config.retrieval_cache_path or os.getenv("RETRIEVAL_CACHE_PATH", "")).strip()
    doc_cache_path = os.getenv("RETRIEVAL_DOC_CACHE_PATH", "").strip()
    if hyre_path:
        _load_hyre_cache(hyre_path)
    if retrieval_path:
        _load_retrieval_cache(retrieval_path)
    if doc_cache_path:
        _load_retrieval_doc_cache(doc_cache_path)


def _row_subject(row: pd.Series, config: EvalConfig) -> str:
    if is_beir_dataset(config.dataset):
        return BEIR_DATASETS[config.dataset]
    if config.dataset == "housing":
        return str(row.get("state", "unknown"))
    if config.dataset == "casehold":
        return "casehold"
    if config.dataset == "legal_rag":
        return "crim_law"
    if config.dataset == "legal_rag_bench":
        return "victorian_crim_law"
    if config.dataset == "mas_legal_bench":
        return str(row.get("source", "mas_legal_bench"))
    if config.dataset == "legal_link_eu":
        return str(row.get("relation_type", "legal_link_eu"))
    if config.dataset == "australian":
        return str(row.get("jurisdiction", "unknown"))
    if config.dataset == "musique":
        return f"{int(row.get('n_hops', 0))}-hop"
    if config.dataset == "medqa":
        return str(row.get("meta_info", "medqa"))
    return str(row.get("subject", "unknown"))


def _row_gold_answer(row: pd.Series, config: EvalConfig) -> str:
    if "answer" not in row:
        return ""
    gold = str(row["answer"]).strip()
    if config.dataset == "housing":
        return gold.capitalize()
    if config.dataset in ("barexam", "casehold", "legalbench_scalr", "mas_legal_bench", "legal_link_eu", "medqa"):
        return gold.upper()
    return gold


def _print_row_status(order_i: int, n: int, record: dict, is_open_ended: bool) -> None:
    status = "PASS" if record.get("is_correct") else "FAIL"
    if record.get("error"):
        status = "ERROR"
    if is_open_ended:
        print(
            f"[{order_i+1}/{n}] {record.get('label', ''):<35} {status:<6} "
            f"({float(record.get('elapsed_sec') or 0):.1f}s, {int(record.get('llm_calls') or 0)} calls)",
            flush=True,
        )
    else:
        print(
            f"[{order_i+1}/{n}] {record.get('label', ''):<35} {status:<6} "
            f"gold={record.get('correct_answer')} pred={record.get('predicted_answer')} "
            f"({float(record.get('elapsed_sec') or 0):.1f}s, {int(record.get('llm_calls') or 0)} calls)",
            flush=True,
        )


def _evaluate_one_row(
    order_i: int,
    row: pd.Series,
    *,
    config: EvalConfig,
    runner,
    provider_route: dict,
    embedding_model: str | None,
    is_open_ended: bool,
    is_short_span: bool,
) -> tuple[int, dict]:
    _reset_llm_call_counter()
    _reset_call_trace()
    _reset_trace_events()
    q_start = time.time()

    subject = _row_subject(row, config)
    label = _row_label(row, config, order_i)
    idx = str(row.get("idx", order_i))
    gold = _row_gold_answer(row, config)

    try:
        result = runner(row, config)
        answer_text = result.get("final_answer", "")
        predicted = _extract_answer(answer_text, config)
        final_line_prediction = _extract_required_final_line_prediction(answer_text, config)
        if final_line_prediction is not None:
            predicted = final_line_prediction
        answer_text, predicted = _maybe_retry_final_answer_format(
            row,
            config,
            result,
            answer_text,
            predicted,
        )

        if is_open_ended:
            is_correct = _judge_open_answer(row["question"], gold, answer_text, config)
            result["judge_score"] = is_correct
        elif is_short_span:
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
    except Exception as exc:
        result = {}
        answer_text = ""
        predicted = None
        is_correct = False
        error = str(exc)

    elapsed = time.time() - q_start
    metrics = _get_metrics()

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
        "gold_idx": _gold_idx_string(row),
        "final_answer": answer_text[:500] if is_open_ended else answer_text,
        "mode": config.mode,
        "provider": config.provider,
        "provider_route": provider_route,
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
        if _housing_state_filter_enabled(config):
            record["housing_state_filter"] = True
    elif config.dataset in ("casehold", "legalbench_scalr"):
        record["choices"] = _record_choices(row, config.dataset)
        record["gold_passage"] = _gold_choice_text(row, gold)
    elif config.dataset == "mas_legal_bench":
        record["choices"] = _record_choices(row, config.dataset)
        record["source"] = str(row.get("source", ""))
        source_context_ids = _coerce_gold_ids(row.get("source_context_ids", ""))
        record["source_context_count"] = len(source_context_ids)
        record["source_context_ids_preview"] = source_context_ids[:20]
        record["gold_passage"] = ""
    elif config.dataset == "legal_link_eu":
        record["choices"] = _record_choices(row, config.dataset)
        record["relation_type"] = str(row.get("relation_type", ""))
        record["source_doc"] = str(row.get("source_doc", ""))
        record["target_doc"] = str(row.get("target_doc", ""))
        record["gold_passage"] = str(row.get("gold_passage", ""))[:500]
    elif config.dataset == "australian":
        record["jurisdiction"] = str(row.get("jurisdiction", ""))
    elif config.dataset in ("legal_rag", "legal_rag_bench"):
        record["relevant_passages"] = str(row.get("relevant_passages", ""))
    elif config.dataset == "medqa":
        record["choices"] = _record_choices(row, config.dataset)
        record["meta_info"] = str(row.get("meta_info", ""))
        record["answer_text"] = str(row.get("answer_text", ""))[:500]
        record["gold_passage"] = ""
    elif is_beir_dataset(config.dataset):
        record["beir_subset"] = BEIR_DATASETS[config.dataset]
        record["gold_count"] = int(row.get("gold_count", 0) or 0)
        record["gold_passage"] = ""
    else:
        record["choices"] = _record_choices(row, config.dataset)
        record["gold_passage"] = str(row.get("gold_passage", ""))[:500]

    for k, v in result.items():
        if k != "final_answer" and k not in record:
            record[k] = v

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

    if _no_silent_fallback_enabled():
        violations = _fallback_guard_violations(record, config)
        if violations:
            record["tag"] = config.tag
            record["no_silent_fallback_violations"] = violations
            raise NoSilentFallbackViolation(label, _serialize_result(record), violations)

    record["tag"] = config.tag
    return order_i, _serialize_result(record)


def run_eval(config: EvalConfig):
    """Run evaluation with the given config."""
    if config.mode not in MODE_RUNNERS:
        print(f"Unknown mode '{config.mode}'. Available: {', '.join(MODE_RUNNERS)}")
        sys.exit(1)

    if (
        config.dataset == "housing"
        and _housing_retrieval_mode(config.mode)
        and not _housing_state_filter_enabled(config)
        and not _allow_unfiltered_housing_retrieval()
    ):
        raise SystemExit(
            "HousingQA retrieval modes must use --housing-state-filter. "
            "The national HousingQA corpus contains every state, so unfiltered retrieval "
            "is a provenance/ablation path only. Set "
            "EVAL_ALLOW_UNFILTERED_HOUSING_RETRIEVAL=1 for an explicit unfiltered run."
        )

    if (
        _canonical_answer_mode(config.mode)
        and not _no_silent_fallback_enabled()
        and not _env_truthy("EVAL_ALLOW_SILENT_FALLBACK")
    ):
        raise SystemExit(
            "NO_SILENT_FALLBACK=1 is required for canonical answer modes. "
            "Set EVAL_ALLOW_SILENT_FALLBACK=1 only for exploratory debugging."
        )

    _setup_provider(config)
    runner = MODE_RUNNERS[config.mode]
    provider_info = get_provider_info()
    provider_route = _provider_route_metadata()
    embedding_model = os.getenv("EVAL_EMBEDDING_MODEL", "").strip() or None

    qa = load_questions(config)
    if config.sample_start or config.sample_end is not None:
        start = max(0, int(config.sample_start or 0))
        end = None if config.sample_end is None else max(start, int(config.sample_end))
        original_n = len(qa)
        qa = qa.iloc[start:end].reset_index(drop=True)
        print(f"[sample-slice] selected rows [{start}:{end if end is not None else ''}] from sampled set of {original_n}")
    n = len(qa)

    print(f"\n{'=' * 70}")
    filter_str = f" | filter={config.source_filter}" if config.source_filter else ""
    if config.dataset == "housing" and _housing_state_filter_enabled(config):
        filter_str += " | housing_state_filter=on"
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
    skip_collection_preflight = os.getenv("SKIP_EVAL_COLLECTION_PREFLIGHT", "").strip().lower() in {"1", "true", "yes", "on"}
    if skip_collection_preflight:
        print("[preflight] collection count skipped by SKIP_EVAL_COLLECTION_PREFLIGHT=1")
    elif config.dataset != "musique" and config.mode not in _NO_CHROMA_MODES:
        try:
            from rag_utils import get_vectorstore
            _coll_name = _collection_for_config(config) if "_collection_for_config" in globals() else "legal_passages"
            _vs = get_vectorstore(_coll_name, embedding_model=os.getenv("EVAL_EMBEDDING_MODEL", "").strip() or None)
            _coll_count = _vs._collection.count() if hasattr(_vs, "_collection") else None
            if _coll_count is not None and _coll_count == 0:
                print(f"[preflight] FAILED: collection '{_coll_name}' is EMPTY (0 docs).")
                print(f"[preflight] mode={config.mode} requires retrieval but corpus is missing.")
                print(f"[preflight] Rebuild via: uv run python utils/fast_embed.py {config.dataset}")
                raise SystemExit(4)
            elif _coll_count is not None:
                print(f"[preflight] collection={_coll_name} has {_coll_count:,} docs OK")
        except SystemExit:
            raise
        except Exception as exc:
            print(f"[preflight] WARNING: could not verify collection: {exc}")

    run_ts = time.strftime("%Y%m%d_%H%M%S")
    detail_path = _detail_log_path(config, run_ts)
    os.makedirs("logs", exist_ok=True)
    if os.path.exists(detail_path):
        raise SystemExit(f"Refusing to overwrite existing detail log: {detail_path}")
    provider_name = provider_info["provider"]
    concurrency = _resolved_concurrency(config, provider_name)
    use_parallel = _is_openrouter_provider(provider_name) and concurrency > 1
    if use_parallel:
        os.environ["EVAL_CONCURRENCY"] = str(concurrency)
    print(f"[concurrency] provider={provider_name} workers={concurrency if use_parallel else 1}")
    if use_parallel:
        _preload_eval_caches(config)
        print(f"[detail-log] collecting rows concurrently; writing deterministic order to {detail_path}")
    else:
        print(f"[detail-log] streaming rows to {detail_path}")

    results_by_order: dict[int, dict] = {}
    results = []
    correct = 0
    total_start = time.time()
    consecutive_errors = 0

    is_open_ended = config.dataset in ("legal_rag", "legal_rag_bench", "australian") or is_beir_dataset(config.dataset)
    is_short_span = config.dataset == "musique"

    row_items = list(qa.iterrows())

    if use_parallel:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(
                    _evaluate_one_row,
                    i,
                    row,
                    config=config,
                    runner=runner,
                    provider_route=provider_route,
                    embedding_model=embedding_model,
                    is_open_ended=is_open_ended,
                    is_short_span=is_short_span,
                ): i
                for i, row in row_items
            }
            error_count = 0
            for future in concurrent.futures.as_completed(futures):
                try:
                    order_i, record = future.result()
                except NoSilentFallbackViolation as exc:
                    _append_detail_record(detail_path, exc.record)
                    for pending in futures:
                        pending.cancel()
                    print(
                        f"\n[no-silent-fallback] blocked {exc.label}: "
                        + "; ".join(exc.violations[:10])
                    )
                    raise SystemExit(5) from exc
                results_by_order[order_i] = record
                if record.get("error"):
                    error_count += 1
                    if error_count >= 5:
                        for pending in futures:
                            pending.cancel()
                        print(
                            f"\n[circuit-breaker] {error_count} row errors in parallel run. "
                            f"Last error: {str(record.get('error'))[:200]}\n"
                            "[circuit-breaker] Aborting to avoid garbage results."
                        )
                        raise SystemExit(3)

        for order_i in sorted(results_by_order):
            record = results_by_order[order_i]
            results.append(record)
            if record.get("is_correct"):
                correct += 1
            _append_detail_record(detail_path, record)
            _print_row_status(order_i, n, record, is_open_ended)
    else:
        for i, row in row_items:
            try:
                order_i, record = _evaluate_one_row(
                    i,
                    row,
                    config=config,
                    runner=runner,
                    provider_route=provider_route,
                    embedding_model=embedding_model,
                    is_open_ended=is_open_ended,
                    is_short_span=is_short_span,
                )
            except NoSilentFallbackViolation as exc:
                _append_detail_record(detail_path, exc.record)
                print(
                    f"\n[no-silent-fallback] blocked {exc.label}: "
                    + "; ".join(exc.violations[:10])
                )
                raise SystemExit(5) from exc

            if record.get("error"):
                consecutive_errors += 1
                if consecutive_errors >= 5:
                    print(
                        f"\n[circuit-breaker] {consecutive_errors} consecutive errors. "
                        f"Last error: {str(record.get('error'))[:200]}\n"
                        f"[circuit-breaker] Aborting at question {i+1} to avoid garbage results."
                    )
                    raise SystemExit(3)
            else:
                consecutive_errors = 0

            if record.get("is_correct"):
                correct += 1
            results.append(record)
            _append_detail_record(detail_path, record)
            _print_row_status(order_i, n, record, is_open_ended)

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
    question_set = config.questions if config.questions in ("curated", "full") else f"n{config.questions}"

    with open(detail_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nDetail log: {detail_path}")

    # --- Append to experiments.jsonl ---
    run_id = f"{run_ts}_{config.mode}_{config.provider}"
    if config.tag:
        run_id += f"_{config.tag}"

    summary = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": config.mode,
        "dataset": config.dataset,
        "provider": provider_info["provider"],
        "model": provider_info["model"],
        "provider_route": provider_route,
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
        "housing_state_filter": bool(config.dataset == "housing" and _housing_state_filter_enabled(config)),
        "detail_log": detail_path,
        "git_commit": _git_commit_short(),
        "concurrency": concurrency if use_parallel else 1,
        "parallel_openrouter": bool(use_parallel),
    }
    if config.sample_start or config.sample_end is not None:
        summary["sample_start"] = int(config.sample_start or 0)
        summary["sample_end"] = config.sample_end

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
        retrieval_checked = [
            r for r in results
            if not (
                config.mode == "adaptive_snap_route"
                and r.get("route_decision") == "SUFFICIENT"
            )
        ]
        empty_ret = sum(1 for r in retrieval_checked if r.get("retrieved_ids") == [])
        empty_ret_rate = empty_ret / max(len(retrieval_checked), 1)
        if empty_ret_rate > 0.5:
            summary["tag"] = (config.tag + "_FAILED-EMPTY-RETRIEVAL").lstrip("_")
            summary["empty_retrieval_rate"] = round(empty_ret_rate, 3)
            summary["empty_retrieval_count"] = empty_ret
            summary["empty_retrieval_checked_count"] = len(retrieval_checked)
            print(
                f"\n[summary-guard] {empty_ret}/{len(retrieval_checked)} ({empty_ret_rate:.0%}) "
                f"retrieval-attempted records had empty retrieval. "
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
                        choices=["barexam", "housing", "legal_rag", "legal_rag_bench", "mas_legal_bench", "legal_link_eu", "australian", "casehold", "musique", "legalbench_scalr", "medqa", *BEIR_DATASETS.keys()],
                        help="Dataset to evaluate on (default: barexam)")
    parser.add_argument("--retrieval-k", type=int, default=5,
                        help="Final top-k after rerank for retrieval modes (default 5; meeting ask: top-1 vs top-5 ablation)")
    parser.add_argument("--sample-start", type=int, default=0,
                        help="Optional start offset after deterministic question sampling")
    parser.add_argument("--sample-end", type=int, default=None,
                        help="Optional end offset after deterministic question sampling")
    parser.add_argument("--hyre-cache-path", default="",
                        help="Optional JSONL cache of snap/HyRE generations keyed by detail-log label")
    parser.add_argument("--retrieval-cache-path", default="",
                        help="Optional JSONL cache of retrieved passage ids for top-k replay")
    parser.add_argument("--housing-state-filter", action="store_true",
                        help="For HousingQA retrieval modes, constrain Chroma retrieval to the question state")
    parser.add_argument("--passage-style-variant", default="",
                        help="Probe-only exemplar style variant: single or multi3")
    parser.add_argument("--exclude-gold-ids", default="",
                        help="Comma/whitespace-separated gold ids to exclude from question loading")
    parser.add_argument("--exclude-gold-ids-path", default="",
                        help="JSON/TXT file of gold ids to exclude from question loading")
    parser.add_argument("--concurrency", type=int, default=0,
                        help="OpenRouter worker count; 0 uses EVAL_CONCURRENCY or the provider default")

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
        sample_start=args.sample_start,
        sample_end=args.sample_end,
        hyre_cache_path=args.hyre_cache_path,
        retrieval_cache_path=args.retrieval_cache_path,
        housing_state_filter=args.housing_state_filter,
        passage_style_variant=args.passage_style_variant,
        exclude_gold_ids=args.exclude_gold_ids,
        exclude_gold_ids_path=args.exclude_gold_ids_path,
        concurrency=args.concurrency,
    )

    run_eval(config)


if __name__ == "__main__":
    main()
