#!/usr/bin/env python3
"""Run answer-free retrieval probes for choice-aware HyRE variants.

This script is intentionally narrower than the full answer harness: it uses the
same question formatting, generation prompts, Chroma collections, and retrieval
helpers, but stops before the final answer LLM call. The output is a JSONL file
with retrieved IDs plus Hit/Recall/MRR summaries at configurable k values.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from eval_config import EvalConfig, load_questions  # noqa: E402
from eval_harness import (  # noqa: E402
    _collection_for_config,
    _contains_answer_artifact,
    _choice_texts,
    _coerce_gold_ids,
    _extract_answer,
    _extract_predicted_answer,
    _fmt,
    _fmt_intermediate,
    _generate_hyde,
    _generate_snap_hyre_blocks,
    _get_call_trace,
    _get_metrics,
    _get_trace_events,
    _gold_ids,
    _orthogonal_passage_style_signals,
    _question_only_hyde_user,
    _reset_call_trace,
    _reset_llm_call_counter,
    _reset_trace_events,
    _retrieve_and_format,
    _retrieval_where_for_row,
    _retrieval_question,
    _row_label,
    _sanitize_intermediate_text,
    _setup_provider,
    _snap_choice_hyre_system,
    _snap_hyde_2call_system,
    _split_choice_hyre,
    _split_snap_and_hyde,
    _llm_call,
)


DEFAULT_PROBE_MODES = (
    "rag_simple",
    "rag_choice_simple",
    "rag_hyde_blind",
    "rag_hyde_choice",
    "snap_hyre",
    "snap_hyre_anchor",
    "multi_hyde_diverse",
    "snap_choice_hyre",
)

CHOICE_PROBE_MODES = DEFAULT_PROBE_MODES + (
    "rag_choice_fused",
    "rag_hyde",
    "snap_issue_hyre",
    "snap_hyre_exemplar_parallel3",
)

_GENERATION_MEMO: dict[tuple[str, str, str], dict[str, Any]] = {}


def _truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _row_idx(row: Any, fallback_i: int) -> str:
    value = row.get("idx", fallback_i)
    try:
        if value != value:
            return str(fallback_i)
    except Exception:
        pass
    return str(value)


def _choice_label_artifact(text: str) -> bool:
    if not text:
        return False
    return bool(
        re.search(r"(?im)^\s*(?:\([A-E]\)|[A-E][.)])\s+", text)
        or re.search(r"(?i)\b(?:option|choice)\s+[A-E]\b", text)
    )


def _generated_text_violations(texts: list[str], *, allow_snap_block: bool = False) -> list[str]:
    violations: list[str] = []
    for i, text in enumerate(texts, 1):
        if not str(text or "").strip():
            violations.append(f"generated_text_{i}_empty")
        if "<think>" in str(text or "").lower():
            violations.append(f"generated_text_{i}_think_tag")
        if _contains_answer_artifact(text):
            violations.append(f"generated_text_{i}_answer_artifact")
        if not allow_snap_block and _choice_label_artifact(text):
            violations.append(f"generated_text_{i}_choice_label_artifact")
    return violations


def _split_multi_hyde(raw: str, fallback: str) -> tuple[list[str], list[str], bool]:
    raw_passages = [
        p.strip()
        for p in str(raw or "").split("\n\n")
        if p.strip() and len(p.strip()) > 30
    ]
    passages = [
        _sanitize_intermediate_text(p, fallback=fallback)
        for p in raw_passages
    ]
    passages = [p for p in passages if p.strip()]
    return raw_passages, passages, len(passages) >= 3


def _context_only_prompt(row: Any, config: EvalConfig) -> str:
    """Retrieval-generation context without answer options/candidate holdings."""
    if config.dataset in ("casehold", "legalbench_scalr"):
        return "\n\n".join([
            "The following excerpt from a court opinion cites a legal holding.",
            f"## Citing Context\n{str(row.get('question', '')).strip()}",
        ])
    return _retrieval_question(row)


def _memo_key(row: Any, config: EvalConfig, kind: str) -> tuple[str, str, str]:
    return (config.dataset, str(row.get("idx", "")), kind)


def _snap_issue_hyre_system(config: EvalConfig) -> str:
    return (
        "You are a legal research assistant preparing a retrieval query for a legal QA task. "
        "Reason briefly about the controlling legal issue, without seeing or choosing answer "
        "options. Then write one retrieval passage.\n\n"
        "Output exactly this structure:\n"
        "ISSUE: <one sentence identifying the controlling legal issue>\n\n"
        "## Passage\n"
        "<2-3 sentence legal-reference passage about the controlling doctrine, rule, holding, "
        "or distinction that would help retrieve relevant corpus evidence>\n\n"
        "Passage rules:\n"
        "- Do not include answer letters, Yes/No labels, or 'Answer:' inside the passage.\n"
        "- Do not mention option labels, candidate answers, or that choices exist.\n"
        "- Write in legal reference, case holding, or treatise style.\n"
        "- Stop immediately after the passage."
    )


def _build_rag_hyde_queries(
    row: Any,
    config: EvalConfig,
    *,
    question: str,
    question_intermediate: str,
    query_context: str,
    memo_kind: str,
    source_mode: str,
) -> dict[str, Any]:
    key = _memo_key(row, config, memo_kind)
    payload = _GENERATION_MEMO.get(key)
    if payload is None:
        hyde = _generate_hyde(
            config,
            "hyde",
            _question_only_hyde_user(query_context),
            label=f"{source_mode}/generate",
            fallback=query_context,
        )
        payload = {
            "hyde_passage": hyde["text"],
            "hyde_passage_raw": hyde["raw"],
            "hyde_contains_answer_artifact": hyde["contains_answer"],
            "hyde_used_fallback": hyde.get("used_fallback", False),
            "generation_parse_ok": bool(hyde["text"]),
        }
        _GENERATION_MEMO[key] = dict(payload)
    return {
        "source_mode": source_mode,
        "retrieval_queries": [payload["hyde_passage"]],
        "formatted_question": question,
        "intermediate_question": question_intermediate,
        "query_context": query_context,
        "generated_passages": [payload["hyde_passage"]],
        **payload,
    }


def _build_snap_hyre_queries(
    row: Any,
    config: EvalConfig,
    *,
    question: str,
    question_intermediate: str,
    source_mode: str,
    include_anchor: bool,
) -> dict[str, Any]:
    key = _memo_key(row, config, "snap_hyre")
    payload = _GENERATION_MEMO.get(key)
    if payload is None:
        raw = _llm_call(_snap_hyde_2call_system(config), question, label="snap_hyre/snap_and_hyre")
        snap_block, hyre_passage, parse_ok = _split_snap_and_hyde(
            raw,
            fallback_passage=question_intermediate,
        )
        payload = {
            "generation_parse_ok": parse_ok,
            "snap_answer": snap_block,
            "snap_letter": _extract_answer(snap_block, config),
            "snap_and_hyre_raw": raw,
            "hyde_passage": hyre_passage,
            "hyde_passage_raw": raw,
            "hyde_contains_answer_artifact": _contains_answer_artifact(hyre_passage),
        }
        _GENERATION_MEMO[key] = dict(payload)
    queries = [payload["hyde_passage"]]
    if include_anchor:
        queries.append(question_intermediate)
    return {
        "source_mode": source_mode,
        "retrieval_queries": queries,
        "formatted_question": question,
        "intermediate_question": question_intermediate,
        "raw_anchor_included": include_anchor,
        "generated_passages": [payload["hyde_passage"]],
        **payload,
    }


def _build_snap_hyre_exemplar_parallel3_queries(
    row: Any,
    config: EvalConfig,
    *,
    question: str,
    question_intermediate: str,
) -> dict[str, Any]:
    signals = _orthogonal_passage_style_signals(config)
    if len(signals) < 3:
        raise ValueError(f"no three-signal exemplar bank for dataset={config.dataset}")

    raw_outputs: list[str] = []
    snap_blocks: list[str] = []
    snap_letters: list[str] = []
    passages: list[str] = []
    parse_flags: list[bool] = []
    retry_meta: list[dict[str, Any]] = []
    signal_keys: list[str] = []
    signal_ids: list[list[str]] = []

    for signal in signals[:3]:
        key = str(signal.get("key") or f"signal_{len(signal_keys) + 1}")
        raw, snap_block, hyre_passage, parse_ok, meta = _generate_snap_hyre_blocks(
            config,
            question=question,
            fallback_passage=question_intermediate,
            label=f"snap_hyre_exemplar_parallel3/{key}",
            use_style_signal=True,
            style_signal_override=str(signal.get("signal") or ""),
        )
        raw_outputs.append(raw)
        snap_blocks.append(snap_block)
        snap_letters.append(str(_extract_answer(snap_block, config) or ""))
        passages.append(hyre_passage)
        parse_flags.append(bool(parse_ok))
        retry_meta.append(meta)
        signal_keys.append(key)
        signal_ids.append([str(value) for value in signal.get("ids", [])])

    return {
        "source_mode": "snap_hyre_exemplar_parallel3",
        "retrieval_queries": passages,
        "formatted_question": question,
        "intermediate_question": question_intermediate,
        "generation_parse_ok": all(parse_flags),
        "parallel_exemplar_signal_keys": signal_keys,
        "parallel_exemplar_signal_ids": signal_ids,
        "snap_answers": snap_blocks,
        "snap_letters": snap_letters,
        "snap_and_hyre_raws": raw_outputs,
        "hyde_passages": passages,
        "n_hyde_passages": len(passages),
        "hyde_contains_answer_artifact": any(_contains_answer_artifact(p) for p in passages),
        "raw_anchor_included": False,
        "parallel_generation_calls": len(passages),
        "parallel_generation_parse_flags": parse_flags,
        "parallel_generation_retry_meta": retry_meta,
        "generated_passages": passages,
    }


def _build_queries(row: Any, config: EvalConfig, mode: str) -> dict[str, Any]:
    question = _fmt(row, config)
    question_intermediate = _fmt_intermediate(row, config)
    context_only = _context_only_prompt(row, config)

    if mode == "rag_simple":
        return {
            "source_mode": mode,
            "retrieval_queries": [_retrieval_question(row)],
            "formatted_question": question,
            "intermediate_question": question_intermediate,
            "generation_parse_ok": True,
            "generated_passages": [],
        }

    if mode == "rag_choice_fused":
        return {
            "source_mode": mode,
            "retrieval_queries": [question_intermediate],
            "formatted_question": question,
            "intermediate_question": question_intermediate,
            "generation_parse_ok": True,
            "choice_query_mode": "fused_unlabeled_choices",
            "generated_passages": [],
        }

    if mode == "rag_choice_simple":
        choices = list(_choice_texts(row, config).values())
        queries = [_retrieval_question(row)] + choices
        return {
            "source_mode": mode,
            "retrieval_queries": queries,
            "formatted_question": question,
            "intermediate_question": question_intermediate,
            "generation_parse_ok": True,
            "choice_query_mode": "raw_question_plus_unlabeled_choice_texts",
            "n_choice_queries": len(choices),
            "generated_passages": [],
        }

    if mode == "rag_hyde_blind":
        return _build_rag_hyde_queries(
            row,
            config,
            question=question,
            question_intermediate=question_intermediate,
            query_context=context_only,
            memo_kind="rag_hyde_blind",
            source_mode=mode,
        )

    if mode in {"rag_hyde", "rag_hyde_choice"}:
        return _build_rag_hyde_queries(
            row,
            config,
            question=question,
            question_intermediate=question_intermediate,
            query_context=question_intermediate,
            memo_kind="rag_hyde_choice",
            source_mode=mode,
        )

    if mode == "snap_issue_hyre":
        raw = _llm_call(_snap_issue_hyre_system(config), context_only, label="snap_issue_hyre/generate")
        snap_block, hyre_passage, parse_ok = _split_snap_and_hyde(
            raw,
            fallback_passage=context_only,
        )
        return {
            "source_mode": mode,
            "retrieval_queries": [hyre_passage],
            "formatted_question": question,
            "intermediate_question": question_intermediate,
            "query_context": context_only,
            "generation_parse_ok": parse_ok,
            "snap_answer": snap_block,
            "snap_and_hyre_raw": raw,
            "hyde_passage": hyre_passage,
            "hyde_passage_raw": raw,
            "hyde_contains_answer_artifact": _contains_answer_artifact(hyre_passage),
            "generated_passages": [hyre_passage],
        }

    if mode == "snap_hyre":
        return _build_snap_hyre_queries(
            row,
            config,
            question=question,
            question_intermediate=question_intermediate,
            source_mode=mode,
            include_anchor=False,
        )

    if mode == "snap_hyre_anchor":
        return _build_snap_hyre_queries(
            row,
            config,
            question=question,
            question_intermediate=question_intermediate,
            source_mode=mode,
            include_anchor=True,
        )

    if mode == "snap_hyre_exemplar_parallel3":
        return _build_snap_hyre_exemplar_parallel3_queries(
            row,
            config,
            question=question,
            question_intermediate=question_intermediate,
        )

    if mode == "multi_hyde_diverse":
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
        raw_passages, passages, parse_ok = _split_multi_hyde(raw, question_intermediate)
        queries = passages + [question_intermediate]
        return {
            "source_mode": mode,
            "retrieval_queries": queries,
            "formatted_question": question,
            "intermediate_question": question_intermediate,
            "generation_parse_ok": parse_ok,
            "multi_hyde_raw": raw,
            "hyde_passages": passages,
            "hyde_passages_raw": raw_passages,
            "n_hyde_passages": len(passages),
            "hyde_contains_answer_artifact": any(_contains_answer_artifact(p) for p in passages),
            "raw_anchor_included": True,
            "generated_passages": passages,
        }

    if mode == "snap_choice_hyre":
        raw = _llm_call(
            _snap_choice_hyre_system(config),
            question,
            label="snap_choice_hyre/snap_and_choice_hyre",
        )
        snap_block, passages, parse_ok = _split_choice_hyre(raw, fallback_passage=question_intermediate)
        queries = passages + [question_intermediate]
        return {
            "source_mode": mode,
            "retrieval_queries": queries,
            "formatted_question": question,
            "intermediate_question": question_intermediate,
            "generation_parse_ok": parse_ok,
            "choice_hyre_generation_raw": raw,
            "snap_answer": snap_block,
            "snap_letter": _extract_predicted_answer(snap_block, config),
            "choice_hyre_passages": passages,
            "hyde_passages": passages,
            "n_choice_hyre_passages": len(passages),
            "hyde_contains_answer_artifact": any(_contains_answer_artifact(p) for p in passages),
            "raw_anchor_included": True,
            "generated_passages": passages,
        }

    raise ValueError(f"unsupported mode: {mode}")


def _mrr(retrieved: list[str], gold: set[str], k: int) -> float:
    for rank, doc_id in enumerate(retrieved[:k], 1):
        if doc_id in gold:
            return 1.0 / rank
    return 0.0


def _row_metrics(retrieved: list[str], gold_ids: list[str], ks: list[int]) -> dict[str, Any]:
    gold = {str(value) for value in gold_ids if str(value)}
    out: dict[str, Any] = {"scored": bool(gold)}
    if not gold:
        for k in ks:
            out[f"hit@{k}"] = None
            out[f"recall@{k}"] = None
            out[f"mrr@{k}"] = None
        return out
    retrieved = [str(value) for value in retrieved]
    for k in ks:
        top = retrieved[:k]
        hits = len(gold & set(top))
        out[f"hit@{k}"] = 1.0 if hits else 0.0
        out[f"recall@{k}"] = hits / len(gold)
        out[f"mrr@{k}"] = _mrr(top, gold, k)
    return out


def _retrieval_proxy_metrics(row: Any, retrieval: dict[str, Any], ks: list[int]) -> dict[str, Any]:
    evidence = list(retrieval.get("evidence_store") or [])
    retrieved = [str(ev.get("idx", "")) for ev in evidence]
    row_source = str(row.get("source", "") or "").strip()
    source_context_ids = _coerce_gold_ids(row.get("source_context_ids", ""))
    source_doc = str(row.get("source_doc", "") or "").strip()
    target_doc = str(row.get("target_doc", "") or "").strip()

    out: dict[str, Any] = {
        "row_source": row_source,
        "same_source_retrieved": bool(retrieval.get("same_source_retrieved", False)),
        "same_source_retrieved_ids": retrieval.get("same_source_retrieved_ids", []),
        "source_context_ids": source_context_ids,
        "source_doc": source_doc,
        "source_doc_retrieved": bool(retrieval.get("source_doc_retrieved", False)),
        "source_doc_retrieved_ids": retrieval.get("source_doc_retrieved_ids", []),
        "target_doc": target_doc,
        "target_doc_retrieved": bool(retrieval.get("target_doc_retrieved", False)),
        "target_doc_retrieved_ids": retrieval.get("target_doc_retrieved_ids", []),
        "cross_encoder_doc_truncated_count": retrieval.get("cross_encoder_doc_truncated_count", 0),
        "cross_encoder_query_truncated": bool(retrieval.get("cross_encoder_query_truncated", False)),
        "cross_encoder_max_chars": retrieval.get("cross_encoder_max_chars", ""),
    }

    if row_source:
        for k in ks:
            top = evidence[:k]
            hits = [
                ev.get("idx", "")
                for ev in top
                if str(ev.get("source", "") or "").strip() == row_source
            ]
            out[f"same_source_hit@{k}"] = 1.0 if hits else 0.0
            out[f"same_source_count@{k}"] = len(hits)

    if source_context_ids:
        context_set = {str(value) for value in source_context_ids if str(value)}
        for k in ks:
            top = retrieved[:k]
            hits = len(context_set & set(top))
            out[f"source_context_hit@{k}"] = 1.0 if hits else 0.0
            out[f"source_context_recall@{k}"] = hits / len(context_set)
            out[f"source_context_mrr@{k}"] = _mrr(top, context_set, k)

    if source_doc or target_doc:
        for k in ks:
            source_at_k = False
            target_at_k = False
            for ev in evidence[:k]:
                ev_source = str(ev.get("source", "") or "").strip()
                ev_citation = str(ev.get("citation", "") or "").strip()
                ev_text_id = ev_source or ev_citation
                if source_doc and (
                    ev_text_id == source_doc
                    or ev_source == source_doc
                    or ev_citation == source_doc
                ):
                    source_at_k = True
                if target_doc and (
                    ev_text_id == target_doc
                    or ev_source == target_doc
                    or ev_citation == target_doc
                ):
                    target_at_k = True
            out[f"source_doc_hit@{k}"] = 1.0 if source_at_k else 0.0
            out[f"target_doc_hit@{k}"] = 1.0 if target_at_k else 0.0
            out[f"source_target_doc_hit@{k}"] = 1.0 if source_at_k and target_at_k else 0.0

    return out


def _strict_violations(record: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    if record.get("error"):
        violations.append(f"error={str(record.get('error'))[:160]}")
    if not record.get("retrieved_ids"):
        violations.append("empty_retrieval")
    if record.get("generation_parse_ok") is False:
        violations.append("generation_parse_ok=False")
    for key, value in record.items():
        if (key.endswith("_fallback") or key.endswith("_used_fallback")) and value:
            violations.append(f"{key}={value}")
        if key.endswith("_contains_answer_artifact") and value:
            violations.append(f"{key}=True")
    generated = [str(value) for value in record.get("generated_passages") or []]
    violations.extend(_generated_text_violations(generated))
    return violations


def _summarize(rows: list[dict[str, Any]], ks: list[int]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), str(row["mode"]))].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (dataset, mode), group in sorted(grouped.items()):
        scored = [row for row in group if row.get("scored")]
        item: dict[str, Any] = {
            "dataset": dataset,
            "mode": mode,
            "rows": len(group),
            "scored_rows": len(scored),
            "errors": sum(1 for row in group if row.get("error")),
            "parse_failures": sum(1 for row in group if row.get("generation_parse_ok") is False),
            "artifact_rows": sum(1 for row in group if row.get("hyde_contains_answer_artifact")),
            "avg_llm_calls": round(
                sum(float(row.get("llm_calls") or 0) for row in group) / max(1, len(group)),
                3,
            ),
        }
        for k in ks:
            for metric in ("hit", "recall", "mrr"):
                key = f"{metric}@{k}"
                values = [float(row[key]) for row in scored if row.get(key) is not None]
                item[key] = round(sum(values) / len(values), 4) if values else None
        summary_rows.append(item)
    return summary_rows


def _write_summary_markdown(path: Path, summary_rows: list[dict[str, Any]], ks: list[int]) -> None:
    header = ["dataset", "mode", "rows", "scored", "errors", "parse_fail", "artifacts", "calls"]
    for k in ks:
        header.extend([f"Hit@{k}", f"Recall@{k}", f"MRR@{k}"])
    lines = [
        "# Choice-Aware Retrieval Probe",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in summary_rows:
        values = [
            row["dataset"],
            row["mode"],
            str(row["rows"]),
            str(row["scored_rows"]),
            str(row["errors"]),
            str(row["parse_failures"]),
            str(row["artifact_rows"]),
            f"{float(row['avg_llm_calls']):.2f}",
        ]
        for k in ks:
            values.extend(
                "n/a" if row.get(f"{metric}@{k}") is None else f"{float(row[f'{metric}@{k}']):.4f}"
                for metric in ("hit", "recall", "mrr")
            )
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="or-gemma4-26b")
    parser.add_argument("--dataset", required=True, choices=[
        "barexam", "housing", "casehold", "legalbench_scalr",
        "legal_link_eu", "mas_legal_bench",
    ])
    parser.add_argument("--questions", default="20")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--sample-end", type=int)
    parser.add_argument("--modes", nargs="+", default=list(DEFAULT_PROBE_MODES), choices=CHOICE_PROBE_MODES)
    parser.add_argument("--max-k", type=int, default=10)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--tag", default="")
    parser.add_argument("--source-filter", default="")
    parser.add_argument("--housing-state-filter", action="store_true",
                        help="For HousingQA, constrain retrieval to each question's state metadata")
    parser.add_argument("--exclude-gold-ids", default="",
                        help="Comma/whitespace-separated gold ids to exclude from question loading")
    parser.add_argument("--exclude-gold-ids-path", default="",
                        help="JSON/TXT file of gold ids to exclude from question loading")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--trace-calls", action="store_true")
    parser.add_argument("--trace-events", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_k <= 0:
        raise SystemExit("--max-k must be positive")
    ks = sorted(set(args.ks))
    if any(k <= 0 or k > args.max_k for k in ks):
        raise SystemExit("--ks values must be between 1 and --max-k")
    if args.trace_calls:
        os.environ["EVAL_TRACE_CALLS"] = "1"
    if args.trace_events:
        os.environ["EVAL_TRACE_EVENTS"] = "1"

    config = EvalConfig(
        mode="rag_simple",
        provider=args.provider,
        questions=args.questions,
        seed=args.seed,
        dataset=args.dataset,
        sample_start=args.sample_start,
        sample_end=args.sample_end,
        source_filter=args.source_filter,
        retrieval_k=args.max_k,
        tag=args.tag,
        housing_state_filter=args.housing_state_filter,
        exclude_gold_ids=args.exclude_gold_ids,
        exclude_gold_ids_path=args.exclude_gold_ids_path,
    )
    _setup_provider(config)
    questions = load_questions(config)
    if args.sample_start or args.sample_end is not None:
        start = max(0, int(args.sample_start or 0))
        end = None if args.sample_end is None else max(start, int(args.sample_end))
        questions = questions.iloc[start:end].reset_index(drop=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    with args.out.open("w") as f:
        for mode in args.modes:
            config.mode = mode
            print(f"mode={mode} dataset={args.dataset} rows={len(questions)}", flush=True)
            for fallback_i, row in questions.iterrows():
                label = _row_label(row, config, fallback_i=fallback_i)
                _reset_llm_call_counter()
                _reset_call_trace()
                _reset_trace_events()
                start_time = time.time()
                error = ""
                payload: dict[str, Any] = {}
                retrieved_ids: list[str] = []
                try:
                    payload = _build_queries(row, config, mode)
                    retrieval = _retrieve_and_format(
                        row,
                        payload["retrieval_queries"],
                        k=args.max_k,
                        label_prefix=mode,
                        where=_retrieval_where_for_row(row, config),
                        collection=_collection_for_config(config),
                    )
                    retrieved_ids = [str(value) for value in retrieval["retrieved_ids"]]
                    payload.update({
                        "evidence_store": retrieval["evidence_store"],
                        "retrieved_ids": retrieved_ids,
                        "gold_retrieved": retrieval["gold_retrieved"],
                        "retrieval_cache_hit": retrieval["retrieval_cache_hit"],
                        "retrieval_query_hash": retrieval["retrieval_query_hash"],
                        **_retrieval_proxy_metrics(row, retrieval, ks),
                    })
                except Exception as exc:
                    error = str(exc)

                metrics = _get_metrics()
                gold_ids = _gold_ids(row)
                record: dict[str, Any] = {
                    "label": label,
                    "idx": _row_idx(row, fallback_i),
                    "dataset": args.dataset,
                    "mode": mode,
                    "provider": args.provider,
                    "tag": args.tag,
                    "elapsed_sec": round(time.time() - start_time, 2),
                    "error": error,
                    "gold_ids": gold_ids,
                    "effective_retrieved_ids": retrieved_ids,
                    "llm_calls": metrics["count"],
                    "input_tokens": metrics["input_tokens"],
                    "output_tokens": metrics["output_tokens"],
                    **payload,
                    **_row_metrics(retrieved_ids, gold_ids, ks),
                }
                if args.trace_calls:
                    record["call_trace"] = _get_call_trace()
                if args.trace_events:
                    record["trace_events"] = _get_trace_events()
                    record["trace_schema_version"] = 1

                if _truthy("NO_SILENT_FALLBACK"):
                    violations = _strict_violations(record)
                    if violations:
                        raise SystemExit(
                            f"NO_SILENT_FALLBACK blocked retrieval probe row {label} "
                            f"mode={mode}: " + "; ".join(violations)
                        )

                f.write(json.dumps(record, sort_keys=True) + "\n")
                f.flush()
                rows.append(record)
                status = "ERR" if error else "OK"
                print(
                    f"[{len(rows)}] {mode:<20} {label:<35} {status:<3} "
                    f"hit@5={record.get('hit@5')} calls={metrics['count']} "
                    f"({record['elapsed_sec']:.1f}s)",
                    flush=True,
                )

    summary_rows = _summarize(rows, ks)
    print(json.dumps(summary_rows, indent=2, sort_keys=True))
    if args.summary_out:
        _write_summary_markdown(args.summary_out, summary_rows, ks)
        print(f"summary={args.summary_out}")
    print(f"rows={len(rows)} out={args.out}")


if __name__ == "__main__":
    main()
