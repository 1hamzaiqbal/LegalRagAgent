#!/usr/bin/env python3
"""Build CSQE generation caches for legal regime-sweep datasets.

CSQE here means corpus-steered query expansion: start from the raw-question
top-k cache, show those real retrieved passages to the generator, ask it to
extract pivotal real sentences, then use the resulting search passage as the
generated query consumed by ``scripts/build_retrieval_cache.py --query-type
csqe_cache``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from eval_config import EvalConfig, load_questions  # noqa: E402
from eval_harness import (  # noqa: E402
    _contains_answer_artifact,
    _fmt_intermediate,
    _get_metrics,
    _llm_call,
    _provider_route_metadata,
    _reset_call_trace,
    _reset_llm_call_counter,
    _reset_trace_events,
    _retrieval_question,
    _row_label,
    _setup_provider,
)
from rag_utils import get_documents_by_idx  # noqa: E402


MODEL = "or-gemma4-26b"
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class DatasetSpec:
    dataset: str
    display: str
    collection: str
    raw_cache: Path
    out: Path
    housing_state_filter: bool = False


def dataset_specs() -> dict[str, DatasetSpec]:
    return {
        "barexam": DatasetSpec(
            dataset="barexam",
            display="BarExamQA",
            collection="legal_passages",
            raw_cache=ROOT / "caches/retrieval/full/barexam_qfull_seed42_raw_question_k10.jsonl",
            out=ROOT / f"caches/generation/full/barexam_qfull_seed42_{MODEL}_csqe.jsonl",
        ),
        "housing": DatasetSpec(
            dataset="housing",
            display="HousingQA",
            collection="housing_statutes",
            raw_cache=ROOT / "caches/retrieval/full/housing_qfull_seed42_statefilter_raw_question_k10.jsonl",
            out=ROOT / f"caches/generation/full/housing_qfull_seed42_statefilter_{MODEL}_csqe.jsonl",
            housing_state_filter=True,
        ),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _row_idx(row: Any, fallback_i: int) -> str:
    value = row.get("idx", fallback_i)
    try:
        if value != value:
            return str(fallback_i)
    except Exception:
        pass
    return str(value)


def _load_existing(path: Path) -> set[str]:
    if not path.exists():
        return set()
    labels: set[str] = set()
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            label = row.get("label")
            if label:
                labels.add(str(label))
    return labels


def _doc_text(title: str, text: str) -> str:
    title = str(title or "").strip()
    text = str(text or "").strip()
    if title and title not in text[: max(80, len(title) + 10)]:
        return f"{title}. {text}"
    return text or title


def _sanitize_text(text: str, limit: int = 900) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].strip()
    return cut + "."


def _doc_excerpt(text: str, limit: int = 900) -> str:
    text = _sanitize_text(text, limit=limit)
    if not text:
        return ""
    sentences = [part.strip() for part in SENTENCE_RE.split(text) if part.strip()]
    if not sentences:
        return text
    out: list[str] = []
    chars = 0
    for sentence in sentences:
        if chars and chars + len(sentence) + 1 > limit:
            break
        out.append(sentence)
        chars += len(sentence) + 1
    return " ".join(out).strip() or text


def _build_doc_lookup(collection: str, ids: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    unique = list(dict.fromkeys(str(idx) for idx in ids if str(idx)))
    for start in range(0, len(unique), 5000):
        chunk = unique[start : start + 5000]
        docs = get_documents_by_idx(collection, chunk)
        for doc in docs:
            idx = str(doc.metadata.get("idx") or "")
            if idx:
                out[idx] = _doc_text(str(doc.metadata.get("title") or ""), doc.page_content)
        print(f"[docs] {collection}: {min(start + len(chunk), len(unique))}/{len(unique)} ids", flush=True)
    return out


def _question_rows(config: EvalConfig) -> dict[str, tuple[int, Any]]:
    return {
        _row_label(row, config, fallback_i=fallback_i): (fallback_i, row)
        for fallback_i, row in load_questions(config).iterrows()
    }


def _parse_csqe(raw: str) -> tuple[list[str], str, bool]:
    text = str(raw or "").strip()
    if not text:
        return [], "", False
    passage_match = re.search(r"(?is)##\s*Search\s+Passage\s*\n(.+)$", text)
    passage = passage_match.group(1).strip() if passage_match else ""
    if passage:
        passage = re.split(r"(?is)\n\s*##\s+", passage, maxsplit=1)[0].strip()
    else:
        passage = text
    passage = re.sub(r"(?im)^\s*(?:search\s+passage|passage)\s*:\s*", "", passage).strip()
    passage = re.sub(r"(?im)^\s*[-*]\s*", "", passage).strip()
    passage = _sanitize_text(passage, limit=1600)

    sent_block = ""
    sent_match = re.search(r"(?is)##\s*(?:Extracted\s+)?Sentences\s*\n(.+?)(?:\n\s*##\s*Search\s+Passage|\Z)", text)
    if sent_match:
        sent_block = sent_match.group(1)
    sentences = []
    for line in sent_block.splitlines():
        cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
        cleaned = re.sub(r"^\[[^\]]+\]\s*", "", cleaned).strip()
        if cleaned:
            sentences.append(_sanitize_text(cleaned, limit=320))
    valid = bool(passage) and not _contains_answer_artifact(passage)
    return sentences[:5], passage, valid


def _system_prompt() -> str:
    return (
        "You build corpus-steered retrieval queries. Use only the provided real retrieved "
        "corpus passages. Extract pivotal sentences or sentence fragments that preserve the "
        "source wording and terminology, then combine them into a concise search passage. "
        "Do not answer the question, do not mention answer choices, and do not invent law or facts."
    )


def _build_user_prompt(*, question: str, docs: list[tuple[str, str]], retry_raw: str = "") -> str:
    parts = [
        "## Question",
        question,
        "",
        "## Raw-Retrieved Corpus Passages",
    ]
    for rank, (doc_id, text) in enumerate(docs, 1):
        parts.extend([f"[Doc {rank}: {doc_id}]", _doc_excerpt(text), ""])
    parts.extend([
        "## Required Output",
        "Return exactly these two sections:",
        "## Sentences",
        "- [Doc id] one pivotal real sentence or sentence fragment",
        "- [Doc id] another pivotal real sentence or sentence fragment",
        "## Search Passage",
        "A 2-4 sentence retrieval passage built only from those real corpus snippets.",
    ])
    if retry_raw:
        parts.extend([
            "",
            "## Previous Malformed Output",
            _sanitize_text(retry_raw, limit=1200),
            "",
            "Repair it now. Keep the same two section headers and do not include an answer.",
        ])
    return "\n".join(parts)


def _build_one(
    order_i: int,
    label: str,
    row: Any,
    raw_row: dict[str, Any],
    *,
    config: EvalConfig,
    spec: DatasetSpec,
    docs_by_id: dict[str, str],
    raw_top_k: int,
) -> tuple[int, dict[str, Any]]:
    _reset_llm_call_counter()
    _reset_call_trace()
    _reset_trace_events()
    start = time.time()
    retrieved_ids = [str(idx) for idx in (raw_row.get("retrieved_ids") or [])[:raw_top_k]]
    docs = [(idx, docs_by_id.get(idx, "")) for idx in retrieved_ids if docs_by_id.get(idx, "")]
    if not docs:
        raise RuntimeError(f"{label}: no raw-retrieved documents available for CSQE")
    question = _retrieval_question(row)
    raw = _llm_call(
        _system_prompt(),
        _build_user_prompt(question=question, docs=docs),
        label="csqe/extract",
    )
    sentences, passage, valid = _parse_csqe(raw)
    retry_used = False
    retry_valid = False
    if not valid and os.getenv("EVAL_GENERATION_FORMAT_RETRY", "").strip().lower() in {"1", "true", "yes", "on"}:
        retry_used = True
        raw_retry = _llm_call(
            _system_prompt(),
            _build_user_prompt(question=question, docs=docs, retry_raw=raw),
            label="csqe/extract/format_retry",
        )
        retry_sentences, retry_passage, retry_valid = _parse_csqe(raw_retry)
        if retry_valid:
            raw = raw_retry
            sentences = retry_sentences
            passage = retry_passage
            valid = True

    contains_answer = _contains_answer_artifact(passage)
    metrics = _get_metrics()
    record = {
        "label": label,
        "idx": _row_idx(row, order_i),
        "dataset": spec.dataset,
        "mode": "csqe",
        "source_mode": "csqe",
        "provider": config.provider,
        "provider_route": _provider_route_metadata(),
        "elapsed_sec": round(time.time() - start, 1),
        "llm_calls": metrics["count"],
        "input_tokens": metrics["input_tokens"],
        "output_tokens": metrics["output_tokens"],
        "hyde_passage": passage,
        "hyde_passage_raw": raw,
        "hyde_contains_answer_artifact": contains_answer,
        "hyde_used_fallback": False,
        "hyde_parse_ok": valid,
        "csqe_raw_top_k": raw_top_k,
        "csqe_source_ids": retrieved_ids,
        "csqe_extracted_sentences": sentences,
        "csqe_format_retry": retry_used,
        "csqe_format_retry_valid": retry_valid,
        "csqe_question_hash": str(raw_row.get("question_hash") or ""),
    }
    violations: list[str] = []
    if not passage:
        violations.append("missing_hyde_passage")
    if not valid:
        violations.append("csqe_parse_ok=False")
    if contains_answer:
        violations.append("hyde_contains_answer_artifact=True")
    if os.getenv("NO_SILENT_FALLBACK", "").strip().lower() in {"1", "true", "yes", "on"} and violations:
        raise RuntimeError(f"{label}: " + "; ".join(violations))
    return order_i, record


def _chunks(values: list[tuple[int, str, Any, dict[str, Any]]], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def build_dataset(args: argparse.Namespace, spec: DatasetSpec) -> None:
    config = EvalConfig(
        mode="rag_simple",
        provider=args.provider,
        questions="full",
        seed=args.seed,
        dataset=spec.dataset,
        housing_state_filter=spec.housing_state_filter,
        concurrency=args.concurrency,
    )
    _setup_provider(config)
    raw_rows = read_jsonl(spec.raw_cache)
    raw_by_label = {str(row["label"]): row for row in raw_rows}
    q_by_label = _question_rows(config)
    labels = list(q_by_label)
    missing = [label for label in labels if label not in raw_by_label]
    if missing:
        raise SystemExit(f"{spec.dataset}: raw retrieval cache missing {len(missing)} labels, first={missing[:5]}")

    all_doc_ids = [
        str(idx)
        for row in raw_rows
        for idx in (row.get("retrieved_ids") or [])[: args.raw_top_k]
    ]
    docs_by_id = _build_doc_lookup(spec.collection, all_doc_ids)
    if not docs_by_id:
        raise SystemExit(f"{spec.dataset}: no raw-retrieved docs resolved from {spec.collection}")

    spec.out.parent.mkdir(parents=True, exist_ok=True)
    done = _load_existing(spec.out) if args.resume else set()
    pending = [
        (order_i, label, row, raw_by_label[label])
        for order_i, label in enumerate(labels)
        for _, row in [q_by_label[label]]
        if label not in done
    ]
    open_mode = "a" if args.resume else "w"
    print(
        f"[csqe] dataset={spec.dataset} pending={len(pending)} done={len(done)} "
        f"workers={args.concurrency} out={spec.out}",
        flush=True,
    )
    wrote = 0
    with spec.out.open(open_mode) as f:
        for batch in _chunks(pending, args.batch_size):
            records: dict[int, dict[str, Any]] = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                futures = {
                    executor.submit(
                        _build_one,
                        order_i,
                        label,
                        row,
                        raw_row,
                        config=config,
                        spec=spec,
                        docs_by_id=docs_by_id,
                        raw_top_k=args.raw_top_k,
                    ): label
                    for order_i, label, row, raw_row in batch
                }
                for future in concurrent.futures.as_completed(futures):
                    label = futures[future]
                    try:
                        order_i, record = future.result()
                    except Exception as exc:
                        for pending_future in futures:
                            pending_future.cancel()
                        raise SystemExit(f"CSQE generation failed for {label}: {exc}") from exc
                    records[order_i] = record
            for order_i in sorted(records):
                f.write(json.dumps(records[order_i], sort_keys=True) + "\n")
                f.flush()
                wrote += 1
                if args.progress_interval > 0 and wrote % args.progress_interval == 0:
                    rec = records[order_i]
                    print(
                        f"[{wrote}] {rec['label']:<35} OK "
                        f"({rec['elapsed_sec']:.1f}s, {rec['llm_calls']} calls)",
                        flush=True,
                    )
    print(f"wrote={wrote} out={spec.out}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["barexam", "housing", "all"], default="all")
    parser.add_argument("--provider", default=MODEL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--raw-top-k", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=int(os.getenv("EVAL_CONCURRENCY", "8") or 8))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-interval", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.concurrency <= 0:
        raise SystemExit("--concurrency must be positive")
    specs = dataset_specs()
    selected = list(specs) if args.dataset == "all" else [args.dataset]
    for key in selected:
        build_dataset(args, specs[key])


if __name__ == "__main__":
    main()
