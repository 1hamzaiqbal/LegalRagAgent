#!/usr/bin/env python3
"""Build and analyze 3SCOPE+raw CE-reranked retrieval pools.

The new arm generates three exemplar-anchored SCOPE passages per query, retrieves
top-k dense candidates independently for the raw question and the three passages,
pools unique candidates, and CE-reranks the pool against the raw question.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_beir_phase1 import fetch_docs_by_idx, generation_passage, score_best_gold_ce  # noqa: E402
from eval_config import BEIR_DATASETS, EvalConfig, load_questions  # noqa: E402
from eval_harness import (  # noqa: E402
    _contains_answer_artifact,
    _gold_ids,
    _llm_call,
    _orthogonal_passage_style_signals,
    _provider_route_metadata,
    _retrieval_question,
    _retrieval_where_for_row,
    _row_label,
    _setup_provider,
)
from rag_utils import _retrieve_dense, get_cross_encoder, get_vectorstore, rerank_with_cross_encoder  # noqa: E402


MODEL = "or-gemma4-26b"
BEIR_ORDER = ["beir_scifact", "beir_nfcorpus", "beir_fiqa", "beir_trec_covid", "beir_scidocs"]
LEGAL_ORDER = ["barexam", "housing"]
ALL_ORDER = BEIR_ORDER + LEGAL_ORDER
DISPLAY = {
    "beir_scifact": "SciFact",
    "beir_nfcorpus": "NFCorpus",
    "beir_fiqa": "FiQA",
    "beir_trec_covid": "TREC-COVID",
    "beir_scidocs": "SciDocs",
    "barexam": "BarExamQA",
    "housing": "HousingQA state-filtered",
}
COLLECTION = {
    **{key: key for key in BEIR_ORDER},
    "barexam": "legal_passages",
    "housing": "housing_statutes",
}
JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
BAD_JSON_ESCAPE_RE = re.compile(r'\\(?!["\\/bfnrtu])')


@dataclass(frozen=True)
class ArmSpec:
    key: str
    display: str
    retrieval: Path
    generation: Path | None = None


def p(path: str) -> Path:
    return ROOT / path


def prefix(dataset: str) -> str:
    if dataset == "housing":
        return "housing_qfull_seed42_statefilter"
    return f"{dataset}_qfull_seed42"


def generation_path(dataset: str) -> Path:
    return p(f"caches/generation/full/{prefix(dataset)}_{MODEL}_3scope_raw.jsonl")


def pool_path(dataset: str, arm: str) -> Path:
    return p(f"caches/retrieval/full/{prefix(dataset)}_{MODEL}_{arm}_k5.jsonl")


def arm_specs(dataset: str) -> dict[str, ArmSpec]:
    if dataset in BEIR_ORDER:
        pre = prefix(dataset)
        return {
            "raw": ArmSpec("raw", "Raw", p(f"caches/retrieval/full/{pre}_raw_question_k10.jsonl")),
            "hyde": ArmSpec(
                "hyde",
                "HyDE",
                p(f"caches/retrieval/full/{pre}_{MODEL}_rag_hyde_k10.jsonl"),
                p(f"caches/generation/full/{pre}_{MODEL}_rag_hyde.jsonl"),
            ),
            "scope": ArmSpec(
                "scope",
                "SCOPE",
                p(f"caches/retrieval/full/{pre}_{MODEL}_snap_hyre_k10.jsonl"),
                p(f"caches/generation/full/{pre}_{MODEL}_snap_hyre.jsonl"),
            ),
            "csqe": ArmSpec(
                "csqe",
                "CSQE",
                p(f"caches/retrieval/full/{pre}_csqe_k10.jsonl"),
                p(f"caches/generation/full/{pre}_csqe.jsonl"),
            ),
            "raw_scope_pool": ArmSpec("raw_scope_pool", "raw∪SCOPE-pool", pool_path(dataset, "raw_scope_pool")),
            "three_scope_raw": ArmSpec("three_scope_raw", "3SCOPE+raw", pool_path(dataset, "3scope_raw_pool")),
        }
    if dataset == "barexam":
        pre = prefix(dataset)
        return {
            "raw": ArmSpec("raw", "Raw", p(f"caches/retrieval/full/{pre}_raw_question_k10.jsonl")),
            "hyde": ArmSpec(
                "hyde",
                "HyDE",
                p(f"caches/retrieval/full/{pre}_{MODEL}_rag_hyde_k10.jsonl"),
                p(f"caches/hyre/full/{pre}_{MODEL}_rag_hyde.jsonl"),
            ),
            "scope": ArmSpec(
                "scope",
                "SCOPE",
                p(f"caches/retrieval/full/{pre}_{MODEL}_snap_hyre_k10.jsonl"),
                p(f"caches/hyre/full/{pre}_{MODEL}_snap_hyre.jsonl"),
            ),
            "csqe": ArmSpec(
                "csqe",
                "CSQE",
                p(f"caches/retrieval/full/{pre}_{MODEL}_csqe_k10.jsonl"),
                p(f"caches/generation/full/{pre}_{MODEL}_csqe.jsonl"),
            ),
            "raw_scope_pool": ArmSpec("raw_scope_pool", "raw∪SCOPE-pool", pool_path(dataset, "raw_scope_pool")),
            "three_scope_raw": ArmSpec("three_scope_raw", "3SCOPE+raw", pool_path(dataset, "3scope_raw_pool")),
        }
    pre = prefix(dataset)
    gen_pre = "housing_qfull_seed42"
    return {
        "raw": ArmSpec("raw", "Raw", p(f"caches/retrieval/full/{pre}_raw_question_k10.jsonl")),
        "hyde": ArmSpec(
            "hyde",
            "HyDE",
            p(f"caches/retrieval/full/{pre}_{MODEL}_rag_hyde_k10.jsonl"),
            p(f"caches/hyre/full/{gen_pre}_{MODEL}_rag_hyde.jsonl"),
        ),
        "scope": ArmSpec(
            "scope",
            "SCOPE",
            p(f"caches/retrieval/full/{pre}_{MODEL}_snap_hyre_k10.jsonl"),
            p(f"caches/hyre/full/{gen_pre}_{MODEL}_snap_hyre.jsonl"),
        ),
        "csqe": ArmSpec(
            "csqe",
            "CSQE",
            p(f"caches/retrieval/full/{pre}_{MODEL}_csqe_k10.jsonl"),
            p(f"caches/generation/full/{pre}_{MODEL}_csqe.jsonl"),
        ),
        "raw_scope_pool": ArmSpec("raw_scope_pool", "raw∪SCOPE-pool", pool_path(dataset, "raw_scope_pool")),
        "three_scope_raw": ArmSpec("three_scope_raw", "3SCOPE+raw", pool_path(dataset, "3scope_raw_pool")),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def load_by_label(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("label") or row.get("idx")): row for row in read_jsonl(path)}


def load_existing_labels(path: Path) -> set[str]:
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
            labels.add(str(row.get("label") or row.get("idx")))
    return labels


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if finite(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def fmt(value: Any, digits: int = 3) -> str:
    if not finite(value):
        return "--"
    return f"{float(value):.{digits}f}"


def pct(value: Any) -> str:
    if not finite(value):
        return "--"
    return f"{100.0 * float(value):.1f}%"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def selected_datasets(phase: str) -> list[str]:
    if phase == "beir":
        return list(BEIR_ORDER)
    if phase == "legal":
        return list(LEGAL_ORDER)
    if phase == "all":
        return list(ALL_ORDER)
    raise SystemExit(f"unknown phase {phase!r}")


def load_exemplar_payload() -> dict[str, Any]:
    path = p("caches/exemplars/beir_orthogonal3_exemplars_2026-05-26.json")
    if not path.exists():
        return {"datasets": {}}
    payload = json.loads(path.read_text())
    return payload.get("datasets", payload)


def exemplar_signals(dataset: str) -> tuple[list[dict[str, Any]], list[str]]:
    if dataset in BEIR_ORDER:
        record = load_exemplar_payload().get(dataset, {})
        signals = []
        for i, ex in enumerate(record.get("exemplars", [])[:3], 1):
            idx = str(ex.get("idx") or "")
            excerpt = str(ex.get("excerpt") or "").strip()
            title = str(ex.get("title") or "").strip()
            source = str(ex.get("source") or "").strip()
            signal = (
                f"A useful retrieval passage for {DISPLAY[dataset]} should match this corpus style "
                "and topical specificity. This exemplar is only a style signal, not evidence for "
                "the current query; do not copy it.\n\n"
                f"Corpus passage excerpt {i}"
                f"{f' ({source})' if source else ''}"
                f"{f', title: {title}' if title else ''}"
                f", doc id {idx}: {excerpt}"
            )
            signals.append({"key": f"exemplar_{i}", "ids": [idx], "signal": signal})
        excludes = [str(idx) for idx in record.get("eval_exclude_gold_ids", []) if str(idx)]
        return signals, excludes
    config = EvalConfig(dataset=dataset, questions="full", seed=42, housing_state_filter=(dataset == "housing"))
    signals = _orthogonal_passage_style_signals(config)
    excludes = [idx for sig in signals for idx in sig.get("ids", [])]
    return signals, [str(idx) for idx in excludes if str(idx)]


def question_config(dataset: str, *, exclude_ids: list[str] | None = None) -> EvalConfig:
    return EvalConfig(
        dataset=dataset,
        questions="full",
        seed=42,
        housing_state_filter=(dataset == "housing"),
        exclude_gold_ids=",".join(exclude_ids or []),
    )


def load_question_rows(dataset: str) -> dict[str, tuple[int, Any, str, list[str]]]:
    _, excludes = exemplar_signals(dataset)
    config = question_config(dataset, exclude_ids=excludes)
    out: dict[str, tuple[int, Any, str, list[str]]] = {}
    for fallback_i, row in load_questions(config).iterrows():
        label = _row_label(row, config, fallback_i=fallback_i)
        out[label] = (
            fallback_i,
            row,
            _retrieval_question(row),
            [str(idx) for idx in _gold_ids(row) if str(idx)],
        )
    return out


def sanitize_text(text: Any, limit: int = 1600) -> str:
    out = re.sub(r"\s+", " ", str(text or "")).strip()
    out = re.sub(r"(?im)^\s*(?:answer|final answer)\s*:.*$", "", out).strip()
    if len(out) <= limit:
        return out
    cut = out[:limit].rsplit(" ", 1)[0].strip()
    return cut + "."


def system_prompt(signal: dict[str, Any]) -> str:
    return (
        "You generate SCOPE retrieval passages for evidence search. Use the exemplar only "
        "as a corpus-style and specificity signal; it is not evidence for the current query. "
        "Given the query, form a concise draft focus, then write one neutral retrieval passage "
        "that could retrieve the relevant corpus evidence. Do not copy the exemplar. Do not "
        "include answer labels, choice letters, yes/no answers, or `Answer:` in the passage.\n\n"
        f"## Exemplar Signal\n{signal.get('signal', '')}\n\n"
        "Return strict JSON only with keys `draft_focus` and `scope_passage`."
    )


def user_prompt(question: str, retry_raw: str = "") -> str:
    body = [
        "## Query",
        question,
        "",
        "## JSON Schema",
        '{"draft_focus":"one concise non-answer description","scope_passage":"2-3 sentence neutral corpus-style retrieval passage"}',
    ]
    if retry_raw:
        body.extend([
            "",
            "## Previous malformed output",
            sanitize_text(retry_raw, limit=1000),
            "",
            "Repair the output as strict JSON. Keep the passage neutral and free of answer labels.",
        ])
    return "\n".join(body)


def parse_scope_json(raw: str) -> tuple[str, str]:
    text = str(raw or "").strip()
    match = JSON_RE.search(text)
    candidate = match.group(0) if match else text
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        data = json.loads(BAD_JSON_ESCAPE_RE.sub(r"\\\\", candidate))
    focus = sanitize_text(data.get("draft_focus", ""), limit=500)
    passage = sanitize_text(data.get("scope_passage", ""), limit=1200)
    if not passage:
        raise ValueError("missing scope_passage")
    if _contains_answer_artifact(passage):
        raise ValueError("scope_passage contains answer artifact")
    return focus, passage


def call_scope(signal: dict[str, Any], question: str, label: str, max_retries: int) -> dict[str, Any]:
    last_raw = ""
    last_error = ""
    for attempt in range(max_retries + 1):
        try:
            raw = _llm_call(
                system_prompt(signal),
                user_prompt(question, retry_raw=last_raw if attempt else ""),
                label=label if attempt == 0 else f"{label}/format_retry",
            )
            focus, passage = parse_scope_json(raw)
            return {
                "draft_focus": focus,
                "scope_passage": passage,
                "raw": raw,
                "retry_count": attempt,
            }
        except Exception as exc:
            last_raw = locals().get("raw", last_raw)
            last_error = str(exc)
            time.sleep(min(8.0, 1.0 + attempt))
    raise RuntimeError(last_error)


def generate_one(
    order_i: int,
    label: str,
    row: Any,
    question: str,
    *,
    dataset: str,
    signals: list[dict[str, Any]],
    max_retries: int,
) -> tuple[int, dict[str, Any]]:
    start = time.time()
    outputs = [
        call_scope(signal, question, f"3scope_raw/{dataset}/{signal.get('key', i)}", max_retries)
        for i, signal in enumerate(signals[:3], 1)
    ]
    passages = [out["scope_passage"] for out in outputs]
    record = {
        "label": label,
        "idx": str(row.get("idx", order_i)),
        "dataset": dataset,
        "mode": "3scope_raw",
        "source_mode": "3scope_raw",
        "provider": MODEL,
        "provider_route": _provider_route_metadata(),
        "elapsed_sec": round(time.time() - start, 3),
        "scope_passages": passages,
        "draft_focuses": [out["draft_focus"] for out in outputs],
        "scope_raws": [out["raw"] for out in outputs],
        "scope_retry_counts": [out["retry_count"] for out in outputs],
        "scope_passage_hashes": [
            hashlib.sha256(passage.encode("utf-8", errors="ignore")).hexdigest()[:16]
            for passage in passages
        ],
        "exemplar_signal_keys": [str(sig.get("key") or f"signal_{i}") for i, sig in enumerate(signals[:3], 1)],
        "exemplar_signal_ids": [[str(idx) for idx in sig.get("ids", [])] for sig in signals[:3]],
        "n_scope_passages": len(passages),
        "question_hash": hashlib.sha256(question.encode("utf-8", errors="ignore")).hexdigest()[:16],
        "contains_answer_artifact": any(_contains_answer_artifact(passage) for passage in passages),
    }
    if len(passages) != 3 or record["contains_answer_artifact"]:
        raise RuntimeError(f"{label}: invalid generated passages")
    return order_i, record


def generate_phase(args: argparse.Namespace) -> None:
    config = EvalConfig(provider=MODEL, dataset="barexam", questions="full", concurrency=args.concurrency)
    _setup_provider(config)
    for dataset in selected_datasets(args.phase):
        signals, _ = exemplar_signals(dataset)
        if len(signals) < 3:
            raise SystemExit(f"{dataset}: expected 3 exemplar signals, got {len(signals)}")
        questions = load_question_rows(dataset)
        labels = list(questions)
        if args.limit:
            labels = labels[: min(args.limit, len(labels))]
        out_path = generation_path(dataset)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        done = load_existing_labels(out_path) if args.resume else set()
        pending = [
            (i, label, questions[label][1], questions[label][2])
            for i, label in enumerate(labels)
            if label not in done
        ]
        print(
            f"[generate] {dataset} labels={len(labels)} done={len(done)} pending={len(pending)} "
            f"workers={args.concurrency} out={rel(out_path)}",
            flush=True,
        )
        mode = "a" if args.resume and out_path.exists() else "w"
        with out_path.open(mode) as f:
            for batch_start in range(0, len(pending), args.batch_size):
                batch = pending[batch_start:batch_start + args.batch_size]
                records: dict[int, dict[str, Any]] = {}
                with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                    futures = {
                        executor.submit(
                            generate_one,
                            order_i,
                            label,
                            row,
                            question,
                            dataset=dataset,
                            signals=signals,
                            max_retries=args.max_retries,
                        ): label
                        for order_i, label, row, question in batch
                    }
                    for future in concurrent.futures.as_completed(futures):
                        label = futures[future]
                        try:
                            order_i, record = future.result()
                        except Exception as exc:
                            for pending_future in futures:
                                pending_future.cancel()
                            raise SystemExit(f"3SCOPE generation failed for {dataset}/{label}: {exc}") from exc
                        records[order_i] = record
                for order_i in sorted(records):
                    f.write(json.dumps(records[order_i], sort_keys=True) + "\n")
                    f.flush()
                    if args.progress_interval and (len(done) + len(records)) % args.progress_interval == 0:
                        pass
                wrote = min(batch_start + len(batch), len(pending))
                if args.progress_interval and wrote % args.progress_interval == 0:
                    print(f"[generate] {dataset} wrote_new={wrote}/{len(pending)}", flush=True)


def doc_id(doc: Any) -> str:
    return str(getattr(doc, "metadata", {}).get("idx") or "")


def dense_pool(
    *,
    queries: list[str],
    question: str,
    vectorstore: Any,
    where: dict[str, Any] | None,
    top_k: int,
) -> tuple[list[Any], list[list[str]], int]:
    lists = [_retrieve_dense(query, k=top_k, vectorstore=vectorstore, where=where) for query in queries]
    source_ids = [[doc_id(doc) for doc in docs] for docs in lists]
    seen: set[str] = set()
    pool: list[Any] = []
    for docs in lists:
        for doc in docs:
            idx = doc_id(doc)
            if idx and idx not in seen:
                seen.add(idx)
                pool.append(doc)
    ranked = rerank_with_cross_encoder(question, pool, top_k=5)
    return ranked, source_ids, len(pool)


def build_pool_phase(args: argparse.Namespace) -> None:
    for dataset in selected_datasets(args.phase):
        config = question_config(dataset, exclude_ids=exemplar_signals(dataset)[1])
        questions = load_question_rows(dataset)
        labels = list(questions)
        if args.limit:
            labels = labels[: min(args.limit, len(labels))]
        three_gen = load_by_label(generation_path(dataset))
        scope_gen = load_by_label(arm_specs(dataset)["scope"].generation)
        vectorstore = get_vectorstore(COLLECTION[dataset])
        outputs = {
            "3scope_raw_pool": pool_path(dataset, "3scope_raw_pool"),
            "raw_scope_pool": pool_path(dataset, "raw_scope_pool"),
        }
        for arm, out_path in outputs.items():
            out_path.parent.mkdir(parents=True, exist_ok=True)
            done = load_existing_labels(out_path) if args.resume else set()
            mode = "a" if args.resume and out_path.exists() else "w"
            wrote = 0
            with out_path.open(mode) as f:
                for row_i, label in enumerate(labels):
                    if label in done:
                        continue
                    fallback_i, row, question, gold_ids = questions[label]
                    where = _retrieval_where_for_row(row, config)
                    if arm == "3scope_raw_pool":
                        gen_row = three_gen.get(label)
                        if not gen_row:
                            raise SystemExit(f"{dataset}: missing 3SCOPE generation for {label}")
                        passages = [str(x) for x in gen_row.get("scope_passages", [])[:3]]
                        queries = [question] + passages
                    else:
                        gen_row = scope_gen.get(label)
                        if not gen_row:
                            raise SystemExit(f"{dataset}: missing canonical SCOPE generation for {label}")
                        queries = [question, generation_passage(gen_row)]
                    ranked, source_ids, pool_size = dense_pool(
                        queries=queries,
                        question=question,
                        vectorstore=vectorstore,
                        where=where,
                        top_k=args.retrieve_k,
                    )
                    retrieved_ids = [doc_id(doc) for doc in ranked]
                    scores = [float(doc.metadata.get("cross_encoder_score", 0.0) or 0.0) for doc in ranked]
                    record = {
                        "label": label,
                        "idx": str(row.get("idx", fallback_i)),
                        "dataset": dataset,
                        "query_type": arm,
                        "label_prefix": arm,
                        "provider": MODEL,
                        "collection": COLLECTION[dataset],
                        "where": where or {},
                        "housing_state_filter": bool(dataset == "housing"),
                        "max_k": 5,
                        "component_top_k": args.retrieve_k,
                        "component_count": len(queries),
                        "pool_size": pool_size,
                        "component_retrieved_ids": source_ids,
                        "retrieved_ids": retrieved_ids,
                        "scores": scores,
                        "gold_ids": gold_ids,
                        "gold_retrieved": bool(set(gold_ids) & set(retrieved_ids[:5])),
                        "ce_rerank_coverage": len(retrieved_ids) / min(5, pool_size) if pool_size else 0.0,
                        "question_hash": hashlib.sha256(question.encode("utf-8", errors="ignore")).hexdigest()[:16],
                    }
                    f.write(json.dumps(record, sort_keys=True) + "\n")
                    f.flush()
                    wrote += 1
                    if args.progress_interval and wrote % args.progress_interval == 0:
                        print(f"[pool] {dataset}/{arm} wrote={wrote}", flush=True)
            print(f"[pool] {dataset}/{arm} wrote={wrote} out={rel(out_path)}", flush=True)


def hit_at(row: dict[str, Any], k: int = 5) -> int:
    gold = {str(idx) for idx in row.get("gold_ids", []) if str(idx)}
    got = {str(idx) for idx in (row.get("retrieved_ids") or [])[:k]}
    return int(bool(gold & got)) if gold else 0


def pair_ri(arm_hits: list[int], raw_hits: list[int]) -> dict[str, Any]:
    help_n = sum(1 for arm, raw in zip(arm_hits, raw_hits) if arm == 1 and raw == 0)
    hurt_n = sum(1 for arm, raw in zip(arm_hits, raw_hits) if arm == 0 and raw == 1)
    n = len(arm_hits)
    return {"help": help_n, "hurt": hurt_n, "ri": (help_n - hurt_n) / n if n else float("nan")}


def ce_scores_for_dataset(dataset: str, labels: list[str], questions: dict[str, tuple[int, Any, str, list[str]]], args: argparse.Namespace) -> dict[str, dict[str, float]]:
    arms = arm_specs(dataset)
    generation: dict[str, dict[str, dict[str, Any]]] = {}
    for key in ["hyde", "scope", "csqe"]:
        if arms[key].generation and arms[key].generation.exists():
            generation[key] = load_by_label(arms[key].generation)
    generation["three_scope_raw"] = load_by_label(generation_path(dataset))
    gold_ids = sorted({gid for label in labels for gid in questions[label][3]})
    gold_docs = fetch_docs_by_idx(COLLECTION[dataset], gold_ids, batch_size=args.doc_batch_size)
    ce = get_cross_encoder()
    scores: dict[str, dict[str, float]] = {}
    raw_items = [(label, questions[label][2], questions[label][3]) for label in labels]
    scores["raw"] = {
        label: score
        for label, (score, _) in score_best_gold_ce(
            ce=ce,
            items=raw_items,
            gold_docs=gold_docs,
            batch_size=args.ce_batch_size,
            chunk_size=args.ce_chunk_size,
            tag=f"{DISPLAY[dataset]}/raw",
        ).items()
    }
    for key in ["hyde", "scope", "csqe"]:
        rows = generation.get(key, {})
        items = [(label, generation_passage(rows[label]), questions[label][3]) for label in labels if label in rows]
        scores[key] = {
            label: score
            for label, (score, _) in score_best_gold_ce(
                ce=ce,
                items=items,
                gold_docs=gold_docs,
                batch_size=args.ce_batch_size,
                chunk_size=args.ce_chunk_size,
                tag=f"{DISPLAY[dataset]}/{key}",
            ).items()
        }
    rows = generation["three_scope_raw"]
    for i in range(3):
        items = []
        for label in labels:
            passages = rows[label].get("scope_passages", [])
            if len(passages) > i:
                items.append((label, str(passages[i]), questions[label][3]))
        scores[f"three_scope_{i + 1}"] = {
            label: score
            for label, (score, _) in score_best_gold_ce(
                ce=ce,
                items=items,
                gold_docs=gold_docs,
                batch_size=args.ce_batch_size,
                chunk_size=args.ce_chunk_size,
                tag=f"{DISPLAY[dataset]}/3scope_{i + 1}",
            ).items()
        }
    return scores


def build_points_for_dataset(dataset: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    questions = load_question_rows(dataset)
    labels = list(questions)
    arms = arm_specs(dataset)
    retrieval = {key: load_by_label(spec.retrieval) for key, spec in arms.items() if spec.retrieval.exists()}
    missing_arms = [key for key in arms if key not in retrieval]
    if missing_arms:
        raise SystemExit(f"{dataset}: missing retrieval caches for {missing_arms}")
    for key, rows in retrieval.items():
        missing = [label for label in labels if label not in rows]
        if missing:
            raise SystemExit(f"{dataset}/{key}: missing {len(missing)} retrieval rows, first={missing[:5]}")
    ce = ce_scores_for_dataset(dataset, labels, questions, args)
    raw_hits = [hit_at(retrieval["raw"][label]) for label in labels]
    points: list[dict[str, Any]] = []
    for label_i, label in enumerate(labels):
        ce_raw = ce["raw"].get(label, float("nan"))
        deltas = [
            ce.get(f"three_scope_{i}", {}).get(label, float("nan")) - ce_raw
            for i in range(1, 4)
        ]
        points.append({
            "dataset": dataset,
            "dataset_display": DISPLAY[dataset],
            "label": label,
            "idx": str(questions[label][1].get("idx", questions[label][0])),
            "gold_ids": questions[label][3],
            "hits": {key: hit_at(retrieval[key][label]) for key in arms},
            "raw_hit": raw_hits[label_i],
            "pool_size": {
                "raw_scope_pool": float(retrieval["raw_scope_pool"][label].get("pool_size", float("nan"))),
                "three_scope_raw": float(retrieval["three_scope_raw"][label].get("pool_size", float("nan"))),
            },
            "ce_rerank_coverage": {
                "raw_scope_pool": float(retrieval["raw_scope_pool"][label].get("ce_rerank_coverage", float("nan"))),
                "three_scope_raw": float(retrieval["three_scope_raw"][label].get("ce_rerank_coverage", float("nan"))),
            },
            "ce_gold": {
                "raw": ce_raw,
                "hyde": ce.get("hyde", {}).get(label, float("nan")),
                "scope": ce.get("scope", {}).get(label, float("nan")),
                "csqe": ce.get("csqe", {}).get(label, float("nan")),
                "three_scope_1": ce.get("three_scope_1", {}).get(label, float("nan")),
                "three_scope_2": ce.get("three_scope_2", {}).get(label, float("nan")),
                "three_scope_3": ce.get("three_scope_3", {}).get(label, float("nan")),
            },
            "ce_delta_vs_raw": {
                "hyde": ce.get("hyde", {}).get(label, float("nan")) - ce_raw,
                "scope": ce.get("scope", {}).get(label, float("nan")) - ce_raw,
                "csqe": ce.get("csqe", {}).get(label, float("nan")) - ce_raw,
                "raw_scope_pool": ce.get("scope", {}).get(label, float("nan")) - ce_raw,
                "three_scope_raw": mean(deltas),
                "three_scope_1": deltas[0],
                "three_scope_2": deltas[1],
                "three_scope_3": deltas[2],
            },
        })
    return points


def summarize(points: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    raw_hits = [int(row["hits"]["raw"]) for row in points]
    hits = [int(row["hits"][arm]) for row in points]
    ri = pair_ri(hits, raw_hits)
    return {
        "n": len(points),
        "hits": sum(hits),
        "hit5": sum(hits) / len(points) if points else float("nan"),
        "ri": ri["ri"],
        "help": ri["help"],
        "hurt": ri["hurt"],
        "ce_delta": mean(row["ce_delta_vs_raw"].get(arm, 0.0 if arm == "raw" else float("nan")) for row in points),
        "pool_size": mean(row["pool_size"].get(arm, float("nan")) for row in points),
        "coverage": mean(row["ce_rerank_coverage"].get(arm, float("nan")) for row in points),
    }


def verdicts(summaries: dict[str, dict[str, dict[str, Any]]]) -> list[tuple[str, str, str]]:
    beir = [summaries[d] for d in BEIR_ORDER if d in summaries]
    legal = {d: summaries[d] for d in LEGAL_ORDER if d in summaries}
    rows: list[tuple[str, str, str]] = []
    if beir:
        raw = mean(s["raw"]["hit5"] for s in beir)
        three = mean(s["three_scope_raw"]["hit5"] for s in beir)
        diff = three - raw
        status = "supported" if diff >= -0.02 else "killed"
        rows.append(("H-strong-noregress", status, f"BEIR macro Hit@5 3SCOPE+raw {pct(three)} vs raw {pct(raw)} (delta {pct(diff)})."))
        csqe = mean(s["csqe"]["hit5"] for s in beir)
        status = "supported" if three >= csqe - 0.02 else ("mixed" if three >= csqe - 0.05 else "killed")
        rows.append(("H-vs-CSQE strong side", status, f"BEIR macro 3SCOPE+raw {pct(three)} vs CSQE {pct(csqe)}."))
        raw_scope = mean(s["raw_scope_pool"]["hit5"] for s in beir)
        status = "supported" if three >= raw_scope - 1e-12 else "killed"
        rows.append(("H-vs-raw∪SCOPE", status, f"BEIR macro 3SCOPE+raw {pct(three)} vs raw∪SCOPE {pct(raw_scope)}."))
        positives = [DISPLAY[d] for d in BEIR_ORDER if d in summaries and summaries[d]["three_scope_raw"]["ri"] > 0]
        status = "supported" if positives else "killed"
        rows.append(("H-net-positive", status, "Positive RI on " + (", ".join(positives) if positives else "no BEIR set") + "."))
    if "barexam" in legal:
        b = legal["barexam"]
        status = "supported" if b["three_scope_raw"]["hit5"] >= b["scope"]["hit5"] else "killed"
        rows.append(("H-weak-help", status, f"BarExam 3SCOPE+raw {pct(b['three_scope_raw']['hit5'])} vs SCOPE {pct(b['scope']['hit5'])}."))
        status = "supported" if b["three_scope_raw"]["hit5"] >= b["csqe"]["hit5"] and b["three_scope_raw"]["hit5"] >= b["scope"]["hit5"] else "mixed"
        rows.append(("H-vs-CSQE weak side", status, f"BarExam 3SCOPE+raw {pct(b['three_scope_raw']['hit5'])} vs CSQE {pct(b['csqe']['hit5'])}."))
    return rows


def load_cached_points(path: Path) -> dict[str, list[dict[str, Any]]]:
    if not path.exists():
        return {}
    out: dict[str, list[dict[str, Any]]] = {}
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            dataset = str(row.get("dataset") or "")
            if dataset:
                out.setdefault(dataset, []).append(row)
    return out


def cached_points_match_dataset(dataset: str, points: list[dict[str, Any]]) -> bool:
    expected = set(load_question_rows(dataset))
    observed = {str(row.get("label") or "") for row in points}
    return bool(expected) and observed == expected and len(points) == len(expected)


def write_report(args: argparse.Namespace) -> None:
    datasets = [
        d for d in selected_datasets(args.phase)
        if generation_path(d).exists() and pool_path(d, "3scope_raw_pool").exists()
    ]
    cached = load_cached_points(args.points_out)
    all_points: list[dict[str, Any]] = []
    for dataset in datasets:
        points = cached.get(dataset, [])
        if cached_points_match_dataset(dataset, points):
            print(f"[report] {dataset}: reuse cached points", flush=True)
            all_points.extend(points)
            continue
        print(f"[report] {dataset}", flush=True)
        all_points.extend(build_points_for_dataset(dataset, args))
    args.points_out.parent.mkdir(parents=True, exist_ok=True)
    with args.points_out.open("w") as f:
        for row in all_points:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    by_dataset: dict[str, list[dict[str, Any]]] = {d: [] for d in datasets}
    for row in all_points:
        by_dataset[str(row["dataset"])].append(row)
    summaries = {
        dataset: {arm: summarize(points, arm) for arm in ["raw", "hyde", "scope", "csqe", "raw_scope_pool", "three_scope_raw"]}
        for dataset, points in by_dataset.items()
    }
    if any(d in summaries for d in BEIR_ORDER):
        beir_points = [row for row in all_points if row["dataset"] in BEIR_ORDER]
        summaries["beir_pooled"] = {
            arm: summarize(beir_points, arm)
            for arm in ["raw", "hyde", "scope", "csqe", "raw_scope_pool", "three_scope_raw"]
        }
    if any(d in summaries for d in LEGAL_ORDER):
        legal_points = [row for row in all_points if row["dataset"] in LEGAL_ORDER]
        summaries["legal_pooled"] = {
            arm: summarize(legal_points, arm)
            for arm in ["raw", "hyde", "scope", "csqe", "raw_scope_pool", "three_scope_raw"]
        }

    lines = [
        "# 3SCOPE + Raw Pool - 2026-05-28",
        "",
        (
            "This report evaluates the 3SCOPE+raw arm: raw query plus three independently "
            "generated exemplar-anchored SCOPE passages, dense top-10 retrieval for each "
            "representation, unique-document pooling, and MiniLM CE reranking to top-5. "
            "No files under `paper/` were edited."
        ),
        "",
        "## Hypothesis Verdicts",
        "",
        "| Hypothesis | Verdict | Key read |",
        "|---|---|---|",
    ]
    for name, status, read in verdicts({k: v for k, v in summaries.items() if k in ALL_ORDER}):
        lines.append(f"| {name} | **{status}** | {read} |")

    lines.extend([
        "",
        "## Regime Table",
        "",
        "| Dataset | Arm | N | Hit@5 | Hits | RI vs raw | Help | Hurt | Mean CE delta vs raw | Avg pool size | CE coverage |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    order = [d for d in ALL_ORDER if d in summaries]
    if "beir_pooled" in summaries:
        order.append("beir_pooled")
    if "legal_pooled" in summaries:
        order.append("legal_pooled")
    display_extra = {"beir_pooled": "BEIR pooled", "legal_pooled": "Legal pooled"}
    for dataset in order:
        for arm in ["raw", "hyde", "scope", "csqe", "raw_scope_pool", "three_scope_raw"]:
            s = summaries[dataset][arm]
            arm_name = {
                "raw": "Raw",
                "hyde": "HyDE",
                "scope": "SCOPE",
                "csqe": "CSQE",
                "raw_scope_pool": "raw∪SCOPE-pool",
                "three_scope_raw": "3SCOPE+raw",
            }[arm]
            lines.append(
                f"| {display_extra.get(dataset, DISPLAY.get(dataset, dataset))} | {arm_name} | {s['n']} | "
                f"{pct(s['hit5'])} | {s['hits']} | {fmt(s['ri'], 3)} | {s['help']} | {s['hurt']} | "
                f"{fmt(s['ce_delta'], 3)} | {fmt(s['pool_size'], 2)} | {pct(s['coverage'])} |"
            )

    lines.extend([
        "",
        "## 3SCOPE Candidate Drift",
        "",
        "| Dataset | N | Mean delta s1 | Mean delta s2 | Mean delta s3 | Mean delta avg |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for dataset in [d for d in ALL_ORDER if d in by_dataset]:
        points = by_dataset[dataset]
        vals = [mean(row["ce_delta_vs_raw"].get(f"three_scope_{i}", float("nan")) for row in points) for i in range(1, 4)]
        lines.append(
            f"| {DISPLAY[dataset]} | {len(points)} | {fmt(vals[0], 3)} | {fmt(vals[1], 3)} | "
            f"{fmt(vals[2], 3)} | {fmt(mean(vals), 3)} |"
        )

    lines.extend([
        "",
        "## Sources",
        "",
        f"- Row-level points: `{rel(args.points_out)}`",
        "- Exemplar source: `caches/exemplars/beir_orthogonal3_exemplars_2026-05-26.json` plus the built-in BarExam/Housing orthogonal signal bank in `eval/eval_harness.py`.",
    ])
    for dataset in datasets:
        lines.append(f"- {DISPLAY[dataset]} 3SCOPE generation: `{rel(generation_path(dataset))}`")
        lines.append(f"- {DISPLAY[dataset]} 3SCOPE+raw pool: `{rel(pool_path(dataset, '3scope_raw_pool'))}`")
        lines.append(f"- {DISPLAY[dataset]} raw∪SCOPE pool: `{rel(pool_path(dataset, 'raw_scope_pool'))}`")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines).rstrip() + "\n")
    print(args.out)
    print(args.points_out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ["generate", "pool", "report"]:
        p_action = sub.add_parser(name)
        p_action.add_argument("--phase", choices=["beir", "legal", "all"], default="all")
        p_action.add_argument("--limit", type=int, default=0)
        p_action.add_argument("--resume", action="store_true")
        p_action.add_argument("--concurrency", type=int, default=int(os.getenv("EVAL_CONCURRENCY", "8") or 8))
        p_action.add_argument("--batch-size", type=int, default=128)
        p_action.add_argument("--max-retries", type=int, default=2)
        p_action.add_argument("--retrieve-k", type=int, default=10)
        p_action.add_argument("--progress-interval", type=int, default=250)
        p_action.add_argument("--doc-batch-size", type=int, default=5000)
        p_action.add_argument("--ce-batch-size", type=int, default=32)
        p_action.add_argument("--ce-chunk-size", type=int, default=4096)
        p_action.add_argument("--out", type=Path, default=ROOT / "docs/generated/3scope_raw_pool_2026-05-28.md")
        p_action.add_argument("--points-out", type=Path, default=ROOT / "docs/generated/3scope_raw_pool_2026-05-28_points.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.action == "generate":
        if args.concurrency <= 0:
            raise SystemExit("--concurrency must be positive")
        generate_phase(args)
    elif args.action == "pool":
        build_pool_phase(args)
    elif args.action == "report":
        write_report(args)


if __name__ == "__main__":
    main()
