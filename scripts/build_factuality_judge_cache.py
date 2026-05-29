#!/usr/bin/env python3
"""LLM-as-judge corpus-supportedness cache for generated passages.

This is a results-lane analysis helper. It reads existing HyDE/SCOPE generation
and retrieval caches, fetches existing corpus passages, and writes one stable
judge record per generated passage and premise kind. It does not create new
generation or retrieval caches.
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
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_config import EvalConfig, load_questions  # noqa: E402
from eval_harness import (  # noqa: E402
    _gold_ids,
    _llm_call,
    _provider_route_metadata,
    _retrieval_question,
    _row_label,
    _setup_provider,
)
from llm_config import get_provider_info  # noqa: E402
from rag_utils import get_documents_by_idx  # noqa: E402


MODEL = "or-gemma4-26b"
DEFAULT_OUT = ROOT / "docs/generated/factuality_judge_q200_2026-05-28.jsonl"
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
TRANSIENT_RE = re.compile(r"(429|rate|timeout|temporar|overload|upstream|connection)", re.I)
BAD_JSON_ESCAPE_RE = re.compile(r'\\(?!["\\/bfnrtu])')


@dataclass(frozen=True)
class ExpansionSpec:
    key: str
    display: str
    generation: Path
    retrieval: Path


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display: str
    collection: str
    raw_cache: Path
    expansions: dict[str, ExpansionSpec]
    housing_state_filter: bool = False


def p(path: str) -> Path:
    return ROOT / path


def dataset_specs() -> dict[str, DatasetSpec]:
    beir: dict[str, tuple[str, str]] = {
        "beir_scifact": ("SciFact", "scifact"),
        "beir_nfcorpus": ("NFCorpus", "nfcorpus"),
        "beir_fiqa": ("FiQA", "fiqa"),
        "beir_trec_covid": ("TREC-COVID", "trec_covid"),
        "beir_scidocs": ("SciDocs", "scidocs"),
    }
    specs: dict[str, DatasetSpec] = {}
    for key, (display, file_key) in beir.items():
        specs[key] = DatasetSpec(
            key=key,
            display=display,
            collection=key,
            raw_cache=p(f"caches/retrieval/full/{key}_qfull_seed42_raw_question_k10.jsonl"),
            expansions={
                "hyde": ExpansionSpec(
                    key="hyde",
                    display="HyDE",
                    generation=p(f"caches/generation/full/{key}_qfull_seed42_{MODEL}_rag_hyde.jsonl"),
                    retrieval=p(f"caches/retrieval/full/{key}_qfull_seed42_{MODEL}_rag_hyde_k10.jsonl"),
                ),
                "scope": ExpansionSpec(
                    key="scope",
                    display="SCOPE",
                    generation=p(f"caches/generation/full/{key}_qfull_seed42_{MODEL}_snap_hyre.jsonl"),
                    retrieval=p(f"caches/retrieval/full/{key}_qfull_seed42_{MODEL}_snap_hyre_k10.jsonl"),
                ),
            },
        )
        _ = file_key
    specs["barexam"] = DatasetSpec(
        key="barexam",
        display="BarExamQA",
        collection="legal_passages",
        raw_cache=p("caches/retrieval/full/barexam_qfull_seed42_raw_question_k10.jsonl"),
        expansions={
            "hyde": ExpansionSpec(
                key="hyde",
                display="HyDE",
                generation=p(f"caches/hyre/full/barexam_qfull_seed42_{MODEL}_rag_hyde.jsonl"),
                retrieval=p(f"caches/retrieval/full/barexam_qfull_seed42_{MODEL}_rag_hyde_k10.jsonl"),
            ),
            "scope": ExpansionSpec(
                key="scope",
                display="SCOPE",
                generation=p(f"caches/hyre/full/barexam_qfull_seed42_{MODEL}_snap_hyre.jsonl"),
                retrieval=p(f"caches/retrieval/full/barexam_qfull_seed42_{MODEL}_snap_hyre_k10.jsonl"),
            ),
        },
    )
    specs["housing"] = DatasetSpec(
        key="housing",
        display="HousingQA state-filtered",
        collection="housing_statutes",
        raw_cache=p("caches/retrieval/full/housing_qfull_seed42_statefilter_raw_question_k10.jsonl"),
        housing_state_filter=True,
        expansions={
            "hyde": ExpansionSpec(
                key="hyde",
                display="HyDE",
                generation=p(f"caches/hyre/full/housing_qfull_seed42_{MODEL}_rag_hyde.jsonl"),
                retrieval=p(f"caches/retrieval/full/housing_qfull_seed42_statefilter_{MODEL}_rag_hyde_k10.jsonl"),
            ),
            "scope": ExpansionSpec(
                key="scope",
                display="SCOPE",
                generation=p(f"caches/hyre/full/housing_qfull_seed42_{MODEL}_snap_hyre.jsonl"),
                retrieval=p(f"caches/retrieval/full/housing_qfull_seed42_statefilter_{MODEL}_snap_hyre_k10.jsonl"),
            ),
        },
    )
    return specs


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def load_by_label(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("label") or row.get("idx")): row for row in read_jsonl(path)}


def stable_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("dataset")),
        str(row.get("label")),
        str(row.get("expansion")),
        str(row.get("premise_kind")),
    )


def load_existing(path: Path) -> set[tuple[str, str, str, str]]:
    if not path.exists():
        return set()
    out: set[tuple[str, str, str, str]] = set()
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                out.add(stable_key(json.loads(line)))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return out


def normalize_text(text: Any) -> str:
    return " ".join(str(text or "").split())


def excerpt(text: str, limit: int) -> str:
    text = normalize_text(text)
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].strip()
    return cut + " ..."


def doc_text(doc: Any) -> str:
    title = normalize_text(getattr(doc, "metadata", {}).get("title", ""))
    text = normalize_text(getattr(doc, "page_content", ""))
    if title and title.lower() not in text[: max(80, len(title) + 20)].lower():
        return f"{title}. {text}"
    return text or title


def fetch_doc_lookup(collection: str, ids: list[str]) -> dict[str, str]:
    unique = list(dict.fromkeys(str(idx) for idx in ids if str(idx)))
    out: dict[str, str] = {}
    for start in range(0, len(unique), 5000):
        chunk = unique[start : start + 5000]
        docs = get_documents_by_idx(collection, chunk)
        for doc in docs:
            meta_idx = str(getattr(doc, "metadata", {}).get("idx") or "")
            if not meta_idx:
                continue
            out[meta_idx] = doc_text(doc)
        print(f"[docs] {collection}: {min(start + len(chunk), len(unique))}/{len(unique)}", flush=True)
    missing = [idx for idx in unique if idx not in out]
    if missing:
        raise RuntimeError(f"{collection}: missing document text for {missing[:8]} n={len(missing)}")
    return out


def load_questions_for(spec: DatasetSpec) -> dict[str, dict[str, Any]]:
    config = EvalConfig(
        dataset=spec.key,
        questions="full",
        seed=42,
        retrieval_k=5,
        housing_state_filter=spec.housing_state_filter,
    )
    out: dict[str, dict[str, Any]] = {}
    for fallback_i, row in load_questions(config).iterrows():
        label = _row_label(row, config, fallback_i=fallback_i)
        out[label] = {
            "label": label,
            "idx": str(row.get("idx", fallback_i)),
            "question": _retrieval_question(row),
            "gold_ids": [str(idx) for idx in _gold_ids(row) if str(idx)],
        }
    return out


def generation_passage(row: dict[str, Any]) -> str:
    return str(row.get("hyde_passage") or row.get("hypothetical_passage") or row.get("passage") or "")


def load_ce_best_gold_ids(paths: list[Path]) -> dict[tuple[str, str, str], str]:
    out: dict[tuple[str, str, str], str] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                dataset = str(row.get("dataset") or "")
                label = str(row.get("label") or "")
                expansion = str(row.get("expansion") or "")
                if expansion:
                    gid = str(row.get("ce_exp_gold_id") or "")
                    if gid:
                        out[(dataset, label, expansion)] = gid
                if row.get("model") == MODEL and row.get("ce_scope_gold_id"):
                    out[(dataset, label, "scope")] = str(row["ce_scope_gold_id"])
    return out


def select_gold_ids(
    *,
    all_gold_ids: list[str],
    ce_best: str,
    max_gold_passages: int,
) -> tuple[list[str], str]:
    if len(all_gold_ids) <= max_gold_passages:
        return all_gold_ids, "all_gold"
    if ce_best:
        return [ce_best], "ce_best_gold_proxy"
    return all_gold_ids[:max_gold_passages], "first_gold_cap"


def build_premise(ids: list[str], docs_by_id: dict[str, str], per_doc_limit: int) -> str:
    parts: list[str] = []
    for rank, idx in enumerate(ids, 1):
        text = docs_by_id.get(str(idx), "")
        if not text:
            continue
        parts.append(f"[Passage {rank}: {idx}]\n{excerpt(text, per_doc_limit)}")
    return "\n\n".join(parts)


def system_prompt() -> str:
    return (
        "You are a strict corpus-supportedness judge. Decide whether the generated "
        "passage is supported by the premise passage set. Judge only factual or "
        "domain claims in the generated passage. Do not reward plausible knowledge "
        "that is absent from the premise. Return JSON only."
    )


def user_prompt(*, premise_kind: str, premise: str, generated_passage: str) -> str:
    return "\n".join([
        "## Premise passage set",
        premise,
        "",
        "## Generated passage to judge",
        generated_passage,
        "",
        "## Verdict labels",
        "entailed: every material claim is directly supported by the premise set.",
        "partially: at least one central claim is supported, but some material claims are unsupported or too broad.",
        "not_entailed: the premise set does not support the material claims, but there is no direct contradiction.",
        "contradicted: at least one material claim conflicts with the premise set.",
        "",
        "## Required JSON",
        '{"verdict":"entailed|partially|not_entailed|contradicted","rationale":"one short sentence","unsupported_claims":["short claim if any"],"premise_kind":"' + premise_kind + '"}',
    ])


def parse_judgment(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    match = JSON_RE.search(text)
    candidate = match.group(0) if match else text
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        # Some otherwise valid judge responses contain prose escapes such as
        # "\S" inside rationale strings. Preserve the raw output in the record,
        # but tolerate those invalid JSON escapes when extracting the verdict.
        data = json.loads(BAD_JSON_ESCAPE_RE.sub(r"\\\\", candidate))
    verdict = str(data.get("verdict") or "").strip().lower().replace("-", "_")
    aliases = {
        "not entailed": "not_entailed",
        "not_supported": "not_entailed",
        "unsupported": "not_entailed",
        "partial": "partially",
        "partly": "partially",
        "contradiction": "contradicted",
    }
    verdict = aliases.get(verdict, verdict)
    if verdict not in {"entailed", "partially", "not_entailed", "contradicted"}:
        raise ValueError(f"invalid verdict {verdict!r}")
    score = {
        "entailed": 1.0,
        "partially": 0.5,
        "not_entailed": 0.0,
        "contradicted": 0.0,
    }[verdict]
    return {
        "verdict": verdict,
        "score": score,
        "rationale": str(data.get("rationale") or "")[:500],
        "unsupported_claims": data.get("unsupported_claims") if isinstance(data.get("unsupported_claims"), list) else [],
    }


def call_with_retries(system: str, user: str, label: str, max_retries: int) -> tuple[str, dict[str, Any], int]:
    last_error = ""
    for attempt in range(max_retries + 1):
        try:
            raw = _llm_call(system, user, label=label)
            return raw, parse_judgment(raw), attempt
        except Exception as exc:
            last_error = str(exc)
            if attempt >= max_retries:
                raise
            sleep_s = min(60.0, 2.0 * (2 ** attempt)) if TRANSIENT_RE.search(last_error) else 1.0
            print(f"[retry] {label} attempt={attempt + 1} sleep={sleep_s:.1f}s error={last_error[:160]}", flush=True)
            time.sleep(sleep_s)
    raise RuntimeError(last_error)


def make_task_records(args: argparse.Namespace, spec: DatasetSpec) -> tuple[list[dict[str, Any]], dict[str, str]]:
    questions = load_questions_for(spec)
    labels = list(questions)
    if args.limit:
        labels = labels[: min(args.limit, len(labels))]

    raw_cache = load_by_label(spec.raw_cache)
    generation_by_exp = {key: load_by_label(exp.generation) for key, exp in spec.expansions.items()}
    retrieval_by_exp = {key: load_by_label(exp.retrieval) for key, exp in spec.expansions.items()}
    ce_best = load_ce_best_gold_ids([
        Path("/tmp/beir_phase1_verification_2026-05-26_points.jsonl"),
        Path("/tmp/beir_phase1b_model_breadth_2026-05-26_points.jsonl"),
        Path("/tmp/affinity_margin_oncache_2026-05-26_points.jsonl"),
    ])
    doc_ids: list[str] = []
    task_rows: list[dict[str, Any]] = []
    for label in labels:
        if label not in raw_cache:
            raise RuntimeError(f"{spec.key}: raw cache missing {label}")
        qrow = questions[label]
        _, gold_strategy = select_gold_ids(
            all_gold_ids=qrow["gold_ids"],
            ce_best="",
            max_gold_passages=args.max_gold_passages,
        )
        raw_top3 = [str(idx) for idx in (raw_cache[label].get("retrieved_ids") or [])[:3]]
        doc_ids.extend(raw_top3)
        for exp_key, exp in spec.expansions.items():
            gen = generation_by_exp[exp_key]
            ret = retrieval_by_exp[exp_key]
            if label not in gen:
                raise RuntimeError(f"{spec.key}/{exp_key}: generation cache missing {label}")
            if label not in ret:
                raise RuntimeError(f"{spec.key}/{exp_key}: retrieval cache missing {label}")
            gold_ids_for_exp, strategy = select_gold_ids(
                all_gold_ids=qrow["gold_ids"],
                ce_best=ce_best.get((spec.key, label, exp_key), ""),
                max_gold_passages=args.max_gold_passages,
            )
            doc_ids.extend(gold_ids_for_exp)
            passage = generation_passage(gen[label])
            if not passage:
                raise RuntimeError(f"{spec.key}/{exp_key}/{label}: missing generated passage")
            base = {
                "dataset": spec.key,
                "dataset_display": spec.display,
                "label": label,
                "idx": qrow["idx"],
                "expansion": exp_key,
                "expansion_display": exp.display,
                "model": MODEL,
                "question": qrow["question"],
                "generated_passage": passage,
                "generated_passage_chars": len(passage),
                "all_gold_count": len(qrow["gold_ids"]),
                "raw_top3_ids": raw_top3,
            }
            task_rows.append({
                **base,
                "premise_kind": "gold",
                "premise_ids": gold_ids_for_exp,
                "gold_strategy": strategy,
            })
            task_rows.append({
                **base,
                "premise_kind": "raw_top3",
                "premise_ids": raw_top3,
                "gold_strategy": gold_strategy,
            })
    docs_by_id = fetch_doc_lookup(spec.collection, doc_ids)
    return task_rows, docs_by_id


def judge_one(order_i: int, task: dict[str, Any], docs_by_id: dict[str, str], args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    premise = build_premise(task["premise_ids"], docs_by_id, args.per_doc_chars)
    if not premise:
        raise RuntimeError(f"{task['dataset']}/{task['label']}/{task['premise_kind']}: empty premise")
    start = time.time()
    provider_info = get_provider_info()
    raw, parsed, retry_count = call_with_retries(
        system_prompt(),
        user_prompt(
            premise_kind=str(task["premise_kind"]),
            premise=premise,
            generated_passage=excerpt(str(task["generated_passage"]), args.generated_chars),
        ),
        label=f"factuality/{task['dataset']}/{task['expansion']}/{task['premise_kind']}",
        max_retries=args.max_retries,
    )
    record = {
        "dataset": task["dataset"],
        "dataset_display": task["dataset_display"],
        "label": task["label"],
        "idx": task["idx"],
        "expansion": task["expansion"],
        "expansion_display": task["expansion_display"],
        "model": task["model"],
        "premise_kind": task["premise_kind"],
        "premise_ids": task["premise_ids"],
        "premise_count": len(task["premise_ids"]),
        "all_gold_count": task["all_gold_count"],
        "gold_strategy": task["gold_strategy"],
        "raw_top3_ids": task["raw_top3_ids"],
        "generated_passage_chars": task["generated_passage_chars"],
        "generated_passage_hash": hashlib.sha256(str(task["generated_passage"]).encode("utf-8")).hexdigest()[:16],
        "verdict": parsed["verdict"],
        "score": parsed["score"],
        "rationale": parsed["rationale"],
        "unsupported_claims": parsed["unsupported_claims"],
        "judge_raw": raw,
        "judge_provider": provider_info.get("provider", args.provider),
        "judge_model": provider_info.get("model", args.provider),
        "provider_route": _provider_route_metadata(),
        "retry_count": retry_count,
        "elapsed_sec": round(time.time() - start, 3),
    }
    return order_i, record


def chunks(values: list[tuple[int, dict[str, Any]]], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def build_dataset(args: argparse.Namespace, spec: DatasetSpec, out_path: Path) -> None:
    task_rows, docs_by_id = make_task_records(args, spec)
    existing = load_existing(out_path) if args.resume else set()
    pending = [
        (i, task)
        for i, task in enumerate(task_rows)
        if stable_key(task) not in existing
    ]
    print(
        f"[dataset] {spec.key}: tasks={len(task_rows)} pending={len(pending)} done={len(existing)} "
        f"workers={args.concurrency}",
        flush=True,
    )
    mode = "a" if args.resume and out_path.exists() else "w"
    with out_path.open(mode) as f:
        wrote = 0
        for batch in chunks(pending, args.batch_size):
            records: dict[int, dict[str, Any]] = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                futures = {
                    executor.submit(judge_one, order_i, task, docs_by_id, args): (order_i, task)
                    for order_i, task in batch
                }
                for future in concurrent.futures.as_completed(futures):
                    order_i, task = futures[future]
                    try:
                        rec_order, record = future.result()
                    except Exception as exc:
                        for pending_future in futures:
                            pending_future.cancel()
                        key = f"{task['dataset']}/{task['label']}/{task['expansion']}/{task['premise_kind']}"
                        raise SystemExit(f"factuality judge failed for {key}: {exc}") from exc
                    records[rec_order] = record
            for order_i in sorted(records):
                f.write(json.dumps(records[order_i], sort_keys=True) + "\n")
                f.flush()
                wrote += 1
                if args.progress_interval and wrote % args.progress_interval == 0:
                    rec = records[order_i]
                    print(
                        f"[judge] wrote={wrote} {rec['dataset']}/{rec['label']} "
                        f"{rec['expansion']}/{rec['premise_kind']} score={rec['score']}",
                        flush=True,
                    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["all"])
    parser.add_argument("--limit", type=int, default=200, help="Rows per dataset; 0 means full.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--provider", default=MODEL)
    parser.add_argument("--concurrency", type=int, default=int(os.getenv("EVAL_CONCURRENCY", "8") or 8))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--max-gold-passages", type=int, default=12)
    parser.add_argument("--per-doc-chars", type=int, default=1600)
    parser.add_argument("--generated-chars", type=int, default=2200)
    parser.add_argument("--progress-interval", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.concurrency <= 0:
        raise SystemExit("--concurrency must be positive")
    if args.limit < 0:
        raise SystemExit("--limit must be >= 0")
    config = EvalConfig(provider=args.provider, dataset="barexam", questions="full", concurrency=args.concurrency)
    _setup_provider(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    specs = dataset_specs()
    selected = list(specs) if args.datasets == ["all"] else args.datasets
    unknown = [key for key in selected if key not in specs]
    if unknown:
        raise SystemExit(f"unknown datasets: {unknown}")
    if args.limit == 0:
        args.limit = None
    for key in selected:
        build_dataset(args, specs[key], args.output)
    print(f"[done] output={args.output}", flush=True)


if __name__ == "__main__":
    main()
