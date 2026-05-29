#!/usr/bin/env python3
"""Build the CaseHOLD raw+SCOPE pool and regime-gradient report.

This is read-only over existing retrieval/generation caches: it does not call
any LLM. The only new artifact is a deterministic CE-reranked retrieval pool.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_beir_phase1 import fetch_docs_by_idx, truncate_for_ce  # noqa: E402
from eval_config import EvalConfig, load_questions  # noqa: E402
from eval_harness import _gold_ids, _retrieval_question, _row_label  # noqa: E402
from rag_utils import get_cross_encoder  # noqa: E402


RAW_CACHE = ROOT / "caches/retrieval/full/casehold_qfull_seed42_raw_question_k10.jsonl"
SCOPE_CACHE = ROOT / "caches/retrieval/full/casehold_qfull_seed42_groq-llama70b_snap_hyre_k10.jsonl"
POOL_CACHE = ROOT / "caches/retrieval/full/casehold_qfull_seed42_groq-llama70b_raw_scope_pool_k5.jsonl"
PRIOR_POINTS = ROOT / "docs/generated/3scope_raw_pool_2026-05-28_points.jsonl"
REPORT = ROOT / "docs/generated/casehold_pool_test_2026-05-28.md"
POINTS = ROOT / "docs/generated/casehold_pool_test_2026-05-28_points.jsonl"
COLLECTION = "casehold_holdings"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def load_by_label(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("label") or row.get("idx")): row for row in read_jsonl(path)}


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def pct(value: Any) -> str:
    if not finite(value):
        return "--"
    return f"{100.0 * float(value):.1f}%"


def fmt(value: Any, digits: int = 3) -> str:
    if not finite(value):
        return "--"
    return f"{float(value):.{digits}f}"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_casehold_questions() -> dict[str, dict[str, Any]]:
    config = EvalConfig(dataset="casehold", questions="full", seed=42)
    out: dict[str, dict[str, Any]] = {}
    for fallback_i, row in load_questions(config).iterrows():
        label = _row_label(row, config, fallback_i=fallback_i)
        out[label] = {
            "idx": str(row.get("idx", fallback_i)),
            "question": _retrieval_question(row),
            "gold_ids": [str(idx) for idx in _gold_ids(row) if str(idx)],
        }
    return out


def hit_ids(retrieved_ids: Iterable[str], gold_ids: Iterable[str], k: int = 5) -> int:
    gold = {str(idx) for idx in gold_ids if str(idx)}
    got = {str(idx) for idx in list(retrieved_ids)[:k] if str(idx)}
    return int(bool(gold & got)) if gold else 0


def hit_row(row: dict[str, Any], k: int = 5) -> int:
    return hit_ids(row.get("retrieved_ids") or [], row.get("gold_ids") or [], k=k)


def pair_ri(arm_hits: list[int], baseline_hits: list[int]) -> dict[str, Any]:
    help_n = sum(1 for arm, base in zip(arm_hits, baseline_hits) if arm == 1 and base == 0)
    hurt_n = sum(1 for arm, base in zip(arm_hits, baseline_hits) if arm == 0 and base == 1)
    n = len(arm_hits)
    return {"help": help_n, "hurt": hurt_n, "ri": (help_n - hurt_n) / n if n else float("nan")}


def summarize_hits(raw_hits: list[int], scope_hits: list[int], pool_hits: list[int]) -> dict[str, Any]:
    n = len(pool_hits)
    raw_ri = pair_ri(pool_hits, raw_hits)
    scope_ri = pair_ri(pool_hits, scope_hits)
    return {
        "n": n,
        "raw_hit5": sum(raw_hits) / n if n else float("nan"),
        "scope_hit5": sum(scope_hits) / n if n else float("nan"),
        "pool_hit5": sum(pool_hits) / n if n else float("nan"),
        "pool_hits": sum(pool_hits),
        "ri_vs_raw": raw_ri["ri"],
        "help_vs_raw": raw_ri["help"],
        "hurt_vs_raw": raw_ri["hurt"],
        "ri_vs_scope": scope_ri["ri"],
        "help_vs_scope": scope_ri["help"],
        "hurt_vs_scope": scope_ri["hurt"],
    }


def validate_inputs(labels: list[str], raw: dict[str, dict[str, Any]], scope: dict[str, dict[str, Any]]) -> None:
    for name, rows in [("raw", raw), ("scope", scope)]:
        missing = [label for label in labels if label not in rows]
        extra = sorted(set(rows) - set(labels))
        if missing:
            raise SystemExit(f"{name}: missing {len(missing)} labels, first={missing[:5]}")
        if extra:
            raise SystemExit(f"{name}: {len(extra)} extra labels, first={extra[:5]}")


def unique_union(*id_lists: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for ids in id_lists:
        for idx in ids:
            idx = str(idx)
            if idx and idx not in seen:
                seen.add(idx)
                out.append(idx)
    return out


def build_casehold_pool(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    questions = load_casehold_questions()
    labels = list(questions)
    raw = load_by_label(RAW_CACHE)
    scope = load_by_label(SCOPE_CACHE)
    validate_inputs(labels, raw, scope)

    pool_ids_by_label = {
        label: unique_union(raw[label].get("retrieved_ids") or [], scope[label].get("retrieved_ids") or [])
        for label in labels
    }
    all_doc_ids = sorted({idx for ids in pool_ids_by_label.values() for idx in ids})
    doc_text = fetch_docs_by_idx(COLLECTION, all_doc_ids, batch_size=args.doc_batch_size)

    ce = get_cross_encoder()
    pairs: list[tuple[str, str]] = []
    meta: list[tuple[str, str]] = []
    for label in labels:
        question = truncate_for_ce(questions[label]["question"])
        for idx in pool_ids_by_label[label]:
            pairs.append((question, truncate_for_ce(doc_text[idx])))
            meta.append((label, idx))

    scores_by_label: dict[str, dict[str, float]] = {label: {} for label in labels}
    total = len(pairs)
    for start in range(0, total, args.ce_chunk_size):
        end = min(start + args.ce_chunk_size, total)
        print(f"[ce] CaseHOLD raw+SCOPE pool: {end}/{total}", flush=True)
        scores = ce.predict(pairs[start:end], batch_size=args.ce_batch_size, show_progress_bar=False)
        for (label, idx), score in zip(meta[start:end], scores):
            scores_by_label[label][idx] = float(score)

    rows: list[dict[str, Any]] = []
    points: list[dict[str, Any]] = []
    raw_hits: list[int] = []
    scope_hits: list[int] = []
    pool_hits: list[int] = []
    for label in labels:
        ranked = sorted(
            pool_ids_by_label[label],
            key=lambda idx: scores_by_label[label].get(idx, float("-inf")),
            reverse=True,
        )
        top_ids = ranked[:5]
        top_scores = [scores_by_label[label][idx] for idx in top_ids]
        gold_ids = questions[label]["gold_ids"]
        raw_hit = hit_row(raw[label], 5)
        scope_hit = hit_row(scope[label], 5)
        pool_hit = hit_ids(top_ids, gold_ids, 5)
        raw_hits.append(raw_hit)
        scope_hits.append(scope_hit)
        pool_hits.append(pool_hit)
        rows.append({
            "label": label,
            "idx": questions[label]["idx"],
            "dataset": "casehold",
            "query_type": "raw_scope_pool",
            "label_prefix": "raw_scope_pool",
            "provider": "groq-llama70b",
            "collection": COLLECTION,
            "max_k": 5,
            "component_top_k": 10,
            "component_count": 2,
            "pool_size": len(pool_ids_by_label[label]),
            "component_retrieved_ids": [
                [str(idx) for idx in (raw[label].get("retrieved_ids") or [])[:10]],
                [str(idx) for idx in (scope[label].get("retrieved_ids") or [])[:10]],
            ],
            "retrieved_ids": top_ids,
            "scores": top_scores,
            "gold_ids": gold_ids,
            "gold_retrieved": bool(pool_hit),
            "where": {},
            "ce_rerank_coverage": len(top_ids) / min(5, len(pool_ids_by_label[label])) if pool_ids_by_label[label] else 0.0,
            "question_hash": raw[label].get("question_hash") or scope[label].get("question_hash"),
        })
        points.append({
            "dataset": "casehold",
            "label": label,
            "idx": questions[label]["idx"],
            "gold_ids": gold_ids,
            "hits": {"raw": raw_hit, "scope": scope_hit, "raw_scope_pool": pool_hit},
            "pool_size": len(pool_ids_by_label[label]),
        })

    summary = summarize_hits(raw_hits, scope_hits, pool_hits)
    POOL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with POOL_CACHE.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    POINTS.parent.mkdir(parents=True, exist_ok=True)
    with POINTS.open("w") as f:
        for row in points:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    return points, summary


def summarize_prior(points: list[dict[str, Any]]) -> dict[str, Any]:
    raw_hits = [int(row["hits"]["raw"]) for row in points]
    scope_hits = [int(row["hits"]["scope"]) for row in points]
    pool_hits = [int(row["hits"]["raw_scope_pool"]) for row in points]
    return summarize_hits(raw_hits, scope_hits, pool_hits)


def load_prior_summaries() -> dict[str, dict[str, Any]]:
    if not PRIOR_POINTS.exists():
        raise SystemExit(f"missing prior points file: {PRIOR_POINTS}")
    grouped: dict[str, list[dict[str, Any]]] = {
        "BarExamQA": [],
        "HousingQA state-filtered": [],
        "BEIR pooled": [],
    }
    for row in read_jsonl(PRIOR_POINTS):
        dataset = str(row.get("dataset") or "")
        if dataset == "barexam":
            grouped["BarExamQA"].append(row)
        elif dataset == "housing":
            grouped["HousingQA state-filtered"].append(row)
        elif dataset.startswith("beir_"):
            grouped["BEIR pooled"].append(row)
    missing = [name for name, rows in grouped.items() if not rows]
    if missing:
        raise SystemExit(f"prior points missing groups: {missing}")
    return {name: summarize_prior(rows) for name, rows in grouped.items()}


def verdict(casehold: dict[str, Any]) -> tuple[str, str, str]:
    pool = casehold["pool_hit5"]
    scope = casehold["scope_hit5"]
    if pool < 0.30:
        return (
            "killed",
            "CaseHOLD raw+SCOPE pool collapses below 30% Hit@5, so pooling fails even in the intermediate-weak regime.",
            "binary",
        )
    if pool >= scope - 0.05:
        return (
            "supported",
            f"CaseHOLD raw+SCOPE pool is within 5pp of SCOPE ({pct(pool)} vs {pct(scope)}), so it preserves the intermediate-weak lift.",
            "gradient",
        )
    return (
        "mixed",
        f"CaseHOLD raw+SCOPE pool stays above the collapse floor but loses more than 5pp to SCOPE ({pct(pool)} vs {pct(scope)}).",
        "gradient with a penalty",
    )


def write_report(casehold_summary: dict[str, Any], prior: dict[str, dict[str, Any]]) -> None:
    status, read, regime = verdict(casehold_summary)
    rows = [
        ("BarExamQA", prior["BarExamQA"], "Gemma-26B SCOPE/pool"),
        ("CaseHOLD", casehold_summary, "Llama-70B SCOPE; no new generation"),
        ("HousingQA state-filtered", prior["HousingQA state-filtered"], "Gemma-26B SCOPE/pool"),
        ("BEIR pooled", prior["BEIR pooled"], "Gemma-26B SCOPE/pool"),
    ]

    lines = [
        "# CaseHOLD raw+SCOPE Pool Test - 2026-05-28",
        "",
        (
            "This test pools existing CaseHOLD raw-question top-10 and Llama-70B SCOPE top-10 "
            "retrieval caches, deduplicates by document id, and reranks the union with "
            "`cross-encoder/ms-marco-MiniLM-L-6-v2` to top-5. No LLM generation or answer "
            "calls were run, and no files under `paper/` were edited."
        ),
        "",
        "## Verdict",
        "",
        f"- **H-pool-intermediate-weak: {status}.** {read}",
        (
            f"- Regime read: **{regime}**. BarExam remains the extreme-weak failure point, "
            "CaseHOLD tests the intermediate-weak band, and Housing/BEIR show pool gains once "
            "raw retrieval has enough useful candidates for CE reranking."
        ),
        "",
        "## Regime Gradient",
        "",
        "| Regime | N | Raw Hit@5 | SCOPE Hit@5 | raw+SCOPE pool Hit@5 | Pool hits | RI vs raw | Help/Hurt vs raw | RI vs SCOPE | Help/Hurt vs SCOPE | Note |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, summary, note in rows:
        lines.append(
            f"| {name} | {summary['n']} | {pct(summary['raw_hit5'])} | "
            f"{pct(summary['scope_hit5'])} | {pct(summary['pool_hit5'])} | "
            f"{summary['pool_hits']} | {fmt(summary['ri_vs_raw'], 3)} | "
            f"{summary['help_vs_raw']}/{summary['hurt_vs_raw']} | "
            f"{fmt(summary['ri_vs_scope'], 3)} | "
            f"{summary['help_vs_scope']}/{summary['hurt_vs_scope']} | {note} |"
        )

    lines.extend([
        "",
        "## Reading",
        "",
        (
            f"- CaseHOLD raw retrieval is {pct(casehold_summary['raw_hit5'])}, far above "
            "BarExam's 1.4% but far below Housing and BEIR. This is the intended "
            "intermediate-weak point."
        ),
        (
            f"- The CaseHOLD pool reaches {pct(casehold_summary['pool_hit5'])} Hit@5. "
            f"It helps {casehold_summary['help_vs_raw']} rows over raw and hurts "
            f"{casehold_summary['hurt_vs_raw']} raw-hit rows, for RI={fmt(casehold_summary['ri_vs_raw'], 3)}."
        ),
        (
            f"- Relative to SCOPE, the pool helps {casehold_summary['help_vs_scope']} rows "
            f"and hurts {casehold_summary['hurt_vs_scope']}, giving RI="
            f"{fmt(casehold_summary['ri_vs_scope'], 3)}. This is the direct preservation check."
        ),
        (
            "- Generator mismatch caveat: CaseHOLD SCOPE is the existing `groq-llama70b` cache, "
            "while the BarExam, Housing, and BEIR pool rows in the comparison table use "
            "`or-gemma4-26b`. The pooling/reranking mechanism is generator-agnostic, but this "
            "is not a strict generator-controlled comparison."
        ),
        "",
        "## Sources",
        "",
        f"- CaseHOLD raw cache: `{rel(RAW_CACHE)}`",
        f"- CaseHOLD SCOPE cache: `{rel(SCOPE_CACHE)}`",
        f"- CaseHOLD pool cache: `{rel(POOL_CACHE)}`",
        f"- CaseHOLD row-level points: `{rel(POINTS)}`",
        f"- Prior BarExam/Housing/BEIR points: `{rel(PRIOR_POINTS)}`",
    ])
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines).rstrip() + "\n")
    print(REPORT)
    print(POINTS)
    print(POOL_CACHE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc-batch-size", type=int, default=5000)
    parser.add_argument("--ce-batch-size", type=int, default=32)
    parser.add_argument("--ce-chunk-size", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, casehold_summary = build_casehold_pool(args)
    prior = load_prior_summaries()
    write_report(casehold_summary, prior)


if __name__ == "__main__":
    main()
