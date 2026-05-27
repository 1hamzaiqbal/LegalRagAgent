#!/usr/bin/env python3
"""Analyze the legal/BEIR CSQE regime sweep from cached retrieval results."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_beir_phase1 import fetch_docs_by_idx, generation_passage, score_best_gold_ce  # noqa: E402
from eval_config import EvalConfig, load_questions  # noqa: E402
from eval_harness import _gold_ids, _retrieval_question, _row_label  # noqa: E402
from rag_utils import get_cross_encoder  # noqa: E402


MODEL = "or-gemma4-26b"


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display: str
    regime: str
    collection: str
    housing_state_filter: bool = False


@dataclass(frozen=True)
class ArmSpec:
    key: str
    display: str
    retrieval_path: Path
    generation_path: Path | None = None


LEGAL_SPECS = [
    DatasetSpec("barexam", "BarExamQA", "weak", "legal_passages"),
    DatasetSpec("housing", "HousingQA state-filtered", "intermediate", "housing_statutes", True),
]

BEIR_DISPLAY = {
    "beir_scifact": "SciFact",
    "beir_nfcorpus": "NFCorpus",
    "beir_fiqa": "FiQA",
    "beir_trec_covid": "TREC-COVID",
    "beir_scidocs": "SciDocs",
}
BEIR_DATASETS = list(BEIR_DISPLAY)


def cache_path(path: str) -> Path:
    return ROOT / path


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


def hit_at(row: dict[str, Any], k: int = 5) -> int:
    gold = {str(idx) for idx in row.get("gold_ids", []) if str(idx)}
    got = {str(idx) for idx in (row.get("retrieved_ids") or [])[:k]}
    return int(bool(gold & got)) if gold else 0


def pair_ri(arm_hits: list[int], raw_hits: list[int]) -> dict[str, Any]:
    help_n = sum(1 for arm, raw in zip(arm_hits, raw_hits) if arm == 1 and raw == 0)
    hurt_n = sum(1 for arm, raw in zip(arm_hits, raw_hits) if arm == 0 and raw == 1)
    n = len(arm_hits)
    return {"help": help_n, "hurt": hurt_n, "ri": (help_n - hurt_n) / n if n else float("nan")}


def legal_arm_specs(dataset: str) -> dict[str, ArmSpec]:
    if dataset == "barexam":
        prefix = "barexam_qfull_seed42"
        return {
            "raw": ArmSpec("raw", "Raw", cache_path(f"caches/retrieval/full/{prefix}_raw_question_k10.jsonl")),
            "hyde": ArmSpec(
                "hyde",
                "HyDE",
                cache_path(f"caches/retrieval/full/{prefix}_{MODEL}_rag_hyde_k10.jsonl"),
                cache_path(f"caches/hyre/full/{prefix}_{MODEL}_rag_hyde.jsonl"),
            ),
            "scope": ArmSpec(
                "scope",
                "SCOPE",
                cache_path(f"caches/retrieval/full/{prefix}_{MODEL}_snap_hyre_k10.jsonl"),
                cache_path(f"caches/hyre/full/{prefix}_{MODEL}_snap_hyre.jsonl"),
            ),
            "csqe": ArmSpec(
                "csqe",
                "CSQE",
                cache_path(f"caches/retrieval/full/{prefix}_{MODEL}_csqe_k10.jsonl"),
                cache_path(f"caches/generation/full/{prefix}_{MODEL}_csqe.jsonl"),
            ),
        }
    prefix = "housing_qfull_seed42_statefilter"
    gen_prefix = "housing_qfull_seed42"
    return {
        "raw": ArmSpec("raw", "Raw", cache_path(f"caches/retrieval/full/{prefix}_raw_question_k10.jsonl")),
        "hyde": ArmSpec(
            "hyde",
            "HyDE",
            cache_path(f"caches/retrieval/full/{prefix}_{MODEL}_rag_hyde_k10.jsonl"),
            cache_path(f"caches/hyre/full/{gen_prefix}_{MODEL}_rag_hyde.jsonl"),
        ),
        "scope": ArmSpec(
            "scope",
            "SCOPE",
            cache_path(f"caches/retrieval/full/{prefix}_{MODEL}_snap_hyre_k10.jsonl"),
            cache_path(f"caches/hyre/full/{gen_prefix}_{MODEL}_snap_hyre.jsonl"),
        ),
        "csqe": ArmSpec(
            "csqe",
            "CSQE",
            cache_path(f"caches/retrieval/full/{prefix}_{MODEL}_csqe_k10.jsonl"),
            cache_path(f"caches/generation/full/{prefix}_{MODEL}_csqe.jsonl"),
        ),
    }


def beir_arm_specs(dataset: str) -> dict[str, ArmSpec]:
    return {
        "raw": ArmSpec("raw", "Raw", cache_path(f"caches/retrieval/full/{dataset}_qfull_seed42_raw_question_k10.jsonl")),
        "hyde": ArmSpec(
            "hyde",
            "HyDE",
            cache_path(f"caches/retrieval/full/{dataset}_qfull_seed42_{MODEL}_rag_hyde_k10.jsonl"),
            cache_path(f"caches/generation/full/{dataset}_qfull_seed42_{MODEL}_rag_hyde.jsonl"),
        ),
        "scope": ArmSpec(
            "scope",
            "SCOPE",
            cache_path(f"caches/retrieval/full/{dataset}_qfull_seed42_{MODEL}_snap_hyre_k10.jsonl"),
            cache_path(f"caches/generation/full/{dataset}_qfull_seed42_{MODEL}_snap_hyre.jsonl"),
        ),
        "csqe": ArmSpec(
            "csqe",
            "CSQE",
            cache_path(f"caches/retrieval/full/{dataset}_qfull_seed42_csqe_k10.jsonl"),
            cache_path(f"caches/generation/full/{dataset}_qfull_seed42_csqe.jsonl"),
        ),
    }


def load_legal_questions(spec: DatasetSpec) -> dict[str, dict[str, Any]]:
    config = EvalConfig(
        dataset=spec.key,
        questions="full",
        seed=42,
        housing_state_filter=spec.housing_state_filter,
    )
    out: dict[str, dict[str, Any]] = {}
    for fallback_i, row in load_questions(config).iterrows():
        label = _row_label(row, config, fallback_i=fallback_i)
        out[label] = {
            "idx": str(row.get("idx", "")),
            "question": _retrieval_question(row),
            "gold_ids": [str(idx) for idx in _gold_ids(row) if str(idx)],
        }
    return out


def build_legal_points(spec: DatasetSpec, args: argparse.Namespace) -> list[dict[str, Any]]:
    questions = load_legal_questions(spec)
    arms = legal_arm_specs(spec.key)
    retrieval = {key: load_by_label(arm.retrieval_path) for key, arm in arms.items()}
    generation = {
        key: load_by_label(arm.generation_path)
        for key, arm in arms.items()
        if arm.generation_path is not None
    }
    labels = list(questions)
    for key, rows in retrieval.items():
        missing = [label for label in labels if label not in rows]
        if missing:
            raise RuntimeError(f"{spec.key}/{key}: retrieval cache missing {len(missing)} labels, first={missing[:5]}")
    for key in ["hyde", "scope", "csqe"]:
        missing = [label for label in labels if label not in generation[key]]
        if missing:
            raise RuntimeError(f"{spec.key}/{key}: generation cache missing {len(missing)} labels, first={missing[:5]}")

    gold_ids = sorted({gid for row in questions.values() for gid in row["gold_ids"]})
    gold_docs = fetch_docs_by_idx(spec.collection, gold_ids, batch_size=args.doc_batch_size)
    ce = get_cross_encoder()
    ce_scores: dict[str, dict[str, float]] = {}
    for key, arm in arms.items():
        if key == "raw":
            items = [(label, questions[label]["question"], questions[label]["gold_ids"]) for label in labels]
        else:
            items = [
                (label, generation_passage(generation[key][label]), questions[label]["gold_ids"])
                for label in labels
            ]
        scored = score_best_gold_ce(
            ce=ce,
            items=items,
            gold_docs=gold_docs,
            batch_size=args.ce_batch_size,
            chunk_size=args.ce_chunk_size,
            tag=f"{spec.display}/{arm.display}",
        )
        ce_scores[key] = {label: score for label, (score, _) in scored.items()}

    points: list[dict[str, Any]] = []
    for label in labels:
        points.append({
            "dataset": spec.key,
            "dataset_display": spec.display,
            "regime": spec.regime,
            "label": label,
            "idx": questions[label]["idx"],
            "hits": {key: hit_at(retrieval[key][label], 5) for key in arms},
            "ce": {key: ce_scores[key].get(label, float("nan")) for key in arms},
        })
    return points


def load_beir_reference_points(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        raise SystemExit(
            f"BEIR points file not found: {path}. Re-run scripts/analyze_exemplar_scope_select.py "
            "or pass --skip-beir-reference."
        )
    for row in read_jsonl(path):
        dataset = str(row.get("dataset") or "")
        if dataset not in BEIR_DISPLAY:
            continue
        hits = row.get("hits") or {}
        ce = row.get("ce") or {}
        rows.append({
            "dataset": dataset,
            "dataset_display": BEIR_DISPLAY[dataset],
            "regime": "strong",
            "label": str(row.get("label") or ""),
            "idx": str(row.get("idx") or ""),
            "hits": {key: int(hits.get(key, 0)) for key in ["raw", "hyde", "scope", "csqe"]},
            "ce": {key: float(ce.get(key, float("nan"))) for key in ["raw", "hyde", "scope", "csqe"]},
        })
    if not rows:
        raise SystemExit(f"No BEIR reference rows loaded from {path}")
    return rows


def write_points(path: Path, points: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for point in points:
            f.write(json.dumps(point, sort_keys=True) + "\n")


def summarize(points: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    hits = [int(p["hits"][arm]) for p in points]
    raw_hits = [int(p["hits"]["raw"]) for p in points]
    raw_ce = [float(p["ce"]["raw"]) for p in points]
    arm_ce = [float(p["ce"][arm]) for p in points]
    ri = pair_ri(hits, raw_hits)
    return {
        "n": len(points),
        "hit5": mean(hits),
        "correct": sum(hits),
        "ri": 0.0 if arm == "raw" else ri["ri"],
        "help": 0 if arm == "raw" else ri["help"],
        "hurt": 0 if arm == "raw" else ri["hurt"],
        "ce_delta": 0.0 if arm == "raw" else mean(a - r for a, r in zip(arm_ce, raw_ce)),
    }


def best_arm(points: list[dict[str, Any]]) -> str:
    vals = {arm: summarize(points, arm)["hit5"] for arm in ["raw", "hyde", "scope", "csqe"]}
    return max(vals, key=lambda key: vals[key])


def verdict_rows(points: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    by_dataset = {name: [p for p in points if p["dataset"] == name] for name in ["barexam", "housing"]}
    beir = [p for p in points if str(p["dataset"]).startswith("beir_")]
    barexam = by_dataset["barexam"]
    housing = by_dataset["housing"]
    b = {arm: summarize(barexam, arm) for arm in ["raw", "hyde", "scope", "csqe"]}
    h = {arm: summarize(housing, arm) for arm in ["raw", "hyde", "scope", "csqe"]}
    strong = {arm: summarize(beir, arm) for arm in ["raw", "hyde", "scope", "csqe"]} if beir else {}

    h_collapse = "supported" if b["csqe"]["hit5"] <= b["raw"]["hit5"] + 0.02 and b["csqe"]["hit5"] < max(b["hyde"]["hit5"], b["scope"]["hit5"]) - 0.05 else "killed"
    h_scope = "supported" if max(b["hyde"]["hit5"], b["scope"]["hit5"]) >= b["csqe"]["hit5"] + 0.05 else "killed"
    h_housing = "supported" if h["csqe"]["hit5"] >= h["scope"]["hit5"] else "killed"
    if strong:
        weak_best = best_arm(barexam)
        strong_best = best_arm(beir)
        strong_expansion_best = max(["hyde", "scope", "csqe"], key=lambda key: strong[key]["hit5"])
        crossover = "supported" if strong_expansion_best == "csqe" and weak_best in {"hyde", "scope"} and b["csqe"]["hit5"] < max(b["hyde"]["hit5"], b["scope"]["hit5"]) else "mixed"
        if strong_best == "raw" and crossover == "supported":
            crossover = "mixed"
        strong_note = (
            f"BEIR pooled all-arm best={strong_best}; expansion-arm best={strong_expansion_best}. "
            f"Raw {pct(strong['raw']['hit5'])}, CSQE {pct(strong['csqe']['hit5'])}, "
            f"SCOPE {pct(strong['scope']['hit5'])}, HyDE {pct(strong['hyde']['hit5'])}."
        )
    else:
        crossover = "not tested"
        strong_note = "BEIR reference skipped."

    return [
        (
            "H-collapse",
            h_collapse,
            f"BarExam CSQE {pct(b['csqe']['hit5'])} vs Raw {pct(b['raw']['hit5'])}; HyDE {pct(b['hyde']['hit5'])}, SCOPE {pct(b['scope']['hit5'])}.",
        ),
        (
            "H-scope-wins-weak",
            h_scope,
            f"On BarExam, best parametric expansion is {pct(max(b['hyde']['hit5'], b['scope']['hit5']))} vs CSQE {pct(b['csqe']['hit5'])}.",
        ),
        (
            "H-csqe-strong",
            h_housing,
            f"Housing retrieval-only: CSQE {pct(h['csqe']['hit5'])} vs SCOPE {pct(h['scope']['hit5'])}, Raw {pct(h['raw']['hit5'])}.",
        ),
        (
            "Net crossover",
            crossover,
            strong_note + " BarExam remains the single weak-query legal set here, so treat the weak-end read as provisional.",
        ),
    ]


def source_paths(include_beir: bool = True) -> list[str]:
    paths: list[str] = []
    for spec in LEGAL_SPECS:
        for arm in legal_arm_specs(spec.key).values():
            for path in (arm.retrieval_path, arm.generation_path):
                if path is not None:
                    paths.append(str(path.relative_to(ROOT)))
    if include_beir:
        paths.append("docs/generated/exemplar_scope_select_2026-05-26.md")
        for dataset in BEIR_DATASETS:
            for arm in beir_arm_specs(dataset).values():
                for path in (arm.retrieval_path, arm.generation_path):
                    if path is not None:
                        paths.append(str(path.relative_to(ROOT)))
    return list(dict.fromkeys(paths))


def write_report(path: Path, points: list[dict[str, Any]], *, include_beir: bool, beir_points_path: Path) -> None:
    arms = ["raw", "hyde", "scope", "csqe"]
    arm_names = {"raw": "Raw", "hyde": "HyDE", "scope": "SCOPE", "csqe": "CSQE"}
    groups: list[tuple[str, str, list[dict[str, Any]]]] = [
        ("BarExamQA", "weak", [p for p in points if p["dataset"] == "barexam"]),
        ("HousingQA state-filtered", "intermediate", [p for p in points if p["dataset"] == "housing"]),
    ]
    if include_beir:
        for dataset, display in BEIR_DISPLAY.items():
            groups.append((display, "strong", [p for p in points if p["dataset"] == dataset]))
        groups.append(("BEIR pooled reference", "strong", [p for p in points if str(p["dataset"]).startswith("beir_")]))

    lines: list[str] = []
    lines.append("# CSQE Regime Sweep - 2026-05-26")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(
        "This is a read-from-cache regime sweep for corpus-steered query expansion (CSQE). "
        "The only new arm is CSQE on BarExamQA and HousingQA state-filtered; raw, HyDE, "
        "and SCOPE use the existing signed retrieval caches. The BEIR rows are the strong-query "
        "Phase-A reference from `docs/generated/exemplar_scope_select_2026-05-26.md`."
    )
    lines.append("")
    lines.append("HousingQA is interpreted as retrieval-only here: prior answer results showed answer conversion is the binding issue, so this table should not be read as downstream answer accuracy.")
    lines.append("")
    lines.append("## Verdicts")
    lines.append("")
    lines.append("| Hypothesis | Verdict | Key read |")
    lines.append("|---|---|---|")
    for hyp, verdict, note in verdict_rows(points):
        lines.append(f"| {hyp} | **{verdict}** | {note} |")
    lines.append("")
    lines.append("## Regime Sweep Table")
    lines.append("")
    lines.append("| Dataset | Regime | Arm | N | Hit@5 | Correct | RI vs raw | Help | Hurt | Mean CE gold-affinity delta vs raw |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, regime, gpoints in groups:
        for arm in arms:
            s = summarize(gpoints, arm)
            lines.append(
                f"| {name} | {regime} | {arm_names[arm]} | {s['n']} | {pct(s['hit5'])} | "
                f"{s['correct']} | {fmt(s['ri'])} | {s['help']} | {s['hurt']} | "
                f"{'--' if arm == 'raw' else fmt(s['ce_delta'])} |"
            )
    lines.append("")
    lines.append("## Reading")
    lines.append("")
    lines.append(
        "The weak-query BarExamQA control is the sharpest legal check: raw retrieval almost never exposes gold evidence, "
        "so CSQE has little useful real text to extract from the raw top-k. A CSQE gain there would have killed the collapse hypothesis."
    )
    lines.append("")
    lines.append(
        "HousingQA sits between regimes. CSQE can reuse top-ranked state-filtered statutory language, but it is still not a downstream answer claim. "
        "Treat it as evidence about retrieval exposure only."
    )
    if include_beir:
        lines.append("")
        lines.append(
            "The BEIR reference is stronger-query retrieval: raw is already competitive, and CSQE is the expansion-style arm that preserves most of that raw strength. "
            "That makes the aggregate crossover a retrieval-regime pattern rather than a universal CSQE win."
        )
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Per-row legal plus BEIR summary points: `{path.with_name(path.stem + '_points.jsonl').relative_to(ROOT)}`")
    if include_beir:
        lines.append(f"- BEIR point source used for this run: `{beir_points_path}`")
    for source in source_paths(include_beir=include_beir):
        lines.append(f"- `{source}`")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "docs/generated/csqe_regime_sweep_2026-05-26.md")
    parser.add_argument("--points-out", type=Path, default=None)
    parser.add_argument("--beir-points", type=Path, default=Path("/tmp/exemplar_scope_select_2026-05-26_points.jsonl"))
    parser.add_argument("--skip-beir-reference", action="store_true")
    parser.add_argument("--ce-batch-size", type=int, default=64)
    parser.add_argument("--ce-chunk-size", type=int, default=5000)
    parser.add_argument("--doc-batch-size", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    points: list[dict[str, Any]] = []
    for spec in LEGAL_SPECS:
        points.extend(build_legal_points(spec, args))
    include_beir = not args.skip_beir_reference
    if include_beir:
        points.extend(load_beir_reference_points(args.beir_points))
    points_out = args.points_out or args.out.with_name(args.out.stem + "_points.jsonl")
    write_points(points_out, points)
    write_report(args.out, points, include_beir=include_beir, beir_points_path=args.beir_points)
    print(f"wrote {args.out}")
    print(f"wrote {points_out}")


if __name__ == "__main__":
    main()
