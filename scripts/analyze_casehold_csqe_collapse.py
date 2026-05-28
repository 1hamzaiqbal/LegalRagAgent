#!/usr/bin/env python3
"""Analyze CaseHOLD CSQE collapse against existing raw/HyDE/SCOPE caches."""

from __future__ import annotations

import argparse
import json
import math
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


@dataclass(frozen=True)
class ArmSpec:
    key: str
    display: str
    retrieval: Path
    generation: Path | None
    provider_note: str


def p(path: str) -> Path:
    return ROOT / path


ARMS: dict[str, ArmSpec] = {
    "raw": ArmSpec(
        "raw",
        "Raw question",
        p("caches/retrieval/full/casehold_qfull_seed42_raw_question_k10.jsonl"),
        None,
        "raw question",
    ),
    "hyde": ArmSpec(
        "hyde",
        "Llama-70B HyDE",
        p("caches/retrieval/full/casehold_qfull_seed42_groq-llama70b_rag_hyde_k10.jsonl"),
        p("caches/hyre/full/casehold_qfull_seed42_groq-llama70b_rag_hyde.jsonl"),
        "groq-llama70b",
    ),
    "scope": ArmSpec(
        "scope",
        "Llama-70B SCOPE",
        p("caches/retrieval/full/casehold_qfull_seed42_groq-llama70b_snap_hyre_k10.jsonl"),
        p("caches/hyre/full/casehold_qfull_seed42_groq-llama70b_snap_hyre.jsonl"),
        "groq-llama70b",
    ),
    "csqe": ArmSpec(
        "csqe",
        "Gemma-26B CSQE",
        p("caches/retrieval/full/casehold_qfull_seed42_or-gemma4-26b_csqe_k10.jsonl"),
        p("caches/generation/full/casehold_qfull_seed42_or-gemma4-26b_csqe.jsonl"),
        "or-gemma4-26b",
    ),
}


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


def hit_at(row: dict[str, Any], k: int = 5) -> int:
    gold = {str(idx) for idx in row.get("gold_ids", []) if str(idx)}
    got = {str(idx) for idx in (row.get("retrieved_ids") or [])[:k]}
    return int(bool(gold & got)) if gold else 0


def pair_ri(arm_hits: list[int], raw_hits: list[int]) -> dict[str, Any]:
    help_n = sum(1 for arm, raw in zip(arm_hits, raw_hits) if arm == 1 and raw == 0)
    hurt_n = sum(1 for arm, raw in zip(arm_hits, raw_hits) if arm == 0 and raw == 1)
    n = len(arm_hits)
    return {"help": help_n, "hurt": hurt_n, "ri": (help_n - hurt_n) / n if n else float("nan")}


def build_points(args: argparse.Namespace) -> list[dict[str, Any]]:
    questions = load_casehold_questions()
    labels = list(questions)
    retrieval = {key: load_by_label(arm.retrieval) for key, arm in ARMS.items()}
    generation = {
        key: load_by_label(arm.generation)
        for key, arm in ARMS.items()
        if arm.generation is not None
    }
    for key, rows in retrieval.items():
        missing = [label for label in labels if label not in rows]
        if missing:
            raise SystemExit(f"{key}: retrieval cache missing {len(missing)} labels, first={missing[:5]}")
    for key, rows in generation.items():
        missing = [label for label in labels if label not in rows]
        if missing:
            raise SystemExit(f"{key}: generation cache missing {len(missing)} labels, first={missing[:5]}")

    gold_ids = sorted({gid for row in questions.values() for gid in row["gold_ids"]})
    gold_docs = fetch_docs_by_idx("casehold_holdings", gold_ids, batch_size=args.doc_batch_size)
    ce = get_cross_encoder()
    ce_scores: dict[str, dict[str, float]] = {}
    for key, arm in ARMS.items():
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
            tag=f"CaseHOLD/{arm.display}",
        )
        ce_scores[key] = {label: score for label, (score, _) in scored.items()}

    points: list[dict[str, Any]] = []
    for label in labels:
        points.append({
            "dataset": "casehold",
            "label": label,
            "idx": questions[label]["idx"],
            "gold_ids": questions[label]["gold_ids"],
            "hits": {key: hit_at(retrieval[key][label], 5) for key in ARMS},
            "ce": {key: ce_scores[key].get(label, float("nan")) for key in ARMS},
            "ce_delta_vs_raw": {
                key: ce_scores[key].get(label, float("nan")) - ce_scores["raw"].get(label, float("nan"))
                for key in ARMS
            },
        })
    return points


def summarize(points: list[dict[str, Any]], arm_key: str) -> dict[str, Any]:
    raw_hits = [int(row["hits"]["raw"]) for row in points]
    arm_hits = [int(row["hits"][arm_key]) for row in points]
    ri = pair_ri(arm_hits, raw_hits)
    deltas = [float(row["ce_delta_vs_raw"][arm_key]) for row in points]
    ces = [float(row["ce"][arm_key]) for row in points]
    return {
        "n": len(points),
        "hit_n": sum(arm_hits),
        "hit5": sum(arm_hits) / len(points) if points else float("nan"),
        "help": ri["help"],
        "hurt": ri["hurt"],
        "ri": ri["ri"],
        "mean_ce": mean(ces),
        "mean_ce_delta": mean(deltas),
    }


def write_report(points: list[dict[str, Any]], args: argparse.Namespace) -> None:
    summaries = {key: summarize(points, key) for key in ARMS}
    csqe_hit = summaries["csqe"]["hit5"]
    parametric_best = max(summaries["hyde"]["hit5"], summaries["scope"]["hit5"])
    supported = csqe_hit <= 0.25 and csqe_hit < parametric_best - 0.05
    killed = csqe_hit > 0.30
    verdict = "supported" if supported else ("killed" if killed else "mixed")

    args.points_out.parent.mkdir(parents=True, exist_ok=True)
    with args.points_out.open("w") as f:
        for row in points:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    lines: list[str] = [
        "# CaseHOLD CSQE Collapse - 2026-05-28",
        "",
        (
            f"Verdict: **{verdict}** for H-collapse-2nd. CaseHOLD CSQE reaches "
            f"{pct(csqe_hit)} Hit@5, only {fmt(summaries['csqe']['ri'], 3)} RI over raw, "
            f"while the existing Llama-70B HyDE/SCOPE caches are {pct(summaries['hyde']['hit5'])} "
            f"and {pct(summaries['scope']['hit5'])} Hit@5. The mechanism sub-check is "
            "**mixed**: CSQE moves toward gold in CE space, but much less than HyDE/SCOPE and "
            "not enough to create a meaningful retrieval lift."
        ),
        "",
        "## Main Table",
        "",
        "| Arm | Generator/source | N | Hit@5 | Hits | RI vs raw | Help | Hurt | Mean CE(gold) | Mean CE delta vs raw |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ["raw", "hyde", "scope", "csqe"]:
        arm = ARMS[key]
        s = summaries[key]
        lines.append(
            f"| {arm.display} | {arm.provider_note} | {s['n']} | {pct(s['hit5'])} | {s['hit_n']} | "
            f"{fmt(s['ri'], 3)} | {s['help']} | {s['hurt']} | {fmt(s['mean_ce'], 3)} | "
            f"{fmt(s['mean_ce_delta'], 3)} |"
        )

    lines.extend([
        "",
        "## Reading",
        "",
        (
            "- CaseHOLD is an intermediate-weak point, not a BarExam clone: raw Hit@5 is "
            f"{pct(summaries['raw']['hit5'])}, versus the prior BarExam raw Hit@5 of 1.4%."
        ),
        (
            "- CSQE barely improves retrieval exposure here: "
            f"{pct(summaries['csqe']['hit5'])} vs raw {pct(summaries['raw']['hit5'])}; "
            f"help={summaries['csqe']['help']} and hurt={summaries['csqe']['hurt']}."
        ),
        (
            "- The parametric expansion arms remain much stronger on the same gold-labeled set: "
            f"HyDE RI={fmt(summaries['hyde']['ri'], 3)} and SCOPE RI={fmt(summaries['scope']['ri'], 3)}."
        ),
        (
            "- The pre-stated near-zero/negative CE-delta mechanism is not literally met: "
            f"CSQE's mean CE gold-affinity delta is positive at {fmt(summaries['csqe']['mean_ce_delta'], 3)}. "
            f"The useful distinction is magnitude: HyDE is {fmt(summaries['hyde']['mean_ce_delta'], 3)} "
            f"and SCOPE is {fmt(summaries['scope']['mean_ce_delta'], 3)}, so CSQE shifts toward gold "
            "but remains much weaker."
        ),
        (
            "- Aggregated with the prior BarExam CSQE sweep, this gives two weak-query legal "
            "sets where CSQE is not the winning expansion arm. BarExam is the extreme weak point; "
            "CaseHOLD shows the gradient point between BarExam and HousingQA."
        ),
        "",
        "## Caveats",
        "",
        (
            "- CSQE was generated with `or-gemma4-26b`, while the HyDE/SCOPE rows are existing "
            "`groq-llama70b` signed caches. Treat this as a mechanism test for CSQE's reliance "
            "on raw-retrieved text, not as a strict model head-to-head."
        ),
        "- Metrics are retrieval exposure only; no downstream CaseHOLD answer cells were run.",
        "",
        "## Sources",
        "",
    ])
    for key in ["raw", "hyde", "scope", "csqe"]:
        arm = ARMS[key]
        lines.append(f"- {arm.display} retrieval: `{rel(arm.retrieval)}`")
        if arm.generation is not None:
            lines.append(f"- {arm.display} generation: `{rel(arm.generation)}`")
    lines.extend([
        f"- Row-level points: `{rel(args.points_out)}`",
        "- Prior BarExam/Housing CSQE context: `docs/generated/csqe_regime_sweep_2026-05-26.md`",
    ])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines).rstrip() + "\n")
    print(args.out)
    print(args.points_out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "docs/generated/casehold_csqe_collapse_2026-05-28.md")
    parser.add_argument("--points-out", type=Path, default=ROOT / "docs/generated/casehold_csqe_collapse_2026-05-28_points.jsonl")
    parser.add_argument("--doc-batch-size", type=int, default=5000)
    parser.add_argument("--ce-batch-size", type=int, default=32)
    parser.add_argument("--ce-chunk-size", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    points = build_points(args)
    write_report(points, args)


if __name__ == "__main__":
    main()
