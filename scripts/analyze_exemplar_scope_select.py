#!/usr/bin/env python3
"""Analyze Phase-A exemplar-grounded SCOPE results on BEIR caches."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import binomtest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "eval"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from analyze_beir_phase1 import (  # noqa: E402
    fetch_docs_by_idx,
    generation_passage,
    score_best_gold_ce,
)
from eval_config import BEIR_DATASETS, EvalConfig, load_questions  # noqa: E402
from eval_harness import _gold_ids, _retrieval_question, _row_label  # noqa: E402
from rag_utils import get_cross_encoder  # noqa: E402


MODEL = "or-gemma4-26b"
DATASETS = ["beir_scifact", "beir_nfcorpus", "beir_fiqa", "beir_trec_covid", "beir_scidocs"]


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    subset: str
    display: str
    collection: str


@dataclass(frozen=True)
class ArmSpec:
    key: str
    display: str
    retrieval_path: Path
    generation_path: Path | None = None
    generation_mode: str = ""


DISPLAY = {
    "beir_scifact": "SciFact",
    "beir_nfcorpus": "NFCorpus",
    "beir_fiqa": "FiQA",
    "beir_trec_covid": "TREC-COVID",
    "beir_scidocs": "SciDocs",
}


def cache_path(path: str) -> Path:
    return REPO_ROOT / path


def specs() -> list[DatasetSpec]:
    return [
        DatasetSpec(key=key, subset=BEIR_DATASETS[key], display=DISPLAY[key], collection=key)
        for key in DATASETS
    ]


def arm_specs(dataset: str) -> dict[str, ArmSpec]:
    return {
        "raw": ArmSpec(
            key="raw",
            display="Raw",
            retrieval_path=cache_path(f"caches/retrieval/full/{dataset}_qfull_seed42_raw_question_k10.jsonl"),
        ),
        "hyde": ArmSpec(
            key="hyde",
            display="HyDE",
            retrieval_path=cache_path(f"caches/retrieval/full/{dataset}_qfull_seed42_{MODEL}_rag_hyde_k10.jsonl"),
            generation_path=cache_path(f"caches/generation/full/{dataset}_qfull_seed42_{MODEL}_rag_hyde.jsonl"),
            generation_mode="rag_hyde",
        ),
        "scope": ArmSpec(
            key="scope",
            display="SCOPE",
            retrieval_path=cache_path(f"caches/retrieval/full/{dataset}_qfull_seed42_{MODEL}_snap_hyre_k10.jsonl"),
            generation_path=cache_path(f"caches/generation/full/{dataset}_qfull_seed42_{MODEL}_snap_hyre.jsonl"),
            generation_mode="snap_hyre",
        ),
        "csqe": ArmSpec(
            key="csqe",
            display="CSQE",
            retrieval_path=cache_path(f"caches/retrieval/full/{dataset}_qfull_seed42_csqe_k10.jsonl"),
            generation_path=cache_path(f"caches/generation/full/{dataset}_qfull_seed42_csqe.jsonl"),
            generation_mode="csqe",
        ),
        "scope_ex": ArmSpec(
            key="scope_ex",
            display="SCOPE-exemplar",
            retrieval_path=cache_path(
                f"caches/retrieval/full/{dataset}_qfull_seed42_{MODEL}_snap_hyre_exemplar_orthogonal3_k10.jsonl"
            ),
            generation_path=cache_path(
                f"caches/generation/full/{dataset}_qfull_seed42_{MODEL}_snap_hyre_exemplar_orthogonal3.jsonl"
            ),
            generation_mode="snap_hyre_exemplar",
        ),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def load_by_label(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        rows[str(row.get("label") or row.get("idx"))] = row
    return rows


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


def pvalue_fmt(value: float) -> str:
    if not finite(value):
        return "--"
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def hit_at(row: dict[str, Any], k: int = 5) -> int:
    gold = {str(idx) for idx in row.get("gold_ids", []) if str(idx)}
    got = {str(idx) for idx in (row.get("retrieved_ids") or [])[:k]}
    return int(bool(gold & got)) if gold else 0


def load_exemplar_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    return payload.get("datasets", payload)


def load_questions_for_spec(spec: DatasetSpec, exclude_gold_ids: list[str]) -> dict[str, dict[str, Any]]:
    config = EvalConfig(
        dataset=spec.key,
        questions="full",
        seed=42,
        exclude_gold_ids=",".join(str(idx) for idx in exclude_gold_ids),
    )
    questions: dict[str, dict[str, Any]] = {}
    for _, row in load_questions(config).iterrows():
        label = _row_label(row, config)
        questions[label] = {
            "label": label,
            "idx": str(row.get("idx", "")),
            "question": _retrieval_question(row),
            "gold_ids": [str(idx) for idx in _gold_ids(row) if str(idx)],
        }
    return questions


def load_phase1_ce(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    if not path.exists():
        return out
    for row in read_jsonl(path):
        out[(str(row["dataset"]), str(row["label"]), str(row["expansion"]))] = row
    return out


def bootstrap_ci(deltas: list[int], *, samples: int = 2000, seed: int = 42) -> tuple[float, float]:
    if not deltas:
        return float("nan"), float("nan")
    arr = np.asarray(deltas, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    n = len(arr)
    for i in range(samples):
        means[i] = float(arr[rng.integers(0, n, size=n)].mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def mcnemar(values: list[tuple[int, int]]) -> dict[str, Any]:
    b = sum(1 for arm, base in values if arm == 1 and base == 0)
    c = sum(1 for arm, base in values if arm == 0 and base == 1)
    p = 1.0 if b + c == 0 else float(binomtest(min(b, c), n=b + c, p=0.5).pvalue)
    return {"arm_only": b, "baseline_only": c, "p": p}


def pair_ri(arm_hits: list[int], baseline_hits: list[int]) -> dict[str, Any]:
    help_n = sum(1 for a, b in zip(arm_hits, baseline_hits) if a == 1 and b == 0)
    hurt_n = sum(1 for a, b in zip(arm_hits, baseline_hits) if a == 0 and b == 1)
    n = len(arm_hits)
    return {
        "help": help_n,
        "hurt": hurt_n,
        "ri": (help_n - hurt_n) / n if n else float("nan"),
    }


def source_paths() -> list[str]:
    paths: list[str] = [
        "caches/exemplars/beir_orthogonal3_exemplars_2026-05-26.json",
        "docs/generated/beir_orthogonal3_exemplars_2026-05-26.md",
    ]
    for spec in specs():
        for arm in arm_specs(spec.key).values():
            for path in (arm.retrieval_path, arm.generation_path):
                if path is None:
                    continue
                rel = str(path.relative_to(REPO_ROOT))
                if rel not in paths:
                    paths.append(rel)
    return paths


def build_dataset_points(
    *,
    spec: DatasetSpec,
    exemplar_record: dict[str, Any],
    phase1_ce: dict[tuple[str, str, str], dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    exclude_ids = [str(idx) for idx in exemplar_record.get("eval_exclude_gold_ids", []) if str(idx)]
    questions = load_questions_for_spec(spec, exclude_ids)
    arms = arm_specs(spec.key)
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
            raise RuntimeError(f"{spec.key}/{key}: retrieval cache missing {missing[:5]} n={len(missing)}")

    ce_scores: dict[str, dict[str, float]] = {key: {} for key in arms}
    for label in labels:
        raw_hyde = phase1_ce.get((spec.key, label, "hyde"))
        raw_scope = phase1_ce.get((spec.key, label, "scope"))
        raw_source = raw_hyde or raw_scope
        if raw_source:
            ce_scores["raw"][label] = float(raw_source["ce_raw_gold"])
        hyde = phase1_ce.get((spec.key, label, "hyde"))
        if hyde:
            ce_scores["hyde"][label] = float(hyde["ce_exp_gold"])
        scope = phase1_ce.get((spec.key, label, "scope"))
        if scope:
            ce_scores["scope"][label] = float(scope["ce_exp_gold"])

    need_raw = [label for label in labels if label not in ce_scores["raw"]]
    need_hyde = [label for label in labels if label not in ce_scores["hyde"]]
    need_scope = [label for label in labels if label not in ce_scores["scope"]]
    needs_model = ["csqe", "scope_ex"]
    if need_raw or need_hyde or need_scope:
        needs_model.extend(
            key for key, needed in (("raw", need_raw), ("hyde", need_hyde), ("scope", need_scope)) if needed
        )

    gold_ids = sorted({gid for row in questions.values() for gid in row["gold_ids"]})
    gold_docs = fetch_docs_by_idx(spec.collection, gold_ids, batch_size=args.doc_batch_size)
    ce = get_cross_encoder()
    for key in needs_model:
        if key == "raw":
            items = [(label, questions[label]["question"], questions[label]["gold_ids"]) for label in labels]
        else:
            gen_rows = generation[key]
            missing = [label for label in labels if label not in gen_rows]
            if missing:
                raise RuntimeError(f"{spec.key}/{key}: generation cache missing {missing[:5]} n={len(missing)}")
            items = [
                (label, generation_passage(gen_rows[label]), questions[label]["gold_ids"])
                for label in labels
            ]
        scored = score_best_gold_ce(
            ce=ce,
            items=items,
            gold_docs=gold_docs,
            batch_size=args.ce_batch_size,
            chunk_size=args.ce_chunk_size,
            tag=f"{spec.display}/{arms[key].display}",
        )
        ce_scores[key].update({label: score for label, (score, _) in scored.items()})

    points: list[dict[str, Any]] = []
    for label in labels:
        hits = {key: hit_at(retrieval[key][label], 5) for key in arms}
        point = {
            "dataset": spec.key,
            "dataset_display": spec.display,
            "label": label,
            "idx": questions[label]["idx"],
            "gold_count": len(questions[label]["gold_ids"]),
            "hits": hits,
            "ce": {key: ce_scores[key].get(label, float("nan")) for key in arms},
        }
        points.append(point)
    return points


def write_points(path: Path, points: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for point in points:
            f.write(json.dumps(point, sort_keys=True) + "\n")


def summarize_hits(points: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    hits = [int(p["hits"][arm]) for p in points]
    raw = [int(p["hits"]["raw"]) for p in points]
    ri = pair_ri(hits, raw)
    raw_ce = [float(p["ce"]["raw"]) for p in points]
    arm_ce = [float(p["ce"][arm]) for p in points]
    return {
        "n": len(points),
        "hit5": mean(hits),
        "count": sum(hits),
        "ri_raw": ri["ri"],
        "help_raw": ri["help"],
        "hurt_raw": ri["hurt"],
        "mean_ce_delta": mean(a - r for a, r in zip(arm_ce, raw_ce)),
    }


def contrast(points: list[dict[str, Any]], arm: str, baseline: str) -> dict[str, Any]:
    arm_hits = [int(p["hits"][arm]) for p in points]
    base_hits = [int(p["hits"][baseline]) for p in points]
    deltas = [a - b for a, b in zip(arm_hits, base_hits)]
    ci_lo, ci_hi = bootstrap_ci(deltas)
    mcn = mcnemar(list(zip(arm_hits, base_hits)))
    return {
        "n": len(points),
        "arm_hit": mean(arm_hits),
        "baseline_hit": mean(base_hits),
        "delta": mean(deltas),
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        **mcn,
    }


def verdicts(points: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    pooled = points
    summaries = {arm: summarize_hits(pooled, arm) for arm in ["hyde", "scope", "csqe", "scope_ex"]}
    sx = summaries["scope_ex"]
    scope = summaries["scope"]
    hyde = summaries["hyde"]
    csqe = summaries["csqe"]

    if sx["hit5"] > scope["hit5"] and sx["hit5"] > hyde["hit5"] and sx["mean_ce_delta"] > scope["mean_ce_delta"]:
        h1 = "supported"
    elif sx["hit5"] > hyde["hit5"] or sx["mean_ce_delta"] > scope["mean_ce_delta"]:
        h1 = "mixed"
    else:
        h1 = "killed"

    positive_cells = []
    for dataset in DATASETS:
        dpoints = [p for p in points if p["dataset"] == dataset]
        for arm in ["hyde", "scope", "csqe", "scope_ex"]:
            s = summarize_hits(dpoints, arm)
            if finite(s["ri_raw"]) and s["ri_raw"] > 0:
                positive_cells.append(f"{DISPLAY[dataset]}/{arm}")

    if positive_cells:
        h3 = "supported"
        h3_note = "positive RI cells: " + ", ".join(positive_cells)
    else:
        h3 = "killed"
        h3_note = "no arm had positive RI vs raw on any strong-query BEIR set"

    if sx["hit5"] >= csqe["hit5"]:
        h4 = "supported"
    elif sx["hit5"] >= csqe["hit5"] - 0.01:
        h4 = "mixed"
    else:
        h4 = "killed"

    return [
        (
            "H1 grounding cuts drift",
            h1,
            "SCOPE-exemplar pooled Hit@5 "
            f"{pct(sx['hit5'])} vs vanilla SCOPE {pct(scope['hit5'])}, HyDE {pct(hyde['hit5'])}; "
            f"mean CE gold delta {fmt(sx['mean_ce_delta'])} vs SCOPE {fmt(scope['mean_ce_delta'])}",
        ),
        (
            "H2 selection helps",
            "not run",
            "Phase B was gated on Phase A promise; Phase A did not beat vanilla SCOPE or CSQE pooled.",
        ),
        ("H3 net-positive on strong-query BEIR", h3, h3_note),
        (
            "H4 snap-answer adds over CSQE",
            h4,
            f"SCOPE-exemplar pooled Hit@5 {pct(sx['hit5'])}; CSQE pooled Hit@5 {pct(csqe['hit5'])}.",
        ),
        (
            "H5 weak-query intact",
            "not run",
            "Weak-query control belongs to Phase B and was not launched after the Phase A stop decision.",
        ),
    ]


def write_report(
    output: Path,
    points: list[dict[str, Any]],
    exemplar_payload: dict[str, Any],
) -> None:
    arms = ["raw", "hyde", "scope", "csqe", "scope_ex"]
    arm_display = {
        "raw": "Raw",
        "hyde": "HyDE",
        "scope": "SCOPE",
        "csqe": "CSQE",
        "scope_ex": "SCOPE-exemplar",
    }

    groups: list[tuple[str, list[dict[str, Any]]]] = [
        (DISPLAY[key], [p for p in points if p["dataset"] == key]) for key in DATASETS
    ]
    groups.append(("Pooled", points))

    lines: list[str] = []
    lines.append("# Exemplar-Grounded SCOPE Phase A - 2026-05-26")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("Phase A tests a single exemplar-grounded SCOPE candidate against raw retrieval, HyDE, vanilla SCOPE, and deterministic CSQE on five BEIR strong-query sets. The only model calls were SCOPE-exemplar query-generation calls; no downstream answer cells were run, and no files under `paper/` were edited.")
    lines.append("")
    lines.append("SciDocs note: every document in the local SciDocs corpus snapshot is a qrels positive for some eval query. To avoid exemplar leakage, the three selected medoid ids are treated as eval exclusions for SciDocs, removing 11/1000 rows from every Phase-A comparison.")
    lines.append("")

    lines.append("## Verdicts")
    lines.append("")
    lines.append("| Hypothesis | Verdict | Key read |")
    lines.append("|---|---|---|")
    for hyp, verdict, note in verdicts(points):
        lines.append(f"| {hyp} | **{verdict}** | {note} |")
    lines.append("")
    lines.append("Phase B decision: **stop for now**. The single-candidate exemplar arm does not beat vanilla SCOPE pooled and is far below CSQE pooled, so the selection arms are not justified under the pre-stated gate.")
    lines.append("")

    lines.append("## Exemplar Guardrail")
    lines.append("")
    lines.append("| Dataset | Source | Exemplar ids | Eval rows excluded |")
    lines.append("|---|---|---|---:|")
    for spec in specs():
        rec = exemplar_payload[spec.key]
        lines.append(
            f"| {spec.display} | {rec.get('embedding_source', '')} | "
            f"`{', '.join(str(x) for x in rec.get('ids', []))}` | {int(rec.get('eval_rows_excluded') or 0)} |"
        )
    lines.append("")

    lines.append("## Hit@5")
    lines.append("")
    lines.append("| Dataset | Arm | N | Hit@5 | Correct | RI vs raw | Help vs raw | Hurt vs raw | Mean CE gold delta vs raw |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for group_name, gpoints in groups:
        for arm in arms:
            s = summarize_hits(gpoints, arm)
            ce_delta = "--" if arm == "raw" else fmt(s["mean_ce_delta"])
            lines.append(
                f"| {group_name} | {arm_display[arm]} | {s['n']} | {pct(s['hit5'])} | {s['count']} | "
                f"{fmt(0.0 if arm == 'raw' else s['ri_raw'])} | "
                f"{0 if arm == 'raw' else s['help_raw']} | {0 if arm == 'raw' else s['hurt_raw']} | {ce_delta} |"
            )
    lines.append("")

    lines.append("## RI Matrix")
    lines.append("")
    lines.append("Each cell is Collins-Thompson `RI=(help-hurt)/N` for the row arm against the column baseline.")
    lines.append("")
    lines.append("| Dataset | Arm | vs Raw | vs HyDE | vs SCOPE | vs CSQE |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for group_name, gpoints in groups:
        for arm in ["hyde", "scope", "csqe", "scope_ex"]:
            arm_hits = [int(p["hits"][arm]) for p in gpoints]
            vals = []
            for base in ["raw", "hyde", "scope", "csqe"]:
                base_hits = [int(p["hits"][base]) for p in gpoints]
                vals.append(fmt(pair_ri(arm_hits, base_hits)["ri"]))
            lines.append(f"| {group_name} | {arm_display[arm]} | " + " | ".join(vals) + " |")
    lines.append("")

    lines.append("## Key Contrasts")
    lines.append("")
    lines.append("| Dataset | Arm | Baseline | N | Delta Hit@5 | 95% bootstrap CI | Arm-only | Baseline-only | McNemar p |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    contrast_specs = [
        ("hyde", "raw"),
        ("scope", "raw"),
        ("csqe", "raw"),
        ("scope_ex", "raw"),
        ("scope_ex", "hyde"),
        ("scope_ex", "scope"),
        ("scope_ex", "csqe"),
    ]
    for group_name, gpoints in groups:
        for arm, base in contrast_specs:
            c = contrast(gpoints, arm, base)
            lines.append(
                f"| {group_name} | {arm_display[arm]} | {arm_display[base]} | {c['n']} | {pct(c['delta'])} | "
                f"[{pct(c['ci_lo'])}, {pct(c['ci_hi'])}] | {c['arm_only']} | {c['baseline_only']} | {pvalue_fmt(c['p'])} |"
            )
    lines.append("")

    lines.append("## Reading")
    lines.append("")
    pooled = {arm: summarize_hits(points, arm) for arm in arms}
    lines.append(f"- CSQE is the strongest expansion-style arm in Phase A: {pct(pooled['csqe']['hit5'])} pooled Hit@5, only {pct(pooled['csqe']['hit5'] - pooled['raw']['hit5'])} behind raw.")
    lines.append(f"- SCOPE-exemplar does not rescue strong-query BEIR: pooled Hit@5 is {pct(pooled['scope_ex']['hit5'])}, slightly below vanilla SCOPE at {pct(pooled['scope']['hit5'])}.")
    lines.append(f"- The exemplar arm does improve over HyDE by {pct(pooled['scope_ex']['hit5'] - pooled['hyde']['hit5'])} pooled Hit@5, but that is not the relevant bar because vanilla SCOPE already does most of that recovery.")
    lines.append("- The snap-answer component is not adding over corpus steering here. CSQE, which uses real raw top-k corpus snippets without a snap answer, is substantially stronger than SCOPE-exemplar on the pooled retrieval metric.")
    lines.append("- The current evidence favors a selective/gated expansion story rather than more ungated generation. Phase B selection may still be interesting later, but Phase A does not justify spending more model calls under the stated gate.")
    lines.append("")

    lines.append("## Sources")
    lines.append("")
    for path in source_paths():
        lines.append(f"- `{path}`")
    lines.append("- `/tmp/beir_phase1_verification_2026-05-26_points.jsonl` for reused raw/HyDE/SCOPE gold-affinity CE scores.")
    lines.append("")

    lines.append("## Reproduction")
    lines.append("")
    lines.append("```bash")
    lines.append("HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CROSS_ENCODER_DEVICE=cuda \\")
    lines.append("uv run python scripts/analyze_exemplar_scope_select.py \\")
    lines.append("  --output docs/generated/exemplar_scope_select_2026-05-26.md")
    lines.append("```")
    lines.append("")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))


def load_points(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return read_jsonl(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "docs/generated/exemplar_scope_select_2026-05-26.md")
    parser.add_argument("--points-cache", type=Path, default=Path("/tmp/exemplar_scope_select_2026-05-26_points.jsonl"))
    parser.add_argument("--reuse-points", action="store_true")
    parser.add_argument("--exemplar-json", type=Path, default=REPO_ROOT / "caches/exemplars/beir_orthogonal3_exemplars_2026-05-26.json")
    parser.add_argument("--phase1-points", type=Path, default=Path("/tmp/beir_phase1_verification_2026-05-26_points.jsonl"))
    parser.add_argument("--doc-batch-size", type=int, default=5000)
    parser.add_argument("--ce-batch-size", type=int, default=64)
    parser.add_argument("--ce-chunk-size", type=int, default=10000)
    args = parser.parse_args()

    exemplar_payload = load_exemplar_payload(args.exemplar_json)
    points = load_points(args.points_cache) if args.reuse_points else []
    if points:
        print(f"[cache] loaded {len(points)} points from {args.points_cache}", flush=True)
    else:
        phase1_ce = load_phase1_ce(args.phase1_points)
        points = []
        for spec in specs():
            print(f"[dataset] {spec.display}", flush=True)
            points.extend(
                build_dataset_points(
                    spec=spec,
                    exemplar_record=exemplar_payload[spec.key],
                    phase1_ce=phase1_ce,
                    args=args,
                )
            )
        write_points(args.points_cache, points)
        print(f"[cache] wrote {args.points_cache}", flush=True)
    write_report(args.output, points, exemplar_payload)
    print(args.output)


if __name__ == "__main__":
    main()
