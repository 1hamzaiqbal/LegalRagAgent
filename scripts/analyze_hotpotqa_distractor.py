#!/usr/bin/env python3
"""Analyze HotpotQA distractor retrieval caches.

The HotpotQA distractor setup has a per-question candidate set. This script
therefore reads the retrieval-cache CE scores directly instead of embedding a
global corpus.
"""
from __future__ import annotations

import csv
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "eval"))

from eval_config import EvalConfig, load_questions as load_eval_questions  # noqa: E402
from eval_harness import _fmt_intermediate, _row_label  # noqa: E402


TOKEN_RE = re.compile(r"[a-z0-9]+")
MODEL = "or-gemma4-26b"
OUT = REPO_ROOT / "docs/generated/hotpotqa_distractor_weakquery_2026-05-26.md"


@dataclass(frozen=True)
class CacheSpec:
    name: str
    display: str
    generation_path: Path | None
    retrieval_path: Path


RAW = CacheSpec(
    name="raw",
    display="Raw question",
    generation_path=None,
    retrieval_path=REPO_ROOT / "caches/retrieval/full/hotpotqa_q1000_seed42_raw_question_k10.jsonl",
)
EXPANSIONS = {
    "hyde": CacheSpec(
        name="hyde",
        display="HyDE",
        generation_path=REPO_ROOT / "caches/generation/full/hotpotqa_q1000_seed42_or-gemma4-26b_rag_hyde.jsonl",
        retrieval_path=REPO_ROOT / "caches/retrieval/full/hotpotqa_q1000_seed42_or-gemma4-26b_rag_hyde_k10.jsonl",
    ),
    "scope": CacheSpec(
        name="scope",
        display="SCOPE / snap_hyre",
        generation_path=REPO_ROOT / "caches/generation/full/hotpotqa_q1000_seed42_or-gemma4-26b_snap_hyre.jsonl",
        retrieval_path=REPO_ROOT / "caches/retrieval/full/hotpotqa_q1000_seed42_or-gemma4-26b_snap_hyre_k10.jsonl",
    ),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def load_by_label(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    return {str(row["label"]): row for row in rows}


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text or "").lower())


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


def signed_pp(value: Any) -> str:
    if not finite(value):
        return "--"
    return f"{100.0 * float(value):+.1f}pp"


def corr(xs: list[float], ys: list[float]) -> dict[str, float]:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if finite(x) and finite(y)]
    if len(pairs) < 3:
        return {"n": len(pairs), "spearman": float("nan"), "kendall": float("nan")}
    x = [p[0] for p in pairs]
    y = [p[1] for p in pairs]
    if len(set(x)) < 2 or len(set(y)) < 2:
        return {"n": len(pairs), "spearman": float("nan"), "kendall": float("nan")}
    s = spearmanr(x, y, nan_policy="omit").statistic
    k = kendalltau(x, y, nan_policy="omit").statistic
    return {
        "n": len(pairs),
        "spearman": float(s) if finite(s) else float("nan"),
        "kendall": float(k) if finite(k) else float("nan"),
    }


def auc_for(features: list[list[float]], target: list[int]) -> float:
    pairs = [(x, y) for x, y in zip(features, target) if all(finite(v) for v in x)]
    if len(pairs) < 10:
        return float("nan")
    x = np.asarray([p[0] for p in pairs], dtype=float)
    y = np.asarray([p[1] for p in pairs], dtype=int)
    if len(set(y.tolist())) < 2:
        return float("nan")
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, solver="liblinear", random_state=42),
    )
    clf.fit(x, y)
    prob = clf.predict_proba(x)[:, 1]
    return float(roc_auc_score(y, prob))


def score_map(row: dict[str, Any]) -> dict[str, float]:
    return {
        str(idx): float(score)
        for idx, score in zip(row.get("retrieved_ids", []) or [], row.get("scores", []) or [])
        if finite(score)
    }


def gold_ids(row: dict[str, Any]) -> list[str]:
    return [str(idx) for idx in row.get("gold_ids", []) or [] if str(idx)]


def max_gold_score(row: dict[str, Any]) -> float:
    scores = score_map(row)
    vals = [scores.get(idx, float("nan")) for idx in gold_ids(row)]
    vals = [v for v in vals if finite(v)]
    return max(vals) if vals else float("nan")


def max_non_gold_score(row: dict[str, Any]) -> float:
    scores = score_map(row)
    gold = set(gold_ids(row))
    vals = [score for idx, score in scores.items() if idx not in gold]
    return max(vals) if vals else float("nan")


def margin(row: dict[str, Any]) -> float:
    g = max_gold_score(row)
    d = max_non_gold_score(row)
    return g - d if finite(g) and finite(d) else float("nan")


def metric_values(row: dict[str, Any], bridge_id: str) -> dict[str, int]:
    ids = [str(idx) for idx in row.get("retrieved_ids", []) or []]
    gold = set(gold_ids(row))
    return {
        "hit@5": int(bool(gold.intersection(ids[:5]))),
        "full@2": int(bool(gold) and gold.issubset(set(ids[:2]))),
        "full@5": int(bool(gold) and gold.issubset(set(ids[:5]))),
        "bridge@2": int(bool(bridge_id) and bridge_id in set(ids[:2])),
        "bridge@5": int(bool(bridge_id) and bridge_id in set(ids[:5])),
    }


def lower_raw_ce_gold(raw_row: dict[str, Any]) -> str:
    scores = score_map(raw_row)
    gold = gold_ids(raw_row)
    if not gold:
        return ""
    return min(gold, key=lambda idx: scores.get(idx, -1e9))


def load_hotpot_questions() -> dict[str, dict[str, Any]]:
    config = EvalConfig(dataset="hotpotqa", questions="1000", seed=42)
    out: dict[str, dict[str, Any]] = {}
    for _, row in load_eval_questions(config).reset_index(drop=True).iterrows():
        label = _row_label(row, config)
        out[label] = {
            "question": _fmt_intermediate(row, config),
            "idx": str(row.get("idx", "")),
            "answer": str(row.get("answer", "")),
            "type": str(row.get("type", "")),
            "level": str(row.get("level", "")),
        }
    return out


def build_lm() -> tuple[Counter[str], int, int]:
    counts: Counter[str] = Counter()
    path = REPO_ROOT / "datasets/hotpotqa_distractor/passages.csv"
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            counts.update(tokenize(row.get("text", "")))
    return counts, sum(counts.values()), max(1, len(counts))


def lm_stats(text: str, counts: Counter[str], total: int, vocab: int) -> tuple[float, float]:
    toks = tokenize(text)
    if not toks:
        return float("nan"), float("nan")
    denom = total + vocab
    log_prob = 0.0
    oov = 0
    for tok in toks:
        c = counts.get(tok, 0)
        if c == 0:
            oov += 1
        log_prob += -math.log((c + 1) / denom)
    return log_prob / len(toks), oov / len(toks)


def summarize_cache(path: Path, mode: str) -> dict[str, Any]:
    rows = read_jsonl(path)
    labels = [str(row.get("label", "")) for row in rows]
    out = {
        "rows": len(rows),
        "duplicates": len(labels) - len(set(labels)),
        "errors": sum(1 for row in rows if row.get("error")),
        "missing_passage": sum(1 for row in rows if mode != "retrieval" and not row.get("hyde_passage")),
        "short_retrieval": sum(1 for row in rows if mode == "retrieval" and len(row.get("retrieved_ids", []) or []) < 10),
        "parse_bad": sum(1 for row in rows if row.get("snap_hyre_parse_ok") is False),
        "answer_artifact": sum(1 for row in rows if row.get("hyde_contains_answer_artifact") is True),
    }
    if mode != "retrieval":
        out["format_retry"] = sum(1 for row in rows if row.get("snap_hyre_format_retry") or row.get("hyde_format_retry"))
    return out


def ri_rows(raw_vals: list[int], exp_vals: list[int]) -> tuple[int, int, float]:
    helps = sum(1 for raw, exp in zip(raw_vals, exp_vals) if exp > raw)
    hurts = sum(1 for raw, exp in zip(raw_vals, exp_vals) if exp < raw)
    n = len(raw_vals)
    return helps, hurts, (helps - hurts) / n if n else float("nan")


def quintile_rows(values: list[float]) -> list[list[int]]:
    finite_pairs = sorted((value, i) for i, value in enumerate(values) if finite(value))
    n = len(finite_pairs)
    bins: list[list[int]] = []
    for b in range(5):
        lo = round(b * n / 5)
        hi = round((b + 1) * n / 5)
        bins.append([idx for _, idx in finite_pairs[lo:hi]])
    return bins


def main() -> None:
    questions = load_hotpot_questions()
    counts, total, vocab = build_lm()
    raw = load_by_label(RAW.retrieval_path)
    exp_rows = {name: load_by_label(spec.retrieval_path) for name, spec in EXPANSIONS.items()}
    labels = sorted(set(raw).intersection(*(set(rows) for rows in exp_rows.values())))

    per_label: dict[str, dict[str, Any]] = {}
    for label in labels:
        q = questions.get(label, {})
        raw_row = raw[label]
        bridge = lower_raw_ce_gold(raw_row)
        logppl, oov = lm_stats(q.get("question", ""), counts, total, vocab)
        raw_metrics = metric_values(raw_row, bridge)
        per_label[label] = {
            "label": label,
            "question": q.get("question", ""),
            "type": q.get("type", ""),
            "level": q.get("level", ""),
            "bridge_id": bridge,
            "logppl": logppl,
            "oov": oov,
            "raw_margin": margin(raw_row),
            "raw_gold": max_gold_score(raw_row),
            "raw_metrics": raw_metrics,
        }
        for name, rows in exp_rows.items():
            row = rows[label]
            exp_metrics = metric_values(row, bridge)
            exp_gold = max_gold_score(row)
            exp_margin = margin(row)
            per_label[label][name] = {
                "gold": exp_gold,
                "margin": exp_margin,
                "delta_gold": exp_gold - per_label[label]["raw_gold"],
                "delta_margin": exp_margin - per_label[label]["raw_margin"],
                "metrics": exp_metrics,
                "gain": {m: exp_metrics[m] - raw_metrics[m] for m in raw_metrics},
            }

    metrics = ["hit@5", "full@2", "full@5", "bridge@2", "bridge@5"]
    lines: list[str] = []
    lines.append("# HotpotQA Distractor Weak-Query Verification")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(
        "The q1000 distractor slice is clean but does **not** show the expected expansion help side. "
        "Raw-question retrieval is already near-saturated on standard Hit@5 and is strongest on the multi-hop metrics. "
        "Because SCOPE and HyDE both net-hurt full-support and bridge recall, I did not scale to full N=7405 and did not run the optional q500 answer EM."
    )
    lines.append("")
    lines.append("## Source Files")
    lines.append("")
    lines.append("| Role | Path |")
    lines.append("|---|---|")
    lines.append(f"| Raw retrieval | `{RAW.retrieval_path.relative_to(REPO_ROOT)}` |")
    for spec in EXPANSIONS.values():
        assert spec.generation_path is not None
        lines.append(f"| {spec.display} generation | `{spec.generation_path.relative_to(REPO_ROOT)}` |")
        lines.append(f"| {spec.display} retrieval | `{spec.retrieval_path.relative_to(REPO_ROOT)}` |")
    lines.append("| Dataset questions | `datasets/hotpotqa_distractor/questions.csv` |")
    lines.append("| Per-question paragraphs | `datasets/hotpotqa_distractor/passages.csv` |")
    lines.append("")
    lines.append("## Cache Health")
    lines.append("")
    lines.append("| Cache | Rows | Duplicates | Errors | Missing passage | Parse bad | Answer artifact | Short retrieval | Format retry |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    raw_health = summarize_cache(RAW.retrieval_path, "retrieval")
    lines.append(f"| Raw retrieval | {raw_health['rows']} | {raw_health['duplicates']} | {raw_health['errors']} | -- | -- | -- | {raw_health['short_retrieval']} | -- |")
    for spec in EXPANSIONS.values():
        gen_health = summarize_cache(spec.generation_path, "generation") if spec.generation_path else {}
        ret_health = summarize_cache(spec.retrieval_path, "retrieval")
        lines.append(
            f"| {spec.display} generation | {gen_health['rows']} | {gen_health['duplicates']} | {gen_health['errors']} | "
            f"{gen_health['missing_passage']} | {gen_health['parse_bad']} | {gen_health['answer_artifact']} | -- | {gen_health.get('format_retry', 0)} |"
        )
        lines.append(f"| {spec.display} retrieval | {ret_health['rows']} | {ret_health['duplicates']} | {ret_health['errors']} | -- | -- | -- | {ret_health['short_retrieval']} | -- |")
    lines.append("")
    lines.append("Short retrieval lists are expected for the six validation rows whose original HotpotQA candidate set has fewer than ten paragraphs.")
    lines.append("")

    lines.append("## Retrieval Metrics")
    lines.append("")
    lines.append("| Method | Hit@5 | Full-support@2 | Full-support@5 | Bridge@2 | Bridge@5 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    method_metric_values: dict[str, dict[str, list[int]]] = {}
    for method_name in ["raw", *EXPANSIONS.keys()]:
        method_metric_values[method_name] = {m: [] for m in metrics}
    for label in labels:
        for m in metrics:
            method_metric_values["raw"][m].append(per_label[label]["raw_metrics"][m])
            for name in EXPANSIONS:
                method_metric_values[name][m].append(per_label[label][name]["metrics"][m])
    lines.append("| Raw question | " + " | ".join(pct(mean(method_metric_values["raw"][m])) for m in metrics) + " |")
    for name, spec in EXPANSIONS.items():
        lines.append("| " + spec.display + " | " + " | ".join(pct(mean(method_metric_values[name][m])) for m in metrics) + " |")
    lines.append("")
    lines.append("Bridge paragraph = the gold paragraph with the lower raw-query CE score for that question.")
    lines.append("")

    lines.append("## Expansion vs Raw")
    lines.append("")
    lines.append("| Method | Metric | Delta | Help rows | Hurt rows | RI |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for name, spec in EXPANSIONS.items():
        for m in metrics:
            raw_vals = method_metric_values["raw"][m]
            exp_vals = method_metric_values[name][m]
            helps, hurts, ri = ri_rows(raw_vals, exp_vals)
            delta = mean(exp_vals) - mean(raw_vals)
            lines.append(f"| {spec.display} | {m} | {signed_pp(delta)} | {helps} | {hurts} | {fmt(ri)} |")
    lines.append("")

    axes = {
        "delta margin": "delta_margin",
        "gold-affinity delta": "delta_gold",
        "raw margin": "raw_margin",
        "log perplexity": "logppl",
        "OOV rate": "oov",
    }
    lines.append("## Mechanism Correlations")
    lines.append("")
    lines.append("| Method | Axis | Gain metric | N | Spearman | Kendall |")
    lines.append("|---|---|---|---:|---:|---:|")
    for name, spec in EXPANSIONS.items():
        for axis_name, axis_key in axes.items():
            xs: list[float] = []
            for label in labels:
                if axis_key in {"raw_margin", "logppl", "oov"}:
                    xs.append(per_label[label][axis_key])
                else:
                    xs.append(per_label[label][name][axis_key])
            for metric in ["full@5", "bridge@5", "hit@5"]:
                ys = [per_label[label][name]["gain"][metric] for label in labels]
                c = corr(xs, ys)
                lines.append(f"| {spec.display} | {axis_name} | {metric} | {int(c['n'])} | {fmt(c['spearman'])} | {fmt(c['kendall'])} |")
    lines.append("")
    lines.append("Margin correlations have N=999 because one q1000 row has no non-gold candidate available for the distractor maximum; retrieval metrics use all 1000 rows.")
    lines.append("")

    lines.append("## Raw-Margin Quintiles")
    lines.append("")
    raw_margins = [per_label[label]["raw_margin"] for label in labels]
    bins = quintile_rows(raw_margins)
    lines.append("| Method | Raw-margin bin | N | Raw full@5 | Method full@5 | Delta | Raw bridge@5 | Method bridge@5 | Delta |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, spec in EXPANSIONS.items():
        for i, idxs in enumerate(bins, 1):
            raw_full = [method_metric_values["raw"]["full@5"][j] for j in idxs]
            exp_full = [method_metric_values[name]["full@5"][j] for j in idxs]
            raw_bridge = [method_metric_values["raw"]["bridge@5"][j] for j in idxs]
            exp_bridge = [method_metric_values[name]["bridge@5"][j] for j in idxs]
            lines.append(
                f"| {spec.display} | Q{i} | {len(idxs)} | {pct(mean(raw_full))} | {pct(mean(exp_full))} | {signed_pp(mean(exp_full) - mean(raw_full))} | "
                f"{pct(mean(raw_bridge))} | {pct(mean(exp_bridge))} | {signed_pp(mean(exp_bridge) - mean(raw_bridge))} |"
            )
    lines.append("")

    lines.append("## P4 Failure AUC")
    lines.append("")
    lines.append("Target is `delta margin < 0`; geometry uses `{M_raw, CE(exp,gold)}`, surprise uses `{OOV, log perplexity}`.")
    lines.append("")
    lines.append("| Method | Failure rate | Geometry AUC | Surprise AUC |")
    lines.append("|---|---:|---:|---:|")
    for name, spec in EXPANSIONS.items():
        target = [int(per_label[label][name]["delta_margin"] < 0) for label in labels]
        geometry = [[per_label[label]["raw_margin"], per_label[label][name]["gold"]] for label in labels]
        surprise = [[per_label[label]["oov"], per_label[label]["logppl"]] for label in labels]
        lines.append(f"| {spec.display} | {pct(mean(target))} | {fmt(auc_for(geometry, target))} | {fmt(auc_for(surprise, target))} |")
    lines.append("")

    lines.append("## SCOPE vs HyDE")
    lines.append("")
    lines.append("| Metric | SCOPE minus HyDE | SCOPE-help rows | HyDE-help rows |")
    lines.append("|---|---:|---:|---:|")
    for m in metrics:
        scope_vals = method_metric_values["scope"][m]
        hyde_vals = method_metric_values["hyde"][m]
        diff = mean(scope_vals) - mean(hyde_vals)
        scope_help = sum(1 for raw_v, exp_v in zip(method_metric_values["raw"][m], scope_vals) if exp_v > raw_v)
        hyde_help = sum(1 for raw_v, exp_v in zip(method_metric_values["raw"][m], hyde_vals) if exp_v > raw_v)
        lines.append(f"| {m} | {signed_pp(diff)} | {scope_help} | {hyde_help} |")
    lines.append("")

    lines.append("## Reading")
    lines.append("")
    lines.append(
        "HotpotQA distractor q1000 is not weak enough in this within-question candidate form. "
        "The raw query sees all ten candidate paragraphs and the CE reranker usually places at least one gold paragraph in the top five; "
        "the harder full-support and bridge metrics are still best under raw retrieval. "
        "SCOPE is less damaging than HyDE on Hit@5 and full-support, but it does not cross into net-help."
    )
    lines.append("")
    lines.append(
        "The margin mechanism is only weakly visible: SCOPE delta margin has the largest positive correlation with gain, but it stays below a strong explanatory threshold "
        "and low raw-margin bins still do not become a positive SCOPE regime. P4 does rule out corpus-surprise as the main failure explanation here: OOV/log-perplexity are near chance, "
        "while geometry is much more predictive of negative margin movement. That geometry result is partly expected because the target is defined from the margin itself, "
        "so it should not be read as evidence that expansion helps in this setting. "
        "For the help-side benchmark, the next better target is a setting with a larger candidate pool or weaker literal query anchoring, such as full-wiki HotpotQA or MuSiQue."
    )
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append("- Stop HotpotQA distractor at q1000 for this lane.")
    lines.append("- Do not scale to full 7405 under the current per-question distractor retrieval setup.")
    lines.append("- Do not run optional q500 answer EM because the retrieval premise is net-negative.")
    lines.append("- Keep the q1000 caches as reusable artifacts for future prompt/selection comparisons.")
    lines.append("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines).rstrip() + "\n")
    print(OUT.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
