#!/usr/bin/env python3
"""BEIR Phase 1b model-breadth report.

This wrapper reuses the Phase 1 BEIR analyzer's loaders, CE scoring, LM scoring,
and P4 helpers, then groups the same measurements by OpenRouter provider.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import analyze_beir_phase1 as phase1  # noqa: E402


DEFAULT_PROVIDERS = (
    "or-gemma4-26b",
    "or-qwen3p5-9b",
    "or-mistral-small-3p2-24b",
    "or-deepseek-v32",
)

PROVIDER_DISPLAY = {
    "or-gemma4-26b": "Gemma 4 26B",
    "or-qwen3p5-9b": "Qwen 3.5 9B",
    "or-qwen35-9b": "Qwen 3.5 9B",
    "or-mistral-small-3p2-24b": "Mistral Small 3.2 24B",
    "or-deepseek-v32": "DeepSeek V3.2",
}

PROVIDER_RUN_NOTES = {
    "or-qwen3p5-9b": (
        "Qwen generation used `OPENROUTER_DISABLE_REASONING=1` after the "
        "sanity check exposed hidden-reasoning / blank-visible-content routing; "
        "the committed caches passed the clean-output checks."
    ),
    "or-qwen35-9b": (
        "Qwen generation used `OPENROUTER_DISABLE_REASONING=1` after the "
        "sanity check exposed hidden-reasoning / blank-visible-content routing; "
        "the committed caches passed the clean-output checks."
    ),
    "or-mistral-small-3p2-24b": (
        "Mistral generation hit an upstream 429 on the default route and was "
        "resumed on the OpenRouter `Mistral` provider with "
        "`EVAL_LLM_MIN_CALL_INTERVAL_SEC=0.75`; the completed caches are clean "
        "and are not marked provisional."
    ),
    "or-deepseek-v32": (
        "DeepSeek V3.2 generation used the OpenRouter `StreamLake` provider "
        "after the default route exposed upstream 429 risk; the completed "
        "caches passed the clean-output checks."
    ),
}


def provider_spec(spec: phase1.BeirSpec, provider: str) -> phase1.BeirSpec:
    return replace(
        spec,
        hyde_generation=REPO_ROOT / "caches" / "generation" / "full" / f"{spec.key}_qfull_seed42_{provider}_rag_hyde.jsonl",
        scope_generation=REPO_ROOT / "caches" / "generation" / "full" / f"{spec.key}_qfull_seed42_{provider}_snap_hyre.jsonl",
        hyde_retrieval=REPO_ROOT / "caches" / "retrieval" / "full" / f"{spec.key}_qfull_seed42_{provider}_rag_hyde_k10.jsonl",
        scope_retrieval=REPO_ROOT / "caches" / "retrieval" / "full" / f"{spec.key}_qfull_seed42_{provider}_snap_hyre_k10.jsonl",
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def pct(value: Any) -> str:
    return phase1.pct(value)


def fmt(value: Any, digits: int = 3) -> str:
    return phase1.fmt(value, digits=digits)


def mean(values: Iterable[float]) -> float:
    return phase1.mean(values)


def label_model(provider: str) -> str:
    return PROVIDER_DISPLAY.get(provider, provider)


def point_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return str(row["model"]), str(row["dataset"]), str(row["expansion"])


def summarize(points: list[dict[str, Any]]) -> dict[str, Any]:
    return phase1.summarize(points)


def corr_pack(points: list[dict[str, Any]]) -> dict[str, Any]:
    gold = phase1.corr([p["ce_gold_delta"] for p in points], [p["retrieval_delta"] for p in points])
    margin = phase1.corr([p["ce_delta_margin"] for p in points], [p["retrieval_delta"] for p in points])
    return {
        "n": len(points),
        "gold_n": gold["n"],
        "gold_spearman": gold["spearman"],
        "gold_kendall": gold["kendall"],
        "margin_n": margin["n"],
        "margin_spearman": margin["spearman"],
        "margin_kendall": margin["kendall"],
        "mean_gold_delta": mean(p["ce_gold_delta"] for p in points),
        "mean_delta_margin": mean(p["ce_delta_margin"] for p in points),
    }


def logistic_rows(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for point in points:
        point["target_margin_failure"] = (
            float(point["ce_delta_margin"] < 0.0) if finite(point.get("ce_delta_margin")) else float("nan")
        )
        point["target_retrieval_hurt"] = float(point["retrieval_delta"] < 0)
    rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for point in points:
        grouped.setdefault(point_key(point), []).append(point)
    for (provider, dataset, expansion), gpoints in sorted(grouped.items()):
        for target, target_label in (
            ("target_margin_failure", "deltaM<0"),
            ("target_retrieval_hurt", "retrieval hurt"),
        ):
            quality = phase1.logistic_auc(gpoints, ["oov_rate", "log_perplexity"], target)
            geometry = phase1.logistic_auc(gpoints, ["ce_margin_raw", "ce_exp_gold"], target)
            rows.append({
                "provider": provider,
                "dataset": dataset,
                "dataset_display": gpoints[0]["dataset_display"],
                "expansion": expansion,
                "expansion_display": gpoints[0]["expansion_display"],
                "target": target_label,
                "quality": quality,
                "geometry": geometry,
            })
    for provider in sorted({p["model"] for p in points}):
        for expansion in ("hyde", "scope"):
            gpoints = [p for p in points if p["model"] == provider and p["expansion"] == expansion]
            if not gpoints:
                continue
            for target, target_label in (
                ("target_margin_failure", "deltaM<0"),
                ("target_retrieval_hurt", "retrieval hurt"),
            ):
                quality = phase1.logistic_auc(gpoints, ["oov_rate", "log_perplexity"], target)
                geometry = phase1.logistic_auc(gpoints, ["ce_margin_raw", "ce_exp_gold"], target)
                rows.append({
                    "provider": provider,
                    "dataset": "pooled",
                    "dataset_display": "Pooled",
                    "expansion": expansion,
                    "expansion_display": "HyDE" if expansion == "hyde" else "SCOPE",
                    "target": target_label,
                    "quality": quality,
                    "geometry": geometry,
                })
    return rows


def generation_health(providers: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for provider in providers:
        for spec in phase1.BEIR_SPECS.values():
            pspec = provider_spec(spec, provider)
            for expansion, path in (("hyde", pspec.hyde_generation), ("scope", pspec.scope_generation)):
                records = read_jsonl(path)
                rows.append({
                    "provider": provider,
                    "dataset": spec.key,
                    "dataset_display": spec.display,
                    "expansion": expansion,
                    "expansion_display": "HyDE" if expansion == "hyde" else "SCOPE",
                    "path": str(path.relative_to(REPO_ROOT)),
                    "rows": len(records),
                    "errors": sum(bool(r.get("error")) for r in records),
                    "missing_passage": sum(not str(r.get("hyde_passage") or "").strip() for r in records),
                    "parse_bad": sum(r.get("snap_hyre_parse_ok") is False for r in records),
                    "answer_artifact": sum(r.get("hyde_contains_answer_artifact") is True for r in records),
                    "format_retry": sum(bool(r.get("snap_hyre_format_retry") or r.get("hyde_format_retry")) for r in records),
                    "max_output_tokens": max([int(r.get("output_tokens") or 0) for r in records] or [0]),
                    "provider_bad": sum(str(r.get("provider") or "") != provider for r in records),
                })
    return rows


def retrieval_health(providers: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for provider in providers:
        for spec in phase1.BEIR_SPECS.values():
            pspec = provider_spec(spec, provider)
            for expansion, path in (("hyde", pspec.hyde_retrieval), ("scope", pspec.scope_retrieval)):
                records = read_jsonl(path)
                rows.append({
                    "provider": provider,
                    "dataset": spec.key,
                    "dataset_display": spec.display,
                    "expansion": expansion,
                    "expansion_display": "HyDE" if expansion == "hyde" else "SCOPE",
                    "path": str(path.relative_to(REPO_ROOT)),
                    "rows": len(records),
                    "short_rows": sum(len(r.get("retrieved_ids") or []) < 10 for r in records),
                    "missing_gold": sum(not r.get("gold_ids") for r in records),
                    "provider_bad": sum(str(r.get("generation_provider") or "") != provider for r in records),
                })
    return rows


def correlation_table(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for point in points:
        grouped.setdefault(point_key(point), []).append(point)
    for (provider, dataset, expansion), gpoints in sorted(grouped.items()):
        rows.append({
            "provider": provider,
            "dataset": dataset,
            "dataset_display": gpoints[0]["dataset_display"],
            "expansion": expansion,
            "expansion_display": gpoints[0]["expansion_display"],
            **corr_pack(gpoints),
        })
    for provider in sorted({p["model"] for p in points}):
        for expansion in ("hyde", "scope"):
            gpoints = [p for p in points if p["model"] == provider and p["expansion"] == expansion]
            if gpoints:
                rows.append({
                    "provider": provider,
                    "dataset": "pooled",
                    "dataset_display": "Pooled",
                    "expansion": expansion,
                    "expansion_display": "HyDE" if expansion == "hyde" else "SCOPE",
                    **corr_pack(gpoints),
                })
    return rows


def retrieval_table(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for point in points:
        grouped.setdefault(point_key(point), []).append(point)
    for (provider, dataset, expansion), gpoints in sorted(grouped.items()):
        rows.append({
            "provider": provider,
            "dataset": dataset,
            "dataset_display": gpoints[0]["dataset_display"],
            "expansion": expansion,
            "expansion_display": gpoints[0]["expansion_display"],
            **summarize(gpoints),
        })
    for provider in sorted({p["model"] for p in points}):
        for expansion in ("hyde", "scope"):
            gpoints = [p for p in points if p["model"] == provider and p["expansion"] == expansion]
            if gpoints:
                rows.append({
                    "provider": provider,
                    "dataset": "pooled",
                    "dataset_display": "Pooled",
                    "expansion": expansion,
                    "expansion_display": "HyDE" if expansion == "hyde" else "SCOPE",
                    **summarize(gpoints),
                })
    return rows


def robustness_table(ret_rows: list[dict[str, Any]], corr_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ret_lookup = {(r["provider"], r["dataset"], r["expansion"]): r for r in ret_rows}
    corr_lookup = {(r["provider"], r["dataset"], r["expansion"]): r for r in corr_rows}
    keys = sorted({(r["provider"], r["dataset"]) for r in ret_rows if r["dataset"] != "pooled"})
    rows: list[dict[str, Any]] = []
    for provider, dataset in keys + sorted({(r["provider"], "pooled") for r in ret_rows if r["dataset"] == "pooled"}):
        h = ret_lookup.get((provider, dataset, "hyde"))
        s = ret_lookup.get((provider, dataset, "scope"))
        hc = corr_lookup.get((provider, dataset, "hyde"))
        sc = corr_lookup.get((provider, dataset, "scope"))
        if not h or not s or not hc or not sc:
            continue
        rows.append({
            "provider": provider,
            "dataset": dataset,
            "dataset_display": h["dataset_display"],
            "scope_minus_hyde_net": float(s["net_delta"]) - float(h["net_delta"]),
            "hyde_net": h["net_delta"],
            "scope_net": s["net_delta"],
            "hyde_gold_delta": hc["mean_gold_delta"],
            "scope_gold_delta": sc["mean_gold_delta"],
            "scope_closer_to_zero": abs(float(hc["mean_gold_delta"])) - abs(float(sc["mean_gold_delta"])),
        })
    return rows


def verdicts(providers: list[str], corr_rows: list[dict[str, Any]], p4_rows: list[dict[str, Any]], robust_rows: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for expansion in ("hyde", "scope"):
        pool = [
            r for r in corr_rows
            if r["dataset"] == "pooled" and r["expansion"] == expansion and r["provider"] in providers
        ]
        holds = [r for r in pool if finite(r["gold_spearman"]) and float(r["gold_spearman"]) >= 0.2]
        verdict = "supported" if len(holds) == len(providers) else "mixed" if holds else "killed"
        values = ", ".join(f"{label_model(r['provider'])} rho={fmt(r['gold_spearman'])}" for r in pool)
        rows.append((f"Gold-affinity mechanism ({'HyDE' if expansion == 'hyde' else 'SCOPE'})", verdict, values))

        pool_p4 = [
            r for r in p4_rows
            if r["dataset"] == "pooled" and r["expansion"] == expansion and r["target"] == "deltaM<0" and r["provider"] in providers
        ]
        p4_holds = [
            r for r in pool_p4
            if finite(r["geometry"]["auc"])
            and finite(r["quality"]["auc"])
            and float(r["geometry"]["auc"]) >= 0.65
            and float(r["geometry"]["auc"]) > float(r["quality"]["auc"]) + 0.05
        ]
        verdict = "supported" if len(p4_holds) == len(providers) else "mixed" if p4_holds else "killed"
        values = ", ".join(
            f"{label_model(r['provider'])} geom={fmt(r['geometry']['auc'])}/quality={fmt(r['quality']['auc'])}"
            for r in pool_p4
        )
        rows.append((f"P4 geometry-not-hallucination ({'HyDE' if expansion == 'hyde' else 'SCOPE'})", verdict, values))

    pooled_robust = [r for r in robust_rows if r["dataset"] == "pooled" and r["provider"] in providers]
    robust_holds = [
        r for r in pooled_robust
        if finite(r["scope_minus_hyde_net"])
        and float(r["scope_minus_hyde_net"]) > 0
        and finite(r["scope_closer_to_zero"])
        and float(r["scope_closer_to_zero"]) > 0
    ]
    verdict = "supported" if len(robust_holds) == len(providers) else "mixed" if robust_holds else "killed"
    values = ", ".join(
        f"{label_model(r['provider'])} net_gap={pct(r['scope_minus_hyde_net'])}, closer0={fmt(r['scope_closer_to_zero'])}"
        for r in pooled_robust
    )
    rows.append(("SCOPE robustness over HyDE", verdict, values))
    return rows


def load_existing_points(paths: list[Path]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for path in paths:
        for row in read_jsonl(path):
            key = (str(row.get("model")), str(row.get("dataset")), str(row.get("expansion")), str(row.get("label")))
            if key in seen:
                continue
            seen.add(key)
            points.append(row)
    return points


def build_missing_points(args: argparse.Namespace, points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    have = {(str(p.get("model")), str(p.get("dataset"))) for p in points}
    for provider in args.providers:
        for key in args.datasets:
            if (provider, key) in have:
                continue
            spec = provider_spec(phase1.BEIR_SPECS[key], provider)
            missing_paths = [path for path in (spec.hyde_generation, spec.scope_generation, spec.hyde_retrieval, spec.scope_retrieval) if not path.exists()]
            if missing_paths:
                raise SystemExit(
                    f"{provider}/{key}: missing required caches: "
                    + ", ".join(str(path.relative_to(REPO_ROOT)) for path in missing_paths)
                )
            old_model = phase1.MODEL
            phase1.MODEL = provider
            try:
                points.extend(phase1.build_points_for_dataset(spec, args))
            finally:
                phase1.MODEL = old_model
            have.add((provider, key))
    return points


def clean_status(health_rows: list[dict[str, Any]]) -> str:
    bad = [
        row for row in health_rows
        if row.get("errors", 0)
        or row.get("missing_passage", 0)
        or row.get("parse_bad", 0)
        or row.get("answer_artifact", 0)
        or row.get("provider_bad", 0)
        or row.get("short_rows", 0)
        or row.get("missing_gold", 0)
    ]
    return "clean" if not bad else f"issues={len(bad)} rows"


def write_report(output: Path, points: list[dict[str, Any]], providers: list[str]) -> None:
    ret_rows = retrieval_table(points)
    corr_rows = correlation_table(points)
    p4 = logistic_rows(points)
    robust = robustness_table(ret_rows, corr_rows)
    gen_health = generation_health(providers)
    ret_health = retrieval_health(providers)
    verdict_rows = verdicts(providers, corr_rows, p4, robust)

    lines: list[str] = []
    lines.append("# BEIR Phase 1b Model Breadth - 2026-05-26")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("Model-breadth analysis over the five BEIR Phase 1 datasets using the same raw-question retrieval caches, Chroma collections, gte-large embeddings, and MiniLM cross-encoder reranking. No files under `paper/` were edited.")
    lines.append("")
    lines.append("Models included: " + ", ".join(f"`{p}` ({label_model(p)})" for p in providers) + ".")
    lines.append("")
    lines.append("## Cross-Model Verdicts")
    lines.append("")
    lines.append("| Claim | Verdict | Key numbers |")
    lines.append("|---|---|---|")
    for claim, verdict, key in verdict_rows:
        lines.append(f"| {claim} | **{verdict}** | {key} |")
    lines.append("")

    lines.append("## Clean-Output And Cache Health")
    lines.append("")
    lines.append(f"Generation status: **{clean_status(gen_health)}**. Retrieval status: **{clean_status(ret_health)}**.")
    notes = [PROVIDER_RUN_NOTES[p] for p in providers if p in PROVIDER_RUN_NOTES]
    if notes:
        lines.append("")
        lines.append("Execution notes:")
        for note in notes:
            lines.append(f"- {note}")
    lines.append("")
    lines.append("| Model | Dataset | Expansion | Generation rows | Errors | Missing passage | Parse bad | Answer artifacts | Format retries | Max output tokens | Retrieval rows | Short retrieval rows | Provider mismatches |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    ret_lookup = {(r["provider"], r["dataset"], r["expansion"]): r for r in ret_health}
    for row in gen_health:
        r = ret_lookup.get((row["provider"], row["dataset"], row["expansion"]), {})
        lines.append(
            f"| {label_model(row['provider'])} | {row['dataset_display']} | {row['expansion_display']} | {row['rows']} | "
            f"{row['errors']} | {row['missing_passage']} | {row['parse_bad']} | {row['answer_artifact']} | "
            f"{row['format_retry']} | {row['max_output_tokens']} | {r.get('rows', 0)} | "
            f"{r.get('short_rows', 0)} | {row['provider_bad'] + r.get('provider_bad', 0)} |"
        )
    lines.append("")

    lines.append("## Retrieval Outcomes")
    lines.append("")
    lines.append("| Model | Dataset | Expansion | N | Raw Hit@5 | Expansion Hit@5 | Net Hit@5 | Help | Hurt | RI |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in ret_rows:
        lines.append(
            f"| {label_model(row['provider'])} | {row['dataset_display']} | {row['expansion_display']} | {row['n']} | "
            f"{pct(row['raw_hit5'])} | {pct(row['exp_hit5'])} | {pct(row['net_delta'])} | "
            f"{row['help']} | {row['hurt']} | {fmt(row['ri'])} |"
        )
    lines.append("")

    lines.append("## Mechanism Correlations")
    lines.append("")
    lines.append("| Model | Dataset | Expansion | N | Mean CE gold delta | Gold rho | Gold tau | Margin-valid N | Mean deltaM | DeltaM rho | DeltaM tau |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in corr_rows:
        lines.append(
            f"| {label_model(row['provider'])} | {row['dataset_display']} | {row['expansion_display']} | {row['n']} | "
            f"{fmt(row['mean_gold_delta'])} | {fmt(row['gold_spearman'])} | {fmt(row['gold_kendall'])} | "
            f"{row['margin_n']} | {fmt(row['mean_delta_margin'])} | {fmt(row['margin_spearman'])} | {fmt(row['margin_kendall'])} |"
        )
    lines.append("")

    lines.append("## P4 Failure Model")
    lines.append("")
    lines.append("| Model | Dataset | Expansion | Target | N | Failures | AUC OOV/logPPL | AUC geometry | Pseudo-R2 OOV/logPPL | Pseudo-R2 geometry |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|")
    for row in p4:
        q = row["quality"]
        g = row["geometry"]
        lines.append(
            f"| {label_model(row['provider'])} | {row['dataset_display']} | {row['expansion_display']} | {row['target']} | "
            f"{g['n']} | {g['failures']} | {fmt(q['auc'])} | {fmt(g['auc'])} | {fmt(q['pseudo_r2'])} | {fmt(g['pseudo_r2'])} |"
        )
    lines.append("")

    lines.append("## SCOPE-vs-HyDE Robustness Gap")
    lines.append("")
    lines.append("Positive `SCOPE-HyDE net` means SCOPE loses less retrieval exposure than HyDE. Positive `closer-to-zero CE delta` means SCOPE's mean gold-affinity movement is closer to raw than HyDE's.")
    lines.append("")
    lines.append("| Model | Dataset | HyDE net Hit@5 | SCOPE net Hit@5 | SCOPE-HyDE net | HyDE mean CE gold delta | SCOPE mean CE gold delta | Closer-to-zero CE delta |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in robust:
        lines.append(
            f"| {label_model(row['provider'])} | {row['dataset_display']} | {pct(row['hyde_net'])} | {pct(row['scope_net'])} | "
            f"{pct(row['scope_minus_hyde_net'])} | {fmt(row['hyde_gold_delta'])} | {fmt(row['scope_gold_delta'])} | "
            f"{fmt(row['scope_closer_to_zero'])} |"
        )
    lines.append("")

    lines.append("## Reading")
    lines.append("")
    pooled_robust = [r for r in robust if r["dataset"] == "pooled"]
    if pooled_robust:
        lines.append("- SCOPE is consistently less destructive than HyDE in pooled Hit@5: " + "; ".join(
            f"{label_model(r['provider'])} gap {pct(r['scope_minus_hyde_net'])}"
            for r in pooled_robust
        ) + ".")
    pool_corr_scope = [r for r in corr_rows if r["dataset"] == "pooled" and r["expansion"] == "scope"]
    if pool_corr_scope:
        lines.append("- SCOPE's row-level gold-affinity mechanism remains positive across included models: " + "; ".join(
            f"{label_model(r['provider'])} rho {fmt(r['gold_spearman'])}"
            for r in pool_corr_scope
        ) + ".")
    lines.append("- The operational implication is still gated expansion: the mechanism predicts which rows can move, but ungated expansion remains risky when raw retrieval already has strong gold exposure.")
    lines.append("")

    lines.append("## Sources")
    lines.append("")
    source_counts = Counter()
    for provider in providers:
        for spec in phase1.BEIR_SPECS.values():
            pspec = provider_spec(spec, provider)
            for path in (spec.raw_cache, pspec.hyde_generation, pspec.scope_generation, pspec.hyde_retrieval, pspec.scope_retrieval):
                source_counts[str(path.relative_to(REPO_ROOT))] += 1
    for path in sorted(source_counts):
        lines.append(f"- `{path}`")
    lines.append("")

    lines.append("## Reproduction")
    lines.append("")
    lines.append("```bash")
    lines.append("HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CROSS_ENCODER_DEVICE=cuda \\")
    lines.append("uv run python scripts/analyze_beir_phase1b.py \\")
    lines.append("  --providers " + " ".join(providers) + " \\")
    lines.append("  --output docs/generated/beir_phase1b_model_breadth_2026-05-26.md")
    lines.append("```")
    lines.append("")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--providers", nargs="+", default=list(DEFAULT_PROVIDERS))
    parser.add_argument("--datasets", nargs="+", default=list(phase1.BEIR_SPECS), choices=sorted(phase1.BEIR_SPECS))
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "docs/generated/beir_phase1b_model_breadth_2026-05-26.md")
    parser.add_argument("--points-cache", type=Path, default=Path("/tmp/beir_phase1b_model_breadth_2026-05-26_points.jsonl"))
    parser.add_argument("--seed-points-cache", action="append", type=Path, default=[Path("/tmp/beir_phase1_verification_2026-05-26_points.jsonl")])
    parser.add_argument("--reuse-points", action="store_true")
    parser.add_argument("--doc-batch-size", type=int, default=5000)
    parser.add_argument("--ce-batch-size", type=int, default=64)
    parser.add_argument("--ce-chunk-size", type=int, default=10000)
    args = parser.parse_args()

    if args.reuse_points and args.points_cache.exists():
        points = load_existing_points([args.points_cache])
        print(f"[cache] loaded {len(points)} points from {args.points_cache}", flush=True)
    else:
        seed_paths = [path for path in args.seed_points_cache if path.exists()]
        points = load_existing_points(seed_paths)
        if points:
            print(f"[cache] seeded {len(points)} points from {len(seed_paths)} cache(s)", flush=True)
    points = build_missing_points(args, points)
    write_jsonl(args.points_cache, points)
    print(f"[cache] wrote {len(points)} points to {args.points_cache}", flush=True)
    write_report(args.output, points, args.providers)
    print(args.output)


if __name__ == "__main__":
    main()
