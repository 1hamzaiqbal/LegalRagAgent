#!/usr/bin/env python3
"""Case-study analysis: find interesting per-question patterns across runs.

Reads all detail JSONL files, cross-references by question idx, and produces
a markdown report with concrete examples for:
  1. RAG-helps  – RAG mode flips a wrong answer to correct
  2. RAG-hurts  – RAG mode flips a correct answer to wrong
  3. Hard Qs    – every run gets the question wrong
  4. Small-vs-Large disagreement

Usage:
    uv run python eval/case_studies.py
"""
from __future__ import annotations

import json
import os
import glob
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
OUTPUT_PATH = LOG_DIR / "case_studies.md"
DATASET_FOCUS = "barexam"
MIN_RUN_SIZE = 100

# Model-size classification – providers mapped to a tier
LARGE_PROVIDERS = {
    "groq-llama70b", "deepseek", "gpt-5.4-mini", "groq-gpt120b",
    "groq-maverick",
}
SMALL_PROVIDERS = {
    "groq-scout", "cerebras", "gpt-5.4-nano", "gpt-4.1-nano",
    "or-qwen3-8b", "or-phi4", "gemma-4b", "gemma", "or-mistral",
    "groq-qwen", "or-gemma4b",
}

RAG_MODES = {
    "rag_simple", "rag_hyde", "rag_snap_hyde", "rag_rewrite",
    "rag_multi_hyde", "rag_hyde_arb", "rag_arbitration",
    "rag_devil_hyde", "rag_top2_hyde",
    "confidence_gated", "decompose_rag", "ce_threshold",
    "ce_threshold_k3", "conf_ce_threshold", "full_pipeline",
    "self_verify", "snap_debate", "double_snap", "snap_hyde_aspect",
}
BASELINE_MODES = {"llm_only"}

MAX_EXAMPLES = 8  # cap examples per section


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_detail_files() -> list[dict]:
    """Load every record from every detail JSONL file."""
    all_records: list[dict] = []
    pattern = str(LOG_DIR / "eval_*_detail.jsonl")
    for path in sorted(glob.glob(pattern)):
        records: list[dict] = []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        all_records.extend(records)
    return all_records


def run_key(rec: dict) -> str:
    """Unique key for a run (mode + provider + dataset)."""
    return f"{rec.get('mode', '?')}|{rec.get('provider', '?')}|{rec.get('dataset', '?')}"


def group_into_runs(records: list[dict]) -> dict[str, list[dict]]:
    """Group records into runs.  Returns {run_key: [records]}."""
    runs: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        runs[run_key(rec)].append(rec)
    return dict(runs)


def is_barexam(rec: dict) -> bool:
    ds = rec.get("dataset", "")
    if ds == DATASET_FOCUS:
        return True
    # Older files without dataset field – detect from idx pattern
    idx = str(rec.get("idx", ""))
    return idx.startswith("mbe_")


def truncate(text: str, maxlen: int = 300) -> str:
    if not text:
        return ""
    text = str(text).replace("\n", " ").strip()
    if len(text) > maxlen:
        return text[:maxlen] + "…"
    return text


def provider_tier(provider: str) -> str | None:
    if provider in LARGE_PROVIDERS:
        return "large"
    if provider in SMALL_PROVIDERS:
        return "small"
    return None


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------
def build_question_index(
    runs: dict[str, list[dict]],
) -> dict[str, dict[str, dict]]:
    """Build idx -> {run_key -> record} mapping for barexam Qs."""
    q_index: dict[str, dict[str, dict]] = defaultdict(dict)
    for rk, recs in runs.items():
        for rec in recs:
            idx = str(rec.get("idx", ""))
            if idx:
                q_index[idx][rk] = rec
    return dict(q_index)


def classify_runs(runs: dict[str, list[dict]]) -> tuple[
    dict[str, list[dict]],  # baseline runs
    dict[str, list[dict]],  # RAG runs
]:
    baselines, rags = {}, {}
    for rk, recs in runs.items():
        mode = recs[0].get("mode", "")
        if mode in BASELINE_MODES:
            baselines[rk] = recs
        elif mode in RAG_MODES:
            rags[rk] = recs
    return baselines, rags


def find_rag_helps(
    q_index: dict[str, dict[str, dict]],
    baseline_keys: set[str],
    rag_keys: set[str],
    run_meta: dict[str, dict],
) -> list[dict]:
    """Find questions where a baseline was wrong but a RAG run was correct."""
    examples = []
    for idx, runs_for_q in q_index.items():
        for bk in baseline_keys & runs_for_q.keys():
            brec = runs_for_q[bk]
            if brec.get("is_correct"):
                continue
            b_provider = brec.get("provider", "")
            # Look for a RAG run with the same provider that flipped to correct
            for rk in rag_keys & runs_for_q.keys():
                rrec = runs_for_q[rk]
                if rrec.get("provider", "") == b_provider and rrec.get("is_correct"):
                    examples.append({
                        "idx": idx,
                        "question": brec.get("question", ""),
                        "gold": brec.get("correct_answer", ""),
                        "choices": brec.get("choices") or rrec.get("choices"),
                        "baseline_mode": brec.get("mode"),
                        "baseline_pred": brec.get("predicted_answer"),
                        "rag_mode": rrec.get("mode"),
                        "rag_pred": rrec.get("predicted_answer"),
                        "provider": b_provider,
                        "snap_answer": rrec.get("snap_answer"),
                        "hyde_passage": rrec.get("hyde_passage"),
                        "evidence_store": rrec.get("evidence_store"),
                        "subject": brec.get("subject"),
                    })
                    break  # one example per (question, baseline) is enough
    # Deduplicate by idx, keep first per idx
    seen = set()
    deduped = []
    for ex in examples:
        if ex["idx"] not in seen:
            seen.add(ex["idx"])
            deduped.append(ex)
    return deduped


def find_rag_hurts(
    q_index: dict[str, dict[str, dict]],
    baseline_keys: set[str],
    rag_keys: set[str],
    run_meta: dict[str, dict],
) -> list[dict]:
    """Find questions where a baseline was correct but a RAG run flipped to wrong."""
    examples = []
    for idx, runs_for_q in q_index.items():
        for bk in baseline_keys & runs_for_q.keys():
            brec = runs_for_q[bk]
            if not brec.get("is_correct"):
                continue
            b_provider = brec.get("provider", "")
            for rk in rag_keys & runs_for_q.keys():
                rrec = runs_for_q[rk]
                if rrec.get("provider", "") == b_provider and not rrec.get("is_correct"):
                    examples.append({
                        "idx": idx,
                        "question": brec.get("question", ""),
                        "gold": brec.get("correct_answer", ""),
                        "choices": brec.get("choices") or rrec.get("choices"),
                        "baseline_mode": brec.get("mode"),
                        "baseline_pred": brec.get("predicted_answer"),
                        "rag_mode": rrec.get("mode"),
                        "rag_pred": rrec.get("predicted_answer"),
                        "provider": b_provider,
                        "snap_answer": rrec.get("snap_answer"),
                        "hyde_passage": rrec.get("hyde_passage"),
                        "evidence_store": rrec.get("evidence_store"),
                        "subject": brec.get("subject"),
                    })
                    break
    seen = set()
    deduped = []
    for ex in examples:
        if ex["idx"] not in seen:
            seen.add(ex["idx"])
            deduped.append(ex)
    return deduped


def find_hard_questions(
    q_index: dict[str, dict[str, dict]],
    valid_keys: set[str],
) -> list[dict]:
    """Questions that every qualifying run gets wrong."""
    hard = []
    for idx, runs_for_q in q_index.items():
        active = {k: v for k, v in runs_for_q.items() if k in valid_keys}
        if len(active) < 3:  # need at least 3 runs to be meaningful
            continue
        if all(not r.get("is_correct") for r in active.values()):
            sample = list(active.values())[0]
            predictions = {
                k: {
                    "mode": v.get("mode"),
                    "provider": v.get("provider"),
                    "pred": v.get("predicted_answer"),
                }
                for k, v in active.items()
            }
            hard.append({
                "idx": idx,
                "question": sample.get("question", ""),
                "gold": sample.get("correct_answer", ""),
                "choices": sample.get("choices"),
                "subject": sample.get("subject"),
                "n_runs_wrong": len(active),
                "predictions": predictions,
            })
    # Sort by most runs wrong
    hard.sort(key=lambda x: -x["n_runs_wrong"])
    return hard


def find_size_disagreements(
    q_index: dict[str, dict[str, dict]],
    valid_keys: set[str],
    run_meta: dict[str, dict],
) -> list[dict]:
    """Questions where small models disagree with large models."""
    examples = []
    for idx, runs_for_q in q_index.items():
        active = {k: v for k, v in runs_for_q.items() if k in valid_keys}
        small_correct = []
        small_wrong = []
        large_correct = []
        large_wrong = []
        for rk, rec in active.items():
            prov = rec.get("provider", "")
            tier = provider_tier(prov)
            if tier == "small":
                (small_correct if rec.get("is_correct") else small_wrong).append(rec)
            elif tier == "large":
                (large_correct if rec.get("is_correct") else large_wrong).append(rec)

        # Case A: large gets it right, small gets it wrong (majority)
        if large_correct and small_wrong and not large_wrong:
            sample_large = large_correct[0]
            sample_small = small_wrong[0]
            examples.append({
                "idx": idx,
                "question": sample_large.get("question", ""),
                "gold": sample_large.get("correct_answer", ""),
                "choices": sample_large.get("choices") or sample_small.get("choices"),
                "subject": sample_large.get("subject"),
                "direction": "large✓ small✗",
                "large_runs": [
                    {"provider": r.get("provider"), "mode": r.get("mode"),
                     "pred": r.get("predicted_answer"), "correct": r.get("is_correct")}
                    for r in large_correct
                ],
                "small_runs": [
                    {"provider": r.get("provider"), "mode": r.get("mode"),
                     "pred": r.get("predicted_answer"), "correct": r.get("is_correct")}
                    for r in small_wrong
                ],
            })
        # Case B: small gets it right, large gets it wrong
        elif small_correct and large_wrong and not small_correct == []:
            if large_wrong and not large_correct:
                sample_small = small_correct[0]
                sample_large = large_wrong[0]
                examples.append({
                    "idx": idx,
                    "question": sample_small.get("question", ""),
                    "gold": sample_small.get("correct_answer", ""),
                    "choices": sample_small.get("choices") or sample_large.get("choices"),
                    "subject": sample_small.get("subject"),
                    "direction": "small✓ large✗",
                    "large_runs": [
                        {"provider": r.get("provider"), "mode": r.get("mode"),
                         "pred": r.get("predicted_answer"), "correct": r.get("is_correct")}
                        for r in large_wrong
                    ],
                    "small_runs": [
                        {"provider": r.get("provider"), "mode": r.get("mode"),
                         "pred": r.get("predicted_answer"), "correct": r.get("is_correct")}
                        for r in small_correct
                    ],
                })
    return examples


# ---------------------------------------------------------------------------
# Markdown report generation
# ---------------------------------------------------------------------------
def fmt_choices(choices: dict | None) -> str:
    if not choices:
        return ""
    lines = []
    for k in sorted(choices.keys()):
        lines.append(f"  - **({k})** {truncate(choices[k], 200)}")
    return "\n".join(lines)


def fmt_evidence(evidence_store: list[dict] | None, limit: int = 3) -> str:
    if not evidence_store:
        return "_No retrieved passages available._"
    lines = []
    for i, ev in enumerate(evidence_store[:limit]):
        score = ev.get("cross_encoder_score")
        score_str = f" (CE={score:.3f})" if score is not None else ""
        src = ev.get("source", "")
        src_str = f" [src: {src}]" if src else ""
        text = truncate(ev.get("text", ""), 250)
        lines.append(f"  {i+1}. {text}{score_str}{src_str}")
    return "\n".join(lines)


def write_report(
    rag_helps: list[dict],
    rag_hurts: list[dict],
    hard_qs: list[dict],
    size_disagree: list[dict],
    run_summary: list[dict],
) -> str:
    """Build the markdown report string."""
    lines: list[str] = []

    lines.append("# Case Study Analysis – Per-Question Patterns\n")
    lines.append(f"_Auto-generated by `eval/case_studies.py` · "
                 f"Dataset: **{DATASET_FOCUS}** · "
                 f"Min run size: {MIN_RUN_SIZE}_\n")

    # ── Summary table ──
    lines.append("## Runs Included\n")
    lines.append("| # | Mode | Provider | N | Acc |")
    lines.append("|---|------|----------|---|-----|")
    for i, rs in enumerate(run_summary, 1):
        acc = rs["correct"] / rs["total"] * 100 if rs["total"] else 0
        lines.append(f"| {i} | {rs['mode']} | {rs['provider']} | {rs['total']} | {acc:.1f}% |")
    lines.append("")

    # ── Stats overview ──
    lines.append("## Pattern Counts\n")
    lines.append(f"| Pattern | Count |")
    lines.append(f"|---------|-------|")
    lines.append(f"| RAG helps (flips wrong→correct) | {len(rag_helps)} |")
    lines.append(f"| RAG hurts (flips correct→wrong) | {len(rag_hurts)} |")
    lines.append(f"| Hard questions (all runs wrong) | {len(hard_qs)} |")
    lines.append(f"| Small vs Large disagreements | {len(size_disagree)} |")
    lines.append("")

    # ── Section 1: RAG helps ──
    lines.append("---\n## 1. RAG Helps – Wrong → Correct\n")
    lines.append(f"_Found **{len(rag_helps)}** questions where RAG flipped a baseline wrong "
                 f"answer to correct (same provider). Showing up to {MAX_EXAMPLES} examples._\n")
    for i, ex in enumerate(rag_helps[:MAX_EXAMPLES], 1):
        lines.append(f"### 1.{i}  Question `{ex['idx']}` ({ex.get('subject', '?')})\n")
        lines.append(f"> {truncate(ex['question'], 500)}\n")
        if ex.get("choices"):
            lines.append(f"**Choices:**\n{fmt_choices(ex['choices'])}\n")
        lines.append(f"**Gold answer:** {ex['gold']}\n")
        lines.append(f"| Run | Prediction | Correct? |")
        lines.append(f"|-----|------------|----------|")
        lines.append(f"| {ex['baseline_mode']} / {ex['provider']} | **{ex['baseline_pred']}** | ✗ |")
        lines.append(f"| {ex['rag_mode']} / {ex['provider']} | **{ex['rag_pred']}** | ✓ |")
        lines.append("")
        if ex.get("snap_answer"):
            lines.append(f"**Snap answer (quick RAG reasoning):**\n> {truncate(ex['snap_answer'], 400)}\n")
        if ex.get("hyde_passage"):
            lines.append(f"**HyDE passage:**\n> {truncate(ex['hyde_passage'], 400)}\n")
        lines.append(f"**Retrieved passages:**\n{fmt_evidence(ex.get('evidence_store'))}\n")

    # ── Section 2: RAG hurts ──
    lines.append("---\n## 2. RAG Hurts – Correct → Wrong\n")
    lines.append(f"_Found **{len(rag_hurts)}** questions where RAG flipped a baseline correct "
                 f"answer to wrong (same provider). Showing up to {MAX_EXAMPLES} examples._\n")
    for i, ex in enumerate(rag_hurts[:MAX_EXAMPLES], 1):
        lines.append(f"### 2.{i}  Question `{ex['idx']}` ({ex.get('subject', '?')})\n")
        lines.append(f"> {truncate(ex['question'], 500)}\n")
        if ex.get("choices"):
            lines.append(f"**Choices:**\n{fmt_choices(ex['choices'])}\n")
        lines.append(f"**Gold answer:** {ex['gold']}\n")
        lines.append(f"| Run | Prediction | Correct? |")
        lines.append(f"|-----|------------|----------|")
        lines.append(f"| {ex['baseline_mode']} / {ex['provider']} | **{ex['baseline_pred']}** | ✓ |")
        lines.append(f"| {ex['rag_mode']} / {ex['provider']} | **{ex['rag_pred']}** | ✗ |")
        lines.append("")
        if ex.get("snap_answer"):
            lines.append(f"**Snap answer (quick RAG reasoning):**\n> {truncate(ex['snap_answer'], 400)}\n")
        if ex.get("hyde_passage"):
            lines.append(f"**HyDE passage:**\n> {truncate(ex['hyde_passage'], 400)}\n")
        lines.append(f"**Retrieved passages:**\n{fmt_evidence(ex.get('evidence_store'))}\n")

    # ── Section 3: Hard questions ──
    lines.append("---\n## 3. Hard Questions – All Runs Wrong\n")
    lines.append(f"_Found **{len(hard_qs)}** questions that every qualifying run got wrong. "
                 f"Showing up to {MAX_EXAMPLES} examples._\n")
    for i, ex in enumerate(hard_qs[:MAX_EXAMPLES], 1):
        lines.append(f"### 3.{i}  Question `{ex['idx']}` ({ex.get('subject', '?')}) "
                     f"— {ex['n_runs_wrong']} runs wrong\n")
        lines.append(f"> {truncate(ex['question'], 500)}\n")
        if ex.get("choices"):
            lines.append(f"**Choices:**\n{fmt_choices(ex['choices'])}\n")
        lines.append(f"**Gold answer:** {ex['gold']}\n")
        lines.append(f"| Mode | Provider | Prediction |")
        lines.append(f"|------|----------|------------|")
        for pk, pv in list(ex["predictions"].items())[:12]:
            lines.append(f"| {pv['mode']} | {pv['provider']} | {pv['pred']} |")
        lines.append("")

    # ── Section 4: Small vs Large ──
    lines.append("---\n## 4. Small vs Large Model Disagreements\n")
    # Split into two sub-sections
    large_wins = [e for e in size_disagree if e["direction"] == "large✓ small✗"]
    small_wins = [e for e in size_disagree if e["direction"] == "small✓ large✗"]
    lines.append(f"_Large correct / Small wrong: **{len(large_wins)}** · "
                 f"Small correct / Large wrong: **{len(small_wins)}**_\n")

    lines.append(f"### 4a. Large Model Correct, Small Model Wrong "
                 f"(showing {min(MAX_EXAMPLES, len(large_wins))})\n")
    for i, ex in enumerate(large_wins[:MAX_EXAMPLES], 1):
        lines.append(f"#### 4a.{i}  Question `{ex['idx']}` ({ex.get('subject', '?')})\n")
        lines.append(f"> {truncate(ex['question'], 400)}\n")
        if ex.get("choices"):
            lines.append(f"**Choices:**\n{fmt_choices(ex['choices'])}\n")
        lines.append(f"**Gold answer:** {ex['gold']}\n")
        lines.append(f"| Tier | Provider | Mode | Pred | ✓/✗ |")
        lines.append(f"|------|----------|------|------|-----|")
        for r in ex["large_runs"][:4]:
            lines.append(f"| Large | {r['provider']} | {r['mode']} | {r['pred']} | ✓ |")
        for r in ex["small_runs"][:4]:
            lines.append(f"| Small | {r['provider']} | {r['mode']} | {r['pred']} | ✗ |")
        lines.append("")

    lines.append(f"### 4b. Small Model Correct, Large Model Wrong "
                 f"(showing {min(MAX_EXAMPLES, len(small_wins))})\n")
    for i, ex in enumerate(small_wins[:MAX_EXAMPLES], 1):
        lines.append(f"#### 4b.{i}  Question `{ex['idx']}` ({ex.get('subject', '?')})\n")
        lines.append(f"> {truncate(ex['question'], 400)}\n")
        if ex.get("choices"):
            lines.append(f"**Choices:**\n{fmt_choices(ex['choices'])}\n")
        lines.append(f"**Gold answer:** {ex['gold']}\n")
        lines.append(f"| Tier | Provider | Mode | Pred | ✓/✗ |")
        lines.append(f"|------|----------|------|------|-----|")
        for r in ex["large_runs"][:4]:
            lines.append(f"| Large | {r['provider']} | {r['mode']} | {r['pred']} | ✗ |")
        for r in ex["small_runs"][:4]:
            lines.append(f"| Small | {r['provider']} | {r['mode']} | {r['pred']} | ✓ |")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading detail files …")
    all_records = load_detail_files()
    print(f"  → {len(all_records)} total records across all files")

    # Filter to barexam
    barexam_records = [r for r in all_records if is_barexam(r)]
    print(f"  → {len(barexam_records)} barexam records")

    # Group into runs & filter by MIN_RUN_SIZE
    runs_raw = group_into_runs(barexam_records)
    runs = {k: v for k, v in runs_raw.items() if len(v) >= MIN_RUN_SIZE}
    print(f"  → {len(runs)} runs with N≥{MIN_RUN_SIZE} (dropped {len(runs_raw) - len(runs)})")

    if not runs:
        print("No qualifying runs found.  Exiting.")
        return

    # Summarise runs
    run_summary = []
    for rk, recs in sorted(runs.items()):
        mode = recs[0].get("mode", "?")
        provider = recs[0].get("provider", "?")
        total = len(recs)
        correct = sum(1 for r in recs if r.get("is_correct"))
        run_summary.append({"mode": mode, "provider": provider,
                            "total": total, "correct": correct})

    # Build question index
    q_index = build_question_index(runs)
    print(f"  → {len(q_index)} unique questions across qualifying runs")

    # Classify runs
    baseline_runs, rag_runs = classify_runs(runs)
    baseline_keys = set(baseline_runs.keys())
    rag_keys = set(rag_runs.keys())
    all_keys = set(runs.keys())

    run_meta = {rk: {"mode": recs[0].get("mode"), "provider": recs[0].get("provider")}
                for rk, recs in runs.items()}

    # ── Analysis ──
    print("\nAnalysing patterns …")

    rag_helps = find_rag_helps(q_index, baseline_keys, rag_keys, run_meta)
    print(f"  RAG helps:  {len(rag_helps)} questions")

    rag_hurts = find_rag_hurts(q_index, baseline_keys, rag_keys, run_meta)
    print(f"  RAG hurts:  {len(rag_hurts)} questions")

    hard_qs = find_hard_questions(q_index, all_keys)
    print(f"  Hard Qs:    {len(hard_qs)} questions")

    size_disagree = find_size_disagreements(q_index, all_keys, run_meta)
    print(f"  Size disagree: {len(size_disagree)} questions")

    # ── Write report ──
    report = write_report(rag_helps, rag_hurts, hard_qs, size_disagree, run_summary)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(report, encoding="utf-8")
    print(f"\n✅  Report written to {OUTPUT_PATH}  ({len(report)} chars)")


if __name__ == "__main__":
    main()
