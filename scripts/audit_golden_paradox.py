"""Audit the BarExam Gemma 4 26B golden_passage paradox.

Meeting 2026-04-27 ask: golden_passage 78.66% < llm_only 79.75% < rag_snap_hyde 81.17%.
A true gold-passage condition should be near-ceiling. Pair the 4 cited Tier 3 logs
by question idx and compute paired transitions to find where the lift comes from.

Outputs:
- transition matrix per pair (llm_only/golden, golden/rag_simple, golden/rag_snap_hyde)
- list of golden_passage failures where llm_only got it right (the paradox cases)
- list of rag_snap_hyde wins where golden_passage failed (lift mechanism)

Default writes a markdown report to docs/golden_paradox_audit_2026-04-27.md.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterator

DEFAULT_LOGS = {
    "llm_only": "logs/eval_llm_only_cluster-vllm_20260426_0027_detail.jsonl",
    "golden_passage": "logs/eval_golden_passage_cluster-vllm_20260426_0224_detail.jsonl",
    "rag_simple": "logs/eval_rag_simple_cluster-vllm_20260425_2020_detail.jsonl",
    "rag_snap_hyde": "logs/eval_rag_snap_hyde_cluster-vllm_20260425_2226_detail.jsonl",
}


def load_jsonl(path: Path) -> Iterator[dict]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def index_by_idx(path: Path) -> dict[str, dict]:
    by_idx: dict[str, dict] = {}
    for rec in load_jsonl(path):
        idx = rec.get("idx")
        if idx is None:
            continue
        by_idx[str(idx)] = rec
    return by_idx


def transitions(a: dict[str, dict], b: dict[str, dict]) -> dict:
    """McNemar-style 2x2 + accuracy on the paired subset."""
    common = sorted(set(a) & set(b))
    both_right = a_right_b_wrong = a_wrong_b_right = both_wrong = 0
    for k in common:
        ar = bool(a[k].get("is_correct"))
        br = bool(b[k].get("is_correct"))
        if ar and br:
            both_right += 1
        elif ar and not br:
            a_right_b_wrong += 1
        elif not ar and br:
            a_wrong_b_right += 1
        else:
            both_wrong += 1
    n = len(common)
    return {
        "n_paired": n,
        "both_right": both_right,
        "a_right_b_wrong": a_right_b_wrong,
        "a_wrong_b_right": a_wrong_b_right,
        "both_wrong": both_wrong,
        "a_acc": (both_right + a_right_b_wrong) / n if n else 0.0,
        "b_acc": (both_right + a_wrong_b_right) / n if n else 0.0,
        "delta_b_minus_a": (a_wrong_b_right - a_right_b_wrong) / n if n else 0.0,
    }


def fmt_pair_table(name_a: str, name_b: str, t: dict) -> str:
    return (
        f"### {name_a} vs {name_b}\n\n"
        f"- Paired N: {t['n_paired']}\n"
        f"- {name_a} acc: {t['a_acc']*100:.2f}% | {name_b} acc: {t['b_acc']*100:.2f}% | "
        f"Δ ({name_b} - {name_a}): {t['delta_b_minus_a']*100:+.2f}pp\n\n"
        f"|  | {name_b} right | {name_b} wrong |\n"
        f"|---|---:|---:|\n"
        f"| **{name_a} right** | {t['both_right']} | {t['a_right_b_wrong']} |\n"
        f"| **{name_a} wrong** | {t['a_wrong_b_right']} | {t['both_wrong']} |\n"
    )


def truncate(text: str, n: int = 400) -> str:
    text = (text or "").strip().replace("\n", " ")
    if len(text) <= n:
        return text
    return text[: n - 3] + "..."


def render_failure_cases(label: str, cases: list[dict], limit: int = 20) -> str:
    """Render up to `limit` paired failure cases for manual inspection."""
    out: list[str] = [f"### {label}", ""]
    for i, case in enumerate(cases[:limit], start=1):
        out.append(
            f"**{i}. idx={case['idx']}** "
            f"(correct={case['correct']!r}, golden→{case['golden_pred']!r}, "
            f"llm_only→{case['llm_only_pred']!r}, snap_hyde→{case['snap_hyde_pred']!r})"
        )
        out.append("")
        out.append(f"- **Q**: {truncate(case['question'])}")
        if case.get("gold_passage"):
            out.append(f"- **Gold passage** ({len(case['gold_passage'])} chars): {truncate(case['gold_passage'])}")
        else:
            out.append("- **Gold passage**: (empty)")
        out.append(f"- **gold_idx**: `{case.get('gold_idx', '')}` | **gold_retrieved (snap_hyde)**: {case.get('snap_hyde_gold_retrieved')}")
        out.append("")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".", help="Repo root (default: cwd)")
    parser.add_argument("--out", default="docs/golden_paradox_audit_2026-04-27.md", help="Output markdown path")
    parser.add_argument("--limit", type=int, default=20, help="Max failure cases per category")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    logs = {name: repo / path for name, path in DEFAULT_LOGS.items()}
    for name, path in logs.items():
        if not path.exists():
            raise SystemExit(f"missing log for {name}: {path}")

    print(f"Loading 4 detail logs from {repo}...")
    indexed = {name: index_by_idx(path) for name, path in logs.items()}
    sizes = {name: len(d) for name, d in indexed.items()}
    print(f"Sizes: {sizes}")

    common = sorted(set.intersection(*[set(d) for d in indexed.values()]))
    print(f"Common idx across all 4: {len(common)}")

    pair_results = {
        ("llm_only", "golden_passage"): transitions(indexed["llm_only"], indexed["golden_passage"]),
        ("rag_simple", "golden_passage"): transitions(indexed["rag_simple"], indexed["golden_passage"]),
        ("golden_passage", "rag_snap_hyde"): transitions(indexed["golden_passage"], indexed["rag_snap_hyde"]),
        ("llm_only", "rag_snap_hyde"): transitions(indexed["llm_only"], indexed["rag_snap_hyde"]),
        ("rag_simple", "rag_snap_hyde"): transitions(indexed["rag_simple"], indexed["rag_snap_hyde"]),
    }

    paradox_cases: list[dict] = []
    snap_hyde_lift_cases: list[dict] = []

    for idx in common:
        ll = indexed["llm_only"][idx]
        gp = indexed["golden_passage"][idx]
        sh = indexed["rag_snap_hyde"][idx]
        if (not bool(gp.get("is_correct"))) and bool(ll.get("is_correct")):
            paradox_cases.append({
                "idx": idx,
                "correct": gp.get("correct_answer"),
                "golden_pred": gp.get("predicted_answer"),
                "llm_only_pred": ll.get("predicted_answer"),
                "snap_hyde_pred": sh.get("predicted_answer"),
                "question": gp.get("question") or ll.get("question") or "",
                "gold_passage": gp.get("gold_passage", ""),
                "gold_idx": gp.get("gold_idx", ""),
                "snap_hyde_gold_retrieved": sh.get("gold_retrieved"),
            })
        if (not bool(gp.get("is_correct"))) and bool(sh.get("is_correct")):
            snap_hyde_lift_cases.append({
                "idx": idx,
                "correct": gp.get("correct_answer"),
                "golden_pred": gp.get("predicted_answer"),
                "llm_only_pred": ll.get("predicted_answer"),
                "snap_hyde_pred": sh.get("predicted_answer"),
                "question": gp.get("question") or "",
                "gold_passage": gp.get("gold_passage", ""),
                "gold_idx": gp.get("gold_idx", ""),
                "snap_hyde_gold_retrieved": sh.get("gold_retrieved"),
            })

    pred_dist_golden = Counter(str(indexed["golden_passage"][k].get("predicted_answer")) for k in common)
    pred_dist_llm = Counter(str(indexed["llm_only"][k].get("predicted_answer")) for k in common)

    out_lines: list[str] = []
    out_lines.append("# Golden-passage paradox audit — Gemma 4 26B-A4B BarExam Tier 3 N=1195\n")
    out_lines.append("Meeting 2026-04-27 #1 ask: explain why `golden_passage` (78.66%) is BELOW `llm_only` (79.75%) ")
    out_lines.append("when both are run on the same model with the same fact pattern, and where the lift to ")
    out_lines.append("`rag_snap_hyde` (81.17%) actually comes from.\n")
    out_lines.append("\nLogs paired:\n")
    for name, path in DEFAULT_LOGS.items():
        out_lines.append(f"- `{name}` → `{path}` (N={sizes[name]})")
    out_lines.append(f"\nCommon idx across all 4 logs: **{len(common)}**\n")

    out_lines.append("## Paired transition tables\n")
    for (a, b), t in pair_results.items():
        out_lines.append(fmt_pair_table(a, b, t))
        out_lines.append("")

    out_lines.append("## Prediction distributions (paired N)\n")
    out_lines.append("| Letter | golden_passage | llm_only |")
    out_lines.append("|---|---:|---:|")
    for letter in sorted(set(pred_dist_golden) | set(pred_dist_llm)):
        out_lines.append(f"| `{letter}` | {pred_dist_golden.get(letter, 0)} | {pred_dist_llm.get(letter, 0)} |")
    out_lines.append("")

    out_lines.append("## The paradox: golden_passage failed while llm_only succeeded\n")
    out_lines.append(f"Total paradox cases: **{len(paradox_cases)}**\n")
    out_lines.append("These are the cases the meeting flagged: gold passage was injected into the prompt, ")
    out_lines.append("but the model did worse than with no context at all. Either the gold passage isn't ")
    out_lines.append("sufficient, the model was distracted/anchored by it, or the gold label is questionable.\n")
    out_lines.append(render_failure_cases("Sample paradox cases", paradox_cases, limit=args.limit))

    out_lines.append("## Lift mechanism: rag_snap_hyde won where golden_passage failed\n")
    out_lines.append(f"Total cases where snap_hyde correct + golden_passage wrong: **{len(snap_hyde_lift_cases)}**\n")
    out_lines.append("These are the cases driving the +2.51pp gap between snap_hyde and golden_passage. ")
    out_lines.append("Inspect to see whether snap_hyde wins by retrieving better evidence than the gold passage, ")
    out_lines.append("by ignoring the noisy gold passage, or by snap-reasoning around it.\n")
    out_lines.append(render_failure_cases("Sample snap_hyde-over-golden cases", snap_hyde_lift_cases, limit=args.limit))

    out_path = repo / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines))
    print(f"\nWrote: {out_path}")
    print(f"Paradox cases: {len(paradox_cases)}")
    print(f"snap_hyde-over-golden lift cases: {len(snap_hyde_lift_cases)}")
    print("\nKey transitions:")
    for (a, b), t in pair_results.items():
        print(f"  {a:>16s} -> {b:<16s} | n={t['n_paired']} | Δ={t['delta_b_minus_a']*100:+.2f}pp | "
              f"a_only={t['a_right_b_wrong']:>3d}, b_only={t['a_wrong_b_right']:>3d}")


if __name__ == "__main__":
    main()
