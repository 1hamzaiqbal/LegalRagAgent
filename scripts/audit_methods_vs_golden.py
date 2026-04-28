"""Cross-method audit: compare every Tier 3 method against golden_passage.

Question (meeting follow-up 2026-04-27): is golden_passage uniquely worse than
expected, or do other methods also under-perform / over-flip the gold-injected
context? Produces a paired-N transition matrix for each method vs golden_passage,
plus an anchoring-rate analysis showing how often each method's wrong predictions
match the gold-passage-induced wrong prediction.

Outputs to docs/methods_vs_golden_audit_2026-04-27.md.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

LOGS = {
    "rag_simple":         "logs/eval_rag_simple_cluster-vllm_20260425_2020_detail.jsonl",
    "rag_hyde":           "logs/eval_rag_hyde_cluster-vllm_20260425_2240_detail.jsonl",
    "rag_snap_hyde":      "logs/eval_rag_snap_hyde_cluster-vllm_20260425_2226_detail.jsonl",
    "snap_only_in_final": "logs/eval_snap_only_in_final_cluster-vllm_20260426_0154_detail.jsonl",
    "llm_only":           "logs/eval_llm_only_cluster-vllm_20260426_0027_detail.jsonl",
    "golden_passage":     "logs/eval_golden_passage_cluster-vllm_20260426_0224_detail.jsonl",
    "subagent_rag":       "logs/eval_subagent_rag_cluster-vllm_20260425_2234_detail.jsonl",
    "subagent_hybrid":    "logs/eval_subagent_hybrid_cluster-vllm_20260426_0254_detail.jsonl",
}


def index_by_idx(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            idx = r.get("idx")
            if idx is not None:
                out[str(idx)] = r
    return out


def transitions(a: dict[str, dict], b: dict[str, dict]) -> dict:
    """a vs b paired transitions on common idx."""
    common = sorted(set(a) & set(b))
    n = len(common)
    if n == 0:
        return {"n": 0}
    both_r = a_only = b_only = both_w = 0
    for k in common:
        ar = bool(a[k].get("is_correct"))
        br = bool(b[k].get("is_correct"))
        if ar and br: both_r += 1
        elif ar: a_only += 1
        elif br: b_only += 1
        else: both_w += 1
    return {
        "n": n, "both_right": both_r, "a_only": a_only, "b_only": b_only, "both_wrong": both_w,
        "a_acc": (both_r + a_only) / n, "b_acc": (both_r + b_only) / n,
        "delta_b_minus_a": (b_only - a_only) / n,
    }


def anchoring_rate(method: dict[str, dict], golden: dict[str, dict]) -> dict:
    """In cases where BOTH method and golden are wrong, how often do they share
    the same wrong prediction? Higher = method 'anchors' to the same wrong
    answer the gold passage induces."""
    common = sorted(set(method) & set(golden))
    both_wrong = same_wrong = diff_wrong = 0
    for k in common:
        m = method[k]
        g = golden[k]
        if (not bool(m.get("is_correct"))) and (not bool(g.get("is_correct"))):
            both_wrong += 1
            if str(m.get("predicted_answer")) == str(g.get("predicted_answer")):
                same_wrong += 1
            else:
                diff_wrong += 1
    return {
        "both_wrong": both_wrong,
        "same_wrong_pred": same_wrong,
        "diff_wrong_pred": diff_wrong,
        "anchor_rate": same_wrong / both_wrong if both_wrong else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--out", default="docs/methods_vs_golden_audit_2026-04-27.md")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    indexed: dict[str, dict[str, dict]] = {}
    for name, path in LOGS.items():
        p = repo / path
        if not p.exists():
            raise SystemExit(f"missing: {p}")
        indexed[name] = index_by_idx(p)
        print(f"  {name:<22s} N={len(indexed[name])}")

    common_all = sorted(set.intersection(*[set(d) for d in indexed.values()]))
    print(f"Common across all 8 modes: {len(common_all)}")
    golden = indexed["golden_passage"]

    method_results = []
    for name in LOGS:
        if name == "golden_passage":
            continue
        t = transitions(golden, indexed[name])
        anc = anchoring_rate(indexed[name], golden)
        method_results.append((name, t, anc))

    method_results.sort(key=lambda x: -x[1]["delta_b_minus_a"])

    out: list[str] = []
    out.append("# Cross-method audit vs golden_passage — Gemma 4 26B-A4B BarExam Tier 3 N=1195\n")
    out.append("Meeting 2026-04-27 follow-up: is golden_passage uniquely worse than expected, or ")
    out.append("do other methods also flip on the same questions? This pairs each of 7 Tier 3 ")
    out.append("methods against `golden_passage` and reports paired transitions plus anchoring rate.\n")
    out.append(f"\nCommon idx across all 8 modes: **{len(common_all)}**\n")

    out.append("## Per-method transitions vs golden_passage\n")
    out.append("Methods sorted by Δ (method - golden_passage). Positive Δ = method beats golden on this paired N.\n")
    out.append("| Method | N paired | golden EM | method EM | Δ | golden-only right | method-only right |")
    out.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, t, _ in method_results:
        out.append(
            f"| `{name}` | {t['n']} | {t['a_acc']*100:.2f}% | {t['b_acc']*100:.2f}% | "
            f"{t['delta_b_minus_a']*100:+.2f}pp | {t['a_only']} | {t['b_only']} |"
        )
    out.append("")

    out.append("## Anchoring rate: when method AND golden are both wrong, how often do they share the same wrong pred?\n")
    out.append("Higher anchor rate = method's failures resemble the gold-passage-induced failures (suggests both ")
    out.append("are misled by the same surface signal in the question, not just random noise).\n")
    out.append("| Method | both wrong | same wrong pred | different wrong pred | anchor rate |")
    out.append("|---|---:|---:|---:|---:|")
    for name, _, anc in method_results:
        out.append(
            f"| `{name}` | {anc['both_wrong']} | {anc['same_wrong_pred']} | "
            f"{anc['diff_wrong_pred']} | {anc['anchor_rate']*100:.1f}% |"
        )
    out.append("")

    out.append("## Symmetric-flip check: does every method flip ~equally with golden?\n")
    out.append("If golden_passage is *uniquely* misleading vs llm_only, it should over-flip llm_only-correct ")
    out.append("answers (high golden-only-right - method-only-right gap). If the flip is symmetric across ALL ")
    out.append("methods, golden is not uniquely bad — it's just one more noisy oracle.\n")
    out.append("| Method | flips that hurt method (golden right, method wrong) | flips that hurt golden (method right, golden wrong) | net (helps method) |")
    out.append("|---|---:|---:|---:|")
    for name, t, _ in method_results:
        out.append(
            f"| `{name}` | {t['a_only']} | {t['b_only']} | {t['b_only'] - t['a_only']:+d} |"
        )
    out.append("")

    out.append("## Reading guide\n")
    out.append("- A **negative** Δ means the method does WORSE than golden_passage on the paired N (these are weak methods).\n")
    out.append("- A **positive** Δ means the method beats golden_passage. The size of `method-only right` shows how ")
    out.append("often the method recovers from the gold-passage anchor.\n")
    out.append("- Anchoring rate of ~25% would be chance (4 MC options); rates well above 25% mean the gold passage ")
    out.append("steers BOTH methods toward the same wrong distractor — i.e., the question contains a real lure.\n")

    p_out = repo / args.out
    p_out.parent.mkdir(parents=True, exist_ok=True)
    p_out.write_text("\n".join(out))
    print(f"\nWrote: {p_out}")
    for name, t, anc in method_results:
        print(f"  {name:<22s} Δ={t['delta_b_minus_a']*100:+.2f}pp  anchor={anc['anchor_rate']*100:.1f}%  "
              f"(method-only:{t['b_only']:>3d}, golden-only:{t['a_only']:>3d})")


if __name__ == "__main__":
    main()
