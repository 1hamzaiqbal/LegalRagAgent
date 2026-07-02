#!/usr/bin/env python
"""Rung 2 analysis — internalized allocation policy vs rung-1 baselines.

Consumes alloc_scores_{trained,zeroshot}.jsonl from the EIT run, rebuilds the
rung-1 cells/splits exactly, and evaluates the 9B-score policy
  a(q) = argmax_a  sigmoid(score_qa) - lam * mean_train_cost_a
on the same test halves as rung 1. sigmoid() is an uncalibrated squash of the
Yes/No logit gap (fine at lam=0 where only ranking matters; cost-aware rows
carry a calibration caveat).
"""
import json
import os
import random
import sys
from math import comb, exp

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from offline_bandit_v0 import CELLS, JDATA, LAMS, build_cell, mcnemar  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_MD = os.path.join(ROOT, "docs", "generated", "alloc_rung2_2026-07-02.md")


def load_scores(tag):
    path = os.path.join(JDATA, f"alloc_scores_{tag}.jsonl")
    d = {}
    for line in open(path):
        r = json.loads(line)
        d[(r["cell"], str(r["question_idx"]), r["action"])] = r["score"]
    return d


def sig(x):
    return 1 / (1 + exp(-x))


def main():
    tags = [t for t in ("trained", "zeroshot")
            if os.path.exists(os.path.join(JDATA, f"alloc_scores_{t}.jsonl"))]
    scores = {t: load_scores(t) for t in tags}
    report = [
        "# Rung 2 — internalized allocation policy (9B, EIT job 93770)",
        "",
        "Policy = argmax_a sigmoid(model score) − λ·mean-train-cost. Same test",
        "halves as rung 1 (seed 0); compare against the best fixed arm and the",
        "rung-1 contextual-features policy per cell.",
    ]
    for cell, cfg in CELLS.items():
        arms, X, R, K, kept = build_cell(cfg)
        n = len(kept)
        order = list(range(n))
        random.Random(0).shuffle(order)
        tr, te = order[: n // 2], order[n // 2:]
        ktr_mean = K[tr].mean(0)
        fixed_acc = {a: R[te][:, ai].mean() for ai, a in enumerate(arms)}
        best_fixed = max(fixed_acc, key=fixed_acc.get)
        bi = arms.index(best_fixed)
        report.append(f"\n## {cell}  (n_test={len(te)}; best fixed: "
                      f"`{best_fixed}` {fixed_acc[best_fixed]:.3f}; "
                      f"oracle {R[te].max(1).mean():.3f})\n")
        report.append("| policy | " + " | ".join(f"acc / ktok @ λ={l}" for l in LAMS) + " | λ=0 vs best fixed |")
        report.append("|---|" + "---|" * (len(LAMS) + 1))
        for tag in tags:
            S = np.array([[scores[tag].get((cell, kept[i], a), float("-inf"))
                           for a in arms] for i in te])
            missing = int(np.sum(~np.isfinite(S)))
            P = np.vectorize(sig)(np.clip(S, -30, 30))
            cells_out = []
            for lam in LAMS:
                act = np.argmax(P - lam * ktr_mean[None, :], axis=1)
                acc = R[te][np.arange(len(te)), act].mean()
                kt = K[te][np.arange(len(te)), act].mean()
                cells_out.append(f"{acc:.3f} / {kt:.2f}")
            act0 = np.argmax(P, axis=1)
            a_pol = R[te][np.arange(len(te)), act0]
            x, y, p = mcnemar(a_pol, R[te][:, bi])
            report.append(f"| 9B-{tag} | " + " | ".join(cells_out) +
                          f" | {a_pol.mean():.3f} vs {fixed_acc[best_fixed]:.3f}, "
                          f"b/c={x}/{y}, p={p:.3f} |")
            if missing:
                report.append(f"  - WARNING: {missing} missing scores in {cell}/{tag}")
            # action distribution at lam=0
            dist = {arms[ai]: int(np.sum(act0 == ai)) for ai in range(len(arms))}
            report.append(f"  - 9B-{tag} λ=0 action distribution: {dist}")
    out = "\n".join(report) + "\n"
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    open(OUT_MD, "w").write(out)
    print(out)
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
