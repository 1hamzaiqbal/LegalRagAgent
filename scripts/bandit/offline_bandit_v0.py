#!/usr/bin/env python
"""Offline contextual-bandit replay v0 — rung 1 of the skill-distillation bridge.

Single-turn retrieve-or-not / which-arm decisions evaluated by replaying the
paired 2026-07-02 detail logs (same questions, llm_only + evidence arms, per-row
correctness and token costs). No new LLM calls.

Per cell (reader x task):
  actions   = llm_only + evidence arms (judge / ce / scope, or raw / hyde / scope)
  reward    = is_correct - lam * ktokens          (lam swept)
  policies  = fixed arms | per-question oracle (ceiling, noise-inflated) |
              judge-score gate (1-D threshold, tuned on train) |
              contextual argmax over per-action logistic reward models (numpy)
  protocol  = 50/50 train/test split (seed 0); everything reported on test.

Outputs: docs/generated/offline_bandit_v0_2026-07-02.md (+ frontier PNG).
"""
import json
import os
import random
from math import comb

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOGS = os.path.join(ROOT, "logs")
JDATA = os.path.join(ROOT, "scripts", "judge_pilot", "data")
OUT_MD = os.path.join(ROOT, "docs", "generated", "offline_bandit_v0_2026-07-02.md")
OUT_PNG = os.path.join(ROOT, "docs", "generated", "offline_bandit_v0_frontier.png")

LAMS = [0.0, 0.005, 0.01, 0.02, 0.05]  # reward penalty per k-token

CELLS = {
    "barexam-70b": {
        "llm_only": "eval_llm_only_groq-llama70b_20260702_043852_barexam_detail.jsonl",
        "judge": "eval_rag_simple_groq-llama70b_20260702_042708_barexam_detail.jsonl",
        "ce": "eval_rag_simple_groq-llama70b_20260702_045017_barexam_detail.jsonl",
        "scope": "eval_rag_simple_groq-llama70b_20260702_050224_barexam_detail.jsonl",
        "pools": "pools_test.jsonl",
        "scores": "scores_trained_9b.json",
    },
    "barexam-8b": {
        "llm_only": "eval_llm_only_groq-llama8b_20260702_112000_barexam_detail.jsonl",
        "judge": "eval_rag_simple_groq-llama8b_20260702_112438_barexam_detail.jsonl",
        "ce": "eval_rag_simple_groq-llama8b_20260702_112926_barexam_detail.jsonl",
        "scope": "eval_rag_simple_groq-llama8b_20260702_113418_barexam_detail.jsonl",
        "pools": "pools_test.jsonl",
        "scores": "scores_trained_9b.json",
    },
    "housing-70b": {
        "llm_only": "eval_llm_only_groq-llama70b_20260702_061002_housing_detail.jsonl",
        "judge": "eval_rag_simple_groq-llama70b_20260702_061742_housing_detail.jsonl",
        "ce": "eval_rag_simple_groq-llama70b_20260702_062828_housing_detail.jsonl",
        "scope": "eval_rag_simple_groq-llama70b_20260702_063923_housing_detail.jsonl",
        "pools": "housing_pools_test.jsonl",
        "scores": "housing_scores_trained.json",
    },
    "housing-8b": {
        "llm_only": "eval_llm_only_groq-llama8b_20260702_113909_housing_detail.jsonl",
        "judge": "eval_rag_simple_groq-llama8b_20260702_114232_housing_detail.jsonl",
        "ce": "eval_rag_simple_groq-llama8b_20260702_114856_housing_detail.jsonl",
        "scope": "eval_rag_simple_groq-llama8b_20260702_115444_housing_detail.jsonl",
        "pools": "housing_pools_test.jsonl",
        "scores": "housing_scores_trained.json",
    },
    "medqa-70b": {
        "llm_only": "eval_llm_only_groq-llama70b_20260702_105258_medqa_detail.jsonl",
        "raw": "eval_rag_simple_groq-llama70b_20260702_112426_medqa_detail.jsonl",
        "hyde": "eval_rag_hyde_groq-llama70b_20260702_115437_medqa_detail.jsonl",
        "scope": "eval_snap_hyre_groq-llama70b_20260702_122522_medqa_detail.jsonl",
        "pools": None,
        "scores": None,
    },
}


def load_rows(fname):
    rows = {}
    for line in open(os.path.join(LOGS, fname)):
        r = json.loads(line)
        rows[str(r["idx"])] = {
            "correct": 1.0 if r["is_correct"] else 0.0,
            "ktok": (r.get("input_tokens", 0) + r.get("output_tokens", 0)) / 1000.0,
            "question": r.get("question", ""),
        }
    return rows


def judge_features(pools_file, scores_file):
    """question_idx -> (max, margin, mean) of trained-judge scores over its pool."""
    pools = [json.loads(l) for l in open(os.path.join(JDATA, pools_file))]
    scores = json.load(open(os.path.join(JDATA, scores_file)))
    feats = {}
    for pi, p in enumerate(pools):
        s = sorted((scores[f"({pi}, {ci})"] for ci in range(len(p["candidates"]))), reverse=True)
        feats[str(p["question_idx"])] = (s[0], s[0] - s[1] if len(s) > 1 else 0.0, float(np.mean(s)))
    return feats


def build_cell(cfg):
    arms = [a for a in cfg if a not in ("pools", "scores")]
    data = {a: load_rows(cfg[a]) for a in arms}
    idxs = sorted(set.intersection(*(set(d) for d in data.values())))
    jf = judge_features(cfg["pools"], cfg["scores"]) if cfg["pools"] else None
    X, R, K = [], [], []
    kept = []
    for q in idxs:
        if jf is not None and q not in jf:
            continue
        qtext = data["llm_only"][q]["question"]
        f = [len(qtext) / 1000.0, len(qtext.split()) / 100.0]
        if jf is not None:
            f += list(jf[q])
        X.append(f)
        R.append([data[a][q]["correct"] for a in arms])
        K.append([data[a][q]["ktok"] for a in arms])
        kept.append(q)
    return arms, np.array(X), np.array(R), np.array(K), kept


def standardize(Xtr, X):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    return (X - mu) / sd


def fit_logistic(X, y, iters=3000, lr=0.5, l2=1e-3):
    Xb = np.hstack([X, np.ones((len(X), 1))])
    w = np.zeros(Xb.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xb @ w))
        g = Xb.T @ (p - y) / len(y) + l2 * w
        w -= lr * g
    return w


def predict_logistic(w, X):
    Xb = np.hstack([X, np.ones((len(X), 1))])
    return 1 / (1 + np.exp(-Xb @ w))


def mcnemar(a, b):
    x = int(np.sum((a == 1) & (b == 0)))
    y = int(np.sum((a == 0) & (b == 1)))
    n = x + y
    if n == 0:
        return x, y, 1.0
    p = sum(comb(n, k) for k in range(min(x, y) + 1)) * 2 / 2 ** n
    return x, y, min(p, 1.0)


def eval_cell(name, cfg, report):
    arms, X, R, K, kept = build_cell(cfg)
    n = len(kept)
    rng = random.Random(0)
    order = list(range(n))
    rng.shuffle(order)
    tr, te = order[: n // 2], order[n // 2:]
    Xtr_raw, Xte_raw = X[tr], X[te]
    Xtr = standardize(Xtr_raw, Xtr_raw)
    Xte = standardize(Xtr_raw, Xte_raw)
    has_judge = cfg["pools"] is not None
    li = arms.index("llm_only")

    report.append(f"\n## {name}  (n_test={len(te)}, arms: {', '.join(arms)})\n")
    report.append("| policy | " + " | ".join(f"acc / ktok / reward @ lam={l}" for l in LAMS) + " |")
    report.append("|---|" + "---|" * len(LAMS))

    # per-action correctness models (train once, reuse across lambdas)
    ws = [fit_logistic(Xtr, R[tr][:, ai]) for ai in range(len(arms))]
    phat = np.stack([predict_logistic(w, Xte) for w in ws], axis=1)  # test
    ktr_mean = K[tr].mean(0)  # arm mean cost from train

    curves = {}

    def add_row(label, pick_fn):
        cells_out, pts = [], []
        for lam in LAMS:
            act = pick_fn(lam)
            acc = R[te][np.arange(len(te)), act].mean()
            kt = K[te][np.arange(len(te)), act].mean()
            rew = acc - lam * kt
            cells_out.append(f"{acc:.3f} / {kt:.2f} / {rew:.3f}")
            pts.append((kt, acc))
        report.append(f"| {label} | " + " | ".join(cells_out) + " |")
        curves[label] = pts
        return pick_fn(0.0)

    fixed_acts0 = {}
    for ai, a in enumerate(arms):
        fixed_acts0[a] = add_row(f"fixed:{a}", lambda lam, ai=ai: np.full(len(te), ai))

    # oracle ceiling
    add_row("oracle (ceiling, noise-inflated)",
            lambda lam: np.argmax(R[te] - lam * K[te], axis=1))

    # judge-score gate: threshold on judge_max between llm_only and best train evidence arm
    if has_judge:
        jmax_tr, jmax_te = Xtr_raw[:, 2], Xte_raw[:, 2]
        def gate_pick(lam):
            rew_tr = R[tr] - lam * K[tr]
            ev = int(np.argmax([rew_tr[:, ai].mean() if ai != li else -9e9 for ai in range(len(arms))]))
            best_t, best_v = None, -9e9
            for t in np.quantile(jmax_tr, np.linspace(0, 1, 41)):
                pick = np.where(jmax_tr >= t, ev, li)
                v = rew_tr[np.arange(len(tr)), pick].mean()
                if v > best_v:
                    best_v, best_t = v, t
            return np.where(jmax_te >= best_t, ev, li)
        gate0 = add_row("gate:judge-max (tuned on train)", gate_pick)

    # contextual argmax policy
    def ctx_pick(lam):
        return np.argmax(phat - lam * ktr_mean[None, :], axis=1)
    ctx0 = add_row("contextual (per-action logistic)", ctx_pick)

    # headline tests at lam=0
    best_fixed = max(fixed_acts0, key=lambda a: R[te][:, arms.index(a)].mean())
    bi = arms.index(best_fixed)
    a_ctx = R[te][np.arange(len(te)), ctx0]
    a_fix = R[te][:, bi]
    x, y, p = mcnemar(a_ctx, a_fix)
    report.append(f"\n- lam=0 headline: contextual {a_ctx.mean():.3f} vs best fixed "
                  f"(`{best_fixed}`) {a_fix.mean():.3f} — McNemar b/c={x}/{y}, p={p:.3f}")
    orc0 = np.argmax(R[te], axis=1)
    match = float(np.mean(ctx0 == orc0))
    report.append(f"- contextual action agrees with per-question oracle action on "
                  f"{match:.1%} of test questions (oracle itself noise-inflated)")
    if has_judge:
        a_gate = R[te][np.arange(len(te)), gate0]
        x, y, p = mcnemar(a_gate, a_fix)
        report.append(f"- gate:judge-max {a_gate.mean():.3f} vs best fixed — b/c={x}/{y}, p={p:.3f}")
    return curves, arms


def main():
    report = [
        "# Offline bandit replay v0 (2026-07-02) — single-turn retrieve-or-not / arm choice",
        "",
        "Rung 1 of the skill-distillation bridge: can a *cheap trained policy* allocate",
        "retrieval per-question better than fixed policies, evaluated by offline replay of",
        "the paired 2026-07-02 arms (no new LLM calls)? Reward = correct − λ·k-tokens.",
        "50/50 train/test split, seed 0; all numbers are test-half. Oracle = per-question",
        "argmax over recorded outcomes — a noise-inflated ceiling (argmax over Bernoulli",
        "draws), reported for scale only.",
    ]
    all_curves = {}
    for name, cfg in CELLS.items():
        curves, arms = eval_cell(name, cfg, report)
        all_curves[name] = curves

    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w") as f:
        f.write("\n".join(report) + "\n")
    print(f"wrote {OUT_MD}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(all_curves), figsize=(4.2 * len(all_curves), 4.2))
    for ax, (name, curves) in zip(np.atleast_1d(axes), all_curves.items()):
        for label, pts in curves.items():
            xs, ys = zip(*pts)
            if label.startswith("fixed"):
                ax.scatter(xs[:1], ys[:1], marker="s", s=45, label=label)
            else:
                ax.plot(xs, ys, marker="o", ms=3.5, lw=1.4, label=label)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("mean k-tokens / question")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=5.5, loc="best")
    np.atleast_1d(axes)[0].set_ylabel("test accuracy")
    fig.suptitle("Offline bandit v0: cost-accuracy frontier (λ sweep moves policies leftward)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
