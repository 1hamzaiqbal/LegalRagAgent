#!/usr/bin/env python
"""Retrieval-side significance tests over all full retrieval caches (queue #2).

Question answered (wiki: [[snap-vs-hyde-ledger]], C7/C11): which cached Hit@5
differences between query-formation arms are statistically significant, per
(dataset, model)? The repo previously had ZERO retrieval-side tests — every
cited Hit@5 delta was a point estimate.

For every discovered pair on the same question set:
  - exact McNemar (binomial) on per-query Hit@5 indicators
  - 95% percentile bootstrap CI (n=2000, seed 42) on the Hit@5 delta

Pairs: snap_hyre vs rag_hyde (same provider), each generated method vs
raw_question, csqe vs raw, raw_scope_pool vs snap_hyre/raw.

Output: docs/generated/retrieval_significance_2026-07-02.md (+ .json)
"""
import glob, json, os, random, re, sys
from math import comb

K = 5
CACHE_DIR = "caches/retrieval/full"
OUT_MD = "docs/generated/retrieval_significance_2026-07-02.md"
OUT_JSON = "docs/generated/retrieval_significance_2026-07-02.json"

PAT = re.compile(
    r"^(?P<dataset>[a-z_]+?)_qfull_seed42(?P<sf>_statefilter)?"
    r"(?:_(?P<provider>groq-[\w-]+|or-[\w.\-]+?))?_(?P<method>raw_question|rag_hyde|snap_hyre|csqe|raw_scope_pool|3scope_raw_pool|golden_neighbors)"
    r"_k(?P<k>\d+)\.jsonl$")


def load_hits(path):
    hits = {}
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            gold = set(r.get("gold_ids") or [])
            if not gold:
                continue
            top = (r.get("retrieved_ids") or [])[:K]
            hits[r["idx"]] = any(g in gold for g in top)
    return hits


def mcnemar(a, b, ids):
    b_ = sum(1 for i in ids if a[i] and not b[i])
    c_ = sum(1 for i in ids if b[i] and not a[i])
    n = b_ + c_
    if n == 0:
        return b_, c_, 1.0
    p = sum(comb(n, k) for k in range(min(b_, c_) + 1)) * 2 / 2 ** n
    return b_, c_, min(p, 1.0)


def bootstrap_ci(a, b, ids, n_boot=2000, seed=42):
    rng = random.Random(seed)
    ids = list(ids)
    deltas = []
    for _ in range(n_boot):
        s = [ids[rng.randrange(len(ids))] for _ in range(len(ids))]
        deltas.append(sum(a[i] for i in s) / len(s) - sum(b[i] for i in s) / len(s))
    deltas.sort()
    return deltas[int(0.025 * n_boot)], deltas[int(0.975 * n_boot)]


def main():
    files = {}
    for p in glob.glob(f"{CACHE_DIR}/*.jsonl"):
        m = PAT.match(os.path.basename(p))
        if not m:
            continue
        d = m.groupdict()
        if d["dataset"].startswith(("beir_", "medqa", "musique", "hotpotqa")):
            pass  # keep BEIR/medqa too — generality matters
        key = (d["dataset"] + ("(sf)" if d["sf"] else ""), d["provider"] or "-", d["method"])
        files[key] = p

    rows = []
    seen_pairs = set()

    def add_pair(ds, prov_a, meth_a, prov_b, meth_b, label):
        ka, kb = (ds, prov_a, meth_a), (ds, prov_b, meth_b)
        if ka not in files or kb not in files:
            return
        pair_id = (ka, kb)
        if pair_id in seen_pairs:
            return
        seen_pairs.add(pair_id)
        A, B = load_hits(files[ka]), load_hits(files[kb])
        ids = sorted(set(A) & set(B))
        if len(ids) < 30:
            return
        ha = sum(A[i] for i in ids) / len(ids)
        hb = sum(B[i] for i in ids) / len(ids)
        b_, c_, p = mcnemar(A, B, ids)
        lo, hi = bootstrap_ci(A, B, ids)
        rows.append(dict(dataset=ds, pair=label, a=f"{prov_a}/{meth_a}", b=f"{prov_b}/{meth_b}",
                         n=len(ids), hit_a=ha, hit_b=hb, delta=ha - hb,
                         mcnemar_b=b_, mcnemar_c=c_, p=p, ci_lo=lo, ci_hi=hi))

    datasets = sorted({k[0] for k in files})
    providers = sorted({k[1] for k in files if k[1] != "-"})
    for ds in datasets:
        for prov in providers:
            add_pair(ds, prov, "snap_hyre", prov, "rag_hyde", "SCOPE vs HyDE")
            add_pair(ds, prov, "snap_hyre", "-", "raw_question", "SCOPE vs raw")
            add_pair(ds, prov, "rag_hyde", "-", "raw_question", "HyDE vs raw")
            add_pair(ds, prov, "raw_scope_pool", prov, "snap_hyre", "pool vs SCOPE")
            add_pair(ds, prov, "raw_scope_pool", "-", "raw_question", "pool vs raw")
        add_pair(ds, "-", "csqe", "-", "raw_question", "CSQE vs raw")

    rows.sort(key=lambda r: (r["dataset"], r["pair"], r["a"]))
    with open(OUT_JSON, "w") as fh:
        json.dump(rows, fh, indent=1)
    with open(OUT_MD, "w") as fh:
        fh.write("# Retrieval-side significance (Hit@5), full caches — 2026-07-02\n\n")
        fh.write("Exact McNemar + 95% percentile bootstrap CI (n=2000, seed 42) on per-query "
                 "Hit@5 indicators; k=5 over cached `retrieved_ids`. Rows with gold ids only. "
                 "Fixes the 'no retrieval-side significance tests exist' gap (C7/C11).\n\n")
        fh.write("| Dataset | Pair | A | B | N | Hit@5 A | Hit@5 B | Δpp | b/c | p | 95% CI (pp) |\n")
        fh.write("|---|---|---|---|---:|---:|---:|---:|---|---|---|\n")
        for r in rows:
            star = "**" if r["p"] < 0.05 else ""
            fh.write(f"| {r['dataset']} | {r['pair']} | {r['a']} | {r['b']} | {r['n']} "
                     f"| {r['hit_a']:.3f} | {r['hit_b']:.3f} | {star}{100*r['delta']:+.2f}{star} "
                     f"| {r['mcnemar_b']}/{r['mcnemar_c']} | {r['p']:.2e} "
                     f"| [{100*r['ci_lo']:+.2f}, {100*r['ci_hi']:+.2f}] |\n")
    print(f"wrote {OUT_MD} ({len(rows)} pairs)")
    sig = [r for r in rows if r["p"] < 0.05]
    print(f"significant at 0.05: {len(sig)}/{len(rows)}")
    for r in sig[:40]:
        print(f"  {r['dataset']:22s} {r['pair']:14s} {r['a']:28s} {100*r['delta']:+.2f}pp p={r['p']:.1e}")


if __name__ == "__main__":
    main()
