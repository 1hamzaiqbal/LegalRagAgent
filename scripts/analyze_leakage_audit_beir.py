#!/usr/bin/env python
"""Yoon-style leakage audit on BEIR (generalization of the BarExamQA audit).

QUESTION (wiki: [[leakage-audit-barexam]] follow-up, thesis-v2 P1): does the
matched/unmatched decomposition of generative-expansion retrieval deltas look
the same OUTSIDE legal — i.e. is the geometry-not-leakage reading
domain-general? On strong-query BEIR expansion is net-negative overall, so
the informative strata are (a) rows where expansion HELPED (is help
leakage-concentrated?) and (b) the lowest raw-margin quintile (the weak-query
pocket inside BEIR — [[beir-phase1]] showed expansion helps there).

Data per dataset in {scifact, nfcorpus, scidocs}: canonical single-passage
generation caches (rag_hyde + snap_hyre, Gemma 4 26B), k10 retrieval caches
(expansion + raw), gold texts from datasets/beir/<subset>/{corpus,qrels}.
Matched = any generated sentence entailed by (CE-best or first) gold doc,
nli-deberta-v3-base, tau=0.8 primary (0.9 sensitivity).

Output: docs/generated/leakage_audit_beir_2026-07-02.md + _points.jsonl
"""
import csv, json, os, re, sys, time
from math import comb

csv.field_size_limit(10 ** 9)
K = 5
SUBSETS = {"scifact": "beir_scifact", "nfcorpus": "beir_nfcorpus", "scidocs": "beir_scidocs"}
GEN_TMPL = "caches/generation/full/{key}_qfull_seed42_or-gemma4-26b_{method}.jsonl"
RET_TMPL = "caches/retrieval/full/{key}_qfull_seed42_or-gemma4-26b_{method}_k10.jsonl"
RAW_TMPL = "caches/retrieval/full/{key}_qfull_seed42_raw_question_k10.jsonl"
OUT_MD = "docs/generated/leakage_audit_beir_2026-07-02.md"
OUT_PTS = "docs/generated/leakage_audit_beir_2026-07-02_points.jsonl"
TAUS = (0.8, 0.9)


def sentences(text, min_len=25):
    parts = re.split(r"(?<=[.!?])\s+", str(text).strip())
    return [p.strip() for p in parts if len(p.strip()) >= min_len][:12]


def load_cache(path):
    d = {}
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            d[str(r["idx"])] = r
    return d


def hit(ids, gold, k=K):
    g = set(gold)
    return any(x in g for x in ids[:k])


def mcnemar(pairs):
    b = sum(1 for a, r in pairs if a and not r)
    c = sum(1 for a, r in pairs if r and not a)
    n = b + c
    p = sum(comb(n, k) for k in range(min(b, c) + 1)) * 2 / 2 ** n if n else 1.0
    return b, c, min(p, 1.0)


def main():
    from sentence_transformers import CrossEncoder
    nli = CrossEncoder("cross-encoder/nli-deberta-v3-base", device="mps")
    ENT = 1
    points = []

    for subset, key in SUBSETS.items():
        base = f"datasets/beir/{subset}"
        if not os.path.exists(f"{base}/corpus.csv"):
            print(f"skip {subset}: corpus missing"); continue
        corpus = {}
        for row in csv.DictReader(open(f"{base}/corpus.csv")):
            corpus[str(row["idx"])] = row["text"]
        gold_by_q = {}
        for row in csv.DictReader(open(f"{base}/qrels_test.csv")):
            if float(row["score"]) > 0:
                gold_by_q.setdefault(str(row["query_id"]), []).append(str(row["corpus_id"]))
        raw = load_cache(RAW_TMPL.format(key=key))
        for method in ("rag_hyde", "snap_hyre"):
            gen = load_cache(GEN_TMPL.format(key=key, method=method))
            ret = load_cache(RET_TMPL.format(key=key, method=method))
            jobs, rows = [], {}
            for qi, g in gen.items():
                if qi not in ret or qi not in raw:
                    continue
                gold = gold_by_q.get(qi) or raw[qi].get("gold_ids") or []
                gold = [str(x) for x in gold]
                gtxt = ""
                for gid in gold[:3]:
                    if gid in corpus:
                        gtxt += corpus[gid][:800] + "\n"
                if not gtxt.strip():
                    continue
                passage = g.get("hyde_passage") or ""
                if not passage:
                    continue
                rows[qi] = {"dataset": key, "method": method,
                            "raw_hit": hit([str(x) for x in raw[qi]["retrieved_ids"]], gold),
                            "exp_hit": hit([str(x) for x in ret[qi]["retrieved_ids"]], gold),
                            "max_ent": 0.0}
                for s in sentences(passage):
                    jobs.append((qi, gtxt[:2000], s))
            print(f"{key}/{method}: {len(rows)} rows, {len(jobs)} NLI pairs", flush=True)
            B = 16
            t0 = time.time()
            for i in range(0, len(jobs), B):
                chunk = jobs[i:i + B]
                probs = nli.predict([(p, h) for _, p, h in chunk], batch_size=B,
                                    convert_to_numpy=True, apply_softmax=True,
                                    show_progress_bar=False)
                for (qi, _, _), pr in zip(chunk, probs):
                    e = float(pr[ENT])
                    if e > rows[qi]["max_ent"]:
                        rows[qi]["max_ent"] = e
                if (i // B) % 100 == 0:
                    print(f"  {min(i+B,len(jobs))}/{len(jobs)} ({(i+B)/max(time.time()-t0,1e-9):.0f}/s)", flush=True)
            points.extend(rows.values())
            with open(OUT_PTS, "w") as fh:
                for r in points:
                    fh.write(json.dumps(r) + "\n")

    lines = ["# Yoon-style leakage audit on BEIR (Gemma 4 26B) - 2026-07-02\n",
             "Question: outside legal, is expansion help leakage-concentrated?",
             "Strata: matched vs unmatched (max sentence entailment vs gold, tau);",
             "reported per dataset x method: Hit@5 deltas and help/hurt counts.\n"]
    for tau in TAUS:
        lines.append(f"\n## tau = {tau}\n")
        lines.append("| Dataset | Method | N | matched | raw Hit@5 | exp Hit@5 (matched) | delta | exp Hit@5 (unmatched) | delta | help_m/help_u | McNemar-unmatched p |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|")
        for key in sorted({r["dataset"] for r in points}):
            for method in ("rag_hyde", "snap_hyre"):
                rs = [r for r in points if r["dataset"] == key and r["method"] == method]
                if not rs:
                    continue
                m = [r for r in rs if r["max_ent"] >= tau]
                u = [r for r in rs if r["max_ent"] < tau]
                def hr(g, f): return sum(r[f] for r in g) / len(g) if g else float("nan")
                help_m = sum(1 for r in m if r["exp_hit"] and not r["raw_hit"])
                help_u = sum(1 for r in u if r["exp_hit"] and not r["raw_hit"])
                b, c, p = mcnemar([(r["exp_hit"], r["raw_hit"]) for r in u])
                lines.append(
                    f"| {key} | {method} | {len(rs)} | {len(m)} ({len(m)/len(rs):.0%}) "
                    f"| {hr(rs,'raw_hit'):.3f} | {hr(m,'exp_hit'):.3f} | {100*(hr(m,'exp_hit')-hr(m,'raw_hit')):+.1f}pp "
                    f"| {hr(u,'exp_hit'):.3f} | {100*(hr(u,'exp_hit')-hr(u,'raw_hit')):+.1f}pp "
                    f"| {help_m}/{help_u} | b/c={b}/{c} p={p:.1e} |")
    open(OUT_MD, "w").write("\n".join(lines))
    print("wrote", OUT_MD)


if __name__ == "__main__":
    main()
