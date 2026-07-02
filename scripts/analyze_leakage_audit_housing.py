#!/usr/bin/env python
"""Yoon-style leakage audit on HousingQA 3SCOPE generations (strong-legal regime).

QUESTION (extends [[leakage-audit-barexam]]): on the strong/intermediate legal
regime, is the (smaller) SCOPE retrieval delta leakage-gated? Housing statutes
are corpus-specific and post-training-cutoff-diverse; the geometry account
predicts matched/unmatched strata behave like BarExamQA's, scaled down.

Data: statefilter 3SCOPE generation cache (3 samples/question, Gemma) matched
to per-sample dense retrieval in the 3scope_raw_pool components
(components = [raw, s1, s2, s3]). Gold texts: judge-pilot housing hydration
(covers the 1,620 split questions — sample restricted to those with hydrated
gold). Subsample 1,200 questions (seed 42) for MPS runtime.
"""
import json, random, re, time

GEN = "caches/generation/full/housing_qfull_seed42_statefilter_or-gemma4-26b_3scope_raw.jsonl"
POOL = "caches/retrieval/full/housing_qfull_seed42_statefilter_or-gemma4-26b_3scope_raw_pool_k5.jsonl"
TEXTS = "scripts/judge_pilot/data/housing_needed_texts.jsonl"
OUT_MD = "docs/generated/leakage_audit_housing_2026-07-02.md"
OUT_PTS = "docs/generated/leakage_audit_housing_2026-07-02_points.jsonl"
TAUS = (0.7, 0.8, 0.9)
K = 5
N_SAMPLE = 1200


def sentences(text, min_len=25):
    parts = re.split(r"(?<=[.!?])\s+", str(text).strip())
    return [p.strip() for p in parts if len(p.strip()) >= min_len][:12]


def main():
    texts = {}
    for line in open(TEXTS):
        r = json.loads(line)
        texts[str(r["id"])] = r["text"]
    gen = {str(json.loads(l)["idx"]): json.loads(l) for l in open(GEN)}
    pool = {str(json.loads(l)["idx"]): json.loads(l) for l in open(POOL)}

    eligible = []
    for qi, p in pool.items():
        if qi not in gen:
            continue
        gids = [str(x) for x in p["gold_ids"]]
        if gids and all(g in texts for g in gids[:2]):
            eligible.append(qi)
    rng = random.Random(42)
    rng.shuffle(eligible)
    sample = eligible[:N_SAMPLE]
    print(f"eligible={len(eligible)} sampled={len(sample)}")

    from sentence_transformers import CrossEncoder
    nli = CrossEncoder("cross-encoder/nli-deberta-v3-base", device="mps")
    ENT = 1
    jobs, rows = [], {}
    for qi in sample:
        p, g = pool[qi], gen[qi]
        gids = [str(x) for x in p["gold_ids"]]
        gtxt = " ".join(texts[x] for x in gids[:2])[:2000]
        comps = p["component_retrieved_ids"]
        gset = set(gids)
        rows[qi] = {"idx": qi,
                    "raw_hit5": any(str(x) in gset for x in comps[0][:K]),
                    "samples": []}
        for si, passage in enumerate(g["scope_passages"][:3]):
            hit = any(str(x) in gset for x in comps[si + 1][:K])
            rows[qi]["samples"].append({"hit5": hit, "max_ent": 0.0})
            for s in sentences(passage):
                jobs.append((qi, si, gtxt, s))
    print(f"NLI pairs: {len(jobs)} over {len(rows)} questions", flush=True)

    B = 16
    t0 = time.time()
    for i in range(0, len(jobs), B):
        chunk = jobs[i:i + B]
        probs = nli.predict([(p_, h) for _, _, p_, h in chunk], batch_size=B,
                            convert_to_numpy=True, apply_softmax=True,
                            show_progress_bar=False)
        for (qi, si, _, _), pr in zip(chunk, probs):
            e = float(pr[ENT])
            if e > rows[qi]["samples"][si]["max_ent"]:
                rows[qi]["samples"][si]["max_ent"] = e
        if (i // B) % 100 == 0:
            print(f"  {min(i+B,len(jobs))}/{len(jobs)} ({(i+B)/max(time.time()-t0,1e-9):.0f}/s)", flush=True)

    with open(OUT_PTS, "w") as fh:
        for r in rows.values():
            fh.write(json.dumps(r) + "\n")

    lines = ["# Yoon-style leakage audit, HousingQA statefilter 3SCOPE (Gemma) - 2026-07-02\n",
             f"Subsample {len(rows)} questions (hydrated-gold, seed 42); dense-stage",
             "per-sample retrieval vs shared raw component. Matched = any generated",
             "sentence entailed by gold statute text (nli-deberta-v3-base).\n"]
    raw_rate = sum(r["raw_hit5"] for r in rows.values()) / len(rows)
    lines.append(f"Raw (dense component) Hit@5: {raw_rate:.4f}\n")
    lines.append("| tau | matched | Hit@5 matched (lift) | unmatched | Hit@5 unmatched (lift) |")
    lines.append("|---|---:|---:|---:|---:|")
    for tau in TAUS:
        m_h, m_r, u_h, u_r = [], [], [], []
        for r in rows.values():
            for s in r["samples"]:
                (m_h if s["max_ent"] >= tau else u_h).append(s["hit5"])
                (m_r if s["max_ent"] >= tau else u_r).append(r["raw_hit5"])
        def rate(x): return sum(x) / len(x) if x else float("nan")
        lines.append(f"| {tau} | {len(m_h)} ({len(m_h)/(len(m_h)+len(u_h)):.1%}) "
                     f"| {rate(m_h):.4f} ({100*(rate(m_h)-rate(m_r)):+.2f}pp) "
                     f"| {len(u_h)} | {rate(u_h):.4f} ({100*(rate(u_h)-rate(u_r)):+.2f}pp) |")
    open(OUT_MD, "w").write("\n".join(lines))
    print("wrote", OUT_MD)
    print("\n".join(lines[-6:]))


if __name__ == "__main__":
    main()
