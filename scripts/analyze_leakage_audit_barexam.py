#!/usr/bin/env python
"""Yoon-style knowledge-leakage audit of SCOPE generations on BarExamQA.

QUESTION (wiki: [[leakage-audit-barexam]]): is the weak-query retrieval lift
of SCOPE-style generation explained by knowledge leakage — the generator
reproducing content entailed by the gold passage (Yoon et al. 2025,
arXiv:2504.14175) — or does the lift survive on unmatched generations, as the
geometry account predicts?

Data: caches/generation/full/barexam_qfull_seed42_or-gemma4-26b_3scope_raw.jsonl
(3 exemplar-anchored SCOPE passages per question, Gemma 4 26B) matched to
per-sample retrieval in .../barexam_qfull_seed42_or-gemma4-26b_3scope_raw_pool_k5.jsonl
(components = [raw, s1, s2, s3], top-10 each). Gold text from the judge-pilot
hydration (scripts/judge_pilot/data/needed_texts.jsonl + qa.csv fallback).

Protocol (Yoon): a generated passage is MATCHED if any of its sentences is
ENTAILED by the gold passage (premise=gold, hypothesis=sentence) at
p(entail) >= tau under cross-encoder/nli-deberta-v3-base. Report tau in
{0.7, 0.8, 0.9}. Split per-sample Hit@5/Hit@10 lift over raw by matched
status.

Caveat recorded in output: these are the 3SCOPE exemplar-anchored generations
(the canonical single-SCOPE texts were never committed); the exemplar prompt
includes real corpus passages from OTHER questions, which if anything should
RAISE matched rates — biasing against the survives-on-unmatched conclusion,
i.e. a conservative test for us.
"""
import json, os, re, sys, time

GEN = "caches/generation/full/barexam_qfull_seed42_or-gemma4-26b_3scope_raw.jsonl"
POOL = "caches/retrieval/full/barexam_qfull_seed42_or-gemma4-26b_3scope_raw_pool_k5.jsonl"
RAW = "caches/retrieval/full/barexam_qfull_seed42_raw_question_k10.jsonl"
TEXTS = "scripts/judge_pilot/data/needed_texts.jsonl"
QA = "datasets/barexam_qa/qa/qa.csv"
OUT_MD = "docs/generated/leakage_audit_barexam_2026-07-02.md"
OUT_PTS = "docs/generated/leakage_audit_barexam_2026-07-02_points.jsonl"
NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
TAUS = (0.7, 0.8, 0.9)
K = 5


def sentences(text, min_len=25):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) >= min_len][:12]


def main():
    import csv
    csv.field_size_limit(10 ** 9)
    gold_text = {}
    for line in open(TEXTS):
        r = json.loads(line)
        gold_text[r["id"]] = r["text"]
    qa = {r["idx"]: r for r in csv.DictReader(open(QA))}
    for r in qa.values():
        gold_text.setdefault(r["gold_idx"], (r.get("gold_passage") or "").strip())

    gen = {json.loads(l)["idx"]: json.loads(l) for l in open(GEN)}
    pool = {json.loads(l)["idx"]: json.loads(l) for l in open(POOL)}
    raw = {json.loads(l)["idx"]: json.loads(l) for l in open(RAW)}

    # sanity: component 0 == raw retrieval
    same = tot = 0
    for qi in list(pool)[:200]:
        if qi in raw:
            tot += 1
            if pool[qi]["component_retrieved_ids"][0][:K] == raw[qi]["retrieved_ids"][:K]:
                same += 1
    print(f"component0==raw top{K} on {same}/{tot} spot-checked rows")

    from sentence_transformers import CrossEncoder
    dev = "mps"
    nli = CrossEncoder(NLI_MODEL, device=dev)
    # label order for this model: contradiction, entailment, neutral
    ENT = 1

    import numpy as np
    jobs = []   # (idx, sample_i, sent_i, premise, hypothesis)
    rows = {}
    for qi, g in gen.items():
        if qi not in pool or qi not in raw:
            continue
        gids = pool[qi]["gold_ids"]
        gtxt = " ".join(gold_text.get(x, "") for x in gids).strip()
        if not gtxt:
            continue
        comps = pool[qi]["component_retrieved_ids"]
        raw_hit = any(x in set(gids) for x in comps[0][:K])
        rows[qi] = {"idx": qi, "raw_hit5": raw_hit, "samples": []}
        for si, passage in enumerate(g["scope_passages"][:3]):
            hit = any(x in set(gids) for x in comps[si + 1][:K])
            sents = sentences(passage)
            rows[qi]["samples"].append({"hit5": hit, "n_sents": len(sents), "max_ent": 0.0})
            for sj, s in enumerate(sents):
                jobs.append((qi, si, gtxt[:3000], s))
    print(f"NLI pairs: {len(jobs)} over {len(rows)} questions")

    B = 128
    t0 = time.time()
    for i in range(0, len(jobs), B):
        chunk = jobs[i:i + B]
        logits = nli.predict([(p, h) for _, _, p, h in chunk],
                             batch_size=B, convert_to_numpy=True,
                             apply_softmax=True, show_progress_bar=False)
        for (qi, si, _, _), probs in zip(chunk, logits):
            e = float(probs[ENT])
            if e > rows[qi]["samples"][si]["max_ent"]:
                rows[qi]["samples"][si]["max_ent"] = e
        if (i // B) % 20 == 0:
            r = (i + B) / max(time.time() - t0, 1e-9)
            print(f"  {min(i+B,len(jobs))}/{len(jobs)} ({r:.0f} pairs/s)", flush=True)

    with open(OUT_PTS, "w") as fh:
        for qi, r in rows.items():
            fh.write(json.dumps(r) + "\n")

    # analysis per tau
    lines = ["# Yoon-style leakage audit, BarExamQA 3SCOPE (Gemma 4 26B) - 2026-07-02\n",
             "Question: does the SCOPE-family weak-query retrieval lift survive on",
             "generations NOT entailed by the gold passage (unmatched), or is it",
             "leakage-concentrated (Yoon et al. 2504.14175)?\n",
             f"Data: 3 exemplar-anchored SCOPE samples/question x {len(rows)} questions;",
             "per-sample Hit@5 vs shared raw-question Hit@5. NLI: nli-deberta-v3-base,",
             "premise=gold passage, hypothesis=generated sentence, matched = max",
             "p(entail) >= tau. CAVEAT: exemplar-anchored variant (canonical texts not",
             "in repo); exemplar prompts bias matched-rate UP -> conservative for the",
             "survives-on-unmatched claim.\n"]
    raw_hit_rate = sum(r["raw_hit5"] for r in rows.values()) / len(rows)
    lines.append(f"Raw-question Hit@5 on these rows: {raw_hit_rate:.4f}\n")
    lines.append("| tau | samples matched | Hit@5 matched | lift matched | samples unmatched | Hit@5 unmatched | lift unmatched |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for tau in TAUS:
        m_h, m_r, u_h, u_r = [], [], [], []
        for r in rows.values():
            for s in r["samples"]:
                (m_h if s["max_ent"] >= tau else u_h).append(s["hit5"])
                (m_r if s["max_ent"] >= tau else u_r).append(r["raw_hit5"])
        def rate(x): return sum(x) / len(x) if x else float("nan")
        lines.append(
            f"| {tau} | {len(m_h)} ({len(m_h)/(len(m_h)+len(u_h)):.1%}) | {rate(m_h):.4f} "
            f"| {100*(rate(m_h)-rate(m_r)):+.2f}pp | {len(u_h)} | {rate(u_h):.4f} "
            f"| {100*(rate(u_h)-rate(u_r)):+.2f}pp |")
    lines.append("\nReading: 'lift' = Hit@5(sample) - Hit@5(raw) within the stratum. Yoon")
    lines.append("predicts unmatched lift <= 0; the geometry account predicts unmatched")
    lines.append("lift > 0 (smaller magnitude allowed).\n")
    open(OUT_MD, "w").write("\n".join(lines))
    print("wrote", OUT_MD)
    print("\n".join(lines[-12:]))


if __name__ == "__main__":
    main()
