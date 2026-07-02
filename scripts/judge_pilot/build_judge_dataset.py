#!/usr/bin/env python
"""Build the BarExamQA relevance-judge dataset for the Tinker pilot (Path C v0).

Judge task: given (fact pattern, question, passage) -> does the passage state
the controlling legal rule? Supervision is free: gold_idx from qa.csv is the
positive; hard negatives are the non-gold passages actually retrieved by the
raw-question and SCOPE retrieval caches (the exact distractors the ms-marco CE
faces at rerank time).

Splits are QUESTION-level (seed 42): train 700 / dev 95 / test 400. The test
artifact is the raw-union-SCOPE candidate pool per question (from the signed
pool cache), so eval reranks the SAME pools the CE reranked — the pilot's
claim is judge-vs-CE on identical candidates.

Outputs (scripts/judge_pilot/data/):
  train.jsonl, dev.jsonl        {question_idx, prompt_text, label}
  pools_test.jsonl              {question_idx, gold_ids, candidates:[{id,text}],
                                 ce_top5, raw_top5, scope_top5}
  split.json                    question-id lists for provenance
"""
import csv, json, os, random, sys

csv.field_size_limit(10**9)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
OUT = "scripts/judge_pilot/data"
os.makedirs(OUT, exist_ok=True)

QA = "datasets/barexam_qa/qa/qa.csv"
CACHE_RAW = "caches/retrieval/full/barexam_qfull_seed42_raw_question_k10.jsonl"
CACHE_SCOPE = "caches/retrieval/full/barexam_qfull_seed42_or-gemma4-26b_snap_hyre_k10.jsonl"
CACHE_POOL = "caches/retrieval/full/barexam_qfull_seed42_or-gemma4-26b_raw_scope_pool_k5.jsonl"

N_NEG_TRAIN = 4          # 2 from raw retrieval, 2 from SCOPE retrieval
MAX_FACTS_CHARS = 1500
MAX_PASSAGE_CHARS = 1500

PROMPT_TMPL = (
    "You are a legal retrieval judge. Decide whether the passage states the "
    "legal rule needed to answer the bar-exam question.\n\n"
    "Fact pattern: {facts}\n"
    "Question: {question}\n\n"
    "Passage: {passage}\n\n"
    "Does this passage state the controlling legal rule for this question? "
    "Answer Yes or No.\nAnswer:"
)


def load_cache(path):
    d = {}
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            d[r["idx"]] = r
    return d


def main():
    qa = list(csv.DictReader(open(QA)))
    assert len(qa) == 1195, len(qa)
    raw = load_cache(CACHE_RAW)
    scope = load_cache(CACHE_SCOPE)
    pool = load_cache(CACHE_POOL)
    print(f"caches: raw={len(raw)} scope={len(scope)} pool={len(pool)}")

    rng = random.Random(42)
    idxs = [r["idx"] for r in qa]
    rng.shuffle(idxs)
    test_ids = set(idxs[:400])
    dev_ids = set(idxs[400:495])
    train_ids = set(idxs[495:])
    json.dump({"train": sorted(train_ids), "dev": sorted(dev_ids),
               "test": sorted(test_ids)}, open(f"{OUT}/split.json", "w"))

    # ---- collect every passage id we need text for ----
    need = set()
    byidx = {r["idx"]: r for r in qa}
    for r in qa:
        qi = r["idx"]
        need.add(r["gold_idx"])
        for cache in (raw, scope):
            if qi in cache:
                need.update(cache[qi]["retrieved_ids"])
        if qi in test_ids and qi in pool:
            for comp in pool[qi]["component_retrieved_ids"]:
                need.update(comp)
            need.update(pool[qi]["retrieved_ids"])
    print(f"passage ids to hydrate: {len(need)}")

    # ---- hydrate: (1) cluster-extracted corpus texts, (2) curated CSV ----
    texts = {}
    extract = f"{OUT}/needed_texts.jsonl"
    if os.path.exists(extract):
        for line in open(extract):
            r = json.loads(line)
            if r["id"] in need:
                texts[r["id"]] = r["text"]
    with open("datasets/barexam_qa/barexam_qa_curated.csv") as fh:
        for row in csv.DictReader(fh):
            if row["idx"] in need and row["idx"] not in texts:
                texts[row["idx"]] = row["text"]
    missing = need - set(texts)
    # gold text fallback from qa.csv's inline gold_passage column
    for r in qa:
        if r["gold_idx"] in missing and r.get("gold_passage", "").strip():
            texts[r["gold_idx"]] = r["gold_passage"].strip()
    missing = need - set(texts)
    print(f"hydrated={len(texts)} missing={len(missing)}")
    if missing:
        print("  sample missing:", sorted(missing)[:5])

    def mk(qrow, passage_text, label):
        facts = (qrow.get("prompt") or "").strip()[:MAX_FACTS_CHARS]
        return {
            "question_idx": qrow["idx"],
            "prompt_text": PROMPT_TMPL.format(
                facts=facts if facts else "(none)",
                question=(qrow.get("question") or "").strip(),
                passage=passage_text.strip()[:MAX_PASSAGE_CHARS]),
            "label": label,
        }

    def negatives(qi, gold, k_each):
        out, seen = [], {gold}
        for cache in (raw, scope):
            got = 0
            if qi not in cache:
                continue
            for pid in cache[qi]["retrieved_ids"]:
                if pid in seen or pid not in texts:
                    continue
                seen.add(pid); out.append(pid); got += 1
                if got >= k_each:
                    break
        return out

    stats = {"train": 0, "dev": 0}
    for name, id_set in (("train", train_ids), ("dev", dev_ids)):
        rows = []
        for r in qa:
            qi = r["idx"]
            if qi not in id_set:
                continue
            gold = r["gold_idx"]
            if gold not in texts:
                continue
            rows.append(mk(r, texts[gold], "Yes"))
            for pid in negatives(qi, gold, N_NEG_TRAIN // 2):
                rows.append(mk(r, texts[pid], "No"))
        rng.shuffle(rows)
        with open(f"{OUT}/{name}.jsonl", "w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        stats[name] = len(rows)

    # ---- test pools ----
    n_pools = n_gold_in_pool = 0
    with open(f"{OUT}/pools_test.jsonl", "w") as fh:
        for r in qa:
            qi = r["idx"]
            if qi not in test_ids or qi not in pool:
                continue
            p = pool[qi]
            cand = []
            seen = set()
            for comp in p["component_retrieved_ids"]:
                for pid in comp:
                    if pid not in seen and pid in texts:
                        seen.add(pid)
                        cand.append({"id": pid, "text": texts[pid][:MAX_PASSAGE_CHARS]})
            gold_ids = p["gold_ids"]
            n_pools += 1
            if any(g in seen for g in gold_ids):
                n_gold_in_pool += 1
            fh.write(json.dumps({
                "question_idx": qi,
                "facts": (r.get("prompt") or "").strip()[:MAX_FACTS_CHARS],
                "question": (r.get("question") or "").strip(),
                "gold_ids": gold_ids,
                "candidates": cand,
                "ce_top5": p["retrieved_ids"],
                "raw_top5": raw[qi]["retrieved_ids"][:5] if qi in raw else [],
                "scope_top5": scope[qi]["retrieved_ids"][:5] if qi in scope else [],
            }) + "\n")

    print(f"train={stats['train']} dev={stats['dev']} pools={n_pools} "
          f"gold-in-pool={n_gold_in_pool} ({n_gold_in_pool/max(n_pools,1):.1%} recall ceiling)")


if __name__ == "__main__":
    main()
