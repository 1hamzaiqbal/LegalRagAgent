#!/usr/bin/env python
"""Build a BEIR (scidocs) relevance-judge dataset (P3 label-semantics resolver: human-relevance qrels).

QUESTION (wiki: [[judge-pilot-scidocs]]): does the same free-label judge
recipe transfer outside legal — and does its gain over the ms-marco CE track
the CE's measurable gold-bury rate? SciDocs chosen for N=1000 and a known
mid-strength regime (raw Hit@5 49%).

Positives = qrels gold; hard negatives = retrieved non-gold from raw + SCOPE
k10 caches; eval pools = the committed raw-union-SCOPE pool cache (Gemma).
Query-level splits (BEIR queries are independent; no group structure).
All-local: corpus/qrels from datasets/beir/scidocs, caches from caches/.
"""
import csv, json, os, random, sys

csv.field_size_limit(10 ** 9)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
OUT = "scripts/judge_pilot/data"
SUBSET = "fiqa"
KEY = "beir_fiqa"
BASE = f"datasets/beir/{SUBSET}"
CACHE_RAW = f"caches/retrieval/full/{KEY}_qfull_seed42_raw_question_k10.jsonl"
CACHE_SCOPE = f"caches/retrieval/full/{KEY}_qfull_seed42_or-gemma4-26b_snap_hyre_k10.jsonl"
CACHE_POOL = f"caches/retrieval/full/{KEY}_qfull_seed42_or-gemma4-26b_raw_scope_pool_k5.jsonl"

N_TEST = 250
MAX_PASSAGE_CHARS = 1500

PROMPT_TMPL = (
    "You are a financial retrieval judge. Decide whether the document answers "
    "or directly addresses the finance question.\n\n"
    "Query: {question}\n\n"
    "Document: {passage}\n\n"
    "Does this document answer or directly address the question? Answer Yes or No.\nAnswer:"
)


def load_cache(path):
    return {str(json.loads(l)["idx"]): json.loads(l) for l in open(path)}


def main():
    corpus = {str(r["idx"]): r["text"] for r in csv.DictReader(open(f"{BASE}/corpus.csv"))}
    questions = {str(r["idx"]): r for r in csv.DictReader(open(f"{BASE}/questions.csv"))}
    gold = {}
    for r in csv.DictReader(open(f"{BASE}/qrels_test.csv")):
        if float(r["score"]) > 0:
            gold.setdefault(str(r["query_id"]), []).append(str(r["corpus_id"]))
    raw, scope, pool = load_cache(CACHE_RAW), load_cache(CACHE_SCOPE), load_cache(CACHE_POOL)
    qids = [q for q in questions if q in pool and q in gold]
    print(f"questions with pool+gold: {len(qids)}")
    rng = random.Random(42)
    rng.shuffle(qids)
    test_ids = qids[:N_TEST]
    train_ids = qids[N_TEST + 60:]
    dev_ids = qids[N_TEST:N_TEST + 60]
    json.dump({"train": train_ids, "dev": dev_ids, "test": test_ids},
              open(f"{OUT}/fiqa_split.json", "w"))

    def qtext(qi):
        return questions[qi].get("question") or questions[qi].get("text") or ""

    def mk(qi, pid, label):
        return {"question_idx": qi,
                "prompt_text": PROMPT_TMPL.format(
                    question=qtext(qi).strip(),
                    passage=corpus[pid].strip()[:MAX_PASSAGE_CHARS]),
                "label": label}

    stats = {}
    for name, ids in (("fiqa_train", train_ids), ("fiqa_dev", dev_ids)):
        rows = []
        for qi in ids:
            gs = [g for g in gold[qi] if g in corpus]
            if not gs:
                continue
            rows.append(mk(qi, gs[0], "Yes"))
            gset = set(gold[qi])
            for cache in (raw, scope):
                got = 0
                if qi not in cache:
                    continue
                for pid in (str(x) for x in cache[qi]["retrieved_ids"]):
                    if pid in gset or pid not in corpus:
                        continue
                    rows.append(mk(qi, pid, "No")); gset.add(pid); got += 1
                    if got >= 2:
                        break
        rng.shuffle(rows)
        with open(f"{OUT}/{name}.jsonl", "w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        stats[name] = len(rows)

    n_pools = n_gold = 0
    with open(f"{OUT}/fiqa_pools_test.jsonl", "w") as fh:
        for qi in test_ids:
            p = pool[qi]
            cand, seen = [], set()
            for comp in p["component_retrieved_ids"]:
                for pid in (str(x) for x in comp):
                    if pid not in seen and pid in corpus:
                        seen.add(pid)
                        cand.append({"id": pid, "text": corpus[pid][:MAX_PASSAGE_CHARS]})
            n_pools += 1
            n_gold += any(g in seen for g in gold[qi])
            fh.write(json.dumps({
                "question_idx": qi, "facts": "(none)", "question": qtext(qi).strip(),
                "gold_ids": gold[qi], "candidates": cand,
                "ce_top5": [str(x) for x in p["retrieved_ids"]],
                "raw_top5": [str(x) for x in raw[qi]["retrieved_ids"][:5]] if qi in raw else [],
                "scope_top5": [str(x) for x in scope[qi]["retrieved_ids"][:5]] if qi in scope else [],
            }) + "\n")
    print(f"train={stats.get('fiqa_train')} dev={stats.get('fiqa_dev')} "
          f"pools={n_pools} gold-in-pool={n_gold} ({n_gold/max(n_pools,1):.1%} ceiling)")


if __name__ == "__main__":
    main()
