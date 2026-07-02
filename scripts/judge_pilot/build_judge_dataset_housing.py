#!/usr/bin/env python
"""Build the HousingQA relevance-judge dataset (thesis-v2 P3, strong-regime test).

QUESTION (wiki: [[judge-pilot-housing]]): does pool + trained-judge reranking
also win on the strong/intermediate regime, where expansion alone hurts and
the raw query is already corpus-shaped? If yes, [[regime-routing]] collapses
into "always pool + trained judge" (thesis-v2 prediction 2).

Same recipe as the BarExamQA pilot: positives = benchmark gold statute ids;
hard negatives = non-gold statutes actually retrieved by the state-filtered
raw and SCOPE caches; eval = rerank the identical raw-union-SCOPE pools the
ms-marco CE reranked (statefilter, Gemma). Splits are QUESTION-GROUP-level
(seed 42) because HousingQA Y/N questions derive from shared source questions
(cross-split leakage otherwise): ~60% train / 10% dev / 30% test groups; test
pools are subsampled to 500 for Tinker budget.

Two-phase because statute texts live in the EIT housing_statutes Chroma
(1.8M docs):
  phase ids   -> data/housing_needed_ids.txt      (run first, extract on EIT)
  phase build -> train/dev/pools JSONLs           (needs housing_needed_texts.jsonl)
"""
import csv, json, os, random, sys

csv.field_size_limit(10 ** 9)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
OUT = "scripts/judge_pilot/data"
os.makedirs(OUT, exist_ok=True)

QA = "datasets/housing_qa/questions.csv"
CACHE_RAW = "caches/retrieval/full/housing_qfull_seed42_statefilter_raw_question_k10.jsonl"
CACHE_SCOPE = "caches/retrieval/full/housing_qfull_seed42_statefilter_or-gemma4-26b_snap_hyre_k10.jsonl"
CACHE_POOL = "caches/retrieval/full/housing_qfull_seed42_statefilter_or-gemma4-26b_raw_scope_pool_k5.jsonl"

N_TEST_POOLS = 500
N_TRAIN_Q = 1000
N_NEG_EACH = 2           # per cache (raw, scope) -> up to 4 negatives/question
MAX_PASSAGE_CHARS = 1500

PROMPT_TMPL = (
    "You are a legal retrieval judge for U.S. housing law. Decide whether the "
    "statute passage provides the legal basis to answer the question for the "
    "given state.\n\n"
    "State: {state}\nQuestion: {question}\n\n"
    "Statute passage: {passage}\n\n"
    "Does this passage provide the controlling statutory basis to answer this "
    "question? Answer Yes or No.\nAnswer:"
)


def load_cache(path):
    d = {}
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            d[str(r["idx"])] = r
    return d


def golds(row):
    return [g.strip() for g in row["gold_idx"].split(",") if g.strip()]


def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else "ids"
    qa = list(csv.DictReader(open(QA)))
    raw = load_cache(CACHE_RAW)
    scope = load_cache(CACHE_SCOPE)
    pool = load_cache(CACHE_POOL)
    print(f"qa={len(qa)} caches: raw={len(raw)} scope={len(scope)} pool={len(pool)}")

    rng = random.Random(42)
    groups = sorted({r["question_group"] for r in qa})
    rng.shuffle(groups)
    n = len(groups)
    test_g = set(groups[: int(0.30 * n)])
    dev_g = set(groups[int(0.30 * n): int(0.40 * n)])
    train_g = set(groups[int(0.40 * n):])

    test_q_all = [r for r in qa if r["question_group"] in test_g and str(r["idx"]) in pool]
    rng.shuffle(test_q_all)
    test_q = test_q_all[:N_TEST_POOLS]
    train_q_all = [r for r in qa if r["question_group"] in train_g]
    rng.shuffle(train_q_all)
    train_q = train_q_all[:N_TRAIN_Q]
    dev_q = [r for r in qa if r["question_group"] in dev_g][:120]
    json.dump({"train": [r["idx"] for r in train_q], "dev": [r["idx"] for r in dev_q],
               "test": [r["idx"] for r in test_q],
               "group_counts": {"train": len(train_g), "dev": len(dev_g), "test": len(test_g)}},
              open(f"{OUT}/housing_split.json", "w"))

    need = set()
    for r in train_q + dev_q:
        qi = str(r["idx"])
        need.update(golds(r))
        for cache in (raw, scope):
            if qi in cache:
                need.update(str(x) for x in cache[qi]["retrieved_ids"][:6])
    for r in test_q:
        qi = str(r["idx"])
        need.update(golds(r))
        p = pool[qi]
        for comp in p["component_retrieved_ids"]:
            need.update(str(x) for x in comp)
        need.update(str(x) for x in p["retrieved_ids"])
    print(f"unique statute ids to hydrate: {len(need)}")

    if phase == "ids":
        with open(f"{OUT}/housing_needed_ids.txt", "w") as fh:
            fh.write("\n".join(sorted(need)))
        print(f"wrote {OUT}/housing_needed_ids.txt — extract on EIT, then run phase build")
        return

    texts = {}
    for line in open(f"{OUT}/housing_needed_texts.jsonl"):
        r = json.loads(line)
        texts[str(r["id"])] = r["text"]
    missing = need - set(texts)
    print(f"hydrated={len(texts & need if isinstance(texts,set) else set(texts) & need)} missing={len(missing)}")

    def mk(qrow, passage_text, label):
        return {"question_idx": qrow["idx"],
                "prompt_text": PROMPT_TMPL.format(
                    state=qrow["state"], question=qrow["question"].strip(),
                    passage=passage_text.strip()[:MAX_PASSAGE_CHARS]),
                "label": label}

    stats = {}
    for name, qs in (("housing_train", train_q), ("housing_dev", dev_q)):
        rows = []
        for r in qs:
            qi = str(r["idx"])
            gs = [g for g in golds(r) if g in texts]
            if not gs:
                continue
            rows.append(mk(r, texts[gs[0]], "Yes"))
            gset = set(golds(r))
            added = 0
            for cache in (raw, scope):
                got = 0
                if qi not in cache:
                    continue
                for pid in (str(x) for x in cache[qi]["retrieved_ids"]):
                    if pid in gset or pid not in texts:
                        continue
                    rows.append(mk(r, texts[pid], "No"))
                    gset.add(pid); got += 1; added += 1
                    if got >= N_NEG_EACH:
                        break
        rng.shuffle(rows)
        with open(f"{OUT}/{name}.jsonl", "w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        stats[name] = len(rows)

    n_pools = n_gold = 0
    with open(f"{OUT}/housing_pools_test.jsonl", "w") as fh:
        for r in test_q:
            qi = str(r["idx"])
            p = pool[qi]
            cand, seen = [], set()
            for comp in p["component_retrieved_ids"]:
                for pid in (str(x) for x in comp):
                    if pid not in seen and pid in texts:
                        seen.add(pid)
                        cand.append({"id": pid, "text": texts[pid][:MAX_PASSAGE_CHARS]})
            gold_ids = golds(r)
            n_pools += 1
            n_gold += any(g in seen for g in gold_ids)
            fh.write(json.dumps({
                "question_idx": r["idx"], "facts": f"State: {r['state']}",
                "state": r["state"], "question": r["question"].strip(),
                "gold_ids": gold_ids, "candidates": cand,
                "ce_top5": [str(x) for x in p["retrieved_ids"]],
                "raw_top5": [str(x) for x in raw[qi]["retrieved_ids"][:5]] if qi in raw else [],
                "scope_top5": [str(x) for x in scope[qi]["retrieved_ids"][:5]] if qi in scope else [],
            }) + "\n")
    print(f"train={stats.get('housing_train')} dev={stats.get('housing_dev')} "
          f"pools={n_pools} gold-in-pool={n_gold} ({n_gold/max(n_pools,1):.1%} ceiling)")


if __name__ == "__main__":
    main()
