#!/usr/bin/env python
"""Build artifacts for the HOUSING judge-evidence answer replay.

QUESTION (wiki: [[judge-answer-conversion]] §Housing): the two-regime
conversion contrast — on Housing, gold evidence is worth ~+5pp (70B signed
rows: llm_only 44.8 → rag_simple 47.3 → golden 67.3) and the trained judge
converts 96.5% of a 57% pool ceiling. Predicted: judge evidence clears the
break-even here, unlike BarExamQA. Four paired arms on the 500 judge-test
questions: llm_only / CE-pool top5 / SCOPE top5 / judge-trained top5.

Emits: housing_qa_test500.csv, housing_{judge,ce,scope}_top5_retrieval.jsonl,
housing_answer_doc_cache.jsonl (full texts, collection=housing_statutes).
"""
import csv, json, os

csv.field_size_limit(10 ** 9)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
DATA = "scripts/judge_pilot/data"

pools = [json.loads(l) for l in open(f"{DATA}/housing_pools_test.jsonl")]
scores = json.load(open(f"{DATA}/housing_scores_trained.json"))
raw_cache = {json.loads(l)["idx"]: json.loads(l)
             for l in open("caches/retrieval/full/housing_qfull_seed42_statefilter_raw_question_k10.jsonl")}

top5 = {}
for pi, p in enumerate(pools):
    order = sorted(range(len(p["candidates"])),
                   key=lambda ci: scores[f"({pi}, {ci})"], reverse=True)
    top5[str(p["question_idx"])] = [p["candidates"][ci]["id"] for ci in order[:5]]

qa_rows = list(csv.DictReader(open("datasets/housing_qa/questions.csv")))
test_rows = [r for r in qa_rows if str(r["idx"]) in top5]
assert len(test_rows) == len(top5), (len(test_rows), len(top5))
with open(f"{DATA}/housing_qa_test500.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(qa_rows[0].keys()))
    w.writeheader()
    for r in test_rows:
        w.writerow(r)

texts = {json.loads(l)["id"]: json.loads(l)["text"]
         for l in open(f"{DATA}/housing_needed_texts.jsonl")}
pool_by_idx = {str(p["question_idx"]): p for p in pools}

def golds(row):
    return [g.strip() for g in row["gold_idx"].split(",") if g.strip()]

arms = {
    "judge": lambda r: top5[str(r["idx"])],
    "ce": lambda r: [str(x) for x in pool_by_idx[str(r["idx"])]["ce_top5"][:5]],
    "scope": lambda r: [str(x) for x in pool_by_idx[str(r["idx"])]["scope_top5"][:5]],
}
need = set()
for arm, fn in arms.items():
    with open(f"{DATA}/housing_{arm}_top5_retrieval.jsonl", "w") as fh:
        n_gold = 0
        for r in test_rows:
            qi = str(r["idx"])
            ids = fn(r)
            need.update(ids)
            src = raw_cache[qi]
            gset = set(golds(r))
            n_gold += any(i in gset for i in ids)
            fh.write(json.dumps({
                "idx": qi, "label": src.get("label") or f"housing_{qi}",
                "label_prefix": "simple", "collection": "housing_statutes",
                "embedding_model": "", "where": src["where"],
                "dataset": "housing", "query_type": f"{arm}_top5_replay",
                "retrieved_ids": ids, "max_k": 5,
                "gold_ids": golds(r),
                "gold_retrieved": any(i in gset for i in ids),
            }) + "\n")
    print(f"{arm}: gold-in-top5 {n_gold}/{len(test_rows)} ({n_gold/len(test_rows):.1%})")

missing = [i for i in sorted(need) if i not in texts]
print(f"doc ids={len(need)} missing={len(missing)}")
assert not missing, missing[:5]
with open(f"{DATA}/housing_answer_doc_cache.jsonl", "w") as fh:
    for pid in sorted(need):
        fh.write(json.dumps({"collection": "housing_statutes", "idx": pid,
                             "text": texts[pid], "metadata": {"idx": pid}}) + "\n")
print("artifacts written")
