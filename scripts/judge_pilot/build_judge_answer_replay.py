#!/usr/bin/env python
"""Build artifacts for the judge-evidence ANSWER replay (thesis-v2: the wall).

QUESTION (wiki: [[judge-answer-conversion]]): does fixing the selector convert
to answers? Evidence = judge-trained top-5 over the raw-union-SCOPE pools,
answered by groq-llama70b via the harness's STRICT retrieval-cache replay
(rag_simple mode), on the 399 held-out judge-test questions.

Emits (scripts/judge_pilot/data/):
  qa_test399.csv            filtered qa.csv (EVAL_QA_CSV override)
  judge_top5_retrieval.jsonl  synthetic retrieval cache (label_prefix=simple,
                              collection=legal_passages, retrieved_ids=judge top5)
  judge_doc_cache.jsonl       full-text doc cache for those ids
"""
import csv, json, os

csv.field_size_limit(10 ** 9)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
DATA = "scripts/judge_pilot/data"

pools = [json.loads(l) for l in open(f"{DATA}/pools_test.jsonl")]
scores = json.load(open(f"{DATA}/scores_trained.json"))

top5 = {}
for pi, p in enumerate(pools):
    order = sorted(range(len(p["candidates"])),
                   key=lambda ci: scores[f"({pi}, {ci})"], reverse=True)
    top5[p["question_idx"]] = [p["candidates"][ci]["id"] for ci in order[:5]]

qa_rows = [r for r in csv.DictReader(open("datasets/barexam_qa/qa/qa.csv"))]
test_rows = [r for r in qa_rows if r["idx"] in top5]
assert len(test_rows) == len(top5), (len(test_rows), len(top5))
with open(f"{DATA}/qa_test399.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(qa_rows[0].keys()))
    w.writeheader()
    for r in test_rows:
        w.writerow(r)

texts = {}
for line in open(f"{DATA}/needed_texts.jsonl"):
    r = json.loads(line)
    texts[r["id"]] = r["text"]
with open("datasets/barexam_qa/barexam_qa_curated.csv") as fh:
    for row in csv.DictReader(fh):
        texts.setdefault(row["idx"], row["text"])
# pool candidate texts as last resort (1500-char capped)
for p in pools:
    for c in p["candidates"]:
        texts.setdefault(c["id"], c["text"])

need = sorted({pid for ids in top5.values() for pid in ids})
missing = [i for i in need if i not in texts]
print(f"doc-cache ids={len(need)} missing={len(missing)}")
assert not missing, missing[:5]

with open(f"{DATA}/judge_top5_retrieval.jsonl", "w") as fh:
    for r in test_rows:
        qi = r["idx"]
        fh.write(json.dumps({
            "idx": qi, "label": f"qa_nan_{qi}", "label_prefix": "simple",
            "collection": "legal_passages", "embedding_model": "", "where": {},
            "dataset": "barexam", "query_type": "judge_trained_top5",
            "retrieved_ids": top5[qi], "max_k": 5,
            "gold_ids": [r["gold_idx"]],
            "gold_retrieved": r["gold_idx"] in top5[qi],
        }) + "\n")

with open(f"{DATA}/judge_doc_cache.jsonl", "w") as fh:
    for pid in need:
        fh.write(json.dumps({"collection": "legal_passages", "idx": pid,
                             "text": texts[pid], "metadata": {"idx": pid}}) + "\n")

n_gold = sum(1 for r in test_rows if r["gold_idx"] in top5[r["idx"]])
print(f"rows={len(test_rows)} gold-in-top5={n_gold} ({n_gold/len(test_rows):.1%}) "
      f"— should match judge-trained Hit@5 0.206")

# --- comparator evidence caches on the same 399 questions (paired arms) ---
for arm_key, field in (("ce", "ce_top5"), ("scope", "scope_top5")):
    with open(f"{DATA}/{arm_key}_top5_retrieval.jsonl", "w") as fh:
        for r in test_rows:
            qi = r["idx"]
            p = next(pp for pp in pools if pp["question_idx"] == qi)
            ids = [str(x) for x in p[field][:5]]
            fh.write(json.dumps({
                "idx": qi, "label": f"qa_nan_{qi}", "label_prefix": "simple",
                "collection": "legal_passages", "embedding_model": "", "where": {},
                "dataset": "barexam", "query_type": f"{arm_key}_top5_replay",
                "retrieved_ids": ids, "max_k": 5,
                "gold_ids": [r["gold_idx"]],
                "gold_retrieved": r["gold_idx"] in ids,
            }) + "\n")
    print(f"wrote {arm_key}_top5_retrieval.jsonl")

# extend doc cache with any ce/scope ids not already present
extra = set()
for p in pools:
    extra.update(str(x) for x in p["ce_top5"][:5])
    extra.update(str(x) for x in p["scope_top5"][:5])
extra -= set(need)
extra_missing = [i for i in sorted(extra) if i not in texts]
print(f"extra ids={len(extra)} missing-text={len(extra_missing)}")
with open(f"{DATA}/judge_doc_cache.jsonl", "a") as fh:
    for pid in sorted(extra):
        if pid in texts:
            fh.write(json.dumps({"collection": "legal_passages", "idx": pid,
                                 "text": texts[pid], "metadata": {"idx": pid}}) + "\n")
