#!/usr/bin/env python
"""Evaluate the Tinker-trained relevance judge as a pool reranker vs the CE.

For each held-out test pool (raw-union-SCOPE candidates, identical to what the
ms-marco cross-encoder reranked in the signed cache), score every candidate
with  s = logprob(" Yes") - logprob(" No")  under the judge, take top-5, and
compare Hit@5 / MRR@5 against the CE's recorded top-5 on the SAME pool.

Arms:
  trained  — LoRA checkpoint from train_tinker_judge.py (train_info.json)
  zeroshot — the untrained base model with the identical prompt (the
             prompted-judge baseline; the Thinking-Machines contrast)

Usage:
  set -a; source .env; set +a
  .venv/bin/python scripts/judge_pilot/eval_judge_rerank.py \
      [--arms trained,zeroshot] [--max-pools N] [--dev-check]
"""
import argparse, json, os, time
from concurrent.futures import wait

import tinker
from tinker import types as T

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
BASE_MODEL = "Qwen/Qwen3.5-9B"
MAX_TOKENS = 1024
WINDOW = 96   # in-flight requests

PROMPT_TMPL = (
    "You are a legal retrieval judge. Decide whether the passage states the "
    "legal rule needed to answer the bar-exam question.\n\n"
    "Fact pattern: {facts}\n"
    "Question: {question}\n\n"
    "Passage: {passage}\n\n"
    "Does this passage state the controlling legal rule for this question? "
    "Answer Yes or No.\nAnswer:"
)


def metrics(ranked_ids_per_pool, pools, k=5):
    hit = mrr = 0
    n = len(pools)
    if n == 0:
        return 0.0, 0.0, 0
    for ranked, p in zip(ranked_ids_per_pool, pools):
        gold = set(p["gold_ids"])
        top = ranked[:k]
        if any(g in gold for g in top):
            hit += 1
            r = next(i for i, pid in enumerate(top, 1) if pid in gold)
            mrr += 1.0 / r
    return hit / n, mrr / n, hit


def cached_list_metrics(key, pools, k=5):
    return metrics([p[key] for p in pools], pools, k)


class Scorer:
    def __init__(self, sampling_client, tok, yes_id, no_id):
        self.sc, self.tok, self.yes, self.no = sampling_client, tok, yes_id, no_id

    def submit(self, prompt_ids):
        # sync method is non-blocking: returns ConcurrentFuture
        f_yes = self.sc.compute_logprobs(T.ModelInput.from_ints(prompt_ids + [self.yes]))
        f_no = self.sc.compute_logprobs(T.ModelInput.from_ints(prompt_ids + [self.no]))
        return (f_yes, f_no)

    @staticmethod
    def resolve(pair):
        (fy, fn) = pair
        ly = fy.result()[-1]
        ln = fn.result()[-1]
        return (ly or 0.0) - (ln or 0.0)


def score_pools(sampling_client, tok, pools, max_pools=None):
    yes_id = tok.encode(" Yes", add_special_tokens=False)[0]
    no_id = tok.encode(" No", add_special_tokens=False)[0]
    sc = Scorer(sampling_client, tok, yes_id, no_id)
    pools = pools[:max_pools] if max_pools else pools
    jobs = []  # (pool_i, cand_i, futures)
    for pi, p in enumerate(pools):
        for ci, c in enumerate(p["candidates"]):
            text = PROMPT_TMPL.format(facts=p["facts"] or "(none)",
                                      question=p["question"], passage=c["text"])
            ids = tok.encode(text, add_special_tokens=False)[-MAX_TOKENS:]
            jobs.append((pi, ci, ids))
    print(f"scoring {len(jobs)} candidates over {len(pools)} pools")
    scores = {}
    t0 = time.time()
    inflight = []
    done = 0
    for j, (pi, ci, ids) in enumerate(jobs):
        inflight.append(((pi, ci), sc.submit(ids)))
        if len(inflight) >= WINDOW or j == len(jobs) - 1:
            for key, pair in inflight:
                scores[key] = Scorer.resolve(pair)
            done += len(inflight)
            inflight = []
            rate = done / max(time.time() - t0, 1e-9)
            print(f"  {done}/{len(jobs)} ({rate:.1f} cand/s)", flush=True)
    ranked = []
    for pi, p in enumerate(pools):
        order = sorted(range(len(p["candidates"])),
                       key=lambda ci: scores[(pi, ci)], reverse=True)
        ranked.append([p["candidates"][ci]["id"] for ci in order])
    return ranked, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="trained,zeroshot")
    ap.add_argument("--max-pools", type=int, default=None)
    ap.add_argument("--dev-check", action="store_true")
    ap.add_argument("--train-info", default=f"{DATA}/train_info.json")
    args = ap.parse_args()

    pools = [json.loads(l) for l in open(f"{DATA}/pools_test.jsonl")]
    if args.max_pools:
        pools = pools[:args.max_pools]
    gold_in_pool = [p for p in pools
                    if any(g in {c['id'] for c in p['candidates']} for g in p['gold_ids'])]
    print(f"pools={len(pools)} gold-in-pool={len(gold_in_pool)} "
          f"(recall ceiling {len(gold_in_pool)/len(pools):.1%})")

    service = tinker.ServiceClient()
    res_path = f"{DATA}/eval_results.json"
    results = {"n_pools": len(pools), "n_gold_in_pool": len(gold_in_pool), "arms": {}}
    if os.path.exists(res_path):
        prev = json.load(open(res_path))
        if prev.get("n_pools") == len(pools):   # only merge same-scope runs
            results["arms"].update(prev.get("arms", {}))

    # reference arms straight from the signed caches
    for key, label in (("ce_top5", "CE (ms-marco, cached)"),
                       ("raw_top5", "raw-question top5 (cached)"),
                       ("scope_top5", "SCOPE top5 (cached)")):
        h, m, nh = cached_list_metrics(key, pools)
        results["arms"][label] = {"hit@5": h, "mrr@5": m, "hits": nh}
        print(f"{label:34s} Hit@5={h:.3f} MRR@5={m:.3f} ({nh}/{len(pools)})")

    for arm in args.arms.split(","):
        arm = arm.strip()
        if arm == "trained":
            info = json.load(open(args.train_info))
            client = service.create_sampling_client(model_path=info["sampler_path"])
        elif arm == "zeroshot":
            client = service.create_sampling_client(base_model=BASE_MODEL)
        else:
            continue
        tok = client.get_tokenizer()
        ranked, scores = score_pools(client, tok, pools)
        h, m, nh = metrics(ranked, pools)
        # conversion on the gold-in-pool subset
        sub = [(r, p) for r, p in zip(ranked, pools)
               if any(g in {c['id'] for c in p['candidates']} for g in p['gold_ids'])]
        hs, ms, nhs = metrics([r for r, _ in sub], [p for _, p in sub])
        results["arms"][f"judge-{arm}"] = {
            "hit@5": h, "mrr@5": m, "hits": nh,
            "gold_in_pool_hit@5": hs, "gold_in_pool_n": len(sub)}
        print(f"judge-{arm:26s} Hit@5={h:.3f} MRR@5={m:.3f} ({nh}/{len(pools)}) | "
              f"conversion on gold-in-pool: {hs:.3f} ({nhs}/{len(sub)})")
        json.dump({str(k): v for k, v in scores.items()},
                  open(f"{DATA}/scores_{arm}.json", "w"))

    json.dump(results, open(f"{DATA}/eval_results.json", "w"), indent=2)
    print("wrote", f"{DATA}/eval_results.json")


if __name__ == "__main__":
    main()
