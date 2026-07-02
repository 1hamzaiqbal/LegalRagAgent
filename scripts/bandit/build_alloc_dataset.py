#!/usr/bin/env python
"""Rung 2 dataset builder — internalized allocation predictor.

Turns the offline-bandit cells into (question, reader, strategy) -> Yes/No
training pairs for the EIT judge-lane recipe: the 9B learns to predict, from
the question text alone, whether a given reader succeeds under a given
retrieval strategy. Splits are IDENTICAL to rung 1 (imports build_cell and
reuses seed-0 shuffles) so policies are directly comparable.

Outputs (scripts/judge_pilot/data/):
  alloc_train.jsonl       train-half pairs, fields: prompt_text, label
  alloc_eval_pairs.jsonl  test-half pairs, fields: cell, question_idx,
                          action, prompt_text (scored on EIT, analyzed here)
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from offline_bandit_v0 import CELLS, JDATA, build_cell  # noqa: E402

READERS = {
    "barexam-70b": "Llama-3.3-70B (a large model)",
    "barexam-8b": "Llama-3.1-8B (a small model)",
    "housing-70b": "Llama-3.3-70B (a large model)",
    "housing-8b": "Llama-3.1-8B (a small model)",
    "medqa-70b": "Llama-3.3-70B (a large model)",
}
TASKS = {
    "barexam": "a bar-exam multiple-choice question",
    "housing": "a housing-law yes/no question",
    "medqa": "a medical-license multiple-choice question",
}
STRATEGIES = {
    "llm_only": "answer directly from its own knowledge, with no retrieved evidence",
    "judge": "answer using the top-5 corpus passages selected by a trained relevance judge",
    "ce": "answer using the top-5 corpus passages selected by a cross-encoder reranker",
    "scope": "answer using the top-5 corpus passages retrieved with a model-generated search passage",
    "raw": "answer using the top-5 corpus passages retrieved with the raw question",
    "hyde": "answer using the top-5 corpus passages retrieved with a hypothetical-document query",
}
MAX_Q_CHARS = 2400  # left-truncation in the trainer keeps the tail; strategy text goes last


def question_texts(cfg, kept):
    """question_idx -> display text (facts + question where pools carry them)."""
    if cfg["pools"]:
        pools = [json.loads(l) for l in open(os.path.join(JDATA, cfg["pools"]))]
        by_idx = {str(p["question_idx"]): p for p in pools}
        return {q: ((by_idx[q].get("facts") or "") + "\n" + by_idx[q]["question"]).strip()
                for q in kept}
    # medqa: take question text from the llm_only detail log
    from offline_bandit_v0 import load_rows
    rows = load_rows(cfg["llm_only"])
    return {q: rows[q]["question"] for q in kept}


def render(cell, qtext, action):
    task = TASKS[cell.split("-")[0]]
    return (
        f"Question ({task}):\n{qtext[-MAX_Q_CHARS:]}\n\n"
        f"Reader model: {READERS[cell]}. "
        f"Strategy: the reader will {STRATEGIES[action]}.\n"
        f"Will the reader answer this question correctly with this strategy? "
        f"Answer Yes or No.\nAnswer:"
    )


def main():
    train_rows, eval_rows = [], []
    for cell, cfg in CELLS.items():
        arms, X, R, K, kept = build_cell(cfg)
        n = len(kept)
        order = list(range(n))
        random.Random(0).shuffle(order)          # identical to rung 1
        tr, te = set(order[: n // 2]), set(order[n // 2:])
        qtexts = question_texts(cfg, kept)
        for i, q in enumerate(kept):
            for ai, a in enumerate(arms):
                row = {
                    "cell": cell,
                    "question_idx": q,
                    "action": a,
                    "prompt_text": render(cell, qtexts[q], a),
                }
                if i in tr:
                    row["label"] = "Yes" if R[i, ai] == 1.0 else "No"
                    train_rows.append(row)
                else:
                    eval_rows.append(row)
        print(f"{cell}: kept={n} train_q={len(tr)} test_q={len(te)} arms={arms}")

    random.Random(1).shuffle(train_rows)
    with open(os.path.join(JDATA, "alloc_train.jsonl"), "w") as f:
        for r in train_rows:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(JDATA, "alloc_eval_pairs.jsonl"), "w") as f:
        for r in eval_rows:
            f.write(json.dumps(r) + "\n")
    ys = sum(1 for r in train_rows if r["label"] == "Yes")
    print(f"train pairs={len(train_rows)} (Yes={ys}, No={len(train_rows)-ys}) "
          f"eval pairs={len(eval_rows)}")


if __name__ == "__main__":
    main()
