#!/usr/bin/env python
"""Train the BarExamQA relevance judge via Tinker LoRA SFT (Path C pilot v0).

Reads scripts/judge_pilot/data/{train,dev}.jsonl (from build_judge_dataset.py),
fine-tunes BASE_MODEL with LoRA on next-token cross-entropy where only the
" Yes"/" No" answer tokens carry loss weight, then saves sampler weights and
records the tinker checkpoint path for eval_judge_rerank.py.

Usage:
  set -a; source .env; set +a
  .venv/bin/python scripts/judge_pilot/train_tinker_judge.py [--smoke]
"""
import argparse, json, os, random, sys, time

import tinker
from tinker import types as T

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
BASE_MODEL = "Qwen/Qwen3.5-9B"
LORA_RANK = 32
LR = 1e-4
EPOCHS = 3
BATCH = 128
MAX_TOKENS = 1024   # prompt truncation guard


def load_jsonl(p):
    return [json.loads(l) for l in open(p)]


def make_datum(tok, ex):
    # loss only on the answer token(s)
    prompt_ids = tok.encode(ex["prompt_text"], add_special_tokens=False)[-MAX_TOKENS:]
    target_ids = tok.encode(" " + ex["label"], add_special_tokens=False)
    full = prompt_ids + target_ids
    weights = [0.0] * len(prompt_ids) + [1.0] * len(target_ids)
    return T.Datum(
        model_input=T.ModelInput.from_ints(full[:-1]),
        loss_fn_inputs={"weights": weights[1:], "target_tokens": full[1:]},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="8 examples, 1 step")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--prefix", default="", help="dataset prefix, e.g. housing_")
    args = ap.parse_args()

    train = load_jsonl(f"{DATA}/{args.prefix}train.jsonl")
    if args.smoke:
        train = train[:8]
    print(f"train examples: {len(train)}")

    sc = tinker.ServiceClient()
    tc = sc.create_lora_training_client(base_model=BASE_MODEL, rank=LORA_RANK)
    tok = tc.get_tokenizer()

    # sanity: single-token targets?
    for lab in (" Yes", " No"):
        print(f"target {lab!r} -> tokens {tok.encode(lab, add_special_tokens=False)}")

    rng = random.Random(0)
    step = 0
    t0 = time.time()
    for epoch in range(1 if args.smoke else args.epochs):
        rng.shuffle(train)
        for i in range(0, len(train), BATCH):
            batch = [make_datum(tok, ex) for ex in train[i:i + BATCH]]
            fb = tc.forward_backward(batch, loss_fn="cross_entropy")
            op = tc.optim_step(T.AdamParams(learning_rate=LR))
            fb_res, _ = fb.result(), op.result()
            # mean loss over weighted tokens
            losses = []
            for out in fb_res.loss_fn_outputs:
                v = out.get("elementwise_loss") if hasattr(out, "get") else None
                if v is not None:
                    data = v.data if hasattr(v, "data") else v
                    losses.extend([x for x in data if x])
            step += 1
            n_w = sum(1 for _ in batch)
            msg = f"epoch {epoch} step {step} n={n_w}"
            if losses:
                msg += f" mean_target_loss={sum(losses)/len(losses):.4f}"
            print(msg, flush=True)

    name = (args.prefix or "barexam-") .rstrip("_") + "-judge-v0" + ("-smoke" if args.smoke else "")
    name = name.replace("--", "-")
    sampling_client = tc.save_weights_and_get_sampling_client(name=name)
    # persist the checkpoint path for the eval script
    info = {
        "base_model": BASE_MODEL,
        "rank": LORA_RANK,
        "lr": LR,
        "epochs": 1 if args.smoke else args.epochs,
        "train_examples": len(train),
        "wall_s": round(time.time() - t0, 1),
        "sampler_path": getattr(sampling_client, "model_path", None) or str(sampling_client),
    }
    # try the documented way to get a durable path
    try:
        resp = tc.save_weights_for_sampler(name=name + "-final").result()
        info["sampler_path"] = resp.path
    except Exception as e:
        print("save_weights_for_sampler fallback:", e)
    out = f"{DATA}/{args.prefix}train_info{'_smoke' if args.smoke else ''}.json"
    json.dump(info, open(out, "w"), indent=2)
    print("saved", out, "->", info["sampler_path"])


if __name__ == "__main__":
    main()
