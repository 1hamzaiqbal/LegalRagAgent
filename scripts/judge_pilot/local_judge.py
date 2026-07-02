#!/usr/bin/env python
"""EIT-local judge lane: HF PEFT LoRA train + Yes/No logit scoring (no Tinker).

Mirrors train_tinker_judge.py / eval_judge_rerank.py semantics exactly:
  train: LoRA (r=32, attn+mlp proj) on next-token CE, loss on " Yes"/" No"
         answer tokens only, 3 epochs, lr 1e-4, effective batch 128.
  score: s = logit(" Yes") - logit(" No") at the final position, one forward
         per (pool, candidate); writes local_scores_<tag>.jsonl + a results
         summary identical in shape to the Tinker eval.

Usage (on an A40 node):
  python local_judge.py train --data-dir D --prefix "" --out-dir CKPT
  python local_judge.py score --data-dir D --prefix "" --ckpt CKPT --tag trained
  python local_judge.py score --data-dir D --prefix "" --tag zeroshot
"""
import argparse, json, os, random, time

import torch

BASE_MODEL = os.getenv("JUDGE_BASE_MODEL", "Qwen/Qwen3.5-9B")
PROMPTS = {
    "": ("You are a legal retrieval judge. Decide whether the passage states the "
         "legal rule needed to answer the bar-exam question.\n\n"
         "Fact pattern: {facts}\nQuestion: {question}\n\n"
         "Passage: {passage}\n\n"
         "Does this passage state the controlling legal rule for this question? "
         "Answer Yes or No.\nAnswer:"),
    "housing_": ("You are a legal retrieval judge for U.S. housing law. Decide whether the "
                 "statute passage provides the legal basis to answer the question for the "
                 "given state.\n\n{facts}\nQuestion: {question}\n\n"
                 "Statute passage: {passage}\n\n"
                 "Does this passage provide the controlling statutory basis to answer this "
                 "question? Answer Yes or No.\nAnswer:"),
}
MAX_TOKENS = 1024


def load_model(ckpt=None, train_mode=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    if ckpt:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, ckpt)
        model = model.merge_and_unload()
    elif train_mode:
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0, bias="none",
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                         "gate_proj", "up_proj", "down_proj"],
                         task_type="CAUSAL_LM")
        model = get_peft_model(model, cfg)
        model.print_trainable_parameters()
    return tok, model


def cmd_train(args):
    tok, model = load_model(train_mode=True)
    rows = [json.loads(l) for l in open(f"{args.data_dir}/{args.prefix}train.jsonl")]
    if args.smoke:
        rows = rows[:32]
    print(f"train examples: {len(rows)}", flush=True)
    model.gradient_checkpointing_enable()
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    rng = random.Random(0)
    micro, accum = 8, 16   # effective batch 128
    step = 0
    t0 = time.time()
    for epoch in range(1 if args.smoke else 3):
        rng.shuffle(rows)
        for i in range(0, len(rows), micro * accum):
            opt.zero_grad()
            chunk = rows[i:i + micro * accum]
            losses = []
            for j in range(0, len(chunk), micro):
                batch = chunk[j:j + micro]
                input_ids, labels = [], []
                for ex in batch:
                    p_ids = tok.encode(ex["prompt_text"], add_special_tokens=False)[-MAX_TOKENS:]
                    t_ids = tok.encode(" " + ex["label"], add_special_tokens=False)
                    input_ids.append(p_ids + t_ids)
                    labels.append([-100] * len(p_ids) + t_ids)
                maxlen = max(len(x) for x in input_ids)
                pad = tok.pad_token_id or tok.eos_token_id
                ids = torch.tensor([x + [pad] * (maxlen - len(x)) for x in input_ids], device="cuda")
                lab = torch.tensor([x + [-100] * (maxlen - len(x)) for x in labels], device="cuda")
                att = (ids != pad).long()
                out = model(input_ids=ids, attention_mask=att, labels=lab)
                (out.loss * len(batch) / len(chunk)).backward()
                losses.append(out.loss.item())
            opt.step()
            step += 1
            print(f"epoch {epoch} step {step} n={len(chunk)} "
                  f"mean_target_loss={sum(losses)/len(losses):.4f} "
                  f"({(time.time()-t0)/60:.1f}m)", flush=True)
    model.save_pretrained(args.out_dir)
    print("saved", args.out_dir, flush=True)


@torch.no_grad()
def cmd_score(args):
    tok, model = load_model(ckpt=args.ckpt)
    model.eval()
    yes_id = tok.encode(" Yes", add_special_tokens=False)[0]
    no_id = tok.encode(" No", add_special_tokens=False)[0]
    tmpl = PROMPTS[args.prefix]
    pools = [json.loads(l) for l in open(f"{args.data_dir}/{args.prefix}pools_test.jsonl")]
    jobs = []
    for pi, p in enumerate(pools):
        for ci, c in enumerate(p["candidates"]):
            jobs.append((pi, ci, tmpl.format(facts=p.get("facts") or "(none)",
                                             question=p["question"], passage=c["text"])))
    print(f"scoring {len(jobs)} candidates over {len(pools)} pools", flush=True)
    scores = {}
    B = 16
    t0 = time.time()
    pad = tok.pad_token_id or tok.eos_token_id
    for i in range(0, len(jobs), B):
        batch = jobs[i:i + B]
        enc = [tok.encode(t, add_special_tokens=False)[-MAX_TOKENS:] for _, _, t in batch]
        maxlen = max(len(x) for x in enc)
        ids = torch.tensor([[pad] * (maxlen - len(x)) + x for x in enc], device="cuda")
        att = torch.tensor([[0] * (maxlen - len(x)) + [1] * len(x) for x in enc], device="cuda")
        logits = model(input_ids=ids, attention_mask=att).logits[:, -1, :]
        for (pi, ci, _), row in zip(batch, logits):
            scores[f"({pi}, {ci})"] = float(row[yes_id] - row[no_id])
        if (i // B) % 50 == 0:
            print(f"  {min(i+B,len(jobs))}/{len(jobs)} ({(i+B)/max(time.time()-t0,1e-9):.1f}/s)", flush=True)
    out = f"{args.data_dir}/local_{args.prefix}scores_{args.tag}.json"
    json.dump(scores, open(out, "w"))
    # summary
    hit = mrr = 0
    for pi, p in enumerate(pools):
        order = sorted(range(len(p["candidates"])),
                       key=lambda ci: scores[f"({pi}, {ci})"], reverse=True)
        top5 = [p["candidates"][ci]["id"] for ci in order[:5]]
        gold = set(p["gold_ids"])
        if any(g in gold for g in top5):
            hit += 1
            mrr += 1.0 / next(i for i, pid in enumerate(top5, 1) if pid in gold)
    print(f"RESULT tag={args.tag} model={BASE_MODEL} ckpt={args.ckpt} "
          f"Hit@5={hit/len(pools):.4f} MRR@5={mrr/len(pools):.4f} ({hit}/{len(pools)})", flush=True)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--data-dir", required=True)
    tr.add_argument("--prefix", default="")
    tr.add_argument("--out-dir", required=True)
    tr.add_argument("--smoke", action="store_true")
    sc = sub.add_parser("score")
    sc.add_argument("--data-dir", required=True)
    sc.add_argument("--prefix", default="")
    sc.add_argument("--ckpt", default=None)
    sc.add_argument("--tag", required=True)
    args = ap.parse_args()
    {"train": cmd_train, "score": cmd_score}[args.cmd](args)
