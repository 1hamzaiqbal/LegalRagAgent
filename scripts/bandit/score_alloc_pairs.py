#!/usr/bin/env python
"""Score allocation eval pairs with the (trained or zero-shot) 9B on EIT.

s = logit(" Yes") - logit(" No") at the final position per pair. Writes
{data_dir}/alloc_scores_{tag}.jsonl with cell/question_idx/action/score.
"""
import argparse
import json
import time

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--base", default="Qwen/Qwen3.5-9B")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="cuda")
    if args.ckpt:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.ckpt).merge_and_unload()
    model.eval()

    yes_id = tok.encode(" Yes", add_special_tokens=False)[0]
    no_id = tok.encode(" No", add_special_tokens=False)[0]
    pairs = [json.loads(l) for l in open(f"{args.data_dir}/alloc_eval_pairs.jsonl")]
    print(f"scoring {len(pairs)} pairs tag={args.tag}", flush=True)

    out_path = f"{args.data_dir}/alloc_scores_{args.tag}.jsonl"
    pad = tok.pad_token_id or tok.eos_token_id
    B, MAXT = 16, 1024
    t0 = time.time()
    with open(out_path, "w") as out, torch.no_grad():
        for i in range(0, len(pairs), B):
            batch = pairs[i:i + B]
            enc = [tok.encode(p["prompt_text"], add_special_tokens=False)[-MAXT:] for p in batch]
            maxlen = max(len(x) for x in enc)
            ids = torch.tensor([[pad] * (maxlen - len(x)) + x for x in enc], device="cuda")
            att = torch.tensor([[0] * (maxlen - len(x)) + [1] * len(x) for x in enc], device="cuda")
            logits = model(input_ids=ids, attention_mask=att).logits[:, -1, :]
            for p, row in zip(batch, logits):
                out.write(json.dumps({
                    "cell": p["cell"], "question_idx": p["question_idx"],
                    "action": p["action"],
                    "score": float(row[yes_id] - row[no_id]),
                }) + "\n")
            if (i // B) % 50 == 0:
                print(f"  {min(i+B,len(pairs))}/{len(pairs)} "
                      f"({(i+B)/max(time.time()-t0,1e-9):.1f}/s)", flush=True)
    print(f"RESULT tag={args.tag} wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
