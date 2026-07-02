#!/usr/bin/env python
"""Single-GPU student trainer for on-policy distillation.

The teacher is a separate vLLM OpenAI-compatible server. This process loads the
student, samples completions on-policy, asks the teacher server for per-token
logprobs on those exact completions, and applies the OPD policy-gradient loss:

  A_t = logp_teacher_t - stopgrad(logp_student_t)
  loss = -mean_t A_t * logp_student_t

Teacher and student must share a tokenizer family because OPD aligns per-token
logprobs. Use Qwen3-to-Qwen3 or Llama-3.x-to-Llama-3.x pairs.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch

try:
    from . import teacher_client
    from .opd_loss import kd_forward_loss, opd_policy_loss, reverse_kl_estimate
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import teacher_client
    from opd_loss import kd_forward_loss, opd_policy_loss, reverse_kl_estimate


LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def log(msg: str) -> None:
    print(msg, flush=True)


def read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "prompt_text" not in row:
                raise ValueError(f"{path}:{line_no} missing required prompt_text")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} contained no task rows")
    return rows


def prompt_stream(rows: list[dict], rng: random.Random):
    rows = list(rows)
    while True:
        rng.shuffle(rows)
        for row in rows:
            yield row


def encode(tokenizer, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def maybe_truncate_prompt(tokenizer, text: str, max_prompt_tokens: int) -> str:
    ids = encode(tokenizer, text)
    if max_prompt_tokens > 0 and len(ids) > max_prompt_tokens:
        ids = ids[-max_prompt_tokens:]
        return tokenizer.decode(ids, skip_special_tokens=False)
    return text


def load_student(args, device: str):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.student)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.student, torch_dtype=dtype)
    model.to(device)
    model.config.use_cache = False

    if args.lora > 0:
        cfg = LoraConfig(
            r=args.lora,
            lora_alpha=2 * args.lora,
            lora_dropout=0.0,
            bias="none",
            target_modules=LORA_TARGETS,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, cfg)
        model.print_trainable_parameters()

    if os.getenv("OPD_GRAD_CKPT", "1") not in ("0", "false", "False"):
        model.gradient_checkpointing_enable()
    return tok, model


@torch.no_grad()
def generate_student_samples(model, tok, prompts: list[str], args, device: str) -> list[dict]:
    model.eval()
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    enc = {k: v.to(device) for k, v in enc.items()}
    prompt_width = enc["input_ids"].shape[1]
    gen = model.generate(
        **enc,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        num_return_sequences=args.group_size,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    samples = []
    for i, row in enumerate(gen):
        prompt_idx = i // args.group_size
        comp_ids = row[prompt_width:].detach().cpu().tolist()
        comp_ids = [t for t in comp_ids if t != tok.pad_token_id]
        if tok.eos_token_id in comp_ids:
            comp_ids = comp_ids[:comp_ids.index(tok.eos_token_id)]
        completion = tok.decode(comp_ids, skip_special_tokens=True)
        if not encode(tok, completion):
            continue
        samples.append({"prompt_text": prompts[prompt_idx], "completion_text": completion})
    if not samples:
        raise RuntimeError("student generated no non-empty completions")
    return samples


def teacher_score_samples(tok, samples: list[dict], args) -> list[dict]:
    scored = []
    timeout = (args.teacher_connect_timeout, args.teacher_read_timeout)
    for sample in samples:
        lps = teacher_client.score_completion_logprobs(
            args.teacher_url,
            args.teacher_model,
            sample["prompt_text"],
            sample["completion_text"],
            tok,
            timeout=timeout,
            retries=args.teacher_retries,
        )
        comp_len = len(encode(tok, sample["completion_text"]))
        if len(lps) != comp_len:
            raise RuntimeError(f"teacher/student token count mismatch: teacher={len(lps)} student={comp_len}")
        row = dict(sample)
        row["teacher_logprobs"] = lps
        scored.append(row)
    return scored


def teacher_kd_samples(rows: list[dict], args) -> list[dict]:
    timeout = (args.teacher_connect_timeout, args.teacher_read_timeout)
    out = []
    for row in rows:
        prompt = row["prompt_text"]
        if row.get("completion_text"):
            completions = [str(row["completion_text"])]
        else:
            completions = teacher_client.sample_from_server(
                args.teacher_url,
                args.teacher_model,
                prompt,
                max_tokens=args.max_new_tokens,
                temperature=args.kd_temperature,
                top_p=args.top_p,
                n=args.group_size,
                timeout=timeout,
                retries=args.teacher_retries,
            )
        for completion in completions:
            if completion:
                out.append({"prompt_text": prompt, "completion_text": completion})
    if not out:
        raise RuntimeError("KD mode produced no non-empty teacher completions")
    return out


def build_batch(tok, samples: list[dict], device: str, require_teacher: bool):
    pad = tok.pad_token_id
    seqs, label_masks, teacher_lps = [], [], []
    for sample in samples:
        p_ids = encode(tok, sample["prompt_text"])
        c_ids = encode(tok, sample["completion_text"])
        if not p_ids:
            raise ValueError("prompt tokenization is empty; cannot score first completion token")
        if not c_ids:
            continue
        seqs.append(p_ids + c_ids)
        label_masks.append([False] * len(p_ids) + [True] * len(c_ids))
        if require_teacher:
            lps = sample.get("teacher_logprobs")
            if lps is None:
                raise ValueError("OPD sample missing teacher_logprobs")
            if len(lps) != len(c_ids):
                raise ValueError(f"teacher logprobs length {len(lps)} != completion tokens {len(c_ids)}")
            teacher_lps.append(list(lps))

    if not seqs:
        raise ValueError("empty training batch after tokenization")

    max_len = max(len(x) for x in seqs)
    max_comp = max(sum(m) for m in label_masks)
    ids = torch.full((len(seqs), max_len), pad, dtype=torch.long, device=device)
    att = torch.zeros((len(seqs), max_len), dtype=torch.long, device=device)
    label_mask = torch.zeros((len(seqs), max_len), dtype=torch.bool, device=device)
    teacher = torch.zeros((len(seqs), max_comp), dtype=torch.float32, device=device)
    comp_mask = torch.zeros((len(seqs), max_comp), dtype=torch.bool, device=device)

    for i, seq in enumerate(seqs):
        n = len(seq)
        ids[i, :n] = torch.tensor(seq, dtype=torch.long, device=device)
        att[i, :n] = 1
        label_mask[i, :n] = torch.tensor(label_masks[i], dtype=torch.bool, device=device)
        comp_len = sum(label_masks[i])
        comp_mask[i, :comp_len] = True
        if require_teacher:
            teacher[i, :comp_len] = torch.tensor(teacher_lps[i], dtype=torch.float32, device=device)

    return ids, att, label_mask, teacher, comp_mask


def current_completion_logprobs(logits: torch.Tensor, input_ids: torch.Tensor, label_mask: torch.Tensor):
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = label_mask[:, 1:]
    log_probs = torch.log_softmax(shift_logits.float(), dim=-1)
    token_logps = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

    max_comp = int(shift_mask.sum(dim=1).max().item())
    out = token_logps.new_zeros((input_ids.shape[0], max_comp))
    mask = torch.zeros((input_ids.shape[0], max_comp), dtype=torch.bool, device=input_ids.device)
    for i in range(input_ids.shape[0]):
        vals = token_logps[i][shift_mask[i]]
        out[i, : vals.numel()] = vals
        mask[i, : vals.numel()] = True
    return out, mask


def kd_loss_for_samples(model, tok, samples: list[dict], device: str) -> tuple[torch.Tensor, int]:
    ids, att, label_mask, _, _ = build_batch(tok, samples, device, require_teacher=False)
    out = model(input_ids=ids, attention_mask=att)
    labels = ids.masked_fill(~label_mask, -100)
    shift_logits = out.logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    loss = kd_forward_loss(shift_logits.float(), shift_labels)
    return loss, int(label_mask.sum().item())


def opd_loss_for_samples(model, tok, samples: list[dict], args, device: str):
    ids, att, label_mask, teacher, teacher_mask = build_batch(tok, samples, device, require_teacher=True)
    out = model(input_ids=ids, attention_mask=att)
    student_lps, student_mask = current_completion_logprobs(out.logits, ids, label_mask)
    if student_lps.shape != teacher.shape:
        raise RuntimeError(f"student/teacher logprob tensor mismatch: {student_lps.shape} vs {teacher.shape}")
    mask = student_mask & teacher_mask
    loss = opd_policy_loss(
        student_lps,
        teacher,
        mask,
        advantage_clip=args.advantage_clip,
        ratio_clip_eps=args.ratio_clip_eps,
    )
    rkl = reverse_kl_estimate(student_lps, teacher, mask)
    return loss, rkl, int(mask.sum().item())


def save_checkpoint(model, tok, out_dir: str, name: str) -> None:
    path = Path(out_dir) / name
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tok.save_pretrained(path)
    log(f"saved {path}")


def run(args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device={device} student={args.student} mode={args.mode}")
    teacher_client.healthcheck(
        args.teacher_url,
        timeout=args.teacher_read_timeout,
        retries=args.teacher_retries,
        raise_on_error=True,
    )

    tok, model = load_student(args, device)
    rows = read_jsonl(args.task_file)
    rng = random.Random(args.seed)
    stream = prompt_stream(rows, rng)
    micro = int(os.getenv("JUDGE_MICRO", "1"))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for step in range(1, args.steps + 1):
        raw_batch = [next(stream) for _ in range(micro)]
        prompt_rows = []
        for row in raw_batch:
            item = dict(row)
            item["prompt_text"] = maybe_truncate_prompt(tok, str(item["prompt_text"]), args.max_prompt_tokens)
            prompt_rows.append(item)

        model.train()
        opt.zero_grad(set_to_none=True)
        if args.mode == "opd":
            prompts = [r["prompt_text"] for r in prompt_rows]
            samples = generate_student_samples(model, tok, prompts, args, device)
            samples = teacher_score_samples(tok, samples, args)
            model.train()
            loss, rkl, ntok = opd_loss_for_samples(model, tok, samples, args, device)
            metric = f"reverse_kl={rkl.item():.4f}"
        else:
            samples = teacher_kd_samples(prompt_rows, args)
            model.train()
            loss, ntok = kd_loss_for_samples(model, tok, samples, device)
            metric = "reverse_kl=NA"

        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        elapsed = max(time.time() - t0, 1e-9)
        log(
            f"step {step}/{args.steps} mode={args.mode} prompts={len(prompt_rows)} "
            f"samples={len(samples)} tokens={ntok} loss={loss.item():.4f} "
            f"{metric} toks_per_s={ntok / elapsed:.1f}"
        )
        t0 = time.time()

        if args.save_every > 0 and step % args.save_every == 0:
            save_checkpoint(model, tok, args.out_dir, f"step_{step:06d}")

    save_checkpoint(model, tok, args.out_dir, "final")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["opd", "kd"], default="opd")
    ap.add_argument("--task-file", required=True)
    ap.add_argument("--student", required=True)
    ap.add_argument("--teacher-url", required=True)
    ap.add_argument("--teacher-model", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--max-prompt-tokens", type=int, default=1536)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lora", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--kd-temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--advantage-clip", type=float, default=5.0)
    ap.add_argument("--ratio-clip-eps", type=float, default=0.2)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--teacher-connect-timeout", type=float, default=10.0)
    ap.add_argument("--teacher-read-timeout", type=float, default=120.0)
    ap.add_argument("--teacher-retries", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
