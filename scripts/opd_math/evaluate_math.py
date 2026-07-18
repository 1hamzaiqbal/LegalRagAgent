#!/usr/bin/env python3
"""Evaluate a pinned base or LoRA model with repeated non-thinking math samples."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import random
import subprocess
import time
from pathlib import Path

import torch

try:
    from .data_contract import iter_jsonl
    from .math_reward import verify_completion
    from .quality_gates import sha256_tree
    from .tokenizer_contract import canonical_sha256, tokenizer_fingerprint
except ImportError:
    from data_contract import iter_jsonl  # type: ignore
    from math_reward import verify_completion  # type: ignore
    from quality_gates import sha256_tree  # type: ignore
    from tokenizer_contract import canonical_sha256, tokenizer_fingerprint  # type: ignore


ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_identity() -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        ).stdout
        return {"commit": commit, "worktree_clean": not status.strip()}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "worktree_clean": False}


def package_versions() -> dict[str, str]:
    names = ("torch", "transformers", "peft", "math-verify")
    return {name: importlib.metadata.version(name) for name in names}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--task-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--samples-per-problem", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--write-completions", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    if args.max_records < 0:
        raise ValueError("--max-records must be nonnegative")
    if args.samples_per_problem <= 0 or args.max_new_tokens <= 0:
        raise ValueError("sample count and completion length must be positive")
    if args.temperature <= 0 or not 0 < args.top_p <= 1 or args.top_k < 0:
        raise ValueError("invalid sampling contract")

    if args.output_dir.is_symlink() or (
        args.output_dir.exists()
        and (not args.output_dir.is_dir() or any(args.output_dir.iterdir()))
    ):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(iter_jsonl(args.task_file))
    if args.max_records > 0:
        rows = rows[: args.max_records]
    if not rows:
        raise ValueError("evaluation task file is empty")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.model_revision,
        local_files_only=args.local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.model_revision,
        local_files_only=args.local_files_only,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="sdpa",
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(args.adapter))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    sample_path = args.output_dir / "samples.jsonl"
    correct = attempted = parse_failed = 0
    unique_prompt_tokens = total_completion_tokens = 0
    total_generation_latency = 0.0
    with sample_path.open("w", encoding="utf-8") as handle, torch.inference_mode():
        for row_index, row in enumerate(rows):
            messages = row.get("prompt")
            if not isinstance(messages, list):
                raise ValueError(f"row {row_index} lacks conversational prompt")
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to(device)
            prompt_width = inputs["input_ids"].shape[1]
            generation_started = time.perf_counter()
            generated = model.generate(
                **inputs,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=args.max_new_tokens,
                num_return_sequences=args.samples_per_problem,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            generation_latency = time.perf_counter() - generation_started
            total_generation_latency += generation_latency
            unique_prompt_tokens += prompt_width
            for sample_idx, output_ids in enumerate(generated):
                completion_ids = output_ids[prompt_width:].detach().cpu().tolist()
                if tokenizer.eos_token_id in completion_ids:
                    completion_ids = completion_ids[: completion_ids.index(tokenizer.eos_token_id) + 1]
                completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
                verdict = verify_completion(completion, row["solution"])
                if verdict["status"] in ("gold_parse_failed", "verifier_error"):
                    raise RuntimeError(f"evaluation verifier failure for {row.get('record_id')}: {verdict}")
                reward = float(verdict["reward"])
                attempted += 1
                correct += int(reward)
                parse_failed += int(verdict["status"] == "prediction_parse_failed")
                total_completion_tokens += len(completion_ids)
                result = {
                    "schema_version": 1,
                    "record_id": row.get("record_id"),
                    "cluster_id": row.get("cluster_id"),
                    "source": row.get("source"),
                    "sample_idx": sample_idx,
                    "reward": reward,
                    "reward_status": verdict["status"],
                    "completion_tokens": len(completion_ids),
                    "prompt_tokens": prompt_width,
                    "generation_batch_latency_seconds": generation_latency,
                    "completion_sha256": hashlib.sha256(completion.encode("utf-8")).hexdigest(),
                }
                if args.write_completions:
                    result["completion_text"] = completion
                handle.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")

    summary = {
        "schema_version": 1,
        "model": args.model,
        "model_revision": args.model_revision,
        "code": {
            "git": git_identity(),
            "evaluator_file_sha256": sha256_file(Path(__file__)),
            "packages": package_versions(),
        },
        "tokenizer_contract_sha256": canonical_sha256(tokenizer_fingerprint(tokenizer)),
        "adapter": None if args.adapter is None else str(args.adapter.resolve()),
        "adapter_tree_sha256": None if args.adapter is None else sha256_tree(args.adapter),
        "task_file": str(args.task_file.resolve()),
        "task_file_sha256": sha256_file(args.task_file),
        "records": len(rows),
        "task_sources": sorted({str(row.get("source")) for row in rows}),
        "task_roles": sorted({str(row.get("role")) for row in rows}),
        "samples_per_problem": args.samples_per_problem,
        "samples": attempted,
        "accuracy": correct / attempted,
        "prediction_parse_failure_fraction": parse_failed / attempted,
        "unique_prompt_tokens": unique_prompt_tokens,
        "expanded_prompt_tokens": unique_prompt_tokens * args.samples_per_problem,
        "total_completion_tokens": total_completion_tokens,
        "total_generation_latency_seconds": total_generation_latency,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
        ),
        "decoding": {
            "thinking": False,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
        },
        "samples_file": str(sample_path.resolve()),
        "samples_file_sha256": sha256_file(sample_path),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
