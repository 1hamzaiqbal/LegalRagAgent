#!/usr/bin/env python3
"""Evaluate one immutable contiguous shard of a pinned math task.

Every record receives a deterministic seed derived from the task-file hash,
global record index, and record ID.  Consequently, changing shard geometry or
retrying a failed shard cannot change the random stream assigned to a record.
Outputs are written to a fresh partial directory and atomically promoted only
after code, task, adapter, package, and Git custody remain stable.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

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
SUMMARY_SCHEMA_VERSION = 2
SAMPLE_SCHEMA_VERSION = 2
EVALUATION_CONTRACT = "opd_math_evaluation_contract_v1"
EVALUATION_SHARD_KIND = "opd_math_evaluation_shard_v1"
EVALUATION_MERGED_KIND = "opd_math_evaluation_merged_v1"
RECORD_SEED_STRATEGY = "task_hash_global_index_record_id_sha256_v1"
SHARD_STRATEGY = "contiguous_balanced_v1"
MERGE_STRATEGY = "ordered_contiguous_shards_v1"
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")


def sha256_file(path: Path) -> str:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected a regular non-symlink file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_identity() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain=v1"],
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


def record_sampling_seed(
    base_seed: int,
    task_file_sha256: str,
    global_record_index: int,
    record_id: str,
) -> int:
    """Derive a stable record seed that is independent of shard geometry."""

    if type(base_seed) is not int or base_seed < 0:
        raise ValueError("base seed must be a nonnegative integer")
    if not isinstance(task_file_sha256, str) or HEX64.fullmatch(task_file_sha256) is None:
        raise ValueError("task_file_sha256 must be a lowercase SHA-256 digest")
    if type(global_record_index) is not int or global_record_index < 0:
        raise ValueError("global_record_index must be a nonnegative integer")
    if not isinstance(record_id, str) or not record_id:
        raise ValueError("record_id must be a non-empty string")
    payload = {
        "strategy": RECORD_SEED_STRATEGY,
        "base_seed": base_seed,
        "task_file_sha256": task_file_sha256,
        "global_record_index": global_record_index,
        "record_id": record_id,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def balanced_shard_bounds(total: int, shard_count: int, shard_index: int) -> tuple[int, int]:
    """Return the exact contiguous slice assigned to one shard."""

    if type(total) is not int or total <= 0:
        raise ValueError("total must be a positive integer")
    if type(shard_count) is not int or shard_count <= 0 or shard_count > total:
        raise ValueError("shard count must be between one and the eligible record count")
    if type(shard_index) is not int or shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard index must be in [0, shard_count)")
    return total * shard_index // shard_count, total * (shard_index + 1) // shard_count


def _checked_record_ids(rows: list[dict[str, Any]], label: str) -> list[str]:
    record_ids: list[str] = []
    for index, row in enumerate(rows):
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"{label} row {index} lacks a stable record_id")
        record_ids.append(record_id)
    if len(record_ids) != len(set(record_ids)):
        raise ValueError(f"{label} contains duplicate record IDs")
    return record_ids


def _checked_adapter(path: Path | None) -> tuple[str | None, str | None]:
    if path is None:
        return None, None
    raw = Path(path).expanduser()
    if raw.is_symlink() or not raw.is_dir():
        raise ValueError(f"adapter must be a regular non-symlink directory: {raw}")
    resolved = raw.resolve()
    return str(resolved), sha256_tree(resolved)


def capture_evaluator_custody(task_file: Path, adapter: Path | None) -> dict[str, Any]:
    adapter_path, adapter_hash = _checked_adapter(adapter)
    return {
        "git": git_identity(),
        "evaluator_file_sha256": sha256_file(Path(__file__).resolve()),
        "packages": package_versions(),
        "task_file": str(Path(task_file).resolve()),
        "task_file_sha256": sha256_file(Path(task_file)),
        "adapter": adapter_path,
        "adapter_tree_sha256": adapter_hash,
    }


def require_clean_stable_custody(
    start: Mapping[str, Any], end: Mapping[str, Any], *, label: str
) -> None:
    start_git = start.get("git")
    end_git = end.get("git")
    if not isinstance(start_git, dict) or not isinstance(end_git, dict):
        raise RuntimeError(f"{label} lacks Git custody")
    commit = start_git.get("commit")
    if not isinstance(commit, str) or HEX40.fullmatch(commit) is None:
        raise RuntimeError(f"{label} lacks an immutable Git commit")
    if start_git.get("worktree_clean") is not True or end_git.get("worktree_clean") is not True:
        raise RuntimeError(f"{label} requires a clean Git worktree at start and end")
    if dict(start) != dict(end):
        changed = sorted(
            key for key in set(start) | set(end) if start.get(key) != end.get(key)
        )
        raise RuntimeError(f"{label} custody changed during execution: {changed}")


def evaluation_contract(
    *,
    model: str,
    model_revision: str,
    adapter: str | None,
    adapter_tree_sha256: str | None,
    task_file: str,
    task_file_sha256: str,
    eligible_record_ids: list[str],
    task_sources: list[str],
    task_roles: list[str],
    samples_per_problem: int,
    decoding: Mapping[str, Any],
    shard_count: int,
    tokenizer_contract_sha256: str,
    custody: Mapping[str, Any],
) -> dict[str, Any]:
    git = custody.get("git")
    if not isinstance(git, dict):
        raise ValueError("evaluation contract lacks Git custody")
    return {
        "schema_version": 1,
        "contract": EVALUATION_CONTRACT,
        "model": model,
        "model_revision": model_revision,
        "adapter": adapter,
        "adapter_tree_sha256": adapter_tree_sha256,
        "task_file": task_file,
        "task_file_sha256": task_file_sha256,
        "eligible_records": len(eligible_record_ids),
        "eligible_record_ids_sha256": canonical_sha256(eligible_record_ids),
        "task_sources": task_sources,
        "task_roles": task_roles,
        "samples_per_problem": samples_per_problem,
        "decoding": dict(decoding),
        "record_seed_contract": {
            "strategy": RECORD_SEED_STRATEGY,
            "base_seed": decoding["seed"],
        },
        "shard": {"strategy": SHARD_STRATEGY, "shard_count": shard_count},
        "tokenizer_contract_sha256": tokenizer_contract_sha256,
        "code": {
            "git_commit": git["commit"],
            "evaluator_file_sha256": custody["evaluator_file_sha256"],
            "packages": custody["packages"],
        },
    }


def custody_manifest(start: Mapping[str, Any], end: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "git_start": start["git"],
        "git_end": end["git"],
        "evaluator_file_sha256_start": start["evaluator_file_sha256"],
        "evaluator_file_sha256_end": end["evaluator_file_sha256"],
        "packages_start": start["packages"],
        "packages_end": end["packages"],
        "task_file_sha256_start": start["task_file_sha256"],
        "task_file_sha256_end": end["task_file_sha256"],
        "adapter_tree_sha256_start": start["adapter_tree_sha256"],
        "adapter_tree_sha256_end": end["adapter_tree_sha256"],
        "stable": True,
    }


def begin_transactional_directory(final_path: Path) -> tuple[Path, Path]:
    """Create a fresh sibling partial directory for an absent final path."""

    raw = Path(final_path).expanduser()
    if raw.is_symlink() or raw.exists():
        raise FileExistsError(f"refusing to overwrite evaluation output: {raw}")
    parent = raw.parent.resolve()
    parent.mkdir(parents=True, exist_ok=True)
    final = parent / raw.name
    if final.is_symlink() or final.exists():
        raise FileExistsError(f"refusing to overwrite evaluation output: {final}")
    partial = Path(tempfile.mkdtemp(prefix=f".{final.name}.partial.", dir=parent))
    return final, partial


def promote_transactional_directory(partial: Path, final: Path) -> None:
    """Atomically promote one completed directory without replacing a peer."""

    partial = Path(partial)
    final = Path(final)
    if partial.is_symlink() or not partial.is_dir():
        raise ValueError(f"partial output is not a regular directory: {partial}")
    lock = final.parent / f".{final.name}.promotion.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise FileExistsError(f"another process is promoting {final}") from exc
    try:
        if final.is_symlink() or final.exists():
            raise FileExistsError(f"refusing to replace completed output: {final}")
        partial.rename(final)
    finally:
        os.close(fd)
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def write_text_fsync(path: Path, content: str) -> None:
    with Path(path).open("x", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_records < 0:
        raise ValueError("--max-records must be nonnegative")
    if args.seed < 0:
        raise ValueError("--seed must be nonnegative")
    if args.samples_per_problem <= 0 or args.max_new_tokens <= 0:
        raise ValueError("sample count and completion length must be positive")
    if args.temperature <= 0 or not 0 < args.top_p <= 1 or args.top_k < 0:
        raise ValueError("invalid sampling contract")

    task_file = Path(args.task_file).expanduser()
    if task_file.is_symlink() or not task_file.is_file():
        raise ValueError(f"task file must be a regular non-symlink file: {task_file}")
    task_file = task_file.resolve()
    adapter = None if args.adapter is None else Path(args.adapter).expanduser().resolve()
    custody_start = capture_evaluator_custody(task_file, adapter)
    require_clean_stable_custody(custody_start, custody_start, label="evaluation start")

    all_rows = list(iter_jsonl(task_file))
    if args.max_records > 0:
        all_rows = all_rows[: args.max_records]
    if not all_rows:
        raise ValueError("evaluation task file is empty")
    eligible_record_ids = _checked_record_ids(all_rows, "selected evaluation task")
    global_records = len(all_rows)
    record_start, record_stop = balanced_shard_bounds(
        global_records, args.shard_count, args.shard_index
    )
    rows = all_rows[record_start:record_stop]
    shard_record_ids = eligible_record_ids[record_start:record_stop]

    final_output, partial_output = begin_transactional_directory(Path(args.output_dir))

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
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    decoding = {
        "thinking": False,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
    }
    tokenizer_hash = canonical_sha256(tokenizer_fingerprint(tokenizer))
    task_sources = sorted({str(row.get("source")) for row in all_rows})
    task_roles = sorted({str(row.get("role")) for row in all_rows})
    contract = evaluation_contract(
        model=args.model,
        model_revision=args.model_revision,
        adapter=custody_start["adapter"],
        adapter_tree_sha256=custody_start["adapter_tree_sha256"],
        task_file=str(task_file),
        task_file_sha256=custody_start["task_file_sha256"],
        eligible_record_ids=eligible_record_ids,
        task_sources=task_sources,
        task_roles=task_roles,
        samples_per_problem=args.samples_per_problem,
        decoding=decoding,
        shard_count=args.shard_count,
        tokenizer_contract_sha256=tokenizer_hash,
        custody=custody_start,
    )
    contract_hash = canonical_sha256(contract)

    sample_path = partial_output / "samples.jsonl"
    correct = attempted = parse_failed = 0
    unique_prompt_tokens = total_completion_tokens = 0
    total_generation_latency = 0.0
    with sample_path.open("x", encoding="utf-8") as handle, torch.inference_mode():
        for local_row_index, row in enumerate(rows):
            global_record_index = record_start + local_row_index
            messages = row.get("prompt")
            if not isinstance(messages, list):
                raise ValueError(f"row {global_record_index} lacks conversational prompt")
            record_id = shard_record_ids[local_row_index]
            record_seed = record_sampling_seed(
                args.seed,
                custody_start["task_file_sha256"],
                global_record_index,
                record_id,
            )
            random.seed(record_seed)
            torch.manual_seed(record_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(record_seed)
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to(device)
            prompt_width = int(inputs["input_ids"].shape[1])
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
                if not completion_ids:
                    raise RuntimeError(f"empty completion for record {record_id}")
                completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
                verdict = verify_completion(completion, row["solution"])
                if verdict["status"] in ("gold_parse_failed", "verifier_error"):
                    raise RuntimeError(f"evaluation verifier failure for {record_id}: {verdict}")
                reward = float(verdict["reward"])
                attempted += 1
                correct += int(reward)
                parse_failed += int(verdict["status"] == "prediction_parse_failed")
                total_completion_tokens += len(completion_ids)
                result = {
                    "schema_version": SAMPLE_SCHEMA_VERSION,
                    "record_id": record_id,
                    "global_record_index": global_record_index,
                    "record_seed": record_seed,
                    "cluster_id": row.get("cluster_id"),
                    "source": row.get("source"),
                    "sample_idx": sample_idx,
                    "reward": reward,
                    "reward_status": verdict["status"],
                    "completion_tokens": len(completion_ids),
                    "prompt_tokens": prompt_width,
                    "generation_batch_latency_seconds": generation_latency,
                    "completion_sha256": hashlib.sha256(
                        completion.encode("utf-8")
                    ).hexdigest(),
                }
                if args.write_completions:
                    result["completion_text"] = completion
                handle.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    custody_end = capture_evaluator_custody(task_file, adapter)
    require_clean_stable_custody(
        custody_start, custody_end, label="evaluation start/end"
    )
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "artifact_kind": EVALUATION_SHARD_KIND,
        "evaluation_contract": contract,
        "evaluation_contract_sha256": contract_hash,
        "model": args.model,
        "model_revision": args.model_revision,
        "code": {
            "git": custody_end["git"],
            "evaluator_file_sha256": custody_end["evaluator_file_sha256"],
            "packages": custody_end["packages"],
        },
        "custody": custody_manifest(custody_start, custody_end),
        "tokenizer_contract_sha256": tokenizer_hash,
        "adapter": custody_end["adapter"],
        "adapter_tree_sha256": custody_end["adapter_tree_sha256"],
        "task_file": str(task_file),
        "task_file_sha256": custody_end["task_file_sha256"],
        "records": len(rows),
        "eligible_records": global_records,
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
        "decoding": decoding,
        "record_seed_contract": contract["record_seed_contract"],
        "shard": {
            "strategy": SHARD_STRATEGY,
            "shard_count": args.shard_count,
            "shard_index": args.shard_index,
            "global_records": global_records,
            "record_start": record_start,
            "record_stop": record_stop,
            "selected_record_ids_sha256": canonical_sha256(shard_record_ids),
        },
        "completion_text_in_samples": bool(args.write_completions),
        "samples_file": "samples.jsonl",
        "samples_file_sha256": sha256_file(sample_path),
    }
    write_text_fsync(
        partial_output / "summary.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    promote_transactional_directory(partial_output, final_output)
    return summary


def parse_args() -> argparse.Namespace:
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
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--write-completions", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    summary = evaluate(parse_args())
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
