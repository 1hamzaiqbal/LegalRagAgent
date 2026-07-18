#!/usr/bin/env python3
"""Train a pinned Qwen3 math teacher with verifiable-reward GRPO."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import subprocess
from pathlib import Path

try:
    from .quality_gates import sha256_tree
except ImportError:
    from quality_gates import sha256_tree  # type: ignore


ROOT = Path(__file__).resolve().parents[2]
PER_DEVICE_TRAIN_BATCH_SIZE = 1
TEACHER_ROLE = "teacher_train"
VALID_SOURCES = ("M", "O")
CANONICAL_TRAINING_PLAN = ROOT / "configs" / "opd_math" / "teacher_training_plan.json"
ALGORITHM_LABEL = "GRPO with DAPO loss normalization; not the complete DAPO recipe"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def max_logged_optimizer_step(log_history: list[dict]) -> int | None:
    """Return the largest integral trainer log step, rejecting malformed values."""

    steps: list[int] = []
    for row in log_history:
        value = row.get("step")
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"trainer log contains a nonnumeric step: {value!r}")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0 or not numeric.is_integer():
            raise ValueError(f"trainer log contains an invalid optimizer step: {value!r}")
        steps.append(int(numeric))
    return max(steps) if steps else None


def git_state() -> dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain=v1"],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        ).stdout
        return {"commit": commit, "dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _read_json_object(path: Path, label: str) -> dict:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def validate_source_manifest_contract(args, prepared: dict) -> dict[str, str]:
    """Bind teacher identity to the exact source manifest used by preparation."""

    source_manifest_path = args.source_manifest.resolve()
    prepared_path = args.prepared_manifest.resolve()
    declared_path = prepared.get("source_manifest_path")
    if not isinstance(declared_path, str) or Path(declared_path).resolve() != source_manifest_path:
        raise ValueError(
            "--source-manifest is not the source manifest bound by prepared data: "
            f"prepared={declared_path!r}, supplied={str(source_manifest_path)!r}"
        )
    source_hash = sha256_file(source_manifest_path)
    if prepared.get("source_manifest_sha256") != source_hash:
        raise ValueError(
            f"source manifest hash does not match prepared data: {prepared_path}"
        )
    source_manifest = _read_json_object(source_manifest_path, "source manifest")
    teacher = source_manifest.get("models", {}).get("teacher")
    if not isinstance(teacher, dict):
        raise ValueError("source manifest lacks models.teacher")
    if teacher.get("id") != args.model or teacher.get("revision") != args.model_revision:
        raise ValueError(
            "teacher identity is not the pinned primary teacher in the source manifest: "
            f"pinned={teacher.get('id')}@{teacher.get('revision')}, "
            f"requested={args.model}@{args.model_revision}"
        )
    if re.fullmatch(r"[0-9a-f]{40}", args.model_revision) is None:
        raise ValueError("--model-revision must be an immutable 40-character lowercase commit")
    return {
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": source_hash,
        "pinned_teacher_model": str(teacher["id"]),
        "pinned_teacher_revision": str(teacher["revision"]),
    }


def read_rows(path: Path, limit: int, source: str) -> tuple[list[dict], int]:
    """Read the selected prefix while validating every row in the bound role file."""
    rows: list[dict] = []
    total_rows = 0
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("prompt"), list) or not row.get("solution"):
                raise ValueError(f"{path}:{line_no}: teacher rows require conversational prompt and solution")
            if row.get("source") != source:
                raise ValueError(
                    f"{path}:{line_no}: expected source={source!r}, got {row.get('source')!r}"
                )
            if row.get("role") != TEACHER_ROLE:
                raise ValueError(
                    f"{path}:{line_no}: expected role={TEACHER_ROLE!r}, got {row.get('role')!r}"
                )
            total_rows += 1
            if len(rows) < limit:
                rows.append(row)
    if not rows:
        raise ValueError(f"{path} contained no usable rows")
    if len(rows) != limit:
        raise ValueError(f"requested --limit={limit}, but {path} contains only {total_rows} rows")
    return rows, total_rows


def gold_parseability(rows: list[dict]) -> tuple[int, int]:
    from math_verify import parse

    # Match the gold side of TRL 1.8's accuracy_reward contract exactly. A gold
    # that fails this parse would receive reward=None and silently carry no
    # policy-gradient signal.
    parseable = sum(bool(parse(str(row["solution"]), parsing_timeout=10)) for row in rows)
    return parseable, len(rows)


def validate_static_args(args) -> dict[str, int]:
    positive_ints = {
        "max_steps": args.max_steps,
        "limit": args.limit,
        "num_generations": args.num_generations,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_prompt_tokens": args.max_prompt_tokens,
        "max_completion_length": args.max_completion_length,
        "lora_r": args.lora_r,
    }
    for name, value in positive_ints.items():
        if value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be a positive integer, got {value}")
    if args.num_generations < 2:
        raise ValueError("GRPO requires at least two generations per prompt")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError(f"--learning-rate must be finite and positive, got {args.learning_rate}")
    if args.seed < 0:
        raise ValueError(f"--seed must be nonnegative, got {args.seed}")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    generation_batch_size = (
        PER_DEVICE_TRAIN_BATCH_SIZE * world_size * args.gradient_accumulation_steps
    )
    if generation_batch_size % args.num_generations != 0:
        raise ValueError(
            "TRL generation batch must be divisible by --num-generations: "
            f"{PER_DEVICE_TRAIN_BATCH_SIZE=} * {world_size=} * "
            f"gradient_accumulation_steps={args.gradient_accumulation_steps} gives "
            f"generation_batch_size={generation_batch_size}, num_generations={args.num_generations}"
        )
    return {"world_size": world_size, "generation_batch_size": generation_batch_size}


def normalized_training_config(args, static_contract: dict[str, int]) -> dict[str, object]:
    """Return the complete source-independent recipe compared across M and O."""

    return {
        "algorithm_label": ALGORITHM_LABEL,
        "attn_implementation": "sdpa",
        "beta": 0.0,
        "budget_mode": args.budget_mode,
        "data_seed": args.seed,
        "dtype": "bfloat16",
        "generation_batch_size": static_contract["generation_batch_size"],
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_checkpointing": True,
        "learning_rate": args.learning_rate,
        "lora_alpha_multiplier": 2,
        "lora_dropout": 0.0,
        "lora_r": args.lora_r,
        "loss_type": "dapo",
        "mask_truncated_completions": not args.smoke,
        "max_completion_length": args.max_completion_length,
        "max_prompt_tokens": args.max_prompt_tokens,
        "max_steps": args.max_steps,
        "min_p": 0.0,
        "num_generations": args.num_generations,
        "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
        "require_informative_reward": args.require_informative_reward,
        "seed": args.seed,
        "shuffle_dataset": True,
        "temperature": 0.7,
        "thinking": False,
        "top_k": 20,
        "top_p": 0.8,
        "use_vllm": False,
        "world_size": static_contract["world_size"],
    }


def validate_training_plan_contract(
    args, static_contract: dict[str, int], *, intended_scientific_run: bool
) -> dict[str, object]:
    """Bind scientific teacher runs to one committed recipe for both sources."""

    plan_path = args.training_plan.resolve()
    if plan_path != CANONICAL_TRAINING_PLAN.resolve():
        raise ValueError(
            "--training-plan must be the canonical tracked teacher plan: "
            f"expected={CANONICAL_TRAINING_PLAN.resolve()}, supplied={plan_path}"
        )
    plan = _read_json_object(plan_path, "teacher training plan")
    if plan.get("schema_version") != 1 or plan.get("plan_id") != "opd_math_teacher_primary_v1":
        raise ValueError("teacher training plan has an unsupported schema or plan_id")
    if plan.get("sources") != list(VALID_SOURCES):
        raise ValueError("teacher training plan must bind exactly the M and O source comparison")
    fixed_config = plan.get("fixed_config")
    if not isinstance(fixed_config, dict) or not fixed_config:
        raise ValueError("teacher training plan lacks fixed_config")
    actual_config = normalized_training_config(args, static_contract)
    plan_compliant = actual_config == fixed_config
    if intended_scientific_run and not plan_compliant:
        differing = sorted(
            key
            for key in set(actual_config) | set(fixed_config)
            if actual_config.get(key) != fixed_config.get(key)
        )
        raise ValueError(
            "primary scientific teacher training differs from the predeclared matched recipe: "
            f"fields={differing}"
        )
    return {
        "training_plan": str(plan_path),
        "training_plan_sha256": sha256_file(plan_path),
        "training_plan_id": plan["plan_id"],
        "training_plan_compliant": plan_compliant,
        "training_plan_config_sha256": canonical_json_sha256(fixed_config),
        "teacher_training_config_sha256": canonical_json_sha256(actual_config),
        "actual_config": actual_config,
    }


def prompt_token_diagnostics(tokenizer, rows: list[dict], max_prompt_tokens: int) -> dict[str, object]:
    """Reject implicit prompt truncation and record the exact rendered length surface."""

    lengths: list[int] = []
    for index, row in enumerate(rows):
        token_ids = tokenizer.apply_chat_template(
            row["prompt"], tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
        if not isinstance(token_ids, list) or any(not isinstance(value, int) for value in token_ids):
            raise RuntimeError(f"tokenizer returned an invalid prompt token sequence at row {index}")
        length = len(token_ids)
        if length > max_prompt_tokens:
            raise RuntimeError(
                f"rendered prompt at selected row {index} has {length} tokens, above "
                f"--max-prompt-tokens={max_prompt_tokens}; teacher prompts are never silently truncated"
            )
        lengths.append(length)
    return {
        "selected_prompts": len(lengths),
        "max_prompt_tokens_allowed": max_prompt_tokens,
        "min_rendered_prompt_tokens": min(lengths),
        "max_rendered_prompt_tokens": max(lengths),
        "mean_rendered_prompt_tokens": sum(lengths) / len(lengths),
        "implicit_truncation_allowed": False,
    }


def validate_prepared_contract(args, prepared: dict) -> tuple[list[dict], dict]:
    if not args.smoke and not prepared.get("scientific_use_allowed"):
        raise ValueError("non-smoke teacher training requires a complete prepared-data collision scan")

    source_contract = validate_source_manifest_contract(args, prepared)
    relative_task = f"roles/{args.source}/{TEACHER_ROLE}.jsonl"
    expected_path = (args.prepared_manifest.resolve().parent / relative_task).resolve()
    actual_path = args.task_file.resolve()
    if actual_path != expected_path:
        raise ValueError(
            "--task-file is not the exact prepared role file for this source: "
            f"expected {expected_path}, got {actual_path}"
        )

    file_entry = prepared.get("files", {}).get(relative_task)
    if not isinstance(file_entry, dict):
        raise ValueError(f"prepared manifest does not register {relative_task}")
    task_hash = sha256_file(actual_path)
    if task_hash != file_entry.get("sha256"):
        raise ValueError(f"task hash does not match prepared manifest entry for {relative_task}")

    declared_rows = file_entry.get("rows")
    if not isinstance(declared_rows, int) or declared_rows <= 0:
        raise ValueError(f"prepared manifest has invalid row count for {relative_task}: {declared_rows!r}")

    primary_limit = prepared.get("primary_matched_budgets", {}).get(TEACHER_ROLE)
    if not isinstance(primary_limit, int) or primary_limit <= 0:
        raise ValueError("prepared manifest lacks a positive primary teacher_train budget")
    if not args.smoke and args.budget_mode == "primary_matched" and args.limit != primary_limit:
        raise ValueError(
            "primary_matched teacher run must use the exact prepared budget: "
            f"--limit={args.limit}, expected {primary_limit}"
        )

    rows, actual_file_rows = read_rows(actual_path, args.limit, args.source)
    if actual_file_rows != declared_rows:
        raise ValueError(
            f"prepared manifest row count drift for {relative_task}: "
            f"manifest={declared_rows}, actual={actual_file_rows}"
        )
    return rows, {
        "relative_task_file": relative_task,
        "task_file_sha256": task_hash,
        "declared_role_rows": declared_rows,
        "primary_matched_limit": primary_limit,
        **source_contract,
    }


def reward_signal_diagnostics(log_history: list[dict]) -> dict:
    zero_std_fractions: list[float] = []
    clipped_ratios: list[float] = []
    reward_stds: list[float] = []
    for row in log_history:
        if isinstance(row.get("frac_reward_zero_std"), (int, float)):
            zero_std_fractions.append(float(row["frac_reward_zero_std"]))
        if isinstance(row.get("completions/clipped_ratio"), (int, float)):
            clipped_ratios.append(float(row["completions/clipped_ratio"]))
        if isinstance(row.get("reward_std"), (int, float)):
            reward_stds.append(float(row["reward_std"]))

    informative = bool(zero_std_fractions) and any(value < 1.0 for value in zero_std_fractions)
    return {
        "informative_reward_observed": informative,
        "reward_log_entries": len(zero_std_fractions),
        "frac_reward_zero_std": zero_std_fractions,
        "max_mixed_reward_sample_fraction": (
            max(1.0 - value for value in zero_std_fractions) if zero_std_fractions else None
        ),
        "reward_std": reward_stds,
        "completion_clipped_ratio": clipped_ratios,
    }


def package_versions() -> dict[str, str]:
    names = (
        "torch",
        "transformers",
        "trl",
        "datasets",
        "peft",
        "accelerate",
        "huggingface-hub",
        "requests",
        "math-verify",
    )
    return {name: importlib.metadata.version(name) for name in names}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--source", choices=VALID_SOURCES, required=True)
    parser.add_argument(
        "--budget-mode", choices=("primary_matched", "dose_response"), default="primary_matched"
    )
    parser.add_argument("--task-file", type=Path, required=True)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--training-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-prompt-tokens", type=int, default=1536)
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--require-informative-reward",
        action="store_true",
        help="fail after persisting diagnostics unless at least one prompt group has nonconstant reward",
    )
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    static_contract = validate_static_args(args)
    intended_scientific_run = not args.smoke and args.budget_mode == "primary_matched"
    plan_contract = validate_training_plan_contract(
        args, static_contract, intended_scientific_run=intended_scientific_run
    )
    if args.output_dir.is_symlink() or (
        args.output_dir.exists()
        and (not args.output_dir.is_dir() or any(args.output_dir.iterdir()))
    ):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prepared = _read_json_object(args.prepared_manifest, "prepared manifest")
    rows, prepared_contract = validate_prepared_contract(args, prepared)
    code_state_start = git_state()
    if intended_scientific_run and code_state_start.get("dirty") is not False:
        raise RuntimeError(
            "primary-matched non-smoke teacher training requires an available, clean Git state"
        )
    parseable, total = gold_parseability(rows)
    if parseable != total:
        raise RuntimeError(
            "every selected gold solution must be parseable by TRL's math-verify contract: "
            f"parseable={parseable}, selected={total}"
        )

    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import GRPOConfig, GRPOTrainer
    from trl.rewards import accuracy_reward
    import torch

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.model_revision,
        local_files_only=args.local_files_only,
    )
    false_render = tokenizer.apply_chat_template(
        rows[0]["prompt"], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    true_render = tokenizer.apply_chat_template(
        rows[0]["prompt"], tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    if false_render == true_render:
        raise RuntimeError("pinned tokenizer ignored enable_thinking=False")
    prompt_diagnostics = prompt_token_diagnostics(tokenizer, rows, args.max_prompt_tokens)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.model_revision,
        local_files_only=args.local_files_only,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    dataset = Dataset.from_list(
        [
            {
                "prompt": row["prompt"],
                "solution": row["solution"],
                "record_id": row.get("record_id"),
                "source": row.get("source"),
            }
            for row in rows
        ]
    )
    config = GRPOConfig(
        output_dir=str(args.output_dir / "trainer"),
        bf16=True,
        max_steps=args.max_steps,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        learning_rate=args.learning_rate,
        beta=0.0,
        loss_type="dapo",
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
        chat_template_kwargs={"enable_thinking": False},
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_vllm=False,
        report_to="none",
        save_strategy="no",
        logging_steps=1,
        logging_first_step=True,
        remove_unused_columns=False,
        mask_truncated_completions=not args.smoke,
        seed=args.seed,
        data_seed=args.seed,
    )
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_r,
        lora_alpha=2 * args.lora_r,
        lora_dropout=0.0,
        target_modules="all-linear",
    )

    run_manifest = {
        "schema_version": 1,
        "stage": "teacher_grpo",
        "status": "contract_validated",
        "scientific_use_allowed": False,
        "intended_scientific_run": intended_scientific_run,
        "claim_boundary": (
            "A completed, informative optimization run is still training evidence only; "
            "teacher quality requires the separate held-out teacher-gap gate."
        ),
        "git_state_start": code_state_start,
        "model": args.model,
        "model_revision": args.model_revision,
        "source": args.source,
        "role": TEACHER_ROLE,
        "budget_mode": args.budget_mode,
        "task_file": str(args.task_file.resolve()),
        "task_file_sha256": prepared_contract["task_file_sha256"],
        "prepared_manifest": str(args.prepared_manifest.resolve()),
        "prepared_manifest_sha256": sha256_file(args.prepared_manifest),
        "source_manifest": prepared_contract["source_manifest"],
        "source_manifest_sha256": prepared_contract["source_manifest_sha256"],
        "training_plan": plan_contract["training_plan"],
        "training_plan_sha256": plan_contract["training_plan_sha256"],
        "training_plan_id": plan_contract["training_plan_id"],
        "training_plan_compliant": plan_contract["training_plan_compliant"],
        "training_plan_config_sha256": plan_contract["training_plan_config_sha256"],
        "teacher_training_config_sha256": plan_contract["teacher_training_config_sha256"],
        "pinned_teacher_model": prepared_contract["pinned_teacher_model"],
        "pinned_teacher_revision": prepared_contract["pinned_teacher_revision"],
        "selected_rows": len(rows),
        "declared_role_rows": prepared_contract["declared_role_rows"],
        "primary_matched_limit": prepared_contract["primary_matched_limit"],
        "gold_parseable_rows": parseable,
        "gold_parseable_fraction": parseable / total,
        "thinking_mode": "disabled",
        "algorithm_label": ALGORITHM_LABEL,
        "prompt_token_diagnostics": prompt_diagnostics,
        "packages": package_versions(),
        "config": plan_contract["actual_config"],
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n")

    trainer = GRPOTrainer(
        model=model,
        args=config,
        reward_funcs=accuracy_reward,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    result = trainer.train()
    log_history = [dict(row) for row in trainer.state.log_history]
    trainer_log_path = args.output_dir / "trainer_log_history.json"
    trainer_log_path.write_text(
        json.dumps(log_history, indent=2, sort_keys=True) + "\n"
    )
    trainer_state_path = args.output_dir / "trainer_state.json"
    trainer.state.save_to_json(str(trainer_state_path))
    signal = reward_signal_diagnostics(log_history)
    actual_optimizer_steps = int(trainer.state.global_step)
    trainer_log_max_step = max_logged_optimizer_step(log_history)
    progress_complete = (
        actual_optimizer_steps == args.max_steps
        and trainer_log_max_step == actual_optimizer_steps
    )
    metrics = dict(result.metrics)
    metrics["gold_parseable_fraction"] = parseable / total
    metrics["actual_optimizer_steps"] = actual_optimizer_steps
    metrics["optimizer_progress_complete"] = progress_complete
    metrics["reward_signal"] = signal
    metrics["peak_cuda_memory_bytes"] = (
        int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
    )
    metrics_path = args.output_dir / "train_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

    run_manifest.update(
        {
            "status": "training_complete_pending_artifact",
            "actual_optimizer_steps": actual_optimizer_steps,
            "trainer_log_max_step": trainer_log_max_step,
            "optimizer_progress_complete": progress_complete,
            "reward_signal": signal,
            "trainer_state": str(trainer_state_path.resolve()),
            "trainer_state_sha256": sha256_file(trainer_state_path),
            "trainer_log_history": str(trainer_log_path.resolve()),
            "trainer_log_history_sha256": sha256_file(trainer_log_path),
            "train_metrics": str(metrics_path.resolve()),
            "train_metrics_sha256": sha256_file(metrics_path),
        }
    )
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n"
    )
    if not progress_complete:
        run_manifest["status"] = "failed_optimizer_progress_gate"
        (args.output_dir / "run_manifest.json").write_text(
            json.dumps(run_manifest, indent=2, sort_keys=True) + "\n"
        )
        raise RuntimeError(
            f"trainer completed {actual_optimizer_steps} optimizer steps; expected exactly {args.max_steps}"
        )
    if args.require_informative_reward and not signal["informative_reward_observed"]:
        run_manifest["status"] = "failed_informative_reward_gate"
        (args.output_dir / "run_manifest.json").write_text(
            json.dumps(run_manifest, indent=2, sort_keys=True) + "\n"
        )
        raise RuntimeError(
            "no mixed-reward prompt group was observed; diagnostics were persisted and no adapter was promoted"
        )

    final_dir = args.output_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(final_dir)
    final_adapter_hash = sha256_tree(final_dir)
    code_state_end = git_state()
    clean_stable_code = (
        code_state_start.get("dirty") is False
        and code_state_end.get("dirty") is False
        and code_state_start.get("commit") is not None
        and code_state_start.get("commit") == code_state_end.get("commit")
    )
    scientific_use_allowed = (
        intended_scientific_run
        and args.budget_mode == "primary_matched"
        and progress_complete
        and signal["informative_reward_observed"]
        and prepared.get("scientific_use_allowed") is True
        and plan_contract["training_plan_compliant"] is True
        and clean_stable_code
    )
    run_manifest.update(
        {
            "status": "completed",
            "scientific_use_allowed": scientific_use_allowed,
            "final_adapter": str(final_dir.resolve()),
            "final_adapter_tree_sha256": final_adapter_hash,
            "git_state_end": code_state_end,
            "clean_stable_code": clean_stable_code,
        }
    )
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"final_adapter": str(final_dir), "metrics": metrics}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
