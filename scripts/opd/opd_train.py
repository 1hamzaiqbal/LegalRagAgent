#!/usr/bin/env python
"""Single-GPU student trainer for task reward plus sampled reverse KL.

The teacher is a separate vLLM OpenAI-compatible server. This process loads the
student, samples completions on-policy, asks the teacher server for per-token
logprobs on those exact token IDs, and applies a score-function reverse-KL
surrogate or its positive-gap-gated variant:

  A_t = logp_teacher_t - stopgrad(logp_student_t)
  loss = -mean_t A_t * logp_student_t

Teacher and student must pass the pinned tokenizer/server fingerprint because
OPD aligns exact token IDs. Matching family names alone is insufficient.

The scientific main arm is ``task_rl_k1_gap``: grouped verifiable task reward
plus a weighted, clipped, positive-gap-gated K1-value score-function auxiliary.
Only its ungated, unclipped, on-policy limit is K4/r-trick gradient-equivalent.
``task_rl`` is the required primary baseline; distillation-only modes remain
plumbing/collapse diagnostics. The sampled K1 value is not a full-vocabulary
KL, and the gate is not an exact reproduction of SDAR.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import random
import re
import stat
import subprocess
import sys
import time
from pathlib import Path

import torch

try:
    from . import teacher_client
    from .opd_loss import (
        kd_forward_loss,
        reverse_kl_score_function_loss,
        sampled_k1_estimate,
        task_reward_policy_loss,
        verl_k1_policy_gradient_loss,
    )
    from .objective_registry import (
        GATED_K1_OBJECTIVE_IDS,
        K1_OBJECTIVE_IDS,
        LOCAL_OBJECTIVE_IDS,
        TASK_AND_K1_OBJECTIVE_IDS,
        TASK_REWARD_OBJECTIVE_IDS,
        load_objective_registry,
        resolve_objective,
    )
    from .trace_metrics import (
        reconstruct_step_metrics,
        validate_recorded_step_metrics,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import teacher_client
    from opd_loss import (
        kd_forward_loss,
        reverse_kl_score_function_loss,
        sampled_k1_estimate,
        task_reward_policy_loss,
        verl_k1_policy_gradient_loss,
    )
    from objective_registry import (  # type: ignore
        GATED_K1_OBJECTIVE_IDS,
        K1_OBJECTIVE_IDS,
        LOCAL_OBJECTIVE_IDS,
        TASK_AND_K1_OBJECTIVE_IDS,
        TASK_REWARD_OBJECTIVE_IDS,
        load_objective_registry,
        resolve_objective,
    )
    from trace_metrics import (  # type: ignore
        reconstruct_step_metrics,
        validate_recorded_step_metrics,
    )

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.opd_math.math_reward import rewards_for_samples
from scripts.opd_math.quality_gates import (
    EVALUATION_CONTRACT,
    STUDENT_GATE_TYPE,
    recompute_student_gate,
    recompute_teacher_gate,
    sha256_tree,
)
from scripts.opd_math.server_scoring_probe import (
    LOCAL_BINDING_SCOPE,
    expected_serve_environment_launcher,
    revalidate_local_process_binding,
)
from scripts.opd_math.verify_environment import (
    reverify_recorded_environment,
    run_external_environment_verification,
    verify_environment as verify_live_environment,
)


LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LEGACY_K1_MODES = {"opd", "opd_gated", "k1_bare", "k1_gap_only", "task_rl_k1_gap"}
LEGACY_GATED_K1_MODES = {"opd_gated", "k1_gap_only", "task_rl_k1_gap"}
LEGACY_TASK_REWARD_MODES = {"task_rl", "task_rl_k1_gap"}
K1_MODES = LEGACY_K1_MODES | set(K1_OBJECTIVE_IDS)
GATED_K1_MODES = LEGACY_GATED_K1_MODES | set(GATED_K1_OBJECTIVE_IDS)
TASK_REWARD_MODES = LEGACY_TASK_REWARD_MODES | set(TASK_REWARD_OBJECTIVE_IDS)
TASK_AND_K1_MODES = {"task_rl_k1_gap"} | set(TASK_AND_K1_OBJECTIVE_IDS)
TEACHER_MODES = K1_MODES | {"kd"}
MERGED_TEACHER_SCHEMA = "opd_math_merged_teacher_v3"
MERGER_FILE = ROOT / "scripts" / "opd_math" / "merge_adapter.py"
CANONICAL_STUDENT_TRAINING_PLAN = (
    ROOT / "configs" / "opd_math" / "student_training_plan.json"
)
ENVIRONMENT_VERIFIER = ROOT / "scripts" / "opd_math" / "verify_environment.py"
EXPECTED_TRAIN_PACKAGES = {
    "torch": "2.11.0",
    "transformers": "4.57.6",
    "peft": "0.19.1",
    "trl": "1.8.0",
    "datasets": "4.8.5",
    "accelerate": "1.14.0",
    "huggingface-hub": "0.36.2",
    "requests": "2.32.5",
    "math-verify": "0.9.0",
}
EXPECTED_MERGE_PACKAGES = {
    name: EXPECTED_TRAIN_PACKAGES[name] for name in ("torch", "transformers", "peft")
}
EXPECTED_SERVE_PACKAGES = {
    "torch": "2.11.0",
    "transformers": "5.12.1",
    "peft": "0.19.1",
    "accelerate": "1.14.0",
    "requests": "2.32.5",
    "vllm": "0.24.0",
}
STUDENT_PRELAUNCH_RECEIPT = "opd_math_o_teacher_student_prelaunch_receipt_v1"


def log(msg: str) -> None:
    print(msg, flush=True)


def canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def task_prompt_sha256(row: dict) -> str:
    prompt = row.get("prompt")
    if prompt is not None:
        if not isinstance(prompt, list):
            raise ValueError("conversational prompt must be a list for prompt identity")
        return canonical_json_sha256(prompt)
    prompt_text = row.get("prompt_text")
    if not isinstance(prompt_text, str) or not prompt_text:
        raise ValueError("task row lacks a stable prompt identity")
    return canonical_json_sha256(prompt_text)


def normalized_student_training_config(args) -> dict:
    return {
        "advantage_clip": args.advantage_clip,
        "attn_implementation": args.attn_implementation,
        "budget_mode": args.budget_mode,
        "enable_thinking": args.enable_thinking,
        "gap_gate_beta": args.gap_gate_beta,
        "grad_clip": args.grad_clip,
        "gradient_checkpointing": args.gradient_checkpointing,
        "group_size": args.group_size,
        "k1_coef": args.k1_coef,
        "learning_rate": args.lr,
        "lora_r": args.lora,
        "max_new_tokens": args.max_new_tokens,
        "max_prompt_tokens": args.max_prompt_tokens,
        "micro_prompts": args.micro_prompts,
        "min_informative_group_fraction": args.min_informative_group_fraction,
        "optimizer_steps": args.steps,
        "seed": args.seed,
        "task_reward_coef": args.task_reward_coef,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
    }


def bind_registered_objective(args) -> dict | None:
    """Bind a registry ID to exact coefficients and local execution semantics.

    Free-form loss flags are not an authorization surface for successor runs.
    Selecting ``--objective-id`` makes the committed registry authoritative and
    overwrites the corresponding argparse defaults before validation or model
    loading.  The registry still cannot authorize a scientific launch by
    itself; that fail-closed boundary is enforced in ``validate_run_contract``.
    """

    objective_id = getattr(args, "objective_id", None)
    if objective_id is None:
        return None
    registry = load_objective_registry()
    objective = resolve_objective(objective_id, registry=registry)
    if objective_id not in LOCAL_OBJECTIVE_IDS or objective.get("local_executable") is not True:
        raise ValueError(
            f"objective {objective_id} must run through the pinned upstream veRL launcher"
        )
    args.mode = objective_id
    args.task_reward_coef = objective["task_reward_coef"]
    args.k1_coef = objective["k1_coef"]
    args.advantage_clip = objective["advantage_clip"]
    args.gap_gate_beta = objective["gap_gate_beta"]
    contract = {
        "registry_id": registry["registry_id"],
        "registry_path": registry["path"],
        "registry_sha256": registry["sha256"],
        "registry_canonical_sha256": registry["canonical_sha256"],
        "registry_status": registry["status"],
        "registry_alone_authorizes_scientific_launch": registry[
            "registry_alone_authorizes_scientific_launch"
        ],
        "objective": objective,
    }
    args.objective_registry_contract = contract
    return contract


def validate_student_training_plan_contract(args) -> dict:
    plan = json.loads(CANONICAL_STUDENT_TRAINING_PLAN.read_text())
    if (
        plan.get("schema_version") != 1
        or plan.get("plan_id") != "opd_math_student_primary_pilot_v1"
        or plan.get("objectives") != ["task_rl", "task_rl_k1_gap"]
    ):
        raise ValueError("student training plan has an unsupported identity")
    fixed = plan.get("fixed_config")
    if not isinstance(fixed, dict) or not fixed:
        raise ValueError("student training plan lacks fixed_config")
    actual = normalized_student_training_config(args)
    compliant = actual == fixed
    if not compliant:
        differing = sorted(
            key for key in set(actual) | set(fixed) if actual.get(key) != fixed.get(key)
        )
        raise ValueError(
            "primary scientific student training differs from the predeclared matched recipe: "
            f"fields={differing}"
        )
    return {
        "path": str(CANONICAL_STUDENT_TRAINING_PLAN.resolve()),
        "sha256": sha256_file(CANONICAL_STUDENT_TRAINING_PLAN),
        "plan_id": plan["plan_id"],
        "plan_config_sha256": canonical_json_sha256(fixed),
        "actual_config_sha256": canonical_json_sha256(actual),
        "config": actual,
        "compliant": compliant,
    }


def read_jsonl(path: str, limit: int = 0) -> list[dict]:
    rows = []
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "prompt_text" not in row and "prompt" not in row:
                raise ValueError(f"{path}:{line_no} missing prompt_text or conversational prompt")
            rows.append(row)
            if limit > 0 and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"{path} contained no task rows")
    return rows


def read_jsonl_objects(path: str | Path) -> list[dict]:
    """Read generic JSONL trace objects without applying the task-row schema."""

    rows: list[dict] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} contains invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no} must contain a JSON object")
            rows.append(row)
    return rows


def count_jsonl_objects(path: str | Path) -> int:
    return len(read_jsonl_objects(path))


def resolve_trace_directory(out_path: str | Path, trace_dir: str | Path | None) -> Path:
    resolved_out = Path(out_path).resolve()
    resolved_trace = Path(trace_dir or (resolved_out / "traces")).resolve()
    canonical_internal_trace = (resolved_out / "traces").resolve()
    if resolved_trace != canonical_internal_trace and (
        resolved_trace.is_relative_to(resolved_out)
        or resolved_out.is_relative_to(resolved_trace)
    ):
        raise ValueError(
            "--trace-dir must be the canonical OUT/traces directory or a disjoint external path"
        )
    return resolved_trace


def prompt_stream(rows: list[dict], rng: random.Random):
    rows = list(rows)
    while True:
        rng.shuffle(rows)
        for row in rows:
            yield row


def encode(tokenizer, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def render_prompt(tokenizer, row: dict, max_prompt_tokens: int, enable_thinking: bool) -> tuple[str, list[int]]:
    if row.get("prompt") is not None:
        prompt = row["prompt"]
        if not isinstance(prompt, list):
            raise ValueError("conversational prompt must be a list of role/content messages")
        text = tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    else:
        text = str(row["prompt_text"])
    ids = encode(tokenizer, text)
    if not ids:
        raise ValueError("prompt tokenization is empty")
    if max_prompt_tokens > 0 and len(ids) > max_prompt_tokens:
        raise ValueError(
            f"rendered prompt has {len(ids)} tokens, above --max-prompt-tokens={max_prompt_tokens}; "
            "the scientific path rejects rather than tail-truncating the chat template"
        )
    return text, ids


def load_student(args, device: str):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        args.student,
        revision=args.student_revision,
        local_files_only=args.local_files_only,
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.student,
        revision=args.student_revision,
        local_files_only=args.local_files_only,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
    )
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

    if args.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    return tok, model


@torch.no_grad()
def trainable_parameter_signature(model) -> dict[str, float | int]:
    total = 0.0
    squared = 0.0
    count = 0
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        values = parameter.detach().float()
        total += float(values.sum().item())
        squared += float(values.square().sum().item())
        count += values.numel()
    if count == 0:
        raise RuntimeError("student model has no trainable parameters")
    if not math.isfinite(total) or not math.isfinite(squared):
        raise RuntimeError("student trainable parameters contain non-finite values")
    return {"elements": count, "sum": total, "squared_l2": squared}


@torch.no_grad()
def trainable_parameter_snapshot(model) -> dict[str, torch.Tensor]:
    snapshot = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not snapshot:
        raise RuntimeError("student model has no trainable parameters")
    for name, value in snapshot.items():
        if not torch.isfinite(value).all():
            raise RuntimeError(f"student trainable parameter is non-finite before step: {name}")
    return snapshot


@torch.no_grad()
def parameter_update_l2(model, before: Mapping[str, torch.Tensor]) -> float:
    current = {
        name: parameter.detach()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if set(current) != set(before):
        raise RuntimeError("trainable parameter identity changed during optimizer step")
    squared = 0.0
    for name, value in current.items():
        if value.shape != before[name].shape:
            raise RuntimeError(f"trainable parameter shape changed during optimizer step: {name}")
        if not torch.isfinite(value).all():
            raise RuntimeError(f"student trainable parameter is non-finite after step: {name}")
        delta = value.float() - before[name].to(device=value.device, dtype=torch.float32)
        if not torch.isfinite(delta).all():
            raise RuntimeError(f"student parameter update is non-finite: {name}")
        squared += float(delta.double().square().sum().item())
    if not math.isfinite(squared):
        raise RuntimeError("student parameter update norm is non-finite")
    return math.sqrt(squared)


@torch.no_grad()
def optimizer_state_signature(optimizer) -> dict[str, float | int]:
    tensors = 0
    elements = 0
    squared = 0.0
    scalars = 0
    for state in optimizer.state.values():
        for name, value in state.items():
            if isinstance(value, torch.Tensor):
                if not torch.isfinite(value).all():
                    raise RuntimeError(f"optimizer state tensor is non-finite: {name}")
                tensors += 1
                elements += value.numel()
                squared += float(value.detach().double().square().sum().item())
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                if not math.isfinite(float(value)):
                    raise RuntimeError(f"optimizer state scalar is non-finite: {name}")
                scalars += 1
    if not math.isfinite(squared):
        raise RuntimeError("optimizer state norm is non-finite")
    return {
        "tensors": tensors,
        "elements": elements,
        "scalars": scalars,
        "squared_l2": squared,
    }


def signatures_differ(before: dict, after: dict) -> bool:
    if before["elements"] != after["elements"]:
        raise RuntimeError("trainable parameter count changed during the run")
    return any(
        not math.isclose(float(before[key]), float(after[key]), rel_tol=0.0, abs_tol=1e-12)
        for key in ("sum", "squared_l2")
    )


@torch.no_grad()
def generate_student_samples(model, tok, prompt_rows: list[dict], args, device: str) -> list[dict]:
    model.eval()
    prompts = [row["prompt_text"] for row in prompt_rows]
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    for i, source in enumerate(prompt_rows):
        actual = enc["input_ids"][i][enc["attention_mask"][i].bool()].tolist()
        if actual != source["prompt_token_ids"]:
            raise RuntimeError(f"batch tokenization drifted for prompt group {i}")
    enc = {k: v.to(device) for k, v in enc.items()}
    prompt_width = enc["input_ids"].shape[1]
    rollout_started = time.perf_counter()
    generation = model.generate(
        **enc,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=0.0,
        typical_p=1.0,
        epsilon_cutoff=0.0,
        eta_cutoff=0.0,
        repetition_penalty=1.0,
        max_new_tokens=args.max_new_tokens,
        num_return_sequences=args.group_size,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
    )
    rollout_latency = time.perf_counter() - rollout_started
    gen = generation.sequences
    behavior_scores = model.compute_transition_scores(
        gen,
        generation.scores,
        normalize_logits=True,
    )
    if behavior_scores.shape[0] != gen.shape[0]:
        raise RuntimeError("rollout behavior-logprob batch shape drifted")
    samples = []
    for i, row in enumerate(gen):
        prompt_idx = i // args.group_size
        sample_idx = i % args.group_size
        comp_ids = [int(t) for t in row[prompt_width:].detach().cpu().tolist()]
        # With Qwen, pad_token_id commonly equals eos_token_id. Retain the
        # first EOS as a scored action and discard only tokens after it.
        terminated_by_eos = tok.eos_token_id in comp_ids
        if terminated_by_eos:
            comp_ids = comp_ids[: comp_ids.index(tok.eos_token_id) + 1]
        elif tok.pad_token_id is not None:
            while comp_ids and comp_ids[-1] == tok.pad_token_id:
                comp_ids.pop()
        if not comp_ids:
            raise RuntimeError(f"empty completion token sequence for prompt group {prompt_idx}")
        behavior_logprobs = [
            float(value)
            for value in behavior_scores[i, : len(comp_ids)].detach().cpu().tolist()
        ]
        if len(behavior_logprobs) != len(comp_ids) or not all(
            math.isfinite(value) for value in behavior_logprobs
        ):
            raise RuntimeError(
                f"invalid rollout behavior log-probabilities for prompt group {prompt_idx}"
            )
        completion = tok.decode(comp_ids, skip_special_tokens=True)
        source = prompt_rows[prompt_idx]
        sample = {
            key: source.get(key)
            for key in ("record_id", "cluster_id", "source", "source_split", "solution", "answer")
        }
        sample.update(
            {
                "group_id": prompt_idx,
                "sample_idx": sample_idx,
                "prompt_text": prompts[prompt_idx],
                "prompt_sha256": task_prompt_sha256(source),
                "prompt_token_ids": list(source["prompt_token_ids"]),
                "completion_text": completion,
                "completion_token_ids": comp_ids,
                "behavior_logprobs": behavior_logprobs,
                "terminated_by_eos": terminated_by_eos,
                "rollout_batch_latency_seconds": rollout_latency,
            }
        )
        samples.append(sample)
    expected = len(prompt_rows) * args.group_size
    if len(samples) != expected:
        raise RuntimeError(f"rollout grouping broke: expected {expected} samples, got {len(samples)}")
    return samples


def teacher_score_samples(samples: list[dict], args) -> list[dict]:
    scored = []
    timeout = (args.teacher_connect_timeout, args.teacher_read_timeout)
    for sample in samples:
        scoring_started = time.perf_counter()
        lps = teacher_client.score_completion_token_logprobs(
            args.teacher_url,
            args.teacher_model,
            sample["prompt_token_ids"],
            sample["completion_token_ids"],
            timeout=timeout,
            retries=args.teacher_retries,
        )
        scoring_latency = time.perf_counter() - scoring_started
        comp_len = len(sample["completion_token_ids"])
        if len(lps) != comp_len:
            raise RuntimeError(f"teacher/student token count mismatch: teacher={len(lps)} student={comp_len}")
        row = dict(sample)
        row["teacher_logprobs"] = lps
        row["teacher_scoring_latency_seconds"] = scoring_latency
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
        p_ids = list(sample.get("prompt_token_ids") or encode(tok, sample["prompt_text"]))
        c_ids = list(sample.get("completion_token_ids") or encode(tok, sample["completion_text"]))
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


def completion_logprobs_for_samples(model, tok, samples: list[dict], device: str, require_teacher: bool):
    ids, att, label_mask, teacher, teacher_mask = build_batch(
        tok, samples, device, require_teacher=require_teacher
    )
    # Gradients work in eval mode. Keeping generation and recomputation in eval
    # avoids dropout-induced behavior/current policy drift.
    model.eval()
    out = model(input_ids=ids, attention_mask=att)
    student_lps, student_mask = current_completion_logprobs(out.logits, ids, label_mask)
    if require_teacher and student_lps.shape != teacher.shape:
        raise RuntimeError(f"student/teacher logprob tensor mismatch: {student_lps.shape} vs {teacher.shape}")
    mask = student_mask & teacher_mask if require_teacher else student_mask
    return student_lps, teacher, mask


def aligned_behavior_logprobs(
    samples: list[dict], mask: torch.Tensor, device: str
) -> torch.Tensor:
    """Align rollout-time log-probabilities to the recomputed response mask."""

    behavior = torch.zeros(mask.shape, dtype=torch.float32, device=device)
    for index, sample in enumerate(samples):
        values = sample.get("behavior_logprobs")
        completion_ids = sample.get("completion_token_ids")
        if (
            not isinstance(values, list)
            or not isinstance(completion_ids, list)
            or len(values) != len(completion_ids)
            or len(values) != int(mask[index].sum().item())
            or any(type(value) not in (int, float) or not math.isfinite(float(value)) for value in values)
        ):
            raise ValueError(
                f"sample {index} lacks exact rollout behavior log-probabilities"
            )
        behavior[index, : len(values)] = torch.tensor(
            values, dtype=torch.float32, device=device
        )
    return behavior


def objective_loss_from_logprobs(
    student_lps: torch.Tensor,
    teacher_lps: torch.Tensor | None,
    mask: torch.Tensor,
    *,
    mode: str,
    task_reward_coef: float,
    k1_coef: float,
    advantage_clip: float | None,
    gap_gate_beta: float | None,
    rewards: torch.Tensor | None = None,
    group_ids: torch.Tensor | None = None,
    behavior_lps: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """Execute one registered or legacy local objective on aligned logprobs."""

    if mode not in TASK_REWARD_MODES | K1_MODES:
        raise ValueError(f"unsupported local objective mode: {mode}")
    if mask.dtype != torch.bool or mask.shape != student_lps.shape or not bool(mask.any()):
        raise ValueError(f"mode {mode} requires a nonempty aligned boolean response mask")
    if not torch.isfinite(student_lps).all():
        raise RuntimeError(f"mode {mode} received non-finite student log-probabilities")
    needs_teacher = mode in K1_MODES
    needs_task_reward = mode in TASK_REWARD_MODES
    if needs_teacher and teacher_lps is None:
        raise ValueError(f"mode {mode} requires teacher log-probabilities")
    if not needs_teacher and teacher_lps is not None:
        raise ValueError(f"mode {mode} unexpectedly received teacher log-probabilities")
    if teacher_lps is not None and (
        teacher_lps.shape != student_lps.shape or not torch.isfinite(teacher_lps).all()
    ):
        raise RuntimeError(f"mode {mode} received invalid teacher log-probabilities")
    if needs_task_reward and (rewards is None or group_ids is None):
        raise ValueError(f"mode {mode} requires task rewards and group IDs")
    if not needs_task_reward and (rewards is not None or group_ids is not None):
        raise ValueError(f"mode {mode} unexpectedly received task rewards")
    if rewards is not None and not torch.isfinite(rewards).all():
        raise RuntimeError(f"mode {mode} received non-finite task rewards")
    verl_compatible_bare = mode == "k1_bare_verl_compatible_clip10"
    if verl_compatible_bare and behavior_lps is None:
        raise ValueError(f"mode {mode} requires rollout behavior log-probabilities")
    if behavior_lps is not None and behavior_lps.shape != student_lps.shape:
        raise ValueError(f"mode {mode} has misaligned rollout behavior log-probabilities")
    if behavior_lps is not None and not torch.isfinite(behavior_lps).all():
        raise RuntimeError(f"mode {mode} received non-finite behavior log-probabilities")

    zero = student_lps.sum() * 0.0
    reverse_kl_sf_loss = zero
    task_loss = zero
    k1_value = None
    gate_mean = None
    positive_gap_fraction = None
    reward_mean = None
    informative_group_fraction = None
    verl_k1_pg_loss = None
    behavior_current_ratio_mean = None

    if needs_teacher:
        gate_beta = gap_gate_beta if mode in GATED_K1_MODES else None
        reverse_kl_sf_loss = reverse_kl_score_function_loss(
            student_lps,
            teacher_lps,
            mask,
            advantage_clip=advantage_clip,
            ratio_clip_eps=None,
            gap_gate_beta=gate_beta,
        )
        k1_value = sampled_k1_estimate(student_lps, teacher_lps, mask)
        gap = teacher_lps.detach() - student_lps.detach()
        executed_advantage = (
            gap
            if advantage_clip is None
            else torch.clamp(gap, min=-advantage_clip, max=advantage_clip)
        )
        positive_gap_fraction = float(gap[mask].gt(0).float().mean().item())
        gate_mean = (
            1.0
            if gate_beta is None
            else float(torch.sigmoid(gate_beta * executed_advantage)[mask].mean().item())
        )

    if needs_task_reward:
        task_loss, _, informative = task_reward_policy_loss(
            student_lps, rewards, group_ids, mask
        )
        reward_mean = float(rewards.mean().item())
        informative_group_fraction = float(informative.float().mean().item())

    if verl_compatible_bare:
        verl_k1_pg_loss = verl_k1_policy_gradient_loss(
            student_lps,
            teacher_lps,
            behavior_lps,
            mask,
            loss_max_clamp=10.0,
            clip_ratio_low=0.2,
            clip_ratio_high=0.2,
            dual_clip_ratio=3.0,
        )
        total = k1_coef * verl_k1_pg_loss
        behavior_current_ratio_mean = float(
            torch.exp(
                torch.clamp(
                    student_lps.detach() - behavior_lps.detach(),
                    min=-20.0,
                    max=20.0,
                )
            )[mask]
            .mean()
            .item()
        )
    elif mode == "task_rl":
        total = task_reward_coef * task_loss
    elif mode in TASK_AND_K1_MODES:
        total = task_reward_coef * task_loss + k1_coef * reverse_kl_sf_loss
    elif mode in K1_OBJECTIVE_IDS:
        total = k1_coef * reverse_kl_sf_loss
    else:
        # Preserve the pre-registry diagnostic behavior. The legacy K1 modes
        # historically reported the unscaled surrogate; registered objectives
        # carry their explicit coefficient in the branch above.
        total = reverse_kl_sf_loss

    if not torch.isfinite(total):
        raise RuntimeError(f"mode {mode} produced a non-finite objective")

    metrics = {
        "task_loss": float(task_loss.detach().item()),
        "reverse_kl_score_function_surrogate": float(reverse_kl_sf_loss.detach().item()),
        "sampled_k1_estimate": None if k1_value is None else float(k1_value.item()),
        "gap_gate_mean": gate_mean,
        "positive_gap_fraction": positive_gap_fraction,
        "reward_mean": reward_mean,
        "informative_group_fraction": informative_group_fraction,
        "verl_compatible_k1_policy_loss": (
            None if verl_k1_pg_loss is None else float(verl_k1_pg_loss.detach().item())
        ),
        "behavior_current_ratio_mean": behavior_current_ratio_mean,
        "tokens": int(mask.sum().item()),
    }
    return total, metrics


def training_loss_for_samples(model, tok, samples: list[dict], args, device: str):
    needs_teacher = args.mode in K1_MODES
    student_lps, teacher, mask = completion_logprobs_for_samples(
        model, tok, samples, device, require_teacher=needs_teacher
    )
    behavior_lps = aligned_behavior_logprobs(samples, mask, device)
    rewards: list[float] | None = None
    reward_statuses: list[str] | None = None
    reward_tensor = group_ids = None
    if args.mode in TASK_REWARD_MODES:
        rewards, reward_statuses = rewards_for_samples(samples)
        reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
        group_ids = torch.tensor(
            [int(sample["group_id"]) for sample in samples],
            dtype=torch.long,
            device=device,
        )
    total, metrics = objective_loss_from_logprobs(
        student_lps,
        teacher if needs_teacher else None,
        mask,
        mode=args.mode,
        task_reward_coef=args.task_reward_coef,
        k1_coef=args.k1_coef,
        advantage_clip=args.advantage_clip,
        gap_gate_beta=args.gap_gate_beta,
        rewards=reward_tensor,
        group_ids=group_ids,
        behavior_lps=behavior_lps,
    )
    return total, metrics, student_lps, teacher if needs_teacher else None, mask, rewards, reward_statuses


def save_checkpoint(model, tok, out_dir: str, name: str) -> None:
    path = Path(out_dir) / name
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tok.save_pretrained(path)
    log(f"saved {path}")


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def write_completion_manifests(trace_dir: Path, run_manifest: dict, completion: dict) -> None:
    trace_artifacts = {}
    for name in ("steps.jsonl", "samples.jsonl"):
        path = (trace_dir / name).resolve()
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as handle:
            rows = sum(bool(line.strip()) for line in handle)
        trace_artifacts[name] = {
            "path": str(path),
            "rows": rows,
            "sha256": sha256_file(path),
        }
    completion["trace_artifacts"] = trace_artifacts
    run_manifest.update(
        {
            "status": completion["status"],
            "scientific_use_allowed": completion["scientific_use_allowed"],
            "training_artifact_eligible_for_held_out_evaluation": completion.get(
                "training_artifact_eligible_for_held_out_evaluation", False
            ),
            "completion": completion,
        }
    )
    (trace_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n"
    )
    (trace_dir / "completion_manifest.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n"
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_prelaunch_receipt(args) -> dict:
    """Reopen the wrapper-sealed preregistration receipt before training."""

    raw = Path(args.prelaunch_receipt).expanduser()
    if raw.is_symlink() or not raw.is_file():
        raise ValueError("primary student run requires a regular prelaunch receipt")
    if raw.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise ValueError("primary student prelaunch receipt must be sealed read-only")
    path = raw.resolve()
    receipt = json.loads(path.read_text())
    if not isinstance(receipt, dict):
        raise ValueError("primary student prelaunch receipt must be a JSON object")
    _expect_equal(receipt, "schema_version", 1, "student prelaunch receipt")
    _expect_equal(
        receipt,
        "receipt",
        STUDENT_PRELAUNCH_RECEIPT,
        "student prelaunch receipt",
    )
    _expect_equal(
        receipt,
        "sealed_before_optimizer_start",
        True,
        "student prelaunch receipt",
    )
    run_key = (
        f"baseline_{args.student_source}"
        if args.mode == "task_rl"
        else args.pair_id
    )
    student_source = (
        args.student_source if args.mode == "task_rl" else str(args.pair_id).split("_")[1]
    )
    for field, expected in (
        ("run_key", run_key),
        ("run_id", args.campaign_run_id),
        ("scheduler_job_id", args.scheduler_job_id),
        ("mode", args.mode),
        ("student_source", student_source),
        ("out_dir", str(Path(args.out_dir).resolve())),
    ):
        _expect_equal(receipt, field, expected, "student prelaunch receipt")
    state = git_state()
    if state.get("dirty") or state.get("commit") != receipt.get("git_commit"):
        raise ValueError("student prelaunch receipt Git identity is not current and clean")
    expected_artifacts = {
        "run_manifest": str(
            (Path(args.out_dir).resolve() / "traces" / "run_manifest.json").resolve()
        ),
        "student_completion_manifest": str(
            (
                Path(args.out_dir).resolve()
                / "traces"
                / "completion_manifest.json"
            ).resolve()
        ),
        "student_adapter": str((Path(args.out_dir).resolve() / "final").resolve()),
        "prelaunch_receipt": str(path),
    }
    _expect_equal(
        receipt,
        "expected_artifacts",
        expected_artifacts,
        "student prelaunch receipt",
    )
    support = receipt.get("student_support")
    if not isinstance(support, dict):
        raise ValueError("student prelaunch receipt lacks support identity")
    _expect_equal(
        support,
        "manifest_sha256",
        sha256_file(args.student_support_manifest),
        "student prelaunch support",
    )
    teacher = receipt.get("o_teacher")
    if args.mode == "task_rl":
        if teacher is not None:
            raise ValueError("baseline prelaunch receipt unexpectedly binds a teacher")
    else:
        if not isinstance(teacher, dict):
            raise ValueError("main-arm prelaunch receipt lacks O-teacher identity")
        _expect_equal(
            teacher,
            "teacher_gap_manifest",
            str(Path(args.teacher_gap_manifest).resolve()),
            "student prelaunch teacher",
        )
        _expect_equal(
            teacher,
            "teacher_gap_manifest_sha256",
            sha256_file(args.teacher_gap_manifest),
            "student prelaunch teacher",
        )
        _expect_equal(
            teacher,
            "merged_checkpoint",
            str(Path(args.teacher_checkpoint).resolve()),
            "student prelaunch teacher",
        )
        _expect_equal(
            teacher,
            "merged_checkpoint_tree_sha256",
            sha256_tree(
                Path(args.teacher_checkpoint),
                exclude_relative_paths=("merge_provenance.json",),
            ),
            "student prelaunch teacher",
        )
        provenance_raw = getattr(args, "teacher_provenance_manifest", None)
        if not provenance_raw:
            raise ValueError(
                "main-arm prelaunch receipt requires teacher provenance custody"
            )
        provenance_path = Path(provenance_raw).expanduser()
        if provenance_path.is_symlink() or not provenance_path.is_file():
            raise ValueError(
                "main-arm prelaunch teacher provenance must be a regular file"
            )
        provenance_path = provenance_path.resolve()
        canonical_provenance_path = (
            Path(args.teacher_checkpoint).resolve() / "merge_provenance.json"
        ).resolve()
        if provenance_path != canonical_provenance_path:
            raise ValueError(
                "main-arm prelaunch teacher provenance is not canonical in checkpoint"
            )
        _expect_equal(
            teacher,
            "merge_provenance_manifest_sha256",
            sha256_file(provenance_path),
            "student prelaunch teacher",
        )
        provenance = json.loads(provenance_path.read_text())
        if not isinstance(provenance, dict):
            raise ValueError("main-arm prelaunch teacher provenance must be a JSON object")
        _expect_equal(
            teacher,
            "merge_provenance_payload_sha256",
            canonical_json_sha256(provenance),
            "student prelaunch teacher",
        )
    for label in ("preregistration", "launch_ledger"):
        binding = receipt.get(label)
        if not isinstance(binding, dict):
            raise ValueError(f"student prelaunch receipt lacks {label} custody")
        bound_path = Path(str(binding.get("path")))
        _expect_equal(
            binding,
            "sha256",
            sha256_file(bound_path),
            f"student prelaunch {label}",
        )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "payload_sha256": canonical_json_sha256(receipt),
        "campaign_id": receipt.get("campaign_id"),
        "run_key": run_key,
        "sealed_before_optimizer_start": True,
        "preregistration": receipt["preregistration"],
        "launch_ledger": receipt["launch_ledger"],
    }


def installed_package_versions(expected: dict[str, str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in expected:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ValueError(f"required training package is not installed: {name}") from exc
    return versions


def _freeze_package_versions(path: Path) -> dict[str, str]:
    versions: dict[str, str] = {}
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "==" not in stripped:
            continue
        name, version = stripped.split("==", 1)
        versions[name.lower().replace("_", "-")] = version
    return versions


def validate_environment_contract(args, *, require_serve: bool) -> dict:
    """Bind exact live train/serve environments to immutable commit freezes."""

    state = git_state()
    commit = state.get("commit")
    if not clean_stable_git_custody(state, state):
        raise ValueError("scientific environment custody requires a clean 40-hex Git commit")

    def checked_freeze(
        raw: str | None, flag: str, filename: str, expected: dict[str, str]
    ) -> dict:
        if not raw:
            raise ValueError(f"scientific runs require {flag}")
        path = Path(raw)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"environment freeze must be a regular non-symlink file: {path}")
        path = path.resolve()
        if (
            path.name != filename
            or path.parent.name != commit
            or path.parent.parent.name != "environment_freezes"
        ):
            raise ValueError(
                f"environment freeze must be the commit-specific {filename} under a {commit} directory"
            )
        frozen = _freeze_package_versions(path)
        mismatched = {
            name: {"expected": version, "actual": frozen.get(name)}
            for name, version in expected.items()
            if frozen.get(name) != version
        }
        if mismatched:
            raise ValueError(f"environment freeze does not contain the pinned package set: {mismatched}")
        return {
            "path": str(path),
            "sha256": sha256_file(path),
            "required_packages": expected,
        }

    def checked_live_environment(
        raw_root: str | None, flag: str, kind: str, freeze: dict
    ) -> dict:
        if not raw_root:
            raise ValueError(f"scientific runs require {flag}")
        root = Path(raw_root).resolve(strict=True)
        expected_executable = root / "bin" / "vllm" if kind == "serve" else None
        verify = (
            verify_live_environment
            if kind == "train"
            else run_external_environment_verification
        )
        verification = verify(
            environment_root=root,
            commit_freeze=freeze["path"],
            expected_commit=commit,
            freeze_kind=kind,
            expected_executable=expected_executable,
        )
        if verification.get("commit_freeze") != {
            "path": freeze["path"],
            "sha256": freeze["sha256"],
            "byte_identical_to_requirements_freeze": True,
        }:
            raise ValueError(f"{kind} verification does not bind the selected commit freeze")
        return verification

    runtime_packages = installed_package_versions(EXPECTED_TRAIN_PACKAGES)
    if runtime_packages != EXPECTED_TRAIN_PACKAGES:
        raise ValueError(
            "live training packages differ from the pinned environment: "
            f"expected={EXPECTED_TRAIN_PACKAGES}, actual={runtime_packages}"
        )
    train_freeze = checked_freeze(
        args.train_environment_freeze,
        "--train-environment-freeze",
        "train.freeze.txt",
        EXPECTED_TRAIN_PACKAGES,
    )
    train_verification = checked_live_environment(
        getattr(args, "train_environment_root", None),
        "--train-environment-root",
        "train",
        train_freeze,
    )
    contract = {
        "schema_version": 2,
        "git_commit": commit,
        "verifier": {
            "path": str(ENVIRONMENT_VERIFIER.resolve()),
            "sha256": sha256_file(ENVIRONMENT_VERIFIER),
        },
        "train_runtime_packages": runtime_packages,
        "train_freeze": train_freeze,
        "train_verification": train_verification,
        "serve_freeze": None,
        "serve_verification": None,
    }
    if require_serve:
        serve_freeze = checked_freeze(
            args.serve_environment_freeze,
            "--serve-environment-freeze",
            "serve.freeze.txt",
            EXPECTED_SERVE_PACKAGES,
        )
        contract["serve_freeze"] = serve_freeze
        contract["serve_verification"] = checked_live_environment(
            getattr(args, "serve_environment_root", None),
            "--serve-environment-root",
            "serve",
            serve_freeze,
        )
    elif getattr(args, "serve_environment_root", None) or args.serve_environment_freeze:
        raise ValueError("task-RL baseline must not bind a teacher serve environment")
    return contract


def environment_contract_unchanged(contract: dict | None) -> bool:
    if not contract:
        return True
    if not isinstance(contract, dict):
        return False
    try:
        if contract.get("schema_version") != 2:
            return False
        verifier = contract.get("verifier")
        if verifier != {
            "path": str(ENVIRONMENT_VERIFIER.resolve()),
            "sha256": sha256_file(ENVIRONMENT_VERIFIER),
        }:
            return False
        for key in ("train_freeze", "serve_freeze"):
            binding = contract.get(key)
            if binding is None:
                continue
            path = Path(binding["path"])
            if (
                path.is_symlink()
                or not path.is_file()
                or sha256_file(path) != binding["sha256"]
            ):
                return False
        runtime = installed_package_versions(EXPECTED_TRAIN_PACKAGES)
        if runtime != contract.get("train_runtime_packages") or runtime != EXPECTED_TRAIN_PACKAGES:
            return False
        for kind in ("train", "serve"):
            freeze = contract.get(f"{kind}_freeze")
            recorded = contract.get(f"{kind}_verification")
            if freeze is None:
                if recorded is not None or kind == "train":
                    return False
                continue
            if not isinstance(recorded, dict):
                return False
            reverify_recorded_environment(recorded, in_process=kind == "train")
            commit_binding = recorded.get("commit_freeze")
            if not isinstance(commit_binding, dict):
                return False
            if (
                commit_binding.get("path") != freeze.get("path")
                or commit_binding.get("sha256") != freeze.get("sha256")
                or commit_binding.get("byte_identical_to_requirements_freeze") is not True
                or recorded.get("expected_commit") != contract.get("git_commit")
                or recorded.get("freeze_kind") != kind
            ):
                return False
    except (OSError, TypeError, ValueError):
        return False
    return True


def final_promotion_custody_failure_status(
    *,
    stable_final_artifact: bool,
    intended_scientific_run: bool,
    clean_stable_code: bool,
    stable_environment_end: bool,
) -> str | None:
    if not stable_final_artifact:
        return "failed_final_artifact_hash_custody"
    if intended_scientific_run and not clean_stable_code:
        return "failed_code_custody_after_promotion"
    if intended_scientific_run and not stable_environment_end:
        return "failed_environment_custody_after_promotion"
    return None


def validate_server_environment_process_binding(
    local_binding: dict, environment_contract: dict
) -> dict:
    verification = environment_contract.get("serve_verification")
    if not isinstance(verification, dict):
        raise ValueError("scientific main arm lacks exact serve-environment verification")
    executable = verification.get("expected_executable")
    if not isinstance(executable, dict):
        raise ValueError("serve-environment verification lacks bin/vllm custody")
    expected = expected_serve_environment_launcher(
        Path(verification["environment_root"])
    )
    if (
        verification.get("live_python") != expected["python"]
        or executable.get("path") != expected["vllm"]
    ):
        raise ValueError("serve-environment verification and launcher identity disagree")
    if local_binding.get("serve_environment_launcher") != expected:
        raise ValueError(
            "live local vLLM PID was not launched by the verified serve environment"
        )
    if local_binding.get("executable") != expected["resolved_python_executable"]:
        raise ValueError(
            "live local vLLM PID executable was not the resolved verified serve "
            "environment interpreter"
        )
    return expected


def git_worktree_is_clean() -> bool:
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return False
    return not status.strip()


def git_state() -> dict[str, str | bool | None]:
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
        return {"commit": commit, "dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def clean_stable_git_custody(start: dict, end: dict) -> bool:
    commit = start.get("commit")
    return bool(
        isinstance(commit, str)
        and re.fullmatch(r"[0-9a-f]{40}", commit)
        and start.get("dirty") is False
        and end.get("dirty") is False
        and end.get("commit") == commit
    )


def immutable_hub_revision(revision: str | None) -> bool:
    return bool(revision and re.fullmatch(r"[0-9a-fA-F]{40}", revision))


def checked_gate(
    path: str | None,
    name: str,
    allow_smoke: bool,
    *,
    expected_gate: str,
) -> dict | None:
    if not path:
        if allow_smoke:
            return None
        raise ValueError(f"{name} is required for this non-smoke run")
    gate_path = Path(path)
    raw = gate_path.read_bytes()
    payload = json.loads(raw)
    if payload.get("gate") != expected_gate:
        raise ValueError(
            f"{name} has gate={payload.get('gate')!r}, expected {expected_gate!r}: {path}"
        )
    if not payload.get("passed"):
        raise ValueError(f"{name} did not pass: {path}")
    payload["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    return payload


def _expect_equal(payload: dict, key: str, expected, label: str) -> None:
    actual = payload.get(key)
    if actual != expected:
        raise ValueError(f"{label} {key} mismatch: expected={expected!r}, actual={actual!r}")


def _manifest_file(prepared: dict, relative_path: str) -> dict:
    files = prepared.get("files")
    if not isinstance(files, dict) or relative_path not in files:
        raise ValueError(f"prepared manifest does not register required role file: {relative_path}")
    entry = files[relative_path]
    if not isinstance(entry, dict) or not entry.get("sha256"):
        raise ValueError(f"prepared manifest has an invalid file entry: {relative_path}")
    return entry


def _pair_by_id(prepared: dict, pair_id: str) -> dict:
    matches = [pair for pair in prepared.get("pairs", []) if pair.get("id") == pair_id]
    if len(matches) != 1:
        raise ValueError(f"prepared manifest does not contain exactly one pair {pair_id!r}")
    return matches[0]


def _validate_role_rows(rows: list[dict], *, source: str, role: str) -> None:
    mismatched = [
        index
        for index, row in enumerate(rows)
        if row.get("source") != source or row.get("role") != role
    ]
    if mismatched:
        raise ValueError(
            f"selected task rows must all be source={source!r}, role={role!r}; "
            f"mismatches at {mismatched[:10]}"
        )


def _validate_full_role_file(path: Path, *, source: str, role: str) -> int:
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("source") != source or row.get("role") != role:
                raise ValueError(
                    f"{path}:{line_number}: expected source={source!r}, role={role!r}"
                )
            count += 1
    if count <= 0:
        raise ValueError(f"registered role file is empty: {path}")
    return count


def _validate_gate_prepared_binding(
    gate: dict,
    *,
    prepared: dict,
    prepared_manifest_path: str,
    relative_task: str,
    label: str,
) -> None:
    manifest_path = Path(prepared_manifest_path).resolve()
    _expect_equal(gate, "prepared_manifest", str(manifest_path), label)
    _expect_equal(gate, "prepared_manifest_sha256", sha256_file(manifest_path), label)
    _expect_equal(gate, "registered_task_file", relative_task, label)
    role_entry = _manifest_file(prepared, relative_task)
    _expect_equal(gate, "registered_task_rows", int(role_entry["rows"]), label)

    source_raw = prepared.get("source_manifest_path")
    if not isinstance(source_raw, str) or not source_raw:
        raise ValueError("prepared manifest lacks source_manifest_path")
    source_path = Path(source_raw)
    if not source_path.is_absolute():
        source_path = manifest_path.parent / source_path
    source_path = source_path.resolve()
    source_hash = sha256_file(source_path)
    _expect_equal(prepared, "source_manifest_sha256", source_hash, "prepared manifest")
    _expect_equal(gate, "source_manifest", str(source_path), label)
    _expect_equal(gate, "source_manifest_sha256", source_hash, label)


def _validate_deterministic_gate_recomputation(gate: dict, *, kind: str) -> None:
    original = dict(gate)
    original.pop("manifest_sha256", None)
    if kind == "student":
        recomputed = recompute_student_gate(original)
    elif kind == "teacher":
        recomputed = recompute_teacher_gate(original)
    else:
        raise ValueError(f"unknown gate recomputation kind: {kind}")
    if recomputed != original:
        changed = sorted(
            key
            for key in set(original) | set(recomputed)
            if original.get(key) != recomputed.get(key)
        )
        raise ValueError(
            f"{kind} gate differs from deterministic recomputation of its bound artifacts; "
            f"changed_fields={changed[:20]}"
        )


def _validate_student_gate(
    gate: dict,
    *,
    args,
    task_hash: str,
    student_source: str,
    prepared: dict,
    current_environment: dict | None,
) -> None:
    _expect_equal(gate, "schema_version", 3, "student support gate")
    _expect_equal(gate, "gate", STUDENT_GATE_TYPE, "student support gate")
    _expect_equal(gate, "student_model", args.student, "student support gate")
    _expect_equal(gate, "student_model_revision", args.student_revision, "student support gate")
    _expect_equal(gate, "task_file_sha256", task_hash, "student support gate")
    _expect_equal(gate, "task_sources", [student_source], "student support gate")
    _expect_equal(gate, "task_roles", ["student_opd"], "student support gate")
    relative_task = f"roles/{student_source}/student_opd.jsonl"
    _validate_gate_prepared_binding(
        gate,
        prepared=prepared,
        prepared_manifest_path=args.prepared_manifest,
        relative_task=relative_task,
        label="student support gate",
    )
    _expect_equal(
        gate,
        "primary_matched_role_budget",
        int(prepared["primary_matched_budgets"]["student_opd"]),
        "student support gate",
    )
    _expect_equal(gate, "pinned_model_kind", "student", "student support gate")
    _expect_equal(gate, "pinned_model", args.student, "student support gate")
    _expect_equal(
        gate, "pinned_model_revision", args.student_revision, "student support gate"
    )
    _expect_equal(gate, "samples_per_problem", args.group_size, "student support gate")
    _expect_equal(
        gate,
        "evaluation_contract",
        EVALUATION_CONTRACT,
        "student support gate",
    )
    if not isinstance(gate.get("evaluation_environment"), dict):
        raise ValueError("student support gate lacks exact evaluation environment custody")
    if not isinstance(gate.get("evaluation_post_promotion_custody"), dict):
        raise ValueError("student support gate lacks post-promotion evaluation custody")
    if not args.allow_ungated_smoke:
        if not isinstance(current_environment, dict):
            raise ValueError("scientific student training lacks its current environment")
        _expect_equal(
            gate,
            "evaluation_git_commit",
            current_environment.get("git_commit"),
            "student support gate/current training commit",
        )
        gate_environment = gate["evaluation_environment"]
        _expect_equal(
            gate_environment,
            "verifier",
            current_environment.get("verifier"),
            "student support gate/current training environment",
        )
        gate_freeze = gate_environment.get("train_freeze")
        current_freeze = current_environment.get("train_freeze")
        if not isinstance(gate_freeze, dict) or not isinstance(current_freeze, dict):
            raise ValueError("student support/current training freeze binding is incomplete")
        for field in ("path", "sha256"):
            _expect_equal(
                gate_freeze,
                field,
                current_freeze.get(field),
                "student support gate/current training freeze",
            )
        _expect_equal(
            gate_environment,
            "train_verification",
            current_environment.get("train_verification"),
            "student support gate/current training environment",
        )
    decoding = gate.get("decoding") or {}
    for key, expected in (
        ("thinking", False),
        ("temperature", args.temperature),
        ("top_p", args.top_p),
        ("top_k", args.top_k),
        ("max_new_tokens", args.max_new_tokens),
        ("seed", args.seed),
    ):
        _expect_equal(decoding, key, expected, "student support gate decoding")
    if not args.allow_ungated_smoke and gate.get("gate_strength") != "scientific":
        raise ValueError("student support gate is not a scientific-strength gate")
    if not args.allow_ungated_smoke and gate.get("authorizes_scientific_training") is not True:
        raise ValueError("student support gate does not authorize scientific training")
    if not args.allow_ungated_smoke:
        _validate_deterministic_gate_recomputation(gate, kind="student")


def _validate_teacher_gate(
    gate: dict, *, pair: dict, prepared: dict, prepared_manifest_path: str
) -> None:
    _expect_equal(gate, "schema_version", 3, "teacher gap gate")
    teacher_source = pair["teacher_source"]
    skill_file = pair["teacher_skill_dev_file"]
    skill_entry = _manifest_file(prepared, skill_file)
    _validate_gate_prepared_binding(
        gate,
        prepared=prepared,
        prepared_manifest_path=prepared_manifest_path,
        relative_task=skill_file,
        label="teacher gap gate",
    )
    _expect_equal(gate, "task_file_sha256", skill_entry["sha256"], "teacher gap gate")
    _expect_equal(gate, "task_sources", [teacher_source], "teacher gap gate")
    _expect_equal(gate, "task_roles", ["teacher_gap_dev"], "teacher gap gate")
    _expect_equal(gate, "pinned_model_kind", "teacher", "teacher gap gate")
    if gate.get("gate_strength") != "scientific":
        raise ValueError("teacher gap gate is not a scientific-strength gate")
    if gate.get("authorizes_scientific_merge") is not True:
        raise ValueError("teacher gap gate does not authorize scientific checkpoint use")
    _validate_deterministic_gate_recomputation(gate, kind="teacher")


def _validate_tokenizer_contract(contract: dict, *, args) -> None:
    student = contract.get("student") or {}
    _expect_equal(student, "model", args.student, "tokenizer contract student")
    _expect_equal(student, "revision", args.student_revision, "tokenizer contract student")
    teacher = contract.get("teacher") or {}
    _expect_equal(teacher, "model", args.teacher_checkpoint, "tokenizer contract teacher")
    if not contract.get("exact_contract_match"):
        raise ValueError("tokenizer contract lacks an exact local tokenizer match")
    _expect_equal(
        contract.get("server") or {},
        "url",
        args.teacher_url.rstrip("/"),
        "tokenizer contract server",
    )
    _expect_equal(
        contract.get("server") or {},
        "model",
        args.teacher_model,
        "tokenizer contract server",
    )
    if not (contract.get("server_probe") or {}).get("matches"):
        raise ValueError("tokenizer contract lacks a passing live server probe")


def _validate_server_scoring_contract(path: str | None, *, args) -> dict:
    if not path:
        if args.allow_ungated_smoke:
            return {}
        raise ValueError("--server-scoring-contract is required for the scientific main arm")
    probe_path = Path(path)
    probe = json.loads(probe_path.read_text())
    _expect_equal(probe, "schema_version", 2, "server scoring contract")
    _expect_equal(probe, "probe", "exact_token_teacher_scoring_v1", "server scoring contract")
    _expect_equal(probe, "passed", True, "server scoring contract")
    _expect_equal(probe, "tokenizer", args.student, "server scoring contract")
    _expect_equal(probe, "tokenizer_revision", args.student_revision, "server scoring contract")
    _expect_equal(
        probe, "server_url", args.teacher_url.rstrip("/"), "server scoring contract"
    )
    _expect_equal(probe, "server_model", args.teacher_model, "server scoring contract")
    if not args.allow_ungated_smoke:
        _expect_equal(
            probe,
            "local_process_binding_validated",
            True,
            "server scoring contract",
        )
        binding = probe.get("local_process_binding")
        if not isinstance(binding, dict):
            raise ValueError("server scoring contract lacks a local process binding")
        _expect_equal(
            binding,
            "scope",
            LOCAL_BINDING_SCOPE,
            "server scoring local process binding",
        )
        if not args.teacher_provenance_manifest:
            raise ValueError(
                "scientific local server revalidation requires --teacher-provenance-manifest"
            )
        revalidate_local_process_binding(
            binding,
            teacher_checkpoint=Path(args.teacher_checkpoint),
            teacher_provenance_manifest=Path(args.teacher_provenance_manifest),
            server_url=args.teacher_url,
            server_model=args.teacher_model,
            server_max_model_len=args.teacher_server_max_model_len,
            serve_environment_root=Path(args.serve_environment_root),
        )
    probe["manifest_sha256"] = sha256_file(probe_path)
    return probe


def _validate_teacher_provenance(path: str | None, teacher_gate: dict, args) -> dict:
    if not path:
        raise ValueError("--teacher-provenance-manifest is required for the scientific main arm")
    provenance_path = Path(path)
    provenance = json.loads(provenance_path.read_text())
    if provenance.get("schema_version") != 1 or provenance.get("schema") != MERGED_TEACHER_SCHEMA:
        raise ValueError("teacher provenance has the wrong schema")
    _expect_equal(provenance, "status", "completed", "teacher provenance")
    _expect_equal(
        provenance,
        "teacher_gap_manifest_sha256",
        teacher_gate["manifest_sha256"],
        "teacher provenance",
    )
    _expect_equal(provenance, "base_model", args.teacher_base_model, "teacher provenance")
    _expect_equal(provenance, "base_revision", args.teacher_base_revision, "teacher provenance")
    _expect_equal(
        provenance,
        "adapter",
        teacher_gate["trained_adapter"],
        "teacher provenance",
    )
    _expect_equal(
        provenance,
        "adapter_tree_sha256",
        teacher_gate["trained_adapter_tree_sha256"],
        "teacher provenance",
    )
    teacher_training_environment = teacher_gate.get("teacher_training_environment")
    if not isinstance(teacher_training_environment, dict):
        raise ValueError("teacher gap lacks exact teacher train-environment custody")
    _expect_equal(
        provenance,
        "teacher_training_environment",
        teacher_training_environment,
        "teacher provenance",
    )
    _expect_equal(
        provenance,
        "output_checkpoint",
        str(Path(args.teacher_checkpoint).resolve()),
        "teacher provenance",
    )
    checkpoint_hash = provenance.get("output_checkpoint_tree_sha256")
    if not isinstance(checkpoint_hash, str) or len(checkpoint_hash) != 64:
        raise ValueError("teacher provenance lacks output checkpoint tree identity")
    live_checkpoint_hash = sha256_tree(
        Path(args.teacher_checkpoint), exclude_relative_paths=("merge_provenance.json",)
    )
    if live_checkpoint_hash != checkpoint_hash:
        raise ValueError("merged teacher checkpoint changed after provenance was written")

    merge_code = provenance.get("merge_code")
    if not isinstance(merge_code, dict):
        raise ValueError("teacher provenance lacks merge-code custody")
    merge_states = []
    for field in (
        "git_state_start",
        "git_state_after_merge",
        "git_state_before_promotion",
        "git_state_end",
    ):
        state = merge_code.get(field)
        if not isinstance(state, dict):
            raise ValueError(f"teacher provenance lacks merge-code {field}")
        merge_states.append(state)
    merge_start = merge_states[0]
    if any(not clean_stable_git_custody(merge_start, state) for state in merge_states):
        raise ValueError("teacher provenance merge Git custody is not clean and stable")
    if merge_code.get("clean_stable_code") is not True:
        raise ValueError("teacher provenance does not attest completed clean merge code")
    current_code = git_state()
    if not clean_stable_git_custody(merge_start, current_code):
        raise ValueError(
            "teacher checkpoint was not merged by the same clean Git commit as student training"
        )
    _expect_equal(
        merge_code,
        "merger_file_sha256",
        sha256_file(MERGER_FILE),
        "teacher provenance merge code",
    )
    _expect_equal(
        merge_code,
        "packages",
        EXPECTED_MERGE_PACKAGES,
        "teacher provenance merge code",
    )
    provenance["manifest_sha256"] = sha256_file(provenance_path)
    return provenance


def validate_run_contract(
    args, rows: list[dict]
) -> tuple[
    dict | None,
    dict | None,
    dict | None,
    dict | None,
    dict | None,
    dict | None,
    dict,
]:
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.max_new_tokens <= 0 or args.max_prompt_tokens <= 0:
        raise ValueError("prompt and completion token limits must be positive")
    if args.lr <= 0 or args.grad_clip <= 0 or args.lora <= 0:
        raise ValueError("learning rate, gradient clip, and LoRA rank must be positive")
    if args.group_size <= 0:
        raise ValueError("--group-size must be positive")
    if args.micro_prompts <= 0:
        raise ValueError("--micro-prompts must be positive")
    if args.advantage_clip is not None and (
        not math.isfinite(args.advantage_clip) or args.advantage_clip <= 0
    ):
        raise ValueError("--advantage-clip must be finite and positive when present")
    if args.mode in K1_MODES and args.advantage_clip is None and args.mode not in {
        "task_rl_k1_ungated_unclipped"
    }:
        raise ValueError("only the registered unclipped objective may omit advantage clipping")
    if not 0.0 <= args.min_informative_group_fraction <= 1.0:
        raise ValueError("--min-informative-group-fraction must be in [0, 1]")
    if args.task_reward_coef <= 0 and args.mode in TASK_REWARD_MODES:
        raise ValueError("task-reward modes require --task-reward-coef > 0")
    if args.k1_coef <= 0 and args.mode in K1_MODES:
        raise ValueError("sampled reverse-KL modes require --k1-coef > 0")
    if args.mode in GATED_K1_MODES and (
        args.gap_gate_beta is None or args.gap_gate_beta <= 0
    ):
        raise ValueError("gap-gated reverse-KL modes require --gap-gate-beta > 0")
    objective_registry_contract = getattr(args, "objective_registry_contract", None)
    if objective_registry_contract is not None and not args.allow_ungated_smoke:
        raise ValueError(
            "registered objective-family scientific launch is blocked until the sealed "
            "successor preregistration and custody validator are implemented"
        )
    if args.teacher_connect_timeout <= 0 or args.teacher_read_timeout <= 0 or args.teacher_retries <= 0:
        raise ValueError("teacher timeout and retry settings must be positive")
    if args.mode in TEACHER_MODES and (not args.teacher_url or not args.teacher_model):
        raise ValueError(f"mode {args.mode} requires --teacher-url and --teacher-model")
    if (
        args.mode == "task_rl_k1_gap"
        and not args.allow_ungated_smoke
        and args.pair_id in {"M_M", "M_O"}
    ):
        raise ValueError("M teacher failed its immutable gate; M_M/M_O are prohibited")
    if args.mode in K1_MODES and not args.teacher_checkpoint:
        raise ValueError(f"mode {args.mode} requires --teacher-checkpoint for identity custody")
    if args.mode in K1_MODES:
        if args.teacher_server_max_model_len <= 0:
            raise ValueError("sampled reverse-KL modes require --teacher-server-max-model-len")
        required_context = args.max_prompt_tokens + args.max_new_tokens + 1
        if required_context > args.teacher_server_max_model_len:
            raise ValueError(
                "teacher context overflow: max_prompt_tokens + max_new_tokens + 1 "
                f"is {required_context}, server limit is {args.teacher_server_max_model_len}"
            )
        if (args.temperature, args.top_p, args.top_k) != (1.0, 1.0, 0):
            raise ValueError(
                "sampled reverse KL requires untruncated on-policy sampling: "
                "--temperature 1 --top-p 1 --top-k 0"
            )
    student_training_plan = None
    if (
        args.mode in TASK_REWARD_MODES
        and not args.allow_ungated_smoke
        and args.budget_mode == "primary_matched"
    ):
        student_training_plan = validate_student_training_plan_contract(args)
    campaign_run_id = getattr(args, "campaign_run_id", None)
    scheduler_job_id = getattr(args, "scheduler_job_id", None)
    if campaign_run_id is not None and not re.fullmatch(
        r"[A-Za-z0-9._-]+", campaign_run_id
    ):
        raise ValueError("--campaign-run-id must be filesystem-safe")
    if scheduler_job_id is not None and not re.fullmatch(r"[1-9][0-9]*", scheduler_job_id):
        raise ValueError("--scheduler-job-id must be a positive decimal Slurm job ID")
    if (
        args.mode in TASK_REWARD_MODES
        and not args.allow_ungated_smoke
        and args.budget_mode == "primary_matched"
        and (campaign_run_id is None or scheduler_job_id is None)
    ):
        raise ValueError(
            "scientific primary runs require preregistered --campaign-run-id "
            "and --scheduler-job-id custody"
        )
    prelaunch_receipt = None
    primary_student_run = (
        args.mode in TASK_REWARD_MODES
        and not args.allow_ungated_smoke
        and args.budget_mode == "primary_matched"
    )
    prelaunch_receipt_arg = getattr(args, "prelaunch_receipt", None)
    if primary_student_run:
        if not prelaunch_receipt_arg:
            raise ValueError(
                "scientific primary runs require a sealed --prelaunch-receipt"
            )
        args.prelaunch_receipt = prelaunch_receipt_arg
        prelaunch_receipt = validate_prelaunch_receipt(args)
    elif prelaunch_receipt_arg:
        raise ValueError(
            "prelaunch receipts are reserved for preregistered primary matched runs"
        )
    if args.mode in TASK_REWARD_MODES:
        if not args.allow_ungated_smoke and args.enable_thinking:
            raise ValueError("scientific OPD-math runs require Qwen3 non-thinking mode")
        if not args.allow_ungated_smoke and not immutable_hub_revision(args.student_revision):
            raise ValueError(
                "scientific OPD-math runs require an immutable 40-hex --student-revision"
            )
        if not args.allow_ungated_smoke and not git_worktree_is_clean():
            raise ValueError("scientific OPD-math runs require a clean Git worktree")
        if args.group_size < 2:
            raise ValueError("task reward requires --group-size >= 2")
        missing = [i for i, row in enumerate(rows) if not row.get("solution")]
        if missing:
            raise ValueError(f"task rows missing solution at indices {missing[:10]}")

    teacher_gate = student_gate = tokenizer_contract = teacher_provenance = None
    server_scoring_contract = None
    prepared_manifest = None
    binding: dict = {
        "pair_id": None,
        "student_source": args.student_source,
        "teacher_source": None,
        "budget_mode": args.budget_mode,
        "campaign_run_id": campaign_run_id,
        "scheduler_job_id": scheduler_job_id,
        "local_checkpoint_custody_validated": False,
        "server_alias_and_token_contract_validated": False,
        "live_local_server_process_binding_validated": False,
        "serve_environment_process_binding_validated": False,
        "server_binding_claim_boundary": (
            "Local Linux process custody is not cryptographic remote attestation."
        ),
        "environment_contract": None,
        "student_training_plan": student_training_plan,
        "prelaunch_receipt": prelaunch_receipt,
        "objective_registry": objective_registry_contract,
    }
    if args.mode in TASK_REWARD_MODES:
        if not args.allow_ungated_smoke:
            binding["environment_contract"] = validate_environment_contract(
                args, require_serve=args.mode == "task_rl_k1_gap"
            )
        if not args.prepared_manifest:
            raise ValueError("task-reward runs require --prepared-manifest")
        if args.budget_mode not in {"primary_matched", "dose_response"}:
            raise ValueError("task-reward runs require an explicit --budget-mode")
        prepared = json.loads(Path(args.prepared_manifest).read_text())
        if not args.allow_ungated_smoke and not prepared.get("scientific_use_allowed"):
            raise ValueError("prepared data manifest is marked non-scientific")
        if args.mode == "task_rl":
            if args.pair_id:
                raise ValueError("task_rl has no teacher coordinate; use --student-source, not --pair-id")
            if args.student_source not in {"M", "O"}:
                raise ValueError("task_rl requires --student-source M or O")
            pair = None
            student_source = args.student_source
            relative_task = f"roles/{student_source}/student_opd.jsonl"
        else:
            if not args.pair_id:
                raise ValueError("task_rl_k1_gap requires --pair-id")
            if args.student_source:
                raise ValueError("task_rl_k1_gap infers student source from --pair-id")
            pair = _pair_by_id(prepared, args.pair_id)
            student_source = pair["opd_source"]
            relative_task = pair["student_opd_file"]
            binding.update(
                {
                    "pair_id": pair["id"],
                    "student_source": student_source,
                    "teacher_source": pair["teacher_source"],
                }
            )
        file_entry = _manifest_file(prepared, relative_task)
        expected_task_path = (Path(args.prepared_manifest).resolve().parent / relative_task).resolve()
        actual_task_path = Path(args.task_file).resolve()
        if actual_task_path != expected_task_path:
            raise ValueError(
                f"student task file must be the exact prepared role path: "
                f"expected={expected_task_path}, actual={actual_task_path}"
            )
        task_hash = sha256_file(args.task_file)
        _expect_equal(file_entry, "sha256", task_hash, "prepared student task file")
        physical_rows = _validate_full_role_file(
            actual_task_path, source=student_source, role="student_opd"
        )
        _expect_equal(file_entry, "rows", physical_rows, "prepared student task file")
        _validate_role_rows(rows, source=student_source, role="student_opd")
        if args.task_limit <= 0:
            raise ValueError("task-reward runs require a positive --task-limit")
        if args.task_limit > int(file_entry["rows"]):
            raise ValueError("--task-limit exceeds the registered student role file")
        matched_limit = int(prepared["primary_matched_budgets"]["student_opd"])
        if args.budget_mode == "primary_matched" and args.task_limit != matched_limit:
            raise ValueError(
                f"primary matched student runs require --task-limit={matched_limit}, "
                f"got {args.task_limit}"
            )
        binding.update(
            {
                "task_role_file": relative_task,
                "task_file_rows": int(file_entry["rows"]),
                "matched_task_limit": matched_limit,
            }
        )
        student_gate = checked_gate(
            args.student_support_manifest,
            "student support manifest",
            args.allow_ungated_smoke,
            expected_gate=STUDENT_GATE_TYPE,
        )
        if student_gate is not None:
            _validate_student_gate(
                student_gate,
                args=args,
                task_hash=task_hash,
                student_source=student_source,
                prepared=prepared,
                current_environment=binding["environment_contract"],
            )
        if args.mode == "task_rl_k1_gap":
            teacher_gate = checked_gate(
                args.teacher_gap_manifest,
                "teacher gap manifest",
                args.allow_ungated_smoke,
                expected_gate="teacher_gap_v1",
            )
            if teacher_gate is not None:
                _validate_teacher_gate(
                    teacher_gate,
                    pair=pair,
                    prepared=prepared,
                    prepared_manifest_path=args.prepared_manifest,
                )
                if not args.allow_ungated_smoke and (
                    not args.teacher_base_model or not args.teacher_base_revision
                ):
                    raise ValueError(
                        "scientific main arm requires teacher base model and revision identity"
                    )
                if args.teacher_base_model:
                    _expect_equal(
                        teacher_gate, "base_model", args.teacher_base_model, "teacher gap gate"
                    )
                if args.teacher_base_revision:
                    _expect_equal(
                        teacher_gate,
                        "base_model_revision",
                        args.teacher_base_revision,
                        "teacher gap gate",
                    )
            tokenizer_contract = checked_gate(
                args.tokenizer_contract,
                "tokenizer contract",
                args.allow_ungated_smoke,
                expected_gate="tokenizer_contract_v1",
            )
            if tokenizer_contract is not None:
                _validate_tokenizer_contract(tokenizer_contract, args=args)
            server_scoring_contract = _validate_server_scoring_contract(
                args.server_scoring_contract, args=args
            )
            if not args.allow_ungated_smoke:
                teacher_provenance = _validate_teacher_provenance(
                    args.teacher_provenance_manifest, teacher_gate, args
                )
                local_binding = server_scoring_contract["local_process_binding"]
                _expect_equal(
                    local_binding,
                    "teacher_checkpoint_tree_sha256",
                    teacher_provenance["output_checkpoint_tree_sha256"],
                    "server scoring local process binding",
                )
                _expect_equal(
                    local_binding,
                    "teacher_provenance_manifest_sha256",
                    teacher_provenance["manifest_sha256"],
                    "server scoring local process binding",
                )
                validate_server_environment_process_binding(
                    local_binding, binding["environment_contract"]
                )
                binding["local_checkpoint_custody_validated"] = True
                binding["server_alias_and_token_contract_validated"] = True
                binding["live_local_server_process_binding_validated"] = True
                binding["serve_environment_process_binding_validated"] = True
        prepared_manifest = {
            "path": str(Path(args.prepared_manifest).resolve()),
            "sha256": sha256_file(args.prepared_manifest),
            "task_role_file": relative_task,
            "task_file_sha256": task_hash,
            "scientific_use_allowed": prepared.get("scientific_use_allowed"),
        }
    elif args.mode in K1_MODES:
        tokenizer_contract = checked_gate(
            args.tokenizer_contract,
            "tokenizer contract",
            args.allow_ungated_smoke,
            expected_gate="tokenizer_contract_v1",
        )
        if tokenizer_contract is not None:
            _validate_tokenizer_contract(tokenizer_contract, args=args)
    return (
        teacher_gate,
        student_gate,
        tokenizer_contract,
        teacher_provenance,
        server_scoring_contract,
        prepared_manifest,
        binding,
    )


def _local_server_process_binding_state(
    mode: str, intended_scientific_run: bool, binding: dict
) -> tuple[bool, bool]:
    """Return whether live binding is required and whether it actually passed.

    An ungated smoke is allowed to omit local process custody, but omission is
    not validation.  Keep these two facts separate in the completion manifest.
    """
    required = intended_scientific_run and mode == "task_rl_k1_gap"
    validated = binding.get("live_local_server_process_binding_validated") is True
    return required, validated


def _server_process_binding_gate_satisfied(required: bool, validated: bool) -> bool:
    """A teacher-server binding must validate only when the mode requires one."""

    return not required or validated


def sample_trace_rows(samples, student_lps, teacher_lps, mask, rewards, statuses, step: int):
    rows = []
    for i, sample in enumerate(samples):
        selected = mask[i]
        student_values = student_lps[i][selected].detach().float()
        teacher_values = None if teacher_lps is None else teacher_lps[i][selected].detach().float()
        if not torch.isfinite(student_values).all() or (
            teacher_values is not None and not torch.isfinite(teacher_values).all()
        ):
            raise RuntimeError("student trace contains non-finite token log-probabilities")
        student_token_logprobs = [float(value) for value in student_values.tolist()]
        teacher_token_logprobs = (
            None
            if teacher_values is None
            else [float(value) for value in teacher_values.tolist()]
        )
        behavior_token_logprobs = sample.get("behavior_logprobs")
        if behavior_token_logprobs is not None:
            if (
                not isinstance(behavior_token_logprobs, list)
                or len(behavior_token_logprobs) != len(student_token_logprobs)
                or any(
                    type(value) not in (int, float) or not math.isfinite(float(value))
                    for value in behavior_token_logprobs
                )
            ):
                raise RuntimeError("sample trace has invalid rollout behavior log-probabilities")
            behavior_token_logprobs = [
                float(value) for value in behavior_token_logprobs
            ]
        student_nll = -sum(student_token_logprobs) / len(student_token_logprobs)
        teacher_nll = (
            None
            if teacher_token_logprobs is None
            else -sum(teacher_token_logprobs) / len(teacher_token_logprobs)
        )
        token_gaps = (
            None
            if teacher_token_logprobs is None
            else [
                teacher_logprob - student_logprob
                for teacher_logprob, student_logprob in zip(
                    teacher_token_logprobs,
                    student_token_logprobs,
                    strict=True,
                )
            ]
        )
        row = {
            "schema_version": 3 if behavior_token_logprobs is not None else 2,
            "step": step,
            "record_id": sample.get("record_id"),
            "source": sample.get("source"),
            "group_id": int(sample["group_id"]),
            "sample_idx": int(sample["sample_idx"]),
            "completion_tokens": int(selected.sum().item()),
            "prompt_tokens": len(sample["prompt_token_ids"]),
            "terminated_by_eos": bool(sample.get("terminated_by_eos")),
            "rollout_batch_latency_seconds": sample.get("rollout_batch_latency_seconds"),
            "teacher_scoring_latency_seconds": sample.get("teacher_scoring_latency_seconds"),
            "completion_sha256": hashlib.sha256(sample["completion_text"].encode("utf-8")).hexdigest(),
            "prompt_sha256": sample["prompt_sha256"],
            "prompt_token_ids": list(sample["prompt_token_ids"]),
            "completion_token_ids": list(sample["completion_token_ids"]),
            "completion_text": sample["completion_text"],
            "student_token_logprobs": student_token_logprobs,
            "behavior_token_logprobs_on_student_trajectory": behavior_token_logprobs,
            "teacher_token_logprobs_on_student_trajectory": teacher_token_logprobs,
            "student_nll": student_nll,
            "teacher_nll_on_student_trajectory": teacher_nll,
            "mean_teacher_student_gap": (
                None if token_gaps is None else sum(token_gaps) / len(token_gaps)
            ),
            "mean_abs_k1_log_ratio": (
                None
                if token_gaps is None
                else sum(abs(value) for value in token_gaps) / len(token_gaps)
            ),
            "min_teacher_student_gap": (
                None if token_gaps is None else min(token_gaps)
            ),
            "max_teacher_student_gap": (
                None if token_gaps is None else max(token_gaps)
            ),
            "positive_teacher_gap_fraction": (
                None
                if token_gaps is None
                else sum(value > 0 for value in token_gaps) / len(token_gaps)
            ),
            "reward": None if rewards is None else float(rewards[i]),
            "reward_status": None if statuses is None else statuses[i],
        }
        rows.append(row)
    return rows


def recompute_student_trace_geometry(
    *,
    steps_path: Path,
    samples_path: Path,
    mode: str,
    expected_steps: int,
    micro_prompts: int,
    group_size: int,
    max_prompt_tokens: int,
    max_completion_tokens: int,
    expected_groups: dict[tuple[int, int], dict],
    tokenizer,
    loss_config: dict | None = None,
    require_behavior_logprobs: bool = False,
) -> dict:
    """Fail closed on the exact student step, group, sample, and prompt trace."""

    step_rows = read_jsonl_objects(steps_path)
    if len(step_rows) != expected_steps:
        raise ValueError("student step trace does not match the planned optimizer-step count")
    for expected_step, row in enumerate(step_rows, 1):
        if (
            row.get("schema_version") != 1
            or row.get("step") != expected_step
            or row.get("mode") != mode
            or row.get("prompts") != micro_prompts
            or row.get("samples") != micro_prompts * group_size
        ):
            raise ValueError(f"student step trace geometry drifted at step {expected_step}")
        for field in ("total_loss", "gradient_norm_before_clip"):
            value = row.get(field)
            if type(value) not in (int, float) or not math.isfinite(float(value)):
                raise ValueError(
                    f"student step trace has invalid {field} at step {expected_step}"
                )
        if require_behavior_logprobs:
            for field in ("parameter_update_l2", "optimizer_state_squared_l2"):
                value = row.get(field)
                if (
                    type(value) not in (int, float)
                    or not math.isfinite(float(value))
                    or float(value) < 0
                ):
                    raise ValueError(
                        f"registered objective step trace has invalid {field} "
                        f"at step {expected_step}"
                    )
            for field in ("optimizer_state_tensors", "optimizer_state_elements"):
                value = row.get(field)
                if type(value) is not int or value <= 0:
                    raise ValueError(
                        f"registered objective step trace has invalid {field} "
                        f"at step {expected_step}"
                    )

    expected_keys = {
        (step, group_id)
        for step in range(1, expected_steps + 1)
        for group_id in range(micro_prompts)
    }
    if set(expected_groups) != expected_keys:
        raise ValueError("in-memory student prompt groups do not match the planned geometry")

    sample_rows = read_jsonl_objects(samples_path)
    expected_samples = expected_steps * micro_prompts * group_size
    if len(sample_rows) != expected_samples:
        raise ValueError("student sample trace does not match the planned rollout count")
    grouped: dict[tuple[int, int], list[dict]] = {}
    completion_tokens = 0
    sample_expanded_prompt_tokens = 0
    for row_number, row in enumerate(sample_rows, 1):
        schema_version = row.get("schema_version")
        if schema_version not in {2, 3}:
            raise ValueError(f"student sample trace row {row_number} has an invalid schema")
        if require_behavior_logprobs and schema_version != 3:
            raise ValueError(
                f"registered objective trace row {row_number} lacks behavior-logprob schema"
            )
        step = row.get("step")
        group_id = row.get("group_id")
        sample_idx = row.get("sample_idx")
        if type(step) is not int or not 1 <= step <= expected_steps:
            raise ValueError(f"student sample trace row {row_number} has an invalid step")
        if type(group_id) is not int or not 0 <= group_id < micro_prompts:
            raise ValueError(f"student sample trace row {row_number} has an invalid group_id")
        if type(sample_idx) is not int or not 0 <= sample_idx < group_size:
            raise ValueError(f"student sample trace row {row_number} has an invalid sample_idx")
        key = (step, group_id)
        expected = expected_groups.get(key)
        if expected is None:
            raise ValueError(f"student sample trace row {row_number} has an unknown prompt group")
        for field in ("record_id", "source", "prompt_sha256"):
            if row.get(field) != expected[field]:
                raise ValueError(
                    f"student sample trace row {row_number} has {field} identity drift"
                )
        prompt_ids = row.get("prompt_token_ids")
        completion_ids = row.get("completion_token_ids")
        if (
            not isinstance(prompt_ids, list)
            or not prompt_ids
            or any(type(value) is not int or value < 0 for value in prompt_ids)
            or len(prompt_ids) > max_prompt_tokens
            or prompt_ids != expected["prompt_token_ids"]
        ):
            raise ValueError(f"student sample trace row {row_number} has invalid prompt tokens")
        if (
            not isinstance(completion_ids, list)
            or not completion_ids
            or any(type(value) is not int or value < 0 for value in completion_ids)
            or len(completion_ids) > max_completion_tokens
        ):
            raise ValueError(
                f"student sample trace row {row_number} has invalid completion tokens"
            )
        if row.get("prompt_tokens") != len(prompt_ids):
            raise ValueError(f"student sample trace row {row_number} has prompt-token drift")
        if row.get("completion_tokens") != len(completion_ids):
            raise ValueError(f"student sample trace row {row_number} has completion-token drift")
        student_logprobs = row.get("student_token_logprobs")
        if (
            not isinstance(student_logprobs, list)
            or len(student_logprobs) != len(completion_ids)
            or any(
                type(value) not in (int, float) or not math.isfinite(float(value))
                for value in student_logprobs
            )
        ):
            raise ValueError(
                f"student sample trace row {row_number} lacks exact student token log-probabilities"
            )
        behavior_logprobs = row.get(
            "behavior_token_logprobs_on_student_trajectory"
        )
        if schema_version == 3:
            if (
                not isinstance(behavior_logprobs, list)
                or len(behavior_logprobs) != len(completion_ids)
                or any(
                    type(value) not in (int, float)
                    or not math.isfinite(float(value))
                    for value in behavior_logprobs
                )
            ):
                raise ValueError(
                    f"student sample trace row {row_number} lacks exact behavior token log-probabilities"
                )
        elif behavior_logprobs is not None:
            raise ValueError(
                f"legacy student sample trace row {row_number} has unversioned behavior log-probabilities"
            )
        teacher_logprobs = row.get("teacher_token_logprobs_on_student_trajectory")
        if mode in K1_MODES:
            if (
                not isinstance(teacher_logprobs, list)
                or len(teacher_logprobs) != len(completion_ids)
                or any(
                    type(value) not in (int, float) or not math.isfinite(float(value))
                    for value in teacher_logprobs
                )
            ):
                raise ValueError(
                    f"student sample trace row {row_number} lacks exact teacher token log-probabilities"
                )
        elif teacher_logprobs is not None:
            raise ValueError(
                f"student sample trace row {row_number} unexpectedly contains teacher log-probabilities"
            )
        recomputed_student_nll = -sum(float(value) for value in student_logprobs) / len(
            student_logprobs
        )
        if not math.isclose(
            float(row.get("student_nll", math.nan)),
            recomputed_student_nll,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError(f"student sample trace row {row_number} has student-NLL drift")
        if mode in K1_MODES:
            recomputed_teacher_nll = -sum(float(value) for value in teacher_logprobs) / len(
                teacher_logprobs
            )
            if not math.isclose(
                float(row.get("teacher_nll_on_student_trajectory", math.nan)),
                recomputed_teacher_nll,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise ValueError(f"student sample trace row {row_number} has teacher-NLL drift")
            token_gaps = [
                float(teacher_logprob) - float(student_logprob)
                for teacher_logprob, student_logprob in zip(
                    teacher_logprobs,
                    student_logprobs,
                    strict=True,
                )
            ]
            expected_gap_metrics = {
                "mean_teacher_student_gap": sum(token_gaps) / len(token_gaps),
                "mean_abs_k1_log_ratio": sum(abs(value) for value in token_gaps)
                / len(token_gaps),
                "min_teacher_student_gap": min(token_gaps),
                "max_teacher_student_gap": max(token_gaps),
                "positive_teacher_gap_fraction": sum(value > 0 for value in token_gaps)
                / len(token_gaps),
            }
            for field, expected_metric in expected_gap_metrics.items():
                if not math.isclose(
                    float(row.get(field, math.nan)),
                    expected_metric,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                ):
                    raise ValueError(
                        f"student sample trace row {row_number} has {field} drift"
                    )
        completion_text = row.get("completion_text")
        if not isinstance(completion_text, str):
            raise ValueError(f"student sample trace row {row_number} lacks completion text")
        if row.get("completion_sha256") != hashlib.sha256(
            completion_text.encode("utf-8")
        ).hexdigest():
            raise ValueError(f"student sample trace row {row_number} has completion drift")
        if tokenizer.decode(completion_ids, skip_special_tokens=True) != completion_text:
            raise ValueError(
                f"student sample trace row {row_number} text does not decode from its token IDs"
            )
        if type(row.get("terminated_by_eos")) is not bool:
            raise ValueError(f"student sample trace row {row_number} lacks EOS termination state")
        for field in ("rollout_batch_latency_seconds",):
            value = row.get(field)
            if type(value) not in (int, float) or not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"student sample trace row {row_number} has invalid {field}")
        teacher_latency = row.get("teacher_scoring_latency_seconds")
        if teacher_latency is not None and (
            type(teacher_latency) not in (int, float)
            or not math.isfinite(float(teacher_latency))
            or teacher_latency < 0
        ):
            raise ValueError(
                f"student sample trace row {row_number} has invalid teacher-scoring latency"
            )
        grouped.setdefault(key, []).append(row)
        completion_tokens += len(completion_ids)
        sample_expanded_prompt_tokens += len(prompt_ids)

    if set(grouped) != expected_keys:
        raise ValueError("student sample trace has missing or unexpected prompt groups")
    realized_record_ids: list[str] = []
    realized_prompt_sequence: list[dict[str, str]] = []
    prompt_group_tokens = 0
    for key in sorted(expected_keys):
        group = sorted(grouped[key], key=lambda row: row["sample_idx"])
        if [row["sample_idx"] for row in group] != list(range(group_size)):
            raise ValueError(f"student sample trace group {key} has missing/duplicate samples")
        expected = expected_groups[key]
        realized_record_ids.append(expected["record_id"])
        realized_prompt_sequence.append(
            {
                "record_id": expected["record_id"],
                "prompt_sha256": expected["prompt_sha256"],
            }
        )
        prompt_group_tokens += len(expected["prompt_token_ids"])

    effective_loss_config = loss_config or {
        "task_reward_coef": 1.0,
        "k1_coef": 0.01,
        "gap_gate_beta": 5.0,
        "advantage_clip": 5.0,
    }
    required_loss_fields = {
        "task_reward_coef",
        "k1_coef",
        "gap_gate_beta",
        "advantage_clip",
    }
    if set(effective_loss_config) != required_loss_fields:
        raise ValueError("student trace loss config lacks the exact audit coefficients")
    samples_by_step: dict[int, list[dict]] = {
        step: [] for step in range(1, expected_steps + 1)
    }
    for row in sample_rows:
        samples_by_step[int(row["step"])].append(row)
    for step, recorded in enumerate(step_rows, 1):
        gap_gate_beta = effective_loss_config["gap_gate_beta"]
        advantage_clip = effective_loss_config["advantage_clip"]
        reconstructed = reconstruct_step_metrics(
            samples_by_step[step],
            mode=mode,
            task_reward_coef=float(effective_loss_config["task_reward_coef"]),
            k1_coef=float(effective_loss_config["k1_coef"]),
            gap_gate_beta=(
                None if gap_gate_beta is None else float(gap_gate_beta)
            ),
            advantage_clip=(
                None if advantage_clip is None else float(advantage_clip)
            ),
        )
        validate_recorded_step_metrics(
            recorded,
            reconstructed,
            label=f"student step trace {step}",
        )

    return {
        "step_trace_rows": len(step_rows),
        "sample_trace_rows": len(sample_rows),
        "prompt_groups": len(expected_keys),
        "rollout_samples": len(sample_rows),
        "unique_training_records": len(set(realized_record_ids)),
        "realized_record_ids_sha256": canonical_json_sha256(realized_record_ids),
        "realized_prompt_sequence_sha256": canonical_json_sha256(
            realized_prompt_sequence
        ),
        "prompt_group_tokens": prompt_group_tokens,
        "sample_expanded_prompt_tokens": sample_expanded_prompt_tokens,
        "completion_tokens": completion_tokens,
        "expected_geometry_observed": True,
    }


def run(args) -> None:
    bind_registered_objective(args)
    code_state_start = git_state()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device={device} student={args.student} mode={args.mode}")
    rows = read_jsonl(args.task_file, args.task_limit)
    (
        teacher_gate,
        student_gate,
        tokenizer_contract,
        teacher_provenance,
        server_scoring_contract,
        prepared_manifest,
        binding,
    ) = validate_run_contract(args, rows)
    intended_scientific_run = (
        args.mode in TASK_REWARD_MODES
        and not args.allow_ungated_smoke
        and args.budget_mode == "primary_matched"
    )
    if intended_scientific_run and not clean_stable_git_custody(
        code_state_start, code_state_start
    ):
        raise ValueError("scientific OPD-math runs require a clean, identifiable Git start state")
    out_path = Path(args.out_dir)
    if out_path.is_symlink() or (
        out_path.exists() and (not out_path.is_dir() or any(out_path.iterdir()))
    ):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {out_path}")
    trace_dir = resolve_trace_directory(out_path, args.trace_dir)
    if trace_dir.is_symlink() or (
        trace_dir.exists()
        and (not trace_dir.is_dir() or any(trace_dir.iterdir()))
    ):
        raise FileExistsError(
            f"refusing to use a symlink or non-empty trace directory: {trace_dir}"
        )
    if args.mode in TEACHER_MODES:
        teacher_client.healthcheck(
            args.teacher_url,
            timeout=args.teacher_read_timeout,
            retries=args.teacher_retries,
            raise_on_error=True,
        )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    tok, model = load_student(args, device)
    rng = random.Random(args.seed)
    stream = prompt_stream(rows, rng)
    micro = args.micro_prompts
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    initial_parameter_signature = trainable_parameter_signature(model)
    out_path.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    registry_contract = getattr(args, "objective_registry_contract", None)
    objective_contract = (
        registry_contract["objective"]["objective_contract"]
        if registry_contract is not None
        else {
            "task_rl": "grouped_verifiable_math_task_reward_v1",
            "task_rl_k1_gap": "grouped_task_reward_plus_clipped_positive_gap_k1_value_reverse_kl_sf_surrogate_v1",
            "kd": "teacher_completion_supervised_nll_v1",
        }.get(args.mode, "sampled_k1_value_reverse_kl_sf_diagnostic_v1")
    )
    run_manifest = {
        "schema_version": 1,
        "objective": args.mode,
        "objective_contract": objective_contract,
        "objective_registry": registry_contract,
        "status": "started",
        "intended_scientific_run": intended_scientific_run,
        "scientific_use_allowed": False,
        "git_commit": code_state_start["commit"],
        "git_worktree_clean": code_state_start["dirty"] is False,
        "git_state_start": code_state_start,
        "task_file": str(Path(args.task_file).resolve()),
        "task_file_sha256": sha256_file(args.task_file),
        "selected_task_rows": len(rows),
        "task_limit": args.task_limit,
        "binding": binding,
        "student": args.student,
        "student_revision": args.student_revision,
        "teacher_model": args.teacher_model,
        "teacher_checkpoint": args.teacher_checkpoint,
        "teacher_base_model": args.teacher_base_model,
        "teacher_base_revision": args.teacher_base_revision,
        "optimizer_steps_planned": args.steps,
        "normalized_training_config": normalized_student_training_config(args),
        "micro_prompts_per_step": micro,
        "planned_rollout_samples": args.steps * micro * args.group_size,
        "seed": args.seed,
        "optimization": {
            "attn_implementation": args.attn_implementation,
            "gradient_checkpointing": args.gradient_checkpointing,
            "learning_rate": args.lr,
            "lora_r": args.lora,
        },
        "generation": {
            "group_size": args.group_size,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "enable_thinking": args.enable_thinking,
        },
        "loss": {
            "task_reward_coef": args.task_reward_coef,
            "k1_coef": args.k1_coef,
            "gap_gate_beta": args.gap_gate_beta,
            "advantage_clip": args.advantage_clip,
        },
        "gates": {
            "prepared_data": prepared_manifest,
            "teacher_gap": teacher_gate,
            "teacher_provenance": teacher_provenance,
            "server_scoring_contract": server_scoring_contract,
            "student_support": student_gate,
            "tokenizer_contract": tokenizer_contract,
        },
    }
    (trace_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n")

    run_started = time.time()
    t0 = run_started
    informative_task_steps = 0
    informative_task_groups = 0
    total_task_groups = 0
    total_scored_completion_tokens = 0
    total_rollout_samples = 0
    prompt_group_tokens = 0
    total_rollout_latency_seconds = 0.0
    total_teacher_scoring_latency_seconds = 0.0
    prompt_groups_seen = 0
    realized_record_ids: list[str] = []
    expected_trace_groups: dict[tuple[int, int], dict] = {}
    gradient_norms: list[float] = []
    parameter_update_norms: list[float] = []
    optimizer_state_signatures: list[dict[str, float | int]] = []
    for step in range(1, args.steps + 1):
        raw_batch = [next(stream) for _ in range(micro)]
        prompt_rows = []
        for group_id, row in enumerate(raw_batch):
            item = dict(row)
            item["prompt_text"], item["prompt_token_ids"] = render_prompt(
                tok, item, args.max_prompt_tokens, args.enable_thinking
            )
            prompt_rows.append(item)
            prompt_groups_seen += 1
            prompt_group_tokens += len(item["prompt_token_ids"])
            record_id = item.get("record_id")
            if intended_scientific_run and (not isinstance(record_id, str) or not record_id):
                raise ValueError("scientific training rows require a stable record_id")
            if isinstance(record_id, str) and record_id:
                realized_record_ids.append(record_id)
            expected_trace_groups[(step, group_id)] = {
                "record_id": record_id,
                "source": item.get("source"),
                "prompt_sha256": task_prompt_sha256(item),
                "prompt_token_ids": list(item["prompt_token_ids"]),
            }

        opt.zero_grad(set_to_none=True)
        if args.mode != "kd":
            samples = generate_student_samples(model, tok, prompt_rows, args, device)
            if args.mode in K1_MODES:
                samples = teacher_score_samples(samples, args)
            loss, metrics, student_lps, teacher_lps, score_mask, rewards, statuses = training_loss_for_samples(
                model, tok, samples, args, device
            )
            ntok = metrics["tokens"]
        else:
            samples = teacher_kd_samples(prompt_rows, args)
            model.eval()
            loss, ntok = kd_loss_for_samples(model, tok, samples, device)
            metrics = {
                "tokens": ntok,
                "task_loss": None,
                "reverse_kl_score_function_surrogate": None,
            }
            student_lps = teacher_lps = score_mask = rewards = statuses = None

        if args.mode in TASK_REWARD_MODES and float(metrics["informative_group_fraction"]) > 0:
            informative_task_steps += 1
        if args.mode in TASK_REWARD_MODES:
            total_task_groups += len(prompt_rows)
            informative_task_groups += int(
                round(float(metrics["informative_group_fraction"]) * len(prompt_rows))
            )
        total_scored_completion_tokens += int(ntok)
        total_rollout_samples += len(samples)
        if args.mode != "kd":
            total_rollout_latency_seconds += float(
                samples[0]["rollout_batch_latency_seconds"]
            )
            total_teacher_scoring_latency_seconds += sum(
                float(sample.get("teacher_scoring_latency_seconds") or 0.0)
                for sample in samples
            )

        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")
        before_step = trainable_parameter_snapshot(model)
        loss.backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip, error_if_nonfinite=True
            ).item()
        )
        if not math.isfinite(gradient_norm):
            raise RuntimeError(
                f"non-finite gradient norm at step {step}; optimizer step was not applied"
            )
        gradient_norms.append(gradient_norm)
        opt.step()
        update_norm = parameter_update_l2(model, before_step)
        optimizer_signature = optimizer_state_signature(opt)
        if optimizer_signature["tensors"] <= 0 or optimizer_signature["elements"] <= 0:
            raise RuntimeError(f"optimizer state was empty after step {step}")
        parameter_update_norms.append(update_norm)
        optimizer_state_signatures.append(optimizer_signature)

        elapsed = max(time.time() - t0, 1e-9)
        step_row = {
            "schema_version": 1,
            "step": step,
            "mode": args.mode,
            "prompts": len(prompt_rows),
            "samples": len(samples),
            "total_loss": float(loss.item()),
            "gradient_norm_before_clip": gradient_norm,
            "parameter_update_l2": update_norm,
            "optimizer_state_tensors": optimizer_signature["tensors"],
            "optimizer_state_elements": optimizer_signature["elements"],
            "optimizer_state_squared_l2": optimizer_signature["squared_l2"],
            "tokens_per_second": ntok / elapsed,
            **metrics,
        }
        append_jsonl(trace_dir / "steps.jsonl", step_row)
        if args.mode != "kd":
            for trace_row in sample_trace_rows(
                samples, student_lps, teacher_lps, score_mask, rewards, statuses, step
            ):
                append_jsonl(trace_dir / "samples.jsonl", trace_row)
        metric = " ".join(
            f"{key}={value:.4f}" for key, value in metrics.items() if isinstance(value, float)
        )
        log(
            f"step {step}/{args.steps} mode={args.mode} prompts={len(prompt_rows)} "
            f"samples={len(samples)} tokens={ntok} loss={loss.item():.4f} "
            f"{metric} toks_per_s={ntok / elapsed:.1f}"
        )
        t0 = time.time()

        if args.save_every > 0 and step % args.save_every == 0:
            save_checkpoint(model, tok, args.out_dir, f"step_{step:06d}")

    final_parameter_signature = trainable_parameter_signature(model)
    parameter_update_observed = signatures_differ(
        initial_parameter_signature, final_parameter_signature
    )
    finite_nonzero_gradient_observed = any(
        math.isfinite(value) and value > 0 for value in gradient_norms
    )
    informative_group_fraction = (
        1.0 if args.mode not in TASK_REWARD_MODES else informative_task_groups / total_task_groups
    )
    task_signal_observed = (
        args.mode not in TASK_REWARD_MODES
        or informative_group_fraction >= args.min_informative_group_fraction
    )
    step_trace_rows = count_jsonl_objects(trace_dir / "steps.jsonl")
    sample_trace_path = trace_dir / "samples.jsonl"
    observed_sample_trace_rows = (
        count_jsonl_objects(sample_trace_path) if sample_trace_path.is_file() else 0
    )
    expected_prompt_groups = args.steps * micro
    expected_rollout_samples = expected_prompt_groups * args.group_size
    selected_record_ids = {
        row.get("record_id")
        for row in rows
        if isinstance(row.get("record_id"), str) and row.get("record_id")
    }
    trace_geometry = None
    if args.mode != "kd":
        trace_geometry = recompute_student_trace_geometry(
            steps_path=trace_dir / "steps.jsonl",
            samples_path=sample_trace_path,
            mode=args.mode,
            expected_steps=args.steps,
            micro_prompts=micro,
            group_size=args.group_size,
            max_prompt_tokens=args.max_prompt_tokens,
            max_completion_tokens=args.max_new_tokens,
            expected_groups=expected_trace_groups,
            tokenizer=tok,
            require_behavior_logprobs=registry_contract is not None,
            loss_config={
                "task_reward_coef": args.task_reward_coef,
                "k1_coef": args.k1_coef,
                "gap_gate_beta": args.gap_gate_beta,
                "advantage_clip": args.advantage_clip,
            },
        )
    realized_record_ids_sha256 = canonical_json_sha256(realized_record_ids)
    realized_prompt_sequence_sha256 = canonical_json_sha256(
        [
            {
                "record_id": expected_trace_groups[key]["record_id"],
                "prompt_sha256": expected_trace_groups[key]["prompt_sha256"],
            }
            for key in sorted(expected_trace_groups)
        ]
    )
    realized_training_geometry_observed = (
        trace_geometry is not None
        and trace_geometry["expected_geometry_observed"] is True
        and prompt_groups_seen == expected_prompt_groups
        and total_task_groups == expected_prompt_groups
        and total_rollout_samples == expected_rollout_samples
        and trace_geometry["step_trace_rows"] == args.steps
        and trace_geometry["sample_trace_rows"] == expected_rollout_samples
        and trace_geometry["completion_tokens"] == total_scored_completion_tokens
        and trace_geometry["prompt_group_tokens"] == prompt_group_tokens
        and trace_geometry["realized_record_ids_sha256"]
        == realized_record_ids_sha256
        and trace_geometry["realized_prompt_sequence_sha256"]
        == realized_prompt_sequence_sha256
        and len(realized_record_ids) == expected_prompt_groups
        and set(realized_record_ids).issubset(selected_record_ids)
    )
    code_state_training_end = git_state()
    clean_stable_training_code = clean_stable_git_custody(
        code_state_start, code_state_training_end
    )
    stable_training_environment = environment_contract_unchanged(
        binding.get("environment_contract")
    )
    (
        server_process_binding_required,
        server_process_binding_validated,
    ) = _local_server_process_binding_state(
        args.mode, intended_scientific_run, binding
    )
    server_process_binding_end = None
    server_process_binding_error = None
    if server_process_binding_required:
        try:
            server_process_binding_end = revalidate_local_process_binding(
                server_scoring_contract["local_process_binding"],
                teacher_checkpoint=Path(args.teacher_checkpoint),
                teacher_provenance_manifest=Path(args.teacher_provenance_manifest),
                server_url=args.teacher_url,
                server_model=args.teacher_model,
                server_max_model_len=args.teacher_server_max_model_len,
                serve_environment_root=Path(args.serve_environment_root),
            )
            validate_server_environment_process_binding(
                server_process_binding_end, binding["environment_contract"]
            )
        except (OSError, ValueError, RuntimeError) as exc:
            server_process_binding_validated = False
            server_process_binding_error = f"{type(exc).__name__}: {exc}"
    require_parameter_update = args.require_parameter_update or intended_scientific_run
    completion = {
        "schema_version": 1,
        "status": (
            "completed"
            if task_signal_observed
            else ("failed_task_signal_gate" if intended_scientific_run else "completed_zero_task_signal_smoke")
        ),
        "objective": args.mode,
        "optimizer_steps_completed": args.steps,
        "rollout_samples": total_rollout_samples,
        "scored_completion_tokens": total_scored_completion_tokens,
        "prompt_group_tokens": prompt_group_tokens,
        "sample_expanded_prompt_tokens": (
            trace_geometry["sample_expanded_prompt_tokens"]
            if trace_geometry is not None
            else prompt_group_tokens * args.group_size
        ),
        "prompt_groups_seen": prompt_groups_seen,
        "step_trace_rows": step_trace_rows,
        "sample_trace_rows": observed_sample_trace_rows,
        "realized_training_geometry_observed": realized_training_geometry_observed,
        "unique_training_records": len(set(realized_record_ids)),
        "realized_record_ids_sha256": realized_record_ids_sha256,
        "realized_prompt_sequence_sha256": realized_prompt_sequence_sha256,
        "total_training_elapsed_seconds": time.time() - run_started,
        "total_rollout_latency_seconds": total_rollout_latency_seconds,
        "total_teacher_scoring_latency_seconds": total_teacher_scoring_latency_seconds,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
        ),
        "intended_scientific_run": intended_scientific_run,
        "informative_task_steps": informative_task_steps,
        "informative_task_groups": informative_task_groups,
        "total_task_groups": total_task_groups,
        "informative_group_fraction": informative_group_fraction,
        "minimum_informative_group_fraction": args.min_informative_group_fraction,
        "task_signal_observed": task_signal_observed,
        "finite_nonzero_gradient_observed": finite_nonzero_gradient_observed,
        "parameter_update_observed": parameter_update_observed,
        "parameter_update_l2_by_step": parameter_update_norms,
        "optimizer_state_signature_final": (
            optimizer_state_signatures[-1] if optimizer_state_signatures else None
        ),
        "git_state_start": code_state_start,
        "git_state_training_end": code_state_training_end,
        "git_state_end": None,
        "clean_stable_code": False,
        "stable_training_environment": stable_training_environment,
        "local_server_process_binding_required": server_process_binding_required,
        "live_local_server_process_binding_validated": server_process_binding_validated,
        "local_server_process_binding_end": server_process_binding_end,
        "local_server_process_binding_error": server_process_binding_error,
        "initial_parameter_signature": initial_parameter_signature,
        "final_parameter_signature": final_parameter_signature,
        "training_artifact_eligible_for_held_out_evaluation": False,
        "scientific_use_allowed": False,
        "claim_boundary": (
            "Completion establishes an optimizer run under the applicable contract; "
            "task performance requires held-out evaluation and uncertainty analysis. "
            + (
                "The validated local server binding is same-host Linux process custody, "
                "not cryptographic remote attestation."
                if server_process_binding_validated
                else "No local server process custody is claimed for this run."
            )
        ),
    }
    if intended_scientific_run and not realized_training_geometry_observed:
        completion["status"] = "failed_realized_training_geometry_gate"
        write_completion_manifests(trace_dir, run_manifest, completion)
        raise RuntimeError(
            "realized prompt/sample/trace geometry differs from the predeclared student plan; "
            "no final adapter was promoted"
        )
    if intended_scientific_run and not task_signal_observed:
        write_completion_manifests(trace_dir, run_manifest, completion)
        raise RuntimeError(
            "the predeclared mixed-reward group fraction was not met; traces were preserved "
            "and no final adapter was promoted"
        )
    if intended_scientific_run and not clean_stable_training_code:
        completion["status"] = "failed_code_custody_gate"
        completion["training_artifact_eligible_for_held_out_evaluation"] = False
        write_completion_manifests(trace_dir, run_manifest, completion)
        raise RuntimeError(
            "Git commit or cleanliness changed during training; no final adapter was promoted"
        )
    if intended_scientific_run and not stable_training_environment:
        completion["status"] = "failed_environment_custody_gate"
        completion["training_artifact_eligible_for_held_out_evaluation"] = False
        write_completion_manifests(trace_dir, run_manifest, completion)
        raise RuntimeError(
            "training package identity or an environment freeze changed during training; "
            "no final adapter was promoted"
        )
    if server_process_binding_required and not server_process_binding_validated:
        completion["status"] = "failed_local_server_process_binding_gate"
        completion["training_artifact_eligible_for_held_out_evaluation"] = False
        write_completion_manifests(trace_dir, run_manifest, completion)
        raise RuntimeError(
            "local vLLM process binding was not validated; no final adapter was promoted"
        )
    if require_parameter_update and (
        not finite_nonzero_gradient_observed or not parameter_update_observed
    ):
        completion["status"] = "failed_parameter_update_gate"
        completion["scientific_use_allowed"] = False
        completion["training_artifact_eligible_for_held_out_evaluation"] = False
        write_completion_manifests(trace_dir, run_manifest, completion)
        raise RuntimeError(
            "no finite nonzero gradient/parameter update was observed; no final adapter was promoted"
        )
    candidate_adapter = (Path(args.out_dir) / "final_candidate").resolve()
    save_checkpoint(model, tok, args.out_dir, candidate_adapter.name)
    candidate_hash = sha256_tree(candidate_adapter)
    code_state_after_save = git_state()
    clean_stable_after_save = clean_stable_git_custody(
        code_state_start, code_state_after_save
    )
    stable_environment_after_save = environment_contract_unchanged(
        binding.get("environment_contract")
    )
    completion["git_state_after_candidate_save"] = code_state_after_save
    completion["candidate_adapter"] = str(candidate_adapter)
    completion["candidate_adapter_tree_sha256"] = candidate_hash
    completion["stable_environment_after_candidate_save"] = stable_environment_after_save
    if intended_scientific_run and not clean_stable_after_save:
        completion["status"] = "failed_code_custody_after_candidate_save"
        completion["git_state_end"] = code_state_after_save
        completion["clean_stable_code"] = False
        write_completion_manifests(trace_dir, run_manifest, completion)
        raise RuntimeError(
            "Git commit or cleanliness changed while saving the candidate adapter; "
            "no final adapter was promoted"
        )
    if intended_scientific_run and not stable_environment_after_save:
        completion["status"] = "failed_environment_custody_after_candidate_save"
        completion["git_state_end"] = code_state_after_save
        completion["clean_stable_code"] = clean_stable_after_save
        write_completion_manifests(trace_dir, run_manifest, completion)
        raise RuntimeError(
            "training package identity or an environment freeze changed while saving the "
            "candidate adapter; no final adapter was promoted"
        )

    final_adapter = (Path(args.out_dir) / "final").resolve()
    candidate_adapter.rename(final_adapter)
    try:
        final_hash = sha256_tree(final_adapter)
    except (OSError, ValueError) as exc:
        rejected_adapter = (Path(args.out_dir) / "rejected_final_artifact_custody").resolve()
        final_adapter.rename(rejected_adapter)
        completion["status"] = "failed_final_artifact_rehash"
        completion["rejected_adapter"] = str(rejected_adapter)
        write_completion_manifests(trace_dir, run_manifest, completion)
        raise RuntimeError(
            "final adapter could not be rehashed after promotion; it was moved to a rejected path"
        ) from exc
    code_state_end = git_state()
    clean_stable_code = clean_stable_git_custody(code_state_start, code_state_end)
    stable_environment_end = environment_contract_unchanged(
        binding.get("environment_contract")
    )
    completion["git_state_end"] = code_state_end
    completion["clean_stable_code"] = clean_stable_code
    completion["stable_environment_end"] = stable_environment_end
    stable_final_artifact = final_hash == candidate_hash
    completion["stable_final_artifact_hash"] = stable_final_artifact
    final_custody_failure = final_promotion_custody_failure_status(
        stable_final_artifact=stable_final_artifact,
        intended_scientific_run=intended_scientific_run,
        clean_stable_code=clean_stable_code,
        stable_environment_end=stable_environment_end,
    )
    if final_custody_failure is not None:
        rejected_adapter = (Path(args.out_dir) / "rejected_final_custody").resolve()
        final_adapter.rename(rejected_adapter)
        completion["status"] = final_custody_failure
        completion["training_artifact_eligible_for_held_out_evaluation"] = False
        completion["scientific_use_allowed"] = False
        completion["rejected_adapter"] = str(rejected_adapter)
        write_completion_manifests(trace_dir, run_manifest, completion)
        raise RuntimeError(
            "Git commit or cleanliness changed during final adapter promotion; "
            "or the bound environment changed; the adapter was moved to a rejected diagnostic path"
        )

    completion["training_artifact_eligible_for_held_out_evaluation"] = (
        intended_scientific_run
        and task_signal_observed
        and finite_nonzero_gradient_observed
        and parameter_update_observed
        and clean_stable_code
        and stable_environment_end
        and stable_final_artifact
        and realized_training_geometry_observed
        and _server_process_binding_gate_satisfied(
            server_process_binding_required, server_process_binding_validated
        )
    )
    completion["final_adapter"] = str(final_adapter)
    completion["final_adapter_tree_sha256"] = final_hash
    completion.pop("candidate_adapter", None)
    completion.pop("candidate_adapter_tree_sha256", None)
    write_completion_manifests(trace_dir, run_manifest, completion)


def parse_args():
    ap = argparse.ArgumentParser()
    objective_selector = ap.add_mutually_exclusive_group(required=True)
    objective_selector.add_argument(
        "--mode",
        choices=["task_rl", "task_rl_k1_gap", "k1_bare", "k1_gap_only", "opd", "opd_gated", "kd"],
    )
    objective_selector.add_argument(
        "--objective-id",
        choices=sorted(LOCAL_OBJECTIVE_IDS | (K1_OBJECTIVE_IDS - LOCAL_OBJECTIVE_IDS)),
        help=(
            "bind exact semantics from configs/opd_math/objective_registry.json; "
            "the upstream veRL objective is rejected by this local trainer"
        ),
    )
    ap.add_argument("--task-file", required=True)
    ap.add_argument("--task-limit", type=int, default=0)
    ap.add_argument("--pair-id", choices=["M_M", "M_O", "O_M", "O_O"])
    ap.add_argument("--student-source", choices=["M", "O"])
    ap.add_argument("--budget-mode", choices=["primary_matched", "dose_response"])
    ap.add_argument("--campaign-run-id")
    ap.add_argument("--scheduler-job-id")
    ap.add_argument("--prelaunch-receipt")
    ap.add_argument("--student", required=True)
    ap.add_argument("--student-revision")
    ap.add_argument("--teacher-url")
    ap.add_argument("--teacher-model")
    ap.add_argument("--teacher-checkpoint")
    ap.add_argument("--teacher-server-max-model-len", type=int, default=0)
    ap.add_argument("--teacher-base-model")
    ap.add_argument("--teacher-base-revision")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--trace-dir")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--micro-prompts", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--max-prompt-tokens", type=int, default=1536)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lora", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--kd-temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--advantage-clip", type=float, default=5.0)
    ap.add_argument(
        "--min-informative-group-fraction",
        type=float,
        default=0.05,
        help="scientific runs fail unless at least this fraction of rollout groups has mixed reward",
    )
    ap.add_argument(
        "--gap-gate-beta",
        type=float,
        default=5.0,
        help="positive sigmoid-gap coefficient used only in opd_gated mode",
    )
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="explicitly bind activation checkpointing; scientific plans require it enabled",
    )
    ap.add_argument("--task-reward-coef", type=float, default=1.0)
    ap.add_argument(
        "--k1-coef",
        type=float,
        default=0.01,
        help="starting auxiliary weight only; must be swept rather than treated as established",
    )
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument(
        "--require-parameter-update",
        action="store_true",
        help="fail after preserving diagnostics unless a finite gradient changes trainable parameters",
    )
    ap.add_argument("--teacher-connect-timeout", type=float, default=10.0)
    ap.add_argument("--teacher-read-timeout", type=float, default=120.0)
    ap.add_argument("--teacher-retries", type=int, default=3)
    ap.add_argument("--teacher-gap-manifest")
    ap.add_argument("--teacher-provenance-manifest")
    ap.add_argument("--server-scoring-contract")
    ap.add_argument("--student-support-manifest")
    ap.add_argument("--tokenizer-contract")
    ap.add_argument("--prepared-manifest")
    ap.add_argument("--train-environment-root")
    ap.add_argument("--train-environment-freeze")
    ap.add_argument("--serve-environment-root")
    ap.add_argument("--serve-environment-freeze")
    ap.add_argument(
        "--allow-ungated-smoke",
        action="store_true",
        help="plumbing only: bypass quality/support/tokenizer manifests and mark output non-scientific",
    )
    ap.add_argument("--enable-thinking", action="store_true", help="default is Qwen3 non-thinking mode")
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--attn-implementation", default="sdpa")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
