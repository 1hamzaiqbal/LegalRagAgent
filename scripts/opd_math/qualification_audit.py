#!/usr/bin/env python3
"""Audit OPD-math teacher signal, truncation, and predecessor baselines.

This is a read-only diagnostic.  It never invokes the symbolic verifier and it
never changes a sealed score.  The audit summarizes the measurements already
stored in immutable traces so that training signal and evaluation failures are
not conflated with task performance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise RuntimeError(f"blank JSONL row at {path}:{line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RuntimeError(f"non-object JSONL row at {path}:{line_number}")
            yield row


def fraction(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def source_receipt(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def require_reward(row: Mapping[str, Any]) -> float:
    reward = row.get("reward")
    if not isinstance(reward, (int, float)) or isinstance(reward, bool):
        raise RuntimeError("sample reward must be numeric")
    return float(reward)


def require_tokens(row: Mapping[str, Any]) -> int:
    tokens = row.get("completion_tokens")
    if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens <= 0:
        raise RuntimeError("completion_tokens must be a positive integer")
    return tokens


def analyze_evaluation_samples(path: Path, *, max_tokens: int) -> dict[str, Any]:
    rows = list(iter_jsonl(path))
    if not rows:
        raise RuntimeError(f"evaluation samples are empty: {path}")
    statuses = Counter(str(row.get("reward_status")) for row in rows)
    tokens = [require_tokens(row) for row in rows]
    at_cap = [row for row in rows if require_tokens(row) >= max_tokens]
    below_cap = [row for row in rows if require_tokens(row) < max_tokens]
    parse_status = "prediction_parse_failed"
    parse_at_cap = sum(str(row.get("reward_status")) == parse_status for row in at_cap)
    parse_below_cap = sum(
        str(row.get("reward_status")) == parse_status for row in below_cap
    )
    correct = sum(require_reward(row) > 0.5 for row in rows)
    record_ids = {str(row.get("record_id")) for row in rows}
    return {
        "source": source_receipt(path),
        "max_completion_tokens": max_tokens,
        "records": len(record_ids),
        "samples": len(rows),
        "accuracy": fraction(correct, len(rows)),
        "status_counts": dict(sorted(statuses.items())),
        "mean_completion_tokens": statistics.fmean(tokens),
        "median_completion_tokens": statistics.median(tokens),
        "at_cap_samples": len(at_cap),
        "at_cap_fraction": fraction(len(at_cap), len(rows)),
        "parse_failures_at_cap": parse_at_cap,
        "parse_failures_below_cap": parse_below_cap,
        "parse_failure_at_cap_fraction": fraction(parse_at_cap, len(at_cap)),
        "parse_failure_below_cap_fraction": fraction(parse_below_cap, len(below_cap)),
        "correct_at_cap": sum(require_reward(row) > 0.5 for row in at_cap),
    }


def analyze_paired_evaluations(
    base_path: Path,
    trained_path: Path,
    *,
    max_tokens: int,
) -> dict[str, Any]:
    def keyed(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
        result: dict[tuple[str, int], dict[str, Any]] = {}
        for row in iter_jsonl(path):
            record_id = row.get("record_id")
            sample_idx = row.get("sample_idx")
            if not isinstance(record_id, str) or not isinstance(sample_idx, int):
                raise RuntimeError(f"invalid paired sample identity in {path}")
            key = (record_id, sample_idx)
            if key in result:
                raise RuntimeError(f"duplicate paired sample identity in {path}: {key}")
            result[key] = row
        return result

    base = keyed(base_path)
    trained = keyed(trained_path)
    if base.keys() != trained.keys():
        raise RuntimeError("paired evaluation sample identities differ")
    transitions: Counter[str] = Counter()
    status_transitions: Counter[str] = Counter()
    cap_transitions: Counter[str] = Counter()
    for key in sorted(base):
        left = base[key]
        right = trained[key]
        left_correct = int(require_reward(left) > 0.5)
        right_correct = int(require_reward(right) > 0.5)
        transitions[f"{left_correct}_to_{right_correct}"] += 1
        status_transitions[
            f"{left.get('reward_status')}_to_{right.get('reward_status')}"
        ] += 1
        left_cap = require_tokens(left) >= max_tokens
        right_cap = require_tokens(right) >= max_tokens
        cap_transitions[f"{int(left_cap)}_to_{int(right_cap)}"] += 1
    return {
        "paired_samples": len(base),
        "reward_transitions": dict(sorted(transitions.items())),
        "reward_net_correct": transitions["0_to_1"] - transitions["1_to_0"],
        "status_transitions": dict(sorted(status_transitions.items())),
        "cap_transitions": dict(sorted(cap_transitions.items())),
    }


def analyze_teacher_run(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "run_manifest.json"
    trace_path = run_dir / "teacher_samples.jsonl"
    log_path = run_dir / "trainer_log_history.json"
    manifest = read_json(manifest_path)
    logs = read_json(log_path)
    if not isinstance(manifest, dict) or not isinstance(logs, list):
        raise RuntimeError("invalid teacher manifest or trainer log")
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise RuntimeError("teacher manifest lacks config")
    max_tokens = config.get("max_completion_length")
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        raise RuntimeError("teacher manifest has invalid max_completion_length")

    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in iter_jsonl(trace_path):
        batch = row.get("reward_batch_index")
        if not isinstance(batch, int) or batch < 0:
            raise RuntimeError("teacher trace has invalid reward_batch_index")
        groups[batch].append(row)
    group_rewards = [[require_reward(row) for row in rows] for rows in groups.values()]
    group_caps = [
        [require_tokens(row) >= max_tokens for row in rows] for rows in groups.values()
    ]
    all_zero = sum(all(reward == 0.0 for reward in rewards) for rewards in group_rewards)
    all_correct = sum(all(reward == 1.0 for reward in rewards) for rewards in group_rewards)
    mixed = sum(len(set(rewards)) > 1 for rewards in group_rewards)
    all_at_cap = sum(all(caps) for caps in group_caps)
    all_at_cap_zero = sum(
        all(caps) and all(reward == 0.0 for reward in rewards)
        for caps, rewards in zip(group_caps, group_rewards, strict=True)
    )
    trace_rows = [row for rows in groups.values() for row in rows]
    at_cap = [row for row in trace_rows if require_tokens(row) >= max_tokens]
    nonzero_gradient = sum(float(row.get("grad_norm", 0.0)) > 0.0 for row in logs)
    selected_rows = manifest.get("selected_rows")
    unique_records = manifest.get("realized_training", {}).get("unique_training_records")
    if not isinstance(selected_rows, int) or not isinstance(unique_records, int):
        raise RuntimeError("teacher manifest lacks selected/realized row counts")
    return {
        "sources": {
            "manifest": source_receipt(manifest_path),
            "trace": source_receipt(trace_path),
            "trainer_log": source_receipt(log_path),
        },
        "algorithm_label": manifest.get("algorithm_label"),
        "config": config,
        "declared_role_rows": manifest.get("declared_role_rows"),
        "selected_training_pool_rows": selected_rows,
        "unique_prompt_groups_seen": unique_records,
        "selected_pool_coverage": fraction(unique_records, selected_rows),
        "prompt_groups": len(groups),
        "completion_samples": len(trace_rows),
        "correct_completion_samples": sum(require_reward(row) > 0.5 for row in trace_rows),
        "reward_mean": statistics.fmean(require_reward(row) for row in trace_rows),
        "all_zero_groups": all_zero,
        "all_correct_groups": all_correct,
        "mixed_reward_groups": mixed,
        "mixed_reward_group_fraction": fraction(mixed, len(groups)),
        "nonzero_gradient_steps": nonzero_gradient,
        "nonzero_gradient_step_fraction": fraction(nonzero_gradient, len(logs)),
        "at_cap_samples": len(at_cap),
        "at_cap_fraction": fraction(len(at_cap), len(trace_rows)),
        "correct_at_cap": sum(require_reward(row) > 0.5 for row in at_cap),
        "all_at_cap_groups": all_at_cap,
        "all_at_cap_zero_reward_groups": all_at_cap_zero,
        "mean_completion_tokens": statistics.fmean(
            require_tokens(row) for row in trace_rows
        ),
        "median_completion_tokens": statistics.median(
            require_tokens(row) for row in trace_rows
        ),
        "intermediate_checkpoints": len(list((run_dir / "trainer").glob("checkpoint-*"))),
        "checkpoint_selection_possible": any(
            (run_dir / "trainer").glob("checkpoint-*")
        ),
    }


def analyze_student_training(run_dir: Path) -> dict[str, Any]:
    trace_dir = run_dir / "traces"
    samples_path = trace_dir / "samples.jsonl"
    steps_path = trace_dir / "steps.jsonl"
    manifest_path = trace_dir / "run_manifest.json"
    manifest = read_json(manifest_path)
    samples = list(iter_jsonl(samples_path))
    steps = list(iter_jsonl(steps_path))
    generation = manifest.get("generation", {})
    max_tokens = generation.get("max_new_tokens")
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        raise RuntimeError("student run manifest lacks max_new_tokens")
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in samples:
        step = row.get("step")
        if not isinstance(step, int):
            raise RuntimeError("student training sample lacks step")
        groups[step].append(row)
    mixed = sum(
        len({require_reward(row) for row in rows}) > 1 for rows in groups.values()
    )
    parse_failed = sum(
        row.get("reward_status") == "prediction_parse_failed" for row in samples
    )
    at_cap = [row for row in samples if require_tokens(row) >= max_tokens]
    return {
        "sources": {
            "manifest": source_receipt(manifest_path),
            "samples": source_receipt(samples_path),
            "steps": source_receipt(steps_path),
        },
        "objective": manifest.get("objective"),
        "optimizer_steps": len(steps),
        "completion_samples": len(samples),
        "reward_mean": statistics.fmean(require_reward(row) for row in samples),
        "mixed_reward_groups": mixed,
        "mixed_reward_group_fraction": fraction(mixed, len(groups)),
        "parse_failure_samples": parse_failed,
        "parse_failure_fraction": fraction(parse_failed, len(samples)),
        "at_cap_samples": len(at_cap),
        "at_cap_fraction": fraction(len(at_cap), len(samples)),
        "parse_failures_at_cap": sum(
            row.get("reward_status") == "prediction_parse_failed" for row in at_cap
        ),
        "terminated_by_eos_fraction": fraction(
            sum(row.get("terminated_by_eos") is True for row in samples), len(samples)
        ),
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    teacher = payload["teacher_training"]
    paired = payload["teacher_heldout"]["paired"]
    lines = [
        "# OPD math qualification audit",
        "",
        "> Read-only reconstruction from sealed traces. No score was recomputed or changed.",
        "",
        "## Teacher training signal",
        "",
        "| Measure | Value |",
        "|---|---:|",
        f"| Prompt groups seen | {teacher['unique_prompt_groups_seen']:,} |",
        f"| Selected-pool coverage | {teacher['selected_pool_coverage']:.2%} |",
        f"| Mixed-reward groups | {teacher['mixed_reward_groups']:,} / {teacher['prompt_groups']:,} ({teacher['mixed_reward_group_fraction']:.2%}) |",
        f"| Nonzero-gradient steps | {teacher['nonzero_gradient_steps']:,} / {teacher['prompt_groups']:,} ({teacher['nonzero_gradient_step_fraction']:.2%}) |",
        f"| Completions at cap | {teacher['at_cap_samples']:,} / {teacher['completion_samples']:,} ({teacher['at_cap_fraction']:.2%}) |",
        f"| Correct capped completions | {teacher['correct_at_cap']:,} |",
        f"| All-capped, zero-reward groups | {teacher['all_at_cap_zero_reward_groups']:,} |",
        "",
        "## Paired held-out teacher movement",
        "",
        f"- Paired samples: {paired['paired_samples']:,}",
        f"- Incorrect-to-correct: {paired['reward_transitions'].get('0_to_1', 0):,}",
        f"- Correct-to-incorrect: {paired['reward_transitions'].get('1_to_0', 0):,}",
        f"- Net additional correct samples: {paired['reward_net_correct']:,}",
        "",
        "## Truncation diagnostics",
        "",
        "| Surface | Accuracy | At cap | Parse failures at cap | Parse failures below cap |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in payload["evaluation_surfaces"].items():
        lines.append(
            f"| {name} | {row['accuracy']:.2%} | {row['at_cap_fraction']:.2%} | "
            f"{row['parse_failures_at_cap']:,} | {row['parse_failures_below_cap']:,} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- Capped-output failures are generation-budget failures, not evidence that the parser should rescue an incomplete trajectory.",
            "- The teacher's held-out gain is retained, but this run is a low-signal pilot rather than evidence of near-optimal teacher training.",
            "- Absolute task-RL held-out scores do not establish improvement unless the raw student is evaluated on the identical held-out records and decoding contract.",
            "- No predecessor artifact demonstrates an OPD student improvement.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=/absolute/path")
    name, raw_path = value.split("=", 1)
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("expected NAME=/absolute/path")
    return name, Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-run-dir", type=Path, required=True)
    parser.add_argument("--teacher-base-samples", type=Path, required=True)
    parser.add_argument("--teacher-trained-samples", type=Path, required=True)
    parser.add_argument("--teacher-max-tokens", type=int, required=True)
    parser.add_argument(
        "--evaluation-surface",
        action="append",
        type=parse_named_path,
        default=[],
        metavar="NAME=PATH",
    )
    parser.add_argument(
        "--evaluation-max-tokens",
        action="append",
        type=int,
        default=[],
        metavar="TOKENS",
    )
    parser.add_argument(
        "--student-training-run",
        action="append",
        type=parse_named_path,
        default=[],
        metavar="NAME=PATH",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()
    if len(args.evaluation_surface) != len(args.evaluation_max_tokens):
        parser.error("each --evaluation-surface requires one --evaluation-max-tokens")
    if args.output_json.exists() or args.output_markdown.exists():
        parser.error("outputs must be fresh paths")

    surfaces = {
        name: analyze_evaluation_samples(path, max_tokens=max_tokens)
        for (name, path), max_tokens in zip(
            args.evaluation_surface, args.evaluation_max_tokens, strict=True
        )
    }
    payload = {
        "schema_version": 1,
        "artifact_type": "opd_math_teacher_evaluator_qualification_audit",
        "claim_boundary": (
            "Read-only diagnostics over sealed traces; no verifier replay, score change, "
            "teacher sufficiency claim, or OPD effectiveness claim."
        ),
        "teacher_training": analyze_teacher_run(args.teacher_run_dir),
        "teacher_heldout": {
            "base": analyze_evaluation_samples(
                args.teacher_base_samples, max_tokens=args.teacher_max_tokens
            ),
            "trained": analyze_evaluation_samples(
                args.teacher_trained_samples, max_tokens=args.teacher_max_tokens
            ),
            "paired": analyze_paired_evaluations(
                args.teacher_base_samples,
                args.teacher_trained_samples,
                max_tokens=args.teacher_max_tokens,
            ),
        },
        "evaluation_surfaces": surfaces,
        "student_training_pilots": {
            name: analyze_student_training(path)
            for name, path in args.student_training_run
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output_markdown.write_text(render_markdown(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
