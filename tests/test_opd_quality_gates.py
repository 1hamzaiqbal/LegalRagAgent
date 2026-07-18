from argparse import Namespace
import json
from pathlib import Path

import pytest

from scripts.opd_math.quality_gates import (
    CANONICAL_TEACHER_TRAINING_PLAN,
    EXPECTED_TEACHER_TRAIN_PACKAGES,
    STUDENT_GATE_TYPE,
    STUDENT_SMOKE_GATE_TYPE,
    TEACHER_GATE_TYPE,
    TEACHER_SMOKE_GATE_TYPE,
    canonical_json_sha256,
    recompute_student_gate,
    sha256_file,
    sha256_tree,
    student_support,
    teacher_gap,
)


MODEL = "Qwen/Qwen3-8B"
REVISION = "a" * 40
STUDENT_MODEL = "Qwen/Qwen3-1.7B"
STUDENT_REVISION = "b" * 40
DECODING = {
    "thinking": False,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "max_new_tokens": 128,
    "seed": 0,
}
EVALUATOR = Path(__file__).resolve().parents[1] / "scripts" / "opd_math" / "evaluate_math.py"


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def write_task(path, count, *, source="M", role="teacher_gap_dev"):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "record_id": f"{source}:{index}",
            "cluster_id": f"cluster:{source}:{index}",
            "source": source,
            "role": role,
            "solution": rf"\boxed{{{index + 1}}}",
        }
        for index in range(count)
    ]
    write_jsonl(path, rows)
    return rows


def write_prepared(tmp_path, *, teacher_gap_count=200, student_count=100, source="M"):
    source_manifest = tmp_path / "source_manifest.json"
    source_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "models": {
                    "teacher": {"id": MODEL, "revision": REVISION},
                    "student": {"id": STUDENT_MODEL, "revision": STUDENT_REVISION},
                },
            },
            sort_keys=True,
        )
        + "\n"
    )
    role_paths = {
        "teacher_train": tmp_path / "roles" / source / "teacher_train.jsonl",
        "teacher_gap_dev": tmp_path / "roles" / source / "teacher_gap_dev.jsonl",
        "student_opd": tmp_path / "roles" / source / "student_opd.jsonl",
    }
    role_rows = {
        "teacher_train": write_task(
            role_paths["teacher_train"], 2, source=source, role="teacher_train"
        ),
        "teacher_gap_dev": write_task(
            role_paths["teacher_gap_dev"],
            teacher_gap_count,
            source=source,
            role="teacher_gap_dev",
        ),
        "student_opd": write_task(
            role_paths["student_opd"], student_count, source=source, role="student_opd"
        ),
    }
    files = {
        f"roles/{source}/{role}.jsonl": {
            "rows": len(role_rows[role]),
            "sha256": sha256_file(path),
        }
        for role, path in role_paths.items()
    }
    prepared = {
        "schema_version": 1,
        "scientific_use_allowed": True,
        "code_git_state": {"commit": "f" * 40, "dirty": False},
        "source_manifest_path": str(source_manifest.resolve()),
        "source_manifest_sha256": sha256_file(source_manifest),
        "primary_matched_budgets": {
            "teacher_train": 2,
            "teacher_gap_dev": teacher_gap_count,
            "student_opd": student_count,
        },
        "files": files,
    }
    manifest = tmp_path / "prepared_manifest.json"
    manifest.write_text(json.dumps(prepared, indent=2, sort_keys=True) + "\n")
    return manifest, prepared, role_paths, role_rows


def write_teacher_run(tmp_path, prepared_manifest, adapter, *, source="M", scientific=True):
    prepared = json.loads(prepared_manifest.read_text())
    source_manifest = prepared["source_manifest_path"]
    training_path = prepared_manifest.parent / "roles" / source / "teacher_train.jsonl"
    state = {"commit": "c" * 40, "dirty": False}
    training_plan = json.loads(CANONICAL_TEACHER_TRAINING_PLAN.read_text())
    fixed_config = training_plan["fixed_config"]
    trainer_state = tmp_path / "trainer_state.json"
    trainer_state.write_text(
        json.dumps({"global_step": fixed_config["max_steps"]}, sort_keys=True) + "\n"
    )
    trainer_log = tmp_path / "trainer_log_history.json"
    trainer_log.write_text(
        json.dumps([{"step": fixed_config["max_steps"], "loss": 0.0}], sort_keys=True)
        + "\n"
    )
    train_metrics = tmp_path / "train_metrics.json"
    train_metrics.write_text(
        json.dumps(
            {
                "actual_optimizer_steps": fixed_config["max_steps"],
                "optimizer_progress_complete": True,
            },
            sort_keys=True,
        )
        + "\n"
    )
    run = {
        "schema_version": 1,
        "stage": "teacher_grpo",
        "status": "completed",
        "scientific_use_allowed": scientific,
        "model": MODEL,
        "model_revision": REVISION,
        "source": source,
        "role": "teacher_train",
        "budget_mode": "primary_matched",
        "task_file": str(training_path.resolve()),
        "task_file_sha256": sha256_file(training_path),
        "prepared_manifest": str(prepared_manifest.resolve()),
        "prepared_manifest_sha256": sha256_file(prepared_manifest),
        "source_manifest": source_manifest,
        "source_manifest_sha256": sha256_file(source_manifest),
        "training_plan": str(CANONICAL_TEACHER_TRAINING_PLAN.resolve()),
        "training_plan_sha256": sha256_file(CANONICAL_TEACHER_TRAINING_PLAN),
        "training_plan_id": training_plan["plan_id"],
        "training_plan_compliant": True,
        "training_plan_config_sha256": canonical_json_sha256(fixed_config),
        "teacher_training_config_sha256": canonical_json_sha256(fixed_config),
        "config": fixed_config,
        "packages": EXPECTED_TEACHER_TRAIN_PACKAGES,
        "pinned_teacher_model": MODEL,
        "pinned_teacher_revision": REVISION,
        "intended_scientific_run": scientific,
        "selected_rows": prepared["primary_matched_budgets"]["teacher_train"],
        "actual_optimizer_steps": fixed_config["max_steps"],
        "trainer_log_max_step": fixed_config["max_steps"],
        "optimizer_progress_complete": True,
        "trainer_state": str(trainer_state.resolve()),
        "trainer_state_sha256": sha256_file(trainer_state),
        "trainer_log_history": str(trainer_log.resolve()),
        "trainer_log_history_sha256": sha256_file(trainer_log),
        "train_metrics": str(train_metrics.resolve()),
        "train_metrics_sha256": sha256_file(train_metrics),
        "prompt_token_diagnostics": {
            "selected_prompts": prepared["primary_matched_budgets"]["teacher_train"],
            "max_prompt_tokens_allowed": fixed_config["max_prompt_tokens"],
            "min_rendered_prompt_tokens": 8,
            "max_rendered_prompt_tokens": 16,
            "mean_rendered_prompt_tokens": 12.0,
            "implicit_truncation_allowed": False,
        },
        "final_adapter": str(adapter.resolve()),
        "final_adapter_tree_sha256": sha256_tree(adapter),
        "git_state_start": state,
        "git_state_end": state,
        "clean_stable_code": True,
    }
    path = tmp_path / "teacher-run.json"
    path.write_text(json.dumps(run, indent=2, sort_keys=True) + "\n")
    return path


def write_evaluation(
    directory,
    name,
    task_path,
    rewards_by_record,
    *,
    model=MODEL,
    revision=REVISION,
    adapter=None,
    source="M",
    role="teacher_gap_dev",
):
    directory.mkdir(parents=True, exist_ok=True)
    samples_path = directory / f"{name}-samples.jsonl"
    summary_path = directory / f"{name}-summary.json"
    task_by_record = {
        row["record_id"]: row
        for row in (json.loads(line) for line in task_path.read_text().splitlines())
    }
    sample_rows = []
    for record_id, rewards in rewards_by_record.items():
        for sample_idx, reward in enumerate(rewards):
            completion = (
                f"Final answer: {task_by_record[record_id]['solution']}."
                if reward
                else r"Final answer: \boxed{-999999}."
            )
            sample_rows.append(
                {
                    "record_id": record_id,
                    "source": source,
                    "sample_idx": sample_idx,
                    "reward": reward,
                    "reward_status": "correct" if reward else "incorrect",
                    "completion_text": completion,
                }
            )
    write_jsonl(samples_path, sample_rows)
    total = len(sample_rows)
    samples_per_problem = len(next(iter(rewards_by_record.values())))
    summary = {
        "schema_version": 1,
        "model": model,
        "model_revision": revision,
        "code": {
            "git": {"commit": "e" * 40, "worktree_clean": True},
            "evaluator_file_sha256": sha256_file(EVALUATOR),
            "packages": {
                "torch": "2.11.0",
                "transformers": "4.57.6",
                "peft": "0.19.1",
                "math-verify": "0.9.0",
            },
        },
        "tokenizer_contract_sha256": "d" * 64,
        "adapter": None if adapter is None else str(adapter.resolve()),
        "adapter_tree_sha256": None if adapter is None else sha256_tree(adapter),
        "task_file": str(task_path.resolve()),
        "task_file_sha256": sha256_file(task_path),
        "records": len(rewards_by_record),
        "task_sources": [source],
        "task_roles": [role],
        "samples_per_problem": samples_per_problem,
        "samples": total,
        "accuracy": sum(row["reward"] for row in sample_rows) / total,
        "prediction_parse_failure_fraction": 0.0,
        "decoding": DECODING,
        "samples_file": str(samples_path.resolve()),
        "samples_file_sha256": sha256_file(samples_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary_path, samples_path


def teacher_args(
    *,
    base_summary,
    base_samples,
    trained_summary,
    trained_samples,
    adapter,
    prepared_manifest,
    teacher_run_manifest,
    smoke=False,
):
    return Namespace(
        base_summary=base_summary,
        base_samples=base_samples,
        trained_summary=trained_summary,
        trained_samples=trained_samples,
        base_model=MODEL,
        base_revision=REVISION,
        trained_adapter=adapter,
        prepared_manifest=prepared_manifest,
        teacher_run_manifest=teacher_run_manifest,
        task_source="M",
        task_role="teacher_gap_dev",
        min_delta=0.0,
        min_records=None,
        bootstrap_draws=1_000,
        seed=0,
        smoke_gate=smoke,
    )


def test_scientific_teacher_gate_requires_records_strict_gain_and_positive_ci(tmp_path):
    prepared_manifest, _, role_paths, role_rows = write_prepared(tmp_path)
    task = role_paths["teacher_gap_dev"]
    task_rows = role_rows["teacher_gap_dev"]
    adapter = tmp_path / "teacher-adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"r": 8}\n')
    teacher_run_manifest = write_teacher_run(tmp_path, prepared_manifest, adapter)
    base_rewards = {row["record_id"]: [0] for row in task_rows}
    tied_rewards = {row["record_id"]: [0] for row in task_rows}
    better_rewards = {row["record_id"]: [1] for row in task_rows}
    base_summary, base_samples = write_evaluation(tmp_path, "base", task, base_rewards)
    tied_summary, tied_samples = write_evaluation(
        tmp_path, "tied", task, tied_rewards, adapter=adapter
    )
    better_summary, better_samples = write_evaluation(
        tmp_path, "better", task, better_rewards, adapter=adapter
    )

    tied = teacher_gap(
        teacher_args(
            base_summary=base_summary,
            base_samples=base_samples,
            trained_summary=tied_summary,
            trained_samples=tied_samples,
            adapter=adapter,
            prepared_manifest=prepared_manifest,
            teacher_run_manifest=teacher_run_manifest,
        )
    )
    assert not tied["passed"]
    assert not tied["requirements"]["strict_delta_met"]

    better = teacher_gap(
        teacher_args(
            base_summary=base_summary,
            base_samples=base_samples,
            trained_summary=better_summary,
            trained_samples=better_samples,
            adapter=adapter,
            prepared_manifest=prepared_manifest,
            teacher_run_manifest=teacher_run_manifest,
        )
    )
    assert better["gate"] == TEACHER_GATE_TYPE
    assert better["gate_strength"] == "scientific"
    assert better["passed"]
    assert better["authorizes_scientific_merge"]
    assert better["shared_records"] == 200
    assert better["bootstrap_95_ci"] == [1.0, 1.0]
    assert len(better["trained_adapter_tree_sha256"]) == 64
    assert better["registered_task_rows"] == 200
    assert better["teacher_training_budget_mode"] == "primary_matched"
    assert better["teacher_training_actual_optimizer_steps"] == 100
    assert (
        better["teacher_training_plan_config_sha256"]
        == better["teacher_training_config_sha256"]
    )

    mismatched_run = json.loads(teacher_run_manifest.read_text())
    mismatched_run["packages"]["trl"] = "0.0.0"
    teacher_run_manifest.write_text(json.dumps(mismatched_run, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="training packages differ"):
        teacher_gap(
            teacher_args(
                base_summary=base_summary,
                base_samples=base_samples,
                trained_summary=better_summary,
                trained_samples=better_samples,
                adapter=adapter,
                prepared_manifest=prepared_manifest,
                teacher_run_manifest=teacher_run_manifest,
            )
        )

    mismatched_run["packages"] = EXPECTED_TEACHER_TRAIN_PACKAGES
    mismatched_run["config"]["max_steps"] = 99
    mismatched_run["teacher_training_config_sha256"] = canonical_json_sha256(
        mismatched_run["config"]
    )
    mismatched_run["actual_optimizer_steps"] = 99
    mismatched_run["training_plan_compliant"] = False
    teacher_run_manifest.write_text(json.dumps(mismatched_run, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="source-independent training plan"):
        teacher_gap(
            teacher_args(
                base_summary=base_summary,
                base_samples=base_samples,
                trained_summary=better_summary,
                trained_samples=better_samples,
                adapter=adapter,
                prepared_manifest=prepared_manifest,
                teacher_run_manifest=teacher_run_manifest,
            )
        )


def test_small_smoke_teacher_gate_has_distinct_non_authorizing_type(tmp_path):
    prepared_manifest, _, role_paths, role_rows = write_prepared(
        tmp_path, teacher_gap_count=2, student_count=2
    )
    task = role_paths["teacher_gap_dev"]
    task_rows = role_rows["teacher_gap_dev"]
    adapter = tmp_path / "smoke-adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}\n")
    teacher_run_manifest = write_teacher_run(
        tmp_path, prepared_manifest, adapter, scientific=False
    )
    base_summary, base_samples = write_evaluation(
        tmp_path, "smoke-base", task, {row["record_id"]: [0] for row in task_rows}
    )
    trained_summary, trained_samples = write_evaluation(
        tmp_path,
        "smoke-trained",
        task,
        {task_rows[0]["record_id"]: [1], task_rows[1]["record_id"]: [0]},
        adapter=adapter,
    )
    args = teacher_args(
        base_summary=base_summary,
        base_samples=base_samples,
        trained_summary=trained_summary,
        trained_samples=trained_samples,
        adapter=adapter,
        prepared_manifest=prepared_manifest,
        teacher_run_manifest=teacher_run_manifest,
        smoke=True,
    )
    result = teacher_gap(args)
    assert result["gate"] == TEACHER_SMOKE_GATE_TYPE
    assert result["passed"]
    assert not result["authorizes_scientific_merge"]
    assert result["bootstrap_95_ci"][0] == 0.0

    args.smoke_gate = False
    with pytest.raises(ValueError, match="eligible teacher run"):
        teacher_gap(args)


def test_scientific_student_support_binds_model_task_and_support(tmp_path):
    prepared_manifest, _, role_paths, role_rows = write_prepared(tmp_path)
    task = role_paths["student_opd"]
    task_rows = role_rows["student_opd"]
    rewards = {
        row["record_id"]: ([0, 1] if index < 50 else [0, 0])
        for index, row in enumerate(task_rows)
    }
    summary, samples = write_evaluation(
        tmp_path,
        "student",
        task,
        rewards,
        model=STUDENT_MODEL,
        revision=STUDENT_REVISION,
        role="student_opd",
    )
    result = student_support(
        Namespace(
            student_summary=summary,
            student_samples=samples,
            student_model=STUDENT_MODEL,
            student_revision=STUDENT_REVISION,
            prepared_manifest=prepared_manifest,
            task_source="M",
            task_role="student_opd",
            min_pass_at_k=0.1,
            min_mixed_group_fraction=0.1,
            min_records=None,
            smoke_gate=False,
        )
    )
    assert result["gate"] == STUDENT_GATE_TYPE
    assert result["passed"]
    assert result["authorizes_scientific_training"]
    assert result["records"] == 100
    assert result["pass_at_k"] == 0.5
    assert result["mixed_reward_group_fraction"] == 0.5
    assert result["task_roles"] == ["student_opd"]
    assert recompute_student_gate(result) == result


def test_small_student_support_is_smoke_only(tmp_path):
    prepared_manifest, _, role_paths, role_rows = write_prepared(
        tmp_path, teacher_gap_count=2, student_count=2
    )
    task = role_paths["student_opd"]
    task_rows = role_rows["student_opd"]
    rewards = {
        task_rows[0]["record_id"]: [0, 1],
        task_rows[1]["record_id"]: [0, 0],
    }
    summary, samples = write_evaluation(
        tmp_path,
        "student-smoke",
        task,
        rewards,
        model=STUDENT_MODEL,
        revision=STUDENT_REVISION,
        role="student_opd",
    )
    args = Namespace(
        student_summary=summary,
        student_samples=samples,
        student_model=STUDENT_MODEL,
        student_revision=STUDENT_REVISION,
        prepared_manifest=prepared_manifest,
        task_source="M",
        task_role="student_opd",
        min_pass_at_k=0.1,
        min_mixed_group_fraction=0.1,
        min_records=None,
        smoke_gate=True,
    )
    result = student_support(args)
    assert result["gate"] == STUDENT_SMOKE_GATE_TYPE
    assert result["passed"]
    assert not result["authorizes_scientific_training"]

    args.smoke_gate = False
    result = student_support(args)
    assert result["gate"] == STUDENT_GATE_TYPE
    assert not result["passed"]
    assert not result["requirements"]["minimum_records_met"]


def test_teacher_gate_rejects_adapter_changed_after_evaluation(tmp_path):
    prepared_manifest, _, role_paths, role_rows = write_prepared(
        tmp_path, teacher_gap_count=2, student_count=2
    )
    task = role_paths["teacher_gap_dev"]
    task_rows = role_rows["teacher_gap_dev"]
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"r": 8}\n')
    base_summary, base_samples = write_evaluation(
        tmp_path, "mutation-base", task, {row["record_id"]: [0] for row in task_rows}
    )
    trained_summary, trained_samples = write_evaluation(
        tmp_path,
        "mutation-trained",
        task,
        {row["record_id"]: [1] for row in task_rows},
        adapter=adapter,
    )
    teacher_run_manifest = write_teacher_run(
        tmp_path, prepared_manifest, adapter, scientific=False
    )
    (adapter / "adapter_config.json").write_text('{"r": 16}\n')
    with pytest.raises(ValueError, match="differs from the identity recorded"):
        teacher_gap(
            teacher_args(
                base_summary=base_summary,
                base_samples=base_samples,
                trained_summary=trained_summary,
                trained_samples=trained_samples,
                adapter=adapter,
                prepared_manifest=prepared_manifest,
                teacher_run_manifest=teacher_run_manifest,
                smoke=True,
            )
        )


def test_scientific_teacher_gate_rejects_prefix_or_ineligible_run(tmp_path):
    prepared_manifest, _, role_paths, role_rows = write_prepared(
        tmp_path, teacher_gap_count=201
    )
    task = role_paths["teacher_gap_dev"]
    selected = role_rows["teacher_gap_dev"][:200]
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}\n")
    run_manifest = write_teacher_run(tmp_path, prepared_manifest, adapter)
    base_summary, base_samples = write_evaluation(
        tmp_path, "prefix-base", task, {row["record_id"]: [0] for row in selected}
    )
    trained_summary, trained_samples = write_evaluation(
        tmp_path,
        "prefix-trained",
        task,
        {row["record_id"]: [1] for row in selected},
        adapter=adapter,
    )
    gate_args = teacher_args(
        base_summary=base_summary,
        base_samples=base_samples,
        trained_summary=trained_summary,
        trained_samples=trained_samples,
        adapter=adapter,
        prepared_manifest=prepared_manifest,
        teacher_run_manifest=run_manifest,
    )
    with pytest.raises(ValueError, match="must use exactly 201"):
        teacher_gap(gate_args)

    # Restore a full evaluation, then prove that a self-labeled dose-response
    # run cannot be promoted through the scientific teacher gate.
    all_rows = role_rows["teacher_gap_dev"]
    base_summary, base_samples = write_evaluation(
        tmp_path, "full-base", task, {row["record_id"]: [0] for row in all_rows}
    )
    trained_summary, trained_samples = write_evaluation(
        tmp_path,
        "full-trained",
        task,
        {row["record_id"]: [1] for row in all_rows},
        adapter=adapter,
    )
    run = json.loads(run_manifest.read_text())
    run["budget_mode"] = "dose_response"
    run_manifest.write_text(json.dumps(run, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="primary_matched"):
        teacher_gap(
            teacher_args(
                base_summary=base_summary,
                base_samples=base_samples,
                trained_summary=trained_summary,
                trained_samples=trained_samples,
                adapter=adapter,
                prepared_manifest=prepared_manifest,
                teacher_run_manifest=run_manifest,
            )
        )


def test_scientific_student_support_rejects_cherry_picked_prefix(tmp_path):
    prepared_manifest, _, role_paths, role_rows = write_prepared(tmp_path, student_count=100)
    task = role_paths["student_opd"]
    selected = role_rows["student_opd"][:99]
    rewards = {row["record_id"]: [0, 1] for row in selected}
    summary, samples = write_evaluation(
        tmp_path,
        "student-prefix",
        task,
        rewards,
        model=STUDENT_MODEL,
        revision=STUDENT_REVISION,
        role="student_opd",
    )
    with pytest.raises(ValueError, match="must use exactly 100"):
        student_support(
            Namespace(
                student_summary=summary,
                student_samples=samples,
                student_model=STUDENT_MODEL,
                student_revision=STUDENT_REVISION,
                prepared_manifest=prepared_manifest,
                task_source="M",
                task_role="student_opd",
                min_pass_at_k=0.1,
                min_mixed_group_fraction=0.1,
                min_records=None,
                smoke_gate=False,
            )
        )
