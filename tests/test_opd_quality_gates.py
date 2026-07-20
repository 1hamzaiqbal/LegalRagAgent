from argparse import Namespace
import hashlib
import json
from pathlib import Path
import sys

import pytest

from tests.opd_evaluation_fixture import write_merged_evaluation
from scripts.opd_math import quality_gates
from scripts.opd_math import plan_evaluation_shards as evaluation_planner

from scripts.opd_math.quality_gates import (
    CANONICAL_TEACHER_TRAINING_PLAN,
    EXPECTED_TEACHER_TRAIN_PACKAGES,
    STUDENT_GATE_TYPE,
    STUDENT_SMOKE_GATE_TYPE,
    STUDENT_SUPPORT_DECODING,
    TEACHER_GATE_TYPE,
    TEACHER_GAP_DECODING,
    TEACHER_SMOKE_GATE_TYPE,
    TEACHER_TARGET_REPORT_RECORDS,
    TEACHER_TARGET_REPORT_TYPE,
    canonical_json_sha256,
    recompute_student_gate,
    recompute_teacher_target_report,
    sha256_file,
    sha256_tree,
    student_support,
    teacher_gap,
    teacher_target_report,
)


MODEL = "Qwen/Qwen3-8B"
REVISION = "a" * 40
STUDENT_MODEL = "Qwen/Qwen3-1.7B"
STUDENT_REVISION = "b" * 40
DECODING = TEACHER_GAP_DECODING
EVALUATOR = Path(__file__).resolve().parents[1] / "scripts" / "opd_math" / "evaluate_math.py"
TEACHER_TRAINING_COMMIT = "c" * 40


@pytest.fixture(autouse=True)
def stub_live_teacher_environment_reverification(monkeypatch):
    monkeypatch.setattr(
        quality_gates,
        "reverify_recorded_environment",
        lambda recorded, *, in_process: dict(recorded),
    )


def teacher_environment_contract(tmp_path, commit):
    freeze = tmp_path / "environment_freezes" / commit / "train.freeze.txt"
    freeze.parent.mkdir(parents=True, exist_ok=True)
    freeze.write_text("exact-test-freeze\n")
    freeze_hash = sha256_file(freeze)
    verification = {
        "schema_version": 1,
        "schema": "opd_math_environment_verification_v1",
        "status": "passed",
        "environment_root": str((tmp_path / "train-env").resolve()),
        "live_python": str((tmp_path / "train-env" / "bin" / "python").resolve()),
        "expected_commit": commit,
        "freeze_kind": "train",
        "installed_distribution_count": 1,
        "installed_distribution_map_sha256": "d" * 64,
        "requirements_freeze": {"path": str(freeze), "sha256": freeze_hash},
        "commit_freeze": {
            "path": str(freeze.resolve()),
            "sha256": freeze_hash,
            "byte_identical_to_requirements_freeze": True,
        },
        "expected_executable": None,
    }
    return {
        "schema_version": 2,
        "git_commit": commit,
        "verifier": {
            "path": str(quality_gates.ENVIRONMENT_VERIFIER.resolve()),
            "sha256": sha256_file(quality_gates.ENVIRONMENT_VERIFIER),
        },
        "train_runtime_packages": EXPECTED_TEACHER_TRAIN_PACKAGES,
        "train_freeze": {
            "path": str(freeze.resolve()),
            "sha256": freeze_hash,
            "required_packages": EXPECTED_TEACHER_TRAIN_PACKAGES,
        },
        "train_verification": verification,
        "serve_freeze": None,
        "serve_verification": None,
    }


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def test_manifest_publication_is_atomic_and_never_overwrites(tmp_path):
    output = tmp_path / "gates" / "gate.json"
    written = quality_gates.write_text_exclusive_fsync(
        output, '{"passed":true}\n', label="test gate"
    )
    assert written == output.resolve()
    assert output.read_text() == '{"passed":true}\n'
    assert not list(output.parent.glob(".gate.json.partial.*"))
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        quality_gates.write_text_exclusive_fsync(
            output, '{"passed":false}\n', label="test gate"
        )
    assert output.read_text() == '{"passed":true}\n'


def test_verifier_error_assignment_supports_monotone_gate_sensitivity():
    grouped = {"a": [0.0, 1.0], "b": [0.0, 0.0]}
    binding = {
        "verifier_error_sample_keys": [
            {"record_id": "a", "sample_idx": 0},
            {"record_id": "b", "sample_idx": 1},
        ]
    }

    pessimistic_base = quality_gates.assign_verifier_error_rewards(
        grouped, binding, 1.0
    )
    pessimistic_trained = quality_gates.assign_verifier_error_rewards(
        grouped, binding, 0.0
    )

    assert pessimistic_base == {"a": [1.0, 1.0], "b": [0.0, 1.0]}
    assert pessimistic_trained == grouped


def test_full_o_teacher_gate_requires_same_revalidated_base_trained_plan(
    tmp_path, monkeypatch
):
    task = tmp_path / "O_teacher_gap_dev.jsonl"
    write_jsonl(
        task,
        [
            {
                "record_id": f"O:{index}",
                "source": "O",
                "role": "teacher_gap_dev",
                "solution": rf"\boxed{{{index}}}",
            }
            for index in range(3)
        ],
    )
    monkeypatch.setattr(
        evaluation_planner, "revalidate_plan_binding", lambda binding: dict(binding)
    )
    shared = {
        "schema_version": 1,
        "binding_kind": "opd_math_o_primary_evaluation_plan_binding_v1",
        "plan": "/plans/O.json",
        "plan_file_sha256": "1" * 64,
        "plan_payload_sha256": "2" * 64,
        "plan_schema_version": 2,
        "plan_kind": evaluation_planner.PLAN_KIND,
        "source": "O",
        "role": "teacher_gap_dev",
        "model": MODEL,
        "model_revision": REVISION,
        "task_file": str(task.resolve()),
        "task_file_sha256": sha256_file(task),
        "max_records": 0,
        "git_commit": TEACHER_TRAINING_COMMIT,
        "train_freeze": "/freezes/train.freeze.txt",
        "train_freeze_sha256": "3" * 64,
        "array_spec": "0-4%4",
        "slurm_array_argument": "--array=0-4%4",
        "array_geometry_sha256": "4" * 64,
        "shard_count": 5,
        "samples_per_problem": 4,
        "decoding": dict(TEACHER_GAP_DECODING),
    }
    base = {**shared, "arm": "base", "adapter": None, "adapter_tree_sha256": None}
    trained = {
        **shared,
        "arm": "trained",
        "adapter": "/adapters/O",
        "adapter_tree_sha256": "5" * 64,
    }
    base_binding = {
        "task_file": str(task.resolve()),
        "records": 3,
        "evaluation_plan": base,
    }
    trained_binding = {
        "task_file": str(task.resolve()),
        "records": 3,
        "evaluation_plan": trained,
    }
    result = quality_gates._require_o_teacher_gap_plan_pair(
        task_source="O",
        base_binding=base_binding,
        trained_binding=trained_binding,
    )
    assert result["plan_payload_sha256"] == "2" * 64
    assert result["base_binding"]["arm"] == "base"
    assert result["trained_binding"]["arm"] == "trained"

    with pytest.raises(ValueError, match="one exact plan"):
        quality_gates._require_o_teacher_gap_plan_pair(
            task_source="O",
            base_binding=base_binding,
            trained_binding={
                **trained_binding,
                "evaluation_plan": {
                    **trained,
                    "plan_payload_sha256": "9" * 64,
                },
            },
        )
    with pytest.raises(ValueError, match="arms must be base then trained"):
        quality_gates._require_o_teacher_gap_plan_pair(
            task_source="O",
            base_binding={
                **base_binding,
                "evaluation_plan": {**base, "arm": "trained"},
            },
            trained_binding=trained_binding,
        )


def test_scientific_student_support_removes_error_induced_mixedness(monkeypatch):
    grouped = {
        f"M:{index}": ([1.0, 1.0, 1.0, 0.0] if index == 0 else [1.0] * 4)
        for index in range(100)
    }
    summary = {
        "adapter": None,
        "adapter_tree_sha256": None,
        "decoding": STUDENT_SUPPORT_DECODING,
    }
    binding = {
        "task_file": "/tmp/student-opd.jsonl",
        "task_file_sha256": "1" * 64,
        "samples_per_problem": 4,
        "verifier_error_samples": 1,
        "verifier_error_sample_keys": [{"record_id": "M:0", "sample_idx": 3}],
        "verifier_error_sample_keys_sha256": "2" * 64,
        "summary": "/tmp/summary.json",
        "summary_sha256": "3" * 64,
        "samples": "/tmp/samples.jsonl",
        "samples_sha256": "4" * 64,
        "evaluation_git_commit": "5" * 40,
        "evaluator_file_sha256": "6" * 64,
        "evaluation_packages": EXPECTED_TEACHER_TRAIN_PACKAGES,
        "tokenizer_contract_sha256": "7" * 64,
        "evaluation_artifact_kind": quality_gates.EVALUATION_MERGED_KIND,
        "evaluation_contract": quality_gates.EVALUATION_CONTRACT,
        "evaluation_contract_sha256": "8" * 64,
        "evaluation_environment": {},
        "evaluation_post_promotion_custody": {},
        "record_seed_contract": {},
        "selected_record_ids_sha256": "9" * 64,
        "evaluation_shard_count": 1,
        "evaluation_shard_strategy": quality_gates.SHARD_STRATEGY,
        "evaluation_merge_strategy": quality_gates.MERGE_STRATEGY,
        "evaluation_merge_provenance_sha256": "a" * 64,
        "evaluation_merge_custody_sha256": "b" * 64,
        "evaluation_merger_file_sha256": "c" * 64,
    }
    monkeypatch.setattr(
        quality_gates,
        "checked_evaluation",
        lambda *_args, **_kwargs: (summary, grouped, binding),
    )
    monkeypatch.setattr(
        quality_gates,
        "_require_scientific_evaluation_contract",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        quality_gates,
        "_prepared_role_binding",
        lambda *_args, **_kwargs: ({}, {}),
    )

    result = student_support(
        Namespace(
            smoke_gate=False,
            min_records=100,
            min_pass_at_k=0.01,
            min_mixed_group_fraction=0.01,
            student_summary=Path("/tmp/summary.json"),
            student_samples=Path("/tmp/samples.jsonl"),
            student_model=STUDENT_MODEL,
            student_revision=STUDENT_REVISION,
            task_source="M",
            task_role="student_opd",
            prepared_manifest=Path("/tmp/prepared.json"),
        )
    )

    assert result["mixed_reward_group_fraction"] == 0.01
    assert result["worst_case_mixed_reward_group_fraction"] == 0.0
    assert result["passed"] is False
    assert result["authorizes_scientific_training"] is False


def teacher_trace_completion(solution, sample_idx):
    return (
        f"Final answer: {solution}."
        if sample_idx == 0
        else r"Final answer: \boxed{-999999}."
    )


def write_task(path, count, *, source="M", role="teacher_gap_dev"):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "record_id": f"{source}:{index}",
            "cluster_id": f"cluster:{source}:{index}",
            "source": source,
            "role": role,
            "prompt": [{"role": "user", "content": f"Solve problem {index}."}],
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
    state = {"commit": TEACHER_TRAINING_COMMIT, "dirty": False}
    environment = teacher_environment_contract(tmp_path, state["commit"])
    training_plan = json.loads(CANONICAL_TEACHER_TRAINING_PLAN.read_text())
    fixed_config = training_plan["fixed_config"]
    trainer_state = tmp_path / "trainer_state.json"
    trainer_state.write_text(
        json.dumps({"global_step": fixed_config["max_steps"]}, sort_keys=True) + "\n"
    )
    trainer_log = tmp_path / "trainer_log_history.json"
    trainer_log.write_text(
        json.dumps(
            [
                {
                    "step": fixed_config["max_steps"],
                    "loss": 0.0,
                    "frac_reward_zero_std": 0.0,
                    "reward_std": 0.5,
                    "completions/clipped_ratio": 0.0,
                }
            ],
            sort_keys=True,
        )
        + "\n"
    )
    reward_signal = {
        "informative_reward_observed": True,
        "reward_log_entries": 1,
        "frac_reward_zero_std": [0.0],
        "max_mixed_reward_sample_fraction": 1.0,
        "reward_std": [0.5],
        "completion_clipped_ratio": [0.0],
    }
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
    teacher_samples = tmp_path / "teacher_samples.jsonl"
    training_rows = [
        json.loads(line) for line in training_path.read_text().splitlines() if line.strip()
    ]
    teacher_samples.write_text(
        "".join(
            json.dumps(
                {
                    "schema_version": 1,
                    "reward_batch_index": step,
                    "sample_idx": sample_idx,
                    "record_id": training_rows[step % len(training_rows)]["record_id"],
                    "source": source,
                    "solution": training_rows[step % len(training_rows)]["solution"],
                    "prompt_sha256": canonical_json_sha256(
                        training_rows[step % len(training_rows)]["prompt"]
                    ),
                    "prompt_tokens": 12,
                    "prompt_token_ids": list(range(12)),
                    "completion_tokens": 32,
                    "completion_token_ids": list(range(32)),
                    "completion_text": teacher_trace_completion(
                        training_rows[step % len(training_rows)]["solution"],
                        sample_idx,
                    ),
                    "completion_sha256": hashlib.sha256(
                        teacher_trace_completion(
                            training_rows[step % len(training_rows)]["solution"],
                            sample_idx,
                        ).encode()
                    ).hexdigest(),
                    "reward": 1.0 if sample_idx == 0 else 0.0,
                },
                sort_keys=True,
            )
            + "\n"
            for step in range(fixed_config["max_steps"])
            for sample_idx in range(fixed_config["num_generations"])
        )
    )
    realized_training = {
        "reward_batches": fixed_config["max_steps"],
        "completion_samples": fixed_config["max_steps"] * fixed_config["num_generations"],
        "unique_training_records": len(training_rows),
        "realized_record_ids_sha256": canonical_json_sha256(
            [
                training_rows[step % len(training_rows)]["record_id"]
                for step in range(fixed_config["max_steps"])
            ]
        ),
        "realized_training_indices_sha256": canonical_json_sha256(
            [step % len(training_rows) for step in range(fixed_config["max_steps"])]
        ),
        "prompt_group_tokens": fixed_config["max_steps"] * 12,
        "sample_expanded_prompt_tokens": (
            fixed_config["max_steps"] * 12 * fixed_config["num_generations"]
        ),
        "total_completion_tokens": (
            fixed_config["max_steps"] * fixed_config["num_generations"] * 32
        ),
        "reward_sum": float(fixed_config["max_steps"]),
        "reward_mean": 1.0 / fixed_config["num_generations"],
        "informative_reward_groups": fixed_config["max_steps"],
        "informative_reward_group_fraction": 1.0,
        "expected_geometry_observed": True,
    }
    train_metrics.write_text(
        json.dumps(
            {
                "actual_optimizer_steps": fixed_config["max_steps"],
                "optimizer_progress_complete": True,
                "reward_signal": reward_signal,
                "realized_training": realized_training,
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
        "reward_signal": reward_signal,
        "trainer_state": str(trainer_state.resolve()),
        "trainer_state_sha256": sha256_file(trainer_state),
        "trainer_log_history": str(trainer_log.resolve()),
        "trainer_log_history_sha256": sha256_file(trainer_log),
        "train_metrics": str(train_metrics.resolve()),
        "train_metrics_sha256": sha256_file(train_metrics),
        "teacher_samples": str(teacher_samples.resolve()),
        "teacher_samples_sha256": sha256_file(teacher_samples),
        "teacher_samples_rows": realized_training["completion_samples"],
        "realized_training": realized_training,
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
        "git_state_before_candidate_save": state,
        "git_state_after_candidate_save": state,
        "git_state_end": state,
        "clean_stable_code": True,
        "environment_contract": environment,
        "stable_environment_before_candidate_save": True,
        "stable_environment_after_candidate_save": True,
        "stable_environment_end": True,
        "stable_final_artifact_hash": True,
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
    decoding=DECODING,
    exact_environment=True,
    git_commit=TEACHER_TRAINING_COMMIT,
):
    directory.mkdir(parents=True, exist_ok=True)
    return write_merged_evaluation(
        directory,
        name,
        task_path,
        rewards_by_record,
        model=model,
        revision=revision,
        adapter=adapter,
        packages={
            "torch": "2.11.0",
            "transformers": "4.57.6",
            "peft": "0.19.1",
            "math-verify": "0.9.0",
        },
        decoding=decoding,
        exact_environment=exact_environment,
        git_commit=git_commit,
    )


def legacy_summary_copy(summary_path, name):
    payload = json.loads(Path(summary_path).read_text())
    payload["schema_version"] = 1
    for field in (
        "artifact_kind",
        "evaluation_contract",
        "evaluation_contract_sha256",
        "merge",
        "merge_custody",
        "record_seed_contract",
    ):
        payload.pop(field, None)
    path = Path(summary_path).parent / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


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


def passing_scientific_teacher_fixture(
    tmp_path,
    *,
    exact_environment=True,
    evaluation_git_commit=TEACHER_TRAINING_COMMIT,
):
    prepared_manifest, _, role_paths, role_rows = write_prepared(tmp_path)
    task = role_paths["teacher_gap_dev"]
    task_rows = role_rows["teacher_gap_dev"]
    adapter = tmp_path / "teacher-adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"r": 8}\n')
    run_manifest = write_teacher_run(tmp_path, prepared_manifest, adapter)
    base_summary, base_samples = write_evaluation(
        tmp_path,
        "base",
        task,
        {row["record_id"]: [0, 0, 0, 0] for row in task_rows},
        exact_environment=exact_environment,
        git_commit=evaluation_git_commit,
    )
    trained_summary, trained_samples = write_evaluation(
        tmp_path,
        "trained",
        task,
        {row["record_id"]: [1, 1, 1, 1] for row in task_rows},
        adapter=adapter,
        exact_environment=exact_environment,
        git_commit=evaluation_git_commit,
    )
    return teacher_args(
        base_summary=base_summary,
        base_samples=base_samples,
        trained_summary=trained_summary,
        trained_samples=trained_samples,
        adapter=adapter,
        prepared_manifest=prepared_manifest,
        teacher_run_manifest=run_manifest,
    )


def teacher_target_args(
    tmp_path,
    *,
    teacher_source="M",
    target_source="O",
    selected_records=TEACHER_TARGET_REPORT_RECORDS,
    base_reward=0,
    trained_reward=1,
    evaluation_git_commit=TEACHER_TRAINING_COMMIT,
):
    prepared_manifest, prepared, role_paths, _ = write_prepared(
        tmp_path,
        teacher_gap_count=TEACHER_TARGET_REPORT_RECORDS,
        source=teacher_source,
    )
    target_relative = f"roles/{target_source}/teacher_gap_dev.jsonl"
    target_path = tmp_path / target_relative
    target_rows = write_task(
        target_path,
        TEACHER_TARGET_REPORT_RECORDS + 1,
        source=target_source,
        role="teacher_gap_dev",
    )
    prepared["files"][target_relative] = {
        "rows": len(target_rows),
        "sha256": sha256_file(target_path),
    }
    teacher_relative = f"roles/{teacher_source}/teacher_gap_dev.jsonl"
    pair = {
        "id": f"{teacher_source}_{target_source}",
        "teacher_source": teacher_source,
        "opd_source": target_source,
        "same_items": False,
        "teacher_skill_dev_limit": TEACHER_TARGET_REPORT_RECORDS,
        "target_gap_dev_limit": TEACHER_TARGET_REPORT_RECORDS,
    }
    for field, relative in (
        ("teacher_skill_dev_file", teacher_relative),
        ("target_gap_dev_file", target_relative),
    ):
        entry = prepared["files"][relative]
        pair[field] = relative
        pair[f"{field}_rows"] = entry["rows"]
        pair[f"{field}_sha256"] = entry["sha256"]
    prepared["pairs"] = [pair]
    prepared_manifest.write_text(json.dumps(prepared, indent=2, sort_keys=True) + "\n")

    adapter = tmp_path / "cross-source-teacher-adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"r": 8}\n')
    teacher_run_manifest = write_teacher_run(
        tmp_path,
        prepared_manifest,
        adapter,
        source=teacher_source,
    )
    selected = target_rows[:selected_records]
    base_summary, base_samples = write_evaluation(
        tmp_path,
        "target-base",
        target_path,
        {
            row["record_id"]: [base_reward] * 4
            for row in selected
        },
        git_commit=evaluation_git_commit,
    )
    trained_summary, trained_samples = write_evaluation(
        tmp_path,
        "target-trained",
        target_path,
        {
            row["record_id"]: [trained_reward] * 4
            for row in selected
        },
        adapter=adapter,
        git_commit=evaluation_git_commit,
    )
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
        teacher_source=teacher_source,
        target_source=target_source,
        task_role="teacher_gap_dev",
        bootstrap_draws=10_000,
        seed=0,
    )


def test_teacher_target_report_is_exact_deterministic_and_non_authorizing(tmp_path):
    args = teacher_target_args(tmp_path)
    report = teacher_target_report(args)

    assert report["report"] == TEACHER_TARGET_REPORT_TYPE
    assert report["report_strength"] == "scientific_measurement"
    assert report["valid"] is True
    assert report["teacher_source"] == "M"
    assert report["target_source"] == "O"
    assert report["pair_id"] == "M_O"
    assert report["shared_records"] == TEACHER_TARGET_REPORT_RECORDS
    assert report["target_gap_dev_limit"] == TEACHER_TARGET_REPORT_RECORDS
    assert report["target_gap_dev_file_rows"] == TEACHER_TARGET_REPORT_RECORDS + 1
    assert report["paired_delta"] == 1.0
    assert report["bootstrap_95_ci"] == [1.0, 1.0]
    assert report["bootstrap_draws"] == 10_000
    assert report["bootstrap_seed"] == 0
    assert report["authorizes_scientific_merge"] is False
    assert report["authorizes_scientific_training"] is False
    assert "gate" not in report
    assert "passed" not in report
    assert recompute_teacher_target_report(report) == report


def test_teacher_target_report_retains_negative_effect_as_valid_measurement(tmp_path):
    report = teacher_target_report(
        teacher_target_args(tmp_path, base_reward=1, trained_reward=0)
    )

    assert report["valid"] is True
    assert report["paired_delta"] == -1.0
    assert report["bootstrap_95_ci"] == [-1.0, -1.0]
    assert report["authorizes_scientific_merge"] is False


def test_teacher_target_report_rejects_source_budget_and_pair_forgery(tmp_path):
    args = teacher_target_args(tmp_path)
    args.teacher_source = args.target_source
    with pytest.raises(ValueError, match="distinct teacher and target sources"):
        teacher_target_report(args)

    args.teacher_source = "M"
    args.bootstrap_draws = 9_999
    with pytest.raises(ValueError, match="exactly 10000"):
        teacher_target_report(args)

    args.bootstrap_draws = 10_000
    prepared = json.loads(args.prepared_manifest.read_text())
    prepared["pairs"][0]["target_gap_dev_limit"] = 352
    args.prepared_manifest.write_text(
        json.dumps(prepared, indent=2, sort_keys=True) + "\n"
    )
    with pytest.raises(ValueError, match="target-gap limit is not exactly 353"):
        teacher_target_report(args)


def test_teacher_target_report_rejects_nonexact_target_prefix(tmp_path):
    args = teacher_target_args(tmp_path, selected_records=352)
    with pytest.raises(ValueError, match="require exactly 353 target records"):
        teacher_target_report(args)


def test_teacher_target_report_rejects_evaluation_commit_mismatch(tmp_path):
    args = teacher_target_args(tmp_path, evaluation_git_commit="e" * 40)
    with pytest.raises(
        ValueError,
        match="target evaluation Git commit differs from teacher training custody",
    ):
        teacher_target_report(args)


def test_teacher_target_report_cli_writes_fresh_sorted_json_and_exits_zero(
    tmp_path, monkeypatch
):
    output = tmp_path / "teacher-target-report.json"
    payload = {
        "schema_version": 3,
        "report": TEACHER_TARGET_REPORT_TYPE,
        "valid": True,
        "authorizes_scientific_merge": False,
        "authorizes_scientific_training": False,
        "paired_delta": -0.25,
    }
    monkeypatch.setattr(
        quality_gates,
        "teacher_target_report",
        lambda args: payload,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "quality_gates.py",
            "teacher-target-report",
            "--base-summary",
            "/tmp/base-summary.json",
            "--base-samples",
            "/tmp/base-samples.jsonl",
            "--trained-summary",
            "/tmp/trained-summary.json",
            "--trained-samples",
            "/tmp/trained-samples.jsonl",
            "--base-model",
            MODEL,
            "--base-revision",
            REVISION,
            "--trained-adapter",
            "/tmp/adapter",
            "--prepared-manifest",
            "/tmp/prepared.json",
            "--teacher-run-manifest",
            "/tmp/teacher-run.json",
            "--teacher-source",
            "M",
            "--target-source",
            "O",
            "--output",
            str(output),
        ],
    )

    assert quality_gates.main() == 0
    assert output.read_text() == json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        quality_gates.main()


def test_scientific_teacher_gate_rejects_legacy_monolithic_summary(tmp_path):
    args = passing_scientific_teacher_fixture(tmp_path)
    args.base_summary = legacy_summary_copy(args.base_summary, "legacy-summary.json")
    with pytest.raises(ValueError, match="requires a schema-v2 merged artifact"):
        teacher_gap(args)


def test_scientific_teacher_gate_rejects_environment_less_v1_contract(tmp_path):
    args = passing_scientific_teacher_fixture(
        tmp_path, exact_environment=False
    )
    base = json.loads(Path(args.base_summary).read_text())
    assert (
        base["evaluation_contract"]["contract"]
        == quality_gates.LEGACY_EVALUATION_CONTRACT
    )
    with pytest.raises(ValueError, match="requires exact train-environment custody"):
        teacher_gap(args)


def test_scientific_teacher_gate_rejects_evaluation_commit_mismatch(tmp_path):
    args = passing_scientific_teacher_fixture(
        tmp_path, evaluation_git_commit="e" * 40
    )
    with pytest.raises(
        ValueError,
        match="teacher evaluation Git commit differs from teacher training custody",
    ):
        teacher_gap(args)


def test_scientific_teacher_gate_requires_untampered_promotion_companions(tmp_path):
    missing_root = tmp_path / "missing"
    missing_root.mkdir()
    missing_args = passing_scientific_teacher_fixture(missing_root)
    merged_dir = Path(missing_args.base_summary).parent
    merged_companion = merged_dir.parent / f"{merged_dir.name}.custody.json"
    assert merged_companion.is_file()
    merged_companion.unlink()
    with pytest.raises(ValueError, match="post-promotion custody companion"):
        teacher_gap(missing_args)

    tampered_root = tmp_path / "tampered"
    tampered_root.mkdir()
    tampered_args = passing_scientific_teacher_fixture(tampered_root)
    merged = json.loads(Path(tampered_args.base_summary).read_text())
    shard_companion = Path(
        merged["merge"]["shards"][0]["post_promotion_custody"]
    )
    companion_payload = json.loads(shard_companion.read_text())
    companion_payload["stable_final_artifact_hash"] = False
    shard_companion.write_text(
        json.dumps(companion_payload, indent=2, sort_keys=True) + "\n"
    )
    with pytest.raises(
        ValueError, match="post-promotion custody companion does not match"
    ):
        teacher_gap(tampered_args)


def test_scientific_student_support_rejects_legacy_monolithic_summary(tmp_path):
    prepared_manifest, _, role_paths, role_rows = write_prepared(
        tmp_path, student_count=100
    )
    task = role_paths["student_opd"]
    rows = role_rows["student_opd"]
    summary, samples = write_evaluation(
        tmp_path,
        "student-legacy",
        task,
        {row["record_id"]: [0, 1, 0, 1] for row in rows},
        model=STUDENT_MODEL,
        revision=STUDENT_REVISION,
        role="student_opd",
        decoding=STUDENT_SUPPORT_DECODING,
        exact_environment=True,
    )
    legacy = legacy_summary_copy(summary, "legacy-summary.json")
    with pytest.raises(ValueError, match="requires a schema-v2 merged artifact"):
        student_support(
            Namespace(
                student_summary=legacy,
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


def rewrite_teacher_trace_claim(run_manifest, trace_rows, realized_training):
    run = json.loads(run_manifest.read_text())
    trace_path = Path(run["teacher_samples"])
    write_jsonl(trace_path, trace_rows)
    metrics_path = Path(run["train_metrics"])
    metrics = json.loads(metrics_path.read_text())
    metrics["realized_training"] = realized_training
    metrics_path.write_text(json.dumps(metrics, sort_keys=True) + "\n")
    run["teacher_samples_sha256"] = sha256_file(trace_path)
    run["teacher_samples_rows"] = len(trace_rows)
    run["realized_training"] = realized_training
    run["train_metrics_sha256"] = sha256_file(metrics_path)
    run_manifest.write_text(json.dumps(run, indent=2, sort_keys=True) + "\n")


def test_scientific_teacher_gate_requires_records_strict_gain_and_positive_ci(tmp_path):
    prepared_manifest, _, role_paths, role_rows = write_prepared(tmp_path)
    task = role_paths["teacher_gap_dev"]
    task_rows = role_rows["teacher_gap_dev"]
    adapter = tmp_path / "teacher-adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"r": 8}\n')
    teacher_run_manifest = write_teacher_run(tmp_path, prepared_manifest, adapter)
    base_rewards = {row["record_id"]: [0, 0, 0, 0] for row in task_rows}
    tied_rewards = {row["record_id"]: [0, 0, 0, 0] for row in task_rows}
    better_rewards = {row["record_id"]: [1, 1, 1, 1] for row in task_rows}
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
    assert better["teacher_training_environment"]["schema_version"] == 2

    intact_run = json.loads(teacher_run_manifest.read_text())
    trace_path = Path(intact_run["teacher_samples"])
    intact_trace = trace_path.read_text()
    trace_path.write_text("".join("{}\n" for _ in range(400)))
    forged_run = dict(intact_run)
    forged_run["teacher_samples_sha256"] = sha256_file(trace_path)
    teacher_run_manifest.write_text(json.dumps(forged_run, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="trace row 1"):
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
    trace_path.write_text(intact_trace)
    teacher_run_manifest.write_text(json.dumps(intact_run, indent=2, sort_keys=True) + "\n")

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


def test_scientific_teacher_gate_rejects_missing_or_mutated_environment_custody(tmp_path):
    args = passing_scientific_teacher_fixture(tmp_path)
    assert teacher_gap(args)["passed"]
    intact = json.loads(args.teacher_run_manifest.read_text())

    missing = dict(intact)
    missing.pop("environment_contract")
    args.teacher_run_manifest.write_text(json.dumps(missing, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="train-environment custody"):
        teacher_gap(args)

    args.teacher_run_manifest.write_text(json.dumps(intact, indent=2, sort_keys=True) + "\n")
    freeze = Path(intact["environment_contract"]["train_freeze"]["path"])
    freeze.write_text("mutated-freeze\n")
    with pytest.raises(ValueError, match="freeze hash has drifted"):
        teacher_gap(args)


@pytest.mark.parametrize(
    "field",
    (
        "stable_environment_before_candidate_save",
        "stable_environment_after_candidate_save",
        "stable_environment_end",
        "stable_final_artifact_hash",
    ),
)
def test_scientific_teacher_gate_requires_each_promotion_custody_attestation(
    tmp_path, field
):
    args = passing_scientific_teacher_fixture(tmp_path)
    run = json.loads(args.teacher_run_manifest.read_text())
    run[field] = False
    args.teacher_run_manifest.write_text(json.dumps(run, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match=field):
        teacher_gap(args)


@pytest.mark.parametrize(
    "field",
    ("git_state_before_candidate_save", "git_state_after_candidate_save"),
)
def test_scientific_teacher_gate_requires_both_candidate_git_states(tmp_path, field):
    args = passing_scientific_teacher_fixture(tmp_path)
    run = json.loads(args.teacher_run_manifest.read_text())
    run.pop(field)
    args.teacher_run_manifest.write_text(json.dumps(run, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="candidate-promotion Git custody"):
        teacher_gap(args)


def test_scientific_teacher_gate_rejects_all_zero_training_trace(tmp_path):
    args = passing_scientific_teacher_fixture(tmp_path)
    assert teacher_gap(args)["passed"]
    run = json.loads(args.teacher_run_manifest.read_text())
    trace_path = Path(run["teacher_samples"])
    trace_rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
    for row in trace_rows:
        row["reward"] = 0.0
    realized = dict(run["realized_training"])
    realized.update(
        {
            "reward_sum": 0.0,
            "reward_mean": 0.0,
            "informative_reward_groups": 0,
            "informative_reward_group_fraction": 0.0,
        }
    )
    rewrite_teacher_trace_claim(args.teacher_run_manifest, trace_rows, realized)

    with pytest.raises(ValueError, match="reward disagrees with TRL accuracy"):
        teacher_gap(args)


def test_scientific_teacher_gate_rejects_repeated_record_sampler_collapse(tmp_path):
    args = passing_scientific_teacher_fixture(tmp_path)
    assert teacher_gap(args)["passed"]
    run = json.loads(args.teacher_run_manifest.read_text())
    trace_path = Path(run["teacher_samples"])
    trace_rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
    first = trace_rows[0]
    for row in trace_rows:
        row["record_id"] = first["record_id"]
        row["solution"] = first["solution"]
        row["prompt_sha256"] = first["prompt_sha256"]
        row["completion_text"] = teacher_trace_completion(
            first["solution"], row["sample_idx"]
        )
        row["completion_sha256"] = hashlib.sha256(
            row["completion_text"].encode()
        ).hexdigest()
    realized = dict(run["realized_training"])
    realized.update(
        {
            "unique_training_records": 1,
            "realized_record_ids_sha256": canonical_json_sha256(
                [first["record_id"]] * 100
            ),
            "realized_training_indices_sha256": canonical_json_sha256([0] * 100),
        }
    )
    rewrite_teacher_trace_claim(args.teacher_run_manifest, trace_rows, realized)

    with pytest.raises(ValueError, match="trace-recomputed prompt geometry"):
        teacher_gap(args)


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
        tmp_path,
        "smoke-base",
        task,
        {row["record_id"]: [0, 0, 0, 0] for row in task_rows},
    )
    trained_summary, trained_samples = write_evaluation(
        tmp_path,
        "smoke-trained",
        task,
        {
            task_rows[0]["record_id"]: [1, 1, 1, 1],
            task_rows[1]["record_id"]: [0, 0, 0, 0],
        },
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
        row["record_id"]: ([0, 1, 0, 1] if index < 50 else [0, 0, 0, 0])
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
        decoding=STUDENT_SUPPORT_DECODING,
        exact_environment=True,
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
    assert result["evaluation_contract"] == quality_gates.EVALUATION_CONTRACT
    assert isinstance(result["evaluation_environment"], dict)
    assert isinstance(result["evaluation_post_promotion_custody"], dict)
    assert recompute_student_gate(result) == result


def test_scientific_student_support_rejects_environment_less_v1_contract(tmp_path):
    prepared_manifest, _, role_paths, role_rows = write_prepared(tmp_path)
    task = role_paths["student_opd"]
    task_rows = role_rows["student_opd"]
    summary, samples = write_evaluation(
        tmp_path,
        "student-environment-less",
        task,
        {row["record_id"]: [0, 1, 0, 1] for row in task_rows},
        model=STUDENT_MODEL,
        revision=STUDENT_REVISION,
        role="student_opd",
        decoding=STUDENT_SUPPORT_DECODING,
        exact_environment=False,
    )
    with pytest.raises(ValueError, match="requires exact train-environment custody"):
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


def test_small_student_support_is_smoke_only(tmp_path):
    prepared_manifest, _, role_paths, role_rows = write_prepared(
        tmp_path, teacher_gap_count=2, student_count=2
    )
    task = role_paths["student_opd"]
    task_rows = role_rows["student_opd"]
    rewards = {
        task_rows[0]["record_id"]: [0, 1, 0, 1],
        task_rows[1]["record_id"]: [0, 0, 0, 0],
    }
    summary, samples = write_evaluation(
        tmp_path,
        "student-smoke",
        task,
        rewards,
        model=STUDENT_MODEL,
        revision=STUDENT_REVISION,
        role="student_opd",
        decoding=STUDENT_SUPPORT_DECODING,
        exact_environment=True,
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
        tmp_path,
        "mutation-base",
        task,
        {row["record_id"]: [0] for row in task_rows},
        exact_environment=False,
    )
    trained_summary, trained_samples = write_evaluation(
        tmp_path,
        "mutation-trained",
        task,
        {row["record_id"]: [1] for row in task_rows},
        adapter=adapter,
        exact_environment=False,
    )
    teacher_run_manifest = write_teacher_run(
        tmp_path, prepared_manifest, adapter, scientific=False
    )
    (adapter / "adapter_config.json").write_text('{"r": 16}\n')
    with pytest.raises(ValueError, match="adapter changed after generation"):
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
        tmp_path,
        "prefix-base",
        task,
        {row["record_id"]: [0, 0, 0, 0] for row in selected},
    )
    trained_summary, trained_samples = write_evaluation(
        tmp_path,
        "prefix-trained",
        task,
        {row["record_id"]: [1, 1, 1, 1] for row in selected},
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
        tmp_path,
        "full-base",
        task,
        {row["record_id"]: [0, 0, 0, 0] for row in all_rows},
    )
    trained_summary, trained_samples = write_evaluation(
        tmp_path,
        "full-trained",
        task,
        {row["record_id"]: [1, 1, 1, 1] for row in all_rows},
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
    rewards = {row["record_id"]: [0, 1, 0, 1] for row in selected}
    summary, samples = write_evaluation(
        tmp_path,
        "student-prefix",
        task,
        rewards,
        model=STUDENT_MODEL,
        revision=STUDENT_REVISION,
        role="student_opd",
        decoding=STUDENT_SUPPORT_DECODING,
        exact_environment=True,
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
