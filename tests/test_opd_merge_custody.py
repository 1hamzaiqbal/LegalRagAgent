from argparse import Namespace
import hashlib
import json
from pathlib import Path

import pytest

from tests.opd_evaluation_fixture import write_merged_evaluation
from scripts.opd_math import quality_gates

from scripts.opd_math.merge_adapter import (
    clean_stable_git_custody,
    require_same_custody,
    validate_teacher_gate_for_merge,
)
from scripts.opd_math.quality_gates import (
    CANONICAL_TEACHER_TRAINING_PLAN,
    EXPECTED_TEACHER_TRAIN_PACKAGES,
    canonical_json_sha256,
    sha256_file,
    sha256_tree,
    teacher_gap,
)


MODEL = "Qwen/Qwen3-8B"
REVISION = "a" * 40
EVALUATOR = Path(__file__).resolve().parents[1] / "scripts" / "opd_math" / "evaluate_math.py"
DECODING = {
    "thinking": False,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "max_new_tokens": 1024,
    "seed": 0,
}


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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def teacher_trace_completion(solution, sample_idx):
    return (
        f"Final answer: {solution}."
        if sample_idx == 0
        else r"Final answer: \boxed{-999999}."
    )


def task_rows(source, role, count):
    return [
        {
            "record_id": f"{source}:{role}:{index}",
            "cluster_id": f"cluster:{source}:{role}:{index}",
            "source": source,
            "role": role,
            "prompt": [{"role": "user", "content": f"Solve problem {index}."}],
            "solution": rf"\boxed{{{index + 1}}}",
        }
        for index in range(count)
    ]


def write_evaluation(tmp_path, name, task, rows, reward, *, adapter=None):
    return write_merged_evaluation(
        tmp_path,
        name,
        task,
        {row["record_id"]: [reward] * 4 for row in rows},
        model=MODEL,
        revision=REVISION,
        adapter=adapter,
        packages={
            "torch": "2.11.0",
            "transformers": "4.57.6",
            "peft": "0.19.1",
            "math-verify": "0.9.0",
        },
        decoding=DECODING,
    )


def write_gate_fixture(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    source_manifest = tmp_path / "source_manifest.json"
    source_manifest.write_text(
        json.dumps(
            {
                "models": {
                    "teacher": {"id": MODEL, "revision": REVISION},
                    "student": {"id": "Qwen/Qwen3-1.7B", "revision": "b" * 40},
                }
            },
            sort_keys=True,
        )
        + "\n"
    )
    train_task = tmp_path / "roles" / "M" / "teacher_train.jsonl"
    gap_task = tmp_path / "roles" / "M" / "teacher_gap_dev.jsonl"
    train_rows = task_rows("M", "teacher_train", 2)
    gap_rows = task_rows("M", "teacher_gap_dev", 200)
    write_jsonl(train_task, train_rows)
    write_jsonl(gap_task, gap_rows)
    prepared = tmp_path / "prepared_manifest.json"
    prepared_payload = {
        "schema_version": 1,
        "scientific_use_allowed": True,
        "code_git_state": {"commit": "f" * 40, "dirty": False},
        "source_manifest_path": str(source_manifest.resolve()),
        "source_manifest_sha256": sha256_file(source_manifest),
        "primary_matched_budgets": {"teacher_train": 2, "teacher_gap_dev": 200},
        "files": {
            "roles/M/teacher_train.jsonl": {
                "rows": 2,
                "sha256": sha256_file(train_task),
            },
            "roles/M/teacher_gap_dev.jsonl": {
                "rows": 200,
                "sha256": sha256_file(gap_task),
            },
        },
    }
    prepared.write_text(json.dumps(prepared_payload, indent=2, sort_keys=True) + "\n")

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"r": 8}\n')
    state = {"commit": "c" * 40, "dirty": False}
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
    write_jsonl(
        teacher_samples,
        [
            {
                "schema_version": 1,
                "reward_batch_index": step,
                "sample_idx": sample_idx,
                "record_id": train_rows[step % len(train_rows)]["record_id"],
                "source": "M",
                "solution": train_rows[step % len(train_rows)]["solution"],
                "prompt_sha256": canonical_json_sha256(
                    train_rows[step % len(train_rows)]["prompt"]
                ),
                "prompt_tokens": 12,
                "prompt_token_ids": list(range(12)),
                "completion_tokens": 32,
                "completion_token_ids": list(range(32)),
                "completion_text": teacher_trace_completion(
                    train_rows[step % len(train_rows)]["solution"], sample_idx
                ),
                "completion_sha256": hashlib.sha256(
                    teacher_trace_completion(
                        train_rows[step % len(train_rows)]["solution"], sample_idx
                    ).encode()
                ).hexdigest(),
                "reward": 1.0 if sample_idx == 0 else 0.0,
            }
            for step in range(fixed_config["max_steps"])
            for sample_idx in range(fixed_config["num_generations"])
        ],
    )
    realized_training = {
        "reward_batches": fixed_config["max_steps"],
        "completion_samples": fixed_config["max_steps"] * fixed_config["num_generations"],
        "unique_training_records": len(train_rows),
        "realized_record_ids_sha256": canonical_json_sha256(
            [
                train_rows[step % len(train_rows)]["record_id"]
                for step in range(fixed_config["max_steps"])
            ]
        ),
        "realized_training_indices_sha256": canonical_json_sha256(
            [step % len(train_rows) for step in range(fixed_config["max_steps"])]
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
    run_manifest = tmp_path / "teacher-run.json"
    run_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stage": "teacher_grpo",
                "status": "completed",
                "scientific_use_allowed": True,
                "model": MODEL,
                "model_revision": REVISION,
                "source": "M",
                "role": "teacher_train",
                "budget_mode": "primary_matched",
                "task_file": str(train_task.resolve()),
                "task_file_sha256": sha256_file(train_task),
                "prepared_manifest": str(prepared.resolve()),
                "prepared_manifest_sha256": sha256_file(prepared),
                "source_manifest": str(source_manifest.resolve()),
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
                "intended_scientific_run": True,
                "selected_rows": 2,
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
                    "selected_prompts": 2,
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
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    base_summary, base_samples = write_evaluation(
        tmp_path, "base", gap_task, gap_rows, 0
    )
    trained_summary, trained_samples = write_evaluation(
        tmp_path, "trained", gap_task, gap_rows, 1, adapter=adapter
    )
    gate = teacher_gap(
        Namespace(
            base_summary=base_summary,
            base_samples=base_samples,
            trained_summary=trained_summary,
            trained_samples=trained_samples,
            base_model=MODEL,
            base_revision=REVISION,
            trained_adapter=adapter,
            prepared_manifest=prepared,
            teacher_run_manifest=run_manifest,
            task_source="M",
            task_role="teacher_gap_dev",
            min_delta=0.0,
            min_records=None,
            bootstrap_draws=1_000,
            seed=0,
            smoke_gate=False,
        )
    )
    manifest = tmp_path / "teacher-gap.json"
    manifest.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
    return manifest, adapter


def test_merge_custody_accepts_only_recomputed_scientific_gate(tmp_path):
    manifest, adapter = write_gate_fixture(tmp_path)
    custody = validate_teacher_gate_for_merge(
        manifest,
        base_model=MODEL,
        base_revision=REVISION,
        adapter=adapter,
    )
    assert custody["adapter_tree_sha256"] == sha256_tree(adapter)
    assert custody["manifest_sha256"] == sha256_file(manifest)
    assert custody["gate"]["teacher_training_environment"]["schema_version"] == 2

    with pytest.raises(ValueError, match="base identity"):
        validate_teacher_gate_for_merge(
            manifest,
            base_model="Qwen/Qwen3-4B",
            base_revision=REVISION,
            adapter=adapter,
        )


def test_merge_custody_rejects_generic_smoke_or_mutated_artifacts(tmp_path):
    manifest, adapter = write_gate_fixture(tmp_path)
    generic = tmp_path / "generic.json"
    generic.write_text('{"passed": true}\n')
    with pytest.raises(ValueError, match="schema_version"):
        validate_teacher_gate_for_merge(
            generic, base_model=MODEL, base_revision=REVISION, adapter=adapter
        )

    payload = json.loads(manifest.read_text())
    payload["gate"] = "teacher_gap_smoke_v1"
    smoke = tmp_path / "smoke.json"
    smoke.write_text(json.dumps(payload) + "\n")
    with pytest.raises(ValueError, match="merge requires gate"):
        validate_teacher_gate_for_merge(
            smoke, base_model=MODEL, base_revision=REVISION, adapter=adapter
        )

    payload = json.loads(manifest.read_text())
    payload.pop("teacher_training_environment")
    missing_environment = tmp_path / "missing-environment.json"
    missing_environment.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="train-environment custody"):
        validate_teacher_gate_for_merge(
            missing_environment,
            base_model=MODEL,
            base_revision=REVISION,
            adapter=adapter,
        )

    (adapter / "adapter_config.json").write_text('{"r": 16}\n')
    with pytest.raises(ValueError, match="adapter tree changed"):
        validate_teacher_gate_for_merge(
            manifest, base_model=MODEL, base_revision=REVISION, adapter=adapter
        )


def test_merge_custody_rejects_changed_or_self_attested_evidence(tmp_path):
    manifest, adapter = write_gate_fixture(tmp_path)
    gate = json.loads(manifest.read_text())
    Path(gate["trained_samples"]).write_text('{"changed": true}\n')
    with pytest.raises(ValueError, match="bound artifact changed"):
        validate_teacher_gate_for_merge(
            manifest, base_model=MODEL, base_revision=REVISION, adapter=adapter
        )

    manifest, adapter = write_gate_fixture(tmp_path / "fabricated")
    payload = json.loads(manifest.read_text())
    payload["paired_delta"] = 0.5
    fabricated = manifest.parent / "fabricated-gate.json"
    fabricated.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="deterministic recomputation"):
        validate_teacher_gate_for_merge(
            fabricated, base_model=MODEL, base_revision=REVISION, adapter=adapter
        )


def test_merge_custody_rejects_rehashed_completion_reward_tampering(tmp_path):
    manifest, adapter = write_gate_fixture(tmp_path)
    gate = json.loads(manifest.read_text())

    trained_samples = Path(gate["trained_samples"])
    rows = [json.loads(line) for line in trained_samples.read_text().splitlines()]
    rows[0]["completion_text"] = r"Final answer: \boxed{-999999}."
    write_jsonl(trained_samples, rows)

    trained_summary = Path(gate["trained_summary"])
    summary = json.loads(trained_summary.read_text())
    summary["samples_file_sha256"] = sha256_file(trained_samples)
    trained_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    # Simulate an attacker updating every directly exposed merged-content hash
    # while retaining the fabricated reward/status pair. Shard reconstruction
    # should reject this even before semantic reward recomputation is needed.
    gate["trained_samples_sha256"] = sha256_file(trained_samples)
    gate["trained_summary_sha256"] = sha256_file(trained_summary)
    forged = tmp_path / "completion-tampered-gate.json"
    forged.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="exact ordered concatenation"):
        validate_teacher_gate_for_merge(
            forged, base_model=MODEL, base_revision=REVISION, adapter=adapter
        )


def test_merge_requires_clean_stable_git_and_unchanged_bound_inputs():
    clean_start = {"commit": "a" * 40, "dirty": False}
    assert clean_stable_git_custody(clean_start, dict(clean_start))
    assert not clean_stable_git_custody(
        clean_start, {"commit": "b" * 40, "dirty": False}
    )
    assert not clean_stable_git_custody(
        clean_start, {"commit": "a" * 40, "dirty": True}
    )

    custody = {
        "manifest": "/tmp/gate.json",
        "manifest_sha256": "c" * 64,
        "adapter": "/tmp/adapter",
        "adapter_tree_sha256": "d" * 64,
    }
    require_same_custody(custody, dict(custody))
    changed = dict(custody, adapter_tree_sha256="e" * 64)
    with pytest.raises(RuntimeError, match="adapter_tree_sha256"):
        require_same_custody(custody, changed)
