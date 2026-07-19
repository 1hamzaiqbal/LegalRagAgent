import importlib.metadata
import json
from argparse import Namespace

import pytest

from scripts.opd_math.train_teacher_grpo import (
    CANONICAL_TRAINING_PLAN,
    ENVIRONMENT_VERIFIER,
    EXPECTED_TEACHER_TRAIN_PACKAGES,
    prompt_token_diagnostics,
    reward_signal_diagnostics,
    sha256_file,
    summarize_teacher_trace,
    teacher_environment_contract_unchanged,
    validate_prepared_contract,
    validate_teacher_environment_contract,
    validate_static_args,
    validate_training_plan_contract,
)


MODEL = "Qwen/Qwen3-8B"
REVISION = "a" * 40


def args(**overrides):
    values = {
        "max_steps": 2,
        "limit": 2,
        "num_generations": 4,
        "gradient_accumulation_steps": 4,
        "max_prompt_tokens": 2304,
        "max_completion_length": 256,
        "learning_rate": 2e-5,
        "lora_r": 16,
        "seed": 0,
        "source": "M",
        "budget_mode": "primary_matched",
        "smoke": False,
        "model": MODEL,
        "model_revision": REVISION,
    }
    values.update(overrides)
    return Namespace(**values)


def prepared_fixture(tmp_path, rows):
    task_file = tmp_path / "roles" / "M" / "teacher_train.jsonl"
    task_file.parent.mkdir(parents=True)
    task_file.write_text("".join(json.dumps(row) + "\n" for row in rows))
    source_manifest = tmp_path / "source_manifest.json"
    source_manifest.write_text(
        json.dumps({"models": {"teacher": {"id": MODEL, "revision": REVISION}}})
    )
    prepared = {
        "schema_version": 1,
        "scientific_use_allowed": True,
        "source_manifest_path": str(source_manifest.resolve()),
        "source_manifest_sha256": sha256_file(source_manifest),
        "primary_matched_budgets": {"teacher_train": 2},
        "files": {
            "roles/M/teacher_train.jsonl": {
                "rows": len(rows),
                "sha256": sha256_file(task_file),
            }
        },
    }
    manifest = tmp_path / "prepared_manifest.json"
    manifest.write_text(json.dumps(prepared))
    return task_file, manifest, source_manifest, prepared


def row(record_id="M:one", source="M", role="teacher_train"):
    return {
        "record_id": record_id,
        "source": source,
        "role": role,
        "prompt": [{"role": "user", "content": "2+2"}],
        "solution": "\\boxed{4}",
    }


def test_static_contract_rejects_nonpositive_and_indivisible(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    assert validate_static_args(args())["generation_batch_size"] == 4
    with pytest.raises(ValueError, match="max-steps"):
        validate_static_args(args(max_steps=0))
    with pytest.raises(ValueError, match="must be divisible"):
        validate_static_args(args(num_generations=3))


def test_scientific_teacher_recipe_is_bound_to_one_source_independent_plan(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    run_args = args(
        training_plan=CANONICAL_TRAINING_PLAN,
        max_steps=100,
        num_generations=4,
        gradient_accumulation_steps=4,
        max_prompt_tokens=2304,
        max_completion_length=1024,
        learning_rate=2e-5,
        lora_r=16,
        seed=0,
        require_informative_reward=True,
    )
    static = validate_static_args(run_args)
    contract = validate_training_plan_contract(
        run_args, static, intended_scientific_run=True
    )
    assert contract["training_plan_compliant"]
    assert contract["training_plan_id"] == "opd_math_teacher_primary_v2"
    assert contract["training_plan_config_sha256"] == contract["teacher_training_config_sha256"]

    run_args.max_steps = 99
    static = validate_static_args(run_args)
    with pytest.raises(ValueError, match="predeclared matched recipe"):
        validate_training_plan_contract(run_args, static, intended_scientific_run=True)


def test_prompt_contract_rejects_implicit_truncation():
    class Tokenizer:
        def apply_chat_template(self, prompt, **kwargs):
            return list(range(prompt[0]["tokens"]))

    rows = [
        {"prompt": [{"tokens": 3}]},
        {"prompt": [{"tokens": 5}]},
    ]
    diagnostics = prompt_token_diagnostics(Tokenizer(), rows, max_prompt_tokens=5)
    assert diagnostics["max_rendered_prompt_tokens"] == 5
    assert diagnostics["implicit_truncation_allowed"] is False
    with pytest.raises(RuntimeError, match="never silently truncated"):
        prompt_token_diagnostics(Tokenizer(), rows, max_prompt_tokens=4)


def test_scientific_teacher_environment_is_commit_specific_and_reverified(
    tmp_path, monkeypatch
):
    commit = "c" * 40
    environment_root = tmp_path / "train-env"
    environment_root.mkdir()
    freeze = tmp_path / "environment_freezes" / commit / "train.freeze.txt"
    freeze.parent.mkdir(parents=True)
    freeze.write_text("exact-freeze\n")
    freeze_hash = sha256_file(freeze)
    verification = {
        "schema_version": 1,
        "schema": "opd_math_environment_verification_v1",
        "status": "passed",
        "environment_root": str(environment_root.resolve()),
        "expected_commit": commit,
        "freeze_kind": "train",
        "commit_freeze": {
            "path": str(freeze.resolve()),
            "sha256": freeze_hash,
            "byte_identical_to_requirements_freeze": True,
        },
        "expected_executable": None,
    }
    monkeypatch.setattr(
        "scripts.opd_math.train_teacher_grpo.package_versions",
        lambda: dict(EXPECTED_TEACHER_TRAIN_PACKAGES),
    )
    monkeypatch.setattr(
        "scripts.opd_math.train_teacher_grpo.verify_environment",
        lambda **kwargs: dict(verification),
    )
    monkeypatch.setattr(
        "scripts.opd_math.train_teacher_grpo.reverify_recorded_environment",
        lambda recorded, *, in_process: dict(recorded),
    )
    run_args = args(
        train_environment_root=environment_root,
        train_environment_freeze=freeze,
    )
    contract = validate_teacher_environment_contract(
        run_args, {"commit": commit, "dirty": False}, required=True
    )
    assert contract["git_commit"] == commit
    assert contract["verifier"] == {
        "path": str(ENVIRONMENT_VERIFIER.resolve()),
        "sha256": sha256_file(ENVIRONMENT_VERIFIER),
    }
    assert teacher_environment_contract_unchanged(contract)

    with monkeypatch.context() as missing_distribution:
        def fail_package_lookup():
            raise importlib.metadata.PackageNotFoundError("torch")

        missing_distribution.setattr(
            "scripts.opd_math.train_teacher_grpo.package_versions",
            fail_package_lookup,
        )
        assert not teacher_environment_contract_unchanged(contract)

    freeze.write_text("mutated\n")
    assert not teacher_environment_contract_unchanged(contract)


def test_scientific_teacher_environment_rejects_missing_or_wrong_commit_freeze(tmp_path):
    commit = "c" * 40
    with pytest.raises(ValueError, match="requires --train-environment-root"):
        validate_teacher_environment_contract(
            args(), {"commit": commit, "dirty": False}, required=True
        )

    environment_root = tmp_path / "train-env"
    environment_root.mkdir()
    wrong = tmp_path / "environment_freezes" / ("d" * 40) / "train.freeze.txt"
    wrong.parent.mkdir(parents=True)
    wrong.write_text("exact-freeze\n")
    with pytest.raises(ValueError, match="commit-specific"):
        validate_teacher_environment_contract(
            args(train_environment_root=environment_root, train_environment_freeze=wrong),
            {"commit": commit, "dirty": False},
            required=True,
        )


def test_primary_contract_binds_exact_role_file_and_budget(tmp_path):
    task_file, manifest, source_manifest, prepared = prepared_fixture(
        tmp_path, [row("M:one"), row("M:two")]
    )
    run_args = args(
        task_file=task_file,
        prepared_manifest=manifest,
        source_manifest=source_manifest,
    )
    selected, contract = validate_prepared_contract(run_args, prepared)
    assert len(selected) == 2
    assert contract["relative_task_file"] == "roles/M/teacher_train.jsonl"

    mismatched = vars(run_args).copy()
    mismatched["limit"] = 1
    with pytest.raises(ValueError, match="exact prepared budget"):
        validate_prepared_contract(args(**mismatched), prepared)


def test_role_validation_scans_rows_beyond_selected_smoke_prefix(tmp_path):
    task_file, manifest, source_manifest, prepared = prepared_fixture(
        tmp_path, [row("M:one"), row("O:two", source="O")]
    )
    run_args = args(
        task_file=task_file,
        prepared_manifest=manifest,
        source_manifest=source_manifest,
        limit=1,
        smoke=True,
    )
    with pytest.raises(ValueError, match="expected source='M'"):
        validate_prepared_contract(run_args, prepared)


def test_teacher_contract_rejects_unpinned_model_or_mutated_source_manifest(tmp_path):
    task_file, manifest, source_manifest, prepared = prepared_fixture(
        tmp_path, [row("M:one"), row("M:two")]
    )
    with pytest.raises(ValueError, match="not the pinned primary teacher"):
        validate_prepared_contract(
            args(
                task_file=task_file,
                prepared_manifest=manifest,
                source_manifest=source_manifest,
                model="Qwen/Qwen3-4B",
            ),
            prepared,
        )
    source_manifest.write_text(json.dumps({"models": {"teacher": {"id": MODEL, "revision": "b" * 40}}}))
    with pytest.raises(ValueError, match="hash does not match"):
        validate_prepared_contract(
            args(
                task_file=task_file,
                prepared_manifest=manifest,
                source_manifest=source_manifest,
            ),
            prepared,
        )


def test_reward_signal_requires_a_mixed_group_entry():
    flat = reward_signal_diagnostics([{"frac_reward_zero_std": 1.0, "reward_std": 0.0}])
    mixed = reward_signal_diagnostics([{"frac_reward_zero_std": 0.5, "reward_std": 0.5}])
    missing = reward_signal_diagnostics([{"loss": 0.0}])
    assert not flat["informative_reward_observed"]
    assert mixed["informative_reward_observed"]
    assert not missing["informative_reward_observed"]


def test_teacher_trace_reports_realized_prompt_geometry(tmp_path):
    path = tmp_path / "teacher_samples.jsonl"
    rows = []
    for step, record_id in enumerate(("M:one", "M:two")):
        for sample_idx in range(4):
            rows.append(
                {
                    "reward_batch_index": step,
                    "sample_idx": sample_idx,
                    "record_id": record_id,
                    "prompt_tokens": 10 + step,
                    "completion_tokens": 20 + sample_idx,
                    "reward": 1.0 if sample_idx == 0 else 0.0,
                }
            )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    record_indices = {"M:one": 0, "M:two": 1}
    summary = summarize_teacher_trace(
        path,
        expected_steps=2,
        num_generations=4,
        record_index_by_id=record_indices,
    )

    assert summary["reward_batches"] == 2
    assert summary["completion_samples"] == 8
    assert summary["unique_training_records"] == 2
    assert summary["prompt_group_tokens"] == 21
    assert summary["sample_expanded_prompt_tokens"] == 84
    assert summary["total_completion_tokens"] == 172
    assert summary["informative_reward_groups"] == 2
    assert summary["reward_sum"] == 2.0
    assert summary["expected_geometry_observed"]

    rows.pop()
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="incomplete generations"):
        summarize_teacher_trace(
            path,
            expected_steps=2,
            num_generations=4,
            record_index_by_id=record_indices,
        )
