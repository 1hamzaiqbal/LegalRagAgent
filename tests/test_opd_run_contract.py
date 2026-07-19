import hashlib
import json
from argparse import Namespace

import pytest

from scripts.opd.opd_train import (
    EXPECTED_TRAIN_PACKAGES,
    _validate_gate_prepared_binding,
    count_jsonl_objects,
    environment_contract_unchanged,
    recompute_student_trace_geometry,
    resolve_trace_directory,
    run,
    sample_trace_rows,
    validate_student_training_plan_contract,
    validate_environment_contract,
    validate_run_contract,
    write_completion_manifests,
)


def test_run_does_not_shadow_sample_trace_rows_callable():
    assert callable(sample_trace_rows)
    assert "sample_trace_rows" not in run.__code__.co_varnames
    assert "sample_trace_rows" in run.__code__.co_names
    assert "observed_sample_trace_rows" in run.__code__.co_varnames


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def args_for(task_file, prepared, **overrides):
    values = {
        "mode": "task_rl_k1_gap",
        "steps": 1,
        "max_new_tokens": 16,
        "max_prompt_tokens": 128,
        "lr": 1e-5,
        "grad_clip": 1.0,
        "gradient_checkpointing": True,
        "lora": 8,
        "group_size": 2,
        "micro_prompts": 1,
        "task_reward_coef": 1.0,
        "k1_coef": 0.01,
        "gap_gate_beta": 5.0,
        "advantage_clip": 5.0,
        "min_informative_group_fraction": 0.05,
        "teacher_connect_timeout": 1.0,
        "teacher_read_timeout": 1.0,
        "teacher_retries": 1,
        "teacher_url": "http://teacher",
        "teacher_model": "served",
        "teacher_checkpoint": "Qwen/Qwen3-8B",
        "teacher_server_max_model_len": 4096,
        "teacher_base_model": None,
        "teacher_base_revision": None,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "prepared_manifest": str(prepared),
        "task_file": str(task_file),
        "task_limit": 1,
        "pair_id": "M_M",
        "student_source": None,
        "budget_mode": "dose_response",
        "student": "Qwen/Qwen3-1.7B",
        "student_revision": "student-revision",
        "student_support_manifest": None,
        "teacher_gap_manifest": None,
        "teacher_provenance_manifest": None,
        "server_scoring_contract": None,
        "tokenizer_contract": None,
        "allow_ungated_smoke": True,
        "enable_thinking": False,
    }
    values.update(overrides)
    return Namespace(**values)


def prepared_fixture(tmp_path):
    source_manifest = tmp_path / "source_manifest.json"
    source_manifest.write_text('{"schema_version":1}\n')
    task = tmp_path / "roles" / "M" / "student_opd.jsonl"
    task.parent.mkdir(parents=True)
    row = {
        "record_id": "M:train:one",
        "source": "M",
        "role": "student_opd",
        "prompt": [{"role": "user", "content": "1+1"}],
        "solution": r"\boxed{2}",
    }
    task.write_text(json.dumps(row) + "\n")
    skill = tmp_path / "roles" / "M" / "teacher_gap_dev.jsonl"
    skill.write_text("{}\n")
    manifest = {
        "scientific_use_allowed": True,
        "source_manifest_path": str(source_manifest.resolve()),
        "source_manifest_sha256": digest(source_manifest),
        "primary_matched_budgets": {"student_opd": 1},
        "files": {
            "roles/M/student_opd.jsonl": {"rows": 1, "sha256": digest(task)},
            "roles/M/teacher_gap_dev.jsonl": {"rows": 1, "sha256": digest(skill)},
            "eval/M_test.jsonl": {"rows": 1, "sha256": digest(task)},
        },
        "pairs": [
            {
                "id": "M_M",
                "teacher_source": "M",
                "opd_source": "M",
                "student_opd_file": "roles/M/student_opd.jsonl",
                "teacher_skill_dev_file": "roles/M/teacher_gap_dev.jsonl",
            }
        ],
    }
    prepared = tmp_path / "prepared.json"
    prepared.write_text(json.dumps(manifest))
    return task, row, prepared


def test_smoke_pair_is_bound_to_exact_student_role(tmp_path):
    task, row, prepared = prepared_fixture(tmp_path)
    result = validate_run_contract(args_for(task, prepared), [row])
    assert result[-1]["task_role_file"] == "roles/M/student_opd.jsonl"
    assert result[-1]["pair_id"] == "M_M"


def test_registered_hash_does_not_authorize_wrong_role(tmp_path):
    task, row, prepared = prepared_fixture(tmp_path)
    row = dict(row, role="external_eval")
    with pytest.raises(ValueError, match="student_opd"):
        validate_run_contract(args_for(task, prepared), [row])


def test_primary_budget_must_match_manifest(tmp_path):
    task, row, prepared = prepared_fixture(tmp_path)
    payload = json.loads(prepared.read_text())
    payload["primary_matched_budgets"]["student_opd"] = 2
    prepared.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="task-limit=2"):
        validate_run_contract(
            args_for(task, prepared, budget_mode="primary_matched", task_limit=1), [row]
        )


def test_task_rl_forbids_fake_teacher_pair(tmp_path):
    task, row, prepared = prepared_fixture(tmp_path)
    with pytest.raises(ValueError, match="no teacher coordinate"):
        validate_run_contract(
            args_for(task, prepared, mode="task_rl", student_source="M", pair_id="M_M"), [row]
        )


def test_context_bound_and_explicit_micro_are_enforced(tmp_path):
    task, row, prepared = prepared_fixture(tmp_path)
    with pytest.raises(ValueError, match="context overflow"):
        validate_run_contract(
            args_for(
                task,
                prepared,
                max_prompt_tokens=100,
                max_new_tokens=28,
                teacher_server_max_model_len=128,
            ),
            [row],
        )
    with pytest.raises(ValueError, match="micro-prompts"):
        validate_run_contract(args_for(task, prepared, micro_prompts=0), [row])


def test_scientific_path_rejects_thinking_before_gate_loading(tmp_path):
    task, row, prepared = prepared_fixture(tmp_path)
    with pytest.raises(ValueError, match="non-thinking"):
        validate_run_contract(
            args_for(task, prepared, allow_ungated_smoke=False, enable_thinking=True), [row]
        )


def test_advantage_clip_and_signal_fraction_are_bounded(tmp_path):
    task, row, prepared = prepared_fixture(tmp_path)
    with pytest.raises(ValueError, match="advantage-clip"):
        validate_run_contract(args_for(task, prepared, advantage_clip=0.0), [row])
    with pytest.raises(ValueError, match="min-informative-group-fraction"):
        validate_run_contract(
            args_for(task, prepared, min_informative_group_fraction=1.1), [row]
        )


def test_primary_student_plan_is_exact_and_predeclared():
    args = Namespace(
        advantage_clip=5.0,
        attn_implementation="sdpa",
        budget_mode="primary_matched",
        enable_thinking=False,
        gap_gate_beta=5.0,
        grad_clip=1.0,
        gradient_checkpointing=True,
        group_size=4,
        k1_coef=0.01,
        lr=1e-5,
        lora=32,
        max_new_tokens=512,
        max_prompt_tokens=1536,
        micro_prompts=1,
        min_informative_group_fraction=0.05,
        steps=100,
        seed=0,
        task_reward_coef=1.0,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
    )
    contract = validate_student_training_plan_contract(args)
    assert contract["plan_id"] == "opd_math_student_primary_pilot_v1"
    assert contract["compliant"]
    assert contract["plan_config_sha256"] == contract["actual_config_sha256"]

    args.steps = 99
    with pytest.raises(ValueError, match="optimizer_steps"):
        validate_student_training_plan_contract(args)

    args.steps = 100
    args.gradient_checkpointing = False
    with pytest.raises(ValueError, match="gradient_checkpointing"):
        validate_student_training_plan_contract(args)


def test_completion_manifest_binds_training_trace_files(tmp_path):
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    (trace_dir / "steps.jsonl").write_text('{"step":1}\n{"step":2}\n')
    (trace_dir / "samples.jsonl").write_text('{"sample":1}\n')
    run = {}
    completion = {
        "status": "completed",
        "scientific_use_allowed": False,
        "training_artifact_eligible_for_held_out_evaluation": True,
    }

    write_completion_manifests(trace_dir, run, completion)

    written = json.loads((trace_dir / "completion_manifest.json").read_text())
    assert written["trace_artifacts"]["steps.jsonl"]["rows"] == 2
    assert written["trace_artifacts"]["samples.jsonl"]["rows"] == 1
    assert len(written["trace_artifacts"]["steps.jsonl"]["sha256"]) == 64
    assert json.loads((trace_dir / "run_manifest.json").read_text())["completion"] == written


def test_generic_trace_counter_accepts_trace_rows_not_task_rows(tmp_path):
    trace = tmp_path / "steps.jsonl"
    trace.write_text('{"step":1}\n\n{"step":2,"samples":4}\n')
    assert count_jsonl_objects(trace) == 2

    trace.write_text('{"step":1}\n[]\n')
    with pytest.raises(ValueError, match="JSON object"):
        count_jsonl_objects(trace)


def test_student_trace_geometry_recomputes_exact_group_and_sample_identity(tmp_path):
    steps = tmp_path / "steps.jsonl"
    samples = tmp_path / "samples.jsonl"
    steps.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "step": 1,
                "mode": "task_rl",
                "prompts": 1,
                "samples": 2,
                "total_loss": 0.5,
                "gradient_norm_before_clip": 1.0,
            }
        )
        + "\n"
    )
    prompt_sha = "a" * 64
    completion = "answer"
    rows = []
    for sample_idx in range(2):
        rows.append(
            {
                "schema_version": 1,
                "step": 1,
                "group_id": 0,
                "sample_idx": sample_idx,
                "record_id": "M:1",
                "source": "M",
                "prompt_sha256": prompt_sha,
                "prompt_token_ids": [1, 2],
                "prompt_tokens": 2,
                "completion_token_ids": [3],
                "completion_tokens": 1,
                "completion_text": completion,
                "completion_sha256": hashlib.sha256(completion.encode()).hexdigest(),
                "terminated_by_eos": True,
                "rollout_batch_latency_seconds": 0.1,
                "teacher_scoring_latency_seconds": None,
            }
        )
    samples.write_text("".join(json.dumps(row) + "\n" for row in rows))

    class FakeTokenizer:
        def decode(self, token_ids, *, skip_special_tokens):
            assert token_ids == [3]
            assert skip_special_tokens is True
            return completion

    kwargs = {
        "steps_path": steps,
        "samples_path": samples,
        "mode": "task_rl",
        "expected_steps": 1,
        "micro_prompts": 1,
        "group_size": 2,
        "max_prompt_tokens": 8,
        "max_completion_tokens": 4,
        "expected_groups": {
            (1, 0): {
                "record_id": "M:1",
                "source": "M",
                "prompt_sha256": prompt_sha,
                "prompt_token_ids": [1, 2],
            }
        },
        "tokenizer": FakeTokenizer(),
    }
    recomputed = recompute_student_trace_geometry(**kwargs)
    assert recomputed["step_trace_rows"] == 1
    assert recomputed["sample_trace_rows"] == 2
    assert recomputed["expected_geometry_observed"] is True

    rows[1]["sample_idx"] = 0
    samples.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="missing/duplicate"):
        recompute_student_trace_geometry(**kwargs)


def test_trace_directory_rejects_reserved_output_subpaths(tmp_path):
    out = tmp_path / "run"
    assert resolve_trace_directory(out, None) == (out / "traces").resolve()
    assert resolve_trace_directory(out, tmp_path / "external") == (
        tmp_path / "external"
    ).resolve()
    for reserved in (out, out / "final", out / "final_candidate"):
        with pytest.raises(ValueError, match="disjoint"):
            resolve_trace_directory(out, reserved)


def test_gate_must_bind_the_current_prepared_and_source_manifests(tmp_path):
    task, _, prepared_path = prepared_fixture(tmp_path)
    prepared = json.loads(prepared_path.read_text())
    gate = {
        "prepared_manifest": str(prepared_path.resolve()),
        "prepared_manifest_sha256": digest(prepared_path),
        "registered_task_file": "roles/M/student_opd.jsonl",
        "registered_task_rows": 1,
        "source_manifest": prepared["source_manifest_path"],
        "source_manifest_sha256": prepared["source_manifest_sha256"],
    }
    _validate_gate_prepared_binding(
        gate,
        prepared=prepared,
        prepared_manifest_path=str(prepared_path),
        relative_task="roles/M/student_opd.jsonl",
        label="test gate",
    )
    gate["prepared_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="prepared_manifest_sha256 mismatch"):
        _validate_gate_prepared_binding(
            gate,
            prepared=prepared,
            prepared_manifest_path=str(prepared_path),
            relative_task="roles/M/student_opd.jsonl",
            label="test gate",
        )


def test_scientific_environment_contract_binds_commit_freeze_and_runtime(tmp_path, monkeypatch):
    commit = "a" * 40
    freeze_dir = tmp_path / "environment_freezes" / commit
    freeze_dir.mkdir(parents=True)
    train_freeze = freeze_dir / "train.freeze.txt"
    train_freeze.write_text(
        "".join(f"{name}=={version}\n" for name, version in EXPECTED_TRAIN_PACKAGES.items())
    )
    monkeypatch.setattr(
        "scripts.opd.opd_train.git_state",
        lambda: {"commit": commit, "dirty": False},
    )
    monkeypatch.setattr(
        "scripts.opd.opd_train.installed_package_versions",
        lambda expected: dict(expected),
    )
    environment_root = tmp_path / "train_environment"
    environment_root.mkdir()
    verification = {
        "schema_version": 1,
        "schema": "opd_math_environment_verification_v1",
        "status": "passed",
        "environment_root": str(environment_root.resolve()),
        "expected_commit": commit,
        "freeze_kind": "train",
        "commit_freeze": {
            "path": str(train_freeze.resolve()),
            "sha256": digest(train_freeze),
            "byte_identical_to_requirements_freeze": True,
        },
    }
    monkeypatch.setattr(
        "scripts.opd.opd_train.verify_live_environment",
        lambda **kwargs: dict(verification),
    )
    monkeypatch.setattr(
        "scripts.opd.opd_train.reverify_recorded_environment",
        lambda recorded, **kwargs: dict(recorded),
    )
    run_args = Namespace(
        train_environment_root=str(environment_root),
        train_environment_freeze=str(train_freeze),
        serve_environment_root=None,
        serve_environment_freeze=None,
    )
    contract = validate_environment_contract(run_args, require_serve=False)
    assert environment_contract_unchanged(contract)

    train_freeze.write_text(train_freeze.read_text() + "tampered==1\n")
    assert not environment_contract_unchanged(contract)
