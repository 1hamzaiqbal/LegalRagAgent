import hashlib
import json
from argparse import Namespace

import pytest

from scripts.opd.opd_train import (
    EXPECTED_TRAIN_PACKAGES,
    _validate_gate_prepared_binding,
    environment_contract_unchanged,
    validate_environment_contract,
    validate_run_contract,
)


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
    freeze_dir = tmp_path / commit
    freeze_dir.mkdir()
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
    run_args = Namespace(
        train_environment_freeze=str(train_freeze),
        serve_environment_freeze=None,
    )
    contract = validate_environment_contract(run_args, require_serve=False)
    assert environment_contract_unchanged(contract)

    train_freeze.write_text(train_freeze.read_text() + "tampered==1\n")
    assert not environment_contract_unchanged(contract)
