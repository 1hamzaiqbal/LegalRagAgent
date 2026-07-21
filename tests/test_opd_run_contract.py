import hashlib
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.opd import opd_train as train_module
from scripts.opd.opd_train import (
    EXPECTED_TRAIN_PACKAGES,
    _validate_gate_prepared_binding,
    count_jsonl_objects,
    environment_contract_unchanged,
    generate_student_samples,
    recompute_student_trace_geometry,
    resolve_trace_directory,
    run,
    sample_trace_rows,
    sha256_tree,
    validate_prelaunch_receipt,
    validate_student_training_plan_contract,
    validate_environment_contract,
    validate_run_contract,
    write_completion_manifests,
    bind_registered_objective,
)
from scripts.opd.trace_metrics import reconstruct_step_metrics
from scripts.opd_math.quality_gates import EVALUATION_CONTRACT, STUDENT_GATE_TYPE


def test_run_does_not_shadow_sample_trace_rows_callable():
    assert callable(sample_trace_rows)
    assert "sample_trace_rows" not in run.__code__.co_varnames
    assert "sample_trace_rows" in run.__code__.co_names
    assert "observed_sample_trace_rows" in run.__code__.co_varnames


def test_rollout_captures_exact_behavior_logprobs_after_eos_and_pad_trimming():
    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 9

        def __call__(self, prompts, **kwargs):
            assert prompts == ["first", "second"]
            assert kwargs == {
                "return_tensors": "pt",
                "padding": True,
                "add_special_tokens": False,
            }
            return {
                "input_ids": torch.tensor([[1, 2], [0, 3]]),
                "attention_mask": torch.tensor([[1, 1], [0, 1]]),
            }

        def decode(self, token_ids, *, skip_special_tokens):
            assert skip_special_tokens is True
            return ":".join(str(value) for value in token_ids)

    class FakeModel:
        def eval(self):
            return self

        def generate(self, **kwargs):
            assert kwargs["return_dict_in_generate"] is True
            assert kwargs["output_scores"] is True
            assert kwargs["num_return_sequences"] == 2
            return SimpleNamespace(
                sequences=torch.tensor(
                    [
                        [1, 2, 5, 9, 0],
                        [1, 2, 6, 7, 0],
                        [0, 3, 8, 9, 0],
                        [0, 3, 4, 0, 0],
                    ]
                ),
                scores=(object(), object(), object()),
            )

        def compute_transition_scores(self, sequences, scores, *, normalize_logits):
            assert sequences.shape == (4, 5)
            assert len(scores) == 3
            assert normalize_logits is True
            return torch.tensor(
                [
                    [-0.1, -0.2, -9.0],
                    [-0.3, -0.4, -9.0],
                    [-0.5, -0.6, -9.0],
                    [-0.7, -9.0, -9.0],
                ]
            )

    prompt_rows = [
        {
            "record_id": "O:1",
            "source": "O",
            "prompt_text": "first",
            "prompt_token_ids": [1, 2],
            "solution": r"\boxed{1}",
        },
        {
            "record_id": "O:2",
            "source": "O",
            "prompt_text": "second",
            "prompt_token_ids": [3],
            "solution": r"\boxed{2}",
        },
    ]
    args = Namespace(
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        max_new_tokens=3,
        group_size=2,
    )
    samples = generate_student_samples(
        FakeModel(), FakeTokenizer(), prompt_rows, args, "cpu"
    )

    assert [sample["completion_token_ids"] for sample in samples] == [
        [5, 9],
        [6, 7],
        [8, 9],
        [4],
    ]
    assert [sample["behavior_logprobs"] for sample in samples] == [
        pytest.approx([-0.1, -0.2]),
        pytest.approx([-0.3, -0.4]),
        pytest.approx([-0.5, -0.6]),
        pytest.approx([-0.7]),
    ]
    assert [sample["group_id"] for sample in samples] == [0, 0, 1, 1]
    assert [sample["sample_idx"] for sample in samples] == [0, 1, 0, 1]


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
        "attn_implementation": "sdpa",
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
        "campaign_run_id": None,
        "scheduler_job_id": None,
        "student": "Qwen/Qwen3-1.7B",
        "student_revision": "student-revision",
        "student_support_manifest": None,
        "teacher_gap_manifest": None,
        "teacher_provenance_manifest": None,
        "server_scoring_contract": None,
        "tokenizer_contract": None,
        "allow_ungated_smoke": True,
        "enable_thinking": False,
        "seed": 0,
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


def test_registered_one_step_task_rl_diagnostic_binds_exact_inputs_without_prelaunch(
    tmp_path, monkeypatch
):
    task, row, prepared = prepared_fixture(tmp_path)
    args = args_for(
        task,
        prepared,
        objective_id="task_rl",
        mode=None,
        objective_family_diagnostic=True,
        objective_family_prompt_plan="/custody/M_seed0.json",
        objective_family_initialization_manifest="/custody/seed0/manifest.json",
        objective_family_launcher=str(train_module.OBJECTIVE_FAMILY_LAUNCHER),
        allow_ungated_smoke=False,
        steps=1,
        max_new_tokens=512,
        max_prompt_tokens=1536,
        lora=32,
        group_size=4,
        student_revision="7" * 40,
        pair_id=None,
        student_source="M",
        budget_mode="primary_matched",
        campaign_run_id=None,
        scheduler_job_id=None,
        prelaunch_receipt=None,
        student_support_manifest="/custody/M_support.json",
        teacher_url=None,
        teacher_model=None,
        teacher_checkpoint=None,
        teacher_gap_manifest=None,
        teacher_provenance_manifest=None,
        tokenizer_contract=None,
        server_scoring_contract=None,
        serve_environment_root=None,
        serve_environment_freeze=None,
    )
    bind_registered_objective(args)
    monkeypatch.setattr(
        train_module,
        "validate_environment_contract",
        lambda selected, require_serve: {"git_commit": "c" * 40},
    )
    monkeypatch.setattr(
        train_module,
        "checked_gate",
        lambda *a, **k: {"gate": STUDENT_GATE_TYPE, "passed": True},
    )
    monkeypatch.setattr(train_module, "_validate_student_gate", lambda *a, **k: None)
    monkeypatch.setattr(train_module, "git_worktree_is_clean", lambda: True)
    monkeypatch.setattr(
        train_module, "git_state", lambda: {"commit": "c" * 40, "dirty": False}
    )
    monkeypatch.setattr(
        train_module,
        "validate_prompt_plan",
        lambda *a, **k: ({"path": "/custody/M_seed0.json"}, [row]),
    )
    monkeypatch.setattr(
        train_module,
        "validate_initialization_manifest",
        lambda *a, **k: {
            "path": "/custody/seed0/manifest.json",
            "adapter_path": "/custody/seed0/adapter",
            "trainable_parameter_signature": {
                "elements": 1,
                "sum": 0.0,
                "squared_l2": 0.0,
            },
        },
    )
    result = validate_run_contract(args, [row])
    binding = result[-1]
    assert binding["objective_family_diagnostic"] is True
    assert binding["student_source"] == "M"
    assert binding["teacher_source"] is None
    assert args.objective_family_ordered_rows == [row]


def test_registered_one_step_bare_k1_diagnostic_routes_only_through_o_teacher(
    tmp_path, monkeypatch
):
    task, row, prepared_path = prepared_fixture(tmp_path)
    prepared = json.loads(prepared_path.read_text())
    o_skill = tmp_path / "roles" / "O" / "teacher_gap_dev.jsonl"
    o_skill.parent.mkdir(parents=True)
    o_skill.write_text("{}\n")
    prepared["files"]["roles/O/teacher_gap_dev.jsonl"] = {
        "rows": 1,
        "sha256": digest(o_skill),
    }
    prepared["pairs"].append(
        {
            "id": "O_M",
            "teacher_source": "O",
            "opd_source": "M",
            "student_opd_file": "roles/M/student_opd.jsonl",
            "teacher_skill_dev_file": "roles/O/teacher_gap_dev.jsonl",
        }
    )
    prepared_path.write_text(json.dumps(prepared))
    args = args_for(
        task,
        prepared_path,
        objective_id="k1_bare_verl_compatible_clip10",
        mode=None,
        objective_family_diagnostic=True,
        objective_family_prompt_plan="/custody/M_seed0.json",
        objective_family_initialization_manifest="/custody/seed0/manifest.json",
        objective_family_launcher=str(train_module.OBJECTIVE_FAMILY_LAUNCHER),
        allow_ungated_smoke=False,
        steps=1,
        max_new_tokens=512,
        max_prompt_tokens=1536,
        lora=32,
        group_size=4,
        student_revision="7" * 40,
        pair_id=None,
        student_source="M",
        budget_mode="primary_matched",
        campaign_run_id=None,
        scheduler_job_id=None,
        prelaunch_receipt=None,
        student_support_manifest="/custody/M_support.json",
        teacher_url="http://127.0.0.1:8000",
        teacher_model="opd-math-teacher",
        teacher_checkpoint="/custody/O_teacher",
        teacher_base_model="Qwen/Qwen3-8B",
        teacher_base_revision="8" * 40,
        teacher_gap_manifest="/custody/O_gap.json",
        teacher_provenance_manifest="/custody/O_teacher/merge_provenance.json",
        tokenizer_contract="/custody/tokenizer.json",
        server_scoring_contract="/custody/server.json",
        serve_environment_root="/custody/serve_env",
        serve_environment_freeze="/custody/serve.freeze.txt",
    )
    bind_registered_objective(args)
    monkeypatch.setattr(
        train_module,
        "validate_environment_contract",
        lambda selected, require_serve: {
            "git_commit": "c" * 40,
            "serve_verification": {},
        },
    )
    monkeypatch.setattr(
        train_module,
        "checked_gate",
        lambda *a, expected_gate, **k: {
            "gate": expected_gate,
            "passed": True,
            **(
                {
                    "base_model": "Qwen/Qwen3-8B",
                    "base_model_revision": "8" * 40,
                }
                if expected_gate == "teacher_gap_v1"
                else {}
            ),
        },
    )
    monkeypatch.setattr(train_module, "_validate_student_gate", lambda *a, **k: None)
    monkeypatch.setattr(train_module, "_validate_teacher_gate", lambda *a, **k: None)
    monkeypatch.setattr(train_module, "_validate_tokenizer_contract", lambda *a, **k: None)
    monkeypatch.setattr(
        train_module,
        "_validate_server_scoring_contract",
        lambda *a, **k: {
            "local_process_binding": {
                "teacher_checkpoint_tree_sha256": "t" * 64,
                "teacher_provenance_manifest_sha256": "p" * 64,
            }
        },
    )
    monkeypatch.setattr(
        train_module,
        "_validate_teacher_provenance",
        lambda *a, **k: {
            "output_checkpoint_tree_sha256": "t" * 64,
            "manifest_sha256": "p" * 64,
        },
    )
    monkeypatch.setattr(
        train_module, "validate_server_environment_process_binding", lambda *a, **k: None
    )
    monkeypatch.setattr(train_module, "git_worktree_is_clean", lambda: True)
    monkeypatch.setattr(
        train_module, "git_state", lambda: {"commit": "c" * 40, "dirty": False}
    )
    monkeypatch.setattr(
        train_module,
        "validate_prompt_plan",
        lambda *a, **k: ({"path": "/custody/M_seed0.json"}, [row]),
    )
    monkeypatch.setattr(
        train_module,
        "validate_initialization_manifest",
        lambda *a, **k: {
            "path": "/custody/seed0/manifest.json",
            "adapter_path": "/custody/seed0/adapter",
            "trainable_parameter_signature": {
                "elements": 1,
                "sum": 0.0,
                "squared_l2": 0.0,
            },
        },
    )
    result = validate_run_contract(args, [row])
    binding = result[-1]
    assert binding["pair_id"] == "O_M"
    assert binding["teacher_source"] == "O"
    assert binding["local_checkpoint_custody_validated"] is True


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


def test_scientific_primary_requires_stable_run_and_scheduler_ids(tmp_path):
    task, row, prepared = prepared_fixture(tmp_path)
    with pytest.raises(ValueError, match="campaign-run-id"):
        validate_run_contract(
            args_for(
                task,
                prepared,
                allow_ungated_smoke=False,
                budget_mode="primary_matched",
                steps=100,
                max_new_tokens=512,
                max_prompt_tokens=1536,
                lora=32,
                group_size=4,
                student_revision="7" * 40,
                pair_id="O_M",
            ),
            [row],
        )


def test_prelaunch_receipt_revalidates_teacher_provenance_before_training(
    tmp_path, monkeypatch
):
    commit = "c" * 40
    checkpoint = tmp_path / "teacher"
    checkpoint.mkdir()
    (checkpoint / "adapter.bin").write_bytes(b"teacher")
    provenance_path = checkpoint / "merge_provenance.json"
    provenance = {"schema_version": 1, "status": "completed"}
    provenance_path.write_text(json.dumps(provenance) + "\n")

    teacher_gate = tmp_path / "teacher_gap.json"
    teacher_gate.write_text('{"gate":"teacher_gap_v1","passed":true}\n')
    support = tmp_path / "student_support.json"
    support.write_text('{"gate":"student_support_v1","passed":true}\n')
    preregistration = tmp_path / "preregistration.json"
    preregistration.write_text('{"preregistration":"sealed"}\n')
    ledger = tmp_path / "launch_ledger.json"
    ledger.write_text('{"ledger":"sealed"}\n')

    out_dir = tmp_path / "run"
    receipt_path = tmp_path / "run.prelaunch.json"
    receipt = {
        "schema_version": 1,
        "receipt": train_module.STUDENT_PRELAUNCH_RECEIPT,
        "sealed_before_optimizer_start": True,
        "campaign_id": "campaign-1",
        "run_key": "O_M",
        "run_id": "run-1",
        "scheduler_job_id": "123",
        "mode": "task_rl_k1_gap",
        "student_source": "M",
        "git_commit": commit,
        "out_dir": str(out_dir.resolve()),
        "expected_artifacts": {
            "run_manifest": str((out_dir / "traces" / "run_manifest.json").resolve()),
            "student_completion_manifest": str(
                (out_dir / "traces" / "completion_manifest.json").resolve()
            ),
            "student_adapter": str((out_dir / "final").resolve()),
            "prelaunch_receipt": str(receipt_path.resolve()),
        },
        "student_support": {"manifest_sha256": digest(support)},
        "o_teacher": {
            "teacher_gap_manifest": str(teacher_gate.resolve()),
            "teacher_gap_manifest_sha256": digest(teacher_gate),
            "merged_checkpoint": str(checkpoint.resolve()),
            "merged_checkpoint_tree_sha256": sha256_tree(
                checkpoint, exclude_relative_paths=("merge_provenance.json",)
            ),
            "merge_provenance_manifest_sha256": digest(provenance_path),
            "merge_provenance_payload_sha256": train_module.canonical_json_sha256(
                provenance
            ),
        },
        "preregistration": {
            "path": str(preregistration.resolve()),
            "sha256": digest(preregistration),
        },
        "launch_ledger": {
            "path": str(ledger.resolve()),
            "sha256": digest(ledger),
        },
    }
    receipt_path.write_text(json.dumps(receipt) + "\n")
    receipt_path.chmod(0o444)
    args = Namespace(
        mode="task_rl_k1_gap",
        pair_id="O_M",
        student_source=None,
        campaign_run_id="run-1",
        scheduler_job_id="123",
        out_dir=str(out_dir),
        prelaunch_receipt=str(receipt_path),
        student_support_manifest=str(support),
        teacher_gap_manifest=str(teacher_gate),
        teacher_checkpoint=str(checkpoint),
        teacher_provenance_manifest=str(provenance_path),
    )
    monkeypatch.setattr(
        train_module, "git_state", lambda: {"commit": commit, "dirty": False}
    )

    binding = validate_prelaunch_receipt(args)
    assert binding["sealed_before_optimizer_start"] is True

    provenance_path.write_text(
        json.dumps({**provenance, "tampered_after_receipt": True}) + "\n"
    )
    with pytest.raises(ValueError, match="merge_provenance_manifest_sha256 mismatch"):
        validate_prelaunch_receipt(args)


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
            args_for(
                task,
                prepared,
                allow_ungated_smoke=False,
                enable_thinking=True,
                pair_id="O_M",
            ),
            [row],
        )


def test_scientific_path_rejects_immutable_failed_m_teacher_arms(tmp_path):
    task, row, prepared = prepared_fixture(tmp_path)
    with pytest.raises(ValueError, match="M_M/M_O are prohibited"):
        validate_run_contract(
            args_for(task, prepared, allow_ungated_smoke=False, pair_id="M_M"),
            [row],
        )


def test_scientific_support_gate_must_share_training_commit_and_environment(
    tmp_path, monkeypatch
):
    task, _, prepared_path = prepared_fixture(tmp_path)
    prepared = json.loads(prepared_path.read_text())
    args = args_for(task, prepared_path, allow_ungated_smoke=False)
    commit = "c" * 40
    verifier = {"path": "/repo/verify_environment.py", "sha256": "1" * 64}
    freeze = {"path": "/freeze/train.freeze.txt", "sha256": "2" * 64}
    verification = {"identity": "same-live-environment"}
    current_environment = {
        "git_commit": commit,
        "verifier": verifier,
        "train_freeze": freeze,
        "train_verification": verification,
    }
    source_path = Path(prepared["source_manifest_path"])
    gate = {
        "schema_version": 3,
        "gate": STUDENT_GATE_TYPE,
        "gate_strength": "scientific",
        "passed": True,
        "authorizes_scientific_training": True,
        "student_model": args.student,
        "student_model_revision": args.student_revision,
        "task_file_sha256": digest(task),
        "task_sources": ["M"],
        "task_roles": ["student_opd"],
        "prepared_manifest": str(prepared_path.resolve()),
        "prepared_manifest_sha256": digest(prepared_path),
        "registered_task_file": "roles/M/student_opd.jsonl",
        "registered_task_rows": 1,
        "source_manifest": str(source_path.resolve()),
        "source_manifest_sha256": digest(source_path),
        "primary_matched_role_budget": 1,
        "pinned_model_kind": "student",
        "pinned_model": args.student,
        "pinned_model_revision": args.student_revision,
        "samples_per_problem": args.group_size,
        "decoding": {
            "thinking": False,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "seed": 0,
        },
        "evaluation_contract": EVALUATION_CONTRACT,
        "evaluation_git_commit": commit,
        "evaluation_environment": {
            "verifier": verifier,
            "train_freeze": freeze,
            "train_verification": verification,
        },
        "evaluation_post_promotion_custody": {"sha256": "3" * 64},
    }
    args.seed = 0
    monkeypatch.setattr(
        train_module,
        "_validate_deterministic_gate_recomputation",
        lambda gate, *, kind: None,
    )

    train_module._validate_student_gate(
        gate,
        args=args,
        task_hash=digest(task),
        student_source="M",
        prepared=prepared,
        current_environment=current_environment,
    )

    gate["evaluation_git_commit"] = "d" * 40
    with pytest.raises(ValueError, match="current training commit"):
        train_module._validate_student_gate(
            gate,
            args=args,
            task_hash=digest(task),
            student_source="M",
            prepared=prepared,
            current_environment=current_environment,
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
                "total_loss": 0.0,
                "gradient_norm_before_clip": 1.0,
                "task_loss": 0.0,
                "reverse_kl_score_function_surrogate": 0.0,
                "sampled_k1_estimate": None,
                "gap_gate_mean": None,
                "positive_gap_fraction": None,
                "reward_mean": 0.5,
                "informative_group_fraction": 1.0,
                "tokens": 2,
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
                "schema_version": 2,
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
                "student_token_logprobs": [-1.0],
                "teacher_token_logprobs_on_student_trajectory": None,
                "student_nll": 1.0,
                "terminated_by_eos": True,
                "rollout_batch_latency_seconds": 0.1,
                "teacher_scoring_latency_seconds": None,
                "reward": float(sample_idx == 0),
                "reward_status": "correct" if sample_idx == 0 else "incorrect",
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

    with pytest.raises(ValueError, match="parameter_update_l2"):
        recompute_student_trace_geometry(**kwargs, require_behavior_logprobs=True)

    rows[1]["sample_idx"] = 0
    samples.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="missing/duplicate"):
        recompute_student_trace_geometry(**kwargs)


def test_main_arm_schema2_reconstructs_teacher_arrays_and_all_step_metrics(tmp_path):
    steps = tmp_path / "steps.jsonl"
    samples = tmp_path / "samples.jsonl"
    prompt_sha = "b" * 64
    student_values = [-2.0, -1.0, -0.5, -3.0]
    teacher_values = [-1.0, -2.0, -0.4, -3.5]
    rewards = [1.0, 0.0, 1.0, 0.0]
    rows = []
    for sample_idx, (student, teacher, reward) in enumerate(
        zip(student_values, teacher_values, rewards, strict=True)
    ):
        completion = f"answer-{sample_idx}"
        gap = teacher - student
        rows.append(
            {
                "schema_version": 2,
                "step": 1,
                "group_id": 0,
                "sample_idx": sample_idx,
                "record_id": "M:main:1",
                "source": "M",
                "prompt_sha256": prompt_sha,
                "prompt_token_ids": [1, 2],
                "prompt_tokens": 2,
                "completion_token_ids": [10 + sample_idx],
                "completion_tokens": 1,
                "completion_text": completion,
                "completion_sha256": hashlib.sha256(completion.encode()).hexdigest(),
                "student_token_logprobs": [student],
                "teacher_token_logprobs_on_student_trajectory": [teacher],
                "student_nll": -student,
                "teacher_nll_on_student_trajectory": -teacher,
                "mean_teacher_student_gap": gap,
                "mean_abs_k1_log_ratio": abs(gap),
                "min_teacher_student_gap": gap,
                "max_teacher_student_gap": gap,
                "positive_teacher_gap_fraction": float(gap > 0),
                "reward": reward,
                "reward_status": "correct" if reward else "incorrect",
                "terminated_by_eos": True,
                "rollout_batch_latency_seconds": 0.1,
                "teacher_scoring_latency_seconds": 0.01,
            }
        )
    loss_config = {
        "task_reward_coef": 1.0,
        "k1_coef": 0.01,
        "gap_gate_beta": 5.0,
        "advantage_clip": 5.0,
    }
    reconstructed = reconstruct_step_metrics(
        rows,
        mode="task_rl_k1_gap",
        **loss_config,
    )
    assert reconstructed["tokens"] == 4
    assert reconstructed["sampled_k1_estimate"] == pytest.approx(0.1)
    assert reconstructed["positive_gap_fraction"] == 0.5
    assert reconstructed["reward_mean"] == 0.5
    assert reconstructed["informative_group_fraction"] == 1.0
    step_row = {
        "schema_version": 1,
        "step": 1,
        "mode": "task_rl_k1_gap",
        "prompts": 1,
        "samples": 4,
        "gradient_norm_before_clip": 1.0,
        **reconstructed,
    }
    steps.write_text(json.dumps(step_row) + "\n")
    samples.write_text("".join(json.dumps(row) + "\n" for row in rows))

    class FakeTokenizer:
        def decode(self, token_ids, *, skip_special_tokens):
            assert skip_special_tokens is True
            return f"answer-{token_ids[0] - 10}"

    kwargs = {
        "steps_path": steps,
        "samples_path": samples,
        "mode": "task_rl_k1_gap",
        "expected_steps": 1,
        "micro_prompts": 1,
        "group_size": 4,
        "max_prompt_tokens": 8,
        "max_completion_tokens": 4,
        "expected_groups": {
            (1, 0): {
                "record_id": "M:main:1",
                "source": "M",
                "prompt_sha256": prompt_sha,
                "prompt_token_ids": [1, 2],
            }
        },
        "tokenizer": FakeTokenizer(),
        "loss_config": loss_config,
    }
    recomputed = recompute_student_trace_geometry(**kwargs)
    assert recomputed["expected_geometry_observed"] is True

    # A coherent per-sample rewrite must still fail if the step-level OPD
    # surrogate/K1/gate/total account is stale.
    rows[0]["teacher_token_logprobs_on_student_trajectory"] = [-4.0]
    gap = -4.0 - student_values[0]
    rows[0].update(
        {
            "teacher_nll_on_student_trajectory": 4.0,
            "mean_teacher_student_gap": gap,
            "mean_abs_k1_log_ratio": abs(gap),
            "min_teacher_student_gap": gap,
            "max_teacher_student_gap": gap,
            "positive_teacher_gap_fraction": float(gap > 0),
        }
    )
    samples.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="differs from trace reconstruction"):
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
