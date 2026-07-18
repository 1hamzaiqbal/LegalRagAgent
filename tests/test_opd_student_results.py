import hashlib
import json
from argparse import Namespace
from pathlib import Path

import pytest

from tests.opd_evaluation_fixture import write_merged_evaluation

from scripts.opd_math import quality_gates
from scripts.opd_math import student_results as results


STUDENT = "Qwen/Qwen3-1.7B"
STUDENT_REVISION = "7" * 40
COMMIT = "c" * 40


@pytest.fixture(autouse=True)
def clean_result_builder(monkeypatch):
    monkeypatch.setattr(
        results, "git_state", lambda: {"commit": COMMIT, "dirty": False}
    )
    monkeypatch.setattr(
        results,
        "reverify_recorded_environment",
        lambda recorded, **kwargs: dict(recorded),
    )


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def fake_verify(completion, _gold):
    correct = r"\boxed{1}" in completion
    return {"reward": float(correct), "status": "correct" if correct else "incorrect"}


def plan_binding():
    plan = json.loads(results.CANONICAL_STUDENT_TRAINING_PLAN.read_text())
    fixed = plan["fixed_config"]
    digest = results.canonical_json_sha256(fixed)
    return {
        "path": str(results.CANONICAL_STUDENT_TRAINING_PLAN.resolve()),
        "sha256": results.sha256_file(results.CANONICAL_STUDENT_TRAINING_PLAN),
        "plan_id": plan["plan_id"],
        "plan_config_sha256": digest,
        "actual_config_sha256": digest,
        "config": fixed,
        "compliant": True,
    }


def write_freeze(path, packages):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{name}=={version}\n" for name, version in packages.items()))
    return {
        "path": str(path.resolve()),
        "sha256": results.sha256_file(path),
        "required_packages": packages,
    }


def write_environment_verification(root, freeze, *, kind):
    root = root.resolve()
    bin_dir = root / "bin"
    bin_dir.mkdir(parents=True)
    python = bin_dir / "python"
    python.write_bytes(b"python")
    python.chmod(0o755)
    requirements = root / "requirements.freeze.txt"
    requirements.write_bytes(Path(freeze["path"]).read_bytes())
    executable = None
    if kind == "serve":
        vllm = bin_dir / "vllm"
        vllm.write_text(f"#!{python}\n")
        vllm.chmod(0o755)
        executable = {
            "path": str(vllm),
            "sha256": results.sha256_file(vllm),
            "shebang": f"#!{python}",
        }
    return {
        "schema_version": 1,
        "schema": results.ENVIRONMENT_VERIFICATION_SCHEMA,
        "status": "passed",
        "environment_root": str(root),
        "live_python": str(python),
        "expected_commit": COMMIT,
        "freeze_kind": kind,
        "installed_distribution_count": len(freeze["required_packages"]),
        "installed_distribution_map_sha256": "a" * 64,
        "requirements_freeze": {
            "path": str(requirements),
            "sha256": results.sha256_file(requirements),
        },
        "commit_freeze": {
            "path": freeze["path"],
            "sha256": freeze["sha256"],
            "byte_identical_to_requirements_freeze": True,
        },
        "expected_executable": executable,
    }


def prepared_fixture(tmp_path, source="M"):
    source_manifest = tmp_path / "source_manifest.json"
    source_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "models": {
                    "student": {"id": STUDENT, "revision": STUDENT_REVISION},
                    "teacher": {"id": "Qwen/Qwen3-8B", "revision": "8" * 40},
                },
            },
            sort_keys=True,
        )
        + "\n"
    )
    train = tmp_path / "roles" / source / "student_opd.jsonl"
    holdout = tmp_path / "roles" / source / "source_holdout.jsonl"
    train_rows = [
        {
            "record_id": f"{source}:train:{index}",
            "cluster_id": f"cluster:{source}:train:{index}",
            "source": source,
            "role": "student_opd",
            "prompt": [{"role": "user", "content": f"problem {index}"}],
            "solution": r"\boxed{1}",
        }
        for index in range(2)
    ]
    holdout_rows = [
        {
            "record_id": f"{source}:holdout:{index}",
            "cluster_id": f"cluster:{source}:holdout:{index}",
            "source": source,
            "role": "source_holdout",
            "solution": r"\boxed{1}",
        }
        for index in range(4)
    ]
    write_jsonl(train, train_rows)
    write_jsonl(holdout, holdout_rows)
    prepared = {
        "schema_version": 1,
        "scientific_use_allowed": True,
        "code_git_state": {"commit": "d" * 40, "dirty": False},
        "source_manifest_path": str(source_manifest.resolve()),
        "source_manifest_sha256": results.sha256_file(source_manifest),
        "primary_matched_budgets": {
            "student_opd": len(train_rows),
            "source_holdout": len(holdout_rows),
        },
        "files": {
            f"roles/{source}/student_opd.jsonl": {
                "rows": len(train_rows),
                "sha256": results.sha256_file(train),
            },
            f"roles/{source}/source_holdout.jsonl": {
                "rows": len(holdout_rows),
                "sha256": results.sha256_file(holdout),
            },
        },
    }
    manifest = tmp_path / "prepared_manifest.json"
    manifest.write_text(json.dumps(prepared, indent=2, sort_keys=True) + "\n")
    return manifest, prepared, train, train_rows, holdout, holdout_rows


def support_gate(prepared_path, train, source="M"):
    return {
        "schema_version": 3,
        "gate": "student_support_v1",
        "gate_strength": "scientific",
        "passed": True,
        "authorizes_scientific_training": True,
        "student_model": STUDENT,
        "student_model_revision": STUDENT_REVISION,
        "task_sources": [source],
        "task_roles": ["student_opd"],
        "task_file_sha256": results.sha256_file(train),
        "prepared_manifest": str(prepared_path.resolve()),
        "prepared_manifest_sha256": results.sha256_file(prepared_path),
        "decoding": results.HELDOUT_DECODING,
        "samples_per_problem": 4,
        "manifest_sha256": "a" * 64,
    }


def student_run_fixture(tmp_path, prepared_path, prepared, train, train_rows, adapter, source="M"):
    trace_dir = tmp_path / "student" / "traces"
    trace_dir.mkdir(parents=True)
    fixed = plan_binding()["config"]
    steps = []
    samples = []
    realized = []
    realized_prompts = []
    for step in range(1, fixed["optimizer_steps"] + 1):
        record = train_rows[(step - 1) % len(train_rows)]
        realized.append(record["record_id"])
        realized_prompts.append(
            {
                "record_id": record["record_id"],
                "prompt_sha256": results._task_prompt_sha256(record),
            }
        )
        steps.append(
            {
                "schema_version": 1,
                "step": step,
                "mode": "task_rl",
                "prompts": 1,
                "samples": 4,
                "total_loss": 0.25,
                "gradient_norm_before_clip": 0.5,
            }
        )
        for sample_idx in range(4):
            completion = (
                r"Final answer: \boxed{1}."
                if sample_idx == 0
                else r"Final answer: \boxed{999}."
            )
            verdict = fake_verify(completion, record["solution"])
            samples.append(
                {
                    "schema_version": 1,
                    "step": step,
                    "record_id": record["record_id"],
                    "source": source,
                    "group_id": 0,
                    "sample_idx": sample_idx,
                    "completion_tokens": 1,
                    "prompt_tokens": 2,
                    "terminated_by_eos": True,
                    "rollout_batch_latency_seconds": 0.1,
                    "teacher_scoring_latency_seconds": None,
                    "completion_sha256": hashlib.sha256(completion.encode()).hexdigest(),
                    "prompt_sha256": results._task_prompt_sha256(record),
                    "prompt_token_ids": [1, 2],
                    "completion_token_ids": [3],
                    "completion_text": completion,
                    "student_nll": 1.0,
                    "teacher_nll_on_student_trajectory": None,
                    "mean_teacher_student_gap": None,
                    "mean_abs_k1_log_ratio": None,
                    "min_teacher_student_gap": None,
                    "max_teacher_student_gap": None,
                    "positive_teacher_gap_fraction": None,
                    "reward": verdict["reward"],
                    "reward_status": verdict["status"],
                }
            )
    steps_path = trace_dir / "steps.jsonl"
    samples_path = trace_dir / "samples.jsonl"
    write_jsonl(steps_path, steps)
    write_jsonl(samples_path, samples)
    state = {"commit": COMMIT, "dirty": False}
    train_freeze = write_freeze(
        tmp_path / "environment_freezes" / COMMIT / "train.freeze.txt",
        results.EXPECTED_TRAIN_PACKAGES,
    )
    train_verification = write_environment_verification(
        tmp_path / "environments" / "train", train_freeze, kind="train"
    )
    environment = {
        "schema_version": 2,
        "git_commit": COMMIT,
        "verifier": {
            "path": str(results.ENVIRONMENT_VERIFIER.resolve()),
            "sha256": results.sha256_file(results.ENVIRONMENT_VERIFIER),
        },
        "train_runtime_packages": results.EXPECTED_TRAIN_PACKAGES,
        "train_freeze": train_freeze,
        "train_verification": train_verification,
        "serve_freeze": None,
        "serve_verification": None,
    }
    adapter_hash = results.sha256_tree(adapter)
    completion = {
        "schema_version": 1,
        "status": "completed",
        "objective": "task_rl",
        "optimizer_steps_completed": 100,
        "rollout_samples": 400,
        "scored_completion_tokens": 400,
        "prompt_group_tokens": 200,
        "sample_expanded_prompt_tokens": 800,
        "prompt_groups_seen": 100,
        "step_trace_rows": 100,
        "sample_trace_rows": 400,
        "realized_training_geometry_observed": True,
        "unique_training_records": 2,
        "realized_record_ids_sha256": results.canonical_json_sha256(realized),
        "realized_prompt_sequence_sha256": results.canonical_json_sha256(
            realized_prompts
        ),
        "total_training_elapsed_seconds": 1.0,
        "total_rollout_latency_seconds": 10.0,
        "total_teacher_scoring_latency_seconds": 0.0,
        "peak_cuda_memory_bytes": 0,
        "intended_scientific_run": True,
        "informative_task_steps": 100,
        "informative_task_groups": 100,
        "total_task_groups": 100,
        "informative_group_fraction": 1.0,
        "minimum_informative_group_fraction": 0.05,
        "task_signal_observed": True,
        "finite_nonzero_gradient_observed": True,
        "parameter_update_observed": True,
        "git_state_start": state,
        "git_state_training_end": state,
        "git_state_after_candidate_save": state,
        "git_state_end": state,
        "clean_stable_code": True,
        "stable_training_environment": True,
        "stable_environment_after_candidate_save": True,
        "stable_environment_end": True,
        "stable_final_artifact_hash": True,
        "local_server_process_binding_required": False,
        "live_local_server_process_binding_validated": False,
        "initial_parameter_signature": {"sum": 0.0},
        "final_parameter_signature": {"sum": 1.0},
        "training_artifact_eligible_for_held_out_evaluation": True,
        "scientific_use_allowed": False,
        "final_adapter": str(adapter.resolve()),
        "final_adapter_tree_sha256": adapter_hash,
        "trace_artifacts": {
            "steps.jsonl": {
                "path": str(steps_path.resolve()),
                "rows": 100,
                "sha256": results.sha256_file(steps_path),
            },
            "samples.jsonl": {
                "path": str(samples_path.resolve()),
                "rows": 400,
                "sha256": results.sha256_file(samples_path),
            },
        },
    }
    train_relative = f"roles/{source}/student_opd.jsonl"
    support = support_gate(prepared_path, train, source)
    binding = {
        "pair_id": None,
        "student_source": source,
        "teacher_source": None,
        "budget_mode": "primary_matched",
        "task_role_file": train_relative,
        "task_file_rows": len(train_rows),
        "matched_task_limit": len(train_rows),
        "local_checkpoint_custody_validated": False,
        "server_alias_and_token_contract_validated": False,
        "live_local_server_process_binding_validated": False,
        "serve_environment_process_binding_validated": False,
        "server_binding_claim_boundary": "test",
        "environment_contract": environment,
        "student_training_plan": plan_binding(),
    }
    run = {
        "schema_version": 1,
        "objective": "task_rl",
        "objective_contract": "grouped_verifiable_math_task_reward_v1",
        "status": "completed",
        "intended_scientific_run": True,
        "scientific_use_allowed": False,
        "training_artifact_eligible_for_held_out_evaluation": True,
        "git_commit": COMMIT,
        "git_worktree_clean": True,
        "git_state_start": state,
        "task_file": str(train.resolve()),
        "task_file_sha256": results.sha256_file(train),
        "selected_task_rows": len(train_rows),
        "task_limit": len(train_rows),
        "binding": binding,
        "student": STUDENT,
        "student_revision": STUDENT_REVISION,
        "teacher_model": None,
        "teacher_checkpoint": None,
        "teacher_base_model": None,
        "teacher_base_revision": None,
        "optimizer_steps_planned": 100,
        "micro_prompts_per_step": 1,
        "planned_rollout_samples": 400,
        "seed": 0,
        "optimization": {
            "attn_implementation": fixed["attn_implementation"],
            "gradient_checkpointing": fixed["gradient_checkpointing"],
            "learning_rate": fixed["learning_rate"],
            "lora_r": fixed["lora_r"],
        },
        "generation": {
            "group_size": 4,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "max_new_tokens": 512,
            "enable_thinking": False,
        },
        "loss": {
            "task_reward_coef": 1.0,
            "k1_coef": 0.01,
            "gap_gate_beta": 5.0,
            "advantage_clip": 5.0,
        },
        "gates": {
            "prepared_data": {
                "path": str(prepared_path.resolve()),
                "sha256": results.sha256_file(prepared_path),
                "task_role_file": train_relative,
                "task_file_sha256": results.sha256_file(train),
                "scientific_use_allowed": True,
            },
            "teacher_gap": None,
            "teacher_provenance": None,
            "server_scoring_contract": None,
            "student_support": support,
            "tokenizer_contract": None,
        },
        "completion": completion,
    }
    run_path = trace_dir / "run_manifest.json"
    completion_path = trace_dir / "completion_manifest.json"
    run_path.write_text(json.dumps(run, indent=2, sort_keys=True) + "\n")
    completion_path.write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n")
    return run_path, completion_path


def evaluation_fixture(tmp_path, holdout, holdout_rows, adapter, source="M", rewards=None):
    rewards = rewards or [[0, 0, 0, 0] for _ in holdout_rows]
    return write_merged_evaluation(
        tmp_path,
        "heldout",
        holdout,
        {
            row["record_id"]: [float(value) for value in row_rewards]
            for row, row_rewards in zip(holdout_rows, rewards)
        },
        model=STUDENT,
        revision=STUDENT_REVISION,
        adapter=adapter,
        packages=quality_gates.EXPECTED_EVALUATION_PACKAGES,
        git_commit=COMMIT,
        tokenizer_contract_sha256="b" * 64,
        decoding=results.HELDOUT_DECODING,
    )


def heldout_args(tmp_path):
    prepared_path, prepared, train, train_rows, holdout, holdout_rows = prepared_fixture(tmp_path)
    adapter = tmp_path / "student" / "final"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text('{"r":32}\n')
    run, completion = student_run_fixture(
        tmp_path, prepared_path, prepared, train, train_rows, adapter
    )
    summary, samples = evaluation_fixture(tmp_path, holdout, holdout_rows, adapter)
    return Namespace(
        matrix_key="baseline_M",
        student_run_manifest=run,
        student_completion_manifest=completion,
        student_summary=summary,
        student_samples=samples,
        trained_adapter=adapter,
        prepared_manifest=prepared_path,
        student_model=STUDENT,
        student_revision=STUDENT_REVISION,
        task_source="M",
    )


def test_heldout_gate_rejects_legacy_monolithic_evaluation(tmp_path):
    args = heldout_args(tmp_path)
    payload = json.loads(Path(args.student_summary).read_text())
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
    legacy = Path(args.student_summary).parent / "legacy-summary.json"
    legacy.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    args.student_summary = legacy
    with pytest.raises(ValueError, match="requires a schema-v2 merged artifact"):
        results.student_heldout_result(args)


def test_zero_accuracy_result_is_authorized_and_deterministically_recomputed(tmp_path, monkeypatch):
    monkeypatch.setattr(results, "verify_completion", fake_verify)
    monkeypatch.setattr(quality_gates, "verify_completion", fake_verify)
    monkeypatch.setattr(results, "recompute_student_gate", lambda gate: gate)
    args = heldout_args(tmp_path)

    gate = results.student_heldout_result(args)

    assert gate["passed"]
    assert gate["authorizes_scientific_matrix_readout"]
    assert gate["authorization_is_independent_of_effect_sign"]
    assert gate["accuracy"] == 0.0
    assert gate["records"] == 4
    assert gate["student_run_binding"]["trace"]["samples"] == 400
    assert results.recompute_student_heldout_result(gate) == gate


def test_heldout_gate_requires_clean_matching_result_builder(tmp_path, monkeypatch):
    monkeypatch.setattr(results, "verify_completion", fake_verify)
    monkeypatch.setattr(quality_gates, "verify_completion", fake_verify)
    monkeypatch.setattr(results, "recompute_student_gate", lambda gate: gate)
    args = heldout_args(tmp_path)
    monkeypatch.setattr(
        results, "git_state", lambda: {"commit": COMMIT, "dirty": True}
    )

    with pytest.raises(ValueError, match="student result builder dirty mismatch"):
        results.student_heldout_result(args)


def test_heldout_gate_rejects_adapter_or_trace_drift(tmp_path, monkeypatch):
    monkeypatch.setattr(results, "verify_completion", fake_verify)
    monkeypatch.setattr(quality_gates, "verify_completion", fake_verify)
    monkeypatch.setattr(results, "recompute_student_gate", lambda gate: gate)
    args = heldout_args(tmp_path)
    gate = results.student_heldout_result(args)

    (args.trained_adapter / "adapter_config.json").write_text('{"r":64}\n')
    with pytest.raises(ValueError, match="identity drifted"):
        results.recompute_student_heldout_result(gate)

    (args.trained_adapter / "adapter_config.json").write_text('{"r":32}\n')
    sample_trace = args.student_completion_manifest.parent / "samples.jsonl"
    sample_trace.write_text(sample_trace.read_text() + "{}\n")
    with pytest.raises(ValueError, match="hash has drifted|identity drifted"):
        results.student_heldout_result(args)


def test_heldout_gate_recomputes_and_requires_realized_training_geometry(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(results, "verify_completion", fake_verify)
    monkeypatch.setattr(quality_gates, "verify_completion", fake_verify)
    monkeypatch.setattr(results, "recompute_student_gate", lambda gate: gate)
    args = heldout_args(tmp_path)
    completion = json.loads(args.student_completion_manifest.read_text())
    completion["realized_training_geometry_observed"] = False
    args.student_completion_manifest.write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n"
    )
    run = json.loads(args.student_run_manifest.read_text())
    run["completion"] = completion
    args.student_run_manifest.write_text(json.dumps(run, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="realized_training_geometry_observed"):
        results.student_heldout_result(args)


@pytest.mark.parametrize("mutation", ["missing", "disabled"])
def test_heldout_gate_requires_registered_checkpointing_record(
    tmp_path, monkeypatch, mutation
):
    monkeypatch.setattr(results, "verify_completion", fake_verify)
    monkeypatch.setattr(quality_gates, "verify_completion", fake_verify)
    monkeypatch.setattr(results, "recompute_student_gate", lambda gate: gate)
    args = heldout_args(tmp_path)
    run = json.loads(args.student_run_manifest.read_text())
    if mutation == "missing":
        run.pop("optimization")
    else:
        run["optimization"]["gradient_checkpointing"] = False
    args.student_run_manifest.write_text(json.dumps(run, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="student run optimization mismatch"):
        results.student_heldout_result(args)


def test_teacher_identity_binds_gate_provenance_checkpoint_and_merge_commit(tmp_path, monkeypatch):
    monkeypatch.setattr(results, "recompute_teacher_gate", lambda gate: gate)
    prepared_path = tmp_path / "prepared_manifest.json"
    prepared_path.write_text('{"schema_version":1}\n')
    prepared_hash = results.sha256_file(prepared_path)
    teacher_adapter = tmp_path / "teacher_adapter"
    teacher_adapter.mkdir()
    (teacher_adapter / "adapter_config.json").write_text("{}\n")
    teacher_gate_disk = {
        "schema_version": 3,
        "gate": "teacher_gap_v1",
        "gate_strength": "scientific",
        "passed": True,
        "authorizes_scientific_merge": True,
        "task_sources": ["M"],
        "task_roles": ["teacher_gap_dev"],
        "prepared_manifest": str(prepared_path.resolve()),
        "prepared_manifest_sha256": prepared_hash,
        "trained_adapter_tree_sha256": results.sha256_tree(teacher_adapter),
    }
    teacher_gate_path = tmp_path / "teacher_gap.json"
    teacher_gate_path.write_text(json.dumps(teacher_gate_disk, indent=2, sort_keys=True) + "\n")
    teacher_gate = dict(teacher_gate_disk)
    teacher_gate["manifest_sha256"] = results.sha256_file(teacher_gate_path)

    checkpoint = tmp_path / "merged_teacher"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"pinned merged weights")
    checkpoint_hash = results.sha256_tree(
        checkpoint, exclude_relative_paths=("merge_provenance.json",)
    )
    state = {"commit": COMMIT, "dirty": False}
    provenance_disk = {
        "schema_version": 1,
        "schema": "opd_math_merged_teacher_v2",
        "status": "completed",
        "output_checkpoint": str(checkpoint.resolve()),
        "output_checkpoint_tree_sha256": checkpoint_hash,
        "teacher_gap_manifest": str(teacher_gate_path.resolve()),
        "teacher_gap_manifest_sha256": results.sha256_file(teacher_gate_path),
        "prepared_manifest": str(prepared_path.resolve()),
        "prepared_manifest_sha256": prepared_hash,
        "base_model": "Qwen/Qwen3-8B",
        "base_revision": "8" * 40,
        "adapter_tree_sha256": teacher_gate["trained_adapter_tree_sha256"],
        "merge_code": {
            "git_state_start": state,
            "git_state_after_merge": state,
            "git_state_before_promotion": state,
            "git_state_end": state,
            "clean_stable_code": True,
            "packages": {
                name: results.EXPECTED_TRAIN_PACKAGES[name]
                for name in ("torch", "transformers", "peft")
            },
        },
    }
    provenance_path = checkpoint / "merge_provenance.json"
    provenance_path.write_text(json.dumps(provenance_disk, indent=2, sort_keys=True) + "\n")
    provenance = dict(provenance_disk)
    provenance["manifest_sha256"] = results.sha256_file(provenance_path)
    run = {
        "teacher_checkpoint": str(checkpoint.resolve()),
        "teacher_base_model": "Qwen/Qwen3-8B",
        "teacher_base_revision": "8" * 40,
        "teacher_model": "teacher-M",
    }
    tokenizer = {
        "gate": "tokenizer_contract_v1",
        "passed": True,
        "exact_contract_match": True,
        "student": {"model": STUDENT, "revision": STUDENT_REVISION},
        "teacher": {"model": str(checkpoint.resolve())},
        "server_probe": {"matches": True},
        "server": {"url": "http://127.0.0.1:1234", "model": "teacher-M"},
        "manifest_sha256": "9" * 64,
    }
    server = {
        "schema_version": 2,
        "probe": "exact_token_teacher_scoring_v1",
        "passed": True,
        "tokenizer": STUDENT,
        "tokenizer_revision": STUDENT_REVISION,
        "server_url": "http://127.0.0.1:1234",
        "server_model": "teacher-M",
        "local_process_binding_validated": True,
        "local_process_binding": {
            "scope": "local_linux_proc_process_binding_not_remote_cryptographic_attestation",
            "validated": True,
            "teacher_checkpoint": str(checkpoint.resolve()),
            "teacher_provenance_manifest": str(provenance_path.resolve()),
            "teacher_checkpoint_tree_sha256": checkpoint_hash,
            "teacher_provenance_manifest_sha256": provenance["manifest_sha256"],
        },
        "manifest_sha256": "0" * 64,
    }

    identity = results._validate_teacher_identity(
        run=run,
        teacher_gate=teacher_gate,
        provenance=provenance,
        tokenizer_contract=tokenizer,
        server_contract=server,
        teacher_source="M",
        student_model=STUDENT,
        student_revision=STUDENT_REVISION,
        prepared_path=prepared_path.resolve(),
        prepared_hash=prepared_hash,
        commit=COMMIT,
    )
    assert identity["merged_checkpoint_tree_sha256"] == checkpoint_hash

    (checkpoint / "model.safetensors").write_bytes(b"mutated weights")
    with pytest.raises(ValueError, match="output_checkpoint_tree_sha256|tree"):
        results._validate_teacher_identity(
            run=run,
            teacher_gate=teacher_gate,
            provenance=provenance,
            tokenizer_contract=tokenizer,
            server_contract=server,
            teacher_source="M",
            student_model=STUDENT,
            student_revision=STUDENT_REVISION,
            prepared_path=prepared_path.resolve(),
            prepared_hash=prepared_hash,
            commit=COMMIT,
        )


def synthetic_matrix_gate(key, rewards, *, support, teacher, root):
    source = results.MATRIX_CONTRACT[key]["student_source"]
    records = {f"{source}:{i}": [float(value)] * 4 for i, value in enumerate(rewards)}
    run_path = root / key / "run_manifest.json"
    adapter_path = root / key / "adapter"
    freeze_root = (root / "environment_freezes" / COMMIT).resolve()
    environment = {
        "verifier": {
            "path": str(results.ENVIRONMENT_VERIFIER.resolve()),
            "sha256": results.sha256_file(results.ENVIRONMENT_VERIFIER),
        },
        "train_freeze": {
            "path": str(freeze_root / "train.freeze.txt"),
            "sha256": "8" * 64,
        },
        "train_verification": {"identity": "shared-train-environment"},
        "serve_freeze": (
            {
                "path": str(freeze_root / "serve.freeze.txt"),
                "sha256": "9" * 64,
            }
            if teacher is not None
            else None
        ),
        "serve_verification": (
            {"identity": "shared-serve-environment"}
            if teacher is not None
            else None
        ),
    }
    return {
        "schema_version": 1,
        "gate": results.STUDENT_HELDOUT_GATE,
        "matrix_key": key,
        "passed": True,
        "authorizes_scientific_matrix_readout": True,
        "authorization_is_independent_of_effect_sign": True,
        "objective": results.MATRIX_CONTRACT[key]["objective"],
        "student_source": source,
        "teacher_source": results.MATRIX_CONTRACT[key]["teacher_source"],
        "student_model": STUDENT,
        "student_model_revision": STUDENT_REVISION,
        "records": len(records),
        "samples_per_problem": 4,
        "record_rewards": records,
        "decoding": results.HELDOUT_DECODING,
        "prepared_binding": {"prepared_manifest_sha256": "1" * 64},
        "evaluation_binding": {
            "task_file_sha256": f"{source.lower()}" * 64,
            "evaluation_git_commit": COMMIT,
            "evaluator_file_sha256": "2" * 64,
            "evaluation_packages": quality_gates.EXPECTED_EVALUATION_PACKAGES,
            "tokenizer_contract_sha256": "3" * 64,
        },
        "student_run_binding": {
            "student_training_plan_sha256": "4" * 64,
            "student_training_config_sha256": "5" * 64,
            "git_commit": COMMIT,
            "student_support": support,
            "teacher": teacher,
            "environment": environment,
            "run_manifest": str(run_path),
            "student_adapter": str(adapter_path),
            "trace": {
                "realized_record_ids_sha256": f"{source.lower()}" * 64,
                "realized_prompt_sequence_sha256": f"{source.lower()}" * 64,
            },
        },
        "result_builder": results._result_builder_custody(COMMIT),
    }


def write_synthetic_matrix(tmp_path, *, teacher_m=None):
    support_m = {"manifest_sha256": "6" * 64, "source": "M"}
    support_o = {"manifest_sha256": "7" * 64, "source": "O"}
    teacher_m = teacher_m or {
        "teacher_source": "M",
        "base_model": "teacher",
        "base_revision": "8" * 40,
        "teacher_gap_manifest_sha256": "a" * 64,
        "teacher_gap_payload_sha256": "b" * 64,
        "merged_checkpoint": "teacher-M",
        "merged_checkpoint_tree_sha256": "c" * 64,
        "merge_provenance_manifest_sha256": "d" * 64,
        "merge_provenance_payload_sha256": "e" * 64,
        "server_scoring_manifest_sha256": "process-specific-M",
    }
    teacher_o = {
        "teacher_source": "O",
        "base_model": "teacher",
        "base_revision": "8" * 40,
        "teacher_gap_manifest_sha256": "f" * 64,
        "teacher_gap_payload_sha256": "0" * 64,
        "merged_checkpoint": "teacher-O",
        "merged_checkpoint_tree_sha256": "1" * 64,
        "merge_provenance_manifest_sha256": "2" * 64,
        "merge_provenance_payload_sha256": "3" * 64,
        "server_scoring_manifest_sha256": "process-specific-O",
    }
    rewards = {
        "baseline_M": [0.25] * 8,
        "baseline_O": [0.50] * 8,
        "M_M": [1.00] * 8,
        "M_O": [0.25] * 8,
        "O_M": [0.00] * 8,
        "O_O": [0.75] * 8,
    }
    paths = {}
    for key in results.MATRIX_CONTRACT:
        teacher = None
        if key.startswith("M_"):
            teacher = teacher_m
        elif key.startswith("O_"):
            teacher = teacher_o
        gate = synthetic_matrix_gate(
            key,
            rewards[key],
            support=support_m if results.MATRIX_CONTRACT[key]["student_source"] == "M" else support_o,
            teacher=teacher,
            root=tmp_path,
        )
        path = tmp_path / f"{key}.json"
        path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
        paths[key] = path
    return paths


def test_matrix_reports_four_baseline_deltas_and_stratified_interaction(tmp_path, monkeypatch):
    monkeypatch.setattr(results, "recompute_student_heldout_result", lambda gate: dict(gate))
    paths = write_synthetic_matrix(tmp_path)

    matrix = results.matrix_readout(paths)

    assert matrix["baseline_deltas"]["M_M"]["estimate"] == 0.75
    assert matrix["baseline_deltas"]["O_M"]["estimate"] == -0.25
    assert matrix["baseline_deltas"]["M_O"]["estimate"] == -0.25
    assert matrix["baseline_deltas"]["O_O"]["estimate"] == 0.25
    assert matrix["baseline_deltas"]["M_M"]["classification"] == "helps"
    assert matrix["baseline_deltas"]["O_M"]["classification"] == "harms"
    assert matrix["same_vs_cross"]["M"]["estimate"] == 1.0
    assert matrix["same_vs_cross"]["O"]["estimate"] == 0.5
    assert matrix["same_vs_cross"]["equal_stratum_mean"]["estimate"] == 0.75
    assert matrix["stratified_interaction"]["estimate"] == 1.5
    assert matrix["bootstrap"]["draws"] == 10_000
    assert matrix["environment_freezes"] == {
        "train": {
            "path": str(
                (tmp_path / "environment_freezes" / COMMIT / "train.freeze.txt").resolve()
            ),
            "sha256": "8" * 64,
        },
        "serve": {
            "path": str(
                (tmp_path / "environment_freezes" / COMMIT / "serve.freeze.txt").resolve()
            ),
            "sha256": "9" * 64,
        },
    }
    assert "OPD math six-run" in results.matrix_markdown(matrix)


def test_matrix_fails_closed_on_wrong_key_set_or_teacher_reuse(tmp_path, monkeypatch):
    monkeypatch.setattr(results, "recompute_student_heldout_result", lambda gate: dict(gate))
    paths = write_synthetic_matrix(tmp_path)
    incomplete = dict(paths)
    incomplete.pop("O_O")
    with pytest.raises(ValueError, match="exactly"):
        results.matrix_readout(incomplete)

    payload = json.loads(paths["M_O"].read_text())
    payload["student_run_binding"]["teacher"]["merged_checkpoint"] = "wrong"
    paths["M_O"].write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="one exact teacher identity"):
        results.matrix_readout(paths)


def test_matrix_requires_matched_realized_training_sequence_within_source(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(results, "recompute_student_heldout_result", lambda gate: dict(gate))
    paths = write_synthetic_matrix(tmp_path)
    payload = json.loads(paths["O_M"].read_text())
    payload["student_run_binding"]["trace"]["realized_record_ids_sha256"] = "f" * 64
    paths["O_M"].write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="realized training sequence"):
        results.matrix_readout(paths)


@pytest.mark.parametrize(
    ("key", "field", "replacement"),
    (
        ("baseline_O", "sha256", "0" * 64),
        ("M_O", "path", "/different/train.freeze.txt"),
    ),
)
def test_matrix_requires_one_exact_train_freeze_across_all_six_arms(
    tmp_path, monkeypatch, key, field, replacement
):
    monkeypatch.setattr(results, "recompute_student_heldout_result", lambda gate: dict(gate))
    paths = write_synthetic_matrix(tmp_path)
    payload = json.loads(paths[key].read_text())
    payload["student_run_binding"]["environment"]["train_freeze"][field] = replacement
    paths[key].write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="train_freeze mismatch"):
        results.matrix_readout(paths)


@pytest.mark.parametrize(
    ("key", "replacement", "message"),
    (
        (
            "baseline_M",
            {"path": "/unexpected/serve.freeze.txt", "sha256": "9" * 64},
            "serve_freeze mismatch",
        ),
        ("M_M", None, "lacks a validated teacher serve environment freeze"),
        (
            "M_O",
            {"path": "/different/serve.freeze.txt", "sha256": "9" * 64},
            "serve_freeze mismatch",
        ),
        (
            "O_O",
            {"path": "/shared/serve.freeze.txt", "sha256": "0" * 64},
            "serve_freeze mismatch",
        ),
    ),
)
def test_matrix_requires_null_baseline_and_one_exact_main_serve_freeze(
    tmp_path, monkeypatch, key, replacement, message
):
    monkeypatch.setattr(results, "recompute_student_heldout_result", lambda gate: dict(gate))
    paths = write_synthetic_matrix(tmp_path)
    payload = json.loads(paths[key].read_text())
    payload["student_run_binding"]["environment"]["serve_freeze"] = replacement
    paths[key].write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match=message):
        results.matrix_readout(paths)


def test_result_output_preflight_rejects_collisions_and_protected_trees(tmp_path):
    protected = tmp_path / "adapter"
    protected.mkdir()
    safe_json = tmp_path / "results" / "matrix.json"
    safe_md = tmp_path / "results" / "matrix.md"
    assert results._preflight_result_outputs([safe_json, safe_md]) == [
        safe_json.resolve(),
        safe_md.resolve(),
    ]

    with pytest.raises(ValueError, match="distinct"):
        results._preflight_result_outputs([safe_json, safe_json])
    with pytest.raises(ValueError, match="protected input tree"):
        results._preflight_result_outputs(
            [protected / "heldout.json"], protected_trees=[protected]
        )
