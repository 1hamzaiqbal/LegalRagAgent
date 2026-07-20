import hashlib
import json
import random
from argparse import Namespace
from pathlib import Path

import pytest

from tests.opd_evaluation_fixture import write_merged_evaluation

from scripts.opd_math import quality_gates
from scripts.opd_math import student_results as results
from scripts.opd.trace_metrics import reconstruct_step_metrics


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
    monkeypatch.setattr(
        quality_gates,
        "reverify_recorded_environment",
        lambda recorded, **kwargs: dict(recorded),
    )
    monkeypatch.setattr(results, "recompute_teacher_gate", lambda gate: dict(gate))
    monkeypatch.setattr(results, "verify_completion", fake_verify)
    monkeypatch.setattr(results, "_legacy_m_strict_verdict", fake_strict_verify)
    monkeypatch.setattr(
        results,
        "_validate_o_m_diagnostic_external_custody",
        lambda **kwargs: (
            kwargs["gates"]["synthetic_support_identity"],
            {"synthetic": "validated-environment"},
            kwargs["gates"]["synthetic_teacher_identity"],
        ),
    )


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def fake_verify(completion, _gold):
    correct = r"\boxed{1}" in completion
    return {"reward": float(correct), "status": "correct" if correct else "incorrect"}


def fake_strict_verify(completion, gold):
    verdict = fake_verify(completion, gold)
    return {
        **verdict,
        "evaluation_verifier_attempts": 1,
        "transient_candidate_error_count": 0,
    }


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
        "gate": quality_gates.STUDENT_GATE_TYPE,
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
        "evaluation_contract": quality_gates.EVALUATION_CONTRACT,
        "evaluation_environment": {"synthetic": True},
        "evaluation_post_promotion_custody": {"synthetic": True},
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
                "total_loss": 0.0,
                "gradient_norm_before_clip": 0.5,
                "task_loss": 0.0,
                "reverse_kl_score_function_surrogate": 0.0,
                "sampled_k1_estimate": None,
                "gap_gate_mean": None,
                "positive_gap_fraction": None,
                "reward_mean": 0.25,
                "informative_group_fraction": 1.0,
                "tokens": 4,
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
                    "schema_version": 2,
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
                    "student_token_logprobs": [-1.0],
                    "teacher_token_logprobs_on_student_trajectory": None,
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
    preregistration_file = tmp_path / "student" / "synthetic_preregistration.json"
    launch_ledger_file = tmp_path / "student" / "synthetic_launch_ledger.json"
    preregistration_file.write_text('{"synthetic":"preregistration"}\n')
    launch_ledger_file.write_text('{"synthetic":"launch-ledger"}\n')
    preregistration_file.chmod(0o444)
    launch_ledger_file.chmod(0o444)
    support_identity = {
        "manifest_sha256": support["manifest_sha256"],
        "payload_sha256": results.canonical_json_sha256(
            results._gate_without_file_hash(support)
        ),
        "source": source,
    }
    prelaunch_path = tmp_path / "student" / "synthetic.prelaunch.json"
    prelaunch = {
        "schema_version": 1,
        "receipt": results.O_TEACHER_PRELAUNCH_RECEIPT,
        "created_utc": "2026-07-20T00:00:00Z",
        "sealed_before_optimizer_start": True,
        "campaign_id": "synthetic-heldout",
        "run_key": f"baseline_{source}",
        "run_id": f"synthetic_{source}",
        "scheduler_job_id": "12345",
        "mode": "task_rl",
        "student_source": source,
        "git_commit": COMMIT,
        "out_dir": str((tmp_path / "student").resolve()),
        "expected_artifacts": {
            "run_manifest": str((trace_dir / "run_manifest.json").resolve()),
            "student_completion_manifest": str(
                (trace_dir / "completion_manifest.json").resolve()
            ),
            "student_adapter": str(adapter.resolve()),
            "prelaunch_receipt": str(prelaunch_path.resolve()),
        },
        "preregistration": {
            "path": str(preregistration_file.resolve()),
            "sha256": results.sha256_file(preregistration_file),
        },
        "launch_ledger": {
            "path": str(launch_ledger_file.resolve()),
            "sha256": results.sha256_file(launch_ledger_file),
        },
        "student_support": support_identity,
        "o_teacher": None,
    }
    prelaunch_path.write_text(json.dumps(prelaunch, indent=2, sort_keys=True) + "\n")
    prelaunch_path.chmod(0o444)
    binding = {
        "pair_id": None,
        "student_source": source,
        "teacher_source": None,
        "budget_mode": "primary_matched",
        "campaign_run_id": f"synthetic_{source}",
        "scheduler_job_id": "12345",
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
        "prelaunch_receipt": {
            "path": str(prelaunch_path.resolve()),
            "sha256": results.sha256_file(prelaunch_path),
            "payload_sha256": results.canonical_json_sha256(prelaunch),
            "campaign_id": prelaunch["campaign_id"],
            "run_key": prelaunch["run_key"],
            "sealed_before_optimizer_start": True,
            "preregistration": prelaunch["preregistration"],
            "launch_ledger": prelaunch["launch_ledger"],
        },
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
        "normalized_training_config": fixed,
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


def evaluation_fixture(
    tmp_path,
    holdout,
    holdout_rows,
    adapter,
    *,
    environment_contract,
    source="M",
    rewards=None,
):
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
        exact_environment=True,
        environment_contract=environment_contract,
    )


def heldout_args(tmp_path):
    prepared_path, prepared, train, train_rows, holdout, holdout_rows = prepared_fixture(tmp_path)
    adapter = tmp_path / "student" / "final"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text('{"r":32}\n')
    run, completion = student_run_fixture(
        tmp_path, prepared_path, prepared, train, train_rows, adapter
    )
    run_environment = json.loads(run.read_text())["binding"]["environment_contract"]
    evaluation_environment = json.loads(json.dumps(run_environment))
    evaluation_environment["train_environment_root"] = evaluation_environment[
        "train_verification"
    ]["environment_root"]
    evaluation_environment["train_runtime_packages"] = (
        quality_gates.EXPECTED_EVALUATION_PACKAGES
    )
    evaluation_environment["train_freeze"]["required_packages"] = (
        quality_gates.EXPECTED_EVALUATION_PACKAGES
    )
    summary, samples = evaluation_fixture(
        tmp_path,
        holdout,
        holdout_rows,
        adapter,
        environment_contract=evaluation_environment,
    )
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


def test_heldout_cli_cannot_publish_inside_evaluation_custody(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(results, "recompute_student_gate", lambda gate: gate)
    args = heldout_args(tmp_path)
    output = Path(args.student_summary).parent / "heldout_gate.json"
    argv = [
        "student_results.py",
        "heldout",
        "--matrix-key",
        args.matrix_key,
        "--student-run-manifest",
        str(args.student_run_manifest),
        "--student-completion-manifest",
        str(args.student_completion_manifest),
        "--student-summary",
        str(args.student_summary),
        "--student-samples",
        str(args.student_samples),
        "--trained-adapter",
        str(args.trained_adapter),
        "--prepared-manifest",
        str(args.prepared_manifest),
        "--student-model",
        args.student_model,
        "--student-revision",
        args.student_revision,
        "--task-source",
        args.task_source,
        "--output",
        str(output),
    ]
    monkeypatch.setattr("sys.argv", argv)

    with pytest.raises(ValueError, match="protected input tree"):
        results.main()
    assert not output.exists()


def test_heldout_cli_remains_recomputable_after_publication(tmp_path, monkeypatch):
    monkeypatch.setattr(results, "recompute_student_gate", lambda gate: gate)
    args = heldout_args(tmp_path)
    output = tmp_path.parent / f"{tmp_path.name}-published" / "heldout_gate.json"
    output.parent.mkdir()
    monkeypatch.setattr(
        "sys.argv",
        [
            "student_results.py",
            "heldout",
            "--matrix-key",
            args.matrix_key,
            "--student-run-manifest",
            str(args.student_run_manifest),
            "--student-completion-manifest",
            str(args.student_completion_manifest),
            "--student-summary",
            str(args.student_summary),
            "--student-samples",
            str(args.student_samples),
            "--trained-adapter",
            str(args.trained_adapter),
            "--prepared-manifest",
            str(args.prepared_manifest),
            "--student-model",
            args.student_model,
            "--student-revision",
            args.student_revision,
            "--task-source",
            args.task_source,
            "--output",
            str(output),
        ],
    )

    assert results.main() == 0
    written = json.loads(output.read_text())
    assert results.recompute_student_heldout_result(written) == written
    assert output.stat().st_mode & 0o222 == 0


def test_heldout_gate_rejects_environment_less_v1_merged_evaluation(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(results, "verify_completion", fake_verify)
    monkeypatch.setattr(quality_gates, "verify_completion", fake_verify)
    monkeypatch.setattr(results, "recompute_student_gate", lambda gate: gate)
    args = heldout_args(tmp_path)
    (tmp_path / "legacy").mkdir()
    prepared_path, _, _, _, holdout, holdout_rows = prepared_fixture(tmp_path / "legacy")
    summary, samples = write_merged_evaluation(
        tmp_path / "legacy",
        "heldout-legacy-v1",
        holdout,
        {row["record_id"]: [0.0, 0.0, 0.0, 0.0] for row in holdout_rows},
        model=STUDENT,
        revision=STUDENT_REVISION,
        adapter=args.trained_adapter,
        packages=quality_gates.EXPECTED_EVALUATION_PACKAGES,
        git_commit=COMMIT,
        tokenizer_contract_sha256="b" * 64,
        decoding=results.HELDOUT_DECODING,
        exact_environment=False,
    )
    args.student_summary = summary
    args.student_samples = samples
    args.prepared_manifest = prepared_path
    with pytest.raises(ValueError, match="exact-environment v2 contract"):
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


def test_result_trace_validator_reconstructs_main_arm_teacher_step_metrics(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(results, "verify_completion", fake_verify)
    prepared_path, prepared, train, train_rows, _, _ = prepared_fixture(tmp_path)
    adapter = tmp_path / "student" / "final"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text('{"r":32}\n')
    _, completion_path = student_run_fixture(
        tmp_path,
        prepared_path,
        prepared,
        train,
        train_rows,
        adapter,
    )
    trace_dir = completion_path.parent
    sample_path = trace_dir / "samples.jsonl"
    step_path = trace_dir / "steps.jsonl"
    sample_rows = [json.loads(line) for line in sample_path.read_text().splitlines()]
    for row in sample_rows:
        student = float(row["student_token_logprobs"][0])
        gap = 0.5 if int(row["sample_idx"]) % 2 == 0 else -0.5
        teacher = student + gap
        row.update(
            {
                "teacher_token_logprobs_on_student_trajectory": [teacher],
                "teacher_nll_on_student_trajectory": -teacher,
                "mean_teacher_student_gap": gap,
                "mean_abs_k1_log_ratio": abs(gap),
                "min_teacher_student_gap": gap,
                "max_teacher_student_gap": gap,
                "positive_teacher_gap_fraction": float(gap > 0),
            }
        )
    fixed = plan_binding()["config"]
    loss_config = {
        "task_reward_coef": fixed["task_reward_coef"],
        "k1_coef": fixed["k1_coef"],
        "gap_gate_beta": fixed["gap_gate_beta"],
        "advantage_clip": fixed["advantage_clip"],
    }
    step_rows = [json.loads(line) for line in step_path.read_text().splitlines()]
    for step, step_row in enumerate(step_rows, 1):
        step_samples = [row for row in sample_rows if row["step"] == step]
        step_row["mode"] = "task_rl_k1_gap"
        step_row.update(
            reconstruct_step_metrics(
                step_samples,
                mode="task_rl_k1_gap",
                **loss_config,
            )
        )
    write_jsonl(sample_path, sample_rows)
    write_jsonl(step_path, step_rows)
    completion = json.loads(completion_path.read_text())
    completion["trace_artifacts"]["samples.jsonl"]["sha256"] = results.sha256_file(
        sample_path
    )
    completion["trace_artifacts"]["steps.jsonl"]["sha256"] = results.sha256_file(
        step_path
    )

    trace = results._validate_trace_artifacts(
        run={},
        completion=completion,
        completion_path=completion_path,
        training_rows=train_rows,
        objective="task_rl_k1_gap",
        source="M",
        fixed=fixed,
    )
    assert trace["steps"] == fixed["optimizer_steps"]
    assert trace["samples"] == fixed["optimizer_steps"] * fixed["group_size"]

    # Keep the per-sample summaries internally coherent, update the file hash,
    # and prove the stale step-level OPD account is still rejected.
    teacher = -4.0
    student = float(sample_rows[0]["student_token_logprobs"][0])
    gap = teacher - student
    sample_rows[0].update(
        {
            "teacher_token_logprobs_on_student_trajectory": [teacher],
            "teacher_nll_on_student_trajectory": -teacher,
            "mean_teacher_student_gap": gap,
            "mean_abs_k1_log_ratio": abs(gap),
            "min_teacher_student_gap": gap,
            "max_teacher_student_gap": gap,
            "positive_teacher_gap_fraction": float(gap > 0),
        }
    )
    write_jsonl(sample_path, sample_rows)
    completion["trace_artifacts"]["samples.jsonl"]["sha256"] = results.sha256_file(
        sample_path
    )
    with pytest.raises(ValueError, match="differs from trace reconstruction"):
        results._validate_trace_artifacts(
            run={},
            completion=completion,
            completion_path=completion_path,
            training_rows=train_rows,
            objective="task_rl_k1_gap",
            source="M",
            fixed=fixed,
        )


def test_teacher_identity_binds_gate_provenance_checkpoint_and_merge_commit(tmp_path, monkeypatch):
    monkeypatch.setattr(results, "recompute_teacher_gate", lambda gate: gate)
    prepared_path = tmp_path / "prepared_manifest.json"
    prepared_path.write_text('{"schema_version":1}\n')
    prepared_hash = results.sha256_file(prepared_path)
    teacher_adapter = tmp_path / "teacher_adapter"
    teacher_adapter.mkdir()
    (teacher_adapter / "adapter_config.json").write_text("{}\n")
    teacher_environment = {"schema_version": 2, "git_commit": COMMIT}
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
        "teacher_training_environment": teacher_environment,
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
        "schema": "opd_math_merged_teacher_v3",
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
        "teacher_training_environment": teacher_environment,
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

    mismatched_disk = dict(provenance_disk)
    mismatched_disk["teacher_training_environment"] = {
        "schema_version": 2,
        "git_commit": "f" * 40,
    }
    provenance_path.write_text(json.dumps(mismatched_disk, indent=2, sort_keys=True) + "\n")
    mismatched_provenance = dict(mismatched_disk)
    mismatched_provenance["manifest_sha256"] = results.sha256_file(provenance_path)
    server["local_process_binding"]["teacher_provenance_manifest_sha256"] = (
        mismatched_provenance["manifest_sha256"]
    )
    with pytest.raises(ValueError, match="teacher_training_environment"):
        results._validate_teacher_identity(
            run=run,
            teacher_gate=teacher_gate,
            provenance=mismatched_provenance,
            tokenizer_contract=tokenizer,
            server_contract=server,
            teacher_source="M",
            student_model=STUDENT,
            student_revision=STUDENT_REVISION,
            prepared_path=prepared_path.resolve(),
            prepared_hash=prepared_hash,
            commit=COMMIT,
        )

    provenance_path.write_text(json.dumps(provenance_disk, indent=2, sort_keys=True) + "\n")
    server["local_process_binding"]["teacher_provenance_manifest_sha256"] = provenance[
        "manifest_sha256"
    ]

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
    completion_path = root / key / "completion_manifest.json"
    adapter_path = root / key / "adapter"
    summary_path = root / key / "heldout" / "summary.json"
    samples_path = root / key / "heldout" / "samples.jsonl"
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
        "record_ids_sha256": results.canonical_json_sha256(sorted(records)),
        "record_accuracy_bounds_under_verifier_uncertainty": {
            record_id: [sum(values) / len(values), sum(values) / len(values)]
            for record_id, values in records.items()
        },
        "decoding": results.HELDOUT_DECODING,
        "prepared_binding": {
            "prepared_manifest_sha256": results.sha256_file(
                root / "data" / "prepared_manifest.json"
            )
        },
        "evaluation_binding": {
            "task_file_sha256": f"{source.lower()}" * 64,
            "evaluation_git_commit": COMMIT,
            "evaluator_file_sha256": "2" * 64,
            "evaluation_packages": quality_gates.EXPECTED_EVALUATION_PACKAGES,
            "tokenizer_contract_sha256": "3" * 64,
            "evaluation_contract": quality_gates.EVALUATION_CONTRACT,
            "evaluation_contract_sha256": hashlib.sha256(key.encode()).hexdigest(),
            "evaluation_environment": {
                "verifier": environment["verifier"],
                "train_freeze": environment["train_freeze"],
                "train_verification": environment["train_verification"],
            },
            "evaluation_post_promotion_custody": {
                "path": str((root / key / "heldout.custody.json").resolve()),
                "sha256": hashlib.sha256(f"custody:{key}".encode()).hexdigest(),
                "tree_sha256": "c" * 64,
            },
        },
        "student_run_binding": {
            "student_training_plan_sha256": results.sha256_file(
                results.CANONICAL_STUDENT_TRAINING_PLAN
            ),
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
        "inputs": {
            "student_run_manifest": str(run_path.resolve()),
            "student_completion_manifest": str(completion_path.resolve()),
            "student_summary": str(summary_path.resolve()),
            "student_samples": str(samples_path.resolve()),
            "trained_adapter": str(adapter_path.resolve()),
            "prepared_manifest": str(
                (root / "data" / "prepared_manifest.json").resolve()
            ),
        },
    }


def write_synthetic_matrix(tmp_path, *, teacher_m=None):
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "prepared_manifest.json").write_text(
        '{"schema_version":1,"scientific_use_allowed":true}\n'
    )
    support_m = {
        "manifest_sha256": "6" * 64,
        "payload_sha256": "8" * 64,
        "source": "M",
    }
    support_o = {
        "manifest_sha256": "7" * 64,
        "payload_sha256": "9" * 64,
        "source": "O",
    }
    teacher_m = teacher_m or {
        "teacher_source": "M",
        "base_model": "teacher",
        "base_revision": "8" * 40,
        "teacher_gap_manifest": str(
            (tmp_path / "teacher" / "M_teacher_gap.json").resolve()
        ),
        "teacher_gap_manifest_sha256": "a" * 64,
        "teacher_gap_payload_sha256": "b" * 64,
        "merged_checkpoint": "teacher-M",
        "merged_checkpoint_tree_sha256": "c" * 64,
        "merge_provenance_manifest_sha256": "d" * 64,
        "merge_provenance_payload_sha256": "e" * 64,
        "tokenizer_contract_manifest_sha256": "4" * 64,
        "tokenizer_contract_payload_sha256": "5" * 64,
        "server_scoring_manifest_sha256": "process-specific-M",
        "server_scoring_payload_sha256": "6" * 64,
    }
    (tmp_path / "teacher").mkdir(parents=True, exist_ok=True)
    (tmp_path / "gates").mkdir(parents=True, exist_ok=True)
    o_teacher_gate = tmp_path / "teacher" / "O_teacher_gap.json"
    o_teacher_gate.write_text('{"gate":"synthetic-O"}\n')
    o_checkpoint = tmp_path / "teacher" / "O_merged"
    o_checkpoint.mkdir()
    teacher_o = {
        "teacher_source": "O",
        "base_model": "teacher",
        "base_revision": "8" * 40,
        "teacher_gap_manifest": str(o_teacher_gate.resolve()),
        "teacher_gap_manifest_sha256": results.sha256_file(o_teacher_gate),
        "teacher_gap_payload_sha256": "0" * 64,
        "merged_checkpoint": str(o_checkpoint.resolve()),
        "merged_checkpoint_tree_sha256": "1" * 64,
        "merge_provenance_manifest_sha256": "2" * 64,
        "merge_provenance_payload_sha256": "3" * 64,
        "tokenizer_contract_manifest_sha256": "4" * 64,
        "tokenizer_contract_payload_sha256": "5" * 64,
        "server_scoring_manifest_sha256": "process-specific-O",
        "server_scoring_payload_sha256": "6" * 64,
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
        path = tmp_path / "gates" / f"{key}.json"
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


def test_matrix_classification_uses_worst_case_verifier_envelope(tmp_path, monkeypatch):
    monkeypatch.setattr(results, "recompute_student_heldout_result", lambda gate: dict(gate))
    paths = write_synthetic_matrix(tmp_path)
    baseline_path = paths["baseline_M"]
    baseline = json.loads(baseline_path.read_text())
    baseline["record_accuracy_bounds_under_verifier_uncertainty"] = {
        record_id: [0.25, 1.0] for record_id in baseline["record_rewards"]
    }
    baseline_path.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n")

    matrix = results.matrix_readout(paths)
    contrast = matrix["baseline_deltas"]["M_M"]

    assert contrast["classification_without_verifier_uncertainty"] == "helps"
    assert contrast["classification"] == "inconclusive"
    assert contrast["verifier_uncertainty_sensitivity"]["estimate_bounds"] == [
        0.0,
        0.75,
    ]
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


def o_teacher_paths(tmp_path):
    all_paths = write_synthetic_matrix(tmp_path)
    return {key: all_paths[key] for key in results.O_TEACHER_CONTRACT}


def synthetic_o_m_diagnostic_identity(
    tmp_path, *, support_identity, teacher_identity
):
    diagnostic_root = tmp_path / "diagnostic"
    trace_root = diagnostic_root / "trace"
    adapter = diagnostic_root / "adapter"
    trace_root.mkdir(parents=True)
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"diagnostic update")
    adapter_hash = results.sha256_tree(adapter)
    state = {"commit": COMMIT, "dirty": False}
    fixed = dict(plan_binding()["config"])
    fixed.update({"budget_mode": "dose_response", "optimizer_steps": 1})
    train = diagnostic_root / "data" / "roles" / "M" / "student_opd.jsonl"
    train_row = {
        "record_id": "M:diagnostic:0",
        "source": "M",
        "role": "student_opd",
        "prompt": [{"role": "user", "content": "Return one."}],
        "solution": r"\boxed{1}",
    }
    write_jsonl(train, [train_row])
    prepared_path = diagnostic_root / "data" / "prepared_manifest.json"
    prepared_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scientific_use_allowed": True,
                "files": {
                    "roles/M/student_opd.jsonl": {
                        "rows": 1,
                        "sha256": results.sha256_file(train),
                    }
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    samples = []
    for sample_idx in range(4):
        completion_text = (
            r"Final answer: \boxed{1}."
            if sample_idx == 0
            else r"Final answer: \boxed{999}."
        )
        reward = float(sample_idx == 0)
        samples.append(
            {
                "schema_version": 2,
                "step": 1,
                "record_id": train_row["record_id"],
                "source": "M",
                "group_id": 0,
                "sample_idx": sample_idx,
                "completion_tokens": 1,
                "prompt_tokens": 2,
                "terminated_by_eos": True,
                "rollout_batch_latency_seconds": 0.1,
                "teacher_scoring_latency_seconds": 0.01,
                "completion_sha256": hashlib.sha256(
                    completion_text.encode()
                ).hexdigest(),
                "prompt_sha256": results._task_prompt_sha256(train_row),
                "prompt_token_ids": [1, 2],
                "completion_token_ids": [3],
                "completion_text": completion_text,
                "student_token_logprobs": [-1.0],
                "teacher_token_logprobs_on_student_trajectory": [-0.5],
                "student_nll": 1.0,
                "teacher_nll_on_student_trajectory": 0.5,
                "mean_teacher_student_gap": 0.5,
                "mean_abs_k1_log_ratio": 0.5,
                "min_teacher_student_gap": 0.5,
                "max_teacher_student_gap": 0.5,
                "positive_teacher_gap_fraction": 1.0,
                "reward": reward,
                "reward_status": "correct" if reward else "incorrect",
            }
        )
    reconstructed = reconstruct_step_metrics(
        samples,
        mode="task_rl_k1_gap",
        task_reward_coef=fixed["task_reward_coef"],
        k1_coef=fixed["k1_coef"],
        gap_gate_beta=fixed["gap_gate_beta"],
        advantage_clip=fixed["advantage_clip"],
    )
    step = {
        "schema_version": 1,
        "step": 1,
        "mode": "task_rl_k1_gap",
        "prompts": 1,
        "samples": 4,
        "gradient_norm_before_clip": 0.5,
        **reconstructed,
    }
    steps_path = trace_root / "steps.jsonl"
    samples_path = trace_root / "samples.jsonl"
    write_jsonl(steps_path, [step])
    write_jsonl(samples_path, samples)
    completion = {
        "schema_version": 1,
        "status": "completed",
        "objective": "task_rl_k1_gap",
        "intended_scientific_run": False,
        "optimizer_steps_completed": 1,
        "rollout_samples": 4,
        "scored_completion_tokens": 4,
        "prompt_group_tokens": 2,
        "sample_expanded_prompt_tokens": 8,
        "prompt_groups_seen": 1,
        "step_trace_rows": 1,
        "sample_trace_rows": 4,
        "realized_training_geometry_observed": True,
        "unique_training_records": 1,
        "realized_record_ids_sha256": results.canonical_json_sha256(
            [train_row["record_id"]]
        ),
        "realized_prompt_sequence_sha256": results.canonical_json_sha256(
            [
                {
                    "record_id": train_row["record_id"],
                    "prompt_sha256": results._task_prompt_sha256(train_row),
                }
            ]
        ),
        "total_rollout_latency_seconds": 0.1,
        "total_teacher_scoring_latency_seconds": 0.04,
        "informative_task_steps": 1,
        "informative_task_groups": 1,
        "total_task_groups": 1,
        "informative_group_fraction": 1.0,
        "minimum_informative_group_fraction": fixed[
            "min_informative_group_fraction"
        ],
        "task_signal_observed": True,
        "finite_nonzero_gradient_observed": True,
        "parameter_update_observed": True,
        "clean_stable_code": True,
        "stable_training_environment": True,
        "stable_environment_after_candidate_save": True,
        "stable_environment_end": True,
        "stable_final_artifact_hash": True,
        "live_local_server_process_binding_validated": True,
        "training_artifact_eligible_for_held_out_evaluation": False,
        "scientific_use_allowed": False,
        "initial_parameter_signature": {"value": 1},
        "final_parameter_signature": {"value": 2},
        "git_state_start": state,
        "git_state_training_end": state,
        "git_state_after_candidate_save": state,
        "git_state_end": state,
        "final_adapter": str(adapter.resolve()),
        "final_adapter_tree_sha256": adapter_hash,
        "trace_artifacts": {
            "steps.jsonl": {
                "path": str(steps_path.resolve()),
                "rows": 1,
                "sha256": results.sha256_file(steps_path),
            },
            "samples.jsonl": {
                "path": str(samples_path.resolve()),
                "rows": 4,
                "sha256": results.sha256_file(samples_path),
            },
        },
    }
    completion_path = trace_root / "completion_manifest.json"
    completion_path.write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n")
    run = {
        "schema_version": 1,
        "status": completion["status"],
        "objective": "task_rl_k1_gap",
        "intended_scientific_run": False,
        "git_commit": COMMIT,
        "git_worktree_clean": True,
        "git_state_start": state,
        "task_file": str(train.resolve()),
        "task_file_sha256": results.sha256_file(train),
        "selected_task_rows": 1,
        "task_limit": 1,
        "student": STUDENT,
        "student_revision": STUDENT_REVISION,
        "normalized_training_config": fixed,
        "optimizer_steps_planned": 1,
        "micro_prompts_per_step": 1,
        "planned_rollout_samples": 4,
        "seed": fixed["seed"],
        "generation": {
            "group_size": fixed["group_size"],
            "temperature": fixed["temperature"],
            "top_p": fixed["top_p"],
            "top_k": fixed["top_k"],
            "max_new_tokens": fixed["max_new_tokens"],
            "enable_thinking": fixed["enable_thinking"],
        },
        "optimization": {
            "attn_implementation": fixed["attn_implementation"],
            "gradient_checkpointing": fixed["gradient_checkpointing"],
            "learning_rate": fixed["learning_rate"],
            "lora_r": fixed["lora_r"],
        },
        "loss": {
            "task_reward_coef": fixed["task_reward_coef"],
            "k1_coef": fixed["k1_coef"],
            "gap_gate_beta": fixed["gap_gate_beta"],
            "advantage_clip": fixed["advantage_clip"],
        },
        "binding": {
            "pair_id": "O_M",
            "student_source": "M",
            "teacher_source": "O",
            "budget_mode": "dose_response",
            "local_checkpoint_custody_validated": True,
            "server_alias_and_token_contract_validated": True,
            "live_local_server_process_binding_validated": True,
            "serve_environment_process_binding_validated": True,
            "environment_contract": {"synthetic": True},
        },
        "gates": {
            "prepared_data": {
                "path": str(prepared_path.resolve()),
                "sha256": results.sha256_file(prepared_path),
                "task_role_file": "roles/M/student_opd.jsonl",
                "task_file_sha256": results.sha256_file(train),
                "scientific_use_allowed": True,
            },
            "synthetic_support_identity": support_identity,
            "synthetic_teacher_identity": teacher_identity,
        },
        "completion": completion,
    }
    run_path = trace_root / "run_manifest.json"
    run_path.write_text(json.dumps(run, indent=2, sort_keys=True) + "\n")
    inputs = {
        "run_manifest": str(run_path.resolve()),
        "run_manifest_sha256": results.sha256_file(run_path),
        "completion_manifest": str(completion_path.resolve()),
        "completion_manifest_sha256": results.sha256_file(completion_path),
        "student_adapter": str(adapter.resolve()),
        "student_adapter_tree_sha256": adapter_hash,
    }
    audit = diagnostic_root / "terminal_audit.json"
    audit.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "audit": results.O_M_ONE_STEP_DIAGNOSTIC_AUDIT,
                "passed": True,
                "diagnostic_clean_before_preregistration": True,
                "plumbing_only": True,
                "scientific_result": False,
                "matrix_key": "O_M",
                "git_commit": COMMIT,
                "inputs": inputs,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    audit.chmod(0o444)
    return {
        "terminal_audit": str(audit.resolve()),
        "terminal_audit_sha256": results.sha256_file(audit),
        **inputs,
        "diagnostic_clean_before_preregistration": True,
        "plumbing_only": True,
        "scientific_result": False,
    }


def synthetic_m_negative_selection_context(tmp_path):
    custody_root = tmp_path / "custody" / "m_legacy_negative"
    task_path = custody_root / "data" / "roles" / "M" / "teacher_gap_dev.jsonl"
    task_rows = []
    for index in range(3):
        task_rows.append(
            {
                "record_id": f"M:synthetic:{index}",
                "source": "M",
                "role": "teacher_gap_dev",
                "solution": r"\boxed{1}",
            }
        )
    write_jsonl(task_path, task_rows)

    adapter = custody_root / "teachers" / "M" / "run_1" / "final_adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter.safetensors").write_bytes(b"synthetic-m-adapter")
    adapter_hash = results.sha256_tree(adapter)
    run_path = adapter.parent / "run_manifest.json"
    clean_state = {"commit": COMMIT, "dirty": False}
    run_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "completed",
                "source": "M",
                "role": "teacher_train",
                "model": "Qwen/Qwen3-8B",
                "model_revision": "b" * 40,
                "final_adapter": str(adapter.resolve()),
                "final_adapter_tree_sha256": adapter_hash,
                "intended_scientific_run": True,
                "scientific_use_allowed": True,
                "optimizer_progress_complete": True,
                "stable_final_artifact_hash": True,
                "git_state_start": clean_state,
                "git_state_end": clean_state,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    decoding = {
        "thinking": False,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "max_new_tokens": 1024,
        "seed": 0,
    }
    stored_surfaces = {}
    evaluation_paths = {}
    for arm in ("base", "trained"):
        merged = custody_root / "evaluations" / arm / "merged"
        samples_path = merged / "samples.jsonl"
        rows = []
        surface = {}
        for record_index, task in enumerate(task_rows):
            rewards = []
            for sample_idx in range(4):
                reward = float(sample_idx == 0)
                answer = r"\boxed{1}" if reward else r"\boxed{0}"
                completion = f"{arm}:{record_index}:{sample_idx} {answer}"
                rewards.append(reward)
                rows.append(
                    {
                        "schema_version": 2,
                        "record_id": task["record_id"],
                        "sample_idx": sample_idx,
                        "global_record_index": record_index,
                        "source": "M",
                        "completion_text": completion,
                        "completion_sha256": hashlib.sha256(
                            completion.encode("utf-8")
                        ).hexdigest(),
                        "reward": reward,
                        "reward_status": "correct" if reward else "incorrect",
                    }
                )
            surface[task["record_id"]] = rewards
        write_jsonl(samples_path, rows)
        accuracy = sum(sum(values) for values in surface.values()) / len(rows)
        summary_path = merged / "summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "artifact_kind": results.EVALUATION_MERGED_KIND,
                    "model": "Qwen/Qwen3-8B",
                    "model_revision": "b" * 40,
                    "task_file": str(task_path.resolve()),
                    "task_file_sha256": results.sha256_file(task_path),
                    "task_sources": ["M"],
                    "task_roles": ["teacher_gap_dev"],
                    "records": len(task_rows),
                    "samples": len(rows),
                    "samples_per_problem": 4,
                    "decoding": decoding,
                    "samples_file": "samples.jsonl",
                    "samples_file_sha256": results.sha256_file(samples_path),
                    "completion_text_in_samples": True,
                    "adapter": None if arm == "base" else str(adapter.resolve()),
                    "adapter_tree_sha256": None if arm == "base" else adapter_hash,
                    "accuracy": accuracy,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        output_tree_hash = results.sha256_tree(merged)
        custody_path = merged.parent / "merged.custody.json"
        custody_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "artifact_kind": results.EVALUATION_MERGED_KIND,
                    "model": "Qwen/Qwen3-8B",
                    "model_revision": "b" * 40,
                    "task_file_sha256": results.sha256_file(task_path),
                    "summary": str(summary_path.resolve()),
                    "summary_sha256": results.sha256_file(summary_path),
                    "samples": str(samples_path.resolve()),
                    "samples_sha256": results.sha256_file(samples_path),
                    "output_dir": str(merged.resolve()),
                    "output_tree_sha256": output_tree_hash,
                    "adapter_tree_sha256": None if arm == "base" else adapter_hash,
                    "publication_commit_point": True,
                    "stable_environment_after_promotion": True,
                    "stable_final_artifact_hash": True,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        stored_surfaces[arm] = surface
        evaluation_paths[arm] = {
            "summary": summary_path,
            "samples": samples_path,
            "custody": custody_path,
            "tree_sha256": output_tree_hash,
            "accuracy": accuracy,
        }

    _, delta, low, high = quality_gates.bootstrap_delta(
        stored_surfaces["base"], stored_surfaces["trained"], 0, 10_000
    )
    requirements = {
        "minimum_records_met": True,
        "strict_delta_met": delta > 0.0,
        "positive_bootstrap_lower_bound_met": low > 0.0,
    }
    gate_path = custody_root / "gates" / "m_teacher_gap.json"
    gate_path.parent.mkdir(parents=True)
    gate = {
        "schema_version": 3,
        "gate": "teacher_gap_v1",
        "gate_strength": "scientific",
        "passed": False,
        "authorizes_scientific_merge": False,
        "shared_records": len(task_rows),
        "base_accuracy": evaluation_paths["base"]["accuracy"],
        "trained_accuracy": evaluation_paths["trained"]["accuracy"],
        "paired_delta": delta,
        "bootstrap_95_ci": [low, high],
        "min_delta": 0.0,
        "min_records": 2,
        "require_positive_ci": True,
        "bootstrap_draws": 10_000,
        "bootstrap_seed": 0,
        "requirements": requirements,
        "base_model": "Qwen/Qwen3-8B",
        "base_model_revision": "b" * 40,
        "task_file": str(task_path.resolve()),
        "task_file_sha256": results.sha256_file(task_path),
        "task_sources": ["M"],
        "task_roles": ["teacher_gap_dev"],
        "decoding": decoding,
        "evaluation_git_commit": COMMIT,
        "teacher_training_git_commit": COMMIT,
        "teacher_run_manifest": str(run_path.resolve()),
        "teacher_run_manifest_sha256": results.sha256_file(run_path),
        "trained_adapter": str(adapter.resolve()),
        "trained_adapter_tree_sha256": adapter_hash,
    }
    for arm in ("base", "trained"):
        paths = evaluation_paths[arm]
        gate[f"{arm}_summary"] = str(paths["summary"].resolve())
        gate[f"{arm}_summary_sha256"] = results.sha256_file(paths["summary"])
        gate[f"{arm}_samples"] = str(paths["samples"].resolve())
        gate[f"{arm}_samples_sha256"] = results.sha256_file(paths["samples"])
        gate[f"{arm}_evaluation_post_promotion_custody"] = {
            "path": str(paths["custody"].resolve()),
            "sha256": results.sha256_file(paths["custody"]),
            "tree_sha256": paths["tree_sha256"],
        }
    gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")

    gate_result = {
        field: gate[field]
        for field in (
            "schema_version",
            "gate",
            "gate_strength",
            "passed",
            "authorizes_scientific_merge",
            "shared_records",
            "base_accuracy",
            "trained_accuracy",
            "paired_delta",
            "bootstrap_95_ci",
            "min_delta",
            "min_records",
            "require_positive_ci",
            "bootstrap_draws",
            "bootstrap_seed",
            "requirements",
        )
    }
    terminal_path = custody_root / "campaigns" / "M_terminal" / "terminal_audit.json"
    terminal_path.parent.mkdir(parents=True)
    artifacts = {
        "gate": {
            "path": str(gate_path.resolve()),
            "sha256": results.sha256_file(gate_path),
        }
    }
    for arm in ("base", "trained"):
        for kind in ("summary", "samples", "custody"):
            artifact = evaluation_paths[arm][kind]
            artifacts[f"{arm}_{kind}"] = {
                "path": str(artifact.resolve()),
                "sha256": results.sha256_file(artifact),
            }
    terminal_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "classification": "scientific_teacher_gap_negative_inconclusive_result",
                "git_commit": COMMIT,
                "artifacts": artifacts,
                "gate_result": gate_result,
                "independent_recompute": {
                    "state": "COMPLETED",
                    "exit_code": "0:0",
                    "recompute_teacher_gate_equal_to_disk": True,
                    "gate_sha256": results.sha256_file(gate_path),
                },
                "downstream_authorization": {
                    "M_teacher_merge": False,
                    "M_M_scientific_arm": False,
                    "M_O_scientific_arm": False,
                    "six_arm_matrix_under_current_campaign": False,
                },
                "forbidden_actions_observed": {
                    "M_main_arm_submitted": False,
                    "M_teacher_merge_presubmitted": False,
                    "merged_M_teacher_output_exists": False,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    payload = results.m_teacher_negative_compatibility_audit(
        Namespace(
            teacher_gap_manifest=gate_path,
            terminal_audit=terminal_path,
            teacher_run_manifest=run_path,
            trained_adapter=adapter,
            created_utc="2026-07-20T00:00:00Z",
        )
    )
    path = custody_root / "m_negative_selection.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    path.chmod(0o444)
    return {"path": str(path.resolve()), "sha256": results.sha256_file(path)}


def o_teacher_preregistration(tmp_path, paths):
    output_root = tmp_path.parent / f"{tmp_path.name}-readout"
    arms = {}
    for key, path in paths.items():
        gate = json.loads(path.read_text())
        arms[key] = {
            "heldout_gate": str(path.resolve()),
            "run_manifest": gate["student_run_binding"]["run_manifest"],
            "student_completion_manifest": gate["inputs"][
                "student_completion_manifest"
            ],
            "student_adapter": gate["student_run_binding"]["student_adapter"],
            "student_eval_summary": gate["inputs"]["student_summary"],
            "student_eval_samples": gate["inputs"]["student_samples"],
            "student_eval_custody": gate["evaluation_binding"][
                "evaluation_post_promotion_custody"
            ]["path"],
            "prelaunch_receipt": str(
                Path(
                    str(Path(gate["student_run_binding"]["student_adapter"]).parent)
                    + ".prelaunch.json"
                ).resolve()
            ),
        }
    o_teacher = json.loads(paths["O_M"].read_text())["student_run_binding"][
        "teacher"
    ]
    stable_o_teacher = {
        field: o_teacher[field] for field in results.O_TEACHER_STABLE_IDENTITY_FIELDS
    }
    support_identities = {
        source: json.loads(paths[key].read_text())["student_run_binding"][
            "student_support"
        ]
        for source, key in (("M", "baseline_M"), ("O", "baseline_O"))
    }
    payload = {
        "schema_version": 1,
        "preregistration": results.O_TEACHER_PREREGISTRATION,
        "campaign_id": "synthetic-o-teacher-campaign",
        "created_utc": "2026-07-20T00:00:00Z",
        "student_outcome_blind": True,
        "sealed_before_student_arm_launch": True,
        "student_arm_outcomes_inspected_before_sealing": False,
        "teacher_selection_condition_known_before_sealing": True,
        "diagnostic_clean_before_preregistration": True,
        "operational_retry_requires_new_preregistration": True,
        "arm_keys": list(results.O_TEACHER_CONTRACT),
        "git_commit": COMMIT,
        "prepared_manifest": {
            "path": str((tmp_path / "data" / "prepared_manifest.json").resolve()),
            "sha256": results.sha256_file(
                tmp_path / "data" / "prepared_manifest.json"
            ),
        },
        "student_training_plan_sha256": results.sha256_file(
            results.CANONICAL_STUDENT_TRAINING_PLAN
        ),
        "one_step_diagnostic": synthetic_o_m_diagnostic_identity(
            tmp_path,
            support_identity=support_identities["M"],
            teacher_identity=stable_o_teacher,
        ),
        "selection_context": synthetic_m_negative_selection_context(tmp_path),
        "o_teacher_stable_identity": stable_o_teacher,
        "student_support_identities": support_identities,
        "arms": arms,
        "outputs": {
            "json": str((output_root / "result.json").resolve()),
            "markdown": str((output_root / "result.md").resolve()),
            "manifest": str(
                (output_root / "result.manifest.json").resolve()
            ),
        },
        "inference": {
            "bootstrap_unit": "paired_record_within_source",
            "draws": 10_000,
            "seed": 0,
            "resampling_order": "M_then_O_single_random.Random_stream",
            "record_order": "lexicographic_record_id",
            "co_primary_contrasts": ["delta_M", "delta_O"],
            "familywise_alpha": 0.05,
            "familywise_interval": "Bonferroni_percentile_97.5",
            "verifier_uncertainty": "binary_worst_case_bootstrap_envelope_v1",
        },
        "claim_boundary": results.O_TEACHER_CLAIM_BOUNDARY,
    }
    custody_dir = tmp_path / "custody"
    custody_dir.mkdir(exist_ok=True)
    path = custody_dir / "o_teacher_preregistration.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    path.chmod(0o444)
    ledger = custody_dir / "o_teacher_launch_ledger.json"
    ledger.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "ledger": results.O_TEACHER_LAUNCH_LEDGER,
                "campaign_id": payload["campaign_id"],
                "created_utc": payload["created_utc"],
                "sealed_before_student_arm_launch": True,
                "student_arm_outcomes_inspected_before_sealing": False,
                "teacher_selection_condition_known_before_sealing": True,
                "diagnostic_clean_before_preregistration": True,
                "preregistration": {
                    "path": str(path.resolve()),
                    "sha256": results.sha256_file(path),
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    ledger.chmod(0o444)
    for key, gate_path in paths.items():
        gate = json.loads(gate_path.read_text())
        gate["student_run_binding"]["prelaunch_receipt"] = {
            "path": arms[key]["prelaunch_receipt"],
            "sha256": "f" * 64,
            "payload_sha256": "e" * 64,
            "campaign_id": payload["campaign_id"],
            "run_key": key,
            "sealed_before_optimizer_start": True,
            "preregistration": {
                "path": str(path.resolve()),
                "sha256": results.sha256_file(path),
            },
            "launch_ledger": {
                "path": str(ledger.resolve()),
                "sha256": results.sha256_file(ledger),
            },
        }
        gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
    return {
        "preregistration_path": path,
        "launch_ledger_path": ledger,
    }


def test_o_teacher_readout_reports_two_conditional_deltas(tmp_path, monkeypatch):
    monkeypatch.setattr(results, "recompute_student_heldout_result", lambda gate: dict(gate))
    paths = o_teacher_paths(tmp_path)

    preregistration = o_teacher_preregistration(tmp_path, paths)
    readout = results.o_teacher_readout(paths, **preregistration)

    assert readout["arm_keys"] == ["baseline_M", "O_M", "baseline_O", "O_O"]
    assert readout["conditional_on_passing_o_teacher_gate"]
    assert readout["not_a_six_arm_matrix"]
    assert readout["primary_results"]["delta_M"]["estimate"] == -0.25
    assert readout["primary_results"]["delta_O"]["estimate"] == 0.25
    assert readout["primary_results"]["delta_M"]["classification"] == "harms"
    assert readout["primary_results"]["delta_O"]["classification"] == "helps"
    assert readout["secondary_cross_source"]["equal_weight_source_average"]["estimate"] == 0.0
    assert readout["secondary_cross_source"]["source_heterogeneity"]["estimate"] == 0.5
    assert readout["secondary_cross_source"]["source_heterogeneity"][
        "not_a_same_source_effect"
    ]
    assert readout["bootstrap"]["draws"] == 10_000
    assert readout["bootstrap"]["seed"] == 0
    markdown = results.o_teacher_markdown(readout)
    assert "conditional on the preregistered O teacher" in markdown
    assert "M_M` and `M_O` were prohibited" in markdown
    assert "Secondary cross-source diagnostics" in markdown
    assert results.recompute_o_teacher_readout(readout) == readout


def test_primary_prelaunch_receipt_consumes_sealed_preregistration(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(results, "recompute_student_gate", lambda gate: dict(gate))
    paths = o_teacher_paths(tmp_path)
    support_path = tmp_path / "custody" / "support_M.json"
    support_path.parent.mkdir(parents=True, exist_ok=True)
    support = {
        "schema_version": 3,
        "gate": quality_gates.STUDENT_GATE_TYPE,
        "gate_strength": "scientific",
        "passed": True,
        "authorizes_scientific_training": True,
        "task_sources": ["M"],
    }
    support_path.write_text(json.dumps(support, indent=2, sort_keys=True) + "\n")
    support_path.chmod(0o444)
    support_identity = {
        "manifest_sha256": results.sha256_file(support_path),
        "payload_sha256": results.canonical_json_sha256(support),
        "source": "M",
    }
    out_dir = tmp_path / "launch" / "baseline_M" / "task_rl" / "run_seed0"
    for key in ("baseline_M", "O_M"):
        gate = json.loads(paths[key].read_text())
        gate["student_run_binding"]["student_support"] = support_identity
        if key == "baseline_M":
            gate["student_run_binding"]["run_manifest"] = str(
                (out_dir / "traces" / "run_manifest.json").resolve()
            )
            gate["student_run_binding"]["student_adapter"] = str(
                (out_dir / "final").resolve()
            )
            gate["inputs"]["student_run_manifest"] = gate["student_run_binding"][
                "run_manifest"
            ]
            gate["inputs"]["student_completion_manifest"] = str(
                (out_dir / "traces" / "completion_manifest.json").resolve()
            )
            gate["inputs"]["trained_adapter"] = gate["student_run_binding"][
                "student_adapter"
            ]
        paths[key].write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
    preregistration = o_teacher_preregistration(tmp_path, paths)
    receipt_path = Path(str(out_dir) + ".prelaunch.json")
    args = Namespace(
        preregistration=preregistration["preregistration_path"],
        launch_ledger=preregistration["launch_ledger_path"],
        run_key="baseline_M",
        run_id="seed0",
        scheduler_job_id="123456",
        mode="task_rl",
        student_source="M",
        out_dir=out_dir,
        student_support_manifest=support_path,
        teacher_gap_manifest=None,
        teacher_checkpoint=None,
        teacher_provenance_manifest=None,
        output=receipt_path,
    )

    receipt = results.o_teacher_prelaunch_receipt(args)
    assert receipt["sealed_before_optimizer_start"]
    assert receipt["run_key"] == "baseline_M"
    assert receipt_path.stat().st_mode & 0o222 == 0
    with pytest.raises(FileExistsError, match="overwrite"):
        results.o_teacher_prelaunch_receipt(args)


def test_o_teacher_preregistration_reopens_exact_diagnostic_traces(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        results, "recompute_student_heldout_result", lambda gate: dict(gate)
    )
    paths = o_teacher_paths(tmp_path)
    preregistration = o_teacher_preregistration(tmp_path, paths)
    prereg = json.loads(Path(preregistration["preregistration_path"]).read_text())
    completion = json.loads(
        Path(prereg["one_step_diagnostic"]["completion_manifest"]).read_text()
    )
    Path(completion["trace_artifacts"]["samples.jsonl"]["path"]).unlink()

    with pytest.raises(ValueError, match="trace samples.jsonl|regular non-symlink"):
        results.o_teacher_readout(paths, **preregistration)


def test_o_teacher_preregistration_binds_exact_future_eval_paths(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        results, "recompute_student_heldout_result", lambda gate: dict(gate)
    )
    paths = o_teacher_paths(tmp_path)
    preregistration = o_teacher_preregistration(tmp_path, paths)
    gate = json.loads(paths["baseline_M"].read_text())
    gate["inputs"]["student_summary"] = str(
        (tmp_path / "posthoc" / "picked_summary.json").resolve()
    )
    paths["baseline_M"].write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="student_summary mismatch"):
        results.o_teacher_readout(paths, **preregistration)


def test_m_negative_selection_context_uses_dedicated_strict_replay(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        results, "recompute_student_heldout_result", lambda gate: dict(gate)
    )
    paths = o_teacher_paths(tmp_path)
    preregistration = o_teacher_preregistration(tmp_path, paths)
    monkeypatch.setattr(
        results,
        "recompute_teacher_gate",
        lambda gate: pytest.fail("legacy compatibility must not call current gate replay"),
    )

    readout = results.o_teacher_readout(paths, **preregistration)
    selection = readout["preregistration"]["selection_context"]

    assert selection["legacy_gate_recomputed_exactly"]
    assert selection["strict_negative_replay_confirmed"]
    assert selection["m_arms_prohibited"]
    assert selection["verifier_error_count"] == 0


@pytest.mark.parametrize(
    "target",
    (
        "base_samples",
        "base_custody",
        "terminal_audit",
        "teacher_run_manifest",
        "trained_adapter",
    ),
)
def test_m_negative_selection_context_rejects_bound_artifact_tamper(
    tmp_path, target
):
    binding = synthetic_m_negative_selection_context(tmp_path)
    audit = json.loads(Path(binding["path"]).read_text())
    artifact = audit["inputs"][target]
    path = Path(artifact["path"])
    if target == "trained_adapter":
        (path / "tamper.bin").write_bytes(b"tamper")
    else:
        path.write_bytes(path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="hash has drifted|tree_sha256 mismatch"):
        results._validate_m_negative_selection_context(binding)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("m_teacher_gate_passed", True),
        ("merge_authorized", True),
        ("m_arms_prohibited", False),
        ("scientific_authorization", True),
        ("m_m_arm_authorized", True),
        ("task_source", "O"),
    ),
)
def test_m_negative_selection_context_rejects_authorization_bypass(
    tmp_path, field, replacement
):
    binding = synthetic_m_negative_selection_context(tmp_path)
    path = Path(binding["path"])
    payload = json.loads(path.read_text())
    payload[field] = replacement
    path.chmod(0o644)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    path.chmod(0o444)
    rewritten = {"path": str(path), "sha256": results.sha256_file(path)}

    with pytest.raises(ValueError, match=f"{field} mismatch"):
        results._validate_m_negative_selection_context(rewritten)


def test_m_negative_selection_context_accepts_conservative_verifier_error(
    tmp_path, monkeypatch
):
    def one_error(completion, gold):
        if completion.startswith("trained:0:1"):
            return {
                "reward": 0.0,
                "status": "verifier_error_zeroed",
                "verifier_stage": "symbolic_verify",
                "verifier_error_type": "SyntheticTimeout",
            }
        return fake_strict_verify(completion, gold)

    monkeypatch.setattr(results, "_legacy_m_strict_verdict", one_error)
    binding = synthetic_m_negative_selection_context(tmp_path)
    validated = results._validate_m_negative_selection_context(binding)

    assert validated["verifier_error_count"] == 1
    assert validated["strict_negative_replay_confirmed"]


def test_m_negative_selection_context_rejects_nonrobust_verifier_errors(
    tmp_path, monkeypatch
):
    def trained_errors(completion, gold):
        if completion.startswith("trained:"):
            return {
                "reward": 0.0,
                "status": "verifier_error_zeroed",
                "verifier_stage": "symbolic_verify",
                "verifier_error_type": "SyntheticTimeout",
            }
        return fake_strict_verify(completion, gold)

    monkeypatch.setattr(results, "_legacy_m_strict_verdict", trained_errors)
    with pytest.raises(ValueError, match="not robust to teacher-favorable"):
        synthetic_m_negative_selection_context(tmp_path)


def test_o_teacher_readout_uses_robust_bonferroni_envelope(tmp_path, monkeypatch):
    monkeypatch.setattr(results, "recompute_student_heldout_result", lambda gate: dict(gate))
    paths = o_teacher_paths(tmp_path)
    baseline_path = paths["baseline_O"]
    baseline = json.loads(baseline_path.read_text())
    baseline["record_accuracy_bounds_under_verifier_uncertainty"] = {
        record_id: [0.5, 1.0] for record_id in baseline["record_rewards"]
    }
    baseline_path.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n")

    readout = results.o_teacher_readout(
        paths, **o_teacher_preregistration(tmp_path, paths)
    )
    contrast = readout["primary_results"]["delta_O"]

    assert contrast["classification_without_verifier_uncertainty"] == "helps"
    assert contrast["classification"] == "inconclusive"
    assert contrast["verifier_uncertainty_sensitivity"]["estimate_bounds"] == [
        -0.25,
        0.25,
    ]
    assert contrast["verifier_uncertainty_sensitivity"][
        "bootstrap_97_5_fwer_envelope"
    ] == [-0.25, 0.25]


def test_o_teacher_readout_locks_paired_rng_order_and_bonferroni_quantiles(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(results, "recompute_student_heldout_result", lambda gate: dict(gate))
    paths = o_teacher_paths(tmp_path)
    count_rules = {
        "baseline_M": lambda index: index % 5,
        "O_M": lambda index: (3 * index + 1) % 5,
        "baseline_O": lambda index: (2 * index) % 5,
        "O_O": lambda index: (4 * index + 2) % 5,
    }
    per_record = {}
    for key, path in paths.items():
        gate = json.loads(path.read_text())
        source = gate["student_source"]
        rewards = {}
        means = []
        for index in range(370):
            correct = count_rules[key](index)
            vector = [1.0] * correct + [0.0] * (4 - correct)
            rewards[f"{source}:{index}"] = vector
            means.append(correct / 4)
        gate["records"] = 370
        gate["record_rewards"] = rewards
        gate["record_ids_sha256"] = results.canonical_json_sha256(sorted(rewards))
        gate["record_accuracy_bounds_under_verifier_uncertainty"] = {
            record_id: [sum(vector) / 4, sum(vector) / 4]
            for record_id, vector in rewards.items()
        }
        path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
        per_record[key] = [
            sum(rewards[record_id]) / 4 for record_id in sorted(rewards)
        ]

    readout = results.o_teacher_readout(
        paths, **o_teacher_preregistration(tmp_path, paths)
    )
    rng = random.Random(0)
    draws = {"M": [], "O": []}
    for _ in range(10_000):
        m_indices = [rng.randrange(370) for _ in range(370)]
        o_indices = [rng.randrange(370) for _ in range(370)]
        draws["M"].append(
            sum(per_record["O_M"][index] for index in m_indices) / 370
            - sum(per_record["baseline_M"][index] for index in m_indices) / 370
        )
        draws["O"].append(
            sum(per_record["O_O"][index] for index in o_indices) / 370
            - sum(per_record["baseline_O"][index] for index in o_indices) / 370
        )
    for source in ("M", "O"):
        ordered = sorted(draws[source])
        expected = [
            ordered[int(0.0125 * 9_999)],
            ordered[int(0.9875 * 9_999)],
        ]
        assert readout["primary_results"][f"delta_{source}"][
            "bootstrap_97_5_fwer_ci"
        ] == pytest.approx(expected, abs=1e-15)


def test_o_teacher_readout_rejects_wrong_arm_set_or_nonfixed_bootstrap(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(results, "recompute_student_heldout_result", lambda gate: dict(gate))
    paths = o_teacher_paths(tmp_path)
    preregistration = o_teacher_preregistration(tmp_path, paths)
    incomplete = dict(paths)
    incomplete.pop("O_O")
    with pytest.raises(ValueError, match="exactly"):
        results.o_teacher_readout(incomplete, **preregistration)
    extra = dict(paths)
    extra["M_M"] = paths["O_M"]
    with pytest.raises(ValueError, match="exactly"):
        results.o_teacher_readout(extra, **preregistration)
    with pytest.raises(ValueError, match="requires seed"):
        results.o_teacher_readout(paths, seed=1, **preregistration)
    with pytest.raises(ValueError, match="requires seed"):
        results.o_teacher_readout(paths, draws=9999, **preregistration)


def test_o_teacher_readout_rejects_posthoc_alternative_valid_gate(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(results, "recompute_student_heldout_result", lambda gate: dict(gate))
    paths = o_teacher_paths(tmp_path)
    preregistration = o_teacher_preregistration(tmp_path, paths)
    alternative = tmp_path / "O_O_alternative_valid_gate.json"
    alternative.write_bytes(paths["O_O"].read_bytes())
    selected = dict(paths)
    selected["O_O"] = alternative

    with pytest.raises(ValueError, match="not fixed by the sealed preregistration"):
        results.o_teacher_readout(selected, **preregistration)


def test_o_teacher_readout_rejects_unsealed_or_mutated_prelaunch_custody(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(results, "recompute_student_heldout_result", lambda gate: dict(gate))
    paths = o_teacher_paths(tmp_path)
    preregistration = o_teacher_preregistration(tmp_path, paths)
    preregistration["launch_ledger_path"].chmod(0o644)
    with pytest.raises(ValueError, match="launch ledger must be sealed read-only"):
        results.o_teacher_readout(paths, **preregistration)

    preregistration["launch_ledger_path"].chmod(0o444)
    preregistration["preregistration_path"].chmod(0o644)
    payload = json.loads(preregistration["preregistration_path"].read_text())
    payload["created_utc"] = "after-outcomes"
    preregistration["preregistration_path"].write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    preregistration["preregistration_path"].chmod(0o444)
    with pytest.raises(ValueError, match="hash differs from the launch ledger"):
        results.o_teacher_readout(paths, **preregistration)


def test_o_teacher_readout_requires_one_exact_teacher_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(results, "recompute_student_heldout_result", lambda gate: dict(gate))
    paths = o_teacher_paths(tmp_path)
    preregistration = o_teacher_preregistration(tmp_path, paths)
    payload = json.loads(paths["O_O"].read_text())
    payload["student_run_binding"]["teacher"]["merged_checkpoint"] = "wrong"
    paths["O_O"].write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="one exact teacher identity"):
        results.o_teacher_readout(paths, **preregistration)


def test_o_teacher_readout_rejects_same_path_teacher_substitution_after_sealing(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(results, "recompute_student_heldout_result", lambda gate: dict(gate))
    paths = o_teacher_paths(tmp_path)
    preregistration = o_teacher_preregistration(tmp_path, paths)
    teacher_gate = tmp_path / "teacher" / "O_teacher_gap.json"
    teacher_gate.chmod(0o644)
    teacher_gate.write_text('{"gate":"substituted-O"}\n')
    replacement_hash = results.sha256_file(teacher_gate)
    for key in ("O_M", "O_O"):
        gate = json.loads(paths[key].read_text())
        teacher = gate["student_run_binding"]["teacher"]
        teacher["teacher_gap_manifest_sha256"] = replacement_hash
        teacher["teacher_gap_payload_sha256"] = "f" * 64
        paths[key].write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="sealed preregistration"):
        results.o_teacher_readout(paths, **preregistration)


def test_o_teacher_readout_rejects_shared_support_substitution_after_sealing(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(results, "recompute_student_heldout_result", lambda gate: dict(gate))
    paths = o_teacher_paths(tmp_path)
    preregistration = o_teacher_preregistration(tmp_path, paths)
    replacement = {
        "manifest_sha256": "a" * 64,
        "payload_sha256": "b" * 64,
        "source": "M",
    }
    for key in ("baseline_M", "O_M"):
        gate = json.loads(paths[key].read_text())
        gate["student_run_binding"]["student_support"] = replacement
        paths[key].write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="support identity differs"):
        results.o_teacher_readout(paths, **preregistration)


def test_o_teacher_readout_requires_matched_training_sequence(tmp_path, monkeypatch):
    monkeypatch.setattr(results, "recompute_student_heldout_result", lambda gate: dict(gate))
    paths = o_teacher_paths(tmp_path)
    preregistration = o_teacher_preregistration(tmp_path, paths)
    payload = json.loads(paths["O_M"].read_text())
    payload["student_run_binding"]["trace"]["realized_record_ids_sha256"] = "f" * 64
    paths["O_M"].write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="realized training sequence"):
        results.o_teacher_readout(paths, **preregistration)


@pytest.mark.parametrize(
    ("key", "field", "replacement", "message"),
    (
        ("baseline_O", "train_freeze", {"path": "/wrong", "sha256": "0" * 64}, "train_freeze mismatch"),
        ("baseline_M", "serve_freeze", {"path": "/wrong", "sha256": "9" * 64}, "serve_freeze mismatch"),
        ("O_M", "serve_freeze", None, "lacks a validated teacher serve environment freeze"),
        ("O_O", "serve_freeze", {"path": "/wrong", "sha256": "9" * 64}, "serve_freeze mismatch"),
    ),
)
def test_o_teacher_readout_requires_exact_environment(
    tmp_path, monkeypatch, key, field, replacement, message
):
    monkeypatch.setattr(results, "recompute_student_heldout_result", lambda gate: dict(gate))
    paths = o_teacher_paths(tmp_path)
    preregistration = o_teacher_preregistration(tmp_path, paths)
    payload = json.loads(paths[key].read_text())
    payload["student_run_binding"]["environment"][field] = replacement
    paths[key].write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match=message):
        results.o_teacher_readout(paths, **preregistration)


def test_o_teacher_markdown_rejects_unknown_readout():
    with pytest.raises(ValueError, match="unknown O-teacher"):
        results.o_teacher_markdown({"readout": "wrong"})


def test_readout_outputs_cannot_mutate_bound_gate_or_adapter_trees(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(results, "recompute_student_heldout_result", lambda gate: dict(gate))
    paths = o_teacher_paths(tmp_path)
    readout = results.o_teacher_readout(
        paths, **o_teacher_preregistration(tmp_path, paths)
    )
    protected = results._readout_protected_trees(readout)
    baseline = json.loads(paths["baseline_M"].read_text())
    adapter = Path(baseline["student_run_binding"]["student_adapter"])

    with pytest.raises(ValueError, match="protected input tree"):
        results._preflight_result_outputs(
            [adapter / "readout.json"], protected_trees=protected
        )
    with pytest.raises(ValueError, match="protected input tree"):
        results._preflight_result_outputs(
            [paths["baseline_M"].parent / "readout.md"],
            protected_trees=protected,
        )
    preregistration = readout["preregistration"]
    with pytest.raises(ValueError, match="protected input tree"):
        results._preflight_result_outputs(
            [Path(preregistration["path"]).parent / "readout.json"],
            protected_trees=protected,
        )
    teacher_gate = Path(
        readout["custody"]["o_teacher_stable_artifact_identity"][
            "teacher_gap_manifest"
        ]
    )
    with pytest.raises(ValueError, match="protected input tree"):
        results._preflight_result_outputs(
            [teacher_gate.parent / "readout.json"], protected_trees=protected
        )


def test_o_teacher_cli_consumes_sealed_launch_ledger_and_fixed_outputs(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(results, "recompute_student_heldout_result", lambda gate: dict(gate))
    paths = o_teacher_paths(tmp_path)
    preregistration = o_teacher_preregistration(tmp_path, paths)
    prereg_payload = json.loads(Path(preregistration["preregistration_path"]).read_text())
    Path(prereg_payload["outputs"]["json"]).parent.mkdir(parents=True)
    monkeypatch.setattr(
        "sys.argv",
        [
            "student_results.py",
            "o-teacher-readout",
            "--baseline-m",
            str(paths["baseline_M"]),
            "--o-m",
            str(paths["O_M"]),
            "--baseline-o",
            str(paths["baseline_O"]),
            "--o-o",
            str(paths["O_O"]),
            "--preregistration",
            str(preregistration["preregistration_path"]),
            "--launch-ledger",
            str(preregistration["launch_ledger_path"]),
            "--output-json",
            prereg_payload["outputs"]["json"],
            "--output-markdown",
            prereg_payload["outputs"]["markdown"],
            "--output-manifest",
            prereg_payload["outputs"]["manifest"],
        ],
    )

    assert results.main() == 0
    written = json.loads(Path(prereg_payload["outputs"]["json"]).read_text())
    assert written["preregistration"]["launch_ledger"]["sealed_read_only"]
    assert Path(prereg_payload["outputs"]["markdown"]).is_file()
    manifest_path = Path(prereg_payload["outputs"]["manifest"])
    manifest = json.loads(manifest_path.read_text())
    assert manifest["result_json"]["sha256"] == results.sha256_file(
        Path(prereg_payload["outputs"]["json"])
    )
    assert manifest["result_markdown"]["sha256"] == results.sha256_file(
        Path(prereg_payload["outputs"]["markdown"])
    )
    for output in prereg_payload["outputs"].values():
        assert Path(output).stat().st_mode & 0o222 == 0


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


def test_readout_protection_covers_diagnostic_and_m_compatibility_inputs(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        results, "recompute_student_heldout_result", lambda gate: dict(gate)
    )
    paths = o_teacher_paths(tmp_path)
    preregistration = o_teacher_preregistration(tmp_path, paths)
    readout = results.o_teacher_readout(paths, **preregistration)
    protected = results._readout_protected_trees(readout)
    prereg = json.loads(Path(preregistration["preregistration_path"]).read_text())
    diagnostic_dir = Path(prereg["one_step_diagnostic"]["terminal_audit"]).parent
    compatibility_path = Path(prereg["selection_context"]["path"])
    compatibility = json.loads(compatibility_path.read_text())
    compatibility_dir = compatibility_path.parent
    base_eval_dir = Path(compatibility["inputs"]["base_summary"]["path"]).parent

    assert diagnostic_dir.resolve() in protected
    assert compatibility_dir.resolve() in protected
    for output in (
        diagnostic_dir / "result.json",
        compatibility_dir / "result.json",
        base_eval_dir / "result.json",
    ):
        with pytest.raises(ValueError, match="protected input tree"):
            results._preflight_result_outputs([output], protected_trees=protected)


def test_readout_bundle_failure_leaves_no_partial_outputs(tmp_path, monkeypatch):
    output_json = tmp_path / "result.json"
    output_markdown = tmp_path / "result.md"
    output_manifest = tmp_path / "result.manifest.json"
    original = results._write_new

    def fail_on_markdown(path, content):
        if Path(path) == output_markdown:
            raise OSError("synthetic Markdown publication failure")
        original(path, content)

    monkeypatch.setattr(results, "_write_new", fail_on_markdown)
    with pytest.raises(OSError, match="synthetic Markdown"):
        results._write_readout_bundle(
            output_json=output_json,
            json_content="{}\n",
            output_markdown=output_markdown,
            markdown_content="# result\n",
            output_manifest=output_manifest,
            payload={"readout": results.O_TEACHER_READOUT},
        )

    assert not output_json.exists()
    assert not output_markdown.exists()
    assert not output_manifest.exists()
