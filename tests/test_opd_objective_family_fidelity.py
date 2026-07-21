import hashlib
import json
from pathlib import Path

import pytest

from scripts.opd import objective_family_fidelity as fidelity
from scripts.opd.objective_family_inputs import sha256_file, sha256_tree
from scripts.opd.objective_registry import load_objective_registry


ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_FIXTURE = ROOT / "configs/opd_math/fidelity/shared_rollout_k1_v1.json"
COMMIT = "a" * 40


def _readonly_json(path, payload):
    path.write_text(json.dumps(payload) + "\n")
    path.chmod(0o444)
    return path


def test_completed_stdout_requires_exact_scheduler_name_and_terminal_marker(tmp_path):
    stdout = tmp_path / "opd_objective_family_123.out"
    stdout.write_text(
        "Objective-family run completed; held-out evaluation remains forbidden until release\n"
    )
    binding = fidelity._bind_completed_stdout(
        stdout, implementation="local", scheduler_job_id="123"
    )
    assert binding["terminal_marker_observed"] is True
    assert binding["sha256"] == sha256_file(stdout)
    assert stdout.stat().st_mode & 0o222 == 0

    wrong = tmp_path / "opd_objective_family_124.out"
    wrong.write_text("Objective-family run completed; held-out evaluation remains forbidden\n")
    with pytest.raises(ValueError, match="filename"):
        fidelity._bind_completed_stdout(
            wrong, implementation="local", scheduler_job_id="123"
        )


def test_diagnostic_receipt_reopens_run_tree_and_stdout(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "artifact.txt").write_text("stable\n")
    stdout = tmp_path / "opd_objective_family_123.out"
    stdout.write_text("done\n")
    stdout.chmod(0o444)
    registry = load_objective_registry()
    payload = {
        "schema_version": 1,
        "receipt": fidelity.DIAGNOSTIC_RECEIPT_ID,
        "status": "passed_plumbing",
        "scientific_use_allowed": False,
        "training_artifact_eligible_for_held_out_evaluation": False,
        "implementation": "local",
        "objective_id": "task_rl",
        "source": "M",
        "seed": 0,
        "git_commit": COMMIT,
        "objective_registry_sha256": registry["sha256"],
        "scheduler_job_id": "123",
        "run": {"root": str(run), "tree_sha256": sha256_tree(run)},
        "slurm_stdout": {
            "path": str(stdout),
            "sha256": sha256_file(stdout),
            "bytes": stdout.stat().st_size,
            "terminal_marker_observed": True,
        },
        "checks": {
            "exact_one_step_four_rollouts": True,
            "finite_nonzero_gradient": True,
            "finite_nonzero_parameter_update": True,
            "finite_nonzero_optimizer_state": True,
            "exact_prompt_and_initialization_bound": True,
            "o_teacher_only_when_required": True,
            "heldout_outcomes_inspected": False,
        },
        "claim_boundary": "plumbing only",
    }
    receipt = _readonly_json(tmp_path / "receipt.json", payload)
    validated = fidelity.validate_diagnostic_receipt(
        receipt, objective_id="task_rl", source="M", commit=COMMIT
    )
    assert validated["payload"]["status"] == "passed_plumbing"

    stdout.chmod(0o644)
    stdout.write_text("changed\n")
    with pytest.raises(ValueError, match="stdout drifted"):
        fidelity.validate_diagnostic_receipt(
            receipt, objective_id="task_rl", source="M", commit=COMMIT
        )


def test_real_model_receipt_requires_real_generated_fixture_and_all_checks(tmp_path):
    source = tmp_path / "samples.jsonl"
    run = tmp_path / "run.json"
    completion = tmp_path / "completion.json"
    for path in (source, run, completion):
        path.write_text("{}\n")
    fixture = json.loads(SYNTHETIC_FIXTURE.read_text())
    fixture["fixture_id"] = "real_model_rollout_k1_v1"
    fixture["status"] = "real_model_stored_tensor_fidelity_only"
    fixture["samples"] = [
        {**fixture["samples"][index % 2], "sample_id": f"real:{index}"}
        for index in range(4)
    ]
    registry = load_objective_registry()
    fixture["provenance"] = {
        "source_samples": str(source),
        "source_samples_sha256": sha256_file(source),
        "run_manifest": str(run),
        "run_manifest_sha256": sha256_file(run),
        "completion_manifest": str(completion),
        "completion_manifest_sha256": sha256_file(completion),
        "local_git_commit": COMMIT,
        "objective_registry_sha256": registry["sha256"],
        "student": "Qwen/Qwen3-1.7B",
        "student_revision": "b" * 40,
        "teacher_checkpoint": "/checkpoint",
        "teacher_checkpoint_tree_sha256": "c" * 64,
        "extractor_sha256": "d" * 64,
        "behavior_logprobs_origin": "generation_transition_scores_before_update",
        "current_student_logprobs_origin": "pre_update_student_forward_on_generated_tokens",
        "teacher_logprobs_origin": "frozen_o_teacher_exact_generated_token_scores",
        "heldout_outcomes_inspected": False,
    }
    fixture_path = _readonly_json(tmp_path / "fixture.json", fixture)
    checks = {
        "local_upstream_scalar_matches": True,
        "local_upstream_gradient_matches": True,
        "local_upstream_adamw_update_matches": True,
        "trace_reconstruction_matches": True,
        "on_policy_score_function_gradient_matches": True,
        "on_policy_score_function_gradient_cosine_pass": True,
        "masked_gradient_zero": True,
    }
    receipt_payload = {
        "schema_version": 1,
        "check_id": fidelity.REAL_RECEIPT_ID,
        "status": "pass",
        "scientific_launch_authorized": False,
        "coverage": {
            "real_model_generated_rollout": True,
            "behavior_scores_from_generation": True,
            "samples": 4,
            "valid_tokens": 10,
        },
        "comparison": checks,
        "custody": {
            "local": {
                "commit": COMMIT,
                "tracked_status": "clean",
                "fixture_path": str(fixture_path),
                "fixture_sha256": sha256_file(fixture_path),
            },
            "upstream_verl": {
                "commit": fidelity.UPSTREAM_VERL_COMMIT,
                "tracked_status": "clean",
            },
        },
    }
    receipt = _readonly_json(tmp_path / "real-receipt.json", receipt_payload)
    assert fidelity.validate_real_model_receipt(receipt, commit=COMMIT)["payload"]["status"] == "pass"

    receipt.chmod(0o644)
    receipt_payload["comparison"]["masked_gradient_zero"] = False
    _readonly_json(receipt, receipt_payload)
    with pytest.raises(ValueError, match="did not fully pass"):
        fidelity.validate_real_model_receipt(receipt, commit=COMMIT)


def test_closure_binding_parser_rejects_duplicates_and_requires_equal_sign():
    assert fidelity._parse_receipt_bindings(["task_rl__M=/tmp/a"]) == {
        "task_rl__M": "/tmp/a"
    }
    with pytest.raises(ValueError, match="invalid"):
        fidelity._parse_receipt_bindings(["task_rl__M=/tmp/a", "task_rl__M=/tmp/b"])
    with pytest.raises(ValueError, match="invalid"):
        fidelity._parse_receipt_bindings(["missing-separator"])
