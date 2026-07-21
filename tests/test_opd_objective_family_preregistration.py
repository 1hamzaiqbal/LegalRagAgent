import json
from argparse import Namespace

import pytest

from scripts.opd import objective_family_preregistration as prereg


def _arm(objective_id, source, seed, tmp_path):
    key = prereg.arm_key(objective_id, source, seed)
    root = tmp_path / key
    return {
        "objective_id": objective_id,
        "implementation": (
            "local" if objective_id in prereg.LOCAL_OBJECTIVE_IDS else "upstream_verl"
        ),
        "source": source,
        "seed": seed,
        "run_id": f"run.{key}",
        "prompt_plan_key": f"{source}_seed{seed}",
        "initial_adapter_key": f"seed{seed}",
        "training_out": str((root / "train").resolve()),
        "prelaunch_receipt": str((root / "prelaunch.json").resolve()),
        "heldout_gate": str((root / "heldout.json").resolve()),
    }


def test_arm_matrix_is_exactly_six_objectives_two_sources_three_seeds():
    assert len(prereg.EXPECTED_ARM_KEYS) == 36
    assert len(set(prereg.EXPECTED_ARM_KEYS)) == 36
    assert not any(key.startswith("M_M") or key.startswith("M_O") for key in prereg.EXPECTED_ARM_KEYS)


def test_prelaunch_receipt_binds_local_arm_and_rejects_run_id_drift(tmp_path, monkeypatch):
    objective_id = "task_rl_k1_ungated_clip5"
    key = prereg.arm_key(objective_id, "M", 1)
    arm = _arm(objective_id, "M", 1, tmp_path)
    receipt_path = tmp_path / "receipt.json"
    arm["prelaunch_receipt"] = str(receipt_path.resolve())
    prompt = {"path": "/inputs/M_seed1.json", "sha256": "a" * 64, "sequence_sha256": "b" * 64}
    initialization = {
        "manifest_path": "/inputs/seed1/manifest.json",
        "manifest_sha256": "c" * 64,
        "adapter_path": "/inputs/seed1/adapter",
        "adapter_tree_sha256": "d" * 64,
    }
    support = {"path": "/gates/M.json", "sha256": "e" * 64, "payload_sha256": "f" * 64, "source": "M"}
    teacher = {"teacher_source": "O"}
    prereg_path = tmp_path / "prereg.json"
    prereg_path.write_text("{}\n")
    launch_path = tmp_path / "launch.json"
    launch_path.write_text("{}\n")
    payload = {
        "campaign_id": "campaign-v1",
        "arms": {key: arm},
    }
    validated_prereg = {
        "path": str(prereg_path.resolve()),
        "sha256": "1" * 64,
        "payload": payload,
        "commit": "9" * 40,
        "student_support": {"M": support},
        "o_teacher": teacher,
        "prompt_plans": {"M_seed1": prompt},
        "initial_adapters": {"seed1": initialization},
    }
    validated_launch = {
        "path": str(launch_path.resolve()),
        "sha256": "2" * 64,
        "payload": {"arm_run_ids": {key: arm["run_id"]}},
    }
    out_dir = arm["training_out"]
    receipt = {
        "schema_version": 1,
        "receipt": prereg.PRELAUNCH_RECEIPT_ID,
        "sealed_before_optimizer_start": True,
        "campaign_id": "campaign-v1",
        "run_key": key,
        "run_id": arm["run_id"],
        "scheduler_job_id": "123",
        "objective_id": objective_id,
        "source": "M",
        "seed": 1,
        "git_commit": "9" * 40,
        "out_dir": out_dir,
        "expected_artifacts": {
            "run_manifest": f"{out_dir}/traces/run_manifest.json",
            "student_completion_manifest": f"{out_dir}/traces/completion_manifest.json",
            "student_adapter": f"{out_dir}/final",
            "prelaunch_receipt": str(receipt_path.resolve()),
        },
        "preregistration": {"path": str(prereg_path.resolve()), "sha256": "1" * 64},
        "launch_plan": {"path": str(launch_path.resolve()), "sha256": "2" * 64},
        "student_support": support,
        "o_teacher": teacher,
        "prompt_plan": prompt,
        "initial_adapter": initialization,
    }
    receipt_path.write_text(json.dumps(receipt) + "\n")
    receipt_path.chmod(0o444)
    monkeypatch.setattr(
        prereg,
        "_binding",
        lambda value, label, **kwargs: (
            prereg_path.resolve() if "preregistration" in label else launch_path.resolve(),
            {},
        ),
    )
    monkeypatch.setattr(prereg, "validate_preregistration", lambda path: validated_prereg)
    monkeypatch.setattr(
        prereg,
        "validate_launch_plan",
        lambda path, preregistration: validated_launch,
    )
    args = Namespace(
        prelaunch_receipt=str(receipt_path),
        objective_registry_contract={
            "objective": {"id": objective_id, "sampled_k1": True}
        },
        student_source="M",
        seed=1,
        out_dir=out_dir,
        campaign_run_id=arm["run_id"],
        scheduler_job_id="123",
    )
    binding = prereg.validate_prelaunch_receipt(args)
    assert binding["run_key"] == key

    args.campaign_run_id = "different"
    with pytest.raises(ValueError, match="run_id"):
        prereg.validate_prelaunch_receipt(args)
