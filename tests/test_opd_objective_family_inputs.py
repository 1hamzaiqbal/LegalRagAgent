import copy
import json
from pathlib import Path

import pytest

from scripts.opd.objective_family_inputs import (
    build_prompt_plan,
    canonical_json_sha256,
    sha256_file,
    sha256_tree,
    validate_initialization_manifest,
    validate_prompt_plan,
)
from scripts.opd_math.quality_gates import sha256_tree as established_sha256_tree


def _rows(source="M", count=120):
    return [
        {
            "record_id": f"{source}:train:{index}",
            "source": source,
            "role": "student_opd",
            "prompt": [{"role": "user", "content": f"problem {index}"}],
            "solution": f"answer {index}",
        }
        for index in range(count)
    ]


def _write_prompt_fixture(tmp_path: Path, monkeypatch):
    task = tmp_path / "student_opd.jsonl"
    rows = _rows()
    task.write_text("".join(json.dumps(row) + "\n" for row in rows))
    prepared = tmp_path / "prepared_manifest.json"
    prepared.write_text("{}\n")
    monkeypatch.setattr(
        "scripts.opd.objective_family_inputs.load_objective_registry",
        lambda: {"sha256": "r" * 64},
    )
    monkeypatch.setattr(
        "scripts.opd.objective_family_inputs.sha256_file",
        lambda path: {
            str(task): "t" * 64,
            str(task.resolve()): "t" * 64,
            str(prepared): "p" * 64,
            str(prepared.resolve()): "p" * 64,
        }.get(str(path), "s" * 64),
    )
    payload = build_prompt_plan(
        rows=rows,
        source="M",
        seed=1,
        task_file=task,
        prepared_manifest=prepared,
        git_commit="a" * 40,
    )
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return rows, task, prepared, path, payload


def test_prompt_plan_binds_exact_sequence_and_allows_only_one_step_diagnostic_prefix(
    tmp_path, monkeypatch
):
    rows, task, prepared, path, payload = _write_prompt_fixture(tmp_path, monkeypatch)
    contract, ordered = validate_prompt_plan(
        path,
        rows=rows,
        source="M",
        seed=1,
        task_file=task,
        prepared_manifest=prepared,
        git_commit="a" * 40,
        steps=100,
        diagnostic=False,
    )
    assert len(ordered) == 100
    assert contract["sequence_sha256"] == payload["sequence_sha256"]
    _, diagnostic = validate_prompt_plan(
        path,
        rows=rows,
        source="M",
        seed=1,
        task_file=task,
        prepared_manifest=prepared,
        git_commit="a" * 40,
        steps=1,
        diagnostic=True,
    )
    assert diagnostic == ordered[:1]

    tampered = copy.deepcopy(payload)
    tampered["sequence"][0]["record_id"] = "M:train:not-present"
    tampered["sequence_sha256"] = canonical_json_sha256(tampered["sequence"])
    path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="row identity"):
        validate_prompt_plan(
            path,
            rows=rows,
            source="M",
            seed=1,
            task_file=task,
            prepared_manifest=prepared,
            git_commit="a" * 40,
            steps=100,
            diagnostic=False,
        )


def test_initialization_manifest_rehashes_adapter_tree(tmp_path, monkeypatch):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"fixed")
    monkeypatch.setattr(
        "scripts.opd.objective_family_inputs.load_objective_registry",
        lambda: {"sha256": "r" * 64},
    )
    monkeypatch.setattr(
        "scripts.opd.objective_family_inputs.sha256_file",
        lambda path: "s" * 64 if str(path).endswith("objective_family_student_plan.json") else hashlib_sha(path),
    )
    payload = {
        "schema_version": 1,
        "initialization": "opd_math_objective_family_initial_adapter_v1",
        "status": "fixed_input_not_launch_authorization",
        "scientific_launch_authorized": False,
        "git_commit": "a" * 40,
        "objective_registry_sha256": "r" * 64,
        "student_training_plan_sha256": "s" * 64,
        "student": "Qwen/Qwen3-1.7B",
        "student_revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        "seed": 0,
        "lora": {
            "r": 32,
            "alpha": 64,
            "dropout": 0.0,
            "bias": "none",
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            "task_type": "CAUSAL_LM",
        },
        "adapter_path": str(adapter.resolve()),
        "adapter_tree_sha256": sha256_tree(adapter),
        "trainable_parameter_signature": {"elements": 2, "sum": 0.0, "squared_l2": 1.0},
    }
    manifest = tmp_path / "initialization_manifest.json"
    manifest.write_text(json.dumps(payload))
    contract = validate_initialization_manifest(
        manifest,
        student=payload["student"],
        student_revision=payload["student_revision"],
        seed=0,
        lora_r=32,
        git_commit="a" * 40,
    )
    assert contract["adapter_tree_sha256"] == payload["adapter_tree_sha256"]
    (adapter / "adapter_model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="tree drifted"):
        validate_initialization_manifest(
            manifest,
            student=payload["student"],
            student_revision=payload["student_revision"],
            seed=0,
            lora_r=32,
            git_commit="a" * 40,
        )


def hashlib_sha(path):
    return __import__("hashlib").sha256(Path(path).read_bytes()).hexdigest()


def test_tree_hash_is_identical_to_established_checkpoint_hash(tmp_path):
    tree = tmp_path / "adapter"
    (tree / "nested").mkdir(parents=True)
    (tree / "adapter_config.json").write_text("{}\n")
    (tree / "nested" / "weights.bin").write_bytes(b"weights")
    assert sha256_tree(tree) == established_sha256_tree(tree)

    (tree / "merge_provenance.json").write_text("{}\n")
    assert sha256_tree(
        tree, exclude_relative_paths=("merge_provenance.json",)
    ) == established_sha256_tree(
        tree, exclude_relative_paths=("merge_provenance.json",)
    )
