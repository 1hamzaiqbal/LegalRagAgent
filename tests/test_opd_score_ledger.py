from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.opd_math import merge_adapter
from scripts.opd_math import score_ledger
from scripts.opd_math.quality_gates import sha256_file, sha256_tree


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def test_gold_only_symbolic_eligibility_rejects_prose_not_numeric_answers():
    numeric = score_ledger.classify_symbolic_gold(answer="2", solution=r"\boxed{2}")
    prose = score_ledger.classify_symbolic_gold(
        answer="z_{0}=1isanessentialsingularityofthefunctionf(z)",
        solution=r"\boxed{z_{0}=1isanessentialsingularityofthefunctionf(z)}",
    )

    assert numeric["eligible"] is True
    assert numeric["reasons"] == []
    assert prose["eligible"] is False
    assert "long_uncommanded_alpha_run" in prose["reasons"]
    assert "prose_dominant_registered_answer" in prose["reasons"]


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter.bin").write_bytes(b"adapter")

    task = tmp_path / "task.jsonl"
    task_rows = [
        {
            "record_id": f"O:{index}",
            "answer": str(index + 1),
            "solution": rf"\boxed{{{index + 1}}}",
        }
        for index in range(4)
    ]
    task_rows.append(
        {
            "record_id": "O:prose",
            "answer": "z_{0}=1isanessentialsingularityofthefunctionf(z)",
            "solution": r"\boxed{z_{0}=1isanessentialsingularityofthefunctionf(z)}",
        }
    )
    _write_jsonl(task, task_rows)

    def sample(record_id: str, reward: float, status: str) -> dict:
        completion = f"answer for {record_id}"
        return {
            "record_id": record_id,
            "sample_idx": 0,
            "reward": reward,
            "reward_status": status,
            "completion_text": completion,
            "completion_sha256": hashlib.sha256(completion.encode()).hexdigest(),
        }

    base_samples = tmp_path / "base_samples.jsonl"
    trained_samples = tmp_path / "trained_samples.jsonl"
    _write_jsonl(
        base_samples,
        [sample(row["record_id"], 0.0, "incorrect") for row in task_rows],
    )
    _write_jsonl(
        trained_samples,
        [sample(row["record_id"], 1.0, "correct") for row in task_rows],
    )

    bound_paths: dict[str, Path] = {
        "task_file": task,
        "base_samples": base_samples,
        "trained_samples": trained_samples,
    }
    for key in (
        "base_summary",
        "trained_summary",
        "prepared_manifest",
        "source_manifest",
        "teacher_run_manifest",
        "teacher_training_task_file",
        "teacher_training_plan",
        "teacher_trainer_state",
        "teacher_trainer_log_history",
        "teacher_train_metrics",
    ):
        path = tmp_path / f"{key}.json"
        _write_json(path, {"fixture": key})
        bound_paths[key] = path

    gate = {
        "schema_version": 3,
        "gate": "teacher_gap_v1",
        "gate_strength": "scientific",
        "passed": True,
        "authorizes_scientific_merge": True,
        "shared_records": len(task_rows),
        "min_records": 1,
        "min_delta": 0.0,
        "bootstrap_draws": 1_000,
        "bootstrap_seed": 0,
        "base_model": "Qwen/Qwen3-8B",
        "base_model_revision": "a" * 40,
        "trained_adapter": str(adapter.resolve()),
        "trained_adapter_tree_sha256": sha256_tree(adapter),
        "task_sources": ["O"],
        "task_roles": ["teacher_gap_dev"],
    }
    for key, path in bound_paths.items():
        gate[key] = str(path.resolve())
        gate[f"{key}_sha256"] = sha256_file(path)
    gate_path = tmp_path / "predecessor_gate.json"
    _write_json(gate_path, gate)

    adjudication = {
        "record_type": "opd_objective_family_manual_verifier_adjudication_v1",
        "decision_id": "fixture-base-adjudication",
        "scope": {
            "task_file_sha256": sha256_file(task),
            "base_samples": str(base_samples.resolve()),
            "base_samples_sha256": sha256_file(base_samples),
            "trained_samples": str(trained_samples.resolve()),
            "trained_samples_sha256": sha256_file(trained_samples),
        },
        "sample": {
            "record_id": "O:0",
            "sample_idx": 0,
            "completion_sha256": hashlib.sha256(b"answer for O:0").hexdigest(),
            "stored_reward": 0.0,
            "stored_reward_status": "incorrect",
        },
        "manual_math_check": {
            "verdict": "correct",
            "reasoning": ["fixture proof"],
        },
        "decision": {"classification": "MANUAL_CORRECT_BASE_SAMPLE"},
        "disclosure": {"post_hoc": True},
    }
    adjudication_path = tmp_path / "adjudication.json"
    _write_json(adjudication_path, adjudication)
    return gate_path, adjudication_path, adapter


def test_bundle_is_deterministically_recomputable_without_verifier(
    tmp_path, monkeypatch
):
    predecessor, adjudication, _adapter = _fixture(tmp_path)
    monkeypatch.setattr(score_ledger, "DEFAULT_TEACHER_MIN_RECORDS", 1)
    monkeypatch.setattr(
        score_ledger,
        "_git_state",
        lambda: {"commit": "b" * 40, "worktree_clean": True},
    )
    output = tmp_path / "bundle"

    result = score_ledger.build_bundle(
        predecessor_gate_path=predecessor,
        adjudication_paths=[adjudication],
        output_dir=output,
        min_eligible_coverage=0.75,
    )
    gate = json.loads((output / "gate.json").read_text())
    recomputed = score_ledger.recompute_score_ledger_gate(gate)

    assert result["passed"] is True
    assert gate == recomputed
    assert gate["total_registered_records"] == 5
    assert gate["eligible_records"] == 4
    assert gate["excluded_records"] == 1
    assert gate["eligible_coverage"] == 0.8
    assert gate["paired_delta"] == 0.75
    assert gate["measurement_policy"] == score_ledger.MEASUREMENT_POLICY
    ledger_rows = [json.loads(line) for line in (output / "score_ledger.jsonl").open()]
    adjudicated = next(row for row in ledger_rows if row["record_id"] == "O:0")
    excluded = next(row for row in ledger_rows if row["record_id"] == "O:prose")
    assert adjudicated["base"][0]["effective_reward"] == 1.0
    assert adjudicated["base"][0]["adjudication_id"] == "fixture-base-adjudication"
    assert excluded["eligibility"]["eligible"] is False

    output.chmod(0o755)
    for path in output.iterdir():
        path.chmod(0o644)


def test_merge_validation_accepts_only_exact_recomputed_score_ledger(
    tmp_path, monkeypatch
):
    predecessor, adjudication, adapter = _fixture(tmp_path)
    monkeypatch.setattr(score_ledger, "DEFAULT_TEACHER_MIN_RECORDS", 1)
    monkeypatch.setattr(
        score_ledger,
        "_git_state",
        lambda: {"commit": "b" * 40, "worktree_clean": True},
    )
    output = tmp_path / "bundle"
    score_ledger.build_bundle(
        predecessor_gate_path=predecessor,
        adjudication_paths=[adjudication],
        output_dir=output,
        min_eligible_coverage=0.75,
    )

    custody = score_ledger.validate_score_ledger_gate_for_merge(
        output / "gate.json",
        base_model="Qwen/Qwen3-8B",
        base_revision="a" * 40,
        adapter=adapter,
    )
    assert custody["manifest_sha256"] == sha256_file(output / "gate.json")

    output.chmod(0o755)
    for path in output.iterdir():
        path.chmod(0o644)


def test_merge_adapter_dispatches_score_ledger_gate_without_legacy_replay(
    tmp_path, monkeypatch
):
    gate_path = tmp_path / "gate.json"
    _write_json(gate_path, {"gate": score_ledger.SCORE_LEDGER_GATE_TYPE})
    expected = {"manifest": str(gate_path.resolve())}
    calls = []

    def validate(path, *, base_model, base_revision, adapter):
        calls.append((path, base_model, base_revision, adapter))
        return expected

    monkeypatch.setattr(
        merge_adapter, "validate_score_ledger_gate_for_merge", validate
    )
    result = merge_adapter.validate_teacher_gate_for_merge(
        gate_path,
        base_model="Qwen/Qwen3-8B",
        base_revision="a" * 40,
        adapter=tmp_path / "adapter",
    )

    assert result is expected
    assert calls == [
        (
            gate_path.resolve(),
            "Qwen/Qwen3-8B",
            "a" * 40,
            tmp_path / "adapter",
        )
    ]


def test_merge_provenance_uses_hash_bound_predecessor_training_custody(tmp_path):
    predecessor = tmp_path / "predecessor.json"
    predecessor_payload = {
        "gate": "teacher_gap_v1",
        "passed": True,
        "base_model": "Qwen/Qwen3-8B",
        "base_model_revision": "a" * 40,
        "trained_adapter": "/custody/O/final_adapter",
        "trained_adapter_tree_sha256": "b" * 64,
        "task_sources": ["O"],
        "task_roles": ["teacher_gap_dev"],
        "teacher_training_plan": "/custody/O/plan.json",
    }
    _write_json(predecessor, predecessor_payload)
    measurement = {
        "gate": score_ledger.SCORE_LEDGER_GATE_TYPE,
        "base_model": predecessor_payload["base_model"],
        "base_model_revision": predecessor_payload["base_model_revision"],
        "trained_adapter": predecessor_payload["trained_adapter"],
        "trained_adapter_tree_sha256": predecessor_payload[
            "trained_adapter_tree_sha256"
        ],
        "task_sources": predecessor_payload["task_sources"],
        "task_roles": predecessor_payload["task_roles"],
        "predecessor_gate": str(predecessor.resolve()),
        "predecessor_gate_sha256": sha256_file(predecessor),
    }

    source = merge_adapter.merge_provenance_source_gate(measurement)

    assert source == predecessor_payload

    measurement["trained_adapter_tree_sha256"] = "c" * 64
    try:
        merge_adapter.merge_provenance_source_gate(measurement)
    except ValueError as exc:
        assert "teacher identity differ" in str(exc)
    else:
        raise AssertionError("identity drift should fail closed")
