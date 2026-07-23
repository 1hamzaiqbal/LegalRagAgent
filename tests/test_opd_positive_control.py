from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.opd import materialize_positive_control as materialize
from scripts.opd import positive_control_gate as gate
from scripts.opd import prepare_opsd_execution_tree as execution_tree
from scripts.opd import verify_positive_control_environment as verify_environment


def test_identifiability_config_is_setup_only() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = json.loads(
        (root / "configs/opd_math/identifiability_v1.json").read_text()
    )
    assert payload["status"] == "setup_only_not_training_authorized"
    assert payload["immutable_boundaries"]["terminal_v2_is_not_reopened"]
    assert payload["immutable_boundaries"][
        "failed_M_teacher_is_never_retrained_merged_or_used"
    ]
    assert payload["positive_control"]["recipe"]["explicit_max_steps"] == 100
    assert payload["successor_cross_scale_pilot"]["status"] == (
        "blocked_on_positive_control"
    )
    assert payload["positive_control"]["stage_release_order"][3] == (
        "base_aime24_reproduction"
    )
    assert payload["positive_control"]["environment"]["vllm"] == "0.11.0"
    assert payload["positive_control"]["execution_hardware"][
        "base_and_all_checkpoint_evaluations_must_use_the_same_gpu_type"
    ]


def test_file_records_are_sorted_and_hashed(tmp_path: Path) -> None:
    (tmp_path / "b.txt").write_text("b")
    (tmp_path / "a.txt").write_text("a")
    records = materialize.file_records(tmp_path)
    assert [row["path"] for row in records] == ["a.txt", "b.txt"]
    assert records[0]["sha256"] == hashlib.sha256(b"a").hexdigest()


def test_pinned_positive_control_sources() -> None:
    assert materialize.TRAIN_REVISION == (
        "1f33e9dc2e8a1c639ca74f8024ad4a9f1f5eae62"
    )
    assert materialize.AIME_REVISION == (
        "2fe88a2f1091d5048c0f36abc874fb997b3dd99a"
    )
    assert materialize.TRAIN_ROWS == 29_434
    assert materialize.AIME_ROWS == 30


def test_gate_reconstructs_all_generations() -> None:
    payload = {
        "results": [
            {
                "generations": [
                    {"correct": generation < 6, "formatted": True}
                    for generation in range(12)
                ]
            }
            for _ in range(30)
        ]
    }
    rebuilt = gate.reconstruct(payload)
    assert rebuilt["generations"] == 360
    assert rebuilt["correct"] == 180
    assert rebuilt["average_at_12_fraction"] == 0.5


def test_execution_tree_edits_are_data_locality_only(tmp_path: Path) -> None:
    train = tmp_path / "opsd_train.py"
    train.write_text(execution_tree.TRAIN_OLD)
    record = execution_tree.replace_once(train, execution_tree.TRAIN_OLD, execution_tree.TRAIN_NEW)
    assert record["before_sha256"] != record["after_sha256"]
    assert "LEGALRAG_OPSD_TRAIN_PARQUET" in train.read_text()
    assert "__import__" in execution_tree.EVAL_NEW


def test_four_gpu_jobs_require_one_node() -> None:
    root = Path(__file__).resolve().parents[1]
    for name in (
        "slurm_opd_positive_control_preflight.sh",
        "slurm_opd_positive_control_base_eval.sh",
    ):
        source = (root / "scripts/hpc" / name).read_text()
        assert "#SBATCH --nodes=1" in source
        assert "#SBATCH --ntasks=1" in source
        assert "#SBATCH --gpus=a100-sxm4:4" in source


def test_resumed_submission_can_select_only_preregistered_gpu_types() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "scripts/hpc/submit_opd_positive_control_preflight.sh"
    ).read_text()
    assert "a100-sxm4|a6000" in source
    assert '--gpus="${GPU_TYPE}:4"' in source


def test_environment_receipt_is_exclusive_json(tmp_path: Path) -> None:
    output = tmp_path / "environment.json"
    verify_environment.write_exclusive(output, {"status": "passed"})
    assert json.loads(output.read_text()) == {"status": "passed"}
    assert output.stat().st_mode & 0o777 == 0o444
