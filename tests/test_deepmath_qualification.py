import copy
import hashlib

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.opd_math.deepmath_qualification import (
    EXPECTED_COLUMNS,
    EXPECTED_SUCCESSOR_MATRIX,
    load_plan,
    validate_plan,
    verify_raw_files,
)


def _payload(plan):
    return {
        key: copy.deepcopy(value)
        for key, value in plan.items()
        if key not in {"path", "sha256", "canonical_sha256"}
    }


def test_plan_pins_candidate_gates_and_never_authorizes_training():
    plan = load_plan()
    assert plan["candidate_source"] == "C"
    assert plan["status"] == "data_only_pre_teacher_not_qualified"
    assert plan["plan_alone_authorizes_teacher_training"] is False
    assert plan["candidate"]["expected_rows"] == 103022
    assert tuple(plan["candidate"]["expected_columns"]) == EXPECTED_COLUMNS
    assert plan["collision_contract"]["allow_skipped_collision_buckets"] is False
    assert plan["data_gates"]["minimum_unique_eligible_clusters"] == 5000
    assert plan["data_gates"]["minimum_gold_parseability"] == 0.99
    assert plan["raw_model_feasibility"]["records"] == 512
    assert tuple(plan["successor_boundary"]["teacher_student_matrix"]) == EXPECTED_SUCCESSOR_MATRIX
    assert plan["successor_boundary"]["MATH_role"] == "external_transfer_target_only"
    assert plan["successor_boundary"]["M_teacher_allowed"] is False


@pytest.mark.parametrize(
    "mutator",
    [
        lambda payload: payload["candidate"].__setitem__("revision", "0" * 40),
        lambda payload: payload["candidate"]["raw_shards"][0].__setitem__("sha256", "0" * 64),
        lambda payload: payload["global_collision_inventory"][1].__setitem__("revision", "0" * 40),
        lambda payload: payload["collision_contract"].__setitem__("allow_skipped_collision_buckets", True),
        lambda payload: payload["data_gates"].__setitem__("minimum_gold_parseability", 0.5),
        lambda payload: payload["raw_model_feasibility"].__setitem__("minimum_student_pass_at_4", 0.0),
        lambda payload: payload["successor_boundary"].__setitem__("M_teacher_allowed", True),
    ],
)
def test_plan_rejects_outcome_adaptive_or_provenance_drift(mutator):
    payload = _payload(load_plan())
    mutator(payload)
    with pytest.raises(ValueError, match="drifted|cannot|required"):
        validate_plan(payload)


def test_raw_verification_checks_all_shard_hashes_schema_and_rows(tmp_path):
    plan = load_plan()
    synthetic = _payload(plan)
    synthetic["candidate"]["expected_rows"] = 10
    shards = []
    for index in range(10):
        path = tmp_path / "data" / f"train-{index:05d}-of-00010.parquet"
        path.parent.mkdir(exist_ok=True)
        table = pa.table(
            {
                "question": [f"question {index}"],
                "final_answer": [str(index)],
                "difficulty": [float(index)],
                "topic": ["algebra"],
                "r1_solution_1": ["forbidden trace one"],
                "r1_solution_2": ["forbidden trace two"],
                "r1_solution_3": ["forbidden trace three"],
            }
        )
        pq.write_table(table, path)
        shards.append(
            {
                "path": f"data/train-{index:05d}-of-00010.parquet",
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    synthetic["candidate"]["raw_shards"] = shards
    synthetic["sha256"] = plan["sha256"]
    synthetic["canonical_sha256"] = plan["canonical_sha256"]

    result = verify_raw_files(synthetic, tmp_path)
    assert result["status"] == "passed"
    assert result["rows"] == 10
    assert result["teacher_training_authorized"] is False
    assert result["scientific_use_allowed"] is False
    assert result["forbidden_training_fields"] == [
        "r1_solution_1",
        "r1_solution_2",
        "r1_solution_3",
    ]

    (tmp_path / "data" / "train-00000-of-00010.parquet").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="byte count drifted|SHA-256 drifted"):
        verify_raw_files(synthetic, tmp_path)
