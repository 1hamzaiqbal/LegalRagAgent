import json
from pathlib import Path

import pytest

from scripts.opd_math.materialize_deepmath_inventory import (
    DEFAULT_PLAN,
    EXPECTED_OUTPUT_COLUMNS,
    _normalized_row,
    load_inventory_plan,
)


def test_inventory_plan_pins_complete_source_surface():
    plan = load_inventory_plan(DEFAULT_PLAN)
    assert plan["expected_total_rows"] == 1_237_750
    assert [item["key"] for item in plan["sources"]] == [
        "C",
        "O",
        "O_lineage",
        "M_train",
        "M_test",
        "eval_math500",
        "eval_aime2024",
        "eval_aime_validation",
        "eval_amc_validation",
        "eval_math_beyond",
    ]
    assert tuple(plan["output_columns"]) == EXPECTED_OUTPUT_COLUMNS
    assert plan["teacher_training_authorized"] is False
    assert plan["scientific_use_allowed"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.update(expected_total_rows=10),
        lambda payload: payload.update(teacher_training_authorized=True),
        lambda payload: payload["sources"][0].update(revision="main"),
        lambda payload: payload["sources"][0]["forbidden_output_fields"].append("problem"),
        lambda payload: payload["sources"][1].update(expected_rows=1),
    ],
)
def test_inventory_plan_semantic_drift_fails_closed(tmp_path, mutation):
    payload = json.loads(DEFAULT_PLAN.read_text())
    mutation(payload)
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_inventory_plan(changed)


def test_candidate_normalization_omits_forbidden_traces_and_hashes_problem():
    spec = load_inventory_plan(DEFAULT_PLAN)["sources"][0]
    raw = {
        "question": "  Find  $x$.  ",
        "final_answer": "2",
        "difficulty": 5.0,
        "topic": "Algebra",
        "r1_solution_1": "forbidden",
        "r1_solution_2": "forbidden",
        "r1_solution_3": "forbidden",
    }
    row, missing = _normalized_row(spec, raw, 7)
    assert missing is False
    assert tuple(row) == EXPECTED_OUTPUT_COLUMNS
    assert row["problem"] == "Find  $x$."
    assert row["answer"] == "2"
    assert row["record_id"].startswith("C:train:")
    assert all("r1_solution" not in key for key in row)


def test_math_solution_answer_extraction_is_explicit():
    spec = load_inventory_plan(DEFAULT_PLAN)["sources"][3]
    row, missing = _normalized_row(
        spec,
        {
            "problem": "What is 1+1?",
            "solution": r"Therefore the answer is \boxed{2}.",
            "level": "Level 1",
            "type": "Algebra",
        },
        0,
    )
    assert missing is False
    assert row["answer"] == "2"

    _, missing = _normalized_row(
        spec,
        {
            "problem": "Prove the claim.",
            "solution": "No boxed final answer.",
            "level": "Level 1",
            "type": "Algebra",
        },
        1,
    )
    assert missing is True
