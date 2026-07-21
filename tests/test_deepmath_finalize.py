import json

import pytest

from scripts.opd_math.finalize_deepmath_audit import load_review_decisions


def test_review_decisions_are_hash_custodied_and_strict(tmp_path):
    path = tmp_path / "decisions.jsonl"
    path.write_text(
        json.dumps({"pair_id": "pair-b", "decision": "distinct"}) + "\n"
        + json.dumps({"pair_id": "pair-a", "decision": "duplicate"}) + "\n"
    )
    rows, receipt = load_review_decisions(path)
    assert rows[0] == {"pair_id": "pair-b", "decision": "distinct"}
    assert receipt["rows"] == 2
    assert len(receipt["sha256"]) == 64


@pytest.mark.parametrize(
    "row",
    [
        {"pair_id": "pair-a", "decision": "maybe"},
        {"pair_id": "pair-a", "decision": "duplicate", "note": "extra"},
        {"decision": "distinct"},
    ],
)
def test_review_decisions_fail_closed_on_schema_drift(tmp_path, row):
    path = tmp_path / "decisions.jsonl"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError):
        load_review_decisions(path)
