import json

import pytest

from scripts.opd_math.finalize_semantic_reviews import POLICY, finalize


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_finalize_requires_complete_unique_reviews_and_applies_override(tmp_path):
    packet = tmp_path / "packet.jsonl"
    review = tmp_path / "review.jsonl"
    override = tmp_path / "override.jsonl"
    pair_a = "a" * 64
    pair_b = "b" * 64
    write_jsonl(packet, [{"pair_id": pair_a}, {"pair_id": pair_b}])
    write_jsonl(
        review,
        [
            {"pair_id": pair_a, "decision": "duplicate", "rationale": "same stem"},
            {"pair_id": pair_b, "decision": "distinct", "rationale": "first pass"},
        ],
    )
    write_jsonl(
        override,
        [{"pair_id": pair_b, "decision": "duplicate", "rationale": "shared diagram"}],
    )

    rows = finalize(packet, [review], [override])

    assert [row["pair_id"] for row in rows] == [pair_a, pair_b]
    assert rows[1]["initial_decision"] == "distinct"
    assert rows[1]["decision"] == "duplicate"
    assert rows[1]["reconciled"] is True
    assert rows[1]["review_policy"] == POLICY

    write_jsonl(review, [{"pair_id": pair_a, "decision": "duplicate", "rationale": "x"}])
    with pytest.raises(ValueError, match="omit"):
        finalize(packet, [review], [])


def test_finalize_rejects_nonhex_pair_ids(tmp_path):
    packet = tmp_path / "packet.jsonl"
    review = tmp_path / "review.jsonl"
    invalid = "z" * 64
    write_jsonl(packet, [{"pair_id": invalid}])
    write_jsonl(
        review,
        [{"pair_id": invalid, "decision": "duplicate", "rationale": "same item"}],
    )

    with pytest.raises(ValueError, match="invalid pair_id"):
        finalize(packet, [review], [])
