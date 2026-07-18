import pytest

from scripts.opd_math.data_contract import (
    ProblemRecord,
    balanced_boxed_answers,
    cluster_and_partition,
    math_answer_from_solution,
    normalize_format_insensitive,
    normalize_problem,
    records_from_math,
    records_from_openr1,
    resolve_semantic_reviews,
    semantic_near_duplicate_edges,
    token_shingles,
    validate_openr1_arrays,
)
from scripts.opd_math.prepare_data import (
    assert_output_parseability,
    filter_reward_parseable,
    register_pair_file,
)


def record(source, split, index, problem, *, answer="2", stratum="algebra"):
    return ProblemRecord(
        source=source,
        source_split=split,
        source_index=index,
        problem=problem,
        answer=answer,
        reference_solution=rf"\boxed{{{answer}}}",
        stratum=stratum,
    )


def openr1_row(**overrides):
    row = {
        "problem": "What is 1+1?",
        "solution": "Two.",
        "answer": "2",
        "problem_type": "Algebra",
        "question_type": "math-word-problem",
        "source": "olympiads",
        "uuid": None,
        "is_reasoning_complete": [True, True],
        "generations": ["g1", "g2"],
        "correctness_math_verify": [True, False],
        "correctness_llama": None,
        "finish_reasons": None,
        "correctness_count": 1,
        "messages": [
            {"role": "user", "content": "What is 1+1?"},
            {"role": "assistant", "content": "2"},
        ],
    }
    row.update(overrides)
    return row


def test_balanced_boxed_answer_handles_nested_latex():
    assert balanced_boxed_answers(r"first \boxed{1}, final \boxed {\frac{1}{2}}") == [
        "1",
        r"\frac{1}{2}",
    ]


def test_math_answer_accepts_pinned_unbraced_math_forms():
    assert math_answer_from_solution(r"Therefore the answer is \boxed 2.") == "2"
    assert math_answer_from_solution(r"Therefore the answer is \boxed 9.") == "9"


def test_normalization_preserves_math_but_collapses_unicode_whitespace():
    assert normalize_problem("  X + １  =  2\n") == "x + 1 = 2"


def test_format_normalization_keeps_semantic_whitespace():
    assert normalize_format_insensitive("Find a b") != normalize_format_insensitive("Find ab")
    assert normalize_format_insensitive(r"$\left(x\right)$") == normalize_format_insensitive("$(x)$")
    assert normalize_format_insensitive("f(6) - f(2) = 12") == normalize_format_insensitive(
        "f(6)-f(2)=12"
    )


def test_cross_source_and_test_collisions_are_quarantined():
    records = [
        record("M", "train", 0, "same problem"),
        record("O", "train", 0, "same   problem"),
        record("M", "test", 0, "test collision"),
        record("O", "train", 1, "test collision"),
        record("M", "train", 2, "unique M"),
        record("O", "train", 2, "unique O"),
    ]
    result = cluster_and_partition(records)
    assert result.stats["quarantine_rows_by_reason"] == {
        "cross_source_collision": 2,
        "touches_M_test": 1,
    }
    kept = sum(len(rows) for source in result.role_rows.values() for rows in source.values())
    assert kept == 2
    assert len(result.external_eval) == 1


def test_openr1_parallel_arrays_must_align():
    row = openr1_row(correctness_math_verify=[True])
    with pytest.raises(ValueError) as caught:
        validate_openr1_arrays(row, 7)
    assert "row 7" in str(caught.value)


def test_openr1_full_schema_and_null_policy_are_enforced():
    validate_openr1_arrays(openr1_row(), 0)
    with pytest.raises(ValueError, match="missing=.*uuid"):
        validate_openr1_arrays({key: value for key, value in openr1_row().items() if key != "uuid"}, 1)
    with pytest.raises(ValueError, match="answer must be a non-null string"):
        validate_openr1_arrays(openr1_row(answer=None), 2)
    with pytest.raises(ValueError, match="messages roles"):
        validate_openr1_arrays(
            openr1_row(
                messages=[
                    {"role": "assistant", "content": "2"},
                    {"role": "user", "content": "What is 1+1?"},
                ]
            ),
            3,
        )
    with pytest.raises(ValueError, match="does not match verifier flags"):
        validate_openr1_arrays(openr1_row(correctness_count=2), 4)


def test_ingestion_exclusions_are_row_addressable():
    rows = [
        {"problem": "good", "level": "Level 1", "solution": r"\boxed 2", "type": "Algebra"},
        {"problem": "bad", "level": "Level 1", "solution": r"\boxed{}", "type": "Algebra"},
    ]
    accepted, stats, excluded = records_from_math(rows, "train")
    assert len(accepted) == 1
    assert accepted[0].answer == "2"
    assert stats == {"seen": 2, "accepted": 1, "excluded": 1}
    assert excluded[0]["source_index"] == 1
    assert excluded[0]["problem"] == "bad"


def test_openr1_ingestion_retains_nullable_uuid_as_metadata():
    accepted, stats, excluded = records_from_openr1([openr1_row(uuid=None)])
    assert stats == {"seen": 1, "accepted": 1, "excluded": 0}
    assert not excluded
    assert accepted[0].source_metadata["uuid"] is None


def test_duplicate_multiplicity_does_not_change_cluster_or_role():
    one = cluster_and_partition([record("M", "train", 0, "duplicate")])
    two = cluster_and_partition(
        [record("M", "train", 0, "duplicate"), record("M", "train", 1, "duplicate")]
    )
    one_rows = [row for rows in one.role_rows["M"].values() for row in rows]
    two_rows = [row for rows in two.role_rows["M"].values() for row in rows]
    assert one_rows[0]["cluster_id"] == two_rows[0]["cluster_id"]
    assert one_rows[0]["role"] == two_rows[0]["role"]


def test_conflicting_duplicate_golds_are_quarantined_and_logged():
    result = cluster_and_partition(
        [
            record("O", "train", 0, "same problem", answer="2"),
            record("O", "train", 1, "same problem", answer="3"),
        ]
    )
    assert result.stats["quarantine_rows_by_reason"] == {"label_conflict": 2}
    assert not any(result.role_rows["O"].values())
    assert any(edge["edge_type"] == "label_conflict" for edge in result.collision_edges)


def test_m_test_cluster_has_one_canonical_row_and_persists_removed_members():
    result = cluster_and_partition(
        [
            record("M", "test", 0, "frozen problem"),
            record("M", "test", 1, "frozen problem"),
            record("M", "train", 2, "frozen problem"),
        ]
    )
    assert len(result.external_eval) == 1
    assert result.stats["duplicate_M_test_rows_removed"] == 1
    assert result.stats["quarantine_rows_by_reason"] == {
        "duplicate_M_test": 1,
        "touches_M_test": 1,
    }


def test_source_stratum_allocation_is_exact_and_order_invariant():
    records = [
        record("M", "train", index, f"unique algebra problem {index}", stratum="Algebra|Level 1")
        for index in range(20)
    ]
    forward = cluster_and_partition(records)
    reverse = cluster_and_partition(list(reversed(records)))
    assert forward.stats["role_counts"]["M"] == {
        "teacher_train": 12,
        "student_opd": 6,
        "teacher_gap_dev": 1,
        "source_holdout": 1,
    }
    forward_roles = {
        row["record_id"]: row["role"] for rows in forward.role_rows["M"].values() for row in rows
    }
    reverse_roles = {
        row["record_id"]: row["role"] for rows in reverse.role_rows["M"].values() for row in rows
    }
    assert forward_roles == reverse_roles


def test_semantic_candidates_fail_closed_until_reviewed():
    records = [
        record("M", "train", 0, "Find x when x plus 1 equals 2 and report the exact result."),
        record("O", "train", 1, "Find x when x plus 1 equals 2; report the exact answer."),
    ]
    edges, ledger, stats = semantic_near_duplicate_edges(
        records,
        candidate_threshold=0.01,
        quarantine_threshold=1.1,
        fingerprint_size=100,
    )
    assert not edges
    assert stats["review_required_edges"] == 1
    assert not stats["complete"]
    _, _, unresolved = resolve_semantic_reviews(records, edges, ledger)
    assert unresolved["unresolved_review_edges"] == 1
    resolved_edges, resolved_ledger, resolved = resolve_semantic_reviews(
        records,
        edges,
        ledger,
        [{"pair_id": ledger[0]["pair_id"], "decision": "duplicate"}],
    )
    assert resolved["review_complete"]
    assert resolved_edges == [(0, 1, "semantic_reviewed_duplicate")]
    assert resolved_ledger[0]["review_decision"] == "duplicate"


def test_semantic_scan_reports_oversized_bucket_as_incomplete():
    records = [
        record("M", "train", index, f"Find x when x plus 1 equals 2 and variant {index}.")
        for index in range(3)
    ]
    _, _, stats = semantic_near_duplicate_edges(
        records, candidate_threshold=0.01, fingerprint_size=100, max_bucket_size=1
    )
    assert stats["skipped_large_bucket_events"] > 0
    assert not stats["scan_complete"]
    assert not stats["complete"]


def test_global_prefix_finds_high_jaccard_pair_missed_by_fixed_bottom_eight():
    def alphabetic_token(index):
        suffix = ""
        value = index
        while True:
            suffix = chr(ord("a") + value % 26) + suffix
            value = value // 26 - 1
            if value < 0:
                return f"term{suffix}"

    base = [alphabetic_token(index) for index in range(500)]
    left_tokens = list(base)
    right_tokens = list(base)
    left_tokens[100], left_tokens[350] = "leftalpha", "leftbeta"
    right_tokens[100], right_tokens[350] = "rightalpha", "rightbeta"
    records = [
        record("M", "train", 0, " ".join(left_tokens)),
        record("O", "train", 1, " ".join(right_tokens)),
    ]

    left_shingles, right_shingles = [token_shingles(item.problem) for item in records]
    similarity = len(left_shingles & right_shingles) / len(left_shingles | right_shingles)
    assert similarity >= 0.95

    # The previous blocker selected only the eight lowest-document-frequency
    # shingles. Both records have ten edit-local shingles, so those two fixed
    # fingerprints are disjoint even though the full sets are >95% identical.
    frequency = {
        key: int(key in left_shingles) + int(key in right_shingles)
        for key in left_shingles | right_shingles
    }
    old_left = set(sorted(left_shingles, key=lambda key: (frequency[key], key))[:8])
    old_right = set(sorted(right_shingles, key=lambda key: (frequency[key], key))[:8])
    assert old_left.isdisjoint(old_right)

    edges, ledger, stats = semantic_near_duplicate_edges(
        records,
        candidate_threshold=0.95,
        quarantine_threshold=0.95,
        fingerprint_size=8,
    )
    assert edges == [(0, 1)]
    assert len(ledger) == 1
    assert ledger[0]["jaccard"] == pytest.approx(similarity)
    assert stats["candidate_pairs"] == 1
    assert stats["blocking_strategy"] == "global_document_frequency_prefix"
    assert stats["minimum_indexed_prefix_size"] > 8
    assert stats["scan_complete"]


def test_math_verify_filter_and_output_assertion_use_exact_boxed_gold():
    records = [
        record("M", "train", 0, "good", answer="2"),
        record("M", "train", 1, "bad", answer="bad"),
    ]

    def fake_parse(gold, *, extraction_mode):
        assert extraction_mode == "first_match"
        return [] if "bad" in gold else [gold]

    accepted, excluded, stats, cache = filter_reward_parseable(
        records, parse_fn=fake_parse, verifier_version="test"
    )
    assert len(accepted) == 1
    assert excluded[0]["record_id"] == records[1].record_id
    assert stats["parseable_records"] == 1
    clustered = cluster_and_partition(accepted)
    output = assert_output_parseability(clustered, parse_fn=fake_parse, cache=cache)
    assert output["all_parseable"]


def test_pair_file_registration_includes_exact_rows_and_hash():
    pair = {}
    files = {"roles/M/teacher_train.jsonl": {"rows": 7, "sha256": "abc"}}
    register_pair_file(pair, "teacher_train_file", "roles/M/teacher_train.jsonl", files)
    assert pair == {
        "teacher_train_file": "roles/M/teacher_train.jsonl",
        "teacher_train_file_rows": 7,
        "teacher_train_file_sha256": "abc",
    }
