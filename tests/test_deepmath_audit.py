import json

import pytest

from scripts.opd_math.audit_deepmath_inventory import (
    DEFAULT_AUDIT_PLAN,
    AuditRecord,
    _cluster_candidate_rows,
    exact_format_edges,
    load_audit_plan,
)
from scripts.opd_math.data_contract import normalize_format_insensitive, normalize_problem, sha256_text


def record(source, index, problem, answer, *, evaluation=False):
    return AuditRecord(
        record_id=f"{source}:train:{index}",
        source=source,
        source_split="train",
        source_index=index,
        problem=problem,
        problem_missing=not bool(problem.strip()),
        answer=answer,
        stratum="",
        is_evaluation=evaluation,
        upstream_id="",
        exact_key=sha256_text(normalize_problem(problem)),
        format_key=sha256_text(normalize_format_insensitive(problem)),
    )


def test_collision_audit_plan_is_fail_closed_and_hash_bound():
    plan = load_audit_plan(DEFAULT_AUDIT_PLAN)
    assert plan["record_scope"] == "all_1237750_materialized_rows"
    assert plan["pair_scope"] == "all_source_pairs_global_document_frequency"
    assert plan["semantic"]["max_bucket_size"] == 4096
    assert plan["semantic"]["allow_skipped_bucket_events"] is False
    assert plan["semantic"]["allow_unresolved_review_edges"] is False
    assert plan["teacher_training_authorized"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.update(teacher_training_authorized=True),
        lambda payload: payload.update(record_scope="C_only"),
        lambda payload: payload["semantic"].update(max_bucket_size=1),
        lambda payload: payload["semantic"].update(allow_skipped_bucket_events=True),
        lambda payload: payload["candidate_gates"].update(max_prompt_tokens=4096),
        lambda payload: payload["candidate_gates"].update(maximum_missing_candidate_problems=1),
    ],
)
def test_collision_audit_plan_mutations_fail_closed(tmp_path, mutation):
    payload = json.loads(DEFAULT_AUDIT_PLAN.read_text())
    mutation(payload)
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_audit_plan(changed)


def test_candidate_clusters_quarantine_cross_source_eval_and_label_conflicts():
    records = [
        record("C", 0, "Cross source problem 1+1", "2"),
        record("O", 0, "Cross source problem 1+1", "2"),
        record("C", 1, "Evaluation problem 2+2", "4"),
        record("eval_math500", 0, "Evaluation problem 2+2", "4", evaluation=True),
        record("C", 2, "Conflicting candidate", "5"),
        record("C", 3, "Conflicting candidate", "6"),
        record("C", 4, "Truly unique candidate", "7"),
        record("C", 5, "Formatting $ x + y $", "8"),
        record("C", 6, "Formatting $x+y$", "8"),
    ]
    union, edges, counts = exact_format_edges(records)
    eligible, quarantine, conflicts, stats = _cluster_candidate_rows(records, union)

    assert counts["exact"] == 3
    assert counts["format_only"] == 1
    assert len(edges) == 4
    eligible_ids = {records[index].record_id for index in eligible}
    assert "C:train:4" in eligible_ids
    assert len(eligible_ids & {"C:train:5", "C:train:6"}) == 1
    assert len(eligible_ids) == 2
    reasons = {row["record_id"]: row["reason"] for row in quarantine}
    assert reasons["C:train:0"] == "cross_source_collision"
    assert reasons["C:train:1"] == "touches_evaluation"
    assert reasons["C:train:2"] == "label_conflict"
    assert reasons["C:train:3"] == "label_conflict"
    assert len(
        [
            record_id
            for record_id in ("C:train:5", "C:train:6")
            if reasons.get(record_id) == "within_C_duplicate"
        ]
    ) == 1
    assert len(conflicts) == 1
    assert conflicts[0]["resolution"] == "candidate_C_cluster_quarantined"
    assert stats["unresolved_label_conflicts"] == 0


def test_missing_candidate_problem_is_quarantined():
    records = [record("C", 0, "", "2")]
    union, _, _ = exact_format_edges(records)
    eligible, quarantine, _, stats = _cluster_candidate_rows(records, union)
    assert eligible == []
    assert quarantine[0]["reason"] == "missing_problem"
    assert stats["quarantine_rows_by_reason"] == {"missing_problem": 1}
