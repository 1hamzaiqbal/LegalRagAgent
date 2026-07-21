#!/usr/bin/env python3
"""Run the outcome-blind DeepMath global collision, label, and prompt audit."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import statistics
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

try:
    from .data_contract import (
        NORMALIZATION_VERSION,
        SEMANTIC_AUDIT_VERSION,
        UnionFind,
        boxed_gold,
        normalize_problem,
        prompt_messages,
        resolve_semantic_reviews,
        semantic_near_duplicate_edges,
        sha256_text,
        stable_rank,
        write_jsonl,
    )
    from .deepmath_qualification import load_plan as load_qualification_plan
    from .materialize_deepmath_inventory import (
        EXPECTED_OUTPUT_COLUMNS,
        EXPECTED_TOTAL_ROWS,
        _load_source_receipt,
        load_inventory_plan,
        sha256_file,
    )
except ImportError:
    from data_contract import (  # type: ignore
        NORMALIZATION_VERSION,
        SEMANTIC_AUDIT_VERSION,
        UnionFind,
        boxed_gold,
        normalize_problem,
        prompt_messages,
        resolve_semantic_reviews,
        semantic_near_duplicate_edges,
        sha256_text,
        stable_rank,
        write_jsonl,
    )
    from deepmath_qualification import load_plan as load_qualification_plan  # type: ignore
    from materialize_deepmath_inventory import (  # type: ignore
        EXPECTED_OUTPUT_COLUMNS,
        EXPECTED_TOTAL_ROWS,
        _load_source_receipt,
        load_inventory_plan,
        sha256_file,
    )


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUDIT_PLAN = ROOT / "configs/opd_math/deepmath_collision_audit_plan.json"
DEFAULT_QUALIFICATION_PLAN = ROOT / "configs/opd_math/deepmath_qualification_plan.json"
DEFAULT_INVENTORY_PLAN = ROOT / "configs/opd_math/deepmath_inventory_plan.json"
AUDIT_ID = "deepmath_C_global_collision_label_prompt_v1"


@dataclass(slots=True)
class AuditRecord:
    record_id: str
    source: str
    source_split: str
    source_index: int
    problem: str
    problem_missing: bool
    answer: str
    stratum: str
    is_evaluation: bool
    upstream_id: str
    exact_key: str
    format_key: str


def canonical_json_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_audit_plan(path: Path = DEFAULT_AUDIT_PLAN) -> dict[str, Any]:
    path = path.resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_top = {
        "schema_version",
        "audit_id",
        "qualification_id",
        "inventory_id",
        "status",
        "teacher_training_authorized",
        "scientific_use_allowed",
        "qualification_plan_sha256",
        "inventory_plan_sha256",
        "record_scope",
        "pair_scope",
        "exact_and_format",
        "semantic",
        "candidate_gates",
        "stage_rules",
    }
    _require(isinstance(payload, dict) and set(payload) == expected_top, "audit plan schema drifted")
    _require(payload["schema_version"] == 1 and payload["audit_id"] == AUDIT_ID, "audit plan identity drifted")
    _require(payload["qualification_id"] == "deepmath_C_data_feasibility_v1", "qualification ID drifted")
    _require(payload["inventory_id"] == "deepmath_C_global_inventory_v1", "inventory ID drifted")
    _require(payload["status"] == "outcome_blind_data_audit_not_teacher_authorization", "audit status drifted")
    _require(payload["teacher_training_authorized"] is False, "audit cannot authorize training")
    _require(payload["scientific_use_allowed"] is False, "audit plan cannot authorize science")
    _require(payload["record_scope"] == "all_1237750_materialized_rows", "record scope drifted")
    _require(payload["pair_scope"] == "all_source_pairs_global_document_frequency", "pair scope drifted")
    exact = payload["exact_and_format"]
    _require(exact == {
        "normalization_version": NORMALIZATION_VERSION,
        "format_normalization_version": "opd-math-format-v1",
        "label_conflict_policy": "quarantine_entire_candidate_C_cluster",
        "quarantine_any_cross_source_C_cluster": True,
        "quarantine_any_evaluation_touching_C_cluster": True,
        "retain_one_canonical_row_per_eligible_C_cluster": True,
    }, "exact/format audit contract drifted")
    semantic = payload["semantic"]
    _require(semantic == {
        "version": SEMANTIC_AUDIT_VERSION,
        "candidate_threshold": 0.85,
        "quarantine_threshold": 0.95,
        "numeric_signature_must_match": True,
        "fingerprint_size": 8,
        "max_bucket_size": 4096,
        "allow_skipped_bucket_events": False,
        "allow_unresolved_review_edges": False,
        "review_decisions": ["duplicate", "distinct"],
    }, "semantic audit contract drifted")
    gates = payload["candidate_gates"]
    _require(gates == {
        "minimum_unique_eligible_clusters": 5000,
        "minimum_gold_parseability": 0.99,
        "maximum_missing_candidate_problems": 0,
        "maximum_unresolved_label_conflicts": 0,
        "maximum_prompt_truncations": 0,
        "max_prompt_tokens": 1536,
        "tokenizer_model": "Qwen/Qwen3-1.7B",
        "tokenizer_revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        "thinking_mode": False,
        "prompt_template": "opd_math_prompt_messages_v1",
    }, "candidate data gates drifted")
    rules = payload["stage_rules"]
    _require(all(rules.get(key) is expected for key, expected in {
        "scan_can_authorize_teacher_training": False,
        "finalize_can_authorize_teacher_training": False,
        "role_freeze_required_after_data_pass": True,
        "raw_model_feasibility_required_after_data_pass": True,
        "C_teacher_gap_required_after_feasibility": True,
    }.items()), "audit stage rules drifted")
    return {
        **payload,
        "path": str(path),
        "sha256": sha256_file(path),
        "canonical_sha256": canonical_json_sha256(payload),
    }


def _git_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout
    if status.strip():
        raise ValueError("DeepMath audit requires a clean Git checkout")
    return {"commit": commit, "clean": True}


def _verify_inventory_manifest(
    manifest_path: Path,
    inventory_plan: Mapping[str, Any],
    qualification_plan: Mapping[str, Any],
    audit_plan: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(payload.get("schema_version") == 1, "inventory manifest schema drifted")
    _require(payload.get("inventory_id") == audit_plan["inventory_id"], "inventory manifest ID drifted")
    _require(payload.get("qualification_id") == audit_plan["qualification_id"], "inventory qualification ID drifted")
    _require(payload.get("status") == "passed", "inventory materialization did not pass")
    _require(payload.get("inventory_plan_sha256") == audit_plan["inventory_plan_sha256"], "audit/inventory plan hash drifted")
    _require(payload.get("inventory_plan_sha256") == inventory_plan["sha256"], "inventory plan bytes drifted")
    _require(payload.get("qualification_plan_sha256") == audit_plan["qualification_plan_sha256"], "audit/qualification plan hash drifted")
    _require(payload.get("qualification_plan_sha256") == qualification_plan["sha256"], "qualification plan bytes drifted")
    _require(payload.get("total_rows") == EXPECTED_TOTAL_ROWS, "inventory total row count drifted")
    _require(tuple(payload.get("output_columns", ())) == EXPECTED_OUTPUT_COLUMNS, "inventory output columns drifted")
    _require(payload.get("forbidden_C_fields_absent") is True, "C forbidden-field gate failed")
    _require(payload.get("teacher_training_authorized") is False, "inventory manifest cannot authorize training")
    _require(payload.get("scientific_use_allowed") is False, "inventory manifest cannot authorize science")
    outputs = payload.get("outputs")
    _require(isinstance(outputs, dict), "inventory manifest lacks outputs")
    manifest_git = payload.get("git")
    _require(
        isinstance(manifest_git, dict)
        and isinstance(manifest_git.get("commit"), str)
        and manifest_git.get("clean") is True,
        "inventory manifest lacks clean Git custody",
    )
    for spec in inventory_plan["sources"]:
        item = outputs.get(spec["key"])
        _require(isinstance(item, dict), f"inventory output missing: {spec['key']}")
        path = Path(item["path"]).resolve()
        _require(path.is_file() and not path.is_symlink(), f"inventory output missing or symlinked: {path}")
        _require(path.stat().st_size == item["bytes"], f"inventory output bytes drifted: {path}")
        _require(sha256_file(path) == item["sha256"], f"inventory output hash drifted: {path}")
        _require(item["rows"] == spec["expected_rows"], f"inventory output rows drifted: {path}")
        _require(tuple(item["columns"]) == EXPECTED_OUTPUT_COLUMNS, f"inventory output schema drifted: {path}")
        _load_source_receipt(
            path.with_name(f"{spec['key']}.receipt.json"),
            plan_sha256=inventory_plan["sha256"],
            git_commit=manifest_git["commit"],
            spec=spec,
            observed_output=item,
        )
    payload["path"] = str(manifest_path)
    payload["sha256"] = sha256_file(manifest_path)
    return payload


def _iter_parquet_rows(path: Path) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    if tuple(parquet.schema_arrow.names) != EXPECTED_OUTPUT_COLUMNS:
        raise ValueError(f"inventory record schema drifted: {path}")
    for batch in parquet.iter_batches(batch_size=4096):
        yield from batch.to_pylist()


def load_records(
    inventory_plan: Mapping[str, Any], inventory_manifest: Mapping[str, Any]
) -> tuple[list[AuditRecord], dict[str, int]]:
    records: list[AuditRecord] = []
    counts: dict[str, int] = {}
    for spec in inventory_plan["sources"]:
        output = inventory_manifest["outputs"][spec["key"]]
        count = 0
        for row in _iter_parquet_rows(Path(output["path"])):
            if set(row) != set(EXPECTED_OUTPUT_COLUMNS):
                raise ValueError("materialized inventory row schema drifted")
            record = AuditRecord(
                record_id=str(row["record_id"]),
                source=str(row["source"]),
                source_split=str(row["source_split"]),
                source_index=int(row["source_index"]),
                problem=str(row["problem"]),
                problem_missing=bool(row["problem_missing"]),
                answer=str(row["answer"]),
                stratum=str(row["stratum"]),
                is_evaluation=bool(row["is_evaluation"]),
                upstream_id=str(row["upstream_id"]),
                exact_key=str(row["canonical_problem_sha256"]),
                format_key=str(row["format_problem_sha256"]),
            )
            if record.source != spec["source"] or record.source_split != spec["split"]:
                raise ValueError(f"materialized inventory source identity drifted: {record.record_id}")
            if record.problem_missing != (not bool(record.problem.strip())):
                raise ValueError(f"materialized missing-problem flag drifted: {record.record_id}")
            if record.exact_key != sha256_text(normalize_problem(record.problem)):
                raise ValueError(f"materialized canonical hash drifted: {record.record_id}")
            records.append(record)
            count += 1
        if count != spec["expected_rows"]:
            raise ValueError(f"loaded inventory rows drifted for {spec['key']}")
        counts[spec["key"]] = count
    if len(records) != EXPECTED_TOTAL_ROWS:
        raise ValueError("loaded global inventory row count drifted")
    if len({record.record_id for record in records}) != len(records):
        raise ValueError("global inventory contains duplicate record IDs")
    return records, counts


def exact_format_edges(records: list[AuditRecord]) -> tuple[UnionFind, list[dict], dict[str, int]]:
    union = UnionFind(len(records))
    edges: list[dict] = []
    counts = {"exact": 0, "format_only": 0}
    first_exact: dict[str, int] = {}
    for index, record in enumerate(records):
        prior = first_exact.get(record.exact_key)
        if prior is None:
            first_exact[record.exact_key] = index
            continue
        union.union(prior, index)
        counts["exact"] += 1
        edges.append(
            {
                "edge_type": "exact",
                "left_record_id": records[prior].record_id,
                "right_record_id": record.record_id,
            }
        )
    first_format: dict[str, int] = {}
    for index, record in enumerate(records):
        prior = first_format.get(record.format_key)
        if prior is None:
            first_format[record.format_key] = index
            continue
        union.union(prior, index)
        if records[prior].exact_key != record.exact_key:
            counts["format_only"] += 1
            edges.append(
                {
                    "edge_type": "format_only",
                    "left_record_id": records[prior].record_id,
                    "right_record_id": record.record_id,
                }
            )
    return union, edges, counts


def _percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    index = min(len(values) - 1, max(0, math.ceil(fraction * len(values)) - 1))
    return sorted(values)[index]


def candidate_parseability(records: list[AuditRecord]) -> tuple[dict[str, Any], list[dict]]:
    from math_verify import parse
    from math_verify.errors import TimeoutException

    cache: dict[str, tuple[bool, str | None]] = {}
    failures = []
    candidate_rows = [record for record in records if record.source == "C"]
    for record in candidate_rows:
        answer = record.answer.strip()
        if answer not in cache:
            if not answer:
                cache[answer] = (False, "empty_answer")
            else:
                try:
                    parsed = parse(
                        boxed_gold(answer),
                        extraction_mode="first_match",
                        parsing_timeout=10,
                        raise_on_error=True,
                    )
                    cache[answer] = (bool(parsed), None if parsed else "parse_failed")
                except (Exception, TimeoutException) as exc:
                    cache[answer] = (False, type(exc).__name__)
        passed, reason = cache[answer]
        if not passed:
            failures.append(
                {
                    "record_id": record.record_id,
                    "answer_sha256": sha256_text(answer),
                    "reason": reason,
                }
            )
    parseable = len(candidate_rows) - len(failures)
    return {
        "candidate_rows": len(candidate_rows),
        "unique_answers": len(cache),
        "parseable_rows": parseable,
        "unparseable_rows": len(failures),
        "parseable_fraction": parseable / len(candidate_rows),
        "verifier": "math-verify",
        "verifier_version": importlib.metadata.version("math-verify"),
    }, failures


def candidate_prompt_surface(
    records: list[AuditRecord], gates: Mapping[str, Any], *, local_files_only: bool
) -> tuple[dict[str, Any], list[dict]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        gates["tokenizer_model"],
        revision=gates["tokenizer_revision"],
        local_files_only=local_files_only,
    )
    lengths: list[int] = []
    truncations = []
    for record in records:
        if record.source != "C":
            continue
        token_ids = tokenizer.apply_chat_template(
            prompt_messages(record.problem),
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if not isinstance(token_ids, list) or any(type(value) is not int for value in token_ids):
            raise ValueError(f"tokenizer returned invalid IDs: {record.record_id}")
        length = len(token_ids)
        lengths.append(length)
        if length > gates["max_prompt_tokens"]:
            truncations.append(
                {
                    "record_id": record.record_id,
                    "prompt_tokens": length,
                    "maximum": gates["max_prompt_tokens"],
                }
            )
    return {
        "candidate_rows": len(lengths),
        "tokenizer_model": gates["tokenizer_model"],
        "tokenizer_revision": gates["tokenizer_revision"],
        "thinking_mode": False,
        "max_prompt_tokens_allowed": gates["max_prompt_tokens"],
        "minimum_prompt_tokens": min(lengths),
        "median_prompt_tokens": statistics.median(lengths),
        "p95_prompt_tokens": _percentile(lengths, 0.95),
        "p99_prompt_tokens": _percentile(lengths, 0.99),
        "maximum_prompt_tokens": max(lengths),
        "prompt_truncations": len(truncations),
        "tokenizer_class": type(tokenizer).__name__,
    }, truncations


def _cluster_candidate_rows(
    records: list[AuditRecord],
    union: UnionFind,
    *,
    candidate_exclusions: Mapping[str, str] | None = None,
) -> tuple[list[int], list[dict], list[dict], dict[str, Any]]:
    candidate_exclusions = candidate_exclusions or {}
    candidate_roots = {union.find(index) for index, record in enumerate(records) if record.source == "C"}
    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        root = union.find(index)
        if root in candidate_roots:
            groups[root].append(index)

    eligible_indices: list[int] = []
    quarantine = []
    label_conflicts = []
    reasons: dict[str, int] = defaultdict(int)
    for indices in groups.values():
        members = [records[index] for index in indices]
        candidate_indices = [index for index in indices if records[index].source == "C"]
        cluster_id = sha256_text("|".join(sorted({record.exact_key for record in members})))
        sources = sorted({record.source for record in members})
        evaluation_touch = any(record.is_evaluation for record in members)
        normalized_answers = {
            normalize_problem(record.answer) for record in members if record.answer.strip()
        }
        label_conflict = len(normalized_answers) > 1
        candidate_missing_problem = any(
            record.source == "C" and record.problem_missing for record in members
        )
        if label_conflict:
            label_conflicts.append(
                {
                    "cluster_id": cluster_id,
                    "member_record_ids": sorted(record.record_id for record in members),
                    "sources": sources,
                    "answer_sha256s": sorted(sha256_text(value) for value in normalized_answers),
                    "resolution": "candidate_C_cluster_quarantined",
                }
            )
        if candidate_missing_problem:
            reason = "missing_problem"
        elif evaluation_touch:
            reason = "touches_evaluation"
        elif any(source != "C" for source in sources):
            reason = "cross_source_collision"
        elif label_conflict:
            reason = "label_conflict"
        else:
            reason = ""
        if reason:
            for index in candidate_indices:
                quarantine.append(
                    {
                        "record_id": records[index].record_id,
                        "cluster_id": cluster_id,
                        "reason": reason,
                        "cluster_sources": sources,
                        "evaluation_touch": evaluation_touch,
                        "label_conflict": label_conflict,
                    }
                )
                reasons[reason] += 1
            continue
        retained_candidate_indices = []
        for index in candidate_indices:
            record = records[index]
            exclusion_reason = candidate_exclusions.get(record.record_id)
            if exclusion_reason is None:
                retained_candidate_indices.append(index)
                continue
            quarantine.append(
                {
                    "record_id": record.record_id,
                    "cluster_id": cluster_id,
                    "reason": exclusion_reason,
                    "cluster_sources": sources,
                    "evaluation_touch": False,
                    "label_conflict": False,
                }
            )
            reasons[exclusion_reason] += 1
        if not retained_candidate_indices:
            continue
        canonical = min(
            retained_candidate_indices,
            key=lambda index: stable_rank(records[index].record_id, AUDIT_ID),
        )
        eligible_indices.append(canonical)
        for index in retained_candidate_indices:
            if index == canonical:
                continue
            quarantine.append(
                {
                    "record_id": records[index].record_id,
                    "cluster_id": cluster_id,
                    "reason": "within_C_duplicate",
                    "representative_record_id": records[canonical].record_id,
                    "cluster_sources": sources,
                    "evaluation_touch": False,
                    "label_conflict": False,
                }
            )
            reasons["within_C_duplicate"] += 1
    eligible_indices.sort(key=lambda index: stable_rank(records[index].record_id, AUDIT_ID))
    quarantine.sort(key=lambda row: (row["reason"], row["record_id"]))
    label_conflicts.sort(key=lambda row: row["cluster_id"])
    return eligible_indices, quarantine, label_conflicts, {
        "candidate_touching_clusters": len(groups),
        "eligible_unique_clusters": len(eligible_indices),
        "quarantined_candidate_rows": len(quarantine),
        "quarantine_rows_by_reason": dict(sorted(reasons.items())),
        "label_conflict_clusters": len(label_conflicts),
        "unresolved_label_conflicts": 0,
    }


def _candidate_output_row(record: AuditRecord) -> dict[str, Any]:
    return {
        "record_id": record.record_id,
        "source": record.source,
        "source_split": record.source_split,
        "source_index": record.source_index,
        "problem": record.problem,
        "problem_missing": record.problem_missing,
        "answer": record.answer,
        "stratum": record.stratum,
        "is_evaluation": record.is_evaluation,
        "upstream_id": record.upstream_id,
        "canonical_problem_sha256": record.exact_key,
        "format_problem_sha256": record.format_key,
    }


def _write_candidate_parquet(path: Path, records: list[AuditRecord], indices: list[int]) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite candidate output: {path}")
    table = pa.Table.from_pylist([_candidate_output_row(records[index]) for index in indices])
    table = table.select(list(EXPECTED_OUTPUT_COLUMNS))
    pq.write_table(table, path, compression="zstd")
    return {
        "path": str(path.resolve()),
        "rows": len(indices),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "columns": list(table.column_names),
    }


def _review_packet(ledger: Iterable[Mapping[str, Any]], record_by_id: Mapping[str, AuditRecord]) -> list[dict]:
    packet = []
    for item in ledger:
        if not item.get("requires_review"):
            continue
        left = record_by_id[item["left_record_id"]]
        right = record_by_id[item["right_record_id"]]
        packet.append(
            {
                "pair_id": item["pair_id"],
                "jaccard": item["jaccard"],
                "identical_numeric_sequence": item["identical_numeric_sequence"],
                "left": {
                    "record_id": left.record_id,
                    "source": left.source,
                    "problem": left.problem,
                    "answer": left.answer,
                },
                "right": {
                    "record_id": right.record_id,
                    "source": right.source,
                    "problem": right.problem,
                    "answer": right.answer,
                },
            }
        )
    packet.sort(key=lambda row: row["pair_id"])
    return packet


def scan(
    audit_plan: Mapping[str, Any],
    inventory_plan: Mapping[str, Any],
    qualification_plan: Mapping[str, Any],
    inventory_manifest: Mapping[str, Any],
    output_dir: Path,
    *,
    local_files_only: bool,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    try:
        output_dir.relative_to(ROOT.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("DeepMath audit output must live outside the Git checkout")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite DeepMath audit output: {output_dir}")
    output_dir.mkdir(parents=True)
    audit_dir = output_dir / "audit"
    records_dir = output_dir / "records"
    audit_dir.mkdir()
    records_dir.mkdir()
    git = _git_state()

    records, source_counts = load_records(inventory_plan, inventory_manifest)
    missing_problems_by_source: dict[str, int] = defaultdict(int)
    for record in records:
        missing_problems_by_source[record.source] += int(record.problem_missing)
    candidate_missing_problems = missing_problems_by_source.get("C", 0)
    parseability, parse_failures = candidate_parseability(records)
    prompt_surface, prompt_truncations = candidate_prompt_surface(
        records, audit_plan["candidate_gates"], local_files_only=local_files_only
    )
    union, deterministic_edges, deterministic_counts = exact_format_edges(records)
    semantic = audit_plan["semantic"]
    auto_edges, semantic_ledger, semantic_stats = semantic_near_duplicate_edges(
        records,
        candidate_threshold=semantic["candidate_threshold"],
        quarantine_threshold=semantic["quarantine_threshold"],
        fingerprint_size=semantic["fingerprint_size"],
        max_bucket_size=semantic["max_bucket_size"],
    )
    for left, right in auto_edges:
        union.union(left, right)
        deterministic_edges.append(
            {
                "edge_type": "semantic_auto_high_confidence",
                "left_record_id": records[left].record_id,
                "right_record_id": records[right].record_id,
            }
        )
    candidate_exclusions = {
        row["record_id"]: "unparseable_gold" for row in parse_failures
    }
    for row in prompt_truncations:
        candidate_exclusions[row["record_id"]] = "prompt_overflow"
    eligible_indices, quarantine, label_conflicts, cluster_stats = _cluster_candidate_rows(
        records,
        union,
        candidate_exclusions=candidate_exclusions,
    )
    record_by_id = {record.record_id: record for record in records}
    review_packet = _review_packet(semantic_ledger, record_by_id)

    files = {}
    for relative, rows in (
        ("audit/collision_edges.jsonl", deterministic_edges),
        ("audit/semantic_candidates.jsonl", semantic_ledger),
        ("audit/semantic_review_packet.jsonl", review_packet),
        ("audit/C_quarantine.jsonl", quarantine),
        ("audit/label_conflicts.jsonl", label_conflicts),
        ("audit/C_parse_failures.jsonl", parse_failures),
        ("audit/C_prompt_truncations.jsonl", prompt_truncations),
    ):
        path = output_dir / relative
        rows_written, digest = write_jsonl(path, rows)
        files[relative] = {
            "path": str(path.resolve()),
            "rows": rows_written,
            "bytes": path.stat().st_size,
            "sha256": digest,
        }
    candidate_output = _write_candidate_parquet(
        records_dir / "C_preliminary_eligible.parquet", records, eligible_indices
    )
    files["records/C_preliminary_eligible.parquet"] = candidate_output

    gates = audit_plan["candidate_gates"]
    blockers = []
    if not semantic_stats["scan_complete"]:
        blockers.append(
            f"semantic scan skipped {semantic_stats['skipped_large_bucket_events']} bucket events"
        )
    if semantic_stats["unresolved_review_edges"]:
        blockers.append(
            f"{semantic_stats['unresolved_review_edges']} semantic review edges unresolved"
        )
    if cluster_stats["eligible_unique_clusters"] < gates["minimum_unique_eligible_clusters"]:
        blockers.append("eligible unique C clusters below minimum")
    if parseability["parseable_fraction"] < gates["minimum_gold_parseability"]:
        blockers.append("C gold parseability below minimum")
    if candidate_missing_problems > gates["maximum_missing_candidate_problems"]:
        blockers.append("C missing problems exceed maximum")
    if cluster_stats["unresolved_label_conflicts"] > gates["maximum_unresolved_label_conflicts"]:
        blockers.append("unresolved C label conflicts exceed maximum")
    if prompt_surface["prompt_truncations"] > gates["maximum_prompt_truncations"]:
        blockers.append("C prompt truncations exceed maximum")
    data_gate_passed = not blockers
    payload = {
        "schema_version": 1,
        "audit_id": AUDIT_ID,
        "qualification_id": audit_plan["qualification_id"],
        "inventory_id": audit_plan["inventory_id"],
        "stage": "global_collision_label_prompt_scan",
        "status": "passed" if data_gate_passed else "blocked",
        "git": git,
        "audit_plan_path": audit_plan["path"],
        "audit_plan_sha256": audit_plan["sha256"],
        "audit_plan_canonical_sha256": audit_plan["canonical_sha256"],
        "qualification_plan_path": qualification_plan["path"],
        "qualification_plan_sha256": qualification_plan["sha256"],
        "inventory_plan_path": inventory_plan["path"],
        "inventory_plan_sha256": inventory_plan["sha256"],
        "inventory_manifest_path": inventory_manifest["path"],
        "inventory_manifest_sha256": inventory_manifest["sha256"],
        "source_counts": source_counts,
        "problem_quality": {
            "missing_problems_by_source": dict(sorted(missing_problems_by_source.items())),
            "missing_candidate_problems": candidate_missing_problems,
        },
        "total_rows": len(records),
        "deterministic_collision_counts": deterministic_counts,
        "semantic": semantic_stats,
        "candidate_clusters": cluster_stats,
        "gold_parseability": parseability,
        "prompt_surface": prompt_surface,
        "files": files,
        "data_gate_passed": data_gate_passed,
        "data_gate_blockers": blockers,
        "teacher_training_authorized": False,
        "scientific_use_allowed": False,
        "remaining_gates": [
            *([] if semantic_stats["unresolved_review_edges"] == 0 else ["complete semantic review"]),
            "freeze deterministic C roles after final data pass",
            "run fixed raw-model feasibility surface",
            "train and independently gap-qualify fresh C teacher",
        ],
        "runtime": {
            "python": sys.version.split()[0],
            "pyarrow": importlib.metadata.version("pyarrow"),
            "transformers": importlib.metadata.version("transformers"),
            "math_verify": importlib.metadata.version("math-verify"),
        },
        "launcher": (
            {
                "path": str(Path(os.environ["OPD_DEEPMATH_AUDIT_LAUNCHER_PATH"]).resolve()),
                "sha256": sha256_file(Path(os.environ["OPD_DEEPMATH_AUDIT_LAUNCHER_PATH"])),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID", "none"),
            }
            if os.environ.get("OPD_DEEPMATH_AUDIT_LAUNCHER_PATH")
            else None
        ),
    }
    manifest_path = output_dir / "audit_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for item in files.values():
        os.chmod(item["path"], 0o444)
    os.chmod(manifest_path, 0o444)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-plan", type=Path, default=DEFAULT_AUDIT_PLAN)
    parser.add_argument("--qualification-plan", type=Path, default=DEFAULT_QUALIFICATION_PLAN)
    parser.add_argument("--inventory-plan", type=Path, default=DEFAULT_INVENTORY_PLAN)
    parser.add_argument("--inventory-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit_plan = load_audit_plan(args.audit_plan)
    qualification_plan = load_qualification_plan(args.qualification_plan)
    inventory_plan = load_inventory_plan(args.inventory_plan)
    inventory_manifest = _verify_inventory_manifest(
        args.inventory_manifest,
        inventory_plan,
        qualification_plan,
        audit_plan,
    )
    result = scan(
        audit_plan,
        inventory_plan,
        qualification_plan,
        inventory_manifest,
        args.output_dir,
        local_files_only=args.local_files_only,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
