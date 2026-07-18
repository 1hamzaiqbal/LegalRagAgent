#!/usr/bin/env python3
"""Pure helpers for the pinned OPD-math data contract.

The primary unit is a problem cluster, not a dataset row.  Exact and
conservative formatting-only duplicates share a cluster so they cannot leak
between teacher training, student OPD, development, and holdout roles.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Mapping


NORMALIZATION_VERSION = "nfkc-casefold-ws-v1"
SEMANTIC_AUDIT_VERSION = "global-prefix-token5-jaccard-v2"
PARTITION_SALT = "legalrag-opd-math-v1"
MATH_COLUMNS = ("problem", "level", "solution", "type")
OPENR1_COLUMNS = (
    "problem",
    "solution",
    "answer",
    "problem_type",
    "question_type",
    "source",
    "uuid",
    "is_reasoning_complete",
    "generations",
    "correctness_math_verify",
    "correctness_llama",
    "finish_reasons",
    "correctness_count",
    "messages",
)
ROLE_RANGES = (
    ("teacher_train", 0, 59),
    ("student_opd", 60, 89),
    ("teacher_gap_dev", 90, 94),
    ("source_holdout", 95, 99),
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_problem(text: str) -> str:
    """Versioned exact-collision normalization.

    Preserve punctuation, numbers, and LaTeX commands; only normalize Unicode,
    case, line endings, and whitespace.  This intentionally favors precision
    over aggressive fuzzy matching.
    """
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return " ".join(text.split())


_SIZE_COMMAND = re.compile(r"\\(?:left|right|big|Big|bigg|Bigg)\b")
_LATEX_SPACING = re.compile(r"(?:\\[,!;:]|\\(?:quad|qquad)\b)")
_MATH_OPERATOR_WS = re.compile(r"\s*([=+\-*/^_(),{}\[\]<>])\s*")
_PUNCTUATION_WS = re.compile(r"\s+([,;:.!?])")


def normalize_format_insensitive(text: str) -> str:
    """Conservative formatting-only key used as a second collision edge."""
    text = normalize_problem(text)
    text = _SIZE_COMMAND.sub("", text)
    text = _LATEX_SPACING.sub("", text)
    text = text.replace("$", "")
    text = _MATH_OPERATOR_WS.sub(r"\1", text)
    text = _PUNCTUATION_WS.sub(r"\1", text)
    return " ".join(text.split())


_BRACED_BOX = re.compile(r"\\boxed\s*\{")
_UNBRACED_BOX = re.compile(
    r"\\boxed(?!\s*\{)\s+(-?(?:\d+(?:\.\d+)?|[A-Za-z]|\\[A-Za-z]+))\b"
)


def balanced_boxed_answers(solution: str) -> list[str]:
    """Extract balanced ``\\boxed {...}`` payloads, including nested braces."""
    text = str(solution)
    out: list[str] = []
    start = 0
    while True:
        match = _BRACED_BOX.search(text, start)
        if match is None:
            break
        content_start = match.end()
        depth = 1
        i = content_start
        while i < len(text) and depth:
            if text[i] == "{" and (i == 0 or text[i - 1] != "\\"):
                depth += 1
            elif text[i] == "}" and (i == 0 or text[i - 1] != "\\"):
                depth -= 1
            i += 1
        if depth == 0:
            out.append(text[content_start : i - 1].strip())
            start = i
        else:
            start = content_start
    return out


def math_answer_from_solution(solution: str) -> str | None:
    answers = balanced_boxed_answers(solution)
    if answers:
        return answers[-1] or None
    # MATH contains a small number of valid unbraced forms such as
    # ``\boxed 2``. Preserve the explicit atom rather than silently dropping it.
    unbraced = _UNBRACED_BOX.findall(str(solution))
    return unbraced[-1] if unbraced else None


def boxed_gold(answer: str) -> str:
    answer = str(answer).strip()
    return rf"\boxed{{{answer}}}"


def prompt_messages(problem: str) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": (
                "Solve the following math problem. Show your reasoning concisely, "
                "and put the final answer inside \\boxed{}.\n\n"
                f"Problem: {str(problem).strip()}"
            ),
        }
    ]


def role_for_cluster(cluster_id: str, salt: str = PARTITION_SALT) -> str:
    bucket = int(sha256_text(f"{salt}:{cluster_id}")[:16], 16) % 100
    for role, low, high in ROLE_RANGES:
        if low <= bucket <= high:
            return role
    raise AssertionError(f"partition bucket outside contract: {bucket}")


def stable_rank(record_id: str, salt: str = PARTITION_SALT) -> str:
    return sha256_text(f"{salt}:rank:{record_id}")


_MATH_TOKEN = re.compile(r"\\[A-Za-z]+|\d+(?:\.\d+)?|[A-Za-z]+|[^\s]")
_NUMBER_TOKEN = re.compile(r"^\d+(?:\.\d+)?$")


def math_tokens(text: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in _MATH_TOKEN.findall(normalize_problem(text)))


def token_shingles(text: str, width: int = 5) -> frozenset[int]:
    tokens = math_tokens(text)
    if not tokens:
        return frozenset()
    parts = (tokens,) if len(tokens) < width else (tokens[i : i + width] for i in range(len(tokens) - width + 1))
    return frozenset(int(sha256_text("\u241f".join(part))[:16], 16) for part in parts)


def numeric_signature(text: str) -> tuple[str, ...]:
    tokens = math_tokens(text)
    values: list[str] = []
    for index, token in enumerate(tokens):
        if not _NUMBER_TOKEN.fullmatch(token):
            continue
        if index > 0 and tokens[index - 1] in {"+", "-"}:
            token = tokens[index - 1] + token
        values.append(token)
    return tuple(values)


def semantic_pair_id(left_record_id: str, right_record_id: str) -> str:
    left, right = sorted((left_record_id, right_record_id))
    return sha256_text(f"{SEMANTIC_AUDIT_VERSION}:{left}\u241f{right}")


def semantic_near_duplicate_edges(
    records: list["ProblemRecord"],
    *,
    candidate_threshold: float = 0.85,
    quarantine_threshold: float = 0.95,
    fingerprint_size: int = 8,
    max_bucket_size: int = 256,
) -> tuple[list[tuple[int, int]], list[dict], dict]:
    """Find conservative math-token near duplicates with prefix filtering.

    High-confidence edges are eligible for clustering only when token-shingle
    Jaccard is at least ``quarantine_threshold`` and the ordered numeric token
    sequence is identical. Candidate-level edges are retained in the audit
    ledger for sensitivity analysis.

    Shingles are sorted by one global, document-frequency-based order. For a
    nonempty shingle set ``S`` and positive candidate threshold ``t``, the
    indexed prefix has at least ``|S| - ceil(t * |S|) + 1`` entries. Standard
    Jaccard prefix filtering then guarantees that every pair with Jaccard at
    least ``t`` shares an indexed shingle. ``fingerprint_size`` is retained as
    a minimum prefix size, not a cap. The only intentional loss of candidate
    recall is the explicit bucket bound, and any such skip marks the scan
    incomplete.
    """
    if not 0.0 < candidate_threshold <= 1.0:
        raise ValueError("candidate_threshold must be in (0, 1]")
    if fingerprint_size <= 0:
        raise ValueError("fingerprint_size must be positive")
    if max_bucket_size <= 0:
        raise ValueError("max_bucket_size must be positive")

    shingles = [token_shingles(record.problem) for record in records]
    numbers = [numeric_signature(record.problem) for record in records]
    document_frequency: dict[int, int] = defaultdict(int)
    for values in shingles:
        for key in values:
            document_frequency[key] += 1
    buckets: dict[int, list[int]] = defaultdict(list)
    candidates: set[tuple[int, int]] = set()
    skipped_large_buckets = 0
    recall_prefix_sizes: list[int] = []
    indexed_prefix_sizes: list[int] = []
    for i, values in enumerate(shingles):
        recall_prefix_size = (
            len(values) - math.ceil(candidate_threshold * len(values)) + 1 if values else 0
        )
        indexed_prefix_size = min(len(values), max(fingerprint_size, recall_prefix_size))
        recall_prefix_sizes.append(recall_prefix_size)
        indexed_prefix_sizes.append(indexed_prefix_size)
        ordered = sorted(values, key=lambda item: (document_frequency[item], item))
        for key in ordered[:indexed_prefix_size]:
            prior = buckets[key]
            if len(prior) < max_bucket_size:
                candidates.update((j, i) for j in prior)
            else:
                skipped_large_buckets += 1
            prior.append(i)

    union_edges: list[tuple[int, int]] = []
    ledger: list[dict] = []
    for left, right in sorted(candidates):
        if records[left].format_key == records[right].format_key:
            # Exact/format-only edges are handled by the deterministic collision
            # pass and do not need to inflate the semantic review surface.
            continue
        a, b = shingles[left], shingles[right]
        if not a or not b:
            continue
        similarity = len(a & b) / len(a | b)
        if similarity < candidate_threshold:
            continue
        same_numbers = numbers[left] == numbers[right]
        auto_quarantine = similarity >= quarantine_threshold and same_numbers
        requires_review = candidate_threshold <= similarity < quarantine_threshold and same_numbers
        pair_id = semantic_pair_id(records[left].record_id, records[right].record_id)
        ledger.append(
            {
                "pair_id": pair_id,
                "left_record_id": records[left].record_id,
                "right_record_id": records[right].record_id,
                "left_source": records[left].source,
                "right_source": records[right].source,
                "jaccard": similarity,
                "identical_numeric_sequence": same_numbers,
                "auto_clustered": auto_quarantine,
                "requires_review": requires_review,
                "review_decision": "duplicate" if auto_quarantine else None,
            }
        )
        if auto_quarantine:
            union_edges.append((left, right))

    review_required = sum(bool(item["requires_review"]) for item in ledger)
    scan_complete = skipped_large_buckets == 0
    stats = {
        "version": SEMANTIC_AUDIT_VERSION,
        "candidate_threshold": candidate_threshold,
        "quarantine_threshold": quarantine_threshold,
        "fingerprint_size": fingerprint_size,
        "blocking_strategy": "global_document_frequency_prefix",
        "recall_prefix_formula": "|S|-ceil(candidate_threshold*|S|)+1",
        "minimum_recall_prefix_size": min(recall_prefix_sizes, default=0),
        "maximum_recall_prefix_size": max(recall_prefix_sizes, default=0),
        "minimum_indexed_prefix_size": min(indexed_prefix_sizes, default=0),
        "maximum_indexed_prefix_size": max(indexed_prefix_sizes, default=0),
        "indexed_shingle_assignments": sum(indexed_prefix_sizes),
        "candidate_pairs": len(candidates),
        "review_edges": len(ledger),
        "auto_cluster_edges": len(union_edges),
        "review_required_edges": review_required,
        "resolved_review_edges": 0,
        "unresolved_review_edges": review_required,
        "skipped_large_bucket_events": skipped_large_buckets,
        "scan_complete": scan_complete,
        "complete": scan_complete and review_required == 0,
    }
    return union_edges, ledger, stats


def resolve_semantic_reviews(
    records: list["ProblemRecord"],
    auto_edges: list[tuple[int, int]],
    ledger: list[dict],
    review_rows: Iterable[Mapping] = (),
) -> tuple[list[tuple[int, int, str]], list[dict], dict]:
    """Resolve all review-required semantic pairs by stable ``pair_id``.

    The preparation manifest remains fail-closed when a required decision is
    absent. Unknown, duplicate, or malformed decisions are hard errors so a
    stale review file cannot silently approve a different dataset revision.
    """
    record_index = {record.record_id: index for index, record in enumerate(records)}
    required = {item["pair_id"]: item for item in ledger if item.get("requires_review")}
    decisions: dict[str, str] = {}
    for row_number, row in enumerate(review_rows, 1):
        pair_id = row.get("pair_id")
        decision = row.get("decision")
        if not isinstance(pair_id, str) or pair_id not in required:
            raise ValueError(f"semantic review row {row_number}: unknown pair_id {pair_id!r}")
        if pair_id in decisions:
            raise ValueError(f"semantic review row {row_number}: duplicate decision for {pair_id}")
        if decision not in {"duplicate", "distinct"}:
            raise ValueError(
                f"semantic review row {row_number}: decision must be 'duplicate' or 'distinct'"
            )
        decisions[pair_id] = decision

    resolved_edges = [
        (left, right, "semantic_auto_high_confidence") for left, right in auto_edges
    ]
    updated_ledger: list[dict] = []
    for item in ledger:
        updated = dict(item)
        if item.get("requires_review"):
            decision = decisions.get(item["pair_id"])
            updated["review_decision"] = decision
            if decision == "duplicate":
                resolved_edges.append(
                    (
                        record_index[item["left_record_id"]],
                        record_index[item["right_record_id"]],
                        "semantic_reviewed_duplicate",
                    )
                )
        updated_ledger.append(updated)

    unresolved = len(required) - len(decisions)
    stats = {
        "resolved_review_edges": len(decisions),
        "reviewed_duplicate_edges": sum(value == "duplicate" for value in decisions.values()),
        "reviewed_distinct_edges": sum(value == "distinct" for value in decisions.values()),
        "unresolved_review_edges": unresolved,
        "review_complete": unresolved == 0,
    }
    return sorted(set(resolved_edges)), updated_ledger, stats


@dataclass
class ProblemRecord:
    source: str
    source_split: str
    source_index: int
    problem: str
    answer: str
    reference_solution: str
    stratum: str
    source_metadata: dict = field(default_factory=dict)

    @property
    def exact_key(self) -> str:
        return sha256_text(normalize_problem(self.problem))

    @property
    def format_key(self) -> str:
        return sha256_text(normalize_format_insensitive(self.problem))

    @property
    def record_id(self) -> str:
        return f"{self.source}:{self.source_split}:{self.exact_key}:{self.source_index}"

    def task_row(self, cluster_id: str, role: str) -> dict:
        return {
            "record_id": self.record_id,
            "cluster_id": cluster_id,
            "question_sha256": self.exact_key,
            "format_sha256": self.format_key,
            "source": self.source,
            "source_split": self.source_split,
            "source_index": self.source_index,
            "stratum": self.stratum,
            "role": role,
            "problem": self.problem,
            "answer": self.answer,
            "solution": boxed_gold(self.answer),
            "reference_solution": self.reference_solution,
            "prompt": prompt_messages(self.problem),
            "source_metadata": self.source_metadata,
        }


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


@dataclass
class ClusteredData:
    role_rows: dict[str, dict[str, list[dict]]]
    external_eval: list[dict]
    quarantined: list[dict]
    collision_edges: list[dict]
    stats: dict


def _stratified_roles(
    eligible: list[tuple[ProblemRecord, str]], salt: str
) -> tuple[dict[str, dict[str, list[dict]]], dict]:
    roles: dict[str, dict[str, list[dict]]] = {
        source: {name: [] for name, _, _ in ROLE_RANGES} for source in ("M", "O")
    }
    weights = {"teacher_train": 0.60, "student_opd": 0.30, "teacher_gap_dev": 0.05, "source_holdout": 0.05}
    strata: dict[tuple[str, str], list[tuple[ProblemRecord, str]]] = defaultdict(list)
    for record, cluster_id in eligible:
        strata[(record.source, record.stratum)].append((record, cluster_id))

    allocation: dict[str, dict] = {}
    for (source, stratum), items in sorted(strata.items()):
        items.sort(key=lambda item: stable_rank(item[1], salt))
        raw = {role: len(items) * weight for role, weight in weights.items()}
        counts = {role: int(value) for role, value in raw.items()}
        remaining = len(items) - sum(counts.values())
        order = sorted(weights, key=lambda role: (-(raw[role] - counts[role]), role))
        for role in order[:remaining]:
            counts[role] += 1
        offset = 0
        for role in weights:
            for record, cluster_id in items[offset : offset + counts[role]]:
                roles[source][role].append(record.task_row(cluster_id, role))
            offset += counts[role]
        allocation[f"{source}|{stratum}"] = {"eligible": len(items), **counts}
    return roles, allocation


def cluster_and_partition(
    records: list[ProblemRecord],
    salt: str = PARTITION_SALT,
    semantic_edges: list[tuple[int, int] | tuple[int, int, str]] | None = None,
) -> ClusteredData:
    """Cluster, quarantine contamination, deduplicate, and assign train roles."""
    uf = UnionFind(len(records))
    collision_edges: list[dict] = []
    for key_name in ("exact_key", "format_key"):
        first: dict[str, int] = {}
        for i, record in enumerate(records):
            key = getattr(record, key_name)
            if key in first:
                left = first[key]
                uf.union(i, left)
                if key_name == "exact_key" or records[left].exact_key != record.exact_key:
                    collision_edges.append(
                        {
                            "edge_type": "exact" if key_name == "exact_key" else "format_only",
                            "left_record_id": records[left].record_id,
                            "right_record_id": record.record_id,
                        }
                    )
            else:
                first[key] = i
    for edge in semantic_edges or []:
        left, right = edge[:2]
        edge_type = edge[2] if len(edge) == 3 else "semantic_high_confidence"
        if not (0 <= left < len(records) and 0 <= right < len(records)):
            raise ValueError(f"semantic edge outside record range: {(left, right)}")
        uf.union(left, right)
        collision_edges.append(
            {
                "edge_type": edge_type,
                "left_record_id": records[left].record_id,
                "right_record_id": records[right].record_id,
            }
        )

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(records)):
        groups[uf.find(i)].append(i)

    external_eval: list[dict] = []
    quarantined: list[dict] = []
    eligible: list[tuple[ProblemRecord, str]] = []
    reasons: dict[str, int] = defaultdict(int)
    duplicate_rows = 0
    duplicate_eval_rows = 0
    cluster_by_record_id: dict[str, str] = {}
    record_by_id = {record.record_id: record for record in records}

    for indices in groups.values():
        members = [records[i] for i in indices]
        # Duplicate multiplicity must not change the cluster identity or its
        # deterministic role assignment.
        cluster_id = sha256_text("|".join(sorted({r.exact_key for r in members})))
        cluster_by_record_id.update({record.record_id: cluster_id for record in members})
        has_m_test = any(r.source == "M" and r.source_split == "test" for r in members)
        train_sources = {r.source for r in members if r.source_split == "train"}
        m_test_members = [r for r in members if r.source == "M" and r.source_split == "test"]
        train_members = [r for r in members if r.source_split == "train"]
        m_test_answer_keys = {normalize_problem(r.answer) for r in m_test_members}
        train_answer_keys = {normalize_problem(r.answer) for r in train_members}

        if len(m_test_answer_keys) > 1:
            collision_edges.append(
                {
                    "edge_type": "label_conflict",
                    "cluster_id": cluster_id,
                    "conflict_scope": "M_test",
                    "member_record_ids": sorted(record.record_id for record in members),
                    "answer_sha256s": sorted({sha256_text(key) for key in m_test_answer_keys}),
                }
            )
            for record in members:
                row = record.task_row(cluster_id, "quarantine")
                row["quarantine_reason"] = "label_conflict_M_test"
                quarantined.append(row)
                reasons["label_conflict_M_test"] += 1
            continue

        if m_test_members:
            canonical_test = min(m_test_members, key=lambda r: stable_rank(r.record_id, salt))
            external_eval.append(canonical_test.task_row(cluster_id, "external_eval"))
            duplicate_eval_rows += len(m_test_members) - 1
            for duplicate in m_test_members:
                if duplicate.record_id == canonical_test.record_id:
                    continue
                row = duplicate.task_row(cluster_id, "quarantine")
                row["quarantine_reason"] = "duplicate_M_test"
                row["representative_record_id"] = canonical_test.record_id
                quarantined.append(row)
                reasons["duplicate_M_test"] += 1

        if not train_members:
            continue
        if has_m_test:
            test_answer_key = next(iter(m_test_answer_keys))
            label_conflict = any(key != test_answer_key for key in train_answer_keys)
            reason = "touches_M_test_label_conflict" if label_conflict else "touches_M_test"
            if label_conflict:
                collision_edges.append(
                    {
                        "edge_type": "label_conflict",
                        "cluster_id": cluster_id,
                        "conflict_scope": "train_vs_M_test",
                        "member_record_ids": sorted(record.record_id for record in members),
                        "answer_sha256s": sorted(
                            {sha256_text(key) for key in m_test_answer_keys | train_answer_keys}
                        ),
                    }
                )
        elif len(train_answer_keys) > 1:
            reason = "label_conflict"
            collision_edges.append(
                {
                    "edge_type": "label_conflict",
                    "cluster_id": cluster_id,
                    "conflict_scope": "train",
                    "member_record_ids": sorted(record.record_id for record in train_members),
                    "answer_sha256s": sorted({sha256_text(key) for key in train_answer_keys}),
                }
            )
        elif len(train_sources) > 1:
            reason = "cross_source_collision"
        else:
            reason = ""

        if reason:
            for record in train_members:
                row = record.task_row(cluster_id, "quarantine")
                row["quarantine_reason"] = reason
                quarantined.append(row)
                reasons[reason] += 1
            continue

        # One canonical row per source/cluster. Stable rank makes the choice
        # invariant to the order in which Hub shards are loaded.
        canonical = min(train_members, key=lambda r: stable_rank(r.record_id, salt))
        duplicate_rows += len(train_members) - 1
        eligible.append((canonical, cluster_id))
        for duplicate in train_members:
            if duplicate.record_id == canonical.record_id:
                continue
            row = duplicate.task_row(cluster_id, "quarantine")
            row["quarantine_reason"] = "within_source_duplicate"
            row["representative_record_id"] = canonical.record_id
            quarantined.append(row)
            reasons["within_source_duplicate"] += 1

    roles, stratum_allocation = _stratified_roles(eligible, salt)

    for source_roles in roles.values():
        for rows in source_roles.values():
            rows.sort(key=lambda row: stable_rank(row["record_id"], salt))
    external_eval.sort(key=lambda row: stable_rank(row["record_id"], salt))
    quarantined.sort(key=lambda row: (row["quarantine_reason"], row["record_id"]))
    for edge in collision_edges:
        if "cluster_id" not in edge:
            edge["cluster_id"] = cluster_by_record_id[edge["left_record_id"]]
        if "left_record_id" in edge:
            edge["left_answer_sha256"] = sha256_text(
                normalize_problem(record_by_id[edge["left_record_id"]].answer)
            )
            edge["right_answer_sha256"] = sha256_text(
                normalize_problem(record_by_id[edge["right_record_id"]].answer)
            )
    collision_edges.sort(
        key=lambda edge: (
            edge["cluster_id"],
            edge["edge_type"],
            edge.get("left_record_id", ""),
            edge.get("right_record_id", ""),
        )
    )

    stats = {
        "normalization_version": NORMALIZATION_VERSION,
        "partition_salt": salt,
        "input_rows": len(records),
        "problem_clusters": len(groups),
        "within_source_duplicate_train_rows_removed": duplicate_rows,
        "duplicate_M_test_rows_removed": duplicate_eval_rows,
        "quarantine_rows_by_reason": dict(sorted(reasons.items())),
        "collision_edges_by_type": {
            edge_type: sum(edge["edge_type"] == edge_type for edge in collision_edges)
            for edge_type in sorted({edge["edge_type"] for edge in collision_edges})
        },
        "stratum_allocation": stratum_allocation,
        "role_counts": {
            source: {role: len(rows) for role, rows in source_roles.items()}
            for source, source_roles in roles.items()
        },
        "external_eval_rows": len(external_eval),
    }
    return ClusteredData(roles, external_eval, quarantined, collision_edges, stats)


def validate_columns(actual: Iterable[str], expected: Iterable[str], source: str) -> None:
    actual_set = set(actual)
    expected_set = set(expected)
    missing = sorted(expected_set - actual_set)
    unexpected = sorted(actual_set - expected_set)
    if missing or unexpected:
        raise ValueError(
            f"{source} schema mismatch: missing={missing}, unexpected={unexpected}"
        )


def _require_string(row: Mapping, name: str, context: str, *, allow_empty: bool = True) -> str:
    value = row[name]
    if not isinstance(value, str):
        raise ValueError(f"{context}: {name} must be a non-null string")
    if not allow_empty and not value.strip():
        raise ValueError(f"{context}: {name} must be non-empty")
    return value


def validate_math_row(row: Mapping, split: str, row_index: int) -> None:
    context = f"MATH {split} row {row_index}"
    validate_columns(row.keys(), MATH_COLUMNS, context)
    for name in MATH_COLUMNS:
        _require_string(row, name, context)


def _validate_list(
    value,
    *,
    name: str,
    context: str,
    expected_length: int | None,
    element_type: type,
    nullable: bool = False,
) -> None:
    if value is None and nullable:
        return
    if not isinstance(value, list):
        raise ValueError(f"{context}: {name} must be a list{' or null' if nullable else ''}")
    if expected_length is not None and len(value) != expected_length:
        raise ValueError(f"{context}: {name} length {len(value)} != {expected_length}")
    if any(type(item) is not element_type for item in value):
        raise ValueError(f"{context}: {name} elements must be {element_type.__name__}")


def validate_openr1_arrays(row: Mapping, row_index: int) -> None:
    context = f"OpenR1 row {row_index}"
    validate_columns(row.keys(), OPENR1_COLUMNS, context)
    for name in ("problem", "solution", "answer", "problem_type", "question_type", "source"):
        _require_string(row, name, context)
    uuid = row["uuid"]
    if uuid is not None and not isinstance(uuid, str):
        raise ValueError(f"{context}: uuid must be string or null")
    generations = row["generations"]
    _validate_list(
        generations,
        name="generations",
        context=context,
        expected_length=None,
        element_type=str,
    )
    expected = len(generations)
    if expected == 0:
        raise ValueError(f"{context}: generations must be non-empty")
    _validate_list(
        row["correctness_math_verify"],
        name="correctness_math_verify",
        context=context,
        expected_length=expected,
        element_type=bool,
    )
    _validate_list(
        row["correctness_llama"],
        name="correctness_llama",
        context=context,
        expected_length=expected,
        element_type=bool,
        nullable=True,
    )
    _validate_list(
        row["finish_reasons"],
        name="finish_reasons",
        context=context,
        expected_length=expected,
        element_type=str,
        nullable=True,
    )
    _validate_list(
        row["is_reasoning_complete"],
        name="is_reasoning_complete",
        context=context,
        expected_length=expected,
        element_type=bool,
    )
    correctness_count = row["correctness_count"]
    if type(correctness_count) is not int or not 1 <= correctness_count <= expected:
        raise ValueError(
            f"{context}: correctness_count must be an integer in [1, {expected}]"
        )
    governing_correctness = (
        row["correctness_llama"]
        if row["correctness_llama"] is not None
        else row["correctness_math_verify"]
    )
    if correctness_count != sum(governing_correctness):
        raise ValueError(
            f"{context}: correctness_count {correctness_count} does not match verifier flags"
        )
    messages = row["messages"]
    if not isinstance(messages, list) or len(messages) != 2:
        raise ValueError(f"{context}: messages must contain exactly user and assistant entries")
    for message_index, message in enumerate(messages):
        if not isinstance(message, Mapping) or set(message) != {"role", "content"}:
            raise ValueError(
                f"{context}: messages[{message_index}] must have exactly role/content strings"
            )
        if not isinstance(message["role"], str) or not isinstance(message["content"], str):
            raise ValueError(f"{context}: messages[{message_index}] role/content must be strings")
    if messages[0]["role"] != "user" or messages[1]["role"] != "assistant":
        raise ValueError(f"{context}: messages roles must be user then assistant")


def records_from_math(rows: Iterable[dict], split: str) -> tuple[list[ProblemRecord], dict, list[dict]]:
    records: list[ProblemRecord] = []
    excluded: list[dict] = []
    seen = 0
    for index, row in enumerate(rows):
        seen += 1
        validate_math_row(row, split, index)
        answer = math_answer_from_solution(row["solution"])
        problem = row["problem"].strip()
        if not problem or answer is None or not answer.strip():
            excluded.append(
                {
                    "source": "M",
                    "source_split": split,
                    "source_index": index,
                    "question_sha256": sha256_text(normalize_problem(problem)),
                    "reason": "empty_problem" if not problem else "missing_or_empty_boxed_answer",
                    "problem": problem,
                    "reference_solution": row["solution"],
                }
            )
            continue
        records.append(
            ProblemRecord(
                source="M",
                source_split=split,
                source_index=index,
                problem=problem,
                answer=answer,
                reference_solution=row["solution"],
                stratum=f"{row['type']}|{row['level']}",
                source_metadata={"type": row["type"], "level": row["level"]},
            )
        )
    return records, {"seen": seen, "accepted": len(records), "excluded": len(excluded)}, excluded


def records_from_openr1(rows: Iterable[dict]) -> tuple[list[ProblemRecord], dict, list[dict]]:
    records: list[ProblemRecord] = []
    excluded: list[dict] = []
    seen = 0
    for index, row in enumerate(rows):
        seen += 1
        validate_openr1_arrays(row, index)
        problem = row["problem"].strip()
        answer = row["answer"].strip()
        if not problem or not answer:
            excluded.append(
                {
                    "source": "O",
                    "source_split": "train",
                    "source_index": index,
                    "question_sha256": sha256_text(normalize_problem(problem)),
                    "reason": "empty_problem" if not problem else "empty_answer",
                    "problem": problem,
                    "answer": answer,
                }
            )
            continue
        records.append(
            ProblemRecord(
                source="O",
                source_split="train",
                source_index=index,
                problem=problem,
                answer=answer,
                reference_solution=row["solution"],
                stratum=f"{row['problem_type']}|{row['question_type']}",
                source_metadata={
                    "problem_type": row.get("problem_type"),
                    "question_type": row.get("question_type"),
                    "original_source": row.get("source"),
                    "uuid": row.get("uuid"),
                    "correctness_count": row.get("correctness_count"),
                },
            )
        )
    return records, {"seen": seen, "accepted": len(records), "excluded": len(excluded)}, excluded


def write_jsonl(path: Path, rows: Iterable[dict]) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            line = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            handle.write(line)
            digest.update(line.encode("utf-8"))
            count += 1
    return count, digest.hexdigest()


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
