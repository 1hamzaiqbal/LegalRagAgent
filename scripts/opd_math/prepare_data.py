#!/usr/bin/env python3
"""Prepare the pinned MATH/OpenR1 role pools and M/O experiment matrix."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
from pathlib import Path

try:
    from .data_contract import (
        MATH_COLUMNS,
        NORMALIZATION_VERSION,
        OPENR1_COLUMNS,
        PARTITION_SALT,
        SEMANTIC_AUDIT_VERSION,
        boxed_gold,
        cluster_and_partition,
        iter_jsonl,
        records_from_math,
        records_from_openr1,
        resolve_semantic_reviews,
        semantic_near_duplicate_edges,
        validate_columns,
        write_jsonl,
    )
except ImportError:
    from data_contract import (  # type: ignore
        MATH_COLUMNS,
        NORMALIZATION_VERSION,
        OPENR1_COLUMNS,
        PARTITION_SALT,
        SEMANTIC_AUDIT_VERSION,
        boxed_gold,
        cluster_and_partition,
        iter_jsonl,
        records_from_math,
        records_from_openr1,
        resolve_semantic_reviews,
        semantic_near_duplicate_edges,
        validate_columns,
        write_jsonl,
    )


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "configs" / "opd_math" / "source_manifest.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state() -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, check=True, text=True, capture_output=True
        ).stdout
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD"], cwd=ROOT, check=True, capture_output=True
        ).stdout
        return {
            "commit": commit,
            "dirty": bool(status.strip()),
            "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
            "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def assert_external_output(path: Path) -> None:
    resolved = path.resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError:
        return
    raise ValueError(f"prepared datasets must live outside the Git worktree: {resolved}")


def load_source(dataset_spec: dict, split: str, limit: int):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("prepare_data.py requires the pinned datasets package") from exc

    dataset = load_dataset(
        dataset_spec["id"],
        dataset_spec["config"],
        split=split,
        revision=dataset_spec["revision"],
    )
    if limit > 0:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return dataset


def _math_verify_parser():
    from math_verify import parse

    return parse, importlib.metadata.version("math-verify")


def filter_reward_parseable(records, *, parse_fn=None, verifier_version: str | None = None):
    if parse_fn is None:
        parse_fn, verifier_version = _math_verify_parser()
    verifier_version = verifier_version or "unknown"

    accepted = []
    excluded = []
    cache: dict[str, bool] = {}
    parse_errors: dict[str, str] = {}
    for record in records:
        gold = boxed_gold(record.answer)
        if gold not in cache:
            try:
                cache[gold] = bool(parse_fn(gold, extraction_mode="first_match"))
            except Exception as exc:  # parser failures are data exclusions, not implicit wrong labels
                cache[gold] = False
                parse_errors[gold] = type(exc).__name__
        if cache[gold]:
            accepted.append(record)
        else:
            excluded.append(
                {
                    "record_id": record.record_id,
                    "source": record.source,
                    "source_split": record.source_split,
                    "source_index": record.source_index,
                    "question_sha256": record.exact_key,
                    "answer_sha256": hashlib.sha256(record.answer.encode("utf-8")).hexdigest(),
                    "answer": record.answer,
                    "reason": (
                        f"gold_parser_error_{parse_errors[gold]}"
                        if gold in parse_errors
                        else f"gold_not_parseable_by_math_verify_{verifier_version}"
                    ),
                }
            )
    stats = {
        "verifier": "math_verify",
        "version": verifier_version,
        "input_records": len(records),
        "unique_boxed_golds": len(cache),
        "parseable_records": len(accepted),
        "excluded_records": len(excluded),
        "parser_error_records": sum(boxed_gold(record.answer) in parse_errors for record in records),
    }
    return accepted, excluded, stats, cache


def assert_output_parseability(clustered, *, parse_fn, cache: dict[str, bool]) -> dict:
    checks: dict[str, dict] = {}
    row_groups = {
        **{
            f"roles/{source}/{role}.jsonl": rows
            for source, source_roles in clustered.role_rows.items()
            for role, rows in source_roles.items()
        },
        "eval/M_test.jsonl": clustered.external_eval,
    }
    all_parseable = True
    for relative, rows in sorted(row_groups.items()):
        failures: list[str] = []
        for row in rows:
            gold = row["solution"]
            if gold not in cache:
                cache[gold] = bool(parse_fn(gold, extraction_mode="first_match"))
            if not cache[gold]:
                failures.append(row["record_id"])
        checks[relative] = {
            "rows": len(rows),
            "parseable_rows": len(rows) - len(failures),
            "unparseable_rows": len(failures),
        }
        all_parseable = all_parseable and not failures
        if failures:
            raise RuntimeError(
                f"output role contains math-verify-unparseable golds: {relative}: {failures[:5]}"
            )
    return {"all_parseable": all_parseable, "files": checks}


def register_pair_file(pair_row: dict, field: str, relative: str, files: dict[str, dict]) -> None:
    if relative not in files:
        raise KeyError(f"pair references unregistered prepared file: {relative}")
    registered = files[relative]
    pair_row[field] = relative
    pair_row[f"{field}_rows"] = registered["rows"]
    pair_row[f"{field}_sha256"] = registered["sha256"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--partition-salt", default=PARTITION_SALT)
    parser.add_argument(
        "--semantic-review-jsonl",
        type=Path,
        help=(
            "optional decisions for every semantic candidate marked requires_review; "
            "rows must contain pair_id and decision=duplicate|distinct"
        ),
    )
    parser.add_argument(
        "--semantic-fingerprint-size",
        type=int,
        default=8,
        help="number of rare deterministic shingle blocks per record (recorded in manifest)",
    )
    parser.add_argument(
        "--semantic-max-bucket-size",
        type=int,
        default=256,
        help="bounded comparisons per shingle block; any skipped block fails scientific use",
    )
    parser.add_argument(
        "--audit-limit-per-split",
        type=int,
        default=0,
        help="nonzero is a plumbing-only partial scan and may not be used for scientific runs",
    )
    args = parser.parse_args()
    if args.semantic_fingerprint_size <= 0 or args.semantic_max_bucket_size <= 0:
        raise ValueError("semantic fingerprint and bucket sizes must be positive")

    output_dir = args.output_dir.resolve()
    assert_external_output(output_dir)
    if output_dir.is_symlink() or (
        output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir()))
    ):
        raise FileExistsError(f"refusing to overwrite non-empty prepared-data directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_manifest_bytes = args.manifest.read_bytes()
    source_manifest = json.loads(source_manifest_bytes)
    partition_contract = source_manifest.get("partition", {})
    if partition_contract.get("normalization_version") != NORMALIZATION_VERSION:
        raise ValueError("source manifest normalization version does not match implementation")
    if partition_contract.get("semantic_audit_version") != SEMANTIC_AUDIT_VERSION:
        raise ValueError("source manifest semantic audit version does not match implementation")
    if args.partition_salt != partition_contract.get("salt"):
        raise ValueError("partition salt override does not match the pinned source manifest")
    datasets = source_manifest["datasets"]

    math_train_ds = load_source(
        datasets["M"], datasets["M"]["train_split"], args.audit_limit_per_split
    )
    math_test_ds = load_source(
        datasets["M"], datasets["M"]["external_eval_split"], args.audit_limit_per_split
    )
    openr1_ds = load_source(
        datasets["O"], datasets["O"]["train_split"], args.audit_limit_per_split
    )
    validate_columns(math_train_ds.column_names, MATH_COLUMNS, "MATH train dataset")
    validate_columns(math_test_ds.column_names, MATH_COLUMNS, "MATH test dataset")
    validate_columns(openr1_ds.column_names, OPENR1_COLUMNS, "OpenR1 train dataset")

    if args.audit_limit_per_split == 0:
        expected = {
            "M_train": datasets["M"]["expected_train_rows"],
            "M_test": datasets["M"]["expected_external_eval_rows"],
            "O_train": datasets["O"]["expected_train_rows"],
        }
        actual = {"M_train": len(math_train_ds), "M_test": len(math_test_ds), "O_train": len(openr1_ds)}
        if actual != expected:
            raise RuntimeError(f"pinned source row counts drifted: expected={expected}, actual={actual}")

    math_train, math_train_stats, math_train_excluded = records_from_math(math_train_ds, "train")
    math_test, math_test_stats, math_test_excluded = records_from_math(math_test_ds, "test")
    openr1_train, openr1_stats, openr1_excluded = records_from_openr1(openr1_ds)
    records = math_train + math_test + openr1_train
    parse_fn, verifier_version = _math_verify_parser()
    records, verifier_excluded, parseability_stats, parseability_cache = filter_reward_parseable(
        records, parse_fn=parse_fn, verifier_version=verifier_version
    )
    ingestion_excluded = math_train_excluded + math_test_excluded + openr1_excluded + verifier_excluded
    m_test_verifier_excluded = any(
        row["source"] == "M" and row["source_split"] == "test" for row in verifier_excluded
    )
    ingestion_excluded.sort(
        key=lambda row: (row["source"], row["source_split"], row["source_index"], row["reason"])
    )
    auto_semantic_edges, semantic_ledger, semantic_stats = semantic_near_duplicate_edges(
        records,
        fingerprint_size=args.semantic_fingerprint_size,
        max_bucket_size=args.semantic_max_bucket_size,
    )
    semantic_review_rows = (
        list(iter_jsonl(args.semantic_review_jsonl)) if args.semantic_review_jsonl else []
    )
    semantic_edges, semantic_ledger, semantic_review_stats = resolve_semantic_reviews(
        records, auto_semantic_edges, semantic_ledger, semantic_review_rows
    )
    semantic_stats.update(semantic_review_stats)
    semantic_stats["complete"] = (
        semantic_stats["scan_complete"] and semantic_stats["review_complete"]
    )
    clustered = cluster_and_partition(records, args.partition_salt, semantic_edges=semantic_edges)
    output_parseability = assert_output_parseability(
        clustered, parse_fn=parse_fn, cache=parseability_cache
    )

    files: dict[str, dict] = {}
    for source, role_rows in clustered.role_rows.items():
        for role, rows in role_rows.items():
            relative = Path("roles") / source / f"{role}.jsonl"
            count, digest = write_jsonl(output_dir / relative, rows)
            files[relative.as_posix()] = {"rows": count, "sha256": digest}

    count, digest = write_jsonl(output_dir / "eval" / "M_test.jsonl", clustered.external_eval)
    files["eval/M_test.jsonl"] = {"rows": count, "sha256": digest}
    count, digest = write_jsonl(output_dir / "audit" / "quarantine.jsonl", clustered.quarantined)
    files["audit/quarantine.jsonl"] = {"rows": count, "sha256": digest}
    count, digest = write_jsonl(output_dir / "audit" / "ingestion_exclusions.jsonl", ingestion_excluded)
    files["audit/ingestion_exclusions.jsonl"] = {"rows": count, "sha256": digest}
    count, digest = write_jsonl(output_dir / "audit" / "collision_edges.jsonl", clustered.collision_edges)
    files["audit/collision_edges.jsonl"] = {"rows": count, "sha256": digest}
    count, digest = write_jsonl(output_dir / "audit" / "semantic_candidates.jsonl", semantic_ledger)
    files["audit/semantic_candidates.jsonl"] = {"rows": count, "sha256": digest}

    pairs = []
    matched_budgets = {
        role: min(len(clustered.role_rows["M"][role]), len(clustered.role_rows["O"][role]))
        for role in ("teacher_train", "student_opd", "teacher_gap_dev", "source_holdout")
    }
    for pair in source_manifest["primary_pairs"]:
        pair_row = dict(pair)
        register_pair_file(
            pair_row,
            "teacher_train_file",
            f"roles/{pair['teacher_source']}/teacher_train.jsonl",
            files,
        )
        register_pair_file(
            pair_row,
            "teacher_skill_dev_file",
            f"roles/{pair['teacher_source']}/teacher_gap_dev.jsonl",
            files,
        )
        register_pair_file(
            pair_row,
            "target_gap_dev_file",
            f"roles/{pair['opd_source']}/teacher_gap_dev.jsonl",
            files,
        )
        register_pair_file(
            pair_row,
            "student_opd_file",
            f"roles/{pair['opd_source']}/student_opd.jsonl",
            files,
        )
        register_pair_file(
            pair_row,
            "student_holdout_file",
            f"roles/{pair['opd_source']}/source_holdout.jsonl",
            files,
        )
        pair_row["same_items"] = False
        pair_row["teacher_example_limit"] = matched_budgets["teacher_train"]
        pair_row["student_opd_pool_limit"] = matched_budgets["student_opd"]
        pair_row["teacher_skill_dev_limit"] = matched_budgets["teacher_gap_dev"]
        pair_row["target_gap_dev_limit"] = matched_budgets["teacher_gap_dev"]
        pair_row["student_holdout_limit"] = matched_budgets["source_holdout"]
        pairs.append(pair_row)

    code_state = git_state()
    scientific_blockers: list[str] = []
    if args.audit_limit_per_split != 0:
        scientific_blockers.append("partial source scan")
    if not semantic_stats["scan_complete"]:
        scientific_blockers.append("semantic scan skipped oversized candidate buckets")
    if not semantic_stats["review_complete"]:
        scientific_blockers.append(
            f"{semantic_stats['unresolved_review_edges']} semantic candidate reviews unresolved"
        )
    if code_state.get("dirty") is not False:
        scientific_blockers.append("preparation code is dirty or Git state is unavailable")
    if not output_parseability["all_parseable"]:
        scientific_blockers.append("one or more output golds are not math-verify parseable")
    if math_test_excluded or m_test_verifier_excluded:
        scientific_blockers.append("frozen MATH test lost one or more records during ingestion")
    if len(clustered.external_eval) != len(math_test):
        scientific_blockers.append("frozen MATH test lost one or more records during partitioning")
    if any(value <= 0 for value in matched_budgets.values()):
        scientific_blockers.append("one or more primary matched role budgets are empty")

    run_manifest = {
        "schema_version": 1,
        "complete_collision_scan": args.audit_limit_per_split == 0,
        "audit_limit_per_split": args.audit_limit_per_split,
        "source_manifest_path": str(args.manifest.resolve()),
        "source_manifest_sha256": hashlib.sha256(source_manifest_bytes).hexdigest(),
        "code_git_state": code_state,
        "hf_home": os.environ.get("HF_HOME"),
        "source_fingerprints": {
            "M_train": getattr(math_train_ds, "_fingerprint", None),
            "M_test": getattr(math_test_ds, "_fingerprint", None),
            "O_train": getattr(openr1_ds, "_fingerprint", None),
        },
        "ingestion": {
            "M_train": math_train_stats,
            "M_test": math_test_stats,
            "O_train": openr1_stats,
        },
        "gold_parseability": {**parseability_stats, "outputs": output_parseability},
        "partition": clustered.stats,
        "semantic_near_duplicate_audit": semantic_stats,
        "semantic_review_input": (
            {
                "path": str(args.semantic_review_jsonl.resolve()),
                "sha256": file_sha256(args.semantic_review_jsonl),
                "rows": len(semantic_review_rows),
            }
            if args.semantic_review_jsonl
            else None
        ),
        "primary_matched_budgets": matched_budgets,
        "files": files,
        "pairs": pairs,
        "scientific_use_allowed": not scientific_blockers,
        "scientific_blockers": scientific_blockers,
        "remaining_gate": None if not scientific_blockers else "; ".join(scientific_blockers),
    }
    manifest_path = output_dir / "prepared_manifest.json"
    manifest_path.write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output_dir": str(output_dir), "manifest": str(manifest_path), **clustered.stats}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
