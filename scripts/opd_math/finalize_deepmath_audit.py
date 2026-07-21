#!/usr/bin/env python3
"""Finalize a completed DeepMath scan from explicit semantic decisions.

This stage never reruns the expensive semantic candidate search.  It verifies
the scan and inventory bytes, applies a complete pair-ID decision file, rebuilds
the final candidate clusters, and emits a new immutable data-only namespace.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    from .audit_deepmath_inventory import (
        AUDIT_ID,
        ROOT,
        _cluster_candidate_rows,
        _verify_inventory_manifest,
        _write_candidate_parquet,
        exact_format_edges,
        load_audit_plan,
        load_records,
    )
    from .data_contract import resolve_semantic_reviews, write_jsonl
    from .deepmath_qualification import load_plan as load_qualification_plan
    from .materialize_deepmath_inventory import load_inventory_plan, sha256_file
except ImportError:
    from audit_deepmath_inventory import (  # type: ignore
        AUDIT_ID,
        ROOT,
        _cluster_candidate_rows,
        _verify_inventory_manifest,
        _write_candidate_parquet,
        exact_format_edges,
        load_audit_plan,
        load_records,
    )
    from data_contract import resolve_semantic_reviews, write_jsonl  # type: ignore
    from deepmath_qualification import load_plan as load_qualification_plan  # type: ignore
    from materialize_deepmath_inventory import load_inventory_plan, sha256_file  # type: ignore


DEFAULT_AUDIT_PLAN = ROOT / "configs/opd_math/deepmath_collision_audit_plan.json"
DEFAULT_QUALIFICATION_PLAN = ROOT / "configs/opd_math/deepmath_qualification_plan.json"
DEFAULT_INVENTORY_PLAN = ROOT / "configs/opd_math/deepmath_inventory_plan.json"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _git_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    _require(not status.strip(), "DeepMath finalization requires a clean Git checkout")
    return {"commit": commit, "clean": True}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    path = path.resolve()
    _require(path.is_file() and not path.is_symlink(), f"JSONL input missing or symlinked: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            _require(isinstance(row, dict), f"JSONL row {line_number} is not an object: {path}")
            rows.append(row)
    return rows


def load_review_decisions(path: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    rows = read_jsonl(path)
    for line_number, row in enumerate(rows, 1):
        _require(
            set(row) == {"pair_id", "decision"},
            f"semantic decision row {line_number} schema drifted",
        )
        _require(isinstance(row["pair_id"], str) and row["pair_id"], "empty semantic pair ID")
        _require(row["decision"] in {"duplicate", "distinct"}, "invalid semantic decision")
    return rows, {
        "path": str(path.resolve()),
        "rows": len(rows),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _verify_scan_file(item: Mapping[str, Any]) -> Path:
    path = Path(str(item["path"])).resolve()
    _require(path.is_file() and not path.is_symlink(), f"scan file missing or symlinked: {path}")
    _require(path.stat().st_size == item["bytes"], f"scan file byte count drifted: {path}")
    _require(sha256_file(path) == item["sha256"], f"scan file hash drifted: {path}")
    return path


def verify_scan_manifest(
    path: Path,
    *,
    audit_plan: Mapping[str, Any],
    inventory_plan: Mapping[str, Any],
    qualification_plan: Mapping[str, Any],
    inventory_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    path = path.resolve()
    _require(path.is_file() and not path.is_symlink(), "scan manifest missing or symlinked")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(payload.get("schema_version") == 1, "scan manifest schema drifted")
    _require(payload.get("audit_id") == AUDIT_ID, "scan audit identity drifted")
    _require(payload.get("stage") == "global_collision_label_prompt_scan", "scan stage drifted")
    _require(payload.get("audit_plan_sha256") == audit_plan["sha256"], "scan audit plan drifted")
    _require(
        payload.get("inventory_plan_sha256") == inventory_plan["sha256"],
        "scan inventory plan drifted",
    )
    _require(
        payload.get("qualification_plan_sha256") == qualification_plan["sha256"],
        "scan qualification plan drifted",
    )
    _require(
        payload.get("inventory_manifest_sha256") == inventory_manifest["sha256"],
        "scan inventory manifest drifted",
    )
    _require(payload.get("teacher_training_authorized") is False, "scan authorized training")
    _require(payload.get("scientific_use_allowed") is False, "scan authorized science")
    files = payload.get("files")
    _require(isinstance(files, dict), "scan manifest lacks file custody")
    required = {
        "audit/collision_edges.jsonl",
        "audit/semantic_candidates.jsonl",
        "audit/semantic_review_packet.jsonl",
        "audit/C_quarantine.jsonl",
        "audit/label_conflicts.jsonl",
        "audit/C_parse_failures.jsonl",
        "audit/C_prompt_truncations.jsonl",
        "records/C_preliminary_eligible.parquet",
    }
    _require(set(files) == required, "scan output file set drifted")
    for item in files.values():
        _verify_scan_file(item)
    payload["path"] = str(path)
    payload["sha256"] = sha256_file(path)
    return payload


def _write_outputs(
    output_dir: Path,
    *,
    records,
    eligible_indices: list[int],
    resolved_ledger: list[dict],
    decisions: list[dict],
    final_edges: list[dict],
    quarantine: list[dict],
    label_conflicts: list[dict],
) -> dict[str, dict[str, Any]]:
    audit_dir = output_dir / "audit"
    records_dir = output_dir / "records"
    audit_dir.mkdir()
    records_dir.mkdir()
    files: dict[str, dict[str, Any]] = {}
    for relative, rows in (
        ("audit/final_collision_edges.jsonl", final_edges),
        ("audit/semantic_candidates_resolved.jsonl", resolved_ledger),
        ("audit/semantic_review_decisions.jsonl", decisions),
        ("audit/C_final_quarantine.jsonl", quarantine),
        ("audit/final_label_conflicts.jsonl", label_conflicts),
    ):
        target = output_dir / relative
        count, digest = write_jsonl(target, rows)
        files[relative] = {
            "path": str(target.resolve()),
            "rows": count,
            "bytes": target.stat().st_size,
            "sha256": digest,
        }
    target = records_dir / "C_data_eligible.parquet"
    files["records/C_data_eligible.parquet"] = _write_candidate_parquet(
        target, records, eligible_indices
    )
    return files


def finalize(
    *,
    audit_plan: Mapping[str, Any],
    inventory_plan: Mapping[str, Any],
    qualification_plan: Mapping[str, Any],
    inventory_manifest: Mapping[str, Any],
    scan_manifest: Mapping[str, Any],
    decisions: list[dict[str, str]],
    decision_receipt: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    try:
        output_dir.relative_to(ROOT.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("DeepMath finalization output must live outside the Git checkout")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite DeepMath finalization output: {output_dir}")
    git = _git_state()

    records, source_counts = load_records(inventory_plan, inventory_manifest)
    index_by_id = {record.record_id: index for index, record in enumerate(records)}
    semantic_path = _verify_scan_file(
        scan_manifest["files"]["audit/semantic_candidates.jsonl"]
    )
    semantic_ledger = read_jsonl(semantic_path)
    auto_edges = []
    for row in semantic_ledger:
        if row.get("auto_clustered"):
            auto_edges.append(
                (index_by_id[row["left_record_id"]], index_by_id[row["right_record_id"]])
            )
    _require(
        len(auto_edges) == scan_manifest["semantic"]["auto_cluster_edges"],
        "scan semantic-auto edge count drifted",
    )
    resolved_edges, resolved_ledger, review_stats = resolve_semantic_reviews(
        records, auto_edges, semantic_ledger, decisions
    )

    union, deterministic_edges, deterministic_counts = exact_format_edges(records)
    _require(
        deterministic_counts == scan_manifest["deterministic_collision_counts"],
        "scan exact/format counts do not reproduce",
    )
    final_edges = list(deterministic_edges)
    for left, right, edge_type in resolved_edges:
        union.union(left, right)
        final_edges.append(
            {
                "edge_type": edge_type,
                "left_record_id": records[left].record_id,
                "right_record_id": records[right].record_id,
            }
        )
    final_edges.sort(
        key=lambda row: (row["edge_type"], row["left_record_id"], row["right_record_id"])
    )

    parse_failures = read_jsonl(
        _verify_scan_file(scan_manifest["files"]["audit/C_parse_failures.jsonl"])
    )
    prompt_truncations = read_jsonl(
        _verify_scan_file(scan_manifest["files"]["audit/C_prompt_truncations.jsonl"])
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

    gates = audit_plan["candidate_gates"]
    parseability = scan_manifest["gold_parseability"]
    prompt_surface = scan_manifest["prompt_surface"]
    problem_quality = scan_manifest["problem_quality"]
    semantic_scan = scan_manifest["semantic"]
    blockers = []
    if not semantic_scan["scan_complete"]:
        blockers.append("semantic scan was incomplete")
    if not review_stats["review_complete"]:
        blockers.append(f"{review_stats['unresolved_review_edges']} semantic reviews unresolved")
    if cluster_stats["eligible_unique_clusters"] < gates["minimum_unique_eligible_clusters"]:
        blockers.append("eligible unique C clusters below minimum")
    if parseability["parseable_fraction"] < gates["minimum_gold_parseability"]:
        blockers.append("C gold parseability below minimum")
    if (
        problem_quality["missing_candidate_problems"]
        > gates["maximum_missing_candidate_problems"]
    ):
        blockers.append("C missing problems exceed maximum")
    if cluster_stats["unresolved_label_conflicts"] > gates["maximum_unresolved_label_conflicts"]:
        blockers.append("unresolved C label conflicts exceed maximum")
    if prompt_surface["prompt_truncations"] > gates["maximum_prompt_truncations"]:
        blockers.append("C prompt truncations exceed maximum")
    data_gate_passed = not blockers

    output_dir.mkdir(parents=True)
    files = _write_outputs(
        output_dir,
        records=records,
        eligible_indices=eligible_indices,
        resolved_ledger=resolved_ledger,
        decisions=sorted(decisions, key=lambda row: row["pair_id"]),
        final_edges=final_edges,
        quarantine=quarantine,
        label_conflicts=label_conflicts,
    )
    payload = {
        "schema_version": 1,
        "audit_id": AUDIT_ID,
        "stage": "global_collision_label_prompt_finalize",
        "status": "passed" if data_gate_passed else "blocked",
        "git": git,
        "audit_plan_sha256": audit_plan["sha256"],
        "qualification_plan_sha256": qualification_plan["sha256"],
        "inventory_plan_sha256": inventory_plan["sha256"],
        "inventory_manifest_path": inventory_manifest["path"],
        "inventory_manifest_sha256": inventory_manifest["sha256"],
        "scan_manifest_path": scan_manifest["path"],
        "scan_manifest_sha256": scan_manifest["sha256"],
        "review_decisions": dict(decision_receipt),
        "source_counts": source_counts,
        "semantic_scan": semantic_scan,
        "semantic_review": review_stats,
        "candidate_clusters": cluster_stats,
        "gold_parseability": parseability,
        "prompt_surface": prompt_surface,
        "problem_quality": problem_quality,
        "files": files,
        "data_gate_passed": data_gate_passed,
        "data_gate_blockers": blockers,
        "teacher_training_authorized": False,
        "scientific_use_allowed": False,
        "remaining_gates": [
            "freeze deterministic C roles after final data pass",
            "run fixed raw-model feasibility surface",
            "train and independently gap-qualify fresh C teacher",
        ],
        "runtime": {"python": sys.version.split()[0]},
        "launcher": (
            {
                "path": str(Path(os.environ["OPD_DEEPMATH_FINALIZE_LAUNCHER_PATH"]).resolve()),
                "sha256": sha256_file(
                    Path(os.environ["OPD_DEEPMATH_FINALIZE_LAUNCHER_PATH"])
                ),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID", "none"),
            }
            if os.environ.get("OPD_DEEPMATH_FINALIZE_LAUNCHER_PATH")
            else None
        ),
    }
    manifest_path = output_dir / "finalization_manifest.json"
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
    parser.add_argument("--scan-manifest", type=Path, required=True)
    parser.add_argument("--review-decisions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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
    scan_manifest = verify_scan_manifest(
        args.scan_manifest,
        audit_plan=audit_plan,
        inventory_plan=inventory_plan,
        qualification_plan=qualification_plan,
        inventory_manifest=inventory_manifest,
    )
    decisions, decision_receipt = load_review_decisions(args.review_decisions)
    result = finalize(
        audit_plan=audit_plan,
        inventory_plan=inventory_plan,
        qualification_plan=qualification_plan,
        inventory_manifest=inventory_manifest,
        scan_manifest=scan_manifest,
        decisions=decisions,
        decision_receipt=decision_receipt,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
