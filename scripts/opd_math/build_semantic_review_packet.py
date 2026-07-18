#!/usr/bin/env python3
"""Materialize human-readable pairs for every unresolved semantic review edge."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .data_contract import iter_jsonl, write_jsonl
except ImportError:
    from data_contract import iter_jsonl, write_jsonl  # type: ignore


def _record_view(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "record_id": row.get("record_id"),
        "source": row.get("source"),
        "source_split": row.get("source_split"),
        "source_index": row.get("source_index"),
        "problem": row.get("problem"),
        "answer": row.get("answer"),
        "solution": row.get("solution"),
        "source_metadata": row.get("source_metadata"),
    }


def build_packet(prepared_dir: Path) -> list[dict[str, Any]]:
    prepared_dir = prepared_dir.resolve()
    manifest_path = prepared_dir / "prepared_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    candidates_path = prepared_dir / "audit" / "semantic_candidates.jsonl"
    candidates = [row for row in iter_jsonl(candidates_path) if row.get("requires_review")]
    required_ids = {
        str(row[field])
        for row in candidates
        for field in ("left_record_id", "right_record_id")
    }

    records: dict[str, dict[str, Any]] = {}
    for relative in sorted(manifest.get("files", {})):
        if not (
            relative.startswith("roles/")
            or relative == "eval/M_test.jsonl"
            or relative == "audit/quarantine.jsonl"
        ):
            continue
        path = prepared_dir / relative
        for row in iter_jsonl(path):
            record_id = row.get("record_id")
            if record_id not in required_ids:
                continue
            if record_id in records:
                raise ValueError(f"review record appears in multiple output files: {record_id}")
            records[str(record_id)] = row

    missing = sorted(required_ids - set(records))
    if missing:
        raise ValueError(f"review packet could not recover {len(missing)} records: {missing[:10]}")

    packet = []
    for candidate in candidates:
        left_id = str(candidate["left_record_id"])
        right_id = str(candidate["right_record_id"])
        packet.append(
            {
                "pair_id": candidate["pair_id"],
                "jaccard": candidate["jaccard"],
                "identical_numeric_sequence": candidate["identical_numeric_sequence"],
                "left": _record_view(records[left_id]),
                "right": _record_view(records[right_id]),
            }
        )
    packet.sort(key=lambda row: str(row["pair_id"]))
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing to overwrite review packet: {args.output}")
    packet = build_packet(args.prepared_dir)
    count, digest = write_jsonl(args.output, packet)
    print(json.dumps({"output": str(args.output.resolve()), "rows": count, "sha256": digest}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
