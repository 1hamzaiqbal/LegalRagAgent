#!/usr/bin/env python3
"""Validate, reconcile, and freeze complete semantic-review decisions."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

try:
    from .data_contract import iter_jsonl, write_jsonl
except ImportError:
    from data_contract import iter_jsonl, write_jsonl  # type: ignore


POLICY = (
    "duplicate includes a shared source item, stem, diagram, equations/data, subquestion "
    "family, OCR/translation variant, or local parameter/target variant whose exposure could "
    "reduce work on the other record; distinct is reserved for generic boilerplate shared by "
    "independent mathematical instances"
)
PAIR_ID = re.compile(r"[0-9a-f]{64}")


def _checked_decision(row: dict[str, Any], *, label: str, row_number: int) -> tuple[str, str, str]:
    pair_id = row.get("pair_id")
    decision = row.get("decision")
    rationale = row.get("rationale")
    if not isinstance(pair_id, str) or PAIR_ID.fullmatch(pair_id) is None:
        raise ValueError(f"{label} row {row_number} has an invalid pair_id")
    if decision not in {"duplicate", "distinct"}:
        raise ValueError(f"{label} row {row_number} has an invalid decision")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError(f"{label} row {row_number} lacks a rationale")
    return pair_id, decision, rationale.strip()


def finalize(
    packet_path: Path, review_paths: list[Path], override_paths: list[Path]
) -> list[dict[str, Any]]:
    packet = list(iter_jsonl(packet_path))
    packet_ids = [row.get("pair_id") for row in packet]
    if any(
        not isinstance(pair_id, str) or PAIR_ID.fullmatch(pair_id) is None
        for pair_id in packet_ids
    ):
        raise ValueError("review packet contains an invalid pair_id")
    if len(set(packet_ids)) != len(packet_ids):
        raise ValueError("review packet contains duplicate pair IDs")

    packet_id_set = set(packet_ids)
    decisions: dict[str, dict[str, Any]] = {}
    for path in review_paths:
        for row_number, row in enumerate(iter_jsonl(path), start=1):
            pair_id, decision, rationale = _checked_decision(
                row, label=str(path), row_number=row_number
            )
            if pair_id not in packet_id_set:
                raise ValueError(f"review contains unknown pair_id: {pair_id}")
            if pair_id in decisions:
                raise ValueError(f"review pair appears more than once: {pair_id}")
            decisions[pair_id] = {
                "pair_id": pair_id,
                "decision": decision,
                "rationale": rationale,
                "initial_decision": decision,
                "review_policy": POLICY,
            }
    missing = set(packet_ids) - set(decisions)
    if missing:
        raise ValueError(f"reviews omit {len(missing)} packet pairs")

    seen_overrides: set[str] = set()
    for path in override_paths:
        for row_number, row in enumerate(iter_jsonl(path), start=1):
            pair_id, decision, rationale = _checked_decision(
                row, label=str(path), row_number=row_number
            )
            if pair_id not in decisions:
                raise ValueError(f"override contains unknown pair_id: {pair_id}")
            if pair_id in seen_overrides:
                raise ValueError(f"pair is overridden more than once: {pair_id}")
            seen_overrides.add(pair_id)
            decisions[pair_id]["decision"] = decision
            decisions[pair_id]["rationale"] = rationale
            decisions[pair_id]["reconciled"] = True

    return [decisions[str(pair_id)] for pair_id in packet_ids]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--review", type=Path, action="append", required=True)
    parser.add_argument("--override", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing to overwrite finalized review file: {args.output}")
    rows = finalize(args.packet, args.review, args.override)
    count, digest = write_jsonl(args.output, rows)
    counts = {decision: sum(row["decision"] == decision for row in rows) for decision in ("duplicate", "distinct")}
    print(json.dumps({"output": str(args.output.resolve()), "rows": count, "sha256": digest, **counts}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
