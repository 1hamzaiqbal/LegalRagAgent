#!/usr/bin/env python3
"""Build a reusable Snap-HyRE generation cache from detail logs.

The cache is keyed by the detail-log `label` so later evals can replay the same
snap reasoning and HyRE passage with `eval_harness.py --hyre-cache-path`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def cache_record(row: dict[str, Any], source: Path) -> dict[str, Any] | None:
    label = row.get("label")
    hyde = row.get("hyde_passage")
    snap = row.get("snap_answer")
    if not label or not hyde or not snap:
        return None
    return {
        "label": label,
        "dataset": row.get("dataset"),
        "source_mode": row.get("mode"),
        "source_log": str(source),
        "snap_answer": snap,
        "snap_and_hyre_raw": row.get("snap_and_hyre_raw") or row.get("hyde_passage_raw") or "",
        "snap_hyre_parse_ok": row.get("snap_hyre_parse_ok", True),
        "hyde_passage": hyde,
        "hyde_passage_raw": row.get("hyde_passage_raw") or row.get("snap_and_hyre_raw") or "",
        "snap_letter": row.get("snap_letter"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", action="append", required=True, type=Path, help="Detail JSONL source log")
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL cache")
    parser.add_argument("--require-parse-ok", action="store_true", help="Skip rows with snap_hyre_parse_ok=false")
    args = parser.parse_args()

    records: dict[str, dict[str, Any]] = {}
    skipped = 0
    for path in args.log:
        for row in load_rows(path):
            if args.require_parse_ok and row.get("snap_hyre_parse_ok") is False:
                skipped += 1
                continue
            rec = cache_record(row, path)
            if rec is None:
                skipped += 1
                continue
            records[str(rec["label"])] = rec

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for label in sorted(records):
            f.write(json.dumps(records[label], sort_keys=True) + "\n")

    print(f"wrote {len(records)} cache rows to {args.output}")
    if skipped:
        print(f"skipped {skipped} rows without usable snap/HyRE fields")


if __name__ == "__main__":
    main()
