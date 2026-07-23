#!/usr/bin/env python3
"""Independently reconstruct and gate an OPSD positive-control evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reconstruct(payload: dict) -> dict:
    results = payload.get("results")
    if not isinstance(results, list) or len(results) != 30:
        raise RuntimeError(f"expected 30 problem records, observed {len(results or [])}")
    total = 0
    correct = 0
    formatted = 0
    for result in results:
        generations = result.get("generations")
        if not isinstance(generations, list) or len(generations) != 12:
            raise RuntimeError("every AIME24 problem must contain exactly 12 generations")
        for generation in generations:
            total += 1
            correct += int(generation.get("correct") is True)
            formatted += int(generation.get("formatted") is True)
    if total != 360:
        raise RuntimeError(f"expected 360 generations, observed {total}")
    average_fraction = correct / total
    return {
        "problems": len(results),
        "generations": total,
        "correct": correct,
        "formatted": formatted,
        "average_at_12_fraction": average_fraction,
        "average_at_12_pct": 100.0 * average_fraction,
    }


def write_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-json", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository-commit", required=True)
    args = parser.parse_args()

    evaluation = json.loads(args.eval_json.read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rebuilt = reconstruct(evaluation)
    reported = float(evaluation["average_at_n_pct"])
    if abs(reported - rebuilt["average_at_12_pct"]) > 1e-9:
        raise RuntimeError(
            f"reported average {reported} disagrees with reconstruction "
            f"{rebuilt['average_at_12_pct']}"
        )
    gate = config["positive_control"]["pass_gate"]
    minimum = float(gate["base_average_at_12_fraction_minimum"])
    maximum = float(gate["base_average_at_12_fraction_maximum"])
    passed = minimum <= rebuilt["average_at_12_fraction"] <= maximum
    receipt = {
        "schema_version": 1,
        "artifact_type": "opd_positive_control_base_gate",
        "campaign_id": config["campaign_id"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit": args.repository_commit,
        "status": "passed" if passed else "failed",
        "decision": "BASELINE_REPRODUCED" if passed else "BASELINE_OUT_OF_RANGE",
        "preregistered_range_fraction": [minimum, maximum],
        "independent_reconstruction": rebuilt,
        "source_eval_json": str(args.eval_json.resolve()),
        "source_eval_sha256": sha256(args.eval_json),
        "config_sha256": sha256(args.config),
    }
    write_exclusive(args.output.resolve(), receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
