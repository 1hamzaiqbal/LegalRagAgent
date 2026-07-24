#!/usr/bin/env python3
"""Independently audit normalized OPSD data with the pinned training runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import datasets
import pyarrow
import pyarrow.parquet as pq
from datasets import load_dataset

try:
    from scripts.opd.normalize_positive_control_data import (
        ALGORITHM,
        EXPECTED_ROWS,
        REQUIRED_COLUMNS,
    )
except ModuleNotFoundError:  # Direct execution by absolute path on EIT.
    from normalize_positive_control_data import (  # type: ignore[no-redef]
        ALGORITHM,
        EXPECTED_ROWS,
        REQUIRED_COLUMNS,
    )


def sha256(path: Path) -> str:
    """Recompute file custody without using the producer implementation."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def combined_row_sequence(paths: list[Path]) -> tuple[int, str]:
    """Independently hash length-framed ordered problem/solution pairs."""
    digest = hashlib.sha256()
    rows = 0
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            columns=list(REQUIRED_COLUMNS), batch_size=1024
        ):
            for problem, solution in zip(
                batch.column(0).to_pylist(),
                batch.column(1).to_pylist(),
                strict=True,
            ):
                for value in (problem, solution):
                    if not isinstance(value, str):
                        raise RuntimeError(
                            "independent audit observed a non-string cell"
                        )
                    encoded = value.encode("utf-8")
                    digest.update(len(encoded).to_bytes(8, "big"))
                    digest.update(encoded)
                rows += 1
    return rows, digest.hexdigest()


def load_object(path: Path, label: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return payload


def write_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def dataset_row_digest(dataset: datasets.Dataset) -> str:
    digest = hashlib.sha256()
    for batch in dataset.iter(batch_size=1024):
        for problem, solution in zip(
            batch["problem"], batch["solution"], strict=True
        ):
            for value in (problem, solution):
                encoded = value.encode("utf-8")
                digest.update(len(encoded).to_bytes(8, "big"))
                digest.update(encoded)
    return digest.hexdigest()


def audit(
    normalized_root: Path,
    producer_commit: str,
    auditor_commit: str,
    output: Path,
) -> dict:
    manifest_path = normalized_root / "manifest.json"
    manifest = load_object(manifest_path, "normalized-data manifest")
    if manifest.get("artifact_type") != "opd_positive_control_normalized_data":
        raise RuntimeError("unexpected normalized-data artifact type")
    if manifest.get("repository_commit") != producer_commit:
        raise RuntimeError("normalized-data producer commit drifted")
    if manifest.get("algorithm") != ALGORITHM:
        raise RuntimeError("normalization algorithm drifted")
    if manifest.get("rows") != EXPECTED_ROWS:
        raise RuntimeError("normalized-data row count drifted")

    source_files = [Path(row["path"]) for row in manifest["source_shards"]]
    target_files = [normalized_root / row["path"] for row in manifest["normalized_shards"]]
    for path, record in zip(source_files, manifest["source_shards"], strict=True):
        if sha256(path) != record["sha256"]:
            raise RuntimeError(f"source shard changed: {path}")
    for path, record in zip(target_files, manifest["normalized_shards"], strict=True):
        if sha256(path) != record["sha256"]:
            raise RuntimeError(f"normalized shard changed: {path}")
        schema = pq.read_schema(path)
        if tuple(schema.names) != REQUIRED_COLUMNS:
            raise RuntimeError(f"normalized columns drifted: {schema.names}")
        if schema.metadata:
            raise RuntimeError(f"normalized metadata is not empty: {path}")

    source_rows, source_digest = combined_row_sequence(source_files)
    target_rows, target_digest = combined_row_sequence(target_files)
    if source_rows != EXPECTED_ROWS or target_rows != EXPECTED_ROWS:
        raise RuntimeError("independent source/target row-count check failed")
    if source_digest != target_digest or target_digest != manifest["row_sequence_sha256"]:
        raise RuntimeError("independent ordered problem/solution digest check failed")

    loaded = load_dataset(
        "parquet",
        data_files={"train": [str(path) for path in target_files]},
        split="train",
    )
    if len(loaded) != EXPECTED_ROWS:
        raise RuntimeError("datasets runtime loaded the wrong row count")
    if tuple(loaded.column_names) != REQUIRED_COLUMNS:
        raise RuntimeError(f"datasets runtime inferred wrong columns: {loaded.column_names}")
    loaded_digest = dataset_row_digest(loaded)
    if loaded_digest != target_digest:
        raise RuntimeError("datasets runtime changed ordered problem/solution rows")

    receipt = {
        "schema_version": 1,
        "artifact_type": "opd_positive_control_normalized_data_audit",
        "campaign_id": "opd_identifiability_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "decision": "NORMALIZED_DATA_LOAD_COMPATIBLE",
        "producer_commit": producer_commit,
        "auditor_commit": auditor_commit,
        "normalized_root": str(normalized_root.resolve()),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256(manifest_path),
        "rows": target_rows,
        "columns": list(REQUIRED_COLUMNS),
        "row_sequence_sha256": target_digest,
        "datasets_version": datasets.__version__,
        "pyarrow_version": pyarrow.__version__,
        "source_shard_sha256": [sha256(path) for path in source_files],
        "normalized_shard_sha256": [sha256(path) for path in target_files],
    }
    write_exclusive(output.resolve(), receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalized-root", type=Path, required=True)
    parser.add_argument("--producer-commit", required=True)
    parser.add_argument("--auditor-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = audit(
        args.normalized_root.resolve(),
        args.producer_commit,
        args.auditor_commit,
        args.output,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
