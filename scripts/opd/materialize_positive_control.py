#!/usr/bin/env python3
"""Materialize and hash the pinned OPSD positive-control datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import load_dataset
from huggingface_hub import snapshot_download


TRAIN_ID = "siyanzhao/Openthoughts_math_30k_opsd"
TRAIN_REVISION = "1f33e9dc2e8a1c639ca74f8024ad4a9f1f5eae62"
TRAIN_ROWS = 29_434
AIME_ID = "HuggingFaceH4/aime_2024"
AIME_REVISION = "2fe88a2f1091d5048c0f36abc874fb997b3dd99a"
AIME_ROWS = 30


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return records


def parquet_files(root: Path) -> list[str]:
    files = sorted(str(path) for path in root.rglob("*.parquet"))
    if not files:
        raise RuntimeError(f"no parquet files found under {root}")
    return files


def validate_dataset(
    root: Path, expected_rows: int, required_columns: set[str]
) -> dict[str, Any]:
    dataset = load_dataset(
        "parquet", data_files={"train": parquet_files(root)}, split="train"
    )
    columns = set(dataset.column_names)
    if len(dataset) != expected_rows:
        raise RuntimeError(
            f"row-count mismatch for {root}: {len(dataset)} != {expected_rows}"
        )
    missing = sorted(required_columns - columns)
    if missing:
        raise RuntimeError(f"missing required columns for {root}: {missing}")
    return {
        "rows": len(dataset),
        "columns": sorted(columns),
        "dataset_fingerprint": dataset._fingerprint,
    }


def download_dataset(repo_id: str, revision: str, target: Path) -> None:
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=target,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repository-commit", required=True)
    args = parser.parse_args()

    output = args.output_root.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to replace positive-control data: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    build = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.building.", dir=output.parent)
    )
    try:
        train_root = build / "opsd_train"
        aime_root = build / "aime24"
        download_dataset(TRAIN_ID, TRAIN_REVISION, train_root)
        download_dataset(AIME_ID, AIME_REVISION, aime_root)

        train_validation = validate_dataset(
            train_root, TRAIN_ROWS, {"problem", "solution"}
        )
        aime_validation = validate_dataset(
            aime_root, AIME_ROWS, {"id", "problem", "answer"}
        )
        manifest = {
            "schema_version": 1,
            "artifact_type": "opd_positive_control_data_manifest",
            "campaign_id": "opd_identifiability_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "repository_commit": args.repository_commit,
            "sources": {
                "opsd_train": {
                    "id": TRAIN_ID,
                    "revision": TRAIN_REVISION,
                    **train_validation,
                },
                "aime24": {
                    "id": AIME_ID,
                    "revision": AIME_REVISION,
                    **aime_validation,
                },
            },
            "files": file_records(build),
        }
        manifest_path = build / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        for path in build.rglob("*"):
            if path.is_file():
                os.chmod(path, 0o444)
        for path in sorted(
            (path for path in build.rglob("*") if path.is_dir()), reverse=True
        ):
            os.chmod(path, 0o555)
        os.chmod(build, 0o555)
        os.replace(build, output)
    except BaseException:
        print(f"preserved failed materialization at {build}")
        raise

    print(json.dumps({"status": "passed", "output_root": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
