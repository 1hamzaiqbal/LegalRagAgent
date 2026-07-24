#!/usr/bin/env python3
"""Project the exact OPSD trainer-visible fields into metadata-free Parquet."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq


EXPECTED_ROWS = 29_434
REQUIRED_COLUMNS = ("problem", "solution", "conversations")
ALGORITHM = "project_opsd_trl026_fields_strip_arrow_metadata_v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_bytes(digest, payload: bytes) -> None:
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def conversation_bytes(value) -> bytes:
    if not isinstance(value, list) or not value:
        raise RuntimeError("conversations must be a nonempty list")
    for message in value:
        if not isinstance(message, dict):
            raise RuntimeError("conversation messages must be objects")
        if not isinstance(message.get("from"), str) or not isinstance(
            message.get("value"), str
        ):
            raise RuntimeError("conversation messages require string from/value")
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def combined_row_sequence(paths: list[Path]) -> tuple[int, str]:
    digest = hashlib.sha256()
    rows = 0
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            columns=list(REQUIRED_COLUMNS), batch_size=512
        ):
            problems = batch.column(0).to_pylist()
            solutions = batch.column(1).to_pylist()
            conversations = batch.column(2).to_pylist()
            for problem, solution, conversation in zip(
                problems, solutions, conversations, strict=True
            ):
                if not isinstance(problem, str) or not isinstance(solution, str):
                    raise RuntimeError("problem and solution must be strings")
                add_bytes(digest, problem.encode("utf-8"))
                add_bytes(digest, solution.encode("utf-8"))
                add_bytes(digest, conversation_bytes(conversation))
                rows += 1
    return rows, digest.hexdigest()


def write_exclusive(path: Path, payload: dict) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def normalize(source_root: Path, output_root: Path, repository_commit: str) -> dict:
    source_files = sorted(source_root.glob("data/*.parquet"))
    if len(source_files) != 2:
        raise RuntimeError(f"expected exactly two source shards, got {source_files}")
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"refusing to replace trainer data: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    build = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.building.", dir=output_root.parent
        )
    )
    try:
        source_records = []
        target_records = []
        for index, source in enumerate(source_files):
            schema = pq.read_schema(source)
            missing = [name for name in REQUIRED_COLUMNS if name not in schema.names]
            if missing:
                raise RuntimeError(f"source shard {source} lacks columns {missing}")
            source_rows, source_digest = combined_row_sequence([source])
            table = pq.read_table(source, columns=list(REQUIRED_COLUMNS))
            table = table.replace_schema_metadata(None)
            target = build / f"train-{index:05d}-of-{len(source_files):05d}.parquet"
            pq.write_table(
                table,
                target,
                compression="zstd",
                use_dictionary=False,
                write_statistics=True,
            )
            target_schema = pq.read_schema(target)
            if target_schema.metadata:
                raise RuntimeError("trainer Parquet unexpectedly retained metadata")
            target_rows, target_digest = combined_row_sequence([target])
            if (target_rows, target_digest) != (source_rows, source_digest):
                raise RuntimeError("trainer projection changed ordered source fields")
            source_records.append(
                {
                    "path": str(source.resolve()),
                    "bytes": source.stat().st_size,
                    "sha256": sha256(source),
                    "rows": source_rows,
                    "trainer_field_sequence_sha256": source_digest,
                }
            )
            target_records.append(
                {
                    "path": target.name,
                    "bytes": target.stat().st_size,
                    "sha256": sha256(target),
                    "rows": target_rows,
                    "trainer_field_sequence_sha256": target_digest,
                    "columns": target_schema.names,
                    "schema_metadata_absent": target_schema.metadata is None,
                }
            )

        source_rows, source_digest = combined_row_sequence(source_files)
        target_files = sorted(build.glob("*.parquet"))
        target_rows, target_digest = combined_row_sequence(target_files)
        if source_rows != EXPECTED_ROWS or target_rows != EXPECTED_ROWS:
            raise RuntimeError(
                f"expected {EXPECTED_ROWS} rows, got source={source_rows}, "
                f"target={target_rows}"
            )
        if source_digest != target_digest:
            raise RuntimeError("combined trainer-field digest changed")
        manifest = {
            "schema_version": 2,
            "artifact_type": "opd_positive_control_trainer_data",
            "campaign_id": "opd_identifiability_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "repository_commit": repository_commit,
            "algorithm": ALGORITHM,
            "required_columns": list(REQUIRED_COLUMNS),
            "rows": target_rows,
            "trainer_field_sequence_sha256": target_digest,
            "source_shards": source_records,
            "trainer_shards": target_records,
        }
        write_exclusive(build / "manifest.json", manifest)
        for path in build.glob("*.parquet"):
            os.chmod(path, 0o444)
        os.replace(build, output_root)
        os.chmod(output_root, 0o555)
        return manifest
    except BaseException:
        print(f"preserved failed trainer-data projection at {build}")
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repository-commit", required=True)
    args = parser.parse_args()
    manifest = normalize(
        args.source_root.resolve(),
        args.output_root.resolve(),
        args.repository_commit,
    )
    print(json.dumps({"status": "passed", "rows": manifest["rows"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
