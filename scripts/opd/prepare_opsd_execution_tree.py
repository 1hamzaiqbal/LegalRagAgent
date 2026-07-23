#!/usr/bin/env python3
"""Create an immutable OPSD execution tree with data-locality-only edits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


PINNED_UPSTREAM_COMMIT = "7448751f307a9cdbcc1246dd1565a1a605b443df"

TRAIN_OLD = 'dataset = load_dataset("siyanzhao/Openthoughts_math_30k_opsd")'
TRAIN_NEW = '''train_parquet = os.environ["LEGALRAG_OPSD_TRAIN_PARQUET"]
    dataset = load_dataset("parquet", data_files={"train": train_parquet})'''

EVAL_OLD = 'dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")'
EVAL_NEW = '''aime_parquet = __import__("os").environ["LEGALRAG_OPSD_AIME24_PARQUET"]
        dataset = load_dataset("parquet", data_files=aime_parquet, split="train")'''


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True
    ).strip()


def replace_once(path: Path, old: str, new: str) -> dict[str, str]:
    source = path.read_text(encoding="utf-8")
    if source.count(old) != 1:
        raise RuntimeError(f"expected exactly one pinned source match in {path}")
    before = sha256(path)
    path.write_text(source.replace(old, new), encoding="utf-8")
    return {"before_sha256": before, "after_sha256": sha256(path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--repository-commit", required=True)
    args = parser.parse_args()

    upstream = args.upstream.resolve()
    target = args.target.resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to replace execution tree: {target}")
    if git(upstream, "rev-parse", "HEAD") != PINNED_UPSTREAM_COMMIT:
        raise RuntimeError("upstream OPSD checkout is not at the pinned commit")
    if git(upstream, "status", "--porcelain=v1", "--untracked-files=no"):
        raise RuntimeError("upstream OPSD checkout has tracked modifications")

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(upstream, target, ignore=shutil.ignore_patterns(".git"))
    changes = {
        "opsd_train.py": replace_once(target / "opsd_train.py", TRAIN_OLD, TRAIN_NEW),
        "eval/evaluate_math.py": replace_once(
            target / "eval/evaluate_math.py", EVAL_OLD, EVAL_NEW
        ),
    }
    manifest = {
        "schema_version": 1,
        "artifact_type": "opsd_data_locality_execution_tree",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit": args.repository_commit,
        "upstream_commit": PINNED_UPSTREAM_COMMIT,
        "semantic_edits": [],
        "harness_edits": [
            "load pinned training parquet from LEGALRAG_OPSD_TRAIN_PARQUET",
            "load pinned AIME24 parquet from LEGALRAG_OPSD_AIME24_PARQUET",
        ],
        "changed_files": changes,
    }
    manifest_path = target / "LEGALRAG_EXECUTION_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.chmod(manifest_path, 0o444)
    print(json.dumps({"status": "passed", "target": str(target)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
