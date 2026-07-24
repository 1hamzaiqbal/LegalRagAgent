#!/usr/bin/env python3
"""Audit OPSD trainer data through the pinned TRL 0.26 preprocessing path."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import datasets
import pyarrow
import pyarrow.parquet as pq
import transformers
import trl
from datasets import load_dataset
from transformers import AutoTokenizer
from trl.data_utils import (
    is_conversational,
    is_conversational_from_value,
    maybe_convert_to_chatml,
)


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
        raise RuntimeError("independent audit found an empty conversation")
    for message in value:
        if not isinstance(message, dict):
            raise RuntimeError("independent audit found a non-object message")
        if not isinstance(message.get("from"), str) or not isinstance(
            message.get("value"), str
        ):
            raise RuntimeError("independent audit requires string from/value fields")
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
            for problem, solution, conversation in zip(
                batch.column(0).to_pylist(),
                batch.column(1).to_pylist(),
                batch.column(2).to_pylist(),
                strict=True,
            ):
                if not isinstance(problem, str) or not isinstance(solution, str):
                    raise RuntimeError("independent audit found non-string task fields")
                add_bytes(digest, problem.encode("utf-8"))
                add_bytes(digest, solution.encode("utf-8"))
                add_bytes(digest, conversation_bytes(conversation))
                rows += 1
    return rows, digest.hexdigest()


def token_sequence(dataset: datasets.Dataset) -> tuple[int, int, int, str]:
    digest = hashlib.sha256()
    sequences = 0
    tokens = 0
    maximum = 0
    for batch in dataset.iter(batch_size=128):
        for input_ids in batch["input_ids"]:
            if not isinstance(input_ids, list) or not input_ids:
                raise RuntimeError("tokenization produced an empty input sequence")
            digest.update(len(input_ids).to_bytes(8, "big"))
            for token_id in input_ids:
                digest.update(int(token_id).to_bytes(8, "big", signed=True))
            sequences += 1
            tokens += len(input_ids)
            maximum = max(maximum, len(input_ids))
    return sequences, tokens, maximum, digest.hexdigest()


def load_collator(upstream: Path):
    path = upstream / "data_collator.py"
    spec = importlib.util.spec_from_file_location("audited_opsd_data_collator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the pinned OPSD data collator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SelfDistillationDataCollator


def write_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def audit(
    trainer_root: Path,
    producer_commit: str,
    auditor_commit: str,
    model_dir: Path,
    upstream: Path,
    output: Path,
) -> dict:
    manifest_path = trainer_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("artifact_type") != "opd_positive_control_trainer_data":
        raise RuntimeError("unexpected trainer-data artifact type")
    if manifest.get("repository_commit") != producer_commit:
        raise RuntimeError("trainer-data producer commit drifted")
    if manifest.get("algorithm") != ALGORITHM:
        raise RuntimeError("trainer-data projection algorithm drifted")
    if manifest.get("required_columns") != list(REQUIRED_COLUMNS):
        raise RuntimeError("trainer-data required columns drifted")

    source_files = [Path(row["path"]) for row in manifest["source_shards"]]
    target_files = [trainer_root / row["path"] for row in manifest["trainer_shards"]]
    for path, record in zip(source_files, manifest["source_shards"], strict=True):
        if sha256(path) != record["sha256"]:
            raise RuntimeError(f"source shard changed: {path}")
    for path, record in zip(target_files, manifest["trainer_shards"], strict=True):
        if sha256(path) != record["sha256"]:
            raise RuntimeError(f"trainer shard changed: {path}")
        schema = pq.read_schema(path)
        if tuple(schema.names) != REQUIRED_COLUMNS or schema.metadata:
            raise RuntimeError(f"trainer shard schema drifted: {path}")

    source_rows, source_digest = combined_row_sequence(source_files)
    target_rows, target_digest = combined_row_sequence(target_files)
    if source_rows != EXPECTED_ROWS or target_rows != EXPECTED_ROWS:
        raise RuntimeError("independent trainer-data row-count check failed")
    if source_digest != target_digest or target_digest != manifest.get(
        "trainer_field_sequence_sha256"
    ):
        raise RuntimeError("independent trainer-field sequence check failed")

    loaded = load_dataset(
        "parquet",
        data_files={"train": [str(path) for path in target_files]},
        split="train",
    )
    if len(loaded) != EXPECTED_ROWS or tuple(loaded.column_names) != REQUIRED_COLUMNS:
        raise RuntimeError("datasets 3.6 loaded an unexpected trainer dataset")
    first = loaded[0]
    if not is_conversational_from_value(first):
        raise RuntimeError("pinned TRL does not recognize source conversations")
    converted = loaded.map(
        maybe_convert_to_chatml,
        remove_columns="conversations",
        desc="Auditing pinned TRL ChatML conversion",
    )
    if not is_conversational(converted[0]):
        raise RuntimeError("pinned TRL ChatML conversion did not produce messages")
    if not {"problem", "solution", "messages"}.issubset(converted.column_names):
        raise RuntimeError("ChatML conversion removed custom-collator task fields")

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        revision=None,
        local_files_only=True,
        trust_remote_code=False,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(example):
        processed = tokenizer.apply_chat_template(
            example["messages"],
            return_dict=True,
            tokenize=True,
            return_assistant_tokens_mask=False,
        )
        return {"input_ids": processed["input_ids"]}

    tokenized = converted.map(
        tokenize_fn,
        desc="Auditing pinned TRL tokenization",
    )
    if not {"problem", "solution", "messages", "input_ids"}.issubset(
        tokenized.column_names
    ):
        raise RuntimeError("tokenization removed custom-collator task fields")
    over_limit = sum(
        len(input_ids) > 20_000
        for batch in tokenized.iter(batch_size=128)
        for input_ids in batch["input_ids"]
    )
    tokenized = tokenized.map(
        lambda example: {"input_ids": example["input_ids"][:20_000]},
        desc="Auditing pinned TRL max-length truncation",
    )
    sequences, tokens, maximum, token_digest = token_sequence(tokenized)
    if sequences != EXPECTED_ROWS:
        raise RuntimeError("tokenization did not cover every trainer row")

    collator_class = load_collator(upstream)
    collator = collator_class(
        tokenizer=tokenizer,
        max_length=20_000,
        reason_first=False,
        student_thinking=False,
        teacher_thinking=True,
    )
    collated = collator([tokenized[index] for index in range(4)])
    required_outputs = {
        "student_prompts",
        "student_prompt_attention_mask",
        "teacher_prompts",
        "teacher_prompt_attention_mask",
    }
    if not required_outputs.issubset(collated):
        raise RuntimeError("pinned custom collator did not emit required tensors")

    receipt = {
        "schema_version": 2,
        "artifact_type": "opd_positive_control_trainer_data_audit",
        "campaign_id": "opd_identifiability_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "decision": "PINNED_TRL026_TRAINER_DATA_COMPATIBLE",
        "producer_commit": producer_commit,
        "auditor_commit": auditor_commit,
        "trainer_root": str(trainer_root.resolve()),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256(manifest_path),
        "rows": target_rows,
        "columns": list(REQUIRED_COLUMNS),
        "trainer_field_sequence_sha256": target_digest,
        "datasets_version": datasets.__version__,
        "pyarrow_version": pyarrow.__version__,
        "transformers_version": transformers.__version__,
        "trl_version": trl.__version__,
        "model_dir": str(model_dir.resolve()),
        "upstream_commit": subprocess.check_output(
            ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_shard_sha256": [sha256(path) for path in source_files],
        "trainer_shard_sha256": [sha256(path) for path in target_files],
        "chatml_conversion_rows": len(converted),
        "tokenized_sequences": sequences,
        "tokenized_tokens": tokens,
        "maximum_tokenized_length": maximum,
        "pretruncate_sequences_over_20000": over_limit,
        "token_sequence_sha256": token_digest,
        "collator_batch_size": int(collated["student_prompts"].shape[0]),
        "collator_outputs": sorted(collated),
    }
    if receipt["upstream_commit"] != "7448751f307a9cdbcc1246dd1565a1a605b443df":
        raise RuntimeError("upstream OPSD commit drifted")
    write_exclusive(output.resolve(), receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainer-root", type=Path, required=True)
    parser.add_argument("--producer-commit", required=True)
    parser.add_argument("--auditor-commit", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = audit(
        args.trainer_root.resolve(),
        args.producer_commit,
        args.auditor_commit,
        args.model_dir.resolve(),
        args.upstream.resolve(),
        args.output,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
