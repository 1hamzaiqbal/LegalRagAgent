#!/usr/bin/env python3
"""Materialize the pinned DeepMath collision inventory without training fields."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

try:
    from .data_contract import (
        NORMALIZATION_VERSION,
        math_answer_from_solution,
        normalize_format_insensitive,
        normalize_problem,
        sha256_text,
    )
    from .deepmath_qualification import load_plan as load_qualification_plan
except ImportError:
    from data_contract import (  # type: ignore
        NORMALIZATION_VERSION,
        math_answer_from_solution,
        normalize_format_insensitive,
        normalize_problem,
        sha256_text,
    )
    from deepmath_qualification import load_plan as load_qualification_plan  # type: ignore


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN = ROOT / "configs/opd_math/deepmath_inventory_plan.json"
DEFAULT_QUALIFICATION_PLAN = ROOT / "configs/opd_math/deepmath_qualification_plan.json"
INVENTORY_ID = "deepmath_C_global_inventory_v1"
EXPECTED_TOTAL_ROWS = 1_237_750
EXPECTED_OUTPUT_COLUMNS = (
    "record_id",
    "source",
    "source_split",
    "source_index",
    "problem",
    "answer",
    "stratum",
    "is_evaluation",
    "upstream_id",
    "canonical_problem_sha256",
    "format_problem_sha256",
)
EXPECTED_SOURCE_KEYS = (
    "C",
    "O",
    "O_lineage",
    "M_train",
    "M_test",
    "eval_math500",
    "eval_aime2024",
    "eval_aime_validation",
    "eval_amc_validation",
    "eval_math_beyond",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_inventory_plan(path: Path = DEFAULT_PLAN) -> dict[str, Any]:
    path = path.resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_top = {
        "schema_version",
        "inventory_id",
        "qualification_id",
        "status",
        "teacher_training_authorized",
        "scientific_use_allowed",
        "normalization_version",
        "format_normalization_version",
        "expected_total_rows",
        "output_columns",
        "sources",
    }
    _require(isinstance(payload, dict) and set(payload) == expected_top, "inventory plan schema drifted")
    _require(payload["schema_version"] == 1, "inventory plan schema version drifted")
    _require(payload["inventory_id"] == INVENTORY_ID, "inventory ID drifted")
    _require(payload["qualification_id"] == "deepmath_C_data_feasibility_v1", "qualification ID drifted")
    _require(payload["status"] == "data_materialization_only_not_qualified", "inventory status drifted")
    _require(payload["teacher_training_authorized"] is False, "inventory plan cannot authorize training")
    _require(payload["scientific_use_allowed"] is False, "inventory plan cannot authorize science")
    _require(payload["normalization_version"] == NORMALIZATION_VERSION, "normalization version drifted")
    _require(payload["format_normalization_version"] == "opd-math-format-v1", "format version drifted")
    _require(payload["expected_total_rows"] == EXPECTED_TOTAL_ROWS, "inventory total row count drifted")
    _require(tuple(payload["output_columns"]) == EXPECTED_OUTPUT_COLUMNS, "output columns drifted")
    sources = payload["sources"]
    _require(isinstance(sources, list), "inventory sources must be a list")
    _require(tuple(item.get("key") for item in sources) == EXPECTED_SOURCE_KEYS, "inventory source keys drifted")
    row_total = 0
    for position, source in enumerate(sources):
        expected_keys = {
            "key",
            "source",
            "loader",
            "dataset_id",
            "revision",
            "config",
            "split",
            "expected_rows",
            "problem_field",
            "answer_field",
            "stratum_fields",
            "upstream_id_field",
            "required_columns",
            "forbidden_output_fields",
            "is_evaluation",
        }
        _require(isinstance(source, dict) and set(source) == expected_keys, f"source[{position}] schema drifted")
        _require(source["loader"] in {
            "pinned_deepmath_raw_parquet",
            "huggingface_datasets",
            "huggingface_datasets_math_solution_answer",
        }, f"source[{position}] loader unsupported")
        _require(isinstance(source["revision"], str) and len(source["revision"]) == 40, f"source[{position}] revision unpinned")
        _require(type(source["expected_rows"]) is int and source["expected_rows"] > 0, f"source[{position}] row count invalid")
        _require(type(source["is_evaluation"]) is bool, f"source[{position}] evaluation flag invalid")
        _require(source["problem_field"] in source["required_columns"], f"source[{position}] problem field missing")
        _require(source["answer_field"] in source["required_columns"], f"source[{position}] answer field missing")
        _require(not set(source["forbidden_output_fields"]) & set(EXPECTED_OUTPUT_COLUMNS), f"source[{position}] forbidden field entered output")
        row_total += source["expected_rows"]
    _require(row_total == EXPECTED_TOTAL_ROWS, "per-source inventory rows do not sum to total")
    return {
        **payload,
        "path": str(path),
        "sha256": sha256_file(path),
        "canonical_sha256": canonical_json_sha256(payload),
    }


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
    if status.strip():
        raise ValueError("inventory materialization requires a clean Git checkout")
    return {"commit": commit, "clean": True}


def _output_schema():
    import pyarrow as pa

    return pa.schema(
        [
            ("record_id", pa.string()),
            ("source", pa.string()),
            ("source_split", pa.string()),
            ("source_index", pa.int64()),
            ("problem", pa.large_string()),
            ("answer", pa.large_string()),
            ("stratum", pa.string()),
            ("is_evaluation", pa.bool_()),
            ("upstream_id", pa.string()),
            ("canonical_problem_sha256", pa.string()),
            ("format_problem_sha256", pa.string()),
        ]
    )


def _stable_string(value: Any, *, allow_empty: bool = True) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        result = value.strip()
    elif isinstance(value, (int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            result = ""
        else:
            result = str(value)
    else:
        result = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    if not allow_empty and not result:
        raise ValueError("required source value is empty")
    return result


def _normalized_row(spec: Mapping[str, Any], row: Mapping[str, Any], index: int) -> tuple[dict, bool]:
    problem = _stable_string(row.get(spec["problem_field"]), allow_empty=False)
    answer_missing = False
    if spec["loader"] == "huggingface_datasets_math_solution_answer":
        solution = _stable_string(row.get(spec["answer_field"]), allow_empty=False)
        extracted = math_answer_from_solution(solution)
        answer = "" if extracted is None else extracted
        answer_missing = extracted is None
    else:
        answer = _stable_string(row.get(spec["answer_field"]))
        answer_missing = not bool(answer)
    stratum = "|".join(
        f"{field}={_stable_string(row.get(field))}" for field in spec["stratum_fields"]
    )
    upstream_field = spec["upstream_id_field"]
    upstream_id = "" if upstream_field is None else _stable_string(row.get(upstream_field))
    exact = sha256_text(normalize_problem(problem))
    format_hash = sha256_text(normalize_format_insensitive(problem))
    source_split = str(spec["split"])
    record_id = f"{spec['source']}:{source_split}:{exact}:{index}"
    return (
        {
            "record_id": record_id,
            "source": str(spec["source"]),
            "source_split": source_split,
            "source_index": index,
            "problem": problem,
            "answer": answer,
            "stratum": stratum,
            "is_evaluation": bool(spec["is_evaluation"]),
            "upstream_id": upstream_id,
            "canonical_problem_sha256": exact,
            "format_problem_sha256": format_hash,
        },
        answer_missing,
    )


def _deepmath_batches(
    spec: Mapping[str, Any], raw_root: Path, qualification_plan_path: Path
) -> tuple[Iterator[list[dict]], dict[str, Any]]:
    import pyarrow.parquet as pq

    qualification = load_qualification_plan(qualification_plan_path)
    if spec["revision"] != qualification["candidate"]["revision"]:
        raise ValueError("DeepMath inventory revision differs from qualification plan")
    paths = [raw_root / shard["path"] for shard in qualification["candidate"]["raw_shards"]]
    for path, expected in zip(paths, qualification["candidate"]["raw_shards"], strict=True):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"DeepMath raw shard unavailable: {path}")
        if path.stat().st_size != expected["bytes"] or sha256_file(path) != expected["sha256"]:
            raise ValueError(f"DeepMath raw shard identity drifted: {path}")

    def iterator() -> Iterator[list[dict]]:
        for path in paths:
            parquet = pq.ParquetFile(path)
            actual = set(parquet.schema_arrow.names)
            if not set(spec["required_columns"]).issubset(actual):
                raise ValueError(f"DeepMath raw schema lacks required columns: {path}")
            for batch in parquet.iter_batches(
                batch_size=4096, columns=list(spec["required_columns"])
            ):
                yield batch.to_pylist()

    return iterator(), {
        "loader": spec["loader"],
        "raw_root": str(raw_root.resolve()),
        "raw_shards": [
            {
                "path": str(path.resolve()),
                "bytes": expected["bytes"],
                "sha256": expected["sha256"],
            }
            for path, expected in zip(paths, qualification["candidate"]["raw_shards"], strict=True)
        ],
        "qualification_plan_sha256": qualification["sha256"],
    }


def _hf_batches(spec: Mapping[str, Any], cache_dir: Path) -> tuple[Iterator[list[dict]], dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(
        spec["dataset_id"],
        spec["config"],
        split=spec["split"],
        revision=spec["revision"],
        cache_dir=str(cache_dir),
    )
    if len(dataset) != spec["expected_rows"]:
        raise ValueError(
            f"{spec['key']} row count drifted: {len(dataset)} != {spec['expected_rows']}"
        )
    missing = sorted(set(spec["required_columns"]) - set(dataset.column_names))
    if missing:
        raise ValueError(f"{spec['key']} lacks required columns: {missing}")

    def iterator() -> Iterator[list[dict]]:
        for start in range(0, len(dataset), 4096):
            stop = min(start + 4096, len(dataset))
            columns = dataset[start:stop]
            yield [
                {name: columns[name][offset] for name in spec["required_columns"]}
                for offset in range(stop - start)
            ]

    cache_files = []
    for item in dataset.cache_files:
        path = Path(item["filename"]).resolve()
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"dataset cache file missing or symlinked: {path}")
        cache_files.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return iterator(), {
        "loader": spec["loader"],
        "dataset_id": spec["dataset_id"],
        "revision": spec["revision"],
        "config": spec["config"],
        "split": spec["split"],
        "fingerprint": getattr(dataset, "_fingerprint", None),
        "columns": list(dataset.column_names),
        "cache_files": cache_files,
    }


def _verify_output(path: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    import pyarrow.parquet as pq

    if not path.is_file() or path.is_symlink():
        raise ValueError(f"materialized source file is missing or symlinked: {path}")
    parquet = pq.ParquetFile(path)
    if tuple(parquet.schema_arrow.names) != EXPECTED_OUTPUT_COLUMNS:
        raise ValueError(f"materialized source schema drifted: {path}")
    if parquet.metadata.num_rows != spec["expected_rows"]:
        raise ValueError(f"materialized source row count drifted: {path}")
    return {
        "path": str(path.resolve()),
        "rows": parquet.metadata.num_rows,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "columns": list(parquet.schema_arrow.names),
    }


def _write_source(
    spec: Mapping[str, Any],
    batches: Iterable[list[dict]],
    output: Path,
) -> tuple[dict[str, Any], dict[str, int]]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    temporary = output.with_name(f".{output.name}.partial")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"stale partial materialization must be inspected: {temporary}")
    writer = pq.ParquetWriter(temporary, _output_schema(), compression="zstd")
    rows = 0
    missing_answers = 0
    try:
        for raw_batch in batches:
            normalized = []
            for raw in raw_batch:
                item, missing = _normalized_row(spec, raw, rows)
                normalized.append(item)
                missing_answers += int(missing)
                rows += 1
            if normalized:
                writer.write_table(pa.Table.from_pylist(normalized, schema=_output_schema()))
    finally:
        writer.close()
    if rows != spec["expected_rows"]:
        raise ValueError(f"{spec['key']} materialized {rows} rows, expected {spec['expected_rows']}")
    temporary.replace(output)
    return _verify_output(output, spec), {
        "input_rows": rows,
        "empty_or_unparsed_answers": missing_answers,
    }


def _source_receipt_path(records_dir: Path, source_key: str) -> Path:
    return records_dir / f"{source_key}.receipt.json"


def _write_source_receipt(
    path: Path,
    *,
    plan_sha256: str,
    git_commit: str,
    spec: Mapping[str, Any],
    source_input: Mapping[str, Any],
    output: Mapping[str, Any],
    ingestion: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "stage": "deepmath_inventory_source_materialization",
        "inventory_id": INVENTORY_ID,
        "inventory_plan_sha256": plan_sha256,
        "git_commit": git_commit,
        "source_key": spec["key"],
        "source_spec_canonical_sha256": canonical_json_sha256(spec),
        "input": dict(source_input),
        "output": dict(output),
        "ingestion": dict(ingestion),
        "teacher_training_authorized": False,
        "scientific_use_allowed": False,
    }
    temporary = path.with_name(f".{path.name}.partial")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite source receipt: {path}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    os.chmod(path, 0o444)
    return payload


def _load_source_receipt(
    path: Path,
    *,
    plan_sha256: str,
    git_commit: str,
    spec: Mapping[str, Any],
    observed_output: Mapping[str, Any],
) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"materialized output lacks a source receipt: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != 1
        or payload.get("stage") != "deepmath_inventory_source_materialization"
        or payload.get("inventory_id") != INVENTORY_ID
        or payload.get("inventory_plan_sha256") != plan_sha256
        or payload.get("git_commit") != git_commit
        or payload.get("source_key") != spec["key"]
        or payload.get("source_spec_canonical_sha256") != canonical_json_sha256(spec)
        or payload.get("teacher_training_authorized") is not False
        or payload.get("scientific_use_allowed") is not False
    ):
        raise ValueError(f"source receipt contract drifted: {path}")
    expected_output = payload.get("output")
    if not isinstance(expected_output, dict):
        raise ValueError(f"source receipt lacks output custody: {path}")
    for key in ("path", "rows", "bytes", "sha256", "columns"):
        if expected_output.get(key) != observed_output.get(key):
            raise ValueError(f"source receipt output identity drifted: {path}: {key}")
    if not isinstance(payload.get("input"), dict) or not isinstance(payload.get("ingestion"), dict):
        raise ValueError(f"source receipt lacks input or ingestion custody: {path}")
    return payload


def materialize(
    plan: Mapping[str, Any],
    qualification_plan_path: Path,
    deepmath_raw_root: Path,
    cache_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    import pyarrow

    output_dir = output_dir.resolve()
    try:
        output_dir.relative_to(ROOT.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("inventory output must live outside the Git checkout")
    if output_dir.is_symlink() or (output_dir.exists() and not output_dir.is_dir()):
        raise ValueError("inventory output path is invalid")
    output_dir.mkdir(parents=True, exist_ok=True)
    records_dir = output_dir / "records"
    records_dir.mkdir(exist_ok=True)
    manifest_path = output_dir / "inventory_manifest.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("inventory_plan_sha256") != plan["sha256"]:
            raise ValueError("existing inventory manifest is bound to another plan")
        manifest_git = payload.get("git")
        if not isinstance(manifest_git, dict) or not isinstance(manifest_git.get("commit"), str):
            raise ValueError("existing inventory manifest lacks Git custody")
        for spec in plan["sources"]:
            observed = _verify_output(records_dir / f"{spec['key']}.parquet", spec)
            _load_source_receipt(
                _source_receipt_path(records_dir, spec["key"]),
                plan_sha256=plan["sha256"],
                git_commit=manifest_git["commit"],
                spec=spec,
                observed_output=observed,
            )
            if payload.get("outputs", {}).get(spec["key"]) != observed:
                raise ValueError("existing inventory manifest output identity drifted")
        return payload

    git = _git_state()
    qualification = load_qualification_plan(qualification_plan_path)
    if qualification["qualification_id"] != plan["qualification_id"]:
        raise ValueError("inventory and qualification plans disagree")
    cache_dir.mkdir(parents=True, exist_ok=True)
    inputs = {}
    outputs = {}
    ingestion = {}
    for spec in plan["sources"]:
        target = records_dir / f"{spec['key']}.parquet"
        receipt_path = _source_receipt_path(records_dir, spec["key"])
        if target.exists():
            observed_output = _verify_output(target, spec)
            receipt = _load_source_receipt(
                receipt_path,
                plan_sha256=plan["sha256"],
                git_commit=git["commit"],
                spec=spec,
                observed_output=observed_output,
            )
            outputs[spec["key"]] = observed_output
            inputs[spec["key"]] = receipt["input"]
            ingestion[spec["key"]] = receipt["ingestion"]
            continue
        if spec["loader"] == "pinned_deepmath_raw_parquet":
            batches, source_input = _deepmath_batches(
                spec, deepmath_raw_root, qualification_plan_path
            )
        else:
            batches, source_input = _hf_batches(spec, cache_dir)
        output, stats = _write_source(spec, batches, target)
        _write_source_receipt(
            receipt_path,
            plan_sha256=plan["sha256"],
            git_commit=git["commit"],
            spec=spec,
            source_input=source_input,
            output=output,
            ingestion=stats,
        )
        os.chmod(target, 0o444)
        inputs[spec["key"]] = source_input
        outputs[spec["key"]] = output
        ingestion[spec["key"]] = stats

    total_rows = sum(item["rows"] for item in outputs.values())
    if total_rows != EXPECTED_TOTAL_ROWS:
        raise ValueError("materialized inventory total row count drifted")
    payload = {
        "schema_version": 1,
        "inventory_id": INVENTORY_ID,
        "qualification_id": plan["qualification_id"],
        "stage": "global_inventory_materialization_only",
        "status": "passed",
        "git": git,
        "inventory_plan_path": plan["path"],
        "inventory_plan_sha256": plan["sha256"],
        "inventory_plan_canonical_sha256": plan["canonical_sha256"],
        "qualification_plan_path": str(qualification_plan_path.resolve()),
        "qualification_plan_sha256": qualification["sha256"],
        "environment": {
            "python": os.sys.version.split()[0],
            "datasets": importlib.metadata.version("datasets"),
            "pyarrow": pyarrow.__version__,
        },
        "launcher": (
            {
                "path": str(Path(os.environ["OPD_INVENTORY_LAUNCHER_PATH"]).resolve()),
                "sha256": sha256_file(Path(os.environ["OPD_INVENTORY_LAUNCHER_PATH"])),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID", "none"),
            }
            if os.environ.get("OPD_INVENTORY_LAUNCHER_PATH")
            else None
        ),
        "cache_dir": str(cache_dir.resolve()),
        "inputs": inputs,
        "ingestion": ingestion,
        "outputs": outputs,
        "total_rows": total_rows,
        "output_columns": list(EXPECTED_OUTPUT_COLUMNS),
        "forbidden_C_fields_absent": True,
        "teacher_training_authorized": False,
        "scientific_use_allowed": False,
        "remaining_gates": [
            "complete global exact/format/semantic collision audit",
            "C gold parseability and prompt-bound audit",
            "raw-model feasibility",
            "deterministic C role freeze",
        ],
    }
    temporary = manifest_path.with_name(f".{manifest_path.name}.partial")
    if temporary.exists():
        raise FileExistsError(f"stale inventory manifest partial exists: {temporary}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    os.chmod(manifest_path, 0o444)
    for item in outputs.values():
        os.chmod(item["path"], 0o444)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--qualification-plan", type=Path, default=DEFAULT_QUALIFICATION_PLAN)
    parser.add_argument("--deepmath-raw-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = load_inventory_plan(args.plan)
    result = materialize(
        plan,
        args.qualification_plan,
        args.deepmath_raw_root,
        args.cache_dir,
        args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
