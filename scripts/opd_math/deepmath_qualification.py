#!/usr/bin/env python3
"""Validate the outcome-blind DeepMath C qualification and raw-file custody.

This module does not qualify a teacher and cannot launch training.  It binds
the candidate, collision inventory, feasibility surface, and terminal rules;
``verify-raw`` verifies only the downloaded source Parquet identities/schema.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = ROOT / "configs" / "opd_math" / "deepmath_qualification_plan.json"
QUALIFICATION_ID = "deepmath_C_data_feasibility_v1"
CANDIDATE_REVISION = "5cf055d1fe3d7a2eb19719ac020211469736ae44"
EXPECTED_COLUMNS = (
    "question",
    "final_answer",
    "difficulty",
    "topic",
    "r1_solution_1",
    "r1_solution_2",
    "r1_solution_3",
)
FORBIDDEN_TRAINING_FIELDS = frozenset(
    {"r1_solution_1", "r1_solution_2", "r1_solution_3"}
)
EXPECTED_SUCCESSOR_MATRIX = ("C_C", "C_O", "O_C", "O_O")
EXPECTED_RAW_SHARDS = tuple(
    {
        "path": f"data/train-{index:05d}-of-00010.parquet",
        "bytes": size,
        "sha256": digest,
    }
    for index, (size, digest) in enumerate(
        (
            (216818587, "d6412432f30425e848a224dc641e681eb1ed51b970d52536eda7daefc01d8c8b"),
            (212374597, "eef9d3012456239eb0f4cd462ac7bebb7d6c4f675c41329c680ef8a506ded512"),
            (213616309, "20e4dc6527d94c2057a9b727f8b58994395de602e284b3294f7d2451f75d681a"),
            (208137532, "c5b20135f93e7890da191973ad77d88e67c95144ccc169ec592489f044e7c38a"),
            (207032084, "4153e531d78ca278d2e12e6d08099b66126b26b99ca6a92b3d6f95c156818749"),
            (207744721, "110586bdc6f35b0434bccbd582f1e97e8328da0752ccd36590a50178885f0360"),
            (207048095, "e39c00ed42a6a1af74ddc042840884a59adcb9139a23f535e399ddf0c292ef76"),
            (207374607, "fdeb213b5c2d0bb50f1081ae48b7ce4fa38147efe623de5d6d836a32f4044dad"),
            (273432098, "d8e5b1417f0364312896d259efbea0600c1fac22eacb2f71187c5b0e9704f388"),
            (182527630, "67639c6620cabce348e91c2cc331c4877307ed1a5d8fbd1c0d7dcca561cea8df"),
        )
    )
)
EXPECTED_COLLISION_INVENTORY = (
    {"source": "C", "dataset_id": "zwhe99/DeepMath-103K", "revision": CANDIDATE_REVISION, "scope": "all_rows"},
    {"source": "O", "dataset_id": "open-r1/OpenR1-Math-220k", "revision": "e4e141ec9dea9f8326f4d347be56105859b2bd68", "scope": "all_rows_and_upstream_problem_ids"},
    {"source": "O_lineage", "dataset_id": "AI-MO/NuminaMath-1.5", "revision": "1b05109f9e5c1ad06c0663519502416c30b300f8", "scope": "all_rows"},
    {"source": "M", "dataset_id": "DigitalLearningGmbH/MATH-lighteval", "revision": "0530c78699ea5e8eb5530600900e1f328b48acad", "scope": "train_and_test"},
    {"source": "eval_math500", "dataset_id": "HuggingFaceH4/MATH-500", "revision": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be", "scope": "all_rows"},
    {"source": "eval_aime2024", "dataset_id": "HuggingFaceH4/aime_2024", "revision": "2fe88a2f1091d5048c0f36abc874fb997b3dd99a", "scope": "all_rows"},
    {"source": "eval_aime_validation", "dataset_id": "AI-MO/aimo-validation-aime", "revision": "13f9e12f613e720c2a2b2f345dd04b998a29494d", "scope": "all_rows"},
    {"source": "eval_amc_validation", "dataset_id": "AI-MO/aimo-validation-amc", "revision": "69d78a4a2c840e82d69af6bc742bda09005f6316", "scope": "all_rows"},
    {"source": "eval_math_beyond", "dataset_id": "brendel-group/MATH-Beyond", "revision": "1a2a6294ff8d673aeb271ab3eff272e8d6b48c2d", "scope": "all_rows"},
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
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


def validate_plan(payload: Mapping[str, Any]) -> dict[str, Any]:
    expected_top = {
        "schema_version",
        "qualification_id",
        "status",
        "candidate_source",
        "plan_alone_authorizes_teacher_training",
        "candidate",
        "global_collision_inventory",
        "collision_contract",
        "data_gates",
        "raw_model_feasibility",
        "role_freeze",
        "successor_boundary",
    }
    _require(set(payload) == expected_top, "DeepMath plan top-level schema drifted")
    _require(payload["schema_version"] == 1, "unsupported DeepMath plan schema")
    _require(payload["qualification_id"] == QUALIFICATION_ID, "DeepMath plan ID drifted")
    _require(
        payload["status"] == "data_only_pre_teacher_not_qualified",
        "DeepMath status must remain pre-teacher and unqualified",
    )
    _require(payload["candidate_source"] == "C", "DeepMath source must be C")
    _require(
        payload["plan_alone_authorizes_teacher_training"] is False,
        "DeepMath plan must not authorize teacher training",
    )

    candidate = payload["candidate"]
    _require(isinstance(candidate, dict), "DeepMath candidate must be an object")
    _require(candidate.get("dataset_id") == "zwhe99/DeepMath-103K", "dataset drifted")
    _require(candidate.get("revision") == CANDIDATE_REVISION, "revision drifted")
    _require(candidate.get("config") == "default" and candidate.get("split") == "train", "split drifted")
    _require(candidate.get("license") == "MIT", "license drifted")
    _require(candidate.get("expected_rows") == 103022, "row count drifted")
    _require(tuple(candidate.get("expected_columns", ())) == EXPECTED_COLUMNS, "columns drifted")
    _require(
        frozenset(candidate.get("forbidden_training_fields", ()))
        == FORBIDDEN_TRAINING_FIELDS,
        "forbidden R1 trace fields drifted",
    )
    approved = frozenset(candidate.get("approved_fields", ()))
    _require(not approved & FORBIDDEN_TRAINING_FIELDS, "R1 traces entered approved fields")
    shards = candidate.get("raw_shards")
    _require(isinstance(shards, list) and len(shards) == 10, "exactly ten raw shards required")
    for index, shard in enumerate(shards):
        _require(isinstance(shard, dict) and set(shard) == {"path", "bytes", "sha256"}, f"raw shard {index} schema drifted")
        _require(shard["path"] == f"data/train-{index:05d}-of-00010.parquet", f"raw shard {index} path drifted")
        _require(type(shard["bytes"]) is int and shard["bytes"] > 0, f"raw shard {index} size invalid")
        _require(isinstance(shard["sha256"], str) and len(shard["sha256"]) == 64, f"raw shard {index} hash invalid")
    _require(tuple(shards) == EXPECTED_RAW_SHARDS, "raw shard custody identities drifted")

    inventory = payload["global_collision_inventory"]
    _require(isinstance(inventory, list), "global collision inventory must be a list")
    for item in inventory:
        _require(set(item) == {"source", "dataset_id", "revision", "scope"}, "collision inventory schema drifted")
        _require(isinstance(item["revision"], str) and len(item["revision"]) == 40, "collision inventory revision is not immutable")
    _require(tuple(inventory) == EXPECTED_COLLISION_INVENTORY, "global collision inventory drifted")

    collision = payload["collision_contract"]
    _require(collision.get("require_complete_inventory") is True, "complete collision inventory is required")
    _require(collision.get("allow_skipped_collision_buckets") is False, "collision buckets cannot be skipped")
    _require(collision.get("allow_unresolved_semantic_candidates") is False, "semantic candidates must be resolved")
    _require(collision.get("allow_unresolved_label_conflicts") is False, "label conflicts must be resolved")
    _require(collision.get("quarantine_any_evaluation_touch") is True, "evaluation-touch clusters must be quarantined")

    gates = payload["data_gates"]
    _require(gates.get("minimum_unique_eligible_clusters") == 5000, "eligible-cluster gate drifted")
    _require(gates.get("minimum_gold_parseability") == 0.99, "gold parseability gate drifted")
    _require(gates.get("maximum_unresolved_label_conflicts") == 0, "label-conflict gate drifted")
    _require(gates.get("maximum_prompt_truncations") == 0, "prompt-truncation gate drifted")
    _require(gates.get("source_independent_max_prompt_tokens") == 1536, "prompt bound drifted")

    feasibility = payload["raw_model_feasibility"]
    expected_feasibility = {
        "records": 512,
        "seed": 0,
        "samples_per_record": 4,
        "thinking_mode": False,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "max_completion_tokens": 512,
        "minimum_teacher_mean_completion_reward": 0.05,
        "maximum_teacher_mean_completion_reward": 0.95,
        "minimum_student_pass_at_4": 0.05,
        "minimum_student_mixed_group_fraction": 0.05,
        "maximum_verifier_error_fraction": 0.001,
    }
    for key, expected in expected_feasibility.items():
        _require(feasibility.get(key) == expected, f"raw-model feasibility {key} drifted")
    _require(
        feasibility.get("selection")
        == "first_512_by_sha256_of_qualification_id_and_canonical_problem_hash_after_collision_quarantine",
        "raw-model feasibility selection drifted",
    )
    _require(feasibility.get("teacher_model") == "Qwen/Qwen3-8B", "teacher model drifted")
    _require(feasibility.get("student_model") == "Qwen/Qwen3-1.7B", "student model drifted")
    _require(len(feasibility.get("teacher_revision", "")) == 40, "teacher revision must be immutable")
    _require(len(feasibility.get("student_revision", "")) == 40, "student revision must be immutable")

    freeze = payload["role_freeze"]
    _require(freeze.get("required_before_teacher_training") is True, "roles must freeze before teacher training")
    _require(freeze.get("failed_teacher_gap_is_terminal") is True, "C teacher gap failure must be terminal")
    successor = payload["successor_boundary"]
    _require(tuple(successor.get("teacher_student_matrix", ())) == EXPECTED_SUCCESSOR_MATRIX, "successor matrix drifted")
    _require(successor.get("MATH_role") == "external_transfer_target_only", "MATH role drifted")
    _require(successor.get("M_teacher_allowed") is False, "M teacher cannot be reintroduced")
    _require(successor.get("M_teacher_arms_allowed") == [], "M teacher arms cannot be reintroduced")
    return dict(payload)


def load_plan(path: str | Path = PLAN_PATH) -> dict[str, Any]:
    plan_path = Path(path).resolve()
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "DeepMath plan must contain a JSON object")
    plan = validate_plan(payload)
    plan["path"] = str(plan_path)
    plan["sha256"] = sha256_file(plan_path)
    plan["canonical_sha256"] = canonical_json_sha256(payload)
    return plan


def verify_raw_files(plan: Mapping[str, Any], data_dir: str | Path) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("verify-raw requires pyarrow") from exc

    root = Path(data_dir).resolve()
    _require(root.is_dir() and not root.is_symlink(), "raw data directory is missing or a symlink")
    total_rows = 0
    observed = []
    for expected in plan["candidate"]["raw_shards"]:
        path = root / expected["path"]
        _require(path.is_file() and not path.is_symlink(), f"raw shard missing or symlinked: {path}")
        size = path.stat().st_size
        digest = sha256_file(path)
        _require(size == expected["bytes"], f"raw shard byte count drifted: {path}")
        _require(digest == expected["sha256"], f"raw shard SHA-256 drifted: {path}")
        parquet = pq.ParquetFile(path)
        columns = tuple(parquet.schema_arrow.names)
        _require(columns == EXPECTED_COLUMNS, f"raw shard schema drifted: {path}")
        total_rows += parquet.metadata.num_rows
        observed.append(
            {
                "path": str(path),
                "bytes": size,
                "sha256": digest,
                "rows": parquet.metadata.num_rows,
                "columns": list(columns),
            }
        )
    _require(total_rows == plan["candidate"]["expected_rows"], "DeepMath total row count drifted")
    return {
        "schema_version": 1,
        "qualification_id": plan["qualification_id"],
        "stage": "raw_file_identity_and_schema_only",
        "status": "passed",
        "candidate_source": "C",
        "plan_sha256": plan["sha256"],
        "plan_canonical_sha256": plan["canonical_sha256"],
        "rows": total_rows,
        "shards": observed,
        "forbidden_training_fields": sorted(FORBIDDEN_TRAINING_FIELDS),
        "teacher_training_authorized": False,
        "scientific_use_allowed": False,
        "remaining_gates": [
            "full candidate label and parseability audit",
            "complete global exact and semantic collision graph",
            "zero prompt truncation audit",
            "raw Qwen3-8B and Qwen3-1.7B feasibility surface",
            "deterministic role freeze",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate-plan")
    verify = subparsers.add_parser("verify-raw")
    verify.add_argument("--data-dir", type=Path, required=True)
    verify.add_argument("--output", type=Path)
    args = parser.parse_args()
    plan = load_plan(args.plan)
    if args.command == "validate-plan":
        result = {
            "qualification_id": plan["qualification_id"],
            "status": plan["status"],
            "plan_sha256": plan["sha256"],
            "plan_canonical_sha256": plan["canonical_sha256"],
            "teacher_training_authorized": False,
        }
    else:
        result = verify_raw_files(plan, args.data_dir)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if getattr(args, "output", None) is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {output}")
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
