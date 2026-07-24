from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.opd import audit_positive_control_normalized_data as audit_data
from scripts.opd import materialize_positive_control as materialize
from scripts.opd import normalize_positive_control_data as normalize_data
from scripts.opd import normalize_positive_control_trainer_data as normalize_trainer_data
from scripts.opd import positive_control_gate as gate
from scripts.opd import positive_control_one_step as one_step
from scripts.opd import positive_control_one_step_terminal_audit as terminal_audit
from scripts.opd import prepare_opsd_execution_tree as execution_tree
from scripts.opd import seal_positive_control_one_step_cache_failure as cache_failure
from scripts.opd import verify_positive_control_environment as verify_environment


def test_identifiability_config_is_setup_only() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = json.loads(
        (root / "configs/opd_math/identifiability_v1.json").read_text()
    )
    assert payload["status"] == "setup_only_not_training_authorized"
    assert payload["immutable_boundaries"]["terminal_v2_is_not_reopened"]
    assert payload["immutable_boundaries"][
        "failed_M_teacher_is_never_retrained_merged_or_used"
    ]
    assert payload["positive_control"]["recipe"]["explicit_max_steps"] == 100
    assert payload["successor_cross_scale_pilot"]["status"] == (
        "blocked_on_positive_control"
    )
    assert payload["positive_control"]["stage_release_order"][3] == (
        "base_aime24_reproduction"
    )
    assert payload["positive_control"]["environment"]["vllm"] == "0.11.0"
    assert payload["positive_control"]["execution_hardware"][
        "base_and_all_checkpoint_evaluations_must_use_the_same_gpu_type"
    ]


def test_file_records_are_sorted_and_hashed(tmp_path: Path) -> None:
    (tmp_path / "b.txt").write_text("b")
    (tmp_path / "a.txt").write_text("a")
    records = materialize.file_records(tmp_path)
    assert [row["path"] for row in records] == ["a.txt", "b.txt"]
    assert records[0]["sha256"] == hashlib.sha256(b"a").hexdigest()


def test_pinned_positive_control_sources() -> None:
    assert materialize.TRAIN_REVISION == (
        "1f33e9dc2e8a1c639ca74f8024ad4a9f1f5eae62"
    )
    assert materialize.AIME_REVISION == (
        "2fe88a2f1091d5048c0f36abc874fb997b3dd99a"
    )
    assert materialize.TRAIN_ROWS == 29_434
    assert materialize.AIME_ROWS == 30


def test_gate_reconstructs_all_generations() -> None:
    payload = {
        "results": [
            {
                "generations": [
                    {"correct": generation < 6, "formatted": True}
                    for generation in range(12)
                ]
            }
            for _ in range(30)
        ]
    }
    rebuilt = gate.reconstruct(payload)
    assert rebuilt["generations"] == 360
    assert rebuilt["correct"] == 180
    assert rebuilt["average_at_12_fraction"] == 0.5


def test_execution_tree_edits_are_data_locality_only(tmp_path: Path) -> None:
    train = tmp_path / "opsd_train.py"
    train.write_text(execution_tree.TRAIN_OLD)
    record = execution_tree.replace_once(train, execution_tree.TRAIN_OLD, execution_tree.TRAIN_NEW)
    assert record["before_sha256"] != record["after_sha256"]
    assert "LEGALRAG_OPSD_TRAIN_PARQUET" in train.read_text()
    assert "__import__" in execution_tree.EVAL_NEW


def test_four_gpu_jobs_require_one_node() -> None:
    root = Path(__file__).resolve().parents[1]
    for name in (
        "slurm_opd_positive_control_preflight.sh",
        "slurm_opd_positive_control_base_eval.sh",
    ):
        source = (root / "scripts/hpc" / name).read_text()
        assert "#SBATCH --nodes=1" in source
        assert "#SBATCH --ntasks=1" in source
        assert "#SBATCH --gpus=a100-sxm4:4" in source


def test_resumed_submission_can_select_only_preregistered_gpu_types() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "scripts/hpc/submit_opd_positive_control_preflight.sh"
    ).read_text()
    assert "a100-sxm4|a6000" in source
    assert '--gpus="${GPU_TYPE}:4"' in source


def test_environment_receipt_is_exclusive_json(tmp_path: Path) -> None:
    output = tmp_path / "environment.json"
    verify_environment.write_exclusive(output, {"status": "passed"})
    assert json.loads(output.read_text()) == {"status": "passed"}
    assert output.stat().st_mode & 0o777 == 0o444


def test_one_step_preregistration_keeps_full_training_blocked() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = json.loads(
        (root / "configs/opd_math/identifiability_v1_one_step.json").read_text()
    )
    assert payload["status"] == (
        "preregistered_diagnostic_only_100_step_training_blocked"
    )
    assert payload["recipe"]["max_steps"] == 1
    assert payload["recipe"]["seed"] == 42
    assert payload["hardware"] == {
        "partition": "general-gpu",
        "gpu_type": "a6000",
        "gpu_count": 4,
        "single_node_required": True,
        "purpose": "one-step training memory and topology gate",
    }
    assert payload["pass_gate"]["terminal_log_hash_required"]
    assert payload["immutable_boundaries"][
        "one_hundred_step_training_is_not_automatically_authorized"
    ]


def test_one_step_retry_is_bound_to_the_audited_normalized_data() -> None:
    root = Path(__file__).resolve().parents[1]
    original = json.loads(
        (root / "configs/opd_math/identifiability_v1_one_step.json").read_text()
    )
    payload = json.loads(
        (
            root
            / "configs/opd_math/identifiability_v1_one_step_retry1.json"
        ).read_text()
    )
    assert payload["status"] == (
        "preregistered_diagnostic_only_100_step_training_blocked"
    )
    assert payload["retry"] == {
        "attempt_id": "normalized_data_retry_1",
        "predecessor_job_id": "132150",
        "allowed_change": "training_parquet_serialization_only",
        "model_recipe_and_ordered_rows_unchanged": True,
    }
    assert payload["training_data"]["rows"] == 29_434
    assert payload["training_data"]["required_columns"] == [
        "problem",
        "solution",
    ]
    assert payload["recipe"]["max_steps"] == 1
    for key in ("upstream", "hardware", "recipe", "pass_gate"):
        assert payload[key] == original[key]
    assert payload["immutable_boundaries"][
        "one_hundred_step_training_is_not_automatically_authorized"
    ]

    retry2 = json.loads(
        (
            root
            / "configs/opd_math/identifiability_v1_one_step_retry2.json"
        ).read_text()
    )
    assert retry2["retry"]["attempt_id"] == "trainer_data_retry_2"
    assert retry2["retry"]["predecessor_job_ids"] == ["132150", "135003"]
    assert retry2["training_data"]["required_columns"] == [
        "problem",
        "solution",
        "conversations",
    ]
    assert retry2["training_data"]["rows"] == 29_434
    assert retry2["training_data"]["pretruncate_sequences_over_20000"] == 0
    for key in ("upstream", "hardware", "recipe", "pass_gate"):
        assert retry2[key] == original[key]


def test_one_step_command_is_exactly_one_step(tmp_path: Path) -> None:
    command = one_step.training_command(
        tmp_path / "env",
        tmp_path / "execution",
        tmp_path / "model",
        tmp_path / "output",
        12345,
    )
    assert command[command.index("--max_steps") + 1] == "1"
    assert command[command.index("--save_steps") + 1] == "1"
    assert command[command.index("--logging_steps") + 1] == "1"
    assert command[command.index("--seed") + 1] == "42"
    assert "--fixed_teacher" in command
    assert "--use_peft" in command
    assert "--use_tinker_loss" not in command


def test_retry_prerequisites_require_sealed_failure_and_normalized_audit(
    tmp_path: Path, monkeypatch
) -> None:
    def record(name: str, payload) -> tuple[str, str]:
        path = tmp_path / name
        if isinstance(payload, str):
            path.write_text(payload)
        else:
            path.write_text(json.dumps(payload))
        return str(path), hashlib.sha256(path.read_bytes()).hexdigest()

    base_eval = record("base.json", {})
    base_gate = record(
        "base_gate.json",
        {
            "status": "passed",
            "decision": "BASELINE_REPRODUCED",
            "repository_commit": "base-commit",
            "independent_reconstruction": {"correct": 193, "generations": 360},
        },
    )
    preflight = record("preflight.json", {"status": "passed"})
    preflight_audit = record("preflight_audit.json", {"status": "passed"})
    data_manifest = record("data_manifest.json", {})
    environment = record("environment.txt", "datasets==3.6.0\n")
    failure = record(
        "failure.json",
        {
            "status": "failed_before_training",
            "decision": "PARQUET_FEATURE_METADATA_INCOMPATIBLE",
            "slurm": {"job_id": "132150"},
            "optimizer_steps": 0,
            "checkpoint_created": False,
            "opd_result_created": False,
        },
    )
    normalized = record(
        "normalized.json",
        {
            "artifact_type": "opd_positive_control_normalized_data",
            "rows": 29_434,
            "row_sequence_sha256": "row-digest",
            "normalized_shards": [{"sha256": "shard-digest"}],
        },
    )
    normalized_audit = record(
        "normalized_audit.json",
        {
            "status": "passed",
            "decision": "NORMALIZED_DATA_LOAD_COMPATIBLE",
            "datasets_version": "3.6.0",
            "rows": 29_434,
            "row_sequence_sha256": "row-digest",
            "columns": ["problem", "solution"],
            "manifest_sha256": normalized[1],
            "normalized_shard_sha256": ["shard-digest"],
            "normalized_root": str(tmp_path / "normalized"),
        },
    )
    normalized_root = tmp_path / "normalized"
    normalized_root.mkdir()
    normalized_shard = normalized_root / "train.parquet"
    normalized_shard.write_bytes(b"normalized shard")
    shard_digest = hashlib.sha256(normalized_shard.read_bytes()).hexdigest()
    normalized_payload = json.loads(Path(normalized[0]).read_text())
    normalized_payload["normalized_shards"][0]["sha256"] = shard_digest
    Path(normalized[0]).write_text(json.dumps(normalized_payload))
    normalized = (normalized[0], hashlib.sha256(Path(normalized[0]).read_bytes()).hexdigest())
    normalized_audit_payload = json.loads(Path(normalized_audit[0]).read_text())
    normalized_audit_payload["manifest_sha256"] = normalized[1]
    normalized_audit_payload["normalized_shard_sha256"] = [shard_digest]
    Path(normalized_audit[0]).write_text(json.dumps(normalized_audit_payload))
    normalized_audit = (
        normalized_audit[0],
        hashlib.sha256(Path(normalized_audit[0]).read_bytes()).hexdigest(),
    )
    prereq = {
        "base_evaluation_repository_commit": "base-commit",
        "base_gate_decision": "BASELINE_REPRODUCED",
        "base_correct": 193,
        "base_generations": 360,
    }
    for key, value in {
        "base_evaluation_json": base_eval,
        "base_gate": base_gate,
        "preflight_receipt": preflight,
        "preflight_independent_audit": preflight_audit,
        "data_manifest": data_manifest,
        "environment_freeze": environment,
        "terminal_failure": failure,
        "normalized_data_manifest": normalized,
        "normalized_data_audit": normalized_audit,
    }.items():
        prereq[key], prereq[f"{key}_sha256"] = value
    parquet_glob = str(normalized_root / "*.parquet")
    monkeypatch.setenv("LEGALRAG_OPSD_TRAIN_PARQUET", parquet_glob)
    config = {
        "status": "preregistered_diagnostic_only_100_step_training_blocked",
        "stage_id": "one_step_real_model_update_diagnostic",
        "immutable_boundaries": {"closed": True},
        "retry": {
            "attempt_id": "normalized_data_retry_1",
            "predecessor_job_id": "132150",
            "allowed_change": "training_parquet_serialization_only",
            "model_recipe_and_ordered_rows_unchanged": True,
        },
        "prerequisites": prereq,
        "training_data": {
            "parquet_glob": parquet_glob,
            "rows": 29_434,
            "required_columns": ["problem", "solution"],
            "row_sequence_sha256": "row-digest",
            "normalized_shard_sha256": [shard_digest],
        },
    }
    custody = one_step.validate_prerequisites(config, "launch-commit")
    assert custody["repository_commit"] == "launch-commit"
    assert "normalized_data_audit" in custody["prerequisite_files"]

    trainer_failure = record(
        "trainer_failure.json",
        {
            "status": "failed_before_optimization",
            "decision": "TRAINER_CHATML_SOURCE_FIELD_MISSING",
            "slurm": {"job_id": "135003"},
            "optimizer_steps": 0,
            "checkpoint_created": False,
            "opd_result_created": False,
        },
    )
    trainer_manifest = record(
        "trainer_manifest.json",
        {
            "artifact_type": "opd_positive_control_trainer_data",
            "rows": 29_434,
            "trainer_field_sequence_sha256": "trainer-row-digest",
            "trainer_shards": [{"sha256": shard_digest}],
        },
    )
    trainer_audit = record(
        "trainer_audit.json",
        {
            "status": "passed",
            "decision": "PINNED_TRL026_TRAINER_DATA_COMPATIBLE",
            "datasets_version": "3.6.0",
            "transformers_version": "4.57.1",
            "trl_version": "0.26.0",
            "upstream_commit": "upstream-commit",
            "rows": 29_434,
            "trainer_field_sequence_sha256": "trainer-row-digest",
            "columns": ["problem", "solution", "conversations"],
            "token_sequence_sha256": "token-digest",
            "tokenized_sequences": 29_434,
            "collator_batch_size": 4,
            "manifest_sha256": trainer_manifest[1],
            "trainer_shard_sha256": [shard_digest],
            "trainer_root": str(normalized_root),
        },
    )
    retry2_prereq = {
        key: value
        for key, value in prereq.items()
        if not key.startswith("terminal_failure")
        and not key.startswith("normalized_data_")
    }
    for key, value in {
        "metadata_failure": failure,
        "trainer_schema_failure": trainer_failure,
        "trainer_data_manifest": trainer_manifest,
        "trainer_data_audit": trainer_audit,
    }.items():
        retry2_prereq[key], retry2_prereq[f"{key}_sha256"] = value
    retry2 = {
        "status": "preregistered_diagnostic_only_100_step_training_blocked",
        "stage_id": "one_step_real_model_update_diagnostic",
        "immutable_boundaries": {"closed": True},
        "retry": {
            "attempt_id": "trainer_data_retry_2",
            "allowed_change": "restore_pinned_trl_conversations_field_only",
            "model_recipe_and_ordered_rows_unchanged": True,
        },
        "upstream": {"repository_commit": "upstream-commit"},
        "prerequisites": retry2_prereq,
        "training_data": {
            "parquet_glob": parquet_glob,
            "rows": 29_434,
            "required_columns": ["problem", "solution", "conversations"],
            "trainer_field_sequence_sha256": "trainer-row-digest",
            "trainer_shard_sha256": [shard_digest],
            "token_sequence_sha256": "token-digest",
        },
    }
    retry2_custody = one_step.validate_prerequisites(retry2, "retry2-commit")
    assert retry2_custody["repository_commit"] == "retry2-commit"
    assert "trainer_data_audit" in retry2_custody["prerequisite_files"]


def test_one_step_audit_proves_nonzero_lora_b_update(tmp_path: Path) -> None:
    import torch
    from safetensors.torch import save_file

    checkpoint = (
        tmp_path
        / "training"
        / one_step.RUN_CONFIG
        / "checkpoint-1"
    )
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text(
        json.dumps(
            {
                "global_step": 1,
                "max_steps": 1,
                "log_history": [
                    {"step": 1, "loss": 0.125, "grad_norm": 0.25}
                ],
            }
        )
    )
    save_file(
        {
            "layer.lora_A.weight": torch.ones(2, 2),
            "layer.lora_B.weight": torch.tensor([[0.0, 0.1], [0.0, 0.0]]),
        },
        str(checkpoint / "adapter_model.safetensors"),
    )
    audit = one_step.audit_training(
        tmp_path,
        {"pass_gate": {"global_step": 1}},
    )
    assert audit["status"] == "passed"
    assert audit["nonzero_lora_B_tensor_count"] == 1
    assert audit["nonzero_lora_B_parameter_count"] == 1


def test_one_step_job_is_single_node_four_a6000s() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "scripts/hpc/slurm_opd_positive_control_one_step.sh"
    ).read_text()
    assert "#SBATCH --nodes=1" in source
    assert "#SBATCH --ntasks=1" in source
    assert "#SBATCH --gpus=a6000:4" in source
    assert "identifiability_v1_one_step_retry2.json" in source
    assert "OPD_IDENT_ONE_STEP_CONFIG" in source
    assert 'export LEGALRAG_OPSD_TRAIN_PARQUET' in source


def test_terminal_accounting_parser_requires_exact_job(monkeypatch) -> None:
    monkeypatch.setattr(
        terminal_audit.subprocess,
        "check_output",
        lambda *args, **kwargs: (
            "131000|COMPLETED|0:0\n"
            "131000.batch|COMPLETED|0:0\n"
            "131000.extern|COMPLETED|0:0\n"
        ),
    )
    assert terminal_audit.exact_job_accounting("131000") == {
        "job_id": "131000",
        "state": "COMPLETED",
        "exit_code": "0:0",
    }


def test_cache_quota_failure_sealer_requires_exact_preoptimization_evidence(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "job_135015"
    run_dir.mkdir()
    (run_dir / "custody_start.json").write_text(
        json.dumps({"slurm_job_id": "135015"})
    )
    (run_dir / "training_exit.json").write_text(
        json.dumps({"slurm_job_id": "135015", "returncode": 1})
    )
    log = tmp_path / "opsd_pc_1step_135015.out"
    log.write_text("\n".join(cache_failure.SIGNATURES))

    receipt = cache_failure.build_receipt(
        job_id="135015",
        run_dir=run_dir,
        slurm_log=log,
        auditor_commit="audit-commit",
        accounting={"job_id": "135015", "state": "FAILED", "exit_code": "1:0"},
    )
    assert receipt["status"] == "failed_before_optimization"
    assert receipt["decision"] == "RUNTIME_COMPILE_CACHE_HOME_QUOTA_EXCEEDED"
    assert receipt["optimizer_steps"] == 0
    assert receipt["checkpoint_created"] is False
    assert receipt["opd_result_created"] is False

    log.write_text("Disk quota exceeded\n")
    try:
        cache_failure.build_receipt(
            job_id="135015",
            run_dir=run_dir,
            slurm_log=log,
            auditor_commit="audit-commit",
            accounting={"job_id": "135015", "state": "FAILED", "exit_code": "1:0"},
        )
    except RuntimeError as error:
        assert "registered signatures" in str(error)
    else:
        raise AssertionError("cache failure was sealed without its registered paths")


def test_one_step_submission_has_no_dependency() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "scripts/hpc/submit_opd_positive_control_one_step.sh"
    ).read_text()
    assert "--dependency" not in source
    assert '"dependent_jobs": []' in source
    assert '"full_training_queued": False' in source
    assert "identifiability_v1_one_step_retry2.json" in source
    assert "OPD_IDENT_ONE_STEP_CONFIG=$CONFIG" in source


def test_normalization_preserves_ordered_required_rows(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    source = tmp_path / "source"
    data = source / "data"
    data.mkdir(parents=True)
    rows = normalize_data.EXPECTED_ROWS
    table = pa.table(
        {
            "problem": [f"p{index}" for index in range(rows)],
            "solution": [f"s{index}" for index in range(rows)],
            "messages": [[{"role": "user", "content": "x"}]] * rows,
            "conversations": [
                [
                    {"from": "user", "value": f"p{index}"},
                    {"from": "assistant", "value": f"s{index}"},
                ]
                for index in range(rows)
            ],
        }
    )
    metadata = {
        b"huggingface": json.dumps(
            {
                "info": {
                    "features": {
                        "messages": {"_type": "List", "feature": {}}
                    }
                }
            }
        ).encode()
    }
    table = table.replace_schema_metadata(metadata)
    midpoint = rows // 2
    pq.write_table(table.slice(0, midpoint), data / "part0.parquet")
    pq.write_table(table.slice(midpoint), data / "part1.parquet")
    output = tmp_path / "normalized"
    manifest = normalize_data.normalize(source, output, "a" * 40)
    assert manifest["rows"] == normalize_data.EXPECTED_ROWS
    assert len(manifest["normalized_shards"]) == 2
    for path in sorted(output.glob("*.parquet")):
        schema = pq.read_schema(path)
        assert schema.names == ["problem", "solution"]
        assert schema.metadata is None
    receipt = audit_data.audit(
        output,
        "a" * 40,
        "b" * 40,
        tmp_path / "audit.json",
    )
    assert receipt["decision"] == "NORMALIZED_DATA_LOAD_COMPATIBLE"
    assert receipt["row_sequence_sha256"] == manifest["row_sequence_sha256"]
    for path in output.iterdir():
        path.chmod(0o644)
    output.chmod(0o755)

    trainer_output = tmp_path / "trainer_normalized"
    trainer_manifest = normalize_trainer_data.normalize(
        source, trainer_output, "c" * 40
    )
    assert trainer_manifest["rows"] == normalize_trainer_data.EXPECTED_ROWS
    assert trainer_manifest["required_columns"] == [
        "problem",
        "solution",
        "conversations",
    ]
    for path in sorted(trainer_output.glob("*.parquet")):
        schema = pq.read_schema(path)
        assert schema.names == ["problem", "solution", "conversations"]
        assert schema.metadata is None
    for path in trainer_output.iterdir():
        path.chmod(0o644)
    trainer_output.chmod(0o755)


def test_normalization_ladder_never_queues_training() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "scripts/hpc/submit_opd_positive_control_normalization.sh"
    ).read_text()
    assert "slurm_opd_positive_control_normalize.sh" in source
    assert "slurm_opd_positive_control_normalized_audit.sh" in source
    assert '"training_queued": False' in source
    assert "one_step" not in source

    trainer_source = (
        root / "scripts/hpc/submit_opd_positive_control_trainer_data.sh"
    ).read_text()
    assert "slurm_opd_positive_control_trainer_data.sh" in trainer_source
    assert "slurm_opd_positive_control_trainer_data_audit.sh" in trainer_source
    assert '"training_queued": False' in trainer_source
    assert "one_step" not in trainer_source
