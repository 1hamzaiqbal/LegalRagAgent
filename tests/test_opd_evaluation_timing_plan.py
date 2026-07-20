import json
from argparse import Namespace
from decimal import Decimal
from pathlib import Path

import pytest

from scripts.opd_math import evaluate_math as evaluation
from scripts.opd_math import merge_evaluations as merger
from scripts.opd_math import plan_evaluation_shards as planner
from tests.opd_evaluation_fixture import write_merged_evaluation


@pytest.fixture(autouse=True)
def stable_exact_timing_environment(monkeypatch):
    monkeypatch.setattr(
        planner, "evaluation_environment_contract_unchanged", lambda contract: True
    )


def _write_task(path: Path, records: int) -> Path:
    rows = [
        {
            "record_id": f"O:train:timing:{index}",
            "cluster_id": f"cluster-{index}",
            "source": "O",
            "role": "teacher_gap_dev",
            "prompt": [{"role": "user", "content": f"Return {index}."}],
            "solution": rf"\boxed{{{index}}}",
        }
        for index in range(records)
    ]
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path.resolve()


def _one_timing_arm(
    tmp_path: Path,
    *,
    task: Path,
    arm: str,
    timing_records: int,
    elapsed: int,
    job_id: int,
    adapter: Path | None,
) -> dict:
    rewards = {
        f"O:train:timing:{index}": [1.0, 0.0, 1.0, 0.0]
        for index in range(timing_records)
    }
    merged_summary, _ = write_merged_evaluation(
        tmp_path,
        f"O-{arm}-timing",
        task,
        rewards,
        model=planner.EXPECTED_MODEL,
        revision=planner.EXPECTED_REVISION,
        adapter=adapter,
        packages=dict(evaluation.EXPECTED_EVALUATION_PACKAGES),
        decoding=dict(planner.EXPECTED_DECODING),
        exact_environment=True,
    )
    shard_dir = merged_summary.parent.parent / "shards" / "shard_00000"
    summary_path = shard_dir / "summary.json"
    companion_path = evaluation.post_promotion_custody_path(shard_dir)
    summary = json.loads(summary_path.read_text())
    stdout = tmp_path / f"opd_math_eval_{job_id}.out"
    stdout.write_text(
        json.dumps(summary, sort_keys=True)
        + "\n"
        + f"PASS evaluation artifact only; no gate inferred: {shard_dir.resolve()}\n",
        encoding="utf-8",
    )
    sacct_raw = tmp_path / f"{arm}-sacct.raw"
    sacct_raw.write_text(
        "|".join(
            (
                str(job_id),
                planner.SACCT_JOB_NAME,
                "COMPLETED",
                "0:0",
                str(elapsed),
                "billing=8,cpu=8,gres/gpu:a100-sxm4=1,gres/gpu=1,mem=96G,node=1",
                str(stdout.resolve()).replace(str(job_id), "%j"),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "summary": summary_path.resolve(),
        "companion": companion_path.resolve(),
        "stdout": stdout.resolve(),
        "sacct_raw": sacct_raw.resolve(),
        "git": summary["code"]["git"],
    }


def _timing_evidence(
    tmp_path: Path,
    *,
    total_records: int = 10,
    timing_records: int = 2,
    base_elapsed: int = 100,
    trained_elapsed: int = 120,
    include_trained: bool = True,
) -> dict:
    task = _write_task(tmp_path / "O_teacher_gap_dev.jsonl", total_records)
    adapter = (tmp_path / "O-teacher-adapter").resolve()
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"r": 8}\n')
    (adapter / "adapter_model.safetensors").write_bytes(b"trained adapter")
    base = _one_timing_arm(
        tmp_path,
        task=task,
        arm="base",
        timing_records=timing_records,
        elapsed=base_elapsed,
        job_id=12345,
        adapter=None,
    )
    trained = None
    if include_trained:
        trained = _one_timing_arm(
            tmp_path,
            task=task,
            arm="trained",
            timing_records=timing_records,
            elapsed=trained_elapsed,
            job_id=12346,
            adapter=adapter,
        )
    return {
        "task": task,
        "adapter": adapter,
        "base": base,
        "trained": trained,
        "git": base["git"],
    }


def _plan_args(evidence: dict, *, include_trained: bool = True) -> dict:
    base = evidence["base"]
    args = {
        "base_timing_summary": base["summary"],
        "base_timing_companion": base["companion"],
        "base_sacct_raw": base["sacct_raw"],
        "base_stdout": base["stdout"],
        "task_file": evidence["task"],
    }
    if include_trained:
        trained = evidence["trained"]
        assert trained is not None
        args.update(
            {
                "trained_timing_summary": trained["summary"],
                "trained_timing_companion": trained["companion"],
                "trained_sacct_raw": trained["sacct_raw"],
                "trained_stdout": trained["stdout"],
            }
        )
    return args


def _launch_settings(spec: dict) -> dict:
    return {
        "array_spec": spec["array_spec"],
        "samples_per_problem": planner.EXPECTED_SAMPLES_PER_PROBLEM,
        "temperature": planner.EXPECTED_DECODING["temperature"],
        "top_p": planner.EXPECTED_DECODING["top_p"],
        "top_k": planner.EXPECTED_DECODING["top_k"],
        "max_new_tokens": planner.EXPECTED_DECODING["max_new_tokens"],
        "seed": planner.EXPECTED_DECODING["seed"],
    }


def test_exact_timing_plan_chooses_smallest_shards_and_binds_arrays(
    tmp_path, monkeypatch
):
    evidence = _timing_evidence(tmp_path)
    monkeypatch.setattr(planner, "git_identity", lambda: dict(evidence["git"]))

    plan = planner.build_plan(
        **_plan_args(evidence),
        total_records=10,
        safety_factor=Decimal("1.25"),
        max_shard_seconds=500,
        max_concurrent=4,
        primary_contract=False,
    )

    # Trained ElapsedRaw=120 exceeds base=100 and therefore controls. S=1
    # projects 5 * 120 * 1.25 = 750 seconds and fails. S=2 is the exact
    # smallest passing count: 3 * 120 * 1.25 = 450.
    array_plan = plan["array_plan"]
    assert array_plan["selected_shard_count"] == 2
    assert array_plan["selected_throttle"] == 2
    assert array_plan["smallest_passing_shard_count"] is True
    assert array_plan["immediately_previous_candidate"] == {
        "shard_count": 1,
        "records_per_shard_ceiling": 10,
        "timing_blocks_per_shard": 5,
        "projected_shard_seconds": 750,
        "exceeds_limit": True,
    }
    assert array_plan["base"] == array_plan["trained"]
    assert array_plan["base_trained_specs_identical"] is True
    assert array_plan["base"]["array_spec"] == "0-1%2"
    assert array_plan["base"]["concurrency_waves"] == 1
    assert array_plan["base"]["projected_shard_seconds"] == 450
    assert array_plan["base"]["projected_gpu_seconds"] == 900
    assert array_plan["projected_gpu_seconds_two_arms"] == 1800
    assert plan["timing"]["planning_elapsed_raw_seconds"] == 120
    assert plan["timing"]["selection_rule"].startswith("max(")
    assert plan["scientific_authorization"] is False
    payload_hash = plan.pop("plan_payload_sha256")
    assert payload_hash == evaluation.canonical_sha256(plan)


def test_plan_output_is_sorted_exclusive_json(tmp_path, monkeypatch):
    evidence = _timing_evidence(tmp_path)
    monkeypatch.setattr(planner, "git_identity", lambda: dict(evidence["git"]))
    plan = planner.build_plan(
        **_plan_args(evidence),
        total_records=10,
        max_shard_seconds=500,
        primary_contract=False,
    )
    output = tmp_path / "decision" / "O_shard_plan.json"

    written = planner.write_plan_exclusive(output, plan)

    assert written == output.resolve()
    assert written.read_text().endswith("\n")
    assert json.loads(written.read_text()) == plan
    assert written.read_text().index('  "array_plan"') < written.read_text().index(
        '  "code"'
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        planner.write_plan_exclusive(output, plan)


def test_plan_rejects_stdout_path_changed_after_sacct_capture(tmp_path, monkeypatch):
    evidence = _timing_evidence(tmp_path)
    monkeypatch.setattr(planner, "git_identity", lambda: dict(evidence["git"]))
    moved = tmp_path / "opd_math_eval_99999.out"
    moved.write_bytes(evidence["base"]["stdout"].read_bytes())

    with pytest.raises(ValueError, match="stdout path differs"):
        planner.build_plan(
            **{
                **_plan_args(evidence),
                "base_stdout": moved,
            },
            total_records=10,
            primary_contract=False,
        )


def test_plan_rejects_a_different_companion(tmp_path, monkeypatch):
    evidence = _timing_evidence(tmp_path)
    monkeypatch.setattr(planner, "git_identity", lambda: dict(evidence["git"]))

    with pytest.raises(ValueError, match="supplied base timing companion"):
        planner.build_plan(
            **{
                **_plan_args(evidence),
                "base_timing_companion": evidence["base"]["sacct_raw"],
            },
            total_records=10,
            primary_contract=False,
        )


def test_geometry_uses_throttle_ceiling_and_rejects_impossible_limit():
    geometry = planner.choose_shard_geometry(
        total_records=100,
        timing_records=10,
        elapsed_raw_seconds=100,
        safety_factor=Decimal("1.0"),
        max_shard_seconds=1000,
        max_concurrent=4,
    )
    assert geometry["selected_shard_count"] == 1
    assert geometry["selected_throttle"] == 1
    assert geometry["base"]["array_spec"] == "0-0%1"
    assert geometry["base"]["concurrency_waves"] == 1

    five_shards = planner.choose_shard_geometry(
        total_records=100,
        timing_records=10,
        elapsed_raw_seconds=100,
        safety_factor=Decimal("1.0"),
        max_shard_seconds=200,
        max_concurrent=4,
    )
    assert five_shards["selected_shard_count"] == 5
    assert five_shards["selected_throttle"] == 4
    assert five_shards["base"]["array_spec"] == "0-4%4"
    assert five_shards["base"]["concurrency_waves"] == 2
    assert five_shards["base"]["projected_gpu_seconds"] == 1000

    with pytest.raises(ValueError, match="no shard count"):
        planner.choose_shard_geometry(
            total_records=100,
            timing_records=10,
            elapsed_raw_seconds=101,
            safety_factor=Decimal("1.0"),
            max_shard_seconds=100,
            max_concurrent=4,
        )


def test_scientific_defaults_are_registered():
    assert planner.DEFAULT_TOTAL_RECORDS == 4585
    assert planner.PRIMARY_TIMING_RECORDS == 32
    assert planner.PRIMARY_MIN_SHARDS == 5
    assert planner.DEFAULT_MAX_SHARD_SECONDS == 64800
    assert planner.DEFAULT_MAX_CONCURRENT == 4
    assert planner.DEFAULT_SAFETY_FACTOR == Decimal("1.25")


def test_primary_contract_rejects_every_geometry_override():
    with pytest.raises(ValueError, match="canonical fixed constraints"):
        planner._require_primary_constraints(
            total_records=4585,
            timing_records=31,
            safety_factor=Decimal("1.25"),
            max_shard_seconds=64800,
            max_concurrent=4,
        )
    with pytest.raises(ValueError, match="canonical fixed constraints"):
        planner._require_primary_constraints(
            total_records=4585,
            timing_records=32,
            safety_factor=Decimal("1.25"),
            max_shard_seconds=64800,
            max_concurrent=4,
            minimum_shard_count=4,
        )
    with pytest.raises(ValueError, match="canonical fixed constraints"):
        planner._require_primary_constraints(
            total_records=4585,
            timing_records=32,
            safety_factor=Decimal("1.0"),
            max_shard_seconds=64800,
            max_concurrent=4,
        )


def test_primary_successor_cannot_be_built_from_base_only_timing(
    tmp_path, monkeypatch
):
    evidence = _timing_evidence(tmp_path, include_trained=False)
    monkeypatch.setattr(planner, "git_identity", lambda: dict(evidence["git"]))

    with pytest.raises(ValueError, match="requires both base and trained"):
        planner.build_plan(**_plan_args(evidence, include_trained=False))


def test_base_only_diagnostic_planning_remains_available(tmp_path, monkeypatch):
    evidence = _timing_evidence(tmp_path, include_trained=False)
    monkeypatch.setattr(planner, "git_identity", lambda: dict(evidence["git"]))

    plan = planner.build_plan(
        **_plan_args(evidence, include_trained=False),
        total_records=10,
        max_shard_seconds=500,
        primary_contract=False,
    )

    assert plan["plan_kind"] == planner.DIAGNOSTIC_PLAN_KIND
    assert plan["inputs"]["trained_timing"] is None
    assert plan["timing"]["selection_rule"] == "base_elapsed_raw_seconds_diagnostic_only"
    assert plan["array_plan"]["registered_minimum_shard_count"] == 1


def test_raw_sacct_job_identity_is_bound_to_stdout(tmp_path, monkeypatch):
    evidence = _timing_evidence(tmp_path)
    monkeypatch.setattr(planner, "git_identity", lambda: dict(evidence["git"]))
    evidence["base"]["sacct_raw"].write_text(
        evidence["base"]["sacct_raw"].read_text().replace(
            "12345|opd_math_eval", "99999|opd_math_eval"
        )
    )
    with pytest.raises(ValueError, match="filename does not bind|stdout path differs"):
        planner.build_plan(
            **_plan_args(evidence),
            total_records=10,
            primary_contract=False,
        )


@pytest.mark.parametrize(
    "alloc_tres",
    (
        "billing=8,gres/gpu:a100-sxm4=1,gres/gpu=2,node=1",
        "billing=8,gres/gpu=2,gres/gpu:a100-sxm4=1,node=1",
    ),
)
def test_raw_sacct_rejects_inconsistent_adjacent_gpu_counts(
    tmp_path, monkeypatch, alloc_tres
):
    evidence = _timing_evidence(tmp_path)
    monkeypatch.setattr(planner, "git_identity", lambda: dict(evidence["git"]))
    fields = evidence["base"]["sacct_raw"].read_text().strip().split("|")
    fields[5] = alloc_tres
    evidence["base"]["sacct_raw"].write_text("|".join(fields) + "\n")
    with pytest.raises(ValueError, match="consistent GPU allocation"):
        planner.build_plan(
            **_plan_args(evidence),
            total_records=10,
            primary_contract=False,
        )


def test_primary_plan_is_consumed_by_exact_base_launch(tmp_path, monkeypatch):
    evidence = _timing_evidence(
        tmp_path,
        total_records=planner.DEFAULT_TOTAL_RECORDS,
        timing_records=32,
        base_elapsed=1000,
        trained_elapsed=1200,
    )
    monkeypatch.setattr(planner, "git_identity", lambda: dict(evidence["git"]))
    plan = planner.build_plan(
        **_plan_args(evidence),
    )
    plan_path = planner.write_plan_exclusive(tmp_path / "primary-plan.json", plan)
    spec = plan["array_plan"]["base"]
    train_freeze = Path(
        plan["code"]["exact_environment_contract"]["train_freeze"]["path"]
    )

    assert plan["schema_version"] == 2
    assert plan["plan_kind"] == planner.PLAN_KIND
    assert plan["timing"]["planning_elapsed_raw_seconds"] == 1200
    assert plan["array_plan"]["smallest_runtime_passing_shard_count"] == 4
    assert plan["array_plan"]["registered_minimum_shard_count"] == 5
    assert plan["array_plan"]["selected_shard_count"] == 5
    assert plan["array_plan"]["smallest_passing_shard_count"] is False
    assert plan["array_plan"]["smallest_authorized_shard_count"] is True
    basis = plan["constraints"]["minimum_shard_count_basis"]
    assert basis["four_shards"]["projected_shard_seconds"] == 75000
    assert basis["four_shards"]["exceeds_primary_limit"] is True
    assert basis["five_shards"]["exceeds_primary_limit"] is False

    binding = planner.validate_launch_against_plan(
        plan_path=plan_path,
        arm="base",
        phase="shard",
        source="O",
        role="teacher_gap_dev",
        model=planner.EXPECTED_MODEL,
        model_revision=planner.EXPECTED_REVISION,
        task_file=evidence["task"],
        max_records=0,
        shard_count=spec["shard_count"],
        git_commit=evidence["git"]["commit"],
        train_freeze=train_freeze,
        adapter=None,
        **_launch_settings(spec),
        array_task_count=spec["shard_count"],
        array_task_min=0,
        array_task_max=spec["shard_index_stop"],
    )
    assert binding["plan_binding"]["plan_payload_sha256"] == plan["plan_payload_sha256"]
    assert binding["plan_binding"]["array_spec"] == spec["array_spec"]
    assert binding["launch_validation"]["phase"] == "shard"
    assert binding["launch_validation"]["declared_array_spec"] == spec["array_spec"]
    assert (
        binding["launch_validation"]["array_spec_source"]
        == "predeclared_OPD_MATH_EVAL_ARRAY_SPEC_v1"
    )

    trained_binding = planner.validate_launch_against_plan(
        plan_path=plan_path,
        arm="trained",
        phase="merge",
        source="O",
        role="teacher_gap_dev",
        model=planner.EXPECTED_MODEL,
        model_revision=planner.EXPECTED_REVISION,
        task_file=evidence["task"],
        max_records=0,
        shard_count=spec["shard_count"],
        git_commit=evidence["git"]["commit"],
        train_freeze=train_freeze,
        adapter=evidence["adapter"],
        **_launch_settings(spec),
    )
    assert trained_binding["plan_binding"]["adapter"] == str(evidence["adapter"])
    assert trained_binding["plan_binding"]["adapter_tree_sha256"] == plan["evaluation"][
        "trained_adapter"
    ]["tree_sha256"]

    with pytest.raises(ValueError, match="shard count differs"):
        planner.validate_launch_against_plan(
            plan_path=plan_path,
            arm="base",
            phase="merge",
            source="O",
            role="teacher_gap_dev",
            model=planner.EXPECTED_MODEL,
            model_revision=planner.EXPECTED_REVISION,
            task_file=evidence["task"],
            max_records=0,
            shard_count=spec["shard_count"] + 1,
            git_commit=evidence["git"]["commit"],
            train_freeze=train_freeze,
            adapter=None,
            **_launch_settings(spec),
        )


def test_validate_launch_rejects_trained_adapter_tree_tamper(tmp_path, monkeypatch):
    evidence = _timing_evidence(
        tmp_path,
        total_records=planner.DEFAULT_TOTAL_RECORDS,
        timing_records=planner.PRIMARY_TIMING_RECORDS,
        base_elapsed=1000,
        trained_elapsed=1200,
    )
    monkeypatch.setattr(planner, "git_identity", lambda: dict(evidence["git"]))
    plan = planner.build_plan(**_plan_args(evidence))
    plan_path = planner.write_plan_exclusive(tmp_path / "primary-plan.json", plan)
    train_freeze = Path(
        plan["code"]["exact_environment_contract"]["train_freeze"]["path"]
    )
    (evidence["adapter"] / "adapter_model.safetensors").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="adapter path/tree binding does not verify"):
        planner.validate_launch_against_plan(
            plan_path=plan_path,
            arm="trained",
            phase="merge",
            source="O",
            role="teacher_gap_dev",
            model=planner.EXPECTED_MODEL,
            model_revision=planner.EXPECTED_REVISION,
            task_file=evidence["task"],
            max_records=0,
            shard_count=plan["array_plan"]["trained"]["shard_count"],
            git_commit=evidence["git"]["commit"],
            train_freeze=train_freeze,
            adapter=evidence["adapter"],
            **_launch_settings(plan["array_plan"]["trained"]),
        )


def test_load_primary_plan_rehashes_both_timing_evidence_trees(
    tmp_path, monkeypatch
):
    evidence = _timing_evidence(
        tmp_path,
        total_records=planner.DEFAULT_TOTAL_RECORDS,
        timing_records=planner.PRIMARY_TIMING_RECORDS,
        base_elapsed=1000,
        trained_elapsed=1200,
    )
    monkeypatch.setattr(planner, "git_identity", lambda: dict(evidence["git"]))
    plan = planner.build_plan(**_plan_args(evidence))
    plan_path = planner.write_plan_exclusive(tmp_path / "primary-plan.json", plan)
    evidence["trained"]["stdout"].write_text(
        evidence["trained"]["stdout"].read_text() + "post-plan drift\n"
    )

    with pytest.raises(ValueError, match="trained timing stdout identity has drifted"):
        planner.load_primary_plan(plan_path)


def test_self_rehashed_plan_cannot_swap_base_for_trained_timing(tmp_path, monkeypatch):
    evidence = _timing_evidence(
        tmp_path,
        total_records=planner.DEFAULT_TOTAL_RECORDS,
        timing_records=planner.PRIMARY_TIMING_RECORDS,
        base_elapsed=1000,
        trained_elapsed=1200,
    )
    monkeypatch.setattr(planner, "git_identity", lambda: dict(evidence["git"]))
    plan = planner.build_plan(**_plan_args(evidence))
    forged = json.loads(json.dumps(plan))
    forged["inputs"]["base_timing"] = forged["inputs"]["trained_timing"]
    payload = dict(forged)
    payload.pop("plan_payload_sha256")
    forged["plan_payload_sha256"] = evaluation.canonical_sha256(payload)
    plan_path = tmp_path / "forged-self-hashed-plan.json"
    plan_path.write_text(json.dumps(forged, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="base timing shard must be an unadapted"):
        planner.load_primary_plan(plan_path)


def test_launch_plan_rejects_array_decoding_and_sample_mismatch(tmp_path, monkeypatch):
    evidence = _timing_evidence(
        tmp_path,
        total_records=planner.DEFAULT_TOTAL_RECORDS,
        timing_records=planner.PRIMARY_TIMING_RECORDS,
        base_elapsed=1000,
        trained_elapsed=1200,
    )
    monkeypatch.setattr(planner, "git_identity", lambda: dict(evidence["git"]))
    plan = planner.build_plan(**_plan_args(evidence))
    plan_path = planner.write_plan_exclusive(tmp_path / "primary-plan.json", plan)
    spec = plan["array_plan"]["base"]
    freeze = Path(plan["code"]["exact_environment_contract"]["train_freeze"]["path"])
    common = {
        "plan_path": plan_path,
        "arm": "base",
        "phase": "merge",
        "source": "O",
        "role": "teacher_gap_dev",
        "model": planner.EXPECTED_MODEL,
        "model_revision": planner.EXPECTED_REVISION,
        "task_file": evidence["task"],
        "max_records": 0,
        "shard_count": spec["shard_count"],
        "git_commit": evidence["git"]["commit"],
        "train_freeze": freeze,
        "adapter": None,
        **_launch_settings(spec),
    }
    valid = planner.validate_launch_against_plan(**common)
    assert planner.revalidate_plan_binding(valid["plan_binding"]) == valid["plan_binding"]
    task_rows = list(merger.iter_jsonl(evidence["task"]))
    environment_contract = plan["code"]["exact_environment_contract"]
    full_contract = evaluation.evaluation_contract(
        model=planner.EXPECTED_MODEL,
        model_revision=planner.EXPECTED_REVISION,
        adapter=None,
        adapter_tree_sha256=None,
        task_file=str(evidence["task"]),
        task_file_sha256=evaluation.sha256_file(evidence["task"]),
        eligible_record_ids=[row["record_id"] for row in task_rows],
        task_sources=["O"],
        task_roles=["teacher_gap_dev"],
        samples_per_problem=planner.EXPECTED_SAMPLES_PER_PROBLEM,
        decoding=dict(planner.EXPECTED_DECODING),
        shard_count=spec["shard_count"],
        tokenizer_contract_sha256="d" * 64,
        custody={
            "git": dict(evidence["git"]),
            "evaluator_file_sha256": evaluation.sha256_file(
                Path(evaluation.__file__)
            ),
            "packages": dict(evaluation.EXPECTED_EVALUATION_PACKAGES),
        },
        environment_contract=environment_contract,
        evaluation_plan=valid["plan_binding"],
    )
    assert planner.revalidate_plan_binding_for_contract(
        valid["plan_binding"], full_contract
    ) == valid["plan_binding"]
    mismatched_contract = json.loads(json.dumps(full_contract))
    mismatched_contract["samples_per_problem"] = 3
    with pytest.raises(ValueError, match="exact artifact contract"):
        planner.revalidate_plan_binding_for_contract(
            valid["plan_binding"], mismatched_contract
        )

    for override, message in (
        ({"array_spec": "0-4%5"}, "literal Slurm array"),
        ({"samples_per_problem": 3}, "sample count"),
        ({"temperature": 0.8}, "decoding differs"),
        ({"seed": False}, "noncanonical numeric types"),
    ):
        with pytest.raises(ValueError, match=message):
            planner.validate_launch_against_plan(**{**common, **override})


def test_complete_o_plan_detection_cannot_be_bypassed_by_positive_budget():
    rows = [
        {"source": "O", "role": "teacher_gap_dev", "record_id": f"O:{index}"}
        for index in range(3)
    ]
    args = Namespace(
        max_records=3,
        shard_plan=None,
        plan_arm=None,
        array_spec=None,
        array_task_count=None,
        array_task_min=None,
        array_task_max=None,
    )
    with pytest.raises(ValueError, match="requires the canonical v2 plan"):
        evaluation.validate_evaluation_plan(
            args,
            task_file=Path("/unused"),
            task_rows=rows,
            physical_record_count=3,
            adapter=None,
            git={"commit": "a" * 40},
        )

    args.shard_plan = Path("/forbidden-plan.json")
    with pytest.raises(ValueError, match="only be supplied"):
        evaluation.validate_evaluation_plan(
            args,
            task_file=Path("/unused"),
            task_rows=rows[:2],
            physical_record_count=3,
            adapter=None,
            git={"commit": "a" * 40},
        )


def test_direct_merge_contract_rejects_complete_o_without_plan(tmp_path):
    task = _write_task(tmp_path / "O.jsonl", 3)
    rows = list(merger.iter_jsonl(task))
    custody = {
        "git": {"commit": "a" * 40, "worktree_clean": True},
        "evaluator_file_sha256": "b" * 64,
        "packages": {"torch": "test"},
    }
    contract = evaluation.evaluation_contract(
        model=planner.EXPECTED_MODEL,
        model_revision=planner.EXPECTED_REVISION,
        adapter=None,
        adapter_tree_sha256=None,
        task_file=str(task),
        task_file_sha256=evaluation.sha256_file(task),
        eligible_record_ids=[row["record_id"] for row in rows],
        task_sources=["O"],
        task_roles=["teacher_gap_dev"],
        samples_per_problem=4,
        decoding=dict(planner.EXPECTED_DECODING),
        shard_count=1,
        tokenizer_contract_sha256="c" * 64,
        custody=custody,
    )
    with pytest.raises(ValueError, match="lacks the canonical v2 plan"):
        merger._validate_contract(
            contract,
            task_file=task,
            task_rows=rows,
            task_hash=evaluation.sha256_file(task),
        )
