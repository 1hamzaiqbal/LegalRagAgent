import json
from decimal import Decimal
from pathlib import Path

import pytest

from scripts.opd_math import evaluate_math as evaluation
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


def _timing_evidence(
    tmp_path: Path, *, total_records: int = 10, timing_records: int = 2, elapsed: int = 100
) -> dict:
    task = _write_task(tmp_path / "O_teacher_gap_dev.jsonl", total_records)
    rewards = {
        f"O:train:timing:{index}": [1.0, 0.0, 1.0, 0.0]
        for index in range(timing_records)
    }
    merged_summary, _ = write_merged_evaluation(
        tmp_path,
        "O-base-timing",
        task,
        rewards,
        model=planner.EXPECTED_MODEL,
        revision=planner.EXPECTED_REVISION,
        adapter=None,
        packages=dict(evaluation.EXPECTED_EVALUATION_PACKAGES),
        decoding=dict(planner.EXPECTED_DECODING),
        exact_environment=True,
    )
    shard_dir = merged_summary.parent.parent / "shards" / "shard_00000"
    summary_path = shard_dir / "summary.json"
    companion_path = evaluation.post_promotion_custody_path(shard_dir)
    summary = json.loads(summary_path.read_text())
    stdout = tmp_path / "opd_math_eval_12345.out"
    stdout.write_text(
        json.dumps(summary, sort_keys=True)
        + "\n"
        + f"PASS evaluation artifact only; no gate inferred: {shard_dir.resolve()}\n",
        encoding="utf-8",
    )
    sacct_raw = tmp_path / "sacct.raw"
    sacct_raw.write_text(
        "|".join(
            (
                "12345",
                planner.SACCT_JOB_NAME,
                "COMPLETED",
                "0:0",
                str(elapsed),
                "billing=8,cpu=8,gres/gpu:a100-sxm4=1,gres/gpu=1,mem=96G,node=1",
                str(stdout.resolve()).replace("12345", "%j"),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "task": task,
        "summary": summary_path.resolve(),
        "companion": companion_path.resolve(),
        "stdout": stdout.resolve(),
        "sacct_raw": sacct_raw.resolve(),
        "git": summary["code"]["git"],
    }


def test_exact_timing_plan_chooses_smallest_shards_and_binds_arrays(
    tmp_path, monkeypatch
):
    evidence = _timing_evidence(tmp_path)
    monkeypatch.setattr(planner, "git_identity", lambda: dict(evidence["git"]))

    plan = planner.build_plan(
        timing_summary=evidence["summary"],
        timing_companion=evidence["companion"],
        task_file=evidence["task"],
        sacct_raw=evidence["sacct_raw"],
        stdout=evidence["stdout"],
        total_records=10,
        safety_factor=Decimal("1.25"),
        max_shard_seconds=500,
        max_concurrent=4,
        primary_contract=False,
    )

    # S=1 projects 5 * 100 * 1.25 = 625 seconds and fails.  S=2 is
    # therefore the exact smallest passing count: 3 * 100 * 1.25 = 375.
    array_plan = plan["array_plan"]
    assert array_plan["selected_shard_count"] == 2
    assert array_plan["selected_throttle"] == 2
    assert array_plan["smallest_passing_shard_count"] is True
    assert array_plan["immediately_previous_candidate"] == {
        "shard_count": 1,
        "records_per_shard_ceiling": 10,
        "timing_blocks_per_shard": 5,
        "projected_shard_seconds": 625,
        "exceeds_limit": True,
    }
    assert array_plan["base"] == array_plan["trained"]
    assert array_plan["base_trained_specs_identical"] is True
    assert array_plan["base"]["array_spec"] == "0-1%2"
    assert array_plan["base"]["concurrency_waves"] == 1
    assert array_plan["base"]["projected_shard_seconds"] == 375
    assert array_plan["base"]["projected_gpu_seconds"] == 750
    assert array_plan["projected_gpu_seconds_two_arms"] == 1500
    assert plan["scientific_authorization"] is False
    payload_hash = plan.pop("plan_payload_sha256")
    assert payload_hash == evaluation.canonical_sha256(plan)


def test_plan_output_is_sorted_exclusive_json(tmp_path, monkeypatch):
    evidence = _timing_evidence(tmp_path)
    monkeypatch.setattr(planner, "git_identity", lambda: dict(evidence["git"]))
    plan = planner.build_plan(
        timing_summary=evidence["summary"],
        timing_companion=evidence["companion"],
        task_file=evidence["task"],
        sacct_raw=evidence["sacct_raw"],
        stdout=evidence["stdout"],
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
    moved.write_bytes(evidence["stdout"].read_bytes())

    with pytest.raises(ValueError, match="stdout path differs"):
        planner.build_plan(
            timing_summary=evidence["summary"],
            timing_companion=evidence["companion"],
            task_file=evidence["task"],
            sacct_raw=evidence["sacct_raw"],
            stdout=moved,
            total_records=10,
            primary_contract=False,
        )


def test_plan_rejects_a_different_companion(tmp_path, monkeypatch):
    evidence = _timing_evidence(tmp_path)
    monkeypatch.setattr(planner, "git_identity", lambda: dict(evidence["git"]))

    with pytest.raises(ValueError, match="supplied timing companion"):
        planner.build_plan(
            timing_summary=evidence["summary"],
            timing_companion=evidence["sacct_raw"],
            task_file=evidence["task"],
            sacct_raw=evidence["sacct_raw"],
            stdout=evidence["stdout"],
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
            safety_factor=Decimal("1.0"),
            max_shard_seconds=64800,
            max_concurrent=4,
        )


def test_raw_sacct_job_identity_is_bound_to_stdout(tmp_path, monkeypatch):
    evidence = _timing_evidence(tmp_path)
    monkeypatch.setattr(planner, "git_identity", lambda: dict(evidence["git"]))
    evidence["sacct_raw"].write_text(
        evidence["sacct_raw"].read_text().replace(
            "12345|opd_math_eval", "99999|opd_math_eval"
        )
    )
    with pytest.raises(ValueError, match="filename does not bind|stdout path differs"):
        planner.build_plan(
            timing_summary=evidence["summary"],
            timing_companion=evidence["companion"],
            task_file=evidence["task"],
            sacct_raw=evidence["sacct_raw"],
            stdout=evidence["stdout"],
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
    fields = evidence["sacct_raw"].read_text().strip().split("|")
    fields[5] = alloc_tres
    evidence["sacct_raw"].write_text("|".join(fields) + "\n")
    with pytest.raises(ValueError, match="consistent GPU allocation"):
        planner.build_plan(
            timing_summary=evidence["summary"],
            timing_companion=evidence["companion"],
            task_file=evidence["task"],
            sacct_raw=evidence["sacct_raw"],
            stdout=evidence["stdout"],
            total_records=10,
            primary_contract=False,
        )


def test_primary_plan_is_consumed_by_exact_base_launch(tmp_path, monkeypatch):
    evidence = _timing_evidence(
        tmp_path,
        total_records=planner.DEFAULT_TOTAL_RECORDS,
        timing_records=32,
        elapsed=1000,
    )
    monkeypatch.setattr(planner, "git_identity", lambda: dict(evidence["git"]))
    plan = planner.build_plan(
        timing_summary=evidence["summary"],
        timing_companion=evidence["companion"],
        task_file=evidence["task"],
        sacct_raw=evidence["sacct_raw"],
        stdout=evidence["stdout"],
    )
    plan_path = planner.write_plan_exclusive(tmp_path / "primary-plan.json", plan)
    spec = plan["array_plan"]["base"]
    train_freeze = Path(
        plan["code"]["exact_environment_contract"]["train_freeze"]["path"]
    )

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
        array_task_count=spec["shard_count"],
        array_task_min=0,
        array_task_max=spec["shard_index_stop"],
    )
    assert binding["plan_payload_sha256"] == plan["plan_payload_sha256"]
    assert binding["array_spec"] == spec["array_spec"]

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
        )
