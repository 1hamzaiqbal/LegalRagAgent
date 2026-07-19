import hashlib
import json
from pathlib import Path

import pytest

from scripts.opd_math import evaluate_math as evaluation
from scripts.opd_math import merge_evaluations as merger
from scripts.opd_math import quality_gates as gates


COMMIT = "c" * 40
MODEL_REVISION = "d" * 40
PACKAGES = {
    "torch": "test",
    "transformers": "test",
    "peft": "test",
    "math-verify": "test",
}


def fake_verify(completion, _solution):
    reward = float(completion == "good")
    return {"reward": reward, "status": "correct" if reward else "incorrect"}


@pytest.fixture(autouse=True)
def stable_merge_runtime(monkeypatch):
    monkeypatch.setattr(merger, "verify_completion", fake_verify)
    monkeypatch.setattr(
        merger,
        "git_identity",
        lambda: {"commit": COMMIT, "worktree_clean": True},
    )
    monkeypatch.setattr(merger, "package_versions", lambda: dict(PACKAGES))
    monkeypatch.setattr(gates, "verify_completion", fake_verify)
    monkeypatch.setattr(gates, "EXPECTED_EVALUATION_PACKAGES", dict(PACKAGES))


def write_task(path: Path, count: int = 7):
    rows = [
        {
            "record_id": f"M:{index}",
            "cluster_id": f"cluster:{index}",
            "source": "M",
            "role": "student_opd",
            "prompt": [{"role": "user", "content": f"problem {index}"}],
            "solution": "gold",
        }
        for index in range(count)
    ]
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return rows


def contract_for(task_path: Path, rows, shard_count: int):
    task_hash = evaluation.sha256_file(task_path)
    custody = {
        "git": {"commit": COMMIT, "worktree_clean": True},
        "evaluator_file_sha256": evaluation.sha256_file(Path(evaluation.__file__)),
        "packages": dict(PACKAGES),
        "task_file": str(task_path.resolve()),
        "task_file_sha256": task_hash,
        "adapter": None,
        "adapter_tree_sha256": None,
    }
    decoding = {
        "thinking": False,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "max_new_tokens": 32,
        "seed": 0,
    }
    return evaluation.evaluation_contract(
        model="Qwen/Qwen3-1.7B",
        model_revision=MODEL_REVISION,
        adapter=None,
        adapter_tree_sha256=None,
        task_file=str(task_path.resolve()),
        task_file_sha256=task_hash,
        eligible_record_ids=[row["record_id"] for row in rows],
        task_sources=["M"],
        task_roles=["student_opd"],
        samples_per_problem=2,
        decoding=decoding,
        shard_count=shard_count,
        tokenizer_contract_sha256="a" * 64,
        custody=custody,
    )


def sample_rows_for(task_path: Path, task_rows, contract, start: int):
    task_hash = evaluation.sha256_file(task_path)
    rows = []
    for local_index, task_row in enumerate(task_rows):
        global_index = start + local_index
        seed = evaluation.record_sampling_seed(
            contract["record_seed_contract"]["base_seed"],
            task_hash,
            global_index,
            task_row["record_id"],
        )
        for sample_idx in range(contract["samples_per_problem"]):
            completion = "good" if (global_index + sample_idx) % 3 else "bad"
            verdict = fake_verify(completion, task_row["solution"])
            rows.append(
                {
                    "schema_version": 2,
                    "record_id": task_row["record_id"],
                    "global_record_index": global_index,
                    "record_seed": seed,
                    "cluster_id": task_row["cluster_id"],
                    "source": task_row["source"],
                    "sample_idx": sample_idx,
                    "reward": verdict["reward"],
                    "reward_status": verdict["status"],
                    "completion_tokens": 2 + sample_idx,
                    "prompt_tokens": 10 + global_index,
                    "generation_batch_latency_seconds": 0.25 + global_index,
                    "completion_text": completion,
                    "completion_sha256": hashlib.sha256(completion.encode()).hexdigest(),
                }
            )
    return rows


def write_valid_shards(root: Path, task_path: Path, task_rows, shard_count: int = 3):
    root.mkdir()
    contract = contract_for(task_path, task_rows, shard_count)
    task_hash = evaluation.sha256_file(task_path)
    for shard_index in range(shard_count):
        start, stop = evaluation.balanced_shard_bounds(
            len(task_rows), shard_count, shard_index
        )
        selected = task_rows[start:stop]
        sample_rows = sample_rows_for(task_path, selected, contract, start)
        shard_dir = root / f"shard_{shard_index:05d}"
        shard_dir.mkdir()
        samples_path = shard_dir / "samples.jsonl"
        samples_path.write_text(
            "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in sample_rows)
        )
        metrics = merger.validate_sample_rows(
            sample_rows,
            task_rows=selected,
            record_start=start,
            samples_per_problem=contract["samples_per_problem"],
            task_hash=task_hash,
            base_seed=contract["record_seed_contract"]["base_seed"],
        )
        git = {"commit": COMMIT, "worktree_clean": True}
        summary = {
            "schema_version": 2,
            "artifact_kind": evaluation.EVALUATION_SHARD_KIND,
            "evaluation_contract": contract,
            "evaluation_contract_sha256": evaluation.canonical_sha256(contract),
            "model": contract["model"],
            "model_revision": contract["model_revision"],
            "code": {
                "git": git,
                "evaluator_file_sha256": contract["code"]["evaluator_file_sha256"],
                "packages": dict(PACKAGES),
            },
            "custody": {
                "git_start": git,
                "git_end": git,
                "evaluator_file_sha256_start": contract["code"]["evaluator_file_sha256"],
                "evaluator_file_sha256_end": contract["code"]["evaluator_file_sha256"],
                "packages_start": dict(PACKAGES),
                "packages_end": dict(PACKAGES),
                "task_file_sha256_start": task_hash,
                "task_file_sha256_end": task_hash,
                "adapter_tree_sha256_start": None,
                "adapter_tree_sha256_end": None,
                "stable": True,
            },
            "tokenizer_contract_sha256": contract["tokenizer_contract_sha256"],
            "adapter": None,
            "adapter_tree_sha256": None,
            "task_file": str(task_path.resolve()),
            "task_file_sha256": task_hash,
            "records": metrics["records"],
            "eligible_records": len(task_rows),
            "task_sources": ["M"],
            "task_roles": ["student_opd"],
            "samples_per_problem": contract["samples_per_problem"],
            "samples": metrics["samples"],
            "accuracy": metrics["accuracy"],
            "prediction_parse_failure_fraction": metrics[
                "prediction_parse_failure_fraction"
            ],
            "unique_prompt_tokens": metrics["unique_prompt_tokens"],
            "expanded_prompt_tokens": metrics["expanded_prompt_tokens"],
            "total_completion_tokens": metrics["total_completion_tokens"],
            "total_generation_latency_seconds": metrics[
                "total_generation_latency_seconds"
            ],
            "peak_cuda_memory_bytes": 100 + shard_index,
            "decoding": contract["decoding"],
            "record_seed_contract": contract["record_seed_contract"],
            "shard": {
                "strategy": evaluation.SHARD_STRATEGY,
                "shard_count": shard_count,
                "shard_index": shard_index,
                "global_records": len(task_rows),
                "record_start": start,
                "record_stop": stop,
                "selected_record_ids_sha256": evaluation.canonical_sha256(
                    [row["record_id"] for row in selected]
                ),
            },
            "completion_text_in_samples": True,
            "samples_file": "samples.jsonl",
            "samples_file_sha256": evaluation.sha256_file(samples_path),
        }
        (shard_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
    return contract


def rewrite_summary_for_samples(shard_dir: Path):
    summary_path = shard_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["samples_file_sha256"] = evaluation.sha256_file(
        shard_dir / "samples.jsonl"
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def test_balanced_shards_cover_every_record_once():
    bounds = [evaluation.balanced_shard_bounds(10, 3, index) for index in range(3)]
    assert bounds == [(0, 3), (3, 6), (6, 10)]
    assert [index for start, stop in bounds for index in range(start, stop)] == list(range(10))
    for args in ((0, 1, 0), (3, 0, 0), (3, 4, 0), (3, 2, 2)):
        with pytest.raises(ValueError):
            evaluation.balanced_shard_bounds(*args)


def test_record_seed_binds_task_index_and_record_but_not_shard_geometry():
    task_hash = "1" * 64
    seed = evaluation.record_sampling_seed(0, task_hash, 5, "M:5")
    assert seed == evaluation.record_sampling_seed(0, task_hash, 5, "M:5")
    assert seed != evaluation.record_sampling_seed(1, task_hash, 5, "M:5")
    assert seed != evaluation.record_sampling_seed(0, "2" * 64, 5, "M:5")
    assert seed != evaluation.record_sampling_seed(0, task_hash, 6, "M:5")
    assert seed != evaluation.record_sampling_seed(0, task_hash, 5, "M:6")


def test_merger_recomputes_rewards_and_emits_ordered_fresh_artifact(tmp_path):
    task_path = tmp_path / "task.jsonl"
    task_rows = write_task(task_path)
    shard_root = tmp_path / "shards"
    contract = write_valid_shards(shard_root, task_path, task_rows)
    output = tmp_path / "merged"

    result = merger.merge_shards(
        shard_root=shard_root,
        shard_count=3,
        task_file=task_path,
        output_dir=output,
    )

    assert result["schema_version"] == 2
    assert result["artifact_kind"] == evaluation.EVALUATION_MERGED_KIND
    assert result["evaluation_contract"] == contract
    assert result["records"] == len(task_rows)
    assert result["samples"] == len(task_rows) * 2
    assert result["samples_file"] == "samples.jsonl"
    assert result["merge"]["selected_record_ids_sha256"] == evaluation.canonical_sha256(
        [row["record_id"] for row in task_rows]
    )
    assert [row["global_record_index"] for row in merger.iter_jsonl(output / "samples.jsonl")][::2] == list(
        range(len(task_rows))
    )
    assert json.loads((output / "summary.json").read_text()) == result
    assert not list(tmp_path.glob(".merged.partial.*"))

    with pytest.raises(FileExistsError, match="overwrite"):
        merger.merge_shards(
            shard_root=shard_root,
            shard_count=3,
            task_file=task_path,
            output_dir=output,
        )


def test_merger_rejects_missing_or_extra_completed_shards(tmp_path):
    task_path = tmp_path / "task.jsonl"
    task_rows = write_task(task_path)
    shard_root = tmp_path / "shards"
    write_valid_shards(shard_root, task_path, task_rows)
    (shard_root / "shard_00002").rename(shard_root / "missing_00002")
    with pytest.raises(ValueError, match="exact completed shard set"):
        merger.merge_shards(
            shard_root=shard_root,
            shard_count=3,
            task_file=task_path,
            output_dir=tmp_path / "merged",
        )

    (shard_root / "missing_00002").rename(shard_root / "shard_00002")
    (shard_root / "shard_00003").mkdir()
    with pytest.raises(ValueError, match="exact completed shard set"):
        merger.merge_shards(
            shard_root=shard_root,
            shard_count=3,
            task_file=task_path,
            output_dir=tmp_path / "merged",
        )


def test_merger_rejects_stale_sample_hash(tmp_path):
    task_path = tmp_path / "task.jsonl"
    task_rows = write_task(task_path)
    shard_root = tmp_path / "shards"
    write_valid_shards(shard_root, task_path, task_rows)
    samples = shard_root / "shard_00001" / "samples.jsonl"
    samples.write_text(samples.read_text() + "{}\n")

    with pytest.raises(ValueError, match="samples_file_sha256"):
        merger.merge_shards(
            shard_root=shard_root,
            shard_count=3,
            task_file=task_path,
            output_dir=tmp_path / "merged",
        )


def test_merger_rejects_forged_reward_even_when_file_hash_is_updated(tmp_path):
    task_path = tmp_path / "task.jsonl"
    task_rows = write_task(task_path)
    shard_root = tmp_path / "shards"
    write_valid_shards(shard_root, task_path, task_rows)
    shard = shard_root / "shard_00000"
    samples = list(merger.iter_jsonl(shard / "samples.jsonl"))
    samples[0]["reward"] = 1.0 - samples[0]["reward"]
    samples[0]["reward_status"] = "correct" if samples[0]["reward"] else "incorrect"
    (shard / "samples.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in samples)
    )
    rewrite_summary_for_samples(shard)

    with pytest.raises(ValueError, match="reward disagrees"):
        merger.merge_shards(
            shard_root=shard_root,
            shard_count=3,
            task_file=task_path,
            output_dir=tmp_path / "merged",
        )


def test_merger_rejects_wrong_record_seed_and_slice_metadata(tmp_path):
    task_path = tmp_path / "task.jsonl"
    task_rows = write_task(task_path)
    shard_root = tmp_path / "shards"
    write_valid_shards(shard_root, task_path, task_rows)
    shard = shard_root / "shard_00001"
    samples = list(merger.iter_jsonl(shard / "samples.jsonl"))
    samples[0]["record_seed"] += 1
    (shard / "samples.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in samples)
    )
    rewrite_summary_for_samples(shard)
    with pytest.raises(ValueError, match="record_seed"):
        merger.merge_shards(
            shard_root=shard_root,
            shard_count=3,
            task_file=task_path,
            output_dir=tmp_path / "merged-seed",
        )

    # Restore valid shards, then mutate only a shard's claimed boundary.
    other_root = tmp_path / "other-shards"
    write_valid_shards(other_root, task_path, task_rows)
    summary_path = other_root / "shard_00001" / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["shard"]["record_start"] -= 1
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="exact task slice"):
        merger.merge_shards(
            shard_root=other_root,
            shard_count=3,
            task_file=task_path,
            output_dir=tmp_path / "merged-boundary",
        )


def test_merger_rejects_contract_or_live_code_identity_drift(tmp_path):
    task_path = tmp_path / "task.jsonl"
    task_rows = write_task(task_path)
    shard_root = tmp_path / "shards"
    write_valid_shards(shard_root, task_path, task_rows)
    monkey_summary_path = shard_root / "shard_00002" / "summary.json"
    monkey_summary = json.loads(monkey_summary_path.read_text())
    monkey_summary["evaluation_contract"]["decoding"]["top_k"] = 7
    monkey_summary["evaluation_contract_sha256"] = evaluation.canonical_sha256(
        monkey_summary["evaluation_contract"]
    )
    monkey_summary["decoding"]["top_k"] = 7
    monkey_summary_path.write_text(json.dumps(monkey_summary, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="one exact evaluation contract"):
        merger.merge_shards(
            shard_root=shard_root,
            shard_count=3,
            task_file=task_path,
            output_dir=tmp_path / "merged-contract",
        )


def test_transactional_directory_keeps_partial_separate_and_refuses_replace(tmp_path):
    final, partial = evaluation.begin_transactional_directory(tmp_path / "result")
    (partial / "artifact").write_text("ok")
    evaluation.promote_transactional_directory(partial, final)
    assert (final / "artifact").read_text() == "ok"
    assert not partial.exists()
    with pytest.raises(FileExistsError, match="overwrite"):
        evaluation.begin_transactional_directory(final)


def test_quality_gate_reconstructs_merged_artifact_and_rejects_raw_shard(tmp_path):
    task_path = tmp_path / "task.jsonl"
    task_rows = write_task(task_path)
    shard_root = tmp_path / "shards"
    write_valid_shards(shard_root, task_path, task_rows)
    output = tmp_path / "merged"
    merger.merge_shards(
        shard_root=shard_root,
        shard_count=3,
        task_file=task_path,
        output_dir=output,
    )

    summary, grouped, binding = gates.checked_evaluation(
        output / "summary.json",
        output / "samples.jsonl",
        expected_model="Qwen/Qwen3-1.7B",
        expected_revision=MODEL_REVISION,
        expected_source="M",
        expected_role="student_opd",
    )
    assert summary["artifact_kind"] == evaluation.EVALUATION_MERGED_KIND
    assert len(grouped) == len(task_rows)
    assert binding["evaluation_shard_count"] == 3
    assert binding["record_seed_contract"]["strategy"] == evaluation.RECORD_SEED_STRATEGY

    with pytest.raises(ValueError, match="incomplete evaluation shard"):
        gates.checked_evaluation(
            shard_root / "shard_00000" / "summary.json",
            shard_root / "shard_00000" / "samples.jsonl",
            expected_model="Qwen/Qwen3-1.7B",
            expected_revision=MODEL_REVISION,
            expected_source="M",
            expected_role="student_opd",
        )


def test_one_shard_still_requires_and_passes_cpu_merge_before_gate(tmp_path):
    task_path = tmp_path / "task.jsonl"
    task_rows = write_task(task_path)
    shard_root = tmp_path / "shards"
    write_valid_shards(shard_root, task_path, task_rows, shard_count=1)
    output = tmp_path / "merged"
    merger.merge_shards(
        shard_root=shard_root,
        shard_count=1,
        task_file=task_path,
        output_dir=output,
    )
    _, grouped, binding = gates.checked_evaluation(
        output / "summary.json",
        output / "samples.jsonl",
        expected_model="Qwen/Qwen3-1.7B",
        expected_revision=MODEL_REVISION,
        expected_source="M",
        expected_role="student_opd",
    )
    assert len(grouped) == len(task_rows)
    assert binding["evaluation_shard_count"] == 1
    assert binding["evaluation_artifact_kind"] == evaluation.EVALUATION_MERGED_KIND


def test_quality_gate_rejects_bound_shard_mutation_after_merge(tmp_path):
    task_path = tmp_path / "task.jsonl"
    task_rows = write_task(task_path)
    shard_root = tmp_path / "shards"
    write_valid_shards(shard_root, task_path, task_rows)
    output = tmp_path / "merged"
    merger.merge_shards(
        shard_root=shard_root,
        shard_count=3,
        task_file=task_path,
        output_dir=output,
    )
    shard_samples = shard_root / "shard_00001" / "samples.jsonl"
    shard_samples.write_text(shard_samples.read_text() + "{}\n")

    with pytest.raises(ValueError, match="changed after merge"):
        gates.checked_evaluation(
            output / "summary.json",
            output / "samples.jsonl",
            expected_model="Qwen/Qwen3-1.7B",
            expected_revision=MODEL_REVISION,
            expected_source="M",
            expected_role="student_opd",
        )


def publication_fixture(tmp_path, name="published"):
    final, partial = evaluation.begin_transactional_directory(tmp_path / name)
    (partial / "samples.jsonl").write_text('{"sample":1}\n')
    summary = {
        "artifact_kind": evaluation.EVALUATION_SHARD_KIND,
        "evaluation_contract": {
            "contract": evaluation.EVALUATION_CONTRACT,
            "eligible_record_ids_sha256": "1" * 64,
        },
        "evaluation_contract_sha256": "2" * 64,
        "model": "test-model",
        "model_revision": MODEL_REVISION,
        "adapter_tree_sha256": None,
        "task_file_sha256": "3" * 64,
        "shard": {"shard_index": 0, "shard_count": 1},
    }
    (partial / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    state = {
        "git": {"commit": COMMIT, "worktree_clean": True},
        "stable_environment": True,
    }

    def require(start, end):
        if dict(start) != dict(end):
            raise RuntimeError("injected custody drift")

    return final, partial, summary, state, require


def test_post_promotion_companion_is_published_last_without_overwrite(tmp_path):
    final, partial, summary, state, require = publication_fixture(tmp_path)
    payload = evaluation.publish_transactional_artifact(
        partial,
        final,
        summary=summary,
        producer="evaluation_shard",
        custody_start=state,
        capture_custody=lambda: dict(state),
        require_stable_custody=require,
    )

    companion = evaluation.post_promotion_custody_path(final)
    assert final.is_dir()
    assert companion.is_file() and not companion.is_symlink()
    assert json.loads(companion.read_text()) == payload
    assert payload["post_promotion_custody_a"] == state
    assert payload["post_promotion_custody_b"] == state
    assert payload["post_promotion_custody_c"] == state
    assert payload["output_tree_sha256"] == gates.sha256_tree(final)
    assert not list(tmp_path.glob(".published.custody.partial.*"))


@pytest.mark.parametrize("drift_capture", [1, 2, 3, 4])
def test_post_promotion_custody_drift_quarantines_without_authorizing(
    tmp_path, drift_capture
):
    final, partial, summary, state, require = publication_fixture(
        tmp_path, f"drift-{drift_capture}"
    )
    calls = 0

    def capture():
        nonlocal calls
        calls += 1
        result = dict(state)
        if calls == drift_capture:
            result["stable_environment"] = False
        return result

    with pytest.raises(RuntimeError, match="custody drift"):
        evaluation.publish_transactional_artifact(
            partial,
            final,
            summary=summary,
            producer="evaluation_shard",
            custody_start=state,
            capture_custody=capture,
            require_stable_custody=require,
        )
    assert not final.exists()
    assert not evaluation.post_promotion_custody_path(final).exists()
    assert list((tmp_path / "rejected").glob(f"drift-{drift_capture}_*"))


def test_post_promotion_tree_mutation_is_quarantined(tmp_path):
    final, partial, summary, state, require = publication_fixture(tmp_path, "tree-drift")
    calls = 0

    def capture():
        nonlocal calls
        calls += 1
        if calls == 2:
            (final / "samples.jsonl").write_text('{"mutated":true}\n')
        return dict(state)

    with pytest.raises(RuntimeError, match="final artifact changed"):
        evaluation.publish_transactional_artifact(
            partial,
            final,
            summary=summary,
            producer="evaluation_shard",
            custody_start=state,
            capture_custody=capture,
            require_stable_custody=require,
        )
    assert not final.exists()
    assert not evaluation.post_promotion_custody_path(final).exists()


def test_target_appearing_after_lock_is_not_quarantined_as_foreign(tmp_path, monkeypatch):
    final, partial, summary, state, require = publication_fixture(tmp_path, "foreign")
    real_open = evaluation.os.open
    injected = False

    def open_with_foreign_target(path, flags, mode=0o777):
        nonlocal injected
        descriptor = real_open(path, flags, mode)
        if Path(path).name == ".foreign.promotion.lock" and not injected:
            final.mkdir()
            (final / "owner.txt").write_text("foreign artifact\n")
            injected = True
        return descriptor

    monkeypatch.setattr(evaluation.os, "open", open_with_foreign_target)
    with pytest.raises(FileExistsError, match="replace published"):
        evaluation.publish_transactional_artifact(
            partial,
            final,
            summary=summary,
            producer="evaluation_shard",
            custody_start=state,
            capture_custody=lambda: dict(state),
            require_stable_custody=require,
        )
    assert (final / "owner.txt").read_text() == "foreign artifact\n"
    assert partial.is_dir()
    assert not (tmp_path / "rejected").exists()


def test_companion_eexist_never_overwrites_and_revokes_output(tmp_path):
    final, partial, summary, state, require = publication_fixture(tmp_path, "eexist")
    companion = evaluation.post_promotion_custody_path(final)
    calls = 0

    def capture():
        nonlocal calls
        calls += 1
        if calls == 4:
            companion.write_text("foreign companion\n")
        return dict(state)

    with pytest.raises(FileExistsError):
        evaluation.publish_transactional_artifact(
            partial,
            final,
            summary=summary,
            producer="evaluation_shard",
            custody_start=state,
            capture_custody=capture,
            require_stable_custody=require,
        )
    assert not final.exists()
    assert not companion.exists()
    rejected_companions = list((tmp_path / "rejected").glob("eexist_*.custody.json"))
    assert len(rejected_companions) == 1
    assert rejected_companions[0].read_text() == "foreign companion\n"


def test_gate_rejects_symlinked_shard_path_before_resolution(tmp_path):
    task_path = tmp_path / "task.jsonl"
    task_rows = write_task(task_path)
    shard_root = tmp_path / "shards"
    write_valid_shards(shard_root, task_path, task_rows, shard_count=1)
    output = tmp_path / "merged"
    merger.merge_shards(
        shard_root=shard_root,
        shard_count=1,
        task_file=task_path,
        output_dir=output,
    )
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text())
    real_shard_summary = shard_root / "shard_00000" / "summary.json"
    linked_summary = tmp_path / "linked-shard-summary.json"
    linked_summary.symlink_to(real_shard_summary)
    summary["merge"]["shards"][0]["summary"] = str(linked_summary)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="regular non-symlink file"):
        gates.checked_evaluation(
            summary_path,
            output / "samples.jsonl",
            expected_model="Qwen/Qwen3-1.7B",
            expected_revision=MODEL_REVISION,
            expected_source="M",
            expected_role="student_opd",
        )
