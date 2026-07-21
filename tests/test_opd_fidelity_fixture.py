import json
from pathlib import Path

import pytest

from scripts.opd.verify_verl_stored_rollout import load_fixture


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "configs/opd_math/fidelity/shared_rollout_k1_v1.json"


def test_stored_rollout_fixture_has_exact_finite_contract():
    payload = load_fixture(FIXTURE)
    assert payload["fixture_id"] == "shared_rollout_k1_v1"
    assert payload["scientific_launch_authorized"] is False
    assert len(payload["samples"]) == 2
    assert sum(sum(sample["response_mask"]) for sample in payload["samples"]) == 5


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda payload: payload.update(scientific_launch_authorized=True), "must not authorize"),
        (
            lambda payload: payload["settings"].update(loss_max_clamp=5.0),
            "settings drifted",
        ),
        (
            lambda payload: payload["samples"][0]["behavior_logprobs"].pop(),
            "exactly 4 values",
        ),
        (
            lambda payload: payload["samples"][1].update(sample_id="fixture:0"),
            "duplicate sample_id",
        ),
    ],
)
def test_stored_rollout_fixture_mutations_fail_closed(tmp_path, mutation, match):
    payload = json.loads(FIXTURE.read_text())
    mutation(payload)
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=match):
        load_fixture(changed)


def test_real_model_fixture_requires_live_hash_bound_provenance(tmp_path):
    source = tmp_path / "samples.jsonl"
    run = tmp_path / "run.json"
    completion = tmp_path / "completion.json"
    for path in (source, run, completion):
        path.write_text("{}\n")
    payload = json.loads(FIXTURE.read_text())
    payload["fixture_id"] = "real_model_rollout_k1_v1"
    payload["status"] = "real_model_stored_tensor_fidelity_only"
    digest = lambda path: __import__("hashlib").sha256(path.read_bytes()).hexdigest()
    payload["provenance"] = {
        "source_samples": str(source),
        "source_samples_sha256": digest(source),
        "run_manifest": str(run),
        "run_manifest_sha256": digest(run),
        "completion_manifest": str(completion),
        "completion_manifest_sha256": digest(completion),
        "local_git_commit": "a" * 40,
        "objective_registry_sha256": "b" * 64,
        "student": "Qwen/Qwen3-1.7B",
        "student_revision": "c" * 40,
        "teacher_checkpoint": "/checkpoint",
        "teacher_checkpoint_tree_sha256": "d" * 64,
        "extractor_sha256": "e" * 64,
        "behavior_logprobs_origin": "generation_transition_scores_before_update",
        "current_student_logprobs_origin": "pre_update_student_forward_on_generated_tokens",
        "teacher_logprobs_origin": "frozen_o_teacher_exact_generated_token_scores",
        "heldout_outcomes_inspected": False,
    }
    fixture = tmp_path / "real.json"
    fixture.write_text(json.dumps(payload))
    assert load_fixture(fixture)["fixture_id"] == "real_model_rollout_k1_v1"
    source.write_text('{"changed":true}\n')
    with pytest.raises(ValueError, match="source_samples binding drifted"):
        load_fixture(fixture)
