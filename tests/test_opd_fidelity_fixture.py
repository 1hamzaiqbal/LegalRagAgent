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
